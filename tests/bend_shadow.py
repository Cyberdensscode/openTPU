"""pytest -p bend_shadow (with tests/ on the path): every isasim Machine.run also runs on the
Bend machine (opentpu.bend_sim) from the same state, and the two must end bit for bit alike:
each slice's DRAM, TMEM, ACT RAM (int8, Sa, offsets), registers, instruction count, WAITW stalls
and HALT CHAIN restarts, or both stop on an error. A mismatch fails the test.

OTPU_BEND_LOG=file appends a JSON line per run (the test, the outcome, isasim's and Bend's
seconds). Runs with a WAITW host hook are compared only when it moves DRAM only (it is called with
the Bend machine's DRAM view), so they are skipped; programs whose units the Bend machine does not
have yet (OTPU_BEND_TODO, opcodes) are skipped too."""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pytest

from opentpu import isa as I
from opentpu import isasim
from opentpu.bend_sim import CTRL, DRAM, ERRORS, TMEM, BendMachine, words_of

TODO = {int(x, 0) for x in os.environ.get("OTPU_BEND_TODO", "").split(",") if x}
# the words of isasim's message that name each Bend error (default: ERRORS' whole text)
KEY = {8: "outside its bound", 14: "HALT CHAIN", 13: "never holds", 24: "multiples of",
       38: "STREAM"}
_orig_run = isasim.Machine.run
_bend: dict = {}                     # the server for the last Config
_test = {"id": ""}


def _server(cfg) -> BendMachine:
    m = _bend.get(cfg)
    if m is None:
        for old in _bend.values():
            old.close()
        _bend.clear()
        m = _bend[cfg] = BendMachine(cfg)
    return m


def _log(rec: dict) -> None:
    p = os.environ.get("OTPU_BEND_LOG")
    if p:
        with open(p, "a") as f:
            f.write(json.dumps(rec) + "\n")


def _ops(progs) -> set:
    return {ins.op for p in progs for ins in p}


def _shadow_run(self, max_steps: int = 10_000_000):
    cfg = self.cfg
    progs = [list(s.prog) for s in self.slices]
    skip = None
    if self.host is not None:
        skip = "host hook"
    elif _ops(progs) & TODO:
        skip = "todo"
    elif any(s.pc or s.stack or s.halted or s.waiting or s.polling for s in self.slices):
        skip = "mid-run state"
    if skip is not None:
        _log({"test": _test["id"], "outcome": "skip", "why": skip})
        return _orig_run(self, max_steps)
    snap = [(s.dram.copy(), s.tmem.copy(), s.act.copy(), s.ascale.copy(), s.adoff.copy(),
             list(s.R)) for s in self.slices]
    chains0 = self.chains                # isasim counts across runs, the Bend machine per run
    t0 = time.perf_counter()
    err = None
    try:
        _orig_run(self, max_steps)
    except isasim.SimError as e:
        err = e
    t1 = time.perf_counter()
    b = _server(cfg)
    for k, (dram, tmem, act, sa, do, R) in enumerate(snap):
        b.put(k, DRAM, 0, dram)
        b.put(k, TMEM, 0, tmem)
        b.set_act(k, act, sa, do)
    b.load(progs, self.args)
    for k, (*_, R) in enumerate(snap):
        b.put(k, CTRL, 0, np.array([0] + list(R[1:16]), np.uint32))
    t2 = time.perf_counter()
    berr = None
    try:
        b.run()
    except isasim.SimError as e:
        berr = e
    t3 = time.perf_counter()
    rec = {"test": _test["id"], "S": cfg.S, "isasim_s": t1 - t0, "bend_s": t3 - t2,
           "icount": [s.icount for s in self.slices]}
    if err is not None or berr is not None:
        aux = int(b.ctrl[:, 4].max()) if berr is not None else 0
        if err is None and "bad opcode" in str(berr) and aux in TODO | {I.MM, I.QACT, I.QST,
                                                                          I.DSTEP, I.STREAM}:
            _log(dict(rec, outcome="skip", why=f"todo op {aux:#x} (chained)"))
            return self
        ok = err is not None and berr is not None
        if ok and getattr(berr, "code", 0) in ERRORS and \
                KEY.get(berr.code, ERRORS[berr.code]) not in str(err):
            ok = False                       # both stopped, on different errors
        _log(dict(rec, outcome="error" if ok else "MISMATCH", isasim=str(err), bend=str(berr)))
        if not ok:
            raise AssertionError(f"bend: isasim {err!r}, Bend {berr!r}")
        raise err
    bad = []
    for k, s in enumerate(self.slices):
        if b.ctrl[k, 1] != s.icount:
            bad.append(f"slice {k} icount {s.icount} vs {int(b.ctrl[k, 1])}")
        regs = b.get(k, CTRL, 0, 64).view(np.uint32)
        if [int(x) for x in regs[1:]] != [int(x) for x in s.R[1:]]:
            bad.append(f"slice {k} R {s.R} vs {regs.tolist()}")
        if not np.array_equal(b.tmem(k), s.tmem):
            d = np.nonzero(b.tmem(k) != s.tmem)[0]
            bad.append(f"slice {k} TMEM {d.size} words differ, first {d[:4].tolist()}")
        if not np.array_equal(b.dram(k), s.dram):
            d = np.nonzero(b.dram(k) != s.dram)[0]
            bad.append(f"slice {k} DRAM {d.size} bytes differ, first {d[:4].tolist()}")
        act, sa, do = b.act(k)
        if not (np.array_equal(act, s.act) and np.array_equal(sa.view(np.uint32),
                                                              s.ascale.view(np.uint32))
                and np.array_equal(do, s.adoff)):
            bad.append(f"slice {k} ACT RAM differs")
        if int(b.ctrl[k, 19]) != s.stalls:
            bad.append(f"slice {k} stalls {s.stalls} vs {int(b.ctrl[k, 19])}")
    if b.chains != self.chains - chains0:
        bad.append(f"chains {self.chains - chains0} vs {b.chains}")
    _log(dict(rec, outcome="MISMATCH" if bad else "same", diff=bad[:6]))
    if bad and os.environ.get("OTPU_BEND_DUMP"):
        import pickle
        with open(os.environ["OTPU_BEND_DUMP"], "wb") as f:   # the run, to replay
            pickle.dump({"cfg": cfg, "progs": [words_of(p) for p in progs],
                         "args": list(self.args or []), "snap": snap}, f)
    if bad:
        raise AssertionError("bend: " + "; ".join(bad[:6]))
    return self


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    _test["id"] = item.nodeid


def pytest_configure(config):
    isasim.Machine.run = _shadow_run


def pytest_unconfigure(config):
    isasim.Machine.run = _orig_run
    for m in _bend.values():
        m.close()
    _bend.clear()
