"""The server's worker (opentpu/serve/scheduler.py) on the ISA simulator with tiny models:
requests submitted together run in turns, each context switched out to the KV store and back,
and every request's greedy tokens equal its own run on a fresh engine, bit for bit (the card's
decode loop and the host's steps alike; Qwen3, LFM2 and Qwen3.5: KV caches only, convolution
rings, DeltaNet states); a prompt that continues a saved context feeds only the rest; stop
strings, max_new and cancels end requests; the store on disk serves a second worker."""
import threading

import numpy as np
import pytest

from opentpu.llm.qwen3 import Engine
from opentpu.serve import scheduler as SC
from opentpu.serve import store as ST

from test_serve_kvio import _model

torch = pytest.importorskip("torch")

CAP = 512


class _Tok:
    """One character per token id."""

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(0x100 + int(i)) for i in ids)


def _engine(spec, W, cfg, card=True):
    return Engine(spec, W, cap=CAP, cfg=cfg, resident=card, prompt_runs=card)


def _ref(spec, W, cfg, prompt, n, stop):
    """Greedy tokens of a fresh engine (stop ids end it, not included)."""
    e = _engine(spec, W, cfg, card=False)
    lg, out = e.prefill(prompt), []
    while len(out) < n:
        t = int(np.argmax(lg))
        if t in stop:
            break
        out.append(t)
        if len(out) == n:
            break
        lg = e.step(t)
    return out


def _prompts(seed, sizes, base=None):
    r = np.random.default_rng(seed)
    return [(list(base) if base else []) + [int(x) for x in r.integers(0, 1000, n)]
            for n in sizes]


@pytest.mark.parametrize("case, card", [("qwen3", True), ("qwen3", False), ("lfm2", True),
                                        ("qwen35", True)])
def test_requests_in_turns_are_their_own_runs(case, card):
    """Four requests at once, quantum 3: each runs in several turns, the others' contexts in
    between; their tokens are their own fresh runs'. Then a fifth whose prompt continues the
    first's: it feeds only what is new."""
    W, spec, cfg = _model(case)
    stop = frozenset({int(spec.eos[0])} if spec.eos else ())
    eng = _engine(spec, W, cfg, card)
    w = SC.Worker(eng, _Tok(), quantum=3, max_running=4).start()
    try:
        ps = _prompts(1, [21, 34, 9, 27])
        reqs = [w.submit(SC.Request(p, max_new=10, stop_ids=stop)) for p in ps]
        for r in reqs:
            r.wait(timeout=600)
        for p, r in zip(ps, reqs):
            assert r.finish in ("length", "stop") and r.error is None, r.error
            assert r.out == _ref(spec, W, cfg, p, 10, stop), r.rid
        assert sum(r.swaps for r in reqs) >= 3 and w.stats["restores"] >= 3
        # the first request's conversation, continued
        nxt = ps[0] + reqs[0].out + _prompts(2, [6])[0]
        r = w.submit(SC.Request(nxt, max_new=8, stop_ids=stop)).wait(timeout=600)
        assert r.out == _ref(spec, W, cfg, nxt, 8, stop)
        if not eng.stateful:
            assert r.cached >= len(ps[0]) + len(reqs[0].out) - 1
        else:                               # (its end saved: the prompt and the reply less one)
            assert r.cached == len(ps[0]) + len(reqs[0].out) - 1 or r.cached == 0
        assert r.prefill_tokens == len(nxt) - r.cached
    finally:
        w.stop()


def test_shared_prefix_from_the_store_and_the_device():
    """Requests sharing a 260-token system prompt: the second takes it from the device (the
    first's context rolled back), the third from the store (another context ran between);
    each feeds only its own part, with its fresh run's tokens."""
    W, spec, cfg = _model("qwen3")
    stop = frozenset()
    w = SC.Worker(_engine(spec, W, cfg), _Tok(), quantum=4).start()
    try:
        sysp = _prompts(5, [260])[0]
        a, b = _prompts(6, [12, 15], base=sysp)
        other = _prompts(7, [30])[0]
        ra = w.submit(SC.Request(a, max_new=5, stop_ids=stop)).wait(600)
        rb = w.submit(SC.Request(b, max_new=5, stop_ids=stop)).wait(600)
        assert rb.cached == 260 and w.stats["device_reuses"] >= 1
        w.submit(SC.Request(other, max_new=3, stop_ids=stop)).wait(600)
        c = sysp + _prompts(8, [10])[0]
        rc = w.submit(SC.Request(c, max_new=5, stop_ids=stop)).wait(600)
        assert rc.cached >= 260 and w.stats["restores"] >= 1
        for p, r in ((a, ra), (b, rb), (c, rc)):
            assert r.out == _ref(spec, W, cfg, p, 5, stop)
        u = w.usage()
        assert u["prefix_cache"]["shared_pages"] >= 1 and u["kv_store"]["ram_bytes"] > 0
    finally:
        w.stop()


def test_stop_strings_max_new_and_cancel():
    W, spec, cfg = _model("qwen3")
    w = SC.Worker(_engine(spec, W, cfg), _Tok(), quantum=4).start()
    try:
        p = _prompts(9, [15])[0]
        want = _ref(spec, W, cfg, p, 12, frozenset())
        text = _Tok().decode(want)
        stop_s = text[5:7]
        r = w.submit(SC.Request(p, max_new=12, stop_ids=frozenset(), stop_strs=(stop_s,)))
        got = "".join(r.stream(timeout=600))
        assert got == text[:text.find(stop_s)] and r.finish == "stop" and stop_s not in got
        r = w.submit(SC.Request(p, max_new=4, stop_ids=frozenset())).wait(600)
        assert r.finish == "length" and r.out == want[:4] and r.text == text[:4]
        # a cancel: before it runs, and while it runs
        gate = threading.Event()
        r1 = SC.Request(_prompts(10, [20])[0], max_new=200, stop_ids=frozenset())
        r1.cancel()
        w.submit(r1).wait(600)
        assert r1.finish == "abort"
        r2 = w.submit(SC.Request(_prompts(11, [20])[0], max_new=300, stop_ids=frozenset()))
        it = r2.stream(timeout=600)
        next(it)
        r2.cancel()
        r2.wait(600)
        assert r2.finish == "abort" and len(r2.out) < 300
        gate.set()
    finally:
        w.stop()


def test_store_on_disk_serves_another_worker(tmp_path):
    """A worker whose store spills to disk; a second worker on the same directory continues the
    first's saved context (its prompt's prefix restored from the files)."""
    W, spec, cfg = _model("qwen3")
    mk = ST.model_key("tiny-qwen3")
    stop = frozenset()
    s1 = ST.BlobStore(64 << 10, 1 << 30, tmp_path)      # 64 KiB of RAM: spills
    w = SC.Worker(_engine(spec, W, cfg), _Tok(), store=s1, model_key=mk).start()
    p = _prompts(12, [300])[0]
    try:
        r = w.submit(SC.Request(p, max_new=4, stop_ids=stop)).wait(600)
        w.submit(SC.Request(_prompts(13, [20])[0], max_new=2, stop_ids=stop)).wait(600)
    finally:
        w.stop()
    s1.flush()
    assert s1.usage()["disk_blobs"] >= 2
    s2 = ST.BlobStore(1 << 30, 1 << 30, tmp_path)
    w2 = SC.Worker(_engine(spec, W, cfg), _Tok(), store=s2, model_key=mk).start()
    try:
        q = p + r.out + _prompts(14, [5])[0]
        r2 = w2.submit(SC.Request(q, max_new=4, stop_ids=stop)).wait(600)
        assert r2.cached >= 300 and r2.out == _ref(spec, W, cfg, q, 4, stop)
        assert s2.usage()["disk_hits"] >= 1
    finally:
        w2.stop()


def test_requests_that_do_not_fit_are_refused():
    W, spec, cfg = _model("qwen3")
    w = SC.Worker(_engine(spec, W, cfg), _Tok())
    with pytest.raises(ValueError):
        w.submit(SC.Request(list(range(CAP)), max_new=1))
    with pytest.raises(ValueError):
        w.submit(SC.Request([], max_new=1))
    r = SC.Request(list(range(CAP - 3)), max_new=50)
    w.submit(r)
    assert r.max_new == 3
