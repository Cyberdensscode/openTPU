"""The ISA simulator in Bend (bend/sim.bend, bend/machine.bend; docs/bend.md), as an Engine
backend: BendBackend runs the same programs as IsaBackend (opentpu.llm.qwen3), bit for bit
isasim's results, in a server process (build/otpu_bend, from bend/server.bend) that keeps the
machine between runs. The host talks to it over two pipes (the protocol: bend/server.bend).

    eng = Engine(spec, W, cfg, backend=bend_backend())

build() compiles the server (Bend 2: ~/.bend/bin/bend, or OTPU_BEND) when its sources are newer
than the binary.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np

from . import isa as I
from . import quant as Q
from .isasim import Config, SimError, _fit_imem

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "bend"
BIN = ROOT / "build" / "otpu_bend"

QUIT, INIT, DATA, LOAD, RUN, HOST = 0, 1, 2, 3, 4, 5
DRAM, TMEM, ACT, CTRL = 0, 1, 2, 3
IMEM = 4096                         # IMEM's first control word (sim.bend IMEM)
RUNNING, DONE, HOSTS, ERROR = 0, 1, 2, 3
POLL = 3                            # a slice's STATE polling a WAITW (sim.bend STATE)
CHUNK = 1 << 28                     # bytes a DATA command moves at most

# the slice's control words 16..39 as RUN returns them (sim.bend)
PC, ICOUNT, STATE, ERR, AUX, PLEN, SP, STALLS, CHAINS, VGEN, STEPS = \
    0, 1, 2, 3, 4, 5, 6, 19, 20, 21, 22

# the errors (sim.bend, machine.bend) as isasim words them (the slice and pc are added)
ERRORS = {
    1: "past end of program", 2: "bad opcode", 3: "LOOP with empty body",
    4: "nested loop bodies end on the same instruction", 5: "loop stack overflow",
    6: "bad DRAM word address", 7: "TMEM index out of range", 8: "RLD outside its bound",
    14: "HALT CHAIN: bad address or length",
    15: "MM: N and KB must be at least 1", 16: "MM: M or ACT RAM range out of bounds",
    17: "MM: RMAX and ASCALE need M <= MCOLS", 18: "MM: PAIR needs 2*M <= MCOLS",
    19: "MM: PAIR needs 4-bit weights", 20: "MM: PAIR needs an even k0",
    21: "MM: RSC does not take NVFP4", 22: "MM: ASCALE needs UNIT and ACC",
    23: "MM: streamed rows must be D-byte aligned and in range",
    24: "MM: ssa and srs must be multiples of the format's alignment",
    25: "MM: scale rows out of range", 26: "MM: an NVFP4 scale with the sign bit set",
    27: "MM: the ACT rows' blocks hold different Sa (more than one QACT)",
    28: "QACT out of ACT RAM bounds", 29: "QST: HALF needs ROW mode",
    30: "QST: byte address out of range", 31: "QST: bad scale address",
    32: "QST: the data and scale ranges overlap", 33: "VOP: bad func",
    34: "VOP read-after-write hazard",
    35: "VOP ARGMAX: rows > 1 needs drs >= 2 (a pair per row)",
    36: "VOP OUTER: bmode must be ROW and cols <= 256",
    37: "DSTEP: rows must be 1..256, cols 64, 128, 192 or 256", 38: "STREAM",
    43: "DSTEP: the state must be DRAM-chunk aligned",
    39: "VOP: a reduction of no columns",
    9: "collective with a halted slice", 10: "slices disagree on the collective instruction",
    11: "GATHER: slices disagree on dst/rows/cols/drs/seg",
    12: "GATHER: the source and destination overlap", 13: "WAITW never holds: the timeout",
    40: "deadlock", 41: "GATHER: too many words for the Bend machine's buffer",
    42: "an instruction too large for the Bend machine's scratch",
}


def _depth(words: int) -> int:
    return max(1, int(words - 1).bit_length())


def build(force: bool = False) -> Path:
    """The server binary, rebuilt when a bend/ source is newer."""
    srcs = list(SRC.glob("*.bend")) + list(SRC.glob("*.c"))
    if not force and BIN.exists() and BIN.stat().st_mtime >= max(p.stat().st_mtime for p in srcs):
        return BIN
    BIN.parent.mkdir(parents=True, exist_ok=True)
    bend = os.environ.get("OTPU_BEND", str(Path.home() / ".bend" / "bin" / "bend"))
    env = dict(os.environ, BEND_NO_TELEMETRY="1")
    subprocess.run([bend, str(SRC / "server.bend"), "-o", str(BIN)], check=True, env=env)
    return BIN


def words_of(prog) -> np.ndarray:
    """A program's words: assembled words (compiler.Words) as they are, instructions encoded."""
    if isinstance(prog, np.ndarray):
        return np.ascontiguousarray(prog, np.uint32).reshape(-1)
    return np.array([w for ins in prog for w in ins.encode()], np.uint32).reshape(-1)


class BendError(SimError):
    """A SimError with the Bend machine's error code (ERRORS) and slice (None: the machine's)."""

    def __init__(self, msg: str, code: int, sid=None):
        super().__init__(msg)
        self.code, self.sid = code, sid


class _Bytes:
    """A slice's DRAM bytes as isasim's Slice.dram reads them (d[a:b], d[a]), from the server."""

    def __init__(self, m: "BendMachine", sid: int):
        self.m, self.sid = m, sid

    def __len__(self) -> int:
        return self.m.cfg.DRAM_BYTES

    def __getitem__(self, k):
        if isinstance(k, slice):
            a, b, step = k.indices(len(self))
            if step != 1:
                raise IndexError("a Bend slice's DRAM reads contiguous bytes")
            return self.m.get(self.sid, DRAM, a, max(0, b - a))
        return self.m.get(self.sid, DRAM, int(k), 1)[0]


class _Words(_Bytes):
    """Its DRAM's 32-bit words (isasim's Slice.m32[i])."""

    def __len__(self) -> int:
        return self.m.cfg.DRAM_BYTES // 4

    def __getitem__(self, i):
        return self.m.get(self.sid, DRAM, 4 * int(i), 4).view(np.uint32)[0]


class BendSlice:
    """One slice of a BendMachine as isasim's Slice shows itself to a WAITW host or a test: the
    WAITW it polls, its registers and its DRAM (words and bytes), read from the server."""

    def __init__(self, m: "BendMachine", sid: int):
        self.m, self.sid = m, sid
        self.m32, self.dram = _Words(m, sid), _Bytes(m, sid)

    @property
    def polling(self) -> I.Instr | None:
        c = self.m.ctrl[self.sid]
        if int(c[STATE]) != POLL or not self.m.programs:
            return None
        pc = int(c[PC])
        return I.Instr.decode(self.m.programs[self.sid][8 * pc:8 * pc + 8])

    def reg(self, r: int) -> int:
        if r == 0:
            return 0
        if self.sid not in self.m.regs:     # (fixed while the machine waits on its host)
            self.m.regs[self.sid] = self.m.get(self.sid, CTRL, 0, 64).view(np.uint32)
        return int(self.m.regs[self.sid][r])

    def holds(self, ins: I.Instr) -> bool:
        """isasim's Slice.holds: whether WAITW ins's condition holds on its DRAM word now."""
        a = (self.reg(ins.ra) + ins.w[0]) & 0xFFFFFFFF
        return I.waitw_holds(int(self.m32[a // 4]), self.reg(ins.rc) + ins.w[2], ins.flags & 3,
                             ins.w[3])


class BendMachine:
    """isasim.Machine's state in the server: S slices' DRAM, TMEM, ACT RAM and control words.
    run() raises SimError as isasim does (the message from the error's code); slices shows
    each slice as isasim's do to a WAITW host (polling, reg, holds, m32, dram)."""

    def __init__(self, cfg: Config, threads: int | None = None, binary: Path | None = None):
        self.cfg = cfg
        if cfg.S < 1 or cfg.IMEM_WORDS % 8 or cfg.TMEM_WORDS & (cfg.TMEM_WORDS - 1) or \
                cfg.DRAM_BYTES & (cfg.DRAM_BYTES - 1) or cfg.DRAM_BYTES > 1 << 32:
            raise ValueError(f"no Bend machine for {cfg}")
        rows, ab, D = cfg.act_rows, cfg.ACT_BLOCKS, cfg.D
        self.aso = rows * ab * D // 4                  # Sa per (row, block)
        self.ado = self.aso + rows * ab                # offsets per (row, block, sub-block)
        self.act_words = self.ado + rows * ab * Q.NSUB
        # OTPU_BEND_BIN: a server built elsewhere (bend -o x.c, then clang on a host without
        # bend); OTPU_BEND_THREADS: the runtime's threads (default: the runtime's choice)
        binary = binary or Path(os.environ.get("OTPU_BEND_BIN", "") or build())
        threads = threads or int(os.environ.get("OTPU_BEND_THREADS", "0")) or None
        # the MXU: each row's outputs in parallel (bend/simpar.bend), or in order (sim.bend's
        # mm.seq) with seq_mxu or OTPU_BEND_SEQ=1; the results are the same
        self.seq_mxu = os.environ.get("OTPU_BEND_SEQ", "") == "1"
        ci, co = os.pipe()                             # commands: host -> server
        ri, ro = os.pipe()                             # replies: server -> host
        env = dict(os.environ, BEND_NO_TELEMETRY="1", OTPU_BEND_IN=str(ci),
                   OTPU_BEND_OUT=str(ro))
        argv = [str(binary)]
        if threads:
            argv += ["+RTS", "--threads", str(int(threads))]
        self.proc = subprocess.Popen(argv, pass_fds=(ci, ro), env=env)
        os.close(ci)
        os.close(ro)
        self._w = os.fdopen(co, "wb", buffering=0)
        self._r = os.fdopen(ri, "rb", buffering=0)
        flags = int(cfg.PAIR) | int(cfg.DSTEP) << 1 | int(cfg.STREAM) << 2
        self._cmd([INIT, cfg.S, D, cfg.MCOLS, ab, cfg.TMEM_WORDS, cfg.DRAM_BYTES - 1,
                   cfg.IMEM_WORDS, rows, flags, _depth(IMEM + cfg.IMEM_WORDS),
                   _depth(cfg.TMEM_WORDS), _depth(self.act_words), _depth(cfg.DRAM_BYTES // 4),
                   self.aso, self.ado])
        if self._reply()[0] != 1:
            raise SimError("the Bend server did not start")
        self.host = None                               # WAITW's host, as isasim.Machine.host
        self.chains = 0
        self.ctrl = np.zeros((cfg.S, 24), np.uint32)
        self.programs: list = []                       # each slice's program's words
        self.regs: dict = {}                           # registers read while the host acts
        self.slices = [BendSlice(self, s) for s in range(cfg.S)]

    # ---------------------------------------------------------------- the pipes
    def _cmd(self, words, data: bytes | memoryview | None = None) -> None:
        h = np.zeros(16, np.uint32)
        h[:len(words)] = [int(x) & 0xFFFFFFFF for x in words]
        self._w.write(h.tobytes())
        if data is not None:
            self._w.write(data)

    def _get(self, n: int) -> bytes:
        b = bytearray(n)
        v, k = memoryview(b), 0
        while k < n:
            r = self._r.readinto(v[k:])
            if not r:
                raise SimError(f"the Bend server stopped (exit {self.proc.poll()})")
            k += r
        return bytes(b)

    def _reply(self) -> np.ndarray:
        return np.frombuffer(self._get(64), np.uint32)

    def close(self) -> None:
        if self.proc.poll() is None:
            try:
                self._cmd([QUIT])
                self.proc.wait(10)
            except (OSError, subprocess.TimeoutExpired):
                self.proc.kill()
        for f in (self._w, self._r):
            f.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # ---------------------------------------------------------------- memories
    def _limit(self, array: int) -> int:
        c = self.cfg
        return {DRAM: c.DRAM_BYTES, TMEM: 4 * c.TMEM_WORDS, ACT: 4 * self.act_words,
                CTRL: 4 * (IMEM + c.IMEM_WORDS)}[array]

    def put(self, s: int, array: int, off: int, data) -> None:
        v = memoryview(np.ascontiguousarray(data)).cast("B")
        if not 0 <= s < self.cfg.S or off < 0 or off + len(v) > self._limit(array):
            raise SimError(f"slice {s}: write of {len(v)} bytes at {off:#x} outside the memory")
        for k in range(0, len(v), CHUNK):
            p = v[k:k + CHUNK]
            self._cmd([DATA, s, array, 0, off + k, len(p)], p)

    def get(self, s: int, array: int, off: int, nbytes: int) -> np.ndarray:
        if not 0 <= s < self.cfg.S or off < 0 or nbytes < 0 or \
                off + nbytes > self._limit(array):
            raise SimError(f"slice {s}: read of {nbytes} bytes at {off:#x} outside the memory")
        out = np.empty(nbytes, np.uint8)
        for k in range(0, nbytes, CHUNK):
            n = min(CHUNK, nbytes - k)
            self._cmd([DATA, s, array, 1, off + k, n])
            out[k:k + n] = np.frombuffer(self._get(n), np.uint8)
        return out

    def dram(self, s: int, addr: int = 0, nbytes: int | None = None) -> np.ndarray:
        n = self.cfg.DRAM_BYTES - addr if nbytes is None else nbytes
        return self.get(s, DRAM, addr, n)

    def tmem(self, s: int) -> np.ndarray:
        return self.get(s, TMEM, 0, 4 * self.cfg.TMEM_WORDS).view(np.uint32)

    def act(self, s: int):
        """isasim's act (int8 [rows][blocks * D]), ascale (fp32 [rows][blocks]) and adoff
        (uint8 [rows][blocks][NSUB])."""
        c, w = self.cfg, self.get(s, ACT, 0, 4 * self.act_words).view(np.uint32)
        rows, ab = c.act_rows, c.ACT_BLOCKS
        act = w[:self.aso].view(np.int8).reshape(rows, ab * c.D)
        sa = w[self.aso:self.ado].view(np.float32).reshape(rows, ab)
        do = w[self.ado:].astype(np.uint8).reshape(rows, ab, Q.NSUB)
        return act, sa, do

    def set_act(self, s: int, act, ascale, adoff) -> None:
        w = np.concatenate([np.ascontiguousarray(act, np.int8).reshape(-1).view(np.uint32),
                            np.asarray(ascale, np.float32).reshape(-1).view(np.uint32),
                            np.asarray(adoff, np.uint8).reshape(-1).astype(np.uint32)])
        self.put(s, ACT, 0, w)

    # ---------------------------------------------------------------- runs
    def load(self, programs, args=None) -> "BendMachine":
        """Start new programs on the same machine (isasim.Machine.load)."""
        assert len(programs) == self.cfg.S
        _fit_imem(self.cfg, programs)
        a = [int(x) & 0xFFFFFFFF for x in (args or [])]
        if len(a) > 8:
            raise ValueError("at most 8 run arguments")
        a += [0] * (8 - len(a))
        self.programs = [words_of(p) for p in programs]
        for s, w in enumerate(self.programs):
            self._cmd([LOAD, s, len(w) // 8] + a, w.tobytes())
        return self

    def _status(self) -> np.ndarray:
        r = self._reply()
        n = int(r[3])
        self.ctrl = np.frombuffer(self._get(96 * n), np.uint32).reshape(n, 24).copy()
        self.regs = {}
        return r

    def run(self, max_steps: int = 0) -> "BendMachine":
        self._cmd([RUN, int(self.seq_mxu)])
        r = self._status()
        while r[0] == HOSTS:
            # every slice that can run polls a WAITW: the host acts once (isasim.Machine.run),
            # then the polling slices read their words again
            if self.host is not None:
                self.host(self)
            self._cmd([HOST, int(self.seq_mxu)])
            r = self._status()
        if r[0] == ERROR:
            self._raise(int(r[1]), int(r[2]))
        self.chains = int(self.ctrl[:, CHAINS].sum())
        return self

    def _raise(self, e: int, es: int) -> None:
        if e:
            raise BendError(ERRORS.get(e, f"machine error {e}"), e)
        c = self.ctrl[es]
        code = int(c[ERR])
        raise BendError(f"slice {es}: {ERRORS.get(code, f'error {code}')} at pc {int(c[PC])} "
                        f"(aux {int(c[AUX]):#x})", code, es)

    @property
    def icounts(self) -> list[int]:
        return [int(x) for x in self.ctrl[:, ICOUNT]]


class BendBackend:
    """The Engine's backend on the Bend machine (as qwen3.IsaBackend on isasim's)."""

    args = True
    generates = True
    chains = True

    def __init__(self, cfg: Config, images: list, threads: int | None = None):
        self.cfg = cfg
        self.machine = BendMachine(cfg, threads)
        for s, m in enumerate(images):
            if m is not None and len(m):
                self.write(s, 0, m)
        self.host = None                    # the Engine's poll (offloaded experts' server)

    @property
    def host(self):
        return self._poll

    @host.setter
    def host(self, poll) -> None:
        """The Engine's poll: the machine's WAITW host polls with it as IsaBackend's does
        (qwen3._isa_host: until a polling slice's WAITW holds or it has nothing left to do)."""
        self._poll = poll
        if poll is None:
            self.machine.host = None
        else:
            from .llm.qwen3 import _isa_host
            self.machine.host = _isa_host(poll)

    def write(self, s: int, addr: int, data: np.ndarray) -> None:
        self.machine.put(s, DRAM, addr, np.ascontiguousarray(data).view(np.uint8).reshape(-1))

    def read(self, s: int, addr: int, nbytes: int) -> np.ndarray:
        return self.machine.get(s, DRAM, addr, nbytes)

    def run(self, programs: list, args=None) -> dict:
        m = self.machine
        m.load(programs, args).run()
        return {"instructions": m.icounts}

    def close(self) -> None:
        self.machine.close()


def bend_backend(threads: int | None = None):
    """Engine(..., backend=bend_backend()): a BendBackend per Engine."""
    return lambda cfg, images: BendBackend(cfg, images, threads)
