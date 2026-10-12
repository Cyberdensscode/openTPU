"""Wait counts: the compiler's half of the sequencer's explicit dependencies (otpu_seq WAITC;
docs/isa.md "Wait counts", docs/seq_vpu.md).

With WAITC the sequencer computes no footprints and compares no ranges. Each instruction carries a
32-bit wait word beside it: seven 4-bit counts, k_j in bits [4j+3:4j]. k_j = k > 0 holds the
instruction until the k-th last instruction of unit U(j) before it (in program order) has
completed, and with it every older one of that unit (each unit completes in order); k = 0 waits
for nothing on U(j).
  j = 0..4: unit j (DMA, MXU, QUANT, VPU, COLL), before the instruction starts (an MM: before its
            release, when it reads the ACT RAM and writes TMEM);
  j = 5, 6: an MM's stream start (its DRAM reads): the DMA's and the quantizer's counts over DRAM
            conflicts only.
An instruction's own unit needs no count (each unit starts in order and, but the VPU, finishes an
instruction before the next touches memory); the VPU's own count is kept. A producer 16 or more
instructions of its unit back has completed: an incomplete one holds a window slot, and so does
each younger instruction of its unit, and the window has 16 slots. So 4 bits are exact.

wait_counts() computes the words from the program alone, conservatively: an abstract execution of
the program (registers as intervals, the run's arguments R8..R15 and every RLD unknown unless
given), each instruction's footprint the otpu_pkg footprint() of every value its registers may
hold, and per unit the last 15 instructions' footprints, joined where paths meet (a loop's exits,
a count the program does not know). A pc's count is the smallest over all its visits. Loops the
budget cannot run (or whose count is unknown beyond CAP iterations) run on widened registers to a
fixed point, which covers every later iteration.

verify() checks words against a concrete execution (the ISA simulator's): every conflict
otpu_pkg conflict() finds with an older instruction that may still be running is waited for.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import isa as I

M32 = 0xFFFFFFFF
TOP = (0, M32)
SP_TMEM, SP_DRAM, SP_ACT = 0, 1, 2
U_DMA, U_MXU, U_Q, U_VPU, U_COLL = 0, 1, 2, 3, 4
NUNITS, NW = 5, 7
WIN = 16
KMAX = WIN - 1                    # ranks a count names; older producers have completed
DEFAULT = 0x0111_1111             # a pc never reached: wait for every unit's last instruction
UNIT = {I.LD: U_DMA, I.ST: U_DMA, I.DSTEP: U_DMA, I.STREAM: U_DMA, I.WAITW: U_DMA,
        I.MM: U_MXU, I.QACT: U_Q, I.QST: U_Q, I.VOP: U_VPU,
        I.BAR: U_COLL, I.GATHER: U_COLL, I.RLD: U_COLL}
WUNIT = (U_DMA, U_MXU, U_Q, U_VPU, U_COLL, U_DMA, U_Q)
CAP = 4                           # iterations of an unknown count run before widening
BUDGET = 4_000_000                # instruction visits before known counts widen too
MAXR = 12                         # ranges a joined footprint keeps per (space, read / write)


# ------------------------------------------------------------------------- values (intervals)
def _add(v, imm):
    lo, hi = v[0] + (imm & M32), v[1] + (imm & M32)
    if (lo >> 32) != (hi >> 32):
        return TOP
    return (lo & M32, hi & M32)


def _mulc(v, c):
    """v * c mod 2^32 (RLD MUL with a constant factor)."""
    if v[0] == v[1]:
        x = (v[0] * c) & M32
        return (x, x)
    lo, hi = v[0] * c, v[1] * c
    return (lo, hi) if hi <= M32 else TOP


def _hull(a, b):
    return (min(a[0], b[0]), max(a[1], b[1]))


# ------------------------------------------------------------------------- footprints
@dataclass(frozen=True)
class Fp:
    """What an instruction may touch: half-open [lo, hi) ranges per space, read and written;
    all: conflicts with everything (BAR)."""
    all: bool = False
    rd: tuple = ()
    wr: tuple = ()


def _mk(sp, base, ln):
    """otpu_pkg mk() over every value of the base: [lo, hi) clamped to the last byte."""
    ln &= M32
    if ln == 0:
        return None
    return (sp, base[0], min(base[1] + ln, M32))


def _beats(r):
    if r is None:
        return None
    sp, lo, hi = r
    return (sp, lo & ~63, hi | 63)


def _p(a, w):
    """otpu_pkg fp_prod: the 24-bit a times the 32-bit w, low 32 bits."""
    return ((a & 0xFFFFFF) * (w & M32)) & M32


def resolve(ins: I.Instr, R) -> list:
    """The register-relative fields as otpu_seq resolves them (w1..w7, intervals)."""
    w = [(x & M32, x & M32) for x in ins.w]
    op = ins.op

    def r(k):
        return (0, 0) if k == 0 else R[k]

    def add(i, k):
        v = r(k)
        if v != (0, 0):
            w[i] = _add(v, w[i][0])

    if op == I.STREAM:
        add(1, ins.ra); add(2, ins.ra); add(3, ins.rb); add(4, ins.rc); add(5, ins.rd)
        return w
    add(0, ins.ra)
    if op != I.QACT:
        add(1, ins.rb)
    if op not in (I.QACT, I.LD, I.ST, I.GATHER):
        add(2, ins.rc)
    if op == I.VOP:
        add(6, ins.rd)
    return w


def footprint(ins: I.Instr, R, D: int = 128, S: int = 1) -> Fp:
    """otpu_pkg footprint() of the instruction for every value its registers may hold (R: 16
    intervals)."""
    op, fl = ins.op, ins.flags
    if op == I.BAR:
        return Fp(all=True)
    w = resolve(ins, R)
    c = [x[0] for x in w]          # immediates (the fields the RTL does not resolve)
    rd, wr = [], []
    if op == I.LD:
        rd.append(_mk(SP_DRAM, w[0], c[2] << 2)); wr.append(_mk(SP_TMEM, w[1], c[2]))
    elif op == I.ST:
        rd.append(_mk(SP_TMEM, w[1], c[2])); wr.append(_mk(SP_DRAM, w[0], c[2] << 2))
    elif op == I.STREAM:
        wr.append(_mk(SP_DRAM, w[1], 4 * 256 * 256)); wr.append(_mk(SP_DRAM, w[2], 4 * 256 * 256))
        wr.append(_mk(SP_TMEM, w[6], 256))
        rd.append(_mk(SP_TMEM, (c[0] & 0xFFFF, c[0] & 0xFFFF), 8))
        rd.append(_mk(SP_TMEM, w[3], 4 * 256)); rd.append(_mk(SP_TMEM, w[4], 256))
        rd.append(_mk(SP_TMEM, w[5], (c[0] >> 16) + 1))
    elif op == I.DSTEP:
        rows, cols = c[3] & 0xFFFF, c[3] >> 16
        if rows and cols:
            p0 = _p(rows - 1, cols)
            wr.append(_mk(SP_DRAM, w[0], (p0 + cols) << 2)); wr.append(_mk(SP_TMEM, w[5], rows))
            rd.append(_mk(SP_TMEM, w[1], cols << 1)); rd.append(_mk(SP_TMEM, w[2], rows))
            rd.append(_mk(SP_TMEM, w[4], c[6] + 1))
    elif op == I.MM:
        n, kb, m = c[3] & 0xFFFF, c[3] >> 16, (c[5] >> 16) & 0xFF
        if n and kb and m:
            p0, p1 = _p(n - 1, c[4]), _p(n - 1, c[6])
            p2, p3 = _p(m - 1, c[5] & 0xFFFF), _p(m, c[5] & 0xFFFF)
            rd.append(_mk(SP_DRAM, w[0], p0 + kb * D))
            if not fl & I.F_UNIT:
                rd.append(_mk(SP_DRAM, w[1], p1 + kb * 4))
            if fl & I.F_ASCALE:
                rd.append(_mk(SP_TMEM, w[1], m))
            ab = (c[5] >> 24) & 0xFF
            rd.append(_mk(SP_ACT, (ab, ab), kb))
            wr.append(_mk(SP_TMEM, w[2], p2 + n))
            if fl & I.F_RMAX:
                wr.append(_mk(SP_TMEM, _add(w[2], p3), m))
    elif op == I.QACT:
        rows, kb = c[1] & 0xFF, c[1] >> 16
        if rows and kb:
            rd.append(_mk(SP_TMEM, w[0], _p(rows - 1, c[2]) + kb * D))
            if fl & I.F_CSCALE:
                rd.append(_mk(SP_TMEM, w[3], kb * D))
            if fl & I.F_RSCALE:
                rd.append(_mk(SP_TMEM, w[4], rows))
            ab = (c[1] >> 8) & 0xFF
            wr.append(_mk(SP_ACT, (ab, ab), kb))
    elif op == I.QST:
        rows, kb = c[3] & 0xFFFF, c[3] >> 16
        if rows and kb:
            p0, p1 = _p(rows - 1, c[4]), _p(rows - 1, c[5])
            p2, p3 = _p(kb * D - 1, c[6]), _p(rows, kb)
            rd.append(_mk(SP_TMEM, w[0], p0 + kb * D))
            wr.append(_beats(_mk(SP_DRAM, w[1], p1 + p2 + 1)))
            wr.append(_beats(_mk(SP_DRAM, w[2], 4 * (rows if fl & I.F_ROW else p3))))
    elif op == I.VOP:
        rows, cols = c[3] & 0xFFFF, c[3] >> 16
        f, bm = (c[5] >> 16) & 0xFF, (c[5] >> 24) & 3
        if rows and cols:
            p0, p1, p2 = _p(rows - 1, c[4] >> 16), _p(rows - 1, c[5] & 0xFFFF), \
                _p(rows - 1, c[4] & 0xFFFF)
            if f == I.V_OUTER:
                if not fl & I.F_DONE:
                    rd.append(_mk(SP_TMEM, w[1], 1 if fl & I.F_DSCALAR else cols))
                rd.append(_mk(SP_TMEM, w[6], cols))
            elif f != I.V_FILL:
                rd.append(_mk(SP_TMEM, w[1], p0 + cols))
            if f in I.BINARY or f in (I.V_RDOT, I.V_OUTER):
                if bm == I.B_FULL:
                    rd.append(_mk(SP_TMEM, w[2], p1 + cols))
                elif bm == I.B_ROW:
                    rd.append(_mk(SP_TMEM, w[2], p1 + 1))
                elif bm == I.B_COL:
                    rd.append(_mk(SP_TMEM, w[2], cols))
            if f == I.V_ARGMAX:
                wr.append(_mk(SP_TMEM, w[0], p2 + 2))
            elif f in I.REDUCE:
                wr.append(_mk(SP_TMEM, w[0], p2 + 1))
            else:
                wr.append(_mk(SP_TMEM, w[0], p2 + cols))
    elif op == I.GATHER:
        rows, cols = c[2] & 0xFFFF, c[2] >> 16
        if rows and cols:
            rd.append(_mk(SP_TMEM, w[0], _p(rows - 1, c[3]) + cols))
            wr.append(_mk(SP_TMEM, w[1], _p(S - 1, c[5]) + _p(rows - 1, c[4]) + cols))
    elif op == I.RLD:
        rd.append(_mk(SP_TMEM, w[0], 1))
    elif op == I.WAITW:
        wr.append(_mk(SP_DRAM, (0, 0), M32)); wr.append(_mk(SP_TMEM, w[1], 1))
    return Fp(False, tuple(x for x in rd if x), tuple(x for x in wr if x))


def _ov(a, b):
    return a[0] == b[0] and a[1] < b[2] and b[1] < a[2]


def conflict(n: Fp, e: Fp, dram_only: bool = False) -> bool:
    """otpu_pkg conflict() (RAW, WAR, WAW); dram_only: conflict_dram()."""
    if n.all or e.all:
        return True
    nr, nw, er, ew = n.rd, n.wr, e.rd, e.wr
    if dram_only:
        nr = [x for x in nr if x[0] == SP_DRAM]; nw = [x for x in nw if x[0] == SP_DRAM]
        er = [x for x in er if x[0] == SP_DRAM]; ew = [x for x in ew if x[0] == SP_DRAM]
    for a in nw:
        for b in er:
            if _ov(a, b):
                return True
        for b in ew:
            if _ov(a, b):
                return True
    for a in ew:
        for b in nr:
            if _ov(a, b):
                return True
    return False


def _coalesce(rs):
    rs = tuple(sorted(set(rs)))
    if len(rs) <= MAXR:
        return rs
    out = []
    for sp in (SP_TMEM, SP_DRAM, SP_ACT):
        x = [r for r in rs if r[0] == sp]
        if x:
            out.append((sp, min(r[1] for r in x), max(r[2] for r in x)))
    return tuple(out)


def _fjoin(a: Fp, b: Fp) -> Fp:
    if a == b:
        return a
    return Fp(a.all or b.all, _coalesce(a.rd + b.rd), _coalesce(a.wr + b.wr))


# ------------------------------------------------------------------------- abstract state
class _State:
    """Registers (intervals) and, per unit, its last KMAX instructions' footprints (newest
    first)."""
    __slots__ = ("R", "H")

    def __init__(self, R, H):
        self.R, self.H = R, H

    def key(self):
        return (tuple(self.R), self.H)


def _join(a: _State | None, b: _State | None) -> _State | None:
    if a is None:
        return b
    if b is None:
        return a
    R = [x if x == y else _hull(x, y) for x, y in zip(a.R, b.R)]
    H = []
    for ha, hb in zip(a.H, b.H):
        if ha == hb:
            H.append(ha)
            continue
        n = max(len(ha), len(hb))
        h = []
        for i in range(n):
            if i >= len(ha):
                h.append(hb[i])
            elif i >= len(hb):
                h.append(ha[i])
            else:
                h.append(_fjoin(ha[i], hb[i]))
        H.append(tuple(h))
    return _State(R, tuple(H))


def _writes(prog, lo, hi) -> set:
    """Registers an instruction in [lo, hi) may write."""
    out = set()
    for ins in prog[lo:hi]:
        if ins.op in (I.LI, I.ADDI, I.RLD) and ins.rd:
            out.add(ins.rd)
    return out


class _Pass:
    def __init__(self, prog, D, S, rld, budget):
        self.prog, self.D, self.S = prog, D, S
        self.rld = rld or {}
        self.kmin = np.full((len(prog), NW), KMAX + 1, np.int64)
        self.seen = np.zeros(len(prog), bool)
        self.visits = 0
        self.budget = budget
        self.widened = []              # (pc of the LOOP, reason)

    # one instruction; returns the state after it (None: the program halted)
    def visit(self, pc, st: _State) -> _State | None:
        ins = self.prog[pc]
        op = ins.op
        self.visits += 1
        if op == I.HALT:
            return None
        if op in (I.NOP,):
            return st
        if op == I.LI:
            if ins.rd:
                R = list(st.R); R[ins.rd] = (ins.w[0] & M32,) * 2
                return _State(R, st.H)
            return st
        if op == I.ADDI:
            if ins.rd:
                R = list(st.R); R[ins.rd] = _add((0, 0) if ins.ra == 0 else st.R[ins.ra], ins.w[0])
                return _State(R, st.H)
            return st
        u = UNIT.get(op)
        if u is None:
            raise ValueError(f"waits: pc {pc}: opcode {op:#x}")
        fp = footprint(ins, st.R, self.D, self.S)
        self.seen[pc] = True
        km = self.kmin[pc]
        for j in range(NW):
            uj = WUNIT[j]
            if j >= NUNITS and op != I.MM:
                continue
            if uj == u and u != U_VPU:
                continue
            h = st.H[uj]
            for i, e in enumerate(h):
                if i + 1 >= km[j]:
                    break
                if conflict(fp, e, dram_only=j >= NUNITS):
                    km[j] = i + 1
                    break
        H = list(st.H)
        H[u] = (fp,) + H[u][:KMAX - 1]
        R = st.R
        if op == I.RLD and ins.rd:
            R = list(R)
            v = self.rld.get(pc, TOP)
            R[ins.rd] = v
        return _State(R, tuple(H))

    def block(self, lo, hi, st: _State | None) -> _State | None:
        pc = lo
        while pc < hi and st is not None:
            ins = self.prog[pc]
            if ins.op == I.LOOP:
                body = ins.w[0]
                st = self.loop(pc, pc + 1, pc + 1 + body, st)
                pc += 1 + body
            else:
                st = self.visit(pc, st)
                pc += 1
        return st

    def loop(self, at, lo, hi, st: _State) -> _State | None:
        ins = self.prog[at]
        self.visits += 1
        cnt = _add((0, 0) if ins.ra == 0 else st.R[ins.ra], ins.w[1])
        clo, chi = cnt
        exits = st if clo == 0 else None
        cap = CAP if clo != chi else chi
        if self.visits + cap * (hi - lo) > self.budget:
            cap = min(cap, CAP)
        s = st
        t = 0
        while t < min(chi, cap) and s is not None:
            prev = s
            s = self.block(lo, hi, s)
            t += 1
            if s is not None and t >= clo:
                exits = _join(exits, s)
            if s is not None and s.key() == prev.key():
                # a fixed point: every later iteration repeats this one
                if chi > t:
                    exits = _join(exits, s)
                return exits
        if s is not None and chi > t:
            self.widened.append((at, "unknown count" if clo != chi else "budget"))
            mod = _writes(self.prog, lo, hi)
            R = [TOP if k in mod else v for k, v in enumerate(s.R)]
            s = _State(R, s.H)
            for _ in range(4 * WIN):
                n = self.block(lo, hi, s)
                if n is None:
                    break
                n = _State([TOP if k in mod else v for k, v in enumerate(n.R)], n.H)
                j = _join(s, n)
                if j.key() == s.key():
                    break
                s = j
            else:
                raise RuntimeError(f"waits: the loop at pc {at} did not reach a fixed point")
            exits = _join(exits, s)
        return exits


def wait_counts(prog: list, D: int = 128, S: int = 1, args=None, rld: dict | None = None,
                budget: int = BUDGET, stats: dict | None = None) -> np.ndarray:
    """The wait words of a program (uint32 per instruction). args: the run's arguments if known
    (R8..R15; default unknown), rld: {pc: (lo, hi)} the values an RLD may load (default
    unknown). stats: filled with the visits and the loops run on widened registers."""
    R = [(0, 0)] * 8 + [TOP] * 8
    if args is not None:
        a = list(args) + [0] * (8 - len(args))
        R = [(0, 0)] * 8 + [(int(x) & M32,) * 2 for x in a[:8]]
    p = _Pass(prog, D, S, rld, budget)
    p.block(0, len(prog), _State(R, ((),) * NUNITS))
    k = np.where(p.kmin > KMAX, 0, p.kmin).astype(np.uint32)
    words = np.zeros(len(prog), np.uint32)
    for j in range(NW):
        words |= k[:, j] << np.uint32(4 * j)
    for pc, ins in enumerate(prog):
        if ins.op in UNIT and not p.seen[pc]:
            words[pc] = DEFAULT
    if stats is not None:
        stats.update(visits=p.visits, widened=p.widened, unreached=int(
            sum(1 for pc, ins in enumerate(prog) if ins.op in UNIT and not p.seen[pc])))
    return words


def counts(word: int) -> list:
    return [(int(word) >> (4 * j)) & 15 for j in range(NW)]


def sync_file(words, path) -> None:
    """otpu_seq's +hz_sync file: lines 'pc k0 .. k6 n' (n unused)."""
    with open(path, "w") as f:
        for pc, w in enumerate(words):
            f.write(f"{pc} " + " ".join(str(x) for x in counts(w)) + " 0\n")


# ------------------------------------------------------------------------- checking
class Checker:
    """Checks wait words on a concrete execution: feed it each unit instruction as it issues in
    program order (pc, the registers then). A conflict with an older instruction of another
    unit (or the VPU's own) within KMAX of its unit that the counts do not wait for raises."""

    def __init__(self, prog, words, D: int = 128, S: int = 1):
        self.prog, self.words, self.D, self.S = prog, words, D, S
        self.H = [[] for _ in range(NUNITS)]
        self.n = 0

    def issue(self, pc, regs) -> None:
        ins = self.prog[pc]
        u = UNIT.get(ins.op)
        if u is None:
            return
        R = [(int(x) & M32,) * 2 for x in regs]
        fp = footprint(ins, R, self.D, self.S)
        k = counts(self.words[pc])
        for j in range(NW):
            uj = WUNIT[j]
            if (j >= NUNITS and ins.op != I.MM) or (uj == u and u != U_VPU):
                continue
            for i, (e, epc) in enumerate(self.H[uj][:KMAX]):
                if conflict(fp, e, dram_only=j >= NUNITS):
                    if k[j] == 0 or k[j] > i + 1:
                        raise AssertionError(
                            f"waits: pc {pc} ({I.OPNAMES.get(ins.op)}) wait {j} k={k[j]} misses "
                            f"the {i + 1}-th last {('DMA', 'MXU', 'QUANT', 'VPU', 'COLL')[uj]} "
                            f"instruction (pc {epc})")
                    break
        self.H[u].insert(0, (fp, pc))
        del self.H[u][KMAX:]
        self.n += 1


def check_isasim(machine, words_per_slice, max_steps: int = 50_000_000) -> int:
    """Run an isasim.Machine to its halt, checking every slice's wait words on the way (no
    chains, no collectives across slices). Returns the instructions checked."""
    sl = machine.slices
    chk = [Checker(s.prog, w, machine.cfg.D, machine.cfg.S) for s, w in zip(sl, words_per_slice)]
    n = 0
    steps = 0
    while not all(s.halted for s in sl):
        progressed = False
        for s, c in zip(sl, chk):
            while not s.halted and s.waiting is None and s.polling is None:
                pc = s.pc
                if 0 <= pc < len(s.prog) and s.prog[pc].op in UNIT:
                    c.issue(pc, [s.reg(r) for r in range(16)])
                s.step()
                steps += 1
                progressed = True
                if steps > max_steps:
                    raise RuntimeError("check_isasim: step limit")
                if s.halted and s.chain is not None:
                    raise RuntimeError("check_isasim: HALT CHAIN")
        live = [s for s in sl if not s.halted]
        if live and all(s.waiting is not None for s in live):
            if any(s.waiting.op == I.GATHER for s in live):
                machine._gather()
            for s in live:
                s.waiting = None
                s.advance()
            progressed = True
        if not progressed:
            for s in sl:
                if s.polling is not None and s.poll(s.polling):
                    s.polling = None
                    s.advance()
                    progressed = True
        if not progressed:
            raise RuntimeError("check_isasim: deadlock")
    return sum(c.n for c in chk)
