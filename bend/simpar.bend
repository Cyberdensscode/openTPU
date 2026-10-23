# The MXU with its rows and each row's outputs in parallel (the server's MM; sim.bend's mm.seq
# computes them in order). The M rows, then row j's N outputs, split in halves, fork-join,
# while a half holds LEAF output blocks (outputs x K-blocks: LEAF * 128 multiply-adds) or more:
# waking a thread costs about as much as a small MM, so a decode step's small MMs stay whole,
# and the LM head's and the prefill's big ones split. Every output is the same sum, in the same
# order, as mm.seq's, so the results are bit for bit the same (tests/test_bend.py runs both).
#
# The halves share the slice's arrays: DRAM, the ACT RAM and TMEM are only read while the
# outputs are computed, and the scratch takes each output at its own word (j * N + n). Bend's
# arrays are affine, so the sharing goes through Array.fork / Array.join (Base, @unsafe: two
# handles to one array). That makes this module unsafe in the checker's sense; sim.bend, which
# the laws and proofs (LAWS.bend) are about, stays pure.
import Base
import ./fp32.bend as F
import ./sim.bend as S

# output blocks (outputs x K-blocks) a task takes at least
def LEAF() -> U32:
  4096

# the output blocks of cnt outputs
def par.work(+o: S.Mk, +cnt: U32) -> U32:
  match o:
    case S.Mk{sa, rs, ssa, srs, out, ors, n, +kb, k0, m, ab, f, wf, dd, nab, ado, aso, msk, tm}:
      (cnt * kb : U32)

def par.f2(rx: Array<U32> & Array<U32>, d1: Array<U32>, d2: Array<U32>, a1: Array<U32>,
  a2: Array<U32>, t1: Array<U32>, t2: Array<U32>) -> S.Mx & S.Mx:
  (x1, x2) = rx
  (S.Mx{d1, a1, t1, x1}, S.Mx{d2, a2, t2, x2})

def par.f1(rd: Array<U32> & Array<U32>, ra: Array<U32> & Array<U32>,
  rt: Array<U32> & Array<U32>, x: Array<U32>) -> S.Mx & S.Mx:
  (d1, d2) = rd
  (a1, a2) = ra
  (t1, t2) = rt
  par.f2(Array.fork(U32, x), d1, d2, a1, a2, t1, t2)

# two handles to each of the four arrays
def par.fork(st: S.Mx) -> S.Mx & S.Mx:
  match st:
    case S.Mx{d, a, t, x}:
      par.f1(Array.fork(U32, d), Array.fork(U32, a), Array.fork(U32, t), x)

def par.join(m1: S.Mx, m2: S.Mx) -> S.Mx:
  match m1 m2:
    case S.Mx{d1, a1, t1, x1} S.Mx{d2, a2, t2, x2}:
      S.Mx{Array.join(U32, d1, d2), Array.join(U32, a1, a2), Array.join(U32, t1, t2),
        Array.join(U32, x1, x2)}

# outputs nn .. nn + cnt of row j: two halves at once while each holds LEAF blocks
# (@unsafe: it and par.two call each other)
@unsafe
def par.ns(+dep: Nat, st: S.Mx, +o: S.Mk, +j: U32, +nn: U32, +cnt: U32, +sj: U32,
  +alpha: U32) -> S.Mx:
  match dep:
    case 0n:
      S.mm.ns(U32.to_nat(cnt), st, o, j, nn, sj, alpha)
    case 1n+p:
      par.sp(p, (par.work(o, cnt) >= (2 * LEAF() : U32) : U32), st, o, j, nn, cnt, sj, alpha)

def par.two(+p: Nat, r: S.Mx & S.Mx, +o: S.Mk, +j: U32, +nn: U32, +cnt: U32, +sj: U32,
  +alpha: U32) -> S.Mx:
  (m1, m2) = r
  +h = (cnt >> 1n : U32)
  y1 y2 = par.ns(p, m1, o, j, nn, h, sj, alpha) par.ns(p, m2, o, j, (nn + h : U32),
    (cnt - h : U32), sj, alpha)
  par.join(y1, y2)

def par.sp(+p: Nat, z: Bool, st: S.Mx, +o: S.Mk, +j: U32, +nn: U32, +cnt: U32, +sj: U32,
  +alpha: U32) -> S.Mx:
  match z:
    case True{}:
      par.two(p, par.fork(st), o, j, nn, cnt, sj, alpha)
    case False{}:
      S.mm.ns(U32.to_nat(cnt), st, o, j, nn, sj, alpha)

# the output blocks of rows j .. j + cnt (each N outputs)
def par.rwork(+o: S.Mk, +cnt: U32) -> U32:
  match o:
    case S.Mk{sa, rs, ssa, srs, out, ors, +n, +kb, k0, m, ab, f, wf, dd, nab, ado, aso, msk, tm}:
      (cnt * n * kb : U32)

# rows j .. j + cnt: two halves at once while each holds LEAF blocks, each row's outputs by
# par.ns (@unsafe: it and par.jtwo call each other)
@unsafe
def par.js(+dep: Nat, st: S.Mx, +o: S.Mk, +j: U32, +cnt: U32) -> S.Mx:
  match dep:
    case 0n:
      S.mm.js(~(n => s2 => o2 => jj => sj => al => par.ns(10n, s2, o2, jj, 0, n, sj, al)),
        U32.to_nat(cnt), st, o, j)
    case 1n+p:
      par.jsp(p, (cnt >= 2 : U32) && (par.rwork(o, cnt) >= (2 * LEAF() : U32) : U32), st, o, j,
        cnt)

def par.jtwo(+p: Nat, r: S.Mx & S.Mx, +o: S.Mk, +j: U32, +cnt: U32) -> S.Mx:
  (m1, m2) = r
  +h = (cnt >> 1n : U32)
  y1 y2 = par.js(p, m1, o, j, h) par.js(p, m2, o, (j + h : U32), (cnt - h : U32))
  par.join(y1, y2)

def par.jsp(+p: Nat, z: Bool, st: S.Mx, +o: S.Mk, +j: U32, +cnt: U32) -> S.Mx:
  match z:
    case True{}:
      par.jtwo(p, par.fork(st), o, j, cnt)
    case False{}:
      S.mm.js(~(n => s2 => o2 => jj => sj => al => par.ns(10n, s2, o2, jj, 0, n, sj, al)),
        U32.to_nat(cnt), st, o, j)

# the M rows
def par.rows(st: S.Mx, +o: S.Mk) -> S.Mx:
  match o:
    case S.Mk{sa, rs, ssa, srs, out, ors, n, kb, k0, +m, ab, f, wf, dd, nab, ado, aso, msk, tm}:
      par.js(10n, st, o, 0, m)

# MM, the parallel MXU (the server's ~mm)
def mm.par(s: S.Sl, +i: S.Ins) -> S.Sl:
  S.mm.ck(~(st => o => par.rows(st, o)), s, i)
