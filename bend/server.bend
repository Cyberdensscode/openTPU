# The Bend ISA simulator as a server: opentpu/bend_sim.py starts it with a command pipe on
# fd 3 and a reply pipe on fd 4, and drives it as isasim's Machine (BendBackend: the Engine's
# write, read and run). Every command is 16 header words, some followed by data:
#   0 QUIT
#   1 INIT   S, D, MCOLS, ACT_BLOCKS, TMEM_WORDS, DRAM_BYTES - 1, IMEM_WORDS, ACT rows, the
#            features (PAIR 1, DSTEP 2, STREAM 4), and the depths (log2 words) of each slice's
#            control, TMEM, ACT and DRAM arrays, the ACT RAM's aso and ado (the first command)
#   2 DATA   slice, array (0 DRAM, 1 TMEM, 2 ACT, 3 control), direction (0 in, 1 out), byte
#            offset, bytes: the bytes follow (in) or are the reply (out)
#   3 LOAD   slice, instructions, the 8 run arguments; the program's words follow
#   4 RUN    the machine runs until every slice halts (DONE), the host's turn (HOST) or an
#            error; the reply: 16 words (state, the machine's error, its slice), then each
#            slice's control words 16..39 (pc, ICOUNT, state, error, ...: sim.bend). Word 1:
#            the MXU, 0 the parallel one (simpar.bend), 1 the sequential (sim.bend's mm.seq)
#   5 HOST   the host has had its turn: the polling slices read again, and the run goes on;
#            the reply as RUN's (word 1 as RUN's)
import Base
import ./fp32.bend as F
import ./sim.bend as S
import ./machine.bend as M
import ./simpar.bend as P

def Srv.hdr() -> IO(U32):
  import "./srv.c"

def Srv.arg(i: U32) -> IO(U32):
  import "./srv.c"

def Srv.mk(depth: U32) -> IO(Array<U32>):
  import "./srv.c"

def Srv.io(a: Array<U32>, dir: U32, off: U32, n: U32) -> IO(Array<U32>):
  import "./srv.c"

def Srv.put(i: U32, v: U32) -> IO(Unit):
  import "./srv.c"

def Srv.flush() -> IO(Unit):
  import "./srv.c"


# ------------------------------------------------------------------------------- slices

def mk.sl(+k: S.Cf, +cd: U32, +td: U32, +ad: U32, +dd: U32) -> IO(S.Sl):
  do IO<S.Sl>:
    c : Array<U32> <- Srv.mk(cd)
    t : Array<U32> <- Srv.mk(td)
    a : Array<U32> <- Srv.mk(ad)
    d : Array<U32> <- Srv.mk(dd)
    h : Array<U32> <- Srv.mk(td)
    x : Array<U32> <- Srv.mk(22)
    return S.Sl{c, t, a, d, h, x, k}

def mk.all(n: Nat, +j: U32, +k: S.Cf, +cd: U32, +td: U32, +ad: U32, +dd: U32) ->
  IO(List<S.Sl>):
  match n:
    case 0n:
      IO.pure(List<S.Sl>, Nil{})
    case 1n+p:
      do IO<List<S.Sl>>:
        s : S.Sl <- mk.sl(S.Cf.with_sid(k, j), cd, td, ad, dd)
        rest : List<S.Sl> <- mk.all(p, (j + 1 : U32), k, cd, td, ad, dd)
        return Con{s, rest}

# one of a slice's arrays in from or out to the host
def sl.io.a(z: Bool, s: S.Sl, +dir: U32, +off: U32, +n: U32) -> IO(S.Sl):
  match z s:
    case True{} S.Sl{c, t, a, d, h, x, k}:
      do IO<S.Sl>:
        a2 : Array<U32> <- Srv.io(a, dir, off, n)
        return S.Sl{c, t, a2, d, h, x, k}
    case False{} S.Sl{c, t, a, d, h, x, k}:
      do IO<S.Sl>:
        c2 : Array<U32> <- Srv.io(c, dir, off, n)
        return S.Sl{c2, t, a, d, h, x, k}

def sl.io.t(z: Bool, s: S.Sl, +w: U32, +dir: U32, +off: U32, +n: U32) -> IO(S.Sl):
  match z s:
    case True{} S.Sl{c, t, a, d, h, x, k}:
      do IO<S.Sl>:
        t2 : Array<U32> <- Srv.io(t, dir, off, n)
        return S.Sl{c, t2, a, d, h, x, k}
    case False{} s:
      sl.io.a(U32.is_eq(w, 2), s, dir, off, n)

def sl.io.d(z: Bool, s: S.Sl, +w: U32, +dir: U32, +off: U32, +n: U32) -> IO(S.Sl):
  match z s:
    case True{} S.Sl{c, t, a, d, h, x, k}:
      do IO<S.Sl>:
        d2 : Array<U32> <- Srv.io(d, dir, off, n)
        return S.Sl{c, t, a, d2, h, x, k}
    case False{} s:
      sl.io.t(U32.is_eq(w, 1), s, w, dir, off, n)

def sl.io(s: S.Sl, +w: U32, +dir: U32, +off: U32, +n: U32) -> IO(S.Sl):
  sl.io.d(U32.is_eq(w, 0), s, w, dir, off, n)

# ------------------------------------------------------------------------------- one slice

# the slices before slice j (reversed) and from it on
def split(n: Nat, xs: List<S.Sl>, ys: List<S.Sl>) -> List<S.Sl> & List<S.Sl>:
  match n:
    case 0n:
      (ys, xs)
    case 1n+p:
      match xs:
        case Nil{}:
          (ys, Nil{})
        case Con{s, rest}:
          split(p, rest, Con{s, ys})

# DATA: one of slice j's arrays in or out
def data.at(r: List<S.Sl> & List<S.Sl>, +w: U32, +dir: U32, +off: U32, +n: U32) ->
  IO(List<S.Sl>):
  (pre, post) = r
  match post:
    case Nil{}:
      IO.pure(List<S.Sl>, M.rev(pre, Nil{}))
    case Con{s, rest}:
      do IO<List<S.Sl>>:
        s2 : S.Sl <- sl.io(s, w, dir, off, n)
        return M.rev(pre, Con{s2, rest})

def data(m: M.Mc) -> IO(M.Mc):
  match m:
    case M.Mc{ss, g, +st, +e, +es}:
      do IO<M.Mc>:
        +j : U32 <- Srv.arg(1)
        +w : U32 <- Srv.arg(2)
        +dir : U32 <- Srv.arg(3)
        +off : U32 <- Srv.arg(4)
        +n : U32 <- Srv.arg(5)
        ys : List<S.Sl> <- data.at(split(U32.to_nat(j), ss, Nil{}), w, dir, off, n)
        return M.Mc{ys, g, st, e, es}

# LOAD: slice j's program (n instructions of 8 words into IMEM) and its run's arguments
def load.sl(s: S.Sl, +n: U32, +a0: U32, +a1: U32, +a2: U32, +a3: U32, +a4: U32, +a5: U32,
  +a6: U32, +a7: U32) -> IO(S.Sl):
  do IO<S.Sl>:
    s2 : S.Sl <- sl.io(s, 3, 0, (S.IMEM() * 4 : U32), (n * 32 : U32))
    return S.load(s2, n, a0, a1, a2, a3, a4, a5, a6, a7)

def load.at(r: List<S.Sl> & List<S.Sl>, +n: U32, +a0: U32, +a1: U32, +a2: U32, +a3: U32,
  +a4: U32, +a5: U32, +a6: U32, +a7: U32) -> IO(List<S.Sl>):
  (pre, post) = r
  match post:
    case Nil{}:
      IO.pure(List<S.Sl>, M.rev(pre, Nil{}))
    case Con{s, rest}:
      do IO<List<S.Sl>>:
        s2 : S.Sl <- load.sl(s, n, a0, a1, a2, a3, a4, a5, a6, a7)
        return M.rev(pre, Con{s2, rest})

def load(m: M.Mc) -> IO(M.Mc):
  match m:
    case M.Mc{ss, g, +st, +e, +es}:
      do IO<M.Mc>:
        +j : U32 <- Srv.arg(1)
        +n : U32 <- Srv.arg(2)
        +a0 : U32 <- Srv.arg(3)
        +a1 : U32 <- Srv.arg(4)
        +a2 : U32 <- Srv.arg(5)
        +a3 : U32 <- Srv.arg(6)
        +a4 : U32 <- Srv.arg(7)
        +a5 : U32 <- Srv.arg(8)
        +a6 : U32 <- Srv.arg(9)
        +a7 : U32 <- Srv.arg(10)
        ys : List<S.Sl> <- load.at(split(U32.to_nat(j), ss, Nil{}), n, a0, a1, a2, a3, a4, a5, a6, a7)
        return M.Mc{ys, g, st, e, es}

# ------------------------------------------------------------------------------- RUN, HOST

# the slices and how many
def len(xs: List<S.Sl>, ys: List<S.Sl>, +n: U32) -> List<S.Sl> & U32:
  match xs:
    case Nil{}:
      (M.rev(ys, Nil{}), n)
    case Con{s, rest}:
      len(rest, Con{s, ys}, (n + 1 : U32))

# each slice's control words 16..39 to the host
def snd(xs: List<S.Sl>, ys: List<S.Sl>) -> IO(List<S.Sl>):
  match xs:
    case Nil{}:
      IO.pure(List<S.Sl>, M.rev(ys, Nil{}))
    case Con{s, rest}:
      do IO<List<S.Sl>>:
        s2 : S.Sl <- sl.io(s, 3, 1, 64, 96)
        zs : List<S.Sl> <- snd(rest, Con{s2, ys})
        return zs

def reply.go(r: List<S.Sl> & U32, g: Array<U32>, +st: U32, +e: U32, +es: U32) -> IO(M.Mc):
  (ss, +n) = r
  do IO<M.Mc>:
    Srv.put(0, st)
    Srv.put(1, e)
    Srv.put(2, es)
    Srv.put(3, n)
    Srv.flush()
    ys : List<S.Sl> <- snd(ss, Nil{})
    return M.Mc{ys, g, st, e, es}

def reply(m: M.Mc) -> IO(M.Mc):
  match m:
    case M.Mc{ss, g, +st, +e, +es}:
      reply.go(len(ss, Nil{}, 0), g, st, e, es)

# ------------------------------------------------------------------------------- the loop

# RUN (host 0) or HOST (host 1) with the sequential MXU (seq) or the parallel one
def cmd.go(seq: Bool, host: Bool, m: M.Mc) -> IO(M.Mc):
  match seq host:
    case True{} True{}:
      reply(M.run.host(~(s => i => S.mm.seq(s, i)), m))
    case True{} False{}:
      reply(M.run(~(s => i => S.mm.seq(s, i)), m))
    case False{} True{}:
      reply(M.run.host(~(s => i => P.mm.par(s, i)), m))
    case False{} False{}:
      reply(M.run(~(s => i => P.mm.par(s, i)), m))

def cmd.rh(m: M.Mc, +op: U32) -> IO(M.Mc):
  do IO<M.Mc>:
    +q : U32 <- Srv.arg(1)
    cmd.go(U32.is_eq(q, 1), U32.is_eq(op, 5), m)

def cmd.h(z: Bool, m: M.Mc) -> IO(M.Mc):
  match z:
    case True{}:
      cmd.rh(m, 5)
    case False{}:
      IO.pure(M.Mc, m)

def cmd.r(z: Bool, m: M.Mc, +op: U32) -> IO(M.Mc):
  match z:
    case True{}:
      cmd.rh(m, 4)
    case False{}:
      cmd.h(U32.is_eq(op, 5), m)

def cmd.l(z: Bool, m: M.Mc, +op: U32) -> IO(M.Mc):
  match z:
    case True{}:
      load(m)
    case False{}:
      cmd.r(U32.is_eq(op, 4), m, op)

def cmd.d(z: Bool, m: M.Mc, +op: U32) -> IO(M.Mc):
  match z:
    case True{}:
      data(m)
    case False{}:
      cmd.l(U32.is_eq(op, 3), m, op)

def cmd(m: M.Mc, +op: U32) -> IO(M.Mc):
  cmd.d(U32.is_eq(op, 2), m, op)

def serve.go(n: Nat, z: Bool, m: M.Mc, +op: U32) -> IO(Unit):
  match n:
    case 0n:
      IO.exit(5)
    case 1n+p:
      match z:
        case True{}:
          IO.exit(0)
        case False{}:
          do IO<Unit>:
            m2 : M.Mc <- cmd(m, op)
            +op2 : U32 <- Srv.hdr()
            serve.go(p, U32.is_eq(op2, 0), m2, op2)

# INIT: the machine (S slices of the geometry the header gives)
def init() -> IO(M.Mc):
  do IO<M.Mc>:
    +ns : U32 <- Srv.arg(1)
    +d : U32 <- Srv.arg(2)
    +mc : U32 <- Srv.arg(3)
    +ab : U32 <- Srv.arg(4)
    +tw : U32 <- Srv.arg(5)
    +db : U32 <- Srv.arg(6)
    +iw : U32 <- Srv.arg(7)
    +ar : U32 <- Srv.arg(8)
    +fe : U32 <- Srv.arg(9)
    +cd : U32 <- Srv.arg(10)
    +td : U32 <- Srv.arg(11)
    +ad : U32 <- Srv.arg(12)
    +dd : U32 <- Srv.arg(13)
    +ao : U32 <- Srv.arg(14)
    +ado : U32 <- Srv.arg(15)
    ss : List<S.Sl> <- mk.all(U32.to_nat(ns), 0,
      S.Cf{ns, d, mc, ab, tw, db, iw, ar, (fe .&. 1 : U32), ((fe >> 1n : U32) .&. 1 : U32),
        ((fe >> 2n : U32) .&. 1 : U32), 0, ao, ado}, cd, td, ad, dd)
    g : Array<U32> <- Srv.mk(22)
    return M.Mc{ss, g, 0, 0, 0}

def main() -> IO(Unit):
  do IO<Unit>:
    op : U32 <- Srv.hdr()
    m : M.Mc <- init()
    Srv.put(0, 1)
    Srv.flush()
    +op2 : U32 <- Srv.hdr()
    serve.go(U32.to_nat(4000000000), U32.is_eq(op2, 0), m, op2)
