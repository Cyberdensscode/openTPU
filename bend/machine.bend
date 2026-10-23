# The openTPU machine (isasim.Machine): S slices that run in parallel between their
# collectives (BAR, GATHER), and the host's turn when every slice that can run waits on a
# WAITW. Pure; the MXU's MM is the template parameter ~mm, as in sim.bend.
import Base
import ./fp32.bend as F
import ./sim.bend as S
import ./word.bend as W

# The slices, the GATHER buffer, and the machine's state: st RUN 0, DONE 1 (every slice
# halted), HOST 2 (the host's turn), ERROR 3; e the machine's own error (9 a collective with
# a halted slice, 10 slices disagree on it, 11 GATHERs that disagree, 12 a GATHER's source
# and destination overlap, 13 a WAITW that never holds, 40 a deadlock) and es its slice
type Mc is Type:
  Mc{ss: List<S.Sl>, g: Array<U32>, st: U32, e: U32, es: U32}

def rev(xs: List<S.Sl>, ys: List<S.Sl>) -> List<S.Sl>:
  match xs:
    case Nil{}:
      ys
    case Con{s, rest}:
      rev(rest, Con{s, ys})

# word i of a slice's control array, and the slice
def peek.go(r: Array<U32> & U32, t: Array<U32>, a: Array<U32>, d: Array<U32>, h: Array<U32>,
  x: Array<U32>, +k: S.Cf) -> S.Sl & U32:
  (c, +v) = r
  (S.Sl{c, t, a, d, h, x, k}, v)

def peek(s: S.Sl, +i: U32) -> S.Sl & U32:
  match s:
    case S.Sl{c, t, a, d, h, x, k}:
      peek.go(Array.get(U32, c, i), t, a, d, h, x, k)

def poke(s: S.Sl, +i: U32, +v: U32) -> S.Sl:
  match s:
    case S.Sl{c, t, a, d, h, x, k}:
      S.Sl{Array.set(U32, c, i, v), t, a, d, h, x, k}

# ------------------------------------------------------------------------------- a round

# every slice runs until it stops; the slices in parallel
def run.one(~mm: S.Sl -> S.Ins -> S.Sl, s: S.Sl) -> S.Sl:
  S.go(~mm, U32.to_nat(4000000000), S.runs(poke(s, S.STEPS(), 0)))

def run.all(~mm: S.Sl -> S.Ins -> S.Sl, xs: List<S.Sl>) -> List<S.Sl>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{s, rest}:
      a b = run.one(~mm, s) run.all(~mm, rest)
      Con{a, b}

# What a round left: slices halted, waiting on a collective, polling a WAITW, stopped by an
# error (the first's index), and whether any instruction ran
type Sm is Data:
  Sm{n: U32, halt: U32, wait: U32, poll: U32, err: U32, es: U32, prog: U32}

def sm.add(+m: Sm, +st: U32, +steps: U32) -> Sm:
  match m:
    case Sm{+n, +halt, +wait, +poll, +err, +es, +prog}:
      Sm{(n + 1 : U32), (halt + F.b2u(U32.is_eq(st, 1)) : U32),
         (wait + F.b2u(U32.is_eq(st, 2)) : U32), (poll + F.b2u(U32.is_eq(st, 3)) : U32),
         (err + F.b2u(U32.is_eq(st, 4)) : U32),
         F.sel(U32.is_eq(err, 0) && U32.is_eq(st, 4), n, es), (prog + steps : U32)}

def scan.st(r: S.Sl & U32, ys: List<S.Sl>, +m: Sm, +st: U32) -> List<S.Sl> & Sm:
  (s, +steps) = r
  (Con{s, ys}, sm.add(m, st, steps))

def scan.go(r: S.Sl & U32, ys: List<S.Sl>, +m: Sm) -> List<S.Sl> & Sm:
  (s, +st) = r
  scan.st(peek(s, S.STEPS()), ys, m, st)

def scan.one(s: S.Sl, r: List<S.Sl> & Sm) -> List<S.Sl> & Sm:
  (ys, +m) = r
  scan.go(peek(s, S.STATE()), ys, m)

# the round's summary; the slices come back reversed
def scan(xs: List<S.Sl>, r: List<S.Sl> & Sm) -> List<S.Sl> & Sm:
  match xs:
    case Nil{}:
      r
    case Con{s, rest}:
      scan(rest, scan.one(s, r))

# ------------------------------------------------------------------------------- collectives

# A collective's fields on one slice (isasim Machine._gather): the opcode, src, dst,
# rows | cols << 16, srs, drs, seg
type Gk is Data:
  Gk{op: U32, src: U32, dst: U32, rc: U32, srs: U32, drs: U32, seg: U32}

def gk.i(r: Array<U32> & S.Ins, t: Array<U32>, a: Array<U32>, d: Array<U32>, h: Array<U32>,
  x: Array<U32>, +k: S.Cf) -> S.Sl & Gk:
  (c, +i) = r
  match i:
    case S.Ins{+w0, +w1, +w2, +w3, +w4, +w5, +w6, w7, +va, +vb, vc, vd}:
      (S.Sl{c, t, a, d, h, x, k}, Gk{S.op(w0), (va + w1 : U32), (vb + w2 : U32), w3, w4, w5, w6})

def gk.pc(r: Array<U32> & U32, t: Array<U32>, a: Array<U32>, d: Array<U32>, h: Array<U32>,
  x: Array<U32>, +k: S.Cf) -> S.Sl & Gk:
  (c, +pc) = r
  gk.i(S.fetch(c, pc), t, a, d, h, x, k)

# the collective a waiting slice stands on
def gk(s: S.Sl) -> S.Sl & Gk:
  match s:
    case S.Sl{c, t, a, d, h, x, k}:
      gk.pc(Array.get(U32, c, S.PC()), t, a, d, h, x, k)

# Whether the slices agree: the first slice's key, and the flags op (a different opcode) and
# key (a GATHER with another dst, rows, cols, drs or seg)
type Ga is Data:
  Ga{n: U32, op: U32, dst: U32, rc: U32, drs: U32, seg: U32, bop: U32, bkey: U32}

def ga.add(+m: Ga, +g: Gk) -> Ga:
  match m g:
    case Ga{+n, +op0, +dst0, +rc0, +drs0, +seg0, +bop, +bkey} Gk{+op, src, +dst, +rc, srs, +drs, +seg}:
      +first = U32.is_eq(n, 0)
      +diff = U32.is_ne(dst, dst0) || U32.is_ne(rc, rc0) || U32.is_ne(drs, drs0)
        || U32.is_ne(seg, seg0)
      Ga{(n + 1 : U32), F.sel(first, op, op0), F.sel(first, dst, dst0), F.sel(first, rc, rc0),
         F.sel(first, drs, drs0), F.sel(first, seg, seg0),
         (bop .|. F.b2u(Bool.not(first) && U32.is_ne(op, op0)) : U32),
         (bkey .|. F.b2u(Bool.not(first) && diff) : U32)}

def ga.st(r: S.Sl & Gk, ys: List<S.Sl>, +m: Ga) -> List<S.Sl> & Ga:
  (s, +g) = r
  (Con{s, ys}, ga.add(m, g))

def ga.one(s: S.Sl, r: List<S.Sl> & Ga) -> List<S.Sl> & Ga:
  (ys, +m) = r
  ga.st(gk(s), ys, m)

def ga(xs: List<S.Sl>, r: List<S.Sl> & Ga) -> List<S.Sl> & Ga:
  match xs:
    case Nil{}:
      r
    case Con{s, rest}:
      ga(rest, ga.one(s, r))

# rows of cols words: dst[j + r*js + q] = src[i + r*si + q]
def cp2(n: Nat, st: Array<U32> & Array<U32>, +i: U32, +si: U32, +j: U32, +js: U32,
  +cols: U32) -> Array<U32> & Array<U32>:
  match n:
    case 0n:
      st
    case 1n+p:
      cp2(p, S.cp(U32.to_nat(cols), st, i, j), (i + si : U32), si, (j + js : U32), js, cols)

# h[b + r*rs + q] = g for r < n, q < cols
def mark(n: Nat, h: Array<U32>, +b: U32, +rs: U32, +cols: U32, +g: U32) -> Array<U32>:
  match n:
    case 0n:
      h
    case 1n+p:
      mark(p, S.fill(U32.to_nat(cols), h, b, g), (b + rs : U32), rs, cols, g)

def seen.step(r: Array<U32> & U32, +f: U32, +g: U32) -> Array<U32> & U32:
  (h, +v) = r
  (h, (f .|. F.b2u(U32.is_eq(v, g)) : U32))

def seen.row(n: Nat, st: Array<U32> & U32, +b: U32, +g: U32) -> Array<U32> & U32:
  match n:
    case 0n:
      st
    case 1n+p:
      (h, +f) = st
      seen.row(p, seen.step(Array.get(U32, h, b), f, g), (b + 1 : U32), g)

# whether a word b + r*rs + q (r < n, q < cols) is stamped g
def seen(n: Nat, st: Array<U32> & U32, +b: U32, +rs: U32, +cols: U32, +g: U32) ->
  Array<U32> & U32:
  match n:
    case 0n:
      st
    case 1n+p:
      seen(p, seen.row(U32.to_nat(cols), st, b, g), (b + rs : U32), rs, cols, g)

# a GATHER's destination on one slice: its nsl segments, seg apart
def mark.segs(n: Nat, h: Array<U32>, +b: U32, +seg: U32, +rows: U32, +drs: U32, +cols: U32,
  +g: U32) -> Array<U32>:
  match n:
    case 0n:
      h
    case 1n+p:
      mark.segs(p, mark(U32.to_nat(rows), h, b, drs, cols, g), (b + seg : U32), seg, rows, drs,
        cols, g)

# The geometry every slice agreed on: dst, rows, cols, drs, seg, and the slice count
type Gg is Data:
  Gg{dst: U32, rows: U32, cols: U32, drs: U32, seg: U32, n: U32}

# a slice, the GATHER buffer and an error flag through phase A
type Gr is Type:
  Gr{s: S.Sl, g: Array<U32>, e: U32}

def gth.fin(r: Array<U32> & Array<U32>, c: Array<U32>, a: Array<U32>, d: Array<U32>,
  h: Array<U32>, x: Array<U32>, +k: S.Cf) -> Gr:
  (t, g) = r
  Gr{S.Sl{c, t, a, d, h, x, k}, g, 0}

# phase A on slice kk, its checks passed or not (e: the error): its source rows into g at
# kk * rows * cols
def gth.cp(z: Bool, s: S.Sl, g: Array<U32>, +src: U32, +srs: U32, +o: U32, +e: U32,
  +geo: Gg) -> Gr:
  match z s geo:
    case True{} S.Sl{c, t, a, d, h, x, k} Gg{dst, +rows, +cols, drs, seg, n}:
      gth.fin(cp2(U32.to_nat(rows), (t, g), src, srs, o, cols, cols), c, a, d, h, x, k)
    case False{} S.Sl{c, t, a, d, h, x, k} geo:
      Gr{S.Sl{S.fail(c, e, src), t, a, d, h, x, k}, g, 1}

def gth.ov(r: Array<U32> & U32, c: Array<U32>, t: Array<U32>, a: Array<U32>, d: Array<U32>,
  x: Array<U32>, +k: S.Cf, b: Array<U32>, +src: U32, +srs: U32, +o: U32, +geo: Gg) -> Gr:
  (h, +f) = r
  gth.cp(U32.is_eq(f, 0), S.Sl{c, t, a, d, h, x, k}, b, src, srs, o, 12, geo)

# the overlap: stamp the destination's words (a fresh VOP number), then look for the stamp
# under the source's
def gth.g(r: Array<U32> & U32, t: Array<U32>, a: Array<U32>, d: Array<U32>, h: Array<U32>,
  x: Array<U32>, +k: S.Cf, b: Array<U32>, +src: U32, +srs: U32, +o: U32, +geo: Gg) -> Gr:
  (c, +g0) = r
  match geo:
    case Gg{+dst, +rows, +cols, +drs, +seg, +n}:
      +g = (g0 + 1 : U32)
      c[37] <- g
      gth.ov(seen(U32.to_nat(rows), (mark.segs(U32.to_nat(n), h, dst, seg, rows, drs, cols, g), 0),
        src, srs, cols, g), c, t, a, d, x, k, b, src, srs, o, geo)

def gth.chk(z: Bool, s: S.Sl, b: Array<U32>, +src: U32, +srs: U32, +o: U32, +geo: Gg) -> Gr:
  match z s:
    case True{} S.Sl{c, t, a, d, h, x, k}:
      gth.g(Array.get(U32, c, S.VGEN()), t, a, d, h, x, k, b, src, srs, o, geo)
    case False{} s:
      gth.cp(False{}, s, b, src, srs, o, 7, geo)

# phase A on one slice: its GATHER's source within TMEM, the destination's segments too
def gA.k(r: S.Sl & Gk, b: Array<U32>, +kk: U32, +geo: Gg) -> Gr:
  (s, +q) = r
  match s q:
    case S.Sl{c, t, a, d, h, x, +k} Gk{op, +src, dst2, rc, +srs, drs2, seg2}:
      match geo:
        case Gg{+dst, +rows, +cols, +drs, +seg, +n}:
          +tl = (S.Cf.tw(k) - 1 : U32)
          +ok = U32.is_eq(rows, 0) || U32.is_eq(cols, 0)
            || ((src <= tl : U32) && W.rng((src + cols - 1 : U32), rows, srs, 1, 0, tl)
              && (dst <= tl : U32) && W.rng((dst + cols - 1 : U32), n, seg, rows, drs, tl))
          gth.chk(ok, S.Sl{c, t, a, d, h, x, k}, b, src, srs, (kk * rows * cols : U32), geo)

# the slices (reversed), the buffer, and whether a slice stopped with an error
type Gs is Type:
  Gs{ys: List<S.Sl>, b: Array<U32>, e: U32}

def gA.st(r: Gr, ys: List<S.Sl>, +e: U32) -> Gs:
  match r:
    case Gr{s, b, +e2}:
      Gs{Con{s, ys}, b, (e .|. e2 : U32)}

def gA.one(s: S.Sl, st: Gs, +kk: U32, +geo: Gg) -> Gs:
  match st:
    case Gs{ys, b, +e}:
      gA.st(gA.k(gk(s), b, kk, geo), ys, e)

def gA(xs: List<S.Sl>, st: Gs, +kk: U32, +geo: Gg) -> Gs:
  match xs:
    case Nil{}:
      st
    case Con{s, rest}:
      gA(rest, gA.one(s, st, kk, geo), (kk + 1 : U32), geo)

# phase B: every slice writes the nsl blocks: T[dst + kk*seg + r*drs + q] = the block of kk
def gB.blocks(n: Nat, st: Array<U32> & Array<U32>, +kk: U32, +geo: Gg) ->
  Array<U32> & Array<U32>:
  match n:
    case 0n:
      st
    case 1n+p:
      match geo:
        case Gg{+dst, +rows, +cols, +drs, +seg, +nn}:
          gB.blocks(p, cp2(U32.to_nat(rows), st, (kk * rows * cols : U32), cols,
            (dst + kk * seg : U32), drs, cols), (kk + 1 : U32), geo)

def gB.fin(r: Array<U32> & Array<U32>, ys: List<S.Sl>, c: Array<U32>, a: Array<U32>,
  d: Array<U32>, h: Array<U32>, x: Array<U32>, +k: S.Cf) -> List<S.Sl> & Array<U32>:
  (b, t) = r
  (Con{S.Sl{c, t, a, d, h, x, k}, ys}, b)

def gB.one(s: S.Sl, st: List<S.Sl> & Array<U32>, +geo: Gg) -> List<S.Sl> & Array<U32>:
  match s:
    case S.Sl{c, t, a, d, h, x, k}:
      (ys, b) = st
      match geo:
        case Gg{dst, rows, cols, drs, seg, +n}:
          gB.fin(gB.blocks(U32.to_nat(n), (b, t), 0, geo), ys, c, a, d, h, x, k)

def gB(xs: List<S.Sl>, st: List<S.Sl> & Array<U32>, +geo: Gg) -> List<S.Sl> & Array<U32>:
  match xs:
    case Nil{}:
      st
    case Con{s, rest}:
      gB(rest, gB.one(s, st, geo), geo)

# ------------------------------------------------------------------------------- the rounds

# the collective done: every slice RUN again, past it
def resume.one(s: S.Sl) -> S.Sl:
  match s:
    case S.Sl{c, t, a, d, h, x, k}:
      c[18] <- 0
      S.Sl{S.adv(c), t, a, d, h, x, k}

def resume(xs: List<S.Sl>) -> List<S.Sl>:
  match xs:
    case Nil{}:
      Nil{}
    case Con{s, rest}:
      Con{resume.one(s), resume(rest)}

# after the host's turn: each polling slice reads its word again; the count that now run
def repoll.go(z: Bool, s: S.Sl) -> S.Sl:
  match z:
    case True{}:
      S.ww.again(s)
    case False{}:
      s

def repoll.st(r: S.Sl & U32) -> S.Sl:
  (s, +st) = r
  repoll.go(U32.is_eq(st, 3), s)

def repoll.prog(r: S.Sl & U32, ys: List<S.Sl>, +n: U32) -> List<S.Sl> & U32:
  (s, +st) = r
  (Con{s, ys}, (n + F.b2u(U32.is_eq(st, 0)) : U32))

def repoll.one(s: S.Sl, r: List<S.Sl> & U32) -> List<S.Sl> & U32:
  (ys, +n) = r
  repoll.prog(peek(repoll.st(peek(s, S.STATE())), S.STATE()), ys, n)

def repoll(xs: List<S.Sl>, r: List<S.Sl> & U32) -> List<S.Sl> & U32:
  match xs:
    case Nil{}:
      r
    case Con{s, rest}:
      repoll(rest, repoll.one(s, r))

def mr.gB(r: List<S.Sl> & Array<U32>) -> Bool & Mc:
  (ys, b) = r
  (True{}, Mc{resume(rev(ys, Nil{})), b, 0, 0, 0})

def mr.gA2(z: Bool, xs: List<S.Sl>, b: Array<U32>, +geo: Gg) -> Bool & Mc:
  match z:
    case True{}:
      mr.gB(gB(xs, (Nil{}, b), geo))
    case False{}:
      (False{}, Mc{xs, b, 3, 0, 0})

def mr.gA(r: Gs, +geo: Gg) -> Bool & Mc:
  match r:
    case Gs{ys, b, +e}:
      mr.gA2(U32.is_eq(e, 0), rev(ys, Nil{}), b, geo)

def mr.big(z: Bool, xs: List<S.Sl>, g: Array<U32>, +geo: Gg) -> Bool & Mc:
  match z:
    case True{}:
      mr.gA(gA(xs, Gs{Nil{}, g, 0}, 0, geo), geo)
    case False{}:
      (False{}, Mc{xs, g, 3, 41, 0})

def mr.gath(z: Bool, xs: List<S.Sl>, g: Array<U32>, +fits: Bool, +geo: Gg) -> Bool & Mc:
  match z:
    case True{}:
      mr.big(fits, xs, g, geo)
    case False{}:
      (False{}, Mc{xs, g, 3, 11, 0})

def mr.bar(z: Bool, xs: List<S.Sl>, g: Array<U32>, +ok: Bool, +fits: Bool, +geo: Gg) ->
  Bool & Mc:
  match z:
    case True{}:
      (True{}, Mc{resume(xs), g, 0, 0, 0})
    case False{}:
      mr.gath(ok, xs, g, fits, geo)

def mr.op(z: Bool, xs: List<S.Sl>, g: Array<U32>, +bar: Bool, +ok: Bool, +fits: Bool,
  +geo: Gg) -> Bool & Mc:
  match z:
    case True{}:
      mr.bar(bar, xs, g, ok, fits, geo)
    case False{}:
      (False{}, Mc{xs, g, 3, 10, 0})

# every live slice waits on a collective: they must agree on it (BAR, or GATHERs of one
# geometry), then it runs (the GATHER's blocks fit the 2^22-word buffer, else error 41)
def mr.coll(r: List<S.Sl> & Ga, g: Array<U32>, +nsl: U32) -> Bool & Mc:
  (ys, +ga) = r
  match ga:
    case Ga{n, +op, +dst, +rc, +drs, +seg, +bop, +bkey}:
      +rows = (rc .&. 65535 : U32)
      +cols = (rc >> 16n : U32)
      +geo = {Gg{dst, rows, cols, drs, seg, nsl} : Gg}
      +fits = ((rows * cols : U32) <= U32.div(4194304, nsl) : U32)
      mr.op(U32.is_eq(bop, 0), rev(ys, Nil{}), g, U32.is_eq(op, 5), U32.is_eq(bkey, 0), fits, geo)

def mr.s6(z: Bool, xs: List<S.Sl>, g: Array<U32>) -> Bool & Mc:
  match z:
    case True{}:
      (False{}, Mc{xs, g, 2, 0, 0})
    case False{}:
      (False{}, Mc{xs, g, 3, 40, 0})

def mr.s5(z: Bool, xs: List<S.Sl>, g: Array<U32>, +poll: U32) -> Bool & Mc:
  match z:
    case True{}:
      (True{}, Mc{xs, g, 0, 0, 0})
    case False{}:
      mr.s6((poll > 0 : U32), xs, g)

def mr.s4(z: Bool, xs: List<S.Sl>, g: Array<U32>, +n: U32) -> Bool & Mc:
  match z:
    case True{}:
      (False{}, Mc{xs, g, 3, 9, 0})
    case False{}:
      mr.coll(ga(xs, (Nil{}, Ga{0, 0, 0, 0, 0, 0, 0, 0})), g, n)

def mr.s3(z: Bool, xs: List<S.Sl>, g: Array<U32>, +n: U32, +halt: U32, +poll: U32,
  +prog: U32) -> Bool & Mc:
  match z:
    case True{}:
      mr.s4((halt > 0 : U32), xs, g, n)
    case False{}:
      mr.s5((prog > 0 : U32), xs, g, poll)

def mr.s2(z: Bool, xs: List<S.Sl>, g: Array<U32>, +n: U32, +halt: U32, +poll: U32,
  +prog: U32, +coll: Bool) -> Bool & Mc:
  match z:
    case True{}:
      (False{}, Mc{xs, g, 1, 0, 0})
    case False{}:
      mr.s3(coll, xs, g, n, halt, poll, prog)

def mr.s1(z: Bool, xs: List<S.Sl>, g: Array<U32>, +n: U32, +halt: U32, +poll: U32,
  +prog: U32, +coll: Bool, +es: U32) -> Bool & Mc:
  match z:
    case True{}:
      (False{}, Mc{xs, g, 3, 0, es})
    case False{}:
      mr.s2(U32.is_eq(halt, n), xs, g, n, halt, poll, prog, coll)

# a round's outcome (isasim Machine.run's loop body): an error stops the machine; every slice
# halted: DONE; every live slice on a collective: run it; no instruction ran: the host's turn
# if a slice polls, else a deadlock
def mr.scan(r: List<S.Sl> & Sm, g: Array<U32>) -> Bool & Mc:
  (ys, +sm) = r
  match sm:
    case Sm{+n, +halt, +wait, +poll, +err, +es, +prog}:
      +live = (n - halt : U32)
      mr.s1((err > 0 : U32), rev(ys, Nil{}), g, n, halt, poll, prog,
        (live > 0 : U32) && U32.is_eq(wait, live), es)

def mround(~mm: S.Sl -> S.Ins -> S.Sl, m: Mc) -> Bool & Mc:
  match m:
    case Mc{ss, g, st, e, es}:
      mr.scan(scan(run.all(~mm, ss), (Nil{}, Sm{0, 0, 0, 0, 0, 0, 0})), g)

def mgo(~mm: S.Sl -> S.Ins -> S.Sl, n: Nat, r: Bool & Mc) -> Mc:
  match n:
    case 0n:
      (b, m) = r
      m
    case 1n+p:
      (b, m) = r
      match b:
        case True{}:
          mgo(~mm, p, mround(~mm, m))
        case False{}:
          m

def host.go(r: List<S.Sl> & U32, g: Array<U32>) -> Bool & Mc:
  (ys, +n) = r
  +go = (n > 0 : U32)
  (go, Mc{rev(ys, Nil{}), g, F.sel(go, 0, 3), F.sel(go, 0, 13), 0})

# after the host's turn: the polling slices read again; none holds: the WAITW's timeout
def host(m: Mc) -> Bool & Mc:
  match m:
    case Mc{ss, g, st, e, es}:
      host.go(repoll(ss, (Nil{}, 0)), g)

# a run until DONE, HOST or ERROR
def run(~mm: S.Sl -> S.Ins -> S.Sl, m: Mc) -> Mc:
  match m:
    case Mc{ss, g, st, e, es}:
      mgo(~mm, U32.to_nat(4000000000), (True{}, Mc{ss, g, 0, 0, 0}))

# the run goes on after the host's turn
def run.host(~mm: S.Sl -> S.Ins -> S.Sl, m: Mc) -> Mc:
  mgo(~mm, U32.to_nat(4000000000), host(m))
