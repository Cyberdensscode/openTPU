# The openTPU ISA in Bend (docs/isa.md): a slice's machine state and its instructions, bit for
# bit the ISA simulator opentpu/isasim.py (tests/test_bend.py compares the two).
#
# Everything here is pure and terminating: LAWS.bend states facts about it and PROOF.bend
# proves them. The MXU's MM is a template parameter (~mm) of the step and the run loops:
# mm.seq below is the sequential one; simpar.bend's splits the output columns over the cores
# (it shares the read-only arrays through Base's @unsafe Array.fork, so it stays out of the
# proofs).
#
# Bend has no hex literals: the constants are decimal, their hex in the comments.
import Base
import ./fp32.bend as F

# ------------------------------------------------------------------------------- the state

# The configuration (isasim.Config) of one slice: S slices, D the MXU depth, MCOLS columns, ab
# ACT RAM blocks, tw TMEM words, dbm1 the last DRAM byte (DRAM_BYTES - 1), iw IMEM words, ar
# the ACT RAM rows, the PAIR / DSTEP / STREAM features (0 or 1), sid the slice's index, and
# where the ACT RAM's scale record starts in a: Sa per (row, block) from aso, the offsets d per
# (row, block, sub-block) from ado.
type Cf is Data:
  Cf{s: U32, d: U32, mcols: U32, ab: U32, tw: U32, dbm1: U32, iw: U32, ar: U32, pair: U32,
     dstep: U32, stream: U32, sid: U32, aso: U32, ado: U32}

# A slice: six word arrays and its configuration.
#   c  control: R0..R15 (words 0..15), the status words below, the loop stack, the run's
#      arguments, and IMEM from word 4096 (8 words an instruction)
#   t  TMEM, an fp32 word per address
#   a  the ACT RAM: int8 elements four to a word, row-major [rows][blocks * D], then the
#      scale record (Cf aso, ado)
#   d  DRAM, little-endian bytes four to a word
#   h  the VPU's hazard stamps, one per TMEM word (the VOP that last wrote it)
#   x  scratch: an MM's outputs, a row being quantized, OUTER's vectors, the reductions'
#      partial sums
type Sl is Type:
  Sl{c: Array<U32>, t: Array<U32>, a: Array<U32>, d: Array<U32>, h: Array<U32>,
     x: Array<U32>, k: Cf}

# the control words
def PC() -> U32:
  16

# instructions executed (ICOUNT)
def ICOUNT() -> U32:
  17

# RUN 0, HALT 1, WAIT 2 (on a collective: BAR, GATHER), POLL 3 (a WAITW that does not hold
# yet), ERROR 4
def STATE() -> U32:
  18

# the error's code (opentpu/bend_sim.py has the messages) and a detail word
def ERR() -> U32:
  19

def AUX() -> U32:
  20

# the program's length in instructions
def PLEN() -> U32:
  21

# loop stack depth; entry j at 23 + 3j: (start, end, count)
def SP() -> U32:
  22

# WAITWs that had to wait, HALT CHAINs taken, the VOP count (the hazard stamps' generation),
# instructions executed in this round of the machine
def STALLS() -> U32:
  35

def CHAINS() -> U32:
  36

def VGEN() -> U32:
  37

def STEPS() -> U32:
  38

# the run's arguments: R8..R15 at the start of every program (8 words)
def ARGS() -> U32:
  40

def IMEM() -> U32:
  4096

def Cf.s(+k: Cf) -> U32:
  match k:
    case Cf{+s, d, mc, ab, tw, db, iw, ar, pr, ds, st, si, ao, ad}:
      s

def Cf.d(+k: Cf) -> U32:
  match k:
    case Cf{s, +d, mc, ab, tw, db, iw, ar, pr, ds, st, si, ao, ad}:
      d

def Cf.mcols(+k: Cf) -> U32:
  match k:
    case Cf{s, d, +mc, ab, tw, db, iw, ar, pr, ds, st, si, ao, ad}:
      mc

def Cf.ab(+k: Cf) -> U32:
  match k:
    case Cf{s, d, mc, +ab, tw, db, iw, ar, pr, ds, st, si, ao, ad}:
      ab

def Cf.tw(+k: Cf) -> U32:
  match k:
    case Cf{s, d, mc, ab, +tw, db, iw, ar, pr, ds, st, si, ao, ad}:
      tw

def Cf.dbm1(+k: Cf) -> U32:
  match k:
    case Cf{s, d, mc, ab, tw, +db, iw, ar, pr, ds, st, si, ao, ad}:
      db

def Cf.iw(+k: Cf) -> U32:
  match k:
    case Cf{s, d, mc, ab, tw, db, +iw, ar, pr, ds, st, si, ao, ad}:
      iw

def Cf.ar(+k: Cf) -> U32:
  match k:
    case Cf{s, d, mc, ab, tw, db, iw, +ar, pr, ds, st, si, ao, ad}:
      ar

def Cf.pair(+k: Cf) -> U32:
  match k:
    case Cf{s, d, mc, ab, tw, db, iw, ar, +pr, ds, st, si, ao, ad}:
      pr

def Cf.dstep(+k: Cf) -> U32:
  match k:
    case Cf{s, d, mc, ab, tw, db, iw, ar, pr, +ds, st, si, ao, ad}:
      ds

def Cf.stream(+k: Cf) -> U32:
  match k:
    case Cf{s, d, mc, ab, tw, db, iw, ar, pr, ds, +st, si, ao, ad}:
      st

def Cf.sid(+k: Cf) -> U32:
  match k:
    case Cf{s, d, mc, ab, tw, db, iw, ar, pr, ds, st, +si, ao, ad}:
      si

def Cf.aso(+k: Cf) -> U32:
  match k:
    case Cf{s, d, mc, ab, tw, db, iw, ar, pr, ds, st, si, +ao, ad}:
      ao

def Cf.ado(+k: Cf) -> U32:
  match k:
    case Cf{s, d, mc, ab, tw, db, iw, ar, pr, ds, st, si, ao, +ad}:
      ad

# ------------------------------------------------------------------------------- words

# the high word of the 64-bit product a * b
def mulhi(+a: U32, +b: U32) -> U32:
  +al = (a .&. 65535 : U32)
  +ah = (a >> 16n : U32)
  +bl = (b .&. 65535 : U32)
  +bh = (b >> 16n : U32)
  +lh = (al * bh : U32)
  +hl = (ah * bl : U32)
  +mid = (((al * bl : U32) >> 16n : U32) + (lh .&. 65535 : U32) + (hl .&. 65535 : U32) : U32)
  ((ah * bh : U32) + (lh >> 16n : U32) + (hl >> 16n : U32) + (mid >> 16n : U32) : U32)

# whether b + (n1 - 1) s1 + (n2 - 1) s2 <= last, exactly (no 32-bit wrap); n1, n2 >= 1. The
# bound of every strided range: TMEM (last = TW - 1), DRAM bytes (DRAM_BYTES - 1)
def rng(+b: U32, +n1: U32, +s1: U32, +n2: U32, +s2: U32, +last: U32) -> Bool:
  +p1 = ((n1 - 1 : U32) * s1 : U32)
  +p2 = ((n2 - 1 : U32) * s2 : U32)
  +u = (b + p1 : U32)
  +v = (u + p2 : U32)
  +hi = (mulhi((n1 - 1 : U32), s1) + mulhi((n2 - 1 : U32), s2) + F.b2u((u < b : U32))
    + F.b2u((v < u : U32)) : U32)
  U32.is_eq(hi, 0) && (v <= last : U32)

# an arithmetic shift right by k in 1..32 (32: the sign word)
def sar(+v: U32, +k: U32) -> U32:
  F.sel((v >= 2147483648 : U32), (F.shr((v .^. 4294967295 : U32), k) .^. 4294967295 : U32),
    F.shr(v, k))

# a byte's value as a signed word (b in 0..255)
def sx8(+b: U32) -> U32:
  ((b .^. 128 : U32) - 128 : U32)

# byte i (0..3) of the word w
def byte(+w: U32, +i: U32) -> U32:
  ((w >> U32.to_nat((i * 8 : U32)) : U32) .&. 255 : U32)

# w with byte i (0..3) replaced by the low byte of v
def put8(+w: U32, +i: U32, +v: U32) -> U32:
  +sh = U32.to_nat((i * 8 : U32))
  ((w .&. ((255 << sh : U32) .^. 4294967295 : U32) : U32) .|. ((v .&. 255 : U32) << sh : U32)
    : U32)

# ------------------------------------------------------------------------------- arrays

# a[i] += k
def bump.go(r: Array<U32> & U32, +i: U32, +k: U32) -> Array<U32>:
  (c, v) = r
  c[i] <- (v + k : U32)

def bump(c: Array<U32>, +i: U32, +k: U32) -> Array<U32>:
  bump.go(Array.get(U32, c, i), i, k)

# the slice stops with error e (aux: a detail for the message)
def fail(c: Array<U32>, +e: U32, +aux: U32) -> Array<U32>:
  c[18] <- 4
  c[19] <- e
  c[20] <- aux

# R[rd] = v; R0 stays 0
def setr.go(z: Bool, c: Array<U32>, +r: U32, +v: U32) -> Array<U32>:
  match z:
    case True{}:
      c
    case False{}:
      c[r] <- v

def setr(c: Array<U32>, +r: U32, +v: U32) -> Array<U32>:
  setr.go(U32.is_eq(r, 0), c, r, v)

# a[i] for the n words i = s, s+1, ...: (array, word)
def cp.step(r: Array<U32> & U32, dst: Array<U32>, +j: U32) -> Array<U32> & Array<U32>:
  (src, v) = r
  (src, Array.set(U32, dst, j, v))

# dst[j + q] = src[i + q] for q < n (two arrays)
def cp(n: Nat, st: Array<U32> & Array<U32>, +i: U32, +j: U32) -> Array<U32> & Array<U32>:
  match n:
    case 0n:
      st
    case 1n+p:
      (src, dst) = st
      cp(p, cp.step(Array.get(U32, src, i), dst, j), (i + 1 : U32), (j + 1 : U32))

def cp1.step(r: Array<U32> & U32, +j: U32) -> Array<U32>:
  (a, v) = r
  a[j] <- v

# a[j + q] = a[i + q] for q < n, in order (one array)
def cp1(n: Nat, a: Array<U32>, +i: U32, +j: U32) -> Array<U32>:
  match n:
    case 0n:
      a
    case 1n+p:
      cp1(p, cp1.step(Array.get(U32, a, i), j), (i + 1 : U32), (j + 1 : U32))

# a[i + q] = v for q < n
def fill(n: Nat, a: Array<U32>, +i: U32, +v: U32) -> Array<U32>:
  match n:
    case 0n:
      a
    case 1n+p:
      fill(p, a[i] <- v, (i + 1 : U32), v)

# ------------------------------------------------------------------------------- fetch

# An instruction's eight words (docs/isa.md "Encoding") and the values of its four register
# fields (R0 reads 0).
type Ins is Data:
  Ins{w0: U32, w1: U32, w2: U32, w3: U32, w4: U32, w5: U32, w6: U32, w7: U32, va: U32,
      vb: U32, vc: U32, vd: U32}

def Ins.w0(+i: Ins) -> U32:
  match i:
    case Ins{+w0, w1, w2, w3, w4, w5, w6, w7, va, vb, vc, vd}:
      w0

def op(+w0: U32) -> U32:
  (w0 .&. 255 : U32)

def fl(+w0: U32) -> U32:
  (w0 >> 24n : U32)

def rd(+w0: U32) -> U32:
  ((w0 >> 20n : U32) .&. 15 : U32)

def fe.vd(r: Array<U32> & U32, +w0: U32, +w1: U32, +w2: U32, +w3: U32, +w4: U32, +w5: U32,
  +w6: U32, +w7: U32, +va: U32, +vb: U32, +vc: U32) -> Array<U32> & Ins:
  (c, vd) = r
  (c, Ins{w0, w1, w2, w3, w4, w5, w6, w7, va, vb, vc, vd})

def fe.vc(r: Array<U32> & U32, +w0: U32, +w1: U32, +w2: U32, +w3: U32, +w4: U32, +w5: U32,
  +w6: U32, +w7: U32, +va: U32, +vb: U32) -> Array<U32> & Ins:
  (c, vc) = r
  fe.vd(Array.get(U32, c, rd(w0)), w0, w1, w2, w3, w4, w5, w6, w7, va, vb, vc)

def fe.vb(r: Array<U32> & U32, +w0: U32, +w1: U32, +w2: U32, +w3: U32, +w4: U32, +w5: U32,
  +w6: U32, +w7: U32, +va: U32) -> Array<U32> & Ins:
  (c, vb) = r
  fe.vc(Array.get(U32, c, ((w0 >> 16n : U32) .&. 15 : U32)), w0, w1, w2, w3, w4, w5, w6, w7,
    va, vb)

def fe.va(r: Array<U32> & U32, +w0: U32, +w1: U32, +w2: U32, +w3: U32, +w4: U32, +w5: U32,
  +w6: U32, +w7: U32) -> Array<U32> & Ins:
  (c, va) = r
  fe.vb(Array.get(U32, c, ((w0 >> 12n : U32) .&. 15 : U32)), w0, w1, w2, w3, w4, w5, w6, w7,
    va)

def fe.w7(r: Array<U32> & U32, +w0: U32, +w1: U32, +w2: U32, +w3: U32, +w4: U32, +w5: U32,
  +w6: U32) -> Array<U32> & Ins:
  (c, w7) = r
  fe.va(Array.get(U32, c, ((w0 >> 8n : U32) .&. 15 : U32)), w0, w1, w2, w3, w4, w5, w6, w7)

def fe.w6(r: Array<U32> & U32, +b: U32, +w0: U32, +w1: U32, +w2: U32, +w3: U32, +w4: U32,
  +w5: U32) -> Array<U32> & Ins:
  (c, w6) = r
  fe.w7(Array.get(U32, c, (b + 7 : U32)), w0, w1, w2, w3, w4, w5, w6)

def fe.w5(r: Array<U32> & U32, +b: U32, +w0: U32, +w1: U32, +w2: U32, +w3: U32, +w4: U32) ->
  Array<U32> & Ins:
  (c, w5) = r
  fe.w6(Array.get(U32, c, (b + 6 : U32)), b, w0, w1, w2, w3, w4, w5)

def fe.w4(r: Array<U32> & U32, +b: U32, +w0: U32, +w1: U32, +w2: U32, +w3: U32) ->
  Array<U32> & Ins:
  (c, w4) = r
  fe.w5(Array.get(U32, c, (b + 5 : U32)), b, w0, w1, w2, w3, w4)

def fe.w3(r: Array<U32> & U32, +b: U32, +w0: U32, +w1: U32, +w2: U32) -> Array<U32> & Ins:
  (c, w3) = r
  fe.w4(Array.get(U32, c, (b + 4 : U32)), b, w0, w1, w2, w3)

def fe.w2(r: Array<U32> & U32, +b: U32, +w0: U32, +w1: U32) -> Array<U32> & Ins:
  (c, w2) = r
  fe.w3(Array.get(U32, c, (b + 3 : U32)), b, w0, w1, w2)

def fe.w1(r: Array<U32> & U32, +b: U32, +w0: U32) -> Array<U32> & Ins:
  (c, w1) = r
  fe.w2(Array.get(U32, c, (b + 2 : U32)), b, w0, w1)

def fe.w0(r: Array<U32> & U32, +b: U32) -> Array<U32> & Ins:
  (c, w0) = r
  fe.w1(Array.get(U32, c, (b + 1 : U32)), b, w0)

# the instruction at pc, its register fields read
def fetch(c: Array<U32>, +pc: U32) -> Array<U32> & Ins:
  +b = (IMEM() + pc * 8 : U32)
  fe.w0(Array.get(U32, c, b), b)

# ------------------------------------------------------------------------------- control

# the next pc (isasim Slice.advance): at a loop body's end, back to its start while the
# count lasts, else pop the loop
def adv.start(r: Array<U32> & U32) -> Array<U32>:
  (c, v) = r
  c[16] <- v

def adv.back(z: Bool, c: Array<U32>, +pc: U32, +e: U32, +n: U32, +sp: U32) -> Array<U32>:
  match z:
    case True{}:
      c[(e + 2 : U32)] <- (n - 1 : U32)
      adv.start(Array.get(U32, c, e))
    case False{}:
      c[22] <- (sp - 1 : U32)
      c[16] <- (pc + 1 : U32)

def adv.cnt(r: Array<U32> & U32, +pc: U32, +e: U32, +sp: U32) -> Array<U32>:
  (c, +n) = r
  adv.back((n > 1 : U32), c, pc, e, n, sp)

def adv.hit(z: Bool, c: Array<U32>, +pc: U32, +e: U32, +sp: U32) -> Array<U32>:
  match z:
    case True{}:
      adv.cnt(Array.get(U32, c, (e + 2 : U32)), pc, e, sp)
    case False{}:
      c[16] <- (pc + 1 : U32)

def adv.end(r: Array<U32> & U32, +pc: U32, +e: U32, +sp: U32) -> Array<U32>:
  (c, +end) = r
  adv.hit(U32.is_eq(end, pc), c, pc, e, sp)

def adv.top(z: Bool, c: Array<U32>, +pc: U32, +sp: U32) -> Array<U32>:
  match z:
    case True{}:
      c[16] <- (pc + 1 : U32)
    case False{}:
      +e = (23 + 3 * (sp - 1 : U32) : U32)
      adv.end(Array.get(U32, c, (e + 1 : U32)), pc, e, sp)

def adv.sp(r: Array<U32> & U32, +pc: U32) -> Array<U32>:
  (c, +sp) = r
  adv.top(U32.is_eq(sp, 0), c, pc, sp)

def adv.pc(r: Array<U32> & U32) -> Array<U32>:
  (c, +pc) = r
  adv.sp(Array.get(U32, c, 22), pc)

def adv(c: Array<U32>) -> Array<U32>:
  adv.pc(Array.get(U32, c, 16))

# R0..R7 = 0, R8..R15 = the run's arguments, pc 0, the loop stack empty, RUN
def restart(c: Array<U32>) -> Array<U32>:
  c = cp1(8n, fill(8n, c, 0, 0), ARGS(), 8)
  c[16] <- 0
  c[22] <- 0
  c[18] <- 0

# LOOP: count R[ra] + w2 (0: skip the body), body w1 instructions after it
def lp.any.step(r: Array<U32> & U32, +end: U32, +f: U32) -> Array<U32> & U32:
  (c, +e) = r
  (c, (f .|. F.b2u(U32.is_eq(e, end)) : U32))

# whether a loop on the stack ends at end
def lp.any(n: Nat, st: Array<U32> & U32, +j: U32, +end: U32) -> Array<U32> & U32:
  match n:
    case 0n:
      st
    case 1n+p:
      (c, +f) = st
      lp.any(p, lp.any.step(Array.get(U32, c, (24 + 3 * j : U32)), end, f), (j + 1 : U32), end)

def lp.push(z: Bool, c: Array<U32>, +body: U32, +n: U32, +pc: U32, +sp: U32) -> Array<U32>:
  match z:
    case True{}:
      fail(c, 5, pc)
    case False{}:
      +e = (23 + 3 * sp : U32)
      c[e] <- (pc + 1 : U32)
      c[(e + 1 : U32)] <- (pc + body : U32)
      c[(e + 2 : U32)] <- n
      c[22] <- (sp + 1 : U32)
      c[16] <- (pc + 1 : U32)

def lp.zero(z: Bool, c: Array<U32>, +body: U32, +n: U32, +pc: U32, +sp: U32) -> Array<U32>:
  match z:
    case True{}:
      c[16] <- (pc + 1 + body : U32)
    case False{}:
      lp.push((sp >= 4 : U32), c, body, n, pc, sp)

def lp.dup(z: Bool, c: Array<U32>, +body: U32, +n: U32, +pc: U32, +sp: U32) -> Array<U32>:
  match z:
    case True{}:
      fail(c, 4, pc)
    case False{}:
      lp.zero(U32.is_eq(n, 0), c, body, n, pc, sp)

def lp.same(r: Array<U32> & U32, +body: U32, +n: U32, +pc: U32, +sp: U32) -> Array<U32>:
  (c, +same) = r
  lp.dup(U32.is_ne(same, 0), c, body, n, pc, sp)

def lp.empty(z: Bool, c: Array<U32>, +body: U32, +n: U32, +pc: U32, +sp: U32) -> Array<U32>:
  match z:
    case True{}:
      fail(c, 3, pc)
    case False{}:
      lp.same(lp.any(U32.to_nat(sp), (c, 0), 0, (pc + body : U32)), body, n, pc, sp)

def lp.sp(r: Array<U32> & U32, +body: U32, +n: U32, +pc: U32) -> Array<U32>:
  (c, +sp) = r
  lp.empty(U32.is_eq(body, 0), c, body, n, pc, sp)

def lp.pc(r: Array<U32> & U32, +body: U32, +n: U32) -> Array<U32>:
  (c, +pc) = r
  lp.sp(Array.get(U32, c, 22), body, n, pc)

def x.loop(s: Sl, +i: Ins) -> Sl:
  match s:
    case Sl{c, t, a, d, h, x, k}:
      match i:
        case Ins{w0, +w1, +w2, w3, w4, w5, w6, w7, +va, vb, vc, vd}:
          Sl{lp.pc(Array.get(U32, c, 16), w1, (va + w2 : U32)), t, a, d, h, x, k}

# LI: R[rd] = w1; ADDI: R[rd] = R[ra] + w1
def x.li(s: Sl, +i: Ins) -> Sl:
  match s:
    case Sl{c, t, a, d, h, x, k}:
      match i:
        case Ins{+w0, +w1, w2, w3, w4, w5, w6, w7, va, vb, vc, vd}:
          Sl{setr(c, rd(w0), w1), t, a, d, h, x, k}

def x.addi(s: Sl, +i: Ins) -> Sl:
  match s:
    case Sl{c, t, a, d, h, x, k}:
      match i:
        case Ins{+w0, +w1, w2, w3, w4, w5, w6, w7, +va, vb, vc, vd}:
          Sl{setr(c, rd(w0), (va + w1 : U32)), t, a, d, h, x, k}

# whether the n DRAM words from byte address b are aligned and in range
def dwords(+b: U32, +n: U32, +k: Cf) -> Bool:
  U32.is_eq((b .&. 3 : U32), 0) && rng(b, n, 4, 1, 0, (Cf.dbm1(k) - 3 : U32))

# whether the TMEM words b + r*s1 + q*s2 (r < n1, q < n2) are in range
def twords(+b: U32, +n1: U32, +s1: U32, +n2: U32, +s2: U32, +k: Cf) -> Bool:
  rng(b, n1, s1, n2, s2, (Cf.tw(k) - 1 : U32))

# LD: T[t + q] = M32[d + 4q]; ST: the other way
def ld.fin(r: Array<U32> & Array<U32>, c: Array<U32>, a: Array<U32>, h: Array<U32>,
  x: Array<U32>, +k: Cf) -> Sl:
  (d, t) = r
  Sl{c, t, a, d, h, x, k}

def st.fin(r: Array<U32> & Array<U32>, c: Array<U32>, a: Array<U32>, h: Array<U32>,
  x: Array<U32>, +k: Cf) -> Sl:
  (t, d) = r
  Sl{c, t, a, d, h, x, k}

def ldst.go(st: Bool, s: Sl, +dd: U32, +tt: U32, +n: U32) -> Sl:
  match st s:
    case False{} Sl{c, t, a, d, h, x, +k}:
      ld.fin(cp(U32.to_nat(n), (d, t), (dd >> 2n : U32), tt), c, a, h, x, k)
    case True{} Sl{c, t, a, d, h, x, +k}:
      st.fin(cp(U32.to_nat(n), (t, d), tt, (dd >> 2n : U32)), c, a, h, x, k)

def ldst.tm(z: Bool, st: Bool, s: Sl, +dd: U32, +tt: U32, +n: U32) -> Sl:
  match z s:
    case True{} s:
      ldst.go(st, s, dd, tt, n)
    case False{} Sl{c, t, a, d, h, x, k}:
      Sl{fail(c, 7, tt), t, a, d, h, x, k}

def ldst.dm(z: Bool, st: Bool, s: Sl, +dd: U32, +tt: U32, +n: U32, +k: Cf) -> Sl:
  match z s:
    case True{} s:
      ldst.tm(twords(tt, n, 1, 1, 0, k), st, s, dd, tt, n)
    case False{} Sl{c, t, a, d, h, x, k2}:
      Sl{fail(c, 6, dd), t, a, d, h, x, k2}

def ldst.n(z: Bool, st: Bool, s: Sl, +dd: U32, +tt: U32, +n: U32) -> Sl:
  match z s:
    case True{} s:
      s
    case False{} Sl{c, t, a, d, h, x, +k}:
      ldst.dm(dwords(dd, n, k), st, Sl{c, t, a, d, h, x, k}, dd, tt, n, k)

def x.ldst(st: Bool, s: Sl, +i: Ins) -> Sl:
  match i:
    case Ins{w0, +w1, +w2, +w3, w4, w5, w6, w7, +va, +vb, vc, vd}:
      ldst.n(U32.is_eq(w3, 0), st, s, (va + w1 : U32), (vb + w2 : U32), w3)

# RLD: R[rd] = f2i(T[R[ra] + w1]) (RAW: the bits), times R[rb] + w2 (MUL), w3 <= it <= w4
# (BOUND)
def rld.set(z: Bool, c: Array<U32>, +r: U32, +v: U32) -> Array<U32>:
  match z:
    case True{}:
      setr(c, r, v)
    case False{}:
      fail(c, 8, v)

def rld.v(r: Array<U32> & U32, c: Array<U32>, a: Array<U32>, d: Array<U32>, h: Array<U32>,
  x: Array<U32>, +k: Cf, +i: Ins) -> Sl:
  (t, +w) = r
  match i:
    case Ins{+w0, w1, +w2, +w3, +w4, w5, w6, w7, va, +vb, vc, vd}:
      +f = fl(w0)
      +v0 = F.sel(U32.is_ne((f .&. 1 : U32), 0), w, F.f2i(w))
      +v = F.sel(U32.is_ne((f .&. 2 : U32), 0), (v0 * (vb + w2 : U32) : U32), v0)
      +ok = U32.is_eq((f .&. 4 : U32), 0) || ((w3 <= v : U32) && (v <= w4 : U32))
      Sl{rld.set(ok, c, rd(w0), v), t, a, d, h, x, k}

def rld.chk(z: Bool, s: Sl, +ta: U32, +i: Ins) -> Sl:
  match z s:
    case True{} Sl{c, t, a, d, h, x, +k}:
      rld.v(Array.get(U32, t, ta), c, a, d, h, x, k, i)
    case False{} Sl{c, t, a, d, h, x, k}:
      Sl{fail(c, 7, ta), t, a, d, h, x, k}

def x.rld(s: Sl, +i: Ins) -> Sl:
  match s:
    case Sl{c, t, a, d, h, x, +k}:
      match i:
        case Ins{w0, +w1, w2, w3, w4, w5, w6, w7, +va, vb, vc, vd}:
          +ta = (va + w1 : U32)
          rld.chk((ta < Cf.tw(k) : U32), Sl{c, t, a, d, h, x, k}, ta, i)

# HALT; CHAIN: load R[rb] instructions from DRAM byte address R[ra] into IMEM and start them
# (docs/isa.md "HALT CHAIN"): R8..R15 the run's arguments again, TMEM, ACT RAM, DRAM kept
def chain.fin(r: Array<U32> & Array<U32>, +n: U32) -> Array<U32> & Array<U32>:
  (d, c) = r
  c[21] <- n
  (d, bump(restart(c), CHAINS(), 1))

def chain.cp(r: Array<U32> & Array<U32>, t: Array<U32>, a: Array<U32>, h: Array<U32>,
  x: Array<U32>, +k: Cf) -> Sl:
  (d, c) = r
  Sl{c, t, a, d, h, x, k}

def chain.dm(z: Bool, s: Sl, +ad: U32, +n: U32) -> Sl:
  match z s:
    case True{} Sl{c, t, a, d, h, x, +k}:
      chain.cp(chain.fin(cp(U32.to_nat((8 * n : U32)), (d, c), (ad >> 2n : U32), IMEM()), n),
        t, a, h, x, k)
    case False{} Sl{c, t, a, d, h, x, k}:
      Sl{fail(c, 6, ad), t, a, d, h, x, k}

def chain.ok(z: Bool, s: Sl, +ad: U32, +n: U32, +k: Cf) -> Sl:
  match z s:
    case True{} s:
      chain.dm(dwords(ad, (8 * n : U32), k), s, ad, n)
    case False{} Sl{c, t, a, d, h, x, k2}:
      Sl{fail(c, 14, ad), t, a, d, h, x, k2}

def halt.go(z: Bool, s: Sl, +ad: U32, +n: U32) -> Sl:
  match z s:
    case True{} Sl{c, t, a, d, h, x, +k}:
      chain.ok(U32.is_eq((ad % Cf.d(k) : U32), 0) && (n > 0 : U32)
        && (n <= (Cf.iw(k) >> 3n : U32) : U32), Sl{c, t, a, d, h, x, k}, ad, n, k)
    case False{} Sl{c, t, a, d, h, x, k}:
      Sl{c[18] <- 1, t, a, d, h, x, k}

def x.halt(s: Sl, +i: Ins) -> Sl:
  match i:
    case Ins{+w0, w1, w2, w3, w4, w5, w6, w7, +va, +vb, vc, vd}:
      halt.go(U32.is_ne((fl(w0) .&. 1 : U32), 0), s, va, vb)

# BAR, GATHER: the slice waits for the others (the machine runs the collective)
def x.wait(s: Sl) -> Sl:
  match s:
    case Sl{c, t, a, d, h, x, k}:
      Sl{c[18] <- 2, t, a, d, h, x, k}

# WAITW's condition on the word it read (isa.waitw_holds): EQ, NE, GE (a signed difference)
def holds(+v: U32, +ref: U32, +cmp: U32, +mask: U32) -> Bool:
  +m = (v .&. mask : U32)
  (U32.is_eq(cmp, 0) && U32.is_eq(m, ref)) || (U32.is_eq(cmp, 1) && U32.is_ne(m, ref))
    || ((cmp >= 2 : U32) && ((m - ref : U32) < 2147483648 : U32))

# WAITW: it holds: T[R[rb] + w2] = the word, go on (RUN); it does not: POLL (first: a step's
# own read, which counts a stall; the machine's later reads do not)
def ww.set(z: Bool, s: Sl, +ta: U32, +v: U32) -> Sl:
  match z s:
    case True{} Sl{c, t, a, d, h, x, k}:
      c[18] <- 0
      Sl{adv(c), Array.set(U32, t, ta, v), a, d, h, x, k}
    case False{} Sl{c, t, a, d, h, x, k}:
      Sl{fail(c, 7, ta), t, a, d, h, x, k}

def ww.hold(z: Bool, s: Sl, +ta: U32, +v: U32, +first: U32) -> Sl:
  match z s:
    case True{} Sl{c, t, a, d, h, x, +k}:
      ww.set((ta < Cf.tw(k) : U32), Sl{c, t, a, d, h, x, k}, ta, v)
    case False{} Sl{c, t, a, d, h, x, k}:
      c[18] <- 3
      Sl{bump(c, STALLS(), first), t, a, d, h, x, k}

def ww.v(r: Array<U32> & U32, c: Array<U32>, t: Array<U32>, a: Array<U32>, h: Array<U32>,
  x: Array<U32>, +k: Cf, +i: Ins, +first: U32) -> Sl:
  (d, +v) = r
  match i:
    case Ins{+w0, w1, +w2, +w3, +w4, w5, w6, w7, va, +vb, +vc, vd}:
      ww.hold(holds(v, (vc + w3 : U32), (fl(w0) .&. 3 : U32), w4), Sl{c, t, a, d, h, x, k},
        (vb + w2 : U32), v, first)

def ww.dm(z: Bool, s: Sl, +ad: U32, +i: Ins, +first: U32) -> Sl:
  match z s:
    case True{} Sl{c, t, a, d, h, x, +k}:
      ww.v(Array.get(U32, d, (ad >> 2n : U32)), c, t, a, h, x, k, i, first)
    case False{} Sl{c, t, a, d, h, x, k}:
      Sl{fail(c, 6, ad), t, a, d, h, x, k}

def x.waitw(s: Sl, +i: Ins, +first: U32) -> Sl:
  match s:
    case Sl{c, t, a, d, h, x, +k}:
      match i:
        case Ins{w0, +w1, w2, w3, w4, w5, w6, w7, +va, vb, vc, vd}:
          +ad = (va + w1 : U32)
          ww.dm(dwords(ad, 1, k), Sl{c, t, a, d, h, x, k}, ad, i, first)

# ------------------------------------------------------------------------------- VPU

# VOP (docs/isa.md "VOP"). The VPU processes a VOP's elements in order: element p reads its
# operands, then writes its result. isasim reads every operand first and raises a hazard if
# an element reads a word an earlier element of the same VOP wrote; without one the two orders
# agree. h stamps each written word with the VOP's number g, so a read of a word stamped g is
# that hazard (error 34).

# TMEM and the stamps through an element, with the word read and the hazard flag
type TV is Type:
  TV{t: Array<U32>, h: Array<U32>, v: U32, z: U32}

def tv.h(r: Array<U32> & U32, t: Array<U32>, +v: U32, +z: U32, +g: U32) -> TV:
  (h, +s) = r
  TV{t, h, v, (z .|. F.b2u(U32.is_eq(s, g)) : U32)}

def tv.t(r: Array<U32> & U32, h: Array<U32>, +i: U32, +z: U32, +g: U32) -> TV:
  (t, +v) = r
  tv.h(Array.get(U32, h, i), t, v, z, g)

# T[i], checked against this VOP's writes; dflt (and no read) when off
def tread(on: Bool, t: Array<U32>, h: Array<U32>, +i: U32, +z: U32, +g: U32, +dflt: U32) ->
  TV:
  match on:
    case True{}:
      tv.t(Array.get(U32, t, i), h, i, z, g)
    case False{}:
      TV{t, h, dflt, z}

# T[i] = v, stamped g
def twrite(r: TV, +i: U32, +v: U32, +g: U32) -> TV:
  match r:
    case TV{t, h, w, +z}:
      TV{Array.set(U32, t, i, v), Array.set(U32, h, i, g), 0, z}

# the elementwise functions: D[r][c] = f(A[r][c], B(r, c))
def ew.b(~f: U32 -> U32 -> U32, r: TV, +av: U32, +di: U32, +g: U32) -> TV:
  match r:
    case TV{t, h, +bv, +z}:
      twrite(TV{t, h, 0, z}, di, f(av, bv), g)

def ew.a(~f: U32 -> U32 -> U32, r: TV, +rb: Bool, +bi: U32, +imm: U32, +di: U32, +g: U32) ->
  TV:
  match r:
    case TV{t, h, +av, +z}:
      ew.b(~f, tread(rb, t, h, bi, z, g, imm), av, di, g)

def ew.cols(~f: U32 -> U32 -> U32, n: Nat, r: TV, +ra: Bool, +rb: Bool, +ai: U32, +bi: U32,
  +bs: U32, +imm: U32, +di: U32, +g: U32) -> TV:
  match n:
    case 0n:
      r
    case 1n+p:
      match r:
        case TV{t, h, v, +z}:
          ew.cols(~f, p, ew.a(~f, tread(ra, t, h, ai, z, g, 0), rb, bi, imm, di, g), ra, rb,
            (ai + 1 : U32), (bi + bs : U32), bs, imm, (di + 1 : U32), g)

# rows of cols elements: A at a + r*ars + c; B at b + r*brs + c*bs (or imm, rb off); D at
# dst + r*drs + c
def ew.rows(~f: U32 -> U32 -> U32, n: Nat, r: TV, +cols: U32, +ra: Bool, +rb: Bool, +a: U32,
  +ars: U32, +b: U32, +brs: U32, +bs: U32, +imm: U32, +dst: U32, +drs: U32, +g: U32) -> TV:
  match n:
    case 0n:
      r
    case 1n+p:
      ew.rows(~f, p, ew.cols(~f, U32.to_nat(cols), r, ra, rb, a, b, bs, imm, dst, g), cols, ra,
        rb, (a + ars : U32), ars, (b + brs : U32), brs, bs, imm, (dst + drs : U32), drs, g)

# A VOP's geometry and operands, resolved
type Vg is Data:
  Vg{dst: U32, a: U32, b: U32, rows: U32, cols: U32, drs: U32, ars: U32, brs: U32, fn: U32,
     bm: U32, imm: U32, fl: U32}

def ew.run(~f: U32 -> U32 -> U32, r: TV, +ra: Bool, +bin: Bool, +v: Vg, +g: U32) -> TV:
  match v:
    case Vg{+dst, +a, +b, +rows, +cols, +drs, +ars, +brs, fn, +bm, +imm, fl}:
      # B: FULL b + r*brs + c, ROW b + r*brs, COL b + c, SCALAR the immediate
      ew.rows(~f, U32.to_nat(rows), r, cols, ra, bin && (bm < 3 : U32), a, ars, b,
        F.sel(U32.is_eq(bm, 2), 0, brs), F.sel(U32.is_eq(bm, 1), 0, 1), imm, dst, drs, g)

# each function its own copy of the loop; z |= 2: not an elementwise function
def ew.bad(r: TV) -> TV:
  match r:
    case TV{t, h, v, +z}:
      TV{t, h, v, (z .|. 2 : U32)}

def ew.f15(z: Bool, r: TV, +v: Vg, +g: U32, +fn: U32) -> TV:
  match z:
    case True{}:
      ew.run(~(p => q => F.log2(p)), r, True{}, False{}, v, g)
    case False{}:
      ew.bad(r)

def ew.f14(z: Bool, r: TV, +v: Vg, +g: U32, +fn: U32) -> TV:
  match z:
    case True{}:
      ew.run(~(p => q => F.exp2(F.sub(p, q))), r, True{}, True{}, v, g)
    case False{}:
      ew.f15(U32.is_eq(fn, 15), r, v, g, fn)

def ew.f13(z: Bool, r: TV, +v: Vg, +g: U32, +fn: U32) -> TV:
  match z:
    case True{}:
      ew.run(~(p => q => F.ftz(q)), r, False{}, True{}, v, g)
    case False{}:
      ew.f14(U32.is_eq(fn, 14), r, v, g, fn)

def ew.f12(z: Bool, r: TV, +v: Vg, +g: U32, +fn: U32) -> TV:
  match z:
    case True{}:
      ew.run(~(p => q => F.fabs(p)), r, True{}, False{}, v, g)
    case False{}:
      ew.f13(U32.is_eq(fn, 13), r, v, g, fn)

def ew.f11(z: Bool, r: TV, +v: Vg, +g: U32, +fn: U32) -> TV:
  match z:
    case True{}:
      ew.run(~(p => q => F.rsqrt(p)), r, True{}, False{}, v, g)
    case False{}:
      ew.f12(U32.is_eq(fn, 12), r, v, g, fn)

def ew.f10(z: Bool, r: TV, +v: Vg, +g: U32, +fn: U32) -> TV:
  match z:
    case True{}:
      ew.run(~(p => q => F.recip(p)), r, True{}, False{}, v, g)
    case False{}:
      ew.f11(U32.is_eq(fn, 11), r, v, g, fn)

def ew.f9(z: Bool, r: TV, +v: Vg, +g: U32, +fn: U32) -> TV:
  match z:
    case True{}:
      ew.run(~(p => q => F.exp2(p)), r, True{}, False{}, v, g)
    case False{}:
      ew.f10(U32.is_eq(fn, 10), r, v, g, fn)

def ew.f8(z: Bool, r: TV, +v: Vg, +g: U32, +fn: U32) -> TV:
  match z:
    case True{}:
      ew.run(~(p => q => F.ftz(p)), r, True{}, False{}, v, g)
    case False{}:
      ew.f9(U32.is_eq(fn, 9), r, v, g, fn)

def ew.f5(z: Bool, r: TV, +v: Vg, +g: U32, +fn: U32) -> TV:
  match z:
    case True{}:
      ew.run(~(p => q => F.fmin(p, q)), r, True{}, True{}, v, g)
    case False{}:
      ew.f8(U32.is_eq(fn, 8), r, v, g, fn)

def ew.f4(z: Bool, r: TV, +v: Vg, +g: U32, +fn: U32) -> TV:
  match z:
    case True{}:
      ew.run(~(p => q => F.fmax(p, q)), r, True{}, True{}, v, g)
    case False{}:
      ew.f5(U32.is_eq(fn, 5), r, v, g, fn)

def ew.f2(z: Bool, r: TV, +v: Vg, +g: U32, +fn: U32) -> TV:
  match z:
    case True{}:
      ew.run(~(p => q => F.sub(q, p)), r, True{}, True{}, v, g)
    case False{}:
      ew.f4(U32.is_eq(fn, 4), r, v, g, fn)

def ew.f1(z: Bool, r: TV, +v: Vg, +g: U32, +fn: U32) -> TV:
  match z:
    case True{}:
      ew.run(~(p => q => F.sub(p, q)), r, True{}, True{}, v, g)
    case False{}:
      ew.f2(U32.is_eq(fn, 2), r, v, g, fn)

def ew.f0(z: Bool, r: TV, +v: Vg, +g: U32, +fn: U32) -> TV:
  match z:
    case True{}:
      ew.run(~(p => q => F.add(p, q)), r, True{}, True{}, v, g)
    case False{}:
      ew.f1(U32.is_eq(fn, 1), r, v, g, fn)

def ew.f3(z: Bool, r: TV, +v: Vg, +g: U32, +fn: U32) -> TV:
  match z:
    case True{}:
      ew.run(~(p => q => F.mul(p, q)), r, True{}, True{}, v, g)
    case False{}:
      ew.f0(U32.is_eq(fn, 0), r, v, g, fn)

# MUL first: the most frequent
def ew(r: TV, +v: Vg, +g: U32, +fn: U32) -> TV:
  ew.f3(U32.is_eq(fn, 3), r, v, g, fn)

# TMEM, the stamps and the scratch through a reduction's or an OUTER's elements
type TX is Type:
  TX{t: Array<U32>, h: Array<U32>, x: Array<U32>, z: U32}

def tx.acc(r: Array<U32> & U32, t: Array<U32>, h: Array<U32>, +z: U32, +j: U32, +e: U32) ->
  TX:
  (x, +s) = r
  TX{t, h, Array.set(U32, x, j, F.add(s, e)), z}

# a sum's term e(A, B) into its partial: x[q mod 64] += e (isum_64, fp32.interleaved_sum)
def rs.b(~f: U32 -> U32 -> U32, r: TV, x: Array<U32>, +av: U32, +sq: Bool, +j: U32) -> TX:
  match r:
    case TV{t, h, +bv, +z}:
      tx.acc(Array.get(U32, x, j), t, h, z, j, f(av, F.sel(sq, av, bv)))

def rs.a(~f: U32 -> U32 -> U32, r: TV, x: Array<U32>, +rb: Bool, +sq: Bool, +bi: U32,
  +imm: U32, +j: U32, +g: U32) -> TX:
  match r:
    case TV{t, h, +av, +z}:
      rs.b(~f, tread(rb, t, h, bi, z, g, imm), x, av, sq, j)

def rs.cols(~f: U32 -> U32 -> U32, n: Nat, r: TX, +rb: Bool, +sq: Bool, +ai: U32, +bi: U32,
  +bs: U32, +imm: U32, +q: U32, +g: U32) -> TX:
  match n:
    case 0n:
      r
    case 1n+p:
      match r:
        case TX{t, h, x, +z}:
          rs.cols(~f, p, rs.a(~f, tread(True{}, t, h, ai, z, g, 0), x, rb, sq, bi, imm,
            (q .&. 63 : U32), g), rb, sq, (ai + 1 : U32), (bi + bs : U32), bs, imm,
            (q + 1 : U32), g)

def tree.step(r: Array<U32> & U32, +i: U32, +w: U32, +u: U32) -> Array<U32>:
  (x, +v) = r
  x[i] <- F.add(u, v)

def tree.lv.u(r: Array<U32> & U32, +i: U32, +w: U32) -> Array<U32>:
  (x, +u) = r
  tree.step(Array.get(U32, x, (i + w : U32)), i, w, u)

# x[i] = x[i] + x[i + w] for i < w
def tree.lv(n: Nat, x: Array<U32>, +i: U32, +w: U32) -> Array<U32>:
  match n:
    case 0n:
      x
    case 1n+p:
      tree.lv(p, tree.lv.u(Array.get(U32, x, i), i, w), (i + 1 : U32), w)

# the folding tree over the 64 partials: x[0] is the sum
def tree(n: Nat, x: Array<U32>, +w: U32) -> Array<U32>:
  match n:
    case 0n:
      x
    case 1n+p:
      tree(p, tree.lv(U32.to_nat(w), x, 0, w), (w >> 1n : U32))

def pad.step(r: Array<U32> & U32, +j: U32) -> Array<U32>:
  (x, +s) = r
  x[j] <- F.add(s, 0)

# the +0 terms that pad the row to a multiple of 64 (they turn a -0 partial into +0)
def pad(n: Nat, x: Array<U32>, +q: U32) -> Array<U32>:
  match n:
    case 0n:
      x
    case 1n+p:
      pad(p, pad.step(Array.get(U32, x, (q .&. 63 : U32)), (q .&. 63 : U32)), (q + 1 : U32))

def tv2tx(r: TV, x: Array<U32>) -> TX:
  match r:
    case TV{t, h, v, +z}:
      TX{t, h, x, z}

# T[di] = x[0], the row's sum
def rs.out(r: Array<U32> & U32, t: Array<U32>, h: Array<U32>, +z: U32, +di: U32, +g: U32) ->
  TX:
  (x, +s) = r
  tv2tx(twrite(TV{t, h, 0, z}, di, s, g), x)

# a row's sum: the partials from +0, the row's terms, the padding, the tree; T[di] = the sum
def rs.fin(r: TX, +cols: U32, +di: U32, +g: U32) -> TX:
  match r:
    case TX{t, h, x, +z}:
      +padn = ((64 - (cols .&. 63 : U32) : U32) .&. 63 : U32)
      rs.out(Array.get(U32, tree(6n, pad(U32.to_nat(padn), x, cols), 32), 0), t, h, z, di, g)

def rs.row(~f: U32 -> U32 -> U32, r: TX, +cols: U32, +rb: Bool, +sq: Bool, +ai: U32,
  +bi: U32, +bs: U32, +imm: U32, +di: U32, +g: U32) -> TX:
  match r:
    case TX{t, h, x, +z}:
      rs.fin(rs.cols(~f, U32.to_nat(cols), TX{t, h, fill(64n, x, 0, 0), z}, rb, sq, ai, bi, bs,
        imm, 0, g), cols, di, g)

def rs.rows(~f: U32 -> U32 -> U32, n: Nat, r: TX, +cols: U32, +rb: Bool, +sq: Bool, +a: U32,
  +ars: U32, +b: U32, +brs: U32, +bs: U32, +imm: U32, +dst: U32, +drs: U32, +g: U32) -> TX:
  match n:
    case 0n:
      r
    case 1n+p:
      rs.rows(~f, p, rs.row(~f, r, cols, rb, sq, a, b, bs, imm, dst, g), cols, rb, sq,
        (a + ars : U32), ars, (b + brs : U32), brs, bs, imm, (dst + drs : U32), drs, g)

def rs.run(~f: U32 -> U32 -> U32, r: TX, +rb: Bool, +sq: Bool, +v: Vg, +g: U32) -> TX:
  match v:
    case Vg{+dst, +a, +b, +rows, +cols, +drs, +ars, +brs, fn, +bm, +imm, fl}:
      rs.rows(~f, U32.to_nat(rows), r, cols, rb && (bm < 3 : U32), sq, a, ars, b,
        F.sel(U32.is_eq(bm, 2), 0, brs), F.sel(U32.is_eq(bm, 1), 0, 1), imm, dst, drs, g)

# the row's maximum in the total order of fp32.gt (VOP RMAX, fp32.chain_max) and the first
# column holding it (ARGMAX)
type TA is Type:
  TA{t: Array<U32>, h: Array<U32>, m: U32, j: U32, z: U32}

def am.v(r: TV, +q: U32, +m: U32, +j: U32) -> TA:
  match r:
    case TV{t, h, +v, +z}:
      +y = F.ftz(v)
      +up = F.gt(y, m)
      TA{t, h, F.sel(up, y, m), F.sel(up, q, j), z}

def am.cols(n: Nat, r: TA, +ai: U32, +q: U32, +g: U32) -> TA:
  match n:
    case 0n:
      r
    case 1n+p:
      match r:
        case TA{t, h, +m, +j, +z}:
          am.cols(p, am.v(tread(True{}, t, h, ai, z, g, 0), q, m, j), (ai + 1 : U32),
            (q + 1 : U32), g)

def am.w2(arg: Bool, r: TV, +di: U32, +j: U32, +base: U32, +g: U32) -> TV:
  match arg:
    case True{}:
      twrite(r, (di + 1 : U32), F.i2f((j + base : U32)), g)
    case False{}:
      r

# T[di] = the maximum (from the key-0 word -NaN 0xFFFFFFFF, below every other), and with arg
# T[di + 1] = i2f(column + base)
def am.out(r: TA, +di: U32, +base: U32, +arg: Bool, +g: U32) -> TV:
  match r:
    case TA{t, h, +m, +j, +z}:
      am.w2(arg, twrite(TV{t, h, 0, z}, di, m, g), di, j, base, g)

def am.row(r: TV, +cols: U32, +ai: U32, +di: U32, +base: U32, +arg: Bool, +g: U32) -> TV:
  match r:
    case TV{t, h, v, +z}:
      am.out(am.cols(U32.to_nat(cols), TA{t, h, 4294967295, 0, z}, ai, 0, g), di, base, arg, g)

def am.rows(n: Nat, r: TV, +cols: U32, +a: U32, +ars: U32, +dst: U32, +drs: U32, +base: U32,
  +arg: Bool, +g: U32) -> TV:
  match n:
    case 0n:
      r
    case 1n+p:
      am.rows(p, am.row(r, cols, a, dst, base, arg, g), cols, (a + ars : U32), ars,
        (dst + drs : U32), drs, base, arg, g)

def am.run(r: TV, +arg: Bool, +v: Vg, +g: U32) -> TV:
  match v:
    case Vg{+dst, +a, b, +rows, +cols, +drs, +ars, brs, fn, bm, +imm, fl}:
      am.rows(U32.to_nat(rows), r, cols, a, ars, dst, drs, imm, arg, g)

def tx.bad(r: TX) -> TX:
  match r:
    case TX{t, h, x, +z}:
      TX{t, h, x, (z .|. 2 : U32)}

def rs.f19(z: Bool, r: TX, +v: Vg, +g: U32) -> TX:
  match z:
    case True{}:
      rs.run(~(p => q => F.mul(p, q)), r, True{}, False{}, v, g)
    case False{}:
      tx.bad(r)

def rs.f18(z: Bool, r: TX, +v: Vg, +g: U32, +fn: U32) -> TX:
  match z:
    case True{}:
      rs.run(~(p => q => F.mul(p, q)), r, False{}, True{}, v, g)
    case False{}:
      rs.f19(U32.is_eq(fn, 19), r, v, g)

def rs.f17(z: Bool, r: TX, +v: Vg, +g: U32, +fn: U32) -> TX:
  match z r:
    case True{} TX{t, h, x, +e}:
      tv2tx(am.run(TV{t, h, 0, e}, False{}, v, g), x)
    case False{} r:
      rs.f18(U32.is_eq(fn, 18), r, v, g, fn)

def rs.f16(z: Bool, r: TX, +v: Vg, +g: U32, +fn: U32) -> TX:
  match z:
    case True{}:
      rs.run(~(p => q => p), r, False{}, False{}, v, g)
    case False{}:
      rs.f17(U32.is_eq(fn, 17), r, v, g, fn)

# the reductions: RSUM 16, RMAX 17, RSSQ 18, RDOT 19
def rs.go(r: TX, +v: Vg, +g: U32, +fn: U32) -> TX:
  rs.f16(U32.is_eq(fn, 16), r, v, g, fn)

# OUTER: D = D * Dv(c) + B(r) * C(c) (fp32.outer). C and Dv are read first into x (C at 0,
# Dv at 256; the RTL buffers them), so they may overlap D; D and B(r) are read with every
# element
def ou.dv(r: Array<U32> & U32, t: Array<U32>, h: Array<U32>, +z: U32, +old: U32, +bv: U32,
  +cv: U32, +di: U32, +g: U32) -> TX:
  (x, +dv) = r
  tv2tx(twrite(TV{t, h, 0, z}, di, F.outer(old, dv, bv, cv), g), x)

def ou.cv(r: Array<U32> & U32, t: Array<U32>, h: Array<U32>, +z: U32, +old: U32, +bv: U32,
  +q: U32, +di: U32, +g: U32) -> TX:
  (x, +cv) = r
  ou.dv(Array.get(U32, x, (256 + q : U32)), t, h, z, old, bv, cv, di, g)

def ou.b(r: TV, x: Array<U32>, +old: U32, +q: U32, +di: U32, +g: U32) -> TX:
  match r:
    case TV{t, h, +bv, +z}:
      ou.cv(Array.get(U32, x, q), t, h, z, old, bv, q, di, g)

def ou.d(r: TV, x: Array<U32>, +bi: U32, +q: U32, +di: U32, +g: U32) -> TX:
  match r:
    case TV{t, h, +old, +z}:
      ou.b(tread(True{}, t, h, bi, z, g, 0), x, old, q, di, g)

def ou.cols(n: Nat, r: TX, +bi: U32, +q: U32, +di: U32, +g: U32) -> TX:
  match n:
    case 0n:
      r
    case 1n+p:
      match r:
        case TX{t, h, x, +z}:
          ou.cols(p, ou.d(tread(True{}, t, h, di, z, g, 0), x, bi, q, di, g), bi, (q + 1 : U32),
            (di + 1 : U32), g)

def ou.rows(n: Nat, r: TX, +cols: U32, +b: U32, +brs: U32, +dst: U32, +drs: U32, +g: U32) ->
  TX:
  match n:
    case 0n:
      r
    case 1n+p:
      ou.rows(p, ou.cols(U32.to_nat(cols), r, b, 0, dst, g), cols, (b + brs : U32), brs,
        (dst + drs : U32), drs, g)

def ou.dsc(r: Array<U32> & U32, x: Array<U32>, +cols: U32) -> Array<U32> & Array<U32>:
  (t, +dv) = r
  (t, fill(U32.to_nat(cols), x, 256, dv))

# Dv into x[256..]: 1.0 (DONE), T[d] for every column (DSCALAR), or T[d + c]
def ou.dvec.go(one: Bool, sc: Bool, t: Array<U32>, x: Array<U32>, +d: U32, +cols: U32) ->
  Array<U32> & Array<U32>:
  match one sc:
    case True{} _:
      (t, fill(U32.to_nat(cols), x, 256, 1065353216))
    case False{} True{}:
      ou.dsc(Array.get(U32, t, d), x, cols)
    case False{} False{}:
      cp(U32.to_nat(cols), (t, x), d, 256)

def ou.dvec(t: Array<U32>, x: Array<U32>, +fl: U32, +d: U32, +cols: U32) ->
  Array<U32> & Array<U32>:
  ou.dvec.go(U32.is_ne((fl .&. 2 : U32), 0), U32.is_ne((fl .&. 1 : U32), 0), t, x, d, cols)

def ou.run2(r: Array<U32> & Array<U32>, h: Array<U32>, +v: Vg, +g: U32) -> TX:
  (t, x) = r
  match v:
    case Vg{+dst, a, +b, +rows, +cols, +drs, ars, +brs, fn, bm, imm, fl}:
      ou.rows(U32.to_nat(rows), TX{t, h, x, 0}, cols, b, brs, dst, drs, g)

def ou.run1(r: Array<U32> & Array<U32>, h: Array<U32>, +v: Vg, +g: U32) -> TX:
  (t, x) = r
  match v:
    case Vg{dst, +a, b, rows, +cols, drs, ars, brs, fn, bm, imm, +fl}:
      ou.run2(ou.dvec(t, x, fl, a, cols), h, v, g)

def ou.run(t: Array<U32>, h: Array<U32>, x: Array<U32>, +v: Vg, +g: U32) -> TX:
  match v:
    case Vg{dst, a, b, rows, +cols, drs, ars, brs, fn, bm, +imm, fl}:
      ou.run1(cp(U32.to_nat(cols), (t, x), imm, 0), h, v, g)

# 0, or the error a VOP stops with before it runs: 36 OUTER's bmode or cols, 35 ARGMAX's drs,
# 39 an RMAX or ARGMAX over no columns (isasim: numpy's error), 7 a word out of TMEM
def vop.chk(+v: Vg, +k: Cf) -> U32:
  match v:
    case Vg{+dst, +a, +b, +rows, +cols, +drs, +ars, +brs, +fn, +bm, +imm, +fl}:
      +outer = U32.is_eq(fn, 6)
      +red = (fn >= 16 : U32) && (fn <= 19 : U32)
      +arg = U32.is_eq(fn, 20)
      +elt = Bool.not(outer) && Bool.not(red) && Bool.not(arg)
      +none = U32.is_eq(cols, 0)
      +ra = Bool.not(outer) && U32.is_ne(fn, 13)
      +rb = ((fn <= 5 : U32) || U32.is_eq(fn, 13) || U32.is_eq(fn, 14) || U32.is_eq(fn, 19))
        && (bm < 3 : U32) && Bool.not(outer)
      +bt = F.sel(U32.is_eq(bm, 0), F.b2u(twords(b, rows, brs, cols, 1, k)),
        F.sel(U32.is_eq(bm, 1), F.b2u(twords(b, rows, brs, 1, 0, k)),
          F.b2u(twords(b, 1, 0, cols, 1, k))))
      +aok = Bool.not(ra) || none || twords(a, rows, ars, cols, 1, k)
      +bok = Bool.not(rb) || none || U32.is_eq(bt, 1)
      +dvok = U32.is_ne((fl .&. 2 : U32), 0)
        || (U32.is_ne((fl .&. 1 : U32), 0) && twords(a, 1, 0, 1, 0, k))
        || (U32.is_eq((fl .&. 1 : U32), 0) && twords(a, 1, 0, cols, 1, k))
      +dok = (elt && (none || twords(dst, rows, drs, cols, 1, k)))
        || (red && twords(dst, rows, drs, 1, 0, k))
        || (arg && twords(dst, rows, drs, 2, 1, k))
        || (outer && (none || (twords(dst, rows, drs, cols, 1, k) && twords(b, rows, brs, 1, 0, k)
          && twords(imm, 1, 0, cols, 1, k) && dvok)))
      F.sel(outer && (U32.is_ne(bm, 1) || (cols > 256 : U32)), 36,
        F.sel(U32.is_eq(rows, 0), 0,
          F.sel(arg && (rows > 1 : U32) && (drs < 2 : U32), 35,
            F.sel((arg || U32.is_eq(fn, 17)) && none, 39,
              F.sel(aok && bok && dok, 0, 7)))))

def vop.err(bad: Bool, hz: Bool, c: Array<U32>) -> Array<U32>:
  match bad hz:
    case True{} _:
      fail(c, 33, 0)
    case False{} True{}:
      fail(c, 34, 0)
    case False{} False{}:
      c

def vop.z(r: TV, c: Array<U32>, a: Array<U32>, d: Array<U32>, x: Array<U32>, +k: Cf) -> Sl:
  match r:
    case TV{t, h, v, +z}:
      Sl{vop.err(U32.is_ne((z .&. 2 : U32), 0), U32.is_ne((z .&. 1 : U32), 0), c), t, a, d, h, x,
        k}

def vop.zx(r: TX, c: Array<U32>, a: Array<U32>, d: Array<U32>, +k: Cf) -> Sl:
  match r:
    case TX{t, h, x, +z}:
      Sl{vop.err(U32.is_ne((z .&. 2 : U32), 0), U32.is_ne((z .&. 1 : U32), 0), c), t, a, d, h, x,
        k}

def Vg.fn(+v: Vg) -> U32:
  match v:
    case Vg{dst, a, b, rows, cols, drs, ars, brs, +fn, bm, imm, fl}:
      fn

# by class: OUTER, the reductions, ARGMAX, the elementwise functions
def vop.c3(z: Bool, s: Sl, +v: Vg, +g: U32, +fn: U32) -> Sl:
  match z s:
    case True{} Sl{c, t, a, d, h, x, k}:
      vop.z(am.run(TV{t, h, 0, 0}, True{}, v, g), c, a, d, x, k)
    case False{} Sl{c, t, a, d, h, x, k}:
      vop.z(ew(TV{t, h, 0, 0}, v, g, fn), c, a, d, x, k)

def vop.c2(z: Bool, s: Sl, +v: Vg, +g: U32, +fn: U32) -> Sl:
  match z s:
    case True{} Sl{c, t, a, d, h, x, k}:
      vop.zx(rs.go(TX{t, h, x, 0}, v, g, fn), c, a, d, k)
    case False{} s:
      vop.c3(U32.is_eq(fn, 20), s, v, g, fn)

def vop.c1(z: Bool, s: Sl, +v: Vg, +g: U32, +fn: U32) -> Sl:
  match z s:
    case True{} Sl{c, t, a, d, h, x, k}:
      vop.zx(ou.run(t, h, x, v, g), c, a, d, k)
    case False{} s:
      vop.c2((fn >= 16 : U32) && (fn <= 19 : U32), s, v, g, fn)

def vop.g(r: Array<U32> & U32, t: Array<U32>, a: Array<U32>, d: Array<U32>, h: Array<U32>,
  x: Array<U32>, +k: Cf, +v: Vg) -> Sl:
  (c, +g0) = r
  +g = (g0 + 1 : U32)
  +fn = Vg.fn(v)
  c[37] <- g
  vop.c1(U32.is_eq(fn, 6), Sl{c, t, a, d, h, x, k}, v, g, fn)

def vop.e(z: Bool, s: Sl, +v: Vg, +e: U32) -> Sl:
  match z s:
    case True{} Sl{c, t, a, d, h, x, k}:
      vop.g(Array.get(U32, c, VGEN()), t, a, d, h, x, k, v)
    case False{} Sl{c, t, a, d, h, x, k}:
      Sl{fail(c, e, 0), t, a, d, h, x, k}

def x.vop(s: Sl, +i: Ins) -> Sl:
  match s:
    case Sl{c, t, a, d, h, x, +k}:
      match i:
        case Ins{+w0, +w1, +w2, +w3, +w4, +w5, +w6, +w7, +va, +vb, +vc, +vd}:
          +v = {Vg{(va + w1 : U32), (vb + w2 : U32), (vc + w3 : U32), (w4 .&. 65535 : U32),
            (w4 >> 16n : U32), (w5 .&. 65535 : U32), (w5 >> 16n : U32), (w6 .&. 65535 : U32),
            ((w6 >> 16n : U32) .&. 255 : U32), ((w6 >> 24n : U32) .&. 3 : U32), (vd + w7 : U32),
            fl(w0)} : Vg}
          +e = vop.chk(v, k)
          vop.e(U32.is_eq(e, 0), Sl{c, t, a, d, h, x, k}, v, e)

# ------------------------------------------------------------------------------- step

def ex.bad(s: Sl, +i: Ins) -> Sl:
  match s:
    case Sl{c, t, a, d, h, x, k}:
      Sl{fail(c, 2, op(Ins.w0(i))), t, a, d, h, x, k}

# ------------------------------------------------------------------------------- quantizer

# docs/isa.md "Quantization" (isasim Slice._quant_rows, fp32.quantize_mx), a row of n values
# at a time: pass 1 puts the row's values (QACT RSCALE: times T[rscale + r], CSCALE: times
# T[cscale + q]), flushed, in x[0..n) and each group of g its max |x| bits at x[n + q/g], and
# keeps the row's amax; pass 2 turns the groups' maxima into inv * 2^d at x[n + q/g] and d at
# x[n + ng + q/g] (d 0 in ROW mode); then q = q8(x * (inv * 2^d)), S = amax * 1/127 and
# inv = 127 * recip(amax) (both 0 for amax 0).

# pass 1's arrays, the row's amax and the group's max so far
type Qr is Type:
  Qr{t: Array<U32>, x: Array<U32>, amax: U32, gm: U32}

def q1.st(r: Array<U32> & U32, x: Array<U32>, +v: U32, +amax: U32, +gm: U32, +c: U32, +n: U32,
  +g: U32, +rv: U32, +f: U32) -> Qr:
  (t, +cv) = r
  +v1 = F.sel(U32.is_ne((f .&. 4 : U32), 0), F.mul(v, rv), v)
  +v2 = F.ftz(F.sel(U32.is_ne((f .&. 2 : U32), 0), F.mul(v1, cv), v1))
  +ab = (v2 .&. 2147483647 : U32)
  +g0 = F.sel(U32.is_eq((c % g : U32), 0), 0, gm)
  +gm2 = F.sel((ab > g0 : U32), ab, g0)
  x[c] <- v2
  Qr{t, Array.set(U32, x, (n + U32.div(c, g) : U32), gm2), F.sel((ab > amax : U32), ab, amax),
    gm2}

def q1.v(r: Array<U32> & U32, x: Array<U32>, +amax: U32, +gm: U32, +c: U32, +n: U32, +g: U32,
  +rv: U32, +f: U32, +cs: U32, +tm: U32) -> Qr:
  (t, +v) = r
  q1.st(Array.get(U32, t, ((cs + c : U32) .&. tm : U32)), x, v, amax, gm, c, n, g, rv, f)

# pass 1 from element c of the row at TMEM word src (f: QACT's flags; QST's are none of these)
def q1(nn: Nat, st: Qr, +src: U32, +c: U32, +n: U32, +g: U32, +rv: U32, +f: U32, +cs: U32,
  +tm: U32) -> Qr:
  match nn:
    case 0n:
      st
    case 1n+p:
      match st:
        case Qr{t, x, +amax, +gm}:
          q1(p, q1.v(Array.get(U32, t, (src + c : U32)), x, amax, gm, c, n, g, rv, f, cs, tm),
            src, (c + 1 : U32), n, g, rv, f, cs, tm)

# S and inv of a row's amax
def q.s(+amax: U32) -> U32:
  F.sel(U32.is_eq(amax, 0), 0, F.mul(amax, 1006699012))

def q.inv(+amax: U32) -> U32:
  F.sel(U32.is_eq(amax, 0), 0, F.mul(1123942400, F.recip(amax)))

def q2.g(r: Array<U32> & U32, +i: U32, +ng: U32, +amax: U32, +inv: U32, +row: Bool) ->
  Array<U32>:
  (x, +gmax) = r
  +d = F.sel(row, 0, F.mx_offset(gmax, amax))
  x[i] <- F.mul(inv, ((127 + d : U32) << 23n : U32))
  x[(i + ng : U32)] <- d

# pass 2 from group i (x index): inv * 2^d and d
def q2(nn: Nat, x: Array<U32>, +i: U32, +ng: U32, +amax: U32, +inv: U32, +row: Bool) ->
  Array<U32>:
  match nn:
    case 0n:
      x
    case 1n+p:
      q2(p, q2.g(Array.get(U32, x, i), i, ng, amax, inv, row), (i + 1 : U32), ng, amax, inv,
        row)

# element c's q (a byte) or'ed into w at byte i
def qp.e(r: Array<U32> & U32, +invg: U32, +w: U32, +i: U32) -> Array<U32> & U32:
  (x, +v) = r
  (x, (w .|. F.shl((F.q8(F.mul(v, invg)) .&. 255 : U32), (8 * i : U32)) : U32))

# the n elements from c (at most 4, one group) as a word of q bytes
def qp(nn: Nat, st: Array<U32> & U32, +c: U32, +invg: U32, +i: U32) -> Array<U32> & U32:
  match nn:
    case 0n:
      st
    case 1n+p:
      (x, +w) = st
      qp(p, qp.e(Array.get(U32, x, c), invg, w, i), (c + 1 : U32), invg, (i + 1 : U32))

# the quantizer's arrays: TMEM and the scratch, and the ACT RAM (QACT) or DRAM (QST)
type Q3 is Type:
  Q3{t: Array<U32>, m: Array<U32>, x: Array<U32>}

# --- QACT

# QACT's operands: rows from TMEM word src (row stride srs), KB blocks into ACT block ab, the
# flags, the row's n = KB * D values in ng groups of g, TMEM's mask, the ACT RAM's row words,
# blocks, Sa's and the offsets' first words, the RSCALE / CSCALE words, and the rows' second
# copy (DUP: rows on; else 0, the same rows written twice)
type Qa is Data:
  Qa{src: U32, srs: U32, rows: U32, kb: U32, ab: U32, f: U32, n: U32, g: U32, ng: U32,
     tm: U32, rw: U32, nab: U32, aso: U32, ado: U32, rsc: U32, cs: U32, dup: U32, dd: U32}

def qa.wd2(r: Array<U32> & U32, a: Array<U32>, +ai: U32, +ai2: U32) -> Array<U32> & Array<U32>:
  (x, +w) = r
  a[ai] <- w
  (x, Array.set(U32, a, ai2, w))

def qa.wd(r: Array<U32> & U32, a: Array<U32>, +c: U32, +ai: U32, +ai2: U32) ->
  Array<U32> & Array<U32>:
  (x, +invg) = r
  qa.wd2(qp(4n, (x, 0), c, invg, 0), a, ai, ai2)

# the row's q words into the ACT RAM from word ai (and ai2)
def qa.ws(nn: Nat, st: Array<U32> & Array<U32>, +c: U32, +n: U32, +g: U32, +ai: U32,
  +ai2: U32) -> Array<U32> & Array<U32>:
  match nn:
    case 0n:
      st
    case 1n+p:
      (x, a) = st
      qa.ws(p, qa.wd(Array.get(U32, x, (n + U32.div(c, g) : U32)), a, c, ai, ai2),
        (c + 4 : U32), n, g, (ai + 1 : U32), (ai2 + 1 : U32))

def qa.fin(r: Array<U32> & Array<U32>, t: Array<U32>) -> Q3:
  (x, a) = r
  Q3{t, a, x}

# row r's S and offsets (the second copy too)
def qa.sc(r: Array<U32> & Array<U32>, t: Array<U32>, +o: Qa, +row: U32, +s: U32) -> Q3:
  (x, a) = r
  match o:
    case Qa{src, srs, rows, +kb, +ab, f, +n, g, +ng, tm, rw, +nab, +aso, +ado, rsc, cs, +dup,
      dd}:
      +b = (row * nab + ab : U32)
      +b2 = ((row + dup : U32) * nab + ab : U32)
      a2 = fill(U32.to_nat(kb), fill(U32.to_nat(kb), a, (aso + b : U32), s), (aso + b2 : U32),
        s)
      qa.fin(cp(U32.to_nat(ng), cp(U32.to_nat(ng), (x, a2), (n + ng : U32), (ado + 4 * b : U32)),
        (n + ng : U32), (ado + 4 * b2 : U32)), t)

def qa.p2(st: Qr, a: Array<U32>, +o: Qa, +row: U32) -> Q3:
  match st:
    case Qr{t, x, +amax, gm}:
      match o:
        case Qa{src, srs, rows, kb, +ab, +f, +n, +g, +ng, tm, +rw, nab, aso, ado, rsc, cs,
          +dup, +dd}:
          x2 = q2(U32.to_nat(ng), x, n, ng, amax, q.inv(amax), U32.is_ne((f .&. 1 : U32), 0))
          +ai = (row * rw + (ab * dd >> 2n : U32) : U32)
          qa.sc(qa.ws(U32.to_nat((n >> 2n : U32)), (x2, a), 0, n, g, ai,
            (ai + dup * rw : U32)), t, o, row, q.s(amax))

def qa.rv(r: Array<U32> & U32, a: Array<U32>, x: Array<U32>, +o: Qa, +row: U32) -> Q3:
  (t, +rv) = r
  match o:
    case Qa{+src, +srs, rows, kb, ab, +f, +n, +g, ng, +tm, rw, nab, aso, ado, rsc, +cs, dup,
      dd}:
      qa.p2(q1(U32.to_nat(n), Qr{t, x, 0, 0}, (src + row * srs : U32), 0, n, g, rv, f, cs, tm),
        a, o, row)

def qa.row(st: Q3, +o: Qa, +row: U32) -> Q3:
  match st:
    case Q3{t, a, x}:
      match o:
        case Qa{src, srs, rows, kb, ab, f, n, g, ng, +tm, rw, nab, aso, ado, +rsc, cs, dup,
          dd}:
          qa.rv(Array.get(U32, t, ((rsc + row : U32) .&. tm : U32)), a, x, o, row)

def qa.rows(nn: Nat, st: Q3, +o: Qa, +row: U32) -> Q3:
  match nn:
    case 0n:
      st
    case 1n+p:
      qa.rows(p, qa.row(st, o, row), o, (row + 1 : U32))

def qa.done(st: Q3, c: Array<U32>, d: Array<U32>, h: Array<U32>, +k: Cf) -> Sl:
  match st:
    case Q3{t, a, x}:
      Sl{c, t, a, d, h, x, k}

def qa.go(z: Bool, s: Sl, +o: Qa) -> Sl:
  match z s:
    case True{} Sl{c, t, a, d, h, x, k}:
      match o:
        case Qa{src, srs, +rows, kb, ab, f, n, g, ng, tm, rw, nab, aso, ado, rsc, cs, dup, dd}:
          qa.done(qa.rows(U32.to_nat(rows), Q3{t, a, x}, o, 0), c, d, h, k)
    case False{} s:
      s

# QACT's checks: 28 the ACT RAM's rows or blocks, 7 the rows' (RSCALE's, CSCALE's) TMEM words,
# 42 a row longer than the scratch holds; none when a row is empty
def qa.chk(+o: Qa, +k: Cf) -> U32:
  match o:
    case Qa{+src, +srs, +rows, +kb, +ab, +f, +n, g, +ng, +tm, rw, nab, aso, ado, +rsc, +cs,
      dup, dd}:
      +dupf = U32.is_ne((f .&. 8 : U32), 0)
      +big = F.sel(dupf, F.b2u(((2 * rows : U32) > Cf.mcols(k) : U32)),
        F.b2u((rows > Cf.ar(k) : U32)))
      +none = U32.is_eq(rows, 0) || U32.is_eq(kb, 0)
      F.sel(U32.is_ne(big, 0) || ((ab + kb : U32) > Cf.ab(k) : U32), 28,
      F.sel(none, 0,
      F.sel(Bool.not(rng(src, rows, srs, n, 1, tm))
        || (U32.is_ne((f .&. 4 : U32), 0) && Bool.not(rng(rsc, rows, 1, 1, 0, tm)))
        || (U32.is_ne((f .&. 2 : U32), 0) && Bool.not(rng(cs, n, 1, 1, 0, tm))), 7,
      F.sel(((n + 2 * ng : U32) > 4194304 : U32) || (kb > 65535 : U32), 42, 0))))

def qa.e(z: Bool, s: Sl, +o: Qa, +e: U32) -> Sl:
  match z s:
    case True{} s:
      match o:
        case Qa{src, srs, +rows, +kb, ab, f, n, g, ng, tm, rw, nab, aso, ado, rsc, cs, dup,
          dd}:
          qa.go(U32.is_ne(rows, 0) && U32.is_ne(kb, 0), s, o)
    case False{} Sl{c, t, a, d, h, x, k}:
      Sl{fail(c, e, 0), t, a, d, h, x, k}

# QACT: rows of TMEM quantized into the ACT RAM (groups of D/4, ROW: one S, d 0); DUP: also
# into rows rows..2*rows-1
def x.qact(s: Sl, +i: Ins) -> Sl:
  match s:
    case Sl{c, t, a, d, h, x, +k}:
      match i:
        case Ins{+w0, +w1, +w2, +w3, +w4, +w5, w6, w7, +va, vb, vc, vd}:
          +f = fl(w0)
          +dd = Cf.d(k)
          +rows = (w2 .&. 255 : U32)
          +kb = (w2 >> 16n : U32)
          +n = (kb * dd : U32)
          +g = (dd >> 2n : U32)
          +o = {Qa{(va + w1 : U32), w3, rows, kb, ((w2 >> 8n : U32) .&. 255 : U32), f, n, g,
            (4 * kb : U32), (Cf.tw(k) - 1 : U32), ((Cf.ab(k) * dd : U32) >> 2n : U32), Cf.ab(k),
            Cf.aso(k), Cf.ado(k), w5, w4, F.sel(U32.is_ne((f .&. 8 : U32), 0), rows, 0), dd}
            : Qa}
          +e = qa.chk(o, k)
          qa.e(U32.is_eq(e, 0), Sl{c, t, a, d, h, x, k}, o, e)

# --- QST

# QST's operands: rows from TMEM word src (stride srs) quantized (groups of D: a d a block) to
# DRAM bytes dst + r * drs + e * es (e < ne: HALF the row's first half), the scale rows from
# sdst (sw bytes each: S, then the KB bytes d padded to words; ROW: S alone)
type Qs is Data:
  Qs{src: U32, srs: U32, rows: U32, kb: U32, f: U32, n: U32, g: U32, ng: U32, tm: U32,
     dst: U32, drs: U32, es: U32, ne: U32, sdst: U32, sw: U32, msk: U32}

def qs.b(r: Array<U32> & U32, +wi: U32, +bs: U32, +q: U32) -> Array<U32>:
  (d, +w) = r
  d[wi] <- put8(w, bs, q)

def qs.e1(r: Array<U32> & U32, d: Array<U32>, +invg: U32, +b: U32) -> Array<U32> & Array<U32>:
  (x, +v) = r
  +wi = (b >> 2n : U32)
  (x, qs.b(Array.get(U32, d, wi), wi, (b .&. 3 : U32), (F.q8(F.mul(v, invg)) .&. 255 : U32)))

def qs.e0(r: Array<U32> & U32, d: Array<U32>, +e: U32, +b: U32) -> Array<U32> & Array<U32>:
  (x, +invg) = r
  qs.e1(Array.get(U32, x, e), d, invg, b)

# the row's bytes from element e, at DRAM byte b on (step es)
def qs.es(nn: Nat, st: Array<U32> & Array<U32>, +e: U32, +b: U32, +es: U32, +n: U32, +g: U32) ->
  Array<U32> & Array<U32>:
  match nn:
    case 0n:
      st
    case 1n+p:
      (x, d) = st
      qs.es(p, qs.e0(Array.get(U32, x, (n + U32.div(e, g) : U32)), d, e, b), (e + 1 : U32),
        (b + es : U32), es, n, g)

def qs.db(r: Array<U32> & U32, +w: U32, +i: U32, +lim: U32, +sh: U32) -> Array<U32> & U32:
  (x, +v) = r
  (x, (w .|. F.sel((i < lim : U32), F.shl(v, sh), 0) : U32))

# a word of the d bytes from x index i (those from lim on are 0)
def qs.dw(nn: Nat, st: Array<U32> & U32, +i: U32, +lim: U32, +sh: U32) -> Array<U32> & U32:
  match nn:
    case 0n:
      st
    case 1n+p:
      (x, +w) = st
      qs.dw(p, qs.db(Array.get(U32, x, i), w, i, lim, sh), (i + 1 : U32), lim, (sh + 8 : U32))

def qs.dput(r: Array<U32> & U32, d: Array<U32>, +wi: U32) -> Array<U32> & Array<U32>:
  (x, +w) = r
  (x, Array.set(U32, d, wi, w))

# the d words from x index i to DRAM word wi on
def qs.ds(nn: Nat, st: Array<U32> & Array<U32>, +i: U32, +lim: U32, +wi: U32) ->
  Array<U32> & Array<U32>:
  match nn:
    case 0n:
      st
    case 1n+p:
      (x, d) = st
      qs.ds(p, qs.dput(qs.dw(4n, (x, 0), i, lim, 0), d, wi), (i + 4 : U32), lim, (wi + 1 : U32))

def qs.fin(r: Array<U32> & Array<U32>, t: Array<U32>) -> Q3:
  (x, d) = r
  Q3{t, d, x}

# the row's scale row at DRAM word si: S, then nw words of the d bytes (x index i on)
def qs.sr(r: Array<U32> & Array<U32>, t: Array<U32>, +si: U32, +s: U32, +nw: U32, +i: U32,
  +kb: U32) -> Q3:
  (x, d) = r
  qs.fin(qs.ds(U32.to_nat(nw), (x, Array.set(U32, d, si, s)), i, (i + kb : U32),
    (si + 1 : U32)), t)

def qs.p2(st: Qr, d: Array<U32>, +o: Qs, +row: U32) -> Q3:
  match st:
    case Qr{t, x, +amax, gm}:
      match o:
        case Qs{src, srs, rows, +kb, +f, +n, +g, +ng, tm, +dst, +drs, +es, +ne, +sdst, +sw,
          msk}:
          +rowm = U32.is_ne((f .&. 1 : U32), 0)
          x2 = q2(U32.to_nat(ng), x, n, ng, amax, q.inv(amax), rowm)
          qs.sr(qs.es(U32.to_nat(ne), (x2, d), 0, (dst + row * drs : U32), es, n, g), t,
            ((sdst + row * sw : U32) >> 2n : U32), q.s(amax),
            F.sel(rowm, 0, ((kb + 3 : U32) >> 2n : U32)), (n + ng : U32), kb)

def qs.row(st: Q3, +o: Qs, +row: U32) -> Q3:
  match st:
    case Q3{t, d, x}:
      match o:
        case Qs{+src, +srs, rows, kb, f, +n, +g, ng, +tm, dst, drs, es, ne, sdst, sw, msk}:
          qs.p2(q1(U32.to_nat(n), Qr{t, x, 0, 0}, (src + row * srs : U32), 0, n, g, 0, 0, 0, tm),
            d, o, row)

def qs.rows(nn: Nat, st: Q3, +o: Qs, +row: U32) -> Q3:
  match nn:
    case 0n:
      st
    case 1n+p:
      qs.rows(p, qs.row(st, o, row), o, (row + 1 : U32))

# whether a row's data bytes base + e * es (e < ne, all in DRAM) meet the scale rows' bytes
# lo..hi
def qs.hit(+base: U32, +es: U32, +ne: U32, +lo: U32, +hi: U32) -> Bool:
  +gap = (lo - base : U32)
  +q = U32.div(gap, F.sel(U32.is_eq(es, 0), 1, es))
  +em = (q + F.b2u(U32.is_ne((gap % F.sel(U32.is_eq(es, 0), 1, es) : U32), 0)) : U32)
  (base <= hi : U32) && ((base >= lo : U32)
    || (U32.is_ne(es, 0) && (em < ne : U32) && ((base + em * es : U32) <= hi : U32)))

def qs.ov(nn: Nat, +row: U32, +o: Qs, +z: Bool) -> Bool:
  match nn:
    case 0n:
      z
    case 1n+p:
      match o:
        case Qs{src, srs, +rows, kb, f, n, g, ng, tm, +dst, +drs, +es, +ne, +sdst, +sw, msk}:
          qs.ov(p, (row + 1 : U32), o, z || qs.hit((dst + row * drs : U32), es, ne, sdst,
            (sdst + rows * sw - 1 : U32)))

# QST's checks: 29 HALF without ROW, 7 the rows' TMEM words, 30 a data byte past DRAM's end,
# 31 the scale rows' alignment or range, 32 the data and scale bytes overlap, 42 a row longer
# than the scratch holds; none when a row is empty
def qs.chk(+o: Qs, +k: Cf) -> U32:
  match o:
    case Qs{+src, +srs, +rows, +kb, +f, +n, g, +ng, +tm, +dst, +drs, +es, +ne, +sdst, +sw, msk}:
      +db = Cf.dbm1(k)
      F.sel(U32.is_ne((f .&. 2 : U32), 0) && U32.is_eq((f .&. 1 : U32), 0), 29,
      F.sel(U32.is_eq(rows, 0) || U32.is_eq(kb, 0), 0,
      F.sel(Bool.not(rng(src, rows, srs, n, 1, tm)), 7,
      F.sel(Bool.not(rng(dst, rows, drs, ne, es, db)), 30,
      F.sel(U32.is_ne((sdst .&. 3 : U32), 0) || Bool.not(rng(sdst, rows, sw, sw, 1, db)), 31,
      F.sel(qs.ov(U32.to_nat(rows), 0, o, False{}), 32,
      F.sel(((n + 2 * ng + 4 : U32) > 4194304 : U32) || (kb > 32767 : U32), 42, 0)))))))

def qs.done(st: Q3, c: Array<U32>, a: Array<U32>, h: Array<U32>, +k: Cf) -> Sl:
  match st:
    case Q3{t, d, x}:
      Sl{c, t, a, d, h, x, k}

def qs.go(z: Bool, s: Sl, +o: Qs) -> Sl:
  match z s:
    case True{} Sl{c, t, a, d, h, x, k}:
      match o:
        case Qs{src, srs, +rows, kb, f, n, g, ng, tm, dst, drs, es, ne, sdst, sw, msk}:
          qs.done(qs.rows(U32.to_nat(rows), Q3{t, d, x}, o, 0), c, a, h, k)
    case False{} s:
      s

def qs.e(z: Bool, s: Sl, +o: Qs, +e: U32) -> Sl:
  match z s:
    case True{} s:
      match o:
        case Qs{src, srs, +rows, +kb, f, n, g, ng, tm, dst, drs, es, ne, sdst, sw, msk}:
          qs.go(U32.is_ne(rows, 0) && U32.is_ne(kb, 0), s, o)
    case False{} Sl{c, t, a, d, h, x, k}:
      Sl{fail(c, e, 0), t, a, d, h, x, k}

# QST: rows of TMEM quantized to DRAM (int8 rows and their scale rows: docs/isa.md "QST")
def x.qst(s: Sl, +i: Ins) -> Sl:
  match s:
    case Sl{c, t, a, d, h, x, +k}:
      match i:
        case Ins{+w0, +w1, +w2, +w3, +w4, +w5, +w6, +w7, +va, +vb, +vc, vd}:
          +f = fl(w0)
          +dd = Cf.d(k)
          +kb = (w4 >> 16n : U32)
          +n = (kb * dd : U32)
          +rowm = U32.is_ne((f .&. 1 : U32), 0)
          +o = {Qs{(va + w1 : U32), w5, (w4 .&. 65535 : U32), kb, f, n, dd, kb,
            (Cf.tw(k) - 1 : U32), (vb + w2 : U32), w6, w7,
            F.sel(U32.is_ne((f .&. 2 : U32), 0), (n >> 1n : U32), n), (vc + w3 : U32),
            F.sel(rowm, 4, (4 + 4 * ((kb + 3 : U32) >> 2n : U32) : U32)),
            (Cf.dbm1(k) >> 2n : U32)} : Qs}
          +e = qs.chk(o, k)
          qs.e(U32.is_eq(e, 0), Sl{c, t, a, d, h, x, k}, o, e)

# DSTEP (docs/isa.md "DSTEP"; isasim Slice._dstep): one Gated DeltaNet head step on a state S
# of rows x cols fp32 in DRAM from byte d (DZERO: +0, not read): kv = S k, dd = (v - kv * e) *
# beta, S = outer(S, e, dd, k) (S * e + dd k, fp32.outer), o = S q; the sums are VOP RDOT's (64
# interleaved partials from +0, then the folding tree). q and k (TMEM qk, qk + cols), v, e
# (T[g]) and beta (T[g + gs]) are read first; the state is written back, then o to T[o..].
#   the scratch: x[0..64) the partials, q from 64 (k after it), v at 576, o at 832, the state
#   from 2048

def ds.p3(r: Array<U32> & U32, +j: U32, +u: U32) -> Array<U32>:
  (x, +p) = r
  x[j] <- F.add(p, u)

def ds.p2(r: Array<U32> & U32, +j: U32, +sv: U32) -> Array<U32>:
  (x, +vv) = r
  ds.p3(Array.get(U32, x, j), j, F.mul(sv, vv))

def ds.p1(r: Array<U32> & U32, +vi: U32, +j: U32) -> Array<U32>:
  (x, +sv) = r
  ds.p2(Array.get(U32, x, vi), j, sv)

def ds.dot(n: Nat, x: Array<U32>, +si: U32, +vi: U32, +q: U32) -> Array<U32>:
  match n:
    case 0n:
      x
    case 1n+p:
      ds.dot(p, ds.p1(Array.get(U32, x, si), vi, (q .&. 63 : U32)), (si + 1 : U32), (vi + 1 : U32),
        (q + 1 : U32))

# RDOT of the scratch's n words at si with those at vi (n a multiple of 64: no padding)
def ds.rdot(x: Array<U32>, +si: U32, +vi: U32, +n: U32) -> Array<U32> & U32:
  Array.get(U32, tree(6n, ds.dot(U32.to_nat(n), fill(64n, x, 0, 0), si, vi, 0), 32), 0)

def ds.u2(r: Array<U32> & U32, +i: U32, +sv: U32, +e: U32, +dd: U32) -> Array<U32>:
  (x, +kv) = r
  x[i] <- F.outer(sv, e, dd, kv)

def ds.u1(r: Array<U32> & U32, +i: U32, +ki: U32, +e: U32, +dd: U32) -> Array<U32>:
  (x, +sv) = r
  ds.u2(Array.get(U32, x, ki), i, sv, e, dd)

def ds.upd(n: Nat, x: Array<U32>, +i: U32, +ki: U32, +e: U32, +dd: U32) -> Array<U32>:
  match n:
    case 0n:
      x
    case 1n+p:
      ds.upd(p, ds.u1(Array.get(U32, x, i), i, ki, e, dd), (i + 1 : U32), (ki + 1 : U32), e, dd)

def ds.row3(r: Array<U32> & U32, +oi: U32) -> Array<U32>:
  (x, +o) = r
  x[oi] <- o

def ds.row2(r: Array<U32> & U32, +si: U32, +cols: U32, +vr: U32, +e: U32, +beta: U32,
  +oi: U32) -> Array<U32>:
  (x, +kv) = r
  ds.row3(ds.rdot(ds.upd(U32.to_nat(cols), x, si, (64 + cols : U32), e,
    F.mul(F.sub(vr, F.mul(kv, e)), beta)), si, 64, cols), oi)

def ds.row1(r: Array<U32> & U32, +si: U32, +cols: U32, +e: U32, +beta: U32, +oi: U32) ->
  Array<U32>:
  (x, +vr) = r
  ds.row2(ds.rdot(x, si, (64 + cols : U32), cols), si, cols, vr, e, beta, oi)

def ds.rows(n: Nat, x: Array<U32>, +r: U32, +cols: U32, +e: U32, +beta: U32) -> Array<U32>:
  match n:
    case 0n:
      x
    case 1n+p:
      ds.rows(p, ds.row1(Array.get(U32, x, (576 + r : U32)), (2048 + r * cols : U32), cols, e,
        beta, (832 + r : U32)), (r + 1 : U32), cols, e, beta)

# the operands: the state's DRAM word, q / k's, v's, e's and o's TMEM words, gs, rows, cols,
# DZERO
type Dg is Data:
  Dg{dw: U32, qk: U32, v: U32, g: U32, o: U32, gs: U32, rows: U32, cols: U32, zero: U32}

def ds.fin(r: Array<U32> & Array<U32>, c: Array<U32>, a: Array<U32>, d: Array<U32>,
  h: Array<U32>, +k: Cf) -> Sl:
  (x, t) = r
  Sl{c, t, a, d, h, x, k}

# o to TMEM (its range: error 7, after the state is written, as isasim's)
def ds.done(z: Bool, x: Array<U32>, t: Array<U32>, c: Array<U32>, a: Array<U32>,
  d: Array<U32>, h: Array<U32>, +k: Cf, +o: U32, +rows: U32) -> Sl:
  match z:
    case True{}:
      ds.fin(cp(U32.to_nat(rows), (x, t), 832, o), c, a, d, h, k)
    case False{}:
      Sl{fail(c, 7, o), t, a, d, h, x, k}

def ds.go4(r: Array<U32> & Array<U32>, c: Array<U32>, t: Array<U32>, a: Array<U32>,
  h: Array<U32>, +k: Cf, +g: Dg) -> Sl:
  (x, d) = r
  match g:
    case Dg{dw, qk, v, gg, +o, gs, +rows, cols, zero}:
      ds.done(rng(o, rows, 1, 1, 0, (Cf.tw(k) - 1 : U32)), x, t, c, a, d, h, k, o, rows)

# the rows, then the state back to DRAM
def ds.go3(x: Array<U32>, d: Array<U32>, c: Array<U32>, t: Array<U32>, a: Array<U32>,
  h: Array<U32>, +k: Cf, +g: Dg, +e: U32, +beta: U32) -> Sl:
  match g:
    case Dg{+dw, qk, v, gg, o, gs, +rows, +cols, zero}:
      ds.go4(cp(U32.to_nat((rows * cols : U32)), (ds.rows(U32.to_nat(rows), x, 0, cols, e, beta),
        d), 2048, dw), c, t, a, h, k, g)

def ds.go2b(r: Array<U32> & Array<U32>, c: Array<U32>, t: Array<U32>, a: Array<U32>,
  h: Array<U32>, +k: Cf, +g: Dg, +e: U32, +beta: U32) -> Sl:
  (d, x) = r
  ds.go3(x, d, c, t, a, h, k, g, e, beta)

# the state: +0 (DZERO) or DRAM's
def ds.go2(z: Bool, x: Array<U32>, d: Array<U32>, c: Array<U32>, t: Array<U32>,
  a: Array<U32>, h: Array<U32>, +k: Cf, +g: Dg, +e: U32, +beta: U32) -> Sl:
  match z:
    case True{}:
      match g:
        case Dg{dw, qk, v, gg, o, gs, +rows, +cols, zero}:
          ds.go3(fill(U32.to_nat((rows * cols : U32)), x, 2048, 0), d, c, t, a, h, k, g, e, beta)
    case False{}:
      match g:
        case Dg{+dw, qk, v, gg, o, gs, +rows, +cols, zero}:
          ds.go2b(cp(U32.to_nat((rows * cols : U32)), (d, x), dw, 2048), c, t, a, h, k, g, e,
            beta)

def ds.eb2(r: Array<U32> & U32, x: Array<U32>, c: Array<U32>, a: Array<U32>, d: Array<U32>,
  h: Array<U32>, +k: Cf, +g: Dg, +e: U32) -> Sl:
  (t, +beta) = r
  match g:
    case Dg{dw, qk, v, gg, o, gs, rows, cols, +zero}:
      ds.go2(U32.is_ne(zero, 0), x, d, c, t, a, h, k, g, e, beta)

def ds.eb1(r: Array<U32> & U32, x: Array<U32>, c: Array<U32>, a: Array<U32>, d: Array<U32>,
  h: Array<U32>, +k: Cf, +g: Dg) -> Sl:
  (t, +e) = r
  match g:
    case Dg{dw, qk, v, +gg, o, +gs, rows, cols, zero}:
      ds.eb2(Array.get(U32, t, (gg + gs : U32)), x, c, a, d, h, k, g, e)

def ds.ld2(r: Array<U32> & Array<U32>, c: Array<U32>, a: Array<U32>, d: Array<U32>,
  h: Array<U32>, +k: Cf, +g: Dg) -> Sl:
  (t, x) = r
  match g:
    case Dg{dw, qk, v, +gg, o, gs, rows, cols, zero}:
      ds.eb1(Array.get(U32, t, gg), x, c, a, d, h, k, g)

# q and k, then v, into the scratch
def ds.ld(r: Array<U32> & Array<U32>, c: Array<U32>, a: Array<U32>, d: Array<U32>,
  h: Array<U32>, +k: Cf, +g: Dg) -> Sl:
  (t, x) = r
  match g:
    case Dg{dw, qk, +v, gg, o, gs, +rows, cols, zero}:
      ds.ld2(cp(U32.to_nat(rows), (t, x), v, 576), c, a, d, h, k, g)

def ds.e(z: Bool, s: Sl, +g: Dg, +e: U32) -> Sl:
  match z s:
    case True{} Sl{c, t, a, d, h, x, +k}:
      match g:
        case Dg{dw, +qk, v, gg, o, gs, rows, +cols, zero}:
          ds.ld(cp(U32.to_nat((2 * cols : U32)), (t, x), qk, 64), c, a, d, h, k, g)
    case False{} Sl{c, t, a, d, h, x, k}:
      Sl{fail(c, e, 0), t, a, d, h, x, k}

# DSTEP's checks: 37 rows 1..256 and cols 64, 128, 192 or 256, 43 the state's alignment (D
# bytes), 7 q, k, v, e or beta past TMEM, 6 the state past DRAM's end
def x.dstep(s: Sl, +i: Ins) -> Sl:
  match s:
    case Sl{c, t, a, d, h, x, +k}:
      match i:
        case Ins{+w0, +w1, +w2, +w3, +w4, +w5, +w6, +w7, +va, +vb, +vc, vd}:
          +rows = (w4 .&. 65535 : U32)
          +cols = (w4 >> 16n : U32)
          +dd = (va + w1 : U32)
          +qk = (vb + w2 : U32)
          +v = (vc + w3 : U32)
          +tm = (Cf.tw(k) - 1 : U32)
          +e = F.sel(U32.is_eq(rows, 0) || (rows > 256 : U32) || U32.is_ne((cols % 64 : U32), 0)
            || U32.is_eq(cols, 0) || (cols > 256 : U32), 37,
            F.sel(U32.is_ne((dd % Cf.d(k) : U32), 0), 43,
            F.sel(Bool.not(rng(qk, (2 * cols : U32), 1, 1, 0, tm)) || Bool.not(rng(v, rows, 1, 1,
              0, tm)) || (w5 > tm : U32) || Bool.not(rng(w5, 2, w7, 1, 0, tm)), 7,
            F.sel(Bool.not(rng(dd, (rows * cols : U32), 4, 4, 1, Cf.dbm1(k))), 6, 0))))
          ds.e(U32.is_eq(e, 0), Sl{c, t, a, d, h, x, k},
            {Dg{(dd >> 2n : U32), qk, v, w5, w6, w7, rows, cols, (fl(w0) .&. 1 : U32)} : Dg}, e)

# a row's max (VOP RMAX's total order, from the smallest key's word): MM's RMAX, STREAM's A_MAX
def rm.step(r: Array<U32> & U32, +mx: U32) -> Array<U32> & U32:
  (x, +v) = r
  (x, F.fmax(mx, v))

def rm.row(n: Nat, st: Array<U32> & U32, +i: U32) -> Array<U32> & U32:
  match n:
    case 0n:
      st
    case 1n+p:
      (x, +mx) = st
      rm.row(p, rm.step(Array.get(U32, x, i), mx), (i + 1 : U32))

# --- STREAM (docs/stream.md; isasim Slice._stream): the rows S[r] of a stream (DRAM words or
# TMEM, +0 with SZERO) pass, under a descriptor of TMEM words, REDUCE A (A = isum_64(S[r] *
# opA), opA a column slot, S itself or a constant; or the row's max), the SCALAR ops over
# r0..r7, A, X[r], K0..K3, +0, +1 (r4..r7 start as K0..K3 where rinit has their bit, the
# registers carry from row to row), MAP U (Y = S | S*G + D*B | S*G | S + D*B, D = r[d_reg]) and
# Q (O[r] = isum_64(Y * Q), Q a slot or Y; or O[r] = r[o_reg]); the row written back is Y
# (out_p2: Y * slot q). The slots, constants and row scalars are read first, O and the saved
# r4..r7 (rsave) are written last, and every check is made before anything is written.
#
# The rows run one at a time, each read before it is written: isasim's order where the source
# and destination meet, and the same result where they do not. A TMEM stream written in place
# whose rows overlap one another (stride < cols) is read whole before any row is written in
# isasim: those rows' Y wait in the scratch and are written after the last row.
#
# The scratch: the partials x[0..64), r0..r7 at 64, K0..K3 at 72, the descriptor's payloads
# from 80 (its ops from 85), O from 128, X from 4224, the slots from 8320 (slot i at 8320 +
# i * cols), the row S at 24704, Y at 28800, the waiting rows from 32896.

def sq.ok(+w: U32) -> Bool:
  U32.is_eq(((w >> 23n : U32) .&. 255 : U32), 128)

# a float-safe word's 24-bit payload (isa._payload): the sign bit over the 23 mantissa bits
def sq.pl(+w: U32) -> U32:
  (((w >> 31n : U32) << 23n : U32) .|. (w .&. 8388607 : U32) : U32)

def sq.fs1(r: Array<U32> & U32, +i: U32, +bad: U32) -> Array<U32> & U32:
  (x, +w) = r
  (Array.set(U32, x, i, sq.pl(w)), (bad .|. F.b2u(Bool.not(sq.ok(w))) : U32))

# the n words from x[i] to payloads; 1 when one of them is not float-safe
def sq.fs(n: Nat, st: Array<U32> & U32, +i: U32) -> Array<U32> & U32:
  match n:
    case 0n:
      st
    case 1n+p:
      (x, +bad) = st
      sq.fs(p, sq.fs1(Array.get(U32, x, i), i, bad), (i + 1 : U32))

def sq.bits(+w: U32, +k: U32, +n: U32) -> U32:
  (F.shr(w, k) .&. (F.shl(1, n) - 1 : U32) : U32)

# an operand's reads: K0..K3 (bits 4..7), X (bit 1)
def sq.opnd(+c: U32) -> U32:
  F.sel(((c >= 10 : U32) && (c <= 13 : U32)), F.shl(16, (c - 10 : U32)),
    F.sel(U32.is_eq(c, 9), 2, 0))

# an op: bit 0 when it is past EXP2SUB, its operands' reads (b only for a binary op)
def sq.op1(r: Array<U32> & U32, +f: U32) -> Array<U32> & U32:
  (x, +p) = r
  +op = (p .&. 15 : U32)
  +rb = F.sel(((op >= 5 : U32) && (op <= 9 : U32)), 0, sq.opnd(sq.bits(p, 11, 4)))
  (x, (((f .|. F.b2u((op > 10 : U32)) : U32) .|. sq.opnd(sq.bits(p, 7, 4)) : U32) .|. rb : U32))

def sq.ops(n: Nat, st: Array<U32> & U32, +i: U32) -> Array<U32> & U32:
  match n:
    case 0n:
      st
    case 1n+p:
      (x, +f) = st
      sq.ops(p, sq.op1(Array.get(U32, x, i), f), (i + 1 : U32))

# the number of slots or constants from 0 to the highest one in m's bits 0..3
def sq.top(+m: U32) -> U32:
  F.sel(U32.is_ne((m .&. 8 : U32), 0), 4, F.sel(U32.is_ne((m .&. 4 : U32), 0), 3,
    F.sel(U32.is_ne((m .&. 2 : U32), 0), 2, (m .&. 1 : U32))))

# --- the rows

# the row loop's operands: cols, the descriptor's c1 and c2, the source's and destination's
# first word and row stride (DRAM words or TMEM), the mode (bit 0 S = +0, 1 the source is TMEM,
# 2 the destination is TMEM, 3 no destination, 4 the rows wait in the scratch)
type Sr is Data:
  Sr{cols: U32, c1: U32, c2: U32, src: U32, ss: U32, dst: U32, ds: U32, md: U32}

def sq.lt(r: Array<U32> & Array<U32>, c: Array<U32>, a: Array<U32>, d: Array<U32>,
  h: Array<U32>, +k: Cf) -> Sl:
  (t, x) = r
  Sl{c, t, a, d, h, x, k}

def sq.ld(r: Array<U32> & Array<U32>, c: Array<U32>, t: Array<U32>, a: Array<U32>,
  h: Array<U32>, +k: Cf) -> Sl:
  (d, x) = r
  Sl{c, t, a, d, h, x, k}

def sq.st(r: Array<U32> & Array<U32>, c: Array<U32>, a: Array<U32>, d: Array<U32>,
  h: Array<U32>, +k: Cf) -> Sl:
  (x, t) = r
  Sl{c, t, a, d, h, x, k}

def sq.sd(r: Array<U32> & Array<U32>, c: Array<U32>, t: Array<U32>, a: Array<U32>,
  h: Array<U32>, +k: Cf) -> Sl:
  (x, d) = r
  Sl{c, t, a, d, h, x, k}

# row r of the stream to S: +0, TMEM's or DRAM's
def sq.load(z: Bool, s: Sl, y: Bool, +m: Sr, +r: U32) -> Sl:
  match z s:
    case True{} Sl{c, t, a, d, h, x, +k}:
      match m:
        case Sr{+cols, c1, c2, src, ss, dst, ds, md}:
          Sl{c, t, a, d, h, fill(U32.to_nat(cols), x, 24704, 0), k}
    case False{} Sl{c, t, a, d, h, x, +k}:
      match y m:
        case True{} Sr{+cols, c1, c2, +src, +ss, dst, ds, md}:
          sq.lt(cp(U32.to_nat(cols), (t, x), (src + r * ss : U32), 24704), c, a, d, h, k)
        case False{} Sr{+cols, c1, c2, +src, +ss, dst, ds, md}:
          sq.ld(cp(U32.to_nat(cols), (d, x), (src + r * ss : U32), 24704), c, t, a, h, k)

# Y to the destination: none, the scratch (the rows wait), TMEM or DRAM
def sq.store(z: Bool, s: Sl, y: Bool, +m: Sr, w: Bool, +r: U32) -> Sl:
  match z s:
    case True{} Sl{c, t, a, d, h, x, k}:
      Sl{c, t, a, d, h, x, k}
    case False{} Sl{c, t, a, d, h, x, +k}:
      match y m:
        case True{} Sr{+cols, c1, c2, src, ss, dst, ds, md}:
          Sl{c, t, a, d, h, cp1(U32.to_nat(cols), x, 28800, (32896 + r * cols : U32)), k}
        case False{} Sr{+cols, c1, c2, src, ss, +dst, +ds, md}:
          match w:
            case True{}:
              sq.st(cp(U32.to_nat(cols), (x, t), 28800, (dst + r * ds : U32)), c, a, d, h, k)
            case False{}:
              sq.sd(cp(U32.to_nat(cols), (x, d), 28800, (dst + r * ds : U32)), c, t, a, h, k)

# the row's sums: the partials (from +0), the terms s[c] * v[c] (v the words from vi, or the
# scalar k when vs is 0), the +0 padding, the tree (fp32.rdot)
def sq.t2(r: Array<U32> & U32, +j: U32, +u: U32) -> Array<U32>:
  (x, +p) = r
  x[j] <- F.add(p, u)

def sq.t1(r: Array<U32> & U32, +sv: U32, +j: U32, +vs: U32, +k: U32) -> Array<U32>:
  (x, +v) = r
  sq.t2(Array.get(U32, x, j), j, F.mul(sv, F.sel(U32.is_eq(vs, 0), k, v)))

def sq.t0(r: Array<U32> & U32, +vi: U32, +j: U32, +vs: U32, +k: U32) -> Array<U32>:
  (x, +sv) = r
  sq.t1(Array.get(U32, x, vi), sv, j, vs, k)

def sq.cols(n: Nat, x: Array<U32>, +si: U32, +vi: U32, +vs: U32, +k: U32, +q: U32) ->
  Array<U32>:
  match n:
    case 0n:
      x
    case 1n+p:
      sq.cols(p, sq.t0(Array.get(U32, x, si), vi, (q .&. 63 : U32), vs, k), (si + 1 : U32),
        (vi + vs : U32), vs, k, (q + 1 : U32))

def sq.sum(x: Array<U32>, +si: U32, +vi: U32, +vs: U32, +k: U32, +n: U32) ->
  Array<U32> & U32:
  Array.get(U32, tree(6n, pad(U32.to_nat(((64 - (n .&. 63 : U32) : U32) .&. 63 : U32)),
    sq.cols(U32.to_nat(n), fill(64n, x, 0, 0), si, vi, vs, k, 0), n), 32), 0)

# A_SLOT, A_SELF, A_CONST: the row's dot with slot a_idx, itself or K[a_idx]
def sq.ak(r: Array<U32> & U32, +c1: U32, +cols: U32) -> Array<U32> & U32:
  (x, +kv) = r
  +aop = sq.bits(c1, 1, 2)
  sq.sum(x, 24704, F.sel(U32.is_eq(aop, 0), (8320 + sq.bits(c1, 3, 2) * cols : U32), 24704),
    F.b2u(U32.is_ne(aop, 2)), kv, cols)

# A: +0 when off; the row's max (VOP RMAX's total order); its dot
def sq.a(z: Bool, y: Bool, x: Array<U32>, +c1: U32, +cols: U32) -> Array<U32> & U32:
  match z:
    case True{}:
      match y:
        case True{}:
          rm.row(U32.to_nat(cols), (x, 4294967295), 24704)
        case False{}:
          sq.ak(Array.get(U32, x, (72 + sq.bits(c1, 3, 2) : U32)), c1, cols)
    case False{}:
      (x, 0)

# --- SCALAR: dst = op(a, b), each op evaluating only its own function
def sq.f10(+p: U32, +q: U32) -> U32:
  F.exp2(F.sub(p, q))

def sq.f9(z: Bool, +op: U32, +p: U32, +q: U32) -> U32:
  match z:
    case True{}:
      F.log2(p)
    case False{}:
      sq.f10(p, q)

def sq.f8(z: Bool, +op: U32, +p: U32, +q: U32) -> U32:
  match z:
    case True{}:
      F.exp2(p)
    case False{}:
      sq.f9(U32.is_eq(op, 9), op, p, q)

def sq.f7(z: Bool, +op: U32, +p: U32, +q: U32) -> U32:
  match z:
    case True{}:
      F.recip(p)
    case False{}:
      sq.f8(U32.is_eq(op, 8), op, p, q)

def sq.f6(z: Bool, +op: U32, +p: U32, +q: U32) -> U32:
  match z:
    case True{}:
      F.rsqrt(p)
    case False{}:
      sq.f7(U32.is_eq(op, 7), op, p, q)

def sq.f5(z: Bool, +op: U32, +p: U32, +q: U32) -> U32:
  match z:
    case True{}:
      F.ftz(p)
    case False{}:
      sq.f6(U32.is_eq(op, 6), op, p, q)

def sq.f4(z: Bool, +op: U32, +p: U32, +q: U32) -> U32:
  match z:
    case True{}:
      F.fmin(p, q)
    case False{}:
      sq.f5(U32.is_eq(op, 5), op, p, q)

def sq.f3(z: Bool, +op: U32, +p: U32, +q: U32) -> U32:
  match z:
    case True{}:
      F.fmax(p, q)
    case False{}:
      sq.f4(U32.is_eq(op, 4), op, p, q)

def sq.f2(z: Bool, +op: U32, +p: U32, +q: U32) -> U32:
  match z:
    case True{}:
      F.mul(p, q)
    case False{}:
      sq.f3(U32.is_eq(op, 3), op, p, q)

def sq.f1(z: Bool, +op: U32, +p: U32, +q: U32) -> U32:
  match z:
    case True{}:
      F.sub(p, q)
    case False{}:
      sq.f2(U32.is_eq(op, 2), op, p, q)

def sq.f(z: Bool, +op: U32, +p: U32, +q: U32) -> U32:
  match z:
    case True{}:
      F.add(p, q)
    case False{}:
      sq.f1(U32.is_eq(op, 1), op, p, q)

# an operand's value from its word in the scratch: r0..r7, A, X[r], K0..K3, +0, +1
def sq.val(r: Array<U32> & U32, +a: U32, +xv: U32, +c: U32) -> Array<U32> & U32:
  (x, +v) = r
  (x, F.sel((c < 8 : U32), v, F.sel(U32.is_eq(c, 8), a, F.sel(U32.is_eq(c, 9), xv,
    F.sel((c < 14 : U32), v, F.sel(U32.is_eq(c, 14), 0, 1065353216))))))

def sq.get(x: Array<U32>, +a: U32, +xv: U32, +c: U32) -> Array<U32> & U32:
  sq.val(Array.get(U32, x, F.sel((c < 8 : U32), (64 + c : U32), (62 + (c .&. 15 : U32) : U32))),
    a, xv, c)

def sq.ob(r: Array<U32> & U32, +op: U32, +dst: U32, +pa: U32) -> Array<U32>:
  (x, +pb) = r
  x[(64 + dst : U32)] <- sq.f(U32.is_eq(op, 0), op, pa, pb)

def sq.oa(r: Array<U32> & U32, +a: U32, +xv: U32, +op: U32, +dst: U32, +cb: U32) ->
  Array<U32>:
  (x, +pa) = r
  sq.ob(sq.get(x, a, xv, cb), op, dst, pa)

def sq.op(r: Array<U32> & U32, +a: U32, +xv: U32) -> Array<U32>:
  (x, +p) = r
  sq.oa(sq.get(x, a, xv, sq.bits(p, 7, 4)), a, xv, (p .&. 15 : U32), sq.bits(p, 4, 3),
    sq.bits(p, 11, 4))

# the scalar ops from op i on, for a row with A = a and X[r] = xv
def sq.sc(n: Nat, x: Array<U32>, +i: U32, +a: U32, +xv: U32) -> Array<U32>:
  match n:
    case 0n:
      x
    case 1n+p:
      sq.sc(p, sq.op(Array.get(U32, x, (85 + i : U32)), a, xv), (i + 1 : U32), a, xv)

# --- MAP U: Y[c] from S[c], G and B (a slot's word or the row's scalar)
def sq.u(z: Bool, y: Bool, w: Bool, +sv: U32, +g: U32, +dv: U32, +b: U32) -> U32:
  match z:
    case True{}:
      F.ftz(sv)
    case False{}:
      match y:
        case True{}:
          F.outer(sv, g, dv, b)
        case False{}:
          match w:
            case True{}:
              F.mul(sv, g)
            case False{}:
              F.add(sv, F.mul(dv, b))

# a row's registers after its scalar ops: D, G's and B's scalars, O's register
type Rg is Data:
  Rg{dv: U32, gv: U32, bv: U32, ov: U32}

def sq.ub(r: Array<U32> & U32, +yi: U32, +sv: U32, +g: U32, +um: U32, +bsl: U32, +gg: Rg) ->
  Array<U32>:
  (x, +bc) = r
  match gg:
    case Rg{+dv, gv, +bv, ov}:
      x[yi] <- sq.u(U32.is_eq(um, 0), U32.is_eq(um, 1), U32.is_eq(um, 2), sv, g, dv,
        F.sel(U32.is_ne(bsl, 0), bc, bv))

def sq.ug(r: Array<U32> & U32, +yi: U32, +sv: U32, +gsl: U32, +um: U32, +bi: U32, +bsl: U32,
  +gg: Rg) -> Array<U32>:
  (x, +gc) = r
  match gg:
    case Rg{dv, +gv, bv, ov}:
      sq.ub(Array.get(U32, x, bi), yi, sv, F.sel(U32.is_ne(gsl, 0), gc, gv), um, bsl, gg)

def sq.us(r: Array<U32> & U32, +yi: U32, +gi: U32, +gsl: U32, +um: U32, +bi: U32, +bsl: U32,
  +gg: Rg) -> Array<U32>:
  (x, +sv) = r
  sq.ug(Array.get(U32, x, gi), yi, sv, gsl, um, bi, bsl, gg)

# Y[q..] from S at 24704 + q, G's slot at gi + q (its scalar when gsl is 0), B's at bi + q
def sq.map(n: Nat, x: Array<U32>, +q: U32, +gi: U32, +gsl: U32, +um: U32, +bi: U32, +bsl: U32,
  +gg: Rg) -> Array<U32>:
  match n:
    case 0n:
      x
    case 1n+p:
      sq.map(p, sq.us(Array.get(U32, x, (24704 + q : U32)), (28800 + q : U32), (gi + q : U32),
        gsl, um, (bi + q : U32), bsl, gg), (q + 1 : U32), gi, gsl, um, bi, bsl, gg)

# out_p2: Y[c] *= slot q's word
def sq.p2c(r: Array<U32> & U32, +yi: U32, +qv: U32) -> Array<U32>:
  (x, +y) = r
  x[yi] <- F.mul(y, qv)

def sq.p2q(r: Array<U32> & U32, +yi: U32) -> Array<U32>:
  (x, +qv) = r
  sq.p2c(Array.get(U32, x, yi), yi, qv)

def sq.p2(n: Nat, x: Array<U32>, +yi: U32, +qi: U32) -> Array<U32>:
  match n:
    case 0n:
      x
    case 1n+p:
      sq.p2(p, sq.p2q(Array.get(U32, x, qi), yi), (yi + 1 : U32), (qi + 1 : U32))

def sq.put(r: Array<U32> & U32, +oi: U32) -> Array<U32>:
  (x, +o) = r
  x[oi] <- o

# O[r] to the scratch: Q's dot (q_en, Q slot q_idx or Y itself with q_self), the register
def sq.o(z: Bool, y: Bool, x: Array<U32>, +c1: U32, +c2: U32, +cols: U32, +ov: U32,
  +r: U32) -> Array<U32>:
  match z:
    case True{}:
      sq.put(sq.sum(x, 28800, F.sel(U32.is_ne(sq.bits(c2, 12, 1), 0), 28800,
        (8320 + sq.bits(c1, 21, 2) * cols : U32)), 1, 0, cols), (128 + r : U32))
    case False{}:
      match y:
        case True{}:
          x[(128 + r : U32)] <- ov
        case False{}:
          x

# Y, O and out_p2 for the row
def sq.y(+m: Sr, +row: U32, r: Array<U32> & Rg) -> Array<U32>:
  match m:
    case Sr{+cols, +c1, +c2, src, ss, dst, ds, md}:
      (x, +gg) = r
      match gg:
        case Rg{dv, gv, bv, +ov}:
          +gsrc = sq.bits(c1, 7, 2)
          sq.p2(U32.to_nat(F.sel(U32.is_ne(sq.bits(c1, 23, 1), 0), cols, 0)),
            sq.o(U32.is_ne(sq.bits(c1, 20, 1), 0), U32.is_ne(sq.bits(c2, 13, 1), 0),
              sq.map(U32.to_nat(cols), x, 0, (8320 + sq.bits(c1, 9, 3) * cols : U32),
                F.b2u(U32.is_eq(gsrc, 1)), sq.bits(c1, 5, 2),
                (8320 + sq.bits(c1, 14, 3) * cols : U32),
                F.b2u(U32.is_eq(sq.bits(c1, 12, 2), 0)), gg), c1, c2, cols, ov, row), 28800,
            (8320 + sq.bits(c1, 21, 2) * cols : U32))

# the row's scalars after its ops: D = r[d_reg]; G r[g_idx], K[g_idx] or 1.0; B K[b_idx] or
# r[b_idx]; O's r[o_reg]
def sq.r6(r: Array<U32> & U32, +dv: U32, +gv: U32, +bsrc: U32, +br: U32, +ov: U32) ->
  Array<U32> & Rg:
  (x, +kb) = r
  (x, Rg{dv, gv, F.sel(U32.is_eq(bsrc, 1), kb, br), ov})

def sq.r5(r: Array<U32> & U32, +c1: U32, +dv: U32, +gr: U32, +br: U32, +ov: U32) ->
  Array<U32> & Rg:
  (x, +kg) = r
  +gsrc = sq.bits(c1, 7, 2)
  sq.r6(Array.get(U32, x, (72 + sq.bits(c1, 14, 2) : U32)), dv,
    F.sel(U32.is_eq(gsrc, 0), gr, F.sel(U32.is_eq(gsrc, 2), kg, 1065353216)),
    sq.bits(c1, 12, 2), br, ov)

def sq.r4(r: Array<U32> & U32, +c1: U32, +dv: U32, +gr: U32, +br: U32) -> Array<U32> & Rg:
  (x, +ov) = r
  sq.r5(Array.get(U32, x, (72 + sq.bits(c1, 9, 2) : U32)), c1, dv, gr, br, ov)

def sq.r3(r: Array<U32> & U32, +c1: U32, +c2: U32, +dv: U32, +gr: U32) -> Array<U32> & Rg:
  (x, +br) = r
  sq.r4(Array.get(U32, x, (64 + sq.bits(c2, 14, 3) : U32)), c1, dv, gr, br)

def sq.r2(r: Array<U32> & U32, +c1: U32, +c2: U32, +dv: U32) -> Array<U32> & Rg:
  (x, +gr) = r
  sq.r3(Array.get(U32, x, (64 + sq.bits(c1, 14, 3) : U32)), c1, c2, dv, gr)

def sq.r1(r: Array<U32> & U32, +c1: U32, +c2: U32) -> Array<U32> & Rg:
  (x, +dv) = r
  sq.r2(Array.get(U32, x, (64 + sq.bits(c1, 9, 3) : U32)), c1, c2, dv)

def sq.regs(x: Array<U32>, +c1: U32, +c2: U32) -> Array<U32> & Rg:
  sq.r1(Array.get(U32, x, (64 + sq.bits(c1, 17, 3) : U32)), c1, c2)

# the row in the scratch: A, the scalar ops (X[r] from 4224 + r), Y, O
def sq.rx2(r: Array<U32> & U32, +m: Sr, +row: U32, +a: U32) -> Array<U32>:
  (x, +xv) = r
  match m:
    case Sr{cols, +c1, +c2, src, ss, dst, ds, md}:
      sq.y(m, row, sq.regs(sq.sc(U32.to_nat((c2 .&. 15 : U32)), x, 0, a, xv), c1, c2))

def sq.rx1(r: Array<U32> & U32, +m: Sr, +row: U32) -> Array<U32>:
  (x, +a) = r
  sq.rx2(Array.get(U32, x, (4224 + row : U32)), m, row, a)

def sq.mid(s: Sl, +m: Sr, +row: U32) -> Sl:
  match s m:
    case Sl{c, t, a, d, h, x, +k} Sr{+cols, +c1, c2, src, ss, dst, ds, md}:
      Sl{c, t, a, d, h, sq.rx1(sq.a(U32.is_ne(sq.bits(c1, 0, 1), 0),
        U32.is_eq(sq.bits(c1, 1, 2), 3), x, c1, cols), m, row), k}

def sq.row(s: Sl, +m: Sr, +r: U32) -> Sl:
  match m:
    case Sr{cols, c1, c2, src, ss, dst, ds, +md}:
      sq.store(U32.is_ne(sq.bits(md, 3, 1), 0), sq.mid(sq.load(U32.is_ne(sq.bits(md, 0, 1), 0),
        s, U32.is_ne(sq.bits(md, 1, 1), 0), m, r), m, r), U32.is_ne(sq.bits(md, 4, 1), 0), m,
        U32.is_ne(sq.bits(md, 2, 1), 0), r)

def sq.rows(n: Nat, s: Sl, +m: Sr, +r: U32) -> Sl:
  match n:
    case 0n:
      s
    case 1n+p:
      sq.rows(p, sq.row(s, m, r), m, (r + 1 : U32))

# --- the inputs and the last writes

def sq.k1(r: Array<U32> & U32, x: Array<U32>, +i: U32) -> Array<U32> & Array<U32>:
  (t, +v) = r
  (t, Array.set(U32, x, i, v))

def sq.kj(z: Bool, t: Array<U32>, x: Array<U32>, +j: U32, +ka: U32, +ks: U32) ->
  Array<U32> & Array<U32>:
  match z:
    case True{}:
      sq.k1(Array.get(U32, t, (ka + j * ks : U32)), x, (72 + j : U32))
    case False{}:
      (t, x)

# the constants in m's bits to K0..K3
def sq.ks(n: Nat, st: Array<U32> & Array<U32>, +j: U32, +m: U32, +ka: U32, +ks: U32) ->
  Array<U32> & Array<U32>:
  match n:
    case 0n:
      st
    case 1n+p:
      (t, x) = st
      sq.ks(p, sq.kj(U32.is_ne(sq.bits(m, j, 1), 0), t, x, j, ka, ks), (j + 1 : U32), m, ka, ks)

def sq.ri1(r: Array<U32> & U32, +j: U32, +on: U32) -> Array<U32>:
  (x, +v) = r
  x[(68 + j : U32)] <- F.sel(U32.is_ne(on, 0), v, 0)

# r4..r7: K0..K3 where rinit has their bit, else +0
def sq.ri(n: Nat, x: Array<U32>, +j: U32, +m: U32) -> Array<U32>:
  match n:
    case 0n:
      x
    case 1n+p:
      sq.ri(p, sq.ri1(Array.get(U32, x, (72 + j : U32)), j, sq.bits(m, j, 1)), (j + 1 : U32), m)

def sq.sv1(r: Array<U32> & U32, t: Array<U32>, +i: U32) -> Array<U32> & Array<U32>:
  (x, +v) = r
  (x, Array.set(U32, t, i, v))

def sq.svj(z: Bool, x: Array<U32>, t: Array<U32>, +j: U32, +ka: U32, +ks: U32) ->
  Array<U32> & Array<U32>:
  match z:
    case True{}:
      sq.sv1(Array.get(U32, x, (68 + j : U32)), t, (ka + j * ks : U32))
    case False{}:
      (x, t)

# rsave: r4..r7 in m's bits to K0..K3's words
def sq.sv(n: Nat, st: Array<U32> & Array<U32>, +j: U32, +m: U32, +ka: U32, +ks: U32) ->
  Array<U32> & Array<U32>:
  match n:
    case 0n:
      st
    case 1n+p:
      (x, t) = st
      sq.sv(p, sq.svj(U32.is_ne(sq.bits(m, j, 1), 0), x, t, j, ka, ks), (j + 1 : U32), m, ka, ks)

# the waiting rows to TMEM, row r's cols words to dst + r * ds
def sq.wr(n: Nat, st: Array<U32> & Array<U32>, +r: U32, +cols: U32, +dst: U32, +ds: U32) ->
  Array<U32> & Array<U32>:
  match n:
    case 0n:
      st
    case 1n+p:
      sq.wr(p, cp(U32.to_nat(cols), st, (32896 + r * cols : U32), (dst + r * ds : U32)),
        (r + 1 : U32), cols, dst, ds)

# the operands: rows, cols, the descriptor's c1, c2 and strides word, src and dst, the slots'
# vec, X's xa, the constants' ka and stride ks, O's out, the flags, the ops' reads (bit 0 a bad
# op, 1 X, 4..7 K0..K3; bit 8 a bad descriptor)
type Sm is Data:
  Sm{rows: U32, cols: U32, c1: U32, c2: U32, p4: U32, src: U32, dst: U32, vec: U32, xa: U32,
     ka: U32, ks: U32, out: U32, f: U32, of: U32}

# the slots read (a bit each): A's, G's, B's, Q's (q_en without q_self, or out_p2)
def sq.sm(+c1: U32) -> U32:
  +um = sq.bits(c1, 5, 2)
  (F.sel(U32.is_ne(sq.bits(c1, 0, 1), 0) && U32.is_eq(sq.bits(c1, 1, 2), 0),
    F.shl(1, sq.bits(c1, 3, 2)), 0)
  .|. F.sel((U32.is_eq(um, 1) || U32.is_eq(um, 2)) && U32.is_eq(sq.bits(c1, 7, 2), 1),
    F.shl(1, sq.bits(c1, 9, 3)), 0)
  .|. F.sel((U32.is_eq(um, 1) || U32.is_eq(um, 3)) && U32.is_eq(sq.bits(c1, 12, 2), 0),
    F.shl(1, sq.bits(c1, 14, 3)), 0) : U32)

def sq.sq(+c1: U32, +c2: U32) -> U32:
  F.sel((U32.is_ne(sq.bits(c1, 20, 1), 0) && U32.is_eq(sq.bits(c2, 12, 1), 0))
    || U32.is_ne(sq.bits(c1, 23, 1), 0), F.shl(1, sq.bits(c1, 21, 2)), 0)

# the constants read: rinit's, the ops' operands', A's, G's, B's
def sq.km(+c1: U32, +c2: U32, +of: U32) -> U32:
  +um = sq.bits(c1, 5, 2)
  (sq.bits(c2, 4, 4) .|. sq.bits(of, 4, 4)
  .|. F.sel(U32.is_ne(sq.bits(c1, 0, 1), 0) && U32.is_eq(sq.bits(c1, 1, 2), 2),
    F.shl(1, sq.bits(c1, 3, 2)), 0)
  .|. F.sel((U32.is_eq(um, 1) || U32.is_eq(um, 2)) && U32.is_eq(sq.bits(c1, 7, 2), 2),
    F.shl(1, sq.bits(c1, 9, 3)), 0)
  .|. F.sel((U32.is_eq(um, 1) || U32.is_eq(um, 3)) && U32.is_eq(sq.bits(c1, 12, 2), 1),
    F.shl(1, sq.bits(c1, 14, 3)), 0) : U32)

# a row stride: the descriptor's, 0 meaning cols
def sq.rs(+v: U32, +cols: U32) -> U32:
  F.sel(U32.is_eq(v, 0), cols, v)

# whether the rows wait in the scratch: a TMEM stream in place whose rows overlap
def sq.stg(+rows: U32, +cols: U32, +p4: U32, +src: U32, +dst: U32, +f: U32) -> U32:
  +srs = sq.rs(sq.bits(p4, 0, 12), cols)
  F.b2u(U32.is_eq((f .&. 15 : U32), 6) && U32.is_eq(src, dst)
    && U32.is_eq(srs, sq.rs(sq.bits(p4, 12, 12), cols)) && (srs < cols : U32) && (rows > 1 : U32))

# STREAM's checks in isasim's order: 38 the descriptor (not float-safe, fewer words than its
# ops), rows or cols 0, q_en with o_reg_en, a bad op, a slot or constant index past 3; 7 the
# slots, constants or X past TMEM; 38 a DRAM stream not DRAM-chunk aligned; 6 a DRAM stream
# past DRAM; 7 a TMEM source past TMEM; 38 a bad B source; 7 a TMEM destination past TMEM; 42
# waiting rows past the scratch; 7 O or the saved registers past TMEM
def sq.chk(+m: Sm, +k: Cf) -> U32:
  match m:
    case Sm{+rows, +cols, +c1, +c2, +p4, +src, +dst, +vec, +xa, +ka, +ks, +out, +f, +of}:
      +tm = (Cf.tw(k) - 1 : U32)
      +um = sq.bits(c1, 5, 2)
      +ug = U32.is_eq(um, 1) || U32.is_eq(um, 2)
      +ub = U32.is_eq(um, 1) || U32.is_eq(um, 3)
      +gsrc = sq.bits(c1, 7, 2)
      +bsrc = sq.bits(c1, 12, 2)
      +oo = U32.is_ne(sq.bits(c1, 20, 1), 0) || U32.is_ne(sq.bits(c2, 13, 1), 0)
      +nsl = sq.top((sq.sm(c1) .|. sq.sq(c1, c2) : U32))
      +nk = sq.top(sq.km(c1, c2, of))
      +nrs = sq.top(sq.bits(c2, 8, 4))
      +sz = (f .&. 1 : U32)
      +st = sq.bits(f, 1, 1)
      +dt = sq.bits(f, 2, 1)
      +nd = sq.bits(f, 3, 1)
      +sdr = U32.is_eq(st, 0) && U32.is_eq(sz, 0)
      +ddr = U32.is_eq(dt, 0) && U32.is_eq(nd, 0)
      F.sel(U32.is_ne(sq.bits(of, 8, 1), 0) || U32.is_eq(rows, 0) || U32.is_eq(cols, 0)
        || (U32.is_ne(sq.bits(c1, 20, 1), 0) && U32.is_ne(sq.bits(c2, 13, 1), 0))
        || U32.is_ne(sq.bits(of, 0, 1), 0)
        || (ug && (U32.is_eq(gsrc, 1) || U32.is_eq(gsrc, 2)) && (sq.bits(c1, 9, 3) >= 4 : U32))
        || (ub && (bsrc < 2 : U32) && (sq.bits(c1, 14, 3) >= 4 : U32)), 38,
      F.sel(((nsl > 0 : U32) && Bool.not(rng(vec, (nsl * cols : U32), 1, 1, 0, tm)))
        || ((nk > 0 : U32) && Bool.not(rng(ka, nk, ks, 1, 0, tm)))
        || (U32.is_ne(sq.bits(of, 1, 1), 0) && Bool.not(rng(xa, rows, 1, 1, 0, tm))), 7,
      F.sel((sdr && U32.is_ne((src % Cf.d(k) : U32), 0))
        || (ddr && U32.is_ne((dst % Cf.d(k) : U32), 0)), 38,
      F.sel((sdr && Bool.not(rng(src, (rows * cols : U32), 4, 4, 1, Cf.dbm1(k))))
        || (ddr && Bool.not(rng(dst, (rows * cols : U32), 4, 4, 1, Cf.dbm1(k)))), 6,
      F.sel(U32.is_ne(st, 0) && U32.is_eq(sz, 0)
        && Bool.not(rng(src, rows, sq.rs(sq.bits(p4, 0, 12), cols), cols, 1, tm)), 7,
      F.sel(ub && U32.is_eq(bsrc, 3), 38,
      F.sel(U32.is_ne(dt, 0) && U32.is_eq(nd, 0)
        && Bool.not(rng(dst, rows, sq.rs(sq.bits(p4, 12, 12), cols), cols, 1, tm)), 7,
      F.sel(U32.is_ne(sq.stg(rows, cols, p4, src, dst, f), 0)
        && ((rows * cols : U32) > 4161408 : U32), 42,
      F.sel((oo && Bool.not(rng(out, rows, 1, 1, 0, tm)))
        || ((nrs > 0 : U32) && Bool.not(rng(ka, nrs, ks, 1, 0, tm))), 7, 0)))))))))

# the rows done: the waiting rows, O, the saved registers
def sq.e3(r: Array<U32> & Array<U32>, c: Array<U32>, a: Array<U32>, d: Array<U32>,
  h: Array<U32>, +k: Cf, +m: Sm) -> Sl:
  match m:
    case Sm{rows, cols, c1, +c2, p4, src, dst, vec, xa, +ka, +ks, out, f, of}:
      sq.st(sq.sv(4n, r, 0, sq.bits(c2, 8, 4), ka, ks), c, a, d, h, k)

def sq.e2(r: Array<U32> & Array<U32>, c: Array<U32>, a: Array<U32>, d: Array<U32>,
  h: Array<U32>, +k: Cf, +m: Sm) -> Sl:
  match m:
    case Sm{+rows, cols, +c1, +c2, p4, src, dst, vec, xa, ka, ks, +out, f, of}:
      sq.e3(cp(U32.to_nat(F.sel(U32.is_ne(sq.bits(c1, 20, 1), 0)
        || U32.is_ne(sq.bits(c2, 13, 1), 0), rows, 0)), r, 128, out), c, a, d, h, k, m)

def sq.e1(s: Sl, +m: Sm) -> Sl:
  match s m:
    case Sl{c, t, a, d, h, x, +k} Sm{+rows, +cols, c1, c2, +p4, +src, +dst, vec, xa, ka, ks, out,
      +f, of}:
      sq.e2(sq.wr(U32.to_nat(F.sel(U32.is_ne(sq.stg(rows, cols, p4, src, dst, f), 0), rows, 0)),
        (x, t), 0, cols, dst, sq.rs(sq.bits(p4, 12, 12), cols)), c, a, d, h, k, m)

# the inputs in the scratch (the slots, then K0..K3, then X, then r0..r7), the rows
def sq.in3(r: Array<U32> & Array<U32>, c: Array<U32>, a: Array<U32>, d: Array<U32>,
  h: Array<U32>, +k: Cf, +m: Sm) -> Sl:
  (t, x) = r
  match m:
    case Sm{+rows, +cols, +c1, +c2, +p4, +src, +dst, vec, xa, ka, ks, out, +f, of}:
      +st = sq.bits(f, 1, 1)
      +dt = sq.bits(f, 2, 1)
      +sr = {Sr{cols, c1, c2, F.sel(U32.is_ne(st, 0), src, (src >> 2n : U32)),
        F.sel(U32.is_ne(st, 0), sq.rs(sq.bits(p4, 0, 12), cols), cols),
        F.sel(U32.is_ne(dt, 0), dst, (dst >> 2n : U32)),
        F.sel(U32.is_ne(dt, 0), sq.rs(sq.bits(p4, 12, 12), cols), cols),
        ((f .&. 15 : U32) .|. (sq.stg(rows, cols, p4, src, dst, f) << 4n : U32) : U32)} : Sr}
      sq.e1(sq.rows(U32.to_nat(rows), Sl{c, t, a, d, h, sq.ri(4n, fill(4n, x, 64, 0), 0,
        sq.bits(c2, 4, 4)), k}, sr, 0), m)

def sq.in2(r: Array<U32> & Array<U32>, c: Array<U32>, a: Array<U32>, d: Array<U32>,
  h: Array<U32>, +k: Cf, +m: Sm) -> Sl:
  match m:
    case Sm{+rows, cols, c1, c2, p4, src, dst, vec, +xa, ka, ks, out, f, +of}:
      sq.in3(cp(U32.to_nat(F.sel(U32.is_ne(sq.bits(of, 1, 1), 0), rows, 0)), r, xa, 4224), c,
        a, d, h, k, m)

def sq.in1(r: Array<U32> & Array<U32>, c: Array<U32>, a: Array<U32>, d: Array<U32>,
  h: Array<U32>, +k: Cf, +m: Sm) -> Sl:
  match m:
    case Sm{rows, cols, +c1, +c2, p4, src, dst, vec, xa, +ka, +ks, out, f, +of}:
      sq.in2(sq.ks(4n, r, 0, sq.km(c1, c2, of), ka, ks), c, a, d, h, k, m)

def sq.go(z: Bool, s: Sl, +m: Sm, +e: U32) -> Sl:
  match z s:
    case True{} Sl{c, t, a, d, h, x, +k}:
      match m:
        case Sm{rows, +cols, +c1, +c2, p4, src, dst, +vec, xa, ka, ks, out, f, of}:
          sq.in1(cp(U32.to_nat((sq.top((sq.sm(c1) .|. sq.sq(c1, c2) : U32)) * cols : U32)),
            (t, x), vec, 8320), c, a, d, h, k, m)
    case False{} Sl{c, t, a, d, h, x, k}:
      Sl{fail(c, e, 0), t, a, d, h, x, k}

# the descriptor: its first five words' payloads, then its ops'; a bad one sets of's bit 8
def sq.d9(r: Array<U32> & U32, c: Array<U32>, t: Array<U32>, a: Array<U32>, d: Array<U32>,
  h: Array<U32>, +k: Cf, +i: Ins, +e: U32, +p0: U32, +c1: U32, +c2: U32, +p4: U32) -> Sl:
  (x, +of) = r
  match i:
    case Ins{+w0, +w1, +w2, +w3, +w4, +w5, +w6, +w7, +va, +vb, +vc, +vd}:
      +m = {Sm{(p0 .&. 4095 : U32), (p0 >> 12n : U32), c1, c2, p4, (va + w2 : U32),
        (va + w3 : U32), (vb + w4 : U32), (vc + w5 : U32), (vd + w6 : U32), (w1 >> 16n : U32),
        w7, fl(w0), (of .|. (e << 8n : U32) : U32)} : Sm}
      +er = sq.chk(m, k)
      sq.go(U32.is_eq(er, 0), Sl{c, t, a, d, h, x, k}, m, er)

def sq.d8(r: Array<U32> & U32, c: Array<U32>, t: Array<U32>, a: Array<U32>, d: Array<U32>,
  h: Array<U32>, +k: Cf, +i: Ins, +e: U32, +p0: U32, +c1: U32, +c2: U32) -> Sl:
  (x, +p4) = r
  sq.d9(sq.ops(U32.to_nat((c2 .&. 15 : U32)), (x, 0), 85), c, t, a, d, h, k, i, e, p0, c1, c2,
    p4)

def sq.d7(r: Array<U32> & U32, c: Array<U32>, t: Array<U32>, a: Array<U32>, d: Array<U32>,
  h: Array<U32>, +k: Cf, +i: Ins, +e: U32, +p0: U32, +c1: U32) -> Sl:
  (x, +c2) = r
  sq.d8(Array.get(U32, x, 84), c, t, a, d, h, k, i, e, p0, c1, c2)

def sq.d6(r: Array<U32> & U32, c: Array<U32>, t: Array<U32>, a: Array<U32>, d: Array<U32>,
  h: Array<U32>, +k: Cf, +i: Ins, +e: U32, +p0: U32) -> Sl:
  (x, +c1) = r
  sq.d7(Array.get(U32, x, 82), c, t, a, d, h, k, i, e, p0, c1)

def sq.d5(r: Array<U32> & U32, c: Array<U32>, t: Array<U32>, a: Array<U32>, d: Array<U32>,
  h: Array<U32>, +k: Cf, +i: Ins, +e: U32) -> Sl:
  (x, +p0) = r
  sq.d6(Array.get(U32, x, 81), c, t, a, d, h, k, i, e, p0)

def sq.d4(r: Array<U32> & U32, c: Array<U32>, t: Array<U32>, a: Array<U32>, d: Array<U32>,
  h: Array<U32>, +k: Cf, +i: Ins, +e: U32) -> Sl:
  (x, +bad) = r
  sq.d5(Array.get(U32, x, 80), c, t, a, d, h, k, i, (e .|. bad : U32))

# fewer than five words before TMEM's end, a word not float-safe, fewer words than the ops
def sq.d3(r: Array<U32> & U32, c: Array<U32>, t: Array<U32>, a: Array<U32>, d: Array<U32>,
  h: Array<U32>, +k: Cf, +i: Ins, +nav: U32, +bad: U32) -> Sl:
  (x, +c2) = r
  +n = (c2 .&. 15 : U32)
  sq.d4(sq.fs(U32.to_nat(n), (x, 0), 85), c, t, a, d, h, k, i,
    F.b2u((nav < 5 : U32) || U32.is_ne(bad, 0) || ((5 + n : U32) > nav : U32)))

def sq.d2(r: Array<U32> & U32, c: Array<U32>, t: Array<U32>, a: Array<U32>, d: Array<U32>,
  h: Array<U32>, +k: Cf, +i: Ins, +nav: U32) -> Sl:
  (x, +bad) = r
  sq.d3(Array.get(U32, x, 82), c, t, a, d, h, k, i, nav, bad)

def sq.d1(r: Array<U32> & Array<U32>, c: Array<U32>, a: Array<U32>, d: Array<U32>,
  h: Array<U32>, +k: Cf, +i: Ins, +nav: U32) -> Sl:
  (t, x) = r
  sq.d2(sq.fs(5n, (x, 0), 80), c, t, a, d, h, k, i, nav)

def sq.w1(+i: Ins) -> U32:
  match i:
    case Ins{w0, +w1, w2, w3, w4, w5, w6, w7, va, vb, vc, vd}:
      w1

# the descriptor's words before TMEM's end (at most 20) to the scratch
def x.stream(s: Sl, +i: Ins) -> Sl:
  match s:
    case Sl{c, t, a, d, h, x, +k}:
      +at = (sq.w1(i) .&. 65535 : U32)
      +tw = Cf.tw(k)
      +nav = F.sel((at >= tw : U32), 0, F.sel(((tw - at : U32) > 20 : U32), 20, (tw - at : U32)))
      sq.d1(cp(U32.to_nat(nav), (t, x), at, 80), c, a, d, h, k, i, nav)

# the instructions after which the slice goes on to the next pc, by opcode (most frequent
# first): VOP, MM, QACT, LD, QST, ST, ADDI, LI, RLD, NOP, DSTEP, STREAM
def ex.f19(~mm: Sl -> Ins -> Sl, z: Bool, s: Sl, +i: Ins, +o: U32) -> Sl:
  match z:
    case True{}:
      x.stream(s, i)
    case False{}:
      ex.bad(s, i)

def ex.f18(~mm: Sl -> Ins -> Sl, z: Bool, s: Sl, +i: Ins, +o: U32) -> Sl:
  match z:
    case True{}:
      x.dstep(s, i)
    case False{}:
      ex.f19(~mm, U32.is_eq(o, 19), s, i, o)

def ex.f0(~mm: Sl -> Ins -> Sl, z: Bool, s: Sl, +i: Ins, +o: U32) -> Sl:
  match z:
    case True{}:
      s
    case False{}:
      ex.f18(~mm, U32.is_eq(o, 18), s, i, o)

def ex.f6(~mm: Sl -> Ins -> Sl, z: Bool, s: Sl, +i: Ins, +o: U32) -> Sl:
  match z:
    case True{}:
      x.rld(s, i)
    case False{}:
      ex.f0(~mm, U32.is_eq(o, 0), s, i, o)

def ex.f2(~mm: Sl -> Ins -> Sl, z: Bool, s: Sl, +i: Ins, +o: U32) -> Sl:
  match z:
    case True{}:
      x.li(s, i)
    case False{}:
      ex.f6(~mm, U32.is_eq(o, 6), s, i, o)

def ex.f3(~mm: Sl -> Ins -> Sl, z: Bool, s: Sl, +i: Ins, +o: U32) -> Sl:
  match z:
    case True{}:
      x.addi(s, i)
    case False{}:
      ex.f2(~mm, U32.is_eq(o, 2), s, i, o)

def ex.f17(~mm: Sl -> Ins -> Sl, z: Bool, s: Sl, +i: Ins, +o: U32) -> Sl:
  match z:
    case True{}:
      x.ldst(True{}, s, i)
    case False{}:
      ex.f3(~mm, U32.is_eq(o, 3), s, i, o)

def ex.f34(~mm: Sl -> Ins -> Sl, z: Bool, s: Sl, +i: Ins, +o: U32) -> Sl:
  match z:
    case True{}:
      x.qst(s, i)
    case False{}:
      ex.f17(~mm, U32.is_eq(o, 17), s, i, o)

def ex.f16(~mm: Sl -> Ins -> Sl, z: Bool, s: Sl, +i: Ins, +o: U32) -> Sl:
  match z:
    case True{}:
      x.ldst(False{}, s, i)
    case False{}:
      ex.f34(~mm, U32.is_eq(o, 34), s, i, o)

def ex.f33(~mm: Sl -> Ins -> Sl, z: Bool, s: Sl, +i: Ins, +o: U32) -> Sl:
  match z:
    case True{}:
      x.qact(s, i)
    case False{}:
      ex.f16(~mm, U32.is_eq(o, 16), s, i, o)

def ex.f32(~mm: Sl -> Ins -> Sl, z: Bool, s: Sl, +i: Ins, +o: U32) -> Sl:
  match z:
    case True{}:
      mm(s, i)
    case False{}:
      ex.f33(~mm, U32.is_eq(o, 33), s, i, o)

def ex.f48(~mm: Sl -> Ins -> Sl, z: Bool, s: Sl, +i: Ins, +o: U32) -> Sl:
  match z:
    case True{}:
      x.vop(s, i)
    case False{}:
      ex.f32(~mm, U32.is_eq(o, 32), s, i, o)

def ex(~mm: Sl -> Ins -> Sl, s: Sl, +i: Ins, +o: U32) -> Sl:
  ex.f48(~mm, U32.is_eq(o, 48), s, i, o)

# after them, the next pc while the slice runs
def post.go(z: Bool, c: Array<U32>) -> Array<U32>:
  match z:
    case True{}:
      adv(c)
    case False{}:
      c

def post.st(r: Array<U32> & U32) -> Array<U32>:
  (c, +st) = r
  post.go(U32.is_eq(st, 0), c)

def post(s: Sl) -> Sl:
  match s:
    case Sl{c, t, a, d, h, x, k}:
      Sl{post.st(Array.get(U32, c, 18)), t, a, d, h, x, k}

# the rest: HALT (1), LOOP (4), WAITW (7), BAR (5) and GATHER (64) set the pc or the state
# themselves (isasim Slice.step)
def st.coll(~mm: Sl -> Ins -> Sl, z: Bool, s: Sl, +i: Ins, +o: U32) -> Sl:
  match z:
    case True{}:
      x.wait(s)
    case False{}:
      post(ex(~mm, s, i, o))

def st.waitw(~mm: Sl -> Ins -> Sl, z: Bool, s: Sl, +i: Ins, +o: U32) -> Sl:
  match z:
    case True{}:
      x.waitw(s, i, 1)
    case False{}:
      st.coll(~mm, U32.is_eq(o, 5) || U32.is_eq(o, 64), s, i, o)

def st.loop(~mm: Sl -> Ins -> Sl, z: Bool, s: Sl, +i: Ins, +o: U32) -> Sl:
  match z:
    case True{}:
      x.loop(s, i)
    case False{}:
      st.waitw(~mm, U32.is_eq(o, 7), s, i, o)

def st.op(~mm: Sl -> Ins -> Sl, z: Bool, s: Sl, +i: Ins, +o: U32) -> Sl:
  match z:
    case True{}:
      x.halt(s, i)
    case False{}:
      st.loop(~mm, U32.is_eq(o, 4), s, i, o)

def step.i(~mm: Sl -> Ins -> Sl, r: Array<U32> & Ins, t: Array<U32>, a: Array<U32>,
  d: Array<U32>, h: Array<U32>, x: Array<U32>, +k: Cf) -> Sl:
  (c, +i) = r
  +o = op(Ins.w0(i))
  st.op(~mm, U32.is_eq(o, 1), Sl{bump(bump(c, ICOUNT(), 1), STEPS(), 1), t, a, d, h, x, k}, i, o)

def step.ok(~mm: Sl -> Ins -> Sl, z: Bool, c: Array<U32>, t: Array<U32>, a: Array<U32>,
  d: Array<U32>, h: Array<U32>, x: Array<U32>, +k: Cf, +pc: U32) -> Sl:
  match z:
    case True{}:
      step.i(~mm, fetch(c, pc), t, a, d, h, x, k)
    case False{}:
      Sl{fail(c, 1, pc), t, a, d, h, x, k}

def step.n(~mm: Sl -> Ins -> Sl, r: Array<U32> & U32, t: Array<U32>, a: Array<U32>,
  d: Array<U32>, h: Array<U32>, x: Array<U32>, +k: Cf, +pc: U32) -> Sl:
  (c, +n) = r
  step.ok(~mm, (pc < n : U32), c, t, a, d, h, x, k, pc)

def step.pc(~mm: Sl -> Ins -> Sl, r: Array<U32> & U32, t: Array<U32>, a: Array<U32>,
  d: Array<U32>, h: Array<U32>, x: Array<U32>, +k: Cf) -> Sl:
  (c, +pc) = r
  step.n(~mm, Array.get(U32, c, PLEN()), t, a, d, h, x, k, pc)

# one instruction (isasim Slice.step)
def step(~mm: Sl -> Ins -> Sl, s: Sl) -> Sl:
  match s:
    case Sl{c, t, a, d, h, x, k}:
      step.pc(~mm, Array.get(U32, c, PC()), t, a, d, h, x, k)

def runs.go(r: Array<U32> & U32, t: Array<U32>, a: Array<U32>, d: Array<U32>, h: Array<U32>,
  x: Array<U32>, +k: Cf) -> Bool & Sl:
  (c, +st) = r
  (U32.is_eq(st, 0), Sl{c, t, a, d, h, x, k})

# whether the slice runs (RUN), and the slice
def runs(s: Sl) -> Bool & Sl:
  match s:
    case Sl{c, t, a, d, h, x, k}:
      runs.go(Array.get(U32, c, STATE()), t, a, d, h, x, k)

# the slice runs until it stops (HALT, a collective, a WAITW that does not hold, an error) or
# n instructions pass
def go(~mm: Sl -> Ins -> Sl, n: Nat, bs: Bool & Sl) -> Sl:
  match n:
    case 0n:
      (b, s) = bs
      s
    case 1n+p:
      (b, s) = bs
      match b:
        case True{}:
          go(~mm, p, runs(step(~mm, s)))
        case False{}:
          s

def ww.again.i(r: Array<U32> & Ins, t: Array<U32>, a: Array<U32>, d: Array<U32>,
  h: Array<U32>, x: Array<U32>, +k: Cf) -> Sl:
  (c, +i) = r
  x.waitw(Sl{c, t, a, d, h, x, k}, i, 0)

def ww.again.pc(r: Array<U32> & U32, t: Array<U32>, a: Array<U32>, d: Array<U32>,
  h: Array<U32>, x: Array<U32>, +k: Cf) -> Sl:
  (c, +pc) = r
  ww.again.i(fetch(c, pc), t, a, d, h, x, k)

# a polling slice reads its WAITW's word again (the machine, after the host's turn)
def ww.again(s: Sl) -> Sl:
  match s:
    case Sl{c, t, a, d, h, x, k}:
      ww.again.pc(Array.get(U32, c, PC()), t, a, d, h, x, k)

# ------------------------------------------------------------------------------- host side

# a new program of n instructions (its words already in IMEM) with the run's arguments
# (isasim Machine.load): the counters cleared, R8..R15 = the arguments, pc 0, RUN
def load(s: Sl, +n: U32, +a0: U32, +a1: U32, +a2: U32, +a3: U32, +a4: U32, +a5: U32,
  +a6: U32, +a7: U32) -> Sl:
  match s:
    case Sl{c, t, a, d, h, x, k}:
      c[PLEN()] <- n
      c[ICOUNT()] <- 0
      c[STALLS()] <- 0
      c[CHAINS()] <- 0
      c[ERR()] <- 0
      c[AUX()] <- 0
      c[40] <- a0
      c[41] <- a1
      c[42] <- a2
      c[43] <- a3
      c[44] <- a4
      c[45] <- a5
      c[46] <- a6
      c[47] <- a7
      Sl{restart(c), t, a, d, h, x, k}

def Cf.with_sid(+k: Cf, +j: U32) -> Cf:
  match k:
    case Cf{+s, +d, +mc, +ab, +tw, +db, +iw, +ar, +pr, +ds, +st, si, +ao, +ad}:
      Cf{s, d, mc, ab, tw, db, iw, ar, pr, ds, st, j, ao, ad}

# (until the MXU is written)
def mm.todo(s: Sl, +i: Ins) -> Sl:
  ex.bad(s, i)

# ------------------------------------------------------------------------------- MXU

# MM (docs/isa.md "MM"; isasim Slice._mm): out[j][n] = Sw[n] * (Sa[j] * 2^-W) * acc, acc the
# exact sum over the K-blocks and their groups (4, NVFP4 8) of (m << (W - dw - da)) * c, c the
# group's dot product of ACT row j's int8 elements (PAIR: the odd blocks from row j + M) and
# the weights (int8; int4 or E2M1 codes), dw and m the weights' scale record's (int8: a dw
# byte a block; int4 / fp4: a word a block, dw and four 6-bit m; NVFP4: an E4M3 byte a group;
# UNIT, RSC: dw 0, m 1), da the ACT block's offsets; a group whose shift is negative drops.
# acc is a 64-bit integer (hi, lo), rounded to fp32 once. ACC adds the old value, ASCALE
# (with UNIT and ACC) old * alpha[j] + y, alpha at TMEM ssa; RMAX writes each row's max to
# out + M * ors + j after the rows. Every output is computed (its old value read) into the
# scratch before any is written, as isasim's.

# an MM's operands and the slice's geometry: the streamed weights from DRAM byte sa (row
# stride rs), the scale rows from ssa (stride srs), the outputs from TMEM word out (row stride
# ors), N outputs, KB K-blocks (the scale rows' from k0), M rows from ACT block ab, the flags
# and the weights' format wf; D, the ACT RAM's blocks nab, its offsets' and Sa's first words,
# DRAM's last word and TMEM's last word (both masks: their sizes are powers of two)
type Mk is Data:
  Mk{sa: U32, rs: U32, ssa: U32, srs: U32, out: U32, ors: U32, n: U32, kb: U32, k0: U32,
     m: U32, ab: U32, f: U32, wf: U32, dd: U32, nab: U32, ado: U32, aso: U32, msk: U32,
     tm: U32}

# the window W: int8 16, int4 / fp4 11, NVFP4 13
def mm.w(+wf: U32) -> U32:
  F.sel(U32.is_eq(wf, 0), 16, F.sel(U32.is_eq(wf, 3), 13, 11))

# streamed bytes a K-block
def mm.bpb(+wf: U32, +d: U32) -> U32:
  F.sel(U32.is_eq(wf, 0), d, (d >> 1n : U32))

# the operands' checks in isasim's order (0: none): 15 N or KB 0, 16 the ACT rows or blocks,
# 17 RMAX / ASCALE with M > MCOLS, 18 PAIR's 2M > MCOLS, 19 PAIR with int8, 20 PAIR with an
# odd k0, 21 RSC with NVFP4, 22 ASCALE without UNIT and ACC, 23 the streamed rows' alignment
# or range
def mm.chk(+o: Mk, +k: Cf) -> U32:
  match o:
    case Mk{+sa, +rs, ssa, srs, out, ors, +n, +kb, +k0, +m, +ab, +f, +wf, +dd, +nab, ado,
      aso, msk, tm}:
      +pair = U32.is_ne((f .&. 64 : U32), 0)
      +r = F.sel(pair, (2 * m : U32), m)
      +len = (U32.div(((kb * mm.bpb(wf, dd) : U32) + dd - 1 : U32), dd) * dd : U32)
      F.sel(U32.is_eq(n, 0) || U32.is_eq(kb, 0), 15,
      F.sel(U32.is_eq(r, 0) || (r > Cf.ar(k) : U32) || ((ab + kb : U32) > nab : U32), 16,
      F.sel((m > Cf.mcols(k) : U32) && U32.is_ne((f .&. 12 : U32), 0), 17,
      F.sel(pair && (r > Cf.mcols(k) : U32), 18,
      F.sel(pair && U32.is_eq(wf, 0), 19,
      F.sel(pair && U32.is_ne((k0 .&. 1 : U32), 0), 20,
      F.sel(U32.is_ne((f .&. 128 : U32), 0) && U32.is_eq(wf, 3), 21,
      F.sel(U32.is_ne((f .&. 8 : U32), 0) && U32.is_ne((f .&. 3 : U32), 3), 22,
      F.sel(U32.is_ne((sa % dd : U32), 0) || U32.is_ne((rs % dd : U32), 0)
        || Bool.not(rng(sa, n, rs, len, 1, Cf.dbm1(k))), 23, 0)))))))))

# the scale rows' checks (isasim Slice._mx_scales; UNIT has none): 24 their alignment, 6 a
# row's Sw (int4 / fp4: a block's word) past DRAM's end, 25 int8's or NVFP4's bytes past it
def mm.schk(+o: Mk, +k: Cf) -> U32:
  match o:
    case Mk{sa, rs, +ssa, +srs, out, ors, +n, +kb, +k0, m, ab, +f, +wf, dd, nab, ado, aso,
      msk, tm}:
      +rsc = U32.is_ne((f .&. 128 : U32), 0)
      +al = F.sel(U32.is_eq(wf, 3), 16,
        F.sel((U32.is_eq(wf, 1) || U32.is_eq(wf, 2)) && Bool.not(rsc), 8, 4))
      +db = Cf.dbm1(k)
      +kn = (k0 + kb : U32)
      F.sel(U32.is_ne((f .&. 1 : U32), 0), 0,
      F.sel(U32.is_ne((ssa % al : U32), 0) || U32.is_ne((srs % al : U32), 0), 24,
      F.sel(Bool.not(rng(ssa, n, srs, 4, 1, db)), 6,
      F.sel(rsc, 0,
      F.sel(U32.is_eq(wf, 0), F.sel(rng(ssa, n, srs, (4 + kn : U32), 1, db), 0, 25),
      F.sel((wf < 3 : U32), F.sel(rng(ssa, n, srs, (4 * kn + 8 : U32), 1, db), 0, 6),
        F.sel(rng(ssa, n, srs, (8 * kn + 16 : U32), 1, db), 0, 25)))))))

# NVFP4's scale bytes with the sign bit set (error 26): the OR of every block's two words
def nv.b(r: Array<U32> & U32, +z: U32) -> Array<U32> & U32:
  (d, +v) = r
  (d, (z .|. v : U32))

def nv.a(r: Array<U32> & U32, +i: U32, +z: U32) -> Array<U32> & U32:
  (d, +v) = r
  nv.b(Array.get(U32, d, (i + 1 : U32)), (z .|. v : U32))

def nv.scan(n: Nat, st: Array<U32> & U32, +q: U32, +kb: U32, +ssa: U32, +srs: U32,
  +k0: U32) -> Array<U32> & U32:
  match n:
    case 0n:
      st
    case 1n+p:
      (d, +z) = st
      +i = ((ssa + U32.div(q, kb) * srs + 16 + 8 * (k0 + (q % kb : U32) : U32) : U32) >> 2n
        : U32)
      nv.scan(p, nv.a(Array.get(U32, d, i), i, z), (q + 1 : U32), kb, ssa, srs, k0)

# whether every block of the R ACT rows read holds row (r mod M)'s first block's Sa (error 27
# if not: the rows came from more than one QACT)
def sa.v(r: Array<U32> & U32, +ref: U32, +z: U32) -> Array<U32> & U32:
  (a, +v) = r
  (a, (z .|. F.b2u(U32.is_ne(v, ref)) : U32))

def sa.r(r: Array<U32> & U32, +vi: U32, +z: U32) -> Array<U32> & U32:
  (a, +ref) = r
  sa.v(Array.get(U32, a, vi), ref, z)

def sa.scan(n: Nat, st: Array<U32> & U32, +q: U32, +kb: U32, +m: U32, +aso: U32, +nab: U32,
  +ab: U32) -> Array<U32> & U32:
  match n:
    case 0n:
      st
    case 1n+p:
      (a, +z) = st
      +rr = U32.div(q, kb)
      sa.scan(p, sa.r(Array.get(U32, a, (aso + (rr % m : U32) * nab + ab : U32)),
        (aso + rr * nab + ab + (q % kb : U32) : U32), z), (q + 1 : U32), kb, m, aso, nab, ab)

# --- the dot products

# the dot product loops' arrays and sum
type Da is Type:
  Da{d: Array<U32>, a: Array<U32>, c: U32}

# four int8 pairs
def mac4(+w: U32, +v: U32) -> U32:
  ((sx8((w .&. 255 : U32)) * sx8((v .&. 255 : U32)) : U32)
    + (sx8(((w >> 8n : U32) .&. 255 : U32)) * sx8(((v >> 8n : U32) .&. 255 : U32)) : U32)
    + (sx8(((w >> 16n : U32) .&. 255 : U32)) * sx8(((v >> 16n : U32) .&. 255 : U32)) : U32)
    + (sar(w, 24) * sar(v, 24) : U32) : U32)

def d8.a(r: Array<U32> & U32, d: Array<U32>, +w: U32, +c: U32) -> Da:
  (a, +v) = r
  Da{d, a, (c + mac4(w, v) : U32)}

def d8.w(r: Array<U32> & U32, a: Array<U32>, +ai: U32, +c: U32) -> Da:
  (d, +w) = r
  d8.a(Array.get(U32, a, ai), d, w, c)

def d8.st(st: Da, +wi: U32, +ai: U32) -> Da:
  match st:
    case Da{d, a, +c}:
      d8.w(Array.get(U32, d, wi), a, ai, c)

# n words of int8 weights from DRAM word wi with n words of int8 elements from ACT word ai
def dot8(n: Nat, st: Da, +wi: U32, +ai: U32) -> Da:
  match n:
    case 0n:
      st
    case 1n+p:
      dot8(p, d8.st(st, wi, ai), (wi + 1 : U32), (ai + 1 : U32))

# the 4-bit codes' values: int4 two's complement; E2M1 (0, 1, 2, 3, 4, 6, 8, 12 and the sign:
# 0xC8643210 the magnitudes' nibbles)
def dec.i4(+c: U32) -> U32:
  ((c .^. 8 : U32) - 8 : U32)

def dec.f4(+c: U32) -> U32:
  +mag = ((3362009616 >> U32.to_nat(((c .&. 7 : U32) * 4 : U32)) : U32) .&. 15 : U32)
  F.sel(U32.is_eq((c .&. 8 : U32), 0), mag, (0 - mag : U32))

# four 4-bit codes (w's low 16 bits, the first in the low nibble) with four int8 elements
def mac4n(~dec: U32 -> U32, +w: U32, +v: U32) -> U32:
  ((dec((w .&. 15 : U32)) * sx8((v .&. 255 : U32)) : U32)
    + (dec(((w >> 4n : U32) .&. 15 : U32)) * sx8(((v >> 8n : U32) .&. 255 : U32)) : U32)
    + (dec(((w >> 8n : U32) .&. 15 : U32)) * sx8(((v >> 16n : U32) .&. 255 : U32)) : U32)
    + (dec(((w >> 12n : U32) .&. 15 : U32)) * sar(v, 24) : U32) : U32)

def d4.a(~dec: U32 -> U32, r: Array<U32> & U32, d: Array<U32>, +w: U32, +c: U32) -> Da:
  (a, +v) = r
  Da{d, a, (c + mac4n(~dec, w, v) : U32)}

def d4.w(~dec: U32 -> U32, r: Array<U32> & U32, a: Array<U32>, +ai: U32, +q: U32, +c: U32) ->
  Da:
  (d, +w) = r
  d4.a(~dec, Array.get(U32, a, ai), d,
    F.sel(U32.is_eq((q .&. 1 : U32), 0), w, (w >> 16n : U32)), c)

def d4.st(~dec: U32 -> U32, st: Da, +wi: U32, +ai: U32, +q: U32) -> Da:
  match st:
    case Da{d, a, +c}:
      d4.w(~dec, Array.get(U32, d, (wi + (q >> 1n : U32) : U32)), a, ai, q, c)

# n words of int8 elements from ACT word ai, the block's word q of them on, with their codes
# (eight a word: word q / 2's half q mod 2 from the block's DRAM word wi)
def dot4(~dec: U32 -> U32, n: Nat, st: Da, +wi: U32, +ai: U32, +q: U32) -> Da:
  match n:
    case 0n:
      st
    case 1n+p:
      dot4(~dec, p, d4.st(~dec, st, wi, ai, q), wi, (ai + 1 : U32), (q + 1 : U32))

# a group's dot product: int8 (gw words), int4 or E2M1 (gw ACT words) from the block's words
def mg.c(z: Bool, y: Bool, d: Array<U32>, a: Array<U32>, +wi: U32, +ai: U32, +q: U32,
  +gw: U32) -> Da:
  match z:
    case True{}:
      dot8(U32.to_nat(gw), Da{d, a, 0}, (wi + q : U32), (ai + q : U32))
    case False{}:
      match y:
        case True{}:
          dot4(~(c => dec.i4(c)), U32.to_nat(gw), Da{d, a, 0}, wi, (ai + q : U32), q)
        case False{}:
          dot4(~(c => dec.f4(c)), U32.to_nat(gw), Da{d, a, 0}, wi, (ai + q : U32), q)

# --- a block's scale record and the sum

# md 0: int8's dw, byte bsel of w0, m 1; 1: the int4 / fp4 word w0 (dw its low byte, group
# g's m bits 8 + 6g on); 2: NVFP4's eight E4M3 bytes w0, w1 (dw = 15 - max(E, 1), m = 8 + M
# for E > 0, else M); 3: UNIT, RSC (dw 0, m 1)
type Bs is Data:
  Bs{md: U32, w0: U32, w1: U32, bsel: U32}

def bs.e4(+w0: U32, +w1: U32, +g: U32) -> U32:
  byte(F.sel((g < 4 : U32), w0, w1), (g .&. 3 : U32))

def bs.dw(+b: Bs, +g: U32) -> U32:
  match b:
    case Bs{+md, +w0, +w1, +bsel}:
      +ex = ((bs.e4(w0, w1, g) >> 3n : U32) .&. 15 : U32)
      F.sel(U32.is_eq(md, 0), byte(w0, bsel),
      F.sel(U32.is_eq(md, 1), (w0 .&. 255 : U32),
      F.sel(U32.is_eq(md, 2), (15 - F.sel((ex > 1 : U32), ex, 1) : U32), 0)))

def bs.m(+b: Bs, +g: U32) -> U32:
  match b:
    case Bs{+md, +w0, +w1, +bsel}:
      +e = bs.e4(w0, w1, g)
      +mt = (e .&. 7 : U32)
      F.sel(U32.is_eq(md, 1), (F.shr(w0, (8 + 6 * g : U32)) .&. 63 : U32),
      F.sel(U32.is_eq(md, 2), F.sel((e >= 8 : U32), (8 + mt : U32), mt), 1))

# the sum's arrays and its 64-bit value
type Ms is Type:
  Ms{d: Array<U32>, a: Array<U32>, hi: U32, lo: U32}

# (hi, lo) += (c * m) << (w - dw - da), nothing if dw + da > w
def mg.add(r: Da, +hi: U32, +lo: U32, +dw: U32, +m: U32, +da: U32, +w: U32) -> Ms:
  match r:
    case Da{d, a, +c}:
      +ok = ((dw + da : U32) <= w : U32)
      +sh = (w - dw - da : U32)
      +p = (c * m : U32)
      +plo = F.sel(ok, F.shl(p, sh), 0)
      +phi = F.sel(ok, sar(p, (32 - sh : U32)), 0)
      +lo2 = (lo + plo : U32)
      Ms{d, a, (hi + phi + F.b2u((lo2 < lo : U32)) : U32), lo2}

def mg.da(r: Array<U32> & U32, d: Array<U32>, +hi: U32, +lo: U32, +wf: U32, +gw: U32,
  +wi: U32, +ai: U32, +b: Bs, +w: U32, +g: U32) -> Ms:
  (a, +da) = r
  mg.add(mg.c(U32.is_eq(wf, 0), U32.is_eq(wf, 1), d, a, wi, ai, (g * gw : U32), gw), hi, lo,
    bs.dw(b, g), bs.m(b, g), da, w)

# the block's groups from g: gsh 1 for NVFP4's eight (two to an ACT offset), else 0
def mm.gs(n: Nat, st: Ms, +wf: U32, +gw: U32, +wi: U32, +ai: U32, +doff: U32, +b: Bs,
  +w: U32, +g: U32, +gsh: U32) -> Ms:
  match n:
    case 0n:
      st
    case 1n+p:
      match st:
        case Ms{d, a, +hi, +lo}:
          mm.gs(p, mg.da(Array.get(U32, a, (doff + F.shr(g, gsh) : U32)), d, hi, lo, wf, gw,
            wi, ai, b, w, g), wf, gw, wi, ai, doff, b, w, (g + 1 : U32), gsh)

def mm.kb2(r: Array<U32> & U32, a: Array<U32>, +hi: U32, +lo: U32, +w0: U32, +md: U32,
  +bsel: U32, +wf: U32, +gw: U32, +wi: U32, +ai: U32, +doff: U32) -> Ms:
  (d, +w1) = r
  +ng = F.sel(U32.is_eq(wf, 3), 8, 4)
  mm.gs(U32.to_nat(ng), Ms{d, a, hi, lo}, wf, gw, wi, ai, doff, Bs{md, w0, w1, bsel},
    mm.w(wf), 0, F.sel(U32.is_eq(wf, 3), 1, 0))

def mm.kb1(r: Array<U32> & U32, a: Array<U32>, +hi: U32, +lo: U32, +ib: U32, +md: U32,
  +bsel: U32, +wf: U32, +gw: U32, +wi: U32, +ai: U32, +doff: U32) -> Ms:
  (d, +w0) = r
  mm.kb2(Array.get(U32, d, ib), a, hi, lo, w0, md, bsel, wf, gw, wi, ai, doff)

# K-block kk of output (j, nn): its weights' and ACT row's words, its scale record, its groups
def mm.kb(st: Ms, +o: Mk, +j: U32, +nn: U32, +kk: U32) -> Ms:
  match st:
    case Ms{d, a, +hi, +lo}:
      match o:
        case Mk{+sa, +rs, +ssa, +srs, out, ors, n, kb, +k0, +m, +ab, +f, +wf, +dd, +nab,
          +ado, aso, +msk, tm}:
          +odd = U32.is_ne((f .&. 64 : U32), 0) && U32.is_ne((kk .&. 1 : U32), 0)
          +r = (j + F.sel(odd, m, 0) : U32)
          +base = (ssa + nn * srs : U32)
          +kq = (k0 + kk : U32)
          +md = F.sel(U32.is_ne((f .&. 129 : U32), 0), 3,
            F.sel(U32.is_eq(wf, 0), 0, F.sel(U32.is_eq(wf, 3), 2, 1)))
          +ba = F.sel(U32.is_eq(md, 0), (base + 4 + kq : U32),
            F.sel(U32.is_eq(md, 1), (base + 8 + 4 * kq : U32), (base + 16 + 8 * kq : U32)))
          +ia = ((ba >> 2n : U32) .&. msk : U32)
          +gw = F.sel(U32.is_eq(wf, 0), (dd >> 4n : U32),
            U32.div(dd, (4 * F.sel(U32.is_eq(wf, 3), 8, 4) : U32)))
          mm.kb1(Array.get(U32, d, ia), a, hi, lo, ((ia + 1 : U32) .&. msk : U32), md,
            (ba .&. 3 : U32), wf, gw,
            (((sa + nn * rs : U32) >> 2n : U32) + kk * (mm.bpb(wf, dd) >> 2n : U32) : U32),
            (r * (nab * dd >> 2n : U32) + ((ab + kk : U32) * dd >> 2n : U32) : U32),
            (ado + (r * nab + ab + kk : U32) * 4 : U32))

def mm.ks(n: Nat, st: Ms, +o: Mk, +j: U32, +nn: U32, +kk: U32) -> Ms:
  match n:
    case 0n:
      st
    case 1n+p:
      mm.ks(p, mm.kb(st, o, j, nn, kk), o, j, nn, (kk + 1 : U32))

# --- the outputs

# the arrays an MM reads and writes
type Mx is Type:
  Mx{d: Array<U32>, a: Array<U32>, t: Array<U32>, x: Array<U32>}

def mm.y2(r: Array<U32> & U32, d: Array<U32>, a: Array<U32>, x: Array<U32>, +y0: U32,
  +f: U32, +alpha: U32, +xi: U32) -> Mx:
  (t, +old) = r
  +y = F.sel(U32.is_ne((f .&. 8 : U32), 0), F.add(F.mul(old, alpha), y0),
    F.sel(U32.is_ne((f .&. 2 : U32), 0), F.add(old, y0), y0))
  Mx{d, a, t, Array.set(U32, x, xi, y)}

def mm.y(st: Ms, t: Array<U32>, x: Array<U32>, +sw: U32, +sj: U32, +f: U32, +alpha: U32,
  +ti: U32, +xi: U32) -> Mx:
  match st:
    case Ms{d, a, +hi, +lo}:
      mm.y2(Array.get(U32, t, ti), d, a, x, F.mul(F.mul(F.i2f64(hi, lo), sw), sj), f, alpha,
        xi)

def mm.sw(r: Array<U32> & U32, a: Array<U32>, t: Array<U32>, x: Array<U32>, +o: Mk, +j: U32,
  +nn: U32, +sj: U32, +alpha: U32) -> Mx:
  (d, +w) = r
  match o:
    case Mk{sa, rs, ssa, srs, +out, +ors, +n, +kb, k0, m, ab, +f, wf, dd, nab, ado, aso, msk,
      tm}:
      mm.y(mm.ks(U32.to_nat(kb), Ms{d, a, 0, 0}, o, j, nn, 0), t, x,
        F.sel(U32.is_ne((f .&. 1 : U32), 0), 1065353216, w), sj, f, alpha,
        (out + j * ors + nn : U32), (j * n + nn : U32))

def mm.n1(st: Mx, +o: Mk, +j: U32, +nn: U32, +sj: U32, +alpha: U32) -> Mx:
  match st:
    case Mx{d, a, t, x}:
      match o:
        case Mk{sa, rs, +ssa, +srs, out, ors, n, kb, k0, m, ab, f, wf, dd, nab, ado, aso,
          +msk, tm}:
          mm.sw(Array.get(U32, d, (((ssa + nn * srs : U32) >> 2n : U32) .&. msk : U32)), a, t,
            x, o, j, nn, sj, alpha)

# row j's outputs from nn
def mm.ns(n: Nat, st: Mx, +o: Mk, +j: U32, +nn: U32, +sj: U32, +alpha: U32) -> Mx:
  match n:
    case 0n:
      st
    case 1n+p:
      mm.ns(p, mm.n1(st, o, j, nn, sj, alpha), o, j, (nn + 1 : U32), sj, alpha)

def mm.j3(~ns: U32 -> Mx -> Mk -> U32 -> U32 -> U32 -> Mx,
  r: Array<U32> & U32, d: Array<U32>, a: Array<U32>,
  x: Array<U32>, +o: Mk, +j: U32, +n: U32, +sj: U32) -> Mx:
  (t, +alpha) = r
  ns(n, Mx{d, a, t, x}, o, j, sj, alpha)

def mm.j2(~ns: U32 -> Mx -> Mk -> U32 -> U32 -> U32 -> Mx,
  r: Array<U32> & U32, d: Array<U32>, t: Array<U32>,
  x: Array<U32>, +o: Mk, +j: U32, +ti: U32, +n: U32, +wf: U32) -> Mx:
  (a, +saj) = r
  mm.j3(~ns, Array.get(U32, t, ti), d, a, x, o, j, n,
    F.mul(saj, ((127 - mm.w(wf) : U32) << 23n : U32)))

# row j: Sa[j] * 2^-W and alpha[j] (any word when not ASCALE), then its outputs
def mm.j1(~ns: U32 -> Mx -> Mk -> U32 -> U32 -> U32 -> Mx, st: Mx, +o: Mk, +j: U32) -> Mx:
  match st:
    case Mx{d, a, t, x}:
      match o:
        case Mk{sa, rs, +ssa, srs, out, ors, +n, kb, k0, m, +ab, f, +wf, dd, +nab, ado, +aso,
          msk, +tm}:
          mm.j2(~ns, Array.get(U32, a, (aso + j * nab + ab : U32)), d, t, x, o, j,
            ((ssa + j : U32) .&. tm : U32), n, wf)

def mm.js(~ns: U32 -> Mx -> Mk -> U32 -> U32 -> U32 -> Mx, n: Nat, st: Mx, +o: Mk, +j: U32) -> Mx:
  match n:
    case 0n:
      st
    case 1n+p:
      mm.js(~ns, p, mm.j1(~ns, st, o, j), o, (j + 1 : U32))

# the scratch's rows to TMEM, row j's N words to out + j * ors
def mm.wr(n: Nat, st: Array<U32> & Array<U32>, +j: U32, +nn: U32, +out: U32, +ors: U32) ->
  Array<U32> & Array<U32>:
  match n:
    case 0n:
      st
    case 1n+p:
      mm.wr(p, cp(U32.to_nat(nn), st, (j * nn : U32), (out + j * ors : U32)), (j + 1 : U32),
        nn, out, ors)

def rm.put(r: Array<U32> & U32, t: Array<U32>, +ti: U32) -> Array<U32> & Array<U32>:
  (x, +mx) = r
  (x, Array.set(U32, t, ti, mx))

def rm.js(n: Nat, st: Array<U32> & Array<U32>, +j: U32, +nn: U32, +base: U32) ->
  Array<U32> & Array<U32>:
  match n:
    case 0n:
      st
    case 1n+p:
      (x, t) = st
      rm.js(p, rm.put(rm.row(U32.to_nat(nn), (x, 4294967295), (j * nn : U32)), t,
        (base + j : U32)), (j + 1 : U32), nn, base)

# --- MM

def mm.fin(r: Array<U32> & Array<U32>, c: Array<U32>, a: Array<U32>, d: Array<U32>,
  h: Array<U32>, +k: Cf) -> Sl:
  (x, t) = r
  Sl{c, t, a, d, h, x, k}

def mm.rm2(z: Bool, x: Array<U32>, t: Array<U32>, c: Array<U32>, a: Array<U32>,
  d: Array<U32>, h: Array<U32>, +k: Cf, +out: U32, +ors: U32, +n: U32, +m: U32) -> Sl:
  match z:
    case True{}:
      mm.fin(rm.js(U32.to_nat(m), (x, t), 0, n, (out + m * ors : U32)), c, a, d, h, k)
    case False{}:
      Sl{fail(c, 7, out), t, a, d, h, x, k}

def mm.rm1(z: Bool, x: Array<U32>, t: Array<U32>, c: Array<U32>, a: Array<U32>,
  d: Array<U32>, h: Array<U32>, +k: Cf, +out: U32, +ors: U32, +n: U32, +m: U32) -> Sl:
  match z:
    case True{}:
      Sl{c, t, a, d, h, x, k}
    case False{}:
      mm.rm2(rng(out, (m + 1 : U32), ors, m, 1, (Cf.tw(k) - 1 : U32)), x, t, c, a, d, h, k, out,
        ors, n, m)

# RMAX after the rows are written (isasim writes them first too): its words' range, error 7
def mm.rmax(r: Array<U32> & Array<U32>, c: Array<U32>, a: Array<U32>, d: Array<U32>,
  h: Array<U32>, +k: Cf, +o: Mk) -> Sl:
  (x, t) = r
  match o:
    case Mk{sa, rs, ssa, srs, +out, +ors, +n, kb, k0, +m, ab, +f, wf, dd, nab, ado, aso, msk,
      tm}:
      mm.rm1(U32.is_eq((f .&. 4 : U32), 0), x, t, c, a, d, h, k, out, ors, n, m)

def mm.go2(st: Mx, c: Array<U32>, h: Array<U32>, +k: Cf, +o: Mk) -> Sl:
  match st:
    case Mx{d, a, t, x}:
      match o:
        case Mk{sa, rs, ssa, srs, +out, +ors, +n, kb, k0, +m, ab, f, wf, dd, nab, ado, aso,
          msk, tm}:
          mm.rmax(mm.wr(U32.to_nat(m), (x, t), 0, n, out, ors), c, a, d, h, k, o)

def mm.go(~rw: Mx -> Mk -> Mx, s: Sl, +o: Mk) -> Sl:
  match s:
    case Sl{c, t, a, d, h, x, +k}:
      mm.go2(rw(Mx{d, a, t, x}, o), c, h, k, o)

def mm.fl(s: Sl, +e: U32) -> Sl:
  match s:
    case Sl{c, t, a, d, h, x, k}:
      Sl{fail(c, e, 0), t, a, d, h, x, k}

# 27 the rows' Sa differ, 7 alpha's (ASCALE) or the outputs' TMEM range, 42 more outputs than
# the scratch holds (2^22: a limit of this simulator, not of the ISA)
def mm.t2(~rw: Mx -> Mk -> Mx,
  z: Bool, ok: Bool, room: Bool, s: Sl, +o: Mk) -> Sl:
  match z:
    case True{}:
      mm.fl(s, 27)
    case False{}:
      match ok:
        case True{}:
          match room:
            case True{}:
              mm.go(~rw, s, o)
            case False{}:
              mm.fl(s, 42)
        case False{}:
          mm.fl(s, 7)

def mm.e27(~rw: Mx -> Mk -> Mx,
  r: Array<U32> & U32, c: Array<U32>, t: Array<U32>,
  d: Array<U32>, h: Array<U32>, x: Array<U32>, +k: Cf, +o: Mk) -> Sl:
  (a, +z) = r
  match o:
    case Mk{sa, rs, +ssa, srs, +out, +ors, +n, kb, k0, +m, ab, +f, wf, dd, nab, ado, aso, msk,
      +tm}:
      mm.t2(~rw, U32.is_ne(z, 0), (U32.is_eq((f .&. 8 : U32), 0) || rng(ssa, m, 1, 1, 0, tm))
        && rng(out, m, ors, n, 1, tm), ((m * n : U32) <= 4194304 : U32), Sl{c, t, a, d, h, x, k},
        o)

def mm.e3(~rw: Mx -> Mk -> Mx, s: Sl, +o: Mk) -> Sl:
  match s:
    case Sl{c, t, a, d, h, x, +k}:
      match o:
        case Mk{sa, rs, ssa, srs, out, ors, n, +kb, k0, +m, +ab, +f, wf, dd, +nab, ado, +aso,
          msk, tm}:
          +r = F.sel(U32.is_ne((f .&. 64 : U32), 0), (2 * m : U32), m)
          mm.e27(~rw, sa.scan(U32.to_nat((r * kb : U32)), (a, 0), 0, kb, m, aso, nab, ab), c,
            t, d, h, x, k, o)

def mm.e26c(~rw: Mx -> Mk -> Mx, z: Bool, s: Sl, +o: Mk) -> Sl:
  match z:
    case True{}:
      mm.fl(s, 26)
    case False{}:
      mm.e3(~rw, s, o)

def mm.e26b(~rw: Mx -> Mk -> Mx,
  r: Array<U32> & U32, c: Array<U32>, t: Array<U32>,
  a: Array<U32>, h: Array<U32>, x: Array<U32>, +k: Cf, +o: Mk) -> Sl:
  (d, +z) = r
  mm.e26c(~rw, U32.is_ne((z .&. 2155905152 : U32), 0), Sl{c, t, a, d, h, x, k}, o)

# 26 an NVFP4 scale byte with the sign bit set (NVFP4 without UNIT: every block's bytes)
def mm.e26(~rw: Mx -> Mk -> Mx, z: Bool, s: Sl, +o: Mk) -> Sl:
  match z s:
    case True{} Sl{c, t, a, d, h, x, +k}:
      match o:
        case Mk{sa, rs, +ssa, +srs, out, ors, +n, +kb, +k0, m, ab, f, wf, dd, nab, ado, aso,
          msk, tm}:
          mm.e26b(~rw, nv.scan(U32.to_nat((n * kb : U32)), (d, 0), 0, kb, ssa, srs, k0), c, t, a, h,
            x, k, o)
    case False{} s:
      mm.e3(~rw, s, o)

def mm.e2(~rw: Mx -> Mk -> Mx, z: Bool, s: Sl, +o: Mk,
  +e: U32) -> Sl:
  match z:
    case True{}:
      match o:
        case Mk{sa, rs, ssa, srs, out, ors, n, kb, k0, m, ab, +f, +wf, dd, nab, ado, aso, msk,
          tm}:
          mm.e26(~rw, U32.is_eq(wf, 3) && U32.is_eq((f .&. 1 : U32), 0), s, o)
    case False{}:
      mm.fl(s, e)

# MM with its rows computed by ~rw(st, o) (mm.rows in order, or bend/simpar.bend's halves in
# parallel)
def mm.ck(~rw: Mx -> Mk -> Mx, s: Sl, +i: Ins) -> Sl:
  match s:
    case Sl{c, t, a, d, h, x, +k}:
      match i:
        case Ins{+w0, +w1, +w2, +w3, +w4, +w5, +w6, +w7, +va, +vb, +vc, vd}:
          +f = fl(w0)
          +o = {Mk{(va + w1 : U32), w5, (vb + w2 : U32), w7, (vc + w3 : U32),
            (w6 .&. 65535 : U32), (w4 .&. 65535 : U32), ((w4 >> 16n : U32) .&. 255 : U32),
            (w4 >> 24n : U32), ((w6 >> 16n : U32) .&. 255 : U32), (w6 >> 24n : U32), f,
            ((f >> 4n : U32) .&. 3 : U32), Cf.d(k), Cf.ab(k), Cf.ado(k), Cf.aso(k),
            (Cf.dbm1(k) >> 2n : U32), (Cf.tw(k) - 1 : U32)} : Mk}
          +e1 = mm.chk(o, k)
          +e = F.sel(U32.is_eq(e1, 0), mm.schk(o, k), e1)
          mm.e2(~rw, U32.is_eq(e, 0), Sl{c, t, a, d, h, x, k}, o, e)

# the M rows from 0, row j's outputs by ~ns(n, st, o, j, sj, alpha) (its n outputs from 0)
def mm.rows(~ns: U32 -> Mx -> Mk -> U32 -> U32 -> U32 -> Mx, st: Mx, +o: Mk) -> Mx:
  match o:
    case Mk{sa, rs, ssa, srs, out, ors, n, kb, k0, +m, ab, f, wf, dd, nab, ado, aso, msk, tm}:
      mm.js(~ns, U32.to_nat(m), st, o, 0)

# MM, the sequential MXU (the ~mm of the step and run loops)
def mm.seq(s: Sl, +i: Ins) -> Sl:
  mm.ck(~(st => o => mm.rows(~(n => st2 => o2 => j => sj => al => mm.ns(U32.to_nat(n), st2,
    o2, j, 0, sj, al)), st, o)), s, i)
