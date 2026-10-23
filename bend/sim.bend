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
  adv.sp(Array.get(U32, c, 16), pc)

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

# (placeholders until their units are written)
def x.qact(s: Sl, +i: Ins) -> Sl:
  ex.bad(s, i)

def x.qst(s: Sl, +i: Ins) -> Sl:
  ex.bad(s, i)

def x.dstep(s: Sl, +i: Ins) -> Sl:
  ex.bad(s, i)

def x.stream(s: Sl, +i: Ins) -> Sl:
  ex.bad(s, i)

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
