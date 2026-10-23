# The ISA's word helpers (docs/isa.md), pure functions of 32-bit words that sim.bend uses: the
# high word of a product, the exact range check of every strided access, shifts and bytes,
# WAITW's condition, and the register write that keeps R0 at 0. A module of their own so that
# LAWS.bend's laws about them reach --verdict's kernel without the whole of sim.bend.
#
# Bend has no hex literals: the constants are decimal, their hex in the comments.
import Base
import ./fp32.bend as F

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

# R[rd] = v; R0 stays 0
def setr.go(z: Bool, c: Array<U32>, +r: U32, +v: U32) -> Array<U32>:
  match z:
    case True{}:
      c
    case False{}:
      c[r] <- v

def setr(c: Array<U32>, +r: U32, +v: U32) -> Array<U32>:
  setr.go(U32.is_eq(r, 0), c, r, v)

# WAITW's condition on the word it read (isa.waitw_holds): EQ, NE, GE (a signed difference)
def holds(+v: U32, +ref: U32, +cmp: U32, +mask: U32) -> Bool:
  +m = (v .&. mask : U32)
  (U32.is_eq(cmp, 0) && U32.is_eq(m, ref)) || (U32.is_eq(cmp, 1) && U32.is_ne(m, ref))
    || ((cmp >= 2 : U32) && ((m - ref : U32) < 2147483648 : U32))
