# The laws of the Bend ISA simulator (docs/bend.md "Laws"): what sim.bend and fp32.bend must
# satisfy, stated from docs/isa.md. PROOF.bend proves every one: `bend bend/PROOF.bend
# --check-only` prints ALL PROOFS CHECK, and `--verdict` checks the proofs again with BendTT's
# kernel, which has a proof in Lean.
#
# A law with `for` holds for every value of its variables: every 32-bit word, every control
# array. The others are about named inputs, proven by evaluation in the checker: the IEEE-754
# corners of the arithmetic (roundTiesToEven, flush to zero on inputs and outputs, the canonical
# NaN; each value derived by hand and equal to opentpu/fp32.py's), every byte, the range check
# at the 32-bit wrap. prog/LAWS.bend has the laws about whole programs run on a slice (they
# import all of sim.bend, which --verdict's kernel does not fit in memory).
#
# Bend has no hex literals: the words are decimal, their hex or value in the comments.
import Base
import ./fp32.bend as F
import ./word.bend as W

# ------------------------------------------------------------------------------- fp32, every word

# flushing to zero twice is flushing once
law ftz_idem:
  for +x: U32
  {F.ftz(F.ftz(x)) == F.ftz(x) : U32}

# canonicalizing a NaN twice is doing it once
law canon_idem:
  for +x: U32
  {F.canon(F.canon(x)) == F.canon(x) : U32}

# |x| has its sign bit clear
law fabs_sign:
  for +x: U32
  {F.sign(F.fabs(x)) == 0 : U32}

# max and min of a word with itself: the word, flushed
law fmax_self:
  for +x: U32
  {F.fmax(x, x) == F.ftz(x) : U32}

law fmin_self:
  for +x: U32
  {F.fmin(x, x) == F.ftz(x) : U32}

# ------------------------------------------------------------------------------- fp32, corners

# 1 + 2 = 3
law add_exact:
  {F.add(1065353216, 1073741824) == 1077936128 : U32}

# 1 + 2^-24 is a tie between 1 and 1 + 2^-23: to the even, 1
law add_tie_down:
  {F.add(1065353216, 864026624) == 1065353216 : U32}

# (1 + 2^-23) + 2^-24 is a tie: to the even, 1 + 2^-22 (0x3F800002)
law add_tie_up:
  {F.add(1065353217, 864026624) == 1065353218 : U32}

# (2 - 2^-23) + 2^-24 rounds up into the next binade: 2
law add_carry:
  {F.add(1073741823, 864026624) == 1073741824 : U32}

# the largest finite plus itself overflows to +inf
law add_overflow:
  {F.add(2139095039, 2139095039) == 2139095040 : U32}

# +inf + -inf is the canonical NaN 0x7FC00000
law add_inf_minus_inf:
  {F.add(2139095040, 4286578688) == 2143289344 : U32}

# a NaN operand (here a signaling one, 0x7F800001) gives the canonical NaN
law add_nan:
  {F.add(2139095041, 1065353216) == 2143289344 : U32}

# 3 + -3 = +0
law add_cancel:
  {F.add(1077936128, 3225419776) == 0 : U32}

# -0 + -0 = -0; +0 + -0 = +0
law add_neg_zeros:
  {F.add(2147483648, 2147483648) == 2147483648 : U32}

law add_mixed_zeros:
  {F.add(0, 2147483648) == 0 : U32}

# denormal inputs flush: 2^-127 + 2^-127 = +0 (IEEE without the flush: 2^-126)
law add_ftz_in:
  {F.add(4194304, 4194304) == 0 : U32}

# a denormal result flushes: 1.75 * 2^-126 - 2^-126 = +0 (IEEE without the flush: 1.5 * 2^-127)
law add_ftz_out:
  {F.add(14680064, 2155872256) == 0 : U32}

# 1 + 2^-24 (1 + 2^-23) is just above the tie: the sticky bit rounds it up to 1 + 2^-23
law add_sticky:
  {F.add(1065353216, 864026625) == 1065353217 : U32}

# 1.5 - 1.5 = +0
law sub_self:
  {F.sub(1069547520, 1069547520) == 0 : U32}

# 1.5 * 1.5 = 2.25
law mul_exact:
  {F.mul(1069547520, 1069547520) == 1074790400 : U32}

# (1 + 2^-23)^2 = 1 + 2^-22 + 2^-46 rounds to 1 + 2^-22
law mul_round:
  {F.mul(1065353217, 1065353217) == 1065353218 : U32}

# (1 + 2^-23) * 1.5 = 1.5 + 1.5 * 2^-23 is a tie: to the even, 1.5 + 2^-22 (0x3FC00002)
law mul_tie:
  {F.mul(1065353217, 1069547520) == 1069547522 : U32}

# 0x3FAA3400 * 0x3F9E7800: the 24 bits below the significand are 0xC00000, a guard bit and one
# sticky bit just under it, so the product rounds up (to 0x3FD2B7C1), not to the even
law mul_sticky:
  {F.mul(1068119040, 1067350016) == 1070774209 : U32}

# 2^127 * 2 overflows to +inf
law mul_overflow:
  {F.mul(2130706432, 1073741824) == 2139095040 : U32}

# 0 * inf is the canonical NaN
law mul_zero_inf:
  {F.mul(0, 2139095040) == 2143289344 : U32}

# -0 * 5 = -0; -2 * 3 = -6
law mul_neg_zero:
  {F.mul(2147483648, 1084227584) == 2147483648 : U32}

law mul_signs:
  {F.mul(3221225472, 1077936128) == 3233808384 : U32}

# 2^-64 * 2^-64 = 2^-128 flushes to +0
law mul_ftz_out:
  {F.mul(528482304, 528482304) == 0 : U32}

# 2^-64 * ((2 - 2^-23) * 2^-63) = 2^-126 (1 - 2^-24) rounds up to the smallest normal, 2^-126,
# which is kept
law mul_min_normal:
  {F.mul(528482304, 545259519) == 8388608 : U32}

# the order: -0 < +0, the canonical NaN above everything, a NaN with the sign below -inf;
# denormals compare as their zeros
law fmax_zeros:
  {F.fmax(2147483648, 0) == 0 : U32}

law fmin_zeros:
  {F.fmin(2147483648, 0) == 2147483648 : U32}

law fmax_nan:
  {F.fmax(2143289344, 1065353216) == 2143289344 : U32}

law fmin_neg_nan:
  {F.fmin(4290772992, 4286578688) == 4290772992 : U32}

law fmax_denormals:
  {F.fmax(1, 2147483648) == 0 : U32}

# |-3| = 3
law fabs_neg:
  {F.fabs(3225419776) == 1077936128 : U32}

# integers to fp32 round to the even: 2^24 + 1 -> 2^24, 2^24 + 3 -> 2^24 + 4; -2^31 and -1 exact
law i2f_tie:
  {F.i2f(16777217) == 1266679808 : U32}

law i2f_tie_up:
  {F.i2f(16777219) == 1266679810 : U32}

law i2f_min:
  {F.i2f(2147483648) == 3472883712 : U32}

law i2f_minus_one:
  {F.i2f(4294967295) == 3212836864 : U32}

# fp32 to integers truncate toward zero: 2.75 -> 2, -2.75 -> -2; 2^31 and NaN -> 0x80000000
law f2i_trunc:
  {F.f2i(1076887552) == 2 : U32}

law f2i_neg:
  {F.f2i(3224371200) == 4294967294 : U32}

law f2i_big:
  {F.f2i(1325400064) == 2147483648 : U32}

law f2i_nan:
  {F.f2i(2143289344) == 2147483648 : U32}

# floor: -0.5 -> -1, 2.5 -> 2
law floor_neg_half:
  {F.floor_i(3204448256) == 4294967295 : U32}

law floor_pos:
  {F.floor_i(1075838976) == 2 : U32}

# the composite functions (the fixed sequences of docs/isa.md) are exact at powers of two and
# correctly rounded at these: exp2(0) = 1, exp2(3) = 8, exp2(-1) = 0.5; below -126 +0, from 128
# +inf
law exp2_zero:
  {F.exp2(0) == 1065353216 : U32}

law exp2_three:
  {F.exp2(1077936128) == 1090519040 : U32}

law exp2_minus_one:
  {F.exp2(3212836864) == 1056964608 : U32}

law exp2_under:
  {F.exp2(3271426048) == 0 : U32}

law exp2_over:
  {F.exp2(1124073472) == 2139095040 : U32}

# log2(8) = 3, log2(1) = +0, log2(0) = -inf, log2(-1) = NaN
law log2_eight:
  {F.log2(1090519040) == 1077936128 : U32}

law log2_one:
  {F.log2(1065353216) == 0 : U32}

# the significand at sqrt(2)'s (3FB504F3) takes the lower range: log2 0.49999994 (3EFFFFFE);
# taken as 1.414.., it would be 0.5 (the boundary's >= as >)
law log2_sqrt2:
  {F.log2(1068827891) == 1056964606 : U32}

law log2_zero:
  {F.log2(0) == 4286578688 : U32}

law log2_neg:
  {F.log2(3212836864) == 2143289344 : U32}

# 1/2 = 0.5, 1/3 = 0x3EAAAAAB (correctly rounded), 1/sqrt(4) = 0.5, 1/sqrt(2) = 0x3F3504F3
# (correctly rounded)
law recip_two:
  {F.recip(1073741824) == 1056964608 : U32}

law recip_three:
  {F.recip(1077936128) == 1051372203 : U32}

law rsqrt_four:
  {F.rsqrt(1082130432) == 1056964608 : U32}

law rsqrt_two:
  {F.rsqrt(1073741824) == 1060439283 : U32}

# the quantizer: 2.5 -> 2 and -3.5 -> -4 (half to even), 200 -> 127 and -200 -> -127
# (saturation: never -128), NaN -> 0
law q8_half_even:
  {F.q8(1075838976) == 2 : U32}

law q8_half_even_neg:
  {F.q8(3227516928) == 4294967292 : U32}

law q8_sat:
  {F.q8(1128792064) == 127 : U32}

# 127.5 is a tie whose even is 128: saturated to 127
law q8_sat_tie:
  {F.q8(1124007936) == 127 : U32}

law q8_sat_neg:
  {F.q8(3276275712) == 4294967169 : U32}

law q8_nan:
  {F.q8(2143289344) == 0 : U32}

# ------------------------------------------------------------------------------- the machine

# R0 stays 0: writing it leaves the control words as they were
law r0_stays_zero:
  for -c: Array<U32>
  for +v: U32
  {W.setr(c, 0, v) == c : Array<U32>}

# WAITW's conditions (isa.waitw_holds): EQ, NE and GE (the masked word minus ref, signed,
# not negative)
law waitw_eq:
  for +v: U32
  for +ref: U32
  for +mask: U32
  {W.holds(v, ref, 0, mask) == U32.is_eq((v .&. mask : U32), ref) : Bool}

law waitw_ne:
  for +v: U32
  for +ref: U32
  for +mask: U32
  {W.holds(v, ref, 1, mask) == U32.is_ne((v .&. mask : U32), ref) : Bool}

law waitw_ge:
  for +v: U32
  for +ref: U32
  for +mask: U32
  {W.holds(v, ref, 2, mask) == (((v .&. mask : U32) - ref : U32) < 2147483648 : U32) : Bool}

# a byte's signed value: b for b < 128, b - 256 (mod 2^32) above
def int8(+b: U32) -> U32:
  F.sel((b < 128 : U32), b, (b - 256 : U32))

def sx8_ok(+b: U32) -> Bool:
  U32.is_eq(W.sx8(b), int8(b))

# p of the 2^k words base, base + 1, ..., base + 2^k - 1
def every(~p: U32 -> Bool, +k: Nat, +base: U32) -> Bool:
  match k:
    case 0n:
      p(base)
    case 1n+q:
      every(~p, q, base) && every(~p, q, (base + (1 << q : U32) : U32))

# sx8 is the byte's signed value, for every byte
law sx8_every_byte:
  {every(~(b => sx8_ok(b)), 8n, 0) == True{} : Bool}

# the strided range check is exact at the 32-bit wrap: the last byte is in range, a second one
# past it is not; ends of exactly 2^32 (a sum, a product) are not
law rng_last:
  {W.rng(4294967295, 1, 0, 1, 0, 4294967295) == True{} : Bool}

law rng_wrap:
  {W.rng(4294967295, 2, 1, 1, 0, 4294967295) == False{} : Bool}

law rng_sum_wrap:
  {W.rng(0, 2, 2147483648, 2, 2147483648, 4294967295) == False{} : Bool}

law rng_product_wrap:
  {W.rng(0, 65537, 65536, 1, 0, 4294967295) == False{} : Bool}

# 16 + 3 * 4 + 2 * 64 = 156: in range of 156, not of 155
law rng_edge_in:
  {W.rng(16, 4, 4, 3, 64, 156) == True{} : Bool}

law rng_edge_out:
  {W.rng(16, 4, 4, 3, 64, 155) == False{} : Bool}

# the high word of 64-bit products
law mulhi_max:
  {W.mulhi(4294967295, 4294967295) == 4294967294 : U32}

law mulhi_carry:
  {W.mulhi(65536, 65536) == 1 : U32}

law mulhi_mixed:
  {W.mulhi(12345678, 87654321) == 251958 : U32}
