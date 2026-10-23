# The openTPU's bit-exact fp32 (docs/isa.md "Arithmetic") on U32 bit patterns.
#
# IEEE-754 binary32 with roundTiesToEven and flush to zero on inputs and outputs; add / sub /
# mul give the canonical NaN 0x7FC00000; the composite functions (exp2, log2, recip, rsqrt,
# the quantizer's q8) are the fixed sequences of add / mul docs/isa.md defines. Everything is
# integer arithmetic on the words (a softfloat), so the results do not depend on the host's
# FPU, and Bend's checker can evaluate them (LAWS.bend states facts about them, PROOF.bend
# proves them). opentpu/fp32.py is the reference model: tests/test_bend.py compares the two
# bit for bit on the RTL's test vectors.
#
# Bend has no hex literals: the constants are decimal, their hex in the comments.
import Base

# ------------------------------------------------------------------------------- words

def sel(c: Bool, a: U32, b: U32) -> U32:
  match c:
    case True{}:
      a
    case False{}:
      b

def b2u(c: Bool) -> U32:
  match c:
    case True{}:
      1
    case False{}:
      0

def shl.go(c: Bool, x: U32, +s: U32) -> U32:
  match c:
    case True{}:
      (x << U32.to_nat(s) : U32)
    case False{}:
      0

# x << s, 0 for s >= 32 (the guard keeps the checker's Nat small)
def shl(x: U32, +s: U32) -> U32:
  shl.go((s < 32 : U32), x, s)

def shr.go(c: Bool, x: U32, +s: U32) -> U32:
  match c:
    case True{}:
      (x >> U32.to_nat(s) : U32)
    case False{}:
      0

# x >> s (logical), 0 for s >= 32
def shr(x: U32, +s: U32) -> U32:
  shr.go((s < 32 : U32), x, s)

# x >> s with every bit shifted out ORed into bit 0 (the sticky bit)
def shr_sticky(+x: U32, +s: U32) -> U32:
  +lost = (x .&. (shl(1, s) - 1 : U32) : U32)
  (shr(x, s) .|. b2u(U32.is_ne(lost, 0)) : U32)

# count of leading zero bits (32 for 0)
def clz(+x: U32) -> U32:
  +c16 = (x < 65536 : U32)
  +x1 = sel(c16, (x << 16n : U32), x)
  +c8 = (x1 < 16777216 : U32)
  +x2 = sel(c8, (x1 << 8n : U32), x1)
  +c4 = (x2 < 268435456 : U32)
  +x3 = sel(c4, (x2 << 4n : U32), x2)
  +c2 = (x3 < 1073741824 : U32)
  +x4 = sel(c2, (x3 << 2n : U32), x3)
  +c1 = (x4 < 2147483648 : U32)
  +x5 = sel(c1, (x4 << 1n : U32), x4)
  (sel(c16, 16, 0) + sel(c8, 8, 0) + sel(c4, 4, 0) + sel(c2, 2, 0) + sel(c1, 1, 0)
    + b2u(U32.is_eq(x5, 0)) : U32)

# signed compares of two's complement words
def slt(+a: U32, +b: U32) -> Bool:
  ((a .^. 2147483648 : U32) < (b .^. 2147483648 : U32) : U32)

def neg(x: U32) -> U32:
  (0 - x : U32)

# ------------------------------------------------------------------------------- classes

# exponent field 0: a denormal or a zero flushes to the zero of its sign; a NaN keeps its bits
def ftz(+x: U32) -> U32:
  sel(U32.is_zero((x .&. 2139095040 : U32)), (x .&. 2147483648 : U32), x)

def is_nan(+x: U32) -> Bool:
  ((x .&. 2147483647 : U32) > 2139095040 : U32)

def is_inf(+x: U32) -> Bool:
  U32.is_eq((x .&. 2147483647 : U32), 2139095040)

# an exponent field of all ones: an infinity or a NaN
def is_special(+x: U32) -> Bool:
  U32.is_eq((x .&. 2139095040 : U32), 2139095040)

def is_zero(+x: U32) -> Bool:
  U32.is_zero((x .&. 2147483647 : U32))

def sign(+x: U32) -> U32:
  (x .&. 2147483648 : U32)

def expf(+x: U32) -> U32:
  ((x >> 23n : U32) .&. 255 : U32)

def mant(+x: U32) -> U32:
  (x .&. 8388607 : U32)

# a NaN as the canonical 0x7FC00000, every other word as it is (fp32.py's _canon)
def canon(+x: U32) -> U32:
  sel(is_nan(x), 2143289344, x)

# ------------------------------------------------------------------------------- add

# the rounding of a sum: s holds 27 bits, the 24-bit significand over a guard, a round and a
# sticky bit, its leading one at bit 26; e >= 1 the biased exponent of that leading one
def add.round(+sx: U32, +e: U32, +s: U32) -> U32:
  +m = (s >> 3n : U32)
  +g = (s .&. 7 : U32)
  +up = (g > 4 : U32) || (U32.is_eq(g, 4) && U32.is_eq((m .&. 1 : U32), 1))
  +m2 = (m + b2u(up) : U32)
  +ov = U32.is_eq(m2, 16777216)
  +e2 = (e + b2u(ov) : U32)
  +m3 = sel(ov, (m2 >> 1n : U32), m2)
  sel((e2 >= 255 : U32), (sx .|. 2139095040 : U32),
    (sx .|. (e2 << 23n : U32) .|. (m3 .&. 8388607 : U32) : U32))

# a difference normalized to bit 26 (lz left shifts); below the normal range it flushes to the
# zero of the sum's sign (a difference of normals is exact there: no rounding happens)
def add.diff(c: Bool, +sx: U32, +ex: U32, +s: U32) -> U32:
  match c:
    case True{}:
      0
    case False{}:
      +lz = (clz(s) - 5 : U32)
      sel((ex > lz : U32), add.round(sx, (ex - lz : U32), shl(s, lz)), sx)

def add.sum(c: Bool, +sx: U32, +ex: U32, +mx: U32, +ys: U32) -> U32:
  match c:
    case True{}:
      +s = (mx + ys : U32)
      +ov = (s >= 134217728 : U32)
      add.round(sx, (ex + b2u(ov) : U32),
        sel(ov, ((s >> 1n : U32) .|. (s .&. 1 : U32) : U32), s))
    case False{}:
      +s = (mx - ys : U32)
      add.diff(U32.is_eq(s, 0), sx, ex, s)

# |x| >= |y|, both normal
def add.norm(+x: U32, +y: U32) -> U32:
  +ex = expf(x)
  +mx = ((mant(x) .|. 8388608 : U32) << 3n : U32)
  +my = ((mant(y) .|. 8388608 : U32) << 3n : U32)
  +d = (ex - expf(y) : U32)
  add.sum(U32.is_zero(sign((x .^. y : U32))), sign(x), ex, mx, shr_sticky(my, d))

# y is a zero: x, or, with x a zero too, the zero whose sign is both signs' AND
def add.yzero(c: Bool, +x: U32, +y: U32) -> U32:
  match c:
    case True{}:
      sel(is_zero(x), (x .&. y : U32), x)
    case False{}:
      add.norm(x, y)

# both finite (flushed): x the larger magnitude
def add.fin(+a: U32, +b: U32) -> U32:
  +swap = ((a .&. 2147483647 : U32) < (b .&. 2147483647 : U32) : U32)
  +x = sel(swap, b, a)
  +y = sel(swap, a, b)
  add.yzero(is_zero(y), x, y)

# an infinity or a NaN among the operands: the canonical NaN for a NaN or inf - inf, else the
# infinity
def add.inf(+a: U32, +b: U32) -> U32:
  +nan = is_nan(a) || is_nan(b) || (is_inf(a) && is_inf(b) && U32.is_ne(sign(a), sign(b)))
  sel(nan, 2143289344, sel(is_inf(a), a, b))

def add.ab(c: Bool, +a: U32, +b: U32) -> U32:
  match c:
    case True{}:
      add.inf(a, b)
    case False{}:
      add.fin(a, b)

def add(+a: U32, +b: U32) -> U32:
  +x = ftz(a)
  +y = ftz(b)
  add.ab(is_special(x) || is_special(y), x, y)

def sub(+a: U32, +b: U32) -> U32:
  add(a, (ftz(b) .^. 2147483648 : U32))

# ------------------------------------------------------------------------------- mul

# the rounding of a product: h the 24-bit significand (leading one at bit 23), l the next 24
# bits, e the biased exponent + 127. Normal: round to 24 bits. Biased exponent 0: IEEE rounds
# on the subnormal grid (23 bits there), and a result that reaches 2^-126 is kept; anything
# else below 2^-126 flushes to the zero of the sign.
def mul.round(+s: U32, +e: U32, +h: U32, +l: U32) -> U32:
  +g = (l >> 23n : U32)
  +st = U32.is_ne((l .&. 8388607 : U32), 0)
  +up = U32.is_eq(g, 1) && (st || U32.is_eq((h .&. 1 : U32), 1))
  +m = (h + b2u(up) : U32)
  +ov = U32.is_eq(m, 16777216)
  +en = (e - 127 + b2u(ov) : U32)
  +norm = sel((en >= 255 : U32), (s .|. 2139095040 : U32),
    (s .|. (en << 23n : U32) .|. (sel(ov, (m >> 1n : U32), m) .&. 8388607 : U32) : U32))
  +q = (h >> 1n : U32)
  +g0 = U32.is_eq((h .&. 1 : U32), 1)
  +up0 = g0 && (U32.is_ne(l, 0) || U32.is_eq((q .&. 1 : U32), 1))
  +sub0 = sel(U32.is_eq((q + b2u(up0) : U32), 8388608), (s .|. 8388608 : U32), s)
  sel((e >= 128 : U32), norm, sel(U32.is_eq(e, 127), sub0, s))

# both normal: the 48-bit product of the 24-bit significands from 12-bit halves
def mul.norm(+s: U32, +a: U32, +b: U32) -> U32:
  +ma = (mant(a) .|. 8388608 : U32)
  +mb = (mant(b) .|. 8388608 : U32)
  +a1 = (ma >> 12n : U32)
  +a0 = (ma .&. 4095 : U32)
  +b1 = (mb >> 12n : U32)
  +b0 = (mb .&. 4095 : U32)
  +mid = ((a1 * b0 : U32) + (a0 * b1 : U32) : U32)
  +t = ((a0 * b0 : U32) + ((mid .&. 4095 : U32) << 12n : U32) : U32)
  +lo = (t .&. 16777215 : U32)
  +hi = ((a1 * b1 : U32) + (mid >> 12n : U32) + (t >> 24n : U32) : U32)
  +top = (hi >= 8388608 : U32)
  mul.round(s, (expf(a) + expf(b) + b2u(top) : U32),
    sel(top, hi, ((hi << 1n : U32) .|. (lo >> 23n : U32) : U32)),
    sel(top, lo, ((lo << 1n : U32) .&. 16777215 : U32)))

def mul.fin(c: Bool, +s: U32, +a: U32, +b: U32) -> U32:
  match c:
    case True{}:
      s
    case False{}:
      mul.norm(s, a, b)

# an infinity or a NaN: the canonical NaN for a NaN or inf * 0, else the signed infinity
def mul.inf(+s: U32, +a: U32, +b: U32) -> U32:
  +nan = is_nan(a) || is_nan(b) || is_zero(a) || is_zero(b)
  sel(nan, 2143289344, (s .|. 2139095040 : U32))

def mul.ab(c: Bool, +s: U32, +a: U32, +b: U32) -> U32:
  match c:
    case True{}:
      mul.inf(s, a, b)
    case False{}:
      mul.fin(is_zero(a) || is_zero(b), s, a, b)

def mul(+a: U32, +b: U32) -> U32:
  +x = ftz(a)
  +y = ftz(b)
  mul.ab(is_special(x) || is_special(y), sign((x .^. y : U32)), x, y)

# ------------------------------------------------------------------------------- integers

# the low word of the 64-bit (hi:lo) >> k, k in 0..63
def shr64(+hi: U32, +lo: U32, +k: U32) -> U32:
  sel((k < 32 : U32), (shr(lo, k) .|. shl(hi, (32 - k : U32)) : U32), shr(hi, (k - 32 : U32)))

# whether the low k bits of (hi:lo) are not all zero, k in 0..64
def low64(+hi: U32, +lo: U32, +k: U32) -> Bool:
  U32.is_eq(sel((k <= 32 : U32), b2u(U32.is_ne((lo .&. (shl(1, k) - 1 : U32) : U32), 0)),
    b2u(U32.is_ne(lo, 0) || U32.is_ne((hi .&. (shl(1, (k - 32 : U32)) - 1 : U32) : U32), 0))), 1)

# bit k of (hi:lo), k in 0..63
def bit64(+hi: U32, +lo: U32, +k: U32) -> U32:
  (sel((k < 32 : U32), shr(lo, k), shr(hi, (k - 32 : U32))) .&. 1 : U32)

# a magnitude (hi:lo) != 0 with its leading one at bit p: rounded to 24 bits
def i2f.round(+s: U32, +hi: U32, +lo: U32, +p: U32) -> U32:
  +k = (p - 23 : U32)
  +q = shr64(hi, lo, k)
  +up = U32.is_eq(bit64(hi, lo, (k - 1 : U32)), 1)
    && (low64(hi, lo, (k - 1 : U32)) || U32.is_eq((q .&. 1 : U32), 1))
  +m = (q + b2u(up) : U32)
  +ov = U32.is_eq(m, 16777216)
  (s .|. ((127 + p + b2u(ov) : U32) << 23n : U32) .|. (sel(ov, (m >> 1n : U32), m) .&. 8388607 : U32)
    : U32)

def i2f.mag(c: Bool, +s: U32, +hi: U32, +lo: U32) -> U32:
  match c:
    case True{}:
      0
    case False{}:
      +p = sel(U32.is_eq(hi, 0), (31 - clz(lo) : U32), (63 - clz(hi) : U32))
      sel((p <= 23 : U32), (s .|. ((127 + p : U32) << 23n : U32) .|. (shl(lo, (23 - p : U32)) .&. 8388607 : U32) : U32),
        i2f.round(s, hi, lo, p))

# a signed 64-bit integer (hi:lo, two's complement) to fp32, roundTiesToEven
def i2f64(+hi: U32, +lo: U32) -> U32:
  +ng = (hi >= 2147483648 : U32)
  +nlo = neg(lo)
  +nhi = ((hi .^. 4294967295 : U32) + b2u(U32.is_eq(lo, 0)) : U32)
  +mhi = sel(ng, nhi, hi)
  +mlo = sel(ng, nlo, lo)
  i2f.mag(U32.is_eq(mhi, 0) && U32.is_eq(mlo, 0), sel(ng, 2147483648, 0), mhi, mlo)

# a signed 32-bit integer to fp32
def i2f(+x: U32) -> U32:
  i2f64(sel((x >= 2147483648 : U32), 4294967295, 0), x)

# RLD's conversion (isa.f2i): truncation toward zero; |x| < 1 gives 0, and |x| >= 2^31,
# infinities and NaN 0x80000000
def f2i(+x: U32) -> U32:
  +e = expf(x)
  +m = (mant(x) .|. 8388608 : U32)
  +v = sel((e >= 150 : U32), shl(m, (e - 150 : U32)), shr(m, (150 - e : U32)))
  sel((e < 127 : U32), 0, sel((e >= 158 : U32), 2147483648, sel(U32.is_eq(sign(x), 0), v, neg(v))))

# floor of a finite x with |x| < 2^24, as a signed 32-bit integer
def floor_i(+x: U32) -> U32:
  +e = expf(x)
  +m = (mant(x) .|. 8388608 : U32)
  +sh = (150 - e : U32)
  +ip = sel((e < 127 : U32), 0, sel((e >= 150 : U32), shl(m, (e - 150 : U32)), shr(m, sh)))
  +fr = ((e < 127 : U32) && Bool.not(is_zero(x)))
    || ((e >= 127 : U32) && ((e < 150 : U32) && U32.is_ne((m .&. (shl(1, sh) - 1 : U32) : U32), 0)))
  sel(U32.is_eq(sign(x), 0), ip, neg((ip + b2u(fr) : U32)))

# ------------------------------------------------------------------------------- order

# the total order's key: -NaN < -inf < ... < -0 < +0 < ... < +inf < NaN on flushed bits
def key(+x: U32) -> U32:
  +u = ftz(x)
  sel(U32.is_eq(sign(u), 0), (u .|. 2147483648 : U32), (u .^. 4294967295 : U32))

def gt(+a: U32, +b: U32) -> Bool:
  (key(a) > key(b) : U32)

def fmax(+a: U32, +b: U32) -> U32:
  sel(gt(a, b), ftz(a), ftz(b))

def fmin(+a: U32, +b: U32) -> U32:
  sel(gt(b, a), ftz(a), ftz(b))

def fabs(+a: U32) -> U32:
  (ftz(a) .&. 2147483647 : U32)

# ------------------------------------------------------------------------------- composites

# x times 2^k on the exponent field (bits + k << 23, mod 2^32; exact while the field stays in
# 1..254)
def exp_add(+x: U32, +k: U32) -> U32:
  (x + (k << 23n : U32) : U32)

# p = p * f + c, the Horner step (a rounded mul, then a rounded add)
def horner(+p: U32, +f: U32, +c: U32) -> U32:
  add(mul(p, f), c)

# exp2 (docs/isa.md): x < -126 gives +0, x >= 128 (or NaN) +inf; else i = floor(x),
# f = x - i2f(i), p = C7, p = p*f + Ck for k = 6..0, and i added to p's exponent field
def exp2(+x0: U32) -> U32:
  +x = canon(ftz(x0))
  +lo = gt(3271294976, x)
  +hi = Bool.not(gt(1124073472, x))
  +xf = sel(Bool.not(lo) && Bool.not(hi), x, 0)
  +i = floor_i(xf)
  +f = sub(xf, i2f(i))
  +p7 = horner(934528982, f, 957792380)
  +p6 = horner(p7, f, 984605430)
  +p5 = horner(p6, f, 1008567604)
  +p4 = horner(p5, f, 1029921086)
  +p3 = horner(p4, f, 1047920105)
  +p2 = horner(p3, f, 1060205080)
  +p1 = horner(p2, f, 1065353216)
  sel(hi, 2139095040, sel(lo, 0, exp_add(p1, i)))

# log2 (docs/isa.md): x = 2^e * m, m in [sqrt(1/2), sqrt(2)), t = m - 1, q = C9,
# q = q*t + Ck for k = 8..1, the result q*t + i2f(e); +-0 gives -inf, x < 0 the canonical NaN,
# +inf +inf
def log2(+x0: U32) -> U32:
  +x = ftz(x0)
  +fr = mant(x)
  +ex = expf(x)
  +ge = (fr >= 3474675 : U32)
  +e = sel(U32.is_eq(ex, 0), 0, (ex - 127 + b2u(ge) : U32))
  +m = ((sel(ge, 126, 127) << 23n : U32) .|. fr : U32)
  +t = add(m, 3212836864)
  +q8 = horner(1040243571, t, 3193192078)
  +q7 = horner(q8, t, 1046275214)
  +q6 = horner(q7, t, 3195317746)
  +q5 = horner(q6, t, 1049848258)
  +q4 = horner(q5, t, 3199774247)
  +q3 = horner(q4, t, 1056324075)
  +q2 = horner(q3, t, 3208161848)
  +q1 = horner(q2, t, 1069066811)
  +r = horner(q1, t, i2f(e))
  +zero = is_zero(x)
  +r1 = sel(U32.is_eq(ex, 255), canon(x), r)
  +r2 = sel(U32.is_ne(sign(x), 0) && Bool.not(zero), 2143289344, r1)
  ftz(sel(zero, 4286578688, r2))

# y = y * (2 - a*y)
def recip.newton(+a: U32, +y: U32) -> U32:
  mul(y, sub(1073741824, mul(a, y)))

# recip (docs/isa.md): on the significand a in [1, 2): the seed, two Newton steps, the
# correction step y*e + y with e = 1 - a*y, then 127 - (x's field) added to the exponent field
# and x's sign; 0 gives +0, |x| >= 2^126 (inf, NaN bits) a zero of x's sign
def recip(+x0: U32) -> U32:
  +x = ftz(x0)
  +ab = (x .&. 2147483647 : U32)
  +zero = U32.is_eq(ab, 0)
  +big = (ab >= 2122317824 : U32)
  +a = (1065353216 .|. (ab .&. 8388607 : U32) : U32)
  +y1 = recip.newton(a, recip.newton(a, (2129859011 - a : U32)))
  +y = add(mul(y1, sub(1065353216, mul(a, y1))), y1)
  +k = sel(big || zero, 0, (127 - (ab >> 23n : U32) : U32))
  +r = (sel(big, 0, exp_add(y, k)) .|. sign(x) : U32)
  sel(zero, 0, r)

# y = y * (1.5 - h*(y*y))
def rsqrt.newton(+h: U32, +y: U32) -> U32:
  mul(y, sub(1069547520, mul(h, mul(y, y))))

# rsqrt (docs/isa.md): x <= 0 (any sign bit), +0 and +inf give +0. Exponent field <= 2 runs
# on 16x and the result is times 4; field 250..254 on x/16, then / 4.
def rsqrt(+x0: U32) -> U32:
  +x = ftz(x0)
  +f = expf(x)
  +bad = U32.is_ne(sign(x), 0) || U32.is_eq(x, 0) || U32.is_eq(x, 2139095040)
  # k as a signed word: 4, -4 or 0
  +k = sel(bad, 0, sel((f <= 2 : U32), 4, sel((f >= 250 : U32) && (f < 255 : U32), 4294967292, 0)))
  +xs = sel(bad, 1065353216, exp_add(x, k))
  +h = mul(1056964608, xs)
  +y1 = rsqrt.newton(h, rsqrt.newton(h, (1597463007 - (xs >> 1n : U32) : U32)))
  +y = add(mul(y1, sub(1056964608, mul(h, mul(y1, y1)))), y1)
  # k // 2: 2, -2 or 0
  +k2 = sel(U32.is_eq(k, 4), 2, sel(U32.is_eq(k, 0), 0, 4294967294))
  sel(bad, 0, exp_add(y, k2))

# q8: round half to even to an integer, saturate to [-127, 127], NaN 0; a signed word
def q8(+x0: U32) -> U32:
  +x = ftz(x0)
  +a = (x .&. 2147483647 : U32)
  +e = expf(x)
  +m = (mant(x) .|. 8388608 : U32)
  +sh = (150 - e : U32)
  +q = shr(m, sh)
  +rem = (m .&. (shl(1, sh) - 1 : U32) : U32)
  +half = shl(1, (sh - 1 : U32))
  +up = (rem > half : U32) || (U32.is_eq(rem, half) && U32.is_eq((q .&. 1 : U32), 1))
  +r = sel((a >= 1124073472 : U32), 127, sel((e < 126 : U32), 0, U32.min((q + b2u(up) : U32), 127)))
  sel(is_nan(x), 0, sel(U32.is_eq(sign(x), 0), r, neg(r)))

# the quantizer's d (docs/isa.md "Quantization"): the largest d <= 31 with a * 2^d <= amax
# (a <= amax, both flushed and >= 0); 0 where a is 0
def mx_offset(+a: U32, +amax: U32) -> U32:
  +ea = (a >> 23n : U32)
  +em = (amax >> 23n : U32)
  +fl = b2u((mant(a) > mant(amax) : U32))
  +d = (em - ea - fl : U32)
  sel(U32.is_eq(a, 0) || slt(em, (ea + fl : U32)), 0, U32.min(d, 31))

# VOP OUTER's a*d + b*c: two rounded products, then a rounded add
def outer(+a: U32, +d: U32, +b: U32, +c: U32) -> U32:
  add(mul(a, d), mul(b, c))
