# The proofs of LAWS.bend's laws (docs/bend.md "Laws"). `bend bend/PROOF.bend --check-only`
# checks them; `--verdict` checks them again with BendTT's kernel.
#
# Bend has no tactics: a proof is a def whose type is the law. {==} proves an equality whose
# sides compute to the same term (the corners and the programs: the checker runs them), a match
# splits the cases, a recursive call is the induction hypothesis, and %e : P rewrites the goal P
# with e : {a == b : T} (the goal is P with b at the _, and becomes P with a there). The laws
# about every word go down to Base's Word(32n), the bits of a U32: the lemmas below prove the
# bit facts by induction on the word's length.
import Base
import ./fp32.bend as F
import ./sim.bend as S
import ./LAWS.bend as Laws

# ------------------------------------------------------------------------------- lemmas

# AND is associative on words of any length
law and_assoc:
  for n: Nat
  for a: Word(n)
  for b: Word(n)
  for c: Word(n)
  {Word.and(n, Word.and(n, a, b), c) == Word.and(n, a, Word.and(n, b, c)) : Word(n)}

def and_assoc(n, a, b, c):
  match n:
    case 0n:
      match a b c:
        case WNil{} WNil{} WNil{}:
          {==}
    case 1n+p:
      match a b c:
        case WCon{False{}, at} WCon{bh, bt} WCon{ch, ct}:
          %and_assoc(p, at, bt, ct) : {WCon{False{}, Word.and(p, Word.and(p, at, bt), ct)}
            == WCon{False{}, _} : Word(1n+p)}
          {==}
        case WCon{True{}, at} WCon{bh, bt} WCon{ch, ct}:
          %and_assoc(p, at, bt, ct) : {WCon{Bool.and(bh, ch), Word.and(p, Word.and(p, at, bt),
            ct)} == WCon{Bool.and(bh, ch), _} : Word(1n+p)}
          {==}

# x AND 0 = 0
law and_zero:
  for n: Nat
  for a: Word(n)
  {Word.and(n, a, Word.zero(n)) == Word.zero(n) : Word(n)}

def and_zero(n, a):
  match n:
    case 0n:
      match a:
        case WNil{}:
          {==}
    case 1n+p:
      match a:
        case WCon{False{}, at}:
          %and_zero(p, at) : {WCon{False{}, Word.and(p, at, Word.zero(p))}
            == WCon{False{}, _} : Word(1n+p)}
          {==}
        case WCon{True{}, at}:
          %and_zero(p, at) : {WCon{False{}, Word.and(p, at, Word.zero(p))}
            == WCon{False{}, _} : Word(1n+p)}
          {==}

# the two on U32, each also reversed (a rewrite %e turns the goal's right side of e into its
# left side)
law u32_and_assoc:
  for a: U32
  for b: U32
  for c: U32
  {((a .&. b : U32) .&. c : U32) == (a .&. (b .&. c : U32) : U32) : U32}

def u32_and_assoc(a, b, c):
  match a b c:
    case U32{x} U32{y} U32{z}:
      %and_assoc(32n, x, y, z) : {U32{Word.and(32n, Word.and(32n, x, y), z)} == U32{_} : U32}
      {==}

law u32_and_assoc_r:
  for a: U32
  for b: U32
  for c: U32
  {(a .&. (b .&. c : U32) : U32) == ((a .&. b : U32) .&. c : U32) : U32}

def u32_and_assoc_r(a, b, c):
  Equal.sym(U32, ((a .&. b : U32) .&. c : U32), (a .&. (b .&. c : U32) : U32),
    u32_and_assoc(a, b, c))

law u32_and_zero:
  for a: U32
  {(a .&. 0 : U32) == 0 : U32}

def u32_and_zero(a):
  match a:
    case U32{x}:
      %and_zero(32n, x) : {U32{Word.and(32n, x, Word.zero(32n))} == U32{_} : U32}
      {==}

law u32_and_zero_r:
  for a: U32
  {0 == (a .&. 0 : U32) : U32}

def u32_and_zero_r(a):
  Equal.sym(U32, (a .&. 0 : U32), 0, u32_and_zero(a))

# a select between a word and itself is that word
law sel_same:
  for c: Bool
  for -a: U32
  {F.sel(c, a, a) == a : U32}

def sel_same(c, a):
  match c:
    case True{}:
      {==}
    case False{}:
      {==}

# b OR False = b, and reversed
law or_false:
  for b: Bool
  {Bool.or(b, False{}) == b : Bool}

def or_false(b):
  match b:
    case True{}:
      {==}
    case False{}:
      {==}

law or_false_r:
  for b: Bool
  {b == Bool.or(b, False{}) : Bool}

def or_false_r(b):
  Equal.sym(Bool, Bool.or(b, False{}), b, or_false(b))

# ------------------------------------------------------------------------------- fp32, every word

# 2139095040 = 0x7F800000 (the exponent field), 2147483648 = 0x80000000 (the sign),
# 2147483647 = 0x7FFFFFFF (the magnitude), 2143289344 = 0x7FC00000 (the canonical NaN)

# a word's sign alone is its own flush: (x & S) & E = x & (S & E) = x & 0 = 0 selects
# (x & S) & S = x & S
law ftz_sign:
  for +x: U32
  {F.ftz((x .&. 2147483648 : U32)) == (x .&. 2147483648 : U32) : U32}

def ftz_sign(x):
  %u32_and_assoc_r(x, 2147483648, 2139095040) : {F.sel(U32.is_zero(_),
    ((x .&. 2147483648 : U32) .&. 2147483648 : U32), (x .&. 2147483648 : U32))
    == (x .&. 2147483648 : U32) : U32}
  %u32_and_zero_r(x) : {F.sel(U32.is_zero(_),
    ((x .&. 2147483648 : U32) .&. 2147483648 : U32), (x .&. 2147483648 : U32))
    == (x .&. 2147483648 : U32) : U32}
  %u32_and_assoc_r(x, 2147483648, 2147483648) : {_ == (x .&. 2147483648 : U32) : U32}
  {==}

# ftz(x) is x & S (exponent field 0: ftz_sign) or x (else: the same test again)
def ftz_idem.go(+x: U32, c: Bool, e: {U32.is_zero((x .&. 2139095040 : U32)) == c : Bool}) ->
  {F.ftz(F.sel(c, (x .&. 2147483648 : U32), x)) == F.sel(c, (x .&. 2147483648 : U32), x) : U32}:
  match c:
    case True{}:
      ftz_sign(x)
    case False{}:
      %Equal.sym(Bool, U32.is_zero((x .&. 2139095040 : U32)), False{}, e) :
        {F.sel(_, (x .&. 2147483648 : U32), x) == x : U32}
      {==}

def Laws.ftz_idem(x):
  ftz_idem.go(x, U32.is_zero((x .&. 2139095040 : U32)), {==})

# canon(x) is the canonical NaN (a NaN itself) or x (not a NaN: the same test again)
def canon_idem.go(+x: U32, c: Bool, e: {F.is_nan(x) == c : Bool}) ->
  {F.canon(F.sel(c, 2143289344, x)) == F.sel(c, 2143289344, x) : U32}:
  match c:
    case True{}:
      {==}
    case False{}:
      %Equal.sym(Bool, F.is_nan(x), False{}, e) : {F.sel(_, 2143289344, x) == x : U32}
      {==}

def Laws.canon_idem(x):
  canon_idem.go(x, F.is_nan(x), {==})

# (ftz(x) & M) & S = ftz(x) & (M & S) = ftz(x) & 0 = 0
def Laws.fabs_sign(x):
  %u32_and_assoc_r(F.ftz(x), 2147483647, 2147483648) : {_ == 0 : U32}
  u32_and_zero(F.ftz(x))

# both sides of the select are ftz(x)
def Laws.fmax_self(x):
  sel_same(F.gt(x, x), F.ftz(x))

def Laws.fmin_self(x):
  sel_same(F.gt(x, x), F.ftz(x))

# ------------------------------------------------------------------------------- fp32, corners

def Laws.add_exact():
  {==}

def Laws.add_tie_down():
  {==}

def Laws.add_tie_up():
  {==}

def Laws.add_carry():
  {==}

def Laws.add_overflow():
  {==}

def Laws.add_inf_minus_inf():
  {==}

def Laws.add_nan():
  {==}

def Laws.add_cancel():
  {==}

def Laws.add_neg_zeros():
  {==}

def Laws.add_mixed_zeros():
  {==}

def Laws.add_ftz_in():
  {==}

def Laws.add_ftz_out():
  {==}

def Laws.add_sticky():
  {==}

def Laws.sub_self():
  {==}

def Laws.mul_exact():
  {==}

def Laws.mul_round():
  {==}

def Laws.mul_tie():
  {==}

def Laws.mul_sticky():
  {==}

def Laws.mul_overflow():
  {==}

def Laws.mul_zero_inf():
  {==}

def Laws.mul_neg_zero():
  {==}

def Laws.mul_signs():
  {==}

def Laws.mul_ftz_out():
  {==}

def Laws.mul_min_normal():
  {==}

def Laws.fmax_zeros():
  {==}

def Laws.fmin_zeros():
  {==}

def Laws.fmax_nan():
  {==}

def Laws.fmin_neg_nan():
  {==}

def Laws.fmax_denormals():
  {==}

def Laws.fabs_neg():
  {==}

def Laws.i2f_tie():
  {==}

def Laws.i2f_tie_up():
  {==}

def Laws.i2f_min():
  {==}

def Laws.i2f_minus_one():
  {==}

def Laws.f2i_trunc():
  {==}

def Laws.f2i_neg():
  {==}

def Laws.f2i_big():
  {==}

def Laws.f2i_nan():
  {==}

def Laws.floor_neg_half():
  {==}

def Laws.floor_pos():
  {==}

def Laws.exp2_zero():
  {==}

def Laws.exp2_three():
  {==}

def Laws.exp2_minus_one():
  {==}

def Laws.exp2_under():
  {==}

def Laws.exp2_over():
  {==}

def Laws.log2_eight():
  {==}

def Laws.log2_one():
  {==}

def Laws.log2_zero():
  {==}

def Laws.log2_neg():
  {==}

def Laws.recip_two():
  {==}

def Laws.recip_three():
  {==}

def Laws.rsqrt_four():
  {==}

def Laws.rsqrt_two():
  {==}

def Laws.q8_half_even():
  {==}

def Laws.q8_half_even_neg():
  {==}

def Laws.q8_sat():
  {==}

def Laws.q8_sat_tie():
  {==}

def Laws.q8_sat_neg():
  {==}

def Laws.q8_nan():
  {==}

# ------------------------------------------------------------------------------- the machine

# setr's test U32.is_eq(0, 0) computes to True, which keeps the array
def Laws.r0_stays_zero(c, v):
  {==}

# with cmp 0 the condition computes to (EQ OR False) OR False
def Laws.waitw_eq(v, ref, mask):
  %or_false_r(U32.is_eq((v .&. mask : U32), ref)) :
    {Bool.or(_, False{}) == U32.is_eq((v .&. mask : U32), ref) : Bool}
  or_false(U32.is_eq((v .&. mask : U32), ref))

# with cmp 1, (False OR NE) OR False
def Laws.waitw_ne(v, ref, mask):
  or_false(U32.is_ne((v .&. mask : U32), ref))

# with cmp 2, (False OR False) OR GE
def Laws.waitw_ge(v, ref, mask):
  {==}

def Laws.sx8_every_byte():
  {==}

def Laws.rng_last():
  {==}

def Laws.rng_wrap():
  {==}

def Laws.rng_sum_wrap():
  {==}

def Laws.rng_product_wrap():
  {==}

def Laws.rng_edge_in():
  {==}

def Laws.rng_edge_out():
  {==}

def Laws.mulhi_max():
  {==}

def Laws.mulhi_carry():
  {==}

def Laws.mulhi_mixed():
  {==}

# ------------------------------------------------------------------------------- programs

def Laws.prog_scalar():
  {==}

def Laws.prog_loop():
  {==}

def Laws.prog_r0():
  {==}

def Laws.prog_end():
  {==}

def Laws.prog_vop():
  {==}

def Laws.prog_ldst():
  {==}

def Laws.prog_ldst_dram():
  {==}
