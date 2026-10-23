# fp32.bend on a file of test vectors: words (op, a, b, expected) per vector, the count in
# word 0 of the header; writes one result word per vector (tests/test_bend.py).
#   fpvec IN OUT DEPTH
import Base
import ../fp32.bend as F
import ../host.bend as H

def fop.o17(c: Bool, +a: U32, +b: U32) -> U32:
  match c:
    case True{}:
      F.mx_offset(a, b)
    case False{}:
      F.ftz(a)

def fop.o15(c: Bool, +op: U32, +a: U32, +b: U32) -> U32:
  match c:
    case True{}:
      F.f2i(a)
    case False{}:
      fop.o17(U32.is_eq(op, 17), a, b)

def fop.o14(c: Bool, +op: U32, +a: U32, +b: U32) -> U32:
  match c:
    case True{}:
      F.fmin(a, b)
    case False{}:
      fop.o15(U32.is_eq(op, 15), op, a, b)

def fop.o13(c: Bool, +op: U32, +a: U32, +b: U32) -> U32:
  match c:
    case True{}:
      F.fmax(a, b)
    case False{}:
      fop.o14(U32.is_eq(op, 4) || U32.is_eq(op, 14), op, a, b)

def fop.o12(c: Bool, +op: U32, +a: U32, +b: U32) -> U32:
  match c:
    case True{}:
      F.log2(a)
    case False{}:
      fop.o13(U32.is_eq(op, 3) || U32.is_eq(op, 13), op, a, b)

def fop.o11(c: Bool, +op: U32, +a: U32, +b: U32) -> U32:
  match c:
    case True{}:
      F.fabs(a)
    case False{}:
      fop.o12(U32.is_eq(op, 12), op, a, b)

def fop.o10(c: Bool, +op: U32, +a: U32, +b: U32) -> U32:
  match c:
    case True{}:
      F.b2u(F.gt(a, b))
    case False{}:
      fop.o11(U32.is_eq(op, 11), op, a, b)

def fop.o9(c: Bool, +op: U32, +a: U32, +b: U32) -> U32:
  match c:
    case True{}:
      (F.q8(a) .&. 255 : U32)
    case False{}:
      fop.o10(U32.is_eq(op, 10), op, a, b)

def fop.o8(c: Bool, +op: U32, +a: U32, +b: U32) -> U32:
  match c:
    case True{}:
      F.i2f(a)
    case False{}:
      fop.o9(U32.is_eq(op, 9), op, a, b)

def fop.o7(c: Bool, +op: U32, +a: U32, +b: U32) -> U32:
  match c:
    case True{}:
      F.rsqrt(a)
    case False{}:
      fop.o8(U32.is_eq(op, 8), op, a, b)

def fop.o6(c: Bool, +op: U32, +a: U32, +b: U32) -> U32:
  match c:
    case True{}:
      F.recip(a)
    case False{}:
      fop.o7(U32.is_eq(op, 7), op, a, b)

def fop.o5(c: Bool, +op: U32, +a: U32, +b: U32) -> U32:
  match c:
    case True{}:
      F.exp2(a)
    case False{}:
      fop.o6(U32.is_eq(op, 6), op, a, b)

def fop.o2(c: Bool, +op: U32, +a: U32, +b: U32) -> U32:
  match c:
    case True{}:
      F.mul(a, b)
    case False{}:
      fop.o5(U32.is_eq(op, 5), op, a, b)

def fop.o1(c: Bool, +op: U32, +a: U32, +b: U32) -> U32:
  match c:
    case True{}:
      F.sub(a, b)
    case False{}:
      fop.o2(U32.is_eq(op, 2), op, a, b)

def fop.o0(c: Bool, +op: U32, +a: U32, +b: U32) -> U32:
  match c:
    case True{}:
      F.add(a, b)
    case False{}:
      fop.o1(U32.is_eq(op, 1), op, a, b)

def fop(+op: U32, +a: U32, +b: U32) -> U32:
  fop.o0(U32.is_eq(op, 0), op, a, b)

# vector i: words 4 + 4i .. 4 + 4i + 2 (op, a, b)
def step.b(r: Array<U32> & U32, out: Array<U32>, +op: U32, +a: U32, +i: U32) ->
  Array<U32> & Array<U32>:
  (inp, b) = r
  (inp, Array.set(U32, out, i, fop(op, a, b)))

def step.a(r: Array<U32> & U32, out: Array<U32>, +op: U32, +i: U32) -> Array<U32> & Array<U32>:
  (inp, a) = r
  step.b(Array.get(U32, inp, (4 * i + 6 : U32)), out, op, a, i)

def step.op(r: Array<U32> & U32, out: Array<U32>, +i: U32) -> Array<U32> & Array<U32>:
  (inp, op) = r
  step.a(Array.get(U32, inp, (4 * i + 5 : U32)), out, op, i)

def loop(n: Nat, st: Array<U32> & Array<U32>, +i: U32) -> Array<U32> & Array<U32>:
  match n:
    case 0n:
      st
    case 1n+p:
      (inp, out) = st
      loop(p, step.op(Array.get(U32, inp, (4 * i + 4 : U32)), out, i), (i + 1 : U32))

def fin(st: Array<U32> & Array<U32>, +n: U32) -> Array<U32> & U32:
  (inp, out) = st
  (out, n)

def run(r: Array<U32> & U32) -> Array<U32> & U32:
  (inp, +n) = r
  fin(loop(U32.to_nat(n), (inp, [0 : U32^22n]), 0), n)

def save(r: Array<U32> & U32) -> IO(Array<U32>):
  (out, +n) = r
  H.Host.save("out.bin", out, n)

def main() -> IO(Unit):
  do IO<Unit>:
    inp : Array<U32> <- H.Host.load("vec.bin", 22)
    out : Array<U32> <- save(run(Array.get(U32, inp, 0)))
    IO.print("ok")
