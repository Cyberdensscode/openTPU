# The expert cache's sweeps (tools/offload/cache_bend.py): the replays of one router trace at
# several slot counts and policies (cache.bend), all at once, each on its own copy of the trace.
# cache.in (in the working directory) holds
#   words 0..7   t, l, k, n (the per-expert records), the replays (nr), where the requests
#                start (sq), the words in use (m), 0
#   16 + 8i      replay i: its policy, its slots, its warm list's length and start, where its
#                lfu counts start (n words), its experts a layer (lru_layer's e)
#   then         the requests (t l k words), the warm lists and the counts
# cache.out: t + 1 words a replay, the peak and the misses of each token.
import Base
import ../host.bend as H
import ./cache.bend as C

type Hd is Data:
  Hd{t: U32, l: U32, k: U32, n: U32, nr: U32, sq: U32, m: U32}

def rdl.s(r: Array<U32> & U32, xs: List<U32>) -> Array<U32> & List<U32>:
  (a, +v) = r
  (a, v <> xs)

# words i - n .. i - 1 as a list
def rdl(n: Nat, r: Array<U32> & List<U32>, +i: U32) -> Array<U32> & List<U32>:
  match n:
    case 0n:
      r
    case 1n+p:
      (a, xs) = r
      rdl(p, rdl.s(Array.get(U32, a, (i - 1 : U32)), xs), (i - 1 : U32))

def hd.of(xs: List<U32>) -> Hd:
  match xs:
    case Con{t, Con{l, Con{k, Con{n, Con{nr, Con{sq, Con{m, _}}}}}}}:
      Hd{t, l, k, n, nr, sq, m}
    case _:
      Hd{0, 0, 0, 0, 0, 0, 0}

def Hd.lay(+h: Hd, +pol: U32, +cap: U32, +nw: U32, +e: U32) -> C.Ly:
  match h:
    case Hd{+t, +l, +k, +n, nr, sq, m}:
      C.lay(pol, cap, t, l, k, n, e, nw)

def Hd.t(+h: Hd) -> U32:
  match h:
    case Hd{+t, l, k, n, nr, sq, m}:
      t

def Hd.nr(+h: Hd) -> U32:
  match h:
    case Hd{t, l, k, n, +nr, sq, m}:
      nr

def Hd.m(+h: Hd) -> U32:
  match h:
    case Hd{t, l, k, n, nr, sq, +m}:
      m

def Hd.run(src: Array<U32>, +h: Hd, +y: C.Ly, +so: U32, +sp: U32) -> Array<U32>:
  match h:
    case Hd{t, l, k, n, nr, +sq, m}:
      C.run(src, y, sq, so, sp)

def leaf.e(st: Array<U32> & Array<U32>, +pk: U32) -> Array<U32>:
  (w, res) = st
  res[0] <- pk

def leaf.d(r: Array<U32> & U32, +y: C.Ly) -> Array<U32>:
  (w, +pk) = r
  +t = C.Ly.t(y)
  res = Array.new(U32, U32.to_nat(C.depth.of((t + 1 : U32))), 0)
  leaf.e(C.cp(U32.to_nat(t), (w, res), C.Ly.out(y), 1, 1), pk)

def leaf.b(r: Array<U32> & List<U32>, +h: Hd) -> Array<U32>:
  (src, xs) = r
  match xs:
    case Con{+pol, Con{+cap, Con{+nw, Con{+so, Con{+sp, Con{+e, _}}}}}}:
      +y = Hd.lay(h, pol, cap, nw, e)
      leaf.d(Array.get(U32, Hd.run(src, h, y, so, sp), 3), y)
    case _:
      [0 : U32^0n]

# replay i: the peak and the misses of each token
def leaf(src: Array<U32>, +h: Hd, +i: U32) -> Array<U32>:
  leaf.b(rdl(6n, (src, Nil{}), (((i * 8 : U32) + 16 : U32) + 6 : U32)), h)

def par.leaf(z: Bool, a: Array<U32>, +h: Hd, +lo: U32) -> List<Array<U32>>:
  match z:
    case True{}:
      leaf(a, h, lo) <> Nil{}
    case False{}:
      Nil{}

def app(xs: List<Array<U32>>, ys: List<Array<U32>>) -> List<Array<U32>>:
  match xs:
    case Nil{}:
      ys
    case Con{x, rest}:
      x <> app(rest, ys)

# replays lo .. lo + cnt - 1 in halves at once (z: two or more), each half on a copy (r: two)
def par(+f: Nat, z: Bool, r: Array<U32> & Array<U32>, +h: Hd, +lo: U32, +cnt: U32) ->
  List<Array<U32>>:
  match f z:
    case 0n _:
      (a, b) = r
      par.leaf(U32.is_eq(cnt, 1), a, h, lo)
    case 1n+p False{}:
      (a, b) = r
      par.leaf(U32.is_eq(cnt, 1), a, h, lo)
    case 1n+p True{}:
      (a, b) = r
      +hf = (cnt >> 1n : U32)
      +rest = (cnt - hf : U32)
      x y = par(p, (hf >= 2 : U32), Array.clone(U32, a), h, lo, hf) par(p, (rest >= 2 : U32),
        Array.clone(U32, b), h, (lo + hf : U32), rest)
      app(x, y)

def put.d(st: Array<U32> & Array<U32>) -> Array<U32>:
  (res, out) = st
  out

# each replay's words into the output, t1 a replay
def put(xs: List<Array<U32>>, out: Array<U32>, +t1: U32, +j: U32) -> Array<U32>:
  match xs:
    case Nil{}:
      out
    case Con{res, rest}:
      put(rest, put.d(C.cp(U32.to_nat(t1), (res, out), 0, j, 1)), t1, (j + t1 : U32))

def shrink(st: Array<U32> & Array<U32>) -> Array<U32>:
  (big, small) = st
  small

def top.b(r: Array<U32> & List<U32>) -> Array<U32> & U32:
  (inp, xs) = r
  +h = hd.of(xs)
  +t1 = (Hd.t(h) + 1 : U32)
  +nr = Hd.nr(h)
  src = shrink(C.cp(U32.to_nat(Hd.m(h)), (inp, Array.new(U32,
    U32.to_nat(C.depth.of(Hd.m(h))), 0)), 0, 0, 1))
  out = Array.new(U32, U32.to_nat(C.depth.of((t1 * nr : U32))), 0)
  (put(par(32n, (nr >= 2 : U32), Array.clone(U32, src), h, 0, nr), out, t1, 0), (t1 * nr : U32))

# the sweep: its output and its words
def top(inp: Array<U32>) -> Array<U32> & U32:
  top.b(rdl(7n, (inp, Nil{}), 7))

def save(r: Array<U32> & U32) -> IO(Array<U32>):
  (out, +n) = r
  H.Host.save("cache.out", out, n)

def main() -> IO(Unit):
  do IO<Unit>:
    inp : Array<U32> <- H.Host.load("cache.in", 23)
    out : Array<U32> <- save(top(inp))
    IO.print("ok")
