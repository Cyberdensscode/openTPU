# The expert cache in Bend: tools/offload/cachesim.py's replay (simulate) of a MoE model's router
# trace against a cache of expert slots, miss for miss, for the policies static, lru, lru_layer,
# lfu and opt (tests/test_bend.py compares the two). lfu_layer is not here: its decayed use is a
# chain of float64 log2 and exp2 that Bend's 32-bit floats do not reproduce.
#
# The requests: T tokens x L MoE layers x the top-k expert ids, each id global (layer j's expert
# i is j E + i, n = L E of them). The warm list: the experts in the cache at the start, in the
# profile's order (lru_layer takes each layer's first ones from all n).
#
# A replay is one word array (Ly below). Everything here is pure and terminating: main.bend runs
# a sweep's replays at once, LAWS.bend states facts about replays and PROOF.bend proves them.
import Base

# ------------------------------------------------------------------------------- words

def sel(c: Bool, +a: U32, +b: U32) -> U32:
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

def max(+a: U32, +b: U32) -> U32:
  sel((a >= b : U32), a, b)

# a < b for heap entries of three words, compared in order
def lt3(+a0: U32, +a1: U32, +a2: U32, +b0: U32, +b1: U32, +b2: U32) -> Bool:
  (a0 < b0 : U32) || (U32.is_eq(a0, b0) && ((a1 < b1 : U32) || (U32.is_eq(a1, b1)
    && (a2 < b2 : U32))))

# ------------------------------------------------------------------------------- the layout

# The policies: static 0, lru 1, lru_layer 2, lfu 3, opt 4. The replay's counters, words 0..6
# of its array: 0 lru's last stamp, 1 the heap's entries, 2 the experts in the cache, 3 the
# most experts in the cache after a request (the peak), 4 lfu's step (requests so far), 5 the
# request's misses, 6 opt's entries put back.
#
# Where the rest is: pol the policy, cap the cache's slots, t tokens, l MoE layers, k experts a
# request, n experts in all (e a layer), nw the warm list's length, nq = t l k;
#   x   five words for each expert v at x + 5v: [0] in the cache (lru: its last stamp, 0 when
#       out), [1] lfu: its count when pushed, opt: its next use, [2] lfu: the step it was pushed
#       at, opt: its last use + 1 while the next uses are found, [3] lfu: its count, [4] the
#       last request that named it (lfu: the step, opt: the request + 1)
#   g   four words a group of slots (lru: one group, lru_layer: a layer each): its queue's head
#       and tail, its size and its limit
#   q   the queues, pg words a group: (expert, stamp) pairs, oldest first; an entry is current
#       while its stamp is its expert's
#   h   the heap of (a, b, c) entries, the least on top (lfu: count, step, expert; opt: nq + 1
#       less the next use, expert, 0: the furthest next use on top)
#   rq  the requests, ro the warm list, nx opt's next use of each request entry
#   mb  the request's misses (k words) and opt's entries to put back (2k)
#   out the misses of each token
type Ly is Data:
  Ly{pol: U32, cap: U32, t: U32, l: U32, k: U32, n: U32, e: U32, nw: U32, nq: U32, x: U32,
     g: U32, q: U32, pg: U32, h: U32, rq: U32, ro: U32, nx: U32, mb: U32, out: U32}

def Ly.pol(+y: Ly) -> U32:
  match y:
    case Ly{+pol, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _}:
      pol

def Ly.cap(+y: Ly) -> U32:
  match y:
    case Ly{_, +cap, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _}:
      cap

def Ly.t(+y: Ly) -> U32:
  match y:
    case Ly{_, _, +t, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _}:
      t

def Ly.l(+y: Ly) -> U32:
  match y:
    case Ly{_, _, _, +l, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _}:
      l

def Ly.k(+y: Ly) -> U32:
  match y:
    case Ly{_, _, _, _, +k, _, _, _, _, _, _, _, _, _, _, _, _, _, _}:
      k

def Ly.n(+y: Ly) -> U32:
  match y:
    case Ly{_, _, _, _, _, +n, _, _, _, _, _, _, _, _, _, _, _, _, _}:
      n

def Ly.e(+y: Ly) -> U32:
  match y:
    case Ly{_, _, _, _, _, _, +e, _, _, _, _, _, _, _, _, _, _, _, _}:
      e

def Ly.nw(+y: Ly) -> U32:
  match y:
    case Ly{_, _, _, _, _, _, _, +nw, _, _, _, _, _, _, _, _, _, _, _}:
      nw

def Ly.nq(+y: Ly) -> U32:
  match y:
    case Ly{_, _, _, _, _, _, _, _, +nq, _, _, _, _, _, _, _, _, _, _}:
      nq

def Ly.x(+y: Ly) -> U32:
  match y:
    case Ly{_, _, _, _, _, _, _, _, _, +x, _, _, _, _, _, _, _, _, _}:
      x

def Ly.g(+y: Ly) -> U32:
  match y:
    case Ly{_, _, _, _, _, _, _, _, _, _, +g, _, _, _, _, _, _, _, _}:
      g

def Ly.q(+y: Ly) -> U32:
  match y:
    case Ly{_, _, _, _, _, _, _, _, _, _, _, +q, _, _, _, _, _, _, _}:
      q

def Ly.pg(+y: Ly) -> U32:
  match y:
    case Ly{_, _, _, _, _, _, _, _, _, _, _, _, +pg, _, _, _, _, _, _}:
      pg

def Ly.h(+y: Ly) -> U32:
  match y:
    case Ly{_, _, _, _, _, _, _, _, _, _, _, _, _, +h, _, _, _, _, _}:
      h

def Ly.rq(+y: Ly) -> U32:
  match y:
    case Ly{_, _, _, _, _, _, _, _, _, _, _, _, _, _, +rq, _, _, _, _}:
      rq

def Ly.ro(+y: Ly) -> U32:
  match y:
    case Ly{_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, +ro, _, _, _}:
      ro

def Ly.nx(+y: Ly) -> U32:
  match y:
    case Ly{_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, +nx, _, _}:
      nx

def Ly.mb(+y: Ly) -> U32:
  match y:
    case Ly{_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, +mb, _}:
      mb

def Ly.out(+y: Ly) -> U32:
  match y:
    case Ly{_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, +out}:
      out

# the layout of a replay
def lay(+pol: U32, +cap: U32, +t: U32, +l: U32, +k: U32, +n: U32, +e: U32, +nw: U32) -> Ly:
  +nq = ((t * l : U32) * k : U32)
  +layer = U32.is_eq(pol, 2)
  +ng = sel(layer, l, 1)
  +pq = sel(layer, ((((cap + l : U32) - 1 : U32) / l : U32) + (t * k : U32) : U32),
    (nw + nq : U32))
  +pg = sel(U32.is_eq(pol, 1) || layer, (pq * 2 : U32), 0)
  +x = {16 : U32}
  +g = (x + (n * 5 : U32) : U32)
  +q = (g + (ng * 4 : U32) : U32)
  +h = (q + (ng * pg : U32) : U32)
  +rq = (h + sel((pol >= 3 : U32), (((nw + nq : U32) + 1 : U32) * 3 : U32), 0) : U32)
  +ro = (rq + nq : U32)
  +nx = (ro + nw : U32)
  +mb = (nx + sel(U32.is_eq(pol, 4), nq, 0) : U32)
  +out = (mb + (k * 3 : U32) : U32)
  Ly{pol, cap, t, l, k, n, e, nw, nq, x, g, q, pg, h, rq, ro, nx, mb, out}

# the least d with 2^d >= m (z: 2^d >= m already)
def depth(f: Nat, z: Bool, +d: U32, +m: U32) -> U32:
  match f z:
    case 0n _:
      d
    case 1n+p True{}:
      d
    case 1n+p False{}:
      depth(p, (m <= (1 << U32.to_nat((d + 1 : U32)) : U32) : U32), (d + 1 : U32), m)

def depth.of(+m: U32) -> U32:
  depth(31n, (m <= 1 : U32), 0, m)

# expert v's record
def xv(+y: Ly, +v: U32) -> U32:
  (Ly.x(y) + (v * 5 : U32) : U32)

# group j's words
def gj(+y: Ly, +j: U32) -> U32:
  (Ly.g(y) + (j * 4 : U32) : U32)

# ------------------------------------------------------------------------------- arrays

def bump.go(r: Array<U32> & U32, +i: U32, +v: U32) -> Array<U32>:
  (w, +a) = r
  w[i] <- (a + v : U32)

# w[i] += v
def bump(w: Array<U32>, +i: U32, +v: U32) -> Array<U32>:
  bump.go(Array.get(U32, w, i), i, v)

# w[i] -= 1
def drop1(w: Array<U32>, +i: U32) -> Array<U32>:
  bump(w, i, 4294967295)

def cp.step(r: Array<U32> & U32, dst: Array<U32>, +j: U32) -> Array<U32> & Array<U32>:
  (src, v) = r
  (src, Array.set(U32, dst, j, v))

# dst[j + s m] = src[i + m] for m < n (two arrays)
def cp(n: Nat, st: Array<U32> & Array<U32>, +i: U32, +j: U32, +s: U32) ->
  Array<U32> & Array<U32>:
  match n:
    case 0n:
      st
    case 1n+p:
      (src, dst) = st
      cp(p, cp.step(Array.get(U32, src, i), dst, j), (i + 1 : U32), (j + s : U32), s)

def put3(w: Array<U32>, +i: U32, +a: U32, +b: U32, +c: U32) -> Array<U32>:
  w1 = Array.set(U32, w, i, a)
  w2 = Array.set(U32, w1, (i + 1 : U32), b)
  Array.set(U32, w2, (i + 2 : U32), c)

# ------------------------------------------------------------------------------- the heap

# a loop's array and whether it goes on (a sift: the hole at i)
type Hs is Type:
  Hs{w: Array<U32>, go: Bool, i: U32}

def up.d(z: Bool, w: Array<U32>, +hb: U32, +i: U32, +pi: U32, +p0: U32, +p1: U32, +p2: U32)
  -> Hs:
  match z:
    case True{}:
      Hs{put3(w, (hb + (i * 3 : U32) : U32), p0, p1, p2), True{}, pi}
    case False{}:
      Hs{w, False{}, i}

def up.c(r: Array<U32> & U32, +hb: U32, +i: U32, +pi: U32, +a: U32, +b: U32, +c: U32,
  +p0: U32, +p1: U32) -> Hs:
  (w, +p2) = r
  up.d(lt3(a, b, c, p0, p1, p2), w, hb, i, pi, p0, p1, p2)

def up.b(r: Array<U32> & U32, +hb: U32, +i: U32, +pi: U32, +a: U32, +b: U32, +c: U32,
  +p0: U32) -> Hs:
  (w, +p1) = r
  up.c(Array.get(U32, w, (hb + (pi * 3 : U32) + 2 : U32)), hb, i, pi, a, b, c, p0, p1)

def up.a(r: Array<U32> & U32, +hb: U32, +i: U32, +pi: U32, +a: U32, +b: U32, +c: U32) -> Hs:
  (w, +p0) = r
  up.b(Array.get(U32, w, (hb + (pi * 3 : U32) + 1 : U32)), hb, i, pi, a, b, c, p0)

# a sift up's step: the parent's entry comes down into the hole while (a, b, c) is less
def up.step(z: Bool, w: Array<U32>, +hb: U32, +i: U32, +a: U32, +b: U32, +c: U32) -> Hs:
  match z:
    case True{}:
      Hs{w, False{}, i}
    case False{}:
      +pi = ((i - 1 : U32) / 2 : U32)
      up.a(Array.get(U32, w, (hb + (pi * 3 : U32) : U32)), hb, i, pi, a, b, c)

def up(f: Nat, s: Hs, +hb: U32, +a: U32, +b: U32, +c: U32) -> Array<U32>:
  match f:
    case 0n:
      match s:
        case Hs{w, go, +i}:
          put3(w, (hb + (i * 3 : U32) : U32), a, b, c)
    case 1n+p:
      match s:
        case Hs{w, go, +i}:
          match go:
            case True{}:
              up(p, up.step(U32.is_eq(i, 0), w, hb, i, a, b, c), hb, a, b, c)
            case False{}:
              put3(w, (hb + (i * 3 : U32) : U32), a, b, c)

def push.go(r: Array<U32> & U32, +hb: U32, +a: U32, +b: U32, +c: U32) -> Array<U32>:
  (w, +hz) = r
  w1 = Array.set(U32, w, 1, (hz + 1 : U32))
  up(33n, Hs{w1, True{}, hz}, hb, a, b, c)

# (a, b, c) into the heap at hb
def push(w: Array<U32>, +hb: U32, +a: U32, +b: U32, +c: U32) -> Array<U32>:
  push.go(Array.get(U32, w, 1), hb, a, b, c)

def dn.e(z: Bool, w: Array<U32>, +hb: U32, +i: U32, +c: U32, +c0: U32, +c1: U32, +c2: U32)
  -> Hs:
  match z:
    case True{}:
      Hs{put3(w, (hb + (i * 3 : U32) : U32), c0, c1, c2), True{}, c}
    case False{}:
      Hs{w, False{}, i}

# the lesser child (z: the right one) up into the hole while it is less than x
def dn.d(z: Bool, w: Array<U32>, +hb: U32, +i: U32, +l: U32, +x0: U32, +x1: U32, +x2: U32,
  +l0: U32, +l1: U32, +l2: U32, +r0: U32, +r1: U32, +r2: U32) -> Hs:
  match z:
    case True{}:
      dn.e(lt3(r0, r1, r2, x0, x1, x2), w, hb, i, (l + 1 : U32), r0, r1, r2)
    case False{}:
      dn.e(lt3(l0, l1, l2, x0, x1, x2), w, hb, i, l, l0, l1, l2)

def dn.r2(r: Array<U32> & U32, +hb: U32, +sz: U32, +i: U32, +l: U32, +x0: U32, +x1: U32,
  +x2: U32, +l0: U32, +l1: U32, +l2: U32, +r0: U32, +r1: U32) -> Hs:
  (w, +r2) = r
  dn.d(((l + 1 : U32) < sz : U32) && lt3(r0, r1, r2, l0, l1, l2), w, hb, i, l, x0, x1, x2,
    l0, l1, l2, r0, r1, r2)

def dn.r1(r: Array<U32> & U32, +hb: U32, +sz: U32, +i: U32, +l: U32, +x0: U32, +x1: U32,
  +x2: U32, +l0: U32, +l1: U32, +l2: U32, +r0: U32) -> Hs:
  (w, +r1) = r
  dn.r2(Array.get(U32, w, (hb + (l * 3 : U32) + 5 : U32)), hb, sz, i, l, x0, x1, x2, l0, l1,
    l2, r0, r1)

def dn.r0(r: Array<U32> & U32, +hb: U32, +sz: U32, +i: U32, +l: U32, +x0: U32, +x1: U32,
  +x2: U32, +l0: U32, +l1: U32, +l2: U32) -> Hs:
  (w, +r0) = r
  dn.r1(Array.get(U32, w, (hb + (l * 3 : U32) + 4 : U32)), hb, sz, i, l, x0, x1, x2, l0, l1,
    l2, r0)

def dn.l2(r: Array<U32> & U32, +hb: U32, +sz: U32, +i: U32, +l: U32, +x0: U32, +x1: U32,
  +x2: U32, +l0: U32, +l1: U32) -> Hs:
  (w, +l2) = r
  dn.r0(Array.get(U32, w, (hb + (l * 3 : U32) + 3 : U32)), hb, sz, i, l, x0, x1, x2, l0, l1,
    l2)

def dn.l1(r: Array<U32> & U32, +hb: U32, +sz: U32, +i: U32, +l: U32, +x0: U32, +x1: U32,
  +x2: U32, +l0: U32) -> Hs:
  (w, +l1) = r
  dn.l2(Array.get(U32, w, (hb + (l * 3 : U32) + 2 : U32)), hb, sz, i, l, x0, x1, x2, l0, l1)

def dn.l0(r: Array<U32> & U32, +hb: U32, +sz: U32, +i: U32, +l: U32, +x0: U32, +x1: U32,
  +x2: U32) -> Hs:
  (w, +l0) = r
  dn.l1(Array.get(U32, w, (hb + (l * 3 : U32) + 1 : U32)), hb, sz, i, l, x0, x1, x2, l0)

# a sift down's step (z: the hole has a child)
def dn.step(z: Bool, w: Array<U32>, +hb: U32, +sz: U32, +i: U32, +x0: U32, +x1: U32,
  +x2: U32) -> Hs:
  match z:
    case True{}:
      +l = ((i * 2 : U32) + 1 : U32)
      dn.l0(Array.get(U32, w, (hb + (l * 3 : U32) : U32)), hb, sz, i, l, x0, x1, x2)
    case False{}:
      Hs{w, False{}, i}

def dn(f: Nat, s: Hs, +hb: U32, +sz: U32, +x0: U32, +x1: U32, +x2: U32) -> Array<U32>:
  match f:
    case 0n:
      match s:
        case Hs{w, go, +i}:
          put3(w, (hb + (i * 3 : U32) : U32), x0, x1, x2)
    case 1n+p:
      match s:
        case Hs{w, go, +i}:
          match go:
            case True{}:
              dn(p, dn.step((((i * 2 : U32) + 1 : U32) < sz : U32), w, hb, sz, i, x0, x1, x2),
                hb, sz, x0, x1, x2)
            case False{}:
              put3(w, (hb + (i * 3 : U32) : U32), x0, x1, x2)

# the heap's least entry, taken off (ok: there was one)
type Top is Type:
  Top{w: Array<U32>, ok: Bool, a: U32, b: U32, c: U32}

def pop.x2(r: Array<U32> & U32, +hb: U32, +sz: U32, +a: U32, +b: U32, +c: U32, +x0: U32,
  +x1: U32) -> Top:
  (w, +x2) = r
  Top{dn(33n, Hs{w, True{}, 0}, hb, sz, x0, x1, x2), True{}, a, b, c}

def pop.x1(r: Array<U32> & U32, +hb: U32, +sz: U32, +a: U32, +b: U32, +c: U32, +x0: U32)
  -> Top:
  (w, +x1) = r
  pop.x2(Array.get(U32, w, (hb + (sz * 3 : U32) + 2 : U32)), hb, sz, a, b, c, x0, x1)

def pop.x0(r: Array<U32> & U32, +hb: U32, +sz: U32, +a: U32, +b: U32, +c: U32) -> Top:
  (w, +x0) = r
  pop.x1(Array.get(U32, w, (hb + (sz * 3 : U32) + 1 : U32)), hb, sz, a, b, c, x0)

def pop.c(r: Array<U32> & U32, +hb: U32, +sz: U32, +a: U32, +b: U32) -> Top:
  (w, +c) = r
  pop.x0(Array.get(U32, w, (hb + (sz * 3 : U32) : U32)), hb, sz, a, b, c)

def pop.b(r: Array<U32> & U32, +hb: U32, +sz: U32, +a: U32) -> Top:
  (w, +b) = r
  pop.c(Array.get(U32, w, (hb + 2 : U32)), hb, sz, a, b)

def pop.a(r: Array<U32> & U32, +hb: U32, +sz: U32) -> Top:
  (w, +a) = r
  pop.b(Array.get(U32, w, (hb + 1 : U32)), hb, sz, a)

def pop.z(z: Bool, w: Array<U32>, +hb: U32, +hz: U32) -> Top:
  match z:
    case True{}:
      Top{w, False{}, 0, 0, 0}
    case False{}:
      w1 = Array.set(U32, w, 1, (hz - 1 : U32))
      pop.a(Array.get(U32, w1, hb), hb, (hz - 1 : U32))

def pop.go(r: Array<U32> & U32, +hb: U32) -> Top:
  (w, +hz) = r
  pop.z(U32.is_eq(hz, 0), w, hb, hz)

# the least entry of the heap at hb, the last one sifted down from the top in its place
def pop(w: Array<U32>, +hb: U32) -> Top:
  pop.go(Array.get(U32, w, 1), hb)

# ------------------------------------------------------------------------------- a request

def miss.add(r: Array<U32> & U32, +y: Ly, +v: U32) -> Array<U32>:
  (w, +mc) = r
  w1 = Array.set(U32, w, (Ly.mb(y) + mc : U32), v)
  Array.set(U32, w1, 5, (mc + 1 : U32))

def fin.c(r: Array<U32> & U32, +occ: U32) -> Array<U32>:
  (w, +pk) = r
  w[3] <- max(pk, occ)

def fin.b(r: Array<U32> & U32) -> Array<U32>:
  (w, +occ) = r
  fin.c(Array.get(U32, w, 3), occ)

def fin.a(r: Array<U32> & U32, +y: Ly, +t: U32) -> Array<U32>:
  (w, +mc) = r
  w1 = bump(w, (Ly.out(y) + t : U32), mc)
  w2 = Array.set(U32, w1, 5, 0)
  fin.b(Array.get(U32, w2, 2))

# the end of a request of token t: its misses to the token's, the peak
def fin(w: Array<U32>, +y: Ly, +t: U32) -> Array<U32>:
  fin.a(Array.get(U32, w, 5), y, t)

# the requests r .. r + f - 1 (one: a request, by policy)
def reqs(~one: Array<U32> -> Ly -> U32 -> Array<U32>, f: Nat, w: Array<U32>, +y: Ly, +r: U32)
  -> Array<U32>:
  match f:
    case 0n:
      w
    case 1n+p:
      reqs(~one, p, one(w, y, r), y, (r + 1 : U32))

# ------------------------------------------------------------------------------- static

def st.c(r: Array<U32> & U32) -> Array<U32>:
  (w, +a) = r
  bump(w, 5, b2u(U32.is_eq(a, 0)))

def st.b(r: Array<U32> & U32, +y: Ly) -> Array<U32>:
  (w, +v) = r
  st.c(Array.get(U32, w, xv(y, v)))

def st1(f: Nat, w: Array<U32>, +y: Ly, +i: U32) -> Array<U32>:
  match f:
    case 0n:
      w
    case 1n+p:
      st1(p, st.b(Array.get(U32, w, i), y), y, (i + 1 : U32))

# static: a miss for each expert not in the warm set, which never changes
def st.one(w: Array<U32>, +y: Ly, +r: U32) -> Array<U32>:
  fin(st1(U32.to_nat(Ly.k(y)), w, y, (Ly.rq(y) + (r * Ly.k(y) : U32) : U32)), y,
    (r / Ly.l(y) : U32))

def warm.st.a(r: Array<U32> & U32, +y: Ly) -> Array<U32>:
  (w, +v) = r
  bump(Array.set(U32, w, xv(y, v), 1), 2, 1)

def warm.st(f: Nat, w: Array<U32>, +y: Ly, +i: U32) -> Array<U32>:
  match f:
    case 0n:
      w
    case 1n+p:
      warm.st(p, warm.st.a(Array.get(U32, w, (Ly.ro(y) + i : U32)), y), y, (i + 1 : U32))

# ------------------------------------------------------------------------------- lru

def qpush.go(r: Array<U32> & U32, +y: Ly, +j: U32, +v: U32, +s: U32) -> Array<U32>:
  (w, +tl) = r
  +at = ((Ly.q(y) + (j * Ly.pg(y) : U32) : U32) + (tl * 2 : U32) : U32)
  w1 = Array.set(U32, w, (gj(y, j) + 1 : U32), (tl + 1 : U32))
  w2 = Array.set(U32, w1, at, v)
  Array.set(U32, w2, (at + 1 : U32), s)

# (v, s) onto group j's queue
def qpush(w: Array<U32>, +y: Ly, +j: U32, +v: U32, +s: U32) -> Array<U32>:
  qpush.go(Array.get(U32, w, (gj(y, j) + 1 : U32)), y, j, v, s)

def touch.go(r: Array<U32> & U32, +y: Ly, +j: U32, +v: U32) -> Array<U32>:
  (w, +ck) = r
  +s = (ck + 1 : U32)
  w1 = Array.set(U32, w, 0, s)
  qpush(Array.set(U32, w1, xv(y, v), s), y, j, v, s)

# expert v is used: a new stamp, onto its group's queue
def touch(w: Array<U32>, +y: Ly, +j: U32, +v: U32) -> Array<U32>:
  touch.go(Array.get(U32, w, 0), y, j, v)

# expert v into the cache, in group j
def ins(w: Array<U32>, +y: Ly, +j: U32, +v: U32) -> Array<U32>:
  bump(bump(touch(w, y, j, v), (gj(y, j) + 2 : U32), 1), 2, 1)

def ev.d(z: Bool, w: Array<U32>, +y: Ly, +j: U32, +v: U32) -> Hs:
  match z:
    case True{}:
      w1 = Array.set(U32, w, xv(y, v), 0)
      Hs{drop1(drop1(w1, (gj(y, j) + 2 : U32)), 2), False{}, 0}
    case False{}:
      Hs{w, True{}, 0}

def ev.c(r: Array<U32> & U32, +y: Ly, +j: U32, +v: U32, +s: U32) -> Hs:
  (w, +a) = r
  ev.d(U32.is_eq(a, s) && U32.is_ne(s, 0), w, y, j, v)

def ev.b(r: Array<U32> & U32, +y: Ly, +j: U32, +v: U32) -> Hs:
  (w, +s) = r
  ev.c(Array.get(U32, w, xv(y, v)), y, j, v, s)

def ev.a(r: Array<U32> & U32, +y: Ly, +j: U32, +at: U32) -> Hs:
  (w, +v) = r
  ev.b(Array.get(U32, w, (at + 1 : U32)), y, j, v)

def ev.step(r: Array<U32> & U32, +y: Ly, +j: U32) -> Hs:
  (w, +hd) = r
  +at = ((Ly.q(y) + (j * Ly.pg(y) : U32) : U32) + (hd * 2 : U32) : U32)
  w1 = Array.set(U32, w, gj(y, j), (hd + 1 : U32))
  ev.a(Array.get(U32, w1, at), y, j, at)

# group j's oldest current entry out of the cache (the entries before it are stale)
def ev(f: Nat, s: Hs, +y: Ly, +j: U32) -> Array<U32>:
  match f:
    case 0n:
      match s:
        case Hs{w, go, i}:
          w
    case 1n+p:
      match s:
        case Hs{w, go, i}:
          match go:
            case True{}:
              ev(p, ev.step(Array.get(U32, w, gj(y, j)), y, j), y, j)
            case False{}:
              w

def l1.d(z: Bool, w: Array<U32>, +y: Ly, +j: U32, +v: U32) -> Array<U32>:
  match z:
    case True{}:
      miss.add(Array.get(U32, w, 5), y, v)
    case False{}:
      touch(w, y, j, v)

def l1.c(r: Array<U32> & U32, +y: Ly, +j: U32, +v: U32) -> Array<U32>:
  (w, +a) = r
  l1.d(U32.is_eq(a, 0), w, y, j, v)

def l1.b(r: Array<U32> & U32, +y: Ly, +j: U32) -> Array<U32>:
  (w, +v) = r
  l1.c(Array.get(U32, w, xv(y, v)), y, j, v)

# a request's experts in order: the cached ones used, the others misses
def l1(f: Nat, w: Array<U32>, +y: Ly, +j: U32, +i: U32) -> Array<U32>:
  match f:
    case 0n:
      w
    case 1n+p:
      l1(p, l1.b(Array.get(U32, w, i), y, j), y, j, (i + 1 : U32))

def l2.d(z: Bool, w: Array<U32>, +y: Ly, +j: U32, +v: U32) -> Array<U32>:
  match z:
    case True{}:
      ins(ev(U32.to_nat(Ly.pg(y)), Hs{w, True{}, 0}, y, j), y, j, v)
    case False{}:
      ins(w, y, j, v)

def l2.c(r: Array<U32> & U32, +y: Ly, +j: U32, +v: U32, +sz: U32) -> Array<U32>:
  (w, +lim) = r
  l2.d((sz >= lim : U32), w, y, j, v)

def l2.b(r: Array<U32> & U32, +y: Ly, +j: U32, +v: U32) -> Array<U32>:
  (w, +sz) = r
  l2.c(Array.get(U32, w, (gj(y, j) + 3 : U32)), y, j, v, sz)

def l2.a(r: Array<U32> & U32, +y: Ly, +j: U32) -> Array<U32>:
  (w, +v) = r
  l2.b(Array.get(U32, w, (gj(y, j) + 2 : U32)), y, j, v)

# the misses in order: each in, the group's oldest out first when the group is full
def l2(f: Nat, w: Array<U32>, +y: Ly, +j: U32, +m: U32) -> Array<U32>:
  match f:
    case 0n:
      w
    case 1n+p:
      l2(p, l2.a(Array.get(U32, w, (Ly.mb(y) + m : U32)), y, j), y, j, (m + 1 : U32))

def l2.go(r: Array<U32> & U32, +y: Ly, +j: U32) -> Array<U32>:
  (w, +mc) = r
  l2(U32.to_nat(mc), w, y, j, 0)

# lru (one group of cap slots) and lru_layer (a group a layer, each of max(its share, k))
def lru.one(w: Array<U32>, +y: Ly, +r: U32) -> Array<U32>:
  +j = sel(U32.is_eq(Ly.pol(y), 2), (r % Ly.l(y) : U32), 0)
  w1 = l1(U32.to_nat(Ly.k(y)), w, y, j, (Ly.rq(y) + (r * Ly.k(y) : U32) : U32))
  fin(l2.go(Array.get(U32, w1, 5), y, j), y, (r / Ly.l(y) : U32))

def warm.lru.a(r: Array<U32> & U32, +y: Ly) -> Array<U32>:
  (w, +v) = r
  ins(w, y, 0, v)

def warm.lru(f: Nat, w: Array<U32>, +y: Ly, +i: U32) -> Array<U32>:
  match f:
    case 0n:
      w
    case 1n+p:
      warm.lru(p, warm.lru.a(Array.get(U32, w, (Ly.ro(y) + i : U32)), y), y, (i + 1 : U32))

# layer j's share of the slots
def share(+y: Ly, +j: U32) -> U32:
  ((Ly.cap(y) / Ly.l(y) : U32) + b2u((j < (Ly.cap(y) % Ly.l(y) : U32) : U32)) : U32)

def lims(f: Nat, w: Array<U32>, +y: Ly, +j: U32) -> Array<U32>:
  match f:
    case 0n:
      w
    case 1n+p:
      lims(p, Array.set(U32, w, (gj(y, j) + 3 : U32), max(share(y, j), Ly.k(y))), y,
        (j + 1 : U32))

def warm.lay.c(z: Bool, w: Array<U32>, +y: Ly, +j: U32, +v: U32) -> Array<U32>:
  match z:
    case True{}:
      ins(w, y, j, v)
    case False{}:
      w

def warm.lay.b(r: Array<U32> & U32, +y: Ly, +j: U32, +v: U32) -> Array<U32>:
  (w, +sz) = r
  warm.lay.c((sz < share(y, j) : U32), w, y, j, v)

def warm.lay.a(r: Array<U32> & U32, +y: Ly) -> Array<U32>:
  (w, +v) = r
  +j = (v / Ly.e(y) : U32)
  warm.lay.b(Array.get(U32, w, (gj(y, j) + 2 : U32)), y, j, v)

# lru_layer's start: each layer's first experts of the warm list, up to its share
def warm.lay(f: Nat, w: Array<U32>, +y: Ly, +i: U32) -> Array<U32>:
  match f:
    case 0n:
      w
    case 1n+p:
      warm.lay(p, warm.lay.a(Array.get(U32, w, (Ly.ro(y) + i : U32)), y), y, (i + 1 : U32))

# ------------------------------------------------------------------------------- lfu

def fa.c(r: Array<U32> & U32, +y: Ly, +v: U32) -> Array<U32>:
  (w, +a) = r
  bump(bump(w, 5, b2u(U32.is_eq(a, 0))), (xv(y, v) + 3 : U32), 1)

def fa.b(r: Array<U32> & U32, +y: Ly, +s: U32) -> Array<U32>:
  (w, +v) = r
  w1 = Array.set(U32, w, (xv(y, v) + 4 : U32), s)
  fa.c(Array.get(U32, w1, xv(y, v)), y, v)

# the request's experts: named at step s, counted, the uncached ones misses
def fa(f: Nat, w: Array<U32>, +y: Ly, +s: U32, +i: U32) -> Array<U32>:
  match f:
    case 0n:
      w
    case 1n+p:
      fa(p, fa.b(Array.get(U32, w, i), y, s), y, s, (i + 1 : U32))

def fe.e(z: Bool, w: Array<U32>, +y: Ly, +v: U32) -> Hs:
  match z:
    case True{}:
      Hs{drop1(Array.set(U32, w, xv(y, v), 0), 2), True{}, 0}
    case False{}:
      Hs{w, True{}, 0}

def fe.d(r: Array<U32> & U32, +y: Ly, +s: U32, +f: U32, +st: U32, +v: U32, +a: U32, +b: U32,
  +c: U32) -> Hs:
  (w, +m) = r
  fe.e(U32.is_eq(a, 1) && U32.is_eq(b, f) && U32.is_eq(c, st) && U32.is_ne(m, s), w, y, v)

def fe.c(r: Array<U32> & U32, +y: Ly, +s: U32, +f: U32, +st: U32, +v: U32, +a: U32, +b: U32)
  -> Hs:
  (w, +c) = r
  fe.d(Array.get(U32, w, (xv(y, v) + 4 : U32)), y, s, f, st, v, a, b, c)

def fe.b(r: Array<U32> & U32, +y: Ly, +s: U32, +f: U32, +st: U32, +v: U32, +a: U32) -> Hs:
  (w, +b) = r
  fe.c(Array.get(U32, w, (xv(y, v) + 2 : U32)), y, s, f, st, v, a, b)

def fe.a(r: Array<U32> & U32, +y: Ly, +s: U32, +f: U32, +st: U32, +v: U32) -> Hs:
  (w, +a) = r
  fe.b(Array.get(U32, w, (xv(y, v) + 1 : U32)), y, s, f, st, v, a)

# the least entry off the heap: its expert out if the entry is current and the request does
# not name it
def fe.top(t: Top, +y: Ly, +s: U32) -> Hs:
  match t:
    case Top{w, ok, +f, +st, +v}:
      match ok:
        case True{}:
          fe.a(Array.get(U32, w, xv(y, v)), y, s, f, st, v)
        case False{}:
          Hs{w, False{}, 0}

def fe.z(z: Bool, w: Array<U32>, +y: Ly, +s: U32) -> Hs:
  match z:
    case True{}:
      fe.top(pop(w, Ly.h(y)), y, s)
    case False{}:
      Hs{w, False{}, 0}

def fe.m(r: Array<U32> & U32, +y: Ly, +s: U32, +occ: U32) -> Hs:
  (w, +mc) = r
  fe.z(((occ + mc : U32) > Ly.cap(y) : U32), w, y, s)

def fe.step(r: Array<U32> & U32, +y: Ly, +s: U32) -> Hs:
  (w, +occ) = r
  fe.m(Array.get(U32, w, 5), y, s, occ)

# out until the cache and the request's misses fit the slots
def fe(f: Nat, h: Hs, +y: Ly, +s: U32) -> Array<U32>:
  match f:
    case 0n:
      match h:
        case Hs{w, go, i}:
          w
    case 1n+p:
      match h:
        case Hs{w, go, i}:
          match go:
            case True{}:
              fe(p, fe.step(Array.get(U32, w, 2), y, s), y, s)
            case False{}:
              w

def fb.c(r: Array<U32> & U32, +y: Ly, +s: U32, +v: U32) -> Array<U32>:
  (w, +cn) = r
  push(put3(w, xv(y, v), 1, cn, s), Ly.h(y), cn, s, v)

def fb.b(r: Array<U32> & U32, +y: Ly, +s: U32, +v: U32) -> Array<U32>:
  (w, +a) = r
  w1 = bump(w, 2, b2u(U32.is_eq(a, 0)))
  fb.c(Array.get(U32, w1, (xv(y, v) + 3 : U32)), y, s, v)

def fb.a(r: Array<U32> & U32, +y: Ly, +s: U32) -> Array<U32>:
  (w, +v) = r
  fb.b(Array.get(U32, w, xv(y, v)), y, s, v)

# the request's experts in the cache, pushed with their counts at step s
def fb(f: Nat, w: Array<U32>, +y: Ly, +s: U32, +i: U32) -> Array<U32>:
  match f:
    case 0n:
      w
    case 1n+p:
      fb(p, fb.a(Array.get(U32, w, i), y, s), y, s, (i + 1 : U32))

def lfu.go(r: Array<U32> & U32, +y: Ly, +rr: U32) -> Array<U32>:
  (w, +s0) = r
  +s = (s0 + 1 : U32)
  +i = (Ly.rq(y) + (rr * Ly.k(y) : U32) : U32)
  +kn = U32.to_nat(Ly.k(y))
  w1 = fa(kn, Array.set(U32, w, 4, s), y, s, i)
  w2 = fe(U32.to_nat(((Ly.nw(y) + Ly.nq(y) : U32) + 1 : U32)), Hs{w1, True{}, 0}, y, s)
  fin(fb(kn, w2, y, s, i), y, (rr / Ly.l(y) : U32))

# lfu: the least count out (ties: the earliest pushed, then the least expert), never one the
# request names, until the cache and the misses fit cap slots
def lfu.one(w: Array<U32>, +y: Ly, +r: U32) -> Array<U32>:
  lfu.go(Array.get(U32, w, 4), y, r)

def warm.lfu.b(r: Array<U32> & U32, +y: Ly, +v: U32) -> Array<U32>:
  (w, +cn) = r
  push(bump(put3(w, xv(y, v), 1, cn, 0), 2, 1), Ly.h(y), cn, 0, v)

def warm.lfu.a(r: Array<U32> & U32, +y: Ly) -> Array<U32>:
  (w, +v) = r
  warm.lfu.b(Array.get(U32, w, (xv(y, v) + 3 : U32)), y, v)

def warm.lfu(f: Nat, w: Array<U32>, +y: Ly, +i: U32) -> Array<U32>:
  match f:
    case 0n:
      w
    case 1n+p:
      warm.lfu(p, warm.lfu.a(Array.get(U32, w, (Ly.ro(y) + i : U32)), y), y, (i + 1 : U32))

# ------------------------------------------------------------------------------- opt

def nxt.b(r: Array<U32> & U32, +y: Ly, +i: U32, +v: U32) -> Array<U32>:
  (w, +la) = r
  w1 = Array.set(U32, w, (Ly.nx(y) + i : U32), sel(U32.is_eq(la, 0), (Ly.nq(y) + 1 : U32),
    (la - 1 : U32)))
  Array.set(U32, w1, (xv(y, v) + 2 : U32), (i + 1 : U32))

def nxt.a(r: Array<U32> & U32, +y: Ly, +i: U32) -> Array<U32>:
  (w, +v) = r
  nxt.b(Array.get(U32, w, (xv(y, v) + 2 : U32)), y, i, v)

# each request entry's next use (nq + 1: none), from the last back (i: the one after)
def nxt(f: Nat, w: Array<U32>, +y: Ly, +i: U32) -> Array<U32>:
  match f:
    case 0n:
      w
    case 1n+p:
      nxt(p, nxt.a(Array.get(U32, w, ((Ly.rq(y) + i : U32) - 1 : U32)), y, (i - 1 : U32)), y,
        (i - 1 : U32))

def warm.opt.b(r: Array<U32> & U32, +y: Ly, +v: U32) -> Array<U32>:
  (w, +la) = r
  +fu = sel(U32.is_eq(la, 0), (Ly.nq(y) + 1 : U32), (la - 1 : U32))
  w1 = Array.set(U32, Array.set(U32, w, xv(y, v), 1), (xv(y, v) + 1 : U32), fu)
  push(bump(w1, 2, 1), Ly.h(y), ((Ly.nq(y) + 1 : U32) - fu : U32), v, 0)

def warm.opt.a(r: Array<U32> & U32, +y: Ly) -> Array<U32>:
  (w, +v) = r
  warm.opt.b(Array.get(U32, w, (xv(y, v) + 2 : U32)), y, v)

# opt's start: the warm experts, each with its first use
def warm.opt(f: Nat, w: Array<U32>, +y: Ly, +i: U32) -> Array<U32>:
  match f:
    case 0n:
      w
    case 1n+p:
      warm.opt(p, warm.opt.a(Array.get(U32, w, (Ly.ro(y) + i : U32)), y), y, (i + 1 : U32))

def o1.c(r: Array<U32> & U32) -> Array<U32>:
  (w, +a) = r
  bump(w, 5, b2u(U32.is_eq(a, 0)))

def o1.b(r: Array<U32> & U32, +y: Ly, +mk: U32) -> Array<U32>:
  (w, +v) = r
  o1.c(Array.get(U32, Array.set(U32, w, (xv(y, v) + 4 : U32), mk), xv(y, v)))

# the request's experts named (mk), the uncached ones misses
def o1(f: Nat, w: Array<U32>, +y: Ly, +mk: U32, +i: U32) -> Array<U32>:
  match f:
    case 0n:
      w
    case 1n+p:
      o1(p, o1.b(Array.get(U32, w, i), y, mk), y, mk, (i + 1 : U32))

def o2.c(r: Array<U32> & U32, +y: Ly, +v: U32, +u: U32) -> Array<U32>:
  (w, +a) = r
  w1 = bump(w, 2, b2u(U32.is_eq(a, 0)))
  w2 = Array.set(U32, Array.set(U32, w1, xv(y, v), 1), (xv(y, v) + 1 : U32), u)
  push(w2, Ly.h(y), ((Ly.nq(y) + 1 : U32) - u : U32), v, 0)

def o2.b(r: Array<U32> & U32, +y: Ly, +v: U32) -> Array<U32>:
  (w, +u) = r
  o2.c(Array.get(U32, w, xv(y, v)), y, v, u)

def o2.a(r: Array<U32> & U32, +y: Ly, +i: U32) -> Array<U32>:
  (w, +v) = r
  o2.b(Array.get(U32, w, (Ly.nx(y) + i : U32)), y, v)

# the request's experts in the cache with their next uses
def o2(f: Nat, w: Array<U32>, +y: Ly, +i: U32) -> Array<U32>:
  match f:
    case 0n:
      w
    case 1n+p:
      o2(p, o2.a(Array.get(U32, w, (Ly.rq(y) + i : U32)), y, i), y, (i + 1 : U32))

def oe.k(r: Array<U32> & U32, +y: Ly, +a: U32, +v: U32) -> Array<U32>:
  (w, +kc) = r
  +at = ((Ly.mb(y) + Ly.k(y) : U32) + (kc * 2 : U32) : U32)
  Array.set(U32, Array.set(U32, Array.set(U32, w, at, a), (at + 1 : U32), v), 6, (kc + 1 : U32))

def oe.e(z: Bool, cur: Bool, w: Array<U32>, +y: Ly, +a: U32, +v: U32) -> Hs:
  match z cur:
    case True{} True{}:
      Hs{oe.k(Array.get(U32, w, 6), y, a, v), True{}, 0}
    case True{} False{}:
      Hs{drop1(Array.set(U32, w, xv(y, v), 0), 2), True{}, 0}
    case False{} _:
      Hs{w, True{}, 0}

def oe.d(r: Array<U32> & U32, +y: Ly, +mk: U32, +a: U32, +v: U32, +av: U32, +bv: U32) -> Hs:
  (w, +m) = r
  oe.e(U32.is_eq(av, 1) && U32.is_eq(bv, ((Ly.nq(y) + 1 : U32) - a : U32)), U32.is_eq(m, mk),
    w, y, a, v)

def oe.c(r: Array<U32> & U32, +y: Ly, +mk: U32, +a: U32, +v: U32, +av: U32) -> Hs:
  (w, +bv) = r
  oe.d(Array.get(U32, w, (xv(y, v) + 4 : U32)), y, mk, a, v, av, bv)

def oe.b(r: Array<U32> & U32, +y: Ly, +mk: U32, +a: U32, +v: U32) -> Hs:
  (w, +av) = r
  oe.c(Array.get(U32, w, (xv(y, v) + 1 : U32)), y, mk, a, v, av)

# the furthest next use off the heap: kept aside if the request names it, else out
def oe.top(t: Top, +y: Ly, +mk: U32) -> Hs:
  match t:
    case Top{w, ok, +a, +v, c}:
      match ok:
        case True{}:
          oe.b(Array.get(U32, w, xv(y, v)), y, mk, a, v)
        case False{}:
          Hs{w, False{}, 0}

def oe.z(z: Bool, w: Array<U32>, +y: Ly, +mk: U32) -> Hs:
  match z:
    case True{}:
      oe.top(pop(w, Ly.h(y)), y, mk)
    case False{}:
      Hs{w, False{}, 0}

def oe.step(r: Array<U32> & U32, +y: Ly, +mk: U32) -> Hs:
  (w, +occ) = r
  oe.z((occ > Ly.cap(y) : U32), w, y, mk)

# out until the cache fits cap slots
def oe(f: Nat, h: Hs, +y: Ly, +mk: U32) -> Array<U32>:
  match f:
    case 0n:
      match h:
        case Hs{w, go, i}:
          w
    case 1n+p:
      match h:
        case Hs{w, go, i}:
          match go:
            case True{}:
              oe(p, oe.step(Array.get(U32, w, 2), y, mk), y, mk)
            case False{}:
              w

def ok.b(r: Array<U32> & U32, +y: Ly, +a: U32) -> Array<U32>:
  (w, +v) = r
  push(w, Ly.h(y), a, v, 0)

def ok.a(r: Array<U32> & U32, +y: Ly, +m: U32) -> Array<U32>:
  (w, +a) = r
  ok.b(Array.get(U32, w, ((Ly.mb(y) + Ly.k(y) : U32) + ((m * 2 : U32) + 1 : U32) : U32)), y, a)

def okp(f: Nat, w: Array<U32>, +y: Ly, +m: U32) -> Array<U32>:
  match f:
    case 0n:
      w
    case 1n+p:
      okp(p, ok.a(Array.get(U32, w, ((Ly.mb(y) + Ly.k(y) : U32) + (m * 2 : U32) : U32)), y, m),
        y, (m + 1 : U32))

# the entries kept aside back onto the heap
def okp.go(r: Array<U32> & U32, +y: Ly) -> Array<U32>:
  (w, +kc) = r
  okp(U32.to_nat(kc), Array.set(U32, w, 6, 0), y, 0)

# opt (Belady): the cached expert used furthest ahead out (ties: the least expert), never one
# the request names, until the cache fits cap slots
def opt.one(w: Array<U32>, +y: Ly, +r: U32) -> Array<U32>:
  +mk = (r + 1 : U32)
  +i = (r * Ly.k(y) : U32)
  +kn = U32.to_nat(Ly.k(y))
  w1 = o2(kn, o1(kn, w, y, mk, (Ly.rq(y) + i : U32)), y, i)
  w2 = oe(U32.to_nat(((Ly.nw(y) + Ly.nq(y) : U32) + 1 : U32)), Hs{w1, True{}, 0}, y, mk)
  fin(okp.go(Array.get(U32, w2, 6), y), y, (r / Ly.l(y) : U32))

# ------------------------------------------------------------------------------- a replay

def sim.p(s0: Bool, s1: Bool, s2: Bool, s3: Bool, w: Array<U32>, +y: Ly) -> Array<U32>:
  match s0 s1 s2 s3:
    case True{} _ _ _:
      reqs(~(w => y => r => st.one(w, y, r)), U32.to_nat((Ly.t(y) * Ly.l(y) : U32)),
        warm.st(U32.to_nat(Ly.nw(y)), w, y, 0), y, 0)
    case False{} True{} _ _:
      w1 = Array.set(U32, w, (Ly.g(y) + 3 : U32), Ly.cap(y))
      reqs(~(w => y => r => lru.one(w, y, r)), U32.to_nat((Ly.t(y) * Ly.l(y) : U32)),
        warm.lru(U32.to_nat(Ly.nw(y)), w1, y, 0), y, 0)
    case False{} False{} True{} _:
      w1 = lims(U32.to_nat(Ly.l(y)), w, y, 0)
      reqs(~(w => y => r => lru.one(w, y, r)), U32.to_nat((Ly.t(y) * Ly.l(y) : U32)),
        warm.lay(U32.to_nat(Ly.nw(y)), w1, y, 0), y, 0)
    case False{} False{} False{} True{}:
      reqs(~(w => y => r => lfu.one(w, y, r)), U32.to_nat((Ly.t(y) * Ly.l(y) : U32)),
        warm.lfu(U32.to_nat(Ly.nw(y)), w, y, 0), y, 0)
    case False{} False{} False{} False{}:
      w1 = nxt(U32.to_nat(Ly.nq(y)), w, y, Ly.nq(y))
      reqs(~(w => y => r => opt.one(w, y, r)), U32.to_nat((Ly.t(y) * Ly.l(y) : U32)),
        warm.opt(U32.to_nat(Ly.nw(y)), w1, y, 0), y, 0)

# the replay, from its array with the requests, the warm list and lfu's counts in place
def sim(w: Array<U32>, +y: Ly) -> Array<U32>:
  sim.p(U32.is_eq(Ly.pol(y), 0), U32.is_eq(Ly.pol(y), 1), U32.is_eq(Ly.pol(y), 2),
    U32.is_eq(Ly.pol(y), 3), w, y)

def run.d(st: Array<U32> & Array<U32>, +y: Ly) -> Array<U32>:
  (src, w) = st
  sim(w, y)

def run.c(st: Array<U32> & Array<U32>, +y: Ly, +sp: U32) -> Array<U32>:
  (src, w) = st
  run.d(cp(U32.to_nat(sel(U32.is_eq(Ly.pol(y), 3), Ly.n(y), 0)), (src, w), sp,
    (Ly.x(y) + 3 : U32), 5), y)

def run.b(st: Array<U32> & Array<U32>, +y: Ly, +so: U32, +sp: U32) -> Array<U32>:
  (src, w) = st
  run.c(cp(U32.to_nat(Ly.nw(y)), (src, w), so, Ly.ro(y), 1), y, sp)

# The replay of the requests at src[sq ..] (t l k words) against cap slots under policy pol,
# from the warm list at src[so ..] (nw words) and, for lfu, the profile's counts at src[sp ..]
# (n words): its array, the misses of each token from Ly.out
def run(src: Array<U32>, +y: Ly, +sq: U32, +so: U32, +sp: U32) -> Array<U32>:
  w = Array.new(U32, U32.to_nat(depth.of((Ly.out(y) + Ly.t(y) : U32))), 0)
  run.b(cp(U32.to_nat(Ly.nq(y)), (src, w), sq, Ly.rq(y), 1), y, so, sp)

# ------------------------------------------------------------------------------- results

def outs.go(f: Nat, r: Array<U32> & U32, +i: U32, acc: List<U32>) -> List<U32>:
  match f:
    case 0n:
      (w, +v) = r
      v <> acc
    case 1n+p:
      (w, +v) = r
      outs.go(p, Array.get(U32, w, (i - 1 : U32)), (i - 1 : U32), v <> acc)

# the misses of each token
def outs(w: Array<U32>, +y: Ly) -> List<U32>:
  +last = ((Ly.out(y) + Ly.t(y) : U32) - 1 : U32)
  outs.go(U32.to_nat((Ly.t(y) - 1 : U32)), Array.get(U32, w, last), last, Nil{})

def word.go(r: Array<U32> & U32) -> U32:
  (w, +v) = r
  v

# the most experts in the cache after a request
def peak(w: Array<U32>) -> U32:
  word.go(Array.get(U32, w, 3))

# ------------------------------------------------------------------------------- from lists

def arr.go(xs: List<U32>, w: Array<U32>, +i: U32) -> Array<U32>:
  match xs:
    case Nil{}:
      w
    case Con{h, t}:
      arr.go(t, Array.set(U32, w, i, h), (i + 1 : U32))

# a replay from lists: the requests qs (t l k ids), the warm list ws (nw ids), lfu's counts ps
# (n words, or none for no profile)
def of(+y: Ly, qs: List<U32>, ws: List<U32>, ps: List<U32>) -> Array<U32>:
  +nq = Ly.nq(y)
  +nw = Ly.nw(y)
  w = Array.new(U32, U32.to_nat(depth.of(((nq + nw : U32) + Ly.n(y) : U32))), 0)
  run(arr.go(ps, arr.go(ws, arr.go(qs, w, 0), nq), (nq + nw : U32)), y, 0, nq, (nq + nw : U32))

# the misses of each token of a replay from lists
def misses(+y: Ly, qs: List<U32>, ws: List<U32>, ps: List<U32>) -> List<U32>:
  outs(of(y, qs, ws, ps), y)

# the most experts in its cache after a request
def most(+y: Ly, qs: List<U32>, ws: List<U32>, ps: List<U32>) -> U32:
  peak(of(y, qs, ws, ps))
