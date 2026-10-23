# Laws about the expert cache's replays (cache.bend; docs/bend.md "Laws"): each replays a small
# trace, and the values are tools/offload/cachesim.py's for the same trace. PROOF.bend proves
# them (the checker runs the replays): `bend bend/cache/PROOF.bend --check-only`.
#
# C.lay(policy, slots, tokens, layers, k, experts, experts a layer, the warm list's length);
# the policies static 0, lru 1, lru_layer 2, lfu 3, opt 4.
import Base
import ./cache.bend as C

def sum(xs: List<U32>) -> U32:
  match xs:
    case Nil{}:
      0
    case Con{h, t}:
      (h + sum(t) : U32)

# the misses of each token of a trace of one layer, one expert a request, no profile
def one(+pol: U32, +cap: U32, +t: U32, +n: U32, qs: List<U32>, ws: List<U32>, +nw: U32) ->
  List<U32>:
  C.misses(C.lay(pol, cap, t, 1, 1, n, n, nw), qs, ws, [])

# the larger of a word and itself is the word
law max_self:
  for +a: U32
  {C.max(a, a) == a : U32}

# a loop one longer than the cache: LRU misses every request; OPT keeps 0 and evicts 1 at 2
law lru_loop:
  {one(1, 2, 5, 3, [0, 1, 2, 0, 1], [], 0) == [1, 1, 1, 1, 1] : List<U32>}

law opt_loop:
  {one(4, 2, 5, 3, [0, 1, 2, 0, 1], [], 0) == [1, 1, 1, 0, 1] : List<U32>}

# static: the warm set never changes
law static_warm:
  {one(0, 1, 5, 3, [0, 1, 0, 2, 0], [0], 1) == [0, 1, 0, 1, 0] : List<U32>}

# lfu: at equal counts the earlier pushed goes (0 at 2), at 0 the less used (2 against 1)
law lfu_ties:
  {one(3, 2, 5, 3, [0, 1, 2, 1, 0], [], 0) == [1, 1, 1, 0, 1] : List<U32>}

# a cache as large as the experts misses each one once, at its first use
law lru_compulsory:
  {one(1, 3, 6, 3, [0, 1, 0, 1, 2, 0], [], 0) == [1, 1, 0, 0, 1, 0] : List<U32>}

# every expert warm: no misses, whatever the policy
law warm_all:
  {[one(0, 3, 6, 3, [0, 1, 0, 1, 2, 0], [0, 1, 2], 3),
    one(1, 3, 6, 3, [0, 1, 0, 1, 2, 0], [0, 1, 2], 3),
    one(2, 3, 6, 3, [0, 1, 0, 1, 2, 0], [0, 1, 2], 3),
    one(3, 3, 6, 3, [0, 1, 0, 1, 2, 0], [0, 1, 2], 3),
    one(4, 3, 6, 3, [0, 1, 0, 1, 2, 0], [0, 1, 2], 3)]
    == [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0]] : List<List<U32>>}

# lfu holds no more experts than its slots, here three tokens of two new experts and two slots
# (cachesim.py's lfu made room for one miss and took them all in until 2026-10-23: 3)
law lfu_fits:
  {C.most(C.lay(3, 2, 3, 1, 2, 6, 6, 0), [0, 1, 2, 3, 4, 5], [], []) == 2 : U32}

# three layers asking for one expert each and two slots: one LRU over the layers thrashes on
# the sweep, an LRU a layer (lru_layer: max(its share, k) slots) keeps each layer's expert
law layer_sweep:
  {[C.misses(C.lay(1, 2, 4, 3, 1, 6, 2, 0), [0, 2, 4, 0, 2, 4, 0, 2, 4, 0, 2, 4], [], []),
    C.misses(C.lay(2, 2, 4, 3, 1, 6, 2, 0), [0, 2, 4, 0, 2, 4, 0, 2, 4, 0, 2, 4], [], [])]
    == [[3, 3, 3, 3], [3, 0, 0, 0]] : List<List<U32>>}

# A skewed trace: two layers of four experts, top-2, six tokens, four slots, each policy warm
# from the trace's own profile (counts 6 3 2 1 a layer; lru_layer: from all eight).
def skew(+pol: U32, ws: List<U32>, +nw: U32) -> U32:
  sum(C.misses(C.lay(pol, 4, 6, 2, 2, 8, 4, nw), [0, 1, 4, 5, 0, 2, 4, 6, 1, 0, 5, 4, 3, 0, 7,
    4, 0, 1, 4, 6, 2, 0, 4, 5], ws, [6, 3, 2, 1, 6, 3, 2, 1]))

law skew_misses:
  {[skew(0, [0, 4, 1, 5], 4), skew(1, [0, 4, 1, 5], 4), skew(2, [0, 4, 1, 5, 2, 6, 3, 7], 8),
    skew(3, [0, 4, 1, 5], 4), skew(4, [0, 4, 1, 5], 4)] == [6, 10, 10, 9, 8] : List<U32>}

# Belady's bound: OPT misses no more than the caches that take every miss in (lru, lru_layer,
# lfu); static, which takes none in, is not one of them and misses fewer here
law opt_bound:
  {((skew(4, [0, 4, 1, 5], 4) <= skew(1, [0, 4, 1, 5], 4) : U32)
    && (skew(4, [0, 4, 1, 5], 4) <= skew(2, [0, 4, 1, 5, 2, 6, 3, 7], 8) : U32)
    && (skew(4, [0, 4, 1, 5], 4) <= skew(3, [0, 4, 1, 5], 4) : U32)) == True{} : Bool}
