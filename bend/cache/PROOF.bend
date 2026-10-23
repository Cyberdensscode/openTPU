# The proofs of the cache laws (LAWS.bend): the checker runs each replay to its end; max_self
# by the select between a word and itself.
import Base
import ./cache.bend as C
import ./LAWS.bend as Laws

law sel_same:
  for c: Bool
  for -a: U32
  {C.sel(c, a, a) == a : U32}

def sel_same(c, a):
  match c:
    case True{}:
      {==}
    case False{}:
      {==}

def Laws.max_self(a):
  sel_same((a >= a : U32), a)

def Laws.lru_loop():
  {==}

def Laws.opt_loop():
  {==}

def Laws.static_warm():
  {==}

def Laws.lfu_ties():
  {==}

def Laws.lru_compulsory():
  {==}

def Laws.warm_all():
  {==}

def Laws.lfu_fits():
  {==}

def Laws.layer_sweep():
  {==}

def Laws.skew_misses():
  {==}

def Laws.opt_bound():
  {==}
