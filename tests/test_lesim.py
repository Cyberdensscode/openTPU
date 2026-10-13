"""tools/offload/lesim.py (docs/offload.md 15): the decode timeline model and the recall of
next-layer predictions, on small hand-made traces."""
import importlib.util
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "lesim", Path(__file__).resolve().parent.parent / "tools/offload/lesim.py")
ls = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ls)

E = 4


def _req(T, L, rng, k=1):
    """[T, L, k] global ids, each layer's k distinct of its E."""
    return np.stack([np.stack([rng.choice(E, k, replace=False) + j * E for j in range(L)])
                     for _ in range(T)])


def test_all_hits_and_all_misses_follow_the_card_model():
    """Every expert in a slot: a token is L requests of W0 each, then C0. One miss a request
    (one slot, two experts in turn): each request's next post is its critical end + F, or post
    + W0 + the demand's write contention (WM), whichever is later."""
    p = dict(ls.MODELS["g26"])
    T, L = 6, 3
    req = np.array([[[j * E + t % 2] for j in range(L)] for t in range(T)])
    warm = [j * E + e for j in range(L) for e in (0, 1)]
    ms, miss, *_ = ls.run(req, None, 2, warm, E, p)
    assert miss == 0 and np.isclose(ms, L * p["W0"] + p["C0"])
    ms, miss, *_ = ls.run(req, None, 1, [], E, p)
    per = max(p["DET"] + p["HEAD"] + p["REC"] + p["F"], p["W0"] + p["WM"] * p["CT"] * p["MB"])
    assert miss == L and np.isclose(ms, L * per + p["C0"])


def test_the_oracle_prefetch_saves_misses_and_time_without_contention():
    """The next layer's own misses sent after this layer's demand, gated to land before the
    next post: fewer misses, and with no write contention never slower; with contention each
    landed expert costs the card CT x MB."""
    rng = np.random.default_rng(1)
    req = _req(200, 4, rng, k=2)
    p = dict(ls.MODELS["q35"], CT=0.0)
    base = ls.run(req, None, 2, [], E, p)
    orc = ls.run(req, None, 2, [], E, p, "oracle", 8, 2, True)
    assert orc[1] < base[1] and orc[0] <= base[0] and orc[4] > 0
    p["CT"] = 0.17
    orc2 = ls.run(req, None, 2, [], E, p, "oracle", 8, 2, True)
    assert orc2[0] > orc[0]


def test_a_prediction_equal_to_the_routes_recalls_every_expert_and_miss():
    """A predictor whose 8 best are the layer's own top-8 (then 8 others): recall 1 of the
    routes and of the server's misses, and its prefetches are all used."""
    rng = np.random.default_rng(2)
    T, L, En, k = 60, 3, 32, 8
    req = np.stack([np.stack([rng.choice(En, k, replace=False) + j * En for j in range(L)])
                    for _ in range(T)])
    exact = []
    for j in range(L):
        rest = [np.setdiff1d(np.arange(En) + j * En, req[t, j])[:k] for t in range(T)]
        exact.append(np.concatenate([req[:, j], np.array(rest)], 1))
    tr = [dict(E=En, req=req, preds={"pre": exact}, meta={}) for _ in range(2)]
    res = ls.recall(tr, 12, 1.0)
    assert res["pre topk@8"] == 1.0 and res["pre miss@8"] == 1.0 and res["pre miss@16"] == 1.0
    miss, sent, useful = res["pre pf 8 8"]
    assert sent > 0 and useful == sent and miss < res["base"]
