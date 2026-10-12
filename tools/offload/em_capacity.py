"""Expert-major passes and expert weight bytes a prompt row at a pass's row capacity C
(docs/offload.md 13.17), on router traces (offload's MODEL.TEXT.npz: each MoE layer's top-k ids of
2048 tokens, L{layer}_idx): every non-overlapping window of W rows, every layer,
passes(C) = sum_e ceil(n_e / C), and layer-major R = 2 (each row pair's union) for reference;
means over the windows and layers. An expert's bytes are its fp4 slot format's (the 35B's
1671168, the 26B's 3446784, as tools/moe_prefill_cosim.py em reports them).

    python3 tools/offload/em_capacity.py ~/otpu-build/offload/traces [--caps 2,32,64]
        [--speedups 1,4] [--save-route DIR]

The time: a pass of m entries takes max(S, c(m)), S its expert's stream (the co-sim's pass at
C = 2, stream-bound: 0.1118 / 0.2365 ms) and c(m) = ceil(m / 2) x c(2) / x on an MXU x times
today's (--speedups; c(2): the expert's FFN on two rows, 0.100 / 0.210 ms, 13.11; the measured
c(4) and c(8) are 0.97x and 0.93x of this linear model's on the 35B). ms_a_row: a layer's expert
run a prompt row; balance: the C where c(C) meets S, 2 S x / c(2).

--save-route writes each model's layer 0, first window of the prose text as
DIR/MODEL.prose.L0.w0.npy, for tools/moe_prefill_cosim.py em --route.
"""
import argparse
import json
from pathlib import Path

import numpy as np

FB = {"qwen35-35b": 1671168, "gemma26": 3446784}
WIN = {"qwen35-35b": (131, 512), "gemma26": (124, 512)}
S_MS = {"qwen35-35b": 0.1118, "gemma26": 0.2365}       # a pass's stream (co-sim, C = 2)
C2_MS = {"qwen35-35b": 0.100, "gemma26": 0.210}         # c(2), the expert's FFN on two rows


def passes(ids: np.ndarray, E: int, c: int) -> int:
    """Expert-major's passes of up to c entries of one expert."""
    n = np.bincount(ids.reshape(-1), minlength=E)
    return int((-(-n // c)).sum())


def run_ms(ids: np.ndarray, E: int, c: int, s: float, c2: float, x: float) -> float:
    """A layer's expert run, ms: each pass max(stream, ceil(m / 2) c(2) / x), m its entries."""
    n = np.bincount(ids.reshape(-1), minlength=E)
    full, rem = n // c, n % c

    def one(m):
        return max(s, -(-m // 2) * c2 / x)
    return float(full.sum() * one(c) + sum(one(int(r)) for r in rem[rem > 0]))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("traces")
    ap.add_argument("--caps", default="2,32,64")
    ap.add_argument("--speedups", default="1,4", help="the MXU's speed against today's")
    ap.add_argument("--save-route", default=None)
    a = ap.parse_args()
    T, caps = Path(a.traces), [int(c) for c in a.caps.split(",")]
    xs = [float(x) for x in a.speedups.split(",")]
    for m, fb in FB.items():
        texts = sorted(T.glob(f"{m}.*.npz"))
        if not texts:
            continue
        for W in WIN[m]:
            acc, lm, dist = {c: [] for c in caps}, [], []
            ms = {(c, x): [] for c in caps for x in xs}
            for f in texts:
                z = np.load(f, allow_pickle=True)
                meta = json.loads(str(z["meta"]))
                for L in meta["moe_layers"]:
                    ids = z[f"L{L}_idx"].astype(np.int64)
                    for w0 in range(0, len(ids) - W + 1, W):
                        x = ids[w0:w0 + W]
                        for c in caps:
                            acc[c].append(passes(x, meta["experts"], c))
                            for sp in xs:
                                ms[c, sp].append(run_ms(x, meta["experts"], c, S_MS[m],
                                                        C2_MS[m], sp))
                        lm.append(sum(len(np.unique(x[r:r + 2])) for r in range(0, W, 2)))
                        dist.append(len(np.unique(x)))
            rec = dict(model=m, rows=W, windows=len(lm), distinct=round(float(np.mean(dist)), 1),
                       lm2_passes=round(float(np.mean(lm)), 1),
                       lm2_mb_a_row=round(float(np.mean(lm)) * fb / W / 1e6, 2))
            for c in caps:
                p = float(np.mean(acc[c]))
                rec[f"C{c}"] = dict(passes=round(p, 1), mb_a_row=round(p * fb / W / 1e6, 2),
                                    **{f"ms_a_row_x{sp:g}": round(float(np.mean(ms[c, sp])) / W, 4)
                                       for sp in xs})
            rec["balance"] = {f"x{sp:g}": round(2 * S_MS[m] * sp / C2_MS[m], 1) for sp in xs}
            print(json.dumps(rec), flush=True)
        if a.save_route:
            z = np.load(T / f"{m}.prose.npz", allow_pickle=True)
            np.save(Path(a.save_route) / f"{m}.prose.L0.w0.npy",
                    z["L0_idx"][:WIN[m][0]].astype(np.int64))


if __name__ == "__main__":
    main()
