"""Expert-major passes and expert weight bytes a prompt row at a pass's row capacity C
(docs/offload.md 13.17), on router traces (offload's MODEL.TEXT.npz: each MoE layer's top-k ids of
2048 tokens, L{layer}_idx): every non-overlapping window of W rows, every layer,
passes(C) = sum_e ceil(n_e / C), and layer-major R = 2 (each row pair's union) for reference;
means over the windows and layers. An expert's bytes are its fp4 slot format's (the 35B's
1671168, the 26B's 3446784, as tools/moe_prefill_cosim.py em reports them).

    python3 tools/offload/em_capacity.py ~/otpu-build/offload/traces [--caps 2,32,64]
        [--save-route DIR]

--save-route writes each model's layer 0, first window of the prose text as
DIR/MODEL.prose.L0.w0.npy, for tools/moe_prefill_cosim.py em --route.
"""
import argparse
import json
from pathlib import Path

import numpy as np

FB = {"qwen35-35b": 1671168, "gemma26": 3446784}
WIN = {"qwen35-35b": (131, 512), "gemma26": (124, 512)}


def passes(ids: np.ndarray, E: int, c: int) -> int:
    """Expert-major's passes of up to c entries of one expert."""
    n = np.bincount(ids.reshape(-1), minlength=E)
    return int((-(-n // c)).sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("traces")
    ap.add_argument("--caps", default="2,32,64")
    ap.add_argument("--save-route", default=None)
    a = ap.parse_args()
    T, caps = Path(a.traces), [int(c) for c in a.caps.split(",")]
    for m, fb in FB.items():
        texts = sorted(T.glob(f"{m}.*.npz"))
        if not texts:
            continue
        for W in WIN[m]:
            acc, lm, dist = {c: [] for c in caps}, [], []
            for f in texts:
                z = np.load(f, allow_pickle=True)
                meta = json.loads(str(z["meta"]))
                for L in meta["moe_layers"]:
                    ids = z[f"L{L}_idx"].astype(np.int64)
                    for w0 in range(0, len(ids) - W + 1, W):
                        x = ids[w0:w0 + W]
                        for c in caps:
                            acc[c].append(passes(x, meta["experts"], c))
                        lm.append(sum(len(np.unique(x[r:r + 2])) for r in range(0, W, 2)))
                        dist.append(len(np.unique(x)))
            rec = dict(model=m, rows=W, windows=len(lm), distinct=round(float(np.mean(dist)), 1),
                       lm2_passes=round(float(np.mean(lm)), 1),
                       lm2_mb_a_row=round(float(np.mean(lm)) * fb / W / 1e6, 2))
            for c in caps:
                p = float(np.mean(acc[c]))
                rec[f"C{c}"] = dict(passes=round(p, 1), mb_a_row=round(p * fb / W / 1e6, 2))
            print(json.dumps(rec), flush=True)
        if a.save_route:
            z = np.load(T / f"{m}.prose.npz", allow_pickle=True)
            np.save(Path(a.save_route) / f"{m}.prose.L0.w0.npy",
                    z["L0_idx"][:WIN[m][0]].astype(np.int64))


if __name__ == "__main__":
    main()
