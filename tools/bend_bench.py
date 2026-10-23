"""The Bend ISA simulator against isasim on a real model (docs/bend.md): the same Engine (one
image, the same programs) runs a prompt's prefill and greedy decode steps on isasim
(qwen3.IsaBackend) and on the Bend machine (opentpu.bend_sim.BendBackend, at each thread count);
every run's logits must agree bit for bit, and the time inside the backends' run() gives the
speedup (the compiler's time, the same for both, is left out; it is reported apart).

    python3 tools/bend_bench.py [--model qwen3|qwen35|lfm2|DIR] [--prompt 16] [--steps 4]
                                [--threads 0,1,2,4,8] [--rows 8] [--no-isa] [--json out.json]

--threads: the Bend runtime's CPU threads (0: its default); --rows: the prefill's rows per run;
--no-isa: Bend alone (the agreement is then checked between thread counts only).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class Timed:
    """A backend whose run() is timed (the simulator's share of the wall time)."""

    def __init__(self, backend):
        self.b, self.runs, self.seconds = backend, 0, 0.0

    def __getattr__(self, name):
        return getattr(self.b, name)

    def run(self, programs, args=None):
        t = time.perf_counter()
        try:
            return self.b.run(programs, args) if args is not None else self.b.run(programs)
        finally:
            self.seconds += time.perf_counter() - t
            self.runs += 1


def session(spec, W, cap, factory, prompt, steps, rows) -> dict:
    """One backend: the prefill of `prompt` (rows per run), then `steps` greedy decode steps."""
    from opentpu.llm.qwen3 import Engine
    box = {}

    def make(cfg, images):
        inner = factory(cfg, images) if factory else None
        if inner is None:
            from opentpu.llm.qwen3 import IsaBackend
            inner = IsaBackend(cfg, images)
        box["b"] = Timed(inner)
        return box["b"]

    t0 = time.perf_counter()
    eng = Engine(spec, W, cap=cap, backend=make)
    t1 = time.perf_counter()
    b = box["b"]
    out, logits = {}, []
    t = time.perf_counter()
    lg = eng.prefill(prompt, chunk=rows)
    out["prefill_s"], out["prefill_sim_s"], out["prefill_runs"] = \
        time.perf_counter() - t, b.seconds, b.runs
    logits.append(np.asarray(lg, np.float32).copy())
    toks = []
    sim0, n0 = b.seconds, b.runs
    t = time.perf_counter()
    for _ in range(steps):
        tok = int(np.argmax(logits[-1].reshape(-1, logits[-1].shape[-1])[-1]))
        toks.append(tok)
        logits.append(np.asarray(eng.step(tok), np.float32).copy())
    out["decode_s"], out["decode_sim_s"], out["decode_runs"] = \
        time.perf_counter() - t, b.seconds - sim0, b.runs - n0
    out["build_s"], out["tokens"], out["cfg"] = t1 - t0, toks, repr(eng.cfg)
    eng._drain()
    if hasattr(b.b, "close"):
        b.b.close()
    return out, logits


def main() -> None:
    from opentpu.bend_sim import bend_backend
    from opentpu.llm import load_spec, model_dir
    from opentpu.llm.qwen3 import load_weights

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--model", default="qwen3")
    ap.add_argument("--prompt", type=int, default=16, help="prompt tokens")
    ap.add_argument("--steps", type=int, default=4, help="decode steps")
    ap.add_argument("--threads", default="0", help="the Bend runtime's threads (0: default)")
    ap.add_argument("--rows", type=int, default=None, help="prefill rows per run")
    ap.add_argument("--cap", type=int, default=256)
    ap.add_argument("--no-isa", action="store_true")
    ap.add_argument("--json")
    a = ap.parse_args()
    path = model_dir(a.model)
    spec, W = load_spec(path), load_weights(path)
    rng = np.random.default_rng(0)
    prompt = [int(x) for x in rng.integers(100, 20000, a.prompt)]
    res = {"model": path.name, "prompt": a.prompt, "steps": a.steps, "runs": {}}
    ref = None
    names = ([] if a.no_isa else ["isasim"]) + [f"bend-t{n}" for n in map(int, a.threads.split(","))]
    for name in names:
        f = None if name == "isasim" else bend_backend(int(name.split("-t")[1]) or None)
        r, logits = session(spec, W, a.cap, f, prompt, a.steps, a.rows)
        if ref is None:
            ref, r["agree"] = logits, None
        else:
            r["agree"] = len(ref) == len(logits) and all(
                x.shape == y.shape and np.array_equal(x.view(np.uint32), y.view(np.uint32))
                for x, y in zip(ref, logits))
        res["runs"][name] = r
        print(f"{name:9s} prefill {r['prefill_sim_s']:8.2f} s sim ({r['prefill_runs']} runs, "
              f"{r['prefill_s']:.2f} s wall)  decode {r['decode_sim_s'] / max(1, a.steps):7.3f} "
              f"s/token sim ({r['decode_s'] / max(1, a.steps):.3f} wall)  build "
              f"{r['build_s']:.1f} s  bit-identical {r['agree']}", flush=True)
    base = res["runs"].get("isasim")
    if base:
        for name, r in res["runs"].items():
            if name != "isasim":
                r["prefill_speedup"] = base["prefill_sim_s"] / r["prefill_sim_s"]
                r["decode_speedup"] = base["decode_sim_s"] / r["decode_sim_s"]
                print(f"{name}: prefill {r['prefill_speedup']:.1f}x, decode "
                      f"{r['decode_speedup']:.1f}x over isasim")
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1))
    bad = [n for n, r in res["runs"].items() if r["agree"] is False]
    if bad:
        raise SystemExit(f"logits differ: {bad}")


if __name__ == "__main__":
    main()
