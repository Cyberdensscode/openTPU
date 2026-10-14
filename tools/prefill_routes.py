"""The route and rows of a dense model's prompt on the card, bucket by bucket (docs/prefill.md,
docs/tiled_prefill.md): what Engine.prefill takes with resident prompt runs (otpu-chat's), from
prefill.py's own rules on a zero image of the whole model.

    python3 tools/prefill_routes.py --model DIR [--wformat auto] [--head-format F] [--cap 2048]
        [--buckets 1-8] [--imem 4096,8192,16384,0]

For each bucket (positions (b - 1) x 256 ..): the tile's rows (prefill._tile_fit), today's prompt
runs' rows and whether they cover the bucket (prefill._fit), the kind the bucket's runs take
(prefill.tiled: "tile" or "prompt") and r_max, whether a prompt reaching the bucket takes prompt
runs at all (prefill.covers: every bucket it touches covered; else the whole prompt takes
compile_rows' route, Engine.prefill_chunks), and the rows of that route's run at the bucket's
start (qwen3.fit_chunk: up to the image's rows, whole passes of MCOLS, what fits TMEM and IMEM).
With --imem, the tile's rows at each IMEM size in instructions (0: unbounded): what a bigger IMEM
or the looped attention would let the tile take (TMEM and ACT RAM still bound it).

The configuration is the card's bitstream (the target's: OTPU_TARGET, default ypcb-00338 m4;
PAIR, DSTEP, STREAM, SEQ_WC), the engine's rows grown to ACT_ROWS (Engine with tiled prompt
runs), lookup tables (resident). --wformat auto is otpu-chat's default (the model's mix, else
int8). OTPU_TILE must be unset (the default: tiled prompt runs on). One JSON line a bucket.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from opentpu.isasim import board_config  # noqa: E402
from opentpu.llm import formats as FM  # noqa: E402
from opentpu.llm import load_spec, model_dir  # noqa: E402
from opentpu.llm import prefill as PF  # noqa: E402
from opentpu.llm.qwen3 import ATTN_BLOCK, PREFILL_ROWS, Engine, fit_chunk  # noqa: E402


def stub_engine(spec, cfg, cap: int, wformat: str, head_format):
    """An Engine as otpu-chat makes it (resident, prompt runs, the program cache off) for
    prefill.py's choices only: its image, no weights and no backend."""
    eng = object.__new__(Engine)
    rows = max(PREFILL_ROWS, cfg.act_rows) if cfg.S == 1 else PREFILL_ROWS
    img = spec.image(cfg, cap, 1, rows, wformat=wformat, head_format=head_format, lookup=True)
    eng.__dict__.update(
        spec=spec, cfg=cfg, cap=cap, block=ATTN_BLOCK, image=img, rows=rows, batch=1,
        prog_cache=False, prompt_runs=True, tile_runs=None, device_inputs=True, poss=[0],
        backend=SimpleNamespace(), row_server=None,
        _host_rows=getattr(img, "host_rows", None), _prompt_lock=threading.RLock())
    return eng


def bucket(eng, b: int) -> dict:
    p = max((b - 1) * eng.block, PF.conv_k(eng) - 1)
    rec = dict(bucket=b, pos=p)
    t = time.time()
    try:
        R, covered = PF._fit(eng, b)
    except Exception as e:                      # noqa: BLE001 (no prompt run fits the bucket)
        R, covered = 0, False
        rec["prompt_error"] = f"{type(e).__name__}: {e}"[:160]
    T = PF._tile_fit(eng, b)
    tiled = PF.tiled(eng, b)
    rec.update(tile_T=T, today_R=R, today_covered=covered, kind="tile" if tiled else "prompt",
               r_max=PF.r_max(eng, b) if (tiled or R) else 0, covered=tiled or covered)
    # compile_rows' route (the prompt goes there when any bucket it touches is not covered)
    n, _, _ = fit_chunk(eng.image, eng.block, 0, p, eng.rows, 1 << 20, eng.rows,
                        tokens=[0] * eng.rows)
    rec["rows_route_R"] = n
    rec["s"] = round(time.time() - t)
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--wformat", default=FM.AUTO)
    ap.add_argument("--head-format", default=None)
    ap.add_argument("--cap", type=int, default=2048)
    ap.add_argument("--buckets", default="", help="a-b or a list (default: every bucket)")
    ap.add_argument("--imem", default="", help="the tile's rows at these IMEM sizes, in "
                    "instructions (0: unbounded), e.g. 4096,8192,16384,0")
    a = ap.parse_args()
    if os.environ.get("OTPU_TILE", "").strip() not in ("", "1"):
        raise SystemExit("prefill_routes: OTPU_TILE=0 turns the tile off; unset it")
    spec = load_spec(model_dir(a.model))
    wf = FM.auto(spec, a.wformat)
    cfg = board_config(PAIR=True, DSTEP=True, STREAM=True, SEQ_WC=True)
    nb = -(-a.cap // ATTN_BLOCK)
    if not a.buckets:
        bs = list(range(1, nb + 1))
    elif "-" in a.buckets:
        lo, hi = map(int, a.buckets.split("-"))
        bs = list(range(lo, hi + 1))
    else:
        bs = [int(x) for x in a.buckets.split(",")]
    eng = stub_engine(spec, cfg, a.cap, wf, a.head_format)
    imems = [int(x) for x in a.imem.split(",")] if a.imem else []
    alt = {n: stub_engine(spec, dataclasses.replace(cfg, IMEM_WORDS=8 * (n or 1 << 16)),
                          a.cap, wf, a.head_format) for n in imems}
    for b in bs:
        rec = dict(model=Path(a.model).name, wformat=wf, cap=a.cap, **bucket(eng, b))
        for n, e in alt.items():
            rec[f"tile_T_imem{n or 'inf'}"] = PF._probe_tile(e, b)
        print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
