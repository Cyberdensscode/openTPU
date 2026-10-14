"""Prompt-run prefill cycles per row, today's runs against tiled ones (docs/tiled_prefill.md):
RTL co-simulation of one prompt run (docs/prefill.md: a P run at a run-time position, its state
words and tokens from DRAM) of R rows over the first L layers of a model.

    python3 tools/perf_prompt.py --model qwen3.5-0.8b [--layers 4] [--blocks 2]
        [--kernel prompt|tile] [--rows 4,8,16] [--mcols 4] [--act-rows 16] [--tmem 65536]
        [--wformat fp4] [--head-format int8] [--mem ldc|port] [--depth 2] [--ranges]
        [--imem 131072] [--phases] [--compile-only]

--kernel prompt: today's prompt run (the image's compile_prompt_run, R rows); tile: the tiled
prompt run (tiled.compile_tile_run, T rows). The run sits at the start of bucket --blocks
(position (blocks - 1) * 256, tpos 0: it attends over blocks - 1 whole blocks and its own
masked one). ACT_ROWS: --act-rows, at least MCOLS (0: MCOLS, the card's). --depth: the tile's
attention pipeline depth (0: compile_tile_run's choice, at the virtual IMEM ATTN_DEPTH);
--ranges: the tile's range forms (tiled.ranges: W_o's operand a QACT range a part). IMEM
is virtual (--imem words, default 1 << 17: 16K instructions; 1 << 19 holds every tile TMEM
does) so a run that would not fit still measures; "imem_fits" says
whether it fits the card's (4096 instructions). --layers: the model's first L layers (Gemma 4:
Spec.truncated).

The images are zeros (timing does not depend on the data). One JSON line per run: cycles,
cycles per row, the core-port roofline (port-B chunks, the MXU's own cycles), with --phases the
cycles per phase. --mem as tools/perf_tiled.py's.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from opentpu import isa as I  # noqa: E402
from opentpu import rtlsim  # noqa: E402
from opentpu.isasim import board_config  # noqa: E402
from opentpu.llm import load_spec, model_dir  # noqa: E402
from opentpu.profile import ddr3_plusargs, parse  # noqa: E402

from perf_tiled import MHZ, phase  # noqa: E402

CARD_IMEM = 4096 * 8


def cut(spec, n):
    if hasattr(spec, "truncated"):
        return spec.truncated(range(n))
    if hasattr(spec, "kinds"):
        return dataclasses.replace(spec, kinds=spec.kinds[:n])
    return dataclasses.replace(spec, layers=n)


def compile_run(img, kernel, blocks, R, block=256, depth=0, ranges=False):
    if kernel == "prompt":
        return img.compile_prompt_run(blocks, R, "P", block)[0]
    from opentpu.llm import tiled
    if not depth and not ranges:
        return tiled.compile_tile_run(img, blocks, R, "P", block)[0]
    forms = tiled.variants
    extra = dict(({"depth": depth} if depth else {}), **({"whole": False} if ranges else {}))
    try:                                    # the forms at this attention depth / in ranges only
        tiled.variants = lambda *a: [dict(v, **extra) for v in forms(*a)]
        return tiled.compile_tile_run(img, blocks, R, "P", block)[0]
    finally:
        tiled.variants = forms


def one(a, spec, R):
    act = max(a.act_rows or a.mcols, a.mcols)
    kw = dict(MCOLS=a.mcols, ACT_ROWS=act, TMEM_WORDS=a.tmem, IMEM_WORDS=a.imem, PAIR=True,
              DSTEP=True, STREAM=True)
    cap = 256 * (a.blocks + 1)
    rec = dict(model=a.model, layers=a.layers, blocks=a.blocks, R=R, mcols=a.mcols,
               act_rows=act, tmem=a.tmem, wformat=a.wformat, mem=a.mem, kernel=a.kernel,
               depth=getattr(a, "depth", 0), ranges=getattr(a, "ranges", False))
    ikw = dict(rows=max(R, 8), wformat=a.wformat, head_format=a.head_format, lookup=True)
    probe = spec.image(board_config(DRAM_BYTES=1 << 40, **kw), cap, **ikw)
    cfg = board_config(DRAM_BYTES=1 << max(20, (probe.nbytes - 1).bit_length()), **kw)
    img = spec.image(cfg, cap, **ikw)
    t = time.time()
    try:
        progs = compile_run(img, a.kernel, a.blocks, R, depth=getattr(a, "depth", 0),
                            ranges=getattr(a, "ranges", False))
    except Exception as e:                  # TMEM / ACT RAM / rows: the fit answer
        rec["error"] = f"{type(e).__name__}: {e}"[:300]
        return rec
    rec["instructions"] = max(len(p) for p in progs)
    rec["imem_fits"] = rec["instructions"] * 8 <= CARD_IMEM
    rec["mms"] = sum(1 for i in progs[0] if i.op == I.MM)
    rec["compile_s"] = round(time.time() - t, 1)
    if a.compile_only:
        return rec
    dram = np.zeros(img.nbytes, np.uint8)
    if a.mem == "ldc":
        plus = rtlsim.ldc_plusargs(1066, MHZ) + ddr3_plusargs(3200 / 3, MHZ)[2:]
        mem = dict(ldc=1066, bw=100, lat=round(0.3 * MHZ),
                   plusargs=["+axi_dram=1", "+axi_map=1"] + plus)
    else:
        mem = dict(ldc=0, bw=100, lat=a.lat, plusargs=[])
    t = time.time()
    _, _, st = rtlsim.run(cfg, progs, [dram], trace=True, uarch=rtlsim.BOARD_UARCH, axi=True,
                          boot=True, stall=0, max_cycles=1 << 40, **mem)
    rec["sim_s"] = round(time.time() - t)
    p = parse(st["trace"], cfg, progs, "prompt")
    p.cycles = st["cycles"]
    ps = p.roofline()["per_slice"][0]
    rec.update(cycles=p.cycles, per_row=round(p.cycles / R), portb=ps["portb"],
               porta=ps["porta_cycles"], mxu=ps["mxu"],
               busy={k: v for k, v in p.unit_busy(0).items() if v})
    if a.phases:
        ph = p.phases(lambda r: phase(progs[0][r.pc]))
        rec["phases"] = {k: dict(cycles=v["cycles"], portb=v["portb"], VPU=v["VPU"],
                                 MXU=v["MXU"]) for k, v in ph.items()}
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--blocks", type=int, default=2)
    ap.add_argument("--kernel", choices=["prompt", "tile"], default="prompt")
    ap.add_argument("--rows", default="4,8,16")
    ap.add_argument("--mcols", type=int, default=4)
    ap.add_argument("--act-rows", type=int, default=0)
    ap.add_argument("--tmem", type=int, default=1 << 16)
    ap.add_argument("--wformat", default="fp4")
    ap.add_argument("--head-format", default="int8")
    ap.add_argument("--mem", choices=["ldc", "port"], default="ldc")
    ap.add_argument("--lat", type=int, default=40)
    ap.add_argument("--depth", type=int, default=0)
    ap.add_argument("--imem", type=int, default=1 << 17, help="IMEM words (virtual)")
    ap.add_argument("--ranges", action="store_true")
    ap.add_argument("--phases", action="store_true")
    ap.add_argument("--compile-only", action="store_true")
    a = ap.parse_args()
    spec = cut(load_spec(model_dir(a.model)), a.layers)
    for R in [int(x) for x in a.rows.split(",")]:
        print(json.dumps(one(a, spec, R)), flush=True)


if __name__ == "__main__":
    main()
