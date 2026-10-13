"""EXPERIMENTAL (docs/tmem_pump.md): what a larger TMEM holds. For each model, TMEM size and MXU
column count, which rows-runs compile (TMEM, ACT RAM) and how many instructions they take
(IMEM: 4096 instructions on the card), from the model's config.json alone (the layout; no
weights). One JSON line per case.

    PYTHONPATH=. python3 tools/tmem_capacity.py rows MODEL_DIR... [--tmem 65536,131072,262144,393216]
        [--mcols 4,8] [--R 1,2,3,4,6,8,12,16] [--pos 300,3000] [--logits last|all|none]
    PYTHONPATH=. python3 tools/tmem_capacity.py layer MODEL_DIR... [--R 1,2,3,4]   (MoE layer runs)

rows: compile_rows of R rows at positions pos .. pos + R - 1 (cap 4096), logits for the last row
(a prompt's last run: fit_chunk's L), for every row (a batched decode step of R users: each row
its LM head and pick) or none (a prompt run before the last). The board's features (PAIR,
DSTEP, STREAM), fp4 weights with an int8 head. layer: compile_layer_run (layer-major MoE
prefill) of the prefix's last layer (Qwen3.5: 4 layers, the 4th attention; Gemma 4: layers 0
and 5), R rows. A case that fails says why (the compiler's message); "imem" is false where the
program is over IMEM. need: the least TMEM (in 1K-word steps, by bisection up to 1M words) at
which the rows-run compiles, and its instructions there: R_max at any size follows.

    PYTHONPATH=. python3 tools/tmem_capacity.py need MODEL_DIR... [--mcols 4,8] [--R ...] [--pos ...]
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from opentpu.compiler import CompileError  # noqa: E402
from opentpu.isasim import board_config  # noqa: E402
from opentpu.llm import load_spec  # noqa: E402


def _ints(s):
    return [int(x) for x in s.split(",") if x]


def cfg_of(tmem, mcols):
    return board_config(PAIR=True, DSTEP=True, STREAM=True, TMEM_WORDS=tmem, MCOLS=mcols)


def cmd_rows(a):
    for m in a.model:
        spec = load_spec(m)
        for tmem in _ints(a.tmem):
            for mc in _ints(a.mcols):
                cfg = cfg_of(tmem, mc)
                Rs = _ints(a.R)
                img = spec.image(cfg, 4096, wformat="fp4", head_format="int8", rows=max(Rs))
                blk = getattr(img, "block", 256)
                for pos in _ints(a.pos):
                    for R in Rs:
                        rec = dict(model=Path(m).name, tmem=tmem, mcols=mc, R=R, pos=pos,
                                   logits=a.logits)
                        lr = {"last": [R - 1], "all": list(range(R)), "none": []}[a.logits]
                        t = time.time()
                        try:
                            progs = img.compile_rows([(0, pos + j) for j in range(R)], lr, blk)
                            n = max(map(len, progs))
                            rec.update(ok=True, instructions=n,
                                       imem=n * 8 <= cfg.IMEM_WORDS)
                        except (CompileError, ValueError) as e:
                            rec.update(ok=False, err=str(e)[:160])
                        rec["s"] = round(time.time() - t, 2)
                        print(json.dumps(rec), flush=True)


def cmd_need(a):
    for m in a.model:
        spec = load_spec(m)
        for mc in _ints(a.mcols):
            Rs = _ints(a.R)
            imgs = {}

            def compiles(tmem, R, pos):
                if tmem not in imgs:
                    cfg = cfg_of(tmem, mc)
                    imgs[tmem] = (cfg, spec.image(cfg, 4096, wformat="fp4", head_format="int8",
                                                  rows=max(Rs)))
                cfg, img = imgs[tmem]
                lr = {"last": [R - 1], "all": list(range(R)), "none": []}[a.logits]
                try:
                    progs = img.compile_rows([(0, pos + j) for j in range(R)], lr,
                                             getattr(img, "block", 256))
                    return max(map(len, progs))
                except (CompileError, ValueError) as e:
                    if "TMEM" not in str(e):
                        raise
                    return None
            for pos in _ints(a.pos):
                for R in Rs:
                    rec = dict(model=Path(m).name, mcols=mc, R=R, pos=pos, logits=a.logits)
                    t = time.time()
                    try:
                        hi_k = 1 << 10                  # K words: hi_k fits, lo fails
                        n = compiles(hi_k << 10, R, pos)
                        if n is None:
                            rec.update(need=None)
                        else:
                            lo = 4
                            while hi_k - lo > 1:
                                mid = (lo + hi_k) // 2
                                if compiles(mid << 10, R, pos) is None:
                                    lo = mid
                                else:
                                    hi_k = mid
                            n = compiles(hi_k << 10, R, pos)
                            rec.update(need=hi_k << 10, instructions=n,
                                       imem=n * 8 <= cfg_of(65536, mc).IMEM_WORDS)
                    except (CompileError, ValueError) as e:
                        rec.update(err=str(e)[:160])
                    rec["s"] = round(time.time() - t, 1)
                    print(json.dumps(rec), flush=True)
                    imgs.clear()


def cmd_layer(a):
    from opentpu.llm.qwen3 import ATTN_BLOCK
    for m in a.model:
        spec = load_spec(m)
        gem = hasattr(spec, "truncated")
        for tmem in _ints(a.tmem):
            for mc in _ints(a.mcols):
                cfg = cfg_of(tmem, mc)
                if gem:
                    s = spec.truncated([0, 5])
                    img = s.image(cfg, 1024, rows=1, wformat="int8", head_format="fp4",
                                  formats="experts=fp4", experts=s.top_k, lookup=True)
                    li = 1
                else:
                    s = dataclasses.replace(spec, kinds=spec.kinds[:4])
                    img = s.image(cfg, 1024, rows=1, wformat="fp4", head_format="int8",
                                  experts=s.moe.k, lookup=True)
                    li = 3
                block = getattr(img, "block", ATTN_BLOCK)
                for R in _ints(a.R):
                    rec = dict(model=Path(m).name, tmem=tmem, mcols=mc, R=R, layer=li)
                    try:
                        progs, ra = img.compile_layer_run(li, 2, block, R=R, embedded=R > 1)
                        n = max(map(len, progs))
                        rec.update(ok=True, instructions=n, imem=n * 8 <= cfg.IMEM_WORDS)
                    except (CompileError, ValueError) as e:
                        rec.update(ok=False, err=str(e)[:160])
                    print(json.dumps(rec), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("rows", "need", "layer"):
        p = sub.add_parser(name)
        p.add_argument("model", nargs="+")
        p.add_argument("--tmem", default="65536,131072,262144,393216")
        p.add_argument("--mcols", default="4,8")
        p.add_argument("--R", default="1,2,3,4" if name == "layer" else "1,2,3,4,6,8,12,16")
        if name != "layer":
            p.add_argument("--pos", default="300,3000")
            p.add_argument("--logits", default="last", choices=("last", "all", "none"))
    a = ap.parse_args()
    {"rows": cmd_rows, "need": cmd_need, "layer": cmd_layer}[a.cmd](a)


if __name__ == "__main__":
    main()
