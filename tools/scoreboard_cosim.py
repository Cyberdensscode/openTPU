"""RTL co-simulation points for the workload scoreboard (tools/scoreboard.py, docs/scoreboard.md).

    python3 tools/scoreboard_cosim.py run MODEL_DIR --layers 1,2 --rows 0,1,4 --pos 64,544
        [--wformat fp4] [--head-format int8] [--formats F] [--batch] [--no-logits]
        [--mcols 4] [--act-rows 0] [--lanes 8] [--tmem 65536] [--mem ldc|ddr|port]
        [--mhz 133.33] [--mts 1066] [--bw 100] [--lat 40] [--out FILE]
    python3 tools/scoreboard_cosim.py probe MODEL_DIR --tmem 65536,131072,262144
        [--mcols 4] [--rows-max 64] [--buckets 1,4] [--wformat fp4] [--head-format int8]

run: each (layers prefix, rows, position) of the model cut to its first N layers (Gemma 4: a
comma list of checkpoint layers, prefixes separated by ':'), on the board's RTL (rtlsim,
BOARD_UARCH, PAIR / DSTEP / STREAM on, the program booted from DRAM). Timing does not depend on
the data, so the DRAM image is zeros: only config.json is read. rows 0: the decode step at pos
(compile_step); R > 0: a prefill run of R rows at positions pos .. pos + R - 1 with the last
row's logits (--no-logits: none); with --batch, R rows of R sequences, each at pos, every row's
logits (the batched decode step; Qwen3 and the Llama-likes). A layer's cost is the difference
of two prefixes. Memory: ldc, the card's channels (LiteDRAM's controller co-simulated, DDR3-1066,
its clock against the core's --mhz: the YPCB at 133.33 MHz); ddr, the DDR3 bank model at --mts
with the core at --mhz (opentpu.profile.ddr3_plusargs; a proxy for other DRAM rates); port,
the native memory model at --bw percent of a beat per cycle per channel and --lat cycles of
latency (memory that keeps up with the core's 128-byte port: HBM, or DDR4 above the port's
rate). One JSON line per run.

probe: compile only. The most rows of a prefill run (compile_rows, the last row's logits) that
fit TMEM / ACT RAM (tmem_rows) and also IMEM (imem_rows), at the first position of each bucket
(256-token attention blocks), per TMEM size. One JSON line per (TMEM, bucket).
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

from opentpu import rtlsim  # noqa: E402
from opentpu.compiler import CompileError  # noqa: E402
from opentpu.isasim import board_config  # noqa: E402
from opentpu.llm import load_spec  # noqa: E402
from opentpu.profile import ddr3_plusargs  # noqa: E402

DRAM = 1 << 30                  # one simulator build per configuration


def prefix(spec, p: str):
    """The model cut to a prefix: N first layers, or Gemma 4's comma list of layers."""
    layers = [int(x) for x in p.split(",")]
    if hasattr(spec, "truncated"):
        return spec.truncated(layers if len(layers) > 1 else range(layers[0]))
    if len(layers) > 1:
        raise ValueError("a list of layers needs a Gemma 4 checkpoint")
    if hasattr(spec, "kinds"):
        return dataclasses.replace(spec, kinds=spec.kinds[:layers[0]])
    return dataclasses.replace(spec, layers=layers[0])


def config(a, **kw):
    over = dict(PAIR=True, DSTEP=True, STREAM=True, DRAM_BYTES=DRAM, MCOLS=a.mcols,
                LANES=a.lanes, TMEM_WORDS=a.tmem_words, ACT_ROWS=a.act_rows, S=a.S)
    over.update(kw)
    return board_config(**over)


IMEM = 1 << 17                  # runs: 16K instructions, one build for every program size


def image(spec, cfg, a, rows: int, batch: int = 1):
    kw = dict(wformat=a.wformat, head_format=a.head_format)
    if a.formats:
        kw["formats"] = a.formats
    if batch > 1:
        kw["batch"] = batch
    return spec.image(cfg, a.cap, rows=max(1, rows), **kw)


def programs(img, a, R: int, pos: int):
    if R == 0:
        return img.compile_step(pos)
    if a.batch:
        return img.compile_rows([(b, pos) for b in range(R)], list(range(R)))
    return img.compile_rows([(0, pos + r) for r in range(R)], [] if a.no_logits else [R - 1])


def memory(a) -> dict:
    if a.mem == "ldc":
        plus = rtlsim.ldc_plusargs(1066, a.mhz) + ddr3_plusargs(3200 / 3, a.mhz)[2:]
        return dict(ldc=1066, bw=100, lat=round(0.3 * a.mhz),
                    plusargs=["+axi_dram=1", "+axi_map=1"] + plus)
    if a.mem == "ddr":
        mts = {1066: 3200 / 3, 1333: 4000 / 3}.get(int(a.mts), a.mts)
        return dict(ldc=0, bw=100, lat=round(0.3 * a.mhz),
                    plusargs=["+axi_dram=1", "+axi_map=1"] + ddr3_plusargs(mts, a.mhz)[2:])
    return dict(ldc=0, bw=a.bw, lat=a.lat, plusargs=[])


def cmd_run(a):
    spec0 = load_spec(a.model)
    uarch = dict(rtlsim.BOARD_UARCH)
    if a.lanes != 8:
        uarch["ULANES"] = a.lanes
    for p in a.layers.split(":"):
        spec = prefix(spec0, p)
        for R in [int(x) for x in a.rows.split(",")]:
            for pos in [int(x) for x in a.pos.split(",")]:
                rec = dict(model=Path(a.model).name, layers=p, rows=R, pos=pos,
                           batch=bool(a.batch and R), wformat=a.wformat,
                           head_format=a.head_format, formats=a.formats, mcols=a.mcols,
                           act_rows=a.act_rows, lanes=a.lanes, tmem=a.tmem_words, mem=a.mem, S=a.S,
                           mhz=a.mhz, bw=a.bw, lat=a.lat, mts=a.mts)
                try:
                    cfg = config(a, IMEM_WORDS=IMEM)       # the fit is IMEM's 4K: instructions
                    img = image(spec, cfg, a, R, R if a.batch and R else 1)
                    progs = programs(img, a, R, pos)
                except (CompileError, ValueError) as e:
                    rec["error"] = f"{type(e).__name__}: {e}"[:300]
                    print(json.dumps(rec), flush=True)
                    continue
                n = max(len(x) for x in progs)
                rec["instructions"] = n
                rec["imem_over"] = n > 4096
                rec["image_mib"] = round(img.nbytes / 2**20, 1)
                if img.nbytes > DRAM:
                    rec["error"] = f"image {img.nbytes} B over the simulated DRAM"
                    print(json.dumps(rec), flush=True)
                    continue
                if a.compile_only:
                    print(json.dumps(rec), flush=True)
                    continue
                drams = [np.zeros(img.nbytes, np.uint8) for _ in range(cfg.S)]
                t = time.time()
                m = memory(a)
                _, _, st = rtlsim.run(cfg, progs, drams, uarch=uarch, axi=True, boot=True,
                                      stall=0, bw=m["bw"], lat=m["lat"], ldc=m["ldc"],
                                      max_cycles=1 << 40, plusargs=m["plusargs"])
                rec["cycles"] = int(st["cycles"])
                rec["sim_s"] = round(time.time() - t, 1)
                print(json.dumps(rec), flush=True)
                if a.out:
                    with open(a.out, "a") as f:
                        f.write(json.dumps(rec) + "\n")


def cmd_probe(a):
    spec = load_spec(a.model)
    for tw in [int(x) for x in a.tmem.split(",")]:
        a.tmem_words = tw
        cfg = config(a, DRAM_BYTES=1 << 32)             # the board's: compile only
        img = image(spec, cfg, a, a.rows_max)
        for b in [int(x) for x in a.buckets.split(",")]:
            pos = (b - 1) * 256
            rec = dict(model=Path(a.model).name, wformat=a.wformat, formats=a.formats,
                       mcols=a.mcols, act_rows=a.act_rows, tmem=tw, bucket=b, pos=pos)
            tm, im, sizes = 0, 0, {}
            for R in range(1, a.rows_max + 1):
                if pos + R > b * 256:
                    break
                try:
                    progs = img.compile_rows([(0, pos + r) for r in range(R)], [R - 1])
                except CompileError as e:
                    if "TMEM" in str(e) or "ACT RAM" in str(e):
                        rec["stop"] = str(e)[:120]
                        break
                    raise
                except ValueError as e:
                    rec["stop"] = str(e)[:120]
                    break
                n = max(len(x) for x in progs)
                sizes[R] = n
                tm = R
                if 8 * n <= cfg.IMEM_WORDS:
                    im = R
            rec.update(tmem_rows=tm, imem_rows=im, instructions=sizes)
            print(json.dumps(rec), flush=True)
            if a.out:
                with open(a.out, "a") as f:
                    f.write(json.dumps(rec) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "probe"):
        s = sub.add_parser(name)
        s.add_argument("model")
        s.add_argument("--wformat", default="fp4")
        s.add_argument("--head-format", default="int8")
        s.add_argument("--formats", default=None)
        s.add_argument("--mcols", type=int, default=4)
        s.add_argument("--act-rows", type=int, default=0)
        s.add_argument("--lanes", type=int, default=8)
        s.add_argument("-S", type=int, default=1)
        s.add_argument("--cap", type=int, default=2048)
        s.add_argument("--out")
    r = sub.choices["run"]
    r.add_argument("--layers", default="1:2")
    r.add_argument("--rows", default="0")
    r.add_argument("--pos", default="544")
    r.add_argument("--batch", action="store_true")
    r.add_argument("--no-logits", action="store_true")
    r.add_argument("--compile-only", action="store_true")
    r.add_argument("--tmem-words", type=int, default=65536)
    r.add_argument("--mem", choices=("ldc", "ddr", "port"), default="ldc")
    r.add_argument("--mhz", type=float, default=133.33)
    r.add_argument("--mts", type=float, default=1066)
    r.add_argument("--bw", type=int, default=100)
    r.add_argument("--lat", type=int, default=40)
    p = sub.choices["probe"]
    p.add_argument("--tmem", default="65536")
    p.add_argument("--rows-max", type=int, default=16)
    p.add_argument("--buckets", default="1,4")
    a = ap.parse_args(argv)
    {"run": cmd_run, "probe": cmd_probe}[a.cmd](a)


if __name__ == "__main__":
    main()
