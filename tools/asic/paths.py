"""The critical paths of a NanGate45 run (tools/asic/flow.py), grouped by unit and kind.

    python3 tools/asic/paths.py POINT [N]     # N worst endpoints (default 200) -> paths.json

Runs tools/asic/paths.tcl in the ORFS container on the finished design (report_checks, one path
per endpoint), then classifies each path by the hierarchy of its start and end points:

    fpipe    through otpu_fpipe's adder / multiplier (latency LM / LA: a parameter)
    systolic the systolic MXU's cells (otpu_pe, otpu_colend: fixed stages)
    tmem     the TMEM's banks, its bank mux or the slice's arbiter (fixed)
    sram     from or into an SRAM macro (the macro's access)
    dma      the DMA, its chunk buffer, the memory ports
    other    the rest (sequencer, quantizer control, VPU control, ...)

and prints the worst paths with their unit (the first levels of the hierarchy), cell count
(logic levels), arrival and slack.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import flow  # noqa: E402

HERE = Path(__file__).resolve().parent


def run(point: str, n: int) -> Path:
    o = flow.out_dir(point)
    out = o / "paths.txt"
    cmd = ["docker", "run", "--rm", "--platform", "linux/amd64", "--cpus=4",
           "-v", f"{flow.ROOT}:{flow.ROOT}", "-w", str(flow.ROOT), flow.IMAGE, "bash", "-lc",
           f"cd /OpenROAD-flow-scripts && source env.sh >/dev/null 2>&1; cd flow && "
           f"PATHS_N={n} PATHS_OUT={out} make DESIGN_CONFIG={o / 'config.mk'} "
           f"WORK_HOME={o / 'work'} RUN_SCRIPT={HERE / 'paths.tcl'} run"]
    with open(o / "paths.log", "w") as log:
        r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    if r.returncode or not out.exists():
        raise RuntimeError(f"paths.tcl failed: {o / 'paths.log'}")
    return out


def unit(name: str) -> str:
    """The unit of an instance or pin name: its first hierarchy levels below the slice."""
    parts = re.split(r"[./]", name.replace("\\", ""))
    parts = [p for p in parts if p and p != "u_slice"]
    return "/".join(parts[:1]) if parts else name


def kind(start: str, end: str, cells: list[str]) -> str:
    s = (start + " " + end + " " + " ".join(cells)).replace("\\", "")
    if "fakeram45_" in " ".join(cells) or re.search(r"u_m\b|\.u_m[./]|/u_m/", start + end):
        return "sram"
    if re.search(r"otpu_pe|otpu_colend|g_pe|u_colend|g_sys|g_col", s):
        return "systolic"
    if re.search(r"u_fma|u_fadd|u_fmul|fpipe|otpu_fadd|otpu_fmul", s):
        return "fpipe"
    if re.search(r"u_tmem|gnt|arb", s):
        return "tmem"
    if re.search(r"u_dma|u_lb", s):
        return "dma"
    return "other"


def parse(text: str) -> list[dict]:
    paths = []
    for blk in text.split("Startpoint: ")[1:]:
        start = blk.split()[0]
        m = re.search(r"Endpoint: (\S+)", blk)
        end = m.group(1) if m else "?"
        cells = re.findall(r"\(([A-Za-z0-9_]+)\)", blk.split("data arrival time")[0])
        cells = [c for c in cells if not c.startswith(("rising", "falling", "propagated",
                                                       "ideal", "net", "in", "out"))]
        arr = re.search(r"([-\d.]+)\s+data arrival time", blk)
        slack = re.search(r"([-\d.]+)\s+slack", blk)
        paths.append({"start": start, "end": end, "levels": max(0, len(cells) - 2),
                      "cells": cells, "arrival": float(arr.group(1)) if arr else None,
                      "slack": float(slack.group(1)) if slack else None,
                      "unit": f"{unit(start)} -> {unit(end)}", "kind": kind(start, end, cells)})
    return paths


def main(argv: list[str]) -> None:
    point = argv[0]
    n = int(argv[1]) if len(argv) > 1 else 200
    o = flow.out_dir(point)
    txt = o / "paths.txt"
    if not txt.exists() or os.environ.get("PATHS_RERUN"):
        run(point, n)
    paths = parse(txt.read_text())
    (o / "paths.json").write_text(json.dumps(paths, indent=1))
    by = defaultdict(list)
    for p in paths:
        by[p["kind"]].append(p)
    print(f"{point}: {len(paths)} endpoints; worst slack per kind:")
    for k, ps in sorted(by.items(), key=lambda kv: min(p["slack"] for p in kv[1])):
        w = min(ps, key=lambda p: p["slack"])
        print(f"  {k:9s} {len(ps):4d} paths, worst {w['slack']:.3f} ns ({w['levels']} levels): "
              f"{w['start']} -> {w['end']}")
    print("worst 25:")
    for p in sorted(paths, key=lambda p: p["slack"])[:25]:
        print(f"  {p['slack']:7.3f} {p['arrival']:6.3f} {p['levels']:3d} lv  {p['kind']:8s} "
              f"{p['unit']:30s} {p['start']} -> {p['end']}")


if __name__ == "__main__":
    main(sys.argv[1:])
