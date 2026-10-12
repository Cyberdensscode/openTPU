"""The critical paths of a NanGate45 run (tools/asic/flow.py), grouped by unit and stage.

    python3 tools/asic/paths.py POINT [N] [STAGE]   # N worst endpoints (default 200) of STAGE
                                                    # (6_final; 4_cts, 5_route) -> paths.json

Runs tools/asic/paths.tcl in the ORFS container on the design (report_checks, one path per
endpoint), then names each path's start and end register by unit (the instance below the slice:
u_seq, u_vpu, u_quant, u_mxu, u_dma, u_tmem, u_act; "slice" for the slice's own registers, the
TMEM arbiter's) and stage, and gives the path a kind:

    fpipe    into or out of a register of otpu_fmul's / otpu_fadd's fixed stages (fmul s1 = the
             significand product, s2 = normalise and round; fadd s1..s4). LM / LA set these
             units' latency, but the stages beyond 2 / 4 are delay registers appended after the
             logic (otpu_delay u_pad): deeper LM / LA move no logic, so they shorten no path
             unless synthesis retimes (SYNTH_RETIME_MODULES)
    systolic the systolic MXU's cells (otpu_pe, otpu_mxce)
    sram     from or into an SRAM macro (its clock-to-q or setup and the wire to it)
    tmem     the TMEM's bank muxes and its copy selection, the slice's arbiter
    dma      the DMA and its chunk buffer
    control  the rest (sequencer, VPU / quantizer control, ...)

and prints the worst path per kind, then the worst 25 with their logic levels, arrival and slack.
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

# otpu_fpipe's operator instances (otpu_vpu, otpu_quant, otpu_se_comp, otpu_se_tail, otpu_mxu,
# otpu_fmadd / otpu_fmma inside them) and their stage registers
FMUL = {"u_m", "u_c", "u_m1", "u_m2", "u_sq", "u_q", "u_r", "u_c2", "u_inv", "u_sc", "u_y"}
FADD = {"u_a", "u_acc"}
FMUL_STAGE = {"m": "fmul.s1", "r": "fmul.s2"}
FADD_STAGE = {"s1": "fadd.s1", "s2": "fadd.s2", "s3": "fadd.s3", "r": "fadd.s4", "ro": "fadd.s4"}


def run(point: str, n: int, stage: str) -> Path:
    o = flow.out_dir(point)
    out = o / f"paths_{stage}.txt"
    cmd = ["docker", "run", "--rm", "--platform", "linux/amd64", "--cpus=4",
           "-v", f"{flow.ROOT}:{flow.ROOT}", "-w", str(flow.ROOT), flow.IMAGE, "bash", "-lc",
           f"cd /OpenROAD-flow-scripts && source env.sh >/dev/null 2>&1; cd flow && "
           f"PATHS_N={n} PATHS_ALL=100000 PATHS_OUT={out} PATHS_STAGE={stage} make DESIGN_CONFIG={o / 'config.mk'} "
           f"WORK_HOME={o / 'work'} RUN_SCRIPT={HERE / 'paths.tcl'} run"]
    with open(o / "paths.log", "w") as log:
        r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    if r.returncode or not out.exists():
        raise RuntimeError(f"paths.tcl failed: {o / 'paths.log'}")
    return out


def where(pin: str, cell: str = "") -> dict:
    """A start or end point's unit, register and stage: u_slice/u_vpu/g_lane[0].g_ma.u_ma.u_a.s2[5]
    $_DFF_P_ (or its pin .../Q with the cell) -> u_vpu, s2 under u_a: fadd.s2. A top-level port:
    unit "port"."""
    name = pin.replace("\\", "")
    if "/" not in name:
        return {"unit": "port", "reg": re.sub(r"\[\d+\]$", "", name), "stage": "port"}
    if cell:                       # a pin: the instance is the rest
        name = name.rsplit("/", 1)[0]
    name = re.sub(r"\$_\w+$", "", name)
    parts = [p for p in re.split(r"[/.](?![^\[]*\])", name) if p]
    if parts and parts[0] == "u_slice":
        parts = parts[1:]
    unit = parts[0] if len(parts) > 1 and parts[0].startswith("u_") else "slice"
    reg = re.sub(r"(\[\d+\])+$", "", parts[-1]) if parts else name
    parent = parts[-2] if len(parts) > 1 else ""
    stage = reg
    if cell.startswith("fakeram45_") or re.search(r"g_\d+x\d+", name):
        stage = "sram"
    elif parent in FMUL and reg in FMUL_STAGE:
        stage = FMUL_STAGE[reg]
    elif parent in FADD and reg in FADD_STAGE:
        stage = FADD_STAGE[reg]
    elif parent == "u_pad":
        stage = "fpipe.pad"
    elif re.search(r"\bu_pe\b", name):
        stage = "pe"
    elif re.search(r"\bu_ce\b", name):
        stage = "colend"
    return {"unit": unit, "reg": reg, "stage": stage}


def kind(s: dict, e: dict) -> str:
    st = (s["stage"], e["stage"])
    if "port" in st:
        return "io"
    if "sram" in st:
        return "sram"
    if any(x.startswith(("fmul.", "fadd.", "fpipe.")) for x in st):
        return "fpipe"
    if "pe" in st or "colend" in st:
        return "systolic"
    if "u_tmem" in (s["unit"], e["unit"]) or "slice" in (s["unit"], e["unit"]):
        return "tmem"
    if "u_dma" in (s["unit"], e["unit"]):
        return "dma"
    return "control"


def parse(text: str) -> list[dict]:
    """report_checks -format full: the worst paths, with their cells."""
    paths = []
    for blk in text.split("Startpoint: ")[1:]:
        start = blk.split()[0]
        m = re.search(r"Endpoint: (\S+)", blk)
        end = m.group(1) if m else "?"
        body = blk.split("data arrival time")[0]
        # the data path: from the start register's clock pin on (the clock tree before it out)
        ck = body.find("/CK (")
        cells = re.findall(r"\(([A-Za-z0-9_]+)\)", body[ck:] if ck >= 0 else body)
        cells = [c for c in cells if not c.startswith(("rising", "falling", "propagated",
                                                       "ideal", "net", "in", "out"))]
        arr = re.search(r"([-\d.]+)\s+data arrival time", blk)
        slack = re.search(r"([-\d.]+)\s+slack", blk)
        s, e = where(start), where(end)
        paths.append({"start": start, "end": end, "levels": max(0, len(cells) - 2),
                      "cells": cells, "arrival": float(arr.group(1)) if arr else None,
                      "slack": float(slack.group(1)) if slack else None,
                      "from": s, "to": e, "kind": kind(s, e)})
    return paths


def parse_all(text: str) -> list[dict]:
    """report_checks -format summary: every endpoint's worst path (start, end, slack)."""
    out = []
    for m in re.finditer(r"^(\S+) \((\w+)\) (\S+) \((\w+)\)\s+([-\d.]+)$", text, re.M):
        s, e = where(m.group(1), m.group(2)), where(m.group(3), m.group(4))
        out.append({"from": s, "to": e, "slack": float(m.group(5)), "kind": kind(s, e)})
    return out


def parse_stages(text: str) -> list[dict]:
    """paths.tcl's stage classes: the worst path from and to each class's registers."""
    out = []
    for blk in re.split(r"^### ", text, flags=re.M)[1:]:
        head, _, body = blk.partition("\n")
        m = re.match(r"(from|to) (\S+)$", head.strip())
        if not m or "Startpoint" not in body:
            continue
        p = parse(body)[0]
        out.append({"dir": m.group(1), "class": m.group(2), **p})
    return out


def label(w: dict) -> str:
    return f"{w['unit']}:{w['stage']}"


def main(argv: list[str]) -> None:
    point = argv[0]
    n = int(argv[1]) if len(argv) > 1 else 200
    stage = argv[2] if len(argv) > 2 else "6_final"
    o = flow.out_dir(point)
    txt = o / f"paths_{stage}.txt"
    if not txt.exists() or os.environ.get("PATHS_RERUN"):
        run(point, n, stage)
    paths = parse(txt.read_text())
    (o / f"paths_{stage}.json").write_text(json.dumps(paths, indent=1))
    allp = parse_all((o / f"paths_{stage}.txt.all").read_text())
    (o / f"paths_{stage}.json").write_text(json.dumps({"worst": paths, "all": allp}))
    by = defaultdict(list)
    for p in allp:
        by[p["kind"]].append(p)
    print(f"{point} ({stage}): {len(allp)} endpoints; per kind, the worst and its from -> to:")
    for k, ps in sorted(by.items(), key=lambda kv: min(p["slack"] for p in kv[1])):
        w = min(ps, key=lambda p: p["slack"])
        print(f"  {k:9s} {len(ps):6d} endpoints, worst {w['slack']:7.3f} ns: "
              f"{label(w['from'])} -> {label(w['to'])}")
    # the groups: a kind and its start and end unit:stage, worst first
    grp = defaultdict(list)
    for p in allp:
        grp[(p["kind"], label(p["from"]), label(p["to"]))].append(p["slack"])
    print("groups (reg to reg; kind, from -> to: endpoints, worst slack):")
    rows = sorted(((k, f, t, len(v), min(v)) for (k, f, t), v in grp.items() if k != "io"),
                  key=lambda r: r[4])
    for k, f, t, n_, w in rows[:40]:
        print(f"  {k:8s} {f:32s} -> {t:32s} {n_:5d}  {w:7.3f} ns")
    print("worst per kind with levels (full paths):")
    seen = set()
    for p in sorted(paths, key=lambda p: p["slack"]):
        if p["kind"] in seen:
            continue
        seen.add(p["kind"])
        print(f"  {p['kind']:8s} {p['slack']:7.3f} ns {p['levels']:3d} levels: {p['start']} -> {p['end']}")
    st = o / f"paths_{stage}.txt.stages"
    if st.exists():
        print("stage classes (the worst path from / to each class's registers):")
        for p in parse_stages(st.read_text()):
            print(f"  {p['dir']:4s} {p['class']:9s} {p['slack']:7.3f} ns {p['levels']:3d} levels "
                  f"{p['kind']:8s} {label(p['from'])} -> {label(p['to'])}")
    print("worst 25:")
    for p in sorted(paths, key=lambda p: p["slack"])[:25]:
        print(f"  {p['slack']:7.3f} {p['arrival']:6.3f} {p['levels']:3d} lv  {p['kind']:8s} "
              f"{p['start']} -> {p['end']}")


if __name__ == "__main__":
    main(sys.argv[1:])
