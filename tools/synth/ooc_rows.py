"""ACT RAM rows' cost out of context (docs/tiled_prefill.md): Vivado synth + place + route of the
ACT RAM and the MXU at the board's parameters, for each ROWS (the most rows of one MM: ACT_ROWS),
on the card's part at the core clock's period. One JSON line per run: LUTs, FFs, LUT RAM, DSPs,
BRAM36 / 18, WNS and fmax.

    python3 tools/synth/ooc_rows.py OUT [--rows 4,16] [--tops otpu_actram,otpu_mxu]
        [--period 7.5] [--mcols 4] [--param otpu_mxu:IMPL=2 ...]

Run on a Vivado host (VIVADO: the vivado binary) from the tree's root, one job at a time.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

from tourney.synth import parse_ooc, vivado_tcl  # noqa: E402

BASE = ["rtl/vpu/otpu_fp.sv", "rtl/vpu/otpu_fpipe.sv", "rtl/top/otpu_pkg.sv",
        "rtl/tech/generic/otpu_tech_ram.sv"]
TOPS = {   # the board's parameters (rtl/shell/otpu_board.sv, boards/ypcb-00338/Makefile)
    "otpu_actram": (["rtl/mem/otpu_actram.sv"], dict(D=128, BLOCKS=128, LANES=8)),
    "otpu_mxu": (["rtl/mxu/otpu_mxu.sv", "rtl/tech/xilinx7/otpu_tech_dsp.sv"],
                 dict(D=128, DEPTH=1024, LANES=8, IMPL=2, SID=0)),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out")
    ap.add_argument("--rows", default="4,16")
    ap.add_argument("--tops", default="otpu_actram,otpu_mxu")
    ap.add_argument("--period", type=float, default=7.5)
    ap.add_argument("--mcols", type=int, default=4)
    ap.add_argument("--param", action="append", default=[], help="TOP:NAME=VALUE")
    a = ap.parse_args()
    vivado = os.environ.get("VIVADO", "vivado")
    extra: dict = {}
    for p in a.param:
        top, kv = p.split(":", 1)
        k, v = kv.split("=", 1)
        extra.setdefault(top, {})[k] = v
    for top in a.tops.split(","):
        srcs, params = TOPS[top]
        for rows in [int(r) for r in a.rows.split(",")]:
            prm = dict(params, MCOLS=a.mcols, ROWS=rows, **extra.get(top, {}))
            out = Path(a.out) / f"{top}_mc{a.mcols}_rows{rows}"
            out.mkdir(parents=True, exist_ok=True)
            tcl = out / "ooc.tcl"
            tcl.write_text(vivado_tcl(ROOT, top, BASE + srcs, prm, out, a.period))
            t = time.time()
            with (out / "vivado.log").open("w") as f:
                rc = subprocess.run([vivado, "-mode", "batch", "-nojournal", "-nolog", "-source",
                                     str(tcl)], cwd=out, stdout=f,
                                    stderr=subprocess.STDOUT).returncode
            log = (out / "vivado.log").read_text()
            util = (out / "util.txt").read_text() if (out / "util.txt").exists() else ""
            res = dict(top=top, rows=rows, params=prm, rc=rc,
                       minutes=round((time.time() - t) / 60, 1))
            res.update(parse_ooc(log, util, a.period))
            print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
