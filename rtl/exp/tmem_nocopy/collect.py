#!/usr/bin/env python3
"""EXPERIMENTAL (docs/tmem_nocopy.md): the co-simulations of run_cosim.sh as a table.

    python3 rtl/exp/tmem_nocopy/collect.py OUTDIR [OUTDIR ...] [--ref i0] [--json FILE]

Each <workload>__<candidate>.log gives the run's cycles (perf_qwen's "N cycles (", or the sum of
the "cycles" of a moe: / zero: run's JSON lines), whether it matched the ISA simulator
("bit-exact vs ISA: True"; zero: and moe: runs have zero images and are not checked) and its
exit code. Prints, per workload, every candidate's cycles against the reference candidate's
(--ref, the card's TMEM) in percent, and per candidate the geometric mean over the workloads
both have. A later OUTDIR's run of the same tag replaces an earlier one's.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

CYC = re.compile(r": (\d+) cycles \(")


def parse(log: Path) -> dict | None:
    txt = log.read_text(errors="replace")
    rc = re.findall(r"^rc=(\d+)", txt, re.M)
    if not rc:
        return None                                   # still running
    r = {"rc": int(rc[-1]), "cycles": None, "exact": None}
    m = CYC.search(txt)
    if m:
        r["cycles"] = int(m.group(1))
    else:
        tot = 0
        for line in txt.splitlines():
            if line.startswith("{") and '"cycles"' in line:
                try:
                    tot += int(json.loads(line)["cycles"])
                except (ValueError, KeyError):
                    pass
        r["cycles"] = tot or None
    if "bit-exact vs ISA: True" in txt:
        r["exact"] = True
    elif "bit-exact vs ISA: False" in txt:
        r["exact"] = False
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--ref", default="i0")
    ap.add_argument("--json")
    ap.add_argument("--cands", help="comma-separated: only these candidates, in this order")
    a = ap.parse_args()
    runs: dict[str, dict[str, dict]] = {}
    for d in a.dirs:
        for log in sorted(Path(d).glob("*__*.log")):
            wl, cand = log.stem.split("__", 1)
            r = parse(log)
            if r is not None:
                runs.setdefault(wl, {})[cand] = r
    cands = sorted({c for w in runs.values() for c in w})
    if a.cands:
        cands = [c for c in a.cands.split(",") if c in cands]
    elif a.ref in cands:
        cands = [a.ref] + [c for c in cands if c != a.ref]
    wls = sorted(runs)
    out = {"runs": runs, "rel": {}}
    print(f"{'workload':10s}" + "".join(f"{c:>9s}" for c in cands))
    logs: dict[str, list[float]] = {c: [] for c in cands}
    for wl in wls:
        ref = runs[wl].get(a.ref, {}).get("cycles")
        row = f"{wl:10s}"
        for c in cands:
            r = runs[wl].get(c)
            if r is None:
                row += f"{'':>9s}"
                continue
            if r["rc"] != 0 or r["cycles"] is None:
                row += f"{'rc' + str(r['rc']):>9s}"
                continue
            flag = "" if r["exact"] in (True, None) else "!"
            if c == a.ref or not ref:
                row += f"{r['cycles']:>8d}{flag or ' '}"
            else:
                rel = r["cycles"] / ref - 1
                out["rel"].setdefault(c, {})[wl] = rel
                logs[c].append(math.log(r["cycles"] / ref))
                row += f"{100 * rel:>+7.2f}%{flag or ' '}"
        print(row)
    print(f"{'geomean':10s}" + "".join(
        f"{'':>9s}" if c == a.ref or not logs[c] else f"{100 * (math.exp(sum(logs[c]) / len(logs[c])) - 1):>+7.2f}% "
        for c in cands))
    print(f"{'n':10s}" + "".join(f"{len(logs[c]) if c != a.ref else '':>9}" for c in cands))
    bad = [(wl, c) for wl in wls for c, r in runs[wl].items() if r["exact"] is False]
    if bad:
        print("NOT BIT-EXACT:", bad)
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
