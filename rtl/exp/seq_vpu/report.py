"""EXPERIMENTAL (docs/seq_vpu.md): tables from run_cosim.sh's results.

    python3 rtl/exp/seq_vpu/report.py cycles OUTDIR [--ref i0] [--cands a,b,..] [--markdown]
    python3 rtl/exp/seq_vpu/report.py hazards OUTDIR [--cand i0]

cycles: each workload's core cycles under each candidate and the change against --ref (a MoE
run's prefixes and a zero-image run are rows of their own); "!" marks a run that was not
bit-exact against the ISA simulator, "?" one that failed. hazards: otpu_seq's statistics
(+hz_log) of a candidate's runs: each unit's cycles busy / with nothing to start / blocked by a
dependency (and on which units), the MXU's release waits, the dependencies by kind and class,
and the wait-count oracle's ranks.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

UNITS = ["DMA", "MXU", "QUANT", "VPU", "COLL"]
CLS = ["LD", "ST", "DSTEP", "WAITW", "MM", "QACT", "QST", "VOP.ew", "VOP.red", "VOP.comp",
       "RLD", "GATHER/BAR"]


def runs(out: Path) -> dict:
    """{(workload row, candidate): (cycles, flag)}; flag "" ok, "!" not bit-exact, "?" failed."""
    res = {}
    for f in sorted(out.glob("*__*.log")):
        wl, cand = f.stem.split("__", 1)
        t = f.read_text(errors="replace")
        rc = re.search(r"^rc=(\d+)", t, re.M)
        if not rc:
            continue                       # still running
        bad = "" if rc.group(1) == "0" else "?"
        if "bit-exact vs ISA: False" in t:
            bad = "!"
        js = [json.loads(x) for x in re.findall(r"^(\{.*\})$", t, re.M)]
        js = [j for j in js if "cycles" in j]
        if js:
            for j in js:
                row = wl if len(js) == 1 else f"{wl}#{j.get('prefix', j.get('layer'))}"
                res[(row, cand)] = (int(j["cycles"]), bad)
            continue
        m = re.search(r"^\S.*?: (\d+) cycles, roofline", t, re.M)
        if m:
            res[(wl, cand)] = (int(m.group(1)), bad)
        elif bad:
            res[(wl, cand)] = (0, bad)
    return res


def cycles(a) -> None:
    res = runs(Path(a.outdir))
    rows = sorted({r for r, _ in res}, key=_order)
    cands = a.cands.split(",") if a.cands else sorted({c for _, c in res} - {a.ref})
    hdr = ["workload", a.ref] + cands
    lines = []
    for r in rows:
        ref = res.get((r, a.ref))
        cells = [r, f"{ref[0]:,}{ref[1]}" if ref else "-"]
        for c in cands:
            v = res.get((r, c))
            if not v:
                cells.append("")
            elif not ref or not ref[0] or not v[0]:
                cells.append(f"{v[0]:,}{v[1]}")
            else:
                cells.append(f"{100 * (v[0] / ref[0] - 1):+.2f}%{v[1]}")
        lines.append(cells)
    _table(hdr, lines, a.markdown)


def _order(r: str):
    k = {"d": 0, "p": 1, "z": 2, "b": 3, "l": 4, "m": 5}
    return (k.get(r[0], 9), r)


def _table(hdr, lines, md):
    if md:
        print("| " + " | ".join(hdr) + " |")
        print("|" + "---|" * len(hdr))
        for c in lines:
            print("| " + " | ".join(c) + " |")
        return
    w = [max(len(str(x[i])) for x in [hdr] + lines) for i in range(len(hdr))]
    for c in [hdr] + lines:
        print("  ".join(str(x).rjust(w[i]) if i else str(x).ljust(w[i]) for i, x in enumerate(c)))


def parse_hz(f: Path) -> dict:
    d = {"u": {}, "dep": defaultdict(int), "blk": defaultdict(int), "k": {}}
    for line in f.read_text().splitlines():
        t = line.split()
        if not t:
            continue
        if t[0] == "HZ" and t[1] == "cycles":
            kv = dict(zip(t[1::2], t[2::2]))
            d.update({k: int(v) for k, v in kv.items()})
        elif t[0] == "HZU":
            u = int(t[1])
            d["u"][u] = {"busy": int(t[3]), "empty": int(t[5]), "start": int(t[7]),
                         "notrdy": int(t[9]), "blocked": int(t[11]),
                         "by": [int(x) for x in t[13:18]]}
        elif t[0] == "HZREL":
            d["rel"] = int(t[2])
            d["rel_by"] = [int(x) for x in t[4:9]]
        elif t[0] == "HZKIND":
            d["kind"] = dict(zip(t[1::2], map(int, t[2::2])))
        elif t[0] == "HZNU":
            d["nu"] = [int(x) for x in t[1:]]
        elif t[0] == "HZDEP":
            d["dep"][(int(t[1]), int(t[2]))] += int(t[3])
        elif t[0] == "HZBLK":
            d["blk"][(int(t[1]), int(t[2]))] += int(t[3])
        elif t[0] == "HZK":
            d["k"][(int(t[1]), int(t[2]))] = [int(x) for x in t[3:]]
    return d


def hazards(a) -> None:
    out = Path(a.outdir)
    for f in sorted(out.glob(f"*__{a.cand}.*.hz")):
        d = parse_hz(f)
        cyc = d.get("cycles", 0) or 1
        print(f"== {f.name}: {cyc:,} cycles, {d.get('dispatched', 0):,} instructions, "
              f"{100 * d.get('nodep', 0) / max(1, d.get('dispatched', 1)):.0f}% with no "
              f"dependency at dispatch; window full {100 * d.get('wfull', 0) / cyc:.1f}%, "
              f"RLD wait {100 * d.get('rlwait', 0) / cyc:.1f}%")
        for u, s in sorted(d["u"].items()):
            by = ", ".join(f"{UNITS[p]} {100 * n / cyc:.1f}%" for p, n in enumerate(s["by"]) if n)
            print(f"  {UNITS[u]:5s} busy {100 * s['busy'] / cyc:5.1f}%  idle, nothing to start "
                  f"{100 * s['empty'] / cyc:5.1f}%  starting {100 * s['start'] / cyc:4.1f}%  "
                  f"unit not ready {100 * s['notrdy'] / cyc:4.1f}%  blocked "
                  f"{100 * s['blocked'] / cyc:5.1f}% ({by})")
        if "rel" in d:
            by = ", ".join(f"{UNITS[p]} {100 * n / cyc:.1f}%" for p, n in enumerate(d["rel_by"]) if n)
            print(f"  MXU streaming, release waiting {100 * d['rel'] / cyc:.1f}% ({by})")
        if "kind" in d:
            print("  dependencies by kind:", ", ".join(f"{k} {v:,}" for k, v in d["kind"].items()))
        if "nu" in d:
            print("  instructions by units depended on (0..5):", d["nu"])
        top = sorted(d["blk"].items(), key=lambda x: -x[1])[:8]
        print("  blocked cycles by (waiting, producer):",
              "; ".join(f"{CLS[w]}<-{CLS[p]} {100 * n / cyc:.1f}%" for (w, p), n in top))
        top = sorted(d["dep"].items(), key=lambda x: -x[1])[:8]
        print("  dependencies by (class, producer):",
              "; ".join(f"{CLS[w]}<-{CLS[p]} {n:,}" for (w, p), n in top))
        for (u, p), h in sorted(d["k"].items()):
            n = sum(h[1:])
            if n:
                q = [sum(h[1:j + 1]) for j in (1, 2, 4, 8)]
                print(f"  ranks {UNITS[u]:5s}<-{UNITS[p]:5s}: {n:7,} with a conflict, k=1 "
                      f"{100 * q[0] / n:.0f}%, <=2 {100 * q[1] / n:.0f}%, <=4 {100 * q[2] / n:.0f}%,"
                      f" <=8 {100 * q[3] / n:.0f}%, none {h[0]:,}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    c = sp.add_parser("cycles")
    c.add_argument("outdir")
    c.add_argument("--ref", default="i0")
    c.add_argument("--cands")
    c.add_argument("--markdown", action="store_true")
    h = sp.add_parser("hazards")
    h.add_argument("outdir")
    h.add_argument("--cand", default="i0")
    a = ap.parse_args(argv)
    {"cycles": cycles, "hazards": hazards}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
