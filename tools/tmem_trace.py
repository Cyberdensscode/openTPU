#!/usr/bin/env python3
"""EXPERIMENTAL (docs/tmem_nocopy.md): what the units read from TMEM, cycle by cycle, from a
co-simulation's trace (otpu_slice +tmtrace; rtl/exp/tmem_nocopy/run_cosim.sh TRACE=1).

    python3 tools/tmem_trace.py stats  FILE.tt [...]    activity, overlap, streams, reuse
    python3 tools/tmem_trace.py screen FILE.tt [...]    one-copy organizations, no feedback
    python3 tools/tmem_trace.py json   FILE.tt [...]    both, as one JSON line per file

A trace holds one or more simulations (a record of kind 15 starts each). Records are three
little-endian 32-bit words {cycle, info, value}; info[3:0] the kind, [7:4] the port or unit,
[15:8] the lanes (otpu_slice): 0 a read (value lane 0's address; lane l reads value + l), 1 a
write, 2 / 3 one lane of the MXU's read / write ports, 4 an instruction starts ([15:8] flags,
[23:16] VOP func, [31:24] opcode), 5 one completes, 6 groups that asked and were not granted.

`screen` replays the card's accesses (every read in the cycle it happened) against a single
copy organized as banks (word address mod 8) of `nsb` RAMs each, and counts the cycles some RAM
would be asked for more distinct words than it reads per cycle (`rps`), and the extra cycles
that serializing them would take (each such cycle needs max ceil(words / rps) cycles): a lower
bound without the stalls' knock-on effects, for ranking organizations before co-simulating
them (otpu_tmem_model).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter

import numpy as np

PORTS = ["MXU", "DMA", "Q", "VA", "VB", "Q3", "Q2", "COLL"]
UNIT_OF_PORT = [1, 0, 2, 3, 3, 2, 2, 4]          # U_DMA 0, U_MXU 1, U_Q 2, U_VPU 3, U_COLL 4
UNITS = ["DMA", "MXU", "Q", "VPU", "COLL"]
POP = np.array([bin(m).count("1") for m in range(256)], dtype=np.int64)        # lanes in a mask
LOW = np.array([(m & -m).bit_length() - 1 if m else 0 for m in range(256)], dtype=np.int64)
OPS = {0x06: "RLD", 0x07: "WAITW", 0x10: "LD", 0x11: "ST", 0x12: "DSTEP", 0x13: "STREAM",
       0x20: "MM", 0x21: "QACT", 0x22: "QST", 0x30: "VOP", 0x40: "GATHER", 0x05: "BAR"}
VFUNC = {0: "ADD", 1: "SUB", 2: "RSUB", 3: "MUL", 4: "MAX", 5: "MIN", 6: "OUTER", 8: "COPY",
         9: "EXP2", 10: "RECIP", 11: "RSQRT", 12: "ABS", 13: "FILL", 14: "EXP2SUB", 15: "LOG2",
         16: "RSUM", 17: "RMAX", 18: "RSSQ", 19: "RDOT", 20: "ARGMAX"}


def segments(path: str) -> list[np.ndarray]:
    """The trace's simulations, each an (n, 3) uint32 array."""
    a = np.fromfile(path, dtype="<u4")
    a = a[: len(a) // 3 * 3].reshape(-1, 3)
    cut = np.nonzero((a[:, 1] & 15) == 15)[0]
    bounds = list(cut) + [len(a)]
    if not len(cut) or cut[0] != 0:
        bounds = [0] + bounds
    return [a[s + (1 if (a[s, 1] & 15) == 15 else 0): e] for s, e in zip(bounds[:-1], bounds[1:])
            if e - s > 1]


def lanes(seg: np.ndarray, kinds: tuple[int, int]):
    """Every lane access of the given kinds (rotator kind, scattered kind): cycle, port, address."""
    k = seg[:, 1] & 15
    rot = seg[k == kinds[0]]
    gen = seg[k == kinds[1]]
    cyc, port, addr = [], [], []
    m = (rot[:, 1] >> 8) & 255
    for l in range(8):
        on = ((m >> l) & 1).astype(bool)
        cyc.append(rot[on, 0]); port.append((rot[on, 1] >> 4) & 15)
        addr.append(rot[on, 2].astype(np.int64) + l)
    cyc.append(gen[:, 0]); port.append((gen[:, 1] >> 4) & 15); addr.append(gen[:, 2].astype(np.int64))
    c = np.concatenate(cyc).astype(np.int64)
    o = np.argsort(c, kind="stable")
    return c[o], np.concatenate(port).astype(np.int64)[o], np.concatenate(addr)[o]


def instructions(seg: np.ndarray):
    """Per unit: [(start, done, op name, icount)] in start order (a unit completes in order)."""
    k = seg[:, 1] & 15
    out = {}
    for u in range(5):
        st = seg[(k == 4) & (((seg[:, 1] >> 4) & 15) == u)]
        dn = seg[(k == 5) & (((seg[:, 1] >> 4) & 15) == u)]
        n = min(len(st), len(dn))
        names = []
        for r in st[:n]:
            op = int(r[1] >> 24)
            nm = OPS.get(op, f"op{op:02x}")
            if nm == "VOP":
                nm = "VOP." + VFUNC.get(int((r[1] >> 16) & 255), "?")
            names.append(nm)
        out[u] = (st[:n, 0].astype(np.int64), dn[:n, 0].astype(np.int64), names)
    return out


def stats(seg: np.ndarray) -> dict:
    c, p, a = lanes(seg, (0, 2))
    k = seg[:, 1] & 15
    span = int(seg[:, 0].max() - seg[:, 0].min() + 1) if len(seg) else 0
    res = {"cycles": span}
    # per port: cycles it read, words
    pc = np.unique(c * 8 + p)
    res["port_cycles"] = {PORTS[i]: int((pc % 8 == i).sum()) for i in range(8)}
    res["port_words"] = {PORTS[i]: int((p == i).sum()) for i in range(8)}
    # streams per cycle (ports reading)
    cyc_of = pc // 8
    _, nper = np.unique(cyc_of, return_counts=True)
    h = np.bincount(nper, minlength=9)
    h[0] = span - len(nper)
    res["streams_hist"] = [int(x) for x in h]
    # units reading per cycle and the commonest unit sets
    uc = np.unique(c * 8 + np.array(UNIT_OF_PORT)[p])
    cu = uc // 8
    bits = np.zeros(int(cu.max()) + 1 if len(cu) else 1, dtype=np.int64)
    np.add.at(bits, cu, 1 << (uc % 8))
    sets = Counter(bits[bits != 0].tolist())
    res["unit_sets"] = {"+".join(UNITS[u] for u in range(5) if (s >> u) & 1): n
                        for s, n in sets.most_common(12)}
    # the instruction each read belongs to: its unit's latest start at or before it
    ins = instructions(seg)
    byop = Counter()
    for i in range(8):
        sel = p == i
        st, dn, names = ins[UNIT_OF_PORT[i]]
        if not len(st):
            continue
        idx = np.searchsorted(st, c[sel], side="right") - 1
        ok = idx >= 0
        for j, n in zip(*np.unique(idx[ok], return_counts=True)):
            byop[(PORTS[i], names[j])] += int(n)
    res["words_by_port_op"] = {f"{q}:{o}": n for (q, o), n in byop.most_common(40)}
    # streaming and reuse per port (rotator ports: a0 per access)
    rd = seg[k == 0]
    seq = {}
    for i in range(1, 8):
        r = rd[((rd[:, 1] >> 4) & 15) == i]
        if len(r) < 2:
            continue
        a0 = r[:, 2].astype(np.int64)
        mk = (r[:, 1] >> 8) & 255
        nl = POP[mk]
        lo = a0 + LOW[mk]
        hi = lo + nl - 1
        cont = lo[1:] == hi[:-1] + 1
        same = (lo[1:] == lo[:-1]) & (hi[1:] == hi[:-1])
        l0, l1 = lo >> 3, hi >> 3
        # the previous two accesses' lines (a 2-line buffer, approximately LRU)
        prev = [np.roll(l0, 1), np.roll(l1, 1), np.roll(l0, 2), np.roll(l1, 2)]
        def kept(x, depth):
            h = np.zeros(len(x), dtype=bool)
            for q in prev[: 2 * depth]:
                h |= x == q
            return h
        hit1 = kept(l0, 1) & kept(l1, 1)
        hit2 = kept(l0, 2) & kept(l1, 2)
        hit1[:1] = False; hit2[:2] = False
        seq[PORTS[i]] = {"accesses": int(len(r)), "contiguous": float(cont.mean()),
                         "same": float(same.mean()), "hit1": float(hit1.mean()),
                         "hit2": float(hit2.mean())}
    res["streams"] = seq
    # cycles some group was not granted (the card's write-bank arbitration and guest rules)
    lg = seg[k == 6]
    res["lose"] = {g: int((((lg[:, 1] >> 8) >> j) & 1).sum())
                   for j, g in enumerate(["DMA", "COLL", "VPU", "MXU", "Q"])}
    return res


def screen(seg: np.ndarray, cands: list[tuple]) -> dict:
    """Per candidate (name, nsb, sbp, hash, rps, free ports): cycles with a RAM over its reads,
    and the extra cycles serializing them takes. rps "t": a true dual-port RAM, two accesses a
    cycle, reads (distinct words) and writes together."""
    c, p, a = lanes(seg, (0, 2))
    wc, _, wa = lanes(seg, (1, 3))
    out = {}
    for name, nsb, sbp, hsh, rps, free in cands:
        keep = ~np.isin(p, [PORTS.index(f) for f in free]) if free else np.ones(len(p), bool)
        cc, aa = c[keep], a[keep]
        lsb = int(nsb).bit_length() - 1
        def ramof(cy, ad):
            sb = (ad >> sbp)
            if hsh == 1:
                sb = sb ^ (ad >> (sbp + lsb))
            elif hsh == 2 and lsb:
                for f in range(sbp + lsb, 20, lsb):
                    sb = sb ^ (ad >> f)
            sb &= nsb - 1
            return (cy << 9) | ((ad & 7) << 5) | sb
        ram = ramof(cc, aa)
        word = aa >> 3
        if rps == "t":           # writes take a port each (a distinct tag), reads share words
            ram = np.concatenate([ram, ramof(wc, wa)])
            word = np.concatenate([word, (wa >> 3) | (1 << 19)])
            rps = 2
        u = np.unique((ram << 20) | word)
        ram_u, n = np.unique(u >> 20, return_counts=True)
        need = (n + rps - 1) // rps
        cyc = ram_u >> 9
        over = need > 1
        if over.any():
            cy, idx = np.unique(cyc[over], return_index=True)
            mx = np.maximum.reduceat(need[over], idx)
            out[name] = {"cycles_over": int(len(cy)), "extra": int((mx - 1).sum())}
        else:
            out[name] = {"cycles_over": 0, "extra": 0}
    return out


CANDS = [  # name, sub-banks, field bit, hash, reads per RAM, ports off the RAM
    ("c1", 1, 3, 0, 1, ()), ("c1r2", 1, 3, 0, 2, ()), ("c1r2q", 1, 3, 0, 2, ("Q3",)),
    ("c1r3", 1, 3, 0, 3, ()),
    ("s2i", 2, 3, 0, 1, ()), ("s4i", 4, 3, 0, 1, ()), ("s8i", 8, 3, 0, 1, ()),
    ("s16i", 16, 3, 0, 1, ()), ("s32i", 32, 3, 0, 1, ()),
    ("s8i6", 8, 6, 0, 1, ()), ("s8i9", 8, 9, 0, 1, ()),
    ("s8r", 8, 13, 0, 1, ()), ("s16r", 16, 12, 0, 1, ()),
    ("s8h", 8, 3, 1, 1, ()), ("s16h", 16, 3, 1, 1, ()),
    ("s8i2", 8, 3, 0, 2, ()), ("s4i2", 4, 3, 0, 2, ()), ("s2i2", 2, 3, 0, 2, ()),
    ("s8iq", 8, 3, 0, 1, ("Q3",)),
    ("c1t", 1, 3, 0, "t", ()), ("s8t", 8, 3, 0, "t", ()), ("s16t", 16, 3, 0, "t", ()),
    ("s8h2", 8, 3, 2, 1, ()), ("s16h2", 16, 3, 2, 1, ()), ("s8h2t", 8, 3, 2, "t", ()),
    ("s16h2t", 16, 3, 2, "t", ()),
]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["stats", "screen", "json"])
    ap.add_argument("files", nargs="+")
    a = ap.parse_args(argv)
    for f in a.files:
        segs = segments(f)
        for s_i, seg in enumerate(segs):
            res = {"file": f, "sim": s_i}
            if a.cmd in ("stats", "json"):
                res["stats"] = stats(seg)
            if a.cmd in ("screen", "json"):
                res["screen"] = screen(seg, CANDS)
            if a.cmd == "json":
                print(json.dumps(res))
            else:
                print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
