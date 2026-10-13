#!/usr/bin/env python3
"""MoE MTP bound (docs/offload.md 15.6): the 35B's decode with a k = 1 MTP loop on path (a), in
lesim's card timeline (tools/offload/lesim.py; session 16's fit), from the router traces.

A step is a verify run of 2 rows (the token, the draft) through every layer: per layer one
request with both rows' routes (their union is what the card computes and the server streams),
then the MTP layer's draft. Acceptance is a Bernoulli(a) per step (seeded): accepted, the step
emits 2 tokens and the next step starts 2 tokens on; rejected, 1 token and 1 on (the rejected
row's routes are the next token's, the trace having no others). Card time per request:
    W0_2(u) = W0 + DMIX + (u - k) X     (no miss: u distinct experts at X ms each beyond k)
    F_2     = F + DMIX                  (after the last miss lands: the rest, at 2 rows)
a step: the layers' requests, then C0 + DHEAD (the head on 2 rows), then the draft: CDRAFT ms
(the MTP layer on 2 rows, its own misses, the 32K fp4 head). Plain decode (1 row, a = 0, no
draft) is lesim's base. Prints ms a token and the speedup per a.

The defaults are 15.6's: DMIX the co-simulated second row of a layer (moe_prefill_cosim.py layer
at R = 1 and 2 on r2's configuration: DeltaNet 2.579 -> 2.974 ms, attention 2.053 -> 2.328 ms;
30 and 10 of them), X one expert's FFN on 2 rows (moe_prefill_cosim.py expert: 13,669 cycles),
CDRAFT the MTP layer on 2 rows (the attention layer's 2.328 ms, 4.9 union experts more, fc 0.26,
the 32K fp4 head 2.13 ms, ~4 misses of its own at ~0.94 ms). --slots 40: the MTP layer's slots
taken from the 40 others.

    python3 tools/offload/mtp_bound.py TRACE... [--dmix 0.365] [--x 0.1025] [--dhead 1.0]
        [--cdraft 9.1] [--accept 0.75,0.78,0.8,0.85,1.0] [--slots 41]
"""
from __future__ import annotations

import argparse
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lesim as LS  # noqa: E402
from cachesim import decayed, load, victim  # noqa: E402


def run2(req, slots, warm, E, p, a, dmix, X, dhead, cdraft, seed=0, half=32.0, rows=2):
    """ms a token, misses a token, tokens a step, distinct experts a request."""
    T, L, k = req.shape
    rng = np.random.default_rng(seed)
    cs = [OrderedDict() for _ in range(L)]
    for e in warm:
        if len(cs[e // E]) < slots:
            cs[e // E][e] = None
    uses = [dict.fromkeys(c, 0.0) for c in cs]
    cs = [OrderedDict.fromkeys(reversed(c)) for c in cs]
    t = link = 0.0
    tok = toks = steps = miss = dist = 0
    while tok + rows <= T:
        steps += 1
        for j in range(L):
            post = t
            seen = post + p["DET"]
            ids = [int(e) for r in range(rows) for e in req[tok + r, j]]
            u = len(set(ids))
            dist += u
            decayed(uses[j], ids, steps / half)
            c = cs[j]
            plain = 0
            for e in dict.fromkeys(ids):
                if e in c:
                    c.move_to_end(e)
                    continue
                plain += 1
                if len(c) >= max(slots, u):
                    del c[victim(c, ids, uses[j])]
                c[e] = None
            miss += plain
            w0 = p["W0"] + (dmix + (u - k) * X if rows > 1 else 0.0)
            f = p["F"] + (dmix if rows > 1 else 0.0)
            nxt = post + w0 + p["WM"] * p["CT"] * p["MB"] * plain
            if plain:
                crit = max(seen, link) + p["HEAD"] + p["REC"] * plain
                link = crit
                nxt = max(nxt, crit + f)
            t = nxt
        t += p["C0"] + (dhead + cdraft if rows > 1 else 0.0)
        acc = rows > 1 and rng.random() < a
        n = 2 if acc else 1
        tok += n
        toks += n
    return t / toks, miss / toks, toks / steps, dist / (steps * L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--dmix", type=float, default=0.365, help="ms a layer, the second row")
    ap.add_argument("--x", type=float, default=0.1025, help="ms an expert on 2 rows")
    ap.add_argument("--dhead", type=float, default=1.0)
    ap.add_argument("--cdraft", type=float, default=9.1, help="ms a step, the MTP draft")
    ap.add_argument("--accept", default="0.75,0.78,0.8,0.85,1.0")
    ap.add_argument("--slots", type=int, default=41, help="slots a layer of the MTP image")
    ap.add_argument("--slots-plain", type=int, default=None,
                    help="slots a layer of plain decode's image (default --slots)")
    a = ap.parse_args()
    sp = a.slots if a.slots_plain is None else a.slots_plain
    tr = [load(x) for x in a.traces]
    p = dict(LS.MODELS["q35"])
    E = tr[0]["E"]
    base = []
    for i, t in enumerate(tr):
        warm = LS.warm_set(tr, i, sp * t["req"].shape[1])
        base.append(run2(t["req"], sp, warm, E, p, 0.0, 0, 0, 0, 0, rows=1))
    b = np.mean(base, 0)
    print(f"plain decode ({sp} slots a layer): {b[0]:.1f} ms a token ({1000 / b[0]:.2f} tok/s), "
          f"{b[1]:.1f} misses a token; MTP at {a.slots} slots: DMIX {a.dmix} X {a.x:.4f} DHEAD "
          f"{a.dhead} CDRAFT {a.cdraft}")
    print(f"{'a':>5s} {'ms/token':>9s} {'tok/s':>6s} {'speedup':>7s} {'misses/token':>12s} "
          f"{'tokens/step':>11s} {'experts/request':>15s} {'c_2+c_draft':>11s}")
    for acc in [float(x) for x in a.accept.split(",")]:
        r = []
        for i, t in enumerate(tr):
            warm = LS.warm_set(tr, i, a.slots * t["req"].shape[1])
            r.append(run2(t["req"], a.slots, warm, E, p, acc, a.dmix, a.x, a.dhead, a.cdraft,
                          seed=i))
        m = np.mean(r, 0)
        c = m[0] * m[2] / b[0]                  # a step's cost in plain decode steps
        print(f"{acc:5.2f} {m[0]:9.1f} {1000 / m[0]:6.2f} {b[0] / m[0]:7.3f}x {m[1]:12.1f} "
              f"{m[2]:11.2f} {m[3]:15.2f} {c:11.3f}")


if __name__ == "__main__":
    main()
