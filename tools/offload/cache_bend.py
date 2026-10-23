"""cachesim.simulate's replays in Bend (bend/cache/: cache.bend, main.bend; docs/bend.md): the
policies static, lru, lru_layer, lfu and opt, miss for miss the Python's, a sweep's replays of
one trace at once on the cores.

    from cache_bend import sweep
    res = sweep(req, [(C, policy, warm, prior), ...])    # [(misses a token, peak), ...]

The binary (bend/cache/main.bend built with bend -o) is rebuilt when a source is newer;
OTPU_BEND_THREADS sets its threads (default: the runtime's choice).
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "bend" / "cache"
BIN = ROOT / "build" / "cachesim_bend"
POLICIES = {"static": 0, "lru": 1, "lru_layer": 2, "lfu": 3, "opt": 4}
MAX_WORDS = 1 << 23                     # main.bend's Host.load depth


def build(force: bool = False) -> Path:
    srcs = [SRC / "cache.bend", SRC / "main.bend", ROOT / "bend" / "host.bend",
            ROOT / "bend" / "host.c"]
    if not force and BIN.exists() and BIN.stat().st_mtime >= max(p.stat().st_mtime for p in srcs):
        return BIN
    BIN.parent.mkdir(parents=True, exist_ok=True)
    bend = os.environ.get("OTPU_BEND", str(Path.home() / ".bend" / "bin" / "bend"))
    subprocess.run([bend, str(SRC / "main.bend"), "-o", str(BIN)], check=True,
                   env=dict(os.environ, BEND_NO_TELEMETRY="1"))
    return BIN


def sweep(req, jobs, threads: int | None = None) -> list[tuple[np.ndarray, int]]:
    """Each job (C, policy, warm, prior) replayed as cachesim.simulate(req, C, policy, warm,
    prior) would: its misses per token and the most experts its cache held after a request."""
    req = np.asarray(req)
    T, L, k = req.shape
    n = int(req.max()) + 1
    for C, pol, warm, prior in jobs:
        if pol not in POLICIES:
            raise ValueError(f"{pol}: not in Bend (static, lru, lru_layer, lfu, opt)")
        n = max(n, max((int(e) + 1 for e in warm), default=0),
                len(prior) if prior is not None else 0)
    words = [np.zeros(16 + 8 * len(jobs), np.uint32), req.reshape(-1).astype(np.uint32)]
    at = sum(w.size for w in words)
    lists: dict = {}                    # one copy of each warm list and count vector

    def place(key, arr):
        nonlocal at
        if key not in lists:
            a = np.zeros(n, np.uint32)
            a[:len(arr)] = arr
            lists[key] = at
            words.append(a)
            at += a.size
        return lists[key]

    zeros = None
    for i, (C, pol, warm, prior) in enumerate(jobs):
        warm = [int(e) for e in warm]
        so = place(("w", tuple(warm)), np.array(warm, np.uint32))
        if prior is None:
            zeros = place(("p", None), np.zeros(n, np.uint32))
            sp, E = zeros, int(req.max()) // L + 1
        else:
            pr = np.asarray(prior)
            sp, E = place(("p", pr.tobytes()), pr.astype(np.uint32)), len(pr) // L
        words[0][16 + 8 * i:16 + 8 * i + 6] = (POLICIES[pol], C, len(warm), so, sp, E)
    words[0][:7] = (T, L, k, n, len(jobs), 16 + 8 * len(jobs), at)
    if at > MAX_WORDS:
        raise ValueError(f"{at} words: more than main.bend loads ({MAX_WORDS})")
    exe = build()
    threads = threads or int(os.environ.get("OTPU_BEND_THREADS", "0")) or None
    argv = [str(exe)] + (["+RTS", "--threads", str(int(threads))] if threads else [])
    with tempfile.TemporaryDirectory(prefix="cache_bend") as d:
        np.concatenate(words).tofile(Path(d) / "cache.in")
        subprocess.run(argv, cwd=d, check=True, stdout=subprocess.DEVNULL)
        out = np.fromfile(Path(d) / "cache.out", np.uint32).reshape(len(jobs), T + 1)
    return [(o[1:].astype(np.int64), int(o[0])) for o in out]
