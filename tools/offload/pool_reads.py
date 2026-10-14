"""How fast the expert pool's disk reads experts the page cache does not hold (docs/minicpmv.md 6,
pooldisk: the card host's SSD at 13-14.5 ms an expert, against docs/offload.md 10.6's 4.2 ms).
Host only and read only: no card, no engine, the pool file never written.

    python3 tools/offload/pool_reads.py POOL [--n 200] [--queue 8] [--seq-mb 512] [--seed 0]
        [--out JSON]

2 x n experts at random (of the file's entries: its .packed's length, else --slot) are dropped
from the page cache (posix_fadvise DONTNEED over each one's bytes; a page some process maps
stays, so an expert still partly in the page cache after it is left out) and read back:
- one at a time: a preadv of the expert's bytes each (the first n);
- queued: `queue` at a time, each group's reads queued at once (POSIX_FADV_WILLNEED, as
  PoolFile.willneed) then read in order (the next n);
- in order: `seq-mb` contiguous MB from a random entry, dropped, then read in 16 MB reads
  (pool_disk.cache's).
Each: ms an expert and MB/s. Also the file's extents (filefrag -v: how many, their mean size,
how many compressed), its mount (/proc/mounts) and the block devices under it (rotational,
scheduler, read_ahead_kb). The page cache is left as it was found: an expert or a 16 MB read
that was in it before is read in again at the end.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import mmap
import os
import re
import subprocess
import time
from pathlib import Path

import numpy as np

PG = mmap.PAGESIZE
SEQ = 16 << 20


def pages_in(fd: int, size: int) -> np.ndarray | None:
    """The file's pages in the page cache (mincore: 1 a page), or None where the host cannot
    tell."""
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
        mm = mmap.mmap(fd, size, prot=mmap.PROT_READ)
    except (OSError, AttributeError, ValueError):
        return None
    vec = (ctypes.c_ubyte * -(-size // PG))()
    v = np.frombuffer(mm, np.uint8)
    try:
        rc = libc.mincore(ctypes.c_void_p(v.ctypes.data), ctypes.c_size_t(size), vec)
    finally:
        del v
        mm.close()
    return None if rc != 0 else np.frombuffer(vec, np.uint8) & 1


def held(vec, off: int, n: int) -> float | None:
    """The share of [off, off + n)'s pages in the page cache."""
    return None if vec is None else float(vec[off // PG:-(-(off + n) // PG)].mean())


def drop(fd: int, off: int, n: int) -> None:
    os.posix_fadvise(fd, off, n, os.POSIX_FADV_DONTNEED)


def extents(path) -> dict:
    """filefrag -v's extents of the file: how many, their mean KB, how many 'encoded'
    (compressed), or {} where there is no filefrag."""
    try:
        p = subprocess.run(["filefrag", "-v", str(path)], capture_output=True, text=True,
                           timeout=600)
    except (OSError, subprocess.TimeoutExpired):
        return {}
    n = enc = blocks = 0
    for ln in p.stdout.splitlines():
        f = ln.split(":")
        if len(f) >= 4 and f[0].strip().isdigit():
            n += 1
            try:
                blocks += int(f[3].split()[0])
            except (ValueError, IndexError):
                pass
            enc += "encoded" in ln
    m = re.search(r"blocks of (\d+) bytes", p.stdout)
    bs = int(m.group(1)) if m else 4096
    return {"extents": n, "mean_kb": round(blocks * bs / max(n, 1) / 1024, 1), "encoded": enc}


def devices(path) -> dict:
    """The mount under path (device, type, options) and the block devices under it (a
    dm-crypt's slaves): rotational, scheduler, read_ahead_kb, model."""
    real, best = os.path.realpath(path), None
    for ln in Path("/proc/mounts").read_text().splitlines() if Path("/proc/mounts").exists() \
            else []:
        src, mnt, typ, opts = ln.split()[:4]
        if (real == mnt or real.startswith(mnt.rstrip("/") + "/")) and \
                (best is None or len(mnt) > len(best[1])):
            best = (src, mnt, typ, opts)
    if best is None:
        return {}
    out = {"mount": dict(zip(("source", "at", "type", "options"), best)), "blocks": []}

    def walk(name):
        b = Path("/sys/class/block") / name
        if not b.exists():
            return
        q = b / "queue" if (b / "queue").exists() else b.resolve().parent / "queue"
        rd = lambda p: p.read_text().strip() if p.exists() else None      # noqa: E731
        out["blocks"].append({"name": name, "rotational": rd(q / "rotational"),
                              "scheduler": rd(q / "scheduler"),
                              "read_ahead_kb": rd(q / "read_ahead_kb"),
                              "model": rd(b.resolve().parent / "device" / "model")
                              or rd(b / "device" / "model")})
        for s in sorted((b / "slaves").iterdir()) if (b / "slaves").exists() else []:
            walk(s.name)
    walk(os.path.basename(os.path.realpath(best[0])))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("pool")
    ap.add_argument("--slot", type=int, help="an expert's bytes (else the file's size over its "
                                             ".packed's entries)")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--queue", type=int, default=8)
    ap.add_argument("--seq-mb", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-filefrag", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    if not hasattr(os, "posix_fadvise"):
        raise SystemExit("posix_fadvise: Linux only")
    size = os.path.getsize(a.pool)
    pk = Path(a.pool + ".packed")
    slot = a.slot or size // pk.stat().st_size
    E = size // slot
    rng = np.random.default_rng(a.seed)
    ids = rng.choice(E, size=min(2 * a.n, E), replace=False)
    one, queued = ids[:len(ids) // 2], ids[len(ids) // 2:]
    nseq = min(a.seq_mb << 20, size) // SEQ * SEQ
    s0 = int(rng.integers(0, max(1, (size - nseq) // slot))) * slot
    out = {"pool": a.pool, "bytes": size, "slot": slot, "entries": E,
           "disk": devices(a.pool), "file": {} if a.no_filefrag else extents(a.pool)}
    print(f"{a.pool}: {size / 1e9:.2f} GB, {E} entries of {slot} bytes; {out['file']}",
          flush=True)
    print(f"  {out['disk']}", flush=True)
    fd = os.open(a.pool, os.O_RDONLY)
    buf = memoryview(bytearray(max(slot, SEQ)))
    try:
        before = pages_in(fd, size)
        out["cached_before"] = None if before is None else round(float(before.mean()), 4)
        was = {int(g): held(before, int(g) * slot, slot) for g in ids}
        was_seq = held(before, s0, nseq)
        for g in ids:
            drop(fd, int(g) * slot, slot)
        drop(fd, s0, nseq)
        after = pages_in(fd, size)
        gone = lambda g: after is None or held(after, g * slot, slot) == 0     # noqa: E731
        res = {}
        for name, grp, q in (("one at a time", one, 1), ("queued", queued, a.queue)):
            grp = [int(g) for g in grp if gone(int(g))]
            ts = []
            for i in range(0, len(grp), q):
                part = grp[i:i + q]
                t = time.perf_counter()
                if q > 1:
                    for g in part:
                        os.posix_fadvise(fd, g * slot, slot, os.POSIX_FADV_WILLNEED)
                for g in part:
                    os.preadv(fd, [buf[:slot]], g * slot)
                ts.append((time.perf_counter() - t) / len(part))
            if ts:
                ms = np.array(ts) * 1e3
                res[name] = {"experts": len(grp), "ms": round(float(ms.mean()), 2),
                             "ms_p50": round(float(np.median(ms)), 2),
                             "ms_p90": round(float(np.percentile(ms, 90)), 2),
                             "mb_s": round(slot / 1e6 / (ms.mean() / 1e3), 1)}
                print(f"  {name}: {len(grp)} experts, {ms.mean():.2f} ms each (p50 "
                      f"{np.median(ms):.2f}, p90 {np.percentile(ms, 90):.2f}), "
                      f"{res[name]['mb_s']} MB/s", flush=True)
        if after is None or held(after, s0, nseq) == 0:
            t = time.perf_counter()
            for off in range(s0, s0 + nseq, SEQ):
                os.preadv(fd, [buf[:SEQ]], off)
            s = time.perf_counter() - t
            res["in order"] = {"mb": nseq >> 20, "s": round(s, 2),
                               "mb_s": round(nseq / 1e6 / s, 1)}
            print(f"  in order: {nseq >> 20} MB in {s:.2f} s, {nseq / 1e6 / s:.1f} MB/s",
                  flush=True)
        out["dropped"] = {"experts": int(sum(gone(int(g)) for g in ids)), "of": len(ids)}
        out["reads"] = res
        # the page cache as it was: what was in it before, read in again
        for g, h in was.items():
            if h:
                os.preadv(fd, [buf[:slot]], g * slot)
        if was_seq:
            for off in range(s0, s0 + nseq, SEQ):
                os.preadv(fd, [buf[:SEQ]], off)
    finally:
        os.close(fd)
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
