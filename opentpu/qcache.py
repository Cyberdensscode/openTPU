"""A disk cache of the 4-bit weight quantizer's results, so a model's DRAM image builds in a
minute once its matrices have been quantized (a 4-bit Qwen3.5-4B takes ~20 minutes of numpy on
the card host, and every card tool built it again under the card lock).

`quantize_mxu` is `quant.quantize_mxu` through the cache. The images' builds (qwen3, lfm2,
qwen35) quantize every weight matrix with it, so any build of a model and format fills the
cache for every other: perf.py, refs.py card (per-position or resident) and decode_profile
build different layouts (lookup tables, rows, KV capacity) from the same matrices.
`opentpu/host/prebuild.py` builds them ahead, outside otpu-lock (gemma4's build too: its
quantization jobs' matrices go through quantize_mxu in its worker processes).

The key is the matrix's content: a hash of its fp32 bytes and shape, the format, D, the
quantizer's parameters (the scale search), the source of quant.py and the numpy version. Any
change to the quantizer or the weights misses; the layout and the configuration do not matter.
int8 is not cached (it quantizes about as fast as the matrix hashes), nor matrices under
MIN_ELEMS elements (the tests' tiny models).

The cache: OTPU_IMAGE_CACHE=<dir>, or 0 for none; by default ~/otpu-build/qcache/mxu on a host
with ~/otpu-build (the build and card hosts), else none. Entries are .npz files written to a
temporary name, then renamed. The least recently used ones go when the cache passes CAP_BYTES
(OTPU_IMAGE_CACHE_GB), and nothing is written that would leave the disk under FREE_FLOOR free.
`stats` counts hits, misses, writes and writes skipped for disk space.

The quantizer runs on PIECE elements' rows at a time (quantize_rows): its float64 temporaries
then stay in a core's caches, twice as fast on one core and nearly linear in the cores, where
whole matrices left every core waiting on the memory bus (omarchy: 240 -> 110 ns an element on
one core; 16 processes 86 -> 17). Rows are quantized on their own, so the result is the same,
bit for bit. A build quantizes its matrices with a Batch: in build_jobs() worker processes
(OTPU_BUILD_JOBS, default the host's CPUs), the cache looked up and written in the build's
process, only the misses' rows sent to the workers (docs/board.md, "Image builds").
"""
from __future__ import annotations

import hashlib
import os
import shutil
import weakref
from collections import deque
from pathlib import Path

import numpy as np

from . import quant as Q

FORMATS = ("fp4", "int4")       # cached formats (int8 quantizes as fast as it hashes)
MIN_ELEMS = 1 << 20             # smaller matrices are quantized in line
CAP_BYTES = 30 << 30            # the cache's size (OTPU_IMAGE_CACHE_GB overrides)
FREE_FLOOR = 20 << 30           # free disk a write must leave
PIECE = 1 << 15                 # elements per quantizer call (quantize_rows)
JOB = 1 << 22                   # Batch: elements per worker job (a matrix's rows)
INLINE = 1 << 20                # Batch: smaller matrices are quantized in the build's process
AHEAD = 2                       # Batch: jobs queued per worker besides the one it runs
stats = {"hit": 0, "miss": 0, "write": 0, "skip": 0}
_SRC: str | None = None


def cache_dir() -> Path | None:
    """The cache directory, or None when the cache is off."""
    v = os.environ.get("OTPU_IMAGE_CACHE")
    if v == "0":
        return None
    if v:
        return Path(v)
    home = Path.home() / "otpu-build"
    return home / "qcache" / "mxu" if home.is_dir() else None


def _source() -> str:
    """What decides the quantizer's output besides its arguments: quant.py and numpy."""
    global _SRC
    if _SRC is None:
        _SRC = hashlib.blake2b(Path(Q.__file__).read_bytes() + np.__version__.encode(),
                               digest_size=16).hexdigest()
    return _SRC


def key(W, fmt: str, D: int) -> str:
    """The cache key of quantize_mxu(W, fmt, D)."""
    a = np.ascontiguousarray(W, np.float32)
    h = hashlib.blake2b(digest_size=20)
    h.update(repr((a.shape, fmt, D, "search", _source())).encode())
    h.update(memoryview(a.reshape(-1)).cast("B"))
    return h.hexdigest()


def quantize_mxu(W, fmt: str, D: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """quant.quantize_mxu(W, fmt, D), from the cache when it holds the result."""
    W = np.ascontiguousarray(W, np.float32)
    f = _entry(W, fmt, D)
    if f is None:
        return quantize_rows(W, fmt, D)
    hit = _load(f)
    if hit is not None:
        return hit
    q, s = quantize_rows(W, fmt, D)
    _store(f.parent.parent, f, q, s)
    return q, s


def quantize_rows(W, fmt: str, D: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """quant.quantize_mxu(W, fmt, D), bit for bit, its rows PIECE elements at a time."""
    W = np.asarray(W, np.float32)
    N, K = W.shape
    r = max(1, PIECE // K)
    if N <= r:
        return Q.quantize_mxu(W, fmt, D)
    q, s = None, None
    for i in range(0, N, r):
        a, b = Q.quantize_mxu(W[i:i + r], fmt, D)
        if q is None:
            q = np.empty((N,) + a.shape[1:], a.dtype)
            s = np.empty((N,) + b.shape[1:], b.dtype)
        q[i:i + r], s[i:i + r] = a, b
    return q, s


def _entry(W: np.ndarray, fmt: str, D: int) -> Path | None:
    """The cache file of a contiguous fp32 W's result (counted a miss until _load finds it), or
    None when it is not cached: the cache off, int8, a small matrix."""
    d = cache_dir()
    if d is None or fmt not in FORMATS or W.size < MIN_ELEMS:
        return None
    k = key(W, fmt, D)
    return d / k[:2] / f"{k}.npz"


def _load(f: Path):
    try:
        with np.load(f) as z:
            q, s = z["q"], z["s"]
        stats["hit"] += 1
        os.utime(f)                     # recently used (the eviction order)
        return q, s
    except FileNotFoundError:
        pass
    except Exception:                   # a damaged entry: quantize again
        f.unlink(missing_ok=True)
    stats["miss"] += 1
    return None


def build_jobs() -> int:
    """The worker processes an image build quantizes in: OTPU_BUILD_JOBS, else the host's CPUs
    (1: in the build's process)."""
    v = os.environ.get("OTPU_BUILD_JOBS")
    if v:
        return max(1, int(v))
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:              # (macOS)
        return os.cpu_count() or 1


def pool(n: int, **kw):
    """n spawned worker processes (they inherit no device or lock descriptor): Batch's, and
    gemma4's quantization jobs' (kw: ProcessPoolExecutor's initializer, initargs)."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    return ProcessPoolExecutor(n, mp_context=mp.get_context("spawn"), **kw)


class Batch:
    """An image build's matrices, quantized in `jobs` worker processes (default build_jobs())
    through the cache: submit(W, fmt, D, done) calls done(q, s) with quantize_mxu(W, fmt, D)'s
    result, bit for bit, in this process -- at once for a cache hit and a matrix under INLINE
    elements (quantized here, as everything with one job), else from a later submit() or wait(),
    once its rows have come back from the workers in jobs of JOB elements. Each matrix is
    hashed once, here, and only the misses' rows are sent; at most AHEAD jobs wait per worker,
    so a few matrices' fp32 rows are in flight. wait() returns once every done() has run, and
    stops the workers (as does dropping the batch)."""

    def __init__(self, jobs: int | None = None):
        self.n = build_jobs() if jobs is None else jobs
        self._pool = None
        self._queue = deque()           # (Future of a job's rows, its _Rows, first row)

    def submit(self, W, fmt: str, D: int, done) -> None:
        W = np.ascontiguousarray(W, np.float32)
        f = _entry(W, fmt, D)
        hit = None if f is None else _load(f)
        if hit is not None:
            done(*hit)
            return
        if self.n <= 1 or W.size < INLINE:
            q, s = quantize_rows(W, fmt, D)
            if f is not None:
                _store(f.parent.parent, f, q, s)
            done(q, s)
            return
        if self._pool is None:
            self._pool = pool(self.n)
            weakref.finalize(self, self._pool.shutdown, wait=False, cancel_futures=True)
        m = _Rows(len(W), done, f)
        r = max(1, JOB // W.shape[1])
        for i in range(0, len(W), r):
            self._drain(self.n * (AHEAD + 1) - 1)
            self._queue.append((self._pool.submit(quantize_rows, W[i:i + r], fmt, D), m, i))

    def _drain(self, keep: int) -> None:
        while len(self._queue) > keep:
            fut, m, i = self._queue.popleft()
            m.add(i, *fut.result())

    def wait(self) -> None:
        self._drain(0)
        if self._pool is not None:
            self._pool.shutdown()
            self._pool = None


class _Rows:
    """A matrix's quantized rows as its jobs come back (in order); done() and the cache entry
    once they all have."""

    def __init__(self, n: int, done, f: Path | None):
        self.n, self.done, self.f, self.got = n, done, f, 0
        self.q = self.s = None

    def add(self, i: int, q: np.ndarray, s: np.ndarray) -> None:
        if self.q is None:
            self.q = np.empty((self.n,) + q.shape[1:], q.dtype)
            self.s = np.empty((self.n,) + s.shape[1:], s.dtype)
        self.q[i:i + len(q)], self.s[i:i + len(s)] = q, s
        self.got += len(q)
        if self.got == self.n:
            if self.f is not None:
                _store(self.f.parent.parent, self.f, self.q, self.s)
            self.done(self.q, self.s)


def _store(d: Path, f: Path, q: np.ndarray, s: np.ndarray) -> None:
    f.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(f.parent).free - q.nbytes - s.nbytes < FREE_FLOOR:
        stats["skip"] += 1
        return
    tmp = f.with_name(f".{f.stem}.{os.getpid()}.npz")
    np.savez(tmp, q=q, s=s)
    os.replace(tmp, f)
    stats["write"] += 1
    _evict(d)


def _evict(d: Path) -> None:
    """Drop the least recently used entries while the cache is over its size (to 90% of it)."""
    cap = int(float(os.environ.get("OTPU_IMAGE_CACHE_GB", CAP_BYTES / 2**30)) * 2**30)
    ents = []
    for p in d.glob("*/*.npz"):
        if p.name.startswith("."):
            continue
        try:
            st = p.stat()
        except FileNotFoundError:
            continue
        ents.append((st.st_mtime, st.st_size, p))
    total = sum(e[1] for e in ents)
    if total <= cap:
        return
    for _, size, p in sorted(ents):
        if total <= 0.9 * cap:
            break
        p.unlink(missing_ok=True)
        total -= size
