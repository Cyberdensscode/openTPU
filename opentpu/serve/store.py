"""The server's KV store on the host: saved contexts (opentpu/serve/kvio.py's pages and state)
in RAM, spilled to files on disk, and the prefix index that finds the longest saved prefix of a
prompt (RadixAttention's role, at the granularity of a context's pages).

BlobStore holds byte blobs by key: the most recently used in RAM up to `ram_bytes`; past that
the least recently used go to files under `dir` (a writer thread writes them; a blob being
written is still read from memory) up to `disk_bytes`, past which the least recently used are
dropped. A blob read from disk comes back into RAM (its file stays). Pinned blobs (an active
request's context) are never evicted. The files outlive the process: a server started on the
same directory finds the contexts saved before (their entries are blobs too).

PrefixCache keeps contexts as entries: the tokens [0, n), the keys of their whole pages (page i
keyed by the model and the tokens [0, (i + 1) P): contexts with a common prefix share them) and
of the tail (the last partial page and, for a model with state, the state at n). match(tokens)
finds where the device can continue a prompt from: for a model whose KV caches hold the whole
context, any common prefix k of a saved context (its pages up to k; what lies past k is
overwritten); for a model with state, the end of a saved context that the prompt continues.
"""
from __future__ import annotations

import hashlib
import json
import os
import queue
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


class BlobStore:
    def __init__(self, ram_bytes: int, disk_bytes: int = 0, dir=None, on_drop=None):
        self.ram_bytes, self.disk_bytes = int(ram_bytes), int(disk_bytes)
        self.dir = Path(dir) if dir is not None and disk_bytes > 0 else None
        self.on_drop = on_drop
        self.lock = threading.RLock()
        self.ram: OrderedDict = OrderedDict()       # key -> bytes (least recently used first)
        self.spilling: dict = {}                    # key -> bytes being written to disk
        self.disk: OrderedDict = OrderedDict()      # key -> size of its file
        self.pins: dict = {}                        # key -> pin count
        self.ram_used = self.disk_used = self.spill_used = 0
        self.stats = {"put": 0, "ram_hits": 0, "disk_hits": 0, "misses": 0, "spilled": 0,
                      "dropped": 0, "disk_read_s": 0.0}
        self._q: queue.Queue | None = None
        if self.dir is not None:
            self.dir.mkdir(parents=True, exist_ok=True)
            files = sorted(self.dir.glob("*.bin"), key=lambda p: p.stat().st_mtime)
            for p in files:                         # the files of an earlier server
                n = p.stat().st_size
                self.disk[p.stem] = n
                self.disk_used += n
            self._q = queue.Queue()
            threading.Thread(target=self._writer, daemon=True, name="otpu-kv-writer").start()
            self._evict_disk()

    # ---- the interface
    def __contains__(self, key: str) -> bool:
        with self.lock:
            return key in self.ram or key in self.spilling or key in self.disk

    def put(self, key: str, data: np.ndarray) -> None:
        data = np.ascontiguousarray(data).view(np.uint8).reshape(-1)
        with self.lock:
            if key in self.ram:
                self.ram_used -= len(self.ram.pop(key))
            self.ram[key] = data
            self.ram_used += len(data)
            self.stats["put"] += 1
            self._evict_ram()

    def get(self, key: str) -> np.ndarray | None:
        with self.lock:
            if key in self.ram:
                self.ram.move_to_end(key)
                self.stats["ram_hits"] += 1
                return self.ram[key]
            if key in self.spilling:
                self.stats["ram_hits"] += 1
                return self.spilling[key]
            if key not in self.disk:
                self.stats["misses"] += 1
                return None
            t0 = time.perf_counter()
            try:
                data = np.fromfile(self._path(key), np.uint8)
            except OSError:
                self._drop_disk(key)
                self.stats["misses"] += 1
                return None
            self.stats["disk_read_s"] += time.perf_counter() - t0
            self.stats["disk_hits"] += 1
            self.disk.move_to_end(key)
            self.ram[key] = data                    # back in RAM; the file stays
            self.ram_used += len(data)
            self._evict_ram()
            return data

    def pin(self, keys) -> None:
        with self.lock:
            for k in keys:
                self.pins[k] = self.pins.get(k, 0) + 1

    def unpin(self, keys) -> None:
        with self.lock:
            for k in keys:
                n = self.pins.get(k, 0) - 1
                if n > 0:
                    self.pins[k] = n
                else:
                    self.pins.pop(k, None)
            self._evict_ram()
            self._evict_disk()

    def discard(self, key: str) -> None:
        """Remove a blob from every tier (no on_drop call)."""
        with self.lock:
            if key in self.ram:
                self.ram_used -= len(self.ram.pop(key))
            if key in self.spilling:
                self.spill_used -= len(self.spilling.pop(key))
            if key in self.disk:
                self.disk_used -= self.disk.pop(key)
                self._unlink(key)

    def flush(self) -> None:
        """Wait until the blobs being spilled are on disk."""
        if self._q is not None:
            self._q.join()

    def usage(self) -> dict:
        with self.lock:
            return {"ram_bytes": self.ram_used, "ram_blobs": len(self.ram),
                    "spilling_bytes": self.spill_used,
                    "disk_bytes": self.disk_used, "disk_blobs": len(self.disk),
                    "ram_limit": self.ram_bytes, "disk_limit": self.disk_bytes, **self.stats}

    def keys(self) -> list[str]:
        with self.lock:
            return list(dict.fromkeys([*self.disk, *self.spilling, *self.ram]))

    # ---- the tiers
    def _path(self, key: str) -> Path:
        return self.dir / f"{key}.bin"

    def _evict_ram(self) -> None:
        if self.ram_used <= self.ram_bytes:
            return
        for key in list(self.ram):
            if self.ram_used <= self.ram_bytes:
                break
            if key in self.pins:
                continue
            data = self.ram.pop(key)
            self.ram_used -= len(data)
            if key in self.disk:                    # already on disk: drop the RAM copy
                pass
            elif self.dir is not None and len(data) <= self.disk_bytes:
                self.spilling[key] = data           # written by the writer thread
                self.spill_used += len(data)
                self._q.put(key)
                self.stats["spilled"] += 1
            else:
                self._dropped(key)

    def _writer(self) -> None:
        while True:
            key = self._q.get()
            try:
                with self.lock:
                    data = self.spilling.get(key)
                if data is not None:
                    tmp = self.dir / f".{key}.tmp"
                    data.tofile(tmp)
                    os.replace(tmp, self._path(key))
                    with self.lock:
                        if self.spilling.pop(key, None) is not None:
                            self.spill_used -= len(data)
                            self.disk[key] = len(data)
                            self.disk_used += len(data)
                            self._evict_disk()
                        else:                       # discarded meanwhile
                            self._unlink(key)
            except OSError:
                with self.lock:
                    d = self.spilling.pop(key, None)
                    if d is not None:
                        self.spill_used -= len(d)
                        self._dropped(key)
            finally:
                self._q.task_done()

    def _evict_disk(self) -> None:
        if self.disk_used <= self.disk_bytes:
            return
        for key in list(self.disk):
            if self.disk_used <= self.disk_bytes:
                break
            if key in self.pins or key in self.ram:
                continue
            self._drop_disk(key)

    def _drop_disk(self, key: str) -> None:
        self.disk_used -= self.disk.pop(key)
        self._unlink(key)
        if key not in self.ram and key not in self.spilling:
            self._dropped(key)

    def _unlink(self, key: str) -> None:
        try:
            self._path(key).unlink()
        except OSError:
            pass

    def _dropped(self, key: str) -> None:
        self.stats["dropped"] += 1
        if self.on_drop is not None:
            self.on_drop(key)


def model_key(*parts) -> str:
    """A short key of what decides a context's bytes: the checkpoint, the image's formats and
    layout (16 hex digits)."""
    return hashlib.sha256(repr(parts).encode()).hexdigest()[:16]


def _tok_bytes(tokens) -> bytes:
    return np.asarray(tokens, np.int64).astype("<i4").tobytes()


@dataclass
class Entry:
    """A saved context: tokens [0, n), its whole pages' keys, its tail's key (the last partial
    page's m positions, then the state) or None."""
    key: str
    tokens: np.ndarray
    pages: list[str]
    tail: str | None
    m: int = 0
    used: float = field(default_factory=time.monotonic)

    @property
    def n(self) -> int:
        return len(self.tokens)

    def blobs(self) -> list[str]:
        return self.pages + ([self.tail] if self.tail else [])


@dataclass
class Match:
    """Where a prompt continues from: its first k tokens, restored from `entry` (None: nothing
    saved matches)."""
    k: int
    entry: Entry | None = None


class PrefixCache:
    """Contexts saved in a BlobStore for one model's layout (kvio.Layout), by their tokens."""

    def __init__(self, store: BlobStore, layout, mkey: str, max_entries: int = 4096):
        self.store, self.L, self.mk = store, layout, mkey
        self.max_entries = max_entries
        self.entries: OrderedDict = OrderedDict()   # entry key -> Entry (oldest use first)
        self.refs: dict = {}                        # blob key -> entries using it
        self.lock = threading.RLock()
        self.stats = {"saved_pages": 0, "shared_pages": 0, "saved_tails": 0, "restored_pages": 0,
                      "restores": 0, "hit_tokens": 0, "lookups": 0}
        prev = store.on_drop
        store.on_drop = lambda k: (self._blob_dropped(k), prev and prev(k))
        self._load()

    # ---- keys
    def page_key(self, tokens, i: int) -> str:
        P = self.L.P
        return f"{self.mk}-p{i}-" + hashlib.sha256(_tok_bytes(tokens[:(i + 1) * P])).hexdigest()[:32]

    def entry_key(self, tokens) -> str:
        return f"{self.mk}-e-" + hashlib.sha256(_tok_bytes(tokens)).hexdigest()[:32]

    # ---- lookup
    def match(self, tokens, limit: int | None = None) -> Match:
        """The longest prefix of `tokens` (at most `limit` of them) the device can continue
        from a saved context."""
        q = np.asarray(tokens, np.int64)
        limit = len(q) if limit is None else min(limit, len(q))
        best = Match(0)
        with self.lock:
            self.stats["lookups"] += 1
            for e in self.entries.values():
                n = min(e.n, limit)
                if n <= best.k:
                    continue
                eq = e.tokens[:n] == q[:n]
                c = n if eq.all() else int(np.argmin(eq))
                if self.L.stateful:
                    if c == e.n and c > best.k:
                        best = Match(c, e)
                elif c > best.k:
                    best = Match(c, e)
        return best

    # ---- save and restore
    def save(self, io, tokens, pin: bool = False) -> Entry:
        """The device's context (its first len(tokens) positions hold `tokens`) into the store:
        the whole pages it does not have yet and the tail (last partial page, state); returns
        the entry (pinned: its blobs kept until unpin)."""
        P, L = self.L.P, self.L
        toks = np.asarray(tokens, np.int64)
        n = len(toks)
        if n == 0:
            raise ValueError("an empty context")
        key = self.entry_key(toks)
        full, m = n // P, n % P
        pages = [self.page_key(toks, i) for i in range(full)]
        tail = None
        if m or L.stateful:
            tail = f"{self.mk}-t-" + key.rsplit("-", 1)[1]
        with self.lock:
            old = self.entries.get(key)
            if old is not None and all(b in self.store for b in old.blobs()):
                old.used = time.monotonic()
                self.entries.move_to_end(key)
                if pin:
                    self.store.pin(old.blobs())
                return old
            missing = [i for i, k in enumerate(pages) if k not in self.store]
        # the device reads (no lock: the engine's thread)
        got = io.read_pages([(i, P) for i in missing]) if missing else []
        tail_data = None
        if tail is not None:
            parts = io.read_pages([(full, m)]) if m else []
            if L.stateful:
                parts.append(io.read_state())
            tail_data = np.concatenate(parts) if len(parts) > 1 else parts[0]
        e = Entry(key, toks.copy(), pages, tail, m)
        with self.lock:
            for i, d in zip(missing, got):
                self.store.put(pages[i], d)
            self.stats["saved_pages"] += len(missing)
            self.stats["shared_pages"] += len(pages) - len(missing)
            if tail is not None:
                self.store.put(tail, tail_data)
                self.stats["saved_tails"] += 1
            if pin:
                self.store.pin(e.blobs())
            self._add(e)
            self._persist(e)
            if not L.stateful:                  # contexts this one continues: covered by it
                for o in list(self.entries.values()):
                    if o is not e and o.n <= n and not self._is_pinned(o) and \
                            np.array_equal(o.tokens, toks[:o.n]):
                        self._remove(o)
            while len(self.entries) > self.max_entries:
                self._remove(next(iter(self.entries.values())))
        return e

    def restore(self, io, match: Match) -> int:
        """Write the pages (and state) of match.entry for its first match.k tokens into the
        device's caches; returns k (0: nothing restored, a blob gone)."""
        e, k = match.entry, match.k
        if e is None or k == 0:
            return 0
        P = self.L.P
        need = -(-k // P)
        writes = []
        with self.lock:
            for i in range(need):
                if i < len(e.pages):
                    d = self.store.get(e.pages[i])
                    if d is None:
                        self._remove(e)
                        return 0
                    writes.append((i, P, d))
                else:                               # the tail's partial page
                    d = self.store.get(e.tail)
                    if d is None:
                        self._remove(e)
                        return 0
                    writes.append((i, e.m, d[:self.L.page_bytes(e.m)]))
            state = None
            if self.L.stateful:
                d = self.store.get(e.tail)
                if d is None:
                    self._remove(e)
                    return 0
                state = d[len(d) - self.L.state_bytes:]
            e.used = time.monotonic()
            self.entries.move_to_end(e.key)
        io.write_pages(writes)
        if state is not None:
            io.write_state(state)
        with self.lock:
            self.stats["restores"] += 1
            self.stats["restored_pages"] += len(writes)
            self.stats["hit_tokens"] += k
        return k

    def unpin(self, e: Entry) -> None:
        self.store.unpin(e.blobs())

    def clear(self) -> None:
        with self.lock:
            for e in list(self.entries.values()):
                self._remove(e)

    def usage(self) -> dict:
        with self.lock:
            return {"entries": len(self.entries),
                    "tokens": int(sum(e.n for e in self.entries.values())), **self.stats}

    # ---- bookkeeping
    def _is_pinned(self, e: Entry) -> bool:
        return any(b in self.store.pins for b in e.blobs())

    def _add(self, e: Entry) -> None:
        self.entries[e.key] = e
        self.entries.move_to_end(e.key)
        for b in e.blobs():
            self.refs.setdefault(b, set()).add(e.key)

    def _remove(self, e: Entry) -> None:
        if self.entries.pop(e.key, None) is None:
            return
        self.store.discard(e.key)
        for b in e.blobs():
            users = self.refs.get(b)
            if users is not None:
                users.discard(e.key)
                if not users:
                    del self.refs[b]
                    self.store.discard(b)

    def _blob_dropped(self, key: str) -> None:
        with self.lock:
            for ek in list(self.refs.get(key, ())):
                e = self.entries.get(ek)
                if e is not None:
                    self._remove(e)
            e = self.entries.get(key)              # an entry's own record
            if e is not None:
                self._remove(e)

    def _persist(self, e: Entry) -> None:
        if self.store.dir is None:
            return
        meta = json.dumps({"pages": e.pages, "tail": e.tail, "m": e.m}).encode()
        rec = np.frombuffer(len(meta).to_bytes(8, "little") + meta + _tok_bytes(e.tokens),
                            np.uint8)
        self.store.put(e.key, rec)

    def _load(self) -> None:
        """Entries an earlier server saved on disk (whose blobs are all still there)."""
        if self.store.dir is None:
            return
        for k in self.store.keys():
            if not k.startswith(f"{self.mk}-e-"):
                continue
            rec = self.store.get(k)
            try:
                n = int.from_bytes(rec[:8].tobytes(), "little")
                meta = json.loads(rec[8:8 + n].tobytes())
                toks = np.frombuffer(rec[8 + n:].tobytes(), "<i4").astype(np.int64)
            except (ValueError, TypeError):
                self.store.discard(k)
                continue
            e = Entry(k, toks, list(meta["pages"]), meta["tail"], int(meta["m"]))
            if all(b in self.store for b in e.blobs()) and self.entry_key(toks) == k:
                self._add(e)
            else:
                self.store.discard(k)
