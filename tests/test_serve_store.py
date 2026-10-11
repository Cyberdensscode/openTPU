"""The server's KV store (opentpu/serve/store.py): BlobStore's tiers (RAM, spilled to files by
a writer thread, read back; pins; files found by a later store) and PrefixCache's entries on a
fake device whose page bytes are a function of the tokens before (as a KV cache's): the longest
saved prefix, whole pages shared by contexts with a common prefix, partial restores where the
caches hold the whole context and only saved ends where there is state, entries dropped with
their blobs."""
import hashlib

import numpy as np
import pytest

from opentpu.serve import store as ST


def _arr(n, v):
    return np.full(n, v, np.uint8)


def test_blobstore_spills_reads_back_and_drops(tmp_path):
    dropped = []
    s = ST.BlobStore(ram_bytes=300, disk_bytes=500, dir=tmp_path, on_drop=dropped.append)
    for i in range(4):
        s.put(f"b{i}", _arr(100, i))
    s.flush()
    u = s.usage()
    assert u["ram_bytes"] <= 300 and u["disk_blobs"] == 1 and (tmp_path / "b0.bin").exists()
    assert np.array_equal(s.get("b0"), _arr(100, 0))        # read back into RAM
    assert s.usage()["disk_hits"] == 1
    for i in range(4, 12):
        s.put(f"b{i}", _arr(100, i))
    s.flush()
    u = s.usage()
    assert u["ram_bytes"] <= 300 and u["disk_bytes"] <= 500 and dropped
    assert all(k not in s for k in dropped)
    assert s.get(dropped[0]) is None
    # a later store on the same directory finds the files
    s2 = ST.BlobStore(ram_bytes=300, disk_bytes=500, dir=tmp_path)
    on_disk = [k for k in s.keys() if k in s.disk]
    assert on_disk and all(k in s2 for k in on_disk)
    assert np.array_equal(s2.get(on_disk[-1]), s.get(on_disk[-1]))


def test_blobstore_pins_and_no_disk():
    dropped = []
    s = ST.BlobStore(ram_bytes=250, on_drop=dropped.append)
    s.put("a", _arr(100, 1))
    s.pin(["a"])
    for i in range(5):
        s.put(f"x{i}", _arr(100, 2))
    assert "a" in s and "a" not in dropped and s.usage()["disk_blobs"] == 0
    s.unpin(["a"])
    s.put("y", _arr(100, 3))
    assert "a" not in s and "a" in dropped


class _Layout:
    def __init__(self, P=4, stateful=False):
        self.P, self.stateful = P, stateful
        self.state_bytes = 8 if stateful else 0

    def page_bytes(self, m=None):
        return 2 * (self.P if m is None else m)


class _Dev:
    """A device whose position p holds 2 bytes of the hash of tokens [0, p] (a KV cache's
    dependence); the state 8 bytes of the hash of the whole context."""

    def __init__(self, L):
        self.layout, self.toks, self.cells = L, [], {}
        self.reads = 0

    @staticmethod
    def cell(prefix):
        return np.frombuffer(hashlib.sha256(np.asarray(prefix, np.int64).tobytes()).digest()[:2],
                             np.uint8)

    def feed(self, toks):
        for t in toks:
            self.toks.append(t)
            self.cells[len(self.toks) - 1] = self.cell(self.toks)

    def read_pages(self, pages):
        P, out = self.layout.P, []
        for i, m in pages:
            self.reads += 1
            out.append(np.concatenate([self.cells.get(i * P + j, np.zeros(2, np.uint8))
                                       for j in range(m)]))
        return out

    def write_pages(self, pages):
        P = self.layout.P
        for i, m, d in pages:
            for j in range(m):
                self.cells[i * P + j] = d[2 * j:2 * j + 2]

    def read_state(self):
        return np.frombuffer(hashlib.sha256(np.asarray(self.toks, np.int64).tobytes()).digest()
                             [:8], np.uint8).copy()

    def write_state(self, d):
        self.state = d.copy()

    def holds(self, toks):
        return all(np.array_equal(self.cells.get(p), self.cell(toks[:p + 1]))
                   for p in range(len(toks)))


def _cache(stateful=False, ram=1 << 20):
    L = _Layout(stateful=stateful)
    return ST.PrefixCache(ST.BlobStore(ram), L, "m0"), L


def test_prefix_cache_longest_prefix_and_shared_pages():
    pc, L = _cache()
    a = list(range(1, 11))                  # 10 tokens: pages 0, 1 whole, 2 positions in page 2
    dev = _Dev(L)
    dev.feed(a)
    e = pc.save(dev, a)
    assert e.n == 10 and len(e.pages) == 2 and e.m == 2 and e.tail
    b = a[:6] + [99, 98, 97, 96, 95]        # shares page 0 and two positions of page 1
    dev2 = _Dev(L)
    dev2.feed(b)
    reads = dev2.reads
    e2 = pc.save(dev2, b)
    assert dev2.reads - reads == 2          # page 1 and the tail: page 0 was stored
    assert pc.stats["shared_pages"] == 1
    # the longest prefix: a's first 9 tokens of a query continuing a otherwise
    q = a[:9] + [7, 7]
    m = pc.match(q)
    assert m.k == 9 and m.entry is e
    fresh = _Dev(L)
    assert pc.restore(fresh, m) == 9 and fresh.holds(q[:9])
    m = pc.match(b[:6] + [99, 1])
    assert m.k == 7 and m.entry is e2
    assert pc.match([5, 5, 5]).k == 0
    assert pc.match(a, limit=len(a) - 1).k == 9     # a prompt keeps a token to feed


def test_prefix_cache_with_state_restores_saved_ends_only():
    pc, L = _cache(stateful=True)
    a = list(range(1, 11))
    dev = _Dev(L)
    dev.feed(a[:6])
    e6 = pc.save(dev, a[:6])                # a reply's start, say
    dev.feed(a[6:])
    e10 = pc.save(dev, a)
    assert e6.key in pc.entries and e10.key in pc.entries      # both kept (state)
    assert pc.match(a[:9] + [5]).k == 6     # not 9: no state there
    m = pc.match(a + [5])
    assert m.k == 10 and m.entry is e10
    fresh = _Dev(L)
    pc.restore(fresh, m)
    assert fresh.holds(a) and np.array_equal(fresh.state, dev.read_state())


def test_stateless_prefixes_are_covered_and_dropped_blobs_drop_entries():
    pc, L = _cache(ram=40)                  # small: blobs dropped (no disk)
    a = list(range(1, 11))
    dev = _Dev(L)
    dev.feed(a[:5])
    e5 = pc.save(dev, a[:5])
    dev.feed(a[5:])
    pc.save(dev, a)
    assert e5.key not in pc.entries         # covered by the longer context
    for s in range(5):                      # other contexts push a's blobs out
        d = _Dev(L)
        toks = [100 + 10 * s + j for j in range(10)]
        d.feed(toks)
        pc.save(d, toks)
    assert pc.match(a + [0]).k == 0 or pc.match(a + [0]).entry.key in pc.entries
    assert all(all(b in pc.store for b in e.blobs()) for e in pc.entries.values())


def test_prefix_cache_entries_survive_a_restart(tmp_path):
    L = _Layout()
    pc = ST.PrefixCache(ST.BlobStore(1 << 20, 1 << 20, tmp_path), L, "m0")
    a = list(range(1, 11))
    dev = _Dev(L)
    dev.feed(a)
    pc.save(dev, a)
    pc.store.discard("nothing")
    # spill everything to disk: a store with no RAM to speak of
    for k in list(pc.store.ram):
        pc.store.ram_bytes = 0
    pc.store._evict_ram()
    pc.store.flush()
    pc2 = ST.PrefixCache(ST.BlobStore(1 << 20, 1 << 20, tmp_path), L, "m0")
    m = pc2.match(a + [3])
    assert m.k == 10
    fresh = _Dev(L)
    pc2.restore(fresh, m)
    assert fresh.holds(a)
    assert ST.PrefixCache(ST.BlobStore(1 << 20, 1 << 20, tmp_path), L, "m1").match(a).k == 0


def test_pinned_entries_stay():
    pc, L = _cache(ram=60)
    a = list(range(1, 11))
    dev = _Dev(L)
    dev.feed(a)
    e = pc.save(dev, a, pin=True)
    for s in range(6):
        d = _Dev(L)
        toks = [200 + 10 * s + j for j in range(10)]
        d.feed(toks)
        pc.save(d, toks)
    assert e.key in pc.entries and pc.match(a + [1]).k == 10
    pc.unpin(e)
    with pytest.raises(ValueError):
        pc.save(dev, [])
