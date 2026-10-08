"""Rolling a context back (otpu-chat's next turn, docs/host.md): Engine.rollback and
Engine.snapshot.

A KV cache is positional: going back to position k only moves the write position (the next
runs' appends overwrite the rest, and attention masks what lies past a row's position, bit
for bit as in a fresh cache). Some state is not: Qwen3.5's DeltaNet states and convolution
windows and LFM2's convolution rings hold the last positions' rows, and Gemma 4's sliding
K / V rings keep position q in slot q mod ring, which later positions overwrite. A rollback
cannot rebuild that state from the cache; it restores a snapshot instead, a copy of the
image's state regions (Image.state_regions) taken at a position the context returns to (a
chat takes one where the next turn's rendering will start to differ: the start of the reply),
and the context continues from the snapshot's position.

Snapshots live on the device, in an area after the image (Engine(snapshots=n): n of them),
copied by programs of LD / ST pairs through TMEM (DRAM -> TMEM -> DRAM, two buffers), or on the
host (the backend's read and write) when the area does not fit the device's DRAM. A snapshot
stays valid while the context keeps its first `position` tokens: a rollback below it drops it.
"""
from __future__ import annotations

import numpy as np

from .. import isa as I

CHUNK = 16384           # words an LD / ST pair moves (64 KiB; two buffers in TMEM)


def merged(regions) -> list[tuple[int, int]]:
    """The regions (DRAM address, bytes) in their order, each joined to the one before when it
    starts where that one ends. A snapshot holds them in this order, so the bytes of a region
    sit at the same place in it for every DeltaNet state slot (the slots' regions are the same
    sizes in the same order; their joins may differ)."""
    out: list[list[int]] = []
    for a, n in ((int(a), int(n)) for a, n in regions if n):
        if a % 4 or n % 4:
            raise ValueError(f"state region {a:#x} + {n} is not word-aligned")
        if out and out[-1][0] + out[-1][1] == a:
            out[-1][1] += n
        else:
            out.append([a, n])
    ends = sorted(out)
    for (a, n), (b, _) in zip(ends, ends[1:]):
        if a + n > b:
            raise ValueError(f"state regions overlap at {b:#x}")
    return [(a, n) for a, n in out]


def copy_programs(pairs, cfg) -> list[list]:
    """Programs (one list per slice, each fitting IMEM) that copy [(src, dst, bytes)] DRAM to
    DRAM: per chunk of up to CHUNK words an LD into TMEM and an ST out of it, the chunks in
    two TMEM buffers in turn (an LD does not wait for the ST before it to leave the buffer)."""
    ch = min(CHUNK, cfg.TMEM_WORDS // 2)
    body = []
    for src, dst, n in pairs:
        for o in range(0, n // 4, ch):
            w = min(ch, n // 4 - o)
            t = ch * (len(body) // 2 % 2)
            body += [I.ld(src + 4 * o, t, w), I.st(dst + 4 * o, t, w)]
    per = (cfg.IMEM_WORDS // 8 - 1) // 2 * 2       # whole pairs, and the HALT
    progs = [body[i:i + per] + [I.halt()] for i in range(0, len(body), per)] or [[I.halt()]]
    return [[p] * cfg.S for p in progs]


class Snapshots:
    """An Engine's snapshots: `n` slots of the image's state regions, in a device area at
    `base` (copy programs), or on the host with base None; `at` maps a slot to its position.
    The regions are per DeltaNet state slot (Spec.mtp: two; the decoder's committed one is
    saved and restored)."""

    def __init__(self, image, cfg, n: int, base: int | None):
        self.image, self.cfg, self.n, self.base = image, cfg, n, base
        self._regions: dict = {}
        self._progs: dict = {}
        self.host: dict = {}                    # host snapshots: slot -> [per slice bytes]
        self.at: dict = {}                      # slot -> position
        self._age: dict = {}                    # slot -> when taken (the oldest is replaced)
        self._clock = 0
        self.nbytes = sum(n for _, n in self.regions(0))

    @staticmethod
    def area(image, cfg, n: int) -> int | None:
        """Where the device area of n snapshots goes (after the image, page-aligned), or None
        when it and the program area after it (BoardBackend's) do not fit the DRAM."""
        if not n or not stateful(image):
            return None
        need = n * sum(b for _, b in merged(image.state_regions(0)))
        base = -(-image.nbytes // 4096) * 4096
        if -(-(base + need) // 4096) * 4096 + 4 * cfg.IMEM_WORDS > cfg.DRAM_BYTES:
            return None
        return base

    def regions(self, slot: int) -> list:
        if slot not in self._regions:
            self._regions[slot] = merged(self.image.state_regions(slot))
        return self._regions[slot]

    def end(self) -> int:
        """The device area's end (bytes): the image arrays grow to it."""
        return self.base + self.n * self.nbytes

    def _programs(self, save: bool, k: int, slot: int):
        key = (save, k, slot)
        if key not in self._progs:
            a, pairs = self.base + k * self.nbytes, []
            for src, n in self.regions(slot):
                pairs.append((src, a, n) if save else (a, src, n))
                a += n
            self._progs[key] = copy_programs(pairs, self.cfg)
        return self._progs[key]

    def save(self, eng, pos: int, slot: int = 0) -> dict:
        """Take a snapshot at `pos` (the context's state now) in the slot already at `pos`, a
        free one, or the oldest; returns the copy's stats (device: the runs' counters)."""
        k = next((j for j, p in self.at.items() if p == pos), None)
        if k is None:
            free = [j for j in range(self.n) if j not in self.at]
            k = free[0] if free else min(self.at, key=self._age.get)
        self.at.pop(k, None)
        st = self._copy(eng, True, k, slot)
        self.at[k] = pos
        self._clock += 1
        self._age[k] = self._clock
        return st

    def best(self, k: int) -> int | None:
        """The slot of the latest snapshot at a position <= k."""
        ok = [j for j, p in self.at.items() if p <= k]
        return max(ok, key=self.at.get) if ok else None

    def restore(self, eng, j: int, slot: int = 0) -> dict:
        """Write snapshot j back into the state regions of `slot`."""
        return self._copy(eng, False, j, slot)

    def drop_above(self, pos: int) -> None:
        """Forget the snapshots taken after `pos` (the context changes from there)."""
        for j in [j for j, p in self.at.items() if p > pos]:
            del self.at[j]
            self.host.pop(j, None)

    def _copy(self, eng, save: bool, k: int, slot: int) -> dict:
        be, S = eng.backend, self.cfg.S
        if self.base is None:
            regs = self.regions(slot)
            if save:
                self.host[k] = [[be.read(s, a, n) for a, n in regs] for s in range(S)]
            else:
                for s in range(S):
                    for (a, _), v in zip(regs, self.host[k][s]):
                        be.write(s, a, v)
            return {"bytes": self.nbytes * S}
        stats = [be.run(p) for p in self._programs(save, k, slot)]
        return {"bytes": self.nbytes * S, "runs": len(stats),
                "cycles": sum((x or {}).get("cycles", 0) for x in stats)}


def stateful(image) -> bool:
    """The image holds state a rollback cannot rebuild from the KV cache (state_regions)."""
    return hasattr(image, "state_regions") and bool(image.state_regions(0))


def grow(images: list, end: int) -> list:
    """The image arrays grown to `end` bytes (zeros after), in place where they own their
    data."""
    out = []
    for m in images:
        if len(m) >= end:
            out.append(m)
            continue
        if isinstance(m, np.ndarray) and m.dtype == np.uint8 and m.ndim == 1 and m.flags.owndata:
            m.resize(end, refcheck=False)
            out.append(m)
        else:
            g = np.zeros(end, np.uint8)
            g[:len(m)] = np.asarray(m, np.uint8)
            out.append(g)
    return out
