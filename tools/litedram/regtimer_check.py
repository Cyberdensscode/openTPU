#!/usr/bin/env python3
"""regtimer.RefreshTimer (done registered) against LiteDRAM's RefreshTimer, in migen's simulator.

The two timers, at several periods (tREFI-like and short ones, 2 the shortest), get the same
random wait each cycle (long runs of wait, as the refresher holds it, and random toggles), and
done and count must be equal every cycle. A third timer whose done is registered without the
& ~done term (--mutate) must differ: the check sees a done held two cycles.

    python3 regtimer_check.py [--cycles 200000] [--seed 1] [--mutate]   (LiteX at core.json's)
"""
import argparse
import random
import sys
from pathlib import Path

from migen import If, Module, Signal, run_simulation
from litedram.core.refresher import RefreshTimer as Stock

sys.path.insert(0, str(Path(__file__).resolve().parent))
from regtimer import RefreshTimer                                # noqa: E402

PERIODS = (2, 3, 7, 1040, 4161)


class Mutant(RefreshTimer):
    """done <= wait & (count == 1) | wait & done: wrong (done stays up while wait does)."""
    def __init__(self, trefi):
        Module.__init__(self)
        self.wait = Signal()
        self.done = Signal()
        self.count = Signal(max=trefi + 1)
        count = Signal(max=trefi + 1, reset=trefi - 1)
        self.sync += [
            If(self.wait & ~self.done, count.eq(count - 1)).Else(count.eq(count.reset)),
            self.done.eq(self.wait & ((count == 1) | self.done)),
        ]
        self.comb += self.count.eq(count)


class Top(Module):
    def __init__(self, mutate):
        self.pairs = []
        for t in PERIODS:
            a, b = Stock(t), (Mutant(t) if mutate else RefreshTimer(t))
            self.submodules += a, b
            self.pairs.append((t, a, b))


def check(cycles, seed, mutate):
    rng = random.Random(seed)
    top = Top(mutate)
    errs = []
    dones = {t: 0 for t in PERIODS}

    def tb():
        w = {t: 1 for t in PERIODS}
        for c in range(cycles):
            for t, a, b in top.pairs:
                if rng.random() < min(0.05, 0.5 / t):         # runs of wait ~2 periods
                    w[t] ^= 1
                yield a.wait.eq(w[t])
                yield b.wait.eq(w[t])
            yield
            for t, a, b in top.pairs:
                da, db = (yield a.done), (yield b.done)
                ca, cb = (yield a.count), (yield b.count)
                dones[t] += da
                if (da, ca) != (db, cb) and len(errs) < 5:
                    errs.append(f"cycle {c}, period {t}: stock done {da} count {ca}, "
                                f"registered done {db} count {cb}")

    run_simulation(top, tb())
    for e in errs:
        print(e)
    print(f"{cycles} cycles, periods {PERIODS}, stock dones {dones}: "
          + ("PASS" if not errs else "FAIL"))
    return 1 if errs else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cycles", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--mutate", action="store_true")
    a = ap.parse_args()
    rc = check(a.cycles, a.seed, a.mutate)
    sys.exit(rc if not a.mutate else 1 - rc)


if __name__ == "__main__":
    main()
