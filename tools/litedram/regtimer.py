"""The refresher's timers with a registered done (production core, gen_core.py).

LiteDRAM's RefreshTimer counts down from tREFI - 1 (the ZQCS timer's from a second's cycles - 1)
while `wait`, and its `done` is the count's compare with zero (27 bits for the ZQCS timer at
133.33 MHz). The refresher's FSM drops its command's valid in DO-REFRESH on the ZQCS timer's done,
and that valid reaches every bankmachine through the multiplexer's choosers: in r2's PHY-placed m4
build (4398cac9) at 133.33 MHz the sys clock's worst path ran from zqcs_timer_count1[24] through
that compare, the FSM, the chooser and a bankmachine's consume into its command FIFO's read
enable, 15 levels, +0.029 ns. This RefreshTimer keeps `done` in a flip-flop, set the cycle the
count reaches zero: its next value is wait & ~done & (count == 1), which is what (count == 0) is
a cycle later, since the count moves to count - 1 exactly when wait & ~done and otherwise
reloads tREFI - 1 (> 0). The same cycles, the compare before the register.

    with registered_timers(): ...       # LiteDRAMCores built inside use this RefreshTimer

Derived from LiteDRAM's litedram/core/refresher.py (BSD-2-Clause): Copyright (c) 2016-2019
Florent Kermarrec, (c) 2015 Sebastien Bourdeauducq.
"""
from contextlib import contextmanager

from migen import If, Module, Signal, bits_for

import litedram.core.refresher


class RefreshTimer(Module):
    """litedram.core.refresher.RefreshTimer (its name and ports) with done registered."""
    def __init__(self, trefi):
        assert trefi > 1
        self.wait = Signal()
        self.done = Signal()
        self.count = Signal(bits_for(trefi))

        # # #

        count = Signal(bits_for(trefi), reset=trefi - 1)
        self.sync += [
            If(self.wait & ~self.done,
                count.eq(count - 1)
            ).Else(
                count.eq(count.reset)
            ),
            self.done.eq(self.wait & ~self.done & (count == 1)),
        ]
        self.comb += self.count.eq(count)


@contextmanager
def registered_timers():
    stock = litedram.core.refresher.RefreshTimer
    litedram.core.refresher.RefreshTimer = RefreshTimer
    try:
        yield
    finally:
        litedram.core.refresher.RefreshTimer = stock
