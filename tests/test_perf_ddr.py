"""tools/perf_ddr.py's boot image, without a simulation: the program as rtlsim._run boots it."""
import sys

import numpy as np

from opentpu import Config, isa as I, rtlsim
from opentpu.waits import wait_counts

sys.path.insert(0, str(rtlsim.ROOT / "tools"))
import perf_ddr  # noqa: E402


# Under WAITC (the card's otpu_seq since release-2) the image carries the compiler's wait entries
# after the program's last chunk, as rtlsim._run writes them. perf_ddr's image had the program
# alone, so every point stopped at its first dependency ("otpu_seq: WAITC: pc 12's wait 0 (k 0)
# misses slot 0"). A VPU copy of what a load wrote waits for the load, and a store of the copy
# waits for the copy: their entries are not zero.
def test_boot_program_carries_the_wait_entries_under_waitc(monkeypatch):
    cfg = Config(S=1, D=128, ACT_BLOCKS=16)
    prog = [I.ld(0, 0, 64), I.vop(I.V_COPY, 128, 0, 0, 1, 64, 64, 64, 64), I.st(4096, 128, 64),
            I.halt()]
    asm = np.asarray(I.assemble(prog), "<u4")
    ww = wait_counts(prog, cfg.D, cfg.S)
    assert ww.any()
    monkeypatch.setitem(rtlsim.BOARD_UARCH, "SEQ_WC", 1)
    words, n = perf_ddr.boot_program(prog, cfg)
    pad = -len(asm) % (cfg.D // 4)
    assert n == len(prog) and np.array_equal(words[:len(asm)], asm)
    assert not words[len(asm):len(asm) + pad].any()
    assert np.array_equal(words[len(asm) + pad:], ww.astype("<u8").view("<u4"))
    # without WAITC the program alone (the scoreboard needs no entries)
    monkeypatch.setitem(rtlsim.BOARD_UARCH, "SEQ_WC", 0)
    words, n = perf_ddr.boot_program(prog, cfg)
    assert n == len(prog) and np.array_equal(words, asm)
