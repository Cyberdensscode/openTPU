"""opentpu.waits: the compiler's wait counts for otpu_seq WAITC (docs/isa.md "Wait counts")."""
import numpy as np
import pytest

from opentpu import isa as I
from opentpu import waits as W

D = 128


def k(words, pc):
    return W.counts(words[pc])


def vop(func, dst, a, b, rows=1, cols=8, **kw):
    return I.vop(func, dst, a, b, rows, cols, cols, cols, cols, **kw)


def test_raw_across_units_and_none_without_overlap():
    prog = [I.ld(0, 100, 8), vop(I.V_ADD, 200, 100, 100), vop(I.V_ADD, 300, 400, 400),
            I.halt()]
    w = W.wait_counts(prog, args=[0] * 8)
    assert k(w, 1)[W.U_DMA] == 1                    # reads what the LD wrote
    assert k(w, 2)[W.U_DMA] == 0 and k(w, 2)[W.U_VPU] == 0
    assert k(w, 0) == [0] * 7


def test_war_across_units():
    """A younger writer on another unit waits for the reader (one-copy TMEM replay, RPL)."""
    prog = [vop(I.V_ADD, 200, 100, 100), I.ld(0, 100, 8), I.halt()]
    w = W.wait_counts(prog, args=[0] * 8)
    assert k(w, 1)[W.U_VPU] == 1


def test_waw_and_rank():
    prog = [vop(I.V_ADD, 200, 100, 100), vop(I.V_ADD, 300, 100, 100),
            vop(I.V_ADD, 400, 100, 100), I.ld(0, 200, 8), I.halt()]
    w = W.wait_counts(prog, args=[0] * 8)
    assert k(w, 3)[W.U_VPU] == 3                    # the 3rd last VPU instruction wrote 200


def test_own_unit_waits_only_for_the_vpu():
    prog = [I.ld(0, 100, 8), I.st(64, 100, 8), vop(I.V_ADD, 200, 100, 100),
            vop(I.V_ADD, 300, 200, 200), I.halt()]
    w = W.wait_counts(prog, args=[0] * 8)
    assert k(w, 1)[W.U_DMA] == 0                    # the DMA orders its own
    assert k(w, 3)[W.U_VPU] == 1                    # the VPU's pipeline does not


def test_loop_carried_and_registers():
    # each iteration reads the tile the last one's LD wrote: R1 advances by 8
    prog = [I.li(1, 0), I.loop(3, 4),
            I.ld(0, 1000, 8, rb=1),
            vop(I.V_ADD, 2000, 1000, 1000, ra=0, rb=1, rc=1),
            I.addi(1, 1, 8),
            I.halt()]
    w = W.wait_counts(prog, args=[0] * 8)
    assert k(w, 3)[W.U_DMA] == 1
    # the LD of iteration t+1 writes 1008 + 8t, which the VOP of iteration t does not read:
    # no WAR; but the VOPs all write 2000 (WAW with each other, the VPU's own count)
    assert k(w, 2)[W.U_VPU] == 0
    assert k(w, 3)[W.U_VPU] == 1


def test_unknown_count_and_argument_are_conservative():
    prog = [I.loop(2, 0, rcount=8),
            I.ld(0, 1000, 8, rb=9),
            vop(I.V_ADD, 3000, 2000, 2000),
            I.ld(0, 2000, 8),
            I.halt()]
    w = W.wait_counts(prog)                          # R8, R9 unknown
    assert k(w, 3)[W.U_VPU] == 1                     # the loop may have run: WAR on 2000
    assert k(w, 2)[W.U_DMA] == 1                     # R9 unknown: the LD may hit 2000
    st = {}
    W.wait_counts(prog, stats=st)
    assert st["widened"]


def test_mm_stream_start_counts_dram_only():
    st_ = I.st(0x1000, 0, 32)
    mm = I.mm(0x1000, 0x8000, 500, 1, 1, 0, 1, 1, 0, 0)
    prog = [st_, I.qact(600, 1, 0, 1, 0), mm, I.halt()]
    w = W.wait_counts(prog, args=[0] * 8)
    c = k(w, 2)
    assert c[5] == 1 and c[W.U_DMA] == 1             # the weights the ST wrote
    assert c[W.U_Q] == 1 and c[6] == 0               # the ACT RAM: release only


def test_bar_waits_for_everything():
    prog = [I.ld(0, 0, 8), vop(I.V_ADD, 200, 100, 100), I.bar(), I.halt()]
    w = W.wait_counts(prog, args=[0] * 8)
    assert k(w, 2)[W.U_DMA] == 1 and k(w, 2)[W.U_VPU] == 1


def test_rank_beyond_the_window_needs_no_count():
    prog = [I.ld(0, 100, 8)] + [vop(I.V_ADD, 300 + 8 * i, 400, 400) for i in range(16)] + \
        [vop(I.V_ADD, 200, 100, 100), I.halt()]
    w = W.wait_counts(prog, args=[0] * 8)
    assert k(w, 17)[W.U_DMA] == 1                    # the DMA's last: rank 1
    prog = [vop(I.V_ADD, 100, 400, 400)] + [vop(I.V_ADD, 500 + 8 * i, 400, 400)
                                           for i in range(15)] + [I.ld(0, 100, 8), I.halt()]
    w = W.wait_counts(prog, args=[0] * 8)
    assert k(w, 16)[W.U_VPU] == 0                    # 16th last VPU instruction: done


def _random_prog(rng, n):
    prog = [I.li(1, int(rng.integers(0, 4)) * 8)]
    body = []
    for _ in range(n):
        t = lambda: int(rng.integers(0, 12)) * 8      # noqa: E731
        c = rng.integers(0, 5)
        if c == 0:
            body.append(I.ld(int(rng.integers(0, 4)) * 64, t(), 8, rb=int(rng.integers(0, 2))))
        elif c == 1:
            body.append(I.st(int(rng.integers(0, 4)) * 64, t(), 8))
        elif c == 2:
            body.append(vop(I.V_ADD, t(), t(), t(), rb=int(rng.integers(0, 2))))
        elif c == 3:
            body.append(vop(I.V_RSUM, t(), t(), 0))
        else:
            body.append(I.addi(1, 1, 8))
    cut = int(rng.integers(1, len(body)))
    return prog + [I.loop(cut, int(rng.integers(0, 4)), rcount=8)] + body + [I.halt()]


@pytest.mark.parametrize("seed", range(40))
def test_counts_cover_every_conflict_of_a_run(seed):
    """The counts computed without the run's argument (a loop count) cover each concrete run."""
    from opentpu.isasim import Machine, board_config
    rng = np.random.default_rng(seed)
    prog = _random_prog(rng, 24)
    words = W.wait_counts(prog)
    cfg = board_config(S=1, DRAM_BYTES=1 << 16, TMEM_WORDS=1 << 12)
    for n in range(0, 6):
        m = Machine(cfg, [prog], [None], args=[n])
        assert W.check_isasim(m, [words]) > 0


def test_checker_finds_a_missing_wait():
    from opentpu.isasim import Machine, board_config
    prog = [I.ld(0, 100, 8), vop(I.V_ADD, 200, 100, 100), I.halt()]
    words = W.wait_counts(prog, args=[0] * 8)
    words[1] = 0
    cfg = board_config(S=1, DRAM_BYTES=1 << 16, TMEM_WORDS=1 << 12)
    with pytest.raises(AssertionError, match="misses"):
        W.check_isasim(Machine(cfg, [prog], [None], args=[0]), [words])
