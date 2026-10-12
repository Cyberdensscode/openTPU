"""opentpu.waits: the compiler's wait entries for otpu_seq WAITC (docs/isa.md "Wait counts")."""
import numpy as np
import pytest

from opentpu import isa as I
from opentpu import waits as W

D = 128


def k(words, pc):
    return W.counts(words[pc])


def vm(words, pc):
    return W.vmask(words[pc])


def vop(func, dst, a, b, rows=1, cols=8, **kw):
    return I.vop(func, dst, a, b, rows, cols, cols, cols, cols, **kw)


def test_raw_across_units_and_none_without_overlap():
    prog = [I.ld(0, 100, 8), vop(I.V_ADD, 200, 100, 100), vop(I.V_ADD, 300, 400, 400),
            I.halt()]
    w = W.wait_counts(prog, args=[0] * 8)
    assert k(w, 1)[W.U_DMA] == 1                    # reads what the LD wrote
    assert k(w, 2)[W.U_DMA] == 0 and vm(w, 2) == 0
    assert k(w, 0) == [0] * 7 and vm(w, 0) == 0
    assert all(k(w, pc)[W.U_VPU] == 0 for pc in range(3))   # the VPU's is the mask


def test_war_across_units():
    """A younger writer on another unit waits for the reader (one-copy TMEM replay, RPL)."""
    prog = [vop(I.V_ADD, 200, 100, 100), I.ld(0, 100, 8), I.halt()]
    w = W.wait_counts(prog, args=[0] * 8)
    assert vm(w, 1) == 0b1


def test_waw_and_rank():
    prog = [vop(I.V_ADD, 200, 100, 100), vop(I.V_ADD, 300, 100, 100),
            vop(I.V_ADD, 400, 100, 100), I.ld(0, 200, 8), I.halt()]
    w = W.wait_counts(prog, args=[0] * 8)
    assert vm(w, 3) == 0b100                        # the 3rd last VPU instruction wrote 200


def test_the_vpu_mask_names_each_producer():
    """The VPU completes out of order: a consumer waits for its producers, not for the VPU
    instructions between them."""
    prog = [vop(I.V_ADD, 100, 500, 500), vop(I.V_ADD, 300, 500, 500),
            vop(I.V_ADD, 200, 500, 500), vop(I.V_ADD, 400, 500, 500),
            I.st(64, 100, 8), vop(I.V_ADD, 600, 100, 200), I.halt()]
    w = W.wait_counts(prog, args=[0] * 8)
    assert vm(w, 4) == 0b1000                       # the ST reads 100: 4 back
    assert vm(w, 5) == 0b1010                       # 100 (4 back) and 200 (2 back)


def test_own_unit_waits_only_for_the_vpu():
    prog = [I.ld(0, 100, 8), I.st(64, 100, 8), vop(I.V_ADD, 200, 100, 100),
            vop(I.V_ADD, 300, 200, 200), I.halt()]
    w = W.wait_counts(prog, args=[0] * 8)
    assert k(w, 1)[W.U_DMA] == 0                    # the DMA orders its own
    assert vm(w, 3) == 0b1                          # the VPU does not


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
    # no WAR; but the VOPs all write 2000 (WAW with each other: the 4 iterations' VOPs, each
    # with the ones before it in the mask)
    assert vm(w, 2) == 0
    assert vm(w, 3) == 0b111


def test_unknown_count_and_argument_are_conservative():
    prog = [I.loop(2, 0, rcount=8),
            I.ld(0, 1000, 8, rb=9),
            vop(I.V_ADD, 3000, 2000, 2000),
            I.ld(0, 2000, 8),
            I.halt()]
    w = W.wait_counts(prog)                          # R8, R9 unknown
    assert vm(w, 3) == W.VMASK                       # up to 15 iterations ran: WAR on 2000
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
    assert k(w, 2)[W.U_DMA] == 1 and vm(w, 2) == 0b1


def test_rank_beyond_the_window_needs_no_count():
    prog = [I.ld(0, 100, 8)] + [vop(I.V_ADD, 300 + 8 * i, 400, 400) for i in range(16)] + \
        [vop(I.V_ADD, 200, 100, 100), I.halt()]
    w = W.wait_counts(prog, args=[0] * 8)
    assert k(w, 17)[W.U_DMA] == 1                    # the DMA's last: rank 1
    prog = [I.ld(0, 100, 8)] + [I.ld(64, 500 + 8 * i, 8) for i in range(15)] + \
        [vop(I.V_ADD, 200, 100, 100), I.halt()]
    w = W.wait_counts(prog, args=[0] * 8)
    assert k(w, 16)[W.U_DMA] == 0                    # 16th last DMA instruction: done
    # the VPU starts out of order, and the sequencer dispatches a VPU instruction once the one
    # 15 before it has completed: the 15th last may run, the 16th last has completed
    for n, want in ((14, 1 << 14), (15, 0)):
        prog = [vop(I.V_ADD, 100, 400, 400)] + [vop(I.V_ADD, 500 + 8 * i, 400, 400)
                                               for i in range(n)] + [I.ld(0, 100, 8), I.halt()]
        w = W.wait_counts(prog, args=[0] * 8)
        assert vm(w, n + 1) == want


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
    prog = [I.ld(0, 100, 8), vop(I.V_ADD, 200, 100, 100), vop(I.V_ADD, 300, 500, 500),
            vop(I.V_ADD, 400, 200, 200), I.halt()]
    cfg = board_config(S=1, DRAM_BYTES=1 << 16, TMEM_WORDS=1 << 12)
    words = W.wait_counts(prog, args=[0] * 8)
    assert W.check_isasim(Machine(cfg, [prog], [None], args=[0]), [words]) == 4
    w1 = words.copy()
    w1[1] = 0
    with pytest.raises(AssertionError, match="misses the 1-th last DMA"):
        W.check_isasim(Machine(cfg, [prog], [None], args=[0]), [w1])
    w3 = words.copy()
    assert vm(w3, 3) == 0b10
    w3[3] = W.counts(0)[0]                          # no mask: the VOP 2 back is missed
    with pytest.raises(AssertionError, match="misses the 2-th last VPU"):
        W.check_isasim(Machine(cfg, [prog], [None], args=[0]), [w3])


@pytest.mark.parametrize("S,boot", [(1, False), (2, True)])
def test_waitc_rtl_matches_isa(have_verilator, monkeypatch, S, boot):
    """otpu_seq WAITC on the compiler's counts: the RTL's state equals the ISA simulator's (boot:
    the loader reads the wait words from DRAM after the program; else +hz_sync_s<k>), and the
    RTL's own check (every conflict on the window waited for) holds."""
    from conftest import assert_same_state, rel
    from opentpu import Config, rtlsim
    from opentpu.kernels import attention_decode
    from opentpu.runtime import compile_kernel, run
    from test_kernels import attn_args
    monkeypatch.setitem(rtlsim.UARCH, "SEQ_WC", 1)
    monkeypatch.setitem(rtlsim.UARCH, "WIN", 16)          # the card's window (WAITC: 16)
    monkeypatch.setitem(rtlsim.MEMORY, "BOOT", boot)
    monkeypatch.setenv("OTPU_WAITS", "1")
    args, want = attn_args(np.random.default_rng(7), Hq=8, Hkv=2, T=100)
    comp, imgs = compile_kernel(attention_decode, Config(S=S), **args)
    ri = run(comp, [i.copy() for i in imgs], "isa")
    rr = run(comp, [i.copy() for i in imgs], "rtl")
    assert_same_state(ri, rr)
    assert rel(rr.outputs["out"], want) < 0.03


def _ooo_program(n_after):
    """A VOP held by a long LD, then n_after independent VOPs (the VPU starts them around it,
    and from the 15th on each waits for the held one: the sequencer's hold), then a VOP that
    reads the held one's and the 3rd last's results, and an ST of it."""
    prog = [I.ld(0, 0, 4096), vop(I.V_ADD, 5000, 0, 0, rows=8, cols=64)]
    for i in range(n_after):
        prog.append(vop(I.V_MUL, 30000 + 64 * i, 20000 + 64 * i, 20000 + 64 * i, cols=64))
    prog += [vop(I.V_ADD, 40000, 5000, 30000 + 64 * (n_after - 3), cols=64),
             I.st(1 << 16, 40000, 64), I.halt()]
    return prog


@pytest.mark.parametrize("n_after", [6, 20])
def test_waitc_rtl_vpu_out_of_order(have_verilator, monkeypatch, n_after):
    """WAITC: the VPU starts younger VOPs around a held one (a mask per consumer), and holds
    the 16th; the RTL's state equals the ISA simulator's."""
    from opentpu import Config, rtlsim
    from opentpu.isasim import Machine
    monkeypatch.setenv("OTPU_WAITS", "1")
    cfg = Config(S=1)
    prog = _ooo_program(n_after)
    rng = np.random.default_rng(n_after)
    img = np.zeros(1 << 20, np.uint8)
    img[:4 * 4096] = rng.uniform(-1, 1, 4096).astype(np.float32).view(np.uint8)
    img[20000 * 4:21500 * 4] = rng.uniform(-1, 1, 1500).astype(np.float32).view(np.uint8)
    prog.insert(0, I.ld(20000 * 4, 20000, 1500))     # the VOPs' inputs first
    w = W.wait_counts(prog)
    # the held VOP n_after + 1 back (none past 15: it has completed), the 3rd last
    assert vm(w, len(prog) - 3) == ((1 << n_after) if n_after < 15 else 0) | 0b100
    m = Machine(cfg, [prog], [img.copy()]).run()
    drams, tmems, _ = rtlsim.run(cfg, [prog], [img.copy()], uarch={"SEQ_WC": 1, "WIN": 16})
    assert np.array_equal(tmems[0], m.slices[0].tmem)
    assert np.array_equal(drams[0], m.slices[0].dram)


@pytest.mark.parametrize("seed", range(4))
def test_waitc_rtl_hazard_stress(have_verilator, monkeypatch, seed):
    """WAITC on the scoreboard stress programs (RAW / WAR / WAW everywhere): the compiler's
    entries, the RTL's state equals the ISA simulator's and its own check holds."""
    from opentpu import Config, rtlsim
    from opentpu.isasim import Machine
    from test_rtl import _hazard_program, _images
    monkeypatch.setenv("OTPU_WAITS", "1")
    rng = np.random.default_rng(3000 + seed)
    cfg = Config(S=1)
    prog = _hazard_program(rng, cfg)
    imgs = _images(rng, 1)
    m = Machine(cfg, [prog], [i.copy() for i in imgs]).run()
    drams, tmems, _ = rtlsim.run(cfg, [prog], [i.copy() for i in imgs], uarch={"SEQ_WC": 1, "WIN": 16})
    assert np.array_equal(tmems[0], m.slices[0].tmem)
    assert np.array_equal(drams[0], m.slices[0].dram)


@pytest.mark.parametrize("wf,rsc,size", [(I.W8, False, 4 + 4 + 2), (I.W4I, False, 12 + 4 * 6),
                                         (I.W4F, False, 12 + 4 * 6), (I.WNV, False, 24 + 8 * 6),
                                         (I.W4F, True, 4)])
def test_mx_mm_scale_rows(wf, rsc, size):
    """The MX column's otpu_pkg: an MM reads each scale row up to block k0 + KB's (RSC: Sw;
    int8x: Sw and a byte a block; int4x / fp4x6: Sw, a word and a word a block; NVFP4: Sw,
    three words and 8 bytes a block); KB is w4[23:16], k0 w4[31:24]."""
    R = [(0, 0)] * 16
    ins = I.mm(0, 0x10000, 100, 3, 2, 256, 3, 1, 0, 64, wf=wf, k0=4, rsc=rsc)
    fp = W.footprint(ins, R)
    assert (W.SP_DRAM, 0x10000, 0x10000 + 2 * 64 + size) in fp.rd
    assert (W.SP_ACT, 0, 2) in fp.rd                 # ab 0, KB 2 (not w4[31:16])


def test_mx_qst_scale_rows():
    """QST writes a scale row of 1 + ceil(KB / 4) words a row (Sw, then a byte a block)."""
    R = [(0, 0)] * 16
    ins = I.qst(0, 0x1000, 0x8000, 3, 5, 5 * D, 5 * D, 1)
    fp = W.footprint(ins, R)
    assert (W.SP_DRAM, 0x8000, (0x8000 + 4 * 3 * (1 + 2)) | 63) in fp.wr
