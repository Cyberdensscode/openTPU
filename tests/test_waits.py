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
    # 31 (VMAX) before it has completed: the 31st last may run, the 32nd last has completed
    for n, want in ((W.VMAX - 1, 1 << (W.VMAX - 1)), (W.VMAX, 0)):
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
    and from the 31st on each waits for the held one: the sequencer's hold), then a VOP that
    reads the held one's and the 3rd last's results, and an ST of it."""
    prog = [I.ld(0, 0, 4096), vop(I.V_ADD, 5000, 0, 0, rows=8, cols=64)]
    for i in range(n_after):
        prog.append(vop(I.V_MUL, 30000 + 64 * i, 20000 + 64 * i, 20000 + 64 * i, cols=64))
    prog += [vop(I.V_ADD, 40000, 5000, 30000 + 64 * (n_after - 3), cols=64),
             I.st(1 << 16, 40000, 64), I.halt()]
    return prog


@pytest.mark.parametrize("n_after", [6, 20, 40])
def test_waitc_rtl_vpu_out_of_order(have_verilator, monkeypatch, n_after):
    """WAITC: the VPU starts younger VOPs around a held one (a mask per consumer), and holds
    the 32nd; the RTL's state equals the ISA simulator's."""
    from opentpu import Config, rtlsim
    from opentpu.isasim import Machine
    monkeypatch.setenv("OTPU_WAITS", "1")
    cfg = Config(S=1)
    prog = _ooo_program(n_after)
    rng = np.random.default_rng(n_after)
    img = np.zeros(1 << 20, np.uint8)
    img[:4 * 4096] = rng.uniform(-1, 1, 4096).astype(np.float32).view(np.uint8)
    img[20000 * 4:23000 * 4] = rng.uniform(-1, 1, 3000).astype(np.float32).view(np.uint8)
    prog.insert(0, I.ld(20000 * 4, 20000, 3000))     # the VOPs' inputs first
    w = W.wait_counts(prog)
    # the held VOP n_after + 1 back (none past VMAX: it has completed), the 3rd last
    assert vm(w, len(prog) - 3) == ((1 << n_after) if n_after < W.VMAX else 0) | 0b100
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


# ------------------------------------------------------------------------------ the host's loads
# A bitstream whose sequencer takes wait entries (CFG_UARCH bit24, Config.SEQ_WC) loads every
# program with them; the host refuses a program without them there, and changes nothing elsewhere.
def _prog():
    return [I.ld(0, 100, 8), vop(I.V_ADD, 200, 100, 100), vop(I.V_MUL, 300, 200, 100),
            I.st(4096, 300, 8), I.halt()]


def _board(seq_wc: bool):
    from opentpu.host.board import Board
    from opentpu.host.fake import FakeTransport
    t = FakeTransport(regmap=4, devname=None, seq_wc=seq_wc)
    return Board(t), t


def test_a_waitc_bitstream_refuses_a_program_without_entries():
    from opentpu.host import regs as RG
    from opentpu.host.board import ConfigMismatch
    b, t = _board(True)
    assert b.waitc
    words = np.asarray(I.assemble(_prog()), np.uint32)
    with pytest.raises(ConfigMismatch, match="wait entries"):
        b.load_program(0x10000, words)
    with pytest.raises(ValueError, match="do not match"):
        b.load_program(0x10000, words, np.zeros(2, np.uint64))
    assert t.regs[RG.R_PROG_N] == 0                         # nothing was loaded


def test_a_waitc_bitstream_loads_the_entries_after_the_program():
    """Board.load: the words, zeros to the next chunk, the 64-bit entries (waits.loadable), at
    the program's address; PROG_N counts the instructions alone."""
    from opentpu.host import regs as RG
    b, t = _board(True)
    prog = _prog()
    b.load(0x10000, prog)
    words = np.asarray(I.assemble(prog), np.uint32)
    ent = W.wait_counts(prog, D, 1)
    got = b.read(0x10000, 2 * D + 8 * len(prog)).view("<u4")
    assert np.array_equal(got[:len(words)], words)
    assert not got[len(words):2 * D // 4].any()             # 160 bytes of program: to 256
    assert np.array_equal(got[2 * D // 4:].view("<u8"), ent)
    assert vm(ent, 2) == 0b1 and t.regs[RG.R_PROG_N] == len(prog)


def test_a_bitstream_without_seq_wc_loads_the_words_alone():
    """No CFG_UARCH bit24: load_program as before (entries, if given, unused), Board.load the
    same words."""
    b, t = _board(False)
    assert not b.waitc
    prog = _prog()
    words = np.asarray(I.assemble(prog), np.uint32)
    b.write(0x10000, np.full(512, 0xAB, np.uint8))
    b.load_program(0x10000, words)
    got = b.read(0x10000, 512)
    assert np.array_equal(got[:4 * len(words)].view("<u4"), words)
    assert (got[4 * len(words):] == 0xAB).all()             # nothing after the program
    b.load(0x10000, prog)
    assert np.array_equal(b.read(0x10000, 512), got)


def test_the_window_guard_on_the_host(monkeypatch):
    """A WAITC bitstream of a target whose window is over 16 is refused: the entries name a
    unit's last 15 instructions."""
    import dataclasses

    from opentpu import target as T
    from opentpu.host.board import ConfigMismatch
    b, _ = _board(True)
    t = T.current()
    wide = dataclasses.replace(t, uarch={**t.uarch, "WIN": 32})
    monkeypatch.setattr(T, "by_platform", lambda pid: wide)
    with pytest.raises(ConfigMismatch, match="window of 32"):
        b.load(0x10000, _prog())


def test_device_config_takes_seq_wc_and_check_fits_needs_its_room():
    """device_config's SEQ_WC is CFG_UARCH bit24 (the CFG_HASH has it: check_hash passes);
    check_fits refuses a configuration without the entries' room on such a bitstream, and a
    configuration with the room runs on a bitstream without them."""
    import dataclasses

    from opentpu.host.board import ConfigMismatch, check_fits, device_config
    b, _ = _board(True)
    cfg = device_config(b.info())
    assert cfg.SEQ_WC
    check_fits(cfg, b.info())
    with pytest.raises(ConfigMismatch, match="no room for them"):
        check_fits(dataclasses.replace(cfg, SEQ_WC=False), b.info())
    b2, _ = _board(False)
    assert not device_config(b2.info()).SEQ_WC
    check_fits(cfg, b2.info())


def test_a_host_without_seq_wc_refuses_a_waitc_bitstream():
    """The CFG_HASH of a build with SEQ_WC has it (target.arch_hash), so a host that writes no
    entries, whose hash leaves it out, refuses the bitstream (check_hash); without it the hash
    is every earlier build's."""
    import dataclasses

    from opentpu import target as T
    card = T.config("ypcb-00338")
    assert not card.SEQ_WC and f"{T.arch_hash(card):08x}" == "89417856"
    on = dataclasses.replace(card, SEQ_WC=True)
    assert T.arch_hash(on) != T.arch_hash(card)
    assert T.main(["hash", "ypcb-00338", "SEQ_WC=1"]) == 0          # (run_vivado.sh's CFG_HASH)
    assert T.arch_hash(T.build_config("ypcb-00338", SEQ_WC="1")) == T.arch_hash(on)


def test_board_backend_runs_with_the_entries_and_its_chain_slots_hold_them():
    """BoardBackend on a WAITC bitstream: start() loads each program with its entries (those of
    its run_args' bounds, for a worker's words too: compiler.Words), slot_words
    (generate.slot_words) puts them after a chained program; on a bitstream without the bit
    the words alone, as before. A configuration without their room is refused there."""
    from opentpu.compiler import Program, RunVar, Words
    from opentpu.host.board import BoardBackend, ConfigMismatch
    from opentpu.host.fake import FakeTransport
    from opentpu.isasim import board_config
    from opentpu.llm import generate as G
    pos = RunVar("pos", 64)
    # the LD reads below 4 KiB with the bound, anywhere without: the QST's DRAM write waits
    prog = Program([I.addi(1, 15, 0), I.ld(0, 100, 8, ra=1),
                    I.qst(2000, 0x10000, 0x18000, 1, 1, D, D, 1), I.halt()], [(pos, 64)])
    ent = W.wait_counts(prog, D, 1)
    assert k(ent, 2)[W.U_DMA] == 0 and k(W.wait_counts(prog, D, 1, run_args=()), 2)[W.U_DMA] == 1
    words = np.asarray(I.assemble(prog), np.uint32)
    n = 4 * len(words)
    for seq_wc in (True, False):
        t = FakeTransport(regmap=4, devname=None, seq_wc=seq_wc)
        cfg = board_config(DRAM_BYTES=1 << 21, SEQ_WC=True)
        be = BoardBackend(cfg, [np.zeros(1 << 16, np.uint8)], transport=t, status=False)
        assert be.waitc == seq_wc
        for p in ([prog], Words.of(words, prog.run_args)):
            be.board.write(be.prog_at, np.zeros(1024, np.uint8))
            be.start(p)
            be.wait()
            got = be.board.read(be.prog_at, 1024)
            assert np.array_equal(got[:n].view("<u4"), words)
            if seq_wc:                                  # (from the next chunk)
                off = -(-n // D) * D
                assert np.array_equal(got[off:off + 8 * len(prog)].view("<u8"), ent)
            else:
                assert not got[n:].any()
        assert np.array_equal(G.slot_words(be, prog),
                              W.loadable(words, ent, D) if seq_wc else words)
        be.close()
    t = FakeTransport(regmap=4, devname=None, seq_wc=True)
    with pytest.raises(ConfigMismatch, match="no room"):
        BoardBackend(board_config(DRAM_BYTES=1 << 21), [np.zeros(4096, np.uint8)], transport=t,
                     status=False)


def test_the_entries_room():
    """A program of IMEM's instructions with its entries is waits.area's 5 / 4 of its words (IMEM
    fills whole chunks), a chain slot two of them (generate.prog_bytes)."""
    from opentpu.llm import generate as G
    n = 32768 // 8
    words = np.zeros(32768, np.uint32)
    assert 4 * len(W.loadable(words, np.zeros(n, np.uint64), D)) == W.area(32768, True) == 163840
    assert W.area(32768) == 131072
    assert G.prog_bytes(32768) == 1 << 18 and G.prog_bytes(32768, True) == 327680


def test_the_caches_key_on_seq_wc_and_the_waits_pass(monkeypatch):
    """The program and image caches' layouts hold the Config, SEQ_WC in it (its chain slots and
    program area differ); waits.py is in the keyed source (progcache.keyed), so an entry and a
    program compiled before a change to the pass miss; the entries' key holds the words, their
    arguments' bounds, D, S and the source."""
    import dataclasses

    from opentpu import imgcache as IC
    from opentpu import progcache as PC
    from opentpu.compiler import RunVar
    from opentpu.isasim import board_config
    cfg = board_config()
    on = dataclasses.replace(cfg, SEQ_WC=True)
    assert PC.key((cfg,), ("decode", 1)) != PC.key((on,), ("decode", 1))
    img = object()
    assert IC.key(img, (cfg,), "ck") != IC.key(img, (on,), "ck")
    assert "waits.py" in {p.name for p in PC.keyed()}
    words = np.asarray(I.assemble(_prog()), np.uint32)
    k0 = W.entries_key(words, D, 1)
    assert W.entries_key(words, D, 1, [(RunVar("pos", 64), 1)]) != k0
    assert W.entries_key(words, D, 2) != k0
    monkeypatch.setattr(PC, "_SRC", "another tree")
    assert W.entries_key(words, D, 1) != k0


def test_entries_are_kept(monkeypatch, tmp_path):
    """waits.entries computes once per program: then from this process, or the program cache's
    disk (an entry beside its programs)."""
    monkeypatch.setenv("OTPU_PROG_CACHE", str(tmp_path))
    monkeypatch.setattr(W, "_mem", W.OrderedDict())
    prog = _prog()
    words = np.asarray(I.assemble(prog), np.uint32)
    n0 = dict(W.stats)
    e = W.entries(words, D, 1)
    assert np.array_equal(e, W.wait_counts(prog, D, 1, run_args=()))
    assert W.stats["computed"] == n0["computed"] + 1
    assert np.array_equal(W.entries(words, D, 1, prog=prog), e)
    assert W.stats["memory"] == n0["memory"] + 1
    W._mem.clear()
    assert np.array_equal(W.entries(words, D, 1), e)
    assert W.stats["disk"] == n0["disk"] + 1 and len(list(tmp_path.glob("*/*.npz"))) == 1


# ------------------------------------------------------------------------------ off the critical path
def test_words_pickle_with_their_run_args_and_entries():
    """A compile worker's words reach the engine with their run_args and entries (Words)."""
    import pickle

    from opentpu.compiler import RunVar, Words
    w = Words.of(I.assemble(_prog()), [(RunVar("pos", 64), 1)], np.arange(5, dtype=np.uint64))
    r = pickle.loads(pickle.dumps(w))
    assert isinstance(r, Words) and np.array_equal(r, w) and r.run_args[0][1] == 1
    assert r.run_args[0][0].bound == 64 and np.array_equal(r.waits, w.waits)
    assert Words.of(r, r.run_args).waits is r.waits        # (kept through Words.of)


def test_worker_words_carry_the_entries_on_a_waitc_configuration():
    """qwen3.worker_words: the entries computed with the words (in the worker process) when the
    image's configuration takes them, none otherwise; BoardBackend loads a worker's entries as
    they come."""
    from types import SimpleNamespace

    from opentpu.compiler import Program, RunVar
    from opentpu.host.board import BoardBackend
    from opentpu.host.fake import FakeTransport
    from opentpu.isasim import board_config
    from opentpu.llm.qwen3 import worker_words
    prog = Program([I.addi(1, 15, 0), I.ld(0, 100, 8, ra=1),
                    I.qst(2000, 0x10000, 0x18000, 1, 1, D, D, 1), I.halt()],
                   [(RunVar("pos", 64), 64)])
    on = SimpleNamespace(cfg=board_config(SEQ_WC=True))
    w = worker_words(on, prog)
    assert np.array_equal(w.waits, W.wait_counts(prog, D, 1)) and w.run_args == prog.run_args
    assert worker_words(SimpleNamespace(cfg=board_config()), prog).waits is None
    t = FakeTransport(regmap=4, devname=None, seq_wc=True)
    be = BoardBackend(board_config(DRAM_BYTES=1 << 21, SEQ_WC=True), [np.zeros(4096, np.uint8)],
                      transport=t, status=False)
    w.waits = np.full(len(prog), W.DEFAULT, np.uint64)     # (what the worker said: loaded)
    be.start(w)
    be.wait()
    got = be.board.read(be.prog_at + D, 8 * len(prog)).view("<u8")
    assert (got == W.DEFAULT).all()
    be.close()


def test_entries_many_in_a_pool():
    """waits.entries_many: the programs not kept yet computed in the pool at once, then kept."""
    from concurrent.futures import ThreadPoolExecutor
    W._mem.clear()
    progs = [_prog(), [I.ld(0, 300, 8), vop(I.V_ADD, 300, 300, 300), I.halt()]]
    items = [(np.asarray(I.assemble(p), np.uint32), None, p) for p in progs]
    with ThreadPoolExecutor(2) as pool:
        got = W.entries_many(items, D, 1, pool)
    assert all(np.array_equal(g, W.wait_counts(p, D, 1)) for g, p in zip(got, progs))
    n = W.stats["memory"]
    assert all(np.array_equal(a, b) for a, b in zip(W.entries_many(items, D, 1), got))
    assert W.stats["memory"] == n + 2


def test_prebuild_keeps_the_sessions_entries(tmp_path, monkeypatch):
    """prebuild's entries step (waits_ahead) on a WAITC configuration: each bucket's resident
    decode and generate programs and the warm prompt runs, kept on the disk under the keys a
    session's engine looks them up by; nothing without SEQ_WC."""
    import dataclasses

    from opentpu.compiler import Words
    from opentpu.host import prebuild as PB
    from opentpu.llm.qwen3 import PREFILL_ROWS, Engine, device_config
    from test_autodecode import _tiny
    monkeypatch.setenv("OTPU_PROG_CACHE", str(tmp_path))
    monkeypatch.setattr(W, "_mem", W.OrderedDict())
    Wt, spec = _tiny("qwen3")
    cfg = dataclasses.replace(device_config(spec, 512, rows=PREFILL_ROWS, lookup=True),
                              SEQ_WC=True)
    eng = Engine(spec, Wt, cap=512, cfg=cfg, backend=PB._Drop, resident=True, pipeline=False)
    n = PB.waits_ahead(eng)
    assert n["programs"] >= 4 and n["computed"] == n["programs"]
    assert PB.waits_ahead(Engine(spec, Wt, cap=512, cfg=dataclasses.replace(cfg, SEQ_WC=False),
                                 backend=PB._Drop, resident=True, pipeline=False)) is None
    W._mem.clear()                          # (a session: another process, the card's flags)
    ses = Engine(spec, Wt, cap=512, cfg=cfg, backend=PB._Drop, resident=True)
    assert ses.resident
    for b in (1, 2):
        progs, ra = ses._decode((b - 1) * 256)
        w = progs if isinstance(progs, Words) else I.assemble(progs[0])
        assert (tmp_path / (k := W.entries_key(w, cfg.D, cfg.S, ra))[:2] / f"{k}.npz").is_file()
        g = ses._generate_prog(b)[0]
        k = W.entries_key(I.assemble(g), cfg.D, cfg.S, g.run_args)
        assert (tmp_path / k[:2] / f"{k}.npz").is_file()


def test_prebuilds_backend_has_the_cards_flags():
    """prebuild's _Drop compiles what a card session does: BoardBackend's flags on a bitstream
    with run arguments and the decode loop, on a transport that streams."""
    from opentpu.host import prebuild as PB
    from opentpu.host.board import BoardBackend
    from opentpu.host.fake import FakeTransport
    from opentpu.isasim import board_config
    t = FakeTransport(regmap=4, devname=None, args=True, gen=True)
    t.streams = True
    be = BoardBackend(board_config(DRAM_BYTES=1 << 21), [np.zeros(4096, np.uint8)], transport=t,
                      status=False)
    for k in ("args", "chains", "generates", "fills"):
        assert bool(getattr(be, k)) and getattr(PB._Drop, k), k
    be.close()
