"""Tiled prefill (opentpu/llm/tiled.py, docs/tiled_prefill.md): a run of T rows, every weight
chunk streamed once for all of them and the attention's rows packed MCOLS / G to an MM, against
today's prefill runs of MCOLS rows over the same tokens on the ISA simulator: the KV cache, every
layer block and the last row's logits bit for bit."""
import numpy as np
import pytest

from opentpu import isa as I
from opentpu.isasim import Machine, board_config
from opentpu.llm import tiled
from opentpu.llm.qwen3 import Engine, Spec, rope_tables

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(0)
    hc = transformers.Qwen3Config(hidden_size=256, num_hidden_layers=2, num_attention_heads=4,
                                  num_key_value_heads=2, head_dim=128, intermediate_size=512,
                                  vocab_size=1000, rms_norm_eps=1e-6, rope_theta=1e6,
                                  tie_word_embeddings=True, max_position_embeddings=4096)
    m = transformers.Qwen3ForCausalLM(hc).float().eval()
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "norm" in n:
                p.copy_(1 + 0.1 * torch.randn_like(p))
    W = {k: v.float().numpy() for k, v in m.state_dict().items()}
    return m, W, Spec(256, 2, 4, 2, 128, 512, 1000)


def _inputs(eng, toks, p0, n):
    """The I/O area's rows for tokens toks at positions p0 ..: (address, values) pairs."""
    io = eng.image.io
    x = eng.embed[toks]
    cs = [np.stack(a).astype(np.float32)
          for a in zip(*[rope_tables(eng.spec, p0 + r) for r in range(n)])]
    return [(io["x"], x), (io["cos"], cs[0]), (io["sin"], cs[1])]


def _put(dram, parts):
    for addr, v in parts:
        b = np.ascontiguousarray(v, np.float32).view(np.uint8).reshape(-1)
        dram[addr:addr + b.size] = b


def _logits(eng, dram, row):
    v, a = eng.spec.vocab, eng.image.io["logits"]
    return dram[a + 4 * row * v:a + 4 * (row + 1) * v].view(np.float32).copy()


@pytest.mark.parametrize("mcols,wformat,p0,T", [
    (4, "int8", 5, 16),         # 2 rows a packed tile (G = 2), one weight pass for 16 rows
    (4, "fp4", 247, 16),        # 4-bit layers; the tile crosses an attention block (256)
    (2, "int8", 9, 6),          # one row a tile (MCOLS = G), T not a multiple of MCOLS
    (8, "fp4", 120, 14),        # 4 rows a tile, a partial last tile; today's runs of 8 and
])                              # 6 rows (a run of <= MCOLS / 2 4-bit rows would PAIR)
def test_tile_is_todays_runs_bit_for_bit(tiny, mcols, wformat, p0, T):
    """A tile of T rows equals today's prefill runs (compile_rows, MCOLS rows each, its MMs one
    pass of MCOLS rows) over the same tokens: every layer block (weights and KV cache) and the
    last row's logits, bit for bit. The tile's MMs take all T rows (one weight pass, replayed
    groups), its attention P rows a tile with per-row masks."""
    _, W, spec = tiny
    wkw = dict(wformat=wformat, head_format="int8") if wformat != "int8" else {}
    act = max(16, mcols)
    cfg = board_config(MCOLS=mcols, ACT_ROWS=act, DRAM_BYTES=1 << 24,
                       PAIR=wformat != "int8")
    toks = [int(t) for t in np.random.default_rng(p0).integers(0, 1000, p0 + T)]
    eng = Engine(spec, W, cap=512, cfg=cfg, rows=T, **wkw)
    eng.prefill(toks[:p0])                      # the cache the rows attend over
    base = eng.backend.machine.slices[0].dram.copy()
    img = eng.image
    lo, hi = img.layer0, img.head[0]

    tile = tiled.compile_tile(img, p0, T, logit_rows=[T - 1])
    mm = [i for i in tile[0] if i.op == I.MM]
    assert max((i.w[5] >> 16) & 0xFF for i in mm) == T     # the projections: one MM, T rows
    a = base.copy()
    _put(a, _inputs(eng, toks[p0:], p0, T))
    a = Machine(cfg, tile, [a]).run().slices[0].dram

    b = base.copy()                             # today's runs of MCOLS rows
    R = mcols
    for r0 in range(0, T, R):
        n = min(R, T - r0)
        rows = [(0, p0 + r0 + r) for r in range(n)]
        progs = img.compile_rows(rows, [n - 1] if r0 + n == T else [])
        _put(b, _inputs(eng, toks[p0 + r0:p0 + r0 + n], p0 + r0, n))
        b = Machine(cfg, progs, [b]).run().slices[0].dram
        last = n - 1
    assert np.array_equal(a[lo:hi], b[lo:hi])
    la, lb = _logits(eng, a, T - 1), _logits(eng, b, last)
    assert np.isfinite(la).all()
    assert np.array_equal(la.view(np.uint32), lb.view(np.uint32))


def test_tile_seq_masks_each_row_group_to_its_own_tokens():
    """A packed tile's blocks: whole blocks every row sees first (looped), then up to the last
    row's token with each row group's count; a row past the block's start sees none of it."""
    s = tiled.tile_seq(250, 4, 4, 256, 2)       # rows at positions 254 .. 257, G = 2
    assert s.full == 0
    (t0, n, m0), (t1, n1, m1) = s.items
    assert (t0, n, t1, n1) == (0, 256, 256, 2)
    assert m0.keep == [(0, 2, 255), (2, 4, 256), (4, 6, 256), (6, 8, 256)]
    assert m1.keep == [(0, 2, 0), (2, 4, 0), (4, 6, 1), (6, 8, 2)]
    s = tiled.tile_seq(600, 0, 2, 256, 2)       # positions 600, 601: two whole blocks first
    assert s.full == 2 and [(t, n) for t, n, _ in s.items] == [(512, 90)]
    assert s.items[0][2].keep == [(0, 2, 89), (2, 4, 90)]


@pytest.mark.parametrize("mcols,wformat,act_rows,T", [(4, "fp4", 16, 12), (8, "int8", 16, 12),
                                                     (4, "fp4", 4, 4)])
def test_tile_on_rtl(tiny, have_verilator, mcols, wformat, act_rows, T):
    """A tile across an attention block on the RTL (the board's memory path): 12 rows with an
    ACT RAM of 16 (every projection one MM replayed for ceil(12 / MCOLS) groups; the packed
    attention's masked blocks; the column parts), and 4 rows on the card's own ACT RAM of
    MCOLS rows (no replay): DRAM and TMEM as the ISA simulator's."""
    from opentpu import rtlsim
    _, W, spec = tiny
    wkw = dict(wformat=wformat, head_format="int8") if wformat != "int8" else {}
    cfg = board_config(MCOLS=mcols, ACT_ROWS=act_rows, DRAM_BYTES=1 << 24,
                       PAIR=wformat != "int8")
    p0 = 256 - T // 2
    toks = [int(t) for t in np.random.default_rng(7).integers(0, 1000, p0 + T)]
    eng = Engine(spec, W, cap=512, cfg=cfg, rows=T, **wkw)
    eng.prefill(toks[:p0])
    dram = eng.backend.machine.slices[0].dram.copy()
    _put(dram, _inputs(eng, toks[p0:], p0, T))
    progs = tiled.compile_tile(eng.image, p0, T, logit_rows=[T - 1])
    m = Machine(cfg, progs, [dram.copy()]).run()
    drams, tmems, _ = rtlsim.run(cfg, progs, [dram.copy()], uarch=rtlsim.BOARD_UARCH, axi=True,
                                 boot=True)
    n = eng.image.nbytes
    assert np.array_equal(drams[0][:n], m.slices[0].dram[:n])
    assert np.array_equal(tmems[0], m.slices[0].tmem)
