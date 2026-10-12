"""Tiled prefill (opentpu/llm/tiled.py, docs/tiled_prefill.md): a run of T rows, every weight
chunk streamed once for all of them and the attention's rows packed MCOLS / G to an MM, against
today's prefill runs over the same tokens on the ISA simulator: the KV cache, every layer block
and the last row's logits bit for bit, where their 4-bit MMs pair alike (PAIR). Tiled prompt
runs (prefill.py's tile mode) of every dense family against today's prompt runs: Qwen3, the
Llama-likes (SmolLM3, Phi-4-mini's shape), LFM2, Qwen3.5."""
import dataclasses
import sys
from pathlib import Path

import numpy as np
import pytest

from opentpu import isa as I
from opentpu.isasim import Machine, board_config
from opentpu.llm import prefill as PF
from opentpu.llm import tiled
from opentpu.llm.qwen3 import Engine, Spec, rope_tables

sys.path.insert(0, str(Path(__file__).parent))

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


# ------------------------------------------------------------------ tiled prompt runs
def _family(case, tmp_path_factory):
    """(W, spec, cfg, engine kwargs) of a tiny model of each family (test_prefill's, and
    test_llama's Llama-likes read back from their checkpoints)."""
    if case in ("smollm3", "phi3"):
        import test_llama as TLL
        from opentpu.llm import load_spec
        from opentpu.llm.qwen3 import load_weights
        torch.manual_seed(0)
        m = {"smollm3": TLL._smollm3, "phi3": TLL._phi3}[case]().float().eval()
        with torch.no_grad():
            for n, p in m.named_parameters():
                if "norm" in n:
                    p.copy_(1 + 0.1 * torch.randn_like(p))
        path = tmp_path_factory.mktemp(case)
        m.save_pretrained(path)
        return load_weights(path), load_spec(path), board_config(DRAM_BYTES=1 << 26), {}
    from test_prefill import _model
    return _model(case)


def _prompt_engines(case, mcols, act, R, tmp_path_factory, pair=None):
    """Two engines of the case's model with prompt runs, MCOLS and ACT_ROWS as given and the
    image's rows act: tiled (a) and today's held to R rows a run (b)."""
    W, spec, cfg, kw = _family(case, tmp_path_factory)
    cfg = dataclasses.replace(cfg, MCOLS=mcols, ACT_ROWS=act)
    if pair is not None:
        cfg = dataclasses.replace(cfg, PAIR=pair)
    a = Engine(spec, W, cap=512, cfg=cfg, rows=act, resident=True, prompt_runs=True, **kw)
    a.tile_runs = True
    b = Engine(spec, W, cap=512, cfg=cfg, rows=act, resident=True, prompt_runs=True, **kw)
    b.tile_runs = False
    b._prompt_rmax = {k: (R, True) for k in range(3)}
    return spec, a, b


def _dram_but_scratch(eng, P):
    """Each slice's DRAM but the runs' own words (test_prefill._drams), the I/O area's logits
    rows (an L run writes its last row's: row T - 1 or R - 1), its input rows (a tiled run's
    spilled x: tiled.Res) and Qwen3.5's per-row scratch gr / on (the last run's rows)."""
    from test_prefill import _drams
    io, n = eng.image.io, eng.image.nbytes
    out = []
    for d in _drams(eng, P):
        d = d[:n].copy()
        d[io["logits"]:io["logits"] + 4 * eng.image.rows * eng.spec.vocab] = 0
        d[io["x"]:io["x"] + 4 * eng.image.rows * eng.spec.hidden] = 0
        if "gr" in io:
            d[io["gr"]:io["gr"] + 4 * 2 * eng.image.nl * eng.image.grows] = 0
            d[io["on"]:io["on"] + 4 * 4 * eng.spec.lin_dv * eng.image.grows] = 0
        out.append(d)
    return out


@pytest.mark.parametrize("case, mcols, P1, P2, R, T, form", [
    ("qwen3", 4, 250, 20, 8, 16, None),     # 2 rows a packed tile (G = 2), across the bucket's
    ("qwen3", 2, 37, 30, 4, 16, None),      # end; MCOLS = G: a row a tile
    ("qwen3", 8, 37, 30, 8, 16, None),      # 4 rows a tile
    ("smollm3", 4, 37, 30, 8, 16, None),    # no q / k norms, layers without RoPE (G = 4)
    ("phi3", 4, 37, 30, 8, 16, None),       # G = 3, RoPE on 96 of 128 dimensions, LongRoPE
    ("phi3", 8, 37, 30, 8, 16, None),       # 2 rows of 3 heads a tile
    ("lfm2", 4, 250, 20, 8, 16, None),      # convolution blocks by channel parts, 64-wide heads
    ("kh8", 4, 250, 20, 4, 12, None),       # Qwen3.5: gated attention, DeltaNet rows (16:
    ("kh16", 4, 37, 30, 4, 12, None),       # TMEM); 16 DeltaNet heads
    ("qwen3", 4, 250, 20, 8, 16, {"spill": True}),     # x spilled while the heads attend
    ("lfm2", 4, 37, 30, 8, 16, {"spill": True}),
    ("kh8", 4, 250, 20, 4, 12, {"spill": True, "drows": 5}),   # DeltaNet in runs of 5, 5, 2
])
def test_tiled_prompt_runs_are_todays(tmp_path_factory, monkeypatch, case, mcols, P1, P2, R, T,
                                      form):
    """Two prompts (a chat's turns, the second from where the first left) through tiled prompt
    runs of T rows and through today's of R: the last logits and every slice's DRAM (KV caches,
    DeltaNet states and windows, LFM2's rings) bit for bit; today's runs at MCOLS 4 do not pair
    (int8 weights). form: the tiled runs' form (tiled.variants), else the first that fits."""
    if form is not None:
        monkeypatch.setattr(tiled, "variants", lambda fam, T, mc: [form])
    r = np.random.default_rng(P1)
    p1, p2 = ([int(t) for t in r.integers(0, 1000, n)] for n in (P1, P2))
    spec, a, b = _prompt_engines(case, mcols, 16, R, tmp_path_factory)
    got = [a.prefill(p1), a.prefill(p2)]
    assert set(a._prompt_tile.values()) == {T}
    assert max(s["rows"] for s in a.stats) == T
    keys = {k[0] for k in a._prompt_progs}
    assert "prompt tile" in keys
    want = [b.prefill(p1), b.prefill(p2)]
    assert max(s["rows"] for s in b.stats) == R
    for x, y in zip(got, want):
        assert np.array_equal(x.view(np.uint32), y.view(np.uint32))
    assert all(np.array_equal(x, y) for x, y in zip(_dram_but_scratch(a, P1 + P2),
                                                    _dram_but_scratch(b, P1 + P2)))


@pytest.mark.parametrize("pair, P1, P2, same", [
    (True, 36, 24, True),       # today's runs of 4 (and 3) rows, the tile's of 8: none pairs
    (True, 250, 20, False),     # today's 2-row runs (at the bucket's end, the prompt's) pair
    (False, 250, 20, True),     # the same without PAIR: bit for bit
])
def test_tiled_prompt_runs_pair_rounding(tmp_path_factory, pair, P1, P2, same):
    """4-bit Qwen3.5: a tiled run and today's differ exactly where one of them pairs (a run of
    at most MCOLS / 2 rows: a paired MM sums each even K block into the odd one first) -- the
    tile's 8 rows never pair, today's 2-row runs do; without PAIR they are the same bits."""
    r = np.random.default_rng(P1)
    p1, p2 = ([int(t) for t in r.integers(0, 1000, n)] for n in (P1, P2))
    _, a, b = _prompt_engines("fp4", 4, 16, 4, tmp_path_factory, pair=pair)
    got = [a.prefill(p1), a.prefill(p2)]
    want = [b.prefill(p1), b.prefill(p2)]
    eq = all(np.array_equal(x, y) for x, y in zip(_dram_but_scratch(a, P1 + P2),
                                                  _dram_but_scratch(b, P1 + P2)))
    eq = eq and all(np.array_equal(x, y) for x, y in zip(got, want))
    assert eq == same
    if not same:
        assert any(s["rows"] <= 2 for s in b.stats)
        assert all(np.abs(x - y).max() < 1e-3 * np.abs(y).max() for x, y in zip(got, want))


def test_tiled_shallow_pipeline(tmp_path_factory, monkeypatch):
    """A tiled run over IMEM at the attention's depth (ATTN_DEPTH: a head's entries leave up to
    ~3 x depth blocks unrolled) takes the same form at tiled.SHALLOW (over at both: the first,
    for the caller's IMEM check: bucket 8's, 4 blocks' code a packed tile at depth 2 against 8
    at 3); its runs are today's bit for bit (a 1030-token prompt: buckets 1-5, the heads' whole
    blocks looped from bucket 4 at depth 2, none at 3)."""
    W, spec, cfg, kw = _family("qwen3-board", tmp_path_factory)
    cfg = dataclasses.replace(cfg, MCOLS=4, ACT_ROWS=16)
    a, b = (Engine(spec, W, cap=2048, cfg=cfg, rows=16, resident=True, prompt_runs=True, **kw)
            for _ in range(2))
    img, imem = a.image, a.image.cfg
    size = lambda: max(map(len, tiled.compile_tile_run(img, 8, 16, "P")[0]))   # noqa: E731
    forms = tiled.variants
    shallow = lambda *x: [dict(v, depth=tiled.SHALLOW) for v in forms(*x)]   # noqa: E731
    n3 = size()
    monkeypatch.setattr(tiled, "variants", shallow)
    n2 = size()
    monkeypatch.setattr(tiled, "variants", forms)
    assert n2 < n3
    monkeypatch.setattr(img, "cfg", dataclasses.replace(imem, IMEM_WORDS=8 * n2))
    assert size() == n2
    monkeypatch.setattr(img, "cfg", dataclasses.replace(imem, IMEM_WORDS=8 * (n2 - 1)))
    assert size() == n3
    monkeypatch.setattr(img, "cfg", imem)
    monkeypatch.setattr(tiled, "variants", shallow)
    a.tile_runs, b.tile_runs = True, False
    b._prompt_rmax = {k: (8, True) for k in range(6)}
    r = np.random.default_rng(5)
    p1, p2 = ([int(t) for t in r.integers(0, 1000, n)] for n in (1030, 20))
    got = [a.prefill(p1), a.prefill(p2)]
    assert set(a._prompt_tile.values()) == {16}
    want = [b.prefill(p1), b.prefill(p2)]
    for x, y in zip(got, want):
        assert np.array_equal(x.view(np.uint32), y.view(np.uint32))
    assert all(np.array_equal(x, y) for x, y in zip(_dram_but_scratch(a, 1050),
                                                    _dram_but_scratch(b, 1050)))


@pytest.mark.parametrize("case, mcols, act, p0, T, form", [
    ("qwen3", 4, 4, 300, 4, None),      # the card's ACT RAM (4 rows): a whole block, then the
    ("lfm2", 4, 16, 200, 16, None),     # masked one; 16 rows: replayed MMs, the convolution's
    ("kh8", 4, 16, 300, 12, None),      # channel parts; Qwen3.5: DeltaNet rows (DSTEP), gated
    ("gemma4-1", 4, 16, 600, 4, None),  # Gemma 4: the sliding window's start-masked block (IMEM:
    ("gemma4-2", 8, 16, 300, 8, None),  # 4 rows in bucket 3); E4B's groups of 4, two rows a tile
    ("kh8", 4, 16, 300, 12, {"spill": True, "drows": 5}),  # x stored and loaded back in place,
])                                                          # DeltaNet in runs of 5, 5, 2
def test_tiled_prompt_run_on_rtl(tmp_path_factory, monkeypatch, have_verilator, case, mcols,
                                 act, p0, T, form):
    """A tiled prompt run (an L run, at a run-time position from the state's words, its tokens
    from out[]) on the RTL with the board's memory path: DRAM and TMEM as the ISA simulator's,
    on the card's MXU (MCOLS 4) with its own ACT RAM of 4 rows and with 16, and on 8 columns;
    form: the run's form (tiled.variants)."""
    from opentpu import rtlsim
    if form is not None:
        monkeypatch.setattr(tiled, "variants", lambda fam, T, mc: [form])
    if case.startswith("gemma4"):
        (W, spec, cfg), kw, cap = _gemma(tmp_path_factory, int(case[-1])), {}, 1024
    else:
        (W, spec, cfg, kw), cap = _family(case, tmp_path_factory), 512
    cfg = dataclasses.replace(cfg, MCOLS=mcols, ACT_ROWS=act)
    eng = Engine(spec, W, cap=cap, cfg=cfg, rows=act, resident=True, prompt_runs=True, **kw)
    eng.tile_runs = True
    toks = [int(t) for t in np.random.default_rng(p0).integers(0, 1000, p0 + act)]
    eng.prefill(toks[:p0])
    assert PF.r_max(eng, p0 // eng.block + 1) == T
    progs, names = PF.programs(eng, p0, T, "L")
    PF.write_tokens(eng, p0, toks[p0:p0 + T])
    g, w = eng.image.lookup["gen"], PF.words(eng, names, p0, T)
    for a, b in PF._ranges(sorted(w)):
        eng.backend.write(0, g["state"] + 4 * a, np.array([w[i] for i in range(a, b)],
                                                          np.float32))
    dram = eng.backend.machine.slices[0].dram.copy()
    m = Machine(cfg, progs, [dram.copy()]).run()
    drams, tmems, _ = rtlsim.run(cfg, progs, [dram.copy()], uarch=rtlsim.BOARD_UARCH, axi=True,
                                 boot=True)
    n = eng.image.nbytes
    assert np.array_equal(drams[0][:n], m.slices[0].dram[:n])
    assert np.array_equal(tmems[0], m.slices[0].tmem)


def _gemma(tmp_path_factory, nkv: int):
    """test_gemma4's tiny Gemma 4 (sliding and global layers, KV-shared ones, a double-wide
    MLP, per-layer inputs) with nkv KV heads: 8 query heads on 1 (two parts of 4 on MCOLS 4)
    or on 2 (E4B's: groups of 4, two rows a tile at MCOLS 8)."""
    import test_gemma4 as TG
    tf = TG.transformers            # (the module test_gemma4 builds with: an engine's imports
    orig = tf.Gemma4TextConfig      # can put another in sys.modules)
    try:
        tf.Gemma4TextConfig = lambda **k: orig(**{**k, "num_key_value_heads": nkv})
        _, W, spec = TG._model(tmp_path_factory.mktemp("gemma4"))
    finally:
        tf.Gemma4TextConfig = orig
    assert spec.kvh(0) == nkv
    return W, spec, TG._cfg()


@pytest.mark.parametrize("mcols, nkv, wf, host, P1, P2, R, T, form", [
    (4, 1, "int8", "0", 250, 20, 4, 8, None),   # two parts of 4 heads a row, across the
    (8, 2, "int8", "0", 250, 20, 4, 8, None),   # bucket's end; groups of 4: two rows a tile
    #                                             (bucket 1: 16 rows, bucket 2 8: IMEM)
    (4, 1, "fp4", "1", 37, 30, 8, 12, None),    # 4-bit layers, the PLE table on the host
    (8, 2, "int8", "0", 760, 20, 4, 8, None),   # the sliding ring's wrap (768 slots), packed
    (8, 1, "int8", "0", 250, 20, 4, 8, "small"),    # x spilled; query parts of 2 heads (the
])                                                  # global layers' 1), 4 (8) rows a tile
def test_tiled_gemma4_prompt_runs_are_todays(tmp_path_factory, monkeypatch, mcols, nkv, wf,
                                             host, P1, P2, R, T, form):
    """Gemma 4's tiled prompt runs (T rows: 16 overflow IMEM or TMEM) against today's prompt
    runs of R rows: the last logits, the layers' DRAM (weights, KV caches, the sliding rings)
    and the next step's logits bit for bit. form "small": x spilled while the heads attend and
    the query parts held to TMEM / 64 words (tiled.QPART)."""
    monkeypatch.setenv("OTPU_PLE_HOST", host)
    if form == "small":
        monkeypatch.setattr(tiled, "variants", lambda fam, T, mc: [{"spill": True}])
        monkeypatch.setattr(tiled, "QPART", 64)
    W, spec, cfg = _gemma(tmp_path_factory, nkv)
    cfg = dataclasses.replace(cfg, MCOLS=mcols, ACT_ROWS=16)
    kw = dict(wformat=wf, head_format="int8" if wf == "fp4" else None)
    r = np.random.default_rng(P1)
    p1, p2 = ([int(t) for t in r.integers(0, 1000, n)] for n in (P1, P2))
    a = Engine(spec, W, cap=1024, cfg=cfg, rows=16, resident=True, prompt_runs=True, **kw)
    a.tile_runs = True
    got = [a.prefill(p1), a.prefill(p2)]
    assert min(a._prompt_tile.values()) == T
    b = Engine(spec, W, cap=1024, cfg=cfg, rows=16, resident=True, prompt_runs=True, **kw)
    b.tile_runs = False
    b._prompt_rmax = {k: (R, True) for k in range(5)}
    want = [b.prefill(p1), b.prefill(p2)]
    for x, y in zip(got, want):
        assert np.array_equal(x.view(np.uint32), y.view(np.uint32))
    img = a.image

    def layers(e):
        return e.backend.machine.slices[0].dram[img.layer0:img.head[0]]
    assert np.array_equal(layers(a), layers(b))
    assert np.array_equal(a.step(5), b.step(5))
    if form == "small":             # (the global layers' 4 heads a row, 8 at MCOLS 8: parts
        tile = a._prompt_progs      # of 1 and 2 heads, 8 and 4 rows a tile)
        assert any(k[0] == "prompt tile" for k in tile)


@pytest.mark.parametrize("how", ["kwarg", "env"])
def test_engine_tile_runs_rows(tmp_path_factory, monkeypatch, how):
    """Engine(tile_runs=True), or OTPU_TILE=1 where it says nothing: its image's I/O rows grow
    to the configuration's ACT_ROWS (an L run of T rows writes row T - 1's logits) and a
    prompt's runs are tiled ones of up to ACT_ROWS rows; without either, neither."""
    W, spec, cfg, kw = _family("qwen3", tmp_path_factory)
    cfg = dataclasses.replace(cfg, MCOLS=4, ACT_ROWS=16)
    monkeypatch.delenv("OTPU_TILE", raising=False)
    off = Engine(spec, W, cap=512, cfg=cfg, rows=8, resident=True, prompt_runs=True, **kw)
    assert off.rows == 8 and not PF.tiling(off)
    if how == "env":
        monkeypatch.setenv("OTPU_TILE", "1")
        eng = Engine(spec, W, cap=512, cfg=cfg, rows=8, resident=True, prompt_runs=True, **kw)
    else:
        eng = Engine(spec, W, cap=512, cfg=cfg, rows=8, resident=True, prompt_runs=True,
                     tile_runs=True, **kw)
    assert eng.rows == eng.image.rows == 16 and PF.tiling(eng)
    toks = [int(t) for t in np.random.default_rng(3).integers(0, 1000, 40)]
    lg = eng.prefill(toks)
    assert [s["rows"] for s in eng.stats] == [16, 16, 8]
    want = off.prefill(toks)
    assert np.array_equal(lg.view(np.uint32), want.view(np.uint32))


def test_reference_key_tells_tiled_runs():
    """tools/qual/refs.py: OTPU_TILE is in a reference's key (env_key: the hashed sources read
    it), so a tiled qual's references are its own."""
    import importlib.util
    import os
    root = Path(__file__).resolve().parents[1]
    sp = importlib.util.spec_from_file_location("qual_refs_t", root / "tools/qual/refs.py")
    refs = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(refs)
    old = os.environ.pop("OTPU_TILE", None)
    try:
        assert "OTPU_TILE" not in refs.env_key()
        os.environ["OTPU_TILE"] = "1"
        assert refs.env_key()["OTPU_TILE"] == "1"
    finally:
        os.environ.pop("OTPU_TILE", None)
        if old is not None:
            os.environ["OTPU_TILE"] = old
