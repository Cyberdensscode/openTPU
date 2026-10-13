"""MiniCPM-V 4.7 on openTPU (docs/minicpmv.md). Its language model is Qwen3.5-MoE: a tiny random
checkpoint laid out as MiniCPM-V's (model_type minicpmv4_7, the language model under
model.language_model.*, an LM head of its own, the vision tower and projector beside it) loads
as qwen35's Spec and weights, without the vision tensors; a vocabulary that is not a multiple
of D (MiniCPM-V's 248,144 = 1938 x 128 + 80) gives the logits of the head padded with zero rows,
bit for bit."""
import json
from dataclasses import replace

import numpy as np
import pytest

from opentpu.llm import load_spec
from opentpu.llm.moe import MoESpec
from opentpu.llm.qwen3 import Engine, LazyWeights, device_config, load_weights
from opentpu.llm.qwen35 import Spec

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
safetensors = pytest.importorskip("safetensors.torch")

KINDS = ("linear", "linear", "attn", "linear", "linear", "attn")
E, K, F = 8, 2, 256
V = 1000 + 80                   # 8 x 128 + 56: the head's last D block is partial


def _text_config(vocab):
    return transformers.Qwen3_5MoeTextConfig(
        hidden_size=256, num_hidden_layers=len(KINDS), num_attention_heads=8,
        num_key_value_heads=2, head_dim=256, vocab_size=vocab,
        layer_types=["full_attention" if k == "attn" else "linear_attention" for k in KINDS],
        linear_num_key_heads=4, linear_num_value_heads=8, linear_key_head_dim=128,
        linear_value_head_dim=128, linear_conv_kernel_dim=4, tie_word_embeddings=False,
        max_position_embeddings=4096, rms_norm_eps=1e-6, num_experts=E,
        num_experts_per_tok=K, moe_intermediate_size=F, shared_expert_intermediate_size=F,
        rope_parameters={"rope_type": "default", "rope_theta": 1e7, "partial_rotary_factor": 0.25,
                         "mrope_interleaved": True, "mrope_section": [11, 11, 10]})


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    """A tiny Qwen3.5-MoE (untied, V-token vocabulary) saved as MiniCPM-V's checkpoint is laid
    out: (the model, its directory)."""
    torch.manual_seed(0)
    hc = _text_config(V)
    m = transformers.Qwen3_5MoeForCausalLM(hc).float().eval()
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "norm" in n:     # zero-centered norms, but the DeltaNet output norm is plain
                p.copy_((1.0 if n.endswith("linear_attn.norm.weight") else 0.0)
                        + 0.1 * torch.randn_like(p))
            elif "experts" in n or "gate" in n:
                p.copy_(0.05 * torch.randn_like(p))
    d = tmp_path_factory.mktemp("minicpmv")
    sd = {(k if k == "lm_head.weight" else k.replace("model.", "model.language_model.", 1)):
          v.contiguous() for k, v in m.state_dict().items()}
    sd["model.vision_tower.encoder.layers.0.mlp.fc1.weight"] = torch.randn(64, 32)
    sd["model.merger.mlp.0.linear_1.weight"] = torch.randn(32, 32)
    safetensors.save_file(sd, str(d / "model.safetensors"))
    tc = hc.to_dict()
    tc["model_type"] = "qwen3_5_moe_text"
    (d / "config.json").write_text(json.dumps({
        "architectures": ["MiniCPMV4_7ForConditionalGeneration"], "model_type": "minicpmv4_7",
        "tie_word_embeddings": False, "text_config": tc, "insert_layer_id": 0,
        "downsample_mode": "16x", "mrope_mode": "canvas",
        "vision_config": {"hidden_size": 128, "intermediate_size": 256, "num_hidden_layers": 2,
                          "num_attention_heads": 2, "image_size": 56, "patch_size": 14,
                          "model_type": "minicpmv4_7_vision"},
        "image_token_id": 248056, "image_start_id": 248078, "image_end_id": 248079,
        "slice_start_id": 248088, "slice_end_id": 248089, "newline_id": 198}))
    return m, d


def test_a_minicpmv_checkpoint_is_a_qwen35_moe(tiny):
    """load_spec takes model_type minicpmv4_7 to qwen35's Spec from its text_config: the MoE,
    an LM head of its own, the int8 embedding a MoE gathers, 1-D RoPE (mRoPE's text case)."""
    m, d = tiny
    s = load_spec(d)
    assert isinstance(s, Spec) and s.moe == MoESpec(E=E, k=K, ffn=F, rule="softmax", shared=F)
    assert (s.vocab, s.tied, s.embed, s.rope_dim, s.lin_nk) == (V, False, "int8", 64, 4)
    assert s.kinds == tuple("attn" if k == "attn" else "linear" for k in KINDS)


@pytest.mark.parametrize("lazy", [False, True])
def test_the_weights_are_the_language_models_alone(tiny, lazy):
    """The loaders give the language model under a text-only checkpoint's names, and none of
    the vision tower's or projector's tensors (docs/minicpmv.md 4.1)."""
    m, d = tiny
    W = LazyWeights(d) if lazy else load_weights(d)
    assert set(W) == set(m.state_dict())
    for k in ("model.embed_tokens.weight", "lm_head.weight", "model.layers.0.mlp.gate.weight"):
        assert np.array_equal(W[k], m.state_dict()[k].numpy())


def _engine(spec, W):
    cfg = device_config(spec, 256, S=1)
    return Engine(spec, W, cap=256, cfg=cfg)


def test_a_partial_head_block_is_the_padded_heads_logits(tiny):
    """The device's logits for a vocabulary of V = 8 x 128 + 56 rows are bit for bit those of
    the same head padded with zero rows to 9 x 128 (the padding's logits 0): the LM head's
    partial last D block, MiniCPM-V's 248,144 = 1938 x 128 + 80, needs no padding of its own.
    Its last real id is checked on its own too. (The language model alone: vision None.)"""
    m, d = tiny
    spec = replace(load_spec(d), vision=None)
    W = dict(load_weights(d).items())
    Vp = -(-V // 128) * 128
    Wp = dict(W)
    for k in ("lm_head.weight", "model.embed_tokens.weight"):
        Wp[k] = np.concatenate([W[k], np.zeros((Vp - V, spec.hidden), np.float32)])
    toks = [int(t) for t in np.random.default_rng(4).integers(0, V, 6)]
    a, b = _engine(spec, W), _engine(replace(spec, vocab=Vp), Wp)
    for t in toks:
        la, lb = a.step(t), b.step(t)
        assert la.shape == (V,) and lb.shape == (Vp,)
        assert np.array_equal(la.view(np.uint32), lb[:V].view(np.uint32))
        assert la[V - 1].view(np.uint32) == lb[V - 1].view(np.uint32)
        assert not np.any(lb[V:])


# =============================================================================== vision (HF 5.18)
def _hf_tiny():
    """A tiny random MiniCPM-V 4.7 of HF's own classes (transformers >= 5.18), fp32."""
    if not hasattr(transformers, "MiniCPMV4_7ForConditionalGeneration"):
        pytest.skip("transformers without minicpmv4_7 (>= 5.18)")
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    import minicpmv_hf
    import tempfile
    d = minicpmv_hf.tiny_checkpoint(Path(tempfile.mkdtemp()))
    m = transformers.MiniCPMV4_7ForConditionalGeneration.from_pretrained(
        d, dtype=torch.float32).eval()
    return m, d


@pytest.fixture(scope="module")
def hf_tiny():
    return _hf_tiny()


@pytest.mark.parametrize("size", [(8, 12), (16, 8), (4, 4)])
def test_vision_reference_matches_hf(hf_tiny, size):
    """The numpy tower, window merger and projector (minicpmv.vision_reference) give HF's
    get_image_features rows for a crop of h x w patches (fp32)."""
    from opentpu.llm import minicpmv as MV
    m, d = hf_tiny
    vs = MV.VisionSpec.from_hf(d)
    W = MV.load_vision(d)
    h, w = size
    rng = np.random.default_rng(h * 100 + w)
    P = rng.uniform(-1, 1, (h * w, 3 * vs.patch ** 2)).astype(np.float32)
    # HF's packing: [1, 3, patch, N * patch], patch l's block at columns l p .. l p + p
    pv = torch.from_numpy(P.reshape(h * w, 3, vs.patch, vs.patch).transpose(1, 2, 0, 3)
                          .reshape(1, 3, vs.patch, h * w * vs.patch))
    with torch.no_grad():
        hf = m.model.get_image_features(pv, torch.tensor([[h, w]])).pooler_output[0].numpy()
    ours = MV.vision_reference(vs, W, P, (h, w))
    assert ours.shape == hf.shape == (h * w // 16, 256)
    assert np.abs(ours - hf).max() < 2e-5 * max(1.0, np.abs(hf).max())


def test_canvas_positions_match_hf(hf_tiny):
    """minicpmv.rope_index gives HF's get_rope_index: text, one image without slices, one with
    a 2 x 3 grid of slices, the image-id prefix and the newline after it, text between."""
    from opentpu.llm import minicpmv as MV
    m, d = hf_tiny
    vs = MV.VisionSpec.from_hf(d)
    P = vs.image_token

    def image(th, tw, grid=None, sh=0, sw=0, idt=(7, 8, 9)):
        ids = list(idt) + [vs.image_start] + [P] * (th * tw // 16) + [vs.image_end]
        sizes = [(th, tw)]
        if grid:
            rows = []
            for r in range(grid[0]):
                rows.append(([vs.slice_start] + [P] * (sh * sw // 16) + [vs.slice_end])
                            * grid[1])
                sizes += [(sh, sw)] * grid[1]
            ids += sum((r + ([vs.newline] if i + 1 < grid[0] else []) for i, r in
                        enumerate(rows)), [])
        return ids, sizes
    cases = []
    a, sa = image(8, 12)
    b, sb = image(16, 8, (2, 3), 8, 12)
    cases.append(([1, 2, 3] + a + [vs.newline, 5, 6] + b + [vs.newline, 9, 10, 11], sa + sb))
    cases.append(([4] * 5 + b + [12, 13], sb))
    cases.append(([1, 2, 3, 4, 5], []))
    for ids, sizes in cases:
        mm = [1 if t == P else 0 for t in ids]
        kw = {}
        if sizes:
            kw = dict(mm_token_type_ids=torch.tensor([mm]), target_sizes=torch.tensor(sizes))
        else:
            kw = dict(mm_token_type_ids=torch.tensor([mm]), target_sizes=torch.zeros(0, 2,
                                                                                 dtype=torch.long))
        hp, hd = m.model.get_rope_index(torch.tensor([ids]), **kw)
        pos, delta = MV.rope_index(ids, sizes, vs)
        assert np.array_equal(pos, hp[:, 0].numpy()), (ids, pos, hp[:, 0])
        assert delta == int(hd[0, 0])


# =============================================================================== vision on the card
def _tiny_vision_weights(vs, rng):
    """Random weights of a tiny vision tower, window merger and projector (load_vision's
    names)."""
    from opentpu.llm import minicpmv as MV
    C, F, W = vs.hidden, vs.ffn, {}

    def lin(n, o, i):
        W[n + ".weight"] = (rng.standard_normal((o, i)) / np.sqrt(i)).astype(np.float32)
        W[n + ".bias"] = (0.1 * rng.standard_normal(o)).astype(np.float32)

    def ln(n, k):
        W[n + ".weight"] = (1 + 0.1 * rng.standard_normal(k)).astype(np.float32)
        W[n + ".bias"] = (0.1 * rng.standard_normal(k)).astype(np.float32)

    def attn(p):
        for n in ("q_proj", "k_proj", "v_proj", "out_proj"):
            lin(p + "self_attn." + n, C, C)
        ln(p + "layer_norm1", C)
    for i in range(vs.layers):
        p = f"{MV.TOWER}encoder.layers.{i}."
        attn(p)
        ln(p + "layer_norm2", C)
        lin(p + "mlp.fc1", F, C)
        lin(p + "mlp.fc2", C, F)
    p = MV.TOWER + "vit_merger."
    attn(p)
    ln(p + "pre_norm", 4 * C)
    lin(p + "linear_1", 4 * F, 4 * C)
    lin(p + "linear_2", C, 4 * F)
    ln(MV.TOWER + "post_layernorm", C)
    p = MV.MERGER + "mlp.0."
    ln(p + "pre_norm", 4 * C)
    lin(p + "linear_1", 4 * C, 4 * C)
    lin(p + "linear_2", vs.out, 4 * C)
    k = vs.patch
    W[MV.TOWER + "embeddings.patch_embedding.weight"] = \
        (rng.standard_normal((C, 3, k, k)) / 24).astype(np.float32)
    W[MV.TOWER + "embeddings.patch_embedding.bias"] = (0.1 * rng.standard_normal(C)).astype(
        np.float32)
    W[MV.TOWER + "embeddings.position_embedding.weight"] = \
        (0.5 * rng.standard_normal((vs.side ** 2, C))).astype(np.float32)
    return W


@pytest.fixture(scope="module")
def tiny_tower():
    """A tiny tower (C 256, 4 heads of 64: padded to D on the card, an MLP of 300: padded to
    its chunks; 3 layers, the window merger after layer 1) on the ISA simulator."""
    from opentpu.isasim import board_config
    from opentpu.llm import minicpmv as MV
    from opentpu.llm.qwen3 import IsaBackend, _Bump
    vs = MV.VisionSpec(hidden=256, heads=4, ffn=300, layers=3, side=5, insert=1, out=256)
    W = _tiny_vision_weights(vs, np.random.default_rng(0))
    cfg = board_config(DRAM_BYTES=1 << 26)
    vi = MV.VisionImage(vs, cfg, _Bump(), nmax=256, out_rows=64)
    img = np.zeros(cfg.DRAM_BYTES, np.uint8)

    def put(addr, a):
        v = np.ascontiguousarray(a).view(np.uint8).reshape(-1)
        img[addr:addr + v.size] = v
    vi.build(W, put)
    return vs, W, vi, IsaBackend(cfg, [img])


@pytest.mark.parametrize("size", [(4, 4), (8, 12), (16, 12), (12, 20)])
def test_the_tower_on_the_card_is_the_reference(tiny_tower, size):
    """A crop of h x w patches through the card's tower, window merger and projector
    (VisionImage.encode: its programs on the ISA simulator, int8 weights and operands) gives
    vision_reference's rows (fp32) to within int8's error: every row's cosine >= 0.999. Sizes
    of one window block (4 x 4), of less and more than one D-token stretch of the window
    attention (96, 192, 240 patches)."""
    from opentpu.kernels.lib import gelu_tanh
    from opentpu.llm import minicpmv as MV
    vs, W, vi, be = tiny_tower
    h, w = size
    P = np.random.default_rng(h * 100 + w).uniform(-1, 1, (h * w, 3 * vs.patch ** 2)).astype(
        np.float32)
    got = vi.encode(be, P, size, proj_gelu=gelu_tanh)
    ref = MV.vision_reference(vs, W, P, size, gelu=MV._gelu_tanh)
    assert got.shape == ref.shape == (h * w // 16, vs.out)
    cos = (got * ref).sum(-1) / np.linalg.norm(got, axis=-1) / np.linalg.norm(ref, axis=-1)
    assert cos.min() >= 0.999, cos.min()
    assert np.abs(got - ref).max() < 0.05 * np.abs(ref).max()


def test_the_card_order_puts_each_window_in_consecutive_rows():
    """card_order: each 4 x 4 block of the grid row-major, its 2 x 2 windows row-major, their
    4 patches row-major. Windows are consecutive rows (windows()' rows, in their order), and
    so are the blocks (the projector's output rows: the h/4 x w/4 grid row-major)."""
    from opentpu.llm import minicpmv as MV
    for h, w in ((4, 4), (8, 12), (12, 8), (16, 20)):
        o = MV.card_order(h, w)
        assert sorted(o) == list(range(h * w))
        assert {tuple(r) for r in o.reshape(-1, 4)} == {tuple(r) for r in MV.windows(h, w)}
        blocks = o.reshape(-1, 16)
        assert np.array_equal(np.sort(blocks, 1), np.sort(MV.windows(h, w, 4), 1))
        for b in blocks:            # a block's windows row-major: (0, 0), (0, 1), (1, 0), (1, 1)
            r, c = b // w, b % w
            assert list(zip(r[::4] // 2 % 2, c[::4] // 2 % 2)) == [(0, 0), (0, 1), (1, 0), (1, 1)]


def test_the_real_towers_programs_fit_the_card():
    """MiniCPM-V 4.7's tower (C 1152, 16 heads of 72, 27 layers) compiles for the card's
    configuration (TMEM, ACT RAM) at crops of 1024, 1056 and 2048 patches, each program within
    IMEM."""
    from opentpu.isasim import board_config
    from opentpu.llm import minicpmv as MV
    from opentpu.llm.qwen3 import _Bump
    cfg = board_config()
    vi = MV.VisionImage(MV.VisionSpec(), cfg, _Bump(), nmax=2048, out_rows=2048)
    for N in (1024, 1056, 2048):
        progs, runs = vi.programs(N)
        assert len(runs) == 1 + 3 * 7 + 4 + 3 * 20 + 1
        for name, (p, _) in progs.items():
            assert len(p) <= cfg.IMEM_WORDS // 8, (N, name, len(p))


def test_the_card_tower_is_hfs(hf_tiny):
    """HF's tiny MiniCPM-V (its own weights, load_vision's reading of them) on the card's
    tower (int8, ISA simulator): HF's get_image_features rows to within int8's error, two
    crops of a 12 x 16 and a 16 x 8 grid."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    import minicpmv_hf
    from opentpu.llm import minicpmv as MV
    m, d = hf_tiny
    vs = MV.VisionSpec.from_hf(d)
    vi, be = minicpmv_hf.card_tower(vs, MV.load_vision(d))
    for h, w in ((12, 16), (16, 8)):
        P = np.random.default_rng(h + w).uniform(-1, 1, (h * w, 3 * vs.patch ** 2)).astype(
            np.float32)
        pv = torch.from_numpy(P.reshape(h * w, 3, vs.patch, vs.patch).transpose(1, 2, 0, 3)
                              .reshape(1, 3, vs.patch, h * w * vs.patch))
        with torch.no_grad():
            hf = m.model.get_image_features(pv, torch.tensor([[h, w]])).pooler_output[0].numpy()
        got = vi.encode(be, P, (h, w))
        cos = (got * hf).sum(-1) / np.linalg.norm(got, axis=-1) / np.linalg.norm(hf, axis=-1)
        assert cos.min() >= 0.999, cos.min()


# =============================================================================== image rows
@pytest.mark.parametrize("em", [False, True])
def test_image_rows_are_layer_zeros_input(tiny, em, monkeypatch):
    """A prompt's image rows (Engine.prefill_layers' vis: each its row of the image's projector
    output, minicpmv.Image's vis) enter layer 0 as those rows: a copy run into the chunk's
    residual rows (em: expert-major's records) and an embedded layer run. Image rows whose
    values an int8 embedding row holds exactly (each D block q 2^-7 with a 127 in it) give
    the logits, bit for bit, of the same prompt with text ids whose (host-gathered) embedding
    rows are those values; then three decode steps too. 20 image rows from vis row 5, after 9
    text rows, in chunks of 24 rows (the image crosses a chunk), runs of 2 rows."""
    from opentpu.llm import qwen35 as Q35
    monkeypatch.setattr(Q35, "PREFILL_CHUNK", 24)
    m, d = tiny
    spec = replace(load_spec(d), vision_patches=128, vision_rows=64)
    rng = np.random.default_rng(3)
    W = dict(load_weights(d).items())
    W.update(_tiny_vision_weights(spec.vision, rng))
    n, at, v0 = 20, 9, 5
    q = rng.integers(-127, 128, (n, spec.hidden))
    q[:, ::128] = 127
    rows = (q * 2.0 ** -7).astype(np.float32)
    fake = list(range(V - 5 - n, V - 5))       # ids whose embedding rows are the image rows
    W["model.embed_tokens.weight"] = W["model.embed_tokens.weight"].copy()
    W["model.embed_tokens.weight"][fake] = rows
    toks = [int(t) for t in rng.integers(0, V - 40, 31)]
    cfg = device_config(spec, 256, rows=1, lookup=True, S=1, experts=2 * K)
    kw = dict(cap=256, cfg=cfg, rows=1, resident=True, experts=2 * K, layer_major=2)
    if em:
        kw["expert_major"] = True
    a, b = Engine(spec, W, **kw), Engine(spec, W, **kw)
    assert a.image.vision is not None and a.expert_major == em
    a.backend.write(0, a.image.vision.work["vis"] + 4 * v0 * spec.hidden, rows)
    vis = [-1] * at + list(range(v0, v0 + n)) + [-1] * (len(toks) - at)
    la = a.prefill_layers(toks[:at] + [0] * n + toks[at:], vis=vis)
    lb = b.prefill_layers(toks[:at] + fake + toks[at:])
    assert ("v", 2) in a._layer_runs and not any(k[0] == "v" for k in b._layer_runs)
    assert np.array_equal(la.view(np.uint32), lb.view(np.uint32))
    t = int(np.argmax(la))
    for _ in range(3):
        ga, gb = a.step(t), b.step(t)
        assert np.array_equal(ga.view(np.uint32), gb.view(np.uint32))
        t = int(np.argmax(ga))


def test_mrope_rows_gather_the_table():
    """mrope_rows: frequency i of a row is the table's entry at the position of section i % 3
    (t, h, w): the rows of a text position are the table's rows bit for bit, an image row's
    entries each the table's at its own section's position."""
    from opentpu.llm import minicpmv as MV
    rng = np.random.default_rng(0)
    cos, sin = rng.standard_normal((2, 50, 32)).astype(np.float32)
    pos = np.array([[3, 3, 7], [3, 4, 9], [3, 5, 11]])
    c, s_ = MV.mrope_rows((cos, sin), pos)
    assert np.array_equal(c[0], cos[3]) and np.array_equal(s_[0], sin[3])
    for j in (1, 2):
        for i in range(32):
            assert c[j, i] == cos[pos[i % 3, j], i] and s_[j, i] == sin[pos[i % 3, j], i]
    with pytest.raises(ValueError):
        MV.mrope_rows((cos, sin), np.array([[50], [0], [0]]))
