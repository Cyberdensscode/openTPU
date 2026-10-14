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


def test_a_null_eos_id_is_the_default(tiny, tmp_path):
    """A config whose eos_token_id is null (the tiny checkpoints': HF's default) gives the
    language model's end tokens, no None among them (otpu-serve's stops refused one)."""
    m, d = tiny
    top = json.loads((d / "config.json").read_text())
    top["text_config"]["eos_token_id"] = None
    (tmp_path / "config.json").write_text(json.dumps(top))
    assert load_spec(tmp_path).eos == (248046, 248044)
    top["text_config"]["eos_token_id"] = [248046, None]
    (tmp_path / "config.json").write_text(json.dumps(top))
    assert load_spec(tmp_path).eos == (248046,)


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


@pytest.mark.parametrize("size", [(8, 12), (16, 12)])
def test_the_odd_stride_residual_gives_the_same_rows(tiny_tower, size, monkeypatch):
    """vit_out_mlp's residual at an odd row stride (its default: W_down's drain at the array's
    rate) and at its DRAM stride (one LD, the layout before) give the tower's rows bit for bit:
    the same arithmetic in another TMEM layout."""
    from opentpu import isa as I
    from opentpu.kernels import vit as VK
    from opentpu.kernels.lib import gelu_tanh
    vs, W, vi, be = tiny_tower
    h, w = size
    P = np.random.default_rng(7).uniform(-1, 1, (h * w, 3 * vs.patch ** 2)).astype(np.float32)
    lds = lambda: sum(i.op == I.LD for i in vi.programs(h * w)[0]["out"][0])
    n_odd = lds()
    odd = vi.encode(be, P, size, proj_gelu=gelu_tanh)
    real = VK.vit_out_mlp

    class Dense:
        def trace(self, cfg, sid, bound):
            return real.trace(cfg, sid, {**bound, "odd": False})
    monkeypatch.setattr(VK, "vit_out_mlp", Dense())
    assert lds() < n_odd
    dense = vi.encode(be, P, size, proj_gelu=gelu_tanh)
    assert np.array_equal(odd.view(np.uint32), dense.view(np.uint32))


@pytest.mark.parametrize("fmt", ["int8", "attn=fp4,mlp=fp4,wm=fp4,proj=fp4"])
def test_the_piece_table_is_the_static_layout(fmt):
    """The pieced weights' table (kernels/pieces.py): the tower's rows with every piece address
    read from the table, the pieces resident, and with the pieces staged at scattered
    slot-like addresses (shuffled, as an expert server's victims fall), are the rows of the
    same pieces at static addresses, bit for bit: only the address source changes. Pieces of
    at most 300 KB (160 KB at fp4): the window merger's 512-row W1 chunks span two pieces (one
    MM a piece, each writing its columns), several W2 column parts share a piece."""
    from opentpu.isasim import board_config
    from opentpu.kernels.lib import gelu_tanh
    from opentpu.llm import minicpmv as MV
    from opentpu.llm.qwen3 import IsaBackend, _Bump
    vs = MV.VisionSpec(hidden=256, heads=4, ffn=300, layers=3, side=5, insert=1, out=256)
    rng = np.random.default_rng(8)
    W = _tiny_vision_weights(vs, rng)
    cfg = board_config(DRAM_BYTES=1 << 27)
    size = (8, 12)
    P = rng.uniform(-1, 1, (96, 3 * vs.patch ** 2)).astype(np.float32)
    outs = []
    for mode in ("static", "table", "staged"):
        vi = MV.VisionImage(vs, cfg, _Bump(), nmax=256, out_rows=64, fmt=fmt,
                            max_piece=300_000 if fmt == "int8" else 160_000,
                            resident=mode != "staged",
                            table=mode != "static")
        img = np.zeros(cfg.DRAM_BYTES, np.uint8)

        def put(addr, a):
            v = np.ascontiguousarray(a).view(np.uint8).reshape(-1)
            img[addr:addr + v.size] = v
        vi.build(W, put)
        if mode == "staged":
            pcs = vi.piece_bytes(W)
            pitch = -(-max(len(p) for p in pcs) // 4096) * 4096
            base = -(-vi.end // 4096) * 4096
            at = base + rng.permutation(len(pcs)) * pitch
            for a, p in zip(at, pcs):
                put(int(a), p)
            put(vi.table, at.astype(np.uint32))
            vi.set_bound(base, base + (len(pcs) - 1) * pitch)
        outs.append(vi.encode(IsaBackend(cfg, [img]), P, size, proj_gelu=gelu_tanh))
    wm = vi.mmats["wm"]
    assert len(wm["w1"].groups) > vi.FW // vi.WCHUNK
    assert any(b - a > 1 for a, b in wm["w2"].groups + vi.lmats["w2"].groups)
    for got in outs[1:]:
        assert np.array_equal(got.view(np.uint32), outs[0].view(np.uint32))


@pytest.mark.parametrize("pool", [False, True])
def test_staged_vision_is_the_resident_towers(tiny, pool, tmp_path):
    """Staged vision weights (Spec.vision_staged, docs/minicpmv.md 4.2): the pieces the
    programs read are the pool's entries past the experts, placed in expert slots by the
    expert server for an encode (its victims the least used experts, every slot warm here),
    their table written, the slots released after it. Two images' crops give the resident
    tower's rows bit for bit; the server's slots are each layer's again, and a prompt's
    prefill (the evicted experts missing) and two decode steps give the resident engine's
    logits bit for bit. pool: the entries in a pool file (the split format, each packed on
    its first use, as on the card host) instead of host RAM."""
    from types import SimpleNamespace
    from opentpu.llm import minicpmv as MV
    m, d = tiny
    spec = replace(load_spec(d), vision_patches=128, vision_rows=64)
    assert spec.vision_staged                   # (the default)
    W = dict(load_weights(d).items())
    W.update(_tiny_vision_weights(spec.vision, np.random.default_rng(7)))
    cfg = device_config(spec, 256, rows=1, lookup=True, S=1, experts=E)
    kw = dict(cap=256, cfg=cfg, rows=1, resident=True, experts=E, layer_major=2)
    a = Engine(replace(spec, vision_staged=False), W, **kw)
    b = Engine(spec, W, **(dict(kw, pool_file=tmp_path / "pool.bin") if pool else kw))
    with pytest.raises(ValueError, match="staged vision: .* pieces"):   # (2 k slots a layer)
        spec.image(device_config(spec, 256, rows=1, lookup=True, S=1, experts=2 * K), 256, 1, 1,
                   experts=2 * K).check_room()
    va, vb = a.image.vision, b.image.vision
    assert va.resident and not vb.resident and vb.max_piece <= b.image.offload.slot_bytes
    assert b.image.pool_extra[0] == vb.npieces and a.image.pool_extra is None
    rng = np.random.default_rng(9)
    ims = [SimpleNamespace(patches=[rng.uniform(-1, 1, (h * w, 3 * 14 * 14)).astype(
        np.float32)], sizes=[(h, w)], tokens=h * w // 16) for h, w in ((8, 8), (4, 12))]
    rows = []
    for e in (a, b):
        n = MV.Vision(e).encode(ims)
        rows.append(e.backend.read(0, e.image.vision.work["vis"], 4 * n * spec.vision.out))
    assert n == 7 and np.array_equal(rows[0], rows[1])
    srv = b.server
    if pool:                            # (the pieces the programs read packed, in the file)
        pf = srv.pool_file
        G = srv.L.layers * srv.L.E
        assert pf.split and len(pf.packed) == G + vb.npieces
        assert all(pf.packed[G + i] for i in vb.used())
    assert srv.placed == {} and srv.place_victims > 0
    assert srv.place_bytes >= sum(vb.piece_sizes()[i] for i in vb.used())
    assert all(len(srv.lru[j]) + len(srv.free[j]) == E for j in range(srv.L.layers))
    toks = [int(t) for t in rng.integers(0, V - 40, 21)]
    la, lb = a.prefill_layers(toks), b.prefill_layers(toks)
    assert np.array_equal(la.view(np.uint32), lb.view(np.uint32))
    t = int(np.argmax(la))
    for _ in range(2):
        ga, gb = a.step(t), b.step(t)
        assert np.array_equal(ga.view(np.uint32), gb.view(np.uint32))
        t = int(np.argmax(ga))


def test_streamed_engine_opens_staged_vision_past_its_probe(tiny):
    """streamed.engine's own path with staged vision (the default): its probe image (the
    layout alone, k slots a layer: no room past k for the pieces) builds, fit_experts sizes the
    engine's slots, the engine's image has the room (Image.check_room) and a Vision opens on
    it; an engine of k slots a layer is refused when it opens. Session mcpv3's first run
    (2026-10-13) died at the probe: the room was checked in every image's constructor."""
    from opentpu.host import streamed as ST
    from opentpu.isasim import board_config
    from opentpu.llm import minicpmv as MV
    m, d = tiny
    spec = load_spec(d)
    assert spec.vision_staged
    W = dict(load_weights(d).items())
    W.update(_tiny_vision_weights(spec.vision, np.random.default_rng(7)))
    cfg = board_config(DRAM_BYTES=1 << 26)
    kw = dict(wformat="int8", head_format=None)  # (int8 slots: the tiny tower's units fit)
    pr = ST.probe(spec, cfg, 256, lookup=True, **kw)
    assert not pr.vision.resident and {n for _, n in pr.offload.slots} == {K}
    with pytest.raises(ValueError, match="staged vision: .* pieces, 0 expert slots past k"):
        pr.check_room()
    eng = ST.engine(d, cfg, "isa", 256, None, W=W, vision_warm=False, **kw)
    assert not eng.image.vision.resident and min(n for _, n in eng.image.offload.slots) > K
    MV.Vision(eng)
    with pytest.raises(ValueError, match="staged vision: .* pieces"):
        ST.engine(d, cfg, "isa", 256, None, experts=K, W=W, vision_warm=False, **kw)


def test_the_real_staged_layout_has_room_at_the_cards_slots(tmp_path):
    """MiniCPM-V 4.7's own layout (its config: Qwen3.5-35B-A3B's language model, SigLIP2's
    tower; the family's formats, chat's 2,048-token cache) on the card's 4 GiB, as
    streamed.engine opens it: the probe image builds at k slots a layer, the slots that fit
    (fit_experts: 36 a layer, mcpv2's) give the engine's image the room for every staged piece,
    and at k slots a layer they do not fit. Layouts only, no weights."""
    from opentpu.host import streamed as ST
    from opentpu.isasim import board_config
    tc = transformers.Qwen3_5MoeTextConfig(
        hidden_size=2048, num_hidden_layers=40, num_attention_heads=16, num_key_value_heads=2,
        head_dim=256, vocab_size=248144, full_attention_interval=4, attn_output_gate=True,
        linear_num_key_heads=16, linear_num_value_heads=32, linear_key_head_dim=128,
        linear_value_head_dim=128, linear_conv_kernel_dim=4, tie_word_embeddings=False,
        max_position_embeddings=262144, rms_norm_eps=1e-6, num_experts=256,
        num_experts_per_tok=8, moe_intermediate_size=512, shared_expert_intermediate_size=512,
        rope_parameters={"rope_type": "default", "rope_theta": 1e7, "partial_rotary_factor": 0.25,
                         "mrope_interleaved": True, "mrope_section": [11, 11, 10]}).to_dict()
    tc["model_type"] = "qwen3_5_moe_text"
    (tmp_path / "config.json").write_text(json.dumps({
        "architectures": ["MiniCPMV4_7ForConditionalGeneration"], "model_type": "minicpmv4_7",
        "tie_word_embeddings": False, "text_config": tc, "insert_layer_id": 6,
        "downsample_mode": "16x", "mrope_mode": "canvas",
        "vision_config": {"hidden_size": 1152, "intermediate_size": 4304,
                          "num_hidden_layers": 27, "num_attention_heads": 16, "image_size": 980,
                          "patch_size": 14, "model_type": "minicpmv4_7_vision"},
        "image_token_id": 248056, "image_start_id": 248078, "image_end_id": 248079,
        "slice_start_id": 248088, "slice_end_id": 248089, "newline_id": 198}))
    spec = load_spec(tmp_path)
    assert spec.vision_staged and spec.moe.k == 8 and spec.vision.layers == 27
    wformat, head, formats = ST.defaults(spec)
    spec = replace(spec, formats=formats)
    cfg = board_config(DRAM_BYTES=1 << 32, MCOLS=4, ACT_ROWS=16, PAIR=True, DSTEP=True,
                       STREAM=True)
    kw = dict(wformat=wformat, head_format=head, lookup=True)
    pr = ST.probe(spec, cfg, 2048, **kw)
    n = ST.fit_experts(spec, cfg, 2048, pr)
    assert n == 36
    with pytest.raises(ValueError, match="staged vision: .* pieces, 0 expert slots past k"):
        pr.check_room()
    spec.image(cfg, 2048, 1, 1, experts=n, **kw).check_room()


def test_a_pool_takes_the_vision_entries_past_its_experts(tmp_path):
    """open_pool's extra entries (staged vision's pieces, pool_key's "vision"): an experts-only
    pool grows by them, its experts kept and they not packed; a run without them opens it as
    its experts'; the same vision key keeps them; another one replaces them (not packed)."""
    from opentpu.host.offload import Layout
    from opentpu.llm import moe as MO
    lay = Layout.build(4096, 4, 2, (2, 2), 4096)
    key = {"key": 2, "numerics": 1, "quant": "q", "format": "fp4", "D": 128,
           "layout": {"layers": 2}, "checkpoint": None}
    p = tmp_path / "pool.bin"
    n = 8

    def opened(k, extra=0):
        pf = MO.open_pool(lay, p, key=k, extra=extra)
        if pf.warm_t is not None:
            pf.warm_t.join()
        return pf
    pf = opened(key)
    pf.arr[:] = np.arange(n, dtype=np.uint8)[:, None] + 1
    pf.packed[:] = 1
    pf.arr.flush()
    pf.packed.flush()
    vk = {"entries": 3, "format": "int8", "pieces": "a"}
    pf = opened({**key, "vision": vk}, 3)
    assert p.stat().st_size == 11 * 4096 and list(pf.packed) == [1] * n + [0] * 3
    assert all(pf.arr[g][0] == g + 1 for g in range(n)) and not pf.arr[n:].any()
    pf.arr[n:] = 77
    pf.packed[n:] = 1
    pf.arr.flush()
    pf.packed.flush()
    pf = opened(key)                    # a run without vision: its experts
    assert pf.arr.shape[0] == n and p.stat().st_size == 11 * 4096
    assert json.loads((tmp_path / "pool.bin.key").read_text())["vision"] == vk
    pf = opened({**key, "vision": vk}, 3)
    assert list(pf.packed) == [1] * 11 and pf.arr[9][0] == 77
    vk2 = {"entries": 2, "format": "attn=fp4", "pieces": "b"}
    pf = opened({**key, "vision": vk2}, 2)
    assert p.stat().st_size == 10 * 4096 and list(pf.packed) == [1] * n + [0] * 2
    assert not pf.arr[n:].any() and pf.arr[3][0] == 4
    with pytest.raises(ValueError):     # another experts' key is refused as before
        opened({**key, "format": "int8", "vision": vk2}, 2)


def test_pack_pool_keys_the_experts_alone(tiny, tmp_path, monkeypatch):
    """pack_pool.py on MiniCPM-V (whose image stages its vision by default: pool_key's "vision"
    and pool_extra): init writes the experts' key alone, as the file holds the experts alone;
    the first staged open appends the vision entries (open_pool's extra) and keeps the experts
    packed since. With the vision key on an experts-only file, pool_entries would count the
    entries in it, and open_pool would take the file for a new pool and truncate it."""
    import importlib.util
    import sys
    from pathlib import Path
    from opentpu.isasim import board_config
    from opentpu.llm import moe as MO
    _, d = tiny
    pool = tmp_path / "pool.bin"
    tool = Path(__file__).resolve().parents[1] / "tools" / "offload" / "pack_pool.py"
    PP = importlib.util.module_from_spec(importlib.util.spec_from_file_location("pp", tool))
    PP.__spec__.loader.exec_module(PP)
    # (pack_pool's image, the tiny tower's patches cut so that its pieces fit the slots)
    spec = replace(load_spec(d), vision_patches=128, vision_rows=64)
    img = spec.image(replace(board_config(), DRAM_BYTES=1 << 40), 1024, 1, 1, "int8", "int8",
                     lookup=True, experts=spec.moe.k)
    monkeypatch.setattr(PP, "image", lambda model, wformat, mtp=False: img)
    monkeypatch.setattr(sys, "argv", ["pack_pool.py", str(d), str(pool), "init"])
    PP.main()
    lay, (extra, vk) = img.offload, img.pool_extra
    n = lay.layers * lay.E
    key = json.loads((tmp_path / "pool.bin.key").read_text())
    assert "vision" not in key and MO.pool_entries(pool, n) == n
    assert key == {k: v for k, v in MO.pool_key(img, str(d)).items() if k != "vision"}
    pf = MO.open_pool(lay, pool, key=key)              # a pack worker's expert
    pf.warm_t.join()
    pf.arr[3] = 9
    pf.packed[3] = 1
    pf.arr.flush()
    pf.packed.flush()
    del pf
    pf = MO.open_pool(lay, pool, key=MO.pool_key(img, str(d)), extra=extra)
    pf.warm_t.join()
    assert pool.stat().st_size == (n + extra) * lay.slot_bytes
    assert pf.packed[3] == 1 and (pf.arr[3] == 9).all() and not pf.packed[n:].any()
    assert json.loads((tmp_path / "pool.bin.key").read_text())["vision"] == vk


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


def test_the_towers_mms_drain_at_the_arrays_rate():
    """Every MM of the real tower's programs drains at its array's rate (compiler.slow_drains:
    where the array takes fewer cycles a column than its rows, their results are in distinct
    TMEM banks). W_down's ACC into the residual (vit_out_mlp: KB 2, 8 cycles a column for 16
    rows) needs the odd-stride tile it is loaded into: at its DRAM stride (C = 1152, a multiple
    of the banks) W_down's MMs ran at 40% of their pops (docs/minicpmv.md 6). vit_embed's (KB 5)
    keeps the dense stride and its one LD."""
    from conftest import assert_drains_keep_up
    from opentpu.isasim import board_config
    from opentpu.llm import minicpmv as MV
    from opentpu.llm.qwen3 import _Bump
    cfg = board_config()
    vi = MV.VisionImage(MV.VisionSpec(), cfg, _Bump(), nmax=256, out_rows=16)
    P, _ = vi.programs(256)
    assert assert_drains_keep_up([p for p, _ in P.values()], cfg) > 50


@pytest.mark.parametrize("staged", [False, True])
def test_the_towers_wait_entries_are_the_exact_passes(staged):
    """The real tower's programs at 256 patches, resident and staged (the pieces' addresses
    from the table, RLD-bounded): waits' memoized pass (wait_counts' default) gives the exact
    pass's entries."""
    from opentpu import waits as WC
    from opentpu.isasim import board_config
    from opentpu.llm import minicpmv as MV
    from opentpu.llm.qwen3 import _Bump
    cfg = board_config()
    vi = MV.VisionImage(MV.VisionSpec(), cfg, _Bump(), nmax=256, out_rows=16, resident=not staged)
    if staged:
        vi.set_bound(1 << 28, 1 << 30)
    P, _ = vi.programs(256)
    for name, (p, ra) in P.items():
        a = WC.wait_counts(p, cfg.D, cfg.S, run_args=ra, memo=False)
        assert np.array_equal(a, WC.wait_counts(p, cfg.D, cfg.S, run_args=ra)), name


def test_the_warm_counts_are_the_processors():
    """vision_warm.counts: every crop's patch count the processor gives (minicpmv.crops: the
    thumbnail and the slices) of images of odd sizes at 1 and 9 slices is among them, each a
    multiple of 16 (two 2 x 2 merges), 1,008 and 1,024 among the four most frequent."""
    from opentpu.host import vision_warm as VW
    from opentpu.llm import minicpmv as MV
    vs = MV.VisionSpec()
    ns = VW.counts(vs, step=100)
    assert all(n % 16 == 0 for n in ns) and {1008, 1024} <= set(ns[:4])
    for H, W in ((300, 500), (1080, 1920), (3000, 400), (777, 777), (250, 3900)):
        for ms in (1, 9):
            cs, _ = MV.crops(torch.zeros((3, H, W), dtype=torch.uint8), vs, ms)
            for c in cs:
                assert (c.shape[-2] // vs.patch) * (c.shape[-1] // vs.patch) in ns, (H, W, ms)


def test_warm_entries_are_found_by_the_first_start(tiny, tmp_path, monkeypatch):
    """vision_warm.warm: the tower's programs at a patch count traced and their wait entries
    computed into the program cache's disk, where a first start's (BoardBackend._entries:
    waits.entries of the assembled words alone) finds every one, none computed; a second warm
    finds its marker and computes nothing; with an entry gone it computes them again. start()
    leaves a backend that takes no entries (the ISA simulator's) alone."""
    from types import SimpleNamespace
    from opentpu import isa as I
    from opentpu import waits as WC
    from opentpu.compiler import Words
    from opentpu.host import vision_warm as VW
    monkeypatch.setenv("OTPU_PROG_CACHE", str(tmp_path))
    m, d = tiny
    spec = replace(load_spec(d), vision_patches=128, vision_rows=64)
    cfg = device_config(spec, 256, rows=1, lookup=True, S=1, experts=E)
    image = spec.image(cfg, 256, 1, 1, experts=E)
    layout = ("test", spec, cfg)
    assert VW.warm(image, layout, 64)
    WC._mem.clear()
    c0, d0 = WC.stats["computed"], WC.stats["disk"]
    P, _ = image.vision.programs(64)
    for p, ra in P.values():
        WC.entries(Words.of(I.assemble(p), ra), cfg.D, cfg.S)
    assert WC.stats["computed"] == c0 and WC.stats["disk"] == d0 + len(P)
    assert not VW.warm(image, layout, 64)
    next(tmp_path.glob("*/*.npz")).unlink()
    assert VW.warm(image, layout, 64)
    assert VW.start(SimpleNamespace(image=image, backend=SimpleNamespace())) is None
    # the process (spawned: the engine's arguments pickled, opentpu imported anew); the
    # processor's counts exceed this work area's 128 patches, so it builds the image and ends
    p = VW.start(SimpleNamespace(image=image, backend=SimpleNamespace(waitc=True), spec=spec,
                                 cfg=cfg, cap=256, batch=1, rows=1, _image_kw={"experts": E},
                                 layout=layout))
    p.join(120)
    assert p.exitcode == 0


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
    spec = replace(load_spec(d), vision_patches=128, vision_rows=64,
                   vision_staged=False)         # (2 k slots a layer: too few to stage)
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



def test_mtp_image_rows_decode_as_plain(tiny, monkeypatch):
    """MiniCPM-V with the MTP drafter (its base Qwen3.5's mtp.* transplanted, docs/offload.md
    15.6): a prompt's image rows through MTPDecoder.prefill's vis (the layer-by-layer prompt's
    copy runs; the MTP layer's input of an image token is its placeholder's embedding row), then
    the MTP loop on the device: the tokens are plain greedy decode's of the same prompt with
    text ids whose embedding rows are the image rows (test_image_rows_are_layer_zeros_input's),
    whatever the drafts. 20 image rows after 9 text rows, chunks of 24 rows."""
    from opentpu.llm import qwen35 as Q35
    from opentpu.llm.mtp import MTPDecoder, MTPStats, mtp_engine
    from test_mtp_moe import _moe_mtp_weights
    monkeypatch.setattr(Q35, "PREFILL_CHUNK", 24)
    m, d = tiny
    spec = replace(load_spec(d), vision_patches=128, vision_rows=64,
                   vision_staged=False)         # (2 k slots a layer: too few to stage)
    rng = np.random.default_rng(3)
    W = dict(load_weights(d).items())
    W.update(_tiny_vision_weights(spec.vision, rng))
    n, at, v0, N = 20, 9, 5, 10
    q = rng.integers(-127, 128, (n, spec.hidden))
    q[:, ::128] = 127
    rows = (q * 2.0 ** -7).astype(np.float32)
    fake = list(range(V - 5 - n, V - 5))       # ids whose embedding rows are the image rows
    W["model.embed_tokens.weight"] = W["model.embed_tokens.weight"].copy()
    W["model.embed_tokens.weight"][fake] = rows
    W = _moe_mtp_weights(W, spec)
    toks = [int(t) for t in rng.integers(0, V - 40, 31)]
    cfg = device_config(spec, 256, rows=1, lookup=True, S=1, experts=2 * K)
    want = Engine(spec, W, cap=256, cfg=cfg, rows=1, resident=True,
                  experts=2 * K).generate(toks[:at] + fake + toks[at:], max_new=N)
    cfg = device_config(replace(spec, mtp=True), 256, rows=2, lookup=True, S=1, experts=6)
    eng = mtp_engine(spec, W, cap=256, cfg=cfg, rows=2, experts=6, layer_major=2)
    dec = MTPDecoder(eng)
    assert dec.by_layers and eng.image.vision is not None
    eng.backend.write(0, eng.image.vision.work["vis"] + 4 * v0 * spec.hidden, rows)
    vis = [-1] * at + list(range(v0, v0 + n)) + [-1] * (len(toks) - at)
    st = MTPStats()
    a0, d0 = dec.prefill(toks[:at] + [0] * n + toks[at:], st, vis=vis)
    assert ("v", 2) in eng._layer_runs and any(k[0] == "m" for k in eng._layer_runs)
    assert dec.loop_card(a0, d0, N, stop=[], st=st).tokens == want


def test_image_rows_run_in_expert_major_tiles(tiny, monkeypatch):
    """MiniCPM-V's image (Qwen3.5's Image, subclassed) takes the tiled expert-major layer runs
    (tiled.layer_family by its base class; docs/tiled_prefill.md 14): a prompt with 20 image
    rows from vis among 31 text rows, expert-major, its layer runs in tiles of 8 rows (layer 0's
    image rows from vis untiled, 2 a run) against today's runs of 2 rows: the logits and three
    decode steps, bit for bit."""
    from opentpu.llm import qwen35 as Q35
    from opentpu.llm import tiled as TL
    monkeypatch.setattr(Q35, "PREFILL_CHUNK", 24)
    monkeypatch.setenv("OTPU_TILE", "1")
    m, d = tiny
    spec = replace(load_spec(d), vision_patches=128, vision_rows=64, vision_staged=False)
    rng = np.random.default_rng(4)
    W = dict(load_weights(d).items())
    W.update(_tiny_vision_weights(spec.vision, rng))
    n, at, v0 = 20, 9, 5
    rows = (0.5 * rng.standard_normal((n, spec.hidden))).astype(np.float32)
    toks = [int(t) for t in rng.integers(0, V - 40, 31)]
    cfg = device_config(spec, 256, rows=1, lookup=True, S=1, experts=2 * K)
    kw = dict(cap=256, cfg=cfg, rows=1, resident=True, experts=2 * K, layer_major=2,
              expert_major=True)
    a = Engine(spec, W, tile_runs=True, tile_rows=8, **kw)
    b = Engine(spec, W, tile_runs=False, **kw)
    assert TL.layer_family(a.image) == "qwen35" and a.layer_tile and not b.layer_tile
    for e in (a, b):
        e.backend.write(0, e.image.vision.work["vis"] + 4 * v0 * spec.hidden, rows)
    vis = [-1] * at + list(range(v0, v0 + n)) + [-1] * (len(toks) - at)
    prompt = toks[:at] + [0] * n + toks[at:]
    la, lb = a.prefill_layers(prompt, vis=vis), b.prefill_layers(prompt, vis=vis)
    tiles = [k for k in a._layer_runs if isinstance(k, tuple) and k[-1] == "tile"]
    assert max(k[2] for k in tiles) == 8 and ("v", 2) in a._layer_runs
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


def test_an_image_cache_hit_keeps_the_position_table(tiny, tmp_path, monkeypatch):
    """The vision side's host table (the patch position embedding, which host_rows gathers)
    is an entry of an attribute of the image's own (vis_tables): an image cache hit, which
    skips the build, puts it back (opentpu/imgcache.py keeps what a build leaves on the image
    and its dict attributes' entries, not on the objects under them)."""
    from opentpu import imgcache as IC
    monkeypatch.setenv("OTPU_IMG_CACHE", str(tmp_path / "img"))
    monkeypatch.setenv("OTPU_IMG_CACHE_FREE_GB", "0")
    m, d = tiny
    ck = tmp_path / "ck"
    ck.mkdir()
    (ck / "config.json").write_text((d / "config.json").read_text())
    spec = replace(load_spec(ck), vision_patches=128, vision_rows=64,
                   vision_staged=False)         # (2 k slots a layer: too few to stage)
    vw = _tiny_vision_weights(spec.vision, np.random.default_rng(4))
    sd = safetensors.load_file(str(d / "model.safetensors"))
    sd.update({k: torch.from_numpy(v) for k, v in vw.items()})
    safetensors.save_file(sd, str(ck / "model.safetensors"))
    cfg = device_config(spec, 256, rows=1, lookup=True, S=1, experts=2 * K)
    P = np.random.default_rng(5).standard_normal((64, 3 * 14 * 14)).astype(np.float32)
    got = []
    for want in ("miss", "hit"):
        e = Engine(spec, LazyWeights(ck), cap=256, cfg=cfg, rows=1, resident=True,
                   experts=2 * K, layer_major=2)
        assert IC.last[0] == want
        got.append(e.image.vision.host_rows(P, (8, 8)))
    pos = vw["model.vision_tower.embeddings.position_embedding.weight"]
    assert np.array_equal(e.image.vis_tables["pos"], pos)
    for a, b in zip(*got):
        assert np.array_equal(a, b)


def test_the_card_rope_rows_are_hfs_mrope(tiny):
    """Vision.positions writes the device's RoPE table so that slot s holds HF's interleaved
    M-RoPE cos / sin (Qwen3_5MoeTextRotaryEmbedding) at the sequence's canvas positions
    (rope_index; s >= T: s + delta, the decode's), and pristine puts the 1-D rows back: a
    thumbnail and a 1 x 2 grid of slices of different (h, w), between text."""
    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as HM
    from opentpu.llm import minicpmv as MV
    m, d = tiny
    spec = replace(load_spec(d), vision_patches=128, vision_rows=64,
                   vision_staged=False)         # (2 k slots a layer: too few to stage)
    vs = spec.vision
    W = dict(load_weights(d).items())
    W.update(_tiny_vision_weights(vs, np.random.default_rng(6)))
    cap = 256
    cfg = device_config(spec, cap, rows=1, lookup=True, S=1, experts=2 * K)
    eng = Engine(spec, W, cap=cap, cfg=cfg, rows=1, resident=True, experts=2 * K, layer_major=2)
    v = MV.Vision(eng)
    sizes = [(8, 12), (12, 8), (12, 8)]
    P_ = vs.image_token
    ids = [5, 6, 7, vs.image_start] + [P_] * 6 + [vs.image_end] + [vs.slice_start] + \
        [P_] * 6 + [vs.slice_end] + [vs.slice_start] + [P_] * 6 + [vs.slice_end] + [9, 10, 11]
    delta = v.positions(ids, sizes)
    pos, _ = MV.rope_index(ids, sizes, vs)
    T = len(ids)
    rot = HM.Qwen3_5MoeTextRotaryEmbedding(m.config)
    lk = eng.image.lookup
    half = lk["half"]

    def table():
        return [eng.backend.read(0, lk[n], 4 * half * cap).view(np.float32).reshape(cap, half)
                for n in ("cos_t", "sin_t")]

    def hf(p):
        c, s_ = rot(torch.zeros(1), torch.from_numpy(p)[:, None, :])
        return c[0, :, :half].numpy(), s_[0, :, :half].numpy()
    want = hf(np.concatenate([pos, np.tile(np.arange(T, cap) + delta, (3, 1))], 1))
    assert delta != 0 and not np.array_equal(pos[1], pos[2])
    for got, w in zip(table(), want):
        assert np.abs(got - w).max() < 1e-4
    v.pristine()
    for got, w in zip(table(), hf(np.tile(np.arange(cap), (3, 1)))):
        assert np.abs(got - w).max() < 1e-4


# =============================================================================== otpu-serve
class _Tok:
    """One character per token id."""

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(0x100 + int(i)) for i in ids)


def test_serve_keys_contexts_by_their_images(tiny):
    """otpu-serve's worker with images (scheduler.Request.images; prompts keyed by
    minicpmv.keyed): two requests with different images after the same text, in turns, give
    the tokens of their own runs on a fresh engine (the tower, the image rows, each context's
    RoPE rows written at its switch); one that continues the first's conversation feeds only
    what is new; one with a third image after the same text reuses no context past the text
    (with the image tokens' one id it would take a saved one whole); a text-only one after
    them runs on the table's own rows."""
    import hashlib

    from opentpu.llm import minicpmv as MV
    from opentpu.serve import scheduler as SC
    m, d = tiny
    sp = load_spec(d)
    vs = replace(sp.vision, image_token=1000, image_start=1001, image_end=1002,
                 slice_start=1003, slice_end=1004)
    spec = replace(sp, vision=vs, vision_patches=128, vision_rows=64,
                   vision_staged=False)         # (2 k slots a layer: too few to stage)
    rng = np.random.default_rng(7)
    W = dict(load_weights(d).items())
    W.update(_tiny_vision_weights(vs, rng))
    cfg = device_config(spec, 256, rows=1, lookup=True, S=1, experts=2 * K)

    def engine():
        return Engine(spec, W, cap=256, cfg=cfg, rows=1, resident=True, experts=2 * K,
                      layer_major=2)

    def image(seed):
        P = np.random.default_rng(seed).uniform(-1, 1, (96, 3 * 14 * 14)).astype(np.float32)
        return MV.ImageInput([P], [(8, 12)], None, "", hashlib.sha256(P.tobytes()).hexdigest())

    head = [int(t) for t in rng.integers(0, 900, 7)]
    tail = [int(t) for t in rng.integers(0, 900, 5)]

    def prompt(im, more=()):
        ids = head + ([vs.image_start] + [vs.image_token] * im.tokens + [vs.image_end]
                      if im is not None else []) + tail + list(more)
        return ids, ([] if im is None else [im])

    def ref(ids, ims, n):
        """Greedy tokens of a fresh engine, as the worker feeds them (the device's loop: the
        prompt's last token through the decode)."""
        e = engine()
        vis = MV.Vision(e).prompt(ids, ims) if ims else None
        card = getattr(e, "can_generate", False)
        a = len(ids) - 1 if card else len(ids)
        lg = e.prefill_layers(ids[:a], logits=not card,
                              **({} if vis is None else {"vis": vis[:a]}))
        if card:
            lg = e.step(ids[-1])
        out = []
        while True:
            t = int(np.argmax(lg))
            out.append(t)
            if len(out) == n:
                return out
            lg = e.step(t)

    def request(ids, ims, n=6):
        key = MV.keyed(ids, ims, vs.image_token) if ims else ids
        return SC.Request(key, max_new=n, stop_ids=frozenset({-1}), images=ims)

    X, Y, Z = image(1), image(2), image(3)
    w = SC.Worker(engine(), _Tok(), quantum=2, max_running=4).start()
    try:
        pa, pb = prompt(X), prompt(Y)
        ra, rb = w.submit(request(*pa)), w.submit(request(*pb))
        for r in (ra, rb):
            r.wait(timeout=900)
            assert r.error is None and r.finish == "length", r.error
        assert ra.swaps + rb.swaps >= 2
        assert ra.out == ref(*pa, 6) and rb.out == ref(*pb, 6) and ra.out != rb.out
        pc = prompt(X, ra.out + [7, 8, 9])
        rc = w.submit(request(*pc)).wait(timeout=900)
        assert rc.error is None and rc.cached >= len(pa[0])
        assert rc.out == ref(*pc, 6)
        pz = prompt(Z)
        rz = w.submit(request(*pz)).wait(timeout=900)
        assert rz.error is None and rz.cached <= len(head) + 1
        assert rz.out == ref(*pz, 6)
        pt = prompt(None)
        rt = w.submit(request(*pt)).wait(timeout=900)
        assert rt.error is None and rt.out == ref(*pt, 6)
        with pytest.raises(ValueError):    # (keys without their images)
            w.submit(SC.Request(request(*pa).prompt, max_new=2))
    finally:
        w.stop(persist=False)


class _VTok:
    """A character tokenizer with MiniCPM-V's markers as single ids (1000 + i), and a template
    that renders an image part as <|image_pad|>."""
    chat_template = "minicpmv-like"
    SPECIAL = ("<|image_pad|>", "<image>", "</image>", "<slice>", "</slice>", "<image_id>",
               "</image_id>")

    def __call__(self, text):
        import re
        out = []
        for piece in re.split("(" + "|".join(re.escape(s) for s in self.SPECIAL) + ")", text):
            if piece in self.SPECIAL:
                out.append(1000 + self.SPECIAL.index(piece))
            else:
                out += [ord(c) % 900 for c in piece]
        return {"input_ids": out}

    def apply_chat_template(self, msgs, add_generation_prompt, enable_thinking, tokenize):
        def text(c):
            return c if isinstance(c, str) else "".join(
                "<|image_pad|>" if p["type"] == "image" else p["text"] for p in c)
        out = "".join(f"\x01{m['role']}\n{text(m['content'])}\x02\n" for m in msgs)
        if add_generation_prompt:
            out += "\x01assistant\n"
        return out if not tokenize else self(out)["input_ids"]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(0x100 + int(i)) for i in ids)


class _Worker:
    """What Server takes of a scheduler.Worker; submit records the requests."""

    def __init__(self, cap=4096):
        import threading
        from types import SimpleNamespace
        self.tok_lock, self.stateful = threading.Lock(), False
        self.eng = SimpleNamespace(cap=cap)
        self.got = []

    def busy(self):
        return 0

    def submit(self, r):
        self.got.append(r)
        return r


def _png(w, h, seed):
    import base64
    import io
    from PIL import Image as PI
    a = np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)
    buf = io.BytesIO()
    PI.fromarray(a).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def test_serve_takes_image_url_parts():
    """otpu-serve's chat with image_url parts (api.Server on MiniCPM-V's spec): each image
    prepared at the request's max_slices, its placeholder in place of its <|image_pad|>, the
    prompt's key its token ids with each image's tokens their content keys; the same image
    gives the same keys, another image others; refused: an image for a model without a vision
    side, an http(s) URL without --image-fetch, a max_slices out of range, a bad data URL."""
    import base64
    from types import SimpleNamespace

    from opentpu.llm import minicpmv as MV
    from opentpu.serve import api as API
    pytest.importorskip("PIL")              # (the images' PNGs, and their decode without torchvision)
    vs = MV.VisionSpec(image_token=1000, image_start=1001, image_end=1002, slice_start=1003,
                       slice_end=1004)
    spec = SimpleNamespace(vision=vs, vision_rows=2048, eos=[999])
    tok, w = _VTok(), _Worker()
    srv = API.Server(w, tok, spec, "mcpv", dict(temperature=0.0, top_k=20, top_p=1.0,
                                                 repetition_penalty=1.0))
    a, b = _png(1000, 700, 1), _png(1000, 700, 2)

    def chat(*urls, text="What is this?", **kw):
        content = [{"type": "image_url", "image_url": {"url": u}} for u in urls] + \
            [{"type": "text", "text": text}]
        w.got.clear()
        srv.chat({"messages": [{"role": "user", "content": content}], **kw})
        return w.got[0]

    r = chat(a, max_slices=4)
    ims = [MV.prepare(base64.b64decode(a.split(",", 1)[1]), vs, 4, image_id=0)]
    assert len(r.images) == 1 and r.images[0].sizes == ims[0].sizes and len(ims[0].sizes) > 1
    ids = tok(MV.expand(tok.apply_chat_template(
        [{"role": "user", "content": [{"type": "image"}, {"type": "text",
                                                          "text": "What is this?"}]}],
        True, False, False), ims))["input_ids"]
    assert MV.unkeyed(r.prompt, vs.image_token) == ids
    assert r.prompt == MV.keyed(ids, ims, vs.image_token)
    assert sum(t < 0 for t in r.prompt) == ims[0].tokens
    assert chat(a, max_slices=4).prompt == r.prompt             # the same image: the same key
    r2 = chat(b, max_slices=4)
    first = next(i for i, t in enumerate(r.prompt) if t < 0)
    assert r2.prompt[:first] == r.prompt[:first] and r2.prompt[first] != r.prompt[first]
    r3 = chat(a, b, max_slices=1)                                 # two images, numbered
    assert [len(im.sizes) for im in r3.images] == [1, 1]
    assert MV.unkeyed(r3.prompt, vs.image_token).count(1005) == 2   # (<image_id> twice)
    for bad in ({"url": "http://example.com/x.png"}, {"url": "/etc/passwd"},
                {"url": "data:image/png,abc"}):
        with pytest.raises(API.APIError):
            srv.chat({"messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": bad}]}]})
    with pytest.raises(API.APIError):
        chat(a, max_slices=0)
    text = API.Server(w, tok, SimpleNamespace(vision=None, eos=[999]), "t", srv.defaults)
    with pytest.raises(API.APIError):
        w.got.clear()
        text.chat({"messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": a}}]}]})
