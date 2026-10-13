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
        "tie_word_embeddings": False, "text_config": tc, "insert_layer_id": 6,
        "downsample_mode": "16x", "mrope_mode": "canvas",
        "vision_config": {"hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 1,
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
    Its last real id is checked on its own too."""
    m, d = tiny
    spec = load_spec(d)
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
