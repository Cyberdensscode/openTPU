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
