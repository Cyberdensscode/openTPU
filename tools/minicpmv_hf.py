"""Hugging Face's MiniCPM-V 4.7 (transformers >= 5.18, `minicpmv4_7`) on a host without the RAM for
its experts: the reference of docs/minicpmv.md 5. The whole model is HF's own code (processor,
vision tower, projector, canvas M-RoPE, Qwen3.5-MoE language model, generate) but every MoE
layer's experts are LazyExperts: HF's eager expert loop with each expert's weights read from the
safetensors files when a token routes to it, so the 35B-A3B runs in about 12 GB (fp32) beside
its 64 GB of bf16 experts.

    python3 tools/minicpmv_hf.py greedy MODEL OUT.json [--prompt P] [--image IMG] [-n 32]
        [--max-slices 9] [--dtype float32|bfloat16] [--think]
        HF's greedy tokens of a chat prompt (and an image), with each step's 8 best (id, logit):
        tools/offload/moe_card.py --check reads OUT.json (ids, tokens, top)
    python3 tools/minicpmv_hf.py logits MODEL OUT.npz [--prompt P] [--image IMG] [--ids J]
        [--hidden] [--max-slices 9] [--dtype ...]
        one teacher-forced forward of the prompt (or J's "ids" + "tokens": a greedy run's prompt
        and reply): the logits of every position (float32), with --hidden every layer's output
        rows; and for an image the processor's outputs and the canvas positions
    python3 tools/minicpmv_hf.py check
        a tiny random MiniCPM-V checkpoint: LazyExperts give the logits of HF's own loading
    python3 tools/minicpmv_hf.py procheck MODEL --images IMG... [--slices 9 1]
        the host's image port (opentpu.llm.minicpmv) against HF's processor and canvas
        positions, bit for bit (the model directory's tokenizer and configs; no weights)

MODEL: a checkpoint directory (models/MiniCPM-V-4.7-35B-A3B). transformers 5.18 may sit on its
own path (PYTHONPATH=~/mcpv/tf518 on hwe-bench, with pillow and torchvision 0.29.1+cpu for
the image processor).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn


class LazyExperts(nn.Module):
    """A Qwen3_5MoeExperts whose weights stay in the checkpoint: the forward is HF's eager loop
    (modeling_qwen3_5_moe.Qwen3_5MoeExperts.forward, v5.18.0) with expert e's gate_up_proj[e]
    and down_proj[e] read from the safetensors slices (bf16) and cast to `dtype`."""

    def __init__(self, handles: dict, prefix: str, config, dtype):
        super().__init__()
        self.num_experts = config.num_experts
        self.prefix, self.dtype = prefix, dtype
        self.h = {n: handles[prefix + n] for n in ("gate_up_proj", "down_proj")}
        from transformers.activations import ACT2FN
        self.act_fn = ACT2FN[config.hidden_act]

    def weight(self, name: str, e: int) -> torch.Tensor:
        f = self.h[name]
        return f.get_slice(self.prefix + name)[e].to(self.dtype)

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts + 1)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            e = int(expert_idx)
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate, up = nn.functional.linear(current_state,
                                            self.weight("gate_up_proj", e)).chunk(2, dim=-1)
            h = nn.functional.linear(self.act_fn(gate) * up, self.weight("down_proj", e))
            h = h * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, h.to(final_hidden_states.dtype))
        return final_hidden_states


def load(model_dir, dtype=torch.float32, attn: str = "sdpa"):
    """(model, processor): HF's MiniCPMV4_7ForConditionalGeneration built on the meta device,
    its experts LazyExperts, every other tensor read from the checkpoint and cast to dtype; the
    non-persistent RoPE buffers recomputed (to_empty leaves them uninitialized)."""
    from safetensors import safe_open
    from transformers import AutoConfig, AutoProcessor, MiniCPMV4_7ForConditionalGeneration
    model_dir = Path(model_dir)
    config = AutoConfig.from_pretrained(model_dir)
    with torch.device("meta"):
        model = MiniCPMV4_7ForConditionalGeneration._from_config(
            config, attn_implementation=attn, dtype=dtype)
    handles = {}
    for f in sorted(model_dir.glob("*.safetensors")):
        h = safe_open(str(f), "pt")
        for k in h.keys():
            handles[k] = h
    tc = config.text_config
    lm = model.model.language_model
    for i, layer in enumerate(lm.layers):
        layer.mlp.experts = LazyExperts(handles, f"model.language_model.layers.{i}.mlp.experts.",
                                        tc, dtype)
    model.to_empty(device="cpu")
    want = dict(model.named_parameters())
    seen = set()
    with torch.no_grad():
        for k, h in handles.items():
            if ".mlp.experts." in k:
                continue
            if k not in want:
                raise KeyError(f"{k}: in the checkpoint, not in the model")
            want[k].copy_(h.get_tensor(k).to(dtype))
            seen.add(k)
    missing = sorted(set(want) - seen)
    if missing:
        raise KeyError(f"not in the checkpoint: {missing[:8]} ({len(missing)})")
    for name, mod in model.named_modules():         # the buffers to_empty left uninitialized
        if hasattr(mod, "inv_freq"):
            inv, sc = mod.compute_default_rope_parameters(mod.config)
            mod.inv_freq = inv.to(torch.float32)
            mod.original_inv_freq = inv.clone().to(torch.float32)
            mod.attention_scaling = sc
    bufs = [n for n, _ in model.named_buffers() if not n.endswith(("inv_freq",))]
    if bufs:
        raise RuntimeError(f"buffers this loader does not recompute: {bufs[:8]}")
    model.eval()
    return model, AutoProcessor.from_pretrained(model_dir)


def inputs(proc, prompt: str, image=None, max_slices: int = 9, think: bool = False):
    """The processor's model inputs for one user turn (the chat template, generation prompt)."""
    content = [{"type": "text", "text": prompt}]
    if image is not None:
        content = [{"type": "image", "path": str(image)}] + content
    msgs = [{"role": "user", "content": content}]
    return proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                    return_dict=True, return_tensors="pt",
                                    max_slice_nums=max_slices, enable_thinking=think)


def greedy(a) -> dict:
    model, proc = load(a.model, getattr(torch, a.dtype))
    x = inputs(proc, a.prompt, a.image, a.max_slices, a.think)
    ids = [int(t) for t in x["input_ids"][0]]
    t = time.time()
    with torch.no_grad():
        out = model.generate(**x, max_new_tokens=a.n, min_new_tokens=a.n, do_sample=False,
                             output_scores=True, return_dict_in_generate=True)
    new = [int(v) for v in out.sequences[0, len(ids):]]
    top = []
    for sc in out.scores:
        v, i = torch.topk(sc[0].float(), 8)
        top.append([[int(p), round(float(q), 4)] for p, q in zip(i, v)])
    return dict(model=str(a.model), prompt=a.prompt, image=a.image, max_slices=a.max_slices,
                dtype=a.dtype, ids=ids, tokens=new, text=proc.tokenizer.decode(new), top=top,
                seconds=round(time.time() - t, 1))


def logits(a) -> None:
    model, proc = load(a.model, getattr(torch, a.dtype))
    if a.ids:
        j = json.loads(Path(a.ids).read_text())
        if j.get("image") or a.image:
            raise SystemExit("--ids with an image: give the image run's prompt instead")
        x = {"input_ids": torch.tensor([j["ids"] + j.get("tokens", [])])}
        x["attention_mask"] = torch.ones_like(x["input_ids"])
    else:
        x = inputs(proc, a.prompt, a.image, a.max_slices, a.think)
    z = {"ids": x["input_ids"][0].numpy()}
    with torch.no_grad():
        if "target_sizes" in x:
            pos, delta = model.model.get_rope_index(
                x["input_ids"], mm_token_type_ids=x["mm_token_type_ids"],
                target_sizes=x["target_sizes"], attention_mask=x.get("attention_mask"))
            z.update(positions=pos[:, 0].numpy(), rope_delta=delta.numpy(),
                     pixel_values=x["pixel_values"].float().numpy(),
                     target_sizes=x["target_sizes"].numpy(),
                     mm_token_type_ids=x["mm_token_type_ids"][0].numpy())
        out = model(**x, output_hidden_states=a.hidden, use_cache=False)
    z["logits"] = out.logits[0].float().numpy()
    if a.hidden:
        z["hidden"] = torch.stack([h[0] for h in out.hidden_states]).float().numpy()
    np.savez(a.out, **z)
    print(f"{a.out}: {len(z['ids'])} positions", file=sys.stderr)


# =============================================================================== check
def tiny_checkpoint(d: Path, vocab: int = 1080) -> Path:
    """A tiny random MiniCPM-V 4.7 checkpoint in d (config, safetensors, the real tokenizer's
    files are not needed: the check feeds ids)."""
    from transformers import MiniCPMV4_7Config, MiniCPMV4_7ForConditionalGeneration
    torch.manual_seed(0)
    kinds = ["linear_attention", "linear_attention", "full_attention"]
    cfg = MiniCPMV4_7Config(
        text_config=dict(model_type="qwen3_5_moe_text", hidden_size=256, num_hidden_layers=3,
                         num_attention_heads=8, num_key_value_heads=2, head_dim=256,
                         vocab_size=vocab, layer_types=kinds, linear_num_key_heads=4,
                         linear_num_value_heads=8, linear_key_head_dim=128,
                         linear_value_head_dim=128, num_experts=8, num_experts_per_tok=2,
                         moe_intermediate_size=128, shared_expert_intermediate_size=128,
                         rope_parameters={"rope_type": "default", "rope_theta": 1e7,
                                          "partial_rotary_factor": 0.25,
                                          "mrope_interleaved": True,
                                          "mrope_section": [11, 11, 10]}),
        vision_config=dict(hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                           num_attention_heads=4, image_size=70, patch_size=14),
        insert_layer_id=1, image_token_id=1000, video_token_id=1001, image_start_id=1002,
        image_end_id=1003, slice_start_id=1004, slice_end_id=1005, newline_id=198)
    m = MiniCPMV4_7ForConditionalGeneration(cfg).float().eval()
    with torch.no_grad():
        for n, p in m.named_parameters():
            p.copy_(0.05 * torch.randn_like(p) + (1.0 if n.endswith("norm.weight") and
                                                    "linear_attn" in n else 0.0))
    # the real checkpoint's layout: fused experts (save_pretrained would split them per expert)
    from safetensors.torch import save_file
    cfg.save_pretrained(d)
    save_file({k: v.contiguous() for k, v in m.state_dict().items()},
              str(d / "model.safetensors"))
    return d


def check() -> None:
    import tempfile
    from transformers import MiniCPMV4_7ForConditionalGeneration
    with tempfile.TemporaryDirectory() as td:
        d = tiny_checkpoint(Path(td))
        ref = MiniCPMV4_7ForConditionalGeneration.from_pretrained(d, dtype=torch.float32).eval()
        lazy = load_model_only(d, torch.float32)
        ids =torch.tensor([[int(t) for t in np.random.default_rng(0).integers(0, 1000, 20)]])
        with torch.no_grad():
            a = ref(input_ids=ids).logits
            b = lazy(input_ids=ids).logits
        err = float((a - b).abs().max())
        print(f"tiny MiniCPM-V: LazyExperts against from_pretrained, max |logit diff| {err:.3g}")
        assert err == 0.0, err


def procheck(a) -> None:
    """The host's port (opentpu.llm.minicpmv: decode, crops, patches, placeholder text, canvas
    positions) against HF's processor and get_rope_index, bit for bit, on each image at each
    max-slices."""
    from transformers import AutoConfig, AutoProcessor, MiniCPMV4_7ForConditionalGeneration
    from opentpu.llm import minicpmv as MV
    proc = AutoProcessor.from_pretrained(a.model)
    vs = MV.VisionSpec.from_hf(a.model)
    with torch.device("meta"):
        m = MiniCPMV4_7ForConditionalGeneration._from_config(AutoConfig.from_pretrained(a.model))
    tok = proc.tokenizer
    bad = 0
    for img in a.images:
        for ms in a.slices:
            x = inputs(proc, a.prompt, img, ms)
            im = MV.prepare(img, vs, ms)
            msgs = [{"role": "user", "content": [{"type": "image"},
                                                 {"type": "text", "text": a.prompt}]}]
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
            ids = tok(MV.expand(text, [im]))["input_ids"]
            pv = x["pixel_values"][0].float().numpy()            # [3, p, N p]
            p = vs.patch
            hf_rows = pv.reshape(3, p, -1, p).transpose(2, 0, 1, 3).reshape(-1, 3 * p * p)
            ours = np.concatenate(im.patches)
            pos, delta = MV.rope_index(ids, im.sizes, vs)
            hp, hd = m.model.get_rope_index(x["input_ids"],
                                            mm_token_type_ids=x["mm_token_type_ids"],
                                            target_sizes=x["target_sizes"])
            ok = dict(ids=list(x["input_ids"][0]) == ids,
                      sizes=[tuple(s) for s in x["target_sizes"].tolist()] == list(im.sizes),
                      pixels=hf_rows.shape == ours.shape and np.array_equal(hf_rows, ours),
                      positions=np.array_equal(hp[:, 0].numpy(), pos) and int(hd) == delta)
            bad += not all(ok.values())
            print(f"{Path(img).name} max_slices {ms}: grid {im.grid}, crops {im.sizes}, "
                  f"{im.tokens} image tokens, {len(ids)} ids, delta {delta}: "
                  + " ".join(f"{k} {'ok' if v else 'DIFF'}" for k, v in ok.items()))
            if not ok["pixels"] and hf_rows.shape == ours.shape:
                print(f"   pixels: max |diff| {np.abs(hf_rows - ours).max()}, "
                      f"{int((hf_rows != ours).sum())} of {ours.size} differ")
    if bad:
        raise SystemExit(f"{bad} cases differ")


def load_model_only(model_dir, dtype):
    """load() without the processor (a checkpoint without tokenizer files)."""
    import transformers
    real = transformers.AutoProcessor.from_pretrained
    transformers.AutoProcessor.from_pretrained = staticmethod(lambda *a, **k: None)
    try:
        return load(model_dir, dtype)[0]
    finally:
        transformers.AutoProcessor.from_pretrained = real


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("mode", choices=("greedy", "logits", "check", "procheck"))
    ap.add_argument("model", nargs="?")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--prompt", default="What is the capital of France? Answer in one sentence.")
    ap.add_argument("--image")
    ap.add_argument("--ids", help="logits: a greedy OUT.json whose prompt ids and tokens to feed")
    ap.add_argument("-n", type=int, default=32)
    ap.add_argument("--max-slices", type=int, default=9)
    ap.add_argument("--dtype", default="float32", choices=("float32", "bfloat16"))
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--hidden", action="store_true")
    ap.add_argument("--images", nargs="*", default=[], help="procheck: the images")
    ap.add_argument("--slices", nargs="*", type=int, default=[9, 1],
                    help="procheck: the max-slices to check")
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    if a.mode == "check":
        check()
    elif a.mode == "procheck":
        procheck(a)
    elif a.mode == "greedy":
        r = greedy(a)
        Path(a.out).write_text(json.dumps(r))
        print(json.dumps({k: r[k] for k in ("tokens", "text", "seconds")}))
    else:
        logits(a)


if __name__ == "__main__":
    main()
