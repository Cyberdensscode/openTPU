"""Speculative decoding with the MTP drafter on a streamed Qwen3.5-MoE (opentpu/llm/mtp.py,
docs/mtp.md 12, docs/offload.md 15.6) on the ISA simulator: the verify run's two rows go through
each MoE layer as one request for their experts' union (moe.moe_ffn_rows), the MTP layer's MoE
is one more layer of the expert server's (its experts in the pool after the model's), and the
greedy tokens equal plain greedy decode's bit for bit whatever the drafts, with 2k expert slots
a layer (misses on every layer). A tiny random Qwen3.5-MoE (tests/test_qwen35_moe.py) with random
mtp.* weights in the checkpoint's form (its experts a tensor each)."""
import dataclasses

import numpy as np
import pytest

from opentpu.isasim import board_config
from opentpu.llm.mtp import MTPDecoder, mtp_engine
from opentpu.llm.qwen3 import Engine

from test_qwen35_moe import K, _tiny, _untied

SLOTS = 2 * K               # a layer's slots: a 2-row request's union fits (k of E = 8 each)

pytest.importorskip("torch")

CFG = board_config(DRAM_BYTES=1 << 25, DSTEP=True, STREAM=True, PAIR=True)


def _moe_mtp_weights(W, spec, seed=1):
    """Random mtp.* tensors for the tiny MoE (zero-centred norms; the MoE as the 35B's
    checkpoint has it: the router mlp.gate, the shared expert and its gate, each routed
    expert's gate / up / down a tensor of its own)."""
    r = np.random.default_rng(seed)
    H, d, mo = spec.hidden, spec.head_dim, spec.moe

    def lin(o, i, s=1.0):
        return (s * r.uniform(-1, 1, (o, i)) / np.sqrt(i)).astype(np.float32)

    def norm(n):
        return (0.1 * r.standard_normal(n)).astype(np.float32)
    a, m = "mtp.layers.0.self_attn.", "mtp.layers.0.mlp."
    out = {"mtp.pre_fc_norm_embedding.weight": norm(H), "mtp.pre_fc_norm_hidden.weight": norm(H),
           "mtp.fc.weight": lin(H, 2 * H), "mtp.norm.weight": norm(H),
           "mtp.layers.0.input_layernorm.weight": norm(H),
           "mtp.layers.0.post_attention_layernorm.weight": norm(H),
           a + "q_proj.weight": lin(2 * spec.n_q * d, H), a + "k_proj.weight": lin(spec.n_kv * d, H),
           a + "v_proj.weight": lin(spec.n_kv * d, H), a + "o_proj.weight": lin(H, spec.n_q * d),
           a + "q_norm.weight": norm(d), a + "k_norm.weight": norm(d),
           m + "gate.weight": lin(mo.E, H, 4.0), m + "shared_expert_gate.weight": lin(1, H),
           m + "shared_expert.gate_proj.weight": lin(mo.shared, H),
           m + "shared_expert.up_proj.weight": lin(mo.shared, H),
           m + "shared_expert.down_proj.weight": lin(H, mo.shared)}
    for e in range(mo.E):
        out.update({f"{m}experts.{e}.gate_proj.weight": lin(mo.ffn, H),
                    f"{m}experts.{e}.up_proj.weight": lin(mo.ffn, H),
                    f"{m}experts.{e}.down_proj.weight": lin(H, mo.ffn)})
    return {**W, **out}


@pytest.fixture(scope="module")
def tiny_moe_mtp():
    """(weights, spec): the tiny MoE with an LM head of its own (its int8 embedding table on
    the host, as the 35B's) and random mtp.* weights."""
    spec, W = _untied(_tiny())
    return _moe_mtp_weights(W, spec), spec


FP4 = dict(wformat="fp4", head_format="int8")


def _plain(spec, W, prompt, N, cfg=CFG, **kw):
    """Plain greedy decode's tokens (resident decode, the host loop, 2k slots a layer)."""
    return Engine(spec, W, cap=512, cfg=cfg, rows=1, resident=True, experts=SLOTS,
                  **kw).generate(prompt, max_new=N)


@pytest.mark.parametrize("drafter, fmt", [("mtp", "int8"), ("right", "int8"), ("wrong", "int8"),
                                          ("mixed", "int8"), ("mixed", "fp4"), ("mtp", "fp4")])
def test_moe_mtp_greedy_is_plain_greedy(tiny_moe_mtp, drafter, fmt):
    """Tokens equal plain greedy decode's for any drafts (the MTP's, all right, all wrong, right
    but every third) with 2k slots a layer: the verify's two rows request their experts' union
    in each MoE layer, the MTP layer its own (one more layer of the server's), and every layer
    misses on the way. The prompt's 9 tokens prefill in runs of 2 rows. fp4: 4-bit weights,
    whose 2-row MMs pair (PAIR, MCOLS 4) as the decode steps' do."""
    W, spec = tiny_moe_mtp
    kw = FP4 if fmt == "fp4" else {}
    prompt = [int(t) for t in np.random.default_rng(4).integers(0, 1000, 9)]
    N = 16
    want = _plain(spec, W, prompt, N, **kw)
    P = len(prompt)

    def right(q, out):
        return want[q - P] if q - P < len(want) else 0
    drafts = {"mtp": None, "right": right,
              "wrong": lambda q, out: (right(q, out) + 1) % 1000,
              "mixed": lambda q, out: right(q, out) if q % 3 else (right(q, out) + 7) % 1000}
    eng = mtp_engine(spec, W, cap=512, cfg=CFG, rows=2, experts=SLOTS, **kw)
    assert eng.image.offload.layers == spec.layers + 1
    dec = MTPDecoder(eng)
    m0 = eng.server.misses
    st = dec.generate(prompt, max_new=N, drafts=drafts[drafter])
    assert st.tokens == want
    assert eng.server.misses > m0
    if drafter == "right":
        assert all(st.accepted[:-1]) and st.iterations <= N // 2 + 1
    if drafter == "wrong":
        assert not any(st.accepted)
    E = spec.moe.E
    if drafter == "mtp":                    # the MTP layer's experts went through the server
        assert any(g // E == spec.layers for g in eng.server.lru[spec.layers])


def test_the_mtp_layers_experts_pack_from_its_tensors(tiny_moe_mtp):
    """The expert server's layer `layers` is the MTP layer's MoE: its slots after the model's,
    its experts packed from the checkpoint's mtp.layers.0.mlp.experts.E.* tensors."""
    W, spec = tiny_moe_mtp
    eng = mtp_engine(spec, W, cap=512, cfg=CFG, rows=2, experts=SLOTS)
    img, E = eng.image, spec.moe.E
    g = spec.layers * E + 3
    a = img.expert(W, g)
    W2 = dict(W)
    W2["mtp.layers.0.mlp.experts.3.up_proj.weight"] = W["mtp.layers.0.mlp.experts.4.up_proj.weight"]
    assert not np.array_equal(a, img.expert(W2, g))
    assert np.array_equal(img.expert(W, 5), img.expert(W2, 5))
    plain = Engine(spec, W, cap=512, cfg=CFG, rows=1, resident=True, experts=SLOTS).image
    assert img.offload.layers == plain.offload.layers + 1
    assert dataclasses.replace(spec).mtp is False


@pytest.mark.parametrize("P, drafter, fmt, N", [
    (250, "right", "int8", 16),     # V at 254 accepts: the wrap; the last token alone
    (251, "right", "int8", 15),     # 253 -> 255: E, then D1 and the next bucket's V
    (249, "mixed", "int8", 15),     # E at the other parity
    (250, "wrong", "int8", 12),
    (249, "right", "fp4", 15),
    (250, "mtp", "int8", 15)])
def test_moe_mtp_loop_on_the_device_is_plain_greedy(tiny_moe_mtp, P, drafter, fmt, N):
    """The MTP loop on the device (V, E, D, D1 chained through the buckets' programs) on the
    streamed MoE gives plain greedy decode's tokens across a bucket's end, whatever the drafts,
    and leaves the committed DeltaNet states as plain decode does. The embedding rows come from
    the host: the host writes the first V's (a0, d), then each program posts the next one's
    tokens as one request (V: a0, a1 for the D; D: t, d for the next V; RowServer's rows 0, 1
    of the slot), and the expert server serves the MoE layers' misses meanwhile."""
    from test_mtp import _states
    W, spec = tiny_moe_mtp
    kw = FP4 if fmt == "fp4" else {}
    prompt = [int(t) for t in np.random.default_rng(P).integers(0, 1000, P)]
    want = _plain(spec, W, prompt, N, **kw)
    right = np.zeros(514, np.float32)
    right[P:P + len(want)] = want
    drafts = {"right": right, "wrong": (right + 1) % 1000, "mtp": None,
              "mixed": np.where(np.arange(514) % 3, right, (right + 7) % 1000)}[drafter]
    eng = mtp_engine(spec, W, cap=512, cfg=CFG, rows=2, experts=SLOTS, **kw)
    dec = MTPDecoder(eng)
    m0 = eng.server.misses
    eng.row_server.history = []
    st = dec.generate_card(prompt, max_new=N, stop=[], drafts=drafts)
    assert st.tokens == want
    assert eng.server.misses > m0
    assert len(eng.row_server.history) >= 2 * st.iterations - 2     # both rows, each program
    if drafter == "right":
        assert sum(st.accepted) >= len(want) // 2 - 2
    if drafter == "wrong":
        assert not any(st.accepted)
    ref = Engine(spec, W, cap=512, cfg=CFG, rows=1, resident=True, experts=SLOTS, **kw)
    ref.prefill(prompt)
    for t in want[:-1]:
        ref.step(t)
    assert eng.pos == ref.pos == P + len(want) - 1
    assert all(np.array_equal(a, b) for a, b in
               zip(_states(eng, spec, dec.slot), _states(ref, spec, 0)))
