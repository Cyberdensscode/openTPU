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


@pytest.mark.parametrize("P, drafter, sp", [
    (250, "right", (0.8, 5, 0.9, 1.1)),     # penalty: row 1's counts the draft
    (251, "mixed", (1.5, 20, 1.0, 1.0)),    # E at 255; no penalty, top-k 20
    (249, "mtp", (0.8, 5, 0.9, 1.1))])
def test_moe_mtp_sampled_loop_is_plain_sampling(tiny_moe_mtp, P, drafter, sp):
    """The sampled MTP loop on the device on the streamed MoE gives plain sampled decode's
    tokens for the same uniforms whatever the drafts (the verify samples a0 with p + 1's
    uniform and accepts iff a0 is the draft, a1 with p + 2's), and leaves the committed
    states as plain decode does."""
    from opentpu.llm import generate as G
    from test_mtp import _states
    W, spec = tiny_moe_mtp
    samp, N, seed = G.Sampling(*sp), 15, 5
    prompt = [int(t) for t in np.random.default_rng(P).integers(0, 1000, P)]
    ref = Engine(spec, W, cap=512, cfg=CFG, rows=1, resident=True, experts=SLOTS)
    u = np.minimum(np.random.default_rng(seed).random(N).astype(np.float32),
                   np.float32(1 - 2.0 ** -24))
    c, want, lg = list(prompt), [], ref.prefill(prompt)
    for i in range(N):
        want.append(G.reference_pick(lg, samp, c, u[i]))
        c.append(want[-1])
        if i < N - 1:
            lg = ref.step(want[-1])
    assert len(set(want)) > len(want) // 2
    right = np.zeros(514, np.float32)
    right[P:P + len(want)] = want
    drafts = {"right": right, "mtp": None,
              "mixed": np.where(np.arange(514) % 3, right, (right + 7) % 1000)}[drafter]
    eng = mtp_engine(spec, W, cap=512, cfg=CFG, rows=2, experts=SLOTS)
    dec = MTPDecoder(eng)
    st = dec.generate_card(prompt, max_new=N, stop=[], drafts=drafts, sampling=samp,
                           rng=np.random.default_rng(seed))
    assert st.tokens == want
    if drafter == "right":
        assert sum(st.accepted) >= len(want) // 2 - 2
    assert eng.pos == ref.pos == P + len(want) - 1
    assert all(np.array_equal(a, b) for a, b in
               zip(_states(eng, spec, dec.slot), _states(ref, spec, 0)))


def _layered(spec, W, em, cfg=CFG, **kw):
    """An MTP engine whose prompt goes layer by layer (runs of 2 rows), expert-major or not,
    6 slots a layer (expert-major's scratch beside 2 k)."""
    return mtp_engine(spec, W, cap=512, cfg=cfg, rows=2, experts=6, layer_major=2,
                      expert_major=em, **kw)


@pytest.mark.parametrize("em, fmt, drafter", [(False, "int8", "mtp"), (True, "int8", "mtp"),
                                              (True, "fp4", "mixed"), (False, "int8", "wrong")])
def test_moe_mtp_layer_major_prefill(tiny_moe_mtp, em, fmt, drafter, monkeypatch):
    """A streamed MoE's prompt layer by layer under MTP (Engine.prefill_layers with the MTP
    layer's K / V, docs/mtp.md 12): the rows before conv_k - 1 in the rows kernel's runs, then
    chunks of 24 rows through the model a layer at a time (expert-major or not), after each
    chunk's last layer runs of up to 4 rows append the MTP layer's K / V (no queries, no MoE),
    and the last row drafts after the head (its hidden row the head's): the loop's tokens equal
    plain greedy decode's. int8: the MTP layer's KV cache and the first draft equal those of
    the 2-row runs' prefill (the MTP layer over every row) bit for bit."""
    from opentpu.llm import qwen35 as Q35
    from opentpu.llm.mtp import MTPStats
    from test_mtp import _mtp_kv
    monkeypatch.setattr(Q35, "PREFILL_CHUNK", 24)
    W, spec = tiny_moe_mtp
    kw = FP4 if fmt == "fp4" else {}
    prompt = [int(t) for t in np.random.default_rng(11).integers(0, 1000, 40)]
    P, N = len(prompt), 12
    want = _plain(spec, W, prompt, N, **kw)
    right = np.zeros(514, np.float32)
    right[P:P + len(want)] = want
    drafts = {"mtp": None, "wrong": (right + 1) % 1000,
              "mixed": np.where(np.arange(514) % 3, right, (right + 7) % 1000)}[drafter]
    eng = _layered(spec, W, em, **kw)
    dec = MTPDecoder(eng)
    assert dec.by_layers and eng.image.prefill_rows == 24 and eng.expert_major == em
    st = dec.generate_card(prompt, max_new=N, stop=[], drafts=drafts)
    assert st.tokens == want
    m = sorted(k for k in eng._layer_runs if k[0] == "m")
    assert m and max(k[2] for k in m) == 4      # (the host's slot rows)
    if drafter == "wrong":
        assert not any(st.accepted)
    if fmt == "int8" and drafter == "mtp":
        a, b = MTPDecoder(_layered(spec, W, em)), MTPDecoder(mtp_engine(
            spec, W, cap=512, cfg=CFG, rows=2, experts=6))
        assert not b.by_layers
        ra, rb = a.prefill(prompt, MTPStats()), b.prefill(prompt, MTPStats())
        assert ra == rb and ra[0] == want[0]
        assert all(np.array_equal(x, y) for x, y in zip(_mtp_kv(a.eng, P), _mtp_kv(b.eng, P)))


def test_moe_mtp_layer_major_continues_from_slot_1(tiny_moe_mtp, monkeypatch):
    """A chat's next turn on the layer-major path: the first turn's loop (drafts all right)
    leaves the states committed in slot 1, and the next prompt's layer runs read and write that
    slot (the runs of slot 1, compiled apart). The tokens equal one fresh decoder's over the
    whole conversation, and plain greedy decode's."""
    from opentpu.llm import qwen35 as Q35
    monkeypatch.setattr(Q35, "PREFILL_CHUNK", 24)
    W, spec = tiny_moe_mtp
    r = np.random.default_rng(3)
    p1, p2 = [int(t) for t in r.integers(0, 1000, 30)], [int(t) for t in r.integers(0, 1000, 9)]
    w1 = _plain(spec, W, p1, 12)
    right = np.zeros(514, np.float32)
    right[len(p1):len(p1) + 12] = w1
    dec = MTPDecoder(_layered(spec, W, True))
    t1 = dec.generate_card(p1, max_new=12, stop=[], drafts=right).tokens
    assert t1 == w1 and dec.slot == 1
    ctx = p1 + t1 + p2
    st = dec.generate_card([t1[-1]] + p2, max_new=10, stop=[])
    assert any(k[1:] == ("slot", 1) for k in dec.eng._layer_runs if len(k) == 3)
    fresh = MTPDecoder(_layered(spec, W, True)).generate_card(ctx, max_new=10, stop=[])
    assert st.tokens == fresh.tokens == _plain(spec, W, ctx, 10)


def test_streamed_mtp_engine_from_a_checkpoint_and_an_extended_pool(tiny_moe_mtp, tmp_path,
                                                                      monkeypatch):
    """otpu-chat --mtp on a streamed MoE (streamed.engine(mtp=True)): the checkpoint's mtp.*
    tensors (LazyWeights(mtp=True)) and the pool pack_pool.py extend makes from the model's own
    (its experts and marks copied, the MTP layer's after them not packed: the run packs each it
    asks for from the checkpoint). The loop's tokens equal plain greedy decode's from the
    model's pool. MiniCPM-V 4.7's case: a checkpoint without mtp.* is refused, and takes them
    from another (mtp_from), whose identity the pool's key holds: the pool extended for the
    model's own MTP layer is refused, one extended with --mtp-from serves it."""
    import json
    import runpy
    import sys
    from pathlib import Path

    from safetensors.numpy import save_file

    from opentpu.host import streamed as ST
    from opentpu.host.offload import to_split
    from opentpu.llm.qwen3 import LazyWeights
    from test_qwen35_moe import _tiny

    W, spec = tiny_moe_mtp
    hf = _tiny()[0].config
    hf.tie_word_embeddings = False

    def checkpoint(d, names):
        d.mkdir()
        hf.save_pretrained(d)
        save_file({k: np.ascontiguousarray(W[k]) for k in names}, str(d / "model.safetensors"))
        return d
    m = checkpoint(tmp_path / "model", list(W))
    m2 = checkpoint(tmp_path / "nomtp", [k for k in W if not k.startswith("mtp.")])
    tool = str(Path(__file__).resolve().parents[1] / "tools" / "offload" / "pack_pool.py")

    def run(*args):
        monkeypatch.setattr(sys, "argv", ["pack_pool.py", *map(str, args)])
        try:
            runpy.run_path(tool, run_name="__main__")
        finally:
            monkeypatch.undo()

    a, b, c = (tmp_path / f"{x}.bin" for x in "abc")
    run(m, a, "init")
    run(m, a, "pack", 0, 1)
    run(m, b, "extend", a)
    v = tmp_path / "v.bin"                  # a pool with entries after its experts (vision)
    v.write_bytes(a.read_bytes() + bytes(range(256)) * (2 * a.stat().st_size // 48 // 256))
    Path(str(v) + ".packed").write_bytes(Path(str(a) + ".packed").read_bytes() + bytes([1, 1]))
    for x in (".format", ".key"):
        Path(str(v) + x).write_text(Path(str(a) + x).read_text())
    run(m2, c, "extend", v, "--mtp-from", m)
    nE = spec.layers * spec.moe.E
    pa, pb = (np.fromfile(str(p) + ".packed", np.uint8) for p in (a, b))
    assert pa.tolist() == [1] * nE and pb.tolist() == [1] * nE + [0] * spec.moe.E
    slot = a.stat().st_size // nE
    assert b.read_bytes()[:nE * slot] == c.read_bytes()[:nE * slot] == a.read_bytes()
    assert b.stat().st_size == c.stat().st_size == (nE + spec.moe.E) * slot
    assert Path(str(c) + ".packed").read_bytes() == pb.tobytes()
    kb, kc = (json.loads(Path(str(p) + ".key").read_text()) for p in (b, c))
    assert kb["layout"]["layers"] == spec.layers + 1 and "mtp_from" not in kb
    assert kc["mtp_from"] and {k: v for k, v in kc.items() if k not in ("mtp_from", "model")} \
        == {k: v for k, v in kb.items() if k != "model"}

    prompt = [int(t) for t in np.random.default_rng(4).integers(0, 1000, 9)]
    N = 12
    kw = dict(cap=512, experts=SLOTS, wformat="fp4", head_format="int8")     # (the pools')
    want = ST.engine(m, CFG, pool=a, **kw).generate(prompt, max_new=N)
    eng = ST.engine(m, CFG, pool=b, mtp=True, **kw)
    assert eng.image.rows == 2 and eng.image.offload.layers == spec.layers + 1
    dec = MTPDecoder(eng)
    assert dec.generate_card(prompt, max_new=N, stop=[]).tokens == want
    got = np.fromfile(str(b) + ".packed", np.uint8)[nE:]
    assert got.any()                        # the MTP layer's experts it asked for, packed
    Wm, img = LazyWeights(m, mtp=True), eng.image
    pool = np.fromfile(b, np.uint8).reshape(-1, slot)
    for e in np.nonzero(got)[0]:
        assert np.array_equal(pool[nE + e], np.asarray(to_split(img.expert(Wm, nE + e)))
                              .view(np.uint8)), e
    with pytest.raises(ValueError, match="no mtp.. tensors"):
        ST.engine(m2, CFG, pool=c, mtp=True, **kw)
    with pytest.raises(ST.PoolError, match="mtp_from differ"):
        ST.engine(m2, CFG, pool=b, mtp=True, mtp_from=m, **kw)
    eng = ST.engine(m2, CFG, pool=c, mtp=True, mtp_from=m, **kw)
    assert MTPDecoder(eng).generate_card(prompt, max_new=N, stop=[]).tokens == want
