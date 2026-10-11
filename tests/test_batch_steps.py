"""Batched decode steps (opentpu/llm/batch.py) on the ISA simulator: an Engine(batch=4) image
with its lookup tables and batch area; three prompts prefilled into slots 0, 1 and 3 (slot 0
by its prompt runs, the others by the rows kernel with device inputs), then steps that feed a
token to each and pick the next on the device. Greedy rows give each sequence's own greedy
decode (a single-sequence engine's), sampled rows (top-k, top-p, temperature, a repetition
penalty) the decode loop's reference pick on that engine's logits with the row's uniform, bit
for bit; rows join and leave the steps; slot 0 keeps the generate loop."""
import dataclasses

import numpy as np
import pytest

from opentpu.llm import batch as BA
from opentpu.llm import generate as G
from opentpu.llm.qwen3 import PREFILL_ROWS, Engine, device_config

from test_serve_kvio import _model

torch = pytest.importorskip("torch")

CAP = 512


def _engines():
    W, spec, cfg1 = _model("qwen3")
    cfg = device_config(spec, CAP, batch=4, rows=PREFILL_ROWS, lookup=True, S=1)
    cfg = dataclasses.replace(cfg, DRAM_BYTES=2 * cfg.DRAM_BYTES)
    eng = Engine(spec, W, cap=CAP, cfg=cfg, batch=4, resident=True, prompt_runs=True,
                 batch_area=True)

    def single():
        return Engine(spec, W, cap=CAP, cfg=cfg1, resident=True, prompt_runs=True)
    return eng, single, spec


def _prompts(seed, sizes):
    r = np.random.default_rng(seed)
    return [[int(x) for x in r.integers(0, 1000, n)] for n in sizes]


def test_greedy_rows_are_each_sequences_decode():
    eng, single, spec = _engines()
    assert eng.bat is not None and eng.can_generate
    slots = [0, 1, 3]
    ps = _prompts(0, [21, 9, 30])
    st = BA.Steps(eng)
    nxt = {}
    for s, p in zip(slots, ps):
        nxt[s] = int(np.argmax(eng.prefill(p, seq=s)))
        st.set_sampling(s, None)
    want = {}
    for s, p in zip(slots, ps):
        e = single()
        lg, out = e.prefill(p), []
        for _ in range(7):
            t = int(np.argmax(lg))
            out.append(t)
            lg = e.step(t)
        want[s] = out
    got = {s: [nxt[s]] for s in slots}
    live = list(slots)
    for i in range(6):
        if i == 3:
            live = [0, 3]                   # slot 1 leaves; its row comes back after
        if i == 5:
            live = [0, 1, 3]
        if i == 5:                          # slot 1 catches up alone first
            while len(got[1]) < len(got[0]):
                t = st.run([(1, eng.poss[1])], [got[1][-1]])[0]
                got[1].append(t)
        rows = [(s, eng.poss[s]) for s in live if len(got[s]) < 7]
        if not rows:
            break
        out = st.run(rows, [got[s][-1] for s, _ in rows])
        for (s, _), t in zip(rows, out):
            got[s].append(t)
    for s in slots:
        assert got[s][:7] == want[s][:len(got[s][:7])], s
        assert len(got[s]) >= 6


@pytest.mark.parametrize("pen", [1.0, 1.3])
def test_sampled_rows_pick_as_the_decode_loop(pen):
    """Rows with sampling: each step's pick equals generate.reference_pick on the logits of a
    single-sequence engine fed the same tokens, with the same uniform; a greedy row beside."""
    eng, single, spec = _engines()
    slots = [0, 2]
    ps = _prompts(1, [17, 25])
    samps = {0: G.Sampling(0.8, 20, 0.9, pen), 2: None}
    st = BA.Steps(eng)
    refs, ctx, nxt = {}, {}, {}
    for s, p in zip(slots, ps):
        lg = eng.prefill(p, seq=s)
        nxt[s] = int(np.argmax(lg))
        ctx[s] = list(p) + [nxt[s]]
        st.set_sampling(s, samps[s], ctx[s])
        e = single()
        e.prefill(p)
        refs[s] = e
    rng = np.random.default_rng(5)
    for _ in range(6):
        u = rng.random(2).astype(np.float32)
        rows = [(s, eng.poss[s]) for s in slots]
        out = st.run(rows, [nxt[s] for s in slots], uniforms=u)
        for r, s in enumerate(slots):
            lg = refs[s].step(nxt[s])
            if samps[s] is None:
                want = int(np.argmax(lg))
            else:
                want = G.reference_pick(lg, samps[s], ctx[s], np.minimum(
                    u[r], np.float32(1 - 2.0 ** -24)), 1, None)
            assert out[r] == want, (s, r)
            nxt[s] = out[r]
            ctx[s].append(out[r])


def test_slot0_keeps_the_generate_loop_and_steps_follow_it():
    eng, single, spec = _engines()
    p = _prompts(2, [19])[0]
    e = single()
    t0 = int(np.argmax(e.prefill(p)))
    want = e.generate_card(t0, 10)
    t = int(np.argmax(eng.prefill(p)))
    got = eng.generate_card(t, 5)               # the loop on the device ...
    st = BA.Steps(eng)
    st.set_sampling(0, None)
    x = got[-1]
    for _ in range(5):                          # ... then batched steps
        x = st.run([(0, eng.poss[0])], [x])[0]
        got.append(x)
    assert got == want


def test_rows_split_into_groups_that_fit_imem():
    """With IMEM for one row's program only, a step of three rows runs in three groups: the
    same tokens as one run of all three."""
    eng, single, spec = _engines()
    slots, ps = [0, 1, 2], _prompts(3, [11, 14, 9])
    st = BA.Steps(eng)
    nxt = {}
    for s, p in zip(slots, ps):
        nxt[s] = int(np.argmax(eng.prefill(p, seq=s)))
        st.set_sampling(s, None)
    one = len(BA.compile_step(eng.image, eng.bat, [(0, eng.poss[0])], [None], eng.block)[0])
    st.limit = one + 8
    rows = [(s, eng.poss[s]) for s in slots]
    assert [len(g) for g in st.plan(rows)] == [1, 1, 1]
    got = {s: [nxt[s]] for s in slots}
    for _ in range(4):
        out = st.run([(s, eng.poss[s]) for s in slots], [got[s][-1] for s in slots])
        for s, t in zip(slots, out):
            got[s].append(t)
    for s, p in zip(slots, ps):
        e = single()
        lg, want = e.prefill(p), []
        for _ in range(5):
            want.append(int(np.argmax(lg)))
            lg = e.step(want[-1])
        assert got[s] == want, s
    assert st.runs == 12
