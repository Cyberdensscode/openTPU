"""A context's bytes (opentpu/serve/kvio.py): the KV caches' pages and the state saved from one
engine, the device then used by another context, and the bytes written back give the context
fed those tokens, bit for bit: the next prefill's logits and greedy tokens equal a fresh
engine's. Direct (the backend's read / write) and staged (LD / ST programs through a staging
area of the device's DRAM) alike, on the ISA simulator; whole pages shared by contexts with a
common prefix; a partial restore (the first k positions of a longer saved context) for a model
whose KV caches hold the whole context."""
import dataclasses

import numpy as np
import pytest

from opentpu.compiler import vt_tile
from opentpu.llm.qwen3 import PREFILL_ROWS, Engine, device_config
from opentpu.serve import kvio as KV

torch = pytest.importorskip("torch")


def _bits(x):
    return np.asarray(x, np.float32).view(np.uint32)


def _model(case):
    from test_autodecode import _tiny
    from test_mtp import CFG
    from test_qwen35 import _tiny_model
    if case in ("qwen3", "lfm2"):
        W, spec = _tiny(case)
        return W, spec, device_config(spec, 512, rows=PREFILL_ROWS, lookup=True, S=1)
    _, W, spec = _tiny_model(8, init=0.2)
    if case == "qwen35-design":         # two slices
        return W, spec, device_config(spec, 512, rows=PREFILL_ROWS, lookup=True, S=2)
    return W, spec, CFG


def _big(cfg):
    """The configuration with room for a staging area past the image."""
    return dataclasses.replace(cfg, DRAM_BYTES=2 * cfg.DRAM_BYTES)


def test_layout_covers_the_kv_bytes():
    """The arrays of Qwen3's image are its kv_bytes, page by page."""
    W, spec, cfg = _model("qwen3")
    eng = Engine(spec, W, cap=512, cfg=cfg)
    L = KV.Layout(eng.image, cfg.S)
    assert L.P == vt_tile(512) == 256 and L.pages == 2 and not L.stateful
    per_pos = sum(a.w for a in L.arrays)
    assert per_pos * eng.cap == eng.image.kv_bytes
    assert L.page_bytes() == cfg.S * per_pos * L.P
    assert L.split(300) == [(0, 256), (1, 44)]
    with pytest.raises(ValueError):
        L.regions(2)


@pytest.mark.parametrize("case, staged", [
    ("qwen3", False), ("qwen3", True), ("lfm2", False), ("lfm2", True), ("qwen35", False),
    ("qwen35", True), ("qwen35-design", False)])
def test_restore_is_the_context(case, staged):
    """Context A (300 tokens: a whole page and part of one), saved; the device then runs context
    B (another 280 tokens); A written back, at its position: the next 9 tokens' logits and the
    greedy continuation equal a fresh engine fed A, bit for bit. With staging the copies run
    on the simulator through the area."""
    W, spec, cfg = _model(case)
    if staged:
        cfg = _big(cfg)

    def make():
        return Engine(spec, W, cap=512, cfg=cfg, resident=True, prompt_runs=True)
    r = np.random.default_rng(1)
    A = [int(x) for x in r.integers(0, 1000, 300)]
    B = [int(x) for x in r.integers(0, 1000, 280)]
    nxt = [int(x) for x in r.integers(0, 1000, 9)]
    eng = make()
    area = (-(-eng.image.nbytes // 4096) * 4096, cfg.DRAM_BYTES // 4) if staged else None
    io = KV.ContextIO(eng, area=area)
    assert (io.area is not None) == staged
    eng.prefill(A)
    pages = io.layout.split(len(A))
    saved = io.read_pages(pages)
    state = io.read_state()
    assert [len(b) for b in saved] == [io.layout.page_bytes(m) for _, m in pages]
    assert len(state) == io.layout.state_bytes and (len(state) > 0) == io.layout.stateful
    eng.reset()
    eng.prefill(B)
    io.write_pages([(i, m, b) for (i, m), b in zip(pages, saved)])
    io.write_state(state)
    eng.adopt(len(A))
    got = eng.prefill(nxt)
    ref = make()
    want = ref.prefill(A + nxt)
    assert np.array_equal(_bits(got), _bits(want))
    t = int(np.argmax(got))
    for _ in range(4):
        x, y = eng.step(t), ref.step(t)
        assert np.array_equal(_bits(x), _bits(y))
        t = int(np.argmax(x))
    if staged:
        assert io.stats["copies"] >= 2


def test_partial_restore_of_a_longer_context():
    """Qwen3 (the KV caches hold the whole context): the pages of a 300-token context restore
    any prefix k of it (here 270, inside its second page; and 200, inside the first), the
    positions past k overwritten by what follows."""
    W, spec, cfg = _model("qwen3")
    r = np.random.default_rng(2)
    A = [int(x) for x in r.integers(0, 1000, 300)]
    eng = Engine(spec, W, cap=512, cfg=cfg)
    io = KV.ContextIO(eng)
    eng.prefill(A)
    pages = io.layout.split(len(A))
    saved = dict(zip(pages, io.read_pages(pages)))
    for k in (270, 200):
        tail = [int(x) for x in r.integers(0, 1000, 11)]
        eng.reset()
        eng.prefill([int(x) for x in r.integers(0, 1000, 290)])     # another context
        need = -(-k // io.layout.P)
        io.write_pages([(i, m, saved[(i, m)]) for i, m in pages[:need]])
        eng.adopt(k)
        got = eng.prefill(tail)
        want = Engine(spec, W, cap=512, cfg=cfg).prefill(A[:k] + tail)
        assert np.array_equal(_bits(got), _bits(want)), k


def test_pages_of_a_shared_prefix_are_the_same_bytes():
    """Two contexts with the same first 256 tokens: their page 0 bytes are equal (the store keeps
    it once), page 1 differs."""
    W, spec, cfg = _model("qwen3")
    r = np.random.default_rng(3)
    pre = [int(x) for x in r.integers(0, 1000, 256)]
    out = []
    for _ in range(2):
        eng = Engine(spec, W, cap=512, cfg=cfg)
        io = KV.ContextIO(eng)
        eng.prefill(pre + [int(x) for x in r.integers(0, 1000, 40)])
        out.append(io.read_pages(io.layout.split(296)))
    assert np.array_equal(out[0][0], out[1][0])
    assert not np.array_equal(out[0][1], out[1][1])
