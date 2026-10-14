"""Expert-major MoE prefill in tiles of T rows (Engine.tile_rows: docs/tiled_prefill.md 14) beside
the layer-major calls of R rows that a chat turn's short calls keep (streamed.gate_expert_major):
Qwen3.5-MoE's on by default (tiled.LAYER_ROWS), off with OTPU_TILE=0."""
import dataclasses

import numpy as np
import pytest

from opentpu.llm.qwen3 import Engine, device_config


def _tiny(wf: str = "int8", act_rows: int = 16):
    """A tiny Qwen3.5-MoE in the 35B's shape on the card's configuration (MCOLS 4, PAIR,
    DSTEP, STREAM; ACT_ROWS act_rows), chunks of 24 rows: (spec, W, Engine's arguments)."""
    import test_qwen35_moe as TQM
    from opentpu.llm import qwen35 as Q35
    spec, W = TQM._untied(TQM._tiny(4))
    cfg = device_config(spec, 256, rows=1, lookup=True, S=1, experts=6, wformat=wf, MCOLS=4,
                        PAIR=True, DSTEP=True, STREAM=True)
    return spec, W, dict(cap=256, cfg=dataclasses.replace(cfg, ACT_ROWS=act_rows), rows=1,
                         resident=True, experts=6, wformat=wf, layer_major=2,
                         expert_major=True), Q35


def _tiles(e) -> list:
    """The rows of the tiled runs an engine compiled."""
    return sorted({k[2] for k in e._layer_runs if isinstance(k, tuple) and k[-1] == "tile"})


@pytest.mark.parametrize("wf", ["int8", "fp4"])
def test_tiled_expert_major_runs_are_todays_bits(monkeypatch, wf):
    """Qwen3.5-MoE in the 35B's shape (a DeltaNet pair shares its q and k), a prompt's call of
    43 rows expert-major in tiles of 8 rows (tile_rows; layer 0's rows from the host's slot
    untiled, R a run) and a short call of 5 rows layer-major in runs of R = 2
    (gate_expert_major), against the same calls in today's runs of R = 2 rows: the logits, the
    layers' blocks (DeltaNet states and windows, KV caches) and the decode steps after, bit for
    bit. Under the MX numerics a 4-bit MM's output is one exact integer whether it pairs (2-row
    runs) or not (the tile's), so a tile in its whole forms gives today's bits in fp4 as in
    int8 (docs/tiled_prefill.md 11)."""
    import test_qwen35_moe as TQM
    from opentpu.host import streamed as ST
    from opentpu.llm import qwen35 as Q35
    monkeypatch.setattr(Q35, "PREFILL_CHUNK", 24)
    monkeypatch.setenv("OTPU_TILE", "1")
    spec, W = TQM._untied(TQM._tiny(4))
    cfg = device_config(spec, 256, rows=1, lookup=True, S=1, experts=6, wformat=wf, MCOLS=4,
                        PAIR=True, DSTEP=True, STREAM=True)
    kw = dict(cap=256, cfg=dataclasses.replace(cfg, ACT_ROWS=16), rows=1, resident=True,
              experts=6, wformat=wf, layer_major=2, expert_major=True)
    a = Engine(spec, W, tile_runs=True, tile_rows=8, **kw)
    b = Engine(spec, W, tile_runs=False, **kw)
    assert a.layer_tile and not b.layer_tile and a.tile_rows == 8
    em_runs = a._layer_runs                 # (the expert-major calls' programs)
    for e in (a, b):
        ST.gate_expert_major(e, 12)
    rng = np.random.default_rng(12)
    toks = [int(t) for t in rng.integers(0, 1000, 43)]
    la, lb = a.prefill(toks), b.prefill(toks)
    assert a.em_last == (43, True)
    tiles = {k[2] for k in em_runs if isinstance(k, tuple) and k[-1] == "tile"}
    assert 8 in tiles and max(k[2] for k in em_runs if isinstance(k, tuple) and k[0] == 0
                              and k[-1] != "tile" and isinstance(k[2], int)) <= 2
    assert np.array_equal(la.view(np.uint32), lb.view(np.uint32))
    more = [int(t) for t in rng.integers(0, 1000, 5)]
    la, lb = a.prefill(more), b.prefill(more)
    assert a.em_last == (5, False)
    assert np.array_equal(la.view(np.uint32), lb.view(np.uint32))
    img = a.image
    assert np.array_equal(a.backend.machine.slices[0].dram[img.layer0:img.head[0]],
                          b.backend.machine.slices[0].dram[img.layer0:img.head[0]])
    t = int(np.argmax(la))
    for _ in range(2):
        ga, gb = a.step(t), b.step(t)
        assert np.array_equal(ga.view(np.uint32), gb.view(np.uint32))
        t = int(np.argmax(ga))


@pytest.mark.parametrize("act_rows", [16, 4])
def test_tiles_are_on_by_default(monkeypatch, act_rows):
    """With OTPU_TILE unset a Qwen3.5-MoE engine made with Engine's defaults (as
    streamed.engine makes otpu-chat's and otpu-serve's) builds the mask tiles and the I/O
    scratch into its image and runs its expert-major calls in tiles of tiled.LAYER_ROWS = 8
    rows, at most ACT_ROWS (docs/tiled_prefill.md 14)."""
    spec, W, kw, Q35 = _tiny(act_rows=act_rows)
    monkeypatch.setattr(Q35, "PREFILL_CHUNK", 24)
    monkeypatch.delenv("OTPU_TILE", raising=False)
    e = Engine(spec, W, **kw)
    T = min(8, act_rows)
    assert e.layer_tile and e.tile_rows == T
    assert "amask" in e.image.lookup and "o" in e.image.io
    e.prefill([int(t) for t in np.random.default_rng(5).integers(0, 1000, 43)])
    assert _tiles(e) and max(_tiles(e)) == T


def test_otpu_tile_0_gives_todays_runs(monkeypatch):
    """OTPU_TILE=0 as a Qwen3.5-MoE engine is made: today's expert-major runs of layer_major's
    rows, on an image without the mask tiles or the I/O scratch (today's layout, and so its
    expert slots), and the default's logits, layer blocks and decode steps, bit for bit."""
    spec, W, kw, Q35 = _tiny("fp4")
    monkeypatch.setattr(Q35, "PREFILL_CHUNK", 24)
    monkeypatch.delenv("OTPU_TILE", raising=False)
    a = Engine(spec, W, **kw)
    monkeypatch.setenv("OTPU_TILE", "0")
    b = Engine(spec, W, **kw)
    assert a.layer_tile and not b.layer_tile
    assert "amask" not in b.image.lookup and "o" not in b.image.io
    toks = [int(t) for t in np.random.default_rng(6).integers(0, 1000, 43)]
    la, lb = a.prefill(toks), b.prefill(toks)
    assert max(_tiles(a)) == 8 and not _tiles(b)
    assert max(k[2] for k in b._layer_runs if isinstance(k, tuple) and k[0] == 1) == 2
    assert np.array_equal(la.view(np.uint32), lb.view(np.uint32))
    t = int(np.argmax(la))
    for _ in range(2):
        ga, gb = a.step(t), b.step(t)
        assert np.array_equal(ga.view(np.uint32), gb.view(np.uint32))
        t = int(np.argmax(ga))
