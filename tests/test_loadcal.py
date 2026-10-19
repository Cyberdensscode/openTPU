"""opentpu.host.loadcal: the DDR3 read taps centred under the core's load. The window and its
centre, when it is wanted, the scan against ddrcal's PHY model, and the load on the ISA simulator."""
from pathlib import Path

import numpy as np

from opentpu import Config
from opentpu.host import ddrcal as C
from opentpu.host import loadcal as L
from opentpu.isasim import Machine

DATA = Path(__file__).parent / "data" / "litedram"


def test_a_window_s_centre_is_the_left_middle_of_an_even_run():
    assert L.window([1, 0, 0, 0, 0, 0, 1]) == (1, 5, 3)
    assert L.window([1, 1, 0, 0, 0, 0, 1, 0]) == (2, 4, 3)      # 2..5: 3, not 4
    assert L.window([5, 5, 5]) == (0, 0, None)


def test_wanted_when_the_core_clock_is_not_the_controllers():
    assert L.wanted(150000, 133.333333e6, env="")
    assert L.wanted(100000, 133.333333e6, env="")
    assert not L.wanted(133333, 133.333333e6, env="")              # E3: the same clock
    assert not L.wanted(None, 133.333333e6, env="")                 # no CORE_KHZ: as before
    assert L.wanted(133333, 133.333333e6, env="1")
    assert not L.wanted(150000, 133.333333e6, env="0")


def test_the_scan_finds_each_lane_s_window_and_takes_its_centre():
    fake = C.FakeCsr(DATA)
    C.calibrate_channel(fake, DATA, stride=4, log=lambda *_: None)
    cal = list(fake.rd)
    err = L.scan([fake], base=0, beats=1 << 10, hz=133.333e6, reps=1)
    lanes = L.choose(err[0], cal)
    for m, x in enumerate(lanes):
        assert (x["start"], x["taps"]) == (fake.LO[m], min(fake.HI[m], 31) - fake.LO[m] + 1)
        assert x["tap"] == fake.LO[m] + (min(fake.HI[m], 31) - fake.LO[m]) // 2
        assert not x["narrow"]
    L.set_taps(fake, [x["tap"] for x in lanes])
    assert fake.rd == [x["tap"] for x in lanes]
    assert all(fake.good(m) for m in range(fake.nm))


def test_a_lane_narrower_than_the_minimum_keeps_its_calibrated_tap():
    err = [[0 if t in (9, 10) else 5, 0 if 4 <= t <= 12 else 5] for t in range(32)]
    a, b = L.choose(err, [11, 7])
    assert (a["taps"], a["tap"], a["narrow"]) == (2, 11, True)
    assert (b["taps"], b["tap"], b["narrow"]) == (9, 8, False)


def test_a_lane_takes_the_widest_eye_under_load_over_its_bitslips():
    """Lane 0's calibrated eye narrows under load (taps 20-22) while the same data reads at the
    next bitslip a bit time earlier on the IDELAY (taps 8-14): the scan at the bitslips either
    side finds it, the lane moves there and reads right; lane 1's other eye is no wider (the
    lower tap of equals: it moves too); every other lane keeps its bitslip and centre."""
    fake = C.FakeCsr(DATA)
    fake.RB[0], fake.LO[0], fake.HI[0] = 3, 18, 24
    fake.RB[1], fake.LO[1], fake.HI[1] = 4, 16, 21
    res = C.calibrate_channel(fake, DATA, stride=4, log=lambda *_: None)
    cs = ([[x["bitslip"] for x in res["read"]]], [res["bit_offsets"]])
    cal = [x["tap"] for x in res["read"]]
    assert cs[0][0][:2] == [3, 4]
    fake.LO[0], fake.HI[0] = 20, 22                     # the load closes the calibrated eye
    fake.EYES[0] = [(4, 8, 14)]
    fake.EYES[1] = [(3, 4, 9)]                          # as wide as lane 1's own (16-21)
    errs, ok = L.scan_slips([fake], cs, L.SLIPS, base=0, beats=1 << 10, hz=133.333e6, reps=1)
    lanes = L.choose_slips(errs[0], cal, ok[0])
    assert (lanes[0]["slip"], lanes[0]["taps"], lanes[0]["tap"]) == (1, 7, 11)
    assert (lanes[1]["slip"], lanes[1]["taps"], lanes[1]["tap"]) == (-1, 6, 6)
    for m in range(2, fake.nm):
        assert lanes[m]["slip"] == 0
        assert lanes[m]["tap"] == fake.LO[m] + (min(fake.HI[m], 31) - fake.LO[m]) // 2
    L.set_slips(fake, [s + x["slip"] for s, x in zip(cs[0][0], lanes)], cs[1][0])
    L.set_taps(fake, [x["tap"] for x in lanes])
    assert all(fake.good(m) for m in range(fake.nm))


def test_a_lane_at_the_bitslip_range_s_end_does_not_wrap():
    assert L.slips_ok(0, [0] * 8, 1) and not L.slips_ok(0, [0] * 8, -1)
    assert not L.slips_ok(7, [0] * 8, 1) and L.slips_ok(7, [0] * 8, -1)
    assert not L.slips_ok(5, [0, 2, 0, 0, 0, 0, 0, 0], 1)        # its bit 1 would be at 8
    assert L.slips_ok(5, [0, -2, 0, 0, 0, 0, 0, 0], 1)


def test_a_narrow_lane_keeps_its_calibrated_tap_and_bitslip():
    narrow = [[0 if t in (9, 10) else 5] for t in range(32)]
    wide = [[0 if 3 <= t <= 8 else 5] for t in range(32)]
    can = {0: [True], 1: [True], -1: [True]}
    lanes = L.choose_slips({0: narrow, 1: narrow, -1: narrow}, [11], can)
    assert (lanes[0]["slip"], lanes[0]["tap"], lanes[0]["narrow"]) == (0, 11, True)
    lanes = L.choose_slips({0: narrow, 1: wide, -1: narrow}, [11], {**can, 1: [False]})
    assert (lanes[0]["slip"], lanes[0]["narrow"]) == (0, True)            # it could not move by 1
    lanes = L.choose_slips({0: narrow, 1: wide, -1: narrow}, [11], can)
    assert (lanes[0]["slip"], lanes[0]["tap"], lanes[0]["narrow"]) == (1, 5, False)


def test_slips_off_scans_the_calibrated_bitslips_only():
    assert L.slips_wanted(env="") and L.slips_wanted(env="1") and not L.slips_wanted(env="0")


def test_the_load_runs_on_the_isa_simulator():
    """A few iterations of the load (the m8 card's TMEM): the DMA's tiles are the source tile (it
    stores them and reloads the one before), and every row of the MMs' results is written."""
    cfg = Config(S=1, DRAM_BYTES=1 << 25, TMEM_WORDS=131072)
    img, a, prog = L.build(cfg, n_mm=2, n_dma=4, n_iter=3)
    dram = np.concatenate([img, np.zeros(cfg.DRAM_BYTES - len(img), np.uint8)])
    m = Machine(cfg, [prog], [dram]).run()
    got = np.asarray(m.slices[0].dram)
    src = img[a["src"]:a["src"] + L.TILE]
    for k in range(4):
        assert np.array_equal(got[a["dma"] + k * L.TILE:a["dma"] + (k + 1) * L.TILE], src)
    out = got[a["out"]:a["out"] + 4 * 2 * L.ROWS * L.R].view(np.float32).reshape(2 * L.ROWS, L.R)
    assert np.isfinite(out).all() and (np.abs(out).sum(1) > 0).all()
