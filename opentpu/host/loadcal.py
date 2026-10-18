"""The DDR3 read taps centred under the core's load (docs/litedram.md section 10, "Read taps under
the core's load"; docs/timing_ffs.md section 8).

The calibration (the LiteDRAM core's CPU, or ddrcal from the host) puts each lane's read IDELAY
at the centre of the taps a burst reads right at with the core idle. With a core clock that is not
the controllers' sys clock (the m8 card at 150 MHz against 133.33), the core's switching sweeps
across the memory clocks' edges and moves the PHY's capture against the data by a different
amount on every read: under load (the MXU streaming weights, the DMA storing and loading) each
lane's window closes by 2-4 taps, mostly from the right, and at the idle centre a lane reads a
wrong beat now and then (r150fp 1da7d3b9 on the card: ECC errors under inference on four loads
of five; at 133.33 MHz the windows close by about 1 tap around the idle centre).

center() runs such a load on the core (one long hardware loop of turnaround.py's traffic: fp4
weights through the MXU beside 64 KiB DMA stores and loads), scans every lane's read window
meanwhile with LiteDRAM's BIST in the last quarter of each channel (all lanes at each tap in
turn), and leaves each lane at its window's centre under the load: the left of the two middles
of an even window, as the load closes the right edge. A lane whose window under load is
narrower than MIN_WINDOW keeps its calibrated tap (and says so). Board.scrub runs it once per
configuration, between two passes of zeros, when wanted_on(): the core clock is not the
controllers' (OTPU_LOAD_CENTER=1 / 0 forces it on / off). The BIST writes without ECC check bits
and the core's reads at the scan's far taps count ECC errors: the second pass of zeros and the
counters' clear after it (Board.scrub) leave neither behind.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from . import ddrcal

ENV = "OTPU_LOAD_CENTER"
MIN_WINDOW = 3              # taps under load for a lane to take its centre there
NTAPS = 32                  # IDELAYE2 taps
REPS, MIB = 10, 128         # per tap: REPS BIST reads of MIB MiB per channel (~12 s in all)
SEED = 0x0F1E2D3C4B5A6978
LOAD_ITERS = 2_000_000      # the load's loop: minutes, ended by Board.stop after the scan

# the load: turnaround.py's traffic in one hardware loop (an MM of R x H fp4 weights on port A,
# 64 KiB DMA tiles stored and reloaded on port B)
H, R, NS = 1024, 1024, 128
TILE = 4 * NS * NS


def wanted(core_khz, sys_hz: float, env: str | None = None) -> bool:
    """Centre the read taps under load: the core clock (CORE_KHZ) is not the controllers' sys
    clock. OTPU_LOAD_CENTER=1 / 0 (or `env`) forces it on / off."""
    v = os.environ.get(ENV, "") if env is None else env
    if v in ("0", "1"):
        return v == "1"
    return bool(core_khz) and abs(core_khz * 1e3 - sys_hz) > 1e3


def sys_hz(data: Path) -> float:
    """The controllers' sys clock (the build's sdram_init.py)."""
    ns = {}
    exec(Path(ddrcal.load_config(data)).read_text(), ns)
    return float(ns["phy"]["sys_hz"])


def wanted_on(b, data: Path | None = None) -> bool:
    """wanted() for the card behind Board b: a LiteDRAM bitstream with ECC (the board model has
    neither) whose core clock is not its controllers'."""
    from . import memcal
    if not getattr(b.t, "ecc", False) or not memcal.hostcal(b.t):
        return False
    return wanted(b.info().get("core_khz"), sys_hz(data or memcal.DATA))


def window(errs) -> tuple[int, int, int | None]:
    """(start, length, centre) of the longest run of taps with no wrong beat; an even run's
    centre is the left of its two middles."""
    start, n = ddrcal.Dram.windows(errs)
    return start, n, (start + (n - 1) // 2 if n else None)


def set_taps(k, taps) -> None:
    """Each lane m of channel k (ddrcal.Chan, or the PHY model) at read tap taps[m]."""
    for m, v in enumerate(taps):
        k.w("ddrphy_dly_sel", 1 << m)
        k.w("ddrphy_rdly_dq_rst", 1)
        for _ in range(v):
            k.w("ddrphy_rdly_dq_inc", 1)
    k.w("ddrphy_dly_sel", 0)


def _bist(k, base: int, beats: int, mode: int) -> None:
    k.w("bist_base", base)
    k.w("bist_length", beats)
    k.w("bist_mode", mode)
    k.w("bist_seed", SEED)
    k.w("bist_start", 1)


def bist_write(chans, base: int, beats: int, hz: float) -> None:
    """The BIST's data (SEED) at beats base .. base + beats of every channel."""
    for k in chans:
        _bist(k, base, beats, 0)
    for k in chans:
        ddrcal.bist_wait(k, beats, 0, hz, timeout=120)


def scan(chans, base: int, beats: int, hz: float, reps: int = REPS, nm: int = 9,
         ntaps: int = NTAPS) -> list:
    """err[ch][tap][lane]: every lane of every channel at each read tap in turn, `beats` beats
    from `base` (bist_write's) read back and checked, `reps` times over. Leaves the taps at
    ntaps - 1; the caller sets them."""
    err = [[[0] * nm for _ in range(ntaps)] for _ in chans]
    for _ in range(reps):
        for tap in range(ntaps):
            for k in chans:
                set_taps(k, [tap] * nm)
            for k in chans:
                _bist(k, base, beats, 1)
            for i, k in enumerate(chans):
                r = ddrcal.bist_wait(k, beats, 1, hz, timeout=120)
                err[i][tap] = [x + y for x, y in zip(err[i][tap], r["lanes"])]
    return err


def choose(err_ch, cal: list, min_window: int = MIN_WINDOW) -> list[dict]:
    """Per lane of one channel's scan: its window under load, and the tap it takes (the window's
    centre, or the calibrated tap where the window is narrower than min_window)."""
    out = []
    for m, c in enumerate(cal):
        start, n, ctr = window([row[m] for row in err_ch])
        ok = n >= min_window
        out.append({"start": start, "taps": n, "centre": ctr, "calibrated": c,
                    "tap": ctr if ok else c, "narrow": not ok})
    return out


def map_line(err_ch, m: int, total: int) -> str:
    """Lane m's scan, a character per tap: 1 no wrong beat, x some, 0 over 1%."""
    return "".join("1" if row[m] == 0 else "0" if row[m] > total // 100 else "x" for row in err_ch)


# ------------------------------------------------------------------------------ the load
def _kernel():
    from opentpu import language as ol

    @ol.jit
    def load(m, n_mm: int, n_dma: int, n_iter: int):
        xs = ol.quantize(ol.load(m.x))
        y = ol.empty([1, R])
        t = ol.load(m.src)
        u = ol.empty([NS * NS])
        per = -(-n_dma // n_mm)
        for _ in ol.range(n_iter):
            k = 0
            for i in range(n_mm):
                ol.dot(xs, m.w[i * R:(i + 1) * R, :], out=y)        # port A: the weights
                for _ in range(per):
                    if k < n_dma:
                        ol.store(m.dma[k, :], t)                    # port B: writes beside them,
                        if k:
                            ol.load(m.dma[k - 1, :], out=u)         # and reads of the previous tile
                        k += 1
                ol.store(m.out[i:i + 1, :], y)
    return load


def build(cfg, n_mm: int = 4, n_dma: int = 16, n_iter: int = LOAD_ITERS, seed: int = 0):
    """The load's image, layout (byte offsets) and program."""
    from opentpu import quant as Q
    from opentpu.llm.qwen3 import _qdesc, _tdesc
    rb, sb = Q.row_bytes(H, "fp4", 128), Q.scale_bytes(H, "fp4", 128)
    a = {"w": 0, "ws": n_mm * R * rb}
    a["x"] = a["ws"] + n_mm * R * sb
    a["src"] = (a["x"] + 4 * H + (1 << 20) - 1) >> 20 << 20
    a["dma"] = a["src"] + TILE
    a["out"] = a["dma"] + n_dma * TILE
    a["prog"] = (a["out"] + 4 * n_mm * R + 4095) // 4096 * 4096
    img = np.zeros(a["prog"], np.uint8)
    rng = np.random.default_rng(seed)
    q, sc = Q.quantize_mxu(rng.standard_normal((R, H)).astype(np.float32), "fp4", 128)
    q = np.ascontiguousarray(q).view(np.uint8).ravel()
    sc = np.ascontiguousarray(sc).view(np.uint8).ravel()
    for i in range(n_mm):
        img[a["w"] + i * q.size:a["w"] + (i + 1) * q.size] = q
        img[a["ws"] + i * sc.size:a["ws"] + (i + 1) * sc.size] = sc
    img[a["x"]:a["x"] + 4 * H] = rng.standard_normal(H).astype(np.float32).view(np.uint8)
    img[a["src"]:a["src"] + TILE] = rng.integers(0, 1 << 32, NS * NS, np.uint32).view(np.uint8)
    m = SimpleNamespace(w=_qdesc(a["w"], a["ws"], n_mm * R, H, 128, "fp4"),
                        x=_tdesc(a["x"], (1, H)), src=_tdesc(a["src"], (NS * NS,)),
                        dma=_tdesc(a["dma"], (n_dma, NS * NS)), out=_tdesc(a["out"], (n_mm, R)))
    prog = _kernel().trace(cfg, 0, {"m": m, "n_mm": n_mm, "n_dma": n_dma,
                                    "n_iter": n_iter}).finish()
    return img, a, prog


# ------------------------------------------------------------------------------ on the card
def calibrated(c, t, data: Path) -> list[list[int]]:
    """Each channel's calibrated read taps: the core's CPU's result, or the host's last one."""
    from . import memcal, selfcal
    if selfcal.present(c):
        res = selfcal.result(c, data / "sdram_init.py")
    else:
        res = {int(ch): r for ch, r in ((memcal.last(t) or {}).get("channels") or {}).items()}
    taps = [[x["tap"] for x in res.get(ch, {}).get("read", [])] for ch in (0, 1)]
    if any(len(x) != 9 for x in taps):
        raise RuntimeError(f"no calibrated read taps to start from ({taps})")
    return taps


def _path(t) -> Path:
    from .runstate import run_dir
    return run_dir() / f"{getattr(t, 'devname', None) or 'card'}.loadcal.json"


def center(b, log=print, data: Path | None = None, reps: int = REPS, mib: int = MIB) -> dict:
    """Centre every lane's read tap under the load (module docstring) on the card behind Board b.
    Expects DRAM written (Board.scrub's first pass: the load's reads are of beats it wrote).
    Raises, with the calibrated taps put back, when the load or the scan fails."""
    from . import memcal
    from .board import CH_BYTES, device_config
    from .runstate import ensure_run_dir, write_shared
    data = data or memcal.DATA
    t = b.t
    c = memcal.csr(t, data)
    chans = [ddrcal.Chan(c, ch) for ch in (0, 1)]
    cal = calibrated(c, t, data)
    hz = sys_hz(data)
    info = b.info()
    ch_beats = ((info.get("cfg") or {}).get("channel_bytes") or CH_BYTES) // 64
    base, region = ch_beats * 3 // 4, ch_beats // 4      # the last quarter of each channel
    beats = min(region, mib << 14)
    t0 = time.time()
    bist_write(chans, base, region, hz)
    cfg = device_config(info, DRAM_BYTES=1 << 30)
    img, lay, prog = build(cfg)
    b.write(0, img)
    b.load(lay["prog"], prog)
    b.start()
    try:
        err = scan(chans, base, beats, hz, reps)
    except BaseException:
        b.stop()
        for k, taps in zip(chans, cal):
            set_taps(k, taps)
        raise
    if b.stop():                    # it halted by itself: the scan did not all run under load
        for k, taps in zip(chans, cal):
            set_taps(k, taps)
        raise RuntimeError("loadcal: the load ended before the scan did; calibrated taps kept")
    lanes = [choose(err[ch], cal[ch]) for ch in (0, 1)]
    for k, ls in zip(chans, lanes):
        set_taps(k, [x["tap"] for x in ls])
    total = reps * beats
    out = {"time": time.time(), "seconds": round(time.time() - t0, 1),
           "core_khz": info.get("core_khz"), "sys_hz": hz, "reps": reps, "mib": mib,
           "channels": {ch: {"lanes": lanes[ch],
                             "map": [map_line(err[ch], m, total) for m in range(9)]}
                        for ch in (0, 1)}}
    for ch in (0, 1):
        ls = lanes[ch]
        log(f"DDR3 channel {ch}: read taps under the core's load {[x['tap'] for x in ls]} "
            f"(calibrated {cal[ch]}; windows {[x['taps'] for x in ls]} taps)"
            + "".join(f"; lane {m}: window {x['taps']} under load, kept its calibrated tap"
                      for m, x in enumerate(ls) if x["narrow"]))
    try:
        ensure_run_dir()
        write_shared(_path(t), json.dumps(out, indent=1))
    except (OSError, RuntimeError):
        pass                        # the result is also in the log
    return out
