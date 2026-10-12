"""Targets (opentpu/target.py, targets/*.toml) and the configuration registers (register map 4):
the loader, board_config from the selected target, the arch hash, the FPGA flow's generics
matching the RTL's defaults, and the host building its Config from the device."""
import dataclasses
import re
import subprocess
import sys
from pathlib import Path

import pytest

from opentpu import target as T
from opentpu.host import regs as R
from opentpu.host.board import Board, ConfigMismatch, check_fits, device_config
from opentpu.host.fake import FakeTransport
from opentpu.isasim import Config, board_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def no_env(monkeypatch):
    for k in ("OTPU_TARGET", "OTPU_MCOLS", "OTPU_LANES", "OTPU_PAIR", "OTPU_DSTEP", "OTPU_STREAM",
              "OTPU_ACT_ROWS"):
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


def test_every_target_loads():
    assert {"ypcb-00338", "bcu1525", "nangate45", "nangate45-small"} <= set(T.names())
    for n in T.names():
        t = T.load(n)
        cfg = T.config(t)
        assert isinstance(cfg, Config) and cfg.S == t.arch["S"]
        assert set(T.rtl_params(t)) >= {"MCOLS", "PLATFORM", "ARCH_HASH", "TMEM_IMPL"}
        d = t.to_dict()
        assert d["arch_hash"] == f"{T.arch_hash(cfg):08x}"
    ids = [T.load(n).platform_id for n in T.names()]
    assert len(ids) == len(set(ids)), "platform ids are unique"


def test_board_config_is_the_card_by_default(no_env):
    """board_config() before targets: the same Config (its fields are in every cache key), but
    for the MX build's ACT_ROWS 16 (0 before)."""
    legacy = Config(S=1, D=128, MCOLS=4, ACT_BLOCKS=128, LANES=8, ACT_ROWS=16,
                    TMEM_WORDS=1 << 16, IMEM_WORDS=1 << 15, DRAM_BYTES=1 << 32, PAIR=False,
                    DSTEP=False, STREAM=False)
    assert board_config() == legacy
    assert T.current().name == "ypcb-00338"
    no_env.setenv("OTPU_MCOLS", "2")
    no_env.setenv("OTPU_PAIR", "1")
    assert board_config() == dataclasses.replace(legacy, MCOLS=2, PAIR=True)


def test_board_config_follows_otpu_target(no_env):
    no_env.setenv("OTPU_TARGET", "bcu1525")
    c = board_config()
    assert (c.S, c.D, c.MCOLS, c.DRAM_BYTES) == (2, 128, 4, 1 << 32)
    no_env.setenv("OTPU_TARGET", "nangate45-small")
    c = board_config()
    assert (c.D, c.MCOLS, c.LANES, c.TMEM_WORDS, c.IMEM_WORDS) == (32, 2, 4, 16384, 8192)
    no_env.setenv("OTPU_TARGET", "no-such-board")
    with pytest.raises(T.TargetError, match="no target 'no-such-board'"):
        board_config()


def _hash_without_numerics(cfg) -> int:
    """arch_hash before the MX numerics (what today's production host computes)."""
    import hashlib
    import json
    d = dataclasses.asdict(cfg)
    s = json.dumps({k: (int(v) if isinstance(v, bool) else v) for k, v in sorted(d.items())},
                   separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(s.encode()).digest()[:4], "big")


def test_arch_hash_is_stable_and_sensitive():
    card = T.config("ypcb-00338")
    # in the bitstreams' CFG_HASH: keep it (the MX build at ACT_ROWS 16; ff270e6e at ACT_ROWS 4)
    assert f"{T.arch_hash(card):08x}" == "89417856"
    four = dataclasses.replace(card, ACT_ROWS=0)
    assert f"{T.arch_hash(four):08x}" == "ff270e6e"
    # production's CFG_HASH (ACT_ROWS 4, the previous numerics), what a host of the previous
    # numerics computes for it: it refuses MX builds (T.NUMERICS)
    assert f"{_hash_without_numerics(four):08x}" == "e170f967"
    # the ASIC reference keeps the card's arch but for its ACT RAM rows (MCOLS)
    assert T.arch_hash(four) == T.arch_hash(T.config("nangate45")), "same arch, same programs"
    for f in dataclasses.fields(Config):
        v = getattr(card, f.name)
        other = dataclasses.replace(card, **{f.name: (not v) if isinstance(v, bool) else v * 2
                                             if v else 1})
        assert T.arch_hash(other) != T.arch_hash(card), f.name


def test_build_config_applies_the_flow_knobs():
    t = T.load("ypcb-00338")
    assert T.build_config(t, MCOLS=4, LANES=8, ACT_ROWS=16, DSTEP=1) == T.config(t)
    assert T.build_config(t, MCOLS=4, LANES=8, DSTEP=1) == T.config(t)       # the target's 16
    c = T.build_config(t, MCOLS=2, ACT_ROWS=2, DSTEP=0)
    assert (c.MCOLS, c.ACT_ROWS, c.DSTEP, c.STREAM, c.PAIR) == (2, 0, False, False, True)
    assert T.build_config(t, MCOLS=4, ACT_ROWS=8).ACT_ROWS == 8


def _sv_defaults(path: Path) -> dict:
    """`parameter int NAME = value` defaults of a SystemVerilog module (simple expressions)."""
    out = {}
    for m in re.finditer(r"parameter\s+(?:int|bit)\s+(\w+)\s*=\s*([^,;\n/]+)", path.read_text()):
        v = m.group(2).strip().replace("1'b", "")
        try:
            out[m.group(1)] = int(eval(v, {}, dict(out)))     # noqa: S307 (our own RTL)
        except Exception:
            pass
    return out


def test_card_target_matches_the_rtl_defaults():
    """The card's target states what otpu_board / otpu_fpga_top_ld build by default, so passing
    the target's values as generics leaves the bitstream's netlist as it was."""
    p = T.rtl_params("ypcb-00338")
    board = _sv_defaults(ROOT / "rtl/shell/otpu_board.sv")
    for k in ("D", "TMEM_WORDS", "IMEM_WORDS", "ACT_BLOCKS", "FIFO_DEPTH", "WIN", "WPB",
              "TRACE_DEPTH", "PQ_WIN", "CHASH", "DSTEP"):
        assert p[k] == board[k], k
    assert p["RPB"] == 8 * p["LANES"]                       # otpu_board: RPB = 8 * LANES
    top = _sv_defaults(ROOT / "rtl/boards/ypcb-00338/otpu_fpga_top_ld.sv")
    for k in ("MCOLS", "ACT_ROWS", "VPU_CL", "MXU_IMPL", "LANES", "ULANES", "DDR_MTS", "PLATFORM"):
        assert p[k] == top[k], k
    assert T.sim_uarch("ypcb-00338") == {"WIN": 16, "RPB": 128, "WPB": 1, "FIFO_DEPTH": 1024}


def test_target_cli():
    run = lambda *a: subprocess.run([sys.executable, "-m", "opentpu.target", *a], cwd=ROOT,
                                    capture_output=True, text=True, check=True).stdout
    env = dict(line.split("=", 1) for line in run("env", "ypcb-00338").split())
    assert (env["T_MCOLS"], env["T_ACT_ROWS"], env["T_PLATFORM"], env["T_ARCH_HASH"]) == \
        ("4", "16", "1", "89417856")
    assert run("hash", "ypcb-00338", "MCOLS=4", "LANES=8", "ACT_ROWS=16", "DSTEP=1").strip() \
        == "89417856"
    assert run("hash", "ypcb-00338", "MCOLS=4", "LANES=8", "ACT_ROWS=4", "DSTEP=1").strip() \
        == "ff270e6e"
    assert "bcu1525" in run("list")


def test_bad_target_files(tmp_path):
    good = (ROOT / "targets/nangate45-small.toml").read_text()
    for edit, msg in ((("[uarch]", "[uarchx]"), "no \\[uarch\\]"),
                      (("TMEM_WORDS = 16384", "TMEM_WORDZ = 16384"), "not Config fields"),
                      (("DRAM_BYTES = 268435456", "DRAM_BYTES = 8589934592"), "over 4 GiB"),
                      (('kind = "asic"', 'kind = "rocket"'), "kind")):
        p = tmp_path / "nangate45-small.toml"
        p.write_text(good.replace(*edit))
        with pytest.raises(T.TargetError, match=msg):
            T.load(str(p))


# ------------------------------------------------------------------------------ register map 4
def test_cfg_register_codec():
    c = dict(S=2, TMEM_WORDS=1 << 16, IMEM_WORDS=1 << 15, ACT_BLOCKS=128, DRAM_BYTES=1 << 32,
             channels=2, channel_bytes=1 << 31, MXU_IMPL=2, TMEM_IMPL=1, VPU_CL=2, ULANES=8,
             platform=2, hash=0x8B699C81)
    assert R.cfg(R.cfg_words(c)) == c


def test_device_config_from_the_cfg_registers(no_env):
    """Map 4: the sizes come from the device; the hash must be the reported Config's."""
    t = FakeTransport(devname=None, regmap=4, pair=True, dstep=True, stream=True)
    info = Board(t).info()
    assert info["cfg"]["platform"] == 1 and info["cfg"]["hash"] == 0x89417856
    assert info["act_rows"] == 16 and info["caps"]["act_rows"]
    cfg = device_config(info)
    assert cfg == T.config("ypcb-00338")
    small = dict(R.cfg(t.cfg_words), TMEM_WORDS=1 << 15, IMEM_WORDS=1 << 14, ACT_BLOCKS=64,
                 DRAM_BYTES=1 << 30)
    small["hash"] = T.arch_hash(dataclasses.replace(cfg, TMEM_WORDS=1 << 15, IMEM_WORDS=1 << 14,
                                                    ACT_BLOCKS=64, DRAM_BYTES=1 << 30))
    info = Board(FakeTransport(devname=None, regmap=4, pair=True, dstep=True, stream=True,
                               cfg=small)).info()
    c2 = device_config(info)
    assert (c2.TMEM_WORDS, c2.IMEM_WORDS, c2.ACT_BLOCKS, c2.DRAM_BYTES) == \
        (1 << 15, 1 << 14, 64, 1 << 30)
    with pytest.raises(ConfigMismatch, match="TMEM_WORDS=65536 and the device has 32768"):
        check_fits(cfg, info)
    check_fits(dataclasses.replace(c2, DRAM_BYTES=1 << 24), info)     # less DRAM: fine
    info["cfg"]["hash"] ^= 1
    with pytest.raises(ConfigMismatch, match="CFG_HASH"):
        device_config(info)


def test_device_config_before_map_4_is_the_card(no_env):
    info = Board(FakeTransport(devname=None, regmap=3, pair=True, dstep=True,
                               stream=True)).info()
    assert info["cfg"] is None
    assert device_config(info) == T.config("ypcb-00338")
    check_fits(board_config(TMEM_WORDS=1 << 20), info)              # nothing to check against


# ------------------------------------------------------------------------------ RTL file lists
def _make_rtl(path: Path) -> list[str]:
    """The board Makefile's RTL list (rtl/ paths)."""
    m = re.search(r"RTL := \$\(addprefix \$\(ROOT\)/rtl/, (.*?)\)\n", path.read_text(), re.S)
    return ["rtl/" + f for f in m.group(1).replace("\\", " ").split()]


def _tcl_rtl(path: Path) -> list[str]:
    m = re.search(r"set rtl \[list \\(.*?)\]\n", path.read_text(), re.S)
    return m.group(1).replace("\\", " ").split()


def test_rtl_file_lists_agree():
    """Every flow's RTL list is opentpu.target.sources(): the board Makefile's lint and
    create_project.tcl (the board's top last, the Makefile linting it separately), the
    simulators' (rtlsim.RTL_SOURCES: the generic cells, plus the simulation-only DRAM model,
    native adapter and top) and tools/synth's."""
    from opentpu import rtlsim
    card = [f"rtl/{f}" for f in T.sources("ypcb-00338")]
    tcl = _tcl_rtl(ROOT / "boards/ypcb-00338/vivado/create_project.tcl")
    assert tcl == card
    make = _make_rtl(ROOT / "boards/ypcb-00338/Makefile")
    assert make == [f for f in card if not f.endswith("otpu_fpga_top_ld.sv")]
    sim = [f for f in rtlsim.RTL_SOURCES
           if f not in ("mem/otpu_dram.sv", "mem/otpu_native_dram.sv", "top/otpu_top.sv")]
    assert sim == T.sources("nangate45", "generic")
    for f in T.CORE + T.SHELL + sum(T.TECH.values(), []) + sum(T.BOARDS.values(), []):
        assert (ROOT / "rtl" / f).exists(), f
    bcu = [f"rtl/{f}" for f in T.sources("bcu1525")]
    assert "rtl/tech/xilinxus/otpu_tech_dsp.sv" in bcu
    assert _tcl_rtl(ROOT / "boards/bcu1525/vivado/create_project.tcl") == bcu
    assert _make_rtl(ROOT / "boards/bcu1525/Makefile") == \
        [f for f in bcu if not f.endswith("otpu_fpga_top_bcu.sv")]


def test_bcu1525_target_matches_its_rtl_defaults():
    """The BCU1525's top and sys wrapper default to its target's values (run_vivado.sh passes
    them as generics too), with the channel layout otpu_board and otpu_ctrl assume (two channels
    per slice, 2 GiB a slice addresses of each: CFG_DRAM)."""
    p = T.rtl_params("bcu1525")
    for f in ("otpu_fpga_top_bcu.sv", "otpu_bcu_sys.sv"):
        top = _sv_defaults(ROOT / "rtl/boards/bcu1525" / f)
        for k in ("S", "D", "MCOLS", "ACT_BLOCKS", "TMEM_WORDS", "IMEM_WORDS", "FIFO_DEPTH",
                  "LANES", "WIN", "WPB", "MXU_IMPL", "MXU_CL", "VPU_CL", "ULANES", "TRACE_DEPTH",
                  "PQ_WIN", "CHASH", "DSTEP", "DDR_MTS", "PLATFORM"):
            assert p[k] == top[k], (f, k)
    assert p["RPB"] == 8 * p["LANES"]                       # the tops: RPB = 8 * LANES
    assert (p["NCH"], p["CH_LOG2"], p["DRAM_LOG2"]) == (2, 31, 32)


def test_tech_families_define_the_same_cells():
    """rtl/tech/<family>/ files define the same modules with the same ports and parameters."""
    def iface(path: Path) -> dict:
        txt = re.sub(r"//.*", "", path.read_text())
        mods = {}
        for m in re.finditer(r"module\s+(\w+)\s*#\((.*?)\)\s*\((.*?)\);", txt, re.S):
            norm = lambda s: re.sub(r"\s+", " ", s).strip()
            mods[m.group(1)] = (norm(m.group(2)), norm(m.group(3)))
        return mods
    ref = iface(ROOT / "rtl/tech/generic/otpu_tech_dsp.sv")
    assert set(ref) == {"otpu_pe", "otpu_mxce"}
    for fam in ("xilinx7", "xilinxus"):
        assert iface(ROOT / f"rtl/tech/{fam}/otpu_tech_dsp.sv") == ref, fam


def test_config_summary_names_the_target(no_env):
    from opentpu.host.board import config_summary
    kw = dict(devname=None, ddr_mts=1066, pair=True, dstep=True, stream=True)
    old = config_summary(Board(FakeTransport(regmap=3, **kw)).info())
    assert old == "D=128 MCOLS=4 LANES=8, register map 3, core 100 MHz, DDR3-1066, build 1234abcd"
    new = config_summary(Board(FakeTransport(regmap=4, **kw)).info())
    assert "register map 4, ypcb-00338 (S=1, 2 channels x 2048 MiB)" in new and "DDR3-1066" in new
    assert T.by_platform(2).name == "bcu1525" and T.by_platform(999) is None
