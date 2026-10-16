"""tools/asic/flow.py and paths.py without OpenROAD: the design points come from the targets and
name only otpu_asic_top's parameters; the constraints; native ORFS or Docker, and prep's yosys; the
paths' units, stages and kinds."""
import dataclasses
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "asic"))
import flow  # noqa: E402
import paths  # noqa: E402

from opentpu import target as T  # noqa: E402


def test_points_are_the_asic_targets():
    """Every ASIC target is a point; a point's parameters are otpu_asic_top's (names, the
    target's values) and its clock the target's core_mhz."""
    top = (ROOT / "rtl/asic/otpu_asic_top.sv").read_text()
    names = set(re.findall(r"parameter\s+(?:int|bit)\s+(\w+)\s*=", top.split(") (")[0]))
    asic = [t for t in T.names() if T.load(t).kind == "asic"]
    assert asic and set(asic) <= set(flow.POINTS)
    for p in flow.POINTS:
        t = T.load(flow.POINTS[p]["target"])
        prm = flow.params(p)
        assert set(prm) <= names, set(prm) - names
        assert prm["D"] == t.arch["D"] and prm["TMEM_WORDS"] == t.arch["TMEM_WORDS"]
        assert abs(flow.clock_ns(p) * t.platform["core_mhz"] - 1000) < 0.01
    # the small point keeps the banked TMEM (its reference results); the card's is r2's one copy
    # in its ASIC form (two-port macros at the core clock, no pump) with the card's
    # configuration but 8 ACT RAM rows: WAITC, MX E 4, the BAR-only collective unit
    small = flow.params("small")
    assert (small["TMEM_IMPL"], small["RPB"], small["WPB"]) == (1, 1, 1)
    assert not {"SEQ_WC", "COLL_GATHER", "MXU_E"} & set(small)
    card = flow.params("card")
    assert T.arch_hash(T.config("nangate45")) == \
        T.arch_hash(dataclasses.replace(T.config("ypcb-00338"), ACT_ROWS=8))
    assert {k: card[k] for k in ("MCOLS", "ACT_ROWS", "MXU_E", "SEQ_WC", "TMEM_IMPL", "TMEM_NSB",
                                 "TMEM_RPL", "TMEM_TDP", "TMEM_GST", "TMEM_PMP", "RPB",
                                 "COLL_GATHER")} == \
        dict(MCOLS=4, ACT_ROWS=8, MXU_E=4, SEQ_WC=1, TMEM_IMPL=3, TMEM_NSB=2, TMEM_RPL=1,
             TMEM_TDP=1, TMEM_GST=1, TMEM_PMP=0, RPB=64, COLL_GATHER=0)


def test_sources_are_the_targets_and_the_top():
    """The ASIC's sources: the target's list (the macro-tiled RAMs, not the inferred ones), the
    ASIC top; a variant's rewritten files from its src/."""
    s = [str(f) for f in flow.sources("small")]
    assert any(f.endswith("rtl/tech/asic/otpu_tech_ram.sv") for f in s)
    assert not any(f.endswith("rtl/tech/generic/otpu_tech_ram.sv") for f in s)
    assert s[-1].endswith("rtl/asic/otpu_asic_top.sv")
    v = [str(f) for f in flow.sources("small-lm3la6")]
    assert any(f.endswith("build/asic/small-lm3la6/src/otpu_vpu.sv") for f in v)


def test_sram_map_has_the_two_port_leaf():
    """The leaves: a 1RW1R macro size gets its own module (otpu_sram_1rw1r, port 0 a read or a
    write, port 1 a read); a design without one has none."""
    import fakeram
    ms = [fakeram.Macro("1r1w", 128, 32), fakeram.Macro("1rw1r", 4096, 32)]
    v = flow.sram_map(ms)
    assert "module otpu_sram_1rw1r" in v and "fakeram45_1rw1r_4096x32 u_m (.clk, .ce_in(ena)" in v
    assert ".ce1_in(enb), .addr1_in(ab), .rd1_out(rdb)" in v
    assert "module otpu_sram_1rw1r" not in flow.sram_map(ms[:1])


def test_card_places_the_mxu_macros_by_column():
    """The card's ORFS design places the MXU's macros with asic/nangate45/mxu_macros.tcl (the
    macro placer the rest); the small point leaves them all to the macro placer."""
    card = flow.config_mk("card", [])
    m = re.search(r"^export MACRO_PLACEMENT_TCL = (\S+)$", card, re.M)
    assert m and Path(m.group(1)).name == "mxu_macros.tcl" and Path(m.group(1)).exists()
    small = flow.config_mk("small", [])
    assert "MACRO_PLACEMENT_TCL" not in small
    # the TMEM's macros by their readers (mxu_macros.tcl reads TMEM_GAP_UM), global routing's
    # metal4 and up at 7%: the card's only
    assert re.search(r"^export TMEM_GAP_UM = 100$", card, re.M)
    assert "TMEM_GAP_UM" in Path(m.group(1)).read_text()
    m = re.search(r"^export FASTROUTE_TCL = (\S+)$", card, re.M)
    assert m and Path(m.group(1)).exists() and "metal4-" in Path(m.group(1)).read_text()
    assert "TMEM_GAP_UM" not in small and "FASTROUTE_TCL" not in small


def test_ports_are_latency_neutral_after_cts():
    """Every point's ports take the core clock's insertion delay on vclk after CTS (a source
    latency, which holds whatever a later stage propagates); the SDC itself keeps both ideal."""
    for p in ("small", "card"):
        m = re.search(r"^export POST_CTS_TCL = (\S+)$", flow.config_mk(p, []), re.M)
        assert m and Path(m.group(1)).name == "io_latency.tcl"
        tcl = Path(m.group(1)).read_text()
        assert "set_clock_latency -source" in tcl and "[get_clocks vclk]" in tcl
        assert "set_clock_latency" not in flow.sdc(p)
        # ... before the CTS step's repair_timing (ORFS runs POST_CTS_TCL after it)
        m = re.search(r"^export PRE_CTS_TCL = (\S+)$", flow.config_mk(p, []), re.M)
        assert m and Path(m.group(1)).name == "cts_latency.tcl"
        pre = Path(m.group(1)).read_text()
        assert "rename ::repair_timing_helper" in pre and "io_latency.tcl" in pre


def test_sdc_asks_the_targets_clock():
    sdc = flow.sdc("card")
    assert f"set clk_period {flow.clock_ns('card')}" in sdc
    assert "create_clock -name core_clock" in sdc and "[get_ports clk]" in sdc


def _orfs(d: Path) -> Path:
    """A built ORFS as orfs_home checks it: env.sh, the installed openroad."""
    (d / "tools/install/OpenROAD/bin").mkdir(parents=True)
    (d / "env.sh").write_text("")
    (d / "tools/install/OpenROAD/bin/openroad").write_text("")
    return d


def test_run_native_else_docker(tmp_path, monkeypatch):
    """ORFS_HOME (a built ORFS) runs make in it, on every core unless told; without it the
    Docker image on 6 CPUs; an ORFS_HOME without the built tools, or neither, says what to set."""
    cmds = []
    monkeypatch.setattr(flow, "out_dir", lambda p: tmp_path / p)
    monkeypatch.setattr(flow.subprocess, "run",
                        lambda cmd, **k: cmds.append(cmd) or subprocess.CompletedProcess(cmd, 0))
    orfs = _orfs(tmp_path / "orfs")
    monkeypatch.setenv("ORFS_HOME", str(orfs))
    assert flow.run("small") == 0
    assert cmds[-1][:2] == ["bash", "-c"] and cmds[-1][2].startswith(f"cd {orfs} && source env.sh")
    assert f"DESIGN_CONFIG={tmp_path / 'small' / 'config.mk'} " in cmds[-1][2]
    assert f"NUM_CORES={os.cpu_count()} " in cmds[-1][2] and cmds[-1][2].endswith(" finish")
    flow.run("small", 3, "cts")
    assert "NUM_CORES=3 " in cmds[-1][2] and cmds[-1][2].endswith(" cts")
    (orfs / "tools/install/OpenROAD/bin/openroad").unlink()
    with pytest.raises(RuntimeError, match="ORFS_HOME"):
        flow.run("small")
    monkeypatch.delenv("ORFS_HOME")
    monkeypatch.setattr(flow.shutil, "which", lambda n: f"/usr/bin/{n}")
    flow.run("small")
    assert cmds[-1][0] == "docker" and "--cpus=6" in cmds[-1] and flow.IMAGE in cmds[-1]
    assert "NUM_CORES=6 " in cmds[-1][-1]
    monkeypatch.setattr(flow.shutil, "which", lambda n: None)
    with pytest.raises(RuntimeError, match="ORFS_HOME"):
        flow.run("small")


def test_prep_yosys(tmp_path, monkeypatch):
    """YOSYS, else the native ORFS's own yosys, else the one on PATH; none is an error."""
    monkeypatch.delenv("YOSYS", raising=False)
    monkeypatch.delenv("ORFS_HOME", raising=False)
    monkeypatch.setattr(flow.shutil, "which", lambda n: None)
    with pytest.raises(RuntimeError, match="no yosys"):
        flow.yosys()
    orfs = _orfs(tmp_path / "orfs")
    (orfs / "tools/install/yosys/bin").mkdir(parents=True)
    (orfs / "tools/install/yosys/bin/yosys").write_text("")
    monkeypatch.setenv("ORFS_HOME", str(orfs))
    assert flow.yosys() == str(orfs / "tools/install/yosys/bin/yosys")
    monkeypatch.setenv("YOSYS", "/opt/oss-cad-suite/bin/yosys")
    assert flow.yosys() == "/opt/oss-cad-suite/bin/yosys"


def test_paths_units_and_stages():
    """Names as ORFS writes them: hierarchical (a kept module's '/'), flattened ('.'), pins."""
    w = paths.where
    assert w("u_slice/u_vpu/g_lane[0].g_slot[0].g_mma.u_ma.u_m/m[12]$_DFFE_PP_") == \
        {"unit": "u_vpu", "reg": "m", "stage": "fmul.s1"}
    assert w("u_slice/u_quant/u_sc/u_sc/r[3]$_DFFE_PP_/Q", "DFF_X1")["stage"] == "fmul.s2"
    assert w("u_slice/u_vpu/g_lane[1].g_slot[2].g_ma.u_ma.u_a.s2[5]$_DFF_P_")["stage"] == "fadd.s2"
    assert w("u_slice/u_vpu/g_lane[1].g_slot[2].g_ma.u_ma.u_a.r[0]$_DFF_P_")["stage"] == "fadd.s4"
    assert w("u_slice/u_seq/c_fp[162]$_DFFE_PP_") == {"unit": "u_seq", "reg": "c_fp", "stage": "c_fp"}
    assert w("u_slice/u_mxu/g_sys.g_ask[0].g_p[0].u_pe.br[1]$_DFF_P_")["stage"] == "pe"
    assert w("sw_addr[29]") == {"unit": "port", "reg": "sw_addr", "stage": "port"}
    m = w("u_slice/u_tmem/g_port[0].g_copy.g_bank[3].g_ram.u_ram/u_t/g_d[0].g_c[0].g_2p.u_m/"
          "g_4096x32.u_m/rd_out[5]", "fakeram45_1r1w_4096x32")
    assert m["unit"] == "u_tmem" and m["stage"] == "sram"
    assert w("u_slice/gq[3]$_DFF_P_")["unit"] == "slice"


def test_paths_kinds_and_summary():
    k = lambda a, b: paths.kind(paths.where(a), paths.where(b))   # noqa: E731
    assert k("u_slice/u_dma/is_st$_SDFFCE_PP0P_", "sw_req") == "io"
    assert k("u_slice/u_seq/q_pr[32]$_DFFE_PP_", "u_slice/u_seq/c_fp[1]$_DFFE_PP_") == "control"
    assert k("u_slice/u_dma/fs_k[0]$_DFFE_PP_",
             "u_slice/u_vpu/g_lane[0].g_slot[7].g_ma.u_ma.u_m/m[1]$_DFFE_PP_") == "fpipe"
    assert k("u_slice/u_mxu/hi0[0]$_DFF_P_", "u_slice/u_mxu/g_sys.g_ask[0].g_p[1].u_pe.x[3]$_DFF_P_") \
        == "systolic"
    txt = ("u_slice/u_seq/q_pr[32]$_DFFE_PP_/Q (DFF_X1) u_slice/u_seq/c_fp[162]$_DFFE_PP_/D (DFF_X1)"
           "    0.133\nu_slice/u_dma/is_st$_SDFFCE_PP0P_/Q (DFF_X1) sw_req (out)   -0.749\n")
    a = paths.parse_all(txt)
    assert [(p["kind"], p["slack"]) for p in a] == [("control", 0.133), ("io", -0.749)]
