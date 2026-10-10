"""tools/asic/flow.py and paths.py without OpenROAD: the design points come from the targets and
name only otpu_asic_top's parameters; the constraints; the paths' units, stages and kinds."""
import re
import sys
from pathlib import Path

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
        assert (prm["TMEM_IMPL"], prm["RPB"], prm["WPB"]) == (1, 1, 1)
        assert abs(flow.clock_ns(p) * t.platform["core_mhz"] - 1000) < 0.01


def test_sources_are_the_targets_and_the_top():
    """The ASIC's sources: the target's list (the macro-tiled RAMs, not the inferred ones), the
    ASIC top; a variant's rewritten files from its src/."""
    s = [str(f) for f in flow.sources("small")]
    assert any(f.endswith("rtl/tech/asic/otpu_tech_ram.sv") for f in s)
    assert not any(f.endswith("rtl/tech/generic/otpu_tech_ram.sv") for f in s)
    assert s[-1].endswith("rtl/asic/otpu_asic_top.sv")
    v = [str(f) for f in flow.sources("small-lm3la6")]
    assert any(f.endswith("build/asic/small-lm3la6/src/otpu_vpu.sv") for f in v)


def test_sdc_asks_the_targets_clock():
    sdc = flow.sdc("card")
    assert f"set clk_period {flow.clock_ns('card')}" in sdc
    assert "create_clock -name core_clock" in sdc and "[get_ports clk]" in sdc


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
