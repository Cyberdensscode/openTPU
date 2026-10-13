"""tools/asic/fakeram.py: the abstract SRAM macros of the NanGate45 flow (docs/asic.md)."""
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "asic"))
import fakeram as F  # noqa: E402

# fakeram45_* of ORFS's nangate45 platform (bsg_fakeram + CACTI): depth, width -> area (um^2)
REF = {(256, 32): 5065.704, (1024, 32): 16406.082, (2048, 39): 45478.818, (512, 64): 17301.438,
       (64, 7): 387.296, (256, 96): 13701.660, (128, 32): 3295.740}


@pytest.mark.parametrize("depth,width", sorted(REF))
def test_area_model_follows_fakeram45(depth, width):
    """The 1RW model is within 30% of bsg_fakeram's macro of the same size (the fit's worst is
    29%), and the LEF's size holds the model's area (snapped up, never down)."""
    m = F.Macro("1rw", depth, width)
    assert abs(m.area_model / REF[depth, width] - 1) < 0.30
    w, h = m.size
    assert w * h >= m.area_model * 0.999


@pytest.mark.parametrize("kind", ["1rw", "1r1w", "1rw1r"])
@pytest.mark.parametrize("depth,width", [(8192, 32), (1024, 128), (128, 20), (2, 1)])
def test_lef_lib_consistent(tmp_path, kind, depth, width):
    m = F.Macro(kind, depth, width)
    files = F.write(m, tmp_path)
    lef, lib = files["lef"].read_text(), files["lib"].read_text()
    w, h = map(float, re.search(r"SIZE ([\d.]+) BY ([\d.]+)", lef).groups())
    assert abs(w / F.SITE_W - round(w / F.SITE_W)) < 1e-6
    assert abs(h / F.SITE_H - round(h / F.SITE_H)) < 1e-6
    A = m.abits
    pins = re.findall(r"^  PIN (\S+)", lef, re.M)
    want = {"1rw": 3 * width + A + 3, "1r1w": 3 * width + 2 * A + 3,
            "1rw1r": 4 * width + 2 * A + 4}[kind]
    assert len(pins) == want + 2 and {"VDD", "VSS"} <= set(pins)
    for y in map(float, re.findall(r"LAYER metal3 ;\n\s+RECT 0.000 ([\d.]+)", lef)):
        assert F.MARGIN <= y <= h - F.MARGIN
        assert abs(y / 0.14 - round(y / 0.14)) < 1e-6         # on the metal3 track grid
    # the Liberty: the same area and pins (buses by name), balanced braces, a clock arc
    assert lib.count("{") == lib.count("}")
    assert abs(float(re.search(r"area : ([\d.]+)", lib).group(1)) - w * h) < 0.01
    buses = set(re.findall(r"bus\((\w+)\)", lib)) | set(re.findall(r"pin\((\w+)\)", lib))
    names = {p.split("[")[0] for p in pins} - {"VDD", "VSS"}
    assert buses == names
    assert f'cell_rise(scalar) {{ values ("{m.clk_to_q:.3f}")' in lib
    if kind == "1r1w":
        assert "address : r_addr_in" in lib and "address : w_addr_in" in lib
        assert m.area_model == pytest.approx(F.Macro("1rw", depth, width).area_model * 1.5)
    if kind == "1rw1r":                 # two read buses, the write on port 0's address
        assert re.findall(r"address : (\w+)", lib) == ["addr_in", "addr1_in", "addr_in", "addr_in"]
        assert lib.count("timing_type : rising_edge") == 2
        assert m.area_model == pytest.approx(F.Macro("1r1w", depth, width).area_model)


@pytest.mark.skipif(shutil.which("verilator") is None, reason="no verilator")
@pytest.mark.parametrize("kind", ["1rw", "1r1w", "1rw1r"])
def test_verilog_models_lint(tmp_path, kind):
    m = F.Macro(kind, 64, 24)
    files = F.write(m, tmp_path)
    for f in (files["v"], files["bb"]):
        r = subprocess.run(["verilator", "--lint-only", "-Wno-fatal", "-Wno-DECLFILENAME",
                            str(f)], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
    assert "(* blackbox *)" in files["bb"].read_text() and "always" not in files["bb"].read_text()


def test_parse_and_names():
    m = F.parse("1r1w:8192x32")
    assert (m.kind, m.depth, m.width, m.name) == ("1r1w", 8192, 32, "fakeram45_1r1w_8192x32")
    assert F.parse("1rw1r:4096x32").name == "fakeram45_1rw1r_4096x32"
    with pytest.raises(ValueError):
        F.Macro("2rw", 64, 8)
