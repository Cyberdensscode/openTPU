"""The cell families (rtl/tech/<family>): the vendor cells compute what the generic models do,
cycle for cycle. sim/tech/tb_tech_dsp.sv drives every otpu_pe variant and otpu_mxce with random
operands and enables; the trace with rtl/tech/xilinx7 (on yosys's DSP48E1 simulation model) must
equal the trace with rtl/tech/generic. (rtl/tech/xilinxus's DSP48E2 has no open model: Vivado's
simulator checks it, docs/targets.md.)"""
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _yosys_xilinx() -> Path | None:
    try:
        d = subprocess.run(["yosys-config", "--datdir"], capture_output=True, text=True,
                           check=True).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    p = Path(d) / "xilinx" / "cells_sim.v"
    return p if p.exists() else None


def _trace(tmp: Path, family: str, extra: list[Path], seeds) -> list[list[str]]:
    obj = tmp / f"obj_{family}"
    r = subprocess.run(["verilator", "--binary", "--timing", "-Wno-fatal", "-Wno-lint",
                        "-Wno-style", "-Wno-SPECIFYIGN", "--top-module", "tb_tech_dsp",
                        "-Mdir", str(obj), str(ROOT / "sim/tech/tb_tech_dsp.sv"),
                        str(ROOT / f"rtl/tech/{family}/otpu_tech_dsp.sv"), *map(str, extra)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    out = []
    for s in seeds:
        run = subprocess.run([str(obj / "Vtb_tech_dsp"), f"+seed={s}"], capture_output=True,
                             text=True, timeout=120, check=True)
        out.append([l for l in run.stdout.splitlines() if l[:1].isdigit()])
    return out


@pytest.mark.skipif(shutil.which("verilator") is None, reason="no Verilator")
def test_xilinx7_cells_equal_the_generic_models(tmp_path):
    cells = _yosys_xilinx()
    if cells is None:
        pytest.skip("no yosys Xilinx simulation models (yosys-config --datdir)")
    text = cells.read_text()
    i = text.index("module DSP48E1 ")
    model = tmp_path / "dsp48e1.v"
    model.write_text(text[i:text.index("endmodule", i) + len("endmodule")] + "\n")
    seeds = (3, 7, 11)
    gen = _trace(tmp_path, "generic", [], seeds)
    x7 = _trace(tmp_path, "xilinx7", [model], seeds)
    for s, a, b in zip(seeds, gen, x7):
        assert len(a) == 3984 and a == b, f"seed {s}"
