"""tchip fork: ternary (t2) MMs on the RTL, against the ISA simulator, bit for bit (DRAM and
TMEM). Four blocks a chunk, one advance each; replay (M > MCOLS) and UNIT included."""
import numpy as np
import pytest

from opentpu import Config, isa as I, quant as Q, rtlsim
from opentpu.isasim import Machine

X, W, S, OUT = 0, 16384, 98304, 4096


@pytest.fixture(scope="module")
def have_verilator():
    import shutil
    if shutil.which("verilator") is None:
        pytest.skip("verilator not installed")
    return True


def _image(rng, N, KB, D):
    img = np.zeros(1 << 20, np.uint8)
    x = rng.standard_normal(4096).astype(np.float32)
    img[X:X + 16384] = x.view(np.uint8)
    Wf = (rng.standard_normal((N, KB * D)) * 0.02).astype(np.float32)
    rows, srows = Q.quantize_t2(Wf, D, per="row")
    srows[1::2] = (-srows[1::2].view(np.float32)).view(np.uint8)   # Sw of either sign
    assert W + rows.size <= S
    img[W:W + rows.size] = rows.reshape(-1)
    img[S:S + srows.size] = srows.reshape(-1)
    return img, rows.shape[1]


@pytest.mark.parametrize("mcols,M", [(4, 1), (4, 3), (4, 6)])
@pytest.mark.parametrize("KB", [1, 3, 4, 6])
@pytest.mark.parametrize("unit", [False, True])
def test_t2_mm_rtl_is_the_isa(have_verilator, mcols, M, KB, unit):
    cfg = Config(S=1, MCOLS=mcols, ACT_ROWS=8)
    D, N = cfg.D, 11
    img, rs = _image(np.random.default_rng(KB * 10 + M), N, KB, D)
    prog = [I.ld(X, 0, 4096), I.qact(0, M, 0, KB, KB * D),
            I.mm(W, S, OUT, N, KB, rs, N, M, 0, 4, wf=I.WT2, unit=unit),
            I.halt()]
    m = Machine(cfg, [prog], [img.copy()]).run()
    drams, tmems, _ = rtlsim.run(cfg, [prog], [img.copy()])
    assert np.array_equal(tmems[0], m.slices[0].tmem)
    assert np.array_equal(drams[0], m.slices[0].dram)
    y = m.slices[0].tget(OUT + np.arange(M * N))
    assert np.isfinite(y).all() and np.abs(y).max() > 0
