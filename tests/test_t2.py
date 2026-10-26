"""tchip fork: ternary weights (t2) and QUAD column reuse.

A t2 MM multiplies the same integers as an int8 MM with row scales (RSC) whose weights are the
ternary values, so the two give the same bits. QUAD (PAIR on t2) takes four K-blocks per chunk
from ACT rows j + q*M (QACT DUP4 fills them) and must change no bit either."""
import dataclasses

import numpy as np
import pytest

from opentpu import Config, isa as I, quant as Q
from opentpu.isasim import Machine

WA, WB, SA, XA, OUT = 16384, 1 << 18, 1 << 19, (1 << 19) + (1 << 16), 8192


def _run(cfg, dram, prog):
    cfg = dataclasses.replace(cfg, DRAM_BYTES=max(cfg.DRAM_BYTES, dram.size))
    return Machine(cfg, [prog + [I.halt()]], [dram.copy()]).run().slices[0]


def _case(rng, N, KB, M, D):
    W = (rng.standard_normal((N, KB * D)) * 0.02).astype(np.float32)
    rows, srows = Q.quantize_t2(W, D, per="row")
    q = Q.DEC_T2[Q.unpack2(rows)][:, :KB * D]                 # the ternary values
    x = rng.standard_normal((M, KB * D)).astype(np.float32)
    dram = np.zeros(1 << 20, np.uint8)
    dram[XA:XA + x.nbytes] = x.view(np.uint8).reshape(-1)
    dram[WA:WA + rows.size] = rows.reshape(-1)                 # t2 rows, rs = rows.shape[1]
    dram[WB:WB + q.size] = q.view(np.uint8).reshape(-1)        # the same values as int8 rows
    dram[SA:SA + srows.size] = srows.reshape(-1)               # RSC scale rows: Sw, 4 bytes
    return x, W, rows, srows, dram


@pytest.mark.parametrize("KB", [1, 2, 3, 4, 5, 8])
@pytest.mark.parametrize("unit", [False, True])
def test_t2_is_int8_rsc_bit_for_bit(KB, unit):
    D, N, M = 32, 9, 2
    cfg = Config(S=1, D=D, MCOLS=4)
    rng = np.random.default_rng(KB + 10 * unit)
    x, W, rows, srows, dram = _case(rng, N, KB, M, D)
    pre = [I.ld(XA, 0, x.size), I.qact(0, M, 0, KB, KB * D)]
    got = _run(cfg, dram, pre + [I.mm(WA, SA, OUT, N, KB, rows.shape[1], N, M, 0, 4,
                                      unit=unit, wf=I.WT2)]).tget(OUT + np.arange(M * N))
    want = _run(cfg, dram, pre + [I.mm(WB, SA, OUT, N, KB, KB * D, N, M, 0, 4, unit=unit,
                                       wf=I.W8, rsc=True)]).tget(OUT + np.arange(M * N))
    assert np.array_equal(got.view(np.uint32), want.view(np.uint32))


@pytest.mark.parametrize("KB", [1, 3, 4, 5, 8, 9])
@pytest.mark.parametrize("MCOLS,M", [(4, 1), (8, 1), (8, 2)])
def test_quad_changes_no_bit(KB, MCOLS, M):
    D, N = 32, 7
    cfg = Config(S=1, D=D, MCOLS=MCOLS, PAIR=True)
    rng = np.random.default_rng(100 * KB + MCOLS + M)
    x, W, rows, srows, dram = _case(rng, N, KB, M, D)
    mm = lambda pair: I.mm(WA, SA, OUT, N, KB, rows.shape[1], N, M, 0, 4, wf=I.WT2, pair=pair)
    quad = _run(cfg, dram, [I.ld(XA, 0, x.size), I.qact(0, M, 0, KB, KB * D, dup4=True),
                            mm(True)]).tget(OUT + np.arange(M * N))
    plain = _run(cfg, dram, [I.ld(XA, 0, x.size), I.qact(0, M, 0, KB, KB * D),
                             mm(False)]).tget(OUT + np.arange(M * N))
    assert np.array_equal(quad.view(np.uint32), plain.view(np.uint32))


def test_t2_matches_float_reference():
    D, N, KB, M = 128, 64, 20, 1                                # K = 2560, BitNet's width
    cfg = Config(S=1, D=D, MCOLS=4, PAIR=True)
    rng = np.random.default_rng(1)
    x, W, rows, srows, dram = _case(rng, N, KB, M, D)
    y = _run(cfg, dram, [I.ld(XA, 0, x.size), I.qact(0, M, 0, KB, KB * D, dup4=True),
                         I.mm(WA, SA, OUT, N, KB, rows.shape[1], N, M, 0, 4, wf=I.WT2,
                              pair=True)]).tget(OUT + np.arange(N))
    ref = x @ Q.dequantize_t2(rows, srows, KB * D).T            # float activations
    rel = np.linalg.norm(y - ref[0]) / np.linalg.norm(ref[0])
    assert rel < 0.02, rel                                      # only int8 activation error


def test_t2_field_checks():
    with pytest.raises(Exception):
        I.mm(WA, SA, OUT, 4, 3, 128, 4, 1, 0, 4, wf=I.WT2, pair=True, k0=2)   # k0 % 4
    with pytest.raises(Exception):
        I.qact(0, 1, 0, 1, 32, dup=True, dup4=True)
    cfg = Config(S=1, D=32, MCOLS=4, PAIR=True)
    dram = np.zeros(1 << 18, np.uint8)
    with pytest.raises(Exception):                              # QUAD needs 4*M <= MCOLS
        _run(cfg, dram, [I.qact(0, 1, 0, 1, 32),
                         I.mm(WA, SA, OUT, 4, 1, 32, 4, 2, 0, 4, wf=I.WT2, pair=True)])


def test_t2_bytes_are_a_quarter_of_int8():
    assert Q.row_bytes(2560, "t2") == 640                       # a quarter of int8's 2560
    assert Q.row_bytes(6912, "t2") == 1792                      # padded to whole 128-byte chunks
