"""The MX MXU (docs/isa.md "MM", "Weight formats", "Quantization"; docs/mx_mxu.md): the ISA
simulator against a scalar model written from the definition -- QACT's and QST's row scale and
power-of-two group offsets, and MM's exact integer per output (B = m << (W - dw - da), groups
whose shift is negative drop) with one fp32 epilogue, for every weight format, UNIT, RSC, PAIR,
k0, ACC and ASCALE."""
import numpy as np
import pytest

from opentpu import Config, fp32 as F, isa as I, quant as Q
from opentpu.isasim import Machine, SimError

X, WA, SA, OUT, AS = 0, 1 << 15, 3 << 14, 1 << 12, 1 << 11
DRAM = 1 << 17


def _f(v):
    return np.float32(F.ftz(np.float32(v)))


def _amax(v):
    return max((abs(_f(t)) for t in v), key=lambda t: np.float32(t).view(np.uint32))


def _offset(a, amax):
    """The largest d <= 31 with a * 2^d <= amax (a <= amax), 0 for a == 0."""
    if a == 0:
        return 0
    d = 0
    while d < 31 and np.float64(a) * 2.0 ** (d + 1) <= np.float64(amax):
        d += 1
    return d


def ref_quantize(row, g, row_mode):
    """docs/isa.md "Quantization" of one row, scalar: (q, S, d per group)."""
    amax = _amax(row)
    S = np.float32(0) if amax == 0 else F.mul(amax, F.INV127)
    inv = np.float32(0) if amax == 0 else F.mul(np.float32(127), F.recip(np.float32(amax)))
    q, ds = [], []
    for i in range(0, len(row), g):
        d = 0 if row_mode else _offset(_amax(row[i:i + g]), amax)
        ds.append(d)
        ig = F.mul(inv, np.float32(2.0 ** d))
        q += [int(F.q8(F.mul(np.float32(t), ig))) for t in row[i:i + g]]
    return q, np.float32(S), ds


def _run(cfg, dram, prog):
    return Machine(cfg, [prog + [I.halt()]], [dram.copy()]).run().slices[0]


@pytest.mark.parametrize("row_mode", [False, True])
def test_qact_row_scale_and_sub_block_offsets(row_mode):
    cfg = Config(S=1, D=32, MCOLS=4, ACT_BLOCKS=8)
    rng = np.random.default_rng(1)
    rows, KB = 3, 3
    x = (rng.standard_normal((rows, KB * cfg.D)) * np.exp(rng.uniform(-6, 6, (rows, KB * 4, 1)))
         .repeat(cfg.D // 4, axis=2).reshape(rows, -1)).astype(np.float32)
    x[1, 8:16] = 0                                        # a zero sub-block: d = 0
    dram = np.zeros(DRAM, np.uint8)
    dram[:x.nbytes] = x.view(np.uint8).reshape(-1)
    s = _run(cfg, dram, [I.ld(0, 0, x.size), I.qact(0, rows, 2, KB, KB * cfg.D, row=row_mode)])
    for r in range(rows):
        q, S, d = ref_quantize(x[r], cfg.D // 4, row_mode)
        assert s.act[r, 2 * cfg.D:(2 + KB) * cfg.D].tolist() == q
        assert np.array_equal(s.ascale[r, 2:2 + KB].view(np.uint32),
                              np.full(KB, S).view(np.uint32))
        assert s.adoff[r, 2:2 + KB].reshape(-1).tolist() == d


@pytest.mark.parametrize("row_mode", [False, True])
def test_qst_writes_the_int8_scale_rows(row_mode):
    cfg = Config(S=1, D=32, MCOLS=4, ACT_BLOCKS=8)
    rng = np.random.default_rng(2)
    rows, KB = 2, 5
    x = (rng.standard_normal((rows, KB * cfg.D)) * np.exp(rng.uniform(-5, 5, (rows, KB, 1)))
         .repeat(cfg.D, axis=2).reshape(rows, -1)).astype(np.float32)
    dram = np.zeros(DRAM, np.uint8)
    dram[:x.nbytes] = x.view(np.uint8).reshape(-1)
    dram[SA:SA + 64] = 0xAB                                 # the pad bytes are written 0
    s = _run(cfg, dram, [I.ld(0, 0, x.size),
                         I.qst(0, WA, SA, rows, KB, KB * cfg.D, KB * cfg.D, 1, row=row_mode)])
    sr = 4 if row_mode else 4 + 4 * -(-KB // 4)
    for r in range(rows):
        q, S, d = ref_quantize(x[r], cfg.D, row_mode)
        assert s.dram[WA + r * KB * cfg.D:][:KB * cfg.D].view(np.int8).tolist() == q
        row = s.dram[SA + r * sr:SA + (r + 1) * sr]
        assert row[:4].view(np.float32)[0].view(np.uint32) == np.float32(S).view(np.uint32)
        if not row_mode:
            assert row[4:4 + KB].tolist() == d and not row[4 + KB:].any()


W_OF = I.MX_W


def _scale_rows(rng, wf, N, KBr, rsc, big_offsets):
    """Random scale rows (bytes [N, srs]) and their (Sw [N], dw [N, KBr, G], m [N, KBr, G])."""
    G = 8 if wf == I.WNV else 4
    sw = (rng.standard_normal(N) * 0.01).astype(np.float32)
    hi = 20 if big_offsets else 4
    if rsc:
        rows = sw[:, None].view(np.uint8).copy()
        return rows, sw, np.zeros((N, KBr, G), int), np.ones((N, KBr, G), int)
    if wf == I.W8:
        dw = rng.integers(0, hi, (N, KBr))
        srs = 4 + 4 * -(-KBr // 4)
        rows = np.zeros((N, srs), np.uint8)
        rows[:, :4] = sw[:, None].view(np.uint8)
        rows[:, 4:4 + KBr] = dw
        return rows, sw, np.repeat(dw[:, :, None], G, 2), np.ones((N, KBr, G), int)
    if wf in (I.W4I, I.W4F):
        dw = rng.integers(0, hi, (N, KBr))
        m = rng.integers(0, 64, (N, KBr, G))
        word = (dw | sum(m[:, :, b] << (8 + 6 * b) for b in range(G))).astype(np.uint32)
        srs = 8 + 4 * KBr + (4 * KBr) % 8
        rows = np.zeros((N, srs), np.uint8)
        rows[:, :4] = sw[:, None].view(np.uint8)
        rows[:, 8:8 + 4 * KBr] = word.view(np.uint8).reshape(N, -1)
        return rows, sw, np.repeat(dw[:, :, None], G, 2), m
    e4 = rng.integers(0, 0x7F, (N, KBr, G))
    if big_offsets:
        e4[rng.random(e4.shape) < 0.3] &= 0x0F           # small exponents: groups drop
    srs = 16 + 8 * KBr + (8 * KBr) % 16
    rows = np.zeros((N, srs), np.uint8)
    rows[:, :4] = sw[:, None].view(np.uint8)
    rows[:, 16:16 + 8 * KBr] = e4.astype(np.uint8).reshape(N, -1)
    E, Mt = (e4 >> 3) & 15, e4 & 7
    return rows, sw, 15 - np.maximum(E, 1), np.where(E > 0, 8 + Mt, Mt)


def ref_mm(act, sa, da, wel, sw, dw, m, wf, pair, M):
    """MM's outputs from the definition, scalar Python integers: act [R, KB, D] int, sa [R]
    (each row's Sa), da [R, KB, 4], wel [N, KB, D] int, sw [N], dw / m [N, KB, G]."""
    R, KB, D = act.shape
    N, G, W = wel.shape[0], dw.shape[2], W_OF[wf]
    y = np.zeros((M, N), np.float32)
    for j in range(M):
        for n in range(N):
            acc = 0
            for k in range(KB):
                r = j + M if pair and k % 2 else j
                for g in range(G):
                    b = g * 4 // G
                    sl = slice(g * D // G, (g + 1) * D // G)
                    c = int(np.dot(act[r, k, sl].astype(np.int64), wel[n, k, sl].astype(np.int64)))
                    sh = W - int(dw[n, k, g]) - int(da[r, k, b])
                    acc += (int(m[n, k, g]) << sh) * c if sh >= 0 else 0
            assert abs(acc) < 2 ** 47
            y[j, n] = F.mul(F.mul(np.float32(acc), sw[n]),
                            F.mul(sa[j], np.float32(2.0 ** -W)))
    return y


CASES = [  # wf, pair, unit, rsc, k0, big offsets
    (I.W8, False, False, False, 0, False), (I.W8, False, False, False, 3, True),
    (I.W8, False, False, True, 0, False), (I.W8, False, True, False, 0, False),
    (I.W4F, False, False, False, 1, False), (I.W4F, True, False, False, 2, True),
    (I.W4I, True, False, False, 0, False), (I.W4F, True, True, False, 0, False),
    (I.W4F, False, False, True, 0, False), (I.WNV, False, False, False, 1, True),
    (I.WNV, True, False, False, 2, False), (I.WNV, True, True, False, 0, False),
]


@pytest.mark.parametrize("wf,pair,unit,rsc,k0,big", CASES)
@pytest.mark.parametrize("KB", [1, 3])
def test_mm_is_one_integer_per_output(wf, pair, unit, rsc, k0, big, KB):
    cfg = Config(S=1, D=32, MCOLS=4, ACT_BLOCKS=8, PAIR=True)
    D, M, N = cfg.D, 2 if pair else 3, 5
    rng = np.random.default_rng(1000 * wf + 10 * KB + k0 + 7 * pair + 3 * unit + rsc)
    R = 2 * M if pair else M
    x = (rng.standard_normal((M, KB * D)) * np.exp(rng.uniform(-8 if big else -2, 2,
                                                                (M, KB * 4, 1)))
         .repeat(D // 4, axis=2).reshape(M, -1)).astype(np.float32)
    KBr = k0 + KB                                          # the scale rows' blocks
    if wf == I.W8:
        wel = rng.integers(-127, 128, (N, KB, D))
        data = wel.astype(np.int8).view(np.uint8).reshape(N, KB * D)
    else:
        codes = rng.integers(0, 16, (N, KB * D))
        wel = Q.DEC["int4" if wf == I.W4I else "fp4"][codes].reshape(N, KB, D)
        data = Q.pack4(codes)
    rs = -(-data.shape[1] // D) * D
    rows, sw, dw, m = _scale_rows(rng, wf, N, KBr, rsc, big)
    dram = np.zeros(DRAM, np.uint8)
    dram[:x.nbytes] = x.view(np.uint8).reshape(-1)
    for n in range(N):
        dram[WA + n * rs:WA + n * rs + data.shape[1]] = data[n]
    srs = rows.shape[1]
    dram[SA:SA + rows.size] = rows.reshape(-1)
    s = _run(cfg, dram, [I.ld(0, 0, x.size), I.qact(0, M, 1, KB, KB * D, dup=pair),
                         I.mm(WA, SA, OUT, N, KB, rs, N, M, 1, srs, unit=unit, wf=wf, pair=pair,
                              k0=k0, rsc=rsc)])
    if unit:
        sw, dw, m = np.ones(N, np.float32), np.zeros_like(dw), np.ones_like(m)
    act = s.act[:R, D:(1 + KB) * D].reshape(R, KB, D)
    want = ref_mm(act, s.ascale[:R, 1], s.adoff[:R, 1:1 + KB], wel, sw,
                  dw[:, k0:k0 + KB], m[:, k0:k0 + KB], wf, pair, M)
    got = s.tget(OUT + np.arange(M * N)).reshape(M, N)
    assert np.array_equal(got.view(np.uint32), want.view(np.uint32))


def test_pair_changes_no_bit():
    """PAIR sums the same integer: a 4-bit MM gives the same bits with and without it."""
    cfg = Config(S=1, D=32, MCOLS=4, ACT_BLOCKS=8, PAIR=True)
    D, M, N, KB = cfg.D, 2, 4, 5
    rng = np.random.default_rng(5)
    x = rng.standard_normal((M, KB * D)).astype(np.float32)
    codes = rng.integers(0, 16, (N, KB * D))
    data = Q.pack4(codes)
    rs = -(-data.shape[1] // D) * D
    rows, *_ = _scale_rows(rng, I.W4F, N, KB, False, True)
    dram = np.zeros(DRAM, np.uint8)
    dram[:x.nbytes] = x.view(np.uint8).reshape(-1)
    for n in range(N):
        dram[WA + n * rs:WA + n * rs + data.shape[1]] = data[n]
    dram[SA:SA + rows.size] = rows.reshape(-1)
    outs = []
    for pair in (False, True):
        s = _run(cfg, dram, [I.ld(0, 0, x.size), I.qact(0, M, 0, KB, KB * D, dup=pair),
                             I.mm(WA, SA, OUT, N, KB, rs, N, M, 0, rows.shape[1], wf=I.W4F,
                                  pair=pair)])
        outs.append(s.tget(OUT + np.arange(M * N)).view(np.uint32).copy())
    assert np.array_equal(*outs)


def test_acc_and_ascale_use_the_mx_output():
    cfg = Config(S=1, D=32, MCOLS=4, ACT_BLOCKS=8)
    D, M, N, KB = cfg.D, 2, 3, 2
    rng = np.random.default_rng(6)
    x = rng.standard_normal((M, KB * D)).astype(np.float32)
    old = rng.standard_normal((M, N)).astype(np.float32)
    alpha = rng.standard_normal(M).astype(np.float32)
    wel = rng.integers(-127, 128, (N, KB * D)).astype(np.int8)
    dram = np.zeros(DRAM, np.uint8)
    dram[:x.nbytes] = x.view(np.uint8).reshape(-1)
    dram[WA:WA + wel.size] = wel.view(np.uint8).reshape(-1)
    pre = [I.ld(0, 0, x.size), I.qact(0, M, 0, KB, KB * D)]
    mm = dict(sa=WA, ssa=AS, out=OUT, n=N, kb=KB, rs=KB * D, ors=N, m=M, ab=0, srs=0, unit=True)
    base = _run(cfg, dram, pre + [I.mm(**mm)]).tget(OUT + np.arange(M * N)).reshape(M, N)
    s = Machine(cfg, [pre + [I.mm(**mm, acc=True)] + [I.halt()]], [dram.copy()])
    sl = s.slices[0]
    sl.tput(OUT + np.arange(M * N), old.reshape(-1))
    s.run()
    assert np.array_equal(sl.tget(OUT + np.arange(M * N)).reshape(M, N).view(np.uint32),
                          F.add(old, base).view(np.uint32))
    s = Machine(cfg, [pre + [I.mm(**{**mm, "ssa": 0}, acc=True, ascale=AS)] + [I.halt()]],
                [dram.copy()])
    sl = s.slices[0]
    sl.tput(OUT + np.arange(M * N), old.reshape(-1))
    sl.tput(AS + np.arange(M), alpha)
    s.run()
    want = F.add(F.mul(old, alpha[:, None]), base)
    assert np.array_equal(sl.tget(OUT + np.arange(M * N)).reshape(M, N).view(np.uint32),
                          want.view(np.uint32))


def test_an_mm_over_two_quantized_ranges_is_refused():
    """MM reads each row's Sa from its first block: blocks two QACTs wrote (two Sa) raise."""
    cfg = Config(S=1, D=32, MCOLS=4, ACT_BLOCKS=8)
    D = cfg.D
    x = np.arange(2 * D, dtype=np.float32)
    dram = np.zeros(DRAM, np.uint8)
    dram[:x.nbytes] = x.view(np.uint8)
    prog = [I.ld(0, 0, x.size), I.qact(0, 1, 0, 1, 2 * D), I.qact(D, 1, 1, 1, 2 * D),
            I.mm(WA, 0, OUT, 1, 2, 2 * D, 1, 1, 0, 0, unit=True)]
    with pytest.raises(SimError, match="Sa"):
        _run(cfg, dram, prog)


def test_mm_field_checks():
    with pytest.raises(ValueError):
        I.mm(0, 0, 0, 1, 256, 128, 1, 1, 0, 0)               # KB is 8 bits
    with pytest.raises(ValueError):
        I.mm(0, 0, 0, 1, 2, 128, 1, 1, 0, 0, wf=I.W4F, pair=True, k0=1)
    with pytest.raises(ValueError):
        I.mm(0, 0, 0, 1, 2, 128, 1, 1, 0, 0, wf=I.WNV, rsc=True)
    ins = I.mm(0, 0, 0, 7, 3, 128, 1, 1, 0, 0, k0=200)
    assert ins.w[3] == 7 | 3 << 16 | 200 << 24
