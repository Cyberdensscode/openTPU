"""Rows of a quantized table (int8 or 4-bit, an MXU format: opentpu/quant.py) gathered and
dequantized on the device, at a row index known only at run time (the token id): an embedding
straight from a tied LM head, or Gemma 4's per-layer embeddings, with no fp32 copy of the table
and no host work per token.

The MXU does it. A table row is streamed as a matrix of its D-blocks (one block, or for 4-bit
one D-byte chunk of two blocks, per streamed row), against a stationary operand of one-hot
rows: row j of one-hot block k is 1 at element M k + j (M = MCOLS), so one MM gives element
M k + j of every block, dequantized by the MXU's own arithmetic (the one-hot quantizes to 127
with Sa = f32(1/127), every offset 0); D / M MMs give every element (for 4-bit, twice as many,
one for each block of a chunk: the one-hot blocks are interleaved with zero blocks, [Z, O_0, Z,
O_1, ..., O_{D/M-1}, Z], and [O_k, Z] reads a chunk's first block, [Z, O_k] its second). The
operand is the constant `onehot(D, M, fmt)` from DRAM, quantized once (operand; where it does
not fit ACT RAM or TMEM whole, each MM's blocks on their own).

A streamed row's scales are its own, so a table keeps them per streamed row, its gather scales
(quant.gather_scales, next to the scale rows an MM over the table reads, or alone): int8 an
fp32 per block, Sw 2^-dw (MM RSC); 4-bit a scale row per chunk (Sw and the chunk's two blocks'
offsets and multipliers). An element is then rne(rne(i2f(127 w m 2^sh) Sw) (f32(1/127) 2^-W)),
sh = W - dw (MM's window; a block with sh < 0 gives 0).

  gather_row     a row of a row-major table (the LM head): the MMs write the elements
                 transposed, [D, blocks], and one VOP per block copies a column to the row
  gather_record  a row of a table stored for gathering (`pack_records`): the elements are
                 placed in the blocks so that the MMs write the row in order, no copies
                 (Gemma 4's PLE table: 70 blocks, 32 MMs)

dequant_row / dequant_records give the same values on the host, bit for bit (the host-written
inputs of chunked prefill equal the device's gathers).
"""
from __future__ import annotations

import numpy as np

from .. import fp32 as F
from .. import isa as I
from .. import language as ol
from .. import qcache as QC
from .. import quant as Q
from ..compiler import Affine, QTensor


# =============================================================================== the operand
def onehot(D: int, M: int, fmt: str) -> np.ndarray:
    """The one-hot stationary operand [M, nb * D] fp32: int8, D / M blocks, row j of block k 1
    at element M k + j; 4-bit, the same blocks interleaved with zero blocks, Z O_0 Z O_1 ...
    O_{D/M-1} Z (O_k is block 2k + 1)."""
    n = D // M
    four = fmt != "int8"
    nb = 2 * n + 1 if four else n
    a = np.zeros((M, nb * D), np.float32)
    for k in range(n):
        b = 2 * k + 1 if four else k
        for j in range(M):
            a[j, b * D + M * k + j] = 1.0
    return a


def onehot_blocks(D: int, M: int, fmt: str) -> int:
    return onehot(D, M, fmt).shape[1] // D


class _Parts:
    """The one-hot operand where it does not fit whole (D / M blocks, 2 D / M + 1 for 4-bit,
    against ACT_BLOCKS; or its tile, M x the blocks, against TMEM: a small configuration):
    each MM's blocks loaded and quantized on their own (blocks(k, kb): M rows of kb blocks
    from DRAM, a QACT, the operand of one MM)."""

    def __init__(self, desc):
        self.desc, self.rows = desc, desc.shape[0]

    @property
    def src(self):                  # (oh.src.rows: M, as a quantized operand's)
        return self

    def blocks(self, k: int, kb: int = 1):
        D = ol.block_size()
        return ol.quantize(ol.load(self.desc[:, k * D:(k + kb) * D]))


def operand(desc):
    """The gathers' one-hot operand (oh) from its DRAM descriptor (the image's onehot(D, M,
    fmt)): loaded and quantized into ACT RAM once where the configuration holds it whole (its
    blocks in ACT RAM, its tile in at most 3/4 of TMEM, leaving the gather's own tiles room:
    the card's largest, 4-bit at D = 128 and 4 columns, is 51%), else _Parts. The choice is
    the configuration's, not the trace's: a whole operand that finds TMEM full fails the
    compile as it always did (the generate loop's split form, a prefill run's fewer rows)."""
    M, cols = desc.shape
    if cols // ol.block_size() <= ol.act_blocks() and 4 * M * cols <= 3 * ol.tmem_words():
        return ol.quantize(ol.load(desc))
    return _Parts(desc)


def table(data, gscale, n: int, k: int, fmt: str, D: int) -> QTensor:
    """The descriptor of a row-major table [n, k] in MXU format fmt to gather from: its element
    rows (quant.quantize_mxu's) at `data` and their gather scales (quant.gather_scales) at
    `gscale`."""
    return QTensor(Affine.of(data), Affine.of(gscale), (n, k), Q.row_bytes(k, fmt, D),
                   Q.gather_scale_bytes(k, fmt, D), D, wf=Q.mxu_wf(fmt))


def _mms(oh, qv: QTensor, out, fmt: str, half: int = 0):
    """The one-hot MMs of a row viewed as blocks (int8: qv [nb, D]; 4-bit: chunks, qv [nc,
    2D]): element M k + j of streamed row n into out(k)[j, n] (4-bit: the chunks' first blocks
    into out(k, 0), their second into out(k, 1))."""
    D, M = qv.D, oh.src.rows
    for k in range(D // M):
        if fmt == "int8":
            ol.dot(oh.blocks(k), qv, out=out(k, 0))
        else:
            ol.dot(oh.blocks(2 * k + 1, 2), qv, out=out(k, 0))
            ol.dot(oh.blocks(2 * k, 2), qv, out=out(k, 1))


def _chunk_srs(fmt: str, D: int) -> int:
    """A 4-bit chunk's gather scale row: its bytes and its stride."""
    return Q.gather_scale_bytes(2 * D, fmt, D)


# =============================================================================== kernels
def gather_row(oh, tab: QTensor, row, fmt: str):
    """Row `row` (an int or a run-time value) of a row-major table [N, K] (int8 or 4-bit, its
    gather scales: `table`), dequantized: an fp32 [K] tile. oh: quantize(onehot(D, M, fmt)) (a
    live stationary operand)."""
    D, K = tab.D, tab.shape[1]
    nb = K // D
    M = oh.src.rows
    r = tab[row:row + 1, :] if tab.shape[0] > 1 else tab
    if fmt == "int8":
        T = ol.empty([D, nb])                       # T[p, b]: element p of block b
        qv = QTensor(r.data, r.scale, (nb, D), D, 4, D, wf=I.W8, rsc=True)
        _mms(oh, qv, lambda k, h: T[M * k:M * k + M, :], fmt)
        Ts = [(T, b) for b in range(nb)]
    else:
        nc = -(-nb // 2)
        T0, T1 = ol.empty([D, nc]), ol.empty([D, nc])
        qv = QTensor(r.data, r.scale, (nc, 2 * D), D, _chunk_srs(fmt, D), D, wf=tab.wf)
        _mms(oh, qv, lambda k, h: (T0, T1)[h][M * k:M * k + M, :], fmt)
        Ts = [((T0, T1)[b % 2], b // 2) for b in range(nb)]
    out = ol.empty([K])
    for b, (T, c) in enumerate(Ts):                 # the transpose, a column per VOP
        out[b * D:(b + 1) * D].column().set(T[:, c:c + 1])
    return out


def gather_record(oh, tab: QTensor, row, fmt: str, S: int):
    """Row `row` of a table packed by pack_records (S blocks per record; tab.shape[1] = S * D
    elements, tab.scale the records' gather scales), dequantized in order: an fp32 [S * D]
    tile (the first n are the row)."""
    D, M = tab.D, oh.src.rows
    out = ol.empty([S * D])
    v = out.reshape(D, S)                           # v[p, c] = element p S + c
    r = tab[row:row + 1, :] if tab.shape[0] > 1 else tab
    if fmt == "int8":
        qv = QTensor(r.data, r.scale, (S, D), D, 4, D, wf=I.W8, rsc=True)
        _mms(oh, qv, lambda k, h: v[M * k:M * k + M, :], fmt)
    else:
        nc = S // 2
        qv = QTensor(r.data, r.scale, (nc, 2 * D), D, _chunk_srs(fmt, D), D, wf=tab.wf)
        _mms(oh, qv, lambda k, h: v[M * k:M * k + M, h * nc:(h + 1) * nc], fmt)
    return out


# =============================================================================== host side
def record_blocks(n: int, fmt: str) -> int:
    """S, the blocks of a record of n elements: at least n / D (the caller rounds n up to
    blocks), with the MM's output row stride S safe for the TMEM banks: M consecutive rows S
    apart fall in distinct banks when S is odd or S = 2 mod 4 and M <= LANES / 2 (every
    configuration: LANES >= 8, M <= 4 on the board, 8 in the design point). 4-bit records hold
    S / 2 chunks: S = 2 mod 4."""
    s = n
    if fmt == "int8":
        while not (s % 2 == 1 or s % 4 == 2):
            s += 1
    else:
        s += s % 2
        if s % 4 == 0:
            s += 2
    return s


def record_data(S: int, fmt: str, D: int) -> int:
    """Bytes of a record's elements (its gather scales follow them)."""
    return S * D if fmt == "int8" else S * D // 2


def record_bytes(S: int, fmt: str, D: int) -> int:
    """Bytes of one record: the S blocks' data, then their gather scales (int8 an fp32 per
    block, 4-bit a scale row per chunk), padded to D bytes."""
    sc = 4 * S if fmt == "int8" else S // 2 * _chunk_srs(fmt, D)
    return record_data(S, fmt, D) + -(-sc // D) * D


def record_table(base, n: int, S: int, fmt: str, D: int) -> QTensor:
    """The descriptor of n records of S blocks at `base` (pack_records' layout) to gather from."""
    rec = record_bytes(S, fmt, D)
    return QTensor(Affine.of(base), Affine.of(base) + record_data(S, fmt, D), (n, S * D), rec,
                   rec, D, wf=Q.mxu_wf(fmt))


def pack_records(rows: np.ndarray, fmt: str, D: int, S: int) -> np.ndarray:
    """Records [T, record_bytes] (uint8) of fp32 rows [T, n], n <= S D: element i of a row is
    element i // S of block i % S (int8: block c is the c-th D bytes; 4-bit: blocks c and S/2
    + c are the two halves of chunk c), zeros past n; the gather scales follow the data. Each
    block is quantized on its own (int8: a row of one block, its scale Sw), a 4-bit chunk's two
    blocks as one row (one Sw, their offsets and multipliers: the chunk's scale row)."""
    T, n = rows.shape
    x = np.zeros((T, S * D), np.float32)
    x[:, :n] = rows
    blk = x.reshape(T, D, S).transpose(0, 2, 1).reshape(T * S, D)     # [T*S, D]: block c
    rec = np.zeros((T, record_bytes(S, fmt, D)), np.uint8)
    nd = record_data(S, fmt, D)
    if fmt == "int8":
        q, s = QC.quantize_rows(blk, fmt, D)                          # [T*S, D], [T*S, 8]
        rec[:, :nd] = q.reshape(T, nd)
        rec[:, nd:nd + 4 * S] = Q.gather_scales(s, fmt, D, D).view(np.uint8).reshape(T, 4 * S)
        return rec
    nc = S // 2
    pair = blk.reshape(T, 2, nc, D).transpose(0, 2, 1, 3).reshape(T * nc, 2 * D)  # chunk rows
    b, s = QC.quantize_rows(pair, fmt, D)                             # [T*nc, D], [T*nc, srs]
    rec[:, :nd] = b.reshape(T, nd)
    cs = _chunk_srs(fmt, D)
    rec[:, nd:nd + nc * cs] = Q.gather_scales(s, fmt, 2 * D, D).reshape(T, nc * cs)
    return rec


def _onehot_q(D: int):
    """The quantized one-hot: (q, Sa) as QACT makes them (every offset 0)."""
    x = np.zeros((1, D), np.float32)
    x[0, 0] = 1.0
    q, s, _ = F.quantize_mx(x, D // Q.NSUB)
    return int(q[0, 0]), np.float32(s[0])


def dequant_blocks(w, Sw, dw, m, fmt: str) -> np.ndarray:
    """The one-hot MMs' values of blocks [N, D]: their integer elements w [N, D] (int8, or the
    4-bit codes decoded) and per block its Sw [N] and per group (G of them) dw and m [N, G].
    fp32 [N, D], as the MXU computes: rne(rne(i2f(127 w (m << sh)) Sw) (Sa 2^-W)), sh = W - dw
    (sh < 0: 0)."""
    q1, s1 = _onehot_q(np.asarray(w).shape[1])
    W = I.MX_W[Q.mxu_wf(fmt)]
    w = np.asarray(w, np.int64)
    N, D = w.shape
    G = np.asarray(dw).shape[1]
    sh = W - np.asarray(dw, np.int64)
    B = np.where(sh >= 0, np.asarray(m, np.int64) << np.maximum(sh, 0), 0)
    acc = q1 * w * np.repeat(B, D // G, axis=1)
    return F.mul(F.mul(F.i2f(acc), np.asarray(Sw, np.float32)[:, None]),
                 F.mul(s1, np.float32(2.0 ** -W)))


def dequant_gathered(rows: np.ndarray, gsc: np.ndarray, fmt: str, K: int,
                     D: int = 128) -> np.ndarray:
    """The gathered values of table rows [N, K] (fp32): their element rows (row_bytes) and
    gather scales (quant.gather_scales: int8 fp32 [N, KB], 4-bit uint8 [N, chunks, srs])."""
    rows = np.asarray(rows, np.uint8)
    N, KB = rows.shape[0], K // D
    w = Q.elements(rows, fmt, K).reshape(N * KB, D)
    if fmt == "int8":
        Sw = np.asarray(gsc).reshape(N, -1)[:, :KB].view(np.float32) \
            if np.asarray(gsc).dtype == np.uint8 else np.asarray(gsc, np.float32)
        Sw = Sw.reshape(N * KB)
        zero = np.zeros((N * KB, Q.NSUB), np.int64)
        return dequant_blocks(w, Sw, zero, zero + 1, fmt).reshape(N, K)
    g = np.asarray(gsc, np.uint8).reshape(N, -1, _chunk_srs(fmt, D))   # [N, nc, srs]
    nc = g.shape[1]
    Sw, dw, m = Q.unpack_scales(g.reshape(N * nc, -1), fmt, 2)        # [N*nc], [N*nc, 2, G]
    Sw = np.repeat(Sw, 2)[:N * nc * 2].reshape(N, 2 * nc)[:, :KB].reshape(-1)
    dw = dw.reshape(N, 2 * nc, -1)[:, :KB].reshape(N * KB, -1)
    m = m.reshape(N, 2 * nc, -1)[:, :KB].reshape(N * KB, -1)
    return dequant_blocks(w, Sw, dw, m, fmt).reshape(N, K)


def dequant_row(qrow: np.ndarray, gsrow: np.ndarray, fmt: str, D: int, K: int | None = None
                ) -> np.ndarray:
    """gather_row's values of one table row: its MXU bytes (quant.quantize_mxu's row) and its
    gather scales (quant.gather_scales' row)."""
    qrow = np.asarray(qrow, np.uint8)
    if K is None:
        K = len(qrow) if fmt == "int8" else 2 * len(qrow)
    return dequant_gathered(qrow[None], np.asarray(gsrow)[None], fmt, K, D)[0]


def gathered(a: np.ndarray, fmt: str, D: int = 128) -> np.ndarray:
    """The values gather_row gives for fp32 rows [n, K] of a table in MXU format fmt
    (quantize_mxu, gather_scales): [n, K] fp32, bit for bit."""
    a = np.atleast_2d(np.asarray(a, np.float32))
    q, s = Q.quantize_mxu(a, fmt, D)
    return dequant_gathered(q, Q.gather_scales(s, fmt, a.shape[1], D), fmt, a.shape[1], D)


def dequant_records(rec: np.ndarray, fmt: str, D: int, S: int) -> np.ndarray:
    """gather_record's values of records [T, record_bytes]: fp32 [T, S * D]."""
    T = rec.shape[0]
    nd = record_data(S, fmt, D)
    if fmt == "int8":
        q = rec[:, :nd].reshape(T * S, D)
        s = rec[:, nd:nd + 4 * S].copy().view(np.float32).reshape(T * S, 1)
        v = dequant_gathered(q, s, fmt, D, D).reshape(T, S, D)
    else:
        nc = S // 2
        cs = _chunk_srs(fmt, D)
        b = rec[:, :nd].reshape(T * nc, D)                               # a chunk row each
        g = rec[:, nd:nd + nc * cs].reshape(T * nc, 1, cs)
        v = dequant_gathered(b, g, fmt, 2 * D, D).reshape(T, nc, 2, D)
        v = v.transpose(0, 2, 1, 3).reshape(T, S, D)                    # block c, S/2 + c
    return np.ascontiguousarray(v.transpose(0, 2, 1)).reshape(T, S * D)
