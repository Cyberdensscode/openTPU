"""Tiled prefill (docs/tiled_prefill.md): T prompt rows through every layer in one run, each
weight chunk streamed once for all T rows.

A prefill run of today's kernel (qwen3_rows) holds its R rows' activations in TMEM at once (about
7K words a row on Qwen3-0.6B, so R <= 8, R_max 3-4 on most layouts) and attends row by row: each
(row, KV head) pair streams its KV head's K and V^T on its own. This kernel takes T rows (T <=
ACT_ROWS: one stationary operand, every MM replays each streamed chunk for ceil(T / MCOLS) groups,
docs/isa.md MM) with what TMEM holds made independent of the projections' widths:

- the attention's projections run one KV head at a time (its K, V and query group: W_k, W_v
  and W_q split by rows; the MM results per element are the same, docs/isa.md), so TMEM holds
  one head's [T, (G + 2) d] instead of [T, (n_q + 2 n_kv) d];
- each KV head's queries attend in tiles of P = MCOLS / G rows (P G query rows, one MM), so K
  and V^T are streamed once per P rows, not once per row. The rows of a tile see different
  numbers of tokens: their last blocks take a per-row-group additive mask (RowMasks at
  compile-time positions, RowTiles at run-time ones: each row's own tile of the image's mask
  table), which gives bit for bit the online softmax of each row alone. A query group wider
  than the MXU (G > MCOLS) attends a row at a time in parts of MCOLS heads, as today's;
- the MLP keeps every chunk's activation quantized in ACT RAM (F / D blocks of T rows) and runs
  the down projection by parts of its output columns, each part summing the chunks in the same
  order (the same bits as swiglu_down's accumulation), so TMEM holds [T, H / parts], not [T, H];
- LFM2's convolution blocks run by parts of their channels (the convolution is per channel):
  in_proj's B, C and x of a part, its ring rows and taps, the gated product quantized into
  out_proj's operand; then out_proj by output parts, as W_o.

Each row's arithmetic is today's rows kernels' (qwen3_rows, qwen35_rows, lfm2_rows), so the
logits, the KV cache and the recurrent states are bit for bit those of today's prefill runs
over the same rows (tests/test_tiled.py): an MM's output is one exact integer (the MX MXU,
docs/isa.md MM: replayed or paired alike) and each operand is quantized over today's columns
(one QACT, a row's one Sa: the attention's outputs whole into W_o's, a down chunk's activation,
LFM2's gated product whole). Where the outputs whole do not fit TMEM at a layer run's rows the
range forms (ranges) quantize them a part at a time (a QACT range each, W_o an MM a range): not
today's bits, as close to the fp32 forward. Qwen3 and the Llama-likes on its code
(llama.py: Phi-4-mini, SmolLM3), Qwen3.5 (gated attention and the MLP tiled, its DeltaNet rows
today's: they already stream each weight once a run, pair of heads by pair) and LFM2 / LFM2.5.

Prompt runs (opentpu/llm/prefill.py, docs/prefill.md) take this kernel by default (not with the
engine's tile_runs False or OTPU_TILE=0: prefill.tile_flag): compile_tile_run's programs at a
run-time position of a bucket, their tokens from out[], as compile_prompt_run's. A MoE model's
expert-major layer runs (docs/offload.md 13.11) take it when asked for (tile_runs True or
OTPU_TILE=1: prefill.layer_tile_flag; Engine.layer_tile: gemma4_layer_tile, T rows of a
chunk through one layer from their records; compile_layer_tile), the mixer's and the dense
MLP's weights once for the T rows, the experts left to the layer's expert run as today.
"""
from __future__ import annotations

import math

from .. import isa as I
from .. import language as ol
from ..compiler import Affine, CompileError, Tensor, current, trace_fit
from ..kernels.attention import RowMasks, RowTiles, TileSeq, _attend_heads
from ..kernels.lib import sigmoid, silu
from ..kernels.mlp import _chunk
from .qwen3 import (ATTN_BLOCK, ATTN_DEPTH, RunRows, RunWords, _amask, _inputs_rows, _layers,
                    _lm_head_rows, _load_opt, _norm_heads, _padded, _rope_gate,
                    _rope_rows_padded)

QPART = 8       # a query part's [T, heads d] at most TMEM / QPART words (fewer heads, more rows)
SHALLOW = 2     # the attention pipeline's depth of a run over IMEM at ATTN_DEPTH


def packing(mcols: int, G: int) -> int:
    """Query rows of a KV head's group (G each) that one attention MM takes: MCOLS / G."""
    return max(1, mcols // G)


def tile_seq(p0: int, r0: int, n: int, block: int, G: int) -> TileSeq:
    """The blocks a packed query tile attends over: rows r0 .. r0 + n - 1 of a tile at p0
    (positions p0 + r, each counting tokens 0 .. p0 + r), G query rows each. Blocks every row
    sees whole are `full` (looped); the rest up to the last row's token are listed, each with a
    RowMasks of the tokens each row group counts (none: every row counts the whole block)."""
    first, last = p0 + r0, p0 + r0 + n - 1          # the tile's first and last rows' positions
    full = (first + 1) // block                     # blocks before the first row's own block
    items = []
    for t0 in range(full * block, last + 1, block):
        m = min(block, last + 1 - t0)               # the block's tokens up to the last row's
        keep = [(i * G, (i + 1) * G, max(0, min(m, p0 + r0 + i + 1 - t0))) for i in range(n)]
        items.append((t0, m, None if all(k == m for _, _, k in keep) else RowMasks(keep)))
    return TileSeq(full, items)


def run_seq(pos: RunRows, r0: int, n: int, G: int, cap: int) -> TileSeq:
    """tile_seq at a run-time position: rows r0 .. r0 + n - 1 of the run `pos` (a prompt run's
    RunRows, every row in its bucket), G query rows each. The bucket's whole blocks, then its
    last block (up to the cache's end), each row group with its own row's Additive tile
    (RunRows.bucket_row: the image's mask table at the row's position in the block)."""
    if pos.amask is None:
        raise CompileError("a tiled run at a run-time position needs the image's mask tiles "
                           "(RunRows amask)")
    t0 = pos.t0
    last = min(pos.block, cap - t0)
    tiles = [(i * G, (i + 1) * G, pos.bucket_row(r0 + i).add) for i in range(n)]
    return TileSeq(pos.blocks - 1, [(t0, last, RowTiles(tiles))])


def _norm_quantize(x, g, eps: float):
    """quantize(rmsnorm(x, g)): the QACT RSCALE / CSCALE that the peephole makes of it (the
    same bytes and scales), without tracing the [T, H] products (x * x, x * r, x * r * g)."""
    r = ol.rsqrt(ol.sumsq(x) * (1.0 / x.cols) + eps)
    return ol.quantize(x, free=True, rscale=r, cscale=ol.load(g))


class Res:
    """A tiled run's residual rows x [T, H]: in TMEM, or with a DRAM scratch (`spill`: the I/O
    area's input rows, which a prompt run does not read) stored there while a block's
    attention runs, the TMEM free for its queries and outputs, and loaded back for the
    residual add (fp32 both ways: the same bits) into its own words (x is carried across the
    layers' hardware loops: every iteration's instructions address it there)."""

    def __init__(self, x, spill=None):
        self.x, self.scratch = x, spill

    def spill(self) -> None:
        if self.scratch is not None:
            x = self.x
            self.at = (x.base, x.shape, x.rs)
            ol.store(self.scratch, x)
            self.x = None

    def back(self):
        if self.x is None:
            x = self.x = _alloc_at(*self.at)
            if x.contiguous:
                ol.load(self.scratch, out=x)
            else:                               # (rows padded: a load a row)
                for r in range(x.rows):
                    ol.load(self.scratch[r, :], out=x[r, :])
        return self.x


def _alloc_at(base: int, shape, rs: int):
    """A tile at TMEM word `base` (Res.back: a loop-carried tile back in its own words), which
    no live tile may hold."""
    import gc
    import weakref

    from ..compiler import Tile
    b = current()
    n = (shape[0] - 1) * rs + shape[-1]
    for attempt in range(2):
        b.tmem_regions = [r for r in b.tmem_regions if r[2]() is not None]
        if not any(lo < base + n and base < hi for lo, hi, _ in b.tmem_regions):
            break
        gc.collect()
    else:
        raise CompileError("TMEM: the residual rows' words are taken when they come back")
    t = Tile(b, base, shape, rs)
    t.spare = 0
    b.tmem_regions.append((base, base + n, weakref.ref(t.buf)))
    b.tmem_regions.sort(key=lambda r: r[0])
    return t


def _attention_tile(res: Res, lw, c, s_, pos, T: int, spec, block: int, parts: int,
                    gated: bool = False, depth: int = ATTN_DEPTH, whole: bool = True,
                    oscr=None):
    """x += W_o . attention(x) for T rows of sequence 0 at positions pos .. pos + T - 1 (pos an
    int, or a prompt run's RunRows: run-time positions), their K / V appended first, each row
    attending causally over the cache: _attention_rows' arithmetic, one KV head's projections
    at a time, its query group in parts of at most MCOLS heads (a part's [T, heads d] at most
    TMEM / QPART words: Qwen3.5-35B's G = 8) and each part's rows in packed tiles (MCOLS query
    rows). gated (Qwen3.5): each head's outputs times sigmoid(W_gate x) before W_o, as
    _attention_rows' (acc / l) * sigmoid(gate). x: res.x, spilled while the heads attend (Res).
    whole: the outputs [T, n_q d] quantized at once into W_o's operand, as _attention_rows'
    (one QACT: a row's one Sa, one MM a W_o part): each part's stored to `oscr` (the image's
    I/O scratch, _oscr) as it ends and loaded back whole once the heads are done, or without
    oscr gathered in TMEM; else each part's quantized as it ends (quantize_into: a QACT range
    of its own, an MM of W_o's a range, docs/isa.md MM; not the same bits)."""
    d, G, eps = spec.head_dim, spec.n_q // spec.n_kv, spec.eps
    D = ol.block_size()
    dk = -(-d // D) * D
    run = isinstance(pos, RunRows)
    kvc = lw.kvs[0]
    xs = _norm_quantize(res.x, lw.g_in, eps)   # T rows: one pass a weight
    qn, kn = _load_opt(lw.qn), _load_opt(lw.kn)
    c, s_ = _rope_gate(lw, c, s_)
    res.spill()                                # (after the tiles that outlive the heads)
    scale = ol.LOG2E / math.sqrt(d)
    heads = list(kvc.owned_heads(spec.n_kv))
    nq = len(heads) * G
    mc = ol.mxu_columns()
    gs = min(G, mc)                            # a part of a KV head's query group: its heads
    while gs % 2 == 0 and T * gs * dk > ol.tmem_words() // QPART:
        gs //= 2                               # (a part's [T, gs dk] queries in TMEM / QPART)
    P = packing(mc, gs)                        # rows of a tile (P gs <= MCOLS query rows)
    tiles = [(r0, min(P, T - r0)) for r0 in range(0, T, P)]
    gparts = [(g0, min(G, g0 + gs)) for g0 in range(0, G, gs)]
    if whole and oscr is None:                 # the outputs, quantized once (W_o's operand)
        o_all, ost = ol.empty([T, nq * d]), None
    else:                                      # W_o's operand, a part's columns at a time
        o_all, ost = None, None if whole else ol.stationary(T, nq * d)
    if oscr is not None:                       # (the scratch as [T, n_q d]: one load back)
        oscr = Tensor(oscr.base, (T, nq * d), (nq * d, 1))
    made, oh = {}, {}                          # (KV head, part) -> its queries, its outputs
    ent, qhs, kvh, seqs = [], [], [], []

    def part(j, g0, g1):
        """KV head j's projections and K / V appends (with its first part), then the queries
        of heads g0 .. g1 of its group [T (g1 - g0), dk], row (r, g) = r (g1 - g0) + g."""
        hh = heads[j]
        if g0 == 0:
            k = ol.dot(xs, lw.wk[j * d:(j + 1) * d, :])
            v = ol.dot(xs, lw.wv[j * d:(j + 1) * d, :])
            kj = _rope_rows_padded(_norm_heads(k, kn, eps), c, s_)
            vj = _padded(v)
            if run:                            # V^T at a run-time position: a row at a time
                for r in range(T):
                    ol.kv_append(kvc, hh, pos.pos + r, kj[r:r + 1, :], vj[r:r + 1, :])
            else:
                ol.kv_append(kvc, hh, pos, kj, vj)
            del k, v, kj, vj
        w = g1 - g0
        q = ol.dot(xs, lw.wq[(j * G + g0) * d:(j * G + g1) * d, :])    # [T, w d]
        Q = ol.empty([T * w, dk], dense=True)
        if dk > d:                                             # heads padded to whole blocks
            Q.set(0.0)
        Qv = Q.reshape(T, w * dk)
        for g in range(w):                                     # head g of every row at once
            _rope_rows_padded(_norm_heads(q[:, g * d:(g + 1) * d], qn, eps), c, s_,
                              out=Qv[:, g * dk:g * dk + d])
        return Q

    def query(j, g0, g1, r0, n):
        def emit():
            if (j, g0) not in made:
                made[(j, g0)] = part(j, g0, g1)
            Q, w = made[(j, g0)], g1 - g0
            if r0 + n == T:
                made.pop((j, g0))
            return Q[r0 * w:(r0 + n) * w, :]
        return emit

    for j in range(len(heads)):
        for g0, g1 in gparts:
            for r0, n in tiles:
                ent.append((j, g0, g1, r0, n))
                qhs.append(query(j, g0, g1, r0, n))
                kvh.append(heads[j])
                seqs.append(run_seq(pos, r0, n, g1 - g0, kvc.cap) if run else
                            tile_seq(pos, r0, n, block, g1 - g0))

    def emit(i, acc, l):
        j, g0, g1, r0, n = ent[i]
        w = g1 - g0
        if o_all is not None:                  # (the part's columns of the outputs)
            o, o0 = o_all, (j * G + g0) * d
        else:
            if (j, g0) not in oh:
                oh[(j, g0)] = ol.empty([T, w * d])
            o, o0 = oh[(j, g0)], 0
        for a in range(n):
            o[r0 + a, o0:o0 + w * d].reshape(w, d).set(
                acc[a * w:(a + 1) * w, :] / l[a * w:(a + 1) * w][:, None])
        if r0 + n == T:                        # the part's rows done
            if gated:                          # a head at a time (TMEM: [T, d] temporaries)
                for g in range(w):
                    h0 = (j * G + g0 + g) * d
                    og = o[:, o0 + g * d:o0 + (g + 1) * d]
                    og.set(og * sigmoid(ol.dot(xs, lw.wgate[h0:h0 + d, :])))
                    del og
            if o_all is None:                  # to its columns of the scratch, or quantized
                h0 = (j * G + g0) * d          # into them; freed
                if whole:
                    ol.store(oscr[0:T, h0:h0 + w * d], oh.pop((j, g0)))
                else:
                    ol.quantize_into(ost, oh.pop((j, g0)), h0)
        del o

    _attend_heads(qhs, kvc, kvh, seqs, block, scale, depth=depth, emit=emit)
    xs = qn = kn = None                        # (the norm gains too: placed after x's tiles,
    #                                            they split TMEM under the load back)
    if whole:                                  # (the scratch's rows back: the stores before)
        if o_all is None:
            o_all = ol.load(oscr[0:T, 0:nq * d])
        ost = ol.quantize(o_all, free=True)
        del o_all
    _add_parts(res.back(), [ost], [lw.wo], parts)              # x += W_o o, by output parts


def _add_parts(x, acts, ws, parts: int, out=None) -> None:
    """x += sum_i acts[i] . ws[i]^T, the sum in the order of the list (dot, then dot(acc=)),
    by `parts` parts of the output columns: TMEM holds [T, H / parts] of the sum at a time,
    each element summed as one [T, H] result would be. out: the sum into that [T, H] tile
    instead (x untouched: Gemma's post-norms need the whole row)."""
    H = x.cols
    hp = H // parts
    for p in range(parts):
        n0 = p * hp
        y = None
        for a, w in zip(acts, ws):
            if y is None:
                y = ol.dot(a, w[n0:n0 + hp, :])
            else:
                ol.dot(a, w[n0:n0 + hp, :], acc=y)
        if out is None:
            x[:, n0:n0 + hp].set(x[:, n0:n0 + hp] + y)
        else:
            out[:, n0:n0 + hp].set(y)
        del y


def down_chunk(wg, wd) -> int:
    """The W_down chunk swiglu_down sums by (its `chunk`, else _chunk's): W_down's stored
    column parts, else _chunk of the slice's F."""
    if wd.parts is not None:
        return wd.pw
    D = ol.block_size()
    return _chunk(wg.shape[0], D, D if wd.wf == 0 else 2 * D)


def _mlp_tile(x, lw, spec, parts: int, sub: int, act=silu, g=None, out=None, xs=None):
    """x + W_down(act(W_gate xn) * (W_up xn)) for T rows (xn: x normed by g, default
    lw.g_post): every chunk's activation quantized once and kept in ACT RAM (the C columns of
    a W_down part: one QACT, as swiglu_down's, of the chunk made `sub` columns at a time into
    a [T, C] tile when sub < C), then the down projection by `parts` parts of its output
    columns, each summing the chunks in swiglu_down's order (the same bits). out: the MLP's
    output into that [T, H] tile instead (_add_parts). xs: the operand, quantized already (x
    then may be None, with out)."""
    if xs is None:
        xs = _norm_quantize(x, lw.g_post if g is None else g, spec.eps)
    wg, wu, wd = lw.wg, lw.wu, lw.wd
    T = (out if x is None else x).rows
    C = down_chunk(wg, wd)
    starts = list(range(0, wg.shape[0], C))
    pieces = [(c0, s0) for c0 in starts for s0 in range(0, C, sub)]

    def gate_up(c0, s0):
        a = c0 + s0
        return ol.dot(xs, wg[a:a + sub, :]), ol.dot(xs, wu[a:a + sub, :])

    acts, a = {}, None
    nxt = gate_up(*pieces[0])
    for i, (c0, s0) in enumerate(pieces):
        g, u = nxt
        if i + 1 < len(pieces):
            nxt = gate_up(*pieces[i + 1])                      # streams while the VPU works
        if sub == C:                                           # the chunk's activation
            acts[c0] = ol.quantize(act(g) * u, free=True)
        else:
            if s0 == 0:
                a = ol.empty([T, C])
            a[:, s0:s0 + sub].set(act(g) * u)
            if s0 + sub == C:
                acts[c0] = ol.quantize(a, free=True)
                a = None
        del g, u
    xs = nxt = None                     # the operand's ACT blocks are free for the down MMs
    _add_parts(out if x is None else x, [acts[c0] for c0 in starts],
               [wd[:, c0:c0 + C] for c0 in starts], parts, out)
    return x


def mlp_parts(H: int, D: int, T: int, tmem: int) -> int:
    """The output projections' parts: the fewest (whole D blocks each, dividing H) whose [T, H /
    parts] sum takes at most an eighth of TMEM."""
    for parts in range(1, H // D + 1):
        if H % (parts * D) == 0 and T * (H // parts + 1) <= tmem // 8:
            return parts
    return H // D


def mlp_sub(C: int, D: int, T: int, tmem: int) -> int:
    """Gate / up columns an MM: the most (whole D blocks, dividing the chunk C) whose four
    [T, sub] results in flight (this piece's gate and up, the next one's), with the chunk's
    [T, C] activation when sub < C, take at most a quarter of TMEM."""
    for sub in range(C, D - 1, -D):
        if C % sub == 0 and 4 * T * (sub + 1) + (T * (C + 1) if sub < C else 0) <= tmem // 4:
            return sub
    return D


def conv_parts(n: int, K: int, D: int, T: int, tmem: int) -> int:
    """LFM2's convolution channel parts: the fewest (whole D blocks each, dividing n) whose
    rows (K - 1 + T of B * x, the part's taps, three [T, n / parts] products and the sum in
    flight) take at most a quarter of TMEM (beside the [T, n] gated product)."""
    for parts in range(1, n // D + 1):
        cp = n // parts
        if n % (parts * D) == 0 and (K - 1 + T + K + 4 * T) * (cp + 1) <= tmem // 4:
            return parts
    return n // D


def _ring_store_part(lw, last, ringo: int, K: int, n: int, c0: int) -> None:
    """lfm2._ring_store of the channels c0 .. c0 + last.cols of the state ring (`last` their
    B * x rows of the last K positions): the same rotation, each row's part to its columns."""
    b = current()
    cp = last.cols
    win = ol.empty([2 * K - 1, cp])
    win[0:K, :].set(last)
    win[K:2 * K - 1, :].set(last[0:K - 1, :])
    r = b.scratch()
    b.emit(I.rld(r, ringo, mul=win.rs, comment="the ring's rotation"))
    for y in range(1, 2 * K):
        ra, imm = b.addr(Affine.of(lw.state.base) + 4 * (n * y + c0))
        j = y - 1 if y <= K else y - K - 1
        b.emit(I.st(imm, win.base + j * win.rs, cp, ra=ra, rb=r, comment="ring row part"))
    b.unscratch(r)
    del win


def _conv_tile(x, lw, pos: RunRows, spec, parts: int, cparts: int):
    """x += out_proj(conv(B * x) * C) for T rows at a prompt run's run-time positions (LFM2's
    convolution block, lfm2._conv_rows): by `cparts` parts of the channels, each part's B, x
    and C streamed once for the T rows, its ring rows and taps loaded, the ring rewritten from
    its last K rows (the run's rotation), the gated product to its columns of [T, n]; then
    out_proj's operand quantized at once (one QACT, as _conv_rows') and out_proj by output
    parts. Per channel and row _conv_rows' products and sums in its order: the same bits."""
    from .lfm2 import _ring_rows
    K, T = spec.conv_k, x.rows
    n = lw.taps.shape[1]
    if not isinstance(pos, RunRows):
        raise CompileError("a tiled convolution block runs at a prompt run's position")
    xs = _norm_quantize(x, lw.g_in, spec.eps)
    y = ol.empty([T, n])                                       # the gated product
    cp = n // cparts
    rows = _ring_rows(K, pos)[0]
    for q in range(cparts):
        c0 = q * cp
        taps = ol.load(lw.taps[:, c0:c0 + cp])                 # [K, cp]
        E = ol.empty([K - 1 + T, cp])                          # B * x, positions p-K+1 ..
        for j, row in zip(range(1, K), rows):                  # the ring's rows before the run
            ol.load(lw.state[row, c0:c0 + cp], out=E[K - 1 - j, :])
        bx = E[K - 1:K - 1 + T, :]
        bx.set(ol.dot(xs, lw.win[c0:c0 + cp, :]) *
               ol.dot(xs, lw.win[2 * n + c0:2 * n + c0 + cp, :]))
        _ring_store_part(lw, E[T - 1:T - 1 + K, :], pos.words["ringo"], K, n, c0)
        acc = bx * taps[K - 1, :][None, :]
        for j in range(1, K):
            acc = acc + E[K - 1 - j:K - 1 + T - j, :] * taps[K - 1 - j, :][None, :]
        del E, bx, taps
        y[:, c0:c0 + cp].set(acc * ol.dot(xs, lw.win[n + c0:n + c0 + cp, :]))
        del acc
    xs = None
    yst = ol.quantize(y, free=True)
    del y
    _add_parts(x, [yst], [lw.wout], parts)
    return x


def _oscr(m, on: bool = True):
    """The image's I/O scratch for a tiled run's attention outputs ([ACT_ROWS, n_q d], the
    widest layer's: the image's io "o"), or None (`on` False, the form's oscr: the outputs
    gathered in TMEM)."""
    return getattr(m, "oscr", None) if on else None


def _widths(m, T: int):
    """(parts, the MLP's sub of a layer): the output projections' parts and the gate / up
    piece width (mlp_parts, mlp_sub) of this slice's hidden and F."""
    spec = m.spec
    D, W = ol.block_size(), ol.tmem_words()
    parts = mlp_parts(spec.hidden // ol.num_programs(), D, T, W)
    return parts, lambda lw: mlp_sub(down_chunk(lw.wg, lw.wd), D, T, W)


def _pieces(T: int, most: int) -> list:
    """T rows in the fewest runs of at most `most` rows, as even as they go ((r0, n), the
    longer first: 6 by 4 is 3 + 3, not 4 + 2, whose 2 rows a 4-bit MM would pair at MCOLS 4)."""
    k = -(-T // most)
    q, r = divmod(T, k)
    out, r0 = [], 0
    for i in range(k):
        n = q + (i < r)
        out.append((r0, n))
        r0 += n
    return out


def _rows_of(pos, T: int):
    return pos if isinstance(pos, RunRows) else [(0, pos + r) for r in range(T)]


@ol.jit
def qwen3_tile(m, p0, T: int, logit_rows, block: int = ATTN_BLOCK, spill: bool = False,
               depth: int = ATTN_DEPTH, whole: bool = True, oscr: bool = True):
    """T rows at positions p0 .. p0 + T - 1 of sequence 0 (p0 an int: their embedding and RoPE
    rows in the I/O area, as qwen3_rows'; a RunRows: a prompt run's tokens and positions), the
    logits of `logit_rows` (empty: none). spill: x in the I/O area's rows while each layer's
    heads attend (Res). depth: the attention's pipeline (_attend_heads: blocks in flight a
    head; fewer, a smaller program: compile_tile_run's forms). whole: W_o's operand quantized
    at once (_attention_tile's; else a QACT range a part)."""
    spec = m.spec
    x, c, s_ = _inputs_rows(m, _rows_of(p0, T))
    res = Res(x, m.xr[0:T, :] if spill else None)
    del x
    parts, sub = _widths(m, T)
    for lw in _layers(m):
        _attention_tile(res, lw, c, s_, p0, T, spec, block, parts, depth=depth, whole=whole,
                        oscr=_oscr(m, oscr))
        _mlp_tile(res.x, lw, spec, parts, sub(lw))
    _lm_head_rows(res.x, m, spec, logit_rows)


@ol.jit
def qwen35_tile(m, pos, T: int, logit_rows, block: int = ATTN_BLOCK, hidden: bool = False,
                spill: bool = False, drows: int = 0, depth: int = ATTN_DEPTH,
                whole: bool = True, oscr: bool = True):
    """qwen35_rows tiled: T rows of a prompt run (pos a RunRows), its attention layers and
    MLPs tiled, its DeltaNet layers today's rows (_deltanet_rows over the T rows, or in runs
    of `drows` rows one after another: their pair buffers grow with the rows; each weight
    streamed once a run, pair of heads by pair). hidden: MTP's rows (each row's final-normed
    hidden to m.hid, as qwen35_rows'). spill, depth, whole: as qwen3_tile's."""
    from ..kernels.lib import rmsnorm
    from .qwen35 import LIN, _deltanet_rows, run_layers
    spec = m.spec
    if not isinstance(pos, RunRows) or pos.lo < spec.conv_k - 1:
        raise CompileError("a tiled Qwen3.5 run is a prompt run from conv_k - 1")
    x, c, s_ = _inputs_rows(m, pos)
    res = Res(x, m.xr[0:T, :] if spill else None)
    del x
    parts, sub = _widths(m, T)
    dr = drows or T

    def layer(li, it):
        lw = m.layer(li, it)
        x = res.x
        if lw.kind == LIN:
            for r0, n in _pieces(T, dr):
                xr = x[r0:r0 + n, :]
                xr.set(_deltanet_rows(xr, lw, pos.lo, spec, m.gr[0:n, :], m.on[0:n, :],
                                      xs=_norm_quantize(xr, lw.g_in, spec.eps)))
                del xr
            del x
        else:
            del x
            _attention_tile(res, lw, c, s_, pos, T, spec, block, parts, gated=True,
                            depth=depth, whole=whole, oscr=_oscr(m, oscr))
        _mlp_tile(res.x, lw, spec, parts, sub(lw))

    run_layers(m.plan, layer)
    x = res.x
    if hidden:
        g = ol.load(m.g_final)
        for r in range(T):
            ol.store(m.hid[r:r + 1, :], rmsnorm(x[r:r + 1, :], g, spec.eps))
        del g
    _lm_head_rows(x, m, spec, logit_rows)


@ol.jit
def lfm2_tile(m, pos, T: int, logit_rows, block: int = ATTN_BLOCK, spill: bool = False,
              depth: int = ATTN_DEPTH, whole: bool = True, oscr: bool = True):
    """lfm2_rows tiled: T rows of a prompt run (pos a RunRows), its attention layers, its
    convolution blocks and its MLPs tiled. spill, depth, whole: as qwen3_tile's."""
    from .lfm2 import CONV, run_layers
    spec = m.spec
    if not isinstance(pos, RunRows):
        raise CompileError("a tiled LFM2 run is a prompt run")
    x, c, s_ = _inputs_rows(m, pos)
    res = Res(x, m.xr[0:T, :] if spill else None)
    del x
    parts, sub = _widths(m, T)
    D, W = ol.block_size(), ol.tmem_words()

    def layer(li, it):
        lw = m.layer(li, it)
        if lw.kind == CONV:
            n = lw.taps.shape[1]
            _conv_tile(res.x, lw, pos, spec, parts, conv_parts(n, spec.conv_k, D, T, W))
        else:
            _attention_tile(res, lw, c, s_, pos, T, spec, block, parts, depth=depth,
                            whole=whole, oscr=_oscr(m, oscr))
        _mlp_tile(res.x, lw, spec, parts, sub(lw))

    run_layers(m.plan, layer)
    _lm_head_rows(res.x, m, spec, logit_rows)


def pack_seq(seqs, G: int, block: int, cap: int):
    """One packed tile's attention span from its rows' own (at run-time positions: Bucket with
    an Additive tile, or Blocks whose masks are Additive tiles; the same blocks for every row
    of a run, the masks their own): every masked block's RowTiles, row group i (G query rows)
    taking row i's tile. One row: its own span."""
    from ..kernels.attention import Additive, Blocks, Bucket
    s0 = seqs[0]
    if len(seqs) == 1:
        return s0
    if isinstance(s0, Bucket):
        if any(s.add is None for s in seqs):
            raise CompileError("a packed tile's masked block needs the mask tiles (Additive)")
        t0 = (s0.blocks - 1) * block
        return TileSeq(s0.blocks - 1, [(t0, min(block, cap - t0), RowTiles(
            [(i * G, (i + 1) * G, s.add) for i, s in enumerate(seqs)]))])
    if isinstance(s0, Blocks):
        items = []
        for k, (t0, n, m0) in enumerate(s0.items):
            ms = [s.items[k][2] for s in seqs]
            if m0 is None:
                items.append((t0, n, None))
                continue
            if not all(isinstance(mk, Additive) for mk in ms):
                raise CompileError("a packed tile's masked blocks need the mask tiles")
            items.append((t0, n, RowTiles([(i * G, (i + 1) * G, mk)
                                           for i, mk in enumerate(ms)])))
        return Blocks(items)
    raise CompileError(f"no packed span of {type(s0).__name__}")


def _rope_rows_g(x, c, s, out):
    """Gemma 4's _rope with one position per row (c, s: [T, nr]): the pairs (i, i + d / 2), i
    < nr, rotated, the others passing through; into `out`. The same products as _rope's."""
    from ..kernels.lib import rope_rows
    d, nr = x.cols, c.cols
    h = d // 2
    if nr == h:
        return rope_rows(x, c, s, out)
    a, b = x[:, 0:nr], x[:, h:h + nr]
    out[:, 0:nr].set(a * c - b * s)
    out[:, h:h + nr].set(b * c + a * s)
    out[:, nr:h].set(x[:, nr:h])
    out[:, h + nr:d].set(x[:, h + nr:d])
    return out




def _gemma_attention_tile(res: Res, lw, m, pos: RunRows, ropes, T: int, block: int,
                          parts: int, depth: int = ATTN_DEPTH, whole: bool = True,
                          oscr=None):
    """gemma4._attention tiled for T rows of a prompt run: x += norm(W_o attention(norm(x))).
    Per KV head its K / V (own layers: projected for the T rows, normed, K RoPE'd, appended a
    row at a time to its slot: the sliding ring's or the cache's), then its query group in
    parts of MCOLS heads, each part's [T, heads d] projected, normed and RoPE'd in place and
    its rows attending in packed tiles (P = MCOLS / G rows; a row's span is its own: the
    sliding window's Blocks or the global Bucket, their masks the mask tiles), the outputs
    written over the part's queries, then (whole: _attention's one QACT) stored to `oscr` as
    the part ends and loaded back whole into W_o's operand (without oscr gathered in a [T, n_q
    d] tile), or quantized as the part ends (a QACT range each, as _attention_tile's); then W_o
    by output parts into [T, H] and the post-norm. Each row's arithmetic is _attention's. x:
    res.x, spilled while the heads attend (Res)."""
    from ..kernels.lib import rmsnorm
    from .gemma4 import SLIDE, _RowPos, _add_norm, _full_seq, _slide_seq, _slot
    spec = m.spec
    d, nkv, eps = lw.hd, lw.nkv, spec.eps
    G = spec.n_q // nkv
    c0, nr = (0, spec.head_dim // 2) if lw.kind == SLIDE else (spec.head_dim, spec.global_rot)
    c, s_ = ropes[:, c0:c0 + nr], ropes[:, c0 + nr:c0 + 2 * nr]
    kv = lw.kv
    rows = [_RowPos(pos, r) for r in range(T)]
    span = _slide_seq if lw.kind == SLIDE else _full_seq
    rseq = [span(m, p, block) for p in rows]
    xs = _norm_quantize(res.x, lw.g_in, eps)
    qn = ol.load(lw.qn)
    kn = ol.load(lw.kn) if lw.own else None
    res.spill()                                # (after the tiles that outlive the heads)
    mc = ol.mxu_columns()
    gs = min(G, mc)
    while gs % 2 == 0 and T * gs * d > ol.tmem_words() // QPART:
        gs //= 2                               # a part's [T, gs d] queries in an eighth of TMEM
    P = packing(mc, gs)                        # rows of gs heads a tile: K / V streamed G / mc
    #                                            times a row whatever gs (P gs = MCOLS)
    tiles = [(r0, min(P, T - r0)) for r0 in range(0, T, P)]
    gparts = [(g0, min(G, g0 + gs)) for g0 in range(0, G, gs)]
    if whole and oscr is None:                 # the outputs, quantized once (W_o's operand)
        o_all, ost = ol.empty([T, spec.n_q * d]), None
    else:                                      # W_o's operand, a part's columns at a time
        o_all, ost = None, None if whole else ol.stationary(T, spec.n_q * d)
    if oscr is not None:                       # (the scratch as [T, n_q d]: one load back)
        oscr = Tensor(oscr.base, (T, spec.n_q * d), (spec.n_q * d, 1))
    made, ent, qhs, kvh, seqs = {}, [], [], [], []

    def part(j, g0, g1):
        """KV head j's K / V appends (with its first part), the queries of heads g0 .. g1 of
        its group: [T, (g1 - g0) d], read as [T (g1 - g0), d] (row (r, g))."""
        if g0 == 0 and lw.own:
            k = ol.dot(xs, lw.wk[j * d:(j + 1) * d, :])
            v = k if lw.same else ol.dot(xs, lw.wv[j * d:(j + 1) * d, :])
            kr = _rope_rows_g(rmsnorm(k, kn, eps), c, s_, ol.empty([T, d]))
            vr = rmsnorm(v, None, eps)
            for r in range(T):
                ol.kv_append(kv, j, _slot(m, lw.kind, rows[r]), kr[r:r + 1, :], vr[r:r + 1, :])
            del k, v, kr, vr
        h0, h1 = j * G + g0, j * G + g1
        q = ol.empty([T, (g1 - g0) * d], dense=True)          # (rows of heads: reshaped)
        ol.dot(xs, lw.wq[h0 * d:h1 * d, :], out=q)
        for g in range(g1 - g0):
            qg = q[:, g * d:(g + 1) * d]
            _rope_rows_g(rmsnorm(qg, qn, eps), c, s_, qg)
        return q

    def query(j, g0, g1, r0, n):
        def emit():
            if (j, g0) not in made:
                made[(j, g0)] = part(j, g0, g1)
            w = g1 - g0
            return made[(j, g0)].reshape(T * w, d)[r0 * w:(r0 + n) * w, :]
        return emit

    for j in range(nkv):
        for g0, g1 in gparts:
            for r0, n in tiles:
                ent.append((j, g0, g1, r0, n))
                qhs.append(query(j, g0, g1, r0, n))
                kvh.append(j)
                seqs.append(pack_seq(rseq[r0:r0 + n], g1 - g0, block, kv.cap))

    def emit(i, acc, l):
        j, g0, g1, r0, n = ent[i]
        w = g1 - g0
        if o_all is not None:                  # the part's columns of the outputs
            h0 = (j * G + g0) * d
            for a in range(n):
                o_all[r0 + a, h0:h0 + w * d].reshape(w, d).set(
                    acc[a * w:(a + 1) * w, :] / l[a * w:(a + 1) * w][:, None])
            if r0 + n == T:
                del made[(j, g0)]              # (its queries)
            return
        o = made[(j, g0)]                      # the outputs over the part's queries
        o.reshape(T * w, d)[r0 * w:(r0 + n) * w, :].set(acc / l[:, None])
        if r0 + n == T:                        # to the scratch, or quantized into W_o's
            h0 = (j * G + g0) * d              # operand
            if whole:
                ol.store(oscr[0:T, h0:h0 + w * d], made.pop((j, g0)))
            else:
                ol.quantize_into(ost, made.pop((j, g0)), h0)
        del o

    _attend_heads(qhs, kv, kvh, seqs, block, ol.LOG2E, depth=depth, emit=emit)
    xs = qn = kn = None                        # (the norm gains: _attention_tile's)
    if whole:                                  # (the scratch's rows back: the stores before)
        if o_all is None:
            o_all = ol.load(oscr[0:T, 0:spec.n_q * d])
        ost = ol.quantize(o_all, free=True)
        del o_all
    x = res.back()
    y = ol.empty([T, spec.hidden])
    _add_parts(y, [ost], [lw.wo], parts, y)
    ost = None
    _add_norm(x, y, ol.load(lw.g_attn), eps)


@ol.jit
def gemma4_tile(m, pos, T: int, logit_rows, block: int = ATTN_BLOCK, spill: bool = False,
                depth: int = ATTN_DEPTH, whole: bool = True, oscr: bool = True):
    """gemma4_step's rows tiled: T rows of a prompt run (pos a RunRows with the image's mask
    tiles), their tokens gathered from out[] and per-layer inputs as gemma4_step's; per layer
    the tiled attention, the dense MLP (GeGLU) tiled into [T, H] for its post-norm, the
    per-layer input block as gemma4._ple. spill: x in the I/O area's input rows while each
    layer's heads attend and W_o runs (Res). depth, whole: as qwen3_tile's."""
    from ..kernels.lib import gelu_tanh
    from .gemma4 import _add_norm, _gathered_out, _ple, _ple_inputs, _pos_rows
    from .qwen3 import fill_logits
    fill_logits(m)
    spec = m.spec
    if not isinstance(pos, RunRows) or pos.amask is None or spec.experts:
        raise CompileError("a tiled Gemma 4 run is a dense model's prompt run with mask tiles")
    ropes = _pos_rows(m, pos, T, masks=False)   # (first: x's gather leaves the next-fit
    x = _gathered_out(m, pos)                   # cursor past its temporaries, TMEM split)
    x.set(x * math.sqrt(spec.hidden))           # (in place: one [T, H] tile)
    if spec.ple_dim:
        _ple_inputs(m, x, None)
    res = Res(x, m.x[0:T, :] if spill else None)
    del x
    parts, sub = _widths(m, T)

    def layer(li, it=None, jt=None):
        lw = m.layer(li, it, jt)
        _gemma_attention_tile(res, lw, m, pos, ropes, T, block, parts, depth, whole,
                              _oscr(m, oscr))
        x = res.x
        y = ol.empty([T, spec.hidden])
        _mlp_tile(x, lw, spec, parts, sub(lw), act=gelu_tanh, g=lw.g_pre, out=y)
        _add_norm(x, y, ol.load(lw.g_ffn), spec.eps, None if spec.ple_dim else ol.load(lw.ls))
        del y
        if spec.ple_dim:
            _ple(x, lw, spec)

    def unit(first, it=None):
        for e0, su, r, nb in m.subs[first]:
            if r == 1:
                for e_ in range(len(su)):
                    layer(first + e0 + e_, it)
                continue
            for jt in ol.range(r):
                for e_ in range(len(su)):
                    layer(first + e0 + e_, it, (jt, len(su), nb))

    for first, _, reps in m.runs:
        if reps == 1:
            unit(first)
            continue
        for it in ol.range(reps):
            unit(first, it)
    if logit_rows:
        _lm_head_rows(res.x, m, spec, list(logit_rows))


EM_GROUP = 2    # rows a group of an expert-major tiled run's per-row record work ([R, H] tiles)


@ol.jit
def gemma4_layer_tile(m, li: int, pos, row, T: int, block: int = ATTN_BLOCK,
                      spill: bool = False, depth: int = ATTN_DEPTH, whole: bool = True,
                      oscr: bool = True):
    """gemma4_layer_run's expert-major run tiled (a MoE model's layer-major prefill,
    docs/offload.md 13.11): the T rows of the chunk from `row` (a run argument; their records'
    X, moe.em_x) at the run-time positions pos .. pos + T - 1 (a RunRows with the image's mask
    tiles, in one attention block) through layer li. The layer before ends first
    (gemma4._em_end, EM_GROUP rows at a time); the attention tiled (_gemma_attention_tile;
    spill: x in the records' X while the heads attend); x to X; the route and the need lines
    (moe.moe_prologue_rows over groups of rows, LINE / 4 ids a line); then the dense MLP tiled
    over the T rows (its weights streamed once), normed (norm_1), to the records' SH. Each
    row's arithmetic is gemma4_layer_run's, so the chunk's records, KV cache and expert runs
    are those of its runs of fewer rows, bit for bit where their 4-bit MMs pair alike."""
    from ..kernels.lib import gelu_tanh, rmsnorm
    from .gemma4 import _em_end, _pos_rows
    from .moe import LINE, em_row, em_x, moe_prologue_rows
    spec, dev = m.spec, m.moe_dev
    if not isinstance(pos, RunRows) or pos.amask is None or not spec.experts or spec.ple_dim:
        raise CompileError("a tiled Gemma 4 layer run is a MoE model's, at a run-time position "
                           "with the image's mask tiles")
    H, k = spec.hidden, spec.top_k
    X = em_x(dev, row, T, H)
    x = ol.load(X)
    ropes = _pos_rows(m, pos, T, masks=False)
    groups = [(r0, min(EM_GROUP, T - r0)) for r0 in range(0, T, EM_GROUP)]
    if li > 0:                                  # (the layer before: its experts' outputs)
        for r0, n in groups:
            _em_end(x[r0:r0 + n, :], m.layer(li - 1), spec, m, row + r0)
    lw = m.layer(li)
    res = Res(x, X if spill else None)
    del x
    parts, sub = _widths(m, T)
    _gemma_attention_tile(res, lw, m, pos, ropes, T, block, parts, depth, whole, _oscr(m, oscr))
    x = res.x
    ol.store(X, x)
    g = max(1, (LINE // 4) // k)                # rows a need line
    for r0 in range(0, T, g):
        moe_prologue_rows(x[r0:r0 + min(g, T - r0), :], lw, spec.moe, dev, spec.eps, row + r0)
    xs = _norm_quantize(x, lw.g_pre, spec.eps)  # the dense MLP: the rows normed (g_pre)
    res.x = x = None
    y = ol.empty([T, H])
    _mlp_tile(None, lw, spec, parts, sub(lw), act=gelu_tanh, out=y, xs=xs)
    xs = None
    gf = ol.load(lw.g_f1)
    for r0, n in groups:                        # SH: norm_1 of the dense MLP's output
        ol.store(Tensor(em_row(dev, row, r0) + 4 * H * (k + 1), (n, H), (dev.em_rec // 4, 1)),
                 rmsnorm(y[r0:r0 + n, :], gf, spec.eps))


@ol.jit
def qwen35_layer_tile(m, li: int, pos, row, T: int, block: int = ATTN_BLOCK,
                      spill: bool = False, drows: int = 0, depth: int = ATTN_DEPTH,
                      whole: bool = True, oscr: bool = True):
    """qwen35_layer_run's expert-major run tiled (Qwen3.5-MoE's layer-major prefill,
    docs/offload.md 13.11): the T rows of the chunk from `row` (their records' X) at the
    run-time positions pos .. pos + T - 1 (a RunRows with the image's mask tiles) through layer
    li. The MoE layer before ends first (moe.moe_combine_rows, EM_GROUP rows at a time); a
    DeltaNet layer's rows in runs of `drows` (default T; at most the image's gr / on rows),
    an attention layer tiled (_attention_tile, gated; spill: x in the records' X while the
    heads attend); x to X; a MoE layer's route, need lines and shared expert over groups of
    rows (moe.moe_prologue_rows, LINE / 4 ids a line), a dense layer's MLP tiled. Each row's
    arithmetic is qwen35_layer_run's."""
    from .moe import LINE, em_x, moe_combine_rows, moe_prologue_rows
    from .qwen35 import LIN, _deltanet_rows
    spec, dev = m.spec, m.moe_dev
    if not isinstance(pos, RunRows) or pos.amask is None or spec.moe is None:
        raise CompileError("a tiled Qwen3.5 layer run is a MoE model's, at a run-time "
                           "position with the image's mask tiles")
    H, mo, K = spec.hidden, spec.moe, spec.conv_k
    X = em_x(dev, row, T, H)
    x = ol.load(X)
    if li > 0 and m.layer(li - 1).moe:          # (the layer before: its experts' outputs)
        for r0 in range(0, T, EM_GROUP):
            xg = x[r0:min(T, r0 + EM_GROUP), :]
            xg.set(moe_combine_rows(xg, mo, dev, row + r0))
            del xg
    lw = m.layer(li)
    parts, sub = _widths(m, T)
    if lw.kind == LIN:
        for r0, n in _pieces(T, drows or T):
            xr = x[r0:r0 + n, :]
            xr.set(_deltanet_rows(xr, lw, K - 1, spec, m.gr[0:n, :], m.on[0:n, :],
                                  xs=_norm_quantize(xr, lw.g_in, spec.eps)))
            del xr
    else:
        c = ol.load(m.cos_t[pos.pos:pos.pos + T, :])
        s_ = ol.load(m.sin_t[pos.pos:pos.pos + T, :])
        res = Res(x, X if spill else None)
        del x
        _attention_tile(res, lw, c, s_, pos, T, spec, block, parts, gated=True, depth=depth,
                        whole=whole, oscr=_oscr(m, oscr))
        x = res.x
        del c, s_
    if not lw.moe:
        _mlp_tile(x, lw, spec, parts, sub(lw))
        ol.store(X, x)
        return
    ol.store(X, x)
    g = max(1, (LINE // 4) // mo.k)             # rows a need line
    for r0 in range(0, T, g):
        moe_prologue_rows(x[r0:r0 + min(g, T - r0), :], lw, mo, dev, spec.eps, row + r0)


@ol.jit
def tile_run(m, kernel, pos, T: int, kind: str, block: int = ATTN_BLOCK, kw=None):
    """A tiled prompt run (docs/prefill.md): `kernel`'s T rows at the run-time position in the
    generate state's tpos word (pos: a RunRows, toks_at 0), their tokens from out[]; kind "P",
    or "L" (the prompt's last run: its last row's logits). kw: the kernel's own (hidden, spill,
    drows)."""
    with RunWords(m, pos):
        kernel.fn(m, pos, T, [T - 1] if kind == "L" else [], block, **(kw or {}))


KERNELS = {"qwen3": qwen3_tile, "qwen35": qwen35_tile, "lfm2": lfm2_tile,
           "gemma4": gemma4_tile}


def family(img) -> str | None:
    """The image's model family (its module's name) when a tile kernel serves it, else None."""
    name = type(img).__module__.rsplit(".", 1)[-1]
    return name if name in KERNELS else None


def check(img, T: int) -> None:
    """T rows fit the image and the configuration: the I/O area's rows (the logits, Qwen3.5's
    gr / on rows), ACT_ROWS (one stationary of T rows), one slice."""
    cfg = img.cfg
    if T > img.rows:
        raise ValueError(f"{T} rows, the image's I/O area holds {img.rows}")
    if T > cfg.act_rows:
        raise ValueError(f"{T} rows: one MM over them needs ACT_ROWS >= {T} ({cfg.act_rows})")
    if cfg.S != 1:
        raise ValueError("tiled prefill: one slice")


def compile_tile(img, p0: int, T: int, block: int = ATTN_BLOCK, logit_rows=()) -> list:
    """One program per slice: qwen3_tile over the image's I/O rows (img.rows >= T), every MM
    over T rows (the configuration's ACT_ROWS >= T)."""
    check(img, T)
    return [trace_fit(qwen3_tile, img.cfg, s, lambda s=s: {
        "m": img.descriptors(s), "p0": p0, "T": T, "logit_rows": list(logit_rows),
        "block": block}).finish() for s in range(img.cfg.S)]


def compile_tile_run(img, blocks: int, T: int, kind: str, block: int = ATTN_BLOCK,
                     hidden: bool = False, slot: int = 0):
    """tile_run's (programs, run_args): T rows of a prompt at a run-time position of bucket
    `blocks` (from conv_k - 1), as the image's compile_prompt_run's P and L runs. The first
    of the run's forms (variants) that fits TMEM and ACT RAM, its tiles placed next-fit, then
    best-fit (_first_fit), at the attention's pipeline depth (ATTN_DEPTH), or where that is
    over IMEM at SHALLOW (its program grows with the blocks a head's entries leave unrolled,
    up to ~3 x depth a packed tile: past bucket ~3 the bigger tiles); over IMEM at both, the
    first (the caller's IMEM check)."""
    fam = family(img)
    if fam is None:
        raise ValueError(f"no tiled prompt run for {type(img).__module__}")
    if not img.lookup:
        raise ValueError("a prompt run needs lookup tables (lookup=True)")
    if kind not in ("P", "L"):
        raise ValueError(f"tiled prompt run kind {kind!r}")
    spec = img.spec
    if getattr(spec, "moe", None) is not None or getattr(spec, "experts", 0):
        raise ValueError("a MoE model's prompt runs layer by layer")
    if hidden and fam != "qwen35":
        raise ValueError("hidden rows: Qwen3.5's MTP")
    check(img, T)
    lo = max((blocks - 1) * block, getattr(spec, "conv_k", 1) - 1)
    if fam == "gemma4":             # (Gemma 4: its own mask tiles' address and step)
        lk = img.lookup
        if "amask" not in lk:
            raise ValueError("a tiled Gemma 4 run needs the image's mask tiles")
        pos = RunRows(blocks, block, lo, 0, img.cap, T, 0,
                      (lk["amask"], 4 * lk["amask_M"] * (block + 1)))
    else:
        pos = RunRows(blocks, block, lo, img.lookup["zmask"], img.cap, T, 0, _amask(img.lookup))
    desc = (lambda s: img.descriptors(s, slot)) if fam == "qwen35" else img.descriptors

    def trace(kw, fit):
        kw = dict(kw, hidden=True) if hidden else kw
        return [tile_run.trace(img.cfg, s, {
            "m": desc(s), "kernel": KERNELS[fam], "pos": pos, "T": T, "kind": kind,
            "block": block, "kw": kw}, fit=fit) for s in range(img.cfg.S)]

    return _first_fit(trace, variants(fam, T, img.cfg.MCOLS), img.cfg)


def _first_fit(trace, forms, cfg):
    """(programs, run_args) of trace(form, fit) (its builders, one a slice) for the first of
    the forms that fits TMEM, ACT RAM and IMEM: each form at the attention's pipeline depth,
    then over IMEM at SHALLOW, then the next form, its tiles placed next-fit; then the same with
    best-fit placement (slower: best-fit reuses the words just freed, which work still running
    reads; Qwen3-0.6B's 16 rows in co-simulation, MX, the card's configuration: 94.9K cycles
    a row placed best-fit, 87.4K next-fit); over IMEM at every one, the first (the caller's
    IMEM check: prefill.programs); none fitting, the last TMEM / ACT RAM error (or, where no
    form applied, the range forms' QACT range error)."""
    err = over = skip = None
    imem = cfg.IMEM_WORDS // 8
    for fit, v in [(fit, v) for fit in ("next", "best") for v in forms]:
        for f in [v] + ([dict(v, depth=SHALLOW)] if "depth" not in v else []):
            try:
                bs = trace(f, fit)
            except CompileError as e:
                if "QACT range" in str(e):     # (a range form's part at an odd block of
                    skip = e                   # 4-bit weights: the form does not apply)
                    break
                if not any(w in str(e) for w in ("TMEM", "ACT RAM")):
                    raise
                err = e
                break
            progs = [b.finish() for b in bs]
            if max(map(len, progs)) <= imem:
                return progs, list(bs[0].run_args)
            if over is None:
                over = progs, list(bs[0].run_args)
    if over is not None:
        return over
    raise err or skip


def variants(fam: str, T: int, mcols: int) -> list:
    """A tiled run's forms, tried in order until one fits TMEM and ACT RAM: as it is; x spilled
    while the heads attend (Res); Qwen3.5's DeltaNet layers in runs of fewer rows (whole passes
    of MCOLS, then fewer: their pair buffers grow with the rows). The attention's outputs
    whole (W_o's operand one QACT: today's bits), through the image's I/O scratch, then
    gathered in TMEM (wholes); a prompt run whose outputs do not fit takes fewer rows
    (prefill._probe_tile)."""
    out = [{}, {"spill": True}]
    if fam == "qwen35":
        d = T
        while d > 1:
            d = d - mcols if d > mcols else d - 1
            out.append({"spill": True, "drows": d})
    return wholes(out)


def wholes(forms: list) -> list:
    """The forms (the attention's outputs whole through the image's I/O scratch: a part
    stored as it ends, loaded back whole), then each with them gathered in TMEM (oscr False:
    no stores, the [T, n_q d] tile beside the heads' work)."""
    return forms + [dict(f, oscr=False) for f in forms]


def ranges(forms: list) -> list:
    """The forms, then those through the scratch with the outputs quantized a part at a time
    instead (whole False: a QACT range a part, W_o an MM a range; not today's bits)."""
    return forms + [dict(f, whole=False) for f in forms if f.get("oscr", True)]


LAYER_KERNELS = {"gemma4": gemma4_layer_tile, "qwen35": qwen35_layer_tile}


def layer_family(img) -> str | None:
    """The image's family when a tiled expert-major layer run serves it (LAYER_KERNELS: a MoE
    image with prefill rows, the expert server's words and the mask tiles, which a MoE model's
    image holds when OTPU_TILE=1 as it is built: prefill.layer_tile_flag), else None."""
    name = type(img).__module__.rsplit(".", 1)[-1]
    ok = name in LAYER_KERNELS and getattr(img, "prefill_rows", 0) and \
        getattr(img, "offload", None) is not None and "amask" in (img.lookup or {})
    return name if ok else None


def compile_layer_tile(img, li: int, blocks: int, T: int, block: int = ATTN_BLOCK):
    """(programs, run_args): layer li's expert-major run tiled (LAYER_KERNELS), T rows at a
    run-time position of bucket `blocks` (from conv_k - 1) and a run-time chunk row ("row"),
    in place of the image's compile_layer_run(li, blocks, R=T, em=True) (Engine.prefill_layers'
    runs of more than one row with tiling on: Engine.layer_tile); the first of its forms that
    fits TMEM, ACT RAM and IMEM (_first_fit)."""
    from ..compiler import RunVar
    fam = layer_family(img)
    if fam is None:
        raise ValueError(f"no tiled layer run for this image ({type(img).__module__}: an "
                         "expert-major MoE image with the mask tiles, OTPU_TILE=1)")
    cfg, lk = img.cfg, img.lookup
    if not 0 <= li < img.spec.layers or not 1 <= T <= cfg.act_rows or cfg.S != 1:
        raise ValueError(f"layer {li}, {T} rows: a layer's run of at most ACT_ROWS "
                         f"({cfg.act_rows}) rows, one slice")
    lo = max((blocks - 1) * block, getattr(img.spec, "conv_k", 1) - 1)
    amask = (lk["amask"], 4 * lk["amask_M"] * (block + 1))
    pos = RunRows(blocks, block, lo, lk.get("zmask", 0), img.cap, T, 0, amask)
    row = RunVar("row", img.prefill_rows)

    def trace(kw, fit):
        return [LAYER_KERNELS[fam].trace(cfg, 0, {
            "m": img.descriptors(0), "li": li, "pos": pos, "row": row, "T": T, "block": block,
            **kw}, fit=fit)]

    return _first_fit(trace, layer_variants(img, fam, T), cfg)


def untiled_rows(img) -> int:
    """The most rows of an untiled layer run (the image's compile_layer_run: its family's
    RUN_ROWS, and R k <= LINE / 4 ids in its request): layer 0's runs that gather their rows
    from the host's slot (embed_host) stay untiled."""
    import sys
    from .moe import LINE
    spec = img.spec
    k = spec.moe.k if getattr(spec, "moe", None) is not None else getattr(spec, "top_k", 1)
    return max(1, min(sys.modules[type(img).__module__].RUN_ROWS, (LINE // 4) // max(1, k)))


def layer_variants(img, fam: str, T: int) -> list:
    """A tiled layer run's forms (variants'): Qwen3.5's DeltaNet rows in runs of at most the
    image's gr / on rows (grows), then fewer; the run's rows are the engine's (layer_major),
    so then the attention's outputs a part at a time (ranges: the 26B's global layers'
    [8, 16 x 512] outputs alone are TMEM's 64K words)."""
    if fam != "qwen35":
        return ranges(wholes([{}, {"spill": True}]))
    d = min(T, img.grows)
    out = [{"drows": d}, {"spill": True, "drows": d}]
    while d > 1:
        d = d - img.cfg.MCOLS if d > img.cfg.MCOLS else d - 1
        out.append({"spill": True, "drows": d})
    return ranges(wholes(out))
