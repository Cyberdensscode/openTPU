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
over the same rows where their MMs pair alike (PAIR: a 4-bit MM of a stationary of at most
MCOLS / 2 rows sums each even K block into the odd one first; the tile pairs when T <= MCOLS /
2, today's run when R <= MCOLS / 2; tests/test_tiled.py). Qwen3 and the Llama-likes on its code
(llama.py: Phi-4-mini, SmolLM3), Qwen3.5 (gated attention and the MLP tiled, its DeltaNet rows
today's: they already stream each weight once a run, pair of heads by pair) and LFM2 / LFM2.5.

Prompt runs (opentpu/llm/prefill.py, docs/prefill.md) take this kernel when the engine's
tile_runs (or OTPU_TILE=1) is on: compile_tile_run's programs at a run-time position of a
bucket, their tokens from out[], as compile_prompt_run's.
"""
from __future__ import annotations

import math

from .. import isa as I
from .. import language as ol
from ..compiler import Affine, CompileError, current, trace_fit
from ..kernels.attention import RowMasks, RowTiles, TileSeq, _attend_heads
from ..kernels.lib import sigmoid, silu
from ..kernels.mlp import _chunk
from .qwen3 import (ATTN_BLOCK, ATTN_DEPTH, RunRows, RunWords, _amask, _inputs_rows, _layers,
                    _lm_head_rows, _load_opt, _norm_heads, _padded, _rope_gate,
                    _rope_rows_padded)


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
                    gated: bool = False, depth: int = ATTN_DEPTH):
    """x += W_o . attention(x) for T rows of sequence 0 at positions pos .. pos + T - 1 (pos an
    int, or a prompt run's RunRows: run-time positions), their K / V appended first, each row
    attending causally over the cache: _attention_rows' arithmetic, one KV head's projections
    at a time and its queries in packed tiles. gated (Qwen3.5): each head's outputs times
    sigmoid(W_gate x) before W_o, as _attention_rows' (acc / l) * sigmoid(gate). x: res.x,
    spilled while the heads attend (Res)."""
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
    gs = min(G, mc)                            # a row's query heads in one entry
    P = packing(mc, G) if G <= mc else 1       # rows of a tile
    tiles = [(r0, min(P, T - r0)) for r0 in range(0, T, P)]
    gparts = [(g0, min(G, g0 + gs)) for g0 in range(0, G, gs)]
    ost = ol.stationary(T, nq * d)             # W_o's operand, a KV head's columns at a time
    oh = {}                                    # KV head j -> its outputs [T, G d] in progress
    ent, qhs, kvh, seqs = [], [], [], []

    def head(j):
        """KV head j's projections, K / V appends and queries [T G, dk], row (r, g) = r G + g."""
        hh = heads[j]
        k = ol.dot(xs, lw.wk[j * d:(j + 1) * d, :])
        v = ol.dot(xs, lw.wv[j * d:(j + 1) * d, :])
        kj = _rope_rows_padded(_norm_heads(k, kn, eps), c, s_)
        vj = _padded(v)
        if run:                                # V^T at a run-time position: a row at a time
            for r in range(T):
                ol.kv_append(kvc, hh, pos.pos + r, kj[r:r + 1, :], vj[r:r + 1, :])
        else:
            ol.kv_append(kvc, hh, pos, kj, vj)
        del k, v, kj, vj
        q = ol.dot(xs, lw.wq[j * G * d:(j + 1) * G * d, :])    # [T, G d]
        Q = ol.empty([T * G, dk], dense=True)
        if dk > d:                                             # heads padded to whole blocks
            Q.set(0.0)
        Qv = Q.reshape(T, G * dk)
        for g in range(G):                                     # head g of every row at once
            _rope_rows_padded(_norm_heads(q[:, g * d:(g + 1) * d], qn, eps), c, s_,
                              out=Qv[:, g * dk:g * dk + d])
        return Q

    made = {}

    def query(j, r0, n, g0, g1):
        def emit():
            if j not in made:
                made[j] = head(j)
            Q = made[j]
            if r0 + n == T and g1 == G:
                made.pop(j)
            return Q[r0 * G + g0:(r0 + n - 1) * G + g1, :]
        return emit

    for j in range(len(heads)):
        for r0, n in tiles:
            for g0, g1 in gparts:
                ent.append((j, r0, n, g0, g1))
                qhs.append(query(j, r0, n, g0, g1))
                kvh.append(heads[j])
                seqs.append(run_seq(pos, r0, n, g1 - g0, kvc.cap) if run else
                            tile_seq(pos, r0, n, block, g1 - g0))

    def emit(i, acc, l):
        j, r0, n, g0, g1 = ent[i]
        if j not in oh:
            oh[j] = ol.empty([T, G * d])
        w = g1 - g0
        for a in range(n):
            oh[j][r0 + a, g0 * d:g1 * d].reshape(w, d).set(
                acc[a * w:(a + 1) * w, :] / l[a * w:(a + 1) * w][:, None])
        if r0 + n == T and g1 == G:            # the head's rows done: quantized, freed
            o = oh.pop(j)
            if gated:                          # a head at a time (TMEM: [T, d] temporaries)
                for g in range(G):
                    h0 = (j * G + g) * d
                    og = o[:, g * d:(g + 1) * d]
                    og.set(og * sigmoid(ol.dot(xs, lw.wgate[h0:h0 + d, :])))
                    del og
            ol.quantize_into(ost, o, j * G * d)
            del o

    _attend_heads(qhs, kvc, kvh, seqs, block, scale, depth=depth, emit=emit)
    xs = None
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


def _mlp_tile(x, lw, spec, parts: int, sub: int, act=silu, g=None, out=None):
    """x + W_down(act(W_gate xn) * (W_up xn)) for T rows (xn: x normed by g, default
    lw.g_post): every chunk's activation quantized once and kept in ACT RAM (the C columns of
    a W_down part, made `sub` columns at a time: each piece's QACT gives its blocks what one
    of the whole chunk would), then the down projection by `parts` parts of its output
    columns, each summing the chunks in swiglu_down's order (the same bits). out: the MLP's
    output into that [T, H] tile instead (_add_parts)."""
    xs = _norm_quantize(x, lw.g_post if g is None else g, spec.eps)
    wg, wu, wd = lw.wg, lw.wu, lw.wd
    T = x.rows
    C = down_chunk(wg, wd)
    starts = list(range(0, wg.shape[0], C))
    pieces = [(c0, s0) for c0 in starts for s0 in range(0, C, sub)]

    def gate_up(c0, s0):
        a = c0 + s0
        return ol.dot(xs, wg[a:a + sub, :]), ol.dot(xs, wu[a:a + sub, :])

    acts = {}
    nxt = gate_up(*pieces[0])
    for i, (c0, s0) in enumerate(pieces):
        g, u = nxt
        if i + 1 < len(pieces):
            nxt = gate_up(*pieces[i + 1])                      # streams while the VPU works
        if s0 == 0:
            acts[c0] = ol.stationary(T, C)
        ol.quantize_into(acts[c0], act(g) * u, s0)
        del g, u
    xs = nxt = None                     # the operand's ACT blocks are free for the down MMs
    _add_parts(x, [acts[c0] for c0 in starts], [wd[:, c0:c0 + C] for c0 in starts], parts, out)
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
    [T, sub] results in flight (this piece's gate and up, the next one's) take at most a
    quarter of TMEM."""
    for sub in range(C, D - 1, -D):
        if C % sub == 0 and 4 * T * (sub + 1) <= tmem // 4:
            return sub
    return D


def conv_parts(n: int, K: int, D: int, T: int, tmem: int) -> int:
    """LFM2's convolution channel parts: the fewest (whole D blocks each, dividing n) whose
    rows (K - 1 + T of B * x, the part's taps, three [T, n / parts] products and the sum in
    flight) take at most a quarter of TMEM."""
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
    its last K rows (the run's rotation), the gated product quantized into out_proj's operand;
    then out_proj by output parts. Per channel and row _conv_rows' products and sums in its
    order: the same bits."""
    from .lfm2 import _ring_rows
    K, T = spec.conv_k, x.rows
    n = lw.taps.shape[1]
    if not isinstance(pos, RunRows):
        raise CompileError("a tiled convolution block runs at a prompt run's position")
    xs = _norm_quantize(x, lw.g_in, spec.eps)
    yst = ol.stationary(T, n)
    cp = n // cparts
    rows = _ring_rows(K, pos)[0]
    for q in range(cparts):
        c0 = q * cp
        taps = ol.load(lw.taps[:, c0:c0 + cp])                 # [K, cp]
        E = ol.empty([K - 1 + T, cp])                          # B * x, positions p-K+1 ..
        for j, row in zip(range(1, K), rows):                  # the ring's rows before the run
            ol.load(lw.state[row, c0:c0 + cp], out=E[K - 1 - j, :])
        bx = E[K - 1:K - 1 + T, :]
        bx.set(ol.dot(xs, lw.win[c0:c0 + cp, :]) * ol.dot(xs, lw.win[2 * n + c0:2 * n + c0 + cp, :]))
        _ring_store_part(lw, E[T - 1:T - 1 + K, :], pos.words["ringo"], K, n, c0)
        acc = bx * taps[K - 1, :][None, :]
        for j in range(1, K):
            acc = acc + E[K - 1 - j:K - 1 + T - j, :] * taps[K - 1 - j, :][None, :]
        del E, bx, taps
        ol.quantize_into(yst, acc * ol.dot(xs, lw.win[n + c0:n + c0 + cp, :]), c0)
        del acc
    xs = None
    _add_parts(x, [yst], [lw.wout], parts)
    return x


def _widths(m, T: int):
    """(parts, the MLP's sub of a layer): the output projections' parts and the gate / up
    piece width (mlp_parts, mlp_sub) of this slice's hidden and F."""
    spec = m.spec
    D, W = ol.block_size(), ol.tmem_words()
    parts = mlp_parts(spec.hidden // ol.num_programs(), D, T, W)
    return parts, lambda lw: mlp_sub(down_chunk(lw.wg, lw.wd), D, T, W)


def _rows_of(pos, T: int):
    return pos if isinstance(pos, RunRows) else [(0, pos + r) for r in range(T)]


@ol.jit
def qwen3_tile(m, p0, T: int, logit_rows, block: int = ATTN_BLOCK, spill: bool = False,
               depth: int = ATTN_DEPTH):
    """T rows at positions p0 .. p0 + T - 1 of sequence 0 (p0 an int: their embedding and RoPE
    rows in the I/O area, as qwen3_rows'; a RunRows: a prompt run's tokens and positions), the
    logits of `logit_rows` (empty: none). spill: x in the I/O area's rows while each layer's
    heads attend (Res). depth: the attention's pipeline (_attend_heads: blocks in flight a
    head; fewer, a smaller program: compile_tile_run's forms)."""
    spec = m.spec
    x, c, s_ = _inputs_rows(m, _rows_of(p0, T))
    res = Res(x, m.xr[0:T, :] if spill else None)
    del x
    parts, sub = _widths(m, T)
    for lw in _layers(m):
        _attention_tile(res, lw, c, s_, p0, T, spec, block, parts, depth=depth)
        _mlp_tile(res.x, lw, spec, parts, sub(lw))
    _lm_head_rows(res.x, m, spec, logit_rows)


@ol.jit
def qwen35_tile(m, pos, T: int, logit_rows, block: int = ATTN_BLOCK, hidden: bool = False,
                spill: bool = False, drows: int = 0, depth: int = ATTN_DEPTH):
    """qwen35_rows tiled: T rows of a prompt run (pos a RunRows), its attention layers and
    MLPs tiled, its DeltaNet layers today's rows (_deltanet_rows over the T rows, or in runs
    of `drows` rows one after another: their pair buffers grow with the rows; each weight
    streamed once a run, pair of heads by pair). hidden: MTP's rows (each row's final-normed
    hidden to m.hid, as qwen35_rows'). spill, depth: as qwen3_tile's."""
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
            for r0 in range(0, T, dr):
                xr = x[r0:min(T, r0 + dr), :]
                n = xr.rows
                xr.set(_deltanet_rows(xr, lw, pos.lo, spec, m.gr[0:n, :], m.on[0:n, :],
                                      xs=_norm_quantize(xr, lw.g_in, spec.eps)))
                del xr
            del x
        else:
            del x
            _attention_tile(res, lw, c, s_, pos, T, spec, block, parts, gated=True,
                            depth=depth)
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
              depth: int = ATTN_DEPTH):
    """lfm2_rows tiled: T rows of a prompt run (pos a RunRows), its attention layers, its
    convolution blocks and its MLPs tiled. spill, depth: as qwen3_tile's."""
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
            _attention_tile(res, lw, c, s_, pos, T, spec, block, parts, depth=depth)
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


QPART = 8       # Gemma 4: a query part's [T, heads d] at most TMEM / QPART words
SHALLOW = 2     # the attention pipeline's depth of a run over IMEM at ATTN_DEPTH


def _gemma_attention_tile(res: Res, lw, m, pos: RunRows, ropes, T: int, block: int,
                          parts: int, depth: int = ATTN_DEPTH):
    """gemma4._attention tiled for T rows of a prompt run: x += norm(W_o attention(norm(x))).
    Per KV head its K / V (own layers: projected for the T rows, normed, K RoPE'd, appended a
    row at a time to its slot: the sliding ring's or the cache's), then its query group in
    parts of MCOLS heads, each part's [T, heads d] projected, normed and RoPE'd in place and
    its rows attending in packed tiles (P = MCOLS / G rows; a row's span is its own: the
    sliding window's Blocks or the global Bucket, their masks the mask tiles), the outputs
    written over the part's queries and quantized into W_o's operand; then W_o by output parts
    into [T, H] and the post-norm. Each row's arithmetic is _attention's. x: res.x, spilled
    while the heads attend and W_o runs (Res)."""
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
    ost = ol.stationary(T, spec.n_q * d)
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
        o = made[(j, g0)]                      # the outputs over the part's queries
        o.reshape(T * w, d)[r0 * w:(r0 + n) * w, :].set(acc / l[:, None])
        if r0 + n == T:
            ol.quantize_into(ost, made.pop((j, g0)), (j * G + g0) * d)
        del o

    _attend_heads(qhs, kv, kvh, seqs, block, ol.LOG2E, depth=depth, emit=emit)
    xs = None
    x = res.back()
    y = ol.empty([T, spec.hidden])
    _add_parts(y, [ost], [lw.wo], parts, y)
    del ost
    _add_norm(x, y, ol.load(lw.g_attn), eps)


@ol.jit
def gemma4_tile(m, pos, T: int, logit_rows, block: int = ATTN_BLOCK, spill: bool = False,
                depth: int = ATTN_DEPTH):
    """gemma4_step's rows tiled: T rows of a prompt run (pos a RunRows with the image's mask
    tiles), their tokens gathered from out[] and per-layer inputs as gemma4_step's; per layer
    the tiled attention, the dense MLP (GeGLU) tiled into [T, H] for its post-norm, the
    per-layer input block as gemma4._ple. spill: x in the I/O area's input rows while each
    layer's heads attend and W_o runs (Res). depth: as qwen3_tile's."""
    from ..kernels.lib import gelu_tanh
    from .gemma4 import _add_norm, _gathered_out, _ple, _ple_inputs, _pos_rows
    from .qwen3 import fill_logits
    fill_logits(m)
    spec = m.spec
    if not isinstance(pos, RunRows) or pos.amask is None or spec.experts:
        raise CompileError("a tiled Gemma 4 run is a dense model's prompt run with mask tiles")
    x = _gathered_out(m, pos)
    ropes = _pos_rows(m, pos, T, masks=False)
    x.set(x * math.sqrt(spec.hidden))           # (in place: one [T, H] tile)
    if spec.ple_dim:
        _ple_inputs(m, x, None)
    res = Res(x, m.x[0:T, :] if spill else None)
    del x
    parts, sub = _widths(m, T)

    def layer(li, it=None, jt=None):
        lw = m.layer(li, it, jt)
        _gemma_attention_tile(res, lw, m, pos, ropes, T, block, parts, depth)
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
    of the run's forms (variants) that fits TMEM and ACT RAM, at the attention's pipeline depth
    (ATTN_DEPTH), or where that is over IMEM at SHALLOW (its program grows with the blocks a
    head's entries leave unrolled, up to ~3 x depth a packed tile: past bucket ~3 the
    bigger tiles); over IMEM at both, the first (the caller's IMEM check)."""
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
    err = over = None
    imem = img.cfg.IMEM_WORDS // 8
    for v in variants(fam, T, img.cfg.MCOLS):       # the first that fits TMEM, ACT RAM, IMEM
        # (over IMEM: the same form with a shallower attention pipeline, then the next form)
        for f in [v] + ([dict(v, depth=SHALLOW)] if "depth" not in v else []):
            kw = dict(f, hidden=True) if hidden else f
            try:
                bs = [trace_fit(tile_run, img.cfg, s, lambda s=s: {
                    "m": desc(s), "kernel": KERNELS[fam], "pos": pos, "T": T, "kind": kind,
                    "block": block, "kw": kw}) for s in range(img.cfg.S)]
            except CompileError as e:
                if not any(w in str(e) for w in ("TMEM", "ACT RAM")):
                    raise
                err = e
                break
            progs = [b.finish() for b in bs]
            if max(map(len, progs)) <= imem:
                return progs, list(bs[0].run_args)
            if over is None:
                over = progs, list(bs[0].run_args)
    if over is not None:                # (the caller's IMEM check: prefill.programs)
        return over
    raise err


def variants(fam: str, T: int, mcols: int) -> list:
    """A tiled run's forms, tried in order until one fits TMEM and ACT RAM: as it is; x spilled
    while the heads attend (Res); Qwen3.5's DeltaNet layers in runs of fewer rows (whole passes
    of MCOLS, then fewer: their pair buffers grow with the rows)."""
    out = [{}, {"spill": True}]
    if fam == "qwen35":
        d = T
        while d > 1:
            d = d - mcols if d > mcols else d - 1
            out.append({"spill": True, "drows": d})
    return out
