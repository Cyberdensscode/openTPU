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
  numbers of tokens: their last blocks take a per-row-group additive mask (RowMasks), which
  gives bit for bit the online softmax of each row alone;
- the MLP keeps every chunk's activation quantized in ACT RAM (F / D blocks of T rows) and runs
  the down projection by parts of its output columns, each part summing the chunks in the same
  order (the same bits as swiglu_down's accumulation), so TMEM holds [T, H / parts], not [T, H].

Each row's arithmetic is qwen3_rows', so the logits and the KV cache are bit for bit those of
today's prefill runs over the same rows (tests/test_tiled.py). Qwen3 and the Llama-likes on this
code path (llama.py); Qwen3.5's gated attention and DeltaNet rows are not covered here.
"""
from __future__ import annotations

import math

from .. import language as ol
from ..compiler import trace_fit
from ..kernels.attention import RowMasks, TileSeq, _attend_heads
from ..kernels.lib import silu
from .qwen3 import (ATTN_BLOCK, ATTN_DEPTH, _inputs_rows, _layers, _lm_head_rows, _load_opt,
                    _norm_heads, _padded, _rope_gate, _rope_rows_padded)


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


def _norm_quantize(x, g, eps: float):
    """quantize(rmsnorm(x, g)): the QACT RSCALE / CSCALE that the peephole makes of it (the
    same bytes and scales), without tracing the [T, H] products (x * x, x * r, x * r * g)."""
    r = ol.rsqrt(ol.sumsq(x) * (1.0 / x.cols) + eps)
    return ol.quantize(x, free=True, rscale=r, cscale=ol.load(g))


def _attention_tile(x, lw, c, s_, p0: int, T: int, spec, block: int, parts: int):
    """x + W_o . attention(x) for T rows at positions p0 .. p0 + T - 1 of sequence 0 (their K / V
    appended first, each row attending causally over the cache): _attention_rows' arithmetic,
    one KV head's projections at a time and its queries in packed tiles."""
    d, G, eps = spec.head_dim, spec.n_q // spec.n_kv, spec.eps
    D = ol.block_size()
    dk = -(-d // D) * D
    xs = _norm_quantize(x, lw.g_in, eps)       # T rows: one pass a weight
    qn, kn = _load_opt(lw.qn), _load_opt(lw.kn)
    c, s_ = _rope_gate(lw, c, s_)
    scale = ol.LOG2E / math.sqrt(d)
    heads = list(lw.kv.owned_heads(spec.n_kv))
    nq = len(heads) * G
    P = packing(ol.mxu_columns(), G)
    tiles = [(r0, min(P, T - r0)) for r0 in range(0, T, P)]
    ost = ol.stationary(T, nq * d)             # W_o's operand, a KV head's columns at a time
    oh = {}                                    # KV head j -> its outputs [T, G d] in progress
    ent, qhs, kvh, seqs = [], [], [], []

    def head(j):
        """KV head j's projections, K / V appends and queries [T G, dk], row (r, g) = r G + g."""
        hh = heads[j]
        k = ol.dot(xs, lw.wk[j * d:(j + 1) * d, :])
        v = ol.dot(xs, lw.wv[j * d:(j + 1) * d, :])
        kj = _rope_rows_padded(_norm_heads(k, kn, eps), c, s_)
        ol.kv_append(lw.kv, hh, p0, kj, _padded(v))
        del k, v, kj
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

    def query(j, r0, n):
        def emit():
            if j not in made:
                made[j] = head(j)
            Q = made[j]
            if r0 + n == T:
                made.pop(j)
            return Q[r0 * G:(r0 + n) * G, :]
        return emit

    for j in range(len(heads)):
        for r0, n in tiles:
            ent.append((j, r0, n))
            qhs.append(query(j, r0, n))
            kvh.append(heads[j])
            seqs.append(tile_seq(p0, r0, n, block, G))

    def emit(i, acc, l):
        j, r0, n = ent[i]
        if j not in oh:
            oh[j] = ol.empty([T, G * d])
        for a in range(n):
            oh[j][r0 + a, :].reshape(G, d).set(
                acc[a * G:(a + 1) * G, :] / l[a * G:(a + 1) * G][:, None])
        if r0 + n == T:                        # the head's rows done: quantized, freed
            ol.quantize_into(ost, oh.pop(j), j * G * d)

    _attend_heads(qhs, lw.kv, kvh, seqs, block, scale, depth=ATTN_DEPTH, emit=emit)
    _add_parts(x, [ost], [lw.wo], parts)                       # x += W_o o, by output parts
    return x


def _add_parts(x, acts, ws, parts: int) -> None:
    """x += sum_i acts[i] . ws[i]^T, the sum in the order of the list (dot, then dot(acc=)),
    by `parts` parts of the output columns: TMEM holds [T, H / parts] of the sum at a time,
    each element summed as one [T, H] result would be."""
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
        x[:, n0:n0 + hp].set(x[:, n0:n0 + hp] + y)
        del y


def _mlp_tile(x, lw, spec, parts: int, sub: int):
    """x + W_down(silu(W_gate xn) * (W_up xn)) for T rows: every chunk's activation quantized
    once and kept in ACT RAM (the C columns of a W_down part, made `sub` columns at a time:
    each piece's QACT gives its blocks what one of the whole chunk would), then the down
    projection by `parts` parts of its output columns, each summing the chunks in swiglu_down's
    order (the same bits)."""
    xs = _norm_quantize(x, lw.g_post, spec.eps)
    wg, wu, wd = lw.wg, lw.wu, lw.wd
    T = x.rows
    C = wd.pw if wd.parts is not None else wg.shape[0]
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
        ol.quantize_into(acts[c0], silu(g) * u, s0)
        del g, u
    xs = nxt = None                     # the operand's ACT blocks are free for the down MMs
    _add_parts(x, [acts[c0] for c0 in starts], [wd[:, c0:c0 + C] for c0 in starts], parts)
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


@ol.jit
def qwen3_tile(m, p0: int, T: int, logit_rows, block: int = ATTN_BLOCK):
    """T rows at positions p0 .. p0 + T - 1 of sequence 0 (their embedding and RoPE rows in the
    I/O area, as qwen3_rows'), the logits of `logit_rows` (empty: none)."""
    spec = m.spec
    x, c, s_ = _inputs_rows(m, [(0, p0 + r) for r in range(T)])
    D, W = ol.block_size(), ol.tmem_words()
    parts = mlp_parts(spec.hidden // ol.num_programs(), D, T, W)
    for lw in _layers(m):
        _attention_tile(x, lw, c, s_, p0, T, spec, block, parts)
        _mlp_tile(x, lw, spec, parts, mlp_sub(lw.wd.pw or lw.wg.shape[0], D, T, W))
    _lm_head_rows(x, m, spec, logit_rows)


def compile_tile(img, p0: int, T: int, block: int = ATTN_BLOCK, logit_rows=()) -> list:
    """One program per slice: qwen3_tile over the image's I/O rows (img.rows >= T), every MM
    over T rows (the configuration's ACT_ROWS >= T)."""
    cfg = img.cfg
    if T > img.rows:
        raise ValueError(f"{T} rows, the image's I/O area holds {img.rows}")
    if T > cfg.act_rows:
        raise ValueError(f"{T} rows: one MM over them needs ACT_ROWS >= {T} ({cfg.act_rows})")
    if cfg.S != 1:
        raise ValueError("tiled prefill: one slice")
    return [trace_fit(qwen3_tile, cfg, s, lambda s=s: {
        "m": img.descriptors(s), "p0": p0, "T": T, "logit_rows": list(logit_rows),
        "block": block}).finish() for s in range(cfg.S)]
