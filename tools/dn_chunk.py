"""Chunked gated delta rule on the MXU (docs/dataflow.md 4.9.1: the accuracy study): formats_scan's
Qwen3.5 emulation with its DeltaNet recurrence in HF's chunked form
(transformers torch_chunk_gated_delta_rule), each of the chunk's eight matmuls on the MXU's int8
operands, against the sequential recurrence (DSTEP's, fp32: float64 here), by KL against the
float model (HF fp32's emulation) over a text.

Per chunk of C rows and head (q scaled, k l2-normed; g the log decays, beta the gates; S [dk, dv]
the state at the chunk's start, fp32 in DRAM):
    gc = cumsum(g); Gam[r, s] = exp(gc_r - gc_s)
    M1  L = (kb . k^T) * Gam, strictly lower           (d-contraction)
        T = (I + L)^-1                                  (VPU, fp32 forward substitution)
    M2  U0 = T . vb                                     (token contraction)
    M3  Wc = T . (kb * exp(gc))                         (token contraction)
    M5  U  = U0 - Wc . S                                (d-contraction, S's columns streamed)
    M4  A  = (q . k^T) * Gam, lower with the diagonal   (d-contraction)
    M6  O  = (q * exp(gc)) . S                          (d-contraction)
    M7     + A . U                                      (token contraction)
    M8  S' = exp(gc_C) S + (k * exp(gc_C - gc))^T . U   (token contraction; the add on the VPU)
MXU operands as the device takes them today: a d-contraction's left rows QACT (fp32 scale a row,
power-of-two offsets a D/4 group: quant.fake_act), its streamed rows int8 with one scale a row
(ROW: the KV cache's K, qwen3._fake_k at d = D); a token contraction's right operand int8 a token
(one scale a token, folded into the left rows before their QACT: P.V's V, qwen3._v_parts / _pv),
the left rows QACT over the chunk's tokens padded to D. MM's window drops (mx_matmul) included.

Modes (a variant's label ends in ";dn=MODE"; parse): cN chunks of N rows, nothing quantized
(the algorithm alone); cN-q<ids> those matmuls on the MXU's int8 operands, one pass; cN-h<ids>
those in three passes of operands split into their int8 part and the int8 of the remainder
(hi.hi + hi.lo + lo.hi: ~16-bit operands on today's MXU); e.g. c64-q568 the state's matmuls int8,
c64-h12347q568 the intra-chunk ones split.

    python tools/dn_chunk.py check
    python tools/dn_chunk.py ppl MODEL --formats F [--wformat fp4] [--tokens 2000]
                                 [--modes c64,c64-q568,...] [--out J]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))      # (formats_scan)
import formats_scan as FS  # noqa: E402
from opentpu import isa as I  # noqa: E402
from opentpu import quant as Q  # noqa: E402
from opentpu.llm import qwen35 as Q35  # noqa: E402
from opentpu.llm.qwen3 import _v_parts  # noqa: E402

D = 128
W8 = I.MX_W[I.W8]


def _pad(x, K):
    """x padded with zeros to K columns, its fp32 subnormals flushed to zero (as the device's
    fp32 VPU leaves them: a long-decayed row's exp(gc) is under fp32's range)."""
    x = np.where(np.abs(x) < 2.0 ** -126, 0.0, x)
    n = x.shape[-1]
    if n == K:
        return x
    out = np.zeros(x.shape[:-1] + (K,))
    out[..., :n] = x
    return out


def _mm(a, b):
    """a @ b^T per head: a an MXAct [h, R, K], b [h, N, K] (offsets 0: ROW or UNIT), with MM's
    window drops (none unless a group's offset is past the window)."""
    y = np.einsum("hrk,hnk->hrn", np.asarray(a).view(np.ndarray), b)
    if a.da is not None and a.da.max() > W8:
        for h in range(len(b)):
            y[h] = Q.mx_matmul(a.rows(h), b[h], None, W8, D)
    return y


def _mm_d(a, b):
    """a [h, R, d] @ b [h, N, d]^T on the MXU: a's rows QACT, b's rows int8 ROW."""
    K = -(-a.shape[-1] // D) * D
    bq = np.asarray(Q.fake_act(_pad(b, K), D, row=True)).view(np.ndarray)
    return _mm(Q.fake_act(_pad(a, K), D), bq)


def _mm_t(a, x):
    """a [h, R, n] @ x [h, n, m] (a token contraction) on the MXU: x int8 a token, its scales
    folded into a's columns, a's rows QACT (padded to D)."""
    xq, xs = _v_parts(_pad(x, x.shape[-1]))
    K = -(-a.shape[-1] // D) * D
    return _mm(Q.fake_act(_pad(a * xs[:, None, :], K), D),
               _pad(np.swapaxes(xq, 1, 2), K))


def _hi_d(a, b):
    """The operands' int8 parts as _mm_d quantizes them (dequantized)."""
    K = -(-a.shape[-1] // D) * D
    return (np.asarray(Q.fake_act(_pad(a, K), D)).view(np.ndarray)[..., :a.shape[-1]],
            np.asarray(Q.fake_act(_pad(b, K), D, row=True)).view(np.ndarray)[..., :b.shape[-1]])


def _hi_t(a, x):
    """The operands' int8 parts as _mm_t quantizes them (x's a token; a's QACT rows with x's
    scales folded, divided out again)."""
    xq, xs = _v_parts(_pad(x, x.shape[-1]))
    K = -(-a.shape[-1] // D) * D
    sc = np.where(xs == 0, 1.0, xs)[:, None, :]
    ah = np.asarray(Q.fake_act(_pad(a * sc, K), D)).view(np.ndarray)[..., :a.shape[-1]] / sc
    return ah, xq * xs[..., None]


def mm(kind: str, a, b, how):
    """kind "d": a @ b^T over d; "t": a @ b over the chunk's tokens. how None: exact; "q": the
    MXU's int8 operands (one pass); "h": each operand split into its int8 part and the int8 of
    its remainder, three passes (hi.hi + hi.lo + lo.hi: ~16-bit operands on today's MXU)."""
    exact = (lambda x, y: np.einsum("hrk,hnk->hrn", x, y)) if kind == "d" else \
        (lambda x, y: np.einsum("hrn,hnm->hrm", x, y))
    if how is None:
        return exact(a, b)
    one = _mm_d if kind == "d" else _mm_t
    if how == "q":
        return one(a, b)
    ah, bh = (_hi_d if kind == "d" else _hi_t)(a, b)
    return one(ah, bh) + one(ah, b - bh) + one(a - ah, bh)


def _solve(L):
    """(I + L)^-1 of strictly lower L [h, n, n] by forward substitution."""
    n = L.shape[-1]
    T = np.broadcast_to(np.eye(n), L.shape).copy()
    for i in range(1, n):
        T[:, i, :i] = -np.einsum("hs,hsj->hj", L[:, i, :i], T[:, :i, :i])
    return T


def parse(mode: str):
    """(C, {matmul id: how}) of a mode "cN[-<how><ids>...]": N rows a chunk; each how ("q" one
    int8 pass, "h" three passes of split operands) followed by the matmul ids (1-8, module
    doc) it applies to; the rest exact. E.g. c64-q568: the state's matmuls int8, the intra-
    chunk ones exact; c64-h12347q568."""
    import re
    m = re.fullmatch(r"c(\d+)(?:-((?:[qh]\d+)+))?", mode)
    if m is None:
        raise ValueError(f"mode {mode!r}")
    how = {}
    for h, ids in re.findall(r"([qh])(\d+)", m.group(2) or ""):
        how.update({int(i): h for i in ids})
    return int(m.group(1)), how


def recur_seq(q, k, v, g, beta):
    """formats_scan's recurrence: o [T, h, dv]."""
    T, nh, dk = q.shape
    S, o = np.zeros((nh, dk, v.shape[2])), np.zeros((T, nh, v.shape[2]))
    for t in range(T):
        S = S * np.exp(g[t])[:, None, None]
        delta = (v[t] - np.einsum("hij,hi->hj", S, k[t])) * beta[t][:, None]
        S = S + k[t][:, :, None] * delta[:, None, :]
        o[t] = np.einsum("hij,hi->hj", S, q[t])
    return o, S


def recur_chunk(q, k, v, g, beta, mode: str):
    """The chunked recurrence (module doc): o [T, h, dv] and the final state."""
    C, how = parse(mode)
    h = how.get
    T, nh, dk = q.shape
    dv = v.shape[2]
    S, o = np.zeros((nh, dk, dv)), np.zeros((T, nh, dv))
    for c0 in range(0, T, C):
        n = min(C, T - c0)
        tr = lambda a: np.ascontiguousarray(np.swapaxes(a[c0:c0 + n], 0, 1))   # noqa: E731
        Qc, Kc, Vc = tr(q), tr(k), tr(v)                    # [h, n, d]
        b, gc = beta[c0:c0 + n].T, np.cumsum(g[c0:c0 + n].T, -1)      # [h, n]
        low, lowd = np.tril(np.ones((n, n)), -1), np.tril(np.ones((n, n)))
        Gam = np.exp((gc[:, :, None] - gc[:, None, :]) * lowd)  # (HF's: the exponent's upper
        #                                       triangle zeroed first, as it would overflow)
        Kb = Kc * b[..., None]
        Tm = _solve(mm("d", Kb, Kc, h(1)) * Gam * low)                          # M1
        U0 = mm("t", Tm, Vc * b[..., None], h(2))                               # M2
        Wc = mm("t", Tm, Kb * np.exp(gc)[..., None], h(3))                      # M3
        St = np.ascontiguousarray(np.swapaxes(S, 1, 2))     # [h, dv, dk]: S's columns
        U = U0 - mm("d", Wc, St, h(5))                                          # M5
        A = mm("d", Qc, Kc, h(4)) * Gam * lowd                                  # M4
        O = mm("d", Qc * np.exp(gc)[..., None], St, h(6)) + mm("t", A, U, h(7))  # M6, M7
        gl = gc[:, -1]
        Kd = Kc * np.exp(gl[:, None] - gc)[..., None]
        S = np.exp(gl)[:, None, None] * S + mm(
            "t", np.ascontiguousarray(np.swapaxes(Kd, 1, 2)), U, h(8))           # M8
        o[c0:c0 + n] = np.swapaxes(O, 0, 1)
    return o, S


def _mode() -> str:
    lab = FS._AT.get("lab") or ""
    return lab.partition(";dn=")[2]


def _qwen35(spec, W, i, x, w, fq, c, s, D, wf):
    """formats_scan._qwen35 with the recurrence of the running variant's mode (_mode)."""
    T, d, eps, K = len(x), spec.head_dim, spec.eps, spec.conv_k
    if spec.kinds[i] != Q35.LIN or not _mode():
        return _ORIG(spec, W, i, x, w, fq, c, s, D, wf)
    nh, nk, dk, dv = spec.lin_heads, spec.lin_nk, spec.lin_dk, spec.lin_dv
    og = 4 if nh % 4 == 0 else 2
    outr = [og * dv] * (nh // og)
    if og == 4 and nh > 2:
        outr[-1:] = [2 * dv, 2 * dv]
    p = f"model.layers.{i}."

    def g1(n):
        return 1 + np.asarray(W[n], np.float64)

    h = fq(Q35._norm(x, g1(p + "input_layernorm.weight"), eps), D)
    a = p + "linear_attn."
    qkv = h @ w(a + "in_proj_qkv.weight").T
    win = np.concatenate([np.zeros((K - 1, qkv.shape[1])), qkv])
    wc = W[a + "conv1d.weight"][:, 0, :]
    y = Q35._silu(sum(win[j:j + T] * wc[:, j] for j in range(K)))
    q, k, v = np.split(y, [nk * dk, 2 * nk * dk], -1)
    q = np.repeat(Q35._l2norm(q.reshape(T, nk, dk)) / math.sqrt(dk), nh // nk, axis=1)
    k = np.repeat(Q35._l2norm(k.reshape(T, nk, dk)), nh // nk, axis=1)
    v = v.reshape(T, nh, dv)
    z = (h @ w(a + "in_proj_z.weight").T).reshape(T, nh, dv)
    beta = 1 / (1 + np.exp(-(h @ w(a + "in_proj_b.weight").T)))
    g = -np.exp(np.asarray(W[a + "A_log"], np.float64)) * Q35._softplus(
        h @ w(a + "in_proj_a.weight").T + W[a + "dt_bias"])
    o, _ = recur_chunk(q, k, v, g, beta, _mode())
    o = Q35._norm(o, np.asarray(W[a + "norm.weight"], np.float64), eps) * Q35._silu(z)
    x = x + fq(o.reshape(T, -1), D, outr) @ w(a + "out_proj.weight").T
    h = fq(Q35._norm(x, g1(p + "post_attention_layernorm.weight"), eps), D)
    mp = spec.mlp_prefix(p)
    gg, u = h @ w(mp + "gate_proj.weight").T, h @ w(mp + "up_proj.weight").T
    dn = mp + "down_proj.weight"
    y = fq(Q35._silu(gg) * u, D, FS._down(spec, wf, dn, D)) @ w(dn).T
    if spec.moe is not None:
        return FS.MoEIn(x, h, y)
    return x + y


_ORIG = FS._qwen35
FS.LAYER[Q35] = _qwen35


def check():
    """The chunked recurrence against the sequential one on random gates and rows (the 35B's
    head shape): unquantized to rounding; quantized, the outputs' relative error."""
    rng = np.random.default_rng(3)
    T, nh, d = 200, 4, 128
    q = rng.standard_normal((T, nh, d))
    q /= np.linalg.norm(q, axis=-1, keepdims=True) * math.sqrt(d)
    k = rng.standard_normal((T, nh, d))
    k /= np.linalg.norm(k, axis=-1, keepdims=True)
    v = rng.standard_normal((T, nh, d))
    g = -np.exp(rng.uniform(-4, 0.5, (T, nh)))
    g[:, 0] = -np.exp(rng.uniform(2, 6, T))     # (a head that forgets at once: exp(-gc) overflows)
    beta = 1 / (1 + np.exp(-rng.standard_normal((T, nh))))
    o0, S0 = recur_seq(q, k, v, g, beta)
    ok = True
    for m in ("c16", "c32", "c64", "c7", "c16-q12345678", "c64-q12345678", "c64-q12347",
              "c64-q568", "c64-q1", "c64-q23", "c64-q47", "c64-h12345678", "c64-h12347q568",
              "c16-h12345678"):
        o, S = recur_chunk(q, k, v, g, beta, m)
        e = np.abs(o - o0).max() / np.abs(o0).max()
        es = np.abs(S - S0).max() / np.abs(S0).max()
        rel = np.linalg.norm(o - o0) / np.linalg.norm(o0)
        print(f"{m:6s} max|do|/max|o| {e:.3e}  ||do||/||o|| {rel:.3e}  max|dS|/max|S| {es:.3e}")
        if not np.isfinite(o).all() or "-" not in m and (e > 1e-9 or es > 1e-9):
            ok = False
    print("check", "ok" if ok else "FAILED")
    return 0 if ok else 1


def ppl(model, formats: str, wformat: str, n_tok: int, modes: list, out=None):
    path = FS.model_dir(model)
    spec, run, nbytes, _, _ = FS._open(path, D)
    ids = FS.text_ids(path, n_tok)
    t0 = time.time()
    specs = {"float": None, "int8": ("", "int8", None), formats: (formats, wformat, None)}
    specs.update({f"{formats};dn={m}": (formats, wformat, None) for m in modes})
    rows = run(specs, ids, lambda m: print(f"[{time.time() - t0:6.0f} s] {m}", flush=True))
    FS._finish(rows, nbytes)
    base = np.asarray(rows[formats]["kl_tok"])
    for r in rows.values():
        dk = np.asarray(r["kl_tok"]) - base
        r["dkl_prod"], r["se_prod"] = float(dk.mean()), float(dk.std(ddof=1) / math.sqrt(len(dk)))
    if out:
        Path(out).write_text(json.dumps({"model": str(model), "tokens": len(ids),
                                         "formats": formats, "wformat": wformat,
                                         "variants": rows}, indent=1))
    for lab, r in rows.items():
        print(f"{lab:60s} ppl {r['ppl']:8.4f}  KL/float {100 * r['kl_float']:.3f}%  dKL "
              f"{100 * r['dkl']:+.3f}% (+-{100 * r['se_dkl']:.3f})  vs prod "
              f"{100 * r['dkl_prod']:+.4f}% (+-{100 * r['se_prod']:.4f})  top1/int8 "
              f"{r['agree_int8']:.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    a = sub.add_parser("ppl")
    a.add_argument("model")
    a.add_argument("--formats", required=True)
    a.add_argument("--wformat", default="fp4")
    a.add_argument("--tokens", type=int, default=2000)
    a.add_argument("--modes", default="c64,c16-q12345678,c64-q12345678,c64-q568")
    a.add_argument("--out")
    a = ap.parse_args()
    if a.cmd == "check":
        return check()
    ppl(a.model, a.formats, a.wformat, a.tokens, a.modes.split(","), a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
