"""EXPERIMENTAL (docs/mx_mxu.md): the accuracy of an MX MXU's formats, in formats_scan's float64
emulation (tools/formats_scan.py: the same models, text, layers and dKL measure).

An MX MXU accumulates a row's K-blocks in the column end's DSP (48-bit integers) and runs the fp
epilogue once per output. That needs every per-block scale to be a shift: the weight and the
activation scales become a per-row fp32 scale (applied once, in the epilogue) times a power of
two per block or sub-block, and a block whose two exponent offsets add up to more than the
shift window W is dropped (its B operand would need a right shift). Formats:

  weights   int8      today: int8, an fp32 scale per 128 (not a shift: today's MXU only)
            fp4       today: E2M1, bf16 per 128 x u4 multiplier per 32 (opentpu/quant.py)
            int8x     int8, fp32 per row, 2^-d per 128 (d: uint8)                     W 16
            fp4x6     E2M1, fp32 per row, 2^-d per 128, 6-bit multiplier per 32         W 11
            fp4x4     the same with today's 4-bit multipliers                          W 13
            nvfp4     NVIDIA's: E2M1, E4M3 per 16, fp32 per tensor (on an MX column the
                      row's largest exponent becomes a per-row byte, as nvfp4r)
            nvfp4r    E2M1, E4M3 per 16 relative to an fp32 per row: on an MX column the
                      E4M3 mantissa is the 4-bit multiplier, its exponent the shift      W 13
            mxfp4     OCP MXFP4: E2M1, E8M0 per 32 (relative to the row's largest)      W 16
  activations a8      today: int8, fp32 per 128 (QACT)
            a8x128    int8, fp32 per row, 2^-d per 128
            a8x32     int8, fp32 per row, 2^-d per 32 (OCP MXINT8's granularity; the ACT
                      scale word holds four 8-bit offsets)

A variant is "WEIGHTS/HEAD/ACTS[/flush]": the layers' weights, the LM head's, the activations
(every MM input, the head's, q, P and K as the attention reads them; V stays int8 per token),
and with /flush the window's drops emulated exactly (per 16 elements; zero blocks never drop).

    PYTHONPATH=. python3 tools/exp/mx_formats.py MODEL OUT.json [--tokens 2000] VARIANT...
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import formats_scan as FS  # noqa: E402
from opentpu import quant as Q  # noqa: E402
from opentpu.llm import load_spec, model_dir  # noqa: E402
from opentpu.llm import qwen3 as Q3  # noqa: E402
from opentpu.llm.qwen3 import _fake_q, _v_parts, rope_tables  # noqa: E402

G16 = 16                                    # the drop test's granularity (NVFP4's sub-block)
WIN = {"int8x": 16, "fp4x6": 11, "fp4x4": 13, "nvfp4r": 13, "mxfp4": 16}
LOG = print


# ------------------------------------------------------------------ quantized operands
class XAct(np.ndarray):
    """Activations as an MX quantizer leaves them: values (float64), and dg [..., K/16], each
    16 elements' exponent offset below the row's scale."""

    def __new__(cls, x, dg):
        o = np.asarray(x).view(cls)
        o.dg = dg
        return o

    def __array_finalize__(self, obj):
        self.dg = getattr(obj, "dg", None)


class WX:
    """A weight as the device holds it: values deq [N, K] and, for MX formats, the exponent
    offsets dw [N, K/16] and the window W. `h @ w.T` is the MM (with the window's drops when
    the activations are XAct and flush is on)."""
    __array_ufunc__ = None                  # ndarray @ WXT defers to WXT.__rmatmul__

    def __init__(self, deq, dw=None, W=None, flush=False):
        self.deq, self.dw, self.W, self.flush = deq, dw, W, flush

    @property
    def T(self):
        return WXT(self)

    @property
    def shape(self):
        return self.deq.shape


FLUSH = {"label": None}             # the variant being run; {label: [dropped, terms]}


class WXT:
    __array_ufunc__ = None

    def __init__(self, w):
        self.w = w

    def __rmatmul__(self, h):
        w = self.w
        hv = np.asarray(h).view(np.ndarray)
        y = hv @ w.deq.T
        if not (w.flush and w.dw is not None and isinstance(h, XAct) and h.dg is not None):
            return y
        cnt = FLUSH.setdefault(FLUSH["label"], [0, 0])
        hf = hv.reshape(-1, hv.shape[-1])
        dg = np.asarray(h.dg).reshape(-1, h.dg.shape[-1])
        yf = y.reshape(-1, y.shape[-1])
        dwmax = w.dw.max(0)
        for g in range(w.dw.shape[1]):
            rows = np.nonzero(dg[:, g] + dwmax[g] > w.W)[0]
            if not len(rows):
                continue
            m = (dg[rows, g][:, None] + w.dw[None, :, g]) > w.W
            if not m.any():
                continue
            sl = slice(G16 * g, G16 * (g + 1))
            yf[rows] -= (hf[rows, sl] @ w.deq[:, sl].T) * m
            cnt[0] += int(m.sum())
        cnt[1] += hf.shape[0] * w.dw.size
        return y


def _expand(d, g):
    """Offsets per g elements -> per 16 (repeated)."""
    return np.repeat(d, g // G16, axis=-1)


def _row_rel(amax_blocks, amax_row):
    """d = floor(log2(row amax / block amax)), the largest power-of-two step that keeps the
    block's maximum within the row scale's range; zero blocks get 0 (they never drop)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.floor(np.log2(amax_row / amax_blocks))
    return np.where(amax_blocks > 0, np.clip(d, 0, 255), 0)


# ------------------------------------------------------------------ activations
def a8x(x, g):
    """int8 per row (fp32: amax/127) times 2^-d per g elements of the last axis."""
    x = np.asarray(x, np.float64)
    sh = x.shape
    g = min(g, sh[-1])                      # a row shorter than a group (LFM2's 64-wide heads)
    xb = x.reshape(*sh[:-1], sh[-1] // g, g)
    ab = np.abs(xb).max(-1)
    ar = ab.max(-1, keepdims=True)
    d = _row_rel(ab, ar)
    s = np.where(ar == 0, 1.0, ar / 127) * np.exp2(-d)
    q = np.clip(np.rint(xb / s[..., None]), -127, 127) * s[..., None]
    return XAct(q.reshape(sh), _expand(d, g))


def fq_of(kind):
    if kind == "a8":
        return _fake_q
    g = {"a8x128": 128, "a8x32": 32}[kind]
    return lambda x, D=128: a8x(x, g)


# ------------------------------------------------------------------ weights
def w_int8x(a):
    a = np.asarray(a, np.float64)
    N, K = a.shape
    xb = a.reshape(N, K // 128, 128)
    ab = np.abs(xb).max(-1)
    ar = ab.max(-1, keepdims=True)
    d = _row_rel(ab, ar)
    s = (np.where(ar == 0, 1.0, ar / 127) * np.exp2(-d))[..., None]
    q = np.clip(np.rint(xb / s), -127, 127) * s
    return WX(q.reshape(N, K), _expand(d, 128), WIN["int8x"])


def w_fp4x(a, mbits):
    """E2M1, sub-block scale Sw[n] * 2^-d[n, k] * m[n, k, b] (m in 1 .. 2^mbits - 1), the
    multiplier picked per sub-block by squared error over formats' search grid."""
    a = np.asarray(a, np.float64)
    N, K = a.shape
    mmax = (1 << mbits) - 1
    xb = a.reshape(N, K // 128, 4, 32)
    sb = np.abs(xb).max(-1) / Q.E2M1_MAX                     # ideal sub-block scales [N, KB, 4]
    top = sb.max(-1).max(-1, keepdims=True)                  # [N, 1]
    Sw = np.where(top == 0, 1.0, top / mmax)
    bmax = sb.max(-1)                                        # [N, KB]
    with np.errstate(divide="ignore"):
        d = np.floor(np.log2(mmax * Sw / bmax))
    d = np.where(bmax > 0, np.clip(d, 0, 255), 0)
    unit = (Sw * np.exp2(-d))[..., None]                     # [N, KB, 1]
    m0 = sb / unit
    best_q = best_e = None
    for f in Q.SEARCH:
        m = np.clip(np.rint(m0 * f), 1, mmax)
        s = (unit * m)[..., None]
        q = np.sign(xb) * Q.E2M1[Q.e2m1_code(xb / s)] * s
        e = ((q - xb) ** 2).sum(-1)
        if best_e is None:
            best_q, best_e = q, e
        else:
            k = e < best_e
            best_q = np.where(k[..., None], q, best_q)
            best_e = np.where(k, e, best_e)
    return WX(best_q.reshape(N, K), _expand(d, 128), WIN[f"fp4x{mbits}"])


def w_nvfp4(a, per_row):
    """NVFP4 (E2M1 per 16, E4M3 scale, fp32 per tensor; per_row: per row instead), search on.
    dw: the E4M3 exponent's offset below the row's largest (subnormals at -6)."""
    a = np.asarray(a, np.float64)
    if not per_row:
        deq = Q.quantize(a.astype(np.float32), "nvfp4", search=True).deq.astype(np.float64)
        return WX(deq)
    N, K = a.shape
    deqs, dws = [], []
    R = max(1, (1 << 22) // K)
    for r0 in range(0, N, R):
        blk = a[r0:r0 + R]
        n = len(blk)
        ts = np.abs(blk).max(-1, keepdims=True) / (Q.E4M3_MAX * Q.E2M1_MAX)
        ts = np.where(ts == 0, 1.0, ts)
        xb = blk.reshape(n, K // 16, 16) / ts[..., None]
        amax = np.abs(xb).max(-1)
        best = None
        for f in Q.SEARCH:
            s = Q.e4m3(amax * f / Q.E2M1_MAX)
            s = np.where((amax == 0) | (s == 0), 1.0, s)
            q = np.sign(xb) * Q.E2M1[Q.e2m1_code(xb / s[..., None])] * s[..., None]
            e = ((q - xb) ** 2).sum(-1)
            if best is None:
                best = [s, q, e]
            else:
                k = e < best[2]
                best = [np.where(k, s, best[0]), np.where(k[..., None], q, best[1]),
                        np.where(k, e, best[2])]
        s, q, _ = best
        ex = np.maximum(np.floor(np.log2(s)), -6)
        live = amax > 0
        emax = np.where(live, ex, -99).max(-1, keepdims=True)
        dws.append(np.where(live, emax - ex, 0))
        deqs.append((q * ts[..., None]).reshape(n, K))
    return WX(np.concatenate(deqs), np.concatenate(dws), WIN["nvfp4r"])


def w_mxfp4(a):
    a = np.asarray(a, np.float64)
    N, K = a.shape
    deq = Q.quantize(a.astype(np.float32), "mxfp4").deq.astype(np.float64)
    xb = a.reshape(N, K // 32, 32)
    amax = np.abs(xb).max(-1)
    with np.errstate(divide="ignore"):
        e = np.floor(np.log2(np.maximum(amax, 2.0 ** -127)))    # the OCP rule's exponent - 2
    live = amax > 0
    emax = np.where(live, e, -999).max(-1, keepdims=True)
    return WX(deq, _expand(np.where(live, emax - e, 0), 32), WIN["mxfp4"])


def dequant(a, fmt, D=128):
    if fmt is not None and fmt.endswith("+f"):          # the window's drops emulated
        w = dequant(a, fmt[:-2], D)
        w.flush = True
        return w
    if fmt is None:
        return WX(np.asarray(a, np.float64))
    if fmt in ("int8", "fp4", "int4"):
        return WX(FS._DEQUANT(a, fmt, D))
    if fmt == "int8x":
        return w_int8x(a)
    if fmt in ("fp4x6", "fp4x4"):
        return w_fp4x(a, int(fmt[-1]))
    if fmt in ("nvfp4", "nvfp4r"):
        return w_nvfp4(a, fmt == "nvfp4r")
    if fmt == "mxfp4":
        return w_mxfp4(a)
    raise ValueError(fmt)


# ------------------------------------------------------------------ emulation
def _attend(q, K, v, fq, D, qd, scale):
    """formats_scan._attend with V always int8 per token for a quantized variant (its check is
    `fq is _fake_q`) and plain arrays into the products."""
    T, nq, _ = q.shape
    G, Dp = nq // K.shape[1], -(-T // D) * D
    quant = fq is not FS._ident
    V, vs = _v_parts(v) if quant else (v, np.ones(v.shape[:2]))
    K = np.asarray(K).view(np.ndarray)
    causal = np.tri(T, dtype=bool)
    o = np.zeros((T, nq, V.shape[2]))
    for hq in range(nq):
        qq = np.asarray(fq(q[:, hq] / scale, qd)).view(np.ndarray)
        sc = np.where(causal, qq @ K[:, hq // G].T, -np.inf)
        pp = np.exp(sc - sc.max(1, keepdims=True))
        ppad = np.zeros((T, Dp))
        ppad[:, :T] = pp * vs[:, hq // G]
        pq = np.asarray(fq(ppad, D)).view(np.ndarray)
        o[:, hq] = (pq[:, :T] @ V[:, hq // G]) / pp.sum(1, keepdims=True)
    return o


def emulate(spec, W, tokens, variants: dict, D=128):
    """formats_scan.emulate with a per-variant activation quantizer: variants {label: None
    (float) or (wfmt(kind) -> format, fq)}; returns {label: (nll, top1, {ref: KL})}."""
    M = FS.family(spec)
    T = len(tokens)
    x0 = np.asarray(W["model.embed_tokens.weight"][list(tokens)], np.float64)
    xs = {lab: (_fake_q(x0, D) if v is not None and spec.embed == "int8" else x0)
          for lab, v in variants.items()}
    cs = [rope_tables(spec, p) for p in range(T)]
    c = np.stack([a for a, _ in cs]).astype(np.float64)[:, None, :]
    s = np.stack([b for _, b in cs]).astype(np.float64)[:, None, :]
    t0 = time.time()
    for i in range(spec.layers):
        held: dict = {}

        def w(n, fmt):
            if (n, fmt) not in held:
                held[n, fmt] = dequant(W[n], fmt, D)
            return held[n, fmt]

        for lab, v in variants.items():
            FLUSH["label"] = lab
            if v is None:
                fq, wl = FS._ident, (lambda n: w(n, None))
            else:
                wf, fq = v
                wl = (lambda n, wf=wf: w(n, wf(M.weight_kind(n)[0])))
            xs[lab] = np.asarray(FS.LAYER[M](spec, W, i, xs[lab], wl, fq, c, s, D)).view(
                np.ndarray)
        held.clear()
        LOG(f"layer {i + 1}/{spec.layers} ({time.time() - t0:.0f} s)", flush=True)
    hs = {lab: (FS._ident if v is None else v[1])(FS._final(spec, W, xs[lab]), D)
          for lab, v in variants.items()}
    del xs
    head = "model.embed_tokens.weight" if spec.tied else "lm_head.weight"
    Wh = W[head]
    fmts = {lab: None if v is None else v[0]("head") for lab, v in variants.items()}
    return FS._head(hs, fmts, lambda: ((r0, Wh[r0:r0 + FS.HEAD_ROWS])
                                       for r0 in range(0, Wh.shape[0], FS.HEAD_ROWS)),
                    tokens, D, False, ("float", "int8"))


def parse(v: str):
    """'WEIGHTS/HEAD/ACTS[/flush]' -> (wfmt(kind) -> format, fq). WEIGHTS: a format, then
    optionally kind=format overrides ("nvfp4,attn=fp4x6": attention in fp4x6)."""
    p = v.split("/")
    f = "+f" if len(p) > 3 and p[3] == "flush" else ""
    base, *over = p[0].split(",")
    kinds = {k: x + f for k, x in (o.split("=") for o in over)}
    wl, wh, act = base + f, p[1] + f, p[2]
    return (lambda kind, wl=wl, wh=wh: wh if kind == "head" else kinds.get(kind, wl)), fq_of(act)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("variants", nargs="+")
    ap.add_argument("--tokens", type=int, default=2000)
    a = ap.parse_args()
    FS._DEQUANT = FS.dequant
    FS.dequant = dequant                     # _head's weights
    FS._attend = _attend                     # the layers' attention
    path = model_dir(a.model)
    spec = load_spec(path)
    W = Q3.load_weights(path)
    ids = FS.text_ids(path, a.tokens)
    rows = {}
    # float and int8 (today's: int8 weights and head, a8) are every variant's references
    var = {"float": None, "int8": (lambda kind: "int8", _fake_q)}
    var.update({v: parse(v) for v in a.variants})
    t0 = time.time()
    res = emulate(spec, W, ids, var)
    for lab, (nll, top1, kl) in res.items():
        rows[lab] = {"nll": float(nll.mean()), "ppl": float(np.exp(nll.mean())),
                     "top1": top1.tolist(), "nll_tok": np.round(nll, 5).tolist(),
                     "kl_float": float(kl["float"].mean()),
                     "kl_tok": np.round(kl["float"], 7).tolist(),
                     "kl_int8": float(kl["int8"].mean()), "wformat": None, "formats": lab,
                     "head": None, "flush": FLUSH.get(lab)}
    LOG(f"emulated in {time.time() - t0:.0f} s", flush=True)
    FS._finish(rows, lambda v: 0)
    Path(a.out).write_text(json.dumps({"model": a.model, "tokens": len(ids), "variants": rows},
                                      indent=1))
    for lab, r in rows.items():
        print(f"{lab:40s} ppl {r['ppl']:8.4f} (+-{100 * r['se_int8']:.2f}%)  dKL "
              f"{100 * r['dkl']:+.3f}% (+-{100 * r['se_dkl']:.3f})  KL {r['kl_float']:.4f}  "
              f"top1/int8 {r['agree_int8']:.3f}  flush {r['flush']}")


if __name__ == "__main__":
    sys.exit(main())
