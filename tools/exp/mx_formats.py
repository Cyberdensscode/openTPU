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
  activations a8      today: int8, fp32 per 128 (QACT; K per token and 128-block, QST)
            a8x128    int8, fp32 per row, 2^-d per 128
            a8x32     int8, fp32 per row, 2^-d per 32 (OCP MXINT8's granularity; the ACT
                      scale word holds four 8-bit offsets)
            With a8x*, K is quantized as QST writes the K cache: per token and head, one fp32
            and a 2^-d per 128-block (docs/isa.md QST; d = 128: one fp32 a token, RSC).

A variant is "WEIGHTS/HEAD/ACTS[/flush]": the layers' weights, the LM head's, the activations
(every MM input, the head's, q, P and K as the attention reads them; V stays int8 per token),
and with /flush the window's drops emulated exactly (per 16 elements; zero blocks never drop).
WEIGHTS is a format, then optionally kind[@a-b]=format overrides of formats.py's kinds (attn,
mlp: the default of gateup and down, conv, delta, experts, ple; a..b checkpoint layers; a
ranged rule wins over an unranged one): "int8x,mlp@4-29=nvfp4r". A MoE's router is int8 (int8x
with a8x activations) in every variant. A label can be given as "LABEL:VARIANT".

The families: formats_scan's (Qwen3 and the Llama-likes, LFM2, Qwen3.5 with its DeltaNet) and
their MoE models (LFM2-MoE, Qwen3.5-MoE: the experts the tokens route to, each variant routing
on its own activations; the shared expert is the MLP kind), and Gemma 4 (E2B / E4B with their
per-layer embeddings, the 26B-A4B's MoE): tools/gemma4_quant_eval.py's emulation with every
variant run layer after layer together. Every variant's weights of a layer are quantized once
per format, in MX_PROCS worker processes (row pieces: the quantizers are row-wise).

    PYTHONPATH=. python3 tools/exp/mx_formats.py run MODEL OUT.json [--tokens 2000]
                         [--ple FMT] VARIANT...
        float and int8 (int8 weights, head and a8: today's) are every run's references;
        --ple: Gemma 4's PLE table in the quantized variants (the int8 reference: int8)
    PYTHONPATH=. python3 tools/exp/mx_formats.py check
        tiny random models: the MoE and Gemma 4 paths with today's formats against the families'
        emulated_logits / gemma4_quant_eval.emulate, and the K quantizer
    PYTHONPATH=. python3 tools/exp/mx_formats.py report OUT.json... [--pairs A,B ...]
        dKL (SE) of every row, paired differences, and each kind's bytes a token

The study ran against formats_scan.py, opentpu/quant.py and the families' emulations of the MXU
before MX (b370889d's, the "int8" / "fp4" reference rows as today's): run it from a checkout of
that commit (mx-acc's runs: ff4072ea..ddd32f43). Since the MX packers, formats_scan's own int8 /
int4 / fp4 / nvfp4 are the MX formats (int8x, int4x, fp4x6, nvfp4r with a8x32 activations,
device-faithful ranges); tests/test_quant.py holds quant.py's quantizers to this file's w_int8x,
w_fp4x(a, 6) and w_nvfp4(a, per_row=True), which still import from here.
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import formats_scan as FS  # noqa: E402
from opentpu import qcache as QC  # noqa: E402
from opentpu import quant as Q  # noqa: E402
from opentpu.kernels import gather as GA  # noqa: E402
from opentpu.llm import gemma4 as G  # noqa: E402
from opentpu.llm import lfm2 as L2  # noqa: E402
from opentpu.llm import load_spec, model_dir  # noqa: E402
from opentpu.llm import moe as MO  # noqa: E402
from opentpu.llm import qwen3 as Q3  # noqa: E402
from opentpu.llm import qwen35 as Q35  # noqa: E402
from opentpu.llm.qwen3 import _fake_q, _v_parts, rope_tables  # noqa: E402

G16 = 16                                    # the drop test's granularity (NVFP4's sub-block)
WIN = {"int8x": 16, "fp4x6": 11, "fp4x4": 13, "nvfp4r": 13, "mxfp4": 16}
TODAY = ("int8", "fp4", "int4")             # today's weight formats (formats_scan.dequant's)
NPROC = int(os.environ.get("MX_PROCS", "0"))    # quantizer worker processes (0: in line)
JOB = 1 << 22                               # elements a worker job quantizes (a matrix's rows)
PIECE = 1 << 15                             # elements a quantizer call takes (in-cache temps)
LOG = print
LAYERS = int(os.environ.get("MX_LAYERS", "0"))  # a smoke test's layers (0: all; then the head)
_POOL: ProcessPoolExecutor | None = None


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


def take(h, t):
    """Rows t of activations h (an XAct keeps its rows' offsets)."""
    v = np.asarray(h).view(np.ndarray)[t]
    dg = getattr(h, "dg", None)
    return XAct(v, dg[t]) if dg is not None else v


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

    def __mul__(self, s):                   # a scalar factor (exact in float64 for every format)
        return WX(self.deq * s, self.dw, self.W, self.flush)


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


class ActQ:
    """An MX activation quantizer as formats_scan's layers call it, fq(x, D): every MM input
    a8x(x, g), K as QST writes the K cache, a8x(K, 128) (one fp32 per token and head, 2^-d per
    128-block). The layers' one 3-D input is K ([T, n_kv, d]: q, P and the MM inputs are 2-D)."""

    def __init__(self, g):
        self.g = g

    def __call__(self, x, D=128):
        return a8x(x, 128) if np.ndim(x) == 3 else a8x(x, self.g)


# ------------------------------------------------------------------ weights
def _int8x(a):
    N, K = a.shape
    xb = a.reshape(N, K // 128, 128)
    ab = np.abs(xb).max(-1)
    ar = ab.max(-1, keepdims=True)
    d = _row_rel(ab, ar)
    s = (np.where(ar == 0, 1.0, ar / 127) * np.exp2(-d))[..., None]
    q = np.clip(np.rint(xb / s), -127, 127) * s
    return q.reshape(N, K), _expand(d, 128)


def _fp4x(a, mbits):
    """E2M1, sub-block scale Sw[n] * 2^-d[n, k] * m[n, k, b] (m in 1 .. 2^mbits - 1), the
    multiplier picked per sub-block by squared error over formats' search grid."""
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
    return best_q.reshape(N, K), _expand(d, 128)


def _nvfp4r(a):
    """NVFP4 with an fp32 per row (E2M1 per 16, E4M3 scale, search on). dw: the E4M3
    exponent's offset below the row's largest (subnormals at -6)."""
    N, K = a.shape
    ts = np.abs(a).max(-1, keepdims=True) / (Q.E4M3_MAX * Q.E2M1_MAX)
    ts = np.where(ts == 0, 1.0, ts)
    xb = a.reshape(N, K // 16, 16) / ts[..., None]
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
    return (q * ts[..., None]).reshape(N, K), np.where(live, emax - ex, 0)


def w_int8x(a) -> WX:
    """int8x as a WX (values, offsets per 16, window): tests/test_quant.py's reference for
    opentpu.quant's int8."""
    q, dw = _int8x(np.asarray(a, np.float64))
    return WX(q, dw, WIN["int8x"])


def w_fp4x(a, mbits: int) -> WX:
    """fp4x4 / fp4x6 as a WX (tests/test_quant.py's reference for opentpu.quant's fp4)."""
    q, dw = _fp4x(np.asarray(a, np.float64), mbits)
    return WX(q, dw, WIN[f"fp4x{mbits}"])


def w_nvfp4(a, per_row: bool = True) -> WX:
    """NVFP4 with an fp32 per row as a WX (tests/test_quant.py's reference for opentpu.quant's
    nvfp4); only the per-row form is the MXU's."""
    if not per_row:
        raise ValueError("w_nvfp4: the MXU's NVFP4 has an fp32 per row (per_row=True)")
    q, dw = _nvfp4r(np.asarray(a, np.float64))
    return WX(q, dw, WIN["nvfp4r"])


def _mxfp4(a):
    N, K = a.shape
    deq = Q.quantize(a.astype(np.float32), "mxfp4").deq.astype(np.float64)
    xb = a.reshape(N, K // 32, 32)
    amax = np.abs(xb).max(-1)
    with np.errstate(divide="ignore"):
        e = np.floor(np.log2(np.maximum(amax, 2.0 ** -127)))    # the OCP rule's exponent - 2
    live = amax > 0
    emax = np.where(live, e, -999).max(-1, keepdims=True)
    return deq, _expand(np.where(live, emax - e, 0), 32)


def _today(a, fmt, D):
    """formats_scan.dequant's int8 (_fake_q) and 4-bit (the image's rows and scale words,
    dequantized as the MXU does), without the image cache: (deq, None)."""
    if fmt == "int8":
        return _fake_q(np.asarray(a, np.float64), D), None
    rows, words = QC.quantize_rows(np.asarray(a, np.float32), fmt, D)
    return Q.dequantize_w4(rows[:, :a.shape[1] // 2], words, fmt, D).astype(np.float64), None


def _qrows(a, fmt, D):
    """(deq float64, dw int16 or None) of rows a [n, K] (fp32) in format fmt, PIECE elements'
    rows at a time (the quantizers are row-wise: the same result as the whole matrix)."""
    N, K = a.shape
    r = max(1, PIECE // K)
    out = []
    for i in range(0, N, r):
        b = a[i:i + r]
        if fmt in TODAY:
            out.append(_today(b, fmt, D))
            continue
        b = np.asarray(b, np.float64)
        q, dw = (_int8x(b) if fmt == "int8x" else _fp4x(b, int(fmt[-1])) if fmt in
                 ("fp4x6", "fp4x4") else _nvfp4r(b) if fmt == "nvfp4r" else _mxfp4(b))
        out.append((q, dw.astype(np.int16)))
    deq = np.concatenate([o[0] for o in out])
    return deq, None if out[0][1] is None else np.concatenate([o[1] for o in out])


class _Pending:
    """A weight being quantized (dequant_async): get() -> WX."""

    def __init__(self, parts, fmt, flush):
        self.parts, self.fmt, self.flush = parts, fmt, flush

    def get(self) -> WX:
        ps = [p.result() if hasattr(p, "result") else p for p in self.parts]
        deq = np.concatenate([p[0] for p in ps]) if len(ps) > 1 else ps[0][0]
        dw = None if ps[0][1] is None else (np.concatenate([p[1] for p in ps]) if len(ps) > 1
                                            else ps[0][1])
        return WX(deq, dw, WIN.get(self.fmt), self.flush and dw is not None)


def dequant_async(a, fmt, D=128) -> _Pending:
    """dequant, its row jobs queued on the worker processes (when there are, and a is large)."""
    if fmt is None:
        return _Pending([(np.asarray(a, np.float64), None)], None, False)
    flush = fmt.endswith("+f")
    fmt = fmt[:-2] if flush else fmt
    if fmt not in TODAY and fmt not in WIN:
        raise ValueError(f"weight format {fmt!r}")
    a = np.asarray(a, np.float32)
    if fmt in WIN and a.shape[1] % 128:
        raise ValueError(f"{fmt}: K {a.shape[1]} is not whole 128-blocks (pad it first)")
    N, K = a.shape
    if _POOL is None or a.size < (1 << 20):
        return _Pending([_qrows(a, fmt, D)], fmt, flush)
    r = max(1, JOB // K)
    return _Pending([_POOL.submit(_qrows, a[i:i + r], fmt, D) for i in range(0, N, r)], fmt,
                    flush)


def dequant(a, fmt, D=128) -> WX:
    """A weight as the device holds it (fmt None: float; "+f": the window's drops on)."""
    return dequant_async(a, fmt, D).get()


def pool(n: int = NPROC) -> None:
    """Start the quantizer's worker processes (fork: before the weights are opened)."""
    global _POOL
    if n > 0 and _POOL is None:
        _POOL = ProcessPoolExecutor(n, mp_context=mp.get_context("fork"))


# ------------------------------------------------------------------ variants
SUB = {"gateup": "mlp", "down": "mlp"}      # a kind whose default is another kind's format


class Variant:
    """A variant "WEIGHTS/HEAD/ACTS[/flush]" (the module's docstring), None: float (no
    quantization at all). wf(kind, layer) its weight formats (layer None: unranged rules only);
    fq the MM inputs' quantizer as formats_scan's layers call it; act / kq / pq the MM inputs',
    K's and P's for this file's own layers."""

    def __init__(self, text: str | None):
        self.text = text
        self.none = text is None
        if self.none:
            self.fq = FS._ident
            return
        p = text.split("/")
        if len(p) < 3:
            raise ValueError(f"variant {text!r}: WEIGHTS/HEAD/ACTS[/flush]")
        self.flush = len(p) > 3 and p[3] == "flush"
        base, *over = p[0].split(",")
        self.base, self.head, self.acts = base, p[1], p[2]
        self.rules = []
        for o in over:
            key, _, fmt = o.partition("=")
            kind, at, span = key.partition("@")
            lo, dash, hi = span.partition("-")
            self.rules.append((kind, int(lo) if at else 0, int(hi if dash else lo) if at
                               else 1 << 30, bool(at), fmt))
        self.mx = self.acts != "a8"
        if self.mx:
            g = {"a8x128": 128, "a8x32": 32}[self.acts]
            self.fq = ActQ(g)
            self.act = lambda x: a8x(x, g)
            self.kq = lambda k: a8x(k, 128)
            self.pq = lambda x: a8x(x, g)
        else:
            self.fq = _fake_q
            self.act = self.kq = self.pq = lambda x: _fake_q(np.asarray(x, np.float64))

    def _pick(self, kind, layer, default):
        hit = [r for r in self.rules if r[0] == kind and
               (r[3] and layer is not None and r[1] <= layer <= r[2] or not r[3])]
        hit.sort(key=lambda r: not r[3])
        return hit[0][4] if hit else default

    def wf(self, kind: str, layer: int | None = 0) -> str | None:
        if self.none:
            return None
        if kind == "head":
            f = self.head
        elif kind == "router":
            f = "int8x" if self.mx else "int8"
        else:
            f = self._pick(SUB[kind], layer, self.base) if kind in SUB else self.base
            f = self._pick(kind, layer, f)
        return f + "+f" if self.flush and f in WIN else f


def variants_of(texts) -> dict:
    """{label: Variant}: float and int8 (today's) first, then "LABEL:VARIANT" or "VARIANT"."""
    out = {"float": Variant(None), "int8": Variant("int8/int8/a8")}
    for t in texts:
        lab, _, v = t.rpartition(":")
        out[lab or v] = Variant(v)
    return out


# ------------------------------------------------------------------ the dense families
def _attend(q, K, v, fq, D, qd, scale):
    """formats_scan._attend with V always int8 per token for a quantized variant (its check is
    `fq is _fake_q`) and plain arrays into the products."""
    T, nq, _ = q.shape
    G_, Dp = nq // K.shape[1], -(-T // D) * D
    quant = fq is not FS._ident
    V, vs = _v_parts(v) if quant else (v, np.ones(v.shape[:2]))
    K = np.asarray(K).view(np.ndarray)
    causal = np.tri(T, dtype=bool)
    o = np.zeros((T, nq, V.shape[2]))
    for hq in range(nq):
        qq = np.asarray(fq(q[:, hq] / scale, qd)).view(np.ndarray)
        sc = np.where(causal, qq @ K[:, hq // G_].T, -np.inf)
        pp = np.exp(sc - sc.max(1, keepdims=True))
        ppad = np.zeros((T, Dp))
        ppad[:, :T] = pp * vs[:, hq // G_]
        pq = np.asarray(fq(ppad, D)).view(np.ndarray)
        o[:, hq] = (pq[:, :T] @ V[:, hq // G_]) / pp.sum(1, keepdims=True)
    return o


def _is_moe(spec, i) -> bool:
    m = getattr(spec, "moe", None)
    return m is not None and i >= m.first


def _raw(W, n, e=None) -> np.ndarray:
    """Tensor n (expert e of a fused one) as fp32, not kept by W (qwen3.Weights keeps every
    tensor under 16 MB: a MoE's experts would fill the host)."""
    if hasattr(W, "_where"):                                    # qwen3.Weights
        import torch
        h, k, rows = W._where[n]
        t = (h.get_slice(k)[int(e)] if e is not None else h.get_tensor(k) if rows is None
             else h.get_slice(k)[rows[0]:rows[1]])
        return t.to(torch.float32).numpy()
    if e is not None and hasattr(W, "part"):                    # LazyWeights, gemma4.Weights
        return W.part(n, e)
    a = W[n]
    return np.asarray(a if e is None else a[e], np.float32)


def _expert_names(M, p, e):
    """(gate, up, down) of expert e of layer prefix p: names or (name, e, row slice)."""
    if M is L2:
        f = f"{p}feed_forward.experts.{e}."
        return (f + "w1.weight",), (f + "w3.weight",), (f + "w2.weight",)
    m = p + "mlp.experts."
    return (m + "gate_up_proj", e, 0), (m + "gate_up_proj", e, 1), (m + "down_proj", e)


def _read_expert(W, M, p, e):
    """Expert e's W_gate, W_up [F, H], W_down [H, F] (fp32)."""
    if M is L2:
        return [_raw(W, n[0]) for n in _expert_names(M, p, e)]
    gu = _raw(W, p + "mlp.experts.gate_up_proj", e)
    F_ = gu.shape[0] // 2
    return [gu[:F_], gu[F_:], _raw(W, p + "mlp.experts.down_proj", e)]


def _silu(v):
    return v / (1 + np.exp(-v))


def _experts(st: dict, read, act, D, T, H, log=None, shapes=None, layer=None) -> dict:
    """The routed experts' sums {label: [T, H]} of the variants' states st {label: (h [T, H],
    ids [T, k], wts [T, k], expert format, fq)}: each expert any variant routes to read once
    (read(e) -> [gate, up, down] fp32, down's columns whole D-blocks), quantized once per
    format on the workers (a window of experts ahead), its rows' products per variant."""
    out = {lab: np.zeros((T, H)) for lab in st}
    need: dict = {}
    for lab, (h, ids, wts, fmt, fq) in st.items():
        for e in np.unique(ids):
            need.setdefault(int(e), set()).add(fmt)
    order = sorted(need)
    ahead = max(2, 2 * max(NPROC, 1))
    queue: list = []

    def submit(e):
        mats = read(e)
        if shapes is not None and ("experts", layer) not in shapes:
            shapes[("experts", layer)] = {f"expert.{j}": m.shape for j, m in enumerate(mats)}
        queue.append((e, {f: [dequant_async(m, f, D) for m in mats] for f in need[e]}))

    nxt = 0
    t0 = time.time()
    for done in range(len(order)):
        while nxt < len(order) and nxt < done + ahead:
            submit(order[nxt])
            nxt += 1
        e, byf = queue.pop(0)
        for f, pend in byf.items():
            wg, wu, wd = [pnd.get() for pnd in pend]
            for lab, (h, ids, wts, fmt, fq) in st.items():
                if fmt != f:
                    continue
                t, j = np.nonzero(ids == e)
                if not len(t):
                    continue
                FLUSH["label"] = lab
                hs = take(h, t)
                u = act(hs @ wg.T) * (hs @ wu.T)
                u = np.pad(u, ((0, 0), (0, wd.shape[1] - u.shape[1])))
                out[lab][t] += wts[t, j][:, None] * (fq(u) @ wd.T)
        if log and (done + 1) % 64 == 0:
            log(f"  experts {done + 1}/{len(order)} ({time.time() - t0:.0f} s)")
    return out


def _route(lg, bias, mo):
    """MO.route per row: ids [T, k], weights [T, k]."""
    r = [MO.route(row, bias, mo) for row in lg]
    return np.array([a for a, _ in r]), np.array([b for _, b in r])


def emulate(spec, W, tokens, variants: dict, D=128, shapes=None, log=None):
    """formats_scan.emulate with each variant's own quantizers (Variant) and MoE layers:
    {label: (nll, top1, {ref: KL})}. shapes gets {(kind, layer): {name: (rows, cols)}}."""
    M = FS.family(spec)
    T = len(tokens)
    shapes = {} if shapes is None else shapes
    x0 = np.asarray(W["model.embed_tokens.weight"][list(tokens)], np.float64)
    xs = {lab: (_fake_q(x0, D) if not v.none and spec.embed == "int8" else x0)
          for lab, v in variants.items()}
    cs = [rope_tables(spec, p) for p in range(T)]
    c = np.stack([a for a, _ in cs]).astype(np.float64)[:, None, :]
    s = np.stack([b for _, b in cs]).astype(np.float64)[:, None, :]
    dense = replace(spec, moe=None) if getattr(spec, "moe", None) is not None else spec
    t0 = time.time()
    for i in range(LAYERS or spec.layers):
        held: dict = {}
        moe = _is_moe(spec, i)

        def w(n, fmt, a=None):
            if (n, fmt) not in held:
                if a is None and n not in W:        # a MoE layer's dense MLP (run as zeros)
                    held[n, fmt] = WX(np.zeros((128, spec.hidden) if ".w2." not in n and
                                               "down_proj" not in n else (spec.hidden, 128)))
                    return held[n, fmt]
                a = W[n] if a is None else a
                kind, layer = (("router", i) if n.endswith(("mlp.router", "feed_forward.gate"
                                                            ".weight")) else M.weight_kind(n))
                shapes.setdefault((kind, layer), {})[n] = a.shape
                held[n, fmt] = dequant(a, fmt, D)
            return held[n, fmt]

        st, xm = {}, {}
        for lab, v in variants.items():
            FLUSH["label"] = lab
            wl = (lambda n, v=v: w(n, v.wf(*M.weight_kind(n))))
            if not moe:
                xs[lab] = np.asarray(FS.LAYER[M](spec, W, i, xs[lab], wl, v.fq, c, s, D)).view(
                    np.ndarray)
                continue
            # the mixer: formats_scan's layer with the FFN's names absent (zeros: adds 0)
            x = np.asarray(FS.LAYER[M](dense, W, i, xs[lab], wl, v.fq, c, s, D)).view(np.ndarray)
            st[lab], xm[lab] = _moe_route(spec, M, W, i, x, v, w, D)
        if moe:
            p = f"model.layers.{i}."
            ys = _experts({lab: s_[:5] for lab, s_ in st.items()},
                          lambda e: _read_expert(W, M, p, e), _silu, D, T, spec.hidden, log,
                          shapes, i)
            for lab in variants:
                xs[lab] = xm[lab] + st[lab][5] + ys[lab]
        held.clear()
        if log:
            log(f"layer {i + 1}/{spec.layers} ({time.time() - t0:.0f} s)")
    hs = {lab: (FS._ident if v.none else v.fq)(FS._final(spec, W, xs[lab]), D)
          for lab, v in variants.items()}
    del xs
    head = "model.embed_tokens.weight" if spec.tied else "lm_head.weight"
    Wh = W[head]
    shapes[("head", 0)] = {head: Wh.shape}
    fmts = {lab: v.wf("head") for lab, v in variants.items()}
    FLUSH["label"] = "head"             # (the head's drops, every variant's together)
    return FS._head(hs, fmts, lambda: ((r0, Wh[r0:r0 + FS.HEAD_ROWS])
                                       for r0 in range(0, Wh.shape[0], FS.HEAD_ROWS)),
                    tokens, D, False, ("float", "int8"))


def _moe_route(spec, M, W, i, x, v, w, D):
    """A MoE layer's FFN input, router and shared expert for variant v on the rows x (after the
    mixer): ((h, ids, wts, expert format, fq, shared expert's output), x). LFM2: sigmoid
    scores + bias; Qwen3.5: the k largest logits' softmax, the shared expert (the MLP kind)
    times sigmoid of the router's last row (lfm2 / qwen35.emulated_logits)."""
    p = f"model.layers.{i}."
    mo = spec.moe
    fq = (lambda a: a) if v.none else (lambda a: v.fq(a, D))
    rf = v.wf("router", i)
    if M is L2:
        h = fq(L2._norm(x, W[p + "ffn_norm.weight"], spec.eps))
        f = p + "feed_forward."
        lg = np.asarray(h @ w(f + "gate.weight", rf).T)
        ids, wts = _route(lg, W[f + "expert_bias"], mo)
        y = np.zeros_like(x)
    else:
        h = fq(Q35._norm(x, 1 + np.asarray(W[p + "post_attention_layernorm.weight"],
                                           np.float64), spec.eps))
        mp = spec.mlp_prefix(p)
        gg = h @ w(mp + "gate_proj.weight", v.wf("gateup", i)).T
        u = h @ w(mp + "up_proj.weight", v.wf("gateup", i)).T
        y = fq(Q35._silu(gg) * u) @ w(mp + "down_proj.weight", v.wf("down", i)).T
        rn = p + "mlp.router"
        lg = np.asarray(h @ w(rn, rf, np.concatenate(
            [W[p + "mlp.gate.weight"], W[p + "mlp.shared_expert_gate.weight"]])).T)
        y = y / (1 + np.exp(-lg[:, mo.E:mo.E + 1]))
        ids, wts = _route(lg[:, :mo.E], None, mo)
    return (h, ids, wts, v.wf("experts", i), fq, y), x


# ------------------------------------------------------------------ Gemma 4
def _g4_kind(n):
    if ".experts." in n:
        return "experts"
    if ".self_attn." in n:
        return "attn"
    if ".mlp." in n:
        return "down" if ".down_proj" in n else "gateup"
    return "ple"


def emulate_g4(spec, W, tokens, variants: dict, D=128, ple_table="int8", shapes=None, log=None):
    """tools/gemma4_quant_eval.py's emulate (hidden rows) for every variant at once, layer after
    layer (each variant keeps its own K / V for the KV-shared layers), then formats_scan._head
    with the soft cap: {label: (nll, top1, {ref: KL})}. The PLE table: float for float, int8
    for the int8 reference, ple_table for the others (a gather's records: no MX format)."""
    T, H, P, L = len(tokens), spec.hidden, spec.ple_dim, spec.layers
    n_q, eps = spec.n_q, spec.eps
    shapes = {} if shapes is None else shapes
    uniq, inv = np.unique(np.asarray(tokens), return_inverse=True)
    head = "model.embed_tokens.weight" if spec.tied else "lm_head.weight"
    emb = np.asarray(G._rows(W, head, uniq.tolist()), np.float32)
    X = {lab: dequant(emb, v.wf("head"), D).deq[inv] * math.sqrt(H)
         for lab, v in variants.items()}
    del emb
    PLI = {}
    if P:
        cols = np.concatenate([np.arange(spec.src(i) * P, (spec.src(i) + 1) * P)
                               for i in range(L)])
        pe0 = np.asarray(G._rows(W, G.Weights.PLE, uniq.tolist()), np.float32)[:, cols] * \
            np.float32((P / 2) ** 0.5)
        tabs = {}
        wp = W["model.per_layer_model_projection.weight"].reshape(-1, P, H)[
            [spec.src(i) for i in range(L)]].reshape(-1, H)
        gpl = np.asarray(W["model.per_layer_projection_norm.weight"], np.float64) * 2 ** -0.5
        held = {}
        for lab, v in variants.items():
            tf = "none" if v.none else "int8" if lab == "int8" else ple_table
            if tf not in tabs:
                pe = pe0
                if tf != "none":
                    S = GA.record_blocks(-(-L * P // D), tf)
                    pe = GA.dequant_records(GA.pack_records(pe0, tf, D, S), tf, D, S)[:, :L * P]
                tabs[tf] = np.asarray(pe, np.float64)[inv]
            pf = v.wf("ple", None)
            if pf not in held:
                held[pf] = dequant(wp, pf, D) * H ** -0.5
                shapes[("ple", -1)] = {"model.per_layer_model_projection.weight": wp.shape}
            xq = X[lab] if v.none else v.act(X[lab])
            PLI[lab] = G._norm(np.asarray(xq @ held[pf].T).reshape(T, L, P), gpl, eps) + \
                tabs[tf].reshape(T, L, P)
        del pe0, wp, tabs, held
    KV = {lab: {} for lab in variants}
    ii = np.arange(T)[:, None]
    ident = lambda a: np.asarray(a, np.float64)                     # noqa: E731
    t0 = time.time()
    for i in range(LAYERS or L):
        p = f"model.layers.{spec.src(i)}."
        li = spec.src(i)
        a = p + "self_attn."
        kind, d, n_kv = spec.kinds[i], spec.hd(i), spec.kvh(i)
        Gq = n_q // n_kv
        tab = [G.rope_tables(spec, pos, kind) for pos in range(T)]
        c = np.stack([t[0] for t in tab])[:, None, :]
        s_ = np.stack([t[1] for t in tab])[:, None, :]
        held: dict = {}

        def wq(n, fmt, arr=None):
            if (n, fmt) not in held:
                x = np.asarray(W[n] if arr is None else arr, np.float32)
                x = np.pad(x, ((0, 0), (0, -x.shape[1] % D)))        # whole D-blocks
                shapes.setdefault(("router" if ".router." in n else _g4_kind(n), li),
                                  {})[n] = x.shape
                held[n, fmt] = dequant(x, fmt, D)
            return held[n, fmt]

        st, mid = {}, {}
        for lab, v in variants.items():
            FLUSH["label"] = lab
            fq = ident if v.none else v.act
            fk = ident if v.none else v.kq
            fp = ident if v.none else v.pq
            x = X[lab]

            def pw(n, v=v):
                return wq(n, v.wf(_g4_kind(n), li))

            h = fq(G._norm(x, W[p + "input_layernorm.weight"], eps))
            q = np.asarray(h @ pw(a + "q_proj.weight").T).reshape(T, n_q, d)
            q = np.asarray(fq(G._rot(G._norm(q, W[a + "q_norm.weight"], eps), c, s_, d // 2)))
            if spec.kv_src[i] == i:
                k = np.asarray(h @ pw(a + "k_proj.weight").T).reshape(T, n_kv, d)
                vv = k if spec.kv_same(i) else np.asarray(
                    h @ pw(a + "v_proj.weight").T).reshape(T, n_kv, d)
                KV[lab][i] = (np.asarray(fk(G._rot(G._norm(k, W[a + "k_norm.weight"], eps), c,
                                                   s_, d // 2))),
                              _v_parts(G._norm(vv, None, eps)) if not v.none else
                              (ident(G._norm(vv, None, eps)), np.ones((T, n_kv))))
            Kh, (Vq, Vs) = KV[lab][spec.kv_src[i]]
            o = np.zeros((T, n_q, d))
            for hq in range(n_q):
                sc = q[:, hq] @ Kh[:, hq // Gq].T
                vs = Vs[:, hq // Gq]
                if kind == G.SLIDE:
                    Wn = -(-spec.window // D) * D
                    idx = np.maximum(0, ii[:, 0] + 1 - spec.window)[:, None] + \
                        np.arange(Wn)[None, :]
                    ok = idx <= ii
                    scp = np.concatenate([sc, np.full((T, Wn), -np.inf)], 1)
                    sb = np.where(ok, np.take_along_axis(scp, idx, 1), -np.inf)
                    pp = np.exp(sb - sb.max(1, keepdims=True))
                    pf = np.zeros((T, T + Wn))
                    np.put_along_axis(pf, idx, np.asarray(fp(pp * np.pad(vs, (0, Wn))[idx])), 1)
                    pf = pf[:, :T]
                else:
                    Tp = -(-T // D) * D
                    sb = np.full((T, Tp), -np.inf)
                    sb[:, :T] = np.where(np.arange(T)[None, :] <= ii, sc, -np.inf)
                    pp = np.exp(sb - sb.max(1, keepdims=True))
                    pf = np.asarray(fp(pp * np.pad(vs, (0, Tp - T))))[:, :T]
                o[:, hq] = (pf @ Vq[:, hq // Gq]) / pp.sum(1, keepdims=True)
            att = np.asarray(fq(o.reshape(T, -1)) @ pw(a + "o_proj.weight").T)
            x = x + G._norm(att, W[p + "post_attention_layernorm.weight"], eps)
            h = fq(G._norm(x, W[p + "pre_feedforward_layernorm.weight"], eps))
            wd = pw(p + "mlp.down_proj.weight")
            u = G._gelu(np.asarray(h @ pw(p + "mlp.gate_proj.weight").T)) * \
                np.asarray(h @ pw(p + "mlp.up_proj.weight").T)
            m = np.asarray(fq(np.pad(u, ((0, 0), (0, wd.shape[1] - u.shape[1])))) @ wd.T)
            mid[lab] = (x, m)
            if spec.experts:
                st[lab] = _g4_route(spec, W, p, li, x, v, wq, fq, D)
        if spec.experts:
            F, pes = spec.expert_ffn, np.asarray(W[p + "router.per_expert_scale"], np.float32)

            def read(e):
                gu = np.asarray(G._rows(W, p + "experts.gate_up_proj", [e])[0], np.float32) \
                    if isinstance(W, G.Weights) else _raw(W, p + "experts.gate_up_proj", e)
                dn = np.asarray(G._rows(W, p + "experts.down_proj", [e])[0], np.float32) \
                    if isinstance(W, G.Weights) else _raw(W, p + "experts.down_proj", e)
                dn = dn * pes[e]
                return [gu[:F], gu[F:], np.pad(dn, ((0, 0), (0, -dn.shape[1] % D)))]
            ys = _experts(st, read, G._gelu, D, T, H, log, shapes, li)
        for lab, v in variants.items():
            FLUSH["label"] = lab
            fq = ident if v.none else v.act
            x, m = mid[lab]
            if spec.experts:
                m = G._norm(m, W[p + "post_feedforward_layernorm_1.weight"], eps) + \
                    G._norm(ys[lab], W[p + "post_feedforward_layernorm_2.weight"], eps)
            x = x + G._norm(m, W[p + "post_feedforward_layernorm.weight"], eps)
            if P:
                g = G._gelu(np.asarray(fq(x) @ wq(p + "per_layer_input_gate.weight",
                                                  v.wf("ple", li)).T)) * PLI[lab][:, i]
                y = np.asarray(fq(g) @ wq(p + "per_layer_projection.weight", v.wf("ple", li)).T)
                x = x + G._norm(y, W[p + "post_per_layer_input_norm.weight"], eps)
            X[lab] = x * np.asarray(W[p + "layer_scalar"], np.float64)
        held.clear()
        if log:
            log(f"layer {i + 1}/{L} ({time.time() - t0:.0f} s)")
    hs = {lab: (FS._ident if v.none else v.fq)(G._norm(X[lab], W["model.norm.weight"], eps), D)
          for lab, v in variants.items()}
    del X, PLI, KV
    shapes[("head", 0)] = {head: (spec.vocab, H)}

    def chunks():
        t = G._Rows(*W._files[head]) if isinstance(W, G.Weights) else W[head]
        for r0 in range(0, spec.vocab, FS.HEAD_ROWS):
            yield r0, t[r0:r0 + FS.HEAD_ROWS]
    FLUSH["label"] = "head"
    return FS._head(hs, {lab: v.wf("head") for lab, v in variants.items()}, chunks, tokens, D,
                    False, ("float", "int8"), cap=spec.softcap)


def _g4_route(spec, W, p, li, x, v, wq, fq, D):
    """Gemma 4's MoE router for variant v on the rows x (gemma4_quant_eval._moe): the unit norm
    quantized into the router (int8 / int8x, router.scale / sqrt(H) folded into its columns),
    the top k's softmax, the experts' input: the unit norm times pre_feedforward_layernorm_2's
    gain, quantized. -> (xe, ids, wts, expert format, fq)."""
    r = p + "router."
    xn = G._norm(x, None, spec.eps)
    xs = fq(xn)
    wr = np.asarray(W[r + "proj.weight"], np.float32) * (
        np.asarray(W[r + "scale"], np.float32) * np.float32(spec.hidden ** -0.5))[None, :]
    lg = np.asarray(xs @ wq(r + "proj.weight", v.wf("router", li), wr).T)
    top = np.argsort(-lg, axis=1, kind="stable")[:, :spec.top_k]
    w = np.take_along_axis(lg, top, 1)
    w = np.exp(w - w[:, :1])
    w /= w.sum(1, keepdims=True)
    xe = fq(xn * np.asarray(W[p + "pre_feedforward_layernorm_2.weight"], np.float64)[None, :])
    return (xe, top, w, v.wf("experts", li), fq)


# ------------------------------------------------------------------ bytes
def row_bytes(K: int, fmt: str | None) -> float:
    """DRAM bytes of a weight row of K elements: data and its scale row (docs/isa.md; today's
    formats as formats_scan.weight_bytes)."""
    f = (fmt or "none").replace("+f", "")
    KB = -(-K // 128)
    if f in TODAY:
        return Q.row_bytes(KB * 128, f, 128) + 4 * KB
    return {"int8x": KB * 128 + 4 + 4 * -(-KB // 4), "fp4x6": KB * 64 + 8 + 4 * KB,
            "fp4x4": KB * 64 + 8 + 4 * KB, "nvfp4r": KB * 64 + 16 + 8 * KB,
            "mxfp4": KB * 64 + 8 + 4 * KB, "none": KB * 512}[f]


def bytes_by_kind(shapes: dict, v: Variant, per_token: dict) -> dict:
    """{kind: bytes a decode token reads} of variant v: each matrix's rows times its row bytes;
    a MoE's experts per_token["experts"] of them a layer (the routed k), the head once."""
    out: dict = {}
    for (kind, layer), mats in shapes.items():
        f = v.wf(kind, None if layer < 0 else layer) if kind != "head" else v.wf("head")
        n = sum(r * row_bytes(cols, f) for r, cols in mats.values())
        if kind == "experts":
            n *= per_token.get("experts", 1)
        out[kind] = out.get(kind, 0) + n
    return out


# ------------------------------------------------------------------ run / report / check
def finish(rows: dict) -> None:
    """formats_scan._finish without bytes: top-1 agreement, SE of NLL against int8, dKL and its
    SE (paired over the tokens)."""
    ref8, refF = rows["int8"]["top1"], rows["float"]["top1"]
    n8, k8 = np.asarray(rows["int8"]["nll_tok"]), np.asarray(rows["int8"]["kl_tok"])
    for r in rows.values():
        t = r.pop("top1")
        r["agree_int8"] = float(np.mean(np.equal(t, ref8)))
        r["agree_float"] = float(np.mean(np.equal(t, refF)))
        dn = np.asarray(r["nll_tok"]) - n8
        r["se_int8"] = float(dn.std(ddof=1) / math.sqrt(len(dn)))
        dk = np.asarray(r["kl_tok"]) - k8
        r["dkl"], r["se_dkl"] = float(dk.mean()), float(dk.std(ddof=1) / math.sqrt(len(dk)))


def run(model, out, texts, n_tok=2000, ple_table="int8", ids_file=None) -> None:
    path = model_dir(model)
    spec = load_spec(path)
    g4 = isinstance(spec, G.Spec)
    var = variants_of(texts)
    t0 = time.time()

    def log(m):
        print(f"[{time.time() - t0:6.0f} s] {m}", flush=True)

    W = G.load_weights(path) if g4 else Q3.load_weights(path)
    ids = FS.text_ids(path, n_tok, ids_file)
    shapes: dict = {}
    log(f"{model}: {len(ids)} tokens, {len(var)} variants, {NPROC} quantizer processes")
    res = (emulate_g4(spec, W, ids, var, ple_table=ple_table, shapes=shapes, log=log) if g4
           else emulate(spec, W, ids, var, shapes=shapes, log=log))
    rows = {}
    k = (spec.top_k if g4 else getattr(getattr(spec, "moe", None), "k", 1)) or 1
    for lab, (nll, top1, kl) in res.items():
        rows[lab] = {"variant": var[lab].text, "nll": float(nll.mean()),
                     "ppl": float(np.exp(nll.mean())), "top1": top1.tolist(),
                     "nll_tok": np.round(nll, 5).tolist(),
                     "kl_float": float(kl["float"].mean()),
                     "kl_tok": np.round(kl["float"], 7).tolist(),
                     "kl_int8": float(kl["int8"].mean()), "flush": FLUSH.get(lab),
                     "bytes": None if var[lab].none else
                     bytes_by_kind(shapes, var[lab], {"experts": k})}
    finish(rows)
    log(f"emulated in {time.time() - t0:.0f} s")
    rows["int8"]["flush_head"] = FLUSH.get("head")
    sh = {f"{kd}@{ly}": {n: list(s) for n, s in m.items()} for (kd, ly), m in shapes.items()}
    Path(out).write_text(json.dumps({"model": model, "tokens": len(ids), "ple_table": ple_table,
                                     "experts_per_token": k, "shapes": sh, "variants": rows},
                                    indent=1))
    for lab, r in rows.items():
        tot = sum(r["bytes"].values()) / 2 ** 20 if r["bytes"] else 0
        print(f"{lab:28s} ppl {r['ppl']:8.4f}  dKL {100 * r['dkl']:+.3f}% "
              f"(+-{100 * r['se_dkl']:.3f})  top1/int8 {r['agree_int8']:.3f}  "
              f"{tot:8.1f} MiB/token  flush {r['flush']}  {r['variant']}")


def paired(a: dict, b: dict) -> tuple:
    """(mean, SE) of dKL(a) - dKL(b) in points, paired over the tokens."""
    d = np.asarray(a["kl_tok"]) - np.asarray(b["kl_tok"])
    return 100 * d.mean(), 100 * d.std(ddof=1) / math.sqrt(len(d))


def report(files, pairs) -> None:
    for f in files:
        z = json.loads(Path(f).read_text())
        rows = z["variants"]
        print(f"== {z['model']} ({z['tokens']} tokens)")
        for lab, r in rows.items():
            by = r.get("bytes") or {}
            tot = sum(by.values())
            share = " ".join(f"{k} {100 * b / tot:.1f}%" for k, b in sorted(by.items())) \
                if tot else ""
            print(f"  {lab:28s} dKL {100 * r['dkl']:+7.3f} ({100 * r['se_dkl']:.3f})  "
                  f"ppl {r['ppl']:8.3f}  {tot / 2 ** 20:8.1f} MiB  [{share}]")
        for pr in pairs:
            a, b = pr.split(",")
            if a in rows and b in rows:
                m, se = paired(rows[a], rows[b])
                print(f"  {a} - {b}: {m:+.3f} ({se:.3f})")


def _tiny_lfm2_moe():
    import torch
    import transformers
    torch.manual_seed(0)
    E, K_, F = 8, 2, 256
    kinds = ("conv", "attn", "conv", "attn", "conv")
    hc = transformers.Lfm2MoeConfig(
        hidden_size=256, num_hidden_layers=len(kinds), num_attention_heads=4,
        num_key_value_heads=2, intermediate_size=512, moe_intermediate_size=F, num_experts=E,
        num_experts_per_tok=K_, num_dense_layers=1, vocab_size=1000, norm_eps=1e-5,
        layer_types=["full_attention" if k == "attn" else "conv" for k in kinds],
        conv_L_cache=3, conv_bias=False, tie_word_embeddings=True,
        max_position_embeddings=4096, rope_parameters={"rope_type": "default", "rope_theta": 1e6})
    m = transformers.Lfm2MoeForCausalLM(hc).float().eval()
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "norm" in n:
                p.copy_(1 + 0.1 * torch.randn_like(p))
            elif "experts" in n or "gate" in n:
                p.copy_(0.05 * torch.randn_like(p))
        for mod in m.modules():
            if hasattr(mod, "expert_bias"):
                mod.expert_bias.copy_(0.02 * torch.randn_like(mod.expert_bias))
    W = {}
    for k, v in m.state_dict().items():
        v = v.float().numpy()
        if k.endswith("experts.gate_up_proj"):
            p = k[:-len("gate_up_proj")]
            for e in range(E):
                W[f"{p}{e}.w1.weight"], W[f"{p}{e}.w3.weight"] = v[e, :F], v[e, F:]
        elif k.endswith("experts.down_proj"):
            p = k[:-len("down_proj")]
            for e in range(E):
                W[f"{p}{e}.w2.weight"] = v[e]
        else:
            W[k] = v
    spec = L2.Spec(256, kinds, 4, 2, 64, 512, 1000, moe=MO.MoESpec(E=E, k=K_, ffn=F, first=1))
    return W, spec


def _tiny_qwen35_moe():
    import torch
    import transformers
    torch.manual_seed(0)
    E, K_, F = 8, 2, 256
    kinds = ("linear", "linear", "attn", "linear", "linear", "attn")
    hc = transformers.Qwen3_5MoeTextConfig(
        hidden_size=256, num_hidden_layers=len(kinds), num_attention_heads=8,
        num_key_value_heads=2, head_dim=256, vocab_size=1000,
        layer_types=["full_attention" if k == "attn" else "linear_attention" for k in kinds],
        linear_num_key_heads=4, linear_num_value_heads=8, linear_key_head_dim=128,
        linear_value_head_dim=128, linear_conv_kernel_dim=4, tie_word_embeddings=True,
        max_position_embeddings=4096, rms_norm_eps=1e-6, num_experts=E,
        num_experts_per_tok=K_, moe_intermediate_size=F, shared_expert_intermediate_size=F,
        rope_parameters={"rope_type": "default", "rope_theta": 1e7, "partial_rotary_factor": 0.25})
    m = transformers.Qwen3_5MoeForCausalLM(hc).float().eval()
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "norm" in n:
                p.copy_((1.0 if n.endswith("linear_attn.norm.weight") else 0.0)
                        + 0.1 * torch.randn_like(p))
            elif "experts" in n or "gate" in n:
                p.copy_(0.05 * torch.randn_like(p))
    W = {k: v.float().numpy() for k, v in m.state_dict().items()}
    spec = Q35.Spec(256, kinds, 8, 2, 256, 64, 8, 128, 128, F, 1000, lin_kheads=4,
                    moe=MO.MoESpec(E=E, k=K_, ffn=F, rule="softmax", shared=F))
    return W, spec


def _tiny_gemma(tmp, moe: bool):
    import torch
    import transformers
    torch.manual_seed(0)
    S_, F_ = "sliding_attention", "full_attention"
    if moe:
        kinds = (S_, S_, F_, S_, S_, F_)
        hc = transformers.Gemma4TextConfig(
            hidden_size=256, num_hidden_layers=len(kinds), num_attention_heads=4,
            num_key_value_heads=2, num_global_key_value_heads=1, attention_k_eq_v=True,
            head_dim=128, global_head_dim=256, intermediate_size=320, vocab_size=1000,
            hidden_size_per_layer_input=0, layer_types=list(kinds), num_kv_shared_layers=0,
            sliding_window=512, final_logit_softcapping=30.0, max_position_embeddings=4096,
            enable_moe_block=True, num_experts=8, top_k_experts=2, moe_intermediate_size=96)
    else:
        kinds = (S_, S_, F_, S_, S_, F_, S_, F_, F_)
        hc = transformers.Gemma4TextConfig(
            hidden_size=256, num_hidden_layers=len(kinds), num_attention_heads=8,
            num_key_value_heads=1, head_dim=128, global_head_dim=256, intermediate_size=512,
            vocab_size=1000, vocab_size_per_layer_input=1000, hidden_size_per_layer_input=128,
            layer_types=list(kinds), num_kv_shared_layers=3, use_double_wide_mlp=True,
            sliding_window=64, final_logit_softcapping=30.0, max_position_embeddings=4096)
    hc._attn_implementation = "eager"
    m = transformers.models.gemma4.modeling_gemma4.Gemma4ForCausalLM(hc).float().eval()
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "norm" in n or n.endswith("router.scale"):
                k = "k_norm" in n
                p.copy_((0.13 if k else 1.0) + (0.01 if k else 0.1) * torch.randn_like(p))
            elif n.endswith("per_expert_scale"):
                p.copy_(0.5 + torch.rand_like(p))
            elif "experts" in n or "router" in n:
                p.copy_((0.5 if "router.proj" in n else 0.05) * torch.randn_like(p))
        for n, b in m.named_buffers():
            if n.endswith("layer_scalar"):
                b.copy_(0.5 + torch.rand_like(b))
    W = {k: v.float().numpy() for k, v in m.state_dict().items()}
    d = Path(tmp) / ("moe" if moe else "ple")
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps(hc.to_dict()))
    return W, G.Spec.from_hf(d)


def check() -> None:
    """The MoE and Gemma 4 paths with today's formats against the families' references, on
    tiny random models; the K quantizer of ActQ."""
    import importlib.util
    import tempfile
    FS._DEQUANT = FS.dequant
    FS.dequant = dequant
    FS._attend = _attend
    rng = np.random.default_rng(0)
    worst = 0.0
    x = rng.standard_normal((5, 2, 256)) * np.exp(rng.standard_normal((5, 2, 256)))
    kq = ActQ(32)(x)
    assert np.array_equal(np.asarray(kq), np.asarray(a8x(x, 128))), "K: a8x(K, 128)"
    assert np.array_equal(np.asarray(ActQ(32)(x[:, 0])), np.asarray(a8x(x[:, 0], 32)))
    print("ActQ: K (3-D) a8x 128, the rest a8x 32")
    v = Variant("int8x,mlp@1-2=nvfp4r,down@2=fp4x6,experts=fp4x6/int8x/a8x32/flush")
    got = [v.wf("gateup", 0), v.wf("gateup", 1), v.wf("down", 2), v.wf("attn", 2),
           v.wf("experts", 3), v.wf("router", 3), v.wf("head"), v.wf("ple", None)]
    assert got == ["int8x+f", "nvfp4r+f", "fp4x6+f", "int8x+f", "fp4x6+f", "int8x+f",
                   "int8x+f", "int8x+f"], got
    print("Variant.wf:", got)
    for name, (W, spec) in (("lfm2-moe", _tiny_lfm2_moe()), ("qwen35-moe", _tiny_qwen35_moe())):
        M = FS.family(spec)
        toks = [int(t) for t in rng.integers(3, 1000, 140)]
        for wf, hf in (("int8", "int8"), ("fp4", "int8"), ("fp4", "fp4")):
            var = {"float": Variant(None), "int8": Variant("int8/int8/a8"),
                   "v": Variant(f"{wf}/{hf}/a8")}
            # the logits: _head with logits=True through emulate's own path
            lg = _emulate_logits(spec, W, toks, var)
            ref = M.emulated_logits(spec, W, toks, wformat=wf, head_format=hf)
            err = np.abs(lg["v"] - ref).max() / np.abs(ref).max()
            worst = max(worst, err)
            print(f"{name:10s} {wf}/{hf}: logits rel {err:.2e}")
            assert err < 1e-6, (name, wf, hf, err)
        mx = _emulate_logits(spec, W, toks, {"float": Variant(None), "int8": Variant(
            "int8/int8/a8"), "v": Variant("fp4x6,attn=nvfp4r/int8x/a8x32/flush")})
        print(f"{name:10s} MX runs: logits rel to int8 "
              f"{np.abs(mx['v'] - mx['int8']).max() / np.abs(mx['int8']).max():.2e}")
    s = importlib.util.spec_from_file_location("gemma4_quant_eval",
                                               ROOT / "tools/gemma4_quant_eval.py")
    E = importlib.util.module_from_spec(s)
    s.loader.exec_module(E)
    with tempfile.TemporaryDirectory() as tmp:
        for moe in (False, True):
            W, spec = _tiny_gemma(tmp, moe)
            toks = [int(t) for t in rng.integers(3, 1000, 140)]
            cases = [("int8", "int8", "", "int8/int8/a8"), ("fp4", "fp4", "", "fp4/fp4/a8"),
                     ("int8", "int8", "gateup@0-2=fp4" + (",experts=fp4" if moe else ""),
                      "int8,gateup@0-2=fp4" + (",experts=fp4" if moe else "") + "/int8/a8")]
            for wf, hf, fm, vt in cases:
                wmap = dict(kv.split("=") for kv in fm.split(",") if kv)
                ref = E.emulate(spec, W, toks, wformat=wf, hf=hf, ple_format="int8", wmap=wmap)
                var = {"float": Variant(None), "int8": Variant("int8/int8/a8"),
                       "v": Variant(vt)}
                lg = _emulate_logits(spec, W, toks, var, g4=True)
                err = np.abs(lg["v"] - ref).max() / np.abs(ref).max()
                worst = max(worst, err)
                print(f"gemma4 {'moe' if moe else 'ple'} {vt:40s} logits rel {err:.2e}")
                assert err < 1e-6, (moe, vt, err)
            mx = _emulate_logits(spec, W, toks, {"float": Variant(None), "int8": Variant(
                "int8/int8/a8"), "v": Variant("int8x,experts=nvfp4r/fp4x6/a8x32/flush")},
                g4=True)
            print(f"gemma4 {'moe' if moe else 'ple'} MX runs: logits rel to int8 "
                  f"{np.abs(mx['v'] - mx['int8']).max() / np.abs(mx['int8']).max():.2e}")
    print(f"OK (worst logits rel {worst:.2e})")


def _emulate_logits(spec, W, toks, var, g4=False) -> dict:
    """emulate / emulate_g4's logits [T, vocab] per label (the head's logits=True path; Gemma:
    before the soft cap), for check."""
    keep = FS._head

    def head(hs, fmts, chunks, tokens, D=128, logits=False, kl_to=(), cap=None):
        return keep(hs, fmts, chunks, tokens, D, True, ())
    FS._head = head
    try:
        return (emulate_g4(spec, W, toks, var, ple_table="int8") if g4 else
                emulate(spec, W, toks, var))
    finally:
        FS._head = keep


def main():
    argv = sys.argv[1:]
    if argv and argv[0] not in ("run", "check", "report", "-h", "--help"):
        argv = ["run"] + argv                   # (the first version's usage: MODEL OUT VARIANT..)
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("run")
    a.add_argument("model")
    a.add_argument("out")
    a.add_argument("variants", nargs="+")
    a.add_argument("--tokens", type=int, default=2000)
    a.add_argument("--ple", default="int8", help="Gemma 4: the PLE table of the quantized "
                   "variants (the int8 reference's: int8)")
    a.add_argument("--ids", help="the text's token ids (formats_scan --ids)")
    sub.add_parser("check")
    a = sub.add_parser("report")
    a.add_argument("files", nargs="+")
    a.add_argument("--pairs", nargs="*", default=[], help="A,B: dKL(A) - dKL(B), paired")
    a = ap.parse_args(argv)
    if a.cmd == "report":
        return report(a.files, a.pairs)
    if a.cmd == "check":
        return check()
    pool()
    FS._DEQUANT = FS.dequant
    FS.dequant = dequant                     # _head's weights
    FS._attend = _attend                     # the layers' attention
    run(a.model, a.out, a.variants, a.tokens, a.ple, a.ids)


if __name__ == "__main__":
    sys.exit(main())
