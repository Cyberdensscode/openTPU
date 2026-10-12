"""Gemma 4's quantization error, on long sequences: gemma4.emulated_logits (float64, the device's
quantization points and none of its rounding) over the whole sequence at once, with the points
switched one at a time and per-kind weight formats (docs/gemma4.md, "Long context").

    python tools/gemma4_quant_eval.py step MODEL REF.npz STEP WF HEAD [--formats attn=int8]
                                      [--quant act,kv,p]
        the top 5 at REF's prompt (ids0) + its first STEP greedy tokens (gen0), and the rank of
        REF's next token (hf_long-style npz: ids0, gen0)
    python tools/gemma4_quant_eval.py nll MODEL REF.npz WF HEAD OUT.npz [--formats ..] [--quant ..]
        the next-token NLL of the soft-capped logits over ids0: perplexity, and the argmax per
        position into OUT.npz (nll, top1)
    python tools/gemma4_quant_eval.py check TINY_DIR
        the batched emulation against emulated_logits (40 tokens, window 16; equal to float64
        rounding: _fake_q rounds a tie of exactly .5 as the device's quantizer does, whatever
        the summation order)

WF / HEAD: int8, fp4, int4 or none (float weights). The whole sequence runs at once: 900 tokens of
E2B take about 1 min in float, 10 min with fp4 layers (the 4-bit quantization's search).
26B-A4B (its MoE block, K = V global layers, no PLE): --formats experts=fp4 for 4-bit experts
(about 1.2 s of quantization an expert, 3840 of them: THREADS at a time; the 4-bit matrices go
through opentpu.qcache, so a study's later runs read them: OTPU_IMAGE_CACHE=<dir> for a cache of
its own, 13 GB for the 26B's fp4 experts).
"""
import argparse
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import numpy as np

from opentpu import isa as I
from opentpu import qcache as QC
from opentpu import quant as Q
from opentpu.kernels import gather as GA
from opentpu.llm import gemma4 as G
from opentpu.llm import moe as MO
from opentpu.llm.qwen3 import ATTN_BLOCK, _fake_k, _fake_q, _qk, _v_parts

THREADS = 4     # experts quantized at once (numpy's 4-bit search releases the GIL: 2.8x on 4)


def emulate(spec, W, tokens, D=128, wformat="int8", hf=None, ple_format="int8", rows=None,
            quant=None, wmap=None, on_rows=None, ckpt=None, hidden=False):
    """gemma4.emulated_logits over the whole sequence at once: the logits [len(rows), vocab]
    before the soft cap (rows: default all), or with on_rows each group of up to 64 rows handed
    to on_rows(first row, logits), or with hidden the rows the head multiplies [len(rows), H]
    (after the final norm and its quantization; tools/formats_scan.py scores many variants'
    rows in one pass over the head). wformat / hf: the layers' and the head's formats, "none" for
    float weights (the PLE table then float too); wmap: per-kind layer formats {"attn", "mlp"
    (or "down" / "gateup" of it), "ple": format}, a key "kind@a-b" for layers a..b only
    (checkpoint layers; it wins over "kind"). quant: the activation quantization points
    applied, a subset of {"act" (the matmul inputs: the QACT ranges of gemma4.emulated_logits),
    "kv" (K / V), "p" (P, per attention block of absolute positions)}; default all, none with
    wformat "none". ckpt: a file the residual is saved to after each layer and resumed from
    (the same tokens and formats; a model without KV-shared layers): a run killed at the
    memory floor loses at most a layer.

    A MoE block (26B-A4B; docs/offload.md section 11): the weightless RMSNorm of x quantized
    into the router (int8, router.scale / sqrt(H) folded into its columns; the top k of its
    logits, their softmax), and times pre_feedforward_layernorm_2's gain, quantized, into the
    experts (wmap "experts", else wformat; per_expert_scale folded into the down projection,
    the width padded with zeros to whole D-blocks), as moe.moe_ffn runs it (g_exp); _moe's
    docstring says what folding that gain into gate / up instead cost."""
    hf = hf or wformat
    none = wformat == "none"
    quant = (set() if none else {"act", "kv", "p"}) if quant is None else set(quant)
    ident = lambda v, d=D, n=None: np.asarray(v, np.float64)              # noqa: E731
    fq = _fake_q if "act" in quant else ident
    fp = _fake_q if "p" in quant else ident

    wmap = wmap or {}                   # per-kind layer formats: attn, mlp, ple (else wformat)

    def get(k, li, default):    # wmap[k] ("kind" or "kind@a-b": layers a..b only)
        for key, f in wmap.items():
            kind, _, span = key.partition("@")
            if kind == k and span:
                lo, _, hi = span.partition("-")
                if int(lo) <= li <= int(hi or lo):
                    return f
        return wmap.get(k, default)

    def fmt_of(n):              # attn; mlp, or down / gateup within it; ple
        li = int(n.split(".")[2])
        if ".experts." in n:
            return get("experts", li, wformat)
        if ".self_attn." in n:
            return get("attn", li, wformat)
        if ".mlp." in n:
            return get("down" if ".down_proj" in n else "gateup", li, get("mlp", li, wformat))
        return get("ple", li, wformat)

    def wq(n, fmt=wformat, a=None, cols=None):
        if n is not None and fmt == wformat and n.startswith("model.layers."):
            fmt = fmt_of(n)
        a = np.asarray(W[n] if a is None else a, np.float32)
        cols = cols or -(-a.shape[1] // D) * D               # whole D-blocks (MLP widths)
        a = np.pad(a, ((0, 0), (0, cols - a.shape[1])))
        if fmt == "none":
            return a.astype(np.float64)
        # an MXW (the 4-bit quantization through opentpu.qcache: the runs of a study share it)
        return Q.fake_w(a, fmt, D, QC.quantize_mxu(a, fmt, D) if fmt in QC.FORMATS else None)

    def padq(v, n, C):          # an activation [T, f] padded with zeros to n columns, quantized
        return fq(np.pad(v, ((0, 0), (0, n - v.shape[1]))), D, C)      # in ranges of C

    def mlp_width(f, fd):       # (the layer block's width, its down projection's chunks)
        if fd == "none":
            return -(-f // D) * D, None
        fp_ = G._ffn_pad(f, fd, D)
        return fp_, G._mlp_chunk(fp_, D, D if fd == "int8" else 2 * D)

    T, H, P, L = len(tokens), spec.hidden, spec.ple_dim, spec.layers
    n_q, eps = spec.n_q, spec.eps
    uniq, inv = np.unique(np.asarray(tokens), return_inverse=True)
    head = "model.embed_tokens.weight" if spec.tied else "lm_head.weight"
    er = np.asarray(G._rows(W, head, uniq.tolist()), np.float32)     # the device's gathers
    x = (er if hf == "none" else GA.gathered(er, hf, D)).astype(np.float64)[inv] * math.sqrt(H)
    pli = None
    if P:
        cols = np.concatenate([np.arange(spec.src(i) * P, (spec.src(i) + 1) * P)
                               for i in range(L)])
        pe = np.asarray(G._rows(W, G.Weights.PLE, uniq.tolist()), np.float32)[:, cols] * \
            np.float32((P / 2) ** 0.5)
        if not none:
            S = GA.record_blocks(-(-L * P // D), ple_format)
            pe = GA.dequant_records(GA.pack_records(pe, ple_format, D, S), ple_format, D,
                                    S)[:, :L * P]
        pe = np.asarray(pe, np.float64)[inv]
        wp = W["model.per_layer_model_projection.weight"].reshape(-1, P, H)[
            [spec.src(i) for i in range(L)]].reshape(-1, H)
        wpq = wq(None, wmap.get("ple", wformat), a=np.asarray(wp, np.float32) *
                 np.float32(H ** -0.5))                       # (folded, as the image's)
        gpl = np.asarray(W["model.per_layer_projection_norm.weight"], np.float64) * 2 ** -0.5
        pli = G._norm((fq(x) @ wpq.T).reshape(T, L, P), gpl, eps) + pe.reshape(T, L, P)
        del wp, wpq, pe
    ii = np.arange(T)[:, None]
    K, V = {}, {}
    key, start = None, 0
    if ckpt is not None and all(spec.kv_src[i] == i for i in range(L)):
        key = repr((list(tokens), wformat, hf, ple_format, sorted(quant), sorted(wmap.items())))
        try:
            with np.load(ckpt) as z:
                if str(z["key"]) == key:
                    start, x = int(z["layer"]) + 1, z["x"].copy()
                    print(f"  resumed after layer {start - 1}", flush=True)
        except (OSError, KeyError, ValueError):
            pass
    for i in range(start, L):
        p = f"model.layers.{spec.src(i)}."
        a = p + "self_attn."
        kind, d, n_kv = spec.kinds[i], spec.hd(i), spec.kvh(i)
        Gq = n_q // n_kv
        tab = [G.rope_tables(spec, pos, kind) for pos in range(T)]
        c = np.stack([t[0] for t in tab])[:, None, :]
        s_ = np.stack([t[1] for t in tab])[:, None, :]
        h = fq(G._norm(x, W[p + "input_layernorm.weight"], eps))
        q = (h @ wq(a + "q_proj.weight").T).reshape(T, n_q, d)
        q = G._rot(G._norm(q, W[a + "q_norm.weight"], eps), c, s_, d // 2)
        if spec.kv_src[i] == i:
            k = (h @ wq(a + "k_proj.weight").T).reshape(T, n_kv, d)
            v = k if spec.kv_same(i) else (h @ wq(a + "v_proj.weight").T).reshape(T, n_kv, d)
            k = G._rot(G._norm(k, W[a + "k_norm.weight"], eps), c, s_, d // 2)
            K[i] = _fake_k(k, D) if "kv" in quant else (ident(k), None)    # as the cache holds
            v = G._norm(v, None, eps)       # int8 values, per-token scales (folded into P)
            V[i] = _v_parts(v) if "kv" in quant else (ident(v), np.ones((T, n_kv)))
        (Kh, dK), (Vq, Vs) = K[spec.kv_src[i]], V[spec.kv_src[i]]
        ok = np.arange(T)[None, :] <= ii                        # causal, and the window
        if kind == G.SLIDE:
            ok &= np.arange(T)[None, :] >= ii + 1 - spec.window
        Tb = -(-T // ATTN_BLOCK) * ATTN_BLOCK                  # P's attention blocks
        o = np.zeros((T, n_q, d))
        for hq in range(n_q):
            g = hq // Gq
            qq = fq(q[:, hq])
            sc = np.where(ok, _qk(qq, Kh[:, g], None if dK is None else dK[:, g], D), -np.inf)
            pp = np.exp(sc - sc.max(1, keepdims=True))                  # [T, T]
            pf = np.zeros((T, Tb))
            pf[:, :T] = pp * Vs[:, g][None, :]
            vt = np.zeros((Tb, d))
            vt[:T] = Vq[:, g]
            o[:, hq] = Q.mx_matmul(fp(pf, D, ATTN_BLOCK), vt.T, None, I.MX_W[I.W8], D) / \
                pp.sum(1, keepdims=True)
        att = fq(o.reshape(T, -1)) @ wq(a + "o_proj.weight").T
        x = x + G._norm(att, W[p + "post_attention_layernorm.weight"], eps)
        h = fq(G._norm(x, W[p + "pre_feedforward_layernorm.weight"], eps))
        dn = p + "mlp.down_proj.weight"
        fw, C = mlp_width(spec.ffn[i], fmt_of(dn) if wformat != "none" else "none")
        wd = wq(dn, cols=fw)
        m = padq(G._gelu(h @ wq(p + "mlp.gate_proj.weight").T) *
                 (h @ wq(p + "mlp.up_proj.weight").T), fw, C) @ wd.T
        del wd
        if spec.experts:
            m = G._norm(m, W[p + "post_feedforward_layernorm_1.weight"], eps) + \
                G._norm(_moe(spec, W, p, x, wq, fq, padq, none, fmt_of, D), W[
                    p + "post_feedforward_layernorm_2.weight"], eps)
        x = x + G._norm(m, W[p + "post_feedforward_layernorm.weight"], eps)
        if pli is not None:
            g = G._gelu(fq(x) @ wq(p + "per_layer_input_gate.weight").T) * pli[:, i]
            y = fq(g) @ wq(p + "per_layer_projection.weight").T
            x = x + G._norm(y, W[p + "post_per_layer_input_norm.weight"], eps)
        x = x * np.asarray(W[p + "layer_scalar"], np.float64)
        if key is not None:
            tmp = f"{ckpt}.tmp.npz"
            np.savez(tmp, key=key, layer=i, x=x)
            os.replace(tmp, ckpt)
    rows = list(range(T)) if rows is None else rows
    xl = fq(G._norm(x[rows], W["model.norm.weight"], eps))
    if hidden:
        return xl

    def head_rows(r0):          # (a checkpoint's head read 16384 rows at a time: 0.2 GB)
        if isinstance(W, G.Weights):
            f, n = W._files[head]
            return G._Rows(f, n)[r0:r0 + 16384]
        return W[head][r0:r0 + 16384]
    starts = range(0, spec.vocab, 16384)
    if on_rows is None:
        return np.concatenate([xl @ wq(None, hf, head_rows(r0)).T for r0 in starts], 1)
    # the quantized head held in float32 (E4B's in float64 would be 5.4 GB), its offsets kept
    Eq = [wq(None, hf, head_rows(r0)).astype(np.float32).T for r0 in starts]
    for i0 in range(0, len(rows), 64):      # on_rows(first row, logits [<= 64, vocab])
        xr = xl.rows(slice(i0, i0 + 64)) if isinstance(xl, Q.MXAct) else xl[i0:i0 + 64]
        on_rows(i0, np.concatenate([xr @ e for e in Eq], 1))


def _moe(spec, W, p, x, wq, fq, padq, none, fmt_of, D=128):
    """The MoE block of layer prefix p on rows x [T, H] (emulate's docstring), float64: the
    router reads the unit RMSNorm of x quantized; the experts read it times
    pre_feedforward_layernorm_2's gain g2, quantized (g2 is ~0 on the residual's outlier
    channels, up to 92 elsewhere: folded into gate / up instead, the quantization of the
    unit norm and of those columns costs 0.18-0.51 relative error per block in int8 from
    layer 10 on, against 0.007-0.018 with g2 on the activation)."""
    r, F = p + "router.", spec.expert_ffn
    xn = G._norm(x, None, spec.eps)
    xs = fq(xn)
    wr = np.asarray(W[r + "proj.weight"], np.float32) * (
        np.asarray(W[r + "scale"], np.float32) * np.float32(spec.hidden ** -0.5))[None, :]
    lg = xs @ wq(None, "none" if none else "int8", a=wr).T                  # [T, E]
    top = np.argsort(-lg, axis=1, kind="stable")[:, :spec.top_k]
    w = np.take_along_axis(lg, top, 1)
    w = np.exp(w - w[:, :1])
    w /= w.sum(1, keepdims=True)
    xe = fq(xn * np.asarray(W[p + "pre_feedforward_layernorm_2.weight"], np.float64)[None, :])
    pes = np.asarray(W[r + "per_expert_scale"], np.float32)
    n = p + "experts.down_proj"                 # (its name: the experts' format, wmap)
    es = [int(e) for e in np.unique(top)]

    xf = None if none else MO.ExpertFormat(spec.hidden, F, D, fmt_of(n))   # the slot's width
    Fp, C = (-(-F // D) * D, None) if none else (xf.F, xf.C)              # and down chunks

    def fake(gu, dn, e):        # expert e's gate, up, down [H, F padded] as the device holds them
        return wq(n, a=gu[:F]), wq(n, a=gu[F:]), wq(n, a=dn * pes[e], cols=Fp)

    out = np.zeros_like(xs)
    with ThreadPoolExecutor(THREADS) as ex:    # THREADS experts at a time (their fp64 copies)
        for i in range(0, len(es), THREADS):
            batch = es[i:i + THREADS]
            jobs = [ex.submit(fake, G._rows(W, p + "experts.gate_up_proj", [e])[0],
                              G._rows(W, n, [e])[0], e) for e in batch]
            for e, job in zip(batch, jobs):
                gq, uq, dq = job.result()
                t, j = np.nonzero(top == e)
                u = G._gelu(xe[t] @ gq.T) * (xe[t] @ uq.T)
                out[t] += w[t, j][:, None] * (padq(u, dq.shape[1], C) @ dq.T)
    return out


def top(spec, tok, lg, n=5):
    cap = G.softcap(spec, lg)
    t = np.argsort(-lg)[:n]
    return [(int(j), tok.decode([int(j)]), round(float(lg[j]), 4), round(float(cap[j]), 4)) for j in t]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("mode", choices=["step", "nll", "check"])
    ap.add_argument("args", nargs="+")
    ap.add_argument("--formats", default="", help="per-kind layer formats: attn, mlp (or down, "
                    "gateup), ple, experts, e.g. attn=int8,down=int8; kind@a-b: layers a..b "
                    "only (gateup@0-6=fp4)")
    ap.add_argument("--quant", default=None, help="the activation points (default all): act,kv,p")
    ap.add_argument("--out", help="step: save the logits (before the cap) to this npz")
    a = ap.parse_args()
    if a.mode == "check":
        spec, W = G.Spec.from_hf(a.args[0]), G.load_weights(a.args[0])
        spec = replace(spec, window=16)
        toks = np.random.default_rng(1).integers(0, spec.vocab, 40).tolist()
        for wf, hf in (("int8", "int8"), ("fp4", "int8"), ("fp4", "fp4")):
            ref = G.emulated_logits(spec, W, toks, wformat=wf, head_format=hf)
            got = emulate(spec, W, toks, wformat=wf, hf=hf)
            print(f"{wf} / {hf} head: max |diff| {np.abs(ref - got).max():.3g} of "
                  f"{np.abs(ref).max():.3g}")
        ref = G.reference_logits(spec, W, toks)
        print(f"none vs reference_logits (fp32): max |diff| "
              f"{np.abs(ref - emulate(spec, W, toks, wformat='none')).max():.3g}")
        return
    from transformers import AutoTokenizer
    quant = None if a.quant is None else [q for q in a.quant.split(",") if q]
    wmap = dict(kv.split("=") for kv in a.formats.split(",") if kv)
    path, ref = a.args[0], np.load(a.args[1])
    spec, W = G.Spec.from_hf(path), G.load_weights(path)
    wf, hf = (a.args[3], a.args[4]) if a.mode == "step" else (a.args[2], a.args[3])
    pts = a.quant if a.quant is not None else "none" if wf == "none" else "all"
    what = f"{wf} layers{' ' + str(wmap) if wmap else ''}, {hf} head, points {pts or 'none'}"
    t0 = time.time()
    if a.mode == "step":
        step = int(a.args[2])
        ids = [int(t) for t in ref["ids0"]] + [int(t) for t in ref["gen0"][:step]]
        lg = emulate(spec, W, ids, wformat=wf, hf=hf, rows=[len(ids) - 1], quant=quant,
                     wmap=wmap)[0]
        tok = AutoTokenizer.from_pretrained(path)
        want = int(ref["gen0"][step])
        print(f"{what}: {len(ids)} tokens, {time.time() - t0:.0f} s")
        print("  top 5 (token, logit, capped):", top(spec, tok, lg))
        print(f"  REF's token {want} {tok.decode([want])!r}: rank {int((lg > lg[want]).sum()) + 1}, "
              f"{float(lg.max() - lg[want]):.4f} below the top (before the cap)")
        if a.out:
            np.savez(a.out, logits=lg)
        return
    out = a.args[4]
    ids = [int(t) for t in ref["ids0"]]
    nll, top1 = np.zeros(len(ids) - 1), np.zeros(len(ids) - 1, np.int64)

    def on_rows(i0, lg):
        n = min(len(lg), len(ids) - 1 - i0)
        lg = G.softcap(spec, lg[:n])
        m = lg.max(1, keepdims=True)
        nll[i0:i0 + n] = m[:, 0] + np.log(np.exp(lg - m).sum(1)) - \
            lg[np.arange(n), ids[i0 + 1:i0 + 1 + n]]
        top1[i0:i0 + n] = lg.argmax(1)

    ckpt = f"{out}.ckpt.npz"                 # (a run killed at the memory floor resumes)
    emulate(spec, W, ids, wformat=wf, hf=hf, quant=quant, wmap=wmap,
            rows=list(range(len(ids) - 1)), on_rows=on_rows, ckpt=ckpt)
    np.savez(out, nll=nll, top1=top1)
    if os.path.exists(ckpt):
        os.remove(ckpt)
    print(f"{what}: {len(ids) - 1} positions, mean NLL {nll.mean():.4f}, ppl "
          f"{np.exp(nll.mean()):.3f}; {time.time() - t0:.0f} s")


if __name__ == "__main__":
    sys.exit(main())
