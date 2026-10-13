"""Mixture-of-experts FFN with its experts streamed into a DRAM cache: path (a) of
docs/offload.md. The card routes and computes every expert; opentpu.host.offload's server only
moves experts into per-layer LRU slots in the card's DRAM and keeps the directory.

A MoE layer on the device (`moe_ffn`), one token:

1. the router: an MM of the normalized residual -> E logits (and, with a shared expert, its
   gate's logit as row E of the same matrix), and the model's rule (MoESpec.rule);
2. the k best by ARGMAX with knock-out (in order, ties to the first), their global ids
   (the layer's base j * E + index) and weights: LFM2's sigmoid scores, chosen with the
   expert bias, renormalized and scaled; Qwen's softmax of the k largest logits;
3. the fence, WAITW served >= seq (the host has finished every earlier request: none of this
   layer's slots is being replaced, and the one request row and answer are free), then the
   ids and seq + 1 to the mailbox; then what needs no routed expert, while the host streams
   the missing ones: a shared expert (a dense SwiGLU of the layer block, times sigmoid of its
   gate's logit; docs/offload.md 15) and beside()'s work;
4. the directory: each id's present flag;
5. the experts present (a LOOP over the k ids, a LOOP R[present] inside), then the others
   (LOOP R[1 - present]). A present one takes its slot address from its directory entry, a
   missing one from its word of the request's answer (docs/offload.md 10.11), each by WAITW
   (!= 0: at once for an expert present; for the others once the host has written the
   answer) to a TMEM word and RLD (raw) into a register; a missing one then waits for its
   slot's tag word (WAITW != 0: the host writes it in its expert's DMA's last beat). Each
   zeroes its slot's tag (the next nonzero tag there is a later expert's), runs
   kernels.mlp.swiglu_down there and writes its weighted output to its row of a [k, H] tile;
   then the answer's words are zeroed (the host writes the next request's answer only after
   its post);
6. the rows summed in the router's order (the result does not depend on what the cache held),
   then the shared expert's output added (last, as before it was computed early: bit for bit
   the same sum), and the whole to the residual.

It holds one register (the resident decode's run arguments and the layer loops hold the
rest): the per-expert values are the columns of a small tile that each loop rotates, so
expert i is always at column 0 in iteration i.

With MoESpec.hint the layer also posts a prefetch hint before its mixer (`moe_hint`,
docs/offload.md section 12): its router on the layer's input, the k best as a request whose
ids are offset by layers x E; the host replaces their slots' victims at once and moves the
missing experts in the link's idle time, and the route's own request finds them landed or on
their way.

`moe_ffn_rows` is the layer on R rows at once (layer-major prefill, docs/offload.md section 13):
each row routes as moe_ffn's; one request carries the rows' R x k ids (the count at mbox + 4);
the card finds their union (each id's first place among them) and runs each union expert once
on all R rows, storing each row's unweighted output to its (row, rank) place in a DRAM scratch
of [R k + 1, H] (R k <= 16, the request's one line; the last row a sink for the outputs no row
chose); each row then sums its own k in its router's order, so a row's result is moe_ffn's bit
for bit. `moe_hint_rows` is moe_hint on R rows: a layer-major run's hint for the next layer, its
router on the run's output rows (section 13.9).

Expert-major (sections 13.11, 13.13) splits a layer-major MoE layer over a chunk's runs: each
run's `moe_prologue_rows` routes its rows, keeps their ids, weights and norm in the rows' records
of a scratch (em_record; the server carves it from the first expert slots for the prefill), and
posts the experts as a need line; then one `moe_expert_run` a layer runs each expert its rows
chose once, in passes of up to C rows (em_pass_rows, docs/offload.md 13.18; ld-memch's
bucketing), each pass waiting on its directory entry; the next layer's runs end the layer (`moe_combine_rows`). A row's arithmetic is
moe_ffn_rows', bit for bit.

`ExpertFormat` is one expert's slot: gate and up [F, H] and W_down's column parts, laid out as a
layer block's MLP (lfm2.Image), so the FFN is kernels.mlp.swiglu_down at the slot's address
(a compiler.DevVar: the register RLD sets). One slice (S = 1, the board's).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .. import isa as I
from .. import language as ol
from .. import quant as Q
from ..compiler import Affine, CompileError, DevVar, QTensor, Tensor, current
from ..host.offload import LINE, SPLIT, ExpertServer, Layout, PoolFile, dram_of, to_split
from ..kernels import mailbox as MB
from ..kernels.lib import gelu_tanh, rmsnorm, sigmoid, silu
from ..kernels.mlp import _chunk, swiglu_down

NEG = -3.0e38                       # knocked out (below every score)
ACTS = {"silu": silu, "gelu_tanh": gelu_tanh}


@dataclass(frozen=True)
class MoESpec:
    E: int                 # experts per layer
    k: int                 # experts per token
    ffn: int               # an expert's width
    first: int = 0         # the first MoE layer (the ones before it have a dense MLP)
    rule: str = "sigmoid_bias"       # LFM2: sigmoid scores, the choice on score + expert bias;
                                     # "softmax": Qwen, the k largest logits' softmax
    norm: bool = True                # renormalize the chosen weights (softmax: always)
    scale: float = 1.0               # then multiply them by this
    shared: int = 0                  # a shared expert's width (Qwen: the layer's dense MLP,
                                     # times sigmoid(x . w_gate)), 0: none
    act: str = "silu"                # the experts' gate: SiLU, or "gelu_tanh" (Gemma 4)
    hint: bool = False               # each layer posts its router's k best on its input as a
                                     # prefetch hint before its mixer (moe_hint; Qwen3.5)

    def __post_init__(self):
        if self.rule not in ("sigmoid_bias", "softmax"):
            raise ValueError(f"MoE router rule {self.rule!r}")
        if self.act not in ACTS:
            raise ValueError(f"MoE expert activation {self.act!r}")
        if self.rule == "softmax" and not self.norm:
            raise ValueError("a softmax router over all experts is not supported (norm=False)")


class ExpertFormat:
    """One expert in its slot: W_gate, W_up [F, H] (rows, then their scale rows) and W_down
    [H, F] in column parts of `C` (swiglu_down's chunks; its rows quantized whole, each part's
    scale rows the row's Sw and its blocks': quant.column_parts), each D-byte aligned. A width
    that is not a whole number of the format's chunks (Gemma 4's 704) is padded with zero rows
    of W_gate and W_up and zero columns of W_down to `F` (768): exact, act(0) * 0 = 0."""

    def __init__(self, H: int, F: int, D: int, wformat: str):
        rb = lambda k: Q.row_bytes(k, wformat, D)                       # noqa: E731
        q = D if wformat == "int8" else 2 * D
        self.F0, F = F, -(-F // q) * q       # the expert's own width, the slot's
        self.H, self.F, self.D, self.wformat = H, F, D, wformat
        self.C = _chunk(F, D, q)
        a, align = 0, max(LINE, D)          # the MXU streams whole D-byte chunks

        def take(n):
            nonlocal a
            at, a = a, a + -(-n // align) * align
            return at
        sb = lambda k: Q.scale_bytes(k, wformat, D)                     # noqa: E731
        self.wg = (take(F * rb(H)), take(F * sb(H)))
        self.wu = (take(F * rb(H)), take(F * sb(H)))
        self.wd = [(take(H * rb(self.C)), take(H * sb(self.C))) for _ in range(F // self.C)]
        self.nbytes = a

    def ranges(self) -> list[int]:
        """The down projection's QACT ranges over the expert's own width F0, for an emulation
        of the unpadded expert: a chunk of C each, the last cut at F0 (the slot's zero columns
        change no range's amax)."""
        return [min(self.C, self.F0 - c) for c in range(0, self.F0, self.C)]

    def pack(self, wg: np.ndarray, wu: np.ndarray, wd: np.ndarray) -> np.ndarray:
        """The slot's bytes for W_gate, W_up [F0, H] and W_down [H, F0] (float)."""
        out = np.zeros(self.nbytes, np.uint8)
        if self.F != self.F0:                   # the zero padding
            pad = self.F - self.F0
            wg, wu = (np.pad(np.asarray(w, np.float32), ((0, pad), (0, 0))) for w in (wg, wu))
            wd = np.pad(np.asarray(wd, np.float32), ((0, 0), (0, pad)))

        def put(pair, qs):
            for at, v in zip(pair, qs):
                v = np.ascontiguousarray(v).view(np.uint8).reshape(-1)
                out[at:at + v.size] = v
        fm, D = self.wformat, self.D
        put(self.wg, Q.quantize_mxu(wg, fm, D))
        put(self.wu, Q.quantize_mxu(wu, fm, D))
        # W_down's rows quantized whole (one Sw a row), stored in column parts of C
        q, s = Q.quantize_mxu(np.asarray(wd, np.float32), fm, D)
        for pair, qs in zip(self.wd, Q.column_parts(q, s, fm, self.F, self.C, D)):
            put(pair, qs)
        return out

    def descs(self, base) -> SimpleNamespace:
        """wg, wu, wd QTensors of the expert whose slot starts at `base` (an address
        expression: a DevVar for a slot found at run time)."""
        H, F, D, C, fm = self.H, self.F, self.D, self.C, self.wformat
        base = Affine.of(base)

        def q(pair, n, k):
            return QTensor.weight(base + pair[0], base + pair[1], n, k, fm, D)
        parts = tuple(q(p, H, C) for p in self.wd)
        wd = QTensor(parts[0].data, parts[0].scale, (H, F), parts[0].rs, parts[0].srs, D,
                     parts=parts, pw=C, wf=parts[0].wf)
        return SimpleNamespace(wg=q(self.wg, F, H), wu=q(self.wu, F, H), wd=wd)


def gemma_expert(W, p: str, e: int):
    """Gemma 4's routed expert e of the layer at prefix p (`p + "experts.gate_up_proj"` [E, 2F,
    H], gate rows first; `p + "experts.down_proj"` [E, H, F]) as W_gate, W_up [F, H] and W_down
    [H, F] for ExpertFormat.pack, router.per_expert_scale[e] folded into W_down (it multiplies
    the expert's output). pre_feedforward_layernorm_2's gain stays out: moe_ffn applies it to
    the experts' input before quantizing it (lw.g_exp; docs/offload.md 11.2). A lazy W with
    `part` reads the one expert."""
    k = p + "experts."
    get = getattr(W, "part", None)
    gu, dn = ((get(k + "gate_up_proj", e), get(k + "down_proj", e)) if get is not None else
              (W[k + "gate_up_proj"][e], W[k + "down_proj"][e]))
    gu, dn = np.asarray(gu, np.float32), np.asarray(dn, np.float32)
    F = gu.shape[0] // 2
    s = np.asarray(W[p + "router.per_expert_scale"], np.float32)[e]
    return gu[:F], gu[F:], dn * s


def gemma_router(W, p: str) -> np.ndarray:
    """Gemma 4's router [E, H] on the residual's unit RMSNorm (moe_ffn's input): router.proj
    with router.scale and H^-0.5 folded into its columns."""
    w = np.asarray(W[p + "router.proj.weight"], np.float32)
    return w * (np.asarray(W[p + "router.scale"], np.float32) * np.float32(w.shape[1] ** -0.5))


# ---------------------------------------------------------------------------- the pool's key
def quant_source(wformat: str) -> str:
    """The quantizer's code an expert's bytes depend on: quant.py and fp32.py (int8x's
    quantize_mx, the 4-bit formats' scale arithmetic). Not numpy's version (qcache._source's):
    a pool is packed on one host and read on another."""
    from .. import fp32 as F
    h = hashlib.blake2b(digest_size=16)
    for f in (Q.__file__, F.__file__):
        h.update(Path(f).read_bytes())
    return h.hexdigest()


def checkpoint_id(model_dir) -> str | None:
    """A checkpoint's identity: its config.json's hash (strip_experts.py copies the file as it
    is, so the card host's checkpoint without the experts has the packing host's); None
    without one."""
    f = Path(model_dir) / "config.json" if model_dir is not None else None
    if f is None or not f.exists():
        return None
    return hashlib.blake2b(f.read_bytes(), digest_size=16).hexdigest()


def pool_key(image, model_dir=None, mtp_from=None) -> dict:
    """The key of `image`'s expert pool (`<pool>.key`, JSON): what decides its experts' bytes,
    the MXU's numerics (quant.MX_NUMERICS: key 2 and on; a key 1 pool holds the scale words of
    the MXU before MX, docs/mx_mxu.md), the quantizer's code (quant_source), the format, D, the
    layout (the pool's and the slot's: ExpertFormat's parts) and the checkpoint
    (checkpoint_id); "numpy" and "model" are notes, not compared. mtp_from: the checkpoint an
    MTP image's MTP layer (its experts the pool's last layer) comes from when not the model's
    (LazyWeights.mtp_from: MiniCPM-V 4.7's, docs/offload.md 15.6), its checkpoint_id as
    "mtp_from". An image with entries past the experts (`pool_extra`: (their count, their key),
    MiniCPM-V's vision pieces staged through the slots, docs/minicpmv.md 4.2) adds "vision":
    their key, which open_pool compares on its own (key_diff does not: the experts' part of a
    pool is the same)."""
    f, L = image.fmt, image.offload
    key = {"key": 2, "numerics": Q.MX_NUMERICS, "quant": quant_source(f.wformat),
           "format": f.wformat, "D": f.D,
           "layout": {"layers": L.layers, "E": L.E, "slot_bytes": L.slot_bytes, "H": f.H,
                      "F": f.F0, "C": f.C, "wg": list(f.wg), "wu": list(f.wu),
                      "wd": [list(x) for x in f.wd]},
           "checkpoint": checkpoint_id(model_dir), "numpy": np.__version__,
           "model": None if model_dir is None else Path(model_dir).name}
    if mtp_from is not None:
        key["mtp_from"] = checkpoint_id(mtp_from)
    if getattr(image, "pool_extra", None) is not None:
        key["vision"] = image.pool_extra[1]
    return key


def key_diff(have: dict, want: dict) -> list:
    """The fields in which a pool's key `have` differs from `want` (the checkpoint only when
    both know it; the MTP layer's checkpoint, mtp_from, whenever either names one)."""
    out = [k for k in ("key", "numerics", "quant", "format", "D", "layout")
           if have.get(k) != want.get(k)]
    if have.get("mtp_from") != want.get("mtp_from"):
        out.append("mtp_from")
    if None not in (have.get("checkpoint"), want.get("checkpoint")) and \
            have["checkpoint"] != want["checkpoint"]:
        out.append("checkpoint")
    return out


def check_pool_key(pool_file, want: dict) -> None:
    """Refuse an expert pool whose key (`<pool>.key`) is not `want`: its experts were packed by
    another quantizer, for the MXU before MX, in another format or layout, or from another
    checkpoint. A pool without a key is refused too: it was packed before keys, so before the
    MX numerics (its scale words are the former MXU's); pack_pool stamp gives one its key once
    `sums` shows it holds this tree's bytes."""
    kf = Path(str(pool_file) + ".key")
    if not kf.exists():
        raise ValueError(
            f"{pool_file} has no key file: a pool packed before keys, for the MXU before MX "
            f"(docs/mx_mxu.md), whose experts' bytes are not this image's. Repack it: remove "
            f"{pool_file} and its .packed and .format, then python3 tools/offload/pack_pool.py "
            f"MODEL {pool_file} init and its pack workers (or let the run pack the experts it "
            f"asks for); a pool known to hold this tree's bytes (pack_pool.py ... sums) gets "
            f"its key by pack_pool.py MODEL {pool_file} stamp")
    have = json.loads(kf.read_text())
    diff = key_diff(have, want)
    if diff:
        note = {k: (have.get(k), want.get(k)) for k in diff if k != "layout"}
        raise ValueError(
            f"{pool_file} was packed for another model image ({', '.join(diff)} differ"
            f"{': ' + str(note) if note else ''}; {have.get('format')} D {have.get('D')} of "
            f"{have.get('model')}, this run is {want['format']} D {want['D']} of "
            f"{want.get('model')}): its experts' bytes are not this image's. Repack it: remove "
            f"{pool_file} and its .packed, .format and .key, then python3 "
            f"tools/offload/pack_pool.py MODEL {pool_file} init and its pack workers (or let "
            f"the run pack the experts it asks for)")


def pool_entries(path: Path, n: int) -> int:
    """The entries a pool file holds by its key: n experts and its "vision" key's entries."""
    kf = Path(str(path) + ".key")
    have = json.loads(kf.read_text()) if kf.exists() else {}
    return n + int((have.get("vision") or {}).get("entries", 0))


def open_pool(layout: Layout, pool_file, mapped: bool = True,
              key: dict | None = None, extra: int = 0) -> PoolFile:
    """The expert pool file for `layout`, opened, and its packed experts' read into the page
    cache started (PoolFile.warm: the host's RAM tier; the Engine opens it before it builds the
    image, so the read runs during the build). The file is the whole pool's size (sparse until
    packed), and `<pool_file>.packed` marks the experts in it (a file of the right size without
    one is a pool packed whole). A new file is in the split format
    (opentpu.host.offload.split_order), as `<pool_file>.format` says; a file without it holds
    the slot format. With `key` (pool_key) a new file gets it as `<pool_file>.key`, and an
    existing one must have it (check_pool_key; one without a key file is refused).
    The page cache stays the kernel's to reclaim: nothing is pinned. mapped: PoolFile's (its
    reads touched through a read-only map: docs/offload.md 10.7).

    extra: entries past the experts (layers x E on; key["vision"] their key: pool_key), each a
    slot's bytes. A pool whose own extra entries are not these (none, or another key's) keeps
    its experts and gets these, not packed yet (the file grown or cut past its experts, its
    key's "vision" replaced); a run without extra opens a pool with them as its experts'."""
    L = layout
    n = L.layers * L.E
    N = n + extra
    path = Path(pool_file)
    done = Path(str(path) + ".packed")
    kf = Path(str(path) + ".key")
    fresh = not path.exists() or path.stat().st_size != pool_entries(path, n) * L.slot_bytes
    if not fresh and key is not None:
        check_pool_key(path, key)
    want_x = (key or {}).get("vision") if extra else None
    have_x = json.loads(kf.read_text()).get("vision") if not fresh and kf.exists() else None
    if fresh:
        with open(path, "wb") as f:
            f.truncate(N * L.slot_bytes)
    elif extra and have_x != want_x:    # its extra entries replaced, its experts kept
        with open(path, "r+b") as f:
            f.truncate(n * L.slot_bytes)
            f.truncate(N * L.slot_bytes)
        pk = np.fromfile(done, np.uint8)[:n] if done.exists() else np.ones(n, np.uint8)
        done.write_bytes(np.concatenate([pk, np.zeros(N - n, np.uint8)]).tobytes())
        have = json.loads(kf.read_text()) if kf.exists() else dict(key)
        have["vision"] = want_x
        kf.write_text(json.dumps(have, indent=1) + "\n")
    M = pool_entries(path, n) if not fresh else N      # (the file's entries)
    if fresh or not done.exists() or done.stat().st_size != M:
        done.write_bytes(bytes(M) if fresh else bytes([1]) * M)
    fmt = Path(str(path) + ".format")
    if fresh:
        fmt.write_text(SPLIT + "\n")
        if key is not None:
            kf.write_text(json.dumps(key, indent=1) + "\n")
        elif kf.exists():
            kf.unlink()
    pf = PoolFile(path, L.slot_bytes, fmt.exists() and fmt.read_text().strip() == SPLIT,
                  mapped=mapped)
    pf.arr = np.memmap(path, np.uint8, "r+", shape=(N, L.slot_bytes))
    pf.packed = np.memmap(done, np.uint8, "r+", shape=(N,))
    pf.ids = np.nonzero(np.asarray(pf.packed))[0]
    pf.resident_at_open = pf.resident(pf.ids)
    pf.warm(pf.ids)
    return pf


def rewarm(pf: PoolFile):
    """The pool's experts that left the page cache since the open's read began, read into it
    again; the Engine calls it once the image is built and the build's memory freed. A 35B
    MoE image's build peaks at 28-30 GB of the card host's 31 and takes the pool out of the
    page cache again (docs/minicpmv.md 6, pooldisk: its turns then read their experts from the
    disk, ~15-32 ms each against 0.5 ms). The open's read is stopped (it may still run: an
    image from the image cache), then those of the packed experts with bytes out of the page
    cache (mincore) are read in file order on a new thread. Returns the warm thread now: the
    open's when it had ended and nothing left the page cache, or where the host cannot tell
    (no mincore: the open's read is left to run)."""
    t = pf.warm_t
    if pf._pages() is None:                 # (no mincore here; and no map of its own made:
        return t                            # pool_map False keeps none)
    stopped = t is not None and t.is_alive()
    if stopped:
        t.stop.set()
        t.join()
    gone = pf.absent(pf.ids)
    if gone is None and stopped:            # (mincore failed: all of them again)
        gone = pf.ids
    # (a stopped read with nothing left out: an empty one, so the record says done)
    return t if gone is None or not (len(gone) or stopped) else pf.warm(gone)


def serve(layout: Layout, expert, backend, pool_file=None, warm=True,
          policy: str = "lfu") -> ExpertServer:
    """The host's expert server on the backend's DRAM (slice 0); with `warm`, the slots filled
    with each layer's first experts. expert(g): global expert g's slot bytes. An expert is
    packed when it is first asked for and kept in host RAM, or, with `pool_file` (a path, or
    open_pool's PoolFile), in that file (the page cache, or the SSD tier), which keeps it for
    later runs (`server.pool`: the PoolFile). `policy`: the slots' replacement (ExpertServer's;
    least decayed use, docs/offload.md 5.5)."""
    L = layout
    pf = None
    if pool_file is None:
        cache: dict = {}

        def pool(g):
            if g not in cache:
                cache[g] = expert(g).tobytes()
            return cache[g]
    else:
        pf = pool_file if isinstance(pool_file, PoolFile) else open_pool(L, pool_file)

        def pool(g):                    # read from the file (the memmap's writes are in the
            if not pf.packed[g]:        # page cache preadv reads)
                pf.arr[g] = to_split(expert(g)) if pf.split else expert(g)
                pf.packed[g] = 1
            return pf.get(g)
    srv = ExpertServer(dram_of(backend, L), L, pool, policy=policy)
    srv.pool_file = pf
    if pf is not None:                  # (its reads' touches after each request is served)
        pf.defer_touch = True
    srv.pool_warm = None if pf is None else pf.warm_t
    srv.load([j * L.E + e for j in range(L.layers) for e in range(L.E)] if warm else ())
    return srv


# ---------------------------------------------------------------------------- reference
def route(logits: np.ndarray, bias: np.ndarray | None, mo: MoESpec):
    """The model's choice for one token's router logits [E]: (ids in the device's order, their
    weights). The k largest (ties: the first) of: LFM2, sigmoid scores + bias, the chosen
    scores renormalized (+ 1e-6, as Hugging Face) and scaled; softmax (Qwen), the logits, the
    chosen ones' softmax (Hugging Face's softmax over all, renormalized over the k)."""
    lg = np.asarray(logits, np.float64)
    if mo.rule == "softmax":
        s, sel = lg, lg.copy()
    else:
        s = 1.0 / (1.0 + np.exp(-lg))
        sel = s + bias
    ids = []
    for _ in range(mo.k):
        i = int(np.argmax(sel))
        ids.append(i)
        sel[i] = -np.inf
    w = s[ids]
    if mo.rule == "softmax":
        w = np.exp(w - w[0])
        w = w / w.sum()
    elif mo.norm:
        w = w / (w.sum() + 1e-6)
    return ids, w * mo.scale


# ---------------------------------------------------------------------------- the device
def moe_hint(x, lw, mo: MoESpec, dev: SimpleNamespace, eps: float) -> None:
    """Path (a)'s prefetch hint (docs/offload.md section 12), before the layer's mixer: its
    router on the layer's input x through moe_ffn's norm, the k best by the model's rule (no
    weights), posted as a request whose global ids are offset by dev.hint_off (layers x E: the
    host's mark of a hint). The host writes served once it has replaced the hinted experts'
    victims, before their transfers, so the route's fence (moe_ffn) waits for no expert; an
    expert still on its way reads as missing there and is waited for by its answer word and
    its tag. lw and dev as moe_ffn's. One register."""
    b = current()
    E, k = mo.E, mo.k
    xs = ol.quantize(rmsnorm(x, ol.load(lw.g_post), eps))
    lg = ol.dot(xs, lw.router)
    del xs
    sel, pr, ids, tmp = ol.empty((E,)), ol.empty((2,)), ol.empty((k,)), ol.empty((k,))
    if mo.rule == "softmax":
        sel.set(lg[0, 0:E])
    else:
        sel.set(sigmoid(lg[0, 0:E]) + ol.load(lw.ebias))
    del lg
    r = b.scratch()
    lp = b.begin_loop(k)                # the k best, as the route picks them
    b.emit(I.argmax(pr.base, sel.base, 1, E, comment="hint: best"))
    b.emit(I.rld(r, pr.base + 1, bound=(0, E - 1), comment="its index"))
    b.emit(I.vop(I.V_FILL, sel.base, 0, 0, 1, 1, 0, 0, 0, I.B_SCALAR, NEG, ra=r,
                 comment="knock out"))
    if k > 1:
        tmp[0:k - 1].set(ids[1:k])
        ids[0:k - 1].set(tmp[0:k - 1])
    b.emit(I.vop(I.V_COPY, ids.base + k - 1, pr.base + 1, 0, 1, 1, 0, 0, 0, comment="id"))
    b.end_loop(lp)
    b.unscratch(r)
    ids.set(ids + ol.load(lw.gbase) + float(dev.hint_off))
    seq = MB.wait_served(dev.mbox)                  # MB.post, with the request's count
    ol.store(Tensor(Affine(dev.mbox + LINE), (k,), (1,)), ids)
    ol.store(Tensor(Affine(dev.mbox + 4), (1,), (1,)), ol.full((1,), float(k)))
    seq.set(seq + 1.0)
    ol.store(Tensor(Affine(dev.mbox), (1,), (1,)), seq)


def moe_hint_rows(x, lw, mo: MoESpec, dev: SimpleNamespace, eps: float) -> None:
    """moe_hint on R rows at once (x [R, H], R * k <= LINE / 4: one request row): each row's k
    best by the router of lw's layer, through its norm (moe_ffn's route), posted as one hint
    of the rows' R * k ids (repeats included) and their count. A layer-major prefill run's
    hint for the next layer (docs/offload.md 13.9): layer j's run on its output rows, with
    layer j + 1's lw, ahead of layer j + 1's runs. One register."""
    b = current()
    R, E, k = x.rows, mo.E, mo.k
    N = R * k
    if N > LINE // 4:
        raise CompileError(f"{R} rows of {k} experts: more ids than a request row holds")
    xs = ol.quantize(rmsnorm(x, ol.load(lw.g_post), eps))
    lg = ol.dot(xs, lw.router)                          # [R, E] (+ the shared expert's gate)
    del xs
    gid = ol.empty((R, k), dense=True)
    pr, tmp, ids = ol.empty((2,)), ol.empty((k,)), ol.empty((k,))
    off = ol.load(lw.gbase) + float(dev.hint_off)
    r = b.scratch()
    for q in range(R):                                  # each row's k best, as the route's
        sel = ol.empty((E,))
        if mo.rule == "softmax":
            sel.set(lg[q, 0:E])
        else:
            sel.set(sigmoid(lg[q, 0:E]) + ol.load(lw.ebias))
        lp = b.begin_loop(k)
        b.emit(I.argmax(pr.base, sel.base, 1, E, comment="hint: best"))
        b.emit(I.rld(r, pr.base + 1, bound=(0, E - 1), comment="its index"))
        b.emit(I.vop(I.V_FILL, sel.base, 0, 0, 1, 1, 0, 0, 0, I.B_SCALAR, NEG, ra=r,
                     comment="knock out"))
        if k > 1:
            tmp[0:k - 1].set(ids[1:k])
            ids[0:k - 1].set(tmp[0:k - 1])
        b.emit(I.vop(I.V_COPY, ids.base + k - 1, pr.base + 1, 0, 1, 1, 0, 0, 0, comment="id"))
        b.end_loop(lp)
        del sel
        gid[q, :].set(ids + off)
    b.unscratch(r)
    del lg, ids, tmp, off
    seq = MB.wait_served(dev.mbox)                  # MB.post, with the request's count
    ol.store(Tensor(Affine(dev.mbox + LINE), (N,), (1,)), gid.reshape(1, N)[0, :])
    ol.store(Tensor(Affine(dev.mbox + 4), (1,), (1,)), ol.full((1,), float(N)))
    seq.set(seq + 1.0)
    ol.store(Tensor(Affine(dev.mbox), (1,), (1,)), seq)


def _ebound(dev):
    """A directory entry's byte offset (8 a global id): its RLD bound, if the device says how
    many ids there are."""
    n = getattr(dev, "ids", None)
    return (0, 8 * (n - 1)) if n else None


def slot_of(r: int, word, off: int, dev: SimpleNamespace, missing: bool,
            off_bound=None) -> None:
    """R[r] = the slot of the expert whose offset (TMEM word `off`: its directory entry's, or
    for a missing one its answer word's) the loop has at column 0, its data landed, and the
    slot's tag zeroed. Present: WAITW on its entry (at once). Missing: WAITW on its answer
    word (once the host has written the answer), then on the slot's tag (once the expert's
    DMA has landed: the host writes the tag in its last beat). The tag is zeroed for every
    expert the card uses, so a nonzero tag is always a later expert's (docs/offload.md 10.11).
    word: a one-word TMEM tile; off_bound: the offset's RLD bound (isa.rld); the slot's is the
    pool's (dev.slots)."""
    b = current()
    b.emit(I.rld(r, off, bound=off_bound, comment="answer word" if missing else "entry offset"))
    b.waitw(word, dev.answer if missing else dev.dir, 0, I.C_NE, ra=r,
            comment="wait: its slot (the answer)" if missing else "its slot")
    b.rld(r, word, raw=True, bound=getattr(dev, "slots", None), comment="its slot")
    if missing:
        b.waitw(word, dev.tag, 0, I.C_NE, ra=r, comment="wait: its data (the slot's tag)")
    ol.store(Tensor(Affine(dev.tag) + DevVar("expert slot", r), (1,), (1,)), ol.zeros((1,)))


def moe_ffn(x, lw, mo: MoESpec, dev: SimpleNamespace, eps: float, beside=None,
            residual: bool = True, y_first: bool = False):
    """x + the MoE FFN of one token (module docstring). lw: the layer's g_post [H], router
    QTensor [E, H] ([E + 1, H] with a shared expert: its gate is row E), ebias Tensor [E]
    (sigmoid_bias), gbase Tensor [1] (j * E as fp32), and with a shared expert its SwiGLU
    wg, wu, wd; with g_exp [H] (Gemma 4: pre_feedforward_layernorm_2's gain) the routed
    experts read their own quantized input, the norm times g_exp, and the router the norm (the
    norm alone, quantized, has its blocks scaled by the residual's outlier channels, which the
    gain zeroes: docs/offload.md 11.2). dev: the offload words (mbox, served, answer, dir:
    static DRAM addresses), `tag` (a slot's tag word, from the slot) and `fmt`, the
    ExpertFormat.

    beside(): emits work that needs no expert (Gemma 4's dense MLP) right after the request is
    posted, so that it runs while the host streams the missing experts (docs/offload.md 5.3);
    moe_ffn's register is free during it. A shared expert runs there too, before beside()'s
    work (docs/offload.md 15: the card waits for its misses in most MoE layers, and a shared
    expert computed after them lengthened every wait by its time). residual=False: the FFN's
    output without x (Gemma 4 norms the experts' sum before its residual). y_first: the experts' [k, H] outputs take
    their TMEM before the router's input does (Gemma 4 26B-A4B: 22,544 words, which only the
    free space before the step's later tiles holds)."""
    b = current()
    if ol.num_programs() != 1:
        raise CompileError("moe_ffn runs on one slice")
    E, k, H = mo.E, mo.k, x.cols
    y = ol.empty((k, H)) if y_first else None
    if getattr(lw, "g_exp", None) is None:
        xs = xe = ol.quantize(rmsnorm(x, ol.load(lw.g_post), eps))
    else:
        xn = rmsnorm(x, ol.load(lw.g_post), eps)
        xs, xe = ol.quantize(xn), ol.quantize(xn * ol.load(lw.g_exp)[None, :])   # (QACT
        del xn                                                                   # CSCALE)
    lg = ol.dot(xs, lw.router)
    sc, sel = ol.empty((E,)), ol.empty((E,))
    if mo.rule == "softmax":
        sc.set(lg[0, 0:E])
        sel.set(sc)
    else:
        sc.set(sigmoid(lg[0, 0:E]))
        sel.set(sc + ol.load(lw.ebias))
    gate = sigmoid(lg[0, E:E + 1]) if mo.shared else None
    # One register in all (the resident decode's arguments and the layer loops hold the
    # rest): r holds one value at a time (an index, the seq bits, an offset, a count, a slot
    # address, a row). The k experts' values are the columns of pe, a [5, k] tile that every
    # loop over the experts rotates left by one at the end of its body: expert i is at column
    # 0 in iteration i, and after k iterations every row is back in order. (A slot address
    # never passes through the VPU, which would flush it as a denormal: WAITW copies it from
    # the directory to the word `word`, at once for an expert present, and RLD raw takes its
    # bits into r.)
    EP, MISS, OFF, WT, ROW, ANS = range(6)          # pe's rows
    pe, tmp, ids, pr = ol.empty((6, k)), ol.empty((6, k)), ol.empty((k,)), ol.empty((2,))
    word = ol.empty((1,))
    wt = pe[WT, :]

    def rotate(rows=slice(0, 6)):
        a, e = rows.start, rows.stop
        if k > 1:
            tmp[a:e, 0:k - 1].set(pe[a:e, 1:k])
        tmp[a:e, k - 1:k].set(pe[a:e, 0:1])
        pe[a:e, :].set(tmp[a:e, :])

    r = b.scratch()
    # the k best: ARGMAX, knock-out; in the router's order, the new one to column k - 1
    lp = b.begin_loop(k)
    b.emit(I.argmax(pr.base, sel.base, 1, E, comment="moe: best"))
    b.emit(I.rld(r, pr.base + 1, bound=(0, E - 1), comment="its index"))
    b.emit(I.vop(I.V_FILL, sel.base, 0, 0, 1, 1, 0, 0, 0, I.B_SCALAR, NEG, ra=r,
                 comment="knock out"))
    rotate(slice(WT, WT + 1))
    b.emit(I.vop(I.V_COPY, wt.base + k - 1, sc.base, 0, 1, 1, 0, 0, 0, rb=r, comment="weight"))
    if k > 1:
        tmp[0, 0:k - 1].set(ids[1:k])
        ids[0:k - 1].set(tmp[0, 0:k - 1])
    b.emit(I.vop(I.V_COPY, ids.base + k - 1, pr.base + 1, 0, 1, 1, 0, 0, 0, comment="id"))
    b.end_loop(lp)
    if mo.rule == "softmax":                            # column 0 holds the largest
        e = ol.exp2((wt - wt[0:1]) * ol.LOG2E)
        wt.set(e * ol.recip(ol.sum(e)))
    elif mo.norm:
        wt.set(wt * ol.recip(ol.sum(wt) + 1e-6))
    if mo.scale != 1.0:
        wt.set(wt * float(mo.scale))
    gid = ol.empty((k,))
    gid.set(ids + ol.load(lw.gbase))                    # global ids: j * E + index
    pe[OFF, :].set(gid * 8.0)                           # their directory entries' offsets
    if y is None:
        y = ol.empty((k, H))
    for i in range(k):              # y's rows (ol.empty may pad them), the answer's words
        pe[ROW, i:i + 1].set(float(i * y.rs))
        pe[ANS, i:i + 1].set(float(4 * i))
    # the fence, then the request: seq + 1 and the ids
    seq = ol.load(Tensor(Affine(dev.mbox), (1,), (1,)))
    b.rld(r, seq, raw=True, comment="seq (bits)")
    b.waitw(word, dev.served, 0, I.C_GE, rc=r, comment="fence: served >= seq")
    ol.store(Tensor(Affine(dev.mbox + LINE), (k,), (1,)), gid)
    ol.store(Tensor(Affine(dev.mbox + 4), (1,), (1,)), ol.full((1,), float(k)))  # (its count)
    seq.set(seq + 1.0)
    ol.store(Tensor(Affine(dev.mbox), (1,), (1,)), seq)
    sh = None
    if mo.shared or beside is not None:                 # while the host streams: the work
        b.unscratch(r)                                  # that needs no routed expert
        if mo.shared:                                   # (the shared expert, added last)
            sh = swiglu_down(xs, lw.wg, lw.wu, lw.wd) * gate[:, None]
            del gate
        if beside is not None:
            beside()
        r = b.scratch()
    # the directory: each entry's present flag
    col = lambda row: pe.base + row * pe.rs                       # noqa: E731 (column 0)
    lp = b.begin_loop(k)
    b.emit(I.rld(r, col(OFF), bound=_ebound(dev), comment="entry offset"))
    b.emit(I.ld(dev.dir + 4, col(EP), 1, ra=r, comment="entry: present"))
    rotate()
    b.end_loop(lp)
    pe[MISS, :].set(pe[EP, :] * -1.0 + 1.0)
    ex = dev.fmt.descs(DevVar("expert slot", r, align=LINE))   # (slots: LINE-aligned)

    def expert():
        """The expert whose slot is R[r], weighted, into its row of y."""
        o = swiglu_down(xe, ex.wg, ex.wu, ex.wd, act=ACTS[mo.act])
        b.check_live(o)
        b.emit(I.rld(r, col(ROW), bound=(0, (k - 1) * y.rs), comment="its row"))
        b.emit(I.vop(I.V_MUL, y.base, o.base, col(WT), 1, H, 0, 0, 0, I.B_ROW, ra=r,
                     comment="weighted, to its row"))

    for flag, wait in ((EP, False), (MISS, True)):
        lp = b.begin_loop(k)
        b.emit(I.rld(r, col(flag), bound=(0, 1), comment="missing" if wait else "present"))
        inner = b.begin_loop(0, rcount=r)               # (the count is read here: r is free)
        slot_of(r, word, col(ANS) if wait else col(OFF), dev, wait,
                (0, 4 * (k - 1)) if wait else _ebound(dev))
        expert()
        b.end_loop(inner)
        rotate()
        b.end_loop(lp)
    b.unscratch(r)
    ol.store(Tensor(Affine(dev.answer), (k,), (1,)), ol.zeros((k,)))   # (for the next request)
    acc = y[0:1, :]
    for i in range(1, k):
        acc = acc + y[i:i + 1, :]
    if sh is not None:
        acc = acc + sh
    return x + acc if residual else acc


def moe_ffn_rows(x, lw, mo: MoESpec, dev: SimpleNamespace, eps: float, beside=None,
                 residual: bool = True):
    """x + the MoE FFN of R token rows at once (x [R, H], R * k <= LINE / 4: one request row):
    a layer-major prefill run's (docs/offload.md 13). Every row routes as moe_ffn's; one
    request names the rows' R * k ids (repeats included: the host serves each once) and their
    count (the word after seq); each expert of the rows' union runs once, on every row (one MM
    of R rows: with PAIR, R <= 2 cost what one does), and each row that chose it stores its
    output, unweighted, to its place (row, rank) of dev.scratch [R * k + 1, H] (the last row a
    sink for the rows that did not); then each row sums its experts' outputs, weighted, in its
    router's order. So a row's result is moe_ffn's on that row alone, bit for bit, whatever
    the cache held. lw, dev (and dev.scratch), beside and residual as moe_ffn's.

    The union: for each row q, eq[n, j] = 1 - min(1, |id_n - id_q,j|) over every entry n of
    the request (the ids are integers: 1 where equal); its row sums say whether row q chose
    entry n's expert (found) and at which rank (the sum of j eq[n, j]), which gives the byte
    offset of entry n's output for row q in the scratch; an entry whose expert an earlier row
    chose is not computed again. One register, as moe_ffn's."""
    b = current()
    if ol.num_programs() != 1:
        raise CompileError("moe_ffn_rows runs on one slice")
    R, H, E, k = x.rows, x.cols, mo.E, mo.k
    N = R * k
    if N > LINE // 4:
        raise CompileError(f"{R} rows of {k} experts: more ids than a request row holds")
    if getattr(lw, "g_exp", None) is None:
        xs = xe = ol.quantize(rmsnorm(x, ol.load(lw.g_post), eps))
    else:
        xn = rmsnorm(x, ol.load(lw.g_post), eps)
        xs, xe = ol.quantize(xn), ol.quantize(xn * ol.load(lw.g_exp)[None, :])
        del xn
    lg = ol.dot(xs, lw.router)                          # [R, E] (+ the shared expert's gate)
    gid, wts = ol.empty((R, k), dense=True), ol.empty((R, k), dense=True)
    pr, tmp, ids = ol.empty((2,)), ol.empty((k,)), ol.empty((k,))
    gb = ol.load(lw.gbase)
    r = b.scratch()
    for q in range(R):                                  # each row's k best, as moe_ffn's
        sc, sel = ol.empty((E,)), ol.empty((E,))
        if mo.rule == "softmax":
            sc.set(lg[q, 0:E])
            sel.set(sc)
        else:
            sc.set(sigmoid(lg[q, 0:E]))
            sel.set(sc + ol.load(lw.ebias))
        wt = wts[q, :]
        lp = b.begin_loop(k)
        b.emit(I.argmax(pr.base, sel.base, 1, E, comment="moe: best"))
        b.emit(I.rld(r, pr.base + 1, bound=(0, E - 1), comment="its index"))
        b.emit(I.vop(I.V_FILL, sel.base, 0, 0, 1, 1, 0, 0, 0, I.B_SCALAR, NEG, ra=r,
                     comment="knock out"))
        if k > 1:
            tmp[0:k - 1].set(wt[1:k])
            wt[0:k - 1].set(tmp[0:k - 1])
            tmp[0:k - 1].set(ids[1:k])
            ids[0:k - 1].set(tmp[0:k - 1])
        b.emit(I.vop(I.V_COPY, wt.base + k - 1, sc.base, 0, 1, 1, 0, 0, 0, rb=r,
                     comment="weight"))
        b.emit(I.vop(I.V_COPY, ids.base + k - 1, pr.base + 1, 0, 1, 1, 0, 0, 0, comment="id"))
        b.end_loop(lp)
        del sc, sel
        if mo.rule == "softmax":                        # column 0 holds the largest
            e = ol.exp2((wt - wt[0:1]) * ol.LOG2E)
            wt.set(e * ol.recip(ol.sum(e)))
            del e
        elif mo.norm:
            wt.set(wt * ol.recip(ol.sum(wt) + 1e-6))
        if mo.scale != 1.0:
            wt.set(wt * float(mo.scale))
        gid[q, :].set(ids + gb)                         # global ids: j * E + index
    del ids, tmp
    gf = gid.reshape(1, N)[0, :]                        # the request's entries, row by row
    # pe's rows (columns: the entries; every loop over them rotates pe left by one, so entry
    # n is at column 0 in iteration n): present and computed here, missing, the directory
    # offset, and per row q the byte offset of the entry's output for q in the scratch
    EP, MISS, OFF, ANS, TG = 0, 1, 2, 3, 4
    RB, SINK = 4 * H, N * 4 * H
    pe, tpe = ol.empty((TG + R, N)), ol.empty((TG + R, N))
    rank = ol.empty((k,))
    for j in range(k):
        rank[j:j + 1].set(float(j))
    dup = ol.zeros((N,))
    for q in range(R):
        d = ol.empty((N, k))
        d.set(gf[:, None])
        eq = 1.0 - ol.minimum(ol.abs(d - gid[q, :][None, :]), 1.0)
        del d
        found = ol.sum(eq)                              # [N]: row q chose entry n's expert
        at = ol.sum(eq * rank[None, :])                 # its rank in row q
        del eq
        pe[TG + q, :].set(found * ((at + float(q * k)) * float(RB) - float(SINK)) + float(SINK))
        if q + 1 < R:                                   # later rows' entries chosen before
            dup[(q + 1) * k:N].set(dup[(q + 1) * k:N] + found[(q + 1) * k:N])
        del found, at
    pe[OFF, :].set(gf * 8.0)
    for n in range(N):                                  # each entry's word of the answer
        pe[ANS, n:n + 1].set(float(4 * n))

    def rotate():
        tpe[:, 0:N - 1].set(pe[:, 1:N])
        tpe[:, N - 1:N].set(pe[:, 0:1])
        pe[:, :].set(tpe[:, :])

    # the fence, then the request: the ids, their count, seq + 1
    word = ol.empty((1,))
    seq = ol.load(Tensor(Affine(dev.mbox), (1,), (1,)))
    b.rld(r, seq, raw=True, comment="seq (bits)")
    b.waitw(word, dev.served, 0, I.C_GE, rc=r, comment="fence: served >= seq")
    ol.store(Tensor(Affine(dev.mbox + LINE), (N,), (1,)), gf)
    ol.store(Tensor(Affine(dev.mbox + 4), (1,), (1,)), ol.full((1,), float(N)))
    seq.set(seq + 1.0)
    ol.store(Tensor(Affine(dev.mbox), (1,), (1,)), seq)
    sh = None
    if mo.shared or beside is not None:                 # while the host streams: the work
        b.unscratch(r)                                  # that needs no routed expert
        if mo.shared:                                   # (the shared expert, added last)
            g = ol.empty((R,))
            g.column().set(lg[:, E:E + 1])
            gate = sigmoid(g)
            del g
            sh = swiglu_down(xs, lw.wg, lw.wu, lw.wd) * gate[:, None]
            del gate, lg
        if beside is not None:
            beside()
        r = b.scratch()
    col = lambda row: pe.base + row * pe.rs                       # noqa: E731 (column 0)
    lp = b.begin_loop(N)                                # the directory: present flags
    b.emit(I.rld(r, col(OFF), bound=_ebound(dev), comment="entry offset"))
    b.emit(I.ld(dev.dir + 4, col(EP), 1, ra=r, comment="entry: present"))
    rotate()
    b.end_loop(lp)
    keep = 1.0 - ol.minimum(dup, 1.0)                   # entries computed here: the union
    pe[MISS, :].set((pe[EP, :] * -1.0 + 1.0) * keep)
    pe[EP, :].set(pe[EP, :] * keep)
    del keep, dup
    ex = dev.fmt.descs(DevVar("expert slot", r, align=LINE))      # (slots: LINE-aligned)

    def expert():
        """The expert whose slot is R[r] on every row; each row's output to its place."""
        o = swiglu_down(xe, ex.wg, ex.wu, ex.wd, act=ACTS[mo.act])
        b.check_live(o)
        for q in range(R):
            b.emit(I.rld(r, col(TG + q), bound=(0, SINK), comment="its place for this row"))
            ol.store(Tensor(Affine(dev.scratch) + DevVar("place", r), (H,), (1,)), o[q, :])

    for flag, wait in ((EP, False), (MISS, True)):
        lp = b.begin_loop(N)
        b.emit(I.rld(r, col(flag), bound=(0, 1), comment="missing" if wait else "present"))
        inner = b.begin_loop(0, rcount=r)               # (the count is read here: r is free)
        slot_of(r, word, col(ANS) if wait else col(OFF), dev, wait,
                (0, 4 * (N - 1)) if wait else _ebound(dev))
        expert()
        b.end_loop(inner)
        rotate()
        b.end_loop(lp)
    b.unscratch(r)
    ol.store(Tensor(Affine(dev.answer), (N,), (1,)), ol.zeros((N,)))   # (for the next request)
    out = ol.empty((R, H))
    for q in range(R):                                  # each row's sum in its router's order
        acc = None
        for i in range(k):
            y = ol.load(Tensor(Affine(dev.scratch + (q * k + i) * RB), (1, H), (H, 1))) * \
                wts[q, i:i + 1]
            acc = y if acc is None else acc + y
        out[q:q + 1, :].set(acc)
        del acc, y
    if sh is not None:
        out = out + sh
    return x + out if residual else out


# ---------------------------------------------------------------------------- expert-major
EM_RHO = 1.12               # an expert's stream from DRAM, MXU cycles a chunk (the card: 17.07 / 15.2
                            # GB/s); any value in (1, 2) orders the pass sizes alike (em_pass_order)
_TWO23 = float(1 << 23)     # FILL's immediate plus a register: the float 2^23 + R (0 <= R < 2^23)


def em_meta(k: int) -> int:
    """Words of a record's tail: the row's k ids and k weights and its norm's 1 / rms, in whole
    lines."""
    w = LINE // 4
    return -(-(2 * k + 1) // w) * w


def em_record(H: int, k: int) -> int:
    """Bytes of a chunk row's record in the expert-major scratch (docs/offload.md 13.11): the
    row's residual X [H], its k experts' outputs OUT [k, H] (rank order), the shared expert's
    output or the normed dense MLP's SH [H], then its tail (em_meta): global ids [k], weights
    [k] and the 1 / rms of the norm before the router and the experts (fp32)."""
    return 4 * ((k + 2) * H + em_meta(k))


def em_slots(layout: Layout, nbytes: int) -> list:
    """The slots a scratch of nbytes covers: the slot area's first ones, from the first slot on
    (the slots follow one another, layer after layer). The server takes them for the prefill
    and hands them back after it, with their tag beats zeroed."""
    s = layout.all_slots()
    n = -(-nbytes // layout.pitch)
    if n > len(s):
        raise ValueError(f"a scratch of {nbytes} bytes: {n} slots, the image has {len(s)}")
    return s[:n]


def em_row(dev: SimpleNamespace, row, r: int = 0) -> Affine:
    """The byte address of the record of chunk row row + r (row: a run argument or an int)."""
    return Affine(dev.em_base + r * dev.em_rec) + row * dev.em_rec


def em_x(dev: SimpleNamespace, row, R: int, H: int) -> Tensor:
    """The residual rows X of chunk rows row .. row + R - 1, [R, H] (expert-major's residual
    stream: the records' first words)."""
    return Tensor(em_row(dev, row), (R, H), (dev.em_rec // 4, 1))


def moe_prologue_rows(x, lw, mo: MoESpec, dev: SimpleNamespace, eps: float, row,
                      beside=None) -> None:
    """Expert-major's MoE prologue (docs/offload.md 13.11) of a layer-major run's R rows x [R,
    H] (chunk rows from `row`, a run argument): each row's norm and route as moe_ffn_rows'; to
    each row's record tail its k global ids and weights and the norm's 1 / rms (the expert run
    norms the row again from X and it: bit for bit); then the rows' R k ids, offset by
    dev.need_off (2 x layers x E), posted as a need line (a hint's post: the server streams
    the experts no slot holds and answers nothing; the layer's expert run waits on their
    directory entries); then to each row's SH the shared expert's output (SwiGLU of the layer
    block times sigmoid of its gate's logit), or beside()'s ([R, H]: Gemma 4's normed dense
    MLP, while the host streams). The caller stores x itself to the records' X (em_x). One
    register."""
    b = current()
    if ol.num_programs() != 1:
        raise CompileError("moe_prologue_rows runs on one slice")
    R, H, E, k = x.rows, x.cols, mo.E, mo.k
    N = R * k
    if N > LINE // 4:
        raise CompileError(f"{R} rows of {k} experts: more ids than a need line holds")
    ss = ol.sum(x * x, axis=1)                          # rmsnorm's, its 1 / rms kept
    rinv = ol.rsqrt(ss * (1.0 / H) + eps)
    del ss
    g = ol.load(lw.g_post)
    if getattr(lw, "g_exp", None) is None:              # (as moe_ffn_rows' quantize(rmsnorm))
        xs = ol.quantize((x * rinv[:, None]) * g[None, :])
    else:
        xn = (x * rinv[:, None]) * g[None, :]
        xs = ol.quantize(xn)
        del xn
    del g
    lg = ol.dot(xs, lw.router)                          # [R, E] (+ the shared expert's gate)
    gid, wts = ol.empty((R, k), dense=True), ol.empty((R, k), dense=True)
    pr, tmp, ids = ol.empty((2,)), ol.empty((k,)), ol.empty((k,))
    gb = ol.load(lw.gbase)
    r = b.scratch()
    for q in range(R):                                  # each row's k best, as moe_ffn_rows'
        sc, sel = ol.empty((E,)), ol.empty((E,))
        if mo.rule == "softmax":
            sc.set(lg[q, 0:E])
            sel.set(sc)
        else:
            sc.set(sigmoid(lg[q, 0:E]))
            sel.set(sc + ol.load(lw.ebias))
        wt = wts[q, :]
        lp = b.begin_loop(k)
        b.emit(I.argmax(pr.base, sel.base, 1, E, comment="moe: best"))
        b.emit(I.rld(r, pr.base + 1, bound=(0, E - 1), comment="its index"))
        b.emit(I.vop(I.V_FILL, sel.base, 0, 0, 1, 1, 0, 0, 0, I.B_SCALAR, NEG, ra=r,
                     comment="knock out"))
        if k > 1:
            tmp[0:k - 1].set(wt[1:k])
            wt[0:k - 1].set(tmp[0:k - 1])
            tmp[0:k - 1].set(ids[1:k])
            ids[0:k - 1].set(tmp[0:k - 1])
        b.emit(I.vop(I.V_COPY, wt.base + k - 1, sc.base, 0, 1, 1, 0, 0, 0, rb=r,
                     comment="weight"))
        b.emit(I.vop(I.V_COPY, ids.base + k - 1, pr.base + 1, 0, 1, 1, 0, 0, 0, comment="id"))
        b.end_loop(lp)
        del sc, sel
        if mo.rule == "softmax":                        # column 0 holds the largest
            e = ol.exp2((wt - wt[0:1]) * ol.LOG2E)
            wt.set(e * ol.recip(ol.sum(e)))
            del e
        elif mo.norm:
            wt.set(wt * ol.recip(ol.sum(wt) + 1e-6))
        if mo.scale != 1.0:
            wt.set(wt * float(mo.scale))
        gid[q, :].set(ids + gb)                         # global ids: j * E + index
    b.unscratch(r)
    del ids, tmp, pr, gb
    tail = 4 * H * (k + 2)
    for q in range(R):                                  # the record tails
        t = em_row(dev, row, q) + tail
        ol.store(Tensor(t, (k,), (1,)), gid[q, :])
        ol.store(Tensor(t + 4 * k, (k,), (1,)), wts[q, :])
        ol.store(Tensor(t + 8 * k, (1,), (1,)), rinv[q:q + 1])
    del wts, rinv
    seq = MB.wait_served(dev.mbox)                      # the need line (MB.post, its count)
    ol.store(Tensor(Affine(dev.mbox + LINE), (N,), (1,)),
             gid.reshape(1, N)[0, :] + float(dev.need_off))
    ol.store(Tensor(Affine(dev.mbox + 4), (1,), (1,)), ol.full((1,), float(N)))
    seq.set(seq + 1.0)
    ol.store(Tensor(Affine(dev.mbox), (1,), (1,)), seq)
    del gid, seq
    if mo.shared:
        gs = ol.empty((R,))
        gs.column().set(lg[:, E:E + 1])
        gate = sigmoid(gs)
        del gs
        sh = swiglu_down(xs, lw.wg, lw.wu, lw.wd) * gate[:, None]
    elif beside is not None:
        del lg
        sh = beside()
    else:
        return
    for q in range(R):
        ol.store(Tensor(em_row(dev, row, q) + 4 * H * (k + 1), (1, H), (H, 1)), sh[q:q + 1, :])


def moe_combine_rows(x, mo: MoESpec, dev: SimpleNamespace, row, residual: bool = True):
    """Expert-major's end of a MoE layer (docs/offload.md 13.11), in the next layer's run (or
    the prefill's head): for each of the R rows x [R, H] from chunk row `row`, its experts'
    outputs weighted and summed in its router's order, from its record (OUT, the tail's
    weights), then the shared expert's output (SH) added, and x + that: moe_ffn_rows' sums,
    bit for bit. residual=False: the routed sum alone (Gemma 4 norms it beside the dense MLP's,
    which SH holds)."""
    R, H, k = x.rows, x.cols, mo.k
    out = ol.empty((R, H))
    for q in range(R):                                  # each row's sum in its router's order
        t = em_row(dev, row, q)
        wt = ol.load(Tensor(t + 4 * H * (k + 2) + 4 * k, (k,), (1,)))
        acc = None
        for i in range(k):
            y = ol.load(Tensor(t + 4 * H * (1 + i), (1, H), (H, 1))) * wt[i:i + 1]
            acc = y if acc is None else acc + y
        out[q:q + 1, :].set(acc)
        del acc, y, wt
    if not residual:
        return out
    if mo.shared:
        out = out + ol.load(Tensor(em_row(dev, row) + 4 * H * (k + 1), (R, H),
                                   (dev.em_rec // 4, 1)))
    return x + out


def em_mxu_cost(cfg, wformat: str, m: int) -> int:
    """An expert pass's MXU time on m rows, in cycles a chunk of its weights (docs/isa.md MM):
    4-bit weights a chunk a cycle with PAIR (2 m <= MCOLS), half a chunk without; int8 a chunk a
    cycle; the ACT RAM's rows past MCOLS replay each chunk, ceil(m / MCOLS) groups."""
    g = -(-m // cfg.MCOLS)
    if Q.mxu_wf(wformat) == 0:
        return g
    return 1 if cfg.PAIR and 2 * m <= cfg.MCOLS else 2 * g


def em_pass_order(cfg, wformat: str) -> list:
    """Every pass size m <= ACT_ROWS, the best first: the least time an entry,
    max(EM_RHO, em_mxu_cost(m)) / m, the fewest rows on a tie (docs/offload.md 13.17). A pass
    streams its expert's weights once, so it costs the stream until the MXU takes longer; the
    stream is a little under a chunk a cycle (EM_RHO), and any value in (1, 2) orders them the
    same."""
    act = cfg.act_rows
    return sorted(range(1, act + 1),
                  key=lambda m: (max(EM_RHO, em_mxu_cost(cfg, wformat, m)) / m, m))


def em_pass_rows(cfg, wformat: str) -> int:
    """C, the entries of one expert an expert-major pass takes: em_pass_order's first. 4 on the
    card (MCOLS 4, PAIR: 2 rows at the stream's rate, 4 at twice its MXU time), 16 at MCOLS 16
    with ACT_ROWS >= 16, 4 for int8 at MCOLS 4 (a chunk a cycle for every row)."""
    return em_pass_order(cfg, wformat)[0]


def em_short_rows(cfg, wformat: str, c: int) -> int:
    """The most rows a pass of C entries computes for less MXU time than C's (0: none): an
    expert's last pass, if it holds no more entries, computes only that many (on the card 2 of
    C = 4)."""
    cost = em_mxu_cost(cfg, wformat, c)
    return max([m for m in range(1, c) if em_mxu_cost(cfg, wformat, m) < cost], default=0)


def moe_expert_run(lw, mo: MoESpec, dev: SimpleNamespace, rows, entries, C: int = 2,
                   S: int = 0) -> None:
    """Expert-major's expert run of one MoE layer (docs/offload.md 13.11, 13.17), after the
    layer's mixer runs over a chunk's `rows` rows (run arguments: rows, and entries = rows x k):
    every (row, rank) entry's expert once on its row, each expert in passes of up to C of its
    entries, each pass streaming the expert's weights once (C: em_pass_rows).

    1. The tables: per row (a loop of `rows`) its k ids from its record tail, and for each of
       its entries its row and its output's place (the record's OUT row of its rank, in lines).
    2. The bucketing (a loop of `entries`): an entry joins its expert's open pass while it has
       room, else opens one (expert, a short flag 0, and the entry in all C places: a pass of
       fewer entries computes its first row again and stores the same output again); the
       expert's free places, next place and room flag (min(free, 1)) count it, and LOOPs
       counted by the flag and 1 - flag are the branches. Registers reach the tables as
       FILL's 2^23 + R, made exact after. The pass table holds entries / C + E passes at most.
       With S (em_short_rows: a smaller pass's MXU costs less) a loop over the experts then
       flags each one's last pass that holds at most S entries (free >= C - S); the others'
       flags go to a sink row.
    3. The passes (a loop of the passes counted): WAITW on the expert's directory entry (its
       slot word != 0: the server writes an entry after its expert's data and tag, and keeps
       a needed expert's slot until this run has read it), its slot's tag zeroed (as slot_of),
       then the pass's C rows (S for a flagged pass, a LOOP of count flag and one of 1 - flag)
       normed again from their X and 1 / rms and quantized (moe_ffn_rows' expression),
       swiglu_down at the slot, each output row to its place. Each row's quantization and MM
       outputs are its own (docs/isa.md: one exact integer an output), so the outputs are
       C = 2's bit for bit.

    The entry wait stops at MB.TIMEOUT with an error the host sees: the prefill never runs on
    without an expert."""
    b = current()
    if ol.num_programs() != 1:
        raise CompileError("moe_expert_run runs on one slice")
    if not 1 <= C or not 0 <= S < C:
        raise CompileError(f"expert run passes of {C} entries, short ones of {S}")
    E, k, H = mo.E, mo.k, dev.fmt.H
    N, REC, PW = dev.em_rows * k, dev.em_rec, C + 2
    PM = N // C + E
    tail = 4 * H * (k + 2)
    # 1. the tables: ids (global), each entry's row, each entry's place (lines from the base)
    L, QR, PL = ol.empty((N,)), ol.empty((N,)), ol.empty((N,))
    rr = b.arg_reg(rows, 1)
    rd, rt, rq, rpl = (b.scratch() for _ in range(4))
    lp = b.begin_loop(0, rcount=rr)
    b.emit(I.ld(dev.em_base + tail, L.base, k, ra=rd, rb=rt, comment="the row's ids"))
    b.emit(I.vop(I.V_FILL, QR.base, 0, 0, 1, k, 0, 0, 0, I.B_SCALAR, _TWO23, ra=rt, rd=rq,
                 comment="its entries' row"))
    for i in range(k):
        b.emit(I.vop(I.V_FILL, PL.base + i, 0, 0, 1, 1, 0, 0, 0, I.B_SCALAR,
                     _TWO23 + (1 + i) * H // 16, ra=rt, rd=rpl, comment="its place (lines)"))
    b.emit(I.addi(rd, rd, REC, comment="next record"))
    b.emit(I.addi(rt, rt, k, comment="its entries"))
    b.emit(I.addi(rq, rq, 1, comment="next row"))
    b.emit(I.addi(rpl, rpl, REC // LINE, comment="its record (lines)"))
    b.end_loop(lp)
    for r in (rd, rt, rq, rpl):
        b.unscratch(r)
    b.release_arg(rows)
    QR.set(QR - _TWO23)
    PL.set(PL - _TWO23)
    gb = ol.load(lw.gbase)
    L.set(L - gb)                                       # the layer's own indices
    # 2. the bucketing
    FREE, ROOM, POS = ol.zeros((E,)), ol.zeros((E,)), ol.zeros((E,))
    BASE = ol.full((E,), _TWO23 + PM * PW)              # (the sink row, for the unused)
    PT = ol.empty((PM + 1, PW), dense=True)
    re = b.arg_reg(entries, 1)
    rp, rnp, r1, r2, r3, r4 = (b.scratch() for _ in range(6))
    lp = b.begin_loop(0, rcount=re)
    b.emit(I.rld(r1, L.base, ra=rp, comment="its expert"))
    b.emit(I.rld(r2, ROOM.base, ra=r1, comment="room in its open pass"))
    b.emit(I.rld(r3, ROOM.base, ra=r1, mul=-1, comment="-room"))
    join = b.begin_loop(0, rcount=r2)
    b.emit(I.rld(r4, POS.base, ra=r1, comment="its next place (2^23 + word)"))
    b.emit(I.vop(I.V_FILL, PT.base - (1 << 23), 0, 0, 1, 1, 0, 0, 0, I.B_SCALAR, _TWO23,
                 ra=r4, rd=rp, comment="join: the entry"))
    b.emit(I.vop(I.V_ADD, POS.base, POS.base, 0, 1, 1, 0, 0, 0, I.B_SCALAR, 1.0, ra=r1, rb=r1,
                 comment="next place"))
    b.emit(I.vop(I.V_ADD, FREE.base, FREE.base, 0, 1, 1, 0, 0, 0, I.B_SCALAR, -1.0, ra=r1,
                 rb=r1, comment="free - 1"))
    b.emit(I.vop(I.V_MIN, ROOM.base, FREE.base, 0, 1, 1, 0, 0, 0, I.B_SCALAR, 1.0, ra=r1,
                 rb=r1, comment="room = min(free, 1)"))
    b.end_loop(join)
    opn = b.begin_loop(1, rcount=r3)                    # 1 - room
    b.emit(I.vop(I.V_FILL, PT.base, 0, 0, 1, 1, 0, 0, 0, I.B_SCALAR, _TWO23, ra=rnp, rd=r1,
                 comment="open: its expert"))
    b.emit(I.vop(I.V_FILL, PT.base + 1, 0, 0, 1, 1, 0, 0, 0, I.B_SCALAR, _TWO23, ra=rnp,
                 comment="open: not short"))
    b.emit(I.vop(I.V_FILL, PT.base + 2, 0, 0, 1, C, 0, 0, 0, I.B_SCALAR, _TWO23, ra=rnp,
                 rd=rp, comment="open: the entry in every place"))
    b.emit(I.vop(I.V_FILL, BASE.base, 0, 0, 1, 1, 0, 0, 0, I.B_SCALAR, _TWO23, ra=r1, rd=rnp,
                 comment="its open pass"))
    b.emit(I.vop(I.V_FILL, POS.base, 0, 0, 1, 1, 0, 0, 0, I.B_SCALAR, _TWO23 + 3, ra=r1,
                 rd=rnp, comment="its next place"))
    b.emit(I.vop(I.V_FILL, FREE.base, 0, 0, 1, 1, 0, 0, 0, I.B_SCALAR, float(C - 1), ra=r1,
                 comment="free"))
    b.emit(I.vop(I.V_FILL, ROOM.base, 0, 0, 1, 1, 0, 0, 0, I.B_SCALAR, float(min(C - 1, 1)),
                 ra=r1, comment="room"))
    b.emit(I.addi(rnp, rnp, PW, comment="next pass"))
    b.end_loop(opn)
    b.emit(I.addi(rp, rp, 1, comment="next entry"))
    b.end_loop(lp)
    NP = ol.full((1,), _TWO23)
    b.emit(I.vop(I.V_FILL, NP.base, 0, 0, 1, 1, 0, 0, 0, I.B_SCALAR, _TWO23, rd=rnp,
                 comment="passes x PW"))
    for r in (rp, rnp, r1, r2, r3, r4):
        b.unscratch(r)
    b.release_arg(entries)
    del L, ROOM, POS
    if S:                                               # each expert's last pass: short?
        SH = ol.empty((E,))
        b.emit(I.vop(I.V_SUB, SH.base, FREE.base, 0, 1, E, 0, 0, 0, I.B_SCALAR,
                     float(C - S - 1), comment="free - (C - S - 1)"))
        b.emit(I.vop(I.V_MAX, SH.base, SH.base, 0, 1, E, 0, 0, 0, I.B_SCALAR, 0.0))
        b.emit(I.vop(I.V_MIN, SH.base, SH.base, 0, 1, E, 0, 0, 0, I.B_SCALAR, 1.0,
                     comment="short: free >= C - S"))
        re, r4 = b.scratch(), b.scratch()
        lp = b.begin_loop(E)
        b.emit(I.rld(r4, BASE.base, ra=re, comment="its last pass (2^23 + word)"))
        b.emit(I.vop(I.V_ADD, PT.base + 1 - (1 << 23), SH.base, 0, 1, 1, 0, 0, 0, I.B_SCALAR,
                     _TWO23, ra=r4, rb=re, comment="its short flag"))
        b.emit(I.addi(re, re, 1, comment="next expert"))
        b.end_loop(lp)
        b.unscratch(re)
        b.unscratch(r4)
        del SH
    del FREE, BASE
    PT.set(PT - _TWO23)
    PT[:, 0:1].set(PT[:, 0:1] + gb)                     # global ids
    NP.set((NP - _TWO23) * (1.0 / PW))
    del gb
    # 3. the passes
    g = ol.load(lw.g_post)
    ge = ol.load(lw.g_exp) if getattr(lw, "g_exp", None) is not None else None
    word = ol.empty((1,))
    rq, rs, rn = b.scratch(), b.scratch(), b.scratch()

    def rows_of_pass(M: int) -> None:
        """The pass's first M places: their rows normed, quantized, through the expert, out."""
        xr = ol.empty((M, H), dense=True)
        rec = DevVar("record", rn, align=LINE)
        for j in range(M):                              # (moe_ffn_rows' expressions, a row at a
            b.emit(I.rld(rn, PT.base + 2 + j, ra=rq, comment="entry"))      # time: TMEM for C
            b.emit(I.rld(rn, QR.base, ra=rn, mul=REC, comment="its row's record"))  # rows of
            x = ol.load(Tensor(Affine(dev.em_base) + rec, (1, H), (H, 1)))       # the 26B's)
            ri = ol.load(Tensor(Affine(dev.em_base + tail + 8 * k) + rec, (1,), (1,)))
            xr[j:j + 1, :].set((x * ri[:, None]) * g[None, :])
            del x, ri
            if ge is not None:
                xr[j:j + 1, :].set(xr[j:j + 1, :] * ge[None, :])
        xe = ol.quantize(xr)
        del xr
        ex = dev.fmt.descs(DevVar("expert slot", rs, align=LINE))
        o = swiglu_down(xe, ex.wg, ex.wu, ex.wd, act=ACTS[mo.act])
        b.check_live(o)
        for j in range(M):
            b.emit(I.rld(rn, PT.base + 2 + j, ra=rq, comment="entry"))
            b.emit(I.rld(rn, PL.base, ra=rn, mul=LINE, comment="its place"))
            ol.store(Tensor(Affine(dev.em_base) + DevVar("place", rn, align=LINE), (1, H),
                            (H, 1)), o[j:j + 1, :])
        del o, xe

    b.rld(rs, NP, comment="passes")
    lp = b.begin_loop(0, rcount=rs)                     # (the count is read here: rs is free)
    b.emit(I.rld(rs, PT.base, ra=rq, mul=8, comment="its directory entry"))
    b.waitw(word, dev.dir, 0, I.C_NE, ra=rs, timeout=MB.TIMEOUT,
            comment="wait: its slot (the entry)")
    b.rld(rs, word, raw=True, bound=getattr(dev, "slots", None), comment="its slot")
    ol.store(Tensor(Affine(dev.tag) + DevVar("expert slot", rs), (1,), (1,)), ol.zeros((1,)))
    if S:
        rf, rg = b.scratch(), b.scratch()
        b.emit(I.rld(rf, PT.base + 1, ra=rq, comment="short"))
        b.emit(I.rld(rg, PT.base + 1, ra=rq, mul=-1, comment="-short"))
        short = b.begin_loop(0, rcount=rf)
        rows_of_pass(S)
        b.end_loop(short)
        full = b.begin_loop(1, rcount=rg)               # 1 - short
        rows_of_pass(C)
        b.end_loop(full)
        b.unscratch(rf)
        b.unscratch(rg)
    else:
        rows_of_pass(C)
    b.emit(I.addi(rq, rq, PW, comment="next pass"))
    b.end_loop(lp)
    for r in (rq, rs, rn):
        b.unscratch(r)
