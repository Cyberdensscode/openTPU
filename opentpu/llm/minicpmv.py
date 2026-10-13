"""MiniCPM-V 4.7 (openbmb/MiniCPM-V-4.7-35B-A3B) on openTPU: docs/minicpmv.md.

The language model is Qwen3.5-MoE (qwen35.Spec, streamed experts). This module holds what is
MiniCPM-V's own:
  VisionSpec        the vision tower's and projector's dimensions and the image token ids
  host side         what the host does to an image, as a tokenizer does to text (no model
                    math): decode, the slice grid, bicubic resizes, normalization, 14 x 14
                    patches; the NaViT position ids; the placeholder text; the canvas M-RoPE
                    positions (integer index arithmetic). Ports of transformers v5.18.0's
                    MiniCPMV4_6ImageProcessor, MiniCPMV4_6Processor and
                    MiniCPMV4_7Model.get_rope_index (Apache-2.0)
  vision_reference  the SigLIP2 tower, its window merger and the projector, fp32 numpy (HF's
                    math): the reference the card's runs are checked against
"""
from __future__ import annotations

import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

TOWER, MERGER = "model.vision_tower.", "model.merger."


# =============================================================================== the spec
@dataclass(frozen=True)
class VisionSpec:
    hidden: int = 1152              # SigLIP2 so400m
    heads: int = 16                 # of 72
    ffn: int = 4304
    layers: int = 27
    patch: int = 14
    side: int = 70                  # the position table's side: image_size 980 / patch 14
    eps: float = 1e-6               # every LayerNorm's (the projector's too)
    insert: int = 6                 # the window merger after this layer (insert_layer_id)
    window: int = 2                 # its window and the projector's merge: 2 x 2
    out: int = 2048                 # the language model's hidden size
    scale: int = 448                # the processor's scale_resolution
    max_slices: int = 9             # its max_slice_nums (preprocessor_config.json)
    image_token: int = 248056       # <|image_pad|>
    image_start: int = 248078       # <image>
    image_end: int = 248079
    slice_start: int = 248088       # <slice>
    slice_end: int = 248089
    newline: int = 198              # "\n"

    @property
    def head_dim(self) -> int:
        return self.hidden // self.heads

    @property
    def merged(self) -> int:
        """The patches per language-model token: two 2 x 2 merges (16x mode)."""
        return (self.window * self.window) ** 2

    @staticmethod
    def from_hf(model_dir) -> "VisionSpec":
        top = json.loads((Path(model_dir) / "config.json").read_text())
        if top.get("downsample_mode", "16x") != "16x" or top.get("merger_times", 1) != 1:
            raise ValueError(f"{model_dir}: only downsample_mode 16x, merger_times 1")
        v = top["vision_config"]
        if v.get("hidden_act", "gelu_pytorch_tanh") != "gelu_pytorch_tanh":
            raise ValueError(f"{model_dir}: vision activation {v['hidden_act']}")
        pre = Path(model_dir) / "preprocessor_config.json"
        p = json.loads(pre.read_text()) if pre.exists() else {}
        if any(x != 0.5 for x in p.get("image_mean", [0.5]) + p.get("image_std", [0.5])):
            raise ValueError(f"{model_dir}: image mean / std other than 0.5")
        return VisionSpec(hidden=v["hidden_size"], heads=v["num_attention_heads"],
                          ffn=v["intermediate_size"], layers=v["num_hidden_layers"],
                          patch=v["patch_size"], side=v["image_size"] // v["patch_size"],
                          eps=v.get("layer_norm_eps", 1e-6),
                          insert=top.get("insert_layer_id", 6),
                          out=top["text_config"]["hidden_size"],
                          scale=p.get("scale_resolution", 448),
                          max_slices=p.get("max_slice_nums", 9),
                          image_token=top["image_token_id"], image_start=top["image_start_id"],
                          image_end=top["image_end_id"], slice_start=top["slice_start_id"],
                          slice_end=top["slice_end_id"], newline=top["newline_id"])


# =============================================================================== host side
def ensure_divide(length, divisor: int) -> int:
    return max(round(length / divisor) * divisor, divisor)


def best_resize(h, w, scale: int, patch: int, upscale: bool = False) -> tuple:
    """find_best_resize: (h, w) of about scale^2 pixels at the aspect ratio when the image is
    larger (or upscale), each a multiple of 4 patches (the two 2 x 2 merges)."""
    if h * w > scale * scale or upscale:
        ar = w / h
        h = int(scale / math.sqrt(ar))
        w = int(h * ar)
    return ensure_divide(h, patch * 4), ensure_divide(w, patch * 4)


def slice_grid(h: int, w: int, max_slices: int, scale: int):
    """get_sliced_grid: (rows, cols) of the slices, or None (the thumbnail alone)."""
    log_ratio = math.log(w / h)
    multiple = min(math.ceil(w * h / (scale * scale)), max_slices)
    if multiple <= 1:
        return None
    best, err = [1, 1], float("inf")
    for n in (multiple - 1, multiple, multiple + 1):
        if n == 1 or n > max_slices:
            continue
        for rows in range(1, n + 1):
            if n % rows == 0:
                e = abs(log_ratio - math.log((n // rows) / rows))
                if e < err:
                    best, err = [rows, n // rows], e
                elif e == err and rows > best[0]:
                    best = [rows, n // rows]
    return tuple(best)


def refine_size(h: int, w: int, grid, scale: int, patch: int) -> tuple:
    """get_refine_size: the size the image is resized to before it is cut into the grid."""
    rows, cols = grid
    bh, bw = best_resize(ensure_divide(h, rows) / rows, ensure_divide(w, cols) / cols, scale,
                         patch, upscale=True)
    return bh * rows, bw * cols


def decode(src):
    """An image as HF's processor reads it: a path, a data: URL or base64 string (torchvision's
    decode_image, RGB; its EXIF orientation is not applied, as HF's for a path), or a PIL image
    (EXIF-transposed, RGB) -> uint8 [3, H, W] (a torch tensor)."""
    import torch
    try:
        from torchvision.io import ImageReadMode, decode_image
    except ImportError:                     # (no torchvision: PIL's decoder, RGB)
        decode_image = None
    if not isinstance(src, (str, Path)):
        from PIL import ImageOps
        im = ImageOps.exif_transpose(src).convert("RGB")
        return torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).contiguous()
    s = str(src)
    if Path(s).is_file():
        raw = Path(s).read_bytes()
    else:
        import base64
        raw = base64.decodebytes((s.split(",", 1)[1] if s.startswith("data:image/") else s)
                                 .encode())
    if decode_image is not None:
        return decode_image(torch.frombuffer(bytearray(raw), dtype=torch.uint8),
                            mode=ImageReadMode.RGB)
    import io
    from PIL import Image
    im = Image.open(io.BytesIO(raw)).convert("RGB")
    return torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).contiguous()


def _resize(img, h: int, w: int):
    """torchvision's bicubic resize with antialiasing of a uint8 [3, H, W] tensor (uint8), as
    HF's TorchvisionBackend; without torchvision, torch's own (the same kernel in float, rounded
    and clamped: not checked bit for bit)."""
    try:
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms.v2 import functional as tvF
        return tvF.resize(img, [h, w], interpolation=InterpolationMode.BICUBIC, antialias=True)
    except ImportError:
        import torch
        y = torch.nn.functional.interpolate(img[None].float(), size=(h, w), mode="bicubic",
                                            antialias=True, align_corners=False)[0]
        return y.round().clamp(0, 255).to(torch.uint8)


def crops(img, vs: VisionSpec, max_slices: int | None = None):
    """An image (uint8 [3, H, W]) as the processor cuts it: ([the thumbnail, slices...] as
    uint8 tensors, the grid (rows, cols) or None)."""
    max_slices = vs.max_slices if max_slices is None else max_slices
    H, W = img.shape[-2:]
    grid = slice_grid(H, W, max_slices, vs.scale) if max_slices >= 1 else None
    th, tw = best_resize(H, W, vs.scale, vs.patch, upscale=grid is None)
    out = [_resize(img, th, tw)]
    if grid is not None:
        rh, rw = refine_size(H, W, grid, vs.scale, vs.patch)
        ref = _resize(img, rh, rw)
        ph, pw = rh // grid[0], rw // grid[1]
        out += [ref[:, i:i + ph, j:j + pw] for i in range(0, rh, ph) for j in range(0, rw, pw)]
    return out, grid


def patches(crop, patch: int = 14) -> np.ndarray:
    """A crop's patch rows [h w, 3 patch^2] (fp32): its patch x patch blocks row-major, each
    (channel, row, column), normalized (x - 127.5) / 127.5 as HF's fused rescale and
    normalize (mean = std = 0.5): the patch embedding is then a matrix product."""
    x = np.asarray(crop, np.float32)
    x = (x - np.float32(127.5)) / np.float32(127.5)
    C, H, W = x.shape
    h, w = H // patch, W // patch
    return np.ascontiguousarray(x.reshape(C, h, patch, w, patch).transpose(1, 3, 0, 2, 4)
                                .reshape(h * w, C * patch * patch))


def position_ids(h: int, w: int, side: int) -> np.ndarray:
    """The NaViT position ids of an h x w patch grid in the side x side table
    (get_vision_nearest_position_ids: float32 bucketize, as torch computes it)."""
    import torch
    b = torch.arange(1 / side, 1.0, 1 / side)
    bh = torch.bucketize(torch.arange(h) / h, b, right=True)
    bw = torch.bucketize(torch.arange(w) / w, b, right=True)
    return (bh[:, None] * side + bw).flatten().numpy()


@dataclass
class ImageInput:
    """One image as the host prepares it: its crops' patch rows and (h, w) in patches (the
    thumbnail first), the slice grid, and the placeholder text that replaces its
    <|image_pad|> in the prompt."""
    patches: list
    sizes: list
    grid: tuple | None
    text: str

    @property
    def tokens(self) -> int:
        """Its language-model rows (the <|image_pad|> tokens of its placeholder)."""
        return sum(h * w for h, w in self.sizes) // 16


def prepare(src, vs: VisionSpec, max_slices: int | None = None, image_id: int | None = 0,
            pad: str = "<|image_pad|>") -> ImageInput:
    """An image (path, data URL, base64 or PIL) -> ImageInput: replace_image_token's text with
    use_image_id's <image_id>N</image_id> prefix (image_id None: none)."""
    cs, grid = crops(decode(src), vs, max_slices)
    ps = [patches(c, vs.patch) for c in cs]
    sizes = [(c.shape[-2] // vs.patch, c.shape[-1] // vs.patch) for c in cs]
    n = [h * w // vs.merged for h, w in sizes]
    text = "<image>" + pad * n[0] + "</image>"
    if grid is not None:
        row = ("<slice>" + pad * n[1] + "</slice>") * grid[1]
        text += "\n".join([row] * grid[0])
    if image_id is not None:
        text = f"<image_id>{image_id}</image_id>" + text
    return ImageInput(ps, sizes, grid, text)


def expand(text: str, images: list, pad: str = "<|image_pad|>") -> str:
    """The chat template's text with its i-th `pad` replaced by images[i].text (one per image,
    in order: get_text_with_replacements)."""
    parts = text.split(pad)
    if len(parts) - 1 != len(images):
        raise ValueError(f"{len(parts) - 1} image placeholders for {len(images)} images")
    return parts[0] + "".join(im.text + p for im, p in zip(images, parts[1:]))


def _frame_end(last: int, ids: list, markers: tuple, limit: int) -> int:
    e = last
    while e < limit and ids[e] in markers:
        e += 1
    return e


def _groups(ids: list, mm: list, vs: VisionSpec):
    """_group_visual_frames: [(group start, group end, [(thumb start, thumb end, frame end,
    slices [(start, end)])])] of one sequence; mm 1 on the image tokens."""
    markers = (vs.slice_start, vs.slice_end, vs.image_start, vs.image_end, vs.newline)
    groups, open_slices, prev_end = [], None, 0
    for modality, run in itertools.groupby(enumerate(mm), lambda t: t[1]):
        if modality == 0:
            continue
        run = list(run)
        crop = (run[0][0], run[-1][0] + 1)
        if open_slices is not None and ids[crop[0] - 1] == vs.slice_start:
            open_slices.append(crop)
        else:
            open_slices = []
            frame = (crop[0], crop[1], open_slices)
            gap = ids[prev_end:crop[0] - 1]
            if groups and all(t in markers for t in gap):
                groups[-1][1].append(frame)
            else:
                groups.append((crop[0] - 1, [frame]))
        prev_end = crop[1]
    out = []
    for g0, frames in groups:
        fr = [(a, b, _frame_end(sl[-1][1] if sl else b, ids, markers, len(ids)), sl)
              for a, b, sl in frames]
        out.append((g0, fr[-1][2], fr))
    return out


def _grid_positions(o: int, grid, canvas_h: int = 0, canvas_w: int = 0, h_off: int = 0,
                    w_off: int = 0, merge: int = 4):
    """get_vision_position_ids: [3, n] (t, h, w) of a crop placed on the canvas."""
    import torch
    gh, gw = int(grid[0]) // merge, int(grid[1]) // merge
    ch, cw = canvas_h or gh, canvas_w or gw
    hc = torch.linspace(0, ch - 1, gh).round().long() + h_off
    wc = torch.linspace(0, cw - 1, gw).round().long() + w_off
    H, W = torch.meshgrid(hc, wc, indexing="ij")
    return (torch.stack([torch.zeros_like(H), H, W]).reshape(3, -1) + o).numpy()


def rope_index(ids, sizes, vs: VisionSpec, merge: int = 4) -> tuple:
    """Canvas M-RoPE (get_rope_index for one unpadded sequence): (positions [3, T] of (t, h,
    w), delta), delta = max position + 1 - T (decode: slot s is at position s + delta). sizes:
    every crop's (h, w) in patches, in order (thumbnail, slices; image after image). A text-only
    sequence is arange(T) in each row, delta 0."""
    ids = [int(t) for t in ids]
    mm = [1 if t == vs.image_token else 0 for t in ids]
    T = len(ids)
    pos = np.tile(np.arange(T, dtype=np.int64), (3, 1))
    it = iter([tuple(int(v) for v in s) for s in sizes])
    cur, idx = 0, 0
    for g0, g1, frames in _groups(ids, mm, vs):
        if g0 > idx:
            pos[:, idx:g0] = np.arange(g0 - idx) + cur
            cur += g0 - idx
        cursor = g0
        for t0, t1, fend, slices in frames:
            thumb = next(it)
            fstart = t0 - 1
            if fstart > cursor:
                n = fstart - cursor
                pos[:, cursor:fstart] = np.arange(n) + cur
                cur += n + 1
            fend = min(fend, g1)
            o = cur
            halo = max(o - 1, 0)
            sh = sw = rows = cols = 0
            ch, cw = thumb[0] // merge, thumb[1] // merge
            if slices:
                cols = len(slices)
                for k in range(len(slices) - 1):
                    if slices[k + 1][0] - slices[k][1] > 2:
                        cols = k + 1
                        break
                rows = len(slices) // cols if cols > 0 else 1
                if rows * cols != len(slices):
                    rows, cols = 1, len(slices)
                first = next(it)
                sh, sw = first[0] // merge, first[1] // merge
                ch, cw = rows * sh, cols * sw
            pos[:, fstart:fend] = o
            pos[1:, fstart] = halo
            if t1 < fend:
                pos[1, t1], pos[2, t1] = o + ch, o + cw
            pos[:, t0:t1] = _grid_positions(o, thumb, ch, cw, merge=merge)
            for k, (s0, s1) in enumerate(slices):
                ho, wo = (k // cols) * sh, (k % cols) * sw
                sz = first if k == 0 else next(it)
                gh, gw = sz[0] // merge, sz[1] // merge
                if s0 - 1 >= fstart:
                    pos[1, s0 - 1], pos[2, s0 - 1] = o + ho, o + wo
                if s1 < fend:
                    pos[1, s1], pos[2, s1] = o + ho + gh - 1, o + wo + gw - 1
                pos[:, s0:s1] = _grid_positions(o, sz, h_off=ho, w_off=wo, merge=merge)
            for k in range(len(slices) - 1):
                a, b = slices[k][1], slices[k + 1][0]
                if b - a <= 2:
                    continue
                for nl in range(a + 1, b - 1):
                    pos[1, nl] = o + (k // cols + 1) * sh - 1
                    pos[2, nl] = o + cols * sw
            cur = o + max(ch, cw) + 1
            cursor = fend
        if cursor < g1:
            pos[:, cursor:g1] = np.arange(g1 - cursor) + cur
            cur += g1 - cursor
        idx = g1
    if idx < T:
        pos[:, idx:T] = np.arange(T - idx) + cur
    return pos, int(pos.max()) + 1 - T if T else 0


# =============================================================================== reference
def load_vision(model_dir) -> dict:
    """The vision tower's and projector's tensors (fp32 numpy), by their checkpoint names."""
    import torch
    from safetensors import safe_open
    W = {}
    for f in sorted(Path(model_dir).glob("*.safetensors")):
        with safe_open(str(f), "pt") as h:
            for k in h.keys():
                if k.startswith((TOWER, MERGER)):
                    W[k] = h.get_tensor(k).to(torch.float32).numpy()
    return W


def _ln(x, w, b, eps):
    m = x.mean(-1, keepdims=True)
    xc = x - m
    return xc / np.sqrt((xc * xc).mean(-1, keepdims=True) + eps) * w + b


def _gelu_tanh(x):
    return 0.5 * x * (1 + np.tanh(np.float32(math.sqrt(2 / math.pi)) * (x + 0.044715 * x ** 3)))


def gelu_erf(x):
    """nn.GELU(): x Phi(x), the exact erf form (the projector's)."""
    import torch
    t = torch.from_numpy(np.ascontiguousarray(x, np.float32))
    return torch.nn.functional.gelu(t).numpy()


def _lin(x, W, n):
    return x @ W[n + ".weight"].T + W[n + ".bias"]


def _attention(x, W, p, vs: VisionSpec, groups):
    """Bidirectional attention of x's rows within each group of row indices."""
    q, k, v = (_lin(x, W, p + n) for n in ("q_proj", "k_proj", "v_proj"))
    d = vs.head_dim
    o = np.zeros_like(q)
    for g in groups:
        for h in range(vs.heads):
            c = slice(h * d, (h + 1) * d)
            s = q[g, c] @ k[g, c].T * np.float32(d ** -0.5)
            s = np.exp(s - s.max(axis=1, keepdims=True))
            o[g, c] = (s / s.sum(axis=1, keepdims=True)) @ v[g, c]
    return _lin(o, W, p + "out_proj")


def windows(h: int, w: int, k: int = 2) -> np.ndarray:
    """[h w / k^2, k^2]: each k x k window's row indices of the h x w grid (row-major), windows
    row-major: the merges' order (the window merger's attention groups and concatenation, the
    projector's 2 x 2)."""
    i = np.arange(h * w).reshape(h // k, k, w // k, k).transpose(0, 2, 1, 3)
    return i.reshape(-1, k * k)


def _layer(x, W, i, vs, groups):
    p = f"{TOWER}encoder.layers.{i}."
    x = x + _attention(_ln(x, W[p + "layer_norm1.weight"], W[p + "layer_norm1.bias"], vs.eps),
                       W, p + "self_attn.", vs, groups)
    y = _ln(x, W[p + "layer_norm2.weight"], W[p + "layer_norm2.bias"], vs.eps)
    return x + _lin(_gelu_tanh(_lin(y, W, p + "mlp.fc1")), W, p + "mlp.fc2")


def vision_reference(vs: VisionSpec, W: dict, P: np.ndarray, size, capture=None,
                     gelu=gelu_erf) -> np.ndarray:
    """One crop's language-model rows [h w / 16, out] (fp32): P its patch rows [h w, 3 p^2],
    size its (h, w) in patches. capture: a dict that gets the stages' rows ("embed",
    "layer6" .. the input of the window merger, "merged" its output, "post" after
    post_layernorm, "out"). gelu: the projector's (nn.GELU's erf; lib's tanh for the
    measurement of docs/minicpmv.md 4.2)."""
    h, w = size
    cap = capture if capture is not None else {}
    f32 = np.float32
    Wp = W[TOWER + "embeddings.patch_embedding.weight"].reshape(vs.hidden, -1)
    x = (np.asarray(P, f32) @ Wp.T + W[TOWER + "embeddings.patch_embedding.bias"]
         + W[TOWER + "embeddings.position_embedding.weight"][position_ids(h, w, vs.side)])
    cap["embed"] = x
    every = [np.arange(h * w)]
    for i in range(vs.insert + 1):
        x = _layer(x, W, i, vs, every)
    cap[f"layer{vs.insert}"] = x
    m = TOWER + "vit_merger."
    win = windows(h, w, vs.window)
    x = x + _attention(_ln(x, W[m + "layer_norm1.weight"], W[m + "layer_norm1.bias"], vs.eps),
                       W, m + "self_attn.", vs, list(win))
    z = x[win].reshape(len(win), -1)                        # [h w / 4, 4 C]
    y = _ln(z, W[m + "pre_norm.weight"], W[m + "pre_norm.bias"], vs.eps)
    x = _lin(_gelu_tanh(_lin(y, W, m + "linear_1")), W, m + "linear_2") + x[win].mean(axis=1)
    cap["merged"] = x
    h2, w2 = h // 2, w // 2
    every = [np.arange(h2 * w2)]
    for i in range(vs.insert + 1, vs.layers):
        x = _layer(x, W, i, vs, every)
    x = _ln(x, W[TOWER + "post_layernorm.weight"], W[TOWER + "post_layernorm.bias"], vs.eps)
    cap["post"] = x
    z = x[windows(h2, w2, vs.window)].reshape(h2 * w2 // 4, -1)
    p = MERGER + "mlp.0."
    y = _ln(z, W[p + "pre_norm.weight"], W[p + "pre_norm.bias"], vs.eps)
    out = _lin(gelu(_lin(y, W, p + "linear_1")), W, p + "linear_2")
    cap["out"] = out
    return out
