# MiniCPM-V 4.7 (35B-A3B) on openTPU

[openbmb/MiniCPM-V-4.7-35B-A3B](https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B) (Apache-2.0,
BF16 safetensors, transformers >= 5.18 `minicpmv4_7`, `AutoModelForImageTextToText`) is a
vision-language model:
- a SigLIP2 vision tower with LLaVA-UHD v4 token compression inside the ViT;
- a projector into the language model;
- a Qwen3.5-MoE language model (`qwen3_5_moe_text`), the architecture openTPU already runs as
  `qwen35-35b`.

This document covers:
- the model as Hugging Face's code defines it: transformers v5.18.0's `minicpmv4_7`, which
  inherits `minicpmv4_6`; the checkpoint has no remote code;
- how it maps onto openTPU: one bitstream, no new hardware op, the vision tower's math on the card;
- the validation and card plan.

Revision checked: f8ddcae6f94778efa90dca75d7f2714916a8132d (2026-10-13). The model is `minicpmv-35b`
in `opentpu.llm.MODELS`.

## 1. The checkpoint

16 safetensors shards, 70.43 GB (1,152 tensors):

| part | tensors | parameters | bf16 |
|---|---|---|---|
| LLM experts (`mlp.experts.gate_up_proj`, `down_proj`, 40 layers x 256) | 80 | 32.21 B | 64.4 GB |
| LLM embedding + LM head (untied, 248,144 x 2048 each) | 2 | 1.02 B | 2.03 GB |
| LLM layers without experts (attention, DeltaNet, shared expert, router, norms) | ~1,000 | 1.43 B | 2.86 GB |
| vision tower + projector (`model.vision_tower.*`, `model.merger.*`) | 61 | 0.553 B | 1.11 GB |

The language model's tensors are `model.language_model.*` and `lm_head.weight`. Under
`load_weights`' rename to `model.*`, these are the names Qwen3.5-35B-A3B's checkpoint gives.

There are no `mtp.*` tensors, although the config says `mtp_num_hidden_layers: 1`. So MTP
speculative decode (docs/mtp.md) is not available for this model. OpenBMB's DSpark draft model
is a separate checkpoint, left for later.

## 2. The language model against qwen35-35b

The text configs differ in one field that matters: `vocab_size` is 248,144 here against
Qwen3.5-35B-A3B's 248,320. Everything else is the same:
- 40 layers, `(linear x 3, full) x 10`, hidden 2048;
- attention: 16 query heads and 2 KV heads of 256, an output gate, RoPE on 64 of the 256
  dimensions at theta 1e7;
- DeltaNet: 16 key heads and 32 value heads of 128, a conv of 4 taps;
- 256 experts top-8 of width 512, normalized softmax top-k, plus a sigmoid-gated shared expert
  of width 512;
- untied embeddings, RMSNorm eps 1e-6.

The other config differences are `dtype`, `eos_token_id`, `mlp_only_layers: []`,
`output_router_logits` and `tie_word_embeddings`. None of them changes the math: Spec.from_hf
defaults eos to (248046, 248044), which matches `generation_config.json`'s `[248044, 248046]`.
HF's MoE block (`Qwen3_5MoeSparseMoeBlock`) and router (softmax, top-8, renormalized) are the
ones openTPU's `moe.route` (`rule="softmax"`, `norm=True`) follows.

**Vocabulary.** 248,144 = 1,938 x 128 + 80, so the LM head's last 128-row block is partial.
The head streams vocabulary rows (`_lm_head`: chunks of `HEAD_CHUNK` rows, the last one short).
A row count that is not a multiple of D is not a K dimension: MM's K is the hidden size, 2048.
The tiny Qwen3.5 tests already run a 1,000-token vocabulary (1,000 = 7 x 128 + 104). The text
path's test checks it at the partial block: the device logits of the real vocabulary must be
bit-identical to those of the same head padded with zero rows to a multiple of D (the last real
id included). If it ever fails, the fallback is that padded head plus a -inf mask on the padding
ids, so they can never be sampled.

**M-RoPE.** `rope_parameters` has `mrope_interleaved: true`, `mrope_section [11, 11, 10]`
(section 4). The rotary dimension is 64, i.e. 32 frequencies. Frequency i takes the position of
channel T when i mod 3 = 0, of H when i mod 3 = 1, and of W when i mod 3 = 2. In HF's
`recomposition_frequencies`, H takes `slice(1, 33, 3)` and W takes `slice(2, 30, 3)`. A text
token has T = H = W = p, so text-only prompts are ordinary 1-D RoPE at p, the qwen35 tables.

**Chat template** (`chat_template.jinja`): Qwen3.5's `<|im_start|>` / `<|im_end|>` turns.
- The generation prompt opens `<think>\n`, or `<think>\n\n</think>\n\n` with
  `enable_thinking=false`. A past reply's thinking is dropped unless it follows the last user
  query.
- An image part of a message renders as `<|image_pad|>` (248056), joined to the text parts with
  `\n`. A video part renders as `<|video_pad|>` (248057). A system message may not hold images.
- Tools follow Qwen3.5's XML format.

## 3. The vision side

### 3.1 The processor (host)

`MiniCPMV4_6ImageProcessor` with `preprocessor_config.json`: max_slice_nums 9, mean = std = 0.5;
defaults scale_resolution 448, patch 14, slice_mode on, downsample "16x", use_image_id on.
Images are converted to RGB uint8 [3, H, W].

- **Grid.** `get_sliced_grid`: ratio = H W / 448^2 and m = min(ceil(ratio), 9). With m <= 1
  there are no slices. Otherwise the grid (rows, cols), rows x cols in {m - 1, m, m + 1} minus
  {1} minus > 9, minimizes |log(W / H) - log(cols / rows)|; on a tie the one with more rows wins.
- **Thumbnail ("source").** `find_best_resize(H, W)`. When H W > 448^2, or when there is no
  grid (`allow_upscale`):
  - h = int(448 / sqrt(W / H)), w = int(h W / H);
  - then each side goes to `ensure_divide(x, 56) = max(round(x / 56) * 56, 56)`: a multiple of
    4 patches, because the two 2x2 merges need it.

  So the thumbnail is always about 448^2 pixels: about 1,024 patches, 64 LLM tokens.
- **Slices.** `get_refine_size`:
  - the image is first resized to the slice size times the grid;
  - the slice size is `find_best_resize` of (ensure_divide(H, rows) / rows,
    ensure_divide(W, cols) / cols), with upscaling allowed;
  - then the image is cut into rows x cols slices, row-major.

  Each slice is also about 448^2 pixels.
- **Resample.** torchvision `resize`, bicubic, antialias on, on the uint8 tensor (uint8 out).
  Then x / 255, then (x - 0.5) / 0.5.
- **Patches** (`reshape_by_patch`). Each crop's 14 x 14 blocks are taken row-major, each block
  flattened (channel, row, column): 588 values. The crops are concatenated, thumbnail first.
  `target_sizes` holds each crop's (h, w) in patches. The patch embedding (a 14 x 14, stride-14
  Conv2d) is then a matrix product of these rows with `patch_embedding.weight` reshaped to
  [1152, 588].
- **Placeholder text** (`replace_image_token`, `_prepend_local_ids`). Each `<|image_pad|>` of
  the prompt becomes `<image_id>N</image_id>` (N the image's index in its sample), then
  `<image>`, then thumbnail_h thumbnail_w / 16 copies of `<|image_pad|>`, then `</image>`.
  With slices it is followed by the slice rows joined by `\n`, each row cols x (`<slice>` +
  slice_h slice_w / 16 pads + `</slice>`).
- **Ids:** `<image>` 248078, `</image>` 248079, `<slice>` 248088, `</slice>` 248089,
  `<image_id>` 248090, `</image_id>` 248091, `\n` 198. `mm_token_type_ids` is 1 exactly on
  248056.

A 448 x 448 image is one crop of 32 x 32 patches, 64 image tokens and 69 prompt tokens with its
markers. A large image with 9 slices is about 640 image tokens.

### 3.2 The vision tower (`MiniCPMV4_7VisionModel`, per crop: attention never crosses crops)

SigLIP2: hidden 1152 (16 heads of **72**), MLP 4304, 27 layers, gelu_pytorch_tanh, LayerNorm
with bias, eps 1e-6.

1. `x = P Wp^T + bp + E_pos[pid]`. P is the crop's patch rows [N, 588]. `E_pos` is a 70 x 70
   table (image_size 980 / 14). The NaViT position id is `pid = 70 bucket(i / h) +
   bucket(j / w)`, where bucket(f) counts the boundaries 1/70, 2/70, ... at or below f, in
   float32 (`get_vision_nearest_position_ids`).
2. Layers 0 .. 6 on the N tokens. Each layer is
   `x += out(attn(LN1(x)))`, then `x += fc2(gelu_tanh(fc1(LN2(x))))`. Every linear has a bias.
   The attention is bidirectional over the crop's tokens, scale 72^-0.5.
3. **The window merger** after layer 6 (`insert_layer_id`, `MiniCPMV4_7ViTWindowAttentionMerger`):
   - y = LN1(x);
   - attention within each 2x2 window of the patch grid (4 tokens, same q/k/v/out shapes as a
     layer);
   - x = x + attn;
   - each window's 4 rows, in row-major order within the window, are concatenated:
     z [N/4, 4608];
   - out = linear_2(gelu_tanh(linear_1(LN(z)))) + mean of the window's 4 rows, with linear_1
     4608 -> 17216 and linear_2 17216 -> 1152.

   The output is the half-resolution grid, row-major.
4. Layers 7 .. 26 on the N / 4 tokens, then `post_layernorm`.
5. **The merger** (`MiniCPMV4_7Merger`, merger_times 1, `DownsampleMLP`):
   - 2x2 windows of the half grid are concatenated as above, into [N/16, 4608];
   - LN(4608, eps 1e-6);
   - linear 4608 -> 4608, then `nn.GELU()` (the **exact erf** GELU, not tanh);
   - linear 4608 -> 2048.

   That gives one LLM embedding per 4x4 patches: the rows that replace the crop's
   `<|image_pad|>` tokens, in order.

"4x" mode skips step 3 (and the merger merges 2x2 once), 4 times the tokens. openTPU runs
the default 16x.

Parameters:
- 27 layers of 15.24 M each;
- patch embedding and position table 6.3 M;
- window merger 104.5 M;
- merger 30.7 M.

That is 553 M in all, 0.55 GB in int8.

### 3.3 Canvas M-RoPE (`get_rope_index`, `mrope_mode: "canvas"`)

Positions are triples (t, h, w).
- Text tokens take p, p, p, continuing a 1-D counter.
- A *group* is a run of frames separated only by markers. A *frame* is one image: its
  thumbnail, its slices and the markers around them.
- A frame starts at its `<image>` token with o, the counter's value: the canvas origin.
- The canvas size, in LLM tokens:
  - without slices, the thumbnail's grid / 4;
  - with slices, rows x slice_h by cols x slice_w (slice grid / 4).
- Every token of the frame first gets (o, o, o). Its markers, including a trailing `\n` (id 198)
  that closes the frame, keep that unless named below.

| token | t | h | w |
|---|---|---|---|
| `<image>` | o | max(o - 1, 0) | max(o - 1, 0) |
| thumbnail (i, j) | o | o + round(linspace(0, canvas_h - 1, th)[i]) | o + round(linspace(0, canvas_w - 1, tw)[j]) |
| `</image>` | o | o + canvas_h | o + canvas_w |
| slice k = (r, c), token (i, j) | o | o + r sh + i | o + c sw + j |
| `<slice>` of k | o | o + r sh | o + c sw |
| `</slice>` of k | o | o + r sh + sh - 1 | o + c sw + sw - 1 |
| `\n` after slice row r | o | o + (r + 1) sh - 1 | o + cols sw |

After the frame, the counter is o + max(canvas_h, canvas_w) + 1.

Notes:
- Without slices the linspace is an arange.
- The rounding is torch's, float32 linspace, round half to even.
- The text after the image continues 1-D. Decode adds `rope_deltas = max position + 1 -
  sequence length`, which is at most 0: the k-th generated token is at slot n + k and position
  n + k + delta.
- The causal mask is by slot, unchanged.

The host port of this is a direct transcription of `get_rope_index` (integer index arithmetic,
like a tokenizer's). It is checked against HF's function on prompts with and without slices,
and with several images.

## 4. Mapping onto openTPU

### 4.1 Text path (step 2)

- `load_spec`: `model_type` `minicpmv4_7` gives `qwen35.Spec.from_hf`, which already reads
  `text_config`.
- `load_weights` and `LazyWeights` also skip `model.vision_tower.` and `model.merger.` (they skip
  `model.visual.` today). The vision side has its own loader.
- The formats are the qwen35 family's (streamed.FORMATS): fp4 experts, attention / DeltaNet /
  shared expert nvfp4, int8 head, int8 embedding gathered on the device. Validation then checks
  dKL on this checkpoint (section 5).
- The pool is MX, built like `pool-q35-mx` (tools/offload/pack_pool.py, `moe.pool_key` with this
  checkpoint's identity) on hwe-bench. 10,240 experts, about 14.9 GB.
- The slim checkpoint for the card is every tensor but `mlp.experts.*` (the pool holds every
  expert): about 6.0 GB in bf16 (4.9 GB LLM, 1.1 GB vision), plus the tokenizer and configs.

With these, text-only prompts are qwen35-35b's programs over another checkpoint: same kernels,
same bitstream, nothing new on the card.

### 4.2 Vision tower on the card (step 3)

Rule: the host only decodes, resizes, slices, normalizes and cuts patches (3.1), and computes
integer indices (the NaViT position ids, the M-RoPE positions). Every multiply-add of the tower
and projector is on the card.

**Layout.** The vision weights are QTensors in the model's image (`minicpmv.VisionParts`), one
block per encoder layer.
- **Heads.** Every attention head is zero-padded from 72 to 128 (one D block): Wq, Wk, Wv get
  zero rows and zero bias, and Wout gets zero columns. That makes q, k, v [N, 2048] and K/V
  heads of d = 128. The scale stays 72^-0.5, given to the quantizer as the attention kernel takes
  it, so the scores are those of 72-wide heads.
- **Padding** 4304 -> 4352 and 17216 -> 17280 (zero rows of fc1 / linear_1 and zero bias,
  zero columns of fc2 / linear_2; gelu_tanh(0) = 0), and the patch's 588 -> 640.
- **Formats.** int8 to start; fp4 / nvfp4 per kind after the cosine check (5).

**Kernels** (generic, `opentpu/kernels/vit.py`; usable by any pre-norm ViT):
- `layernorm(x, w, b)`: mean = RSUM / n, xc = x - mean (ROW), r = rsqrt(RSSQ(xc) / n + eps),
  then xc r w + b. 6 VOPs, no new op.
- A linear with a bias is the MM plus one VOP add (COL mode). The bias is fp32, not folded
  into an int8 block.
- **Bidirectional attention** over a crop. The crop's K and V are appended to a vision KV
  scratch (`KVDesc`, capacity >= N, positions = token index; int8 K with fp32 row scale, V^T
  tiles, as the LLM's cache). Then each head's queries, in groups of MCOLS rows, run
  `attention._attend_heads` with an int length N: no mask, exactly bidirectional. The tail block
  is padded as today.
- **Window attention.** For a window in grid order, its 4 queries are two loads of 2 contiguous
  rows of the Q scratch. Its keys are `attention.Blocks([(r0, 2, None), (r1, 2, None)])`, the
  two 2-token runs in the grid-order KV scratch. Neither a reorder nor a mask is needed.
- **Window concatenation for the merges.** In the row-major [h, w, C] residual, the windows of
  window row i are two strided views: rows 2i and 2i + 1 of the grid, each [w / 2, 2C] with
  row stride 2C. So a [w / 2, 4C] tile of z is two DMA loads into its halves, with no TMEM copy.
  The window mean is 3 adds and a scale of the four C-wide quarters.
- **GELU.**
  - The ViT MLPs and the window merger use gelu_tanh (`lib.gelu_tanh`, as HF).
  - The projector's GELU is erf. It is measured first: tanh's projector-output cosine and the
    first token's dKL against exact erf, on real images (the lead's decision, 2026-10-13).
  - If it shows, `lib.gelu_erf` composes it from existing VOPs:
    max(x, 0) - |x| / 2 Q(|x| / sqrt 2), with
    Q(z) = (a1 t + ... + a5 t^5) 2^(-z^2 log2 e), t = 1 / (1 + p z) (Abramowitz-Stegun 7.1.26,
    |error| < 1.5e-7). About 20 VOPs per element, only on the projector's 4608 columns. It is a
    numerics choice, reported with its error numbers.

**Programs.** These are host-orchestrated runs over the crop, the qwen35 MoE prefill's model:
a run per phase.
- Row groups of R <= ACT_ROWS rows (16 on the card: the MX MXU replays each weight chunk for
  MCOLS rows at a time) are a hardware loop with affine addresses, so a run fits IMEM at any N.
- Per layer, three runs:
  1. LN1 + Q, K, V (Q scaled to the Q scratch; K, V appended);
  2. attention per head and query group;
  3. out + residual + LN2 + MLP (F in chunks, as `swiglu_down`) + residual.
- The window merger and the projector have their own runs.
- The programs depend on the crop's (h, w) only through compile-time loop counts. They are
  compiled per shape and kept in the program cache: a thumbnail's shape follows the aspect
  ratio, and slices share one shape.
- The residual stream is fp32 rows in DRAM, [N, 1152] (4.7 MB at N = 1024). The Q and O scratch
  are [N, 2048] fp32 each, and the KV scratch 16 x N x 128 x 2 bytes.
- The projector's output rows, [tokens, 2048] fp32, go to the image's `vis` buffer.

**Cost on the card** (MCOLS 4 x D 128, 512 MAC a cycle, MX replay so weights stream once per
16 rows, 133 MHz), per crop of 1,024 patches with padded heads:

| part | MAC |
|---|---|
| layers 0-6 | 170 G |
| window merger | 35 G |
| layers 7-26 at 256 tokens | 105 G |
| projector | 2 G |

That is about 312 G MAC, about 4.6 s per crop at full MXU rate. A 448^2 image is 1 crop and
64 tokens. A large image with 9 slices is 10 crops, about 46 s, and about 640 tokens.

`--max-slices` (HF's `max_slice_nums`) bounds the cost: 1 gives the thumbnail alone. The 72 -> 128
head padding costs about 17% of a layer's MACs. Packing the projections unpadded, padding only
the K/V appends, is a later optimization. The m8 build (MCOLS 8) halves all of it.

**DRAM.** Resident int8 vision weights take about 0.56 GB from the expert slots, about 330 slots
of 1.67 MB, roughly 8 of each layer's ~43. The decode cost of that is measured with
`streamed.probe` and the slot simulator before choosing. Options:
- fp4 / nvfp4 vision weights: about half the bytes;
- vision weights staged into the slot region only while an image is encoded. The slots refill
  on demand; the server marks them empty. That is the ambitious option, for after v1.

### 4.3 Image rows into the prompt (step 4)

- **Rows.** The 35B prefills layer-major (`qwen35_layer_run`: the chunk's residual rows in
  `xbuf`, or expert-major's records). A row's layer-0 input is today written by
  `qwen35_embed_run` from the token's embedding.
  - An image row instead gets a copy run, `vis[src:src + R] -> X[row:row + R]` (both run-time),
    and its layer runs take `embedded=True`. These are device DMA copies of the vision output,
    with no host math.
  - Text rows are unchanged, so a text-only prompt runs exactly qwen35-35b's programs.
- **M-RoPE without kernel changes.** Every kernel reads its RoPE rows from the image's
  `cos_t` / `sin_t` at the token's slot (`m.cos_t[p]`, `RunPos.pos`). For this model those
  tables become per-sequence: row s holds the cos / sin of slot s's M-RoPE position. A pristine
  copy (positions 0 .. cap - 1) stays beside them. Writing the active rows takes two device runs,
  no arithmetic beyond exact selects:
  - **Image rows.** A run reads, per row, the host's integer (t, h, w) from a small table (RLD
    into a `DevVar`, as the expert server's slot addresses). It loads the three pristine rows
    and selects per frequency by the 0/1 column masks of T / H / W. These are exact:
    x * 1 + y * 0 + z * 0 = x.
  - **Text after an image.** `active[s] = pristine[s + delta]`, a contiguous DRAM copy with delta
    a run argument, for the slots up to the capacity.
  - Text-only sequences need neither run: active = pristine, bit-exact.
  - Decode, the generate loop, chunked prefill and snapshots read the active table as they read
    the table today.
  - A chat rollback rewrites the rows past the restored prefix from the host's position map.
  - qwen35 decodes one sequence (no batched decode), so one active table per image suffices.

  Positions never exceed the slot index (the canvas compresses), so pristine rows 0 .. cap - 1
  cover every position.

### 4.4 Host (step 4)

- `opentpu/llm/minicpmv.py` holds:
  - the processor port: torch/torchvision resize as HF does it, normalization, patches, the
    NaViT position ids, the placeholder text and the canvas positions;
  - the numpy references for the vision tower and projector (fp32, checked against HF);
  - the quantized emulation;
  - the image layout's vision part and the runs' driver.
- **otpu-chat** gets `--image PATH` (repeatable; attached to the next user message) and
  `--max-slices`.
- **otpu-serve** accepts OpenAI `image_url` content parts:
  - `data:` URLs (base64 PNG / JPEG / WebP);
  - local paths only with an explicit flag;
  - no remote fetch by default.

  The image tokens are part of the prompt: a request with an image is prefilled with its
  encoding first. A KV snapshot keys on the token ids plus each image's sha256, because the
  `<|image_pad|>` ids alone do not identify the content.

## 5. Validation

**Text.** References:
- openTPU's numpy `qwen35.reference_logits` (fp32, LazyWeights per tensor);
- a layer-streaming HF reference: HF's own `Qwen3_5MoeDecoderLayer` modules, each layer's
  weights loaded, run and freed, so the 35B fits hwe-bench's 62 GB. Its first 2 layers are
  checked against the whole HF model built with 2 layers.

Checks:
- layer-wise hidden-state cosine;
- logits cosine and dKL of the emulation (production mix) and of the ISA simulator against the
  references on real chat prompts;
- the formats choice by dKL, finals at >= 2000 tokens (as the formats scans);
- greedy tokens against HF's.

**Vision.**
- The processor port: pixel_values, target_sizes and placeholder ids equal to HF's processor's,
  bit for bit, on several images and aspect ratios.
- The canvas positions equal to HF's `get_rope_index`.
- The tower and projector: the numpy reference against HF (fp32) on real images; the emulation
  and the ISA simulator against both, by cosine per stage (after layer 6, the merger, layer 26,
  the projector).
- The GELU measurement (4.2), then the first token's dKL with the device's image rows against
  HF's.

**Card.** These are prepared as scripts for the lead's sessions, never run on the card by this
work.

## 6. Card plan

On opentpu, `/data/otpu/minicpmv/` holds:
- the slim checkpoint (no experts, about 6.0 GB);
- the MX pool (about 14.9 GB with its .packed, .format and .key).

Relays go hwe-bench -> Mac -> opentpu. Nothing large goes on /home. Pool and checkpoint
sha256 sums are recorded beside them.

The first session is text-only: the qwen35-35b qualification's prompts on this checkpoint. Then
one image at `--max-slices 1`, then a sliced one.

## 7. Open questions for the lead

1. The projector GELU: tanh or erf, decided by the measurement (4.2).
2. The vision weights' format and residence (4.2 "DRAM"): resident int8 first, then fp4 /
   nvfp4 and the slot cost.
3. A default `--max-slices` for chat on the card: HF's 9 costs about 46 s of vision and about
   640 prompt tokens per large image.
