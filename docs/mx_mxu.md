# MX MXU: accumulate in the column, scales as shifts, one epilogue per output

Question (the user, 2026-10-11): "pumped MXU with systolic and MX (verify how we could make
attention fast on this, and if we need to drop support for NVFP at all or if we can find a
pumped way to improve this) ... Prefill speed and multi user, and MOE are now as important as
decode speed." The design measured here: the MXU column accumulates a whole output in its column
end's DSP as an exact integer, with every block scale applied as a shift (power-of-two block
scales, as in OCP MX), so the fp32 epilogue that today runs per K-block in every column runs once
per output, and E lanes of it serve the whole array. Pumped DSP48E1s (2x clock,
docs/dsp_pumping.md) or packed DSP48E2s give each physical column two stationary rows.

Measured: the formats in float64 emulation (formats_scan's text and dKL, 2000 tokens, three
models); the shared epilogue in the RTL co-sim (Verilator, LiteDRAM's DDR3-1066 controller, the
card's 133.33 MHz) with a timing hack that caps the MXU's drain at E results a cycle; the column
and lanes in xsim (bit-exact) and Yosys (area; Vivado's timing is pending, section 5); the four
workloads through the workloads agent's scoreboard. Nothing here ran on the card.

**Short answer**

- **Accuracy: no loss, a small gain.** With block scales as powers of two under a per-row fp32
  and 6-bit sub-block multipliers (`fp4x6`: 4.25 bits a weight, today's bytes), the 4-bit path is
  at least as accurate as today's fp4 on all three models (paired dKL -0.47 +- 0.36,
  -0.15 +- 0.15, -1.96 +- 0.21 on Qwen3-0.6B, Qwen3.5-0.8B, LFM2.5-230M), and the int8 path is
  better than today's int8 (-0.41, -0.07, -0.12), because the activations get a power of two per
  32 under a per-row fp32 instead of an fp32 per 128. The terms the shift window drops (at most 7
  in 10^5) cost nothing measurable (-0.02 to 0). The fixed-point accumulation is exact; today's
  fp32 order is within ~1e-7 of exact. int8 weights get 2.3% smaller (8.06 bits).
- **NVFP4 does not have to be dropped.** It runs exactly on the MX column: its E4M3 scale's
  mantissa is the multiplier, its exponent the shift (9 column-end DSPs a row instead of 5, or a
  pumped column end). It is 2.6 / 2.0 points better than today's fp4 on Qwen3 / Qwen3.5 and 1.6
  worse on LFM2.5, where it degrades with position: there it helps the attention's weights (-2.6
  points in them alone) and hurts the MLP's (+1.0) (section 3.4). It costs 0.25 bit a weight
  (+5.9% of 4-bit weight bytes). By the formats rule (a point of dKL is worth 10% of speed) it
  pays on Qwen3 and Qwen3.5: a per-model, per-kind choice, as `formats.py` already makes. OCP
  MXFP4 (E8M0 per 32) is 6-8 points worse than today's fp4: not a candidate.
- **Attention stays as fast.** q.K^T gives an output per row per K row (one block at d = 128),
  but attention is bound by the softmax on the VPU, not by the MXU's results: E = MCOLS/2 lanes
  cost 0 to +0.7% in every co-sim (prefill at 4, 8 and 16 rows, at 200, 544 and 4000 tokens,
  decode at 544 and 4000, batched 4 and 8 users, MoE experts of 1-16 rows; Qwen3, Qwen3.5,
  LFM2.5, the 35B and 26B experts), E = MCOLS/4 +0.1 to +1.4% once the MLP's down-projection
  partials are long enough (a compiler rule; MoE experts on 16 rows still +9-11%), one lane +13 to
  +56%. No raw-score path or full-rate q.K^T epilogue is needed. Nor does the MX MXU make
  attention faster: that is the VPU's (pumped VPU, another agent's study).
- **Area** (synthesis estimate, section 5): at today's 4 rows the MX datapath has 31% fewer LUTs,
  42% fewer flip-flops and 264 fewer DSPs; at 8 rows 35% fewer LUTs, 47% fewer flip-flops and 528
  fewer DSPs than today's MCOLS 8. Going from 4 to 8 rows costs +2.2K LUTs and +1.8K flip-flops
  (plus 4.3K SRLs, 1.2K LUT RAM, 24 DSPs and the ACT RAM's width) instead of MCOLS 8's +12.0K and
  +14.2K. Vivado timing is pending (section 5).
- **Workloads** (co-sim, the YPCB's memory path, an MX array of 8 rows with 4 lanes against today's
  MCOLS 4): prefill -28% (Qwen3, 4 rows; a 16-row prefill -38%, -52% on a 16-row array), -17%
  (Qwen3.5), -39% (LFM2.5), -12% at 4000 tokens; batched decode of 8 users -29% a step (-41% on 16
  rows); MoE expert FFNs of 4-16 rows -45 to -48% (-68 to -72% on 16 rows); single-user decode 0.
  Scoreboard (YPCB): 8 rows 1.00x decode, 1.38x prefill, 1.74x batched, 1.00x MoE; 16 rows with 4x
  TMEM 1.00 / 2.09 / 2.78 / 1.00.
- **A side finding**: a TMEM above 64K words serializes the sequencer (16-bit TMEM ranges in its
  dependency window): +70% on an 8-row prefill at 256K words until the ranges are widened
  (section 4).

## 1. Today's numerics (the reference)

`MM` (docs/isa.md, `opentpu/isasim.py` `_mm`), per output row `j`, weight row `n`, K-block `k`
of D = 128 elements:

```
isum      = sum_i ACT[j][k*D + i] * w[i]                              exact integer
  4-bit:    sum_b m_b * sum_{i in b} ACT * w     (b: 4 sub-blocks of 32; m_b unsigned 4-bit)
t[j][k]   = (i2f(isum) * ws[n][k]) * ASCALE[j][k]                    two fp32 roundings
acc[j]    = isum_4(t[j][0..KB-1])                four interleaved fp32 partials, then combined
```

- int8 weights: `ws` fp32 per (row, block), amax / 127 (`runtime.quantize_rows`): 8.25 bits.
- fp4 (E2M1 stored as twice its value, {0, 1, 2, 3, 4, 6, 8, 12}) and int4: `ws` bf16 per (row,
  block) and `m_b` in 1..15 per 32, picked by squared error (`quant.quantize_w4`): 4.25 bits.
- Activations (QACT): int8 per (row, block), `s = amax * (1/127)` in fp32; ROW mode one scale a
  row; CSCALE / RSCALE fold per-column / per-row factors in first.
- KV cache: K int8 per token (QST, a scale per 128-block), V^T int8 with V's per-token scales
  folded into P (QACT CSCALE), so P.V streams V^T with UNIT scales.

Every K-block's integer is converted and scaled before it is added, so every column runs i2f, two
fp32 multipliers (6 DSPs), the PAIR adder, the partial loop and the combine each cycle, and the
drain's read-modify-write lanes run per column too: ~5.1K LUTs and ~5.1K flip-flops a column with
the ACT RAM's share, against ~0.44K LUTs and ~3.1K flip-flops for the 128-DSP column itself
(docs/dsp_pumping.md section 5). That per-result part is what MCOLS 8 costs on the YPCB.

## 2. The MX column

Every per-block scale becomes a power of two under a per-row fp32, so the column end's DSP applies
it as a shift of its B operand, and a whole output accumulates as an exact integer in the DSP's
48-bit P register:

```
weights   Sw[n] fp32 per row; per block k: dw[n][k] (uint8); 4-bit: m[n][k][b], 6 bits per 32
acts      Sa[j] fp32 per row; per 32 (sub-block b): da[j][k][b] (uint8)  (one 32-bit word a block)
c         = sum_{i in b} ACT[j][i] * w[i]                     (today's chains, exact)
sh        = Wf - dw[n][k] - da[j][k][b]                       Wf 11 (fp4x6), 13 (fp4x4), 16 (int8x)
B         = sh >= 0 ? m << sh : 0                              m << sh < 2^17: the 18-bit B port
acc[j][n] = sum_k sum_b B * c                                  exact in 48 bits
y         = rne(rne(rne_f32(acc) * Sw[n]) * (Sa[j] * 2^-Wf))   once per output, then ACC / ASCALE
```

- **Drops.** A term whose two offsets add to more than Wf is dropped (B would need a right shift):
  its scale is under 2^-Wf of the row's largest weight scale times the row's largest activation
  scale. **Bounds**: 4-bit, a sub-block's sum < 2^16, times B < 2^17, 4 sub-blocks and 128 blocks
  (K = 16384): < 2^42; int8 < 2^44. K = 65536 still fits (< 2^46).
- **Column end**: today's four `otpu_colend` DSPs (a sub-block's two chain ends pre-added, times
  the multiplier, PCIN cascade) with B = m << sh instead of m, and a fifth DSP that accumulates over
  K (P = P + PCIN; a row's first block P = PCIN). The shift is an 8-bit subtract, a clamp and a
  17-bit shift of the 6-bit m per sub-block, in fabric. PAIR's two columns accumulate separately;
  the lane adds the two integers.
- **Epilogue lanes**: E lanes shared by the array, each i2f48 (round to nearest even), two fp32
  multipliers and the RMW multiply-add (ACC, ASCALE), fed from a result FIFO of 48-bit integers.
  A lane takes an output a cycle; a row of KB blocks takes KB cycles in the array (PAIR KB/2), so
  an MM keeps full rate while rows / KB <= E.
- **Formats** (bits a weight; the per-row fp32 is negligible):
  - `int8x`: int8, 2^-d per 128 under Sw: 8.06 bits (today 8.25: the block's fp32 becomes 8 bits).
  - `fp4x6`: E2M1, 2^-d per 128 and a 6-bit multiplier per 32 under Sw: one 32-bit word a block,
    4.25 bits, as today. `fp4x4`: today's 4-bit multipliers (4.19 bits).
  - `nvfp4` on the column: E2M1 with an E4M3 scale per 16 under a per-row fp32 (`nvfp4r`; the
    standard's per-tensor fp32 works too, the row's largest exponent becoming a per-row byte).
    Each 16-long chain end gets its own DSP (B = the E4M3 mantissa 8..15, subnormals 0..7, << sh,
    Wf 13), 8 a row plus the accumulator: 4.5 bits.
  - `mxfp4` (OCP): E2M1, E8M0 per 32: m = 1, Wf 16.
  - Activations `a8x32`: int8, 2^-d per 32 under a per-row fp32 (OCP MXINT8's grouping); the ACT
    RAM's scale word holds the four 8-bit offsets, a per-row register the fp32. `a8x128`: one
    offset per block.
- **Single-block outputs.** q.K^T at d = 128 is one block, so dw = da = 0 by construction and the
  lane multiplies by the token's K scale and the q row's scale: today's arithmetic for a
  one-block MM, and the K cache keeps its per-token fp32. Heads of 256 / 512 (Qwen3.5, Gemma 4)
  store K per token as int8x (QST in QACT's X mode).

## 3. Formats and accuracy

### 3.1 The accumulation itself

On random int8 operands with log-normal block scales (K = 1024 / 3072 / 8192), today's order
(per-block fp32 products, four interleaved fp32 partials) is within 5-8e-8 (mean) and 1e-6 (max)
of the exact sum, relative to the mean |output|; the MX order (exact 48-bit integers, then i2f
and two fp32 multiplies) is within 3-4e-8 / 7e-7 of its own exact sum. Both are ~2^-24: the
fixed-point accumulation changes nothing measurable by itself. What moves the model is the
change of formats that shifts require, and the dropped terms.

### 3.2 Method

`tools/exp/mx_formats.py` runs `tools/formats_scan.py`'s float64 emulation (the same layers; the
first 2000 tokens of docs/isa.md; the same head; dKL = mean KL(float || variant) minus int8's, in
percent, SE paired over the tokens) with a quantizer per variant for the weights, the head and the
activations (every MM input: the layers', the head's, q, P and K as attention reads them; V stays
int8 per token), and the window's drops emulated exactly per 16 elements (`XAct` / `WX`: the
dropped sub-products are subtracted from the product). hwe-bench, 2026-10-12, 13-36 min a model.
Variants are weights / head / activations [/ flush: the drops on].

### 3.3 Results (2000 tokens)

dKL in percent (SE); perplexity of the 2000 tokens. **MX** rows run on the proposed column, drops
included; the others need today's per-column epilogue.

| variant | Qwen3-0.6B | Qwen3.5-0.8B | LFM2.5-230M | bits / weight |
|---|---:|---:|---:|---:|
| float | -1.25 (0.05), ppl 42.23 | -0.40 (0.01), 30.96 | -0.52 (0.02), 60.05 | |
| int8 today (int8 / int8 / a8) | 0, ppl 41.87 | 0, 30.91 | 0, 59.19 | 8.25 |
| int8 / int8 / a8x128 | +0.17 (0.04) | +0.12 (0.01) | +0.10 (0.02) | 8.25 |
| int8 / int8 / a8x32 | -0.50 (0.04) | -0.14 (0.01) | -0.22 (0.02) | 8.25 |
| **MX int8: int8x / int8x / a8x32** | **-0.41 (0.04)**, 42.29 | **-0.07 (0.01)**, 31.05 | **-0.12 (0.02)**, 60.53 | 8.06 |
| fp4 today (fp4 / int8 / a8) | +15.99 (0.45), ppl 46.27 | +11.07 (0.23), 35.18 | +15.83 (0.28), 64.68 | 4.25 |
| fp4 / int8 / a8x32 | +15.67 (0.44) | +10.92 (0.23) | +15.59 (0.28) | 4.25 |
| fp4x6 / int8 / a8 | +15.69 (0.44) | +11.11 (0.24) | +14.19 (0.25) | 4.25 |
| fp4x4 / int8 / a8 | +16.44 (0.40) | +11.94 (0.27) | +15.77 (0.35) | 4.19 |
| **MX fp4: fp4x6 / int8x / a8x32** | **+15.52 (0.42)**, 47.28 | **+10.91 (0.23)**, 34.52 | **+13.87 (0.25)**, 66.71 | 4.25 |
| MX fp4x4 / int8x / a8x32 | +15.96 (0.39) | +11.80 (0.28) | +15.63 (0.35) | 4.19 |
| nvfp4 / int8 / a8 (per-tensor fp32) | +13.98 (0.43) | +9.22 (0.21) | +22.01 (0.45) | 4.5 |
| nvfp4r / int8 / a8 (per-row fp32) | +13.28 (0.35) | +9.25 (0.21) | +16.92 (0.32) | 4.5 |
| **MX NVFP4: nvfp4r / int8x / a8x32** | **+13.38 (0.35)**, 47.23 | **+9.07 (0.21)**, 34.34 | **+17.38 (0.33)**, 76.11 | 4.5 |
| mxfp4 / int8 / a8 (OCP) | +24.77 (0.62) | +17.12 (0.38) | +23.03 (0.44) | 4.25 |
| MX mxfp4 / int8x / a8x32 | +24.29 (0.60) | +17.05 (0.38) | +22.84 (0.44) | 4.25 |

Paired differences (points, SE over the tokens):

| | Qwen3-0.6B | Qwen3.5-0.8B | LFM2.5-230M |
|---|---:|---:|---:|
| MX int8 - int8 today | -0.41 (0.04) | -0.07 (0.01) | -0.12 (0.02) |
| MX fp4x6 - fp4 today | -0.47 (0.36) | -0.15 (0.15) | -1.96 (0.21) |
| MX fp4x4 - fp4 today | -0.03 (0.37) | +0.74 (0.21) | -0.20 (0.27) |
| MX NVFP4 - fp4 today | -2.61 (0.41) | -1.99 (0.20) | +1.55 (0.36) |
| MX NVFP4 - MX fp4x6 | -2.14 (0.37) | -1.84 (0.20) | +3.52 (0.33) |
| MX mxfp4 - fp4 today | +8.29 (0.56) | +5.99 (0.35) | +7.01 (0.40) |
| fp4: a8x32 - a8 | -0.33 (0.15) | -0.15 (0.05) | -0.24 (0.06) |
| int8x: drops on - off | -0.02 (0.02) | -0.01 (0.01) | 0 |

Share of (row, output, 16-element) terms dropped: int8x 0 to 7e-7, fp4x6 7e-9 to 7e-5, NVFP4
7e-10 to 2e-5, mxfp4 0 to 2e-7.

- **The activations are the free gain**: a power of two per 32 under a per-row fp32 beats today's
  fp32 per 128 on all three models (-0.14 to -0.50 points at int8): a block's outlier no longer
  sets the scale of the other 96 elements. A power of two per 128 alone loses 0.10-0.17.
- **fp4x6 replaces today's fp4 without loss**: the bf16 block scale's mantissa is not needed when
  the sub-block multiplier has 6 bits. With today's 4-bit multipliers (fp4x4) Qwen3.5 loses 0.74.
- **NVFP4** is the most accurate 4-bit format on Qwen3 and Qwen3.5 (-2.0 to -2.6 points against
  today's fp4), not on LFM2.5 (section 3.4).
- **MXFP4** is far worse everywhere (+6 to +8 points against today's fp4).
- dKL here is the whole model's (formats_scan's), 2000 tokens; per-kind mixes (formats.py) were
  not scanned in the MX formats.

### 3.4 NVFP4 on LFM2.5: the position

dKL per window of 250 tokens (points against int8, same tokens):

| | 0 | 250 | 500 | 750 | 1000 | 1250 | 1500 | 1750 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fp4 today | +13.7 | +14.6 | +19.7 | +13.7 | +18.6 | +15.9 | +16.4 | +14.0 |
| MX fp4x6 | +12.6 | +14.4 | +17.0 | +14.0 | +14.4 | +13.7 | +13.6 | +11.2 |
| nvfp4 (per tensor) | +9.8 | +12.3 | +20.2 | +21.2 | +24.8 | +31.3 | +33.0 | +23.5 |
| MX nvfp4r | +9.9 | +12.4 | +17.5 | +15.9 | +17.7 | +25.1 | +23.3 | +17.2 |

NVFP4 is the best format over the first 500 tokens on LFM2.5 too, then it falls behind from
~750 tokens on; on Qwen3 and Qwen3.5 it stays ahead in every window.

Which weights (per-tensor NVFP4 in one kind, today's fp4 in the others; paired against fp4
everywhere, same tokens):

| NVFP4 in | dKL | against fp4 | windows 0 .. 1750 |
|---|---:|---:|---|
| attention (q, k, v, o) | +13.23 (0.24) | **-2.60 (0.16)** | +12.0 +12.9 +16.0 +11.5 +14.6 +14.0 +14.0 +10.9 |
| conv (in, out) | +14.81 (0.28) | -1.02 (0.16) | +13.6 +14.8 +17.0 +12.7 +16.3 +15.3 +15.7 +13.0 |
| MLP (gate, up, down) | +16.81 (0.32) | **+0.98 (0.29)** | +11.3 +13.7 +22.2 +17.3 +18.1 +19.7 +19.0 +13.3 |
| all but attention | +15.52 (0.30) | -0.31 (0.31) | +11.5 +13.1 +17.8 +15.1 +17.0 +19.0 +18.5 +12.1 |
| all | +22.01 (0.45) | +6.18 | |

NVFP4 helps LFM2.5's attention and conv weights and hurts its MLP, and the kinds do not add: the
attention's NVFP4 gains 2.6 points over fp4 elsewhere but costs 6.5 with the MLP and conv in
NVFP4 too. So on LFM2.5 NVFP4 is a per-kind choice (attention), which `opentpu/llm/formats.py`
already expresses and the MX column supports per MM. (For scale: on the MX column, LFM2.5 with
the attention in int8x and the rest in fp4x6 is at +8.77, -5.10 points against fp4x6 everywhere.)

### 3.5 What the change means

The MM's accumulation order changes (exact integers, then three roundings), and so does every
format's scale, the weights' and the activations'. Every image is re-quantized and every reference
recomputed (refcache keys, validate.py's card comparisons, the token-exact tests' expected tokens).
The user decides (section 7).

## 4. Attention on shared lanes

**The rate.** A lane takes one output a cycle. q.K^T streams one K row (a token) a cycle and gives
an output per stationary row per K row when d = 128 (one block): G rows in decode (Qwen3-0.6B 2;
Qwen3.5-0.8B 4 with d = 256 = 2 blocks; Gemma 4 26B 2 / 8 with d 256 / 512; Qwen3.5-35B 8 with d
256), up to MCOLS in prefill (R x G rows). P.V's outputs take two blocks (256-token attention
blocks). So decode needs ~2 lanes for full rate, prefill MCOLS.

**The co-sim.** `OTPU_EPI_RATE = E` caps the MXU's drain at E results a cycle
(`rtl/mxu/otpu_mxu.sv`, `drain_run`; the rows wait in the result FIFO and the pops stall once it
is full, as for today's bank-conflict runs), passed through `OTPU_VDEFS` (`opentpu/rtlsim.py`).
Qwen3-0.6B fp4 with the int8 head, 2 layers, the YPCB memory path (LiteDRAM DDR3-1066, 133.33 MHz),
PAIR / DSTEP / STREAM on, `tools/perf_qwen.py` (`--batch B`: one row per KV slot); cycles and the
change against E = 0 (today's drain, MCOLS results a cycle):

| run | MCOLS | E = 0 | E = MCOLS/2 | E = MCOLS/4 | E = 1 |
|---|---:|---:|---:|---:|---:|
| prefill 4 rows, pos 544 | 4 | 372,383 | 373,700 (+0.4%) | | 467,233 (+25.5%) |
| prefill 4 rows, pos 544 | 8 | 267,269 | 267,269 (0) | 293,307 (+9.7%) | 404,693 (+51%) |
| the same, chunk 1024 | 8 | 268,146 | | 270,366 (+0.8%) | |
| prefill 8 rows, pos 544 | 8 | 502,925 | 503,044 (+0.02%) | 558,336 (+11.0%) | 786,573 (+56%) |
| prefill 4 rows, pos 4000 | 4 | 842,590 | 842,967 (+0.04%) | | |
| prefill 4 rows, pos 4000 | 8 | 740,566 | 740,566 (0) | 764,535 (+3.2%) | |
| the same, chunk 1024 | 8 | 741,125 | | 742,003 (+0.1%) | |
| batched 4 users + head, pos 544 | 4 | 1,696,116 | 1,694,384 (-0.1%) | | |
| batched 4 users + head | 8 | 1,593,159 | | 1,614,799 (+1.4%) | |
| batched 8 users, no head | 8 | 518,581 | 521,472 (+0.6%) | 570,702 (+10.1%) | |
| batched 8 users, no head | 16 | 429,052 | 429,052 (0: E >= rows) | 453,374 (+5.7%) | |
| batched 8 users, chunk 1024 (1) | 8 | 525,211 | | 528,844 (+0.7%) | |
| batched 8 users, chunk 1024 (1) | 16 | 425,265 | | 427,909 (+0.6%) | |
| prefill 16 rows, pos 200 (1) | 8 | 785,826 | 790,945 (+0.7%) | 900,573 (+14.6%) | |
| the same, chunk 1024 (1) | 8 | 810,064 | | 821,526 (+1.4%) | |
| prefill 16 rows, pos 200 (1) | 16 | 613,106 | 613,106 (0) | 660,953 (+7.8%) | |
| the same, chunk 1024 (1) | 16 | 607,149 | | 611,637 (+0.7%) | |
| decode + head, pos 544 | 4 | 1,471,170 | 1,471,170 (0) | | 1,482,874 (+0.8%) |
| decode + head, pos 4000 | 4 | 1,589,677 | 1,589,677 (0) | | 1,670,786 (+5.1%) |
| Qwen3.5-0.8B decode + head, 4 layers | 4 | 2,571,917 | 2,572,854 (+0.04%) | | 2,578,205 (+0.2%) |
| Qwen3.5-0.8B prefill 4 rows, 4 layers | 8 | 942,492 | 942,492 (0) | 1,006,839 (+6.8%) | |
| the same, chunk 1024 | 8 | 941,973 | | 950,872 (+0.9%) | |
| LFM2.5 prefill 4 rows, 4 layers | 8 | 271,253 | 271,253 (0) | 312,863 (+15.3%) | |
| the same, chunk 1024 | 8 | 269,796 | | 272,042 (+0.8%) | |

(1) A TMEM of 256K words (`OTPU_TMEM_WORDS`, 16 ACT rows), which 16 rows and chunks of 1024 at 8
rows need, with the sequencer's TMEM ranges widened (`OTPU_TAW=18`, below).

**A side finding: a TMEM above 64K words serializes.** `rtl/seq/otpu_seq.sv` keeps 16-bit TMEM
ranges in its dependency window (TAW = 16) and makes any access above 2^16 conflict with
everything. With 256K words the compiler places tiles there, so the same 8-row prefill took
853,599 cycles instead of 502,925 (bit-exact with the ISA, only serialized; 839,408 at 128K), and
502,398 with TAW = 18 (`OTPU_TAW`, an experimental define). A bigger TMEM (the TMEM study, the
scoreboard's TMEM what-ifs) needs TAW = clog2(TMEM_WORDS) in the sequencer.

**Where the E = MCOLS/4 cost is.** Not in attention: at MCOLS 8, 4 rows, E = 2 the attention phase
goes 182,531 -> 184,544 cycles (+1.1%), at 4000 tokens 655,852 -> 655,764 (0). It is in the MLP
(83,897 -> 107,922): `kernels/mlp.swiglu_down` accumulates the down projection in partial MMs over
F chunks of 256 (`_chunk`), two K-blocks each, one PAIR advance, so 4 outputs a cycle. An MM keeps
full rate while rows / E <= KB (KB / 2 with PAIR), so the partial MMs need chunks of at least
D x rows / E (twice that with PAIR). With E = MCOLS/2 today's 256 always meets it (rows <= MCOLS;
PAIR rows <= MCOLS/2); with E = MCOLS/4 it takes 512 or more. With chunks of 1024
(`OTPU_MX_CHUNK=1024`, an experimental override) the MLP's cost falls to +1.0% and the run's to
+0.8% (the "chunk 1024" rows). Bigger chunks need TMEM: batched 8 users and the MoE experts' 8
rows at chunk 1024 do not fit today's 64K words (section 6.4 runs them at 256K).

**Attention is VPU-bound.** One lane (E = 1) costs attention +28% (prefill) to +40% (decode at
4000 tokens: 202,137 -> 283,103 cycles); two or more cost nothing: the softmax (EXP2SUB, the row
sums, P's quantization) sets attention's pace, and q.K^T's results wait for it anyway (the
VPU-first arbiter finding). MCOLS 8 itself takes attention only from 219,275 to 182,531 (-17%) at
544 tokens and 689,473 to 655,852 (-5%) at 4000.

**So:**
- E = MCOLS/2 shared lanes: no measurable cost (0 to +0.7%).
- E = MCOLS/4: +0.1 to +1.4% against the same chunks with the down projection's chunks at
  D x rows / E or more (a compiler rule; it needs TMEM at 8 rows and more), +3 to +15% without.
  The bigger chunks are not free by themselves (16 rows at MCOLS 8: +3.1%; at 16: -1.0%).
- E = 1: no.
- A raw int32 score path (scores without scales, the scale folded into the softmax's pre-scale)
  or full-rate lanes for q.K^T buy nothing on these models: attention's MXU side is not the
  limit. K keeps its per-token fp32 (one block a token at d = 128).

## 5. Area and timing

`rtl/exp/mx_mxu/` (experimental, in no file list): `otpu_mx_x7.sv` (pumped DSP48E1 columns of two
rows, their column ends with the shift and the accumulating DSP, the 266.67 MHz clock),
`otpu_mx_us.sv` (the same on packed DSP48E2s, one clock), `otpu_mx_lane.sv` (a shared lane:
i2f48, the two fp32 multiplies, the RMW multiply-add), `mx_top.sv` (NCOL physical columns, 2 NCOL
rows, PAIR's integer add, the result FIFO, E lanes), `base_top.sv` (today's columns and per-column
epilogue, copied from `otpu_mxu`, with the same surroundings), `tb_mx.sv` (both families against a
reference model), `ooc.tcl` / `run_ooc.sh` (Vivado), `run_xsim.sh`.

**Bit-exact**: `tb_mx.sv` in xsim on the unisim models (omarchy, 2026-10-11): 260 rows, 880 lane
outputs, 416 conversions, 0 mismatches, for the DSP48E1 (pumped) and the DSP48E2 (packed) columns.

**Synthesis estimate** (Yosys 0.64 `synth_xilinx -family xc7 / xcup -flatten` on the Mac, the
sources through sv2v; synthesis only, no placement or timing, and Yosys's LUT counts run above
Vivado's: use the differences). Both tops hold the columns, the column ends, the epilogue, the
result FIFO and the read-modify-write lanes; neither holds the ACT RAM, the issuer, the chunk and
scale FIFOs or TMEM (the same for both, except the ACT RAM's width, which follows the rows):

| family | design | rows | LUT | SRL | LUT RAM | FF | DSP |
|---|---|---:|---:|---:|---:|---:|---:|
| DSP48E1 (7-series) | today, MCOLS 4 | 4 | 15,713 | 6,300 | 576 | 12,049 | 552 |
| | **MX, 2 pumped columns, E 2** | 4 | **10,838** | 6,065 | 960 | **6,993** | **288** |
| | today, MCOLS 8 | 8 | 27,726 | 10,958 | 1,092 | 26,285 | 1,104 |
| | **MX, 4 pumped columns, E 4** | 8 | **17,942** | 10,593 | 1,728 | **13,855** | **576** |
| | MX, 4 pumped columns, E 2 | 8 | 14,865 | 10,499 | 1,728 | 12,478 | 564 |
| DSP48E2 (UltraScale+) | today, MCOLS 4 | 4 | 16,587 | 6,300 | 504 | 12,421 | 552 |
| | **MX, 2 packed columns, E 2** | 4 | **11,204** | 6,129 | 840 | **6,946** | **288** |
| | today, MCOLS 8 | 8 | 28,941 | 10,958 | 936 | 27,029 | 1,104 |
| | **MX, 4 packed columns, E 4** | 8 | **18,529** | 10,705 | 1,488 | **13,700** | **576** |

One lane: 1,167 LUTs, 100 SRLs, 546 flip-flops, 6 DSPs.

- **At today's rate** (4 rows) the MX datapath is 31% fewer LUTs, 42% fewer flip-flops and 48%
  fewer DSPs (-4.9K, -5.1K, -264): the per-column epilogue and drain lanes go, two shared lanes,
  the shifts and the accumulating DSPs come. Pumping alone accounts for -256 DSPs and ~-1.7K
  flip-flops of that (docs/dsp_pumping.md section 5, Vivado); the rest is the epilogue.
- **At MCOLS 8's rate** (8 rows, E 4) it is 35% fewer LUTs, 47% fewer flip-flops and 528 fewer DSPs
  than today's MCOLS 8 (DSP48E2: 36%, 49%, 528). Going from today's 4 to 8 rows costs +2.2K LUTs,
  +4.3K SRLs, +1.2K LUT RAM, +1.8K flip-flops and +24 DSPs on MX (the second activation skew and the
  column ends), against +12.0K, +4.7K, +0.5K, +14.2K and +552 for today's MCOLS 8.
- **NVFP4** adds 4 DSPs and 4 shifts a row to the column end (+16 DSPs at 4 rows, +32 at 8).
- The ACT RAM's read width doubles with 8 rows either way (64 -> ~128 BRAM36, docs/dsp_pumping.md
  4.3).

**Vivado out of context** (`run_ooc.sh`: synthesis, placement, routing at 133.33 MHz, the pumped
columns at 266.67 MHz; the pairs k7-base4 / k7-mx2e2, k7-base8 / k7-mx4e4 on xc7k480tffg1156-2,
the same on xcku5p): not run yet. The first pair has been queued on omarchy since 2026-10-11
23:24 (-03) behind a full build; the timing of the column end's shift into the B port, the
accumulating DSP, the lane's i2f48 and the pumped domain are what it will tell. The pumped
column alone closed at 266.67 MHz with +0.163 ns (docs/dsp_pumping.md section 5).

## 6. Workloads

### 6.1 Co-sim, the YPCB's memory path

MX arrays of 8 rows with 4 lanes and of 16 rows with 8, against today's MCOLS 4:

| workload | today, MCOLS 4 | MX 8 rows, E 4 | change | MX 16 rows, E 8 | change |
|---|---:|---:|---:|---:|---:|
| Qwen3-0.6B fp4 prefill, 4 rows, pos 544 (2 layers) | 372,383 | 267,269 | -28.2% | | |
| Qwen3-0.6B fp4 prefill, 4 rows, pos 4000 | 842,590 | 740,566 | -12.1% | | |
| Qwen3-0.6B fp4 prefill, 16 rows, pos 200 (2) | 1,266,469 | 790,945 | -37.5% | 613,106 | -51.6% |
| Qwen3.5-0.8B fp4 prefill, 4 rows (4 layers) | 1,137,275 (1) | 942,492 | -17.1% | | |
| LFM2.5-230M fp4 prefill, 4 rows (4 layers) | 443,238 (1) | 271,253 | -38.8% | | |
| Qwen3-0.6B batched decode, 4 users + head | 1,696,116 | 1,593,159 | -6.1% | | |
| Qwen3-0.6B batched decode, 8 users, no head | 730,652 | 521,472 | -28.6% | 429,052 | -41.3% |
| Qwen3-0.6B decode + head, pos 544 / 4000 | 1,471,170 / 1,589,677 | unchanged | 0 | | |

(1) docs/dsp_pumping.md section 3 (the same tree and settings). (2) TMEM 256K words, TAW 18
(section 4). Prefill at 8 rows: section 6.2.

### 6.2 An ideal port

The BCU1525's and F2's slices: a 128-byte chunk a core cycle, 60 cycles of latency:

| workload | MCOLS 4 | MX 8 rows | change | MX 16 rows | change |
|---|---:|---:|---:|---:|---:|
| Qwen3-0.6B prefill 4 rows | 365,469 | 256,787 (E 4) | -29.7% | | |
| Qwen3-0.6B prefill 8 rows | 719,211 | 493,946 (E 4) | -31.3% | | |
| Qwen3-0.6B batched 4 users + head | 1,652,489 | 1,543,794 | -6.6% | | |
| Qwen3-0.6B batched 8 users, no head | 715,821 | 504,995 (E 4) | -29.5% | 418,047 (E 0), 442,100 (E 4) | -41.6%, -38.2% |
| Qwen3-0.6B decode + head | 1,433,117 | 1,433,117 (E 2) | 0 | | |

### 6.3 Scoreboard

`tools/scoreboard.py` (the workloads agent's tree, e769bf79 and its edits since), each option
against its own board and clock:

| board | option | decode | prefill | batched | MoE | score |
|---|---|---:|---:|---:|---:|---:|
| YPCB | MX 8 rows | 1.00x | 1.38x | 1.74x | 1.00x | 1.25x |
| YPCB | MX 8 rows, TMEM x4, 16 ACT rows | 1.00x | 1.62x | 1.84x | 1.00x | 1.32x |
| YPCB | MX 16 rows, TMEM x4 | 1.00x | 2.09x | 2.78x | 1.00x | 1.55x |
| BCU1525 DDR4-2400, 200 MHz | MX 8 rows | 1.00x | 1.32x | 1.66x | 1.00x | 1.22x |
| BCU1525 DDR4-2400, 200 MHz | MX 16 rows, TMEM x4 | 1.00x | 1.93x | 2.50x | 1.00x | 1.48x |
| BCU1525 DDR4-2400, 200 MHz | port 256 B + MXU x2 | 1.41x | 1.50x | 1.63x | 1.28x | 1.45x |
| BCU1525 DDR4-2400, 200 MHz | port 256 B + MXU x2, 8 rows | 1.42x | 1.79x | 2.54x | 1.28x | 1.70x |
| F2 (S 4, 250 MHz) | MX 8 rows | 1.00x | 1.26x | 1.57x | 1.00x | 1.19x |
| F2 (S 4, 250 MHz) | MX 16 rows, TMEM x4 | 1.00x | 1.78x | 2.22x | 1.00x | 1.41x |
| F2 (S 4, 250 MHz) | port 256 B + MXU x2 | 1.75x | 1.52x | 1.58x | 1.46x | 1.57x |
| F2 (S 4, 250 MHz) | port 256 B + MXU x2, 8 rows | 1.75x | 1.80x | 2.31x | 1.46x | 1.80x |
| F2 (S 4, 250 MHz) | port 512 B + MXU x4 | 2.51x | 2.00x | 2.21x | 1.80x | 2.11x |

The scoreboard has no lane limit (E = MCOLS); the co-sim puts E = MCOLS/2 within 0.6% of that.
"MXU x2 / x4" is where the MX column matters beyond the row count: two products per DSP (pumped
on the 7-series, packed on DSP48E2) behind an epilogue that does not grow with them. The wider
port is docs/wide_dram.md's separate change; on the YPCB (DDR3-1066) it buys nothing, so its rows
are left out. Decode and MoE on the YPCB are bound by DRAM and the PCIe link (1.00x); MoE's
expert FFNs gain where they run (6.4).

### 6.4 MoE expert FFNs (`tools/moe_prefill_cosim.py expert`: one expert's FFN on M rows, cycles)

| model, M rows | MCOLS 4 | MX 8 rows, E 4 | 8 rows, E 2 | MX 16 rows, E 8 | 16 rows, E 4 | 16 rows, E 4, chunk 1024 |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-35B-A3B, 1 | 13,454 | 13,458 | 13,458 | 13,466 | 13,466 | 13,446 |
| Qwen3.5-35B-A3B, 2 | 13,298 | 13,302 | 13,302 | 13,310 | 13,310 | 13,368 |
| Qwen3.5-35B-A3B, 4 | 25,895 | 13,850 | 17,795 | 13,858 | 13,858 | 14,510 |
| Qwen3.5-35B-A3B, 8 | 49,505 | 27,350 (2) | 35,624 | 16,052 | 20,451 | 17,816 |
| Qwen3.5-35B-A3B, 16 (1) | 104,784 | 56,406 | | 32,578 | 41,090 | 35,515 |
| Gemma 4 26B-A4B, 1 | 27,964 | 27,968 | 27,968 | 27,976 | 27,976 | 27,898 |
| Gemma 4 26B-A4B, 2 | 28,024 | 28,028 | 28,028 | 28,036 | 28,036 | 27,879 |
| Gemma 4 26B-A4B, 4 | 52,407 | 28,813 | 35,923 | 28,821 | 28,821 | 28,806 |
| Gemma 4 26B-A4B, 8 | | 56,408 (2, 3) | 57,167 (3) | | | 32,813 |
| Gemma 4 26B-A4B, 16 (1) | 209,931 | 109,506 | | 58,839 | 76,315 | 65,484 |

(1) TMEM 256K words, TAW 18 (section 4). (2) E 0 (not run at E 4; E = rows or more is no cap
up to 4 rows). (3) Chunks of 1024 (today's do not fit 64K words at 8 rows).

An expert's FFN streams its slot once (1.67 MB for the 35B, 3.4 MB for the 26B): 1-2 rows are bound
by the port at any row count; 4 rows halve at 8 rows (-47% / -45%); 8 rows take -45% at 8 rows and
-68% at 16; 16 rows -46% / -48% at 8 rows and -69% / -72% at 16 (E 8 = E 0 on all of them). On 16
rows, E = MCOLS/4 costs the experts +26% / +30% (+9% / +11% with chunks of 1024, which the 35B's F =
512 caps at 512): E = MCOLS/2. MoE decode (one row) does not change.

## 7. Recommendation and decisions

**Recommendation** (pending Vivado's timing, section 5):

1. **Make the MX column the MXU's next datapath**, with `int8x` and `fp4x6` weights and `a8x32`
   activations, exact accumulation in the column end, and E = MCOLS/2 shared epilogue lanes. It is
   at least as accurate as today on all three models (section 3), it costs nothing measurable in
   any workload (section 4), and it removes the per-column epilogue that makes MCOLS 8 expensive.
2. **Keep NVFP4, as a per-kind option, not as the default.** The column end with a DSP per
   16-long chain (8 + 1 a row, +4 DSPs a row against fp4x6's 5) runs NVFP4 exactly, and the same
   column end runs every other format (two chain ends with equal B). It pays where the formats
   scan says so (Qwen3, Qwen3.5: -1.8 to -2.1 points against fp4x6 for +5.9% of 4-bit bytes; on
   LFM2.5 only in the attention). A pumped column end (the two 16-chains of a sub-block in the
   two fast cycles) would do it with today's DSP count; not built. Drop OCP MXFP4.
3. **No special attention path.** E >= 2 hides q.K^T's one-block outputs behind the VPU-bound
   softmax. Attention's speed is the VPU's (the pumped-VPU study).
4. **Rows**: on the YPCB, the MX datapath first at today's 4 rows (it gives back ~4.9K LUTs and
   ~5.1K flip-flops of the 81%-full part), then 8 rows: the prefill / batched-decode / MoE-expert
   lever (-28 to -48% of those runs' cycles, decode unchanged) for a fifth of the LUTs (under half
   with the SRLs and LUT RAM) and an eighth of the flip-flops that MCOLS 8 costs today. 16 rows with
   a bigger TMEM (and the sequencer's wider ranges) on the bigger boards (scoreboard 1.48x on the
   BCU1525, 1.41x on the F2, before a wider port). Pumped DSP48E1s on the 7-series, packed DSP48E2s
   on UltraScale+.

**Decisions for the user** (not guessed here):

- **The numerics change.** MX changes MM's accumulation order and every format's scale; every image
  and reference is rebuilt and the token streams change (dKL equal or better on the three models
  measured). Today's fp4 (bf16 block scale) cannot run exactly on the MX column: its 8-bit
  mantissa times the 4-bit multiplier would leave Wf = 5 and drop terms, so fp4x6 replaces it
  (same bytes). Accept or not?
- **NVFP4 in hardware** (+4 DSPs a row in the column end, the formats scan choosing per kind) or
  not (then fp4x6 alone).
- **E**: MCOLS/2 (no measured cost; 2 lanes at 4 rows, 4 at 8) or MCOLS/4 (one lane, ~1.3K LUTs,
  fewer; needs the chunk rule and TMEM, and the MoE experts still lose 9-11% on 16 rows).
- **Card time**: a qualification on the card needs a session the lead schedules (none used here).

## 8. Implementation plan

Staged so that each step is checked before the next one costs Vivado time.

**1. ISA, simulator, compiler (no RTL).**
- `docs/isa.md`, `opentpu/isa.py`: MM flag bit 7 `MX` (free today) reinterprets WF: 0 int8x, 1 int4x
  (int4 with fp4x6's scales; not measured here), 2 fp4x6, 3 nvfp4. Scale words: fp4x6 / int4x one
  word a block (`dw` in [7:0], four 6-bit m above it); int8x one byte a block (4 blocks a word);
  nvfp4 two words a block (eight E4M3 bytes) and the row's reference exponent; Sw a word per row
  ahead of the row's scales. QACT flag bit 4 `X`: int8 with four 8-bit offsets in the block's ACT
  scale word and Sa in a per-row register (ASCALE's row); QST `X` for K of d > 128. MM semantics as
  in section 2: exact accumulation, the drop rule, y = rne(rne(i2f(acc) x Sw) x (Sa x 2^-Wf)), then
  ACC / ASCALE / RMAX as today; PAIR's two terms added as integers; UNIT = Sw 1, dw 0, m 1.
- `opentpu/isasim.py` (`_mm`, `_qact`, `_qst`) and `opentpu/fp32.py` (i2f of a 48-bit integer,
  round to nearest even).
- `opentpu/quant.py`: `quantize_int8x`, `quantize_fp4x6` (the multiplier search with m in 1..63),
  `quantize_nvfp4r` (from `tools/exp/mx_formats.py`), `quantize_mxu` / `mxu_wf` / `row_bytes` for
  them; `opentpu/llm/formats.py` names; `runtime.quantize_rows` X mode; every family's
  `emulated_logits` and formats_scan / quant_eval take the MX quantizers.
- `opentpu/kernels/mlp.py`: the chunk rule (D x rows / E, twice with PAIR) when E < MCOLS/2.
- Tests: `tests/test_isa.py` MX MMs against a numpy model (exact accumulation, sh = 0 / -1 at the
  drop boundary, the 48-bit bound at K = 65536, PAIR, UNIT, ACC / ASCALE / RMAX); `test_quant.py`
  and `test_formats.py` for the new formats; the formats gate (dKL on Qwen3-0.6B, Qwen3.5-0.8B,
  LFM2.5 at 2000 tokens no worse than today's); the ISA-level token-exact images rebuilt with new
  refcache keys.
- Before RTL: dKL on bigger models (Qwen3.5-4B, Gemma 4 E2B / E4B, the 26B / 35B MoE experts):
  their scale ranges set how often the window drops.

**2. RTL at today's 4 rows** (`rtl/mxu/otpu_mxu.sv` IMPL 3, the experiment's cells).
- The column end with the shift and the accumulating DSP (`rtl/tech/{xilinx7,xilinxus,generic}`),
  the 48-bit result FIFO, E shared lanes (`otpu_mx_lane`), PAIR's integer add, Sw from the scale
  stream, Sa per row; `rtl/vpu/otpu_quant.sv` X mode (per-32 amax, the offsets against the row's
  largest, the row's scale).
- `tests/test_rtl.py` / `test_quant_rtl.py` (Verilator): IMPL 3 against isasim, bit-exact, random
  MMs at MCOLS 4 / 8, E 1 / 2 / 4, every format; the `*_token_on_rtl` tests on the new refs.
- OOC of `otpu_mxu` IMPL 3 against IMPL 2 (omarchy / opentpu), then a FAST=1 board build at 100
  MHz (dev-builds rule) for function, then 133.33 MHz; card qualification in a session the lead
  schedules.

**3. 8 rows** on the YPCB: the pumped column (`rtl/exp/dsp_pump`, a second clock) or 8 physical
columns, ACT RAM 8 rows a cycle, E = 4; on UltraScale+ the packed DSP48E2 column (`otpu_mx_us`).
16 rows with 4x TMEM where the board has the RAM (BCU1525, F2), with `otpu_seq`'s TAW widened
to clog2(TMEM_WORDS).

**Risks**
- **QACT's X mode** needs the row's largest scale before the offsets: ROW mode's two passes over
  the row instead of one. The co-sim did not model it (QACT timing unchanged); attention is
  VPU-bound, so P's quantization must not get slower. Alternative: the row scale from the previous
  MM's RMAX.
- **Drops on other models**: measured at most 7e-5 of the terms on three small models; a model
  with a wider spread of block scales loses more. Wf is set by the 18-bit B port on both DSP
  families.
- **MoE experts at 16 rows with E = MCOLS/4**: +26-30%, +9-11% with the chunk rule (the 35B's
  F = 512 caps its chunks); E = MCOLS/2 has no such case.
- **A TMEM above 64K words** needs the sequencer's TMEM ranges widened (TAW = clog2(TMEM_WORDS) in
  `otpu_seq`; `OTPU_TAW` here is a co-sim define only), or every access above 2^16 serializes.
- **The pumped clock domain** on the 7-series (docs/dsp_pumping.md section 5: +0.16 ns at 266.67
  MHz out of context; a full build not run).
