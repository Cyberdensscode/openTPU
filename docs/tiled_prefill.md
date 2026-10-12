# Tiled prefill: more rows per weight pass, and the same machinery for MoE and many users

The question (the user, 2026-10-11): prefill, multi-user and MoE are now as important as decode.
Today a prefill run holds its R rows in TMEM, R_max 3-4 a bucket (docs/prefill.md 7), so each
weight chunk read from DRAM serves 3-4 tokens. Would a run of 32-256 rows, every weight chunk
streamed once for all of them, pay; what do the compiler and the ISA need for it; and how does
the same machinery serve MoE prefill and batched decode?

Measured here means an RTL co-simulation (tools/perf_tiled.py: the board's configuration and
memory path, LiteDRAM's controllers at DDR3-1066 with the core at 133.33 MHz, or a memory at the
core port's full rate; zero images, the timing does not depend on the data) or an ISA-simulator
run; estimated means arithmetic on measured parts, named as such.

## 0. Short answer

- **Rows a weight pass are not what prefill lacks.** A pass needs about MCOLS rows (4-bit) or
  2 MCOLS (int8) to reach the MXU's multiply rate (section 2), and the card's prompt runs are
  already 96% MXU-busy at 4 rows. Today's kernel at 8, 16 and 32 rows with MM replay is within
  -1.2% / +1.3% a row of 4 rows at MCOLS 4 (4-bit), while its weight traffic falls 2.8x
  (section 4.1). 32-256-row tiles buy DRAM traffic, not cycles.
- **What pays is the restructured run** (opentpu/llm/tiled.py, bit for bit today's results):
  attention packed MCOLS / G rows to a K / V stream, the next KV head's projections streaming
  while a head attends, the MLP in pipelined pieces, and nothing in TMEM that grows with the
  model's widths. Measured (RTL co-simulation, 2 layers from position 256, cycles a row):
  - Qwen3-0.6B on the card (MCOLS 4): 83.6K today -> 79.5K at T = 4 (-4.9%, today's bitstream:
    no replay) -> 76.4K at T = 16 (-8.6%, ACT_ROWS 16); int8 89.2K -> 76.5K at T = 8 (-14%);
    at position 1024, 110.8K -> 92.4K (-17%);
  - Qwen3-0.6B at MCOLS 8: 57.7K today (4 rows), 53.8K (8 rows) -> 44.4K at T = 16 (-23% /
    -17%); position 1024: 82.0K -> 58.4K (-29%). With the tile MCOLS 8's prefill is 1.88x the
    card's today, against 1.45x with today's kernel;
  - a Qwen3-4B shape: -1.1% at MCOLS 4 (its MLP is 69% of a row and at its floor; its G = 4
    heads already fill the columns), -10.7% at MCOLS 8; Qwen3.5-0.8B is 61-73% DeltaNet, which
    no tile touches (about -4% at MCOLS 8, estimated, not ported);
  - a memory at the port's full rate (the BCU1525's DDR4 at 1600 or 2400, the F2's HBM)
    changes these by 0-4%: prefill is MXU-bound on every board.
- **It fits today's TMEM**: T = 16-24 on Qwen3-0.6B and 8-12 on a 4B shape in 64K words, with
  no DRAM spill; a bigger TMEM adds nothing to the tile (77.0K at 256K words against 76.4K).
  Past ~1K tokens the unrolled packed attention needs the looped attention (IMEM).
- **MoE**: expert-major prefill is already this structure, and the tiled mixer runs belong on
  its path (layer-major runs keep R = 2 for their expert union). Tiling them is worth an
  estimated -2.6 to -3.1 s of gemma-4-26B's 16.25 s pooled prompt on the card (its int8 dense
  weights stream at 2 rows a pass today; -2.2 to -2.6 s at T = 4 on today's bitstream), little
  on Qwen3.5-35B (fp4, DeltaNet-bound). Expert passes: at MCOLS 4 a pair is the floor
  (measured), at MCOLS 8 four rows cost what two do. The dormant expert-major path's own
  predicted -1.7 / -1.9 s is the link side; its card A/B (pfB, staged) comes first and needs
  card time, which the lead schedules.
- **Many users**: one pass of 2 MCOLS users against two runs of MCOLS: -1% (fp4) to -6% (int8).
  Users' attention does not pack; the aggregate's lever is the MXU's rate (MCOLS 8: -36% a row
  for 8 users).
- **Recommendation.** The tile is the prefill half of a faster MXU: build it with the MCOLS 8 /
  pumped-MXU work (+12% to +30% prefill throughput on top) and for int8 layers once
  ACT_ROWS >= 8 (it holds 8-12 rows of a 4B-class layout in 64K words, today's kernel 3-4).
  On today's card alone it is small (Qwen3-0.6B -5%, more at long context; 4B-class fp4 ~0),
  so it should not displace other card work; its card-side win now is gemma-4-26B's mixer
  runs, after the pfB A/B.
- **Found on the way**: an RTL co-simulation with a TMEM over 64K words serialized every
  instruction touching the upper words (otpu_seq's 16-bit hazard ranges; 83.6K -> 112.7K a
  row); the fix is on this branch and on tmem-pump's (the same change, 3.5).

## 1. Where a prefill row's cycles go today

Measured (tools/perf_tiled.py --kernel rows: compile_rows, today's qwen3_rows; Qwen3-0.6B, its
first 2 layers, 4 rows at positions 256-259, fp4 with no logits; the card's configuration:
ACT_ROWS = MCOLS, TMEM 64K words; the card's memory path at DDR3-1066, core 133.33 MHz; a phase
is the cycles from the last instruction end before it to the last end within it, perf_qwen's
split; "attn-proj" is the Q / K / V / O projections and the queries' norms and RoPE, "attn-core"
the flash attention, "mlp" the norm, gate / up, SiLU and down):

| MCOLS | cycles a row (2 layers) | attn-core | attn-proj | mlp | MXU busy | the linear layers' MXU floor |
|---|---|---|---|---|---|---|
| 4 | 83.6K | 14.6K (17%) | 30.7K (37%) | 38.1K (46%) | 80.5K (96%) | 61.4K |
| 8 (virtual, ACT_ROWS 8) | 57.7K | 14.6K (25%) | 21.9K (38%) | 21.0K (36%) | 54.6K (95%) | 30.7K |

- **The MXU is busy 96% of a row.** The MLP runs at 1.03x its weights' floor (38.1K against
  36.9K); the attention block's projections at 1.25x theirs (30.7K against 24.6K: the per-row
  query norms and RoPE between the MMs). What the MXU does past the linear floor (19.1K) is the
  attention's q.K^T and P.V MMs: each (row, KV head) pair is a flash-attention entry of its own,
  M = G = 2 query rows on 4 columns, half the MXU idle.
- **Attention grows with blocks, slowly.** 3.7K a row and layer at position 100 (one block),
  7.3K at 256 and 267 (two): about 0.46K a (row, KV head, block). At position 1024 (five
  blocks) it is 41.1K of 110.8K a row (37%).
- **A run's fixed cost is nil**: 1 layer 41.8K a row, 2 layers 83.6K (the layers are a
  hardware loop).
- **The card agrees within 6%.** int8, 4 rows: the co-simulation gives 81.8K a row at position
  100 and 89.2K at 256 and 267, 1.145M and 1.249M cycles a token over 28 layers. The card's
  prompt runs (docs/prefill.md 7, fmvf 2026-10-02) took 288.0M cycles for 236 tokens at 0-235
  in 59 runs of 4 (1.214M a token, less the last run's logits) and 21.4M for 13 tokens at 267
  in three runs of 4 and one of 1 with logits (~1.33M a token in the 4-row runs, taking the
  1-row run as an int8 decode step's ~5.4M). Prompt runs cost ~6% more than compile_rows'
  programs (run-time positions: RunWords, the mask tiles' LDs, V^T appended a row at a time).
- The whole model at that rate: 1.17M cycles a token (fp4), 114 tokens/s at 133.33 MHz.

## 2. How many rows a weight pass needs

A pass of T rows over one 128-byte weight chunk costs max(stream, MXU) cycles (docs/isa.md MM,
docs/dsp_pumping.md 2 on dsp-pump's branch):
- the stream: 1 / eta, eta the chunks a cycle the memory delivers to the port (at most one: the
  port is 128 bytes a cycle). The card's DDR3-1066 through LiteDRAM: 0.82-0.85 while decoding
  (docs/board.md); the BCU1525's DDR4 (two channels a slice: 25.6 GB/s at 1600, 38.4 at 2400)
  and the F2's HBM deliver more than one slice's port takes (12.8 GB/s at 100 MHz, 17.1 at
  133.33), so eta = 1 there;
- the MXU: int8, ceil(T / MCOLS) cycles (replayed groups); 4-bit, two blocks a chunk: 1 cycle
  with PAIR (T <= MCOLS / 2), else 2 ceil(T / MCOLS).

Cycles a chunk and row, max(stream, MXU) / T (`tools/tile_model.py knee`; estimated from the ISA
rates, the co-simulations of section 4 measure them):

| weights, eta | MCOLS | T = 1 | 2 | 4 | 8 | 16 | 32 | the knee (fewest rows at the floor) | the floor |
|---|---|---|---|---|---|---|---|---|---|
| 4-bit, 0.85 (YPCB) | 4 | 1.176 | 0.588 | **0.500** | 0.500 | 0.500 | 0.500 | 4 | 0.500 |
| | 8 | 1.176 | 0.588 | 0.294 | **0.250** | 0.250 | 0.250 | 8 | 0.250 |
| | 16 | 1.176 | 0.588 | 0.294 | 0.147 | **0.125** | 0.125 | 16 | 0.125 |
| 4-bit, 1.0 (port-bound) | 4 | 1.000 | **0.500** | 0.500 | 0.500 | 0.500 | 0.500 | 2 | 0.500 |
| | 8 | 1.000 | 0.500 | **0.250** | 0.250 | 0.250 | 0.250 | 4 | 0.250 |
| | 16 | 1.000 | 0.500 | 0.250 | **0.125** | 0.125 | 0.125 | 8 | 0.125 |
| int8, 0.85 (YPCB) | 4 | 1.176 | 0.588 | 0.294 | **0.250** | 0.250 | 0.250 | 8 | 0.250 |
| | 8 | 1.176 | 0.588 | 0.294 | 0.147 | **0.125** | 0.125 | 16 | 0.125 |
| | 16 | 1.176 | 0.588 | 0.294 | 0.147 | 0.074 | **0.062** | 32 | 0.062 |
| int8, 1.0 (port-bound) | 4 | 1.000 | 0.500 | **0.250** | 0.250 | 0.250 | 0.250 | 4 | 0.250 |
| | 8 | 1.000 | 0.500 | 0.250 | **0.125** | 0.125 | 0.125 | 8 | 0.125 |
| | 16 | 1.000 | 0.500 | 0.250 | 0.125 | **0.062** | 0.062 | 16 | 0.062 |

- **The linear layers need about MCOLS rows a pass (4-bit) or 2 MCOLS (int8), not 32-256.**
  Past the knee the MXU's multiply rate is the bound and more rows change nothing but the DRAM
  traffic. The knee scales with the MXU's rate per chunk of stream: MCOLS 8, or a 2x-pumped
  MCOLS 4 (docs/dsp_pumping.md), doubles it; MCOLS 16 doubles it again.
- **What today's R = 4 leaves** (at the card's eta): 4-bit at MCOLS 4, nothing; int8 at MCOLS 4,
  15% of the linear layers; 4-bit at MCOLS 8, 15%; int8 at MCOLS 8, 58% (0.294 -> 0.125: four
  rows use half of eight columns). Section 4.1 measures it on whole rows: -6%, -7% and -36%
  from 4 to 8 rows.
- **A wider port** (2 chunks a cycle, which DDR4-2400 and HBM could feed) halves the stream term;
  the knee in rows falls, the floor stays. It speeds decode and below-knee runs, not a pass at
  the floor.
- **What the knee leaves out** is everything that is not a weight stream: attention (its K / V
  streams once per row and KV head today, at G of MCOLS columns), the vector work per row, and
  the dependency chains between them. Those are what the tile restructures.

For a Qwen3-0.6B layer (61.4K chunks of 4-bit weights, 122.9K int8) the floor is 30.7K cycles a
row at MCOLS 4 and 15.4K at MCOLS 8; a Qwen3-4B-shaped layer (394K / 788K chunks) 197K and 99K
(`tools/tile_model.py layers`).

## 3. Design

### 3.1 What "weight-stationary over token tiles" is on this MXU

The MXU streams the weights (one 128-byte chunk a cycle at most) against a stationary operand
of activation rows in the ACT RAM. So a weight chunk is used by every row the ACT RAM holds:
with ACT_ROWS > MCOLS the MXU keeps each streamed chunk in its FIFO and replays it for
ceil(M / MCOLS) groups of MCOLS rows (docs/isa.md, MM: one DRAM pass serves all M rows). A tile
of T <= ACT_ROWS rows is therefore one MM per weight slice, each chunk fetched once and held for
T / MCOLS cycles. The alternative orientation (weights in the ACT RAM, activations streamed)
does not exist on this MXU: the ACT RAM is int8 only (no 4-bit weights), it is written only by
QACT from fp32 TMEM rows, and the scales would multiply in the other order (not the ISA's bits).

What limits T today is not the MXU but where the rows live:
- **TMEM.** Today's kernel (qwen3_rows) holds every row's activations at once: 6.7-7.6K words a
  row on Qwen3-0.6B (compiled: 30.6K words at 4 rows, 55.8K at 8, 107.9K at 16, 215.6K at 32),
  so a 64K-word TMEM takes 8 rows, and 3-4 on the larger layouts.
- **ACT RAM rows.** The card's bitstream has ACT_ROWS = MCOLS = 4. MM replay is in the RTL
  (otpu_actram's groups, otpu_mxu's replay; `make bit ACT_ROWS=8` was built and measured on
  2026-09-27 at MCOLS 2, docs/benchmarks.md) but not in production.
- **IMEM.** Attention is unrolled per row, KV head and block, so a run's program grows with
  its rows and context (docs/prefill.md 10).

### 3.2 The tiled run

`opentpu/llm/tiled.py` (qwen3_tile, compile_tile). For a tile of T rows at positions p0 ..
p0 + T - 1, every layer (the layers' hardware loop as today):

1. **Attention input.** RMSNorm and quantization of all T rows in one QACT (RSCALE / CSCALE,
   the instruction today's peephole makes, emitted directly so no [T, H] product is traced).
2. **Per KV head j** (the next head's projections stream while this head attends):
   - K_j, V_j = MM(xs, W_k[j]), MM(xs, W_v[j]): one weight pass for T rows (W split by rows:
     each output element is the same MM sum, docs/isa.md);
   - K norm and RoPE per row, all T rows' K / V appended (QST) before any row attends;
   - Q_j = MM(xs, W_q[j]) [T, G d], normed and rotated into [T G, d] (row (r, g) = r G + g);
   - the attention in packed tiles of P = MCOLS / G rows: one MM of P G query rows streams the
     head's K (and P.V its V^T) once for P rows, where today each row streams them alone. The
     blocks every row of the packed tile sees whole run in the attention's hardware loop as
     today; the last block(s) take a mask per row group (RowMasks: the score buffer filled with
     -0, then -inf past each row's token, then q.K^T accumulated into it with the row maxima,
     MM ACC + RMAX). A row group that sees none of a block (a packed tile across a block
     boundary) gets all -inf: m stays, p = +0, alpha = 1, acc and l pass unchanged;
   - the head's outputs acc / l quantized into its columns of W_o's operand (ol.stationary +
     ol.quantize_into: per row and block, so the same bytes and scales as quantizing [T, n_q d]
     at once).
3. **Output projection and residual** by parts of the output columns: y_p = MM(o, W_o[p]),
   x[:, p] += y_p. TMEM holds [T, H / parts] of y, parts chosen so that is at most 1/8 of TMEM.
4. **MLP.** xs as in 1; gate and up in pieces of `sub` columns (software-pipelined: the next
   piece streams while the VPU computes silu(g) u), each piece quantized into its chunk's
   operand, which stays in the ACT RAM (F / D blocks of T rows: 24 on the 0.6B, 76 on a 4B);
   then the down projection by output parts, each part summing the chunks in swiglu_down's
   order (dot, then dot(acc=)), so each element is today's sum.

Bit for bit: every row's arithmetic is today's (section 7). What TMEM holds no longer grows
with H, n_q d or F, only with T times a KV head's width, a gate / up piece and an output part.
At T <= MCOLS the tile needs no replay: T = 4 runs on the card's bitstream as it is.

### 3.3 Activations: in TMEM, or through DRAM

Compiled for the card's 64K-word TMEM and 4K-instruction IMEM (the first 2 layers; fp4, int8
head, position 256): Qwen3-0.6B fits T = 16 (2,874 instructions) and T = 24 (3,994), not 32 (the
[32, 1024] residual is half of TMEM); a Qwen3-4B-shaped layout (H 2560, F 9728, 32 / 8 heads)
fits T = 12 (3,956 instructions; today's kernel fits 3-4 rows there, and none at 4 rows past
position ~1000 without fit_chunk's shrinking). So on the dense models up to 4B the tile stays in
TMEM at the T that matters (section 2), with no spill.

Past that, the residual moves to DRAM. A design that needs no new instruction:
- x lives in a DRAM scratch in D-column blocks, [H / D][T][D] fp32 (each block contiguous);
- the norm's sum of squares is accumulated over the blocks in the 64 partials of the ISA's
  isum_64 (partial q gets columns q, q + 64, ... in order: two MUL + ADD passes of [T, 64] per
  block) and folded at the end as RSSQ folds: bit for bit RSSQ's sum (docs/isa.md, Sums);
- the QACT then runs per D-column block (rows 0 .. T - 1, KB = 1, RSCALE / CSCALE): per-block
  quantization, so the same operand;
- the residual adds load and store a block column at a time.
Spill traffic: about 6 x 4H bytes a row and layer (two norms' reads, two residual
read-modify-writes): 24 KB on the 0.6B against 7.9 MB of fp4 weights a layer, i.e. 0.19K
chunks a row against 61.4K / T. At T = 64 that is 1.15K chunks of port traffic a row and layer
in 30.7K MXU cycles (4% of the port): the spill rides in the headroom the MXU-bound phases
leave. Its cost is instructions and latency, not bandwidth. Not built: section 4 shows no T
past 16 that would need it.

### 3.4 What the compiler and the ISA need

| need | today | for tiles |
|---|---|---|
| MM over more rows than MCOLS | replay in the ISA, ISA simulator and RTL (M <= ACT_ROWS <= 255; a replayed row stays in the 1024-chunk FIFO until its last group) | T = MCOLS needs none. T = 16: a bitstream with ACT_ROWS 16: the ACT RAM's RAMs are a 256-bit word per column and 32-byte piece, GROUPS x BLOCKS deep; at MCOLS 4, 4 groups x 128 blocks = 512 words fill the 64 BRAM36 that 128 words use today (estimated from the RAM shapes, not synthesized). ACT_ROWS 32: 128 BRAM36 (+64) |
| MM over more rows than ACT_ROWS | no | not needed: T <= ACT_ROWS (16) is past the knee (section 2) |
| an operand built in parts | no | `ol.stationary(rows, cols)` + `ol.quantize_into(st, x, col)` (prototype) |
| an operand that does not pin its TMEM source | no | `ol.quantize(x, free=True)` (prototype) |
| norms without [T, H] products | the peephole after tracing the products | `ol.quantize(x, rscale=, cscale=)`, `ol.sumsq(x)` (prototype; the same instructions) |
| attention rows of several tokens in one MM | no: RMAX / ASCALE need M <= MCOLS | per-row-group masks (RowMasks, TileSeq; prototype): P = MCOLS / G rows a K stream |
| attention of a whole tile in one MM | no | RMAX and ASCALE over replayed groups (M row maxima and M rescale factors in the epilogue: an RTL change). The MXU's cycles stay those of packed tiles (replay runs MCOLS rows a cycle too); it would cut MMs, their fills and the K / V streams: at most the ~9K a row (2 layers) the tile's attention takes over its MXU work at MCOLS 4 (estimated) |
| address registers | 8 arguments, up to 15 registers | the tile at compile-time positions takes 1 (the layer loop's); at a run-time position its rows share one tpos (RunPos.offset), so the prompt run's 6 coefficients serve T rows |
| IMEM | 4K instructions | T = 16 at MCOLS 4: 2,874 instructions at position 256, 4,282 at 1000, 7,424 at 2000; MCOLS 8 (P = 4): 1,946 / 2,650 / 4,224. Past ~1K tokens the packed tiles need the looped attention of docs/prefill.md 10 (a loop over a head's packed tiles: affine q tile, output rows and mask rows) |
| DMA | LD / ST / QST | T rows' K / V appends (QST of [T, d]; V^T byte-strided as today), the I/O rows; with the spill, block-column LD / ST |

### 3.5 Against a bigger TMEM

A bigger TMEM alone lets today's kernel run more rows: 108K words for 16 rows of Qwen3-0.6B,
216K for 32. Measured (section 4.1), that buys at most 1-2% a row: its linear layers are at the
MXU's rate from MCOLS rows (4-bit) or 2 MCOLS (int8, which 64K words already fit), and its
attention stays per row. The restructured run fits today's 64K words at T = 16-24, and a
TMEM of 256K words gives it nothing (T = 16: 77.0K a row against 76.4K at MCOLS 4, 44.6K
against 44.4K at MCOLS 8; larger MLP pieces pipeline less). A bigger TMEM is worth something
to the tile only past T = 24 (no spill), which section 2 says no MXU here needs below MCOLS 16.

**A co-simulation trap.** Every TMEM above 64K words was co-simulated with otpu_seq tracking
TMEM ranges in 16 bits (TAW): an instruction touching a word past 65,535 conflicted with
everything, as a barrier does. Today's kernel at 4 rows took 112.7K a row with a 256K-word
TMEM against 83.6K with 64K, the same program. This branch makes TAW follow TMEM_WORDS
(otpu_seq's TMEM_WORDS parameter; 16 bits up to 64K words, so the card's build does not
change); tmem-pump's branch has the same change (4eca9a76, with the STREAM descriptors kept in
the first 64K). Every TMEM-256K figure here is with the change.

## 4. Measured: the tiled run against today's

`tools/perf_tiled.py` (section 1's setup): "ldc" is the card's memory path (DDR3-1066 through
LiteDRAM's controllers, the core at 133.33 MHz), "port" a memory that delivers a chunk every
cycle with a 40-cycle latency, which stands in for the BCU1525's DDR4 at 1600 or 2400 and the
F2's HBM (each delivers more than a slice's port takes, section 2: their cycles are the same).
MCOLS 8 is virtual (the RTL at MCOLS 8; the card's part does not fit it). Cycles a row for the
first 2 layers, positions from 256, no logits; tokens/s are the whole model at that rate.

### 4.1 Today's kernel with replay

compile_rows over R rows in one run, ACT_ROWS = max(MCOLS, R), TMEM 64K words up to 8 rows
and 256K past that:

| weights | MCOLS | memory | R = 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| fp4 | 4 | ldc | 83.6K | 82.6K | 83.1K | 84.7K |
| | 4 | port | 82.7K | | | |
| | 8 | ldc | 57.7K | 53.8K | 53.0K | 54.0K |
| | 8 | port | 55.2K | 53.0K | | |
| int8 | 4 | ldc | 89.2K | 83.8K | 83.8K | 85.0K |
| | 8 | ldc | 89.2K | 56.8K | 54.5K | 54.4K |

- **Past the knee more rows a pass give nothing.** The port-B chunks a row fall 2.8x (fp4
  41.2K -> 14.5K, 4 -> 32 rows) and the cycles stay within 2%.
- **Up to the knee they give what section 2 predicts**: int8 at MCOLS 4 -6% (4 -> 8 rows),
  fp4 at MCOLS 8 -7%, int8 at MCOLS 8 -36%. Each fits 64K words of TMEM at 8 rows; what it
  needs is ACT_ROWS 8.
- **The memory is not the bound**: the port's full rate is 1-4% faster.

### 4.2 The tiled run

compile_tile, TMEM 64K words, ACT_ROWS 16 (T = 4 also on ACT_ROWS 4, the card's: 79.5K, the
same). The best of today's kernel at the same configuration for comparison:

| weights | MCOLS | memory | T = 4 | 8 | 16 | today's best (rows) | tile at 16 against it |
|---|---|---|---|---|---|---|---|
| fp4 | 4 | ldc | 79.5K | 76.5K | 76.4K | 82.6K (8) / 83.6K (4) | -7.5% / -8.6% |
| | 4 | port | 78.5K | 76.5K | 76.5K | 82.7K (4) | -7.5% |
| | 8 | ldc | 49.1K | 45.1K | 44.4K | 53.0K (16) / 57.7K (4) | -16% / -23% |
| | 8 | port | 47.9K | 44.6K | 44.4K | 53.0K (8) | -16% |
| int8 | 4 | ldc | 84.7K | 76.5K | 76.4K | 83.8K (8) / 89.2K (4) | -8.8% / -14% |
| | 8 | ldc | 82.5K | 48.8K | 44.4K | 54.4K (32) / 89.2K (4) | -18% / -50% |

Where the cycles go (MCOLS 4, fp4; the tile's attention phase includes the next head's
projections, which it overlaps, so the attention block is compared whole: input norm,
projections, attention, output projection):

| run | attention block | mlp | MXU busy |
|---|---|---|---|
| today, R = 4 | 45.5K | 38.1K | 80.5K |
| tile, T = 4 | 41.5K | 37.9K | 76.5K |
| tile, T = 16 | 38.7K | 37.7K | 74.4K |
| MCOLS 8: today, R = 8 | 33.6K | 20.3K | 50.5K |
| MCOLS 8: tile, T = 16 | 25.1K | 19.3K | 42.4K |

- **The gain is the attention block's.** Packed tiles put MCOLS / G rows' queries on the
  columns that one row's G = 2 left idle (at MCOLS 4: the attention MMs at full width, -4.0K at
  T = 4); at T = 8-16 its projections pass the weights once for all rows and overlap the
  attention (-2.8K more). The MLP is at its weights' floor either way (1.02x).
- **MCOLS 8 is where it matters**: four rows share a K / V stream and the projections reach the
  floor that 4-row runs cannot (PAIR's pairs stream-bound at 0.294 a chunk): -23% against 4-row
  runs, -17% against today's kernel at its best (16 rows with replay).
- **What is left over the floor** at T = 16: 15.0K a row at MCOLS 4 (76.4K against 61.4K),
  13.7K at MCOLS 8: the attention's MXU work (estimated ~5K a row at MCOLS 4: K and V^T take
  ~650 cycles a packed tile and KV head over two blocks) and its MMs' fills and dependency
  chains.

**Longer context** (position 1024, five blocks; fp4):

| MCOLS | today, R = 4 | today, R = 8 | tile, T = 16 | change |
|---|---|---|---|---|
| 4 | 110.8K (attn-core 41.1K) | | 92.4K | -17% |
| 8 | 84.1K | 82.0K | 58.4K | -29% |

The tile's T = 16 program is 4,986 instructions there at MCOLS 4 (over the card's 4K: the
looped attention first, 3.4) and 3,002 at MCOLS 8.

**Larger and hybrid models** (fp4, from position 256; a Qwen3-4B-shaped layout's 2 layers: H
2560, F 9728, 32 / 8 heads (G = 4), d 128; Qwen3.5-0.8B's first 4 layers: 3 DeltaNet, 1 gated
attention):

| model | MCOLS | today, R = 4 | tile, T = 4 | T = 8 | T = 12 | best change |
|---|---|---|---|---|---|---|
| Qwen3-4B shape | 4 | 428.7K | 428.2K | 423.9K | 434.2K | -1.1% |
| | 8 | 254.2K | 249.3K | 226.9K | (TMEM) | -10.7% |
| Qwen3.5-0.8B (today's kernel only) | 4 | 282.4K | | | | |
| | 8 | 233.7K | | | | |

- **The 4B shape is its MLP**: 294.4K of 428.7K a row (69%) at 1.01x its weights' floor; the
  attention is 5% (20.4K) and its G = 4 query heads already fill MCOLS 4's columns (P = 1: no
  packing). So the tile has nothing to take at MCOLS 4. At MCOLS 8, 8 rows reach the 4-bit
  floor that 4-row paired runs miss (stream-bound at 0.294 a chunk): -10.7%, the MLP at its
  floor (147.9K against 145.9K).
- **Qwen3.5-0.8B is its DeltaNet**: the DeltaNet rows (perf_tiled's deltanet phase and the
  time outside the named phases: the recurrence, convolutions and gates) take 173K of 282.4K a
  row (61%; 170K of 233.7K at MCOLS 8, 73%), per row on the DSTEP and the VPU. MCOLS 8 gives
  -17% (docs/prefill.md 8 measured -17.2% on the whole model); a tile (not ported) would add at
  most the linear layers' 15% stream share at MCOLS 8, about -4% (estimated), nothing at
  MCOLS 4. Its prefill lever is the DSTEP (16 lanes, docs/prefill.md 8).

So the tile's gain is the attention's share and the MXU's appetite for rows: largest on small
models with many KV heads (Qwen3-0.6B, G = 2), at long context and at MCOLS 8; about nothing on
a 4B-class fp4 model on the card.

### 4.3 The whole model, by board

Qwen3-0.6B, fp4, prefill tokens a second of device time at positions ~256 (28 layers from the
2-layer cycles; no host time, no logits):

| board | today (R = 4) | today at its best | tile, T = 4 | tile, T = 16 |
|---|---|---|---|---|
| YPCB (MCOLS 4, DDR3-1066, 133.33 MHz) | 114 | 115 (R = 8, ACT_ROWS 8) | 120 (today's bitstream) | 125 (ACT_ROWS 16) |
| the same at position 1024 | 86 | | | 103 |
| MCOLS 8, DDR3-1066, 133.33 MHz | 165 | 180 (R = 16) | 194 | 214 |
| BCU1525 / F2, one slice, MCOLS 4, 100 MHz | 86 | | 91 | 93 |
| BCU1525 / F2, one slice, MCOLS 8, 100 MHz | 129 | 135 (R = 8) | 149 | 161 |

int8 weights (MCOLS 4 at 133.33 MHz): 107 today, 114 at R = 8, 125 with the tile at T = 8;
at MCOLS 8: 107 / 175 (R = 16) / 214 (T = 16).

## 5. MoE

### 5.1 Expert-major is a tiled prefill already

Layer-major prefill with expert-major MoE (docs/offload.md 13.11-13.13, merged, off by default)
runs each MoE layer of a chunk (up to 512 rows, the rows in DRAM records) as mixer runs of R = 2
rows, then one expert run that buckets the chunk's R k entries by expert and makes passes of two
rows of one expert. That is the tiled run's structure with the MoE layer's tile being "the
rows that chose this expert": the rows live in DRAM between the phases (3.3's spill), each pass
loads its rows, quantizes them and streams the expert once. Tiling generalizes it in two places:

- **The mixer runs become tiles.** Their dense weights (attention or DeltaNet projections, the
  shared expert or Gemma 4's dense MLP, the router) are streamed once per run, so at R = 2 a
  run's weight stream is shared by 2 rows. With the tile layout (3.2) a mixer run takes 8-16
  rows, its dense weights once for all of them, its attention packed.
- **The expert passes take MCOLS / 2 rows (4-bit experts, PAIR) or up to ACT_ROWS (replay).**
  A 4-bit expert at MCOLS 4 is at the floor with pairs already (two rows paired cost one
  row's cycles; 4 rows cost two pairs'); at MCOLS 8 a pass of 4 paired rows costs what 2 do
  now. Replay past that saves the expert's re-reads (a 26B expert used by 13 rows is streamed
  7 times in pairs, once in a 16-row pass) but not cycles: it matters only where the stream is
  shared (the host's H2C writes during compute: 0.17 s a GB, docs/offload.md 12.6).

One expert's FFN on M rows, co-simulated (tools/moe_prefill_cosim.py expert: the card's memory
path, ACT_ROWS 16, TMEM 64K; the expert's cycles less the same program without it; ms at
133.33 MHz):

| expert | MCOLS | M = 1 | 2 | 4 | 8 | a row at the floor |
|---|---|---|---|---|---|---|
| Qwen3.5-35B-A3B (1.67 MB fp4) | 4 | 0.101 | 0.100 | 0.194 | 0.389 | 0.049 |
| | 8 | 0.101 | 0.100 | 0.104 | 0.205 | 0.024 |
| gemma-4-26B-A4B (3.45 MB fp4) | 4 | 0.210 | 0.210 | 0.393 | (TMEM) | 0.101 |
| | 8 | 0.210 | 0.210 | 0.216 | (TMEM) | 0.050 |

- **At MCOLS 4 a pair is the floor**: 4 rows cost two pairs (0.194 against 2 x 0.100), 8 rows
  four. Replay saves the stream, not cycles, as section 2 has it.
- **At MCOLS 8, 4 rows cost what 2 do** (0.104 against 0.100): the expert passes of the
  expert-major run should take MCOLS / 2 rows, not 2.
- The 26B's 8 rows (and either model's 16) do not fit 64K words with this kernel (swiglu_down
  keeps the [M, H] input and [M, F] products); the tile's MLP (3.2: pieces, parts) would fit
  them, for the stream's sake only.

### 5.2 What it gives the two MoE models

Each layer's dense weights (from the config.json shapes):

| model | dense weights a layer | format (as run) | a token's dense MACs | expert | experts a token |
|---|---|---|---|---|---|
| gemma-4-26B-A4B | 52.8M (sliding) / 70.2M (global) | int8 | 1.67G (30 layers) | 5.95M (3.45 MB fp4) | 8 of 128 |
| Qwen3.5-35B-A3B | 37.4M (DeltaNet) / 30.9M (attention), with the shared expert | fp4 | 1.43G (40 layers) | 3.15M (1.67 MB) | 8 of 256 |

**YPCB (experts offloaded, Gen2 link).** The measured split of a pooled R = 2 prompt
(docs/offload.md 13.7: 35B 134 tokens, 26B 124 tokens): layer runs' compute 8.29 s (35B) and
13.72 s (26B), of which the mixer ("pre", up to the request) is 1.94 / 1.18 ms a run (35B
DeltaNet / attention) and 3.33 / 3.85 ms (26B sliding / global), plus 3.46 / 2.12 s of exposed
misses.

- **26B: the mixer runs are its int8 dense weights at 2 rows a pass.** 52.8 MB a sliding layer is
  412K chunks: at 3.33 ms a run that is the stream at 93% of the port, so the mixer is
  stream-bound at 1.67 ms a row. In tiles of T >= 8 at MCOLS 4 the same weights take the int8
  floor, 0.25 cycles a chunk and row: 0.77 ms (sliding), 1.03 ms (global) a row. Over 124
  tokens and 30 layers: **about -3.1 s** (estimated) of the pooled prompt's 16.25 s after the
  compile fix (-19%), or -2.6 s if the tile runs these layers at 1.25x the floor as it does
  Qwen3-0.6B's (section 4.2). T = 4 on today's bitstream (stream-bound at 0.294 a chunk and
  row) gets -2.2 to -2.6 s of it; MCOLS 8 (int8 floor 0.125) -4.1 to -4.8 s.
- **The tiled mixer needs expert-major's split.** In today's layer-major runs each union expert
  runs on all R rows of the run, so more rows mean a larger union, and the request is one line
  of R k <= 16 ids (docs/offload.md 13.2: R = 2 is the point for 4-bit experts). Expert-major
  runs the mixer apart from the experts, so its mixer runs can take any T; the tile goes on
  that path.
- **35B: little.** Its dense layers are fp4, where 2 paired rows are at the stream's rate and the
  floor is 15% below; its DeltaNet recurrence (32 value heads, 2,048 cycles each a row on the
  8-lane DSTEP: 0.49 ms a row and layer) does not tile. About -0.3 to -0.5 s (estimated).
- **Expert-major (dormant) is the other half.** Its predicted gain (docs/offload.md 13.11's model,
  needs streamed during the mixer runs, against today's pooled base) is -1.7 s (35B) and -1.9 s
  (26B): the link side, which tiling does not touch. For the 35B it is most of what is left
  without new RTL; for the 26B the tiled mixer is the larger half, and the two add (the tiled
  mixer runs shorten the phase that the need-streaming hides under, so the overlap shrinks:
  the sum is an upper bound).
- **The card A/B of expert-major needs card time** (session pfB, staged on opentpu in
  ~/otpu-build/offload/card2/pfB/README.md with its predicted column; the lead schedules it).
  Nothing in this note changes that session's arms.

**BCU1525 / F2 (experts resident, no link).** Estimated per token and slice at the floor, MCOLS
4, the port's rate (lower bounds: the tile ran Qwen3-0.6B at 1.25x its floor):

| model | dense | experts | DeltaNet steps | today's R = 2 | tiled |
|---|---|---|---|---|---|
| 26B | 3.26M cycles (int8; R = 2 at the port: 6.5M) | 2.79M (fp4: pairs are at the floor) | | ~9.3M + attention | ~6.1M + attention (-35%) |
| 35B | 2.80M (fp4; R = 2 paired: at the floor) | 1.96M | 1.97M (30 layers x 32 heads x 2,048) | ~6.7M | ~6.7M |

At 100 MHz (the BCU1525's development clock) that is ~10.7 -> ~16 tokens/s a slice for the 26B
and ~15 for the 35B. At MCOLS 8 the MXU parts halve and the 35B's DeltaNet steps (1.97M) become
its bound: its prefill lever is the recurrence (a 16-lane DSTEP, docs/prefill.md 8, or the
chunked delta rule as MXU work: an algorithm change, not bit-exact, needing its own accuracy
check), not more rows.

## 6. Many users

Batched decode (opentpu/llm/batch.py, docs/serve.md) runs a step's rows in blocks of MCOLS
slots, one run each, every run streaming every weight. With replay a whole step takes one pass:
rows of different sequences are rows of one stationary operand (up to ACT_ROWS), each row's
attention an entry of its own over its own cache (rows of different sequences share no K / V,
so they do not pack: the tile's main gain does not apply). What one pass of 2 MCOLS rows gives
against two runs of MCOLS is what today's kernel measures at 8 and 16 rows against 4 and 8
(section 4.1; 2 layers at position 256, cycles a row):

| MCOLS | weights | runs of MCOLS rows | one pass of 2 MCOLS rows | change |
|---|---|---|---|---|
| 4 | fp4 | 83.6K | 82.6K | -1.2% |
| 4 | int8 | 89.2K | 83.8K | -6.1% |
| 8 | fp4 | 53.8K | 53.0K | -1.5% |
| 8 | int8 | 56.8K | 54.5K | -4.0% |

- **Longer passes buy 1-6%**, int8 most (its knee is 2 MCOLS). A decode step's int8 LM head
  (18% of a Qwen3-0.6B decode row) is in the same case: about 3% more for a step with logits
  (estimated). It needs ACT_ROWS 8 at MCOLS 4 and nothing else (64K words of TMEM hold 8 rows).
- **What raises the aggregate is the MXU's rate**: 8 users' rows at MCOLS 8 in one run cost
  53.8K a row against 83.6K in two runs at MCOLS 4 (-36%; fp4). So 4-8 users gain from MCOLS 8
  or a pumped MCOLS 4, not from longer passes.
- **The attention is per user.** At context 1024 the card's 4-row step took 60.6 ms against
  46.3 ms at 64 (docs/serve.md): per-row attention, which no tiling across users removes.
- **Mixed runs** (a prompt's rows with other users' decode rows, docs/serve.md) are where the
  tile helps a multi-user server: a prompt's rows ride the decode step's weight pass with their
  attention packed, instead of a feed run of MCOLS rows per pass.

## 7. The prototype

- `opentpu/llm/tiled.py`: qwen3_tile and compile_tile (Qwen3 and the Llama-likes of this code
  path; one slice; compile-time positions). Qwen3.5's gated attention and DeltaNet rows, LFM2
  and Gemma 4 are not ported.
- `opentpu/kernels/attention.py`: RowMasks (a packed tile's masked block) and TileSeq (whole
  blocks looped, then the listed ones); the existing paths are unchanged.
- `opentpu/compiler.py`, `opentpu/language.py`: ol.stationary / ol.quantize_into,
  ol.quantize(free=, rscale=, cscale=), ol.sumsq. Defaults leave every existing program as it
  was.
- `rtl/seq/otpu_seq.sv`, `rtl/top/otpu_slice.sv`: the TMEM hazard ranges' width follows
  TMEM_WORDS (3.5; no change at 64K words).
- `tools/perf_tiled.py` (the co-simulations of section 4), `tools/tile_model.py` (section 2's
  table, the MoE pass counts).
- `tests/test_tiled.py`:
  - a tile against today's prefill runs of MCOLS rows over the same tokens (Engine's cache of
    the tokens before them; the ISA simulator): every layer block (weights and KV cache) and
    the last row's logits bit for bit, at MCOLS 4 int8 (16 rows), MCOLS 4 fp4 (16 rows across
    the attention block at 256), MCOLS 2 int8 (6 rows: one row a packed tile, a partial MXU
    group) and MCOLS 8 fp4 (14 rows: 4 a packed tile, a partial last one);
  - the packed tiles' blocks and masks (tile_seq);
  - the RTL against the ISA simulator (DRAM and TMEM equal), on the board's memory path: a
    12-row tile across a block at MCOLS 4 fp4 and MCOLS 8 int8 (ACT RAM of 16 rows), and a
    4-row tile on the card's own configuration (MCOLS 4, ACT_ROWS 4, fp4).

**The order differences, exactly.** None in the tile itself: an MM's results do not depend on
how many rows share it (docs/isa.md), splitting W_q / W_k / W_v / W_o / W_down by rows gives each
output element the same MM sum, the down projection's parts keep swiglu_down's order of
chunks, per-block QACTs give per-block bytes and scales, and a masked or fully masked block is
exact (section 3.2). The one difference against today's runs is PAIR: a run of at most
MCOLS / 2 rows of 4-bit weights is paired (two blocks a cycle), and PAIR adds a chunk's two
blocks before the partial sums (docs/isa.md, Column reuse), so its sums round differently from
an unpaired MM. A tile pairs exactly when today's run would (2T <= MCOLS); a tile of more rows
matches today's runs of more than MCOLS / 2 rows bit for bit, and differs from a paired run
(e.g. the 2-row runs of the Qwen3.5-4B, docs/prefill.md 7) by that rounding only. The test's
MCOLS 8 case takes runs of 8 and 6 rows for that reason (a run of 4 would pair).

## 8. Plan, and what it gives each workload on each board

In order; each step stands on its own.

1. **The tile kernel in the engine's prompt runs** (software; at T = MCOLS it runs on today's
   bitstream). Scheduled with the MXU work below rather than ahead of other card work: on the
   card alone it is worth -5% on Qwen3-0.6B (-17% at position 1024 with T = 16) and about
   nothing on 4B-class fp4 models.
   - run-time positions: the tile as a prompt run (RunPos: the rows share one tpos), so one
     program serves every tile of a bucket, cached as today's prompt runs. Its masks come from
     a table as the prompt runs' additive masks do (docs/prefill.md 9), a second one: per
     position q, row group i (G rows) masked at q + i, so a packed tile's mask is one LD at
     the table + tpos x its stride (1 MiB at 256 x MCOLS 4, as the first);
   - the looped attention of docs/prefill.md 10 over a head's packed tiles, so the program
     stays under 4K instructions past ~1K tokens;
   - the planner's T: the knee (MCOLS for 4-bit, 2 MCOLS for int8) bounded by what trace_fit
     fits in TMEM and by ACT_ROWS; Engine.prefill takes the tile where its cost a row is lower
     than today's (as prefill.covers decides prompt runs today);
   - Qwen3 and the Llama-likes first, then LFM2, Gemma 4 and Qwen3.5's attention layers (the
     DeltaNet rows stay per row).
2. **With MCOLS 8 or a pumped MXU** (dsp-pump's and mcols' work) the tile is the prefill
   kernel: 4-bit's knee is then 8 rows (no replay at MCOLS 8; ACT_ROWS 8 for a 2x-pumped MCOLS
   4), and it takes Qwen3-0.6B from 1.45x to 1.88x the card's prefill today, the 4B shape from
   1.69x to 1.89x.
3. **ACT_ROWS 8-16 in a card bitstream that has room** (`make bit ACT_ROWS=16`; a FAST=1 build
   at 100 MHz first for the function: tests/test_tiled.py's RTL cases on the card, then a tile
   against compile_rows on a real prompt, token for token; then the timing build): int8 layers
   reach their knee (Qwen3-0.6B int8 -14% with the tile at T = 8), batched decode a pass of 8
   (int8 -6%).
4. **MoE**: the pfB A/B of expert-major first (card time, the lead schedules it), since the
   tiled mixer goes on its path (5.2); then its mixer runs as tiles (Gemma 4 26B first:
   section 5.2's largest estimate), expert passes of MCOLS / 2 rows (2 at MCOLS 4, 4 at 8).
5. **Later, measured first in co-simulation**: RMAX / ASCALE over replayed groups (fewer
   attention MMs; at most ~9K of the 76.4K at MCOLS 4, estimated); the DRAM spill only if a
   tile must outgrow TMEM (MCOLS 16).

What it gives each workload, by board (YPCB: the card, MCOLS 4, DDR3-1066, 133.33 MHz; MCOLS 8:
the faster MXU of dsp-pump's and mcols' work, same memory; BCU1525 and F2: one slice at the
port's rate, 100 MHz; device tokens a second of prefill at positions ~256, fp4; measured = the
co-simulations above, estimated = arithmetic on them):

| workload | YPCB today | YPCB, tile on today's bitstream | YPCB, + ACT_ROWS 16 | MCOLS 8: today's kernel -> tile | BCU1525 / F2 (MCOLS 4 today -> tile; MCOLS 8 tile) |
|---|---|---|---|---|---|
| single-user decode | unchanged (the tile is a prefill kernel) | unchanged | unchanged | unchanged | unchanged |
| prefill, Qwen3-0.6B | 114 | 120 (+5%) | 125 (+9%) | 165 -> 214 (+30%) | 86 -> 93; 161 |
| prefill, Qwen3-0.6B at ~1024 | 86 | | 103 (+20%) | 113 -> 163 (+44%) | |
| prefill, Qwen3-4B shape | 17.3 | 17.3 | 17.5 (-1.1%) | 29.1 -> 32.6 (+12%) | |
| prefill, Qwen3.5-0.8B | 78.7 | not ported (~0) | ~0 | 95.1 -> ~99 (estimated) | |
| multi-user 4 | unchanged | unchanged | unchanged | MCOLS 8: -31% a row (57.7K / 83.6K) | as YPCB |
| multi-user 8 | two runs of 4 | unchanged | one pass of 8: -1% (fp4), -6% (int8) | one run of 8: -36% a row | as YPCB |
| MoE prefill, 26B (offloaded; expert-major's mixer runs as tiles) | 16.25 s (124 tokens, pooled, after compiles) | -2.2 to -2.6 s (estimated) | -2.6 to -3.1 s (estimated) | -4.1 to -4.8 s (estimated) | resident: ~9.3M -> ~6.1M cycles a token (estimated) |
| MoE prefill, 35B | 13.32 s (134 tokens, likewise) | -0.3 to -0.5 s (estimated) | same | DeltaNet-bound | ~6.7M a token either way (estimated) |

Expert-major's -1.7 s (35B) / -1.9 s (26B) adds to the MoE rows once the pfB A/B confirms it.

## 9. Risks and open points

- **No card measurement of a tile.** Every cycle count here is a co-simulation of the card's
  memory path (DDR3-1066 through LiteDRAM's controllers) or of a memory at the port's rate
  standing in for the BCU1525's DDR4 and the F2's HBM. The co-simulation of today's kernel is
  ~6% below the card's prompt runs (section 1); a tile as a prompt run pays the same run-time
  position costs, so its gain is expected within a few points of section 4's.
- **The 2-layer proxy at position 256** has two attention blocks a row; a short prompt's rows
  have one (less attention, so less tile gain), a long one's more (section 4's position 1024:
  more gain).
- **ACT_ROWS 16 on the card is unbuilt.** Replay is in the RTL and ACT_ROWS 8 was built once
  (2026-09-27, at MCOLS 2: docs/benchmarks.md); the BRAM count (unchanged at MCOLS 4) is read
  off the RAM shapes, not synthesized, and the replay mux's timing at 133.33 MHz is unknown. A
  dev build answers both. Step 1 of the plan does not need it.
- **PAIR rounding.** A tile of more than MCOLS / 2 rows matches today's unpaired runs bit for bit
  but not today's paired runs of 2 rows (the Qwen3.5-4B's at MCOLS 4): a model whose reference
  tokens were made with paired runs gets new reference tokens (refcache keys), not a failure.
- **IMEM past ~1K tokens.** The prototype unrolls the packed tiles; a 16-row tile at position
  1024 is 5.0K instructions at MCOLS 4, over the card's 4K. The looped attention of
  docs/prefill.md 10 is needed before long prompts take tiles of 16 (today's 4-row run is
  2,357 instructions there).
- **Models not ported.** The prototype is Qwen3's code path; Qwen3.5 (gated attention with G = 4,
  so no packing at MCOLS 4, and the DeltaNet's per-row DSTEP), LFM2's convolutions and Gemma 4's
  sliding / global layers and per-layer embeddings each need their tile kernel. The MoE numbers
  of section 5 are estimates on measured parts, the expert-major gain a model's prediction
  until the pfB session.
- **Co-simulations with a TMEM over 64K words** made before otpu_seq's fix (3.5) are pessimistic
  wherever the compiler placed tiles past word 65,535; that includes this note's first runs (not
  used here) and any other branch's that did not carry the change.
- **A user decision:** none needed for the software steps. An ACT_ROWS 16 bitstream and its card
  qualification, and the pfB A/B, take card time, which the lead schedules.
