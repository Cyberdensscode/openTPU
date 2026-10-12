# The workload scoreboard

`tools/scoreboard.py` estimates what a hardware configuration gives on four workloads:

- single-user decode;
- prefill (TTFT);
- batched decode;
- MoE.

It uses an analytic model calibrated on the card and on the RTL co-sim. It is pure Python and
needs no checkpoint, RTL or card. `tools/scoreboard_cosim.py` runs the RTL co-sim and the
compiler for the points the model is calibrated on.

## Usage

```sh
# from the repository root; the tool imports nothing from opentpu/
python3 tools/scoreboard.py report                         # the YPCB in production
python3 tools/scoreboard.py report --board bcu1525 --mts 2400 --mhz 200
python3 tools/scoreboard.py report --board f2
python3 tools/scoreboard.py compare --mcols 8 --tmem-words 262144 --rows 16 --board bcu1525
python3 tools/scoreboard.py whatifs                        # the standard what-ifs on the YPCB
python3 tools/scoreboard.py whatifs --board bcu1525        # ... against another board
python3 tools/scoreboard.py validate                       # the model against the card and co-sim
python3 tools/scoreboard.py cosim pts.jsonl                # a co-sim campaign's layers against the model
python3 tools/scoreboard.py boards                         # the presets
python3 tools/scoreboard.py models                         # the model shapes and their overhead fits
python3 tools/scoreboard.py report --markdown --json out.json --models qwen3,phi4-mini
```

- **`report`**: prints the tables for one configuration.
- **`compare`**: prints the ratio of a configuration to its board preset for each workload.
  The overall score is the geometric mean of the four ratios.
- **`whatifs`**: runs the list below and prints one row of ratios for each what-if.

Each option overrides the board preset:

| option | what it changes |
|---|---|
| `--board ypcb\|bcu1525\|f2` | the preset (below) |
| `-S N` | slices, tensor-parallel: weights, KV heads and experts are split |
| `--mcols N` | MXU columns: `MCOLS x D` MACs a cycle and MCOLS rows a pass (int8; a PAIRed 4-bit row takes two) |
| `--act-rows N` | rows a weight pass (MM replay, "virtual MCOLS"): a run of R rows streams the weights `ceil(R / N)` times |
| `--mxu-pump X` | MACs a cycle x X at the same rows a pass (a multi-pumped MXU) |
| `--tmem-words N` | TMEM words: sets the prefill rows a run from the compiler's fit (`tools/scoreboard_rmax.json`) |
| `--rows N` | prefill rows a run, forced (tiling); `--imem 0` lifts the IMEM limit (looped attention) |
| `--batch-rows N` | rows in one batched-decode run (default: rows a pass) |
| `--vpu X` | vector-unit and DSTEP throughput x X |
| `--mhz F` | core clock |
| `--mem ddr3\|ddr4\|hbm --mts N --channels N --mem-eff E --gbps G` | a slice's memory |
| `--port-bytes N` | core port bytes a cycle a slice (128: two 64-byte channel beats) |
| `--pcie-gbps G` | host-link bandwidth for offloaded experts |
| `--fast-gb G` | memory a slice holds and addresses (4: a 32-bit address) |
| `--moe-rows all\|gather` | a MoE layer run computes its union of experts on all R rows, or each row only its own |
| `--wformat fp4\|int8\|auto` | the dense models' weights (fp4 with an int8 head; `auto` = the recommended mix) |
| `--ctx N` | the decode context (default 512) |

The API returns plain dicts, so other tools and the what-if branches can call it directly:

```python
import sys; sys.path.insert(0, "tools")
import scoreboard as SB
hw = SB.board("ypcb", mcols=8)                 # a preset with overrides (an HW dataclass)
SB.decode("qwen3", hw)                         # {"tok_s", "mcycles", "mb_token"}
SB.prefill("phi4-mini", hw, P=1024)            # {"ttft_s", "tok_s", "rows", "runs", ...}
SB.batch("qwen3", hw, users=8)                 # {"total_tok_s", "user_tok_s", ...}
SB.moe_decode("qwen35-35b", hw); SB.moe_prefill("gemma4-26b", hw, P=256)
SB.compare(SB.board("ypcb"), hw)               # {"decode", "prefill", "batch", "moe", "score"}
```

The co-sim driver needs a model's `config.json` only, because it runs zero DRAM images:

```sh
PYTHONPATH=. python3 tools/scoreboard_cosim.py run models/Qwen3-0.6B --layers 1:2 --rows 0,4 \
    --mcols 8 --lanes 16 --tmem-words 262144 --act-rows 16 --mem ldc --out pts.jsonl
PYTHONPATH=. python3 tools/scoreboard_cosim.py probe models/Qwen3-0.6B --tmem 65536,262144 \
    --act-rows 16 --buckets 1,4                # compile only: the rows a prefill run fits
```

Run the co-sim on omarchy through `~/otpu-build/slot.sh`. A two-layer Qwen3 run needs about
4 GB and 10-40 minutes.

## 1. The workloads

All weights are fp4 with an int8 LM head (`--wformat`); the MoE models use their production
formats.

- **Decode**: one user's tokens a second at context 512. This is device time; the card's
  decode loop picks each token itself.
- **Prefill**: a 256- and a 1024-token prompt from position 0, in prompt runs of R rows.
  - A run stays inside its 256-token bucket.
  - R is what the TMEM and IMEM fit allow under the host's rule (section 2).
  - TTFT is the runs' device time plus 0.6 ms of host time a run.
- **Batched decode**: 1, 4, 8 and 16 users at context 512.
  - A step runs the users in blocks of one weight pass (MCOLS rows, or `--batch-rows`), one
    run a block, and picks every row's logits on the card.
  - The table gives the total and the per-user tokens a second.
  - Batched kernels exist today for Qwen3 and the Llama-likes (SmolLM3, Phi-4-mini); the
    table shows those.
- **MoE**: Gemma 4 26B-A4B and Qwen3.5-35B-A3B.
  - Expert slots fill the memory the slices address beside the rest of the image. The other
    experts come over PCIe from the host, or from a second card tier (F2's DDR4).
  - Decode is the card's time with the experts present, plus each layer's link time that the
    card does not hide.
  - Prefill is layer-major, R = 2, with pooled slots (docs/offload.md 13).

The summary line and `compare` reduce each workload to one number, a geometric mean over the
models:

- decode tok/s;
- prefill tok/s at 1024 tokens;
- batched total tok/s at 16 users;
- for MoE, sqrt(decode x prefill 1024) tok/s.

`score` is the geometric mean of the four ratios, so the four workloads weigh the same.

**What binds.** Every result carries each unit's share of its time:

- the weight stream, named `port` or `memory` after whichever is slower;
- `MXU`;
- `attention`, the rows' KV streams;
- `DSTEP`;
- `overhead`, meaning vector unit, control and runs;
- `host link` and `host`.

The unit with the largest share binds. `report` shows it for each model and workload.
`whatifs` and `compare` show, for each option, what binds each workload after the change.

## 2. The model

A run of R rows costs, in each layer and on each slice:

    cycles = max(memory / S, MXU / S) + attention + recurrence + overhead

- **memory** is the layer's DRAM bytes over the slice's stream rate. The rate is the lower of
  the port (`port_bytes x 0.97` a cycle) and the memory (`GB/s x mem_eff / MHz`). The bytes
  are:
  - **Weights**: `ceil(R / rows a pass)` times their bytes. A weight takes 8.25 bits (int8)
    or 4.25 bits (fp4), block scales included.
  - **DeltaNet**: the state, read and written once a run.
  - **LFM2**: the conv window.
  - **MoE**: the routed experts, the union of the run's rows.
- **attention** is a serial phase between the layer's matmuls: q, k and v before it, o after.
  - Each row streams its KV cache: whole 256-token blocks, a sliding layer at most its window
    plus a block, at `2d + 4d/128 + 4` bytes a token and KV head.
  - It streams them once for each group of MCOLS query heads of a KV head, as
    `gemma4._attention` and `qwen3._attend_heads` part them.
  - The card's prompt runs from 1 to 2 blocks (A0, B0, B1) and its batched steps at contexts
    64 and 1024 fit this stream alone, with no latency term (`att_blk` 0). The tiling agent's
    co-sim breakdown agrees: 0.46K cycles a row, KV head and block.
- **MXU**:
  - int8: `params / D x ceil(R / MCOLS)` cycles.
  - 4-bit with PAIR: `params / 2D x ceil(2R / MCOLS)` cycles.
  - Both are divided by `--mxu-pump`.

  The MXU takes D weights a cycle (2D at 4 bits), which is the port's rate. A wider port
  therefore needs a wider MXU (section 6).
- **recurrence**: a DeltaNet layer's first row of DSTEPs hides behind the weight stream.
  Every further row adds `v_heads / S x (dk dv / (8 x vpu) + 200)` cycles.
- **overhead**:

      (kappa_pass x passes + kappa_row x min(R, rows a pass)) x 7000 x hidden / 1024 x (0.5 / vpu + 0.5)

  - The two kappas are fitted for each model on its card points (`scoreboard.py fit`).
  - The rows of a second pass add nothing. The card's two-pass runs of 8 rows show that
    their vector work hides behind the next pass's matmul.
  - Half of the overhead scales with the vector unit. This share is provisional until the
    LANES = 16 co-sim lands (section 8).
  - S > 1 adds 1500 cycles a layer and row for the collectives. This is an estimate.

A run adds 5000 cycles. Logits add the LM head, which streams once a pass of logits rows, plus
2000 cycles a row for argmax or sampling. Gemma 4 adds its PLE projection.

**The tile kernel** (`--kernel tiled`) is the tiling agent's restructured prompt run
(branch tiled-prefill):

- MCOLS / G rows share each KV head's entry (packed attention), with its K and V products on
  the MXU.
- The next head's projections stream meanwhile, and gate/up pieces pipeline against SiLU.
- An attention layer costs `max(memory, linear MXU + attention MXU)`, plus
  `4.9K x R x hidden / 1024` cycles outside the MXU.
  - This is fitted on 14 of its co-sim points (TMEM 64K; MCOLS 4 and 8; T 4-16; int8 and
    fp4; positions 256 and 1024).
  - Mean error is 2.9%. The worst is MCOLS 8 at 1024 (-12.6%), where the model's attention
    is too cheap.
  - On the card it costs x 1.06 over that co-sim: the card's prompt runs cost about 6% more
    than compile_rows, from their RunWords, mask loads and per-row V^T appends. That is
    tiling's slopes against the card's B0, and the scoreboard's today's kernel is fitted on
    the card.
- It applies to prompt runs only. Batched users do not share KV, and MoE layer runs are
  their own kernel. The host keeps today's kernel wherever that is faster.

**Prefill rows.** Prefill takes the rows a run fits from the compiler itself:
`tools/scoreboard_cosim.py probe` compiles each model at each TMEM size and ACT RAM depth
until TMEM, the ACT RAM or IMEM overflows, giving `tools/scoreboard_rmax.json`. Of the rows
that fit, the host's rule takes whole passes above one pass, and an odd R's R - 1 at 4 bits
(PAIR). Of those and of fewer whole passes, the model takes the R it finds fastest a row.

The probe agrees with tmem-pump's own bisection of the TMEM a run needs (Qwen3.5-4B: 54K,
70K and 140K words at R = 3, 4 and 8). The probe gives 3 rows at 64K and 7 at 128K, with IMEM
over 4096 instructions from R = 8.

**MoE:**

- **Misses a decode token**: the per-layer LRU over four 2048-token router traces
  (docs/offload.md 3) at the slots' share of the pool, x 0.88 (decayed use), x 1.2 (the
  card's chats miss more than the traces).
- **Exposure**: a layer's link time is exposed past what the card computes meanwhile, namely
  the dense MLP or shared expert and the hit experts. Each layer with misses adds 0.15 ms of
  host lead.
- **Prompt misses**: each layer's distinct experts over the prompt (`k R^0.6`, at most a
  skewed 75 of 128 (26B) or 145 of 256 (35B)) minus the slots' share, of which 85% is
  exposed.
- **Host time**: 0.2 ms a layer run. A DeltaNet model runs its first `conv_k - 1` rows as
  decode steps.

## 3. Validation

Against the 51 dense card points (decode, prefill, batched), the mean |error| is 0.6% and the largest 3.1%. `scoreboard.py validate` lists every point:

| points | mean \|error\| | max | leave-one-out mean | leave-one-out max |
|---|---:|---:|---:|---:|
| decode (21) | 0.5% | 3.1% | 0.6% | 3.1% |
| prefill (24) | 0.7% | 2.4% | 1.0% | 4.2% |
| batched (6) | 0.2% | 0.4% | 0.3% | 0.4% |
| co-sim (8) | 1.3% | 4.2% | 1.4% | 4.2% |
| co-sim 2-layer (1) | 10.9% | 10.9% | 10.9% | 10.9% |
| co-sim tile kernel (14) | 2.9% | 12.6% | 2.9% | 12.6% |
| co-sim layer (10) | 1.0% | 2.6% | 1.2% | 2.7% |
| co-sim layer (compile_rows) (4) | 7.3% | 9.4% | 7.3% | 9.4% |
| MoE layer run ms (6) | 2.4% | 4.1% | - | - |
| MoE (4) | 2.3% | 3.2% | - | - |

- **Card points**: production builds at 133.33 MHz.
  - Decode: the decode counters (`tools/qual/perf.py`).
  - Prompt runs (docs/prefill.md 7): a first prompt (A0), a 224-291-token prompt (B0) and a
    next turn of 13 tokens at 224-322 (B1), for six models.
  - Batched: otpu-serve's steps.
  - MoE: the streamed MoEs' sessions (docs/offload.md 13.7, 14.4).
- **Co-sim points**: the RTL with LiteDRAM's DDR3-1066 controller.
  - `co-sim`: whole-model decode and prefill runs. It includes a Qwen3.5 prefill run at
    MCOLS 8 that the fit never saw.
  - `co-sim layer`: the per-layer slopes of docs/prefill.md 8. Phi-4-mini's at position 128
    (1 block), SmolLM3's at 3968 (16 blocks), Qwen3.5-4B's at 544.
  - `co-sim 2-layer` and `co-sim tile kernel`: the tiling agent's runs, today's kernel on the
    card's configuration and the tile kernel. They are not in the kappa fit. The tile
    kernel's row constant is fitted on its own points.
- **Leave-one-out**: refits the point's model without that point. The largest is Qwen3.5-2B's
  prefill at 4.2%; it has one prefill point, predicted from its decode points.
- **MoE**: its constants (host lead, exposure, run time) are fitted on the same four MoE
  points, so its 2-3% is in-sample. The zero-miss layer runs (2.7%) check its compute part
  independently.
- **Decode bytes a token**: within 2% of the card's counters, except Gemma 4 E2B. The model
  gives ~5-7% fewer bytes than its counters read (~100 MB a token), and the cause is
  unexplained.

The co-sim campaign's what-if points (section 8) were still queued on omarchy when this was written; `validate` will list them once they are added to `COSIM`.

## 4. Baselines

| configuration | stream GB/s | decode tok/s | prefill 1024 tok/s | batched 16 users tok/s | MoE | score vs YPCB |
|---|---:|---:|---:|---:|---:|---:|
| YPCB (production) | 16.6 | 15.4 | 42.1 | 29.0 | 6.90 | 1.00x |
| BCU1525 DDR4-1600, 100 MHz | 24.8 | 22.0 | 57.3 | 39.8 | 11.25 | 1.44x |
| BCU1525 DDR4-1600, 200 MHz | 48.6 | 43.3 | 112.4 | 78.7 | 21.19 | 2.81x |
| BCU1525 DDR4-2400, 100 MHz | 24.8 | 22.0 | 57.3 | 39.8 | 11.25 | 1.44x |
| BCU1525 DDR4-2400, 200 MHz | 49.7 | 44.1 | 113.2 | 79.0 | 21.49 | 2.84x |
| BCU1525 DDR4-2400, 200 MHz, port 256 + MXU x2 | 73.0 | 62.3 | 170.4 | 128.8 | 27.51 | 4.13x |
| F2 (HBM, S 4, 250 MHz) | 124.2 | 101.9 | 241.2 | 170.4 | 44.25 | 6.14x |
| F2, port 256 + MXU x2 | 248.3 | 177.9 | 366.5 | 269.9 | 64.51 | 9.67x |
| F2, port 512 + MXU x4 | 414.0 | 255.5 | 484.0 | 376.6 | 79.52 | 12.99x |

The presets (`scoreboard.py boards`):

- **ypcb**: the production card. xc7k480t, 2 x DDR3-1066 through LiteDRAM, 133.33 MHz,
  MCOLS 4, TMEM 64K words, PCIe Gen2 x8 (2.2 GB/s for an expert miss).
- **bcu1525**: VU9P with S = 2 slices, each on 2 DDR4 channels. Each slice addresses 4 GiB of
  its 2 x 16 GB.
  - Bring-up is DDR4-1600 at 100 MHz, with mem_eff 0.95 assumed.
  - PCIe Gen3 x16 at 9 GB/s is assumed, not measured.
- **f2**: AWS F2, VU47P. 16 GB of HBM (~460 GB/s, taken as 115 GB/s a slice at 0.90), S = 4
  at 250 MHz.
  - Its 64 GB of DDR4 is a second tier for experts, assumed to be one 19.2 GB/s channel.
  - Its PCIe is taken as 20 GB/s.

Every board here is **port-bound**: a slice streams at most 128 B a core cycle. On the BCU1525
at 100 MHz that is half its DDR4, and on F2 at 250 MHz it is 128 of 460 GB/s. The BCU1525's
DDR4-2400 buys nothing until the port widens.

### YPCB (production)

**ypcb: S 1, MCOLS 4 (rows a pass 4, MXU x1), TMEM 64K words, VPU x1, 133.33 MHz, DDR3-1066.67 x 2 a slice: 16.6 GB/s, 124.2 B a cycle (port-bound), 16.6 GB/s in all; host link 2.2 GB/s**

summary (geometric means, fp4 weights, context 512): decode 15.4 tok/s, prefill 1024 42.1 tok/s, batched 16 users 29.0 tok/s in all, MoE 6.90 (sqrt of decode x prefill 1024 tok/s)

bound (the unit taking the largest share of the time, averaged over the models): decode port 93%, prefill MXU 53%, batch MXU 69%, moe port 69%

| model | decode tok/s | MB/token | prefill 256 tok/s | TTFT 256 s | prefill 1024 tok/s | TTFT 1024 s | rows a run | decode bound | prefill 1024 bound |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | 35.8 | 440 | 122.8 | 2.08 | 105.4 | 9.72 | 8 | port 85% | MXU 68% |
| lfm2 | 97.3 | 165 | 358.9 | 0.71 | 311.3 | 3.29 | 8..4 | port 92% | MXU 74% |
| qwen35 | 27.6 | 571 | 77.3 | 3.31 | 76.8 | 13.34 | 4 | port 95% | MXU 56% |
| qwen35-2b | 12.6 | 1,301 | 40.9 | 6.25 | 40.8 | 25.07 | 4 | port 99% | MXU 83% |
| qwen35-4b | 6.2 | 2,671 | 14.8 | 17.33 | 14.7 | 69.61 | 2 | port 99% | port 89% |
| smollm3 | 8.9 | 1,794 | 22.3 | 11.49 | 21.9 | 46.80 | 4 | port 95% | MXU 90% |
| phi4-mini | 6.7 | 2,397 | 17.8 | 14.37 | 17.4 | 58.97 | 2 | port 95% | port 90% |
| lfm2-2.6b | 11.1 | 1,446 | 26.7 | 9.58 | 24.5 | 41.73 | 4..2 | port 97% | port 72% |
| gemma4 | 10.0 | 1,446 | 29.9 | 8.57 | 29.2 | 35.12 | 4 | port 85% | MXU 80% |

| batched decode, total / each tok/s | 1 users | 4 users | 8 users | 16 users | 16 users bound |
|---|---:|---:|---:|---:|---:|
| qwen3 | 35.8 / 35.8 | 76.3 / 19.1 | 76.3 / 9.5 | 76.3 / 4.8 | MXU 49% |
| smollm3 | 8.9 / 8.9 | 20.0 / 5.0 | 20.0 / 2.5 | 20.0 / 1.2 | MXU 82% |
| phi4-mini | 6.7 / 6.7 | 16.0 / 4.0 | 16.0 / 2.0 | 16.0 / 1.0 | MXU 76% |

| MoE | experts on the card | decode tok/s | misses a token | prefill 256 tok/s | TTFT 256 s | prefill 1024 tok/s | TTFT 1024 s | decode bound | prefill 1024 bound |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gemma4-26b | 14% (535 slots) | 3.56 | 87 | 8.3 | 30.9 | 8.6 | 118.8 | port 64% | port 78% |
| qwen35-35b | 16% (1615 slots) | 5.45 | 104 | 12.0 | 21.4 | 13.6 | 75.4 | port 64% | port 70% |

### BCU1525 DDR4-1600, 100 MHz

**bcu1525: S 2, MCOLS 4 (rows a pass 4, MXU x1), TMEM 64K words, VPU x1, 100 MHz, DDR4-1600 x 2 a slice: 24.3 GB/s, 124.2 B a cycle (port-bound), 24.8 GB/s in all; host link 9 GB/s**

summary (geometric means, fp4 weights, context 512): decode 22.0 tok/s, prefill 1024 57.3 tok/s, batched 16 users 39.8 tok/s in all, MoE 11.25 (sqrt of decode x prefill 1024 tok/s)

bound (the unit taking the largest share of the time, averaged over the models): decode port 89%, prefill MXU 47%, batch MXU 64%, moe port 76%

| model | decode tok/s | MB/token | prefill 256 tok/s | TTFT 256 s | prefill 1024 tok/s | TTFT 1024 s | rows a run | decode bound | prefill 1024 bound |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | 50.1 | 440 | 158.3 | 1.62 | 138.6 | 7.39 | 8 | port 80% | MXU 60% |
| lfm2 | 138.0 | 165 | 460.7 | 0.56 | 394.9 | 2.59 | 8..4 | port 87% | MXU 63% |
| qwen35 | 39.1 | 571 | 98.9 | 2.59 | 98.2 | 10.43 | 4 | port 89% | MXU 48% |
| qwen35-2b | 18.6 | 1,301 | 59.5 | 4.31 | 59.3 | 17.26 | 4 | port 97% | MXU 80% |
| qwen35-4b | 9.2 | 2,671 | 21.9 | 11.71 | 21.8 | 47.01 | 2 | port 99% | port 88% |
| smollm3 | 12.7 | 1,794 | 30.8 | 8.30 | 30.3 | 33.77 | 4 | port 91% | MXU 83% |
| phi4-mini | 9.7 | 2,397 | 25.0 | 10.23 | 24.4 | 41.92 | 2 | port 92% | port 84% |
| lfm2-2.6b | 16.2 | 1,446 | 38.0 | 6.74 | 34.8 | 29.43 | 4..2 | port 93% | port 68% |
| gemma4 | 13.2 | 1,446 | 38.1 | 6.72 | 37.3 | 27.44 | 4 | port 75% | MXU 68% |

| batched decode, total / each tok/s | 1 users | 4 users | 8 users | 16 users | 16 users bound |
|---|---:|---:|---:|---:|---:|
| qwen3 | 50.1 / 50.1 | 98.9 / 24.7 | 98.9 / 12.4 | 98.9 / 6.2 | MXU 43% |
| smollm3 | 12.7 / 12.7 | 27.8 / 7.0 | 27.8 / 3.5 | 27.8 / 1.7 | MXU 76% |
| phi4-mini | 9.7 / 9.7 | 22.9 / 5.7 | 22.9 / 2.9 | 22.9 / 1.4 | MXU 72% |

| MoE | experts on the card | decode tok/s | misses a token | prefill 256 tok/s | TTFT 256 s | prefill 1024 tok/s | TTFT 1024 s | decode bound | prefill 1024 bound |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gemma4-26b | 44% (1694 slots) | 6.49 | 16 | 11.5 | 22.2 | 11.5 | 89.4 | port 78% | port 69% |
| qwen35-35b | 39% (4005 slots) | 11.45 | 41 | 18.2 | 14.0 | 18.8 | 54.4 | port 90% | port 65% |

### BCU1525 DDR4-2400, 200 MHz

**bcu1525: S 2, MCOLS 4 (rows a pass 4, MXU x1), TMEM 64K words, VPU x1, 200 MHz, DDR4-2400 x 2 a slice: 36.5 GB/s, 124.2 B a cycle (port-bound), 49.7 GB/s in all; host link 9 GB/s**

summary (geometric means, fp4 weights, context 512): decode 44.1 tok/s, prefill 1024 113.2 tok/s, batched 16 users 79.0 tok/s in all, MoE 21.49 (sqrt of decode x prefill 1024 tok/s)

bound (the unit taking the largest share of the time, averaged over the models): decode port 89%, prefill MXU 46%, batch MXU 63%, moe port 72%

| model | decode tok/s | MB/token | prefill 256 tok/s | TTFT 256 s | prefill 1024 tok/s | TTFT 1024 s | rows a run | decode bound | prefill 1024 bound |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | 100.2 | 440 | 312.8 | 0.82 | 274.4 | 3.73 | 8 | port 80% | MXU 59% |
| lfm2 | 276.0 | 165 | 890.5 | 0.29 | 750.8 | 1.36 | 8..4 | port 87% | MXU 60% |
| qwen35 | 78.1 | 571 | 194.8 | 1.31 | 193.6 | 5.29 | 4 | port 89% | MXU 47% |
| qwen35-2b | 37.2 | 1,301 | 117.9 | 2.17 | 117.6 | 8.71 | 4 | port 97% | MXU 79% |
| qwen35-4b | 18.5 | 2,671 | 43.5 | 5.89 | 43.3 | 23.66 | 2 | port 99% | port 87% |
| smollm3 | 25.5 | 1,794 | 61.4 | 4.17 | 60.4 | 16.96 | 4 | port 91% | MXU 83% |
| phi4-mini | 19.5 | 2,397 | 49.7 | 5.15 | 48.5 | 21.11 | 2 | port 92% | port 84% |
| lfm2-2.6b | 32.4 | 1,446 | 75.6 | 3.39 | 68.9 | 14.85 | 4..2 | port 93% | port 67% |
| gemma4 | 26.4 | 1,446 | 75.8 | 3.38 | 74.2 | 13.80 | 4 | port 75% | MXU 68% |

| batched decode, total / each tok/s | 1 users | 4 users | 8 users | 16 users | 16 users bound |
|---|---:|---:|---:|---:|---:|
| qwen3 | 100.2 / 100.2 | 194.9 / 48.7 | 194.9 / 24.4 | 194.9 / 12.2 | MXU 42% |
| smollm3 | 25.5 / 25.5 | 55.5 / 13.9 | 55.5 / 6.9 | 55.5 / 3.5 | MXU 76% |
| phi4-mini | 19.5 / 19.5 | 45.7 / 11.4 | 45.7 / 5.7 | 45.7 / 2.9 | MXU 72% |

| MoE | experts on the card | decode tok/s | misses a token | prefill 256 tok/s | TTFT 256 s | prefill 1024 tok/s | TTFT 1024 s | decode bound | prefill 1024 bound |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gemma4-26b | 44% (1694 slots) | 12.97 | 16 | 21.9 | 11.7 | 22.1 | 46.4 | port 78% | port 67% |
| qwen35-35b | 39% (4005 slots) | 21.51 | 41 | 32.7 | 7.8 | 34.7 | 29.5 | port 85% | port 60% |

### F2 (HBM, S 4, 250 MHz)

**f2: S 4, MCOLS 4 (rows a pass 4, MXU x1), TMEM 64K words, VPU x1, 250 MHz, HBM 115 GB/s a slice: 103.5 GB/s, 124.2 B a cycle (port-bound), 124.2 GB/s in all; host link 20 GB/s**

summary (geometric means, fp4 weights, context 512): decode 101.9 tok/s, prefill 1024 241.2 tok/s, batched 16 users 170.4 tok/s in all, MoE 44.25 (sqrt of decode x prefill 1024 tok/s)

bound (the unit taking the largest share of the time, averaged over the models): decode port 83%, prefill MXU 39%, batch MXU 55%, moe port 61%

| model | decode tok/s | MB/token | prefill 256 tok/s | TTFT 256 s | prefill 1024 tok/s | TTFT 1024 s | rows a run | decode bound | prefill 1024 bound |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | 225.0 | 440 | 630.9 | 0.41 | 566.9 | 1.81 | 8 | port 71% | MXU 49% |
| lfm2 | 638.8 | 165 | 1,761.8 | 0.15 | 1,411.5 | 0.73 | 8..4 | port 80% | MXU 45% |
| qwen35 | 177.4 | 571 | 377.5 | 0.68 | 375.6 | 2.73 | 4 | port 81% | overhead 38% |
| qwen35-2b | 90.7 | 1,301 | 277.3 | 0.92 | 276.8 | 3.70 | 4 | port 95% | MXU 75% |
| qwen35-4b | 46.0 | 2,671 | 105.4 | 2.43 | 105.0 | 9.75 | 2 | port 99% | port 85% |
| smollm3 | 58.9 | 1,794 | 133.7 | 1.91 | 131.8 | 7.77 | 4 | port 84% | MXU 72% |
| phi4-mini | 45.9 | 2,397 | 109.7 | 2.33 | 107.4 | 9.53 | 2 | port 87% | port 74% |
| lfm2-2.6b | 76.5 | 1,446 | 171.8 | 1.49 | 154.7 | 6.62 | 4..2 | port 88% | port 61% |
| gemma4 | 53.6 | 1,446 | 146.7 | 1.75 | 144.3 | 7.09 | 4 | port 61% | MXU 53% |

| batched decode, total / each tok/s | 1 users | 4 users | 8 users | 16 users | 16 users bound |
|---|---:|---:|---:|---:|---:|
| qwen3 | 225.0 / 225.0 | 386.9 / 96.7 | 386.9 / 48.4 | 386.9 / 24.2 | overhead 34% |
| smollm3 | 58.9 / 58.9 | 122.3 / 30.6 | 122.3 / 15.3 | 122.3 / 7.6 | MXU 67% |
| phi4-mini | 45.9 / 45.9 | 104.6 / 26.1 | 104.6 / 13.1 | 104.6 / 6.5 | MXU 66% |

| MoE | experts on the card | decode tok/s | misses a token | prefill 256 tok/s | TTFT 256 s | prefill 1024 tok/s | TTFT 1024 s | decode bound | prefill 1024 bound |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gemma4-26b | all | 27.10 | 0 | 42.7 | 6.0 | 42.1 | 24.3 | port 65% | port 51% |
| qwen35-35b | 86% (8786 slots) + 1453 far | 51.78 | 4 | 64.2 | 4.0 | 65.0 | 15.8 | port 82% | port 45% |

### MoE on one slice (today's code) against S slices

| board | model | slices | experts on the card | decode tok/s | prefill 1024 tok/s |
|---|---|---:|---:|---:|---:|
| BCU1525 DDR4-2400, 200 MHz | gemma4-26b | 1 | 14% | 7.21 | 12.9 |
| BCU1525 DDR4-2400, 200 MHz | gemma4-26b | 2 | 44% | 12.97 | 22.1 |
| BCU1525 DDR4-2400, 200 MHz | qwen35-35b | 1 | 16% | 10.99 | 20.4 |
| BCU1525 DDR4-2400, 200 MHz | qwen35-35b | 2 | 39% | 21.51 | 34.7 |
| F2 | gemma4-26b | 1 | 14% + 3304 far | 8.55 | 16.1 |
| F2 | gemma4-26b | 4 | all | 27.10 | 42.1 |
| F2 | qwen35-35b | 1 | 16% + 8624 far | 14.30 | 25.2 |
| F2 | qwen35-35b | 4 | 86% + 1453 far | 51.78 | 65.0 |


## 5. What-ifs

`scoreboard.py whatifs [--board B]` gives each what-if's ratio to the board.

**Against ypcb** (S 1, MCOLS 4 (rows a pass 4, MXU x1), TMEM 64K words, VPU x1, 133.33 MHz, DDR3-1066.67 x 2 a slice: 16.6 GB/s, 124.2 B a cycle (port-bound), 16.6 GB/s in all; host link 2.2 GB/s):

| what-if (against ypcb) | decode | prefill | batched | MoE | score | bound after: decode / prefill / batched / MoE |
|---|---:|---:|---:|---:|---:|---|
| (ypcb itself) | 1.000x | 1.000x | 1.000x | 1.000x | 1.000x | port 93% / MXU 53% / MXU 69% / port 69% |
| MCOLS 8 | 1.001x | 1.377x | 1.736x | 1.003x | 1.245x | port 94% / port 49% / MXU 61% / port 70% |
| TMEM x2 (8 ACT RAM rows) | 1.000x | 1.023x | 1.015x | 1.000x | 1.009x | port 93% / MXU 75% / MXU 83% / port 69% |
| TMEM x4 (16 ACT RAM rows) | 1.000x | 1.023x | 1.018x | 1.000x | 1.010x | port 93% / MXU 75% / MXU 84% / port 69% |
| 64 rows a run, today's kernel | 1.000x | 1.021x | 1.018x | 1.000x | 1.010x | port 93% / MXU 80% / MXU 84% / port 69% |
| tile kernel, 16 rows a run (64K TMEM) | 1.000x | 1.031x | 1.000x | 1.000x | 1.008x | port 93% / MXU 86% / MXU 69% / port 69% |
| tile kernel, 16 rows, MCOLS 8 | 1.001x | 1.788x | 1.736x | 1.003x | 1.329x | port 94% / MXU 77% / MXU 61% / port 70% |
| 8 users in one pass (replay) | 1.000x | 0.993x | 1.015x | 1.000x | 1.002x | port 93% / MXU 52% / MXU 83% / port 69% |
| VPU x2 | 1.010x | 1.045x | 1.017x | 1.027x | 1.024x | port 94% / MXU 55% / MXU 70% / port 71% |
| MXU pumped x2 | 1.000x | 1.323x | 1.460x | 1.000x | 1.179x | port 93% / port 73% / port 77% / port 69% |
| MCOLS 8 + TMEM x4 + VPU x2 | 1.011x | 1.730x | 1.804x | 1.030x | 1.343x | port 94% / MXU 64% / MXU 75% / port 71% |
| MXU x2 + TMEM x4 + VPU x2 (pumped) | 1.010x | 1.754x | 1.804x | 1.027x | 1.346x | port 94% / MXU 62% / MXU 75% / port 71% |
| port 256 B a cycle (memory permitting) | 1.000x | 1.000x | 1.000x | 1.000x | 1.000x | memory 93% / MXU 53% / MXU 69% / memory 69% |
| port 256 B + MXU x2 | 1.000x | 1.323x | 1.460x | 1.000x | 1.179x | memory 93% / memory 73% / memory 77% / memory 69% |
| MoE runs of 8 rows (64-id requests) | 1.000x | 1.000x | 1.000x | 0.928x | 0.981x | port 93% / MXU 53% / MXU 69% / MXU 41% |
| MoE runs of 8 rows, each row its experts | 1.000x | 1.000x | 1.000x | 1.234x | 1.054x | port 93% / MXU 53% / MXU 69% / port 51% |

**Against bcu1525** (S 2, MCOLS 4 (rows a pass 4, MXU x1), TMEM 64K words, VPU x1, 100 MHz, DDR4-1600 x 2 a slice: 24.3 GB/s, 124.2 B a cycle (port-bound), 24.8 GB/s in all; host link 9 GB/s):

| what-if (against bcu1525) | decode | prefill | batched | MoE | score | bound after: decode / prefill / batched / MoE |
|---|---:|---:|---:|---:|---:|---|
| (bcu1525 itself) | 1.000x | 1.000x | 1.000x | 1.000x | 1.000x | port 89% / MXU 47% / MXU 64% / port 76% |
| MCOLS 8 | 1.001x | 1.322x | 1.655x | 1.004x | 1.218x | port 89% / port 45% / MXU 54% / port 76% |
| TMEM x2 (8 ACT RAM rows) | 1.000x | 1.024x | 1.018x | 1.000x | 1.010x | port 89% / MXU 68% / MXU 77% / port 76% |
| TMEM x4 (16 ACT RAM rows) | 1.000x | 1.023x | 1.023x | 1.000x | 1.011x | port 89% / MXU 68% / MXU 77% / port 76% |
| 64 rows a run, today's kernel | 1.000x | 1.030x | 1.023x | 1.000x | 1.013x | port 89% / MXU 74% / MXU 77% / port 76% |
| tile kernel, 16 rows a run (64K TMEM) | 1.000x | 1.041x | 1.000x | 1.000x | 1.010x | port 89% / MXU 77% / MXU 64% / port 76% |
| tile kernel, 16 rows, MCOLS 8 | 1.001x | 1.688x | 1.655x | 1.004x | 1.294x | port 89% / MXU 66% / MXU 54% / port 76% |
| 8 users in one pass (replay) | 1.000x | 0.988x | 1.018x | 1.000x | 1.002x | port 89% / MXU 46% / MXU 77% / port 76% |
| VPU x2 | 1.018x | 1.055x | 1.030x | 1.041x | 1.036x | port 91% / MXU 50% / MXU 65% / port 79% |
| MXU pumped x2 | 1.000x | 1.276x | 1.411x | 1.000x | 1.158x | port 89% / port 65% / port 68% / port 76% |
| MCOLS 8 + TMEM x4 + VPU x2 | 1.019x | 1.672x | 1.756x | 1.044x | 1.330x | port 91% / MXU 56% / MXU 68% / port 79% |
| MXU x2 + TMEM x4 + VPU x2 (pumped) | 1.018x | 1.694x | 1.756x | 1.041x | 1.332x | port 91% / MXU 55% / MXU 68% / port 79% |
| port 256 B a cycle (memory permitting) | 1.105x | 1.059x | 1.054x | 1.093x | 1.078x | MXU 89% / MXU 75% / MXU 79% / MXU 74% |
| port 256 B + MXU x2 | 1.816x | 1.655x | 1.737x | 1.596x | 1.699x | memory 83% / MXU 38% / MXU 56% / memory 62% |
| MoE runs of 8 rows (64-id requests) | 1.000x | 1.000x | 1.000x | 0.956x | 0.989x | port 89% / MXU 47% / MXU 64% / port 42% |
| MoE runs of 8 rows, each row its experts | 1.000x | 1.000x | 1.000x | 1.250x | 1.057x | port 89% / MXU 47% / MXU 64% / port 59% |

**Against f2** (S 4, MCOLS 4 (rows a pass 4, MXU x1), TMEM 64K words, VPU x1, 250 MHz, HBM 115 GB/s a slice: 103.5 GB/s, 124.2 B a cycle (port-bound), 124.2 GB/s in all; host link 20 GB/s):

| what-if (against f2) | decode | prefill | batched | MoE | score | bound after: decode / prefill / batched / MoE |
|---|---:|---:|---:|---:|---:|---|
| (f2 itself) | 1.000x | 1.000x | 1.000x | 1.000x | 1.000x | port 83% / MXU 39% / MXU 55% / port 61% |
| MCOLS 8 | 1.001x | 1.259x | 1.567x | 1.003x | 1.186x | port 83% / port 39% / MXU 45% / port 61% |
| TMEM x2 (8 ACT RAM rows) | 1.000x | 1.036x | 1.034x | 1.000x | 1.018x | port 83% / MXU 59% / MXU 68% / port 61% |
| TMEM x4 (16 ACT RAM rows) | 1.000x | 1.038x | 1.049x | 1.000x | 1.022x | port 83% / MXU 59% / MXU 69% / port 61% |
| 64 rows a run, today's kernel | 1.000x | 1.083x | 1.049x | 1.000x | 1.032x | port 83% / MXU 66% / MXU 69% / port 61% |
| tile kernel, 16 rows a run (64K TMEM) | 1.000x | 1.086x | 1.000x | 1.000x | 1.021x | port 83% / MXU 68% / MXU 55% / port 61% |
| tile kernel, 16 rows, MCOLS 8 | 1.001x | 1.620x | 1.567x | 1.003x | 1.263x | port 83% / MXU 54% / MXU 45% / port 61% |
| 8 users in one pass (replay) | 1.000x | 0.986x | 1.034x | 1.000x | 1.005x | port 83% / MXU 38% / MXU 68% / port 61% |
| VPU x2 | 1.032x | 1.070x | 1.052x | 1.054x | 1.052x | port 85% / MXU 42% / MXU 58% / port 64% |
| MXU pumped x2 | 1.000x | 1.216x | 1.340x | 1.000x | 1.130x | port 83% / port 54% / port 56% / port 61% |
| MCOLS 8 + TMEM x4 + VPU x2 | 1.033x | 1.630x | 1.729x | 1.057x | 1.325x | port 85% / MXU 47% / MXU 58% / port 64% |
| MXU x2 + TMEM x4 + VPU x2 (pumped) | 1.032x | 1.653x | 1.729x | 1.054x | 1.328x | port 85% / MXU 46% / MXU 58% / port 64% |
| port 256 B a cycle (memory permitting) | 1.097x | 1.051x | 1.044x | 1.080x | 1.068x | MXU 83% / MXU 64% / MXU 68% / MXU 59% |
| port 256 B + MXU x2 | 1.747x | 1.520x | 1.584x | 1.458x | 1.574x | port 73% / overhead 33% / MXU 45% / port 45% |
| MoE runs of 8 rows (64-id requests) | 1.000x | 1.000x | 1.000x | 1.056x | 1.014x | port 83% / MXU 39% / MXU 55% / port 37% |
| MoE runs of 8 rows, each row its experts | 1.000x | 1.000x | 1.000x | 1.321x | 1.072x | port 83% / MXU 39% / MXU 55% / port 50% |


What each what-if sets:

- **MCOLS 8**: `--mcols 8`, which also means rows a pass 8.
- **TMEM x2 / x4**: `--tmem-words 131072 / 262144` with ACT RAM rows 8 / 16, giving the
  probe's rows a run.
- **8 users in one pass**: `--batch-rows 8 --act-rows 8`, with the MXU still at MCOLS 4.
- **VPU x2**: `--vpu 2`.
- **MXU pumped x2**: `--mxu-pump 2`.
- **Port 256 B**: `--port-bytes 256`, alone or with an MXU x2.
- **64 rows a run, today's kernel**: `--rows 64 --act-rows 64 --imem 0`.
- **Tile kernel**: `--kernel tiled --rows 16`, which fits 64K TMEM for Qwen3; also with
  MCOLS 8.
- **MoE runs of 8 rows**: `--moe-ids 64`, a request line of 64 expert ids, so R x 8 <= 64.
  It runs either with the run's union of experts on all 8 rows (today's layer run) or with
  `--moe-rows gather`, where each expert computes only the rows that picked it.

## 6. What the numbers say

- **The YPCB's MXU is matched to its port, and prefill and batching are MXU-bound.**
  - The MXU takes one D-weight block a cycle for each group of MCOLS rows (two at 4 bits),
    the rate the port streams them. Decode therefore streams at the port's rate.
  - Prefill and batched decode become MXU-bound from 2 rows (4-bit) or 4 rows (int8).
  - So more rows a run (TMEM x2 / x4, 64 rows of today's kernel, 8-user passes on MCOLS 4)
    gain only their per-run overheads: 2-3% on prefill, 1-2% on batching.
  - The tile kernel gains 3% on prefill at MCOLS 4: today's kernel is already about 96%
    MXU-busy there (tiling's co-sim). Its gain is on long prompts of Qwen3, LFM2 and Gemma 4,
    whose G = 2 packs 2 rows an entry. With MCOLS 8 it packs 4 and reaches 1.79x, against
    MCOLS 8's 1.38x alone.
  - More MACs a cycle gain directly: MCOLS 8 gives prefill 1.38x and batched 1.74x; an MXU
    pumped x2 gives 1.32x and 1.46x.
  - Decode moves with neither.
- **More memory bandwidth needs a wider weight path.**
  - A wider port alone gives at most ~10%, the 4-bit stream's scale bytes, because the MXU
    then bounds decode.
  - A 256-byte port with an MXU x2 gives the BCU1525 1.8x decode and 1.6-1.7x on the other
    workloads.
  - At DDR4-2400 and 200 MHz that is 4.1x the YPCB's score.
- **On a fast board, the fixed costs dominate.**
  - F2 at 512 B a cycle with an MXU x4 streams 3.3x F2's bytes but decodes only 2.5x faster.
  - The vector unit is worth 3.2% decode on F2 against 1% on the YPCB.
  - An MXU x4 needs 4 x 716 DSPs a slice (11.5K for S = 4) against the VU47P's 9,024, so it
    has to be pumped (2x clock) rather than replicated.
- **MoE.**
  - On the YPCB only 14-16% of the experts fit, so a decode token misses 87-104 experts over
    PCIe Gen2.
  - The exposed link takes 26-32% of decode. The rest is mostly the card streaming the
    experts it holds: the port, at 64%.
  - The BCU1525's two 4 GiB slices hold 39-44% and hide the link entirely. F2 holds the whole
    26B and 86% of the 35B, with the rest in its DDR4.
  - MoE prefill is bound by the port re-streaming each run's union of experts. A request
    line of 16 ids caps a layer run at 2 rows (k = 8), so a 1024-token prompt streams about
    12 experts a run, 512 runs a layer.
  - Wider requests (64 ids, 8 rows) only pay if each expert computes just the rows that
    picked it (`--moe-rows gather`): MoE 1.23-1.32x. With today's run, which computes the
    union on all rows, they lose (0.93-1.06x).
  - The MCOLS, TMEM and VPU what-ifs barely move MoE (1.00-1.06x). Memory, the host link, S
    and the request width do.

## 7. Caveats and decisions for the user

- **4 GiB a slice.** Addresses are 32 bits, so a slice holds 4 GiB of model (`--fast-gb`).
  The BCU1525's 2 x 16 GB a slice and F2's DDR4 tier need address windows or far-memory
  DMA, which is an ISA decision.
- **Gemma 4 and the MoE models run on one slice in today's code.** Their S > 1 rows assume
  tensor-parallel slices like the dense models. Section 4 gives MoE at S = 1 for comparison.
- **TMEM above 64K words** needs two RTL fixes before its what-ifs hold.
  - STREAM descriptors hold a 16-bit address, so the STREAM area must stay below word 65536
    (from tmem-pump).
  - `rtl/seq/otpu_seq.sv` tracks TMEM ranges in 16 bits (TAW = 16). Every access at or above
    word 65536 conflicts with every instruction and serializes the pipeline (from tiling).
    The tile kernel at T = 16 takes 104.8K cycles a row at TMEM 256K against 76.4K at 64K.
  - The model assumes both are fixed. Every co-sim above 64K words is pessimistic until TAW
    follows TMEM_WORDS, and the campaign dropped its TMEM 256K arm for this reason.
- **MoE layer runs are capped at R = 2** by the request line (R x k <= 16 expert ids). A
  wider line is `--moe-ids` (section 5).
- **A single pass of more than MCOLS rows** (`--act-rows` above MCOLS, MM replay) costs every
  row's overhead in the model. The card's two-pass runs hide the second pass's.
  - At TMEM 64K an ACT RAM of 32 rows runs today's kernel at R = 4 exactly as 4 rows do
    (83.6K a row, tiling).
  - One pass of 8 rows against two passes of 4 is in the campaign (section 8).
- **fp4 prompt runs of today's kernel**: the model is above the tiling agent's compile_rows
  co-sim. Its layer slopes (Qwen3, R = 4, card configuration) put the model 6-7% above at
  int8 and 9% above at fp4.
  - The card's prompt runs cost about 6% over compile_rows, which the int8 gap matches.
  - That leaves the model ~3% pessimistic on fp4 Qwen3 prefill. Its per-row overhead is
    fitted on int8 card points and hides partly under MXU-bound fp4 MMs.
  - `validate` lists these points as `co-sim layer (compile_rows)`, checked but not fitted.
- **The tile kernel at T = 16** fits 64K TMEM for Qwen3. For the larger models it is
  assumed, not probed.
- **The vector unit's share** of the overhead (0.5) is provisional until the LANES = 16
  co-sim. The conclusion does not hang on it: across shares from 0.25 to 1.0, VPU x2 is worth
  1.7-3.9% of the YPCB's score and 2.9-10.1% of F2's.
- **The S > 1 collective cost** (1500 cycles a layer and row) is an estimate. Between 0 and
  6000 cycles it moves F2's decode by -6% to +2% and its prefill by -12% to +5%.
- **Context-dependent vector work** (softmax over the context) enters only through the KV
  bytes. That fits the card's batched steps at contexts 64 and 1024 to 0.4%.
- **Assumed, not measured**: the BCU1525's PCIe and DDR4 efficiency, F2's DDR4 tier and PCIe,
  and the S > 1 collective cost.
- **Batched decode** exists for Qwen3 and the Llama-likes only. Gemma 4, LFM2 and Qwen3.5
  batching is modelled the same way but has no kernel yet.

## 8. Calibration data and the co-sim campaign

The card and co-sim points are in `tools/scoreboard.py` (`CARD_DECODE`, `CARD_PREFILL`,
`CARD_BATCH`, `COSIM`, `CARD_MOE`, `MOE_LAYER_RUNS`). After adding a point:

1. Run `scoreboard.py fit` to refit the kappas.
2. Copy them into `KAPPA`.
3. Run `scoreboard.py validate`.

The model shapes come from `scoreboard.py gen-models DIR...`, which reads the config.json
and the safetensors headers, into `tools/scoreboard_models.json`. Refresh the rows a run fits
with `tools/scoreboard_cosim.py probe`.

The co-sim campaign (`build/sbcosim.sh` on hwe-bench, two slot jobs) runs a model's first one
and two layers. The difference between them is a layer's cycles, with the run's own cost
removed. In order, it covers:

1. One pass of 8 rows (ACT RAM 8) against two passes of 4, both at TMEM 64K.
2. LANES 16 (decode and R 4, Qwen3 and Qwen3.5).
3. MCOLS 8 (prefill R 8, an 8-row int8 batch).
4. S = 2 and S = 1 at the port's memory rate.
5. The rest of the YPCB baseline.
