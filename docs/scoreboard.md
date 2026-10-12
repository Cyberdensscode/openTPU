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

## 2. The model

A run of R rows costs, in each layer and on each slice:

    cycles = max(memory / S, MXU / S) + recurrence + overhead

- **memory** is the layer's DRAM bytes over the slice's stream rate. The rate is the lower of
  the port (`port_bytes x 0.97` a cycle) and the memory (`GB/s x mem_eff / MHz`). The bytes
  are:
  - **Weights**: `ceil(R / rows a pass)` times their bytes. A weight takes 8.25 bits (int8)
    or 4.25 bits (fp4), block scales included.
  - **KV cache**: each row reads whole 256-token blocks (a sliding layer at most its window
    plus a block). It reads them once for each group of MCOLS query heads of a KV head, at
    `2d + 4d/128 + 4` bytes a token and KV head, as `gemma4._attention` and
    `qwen3._attend_heads` part them.
  - **DeltaNet**: the state, read and written once a run.
  - **LFM2**: the conv window.
  - **MoE**: the routed experts, the union of the run's rows.
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

**Prefill rows.** Prefill takes the rows a run fits from the compiler itself:
`tools/scoreboard_cosim.py probe` compiles each model at each TMEM size and ACT RAM depth
until TMEM, the ACT RAM or IMEM overflows, giving `tools/scoreboard_rmax.json`. Of the rows
that fit, the host's rule takes whole passes above one pass, and an odd R's R - 1 at 4 bits
(PAIR). Of those and of fewer whole passes, the model takes the R it finds fastest a row.

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

Against the 39 dense card points (decode, prefill, batched), the mean |error| is 0.5% and the largest 2.5%. `scoreboard.py validate` lists every point:

| points | mean \|error\| | max | leave-one-out mean | leave-one-out max |
|---|---:|---:|---:|---:|
| decode (21) | 0.4% | 2.3% | 0.5% | 3.1% |
| prefill (12) | 0.8% | 2.5% | 2.6% | 6.0% |
| batched (6) | 0.2% | 0.4% | 0.3% | 0.4% |
| co-sim (8) | 1.1% | 2.6% | 1.2% | 2.6% |
| MoE layer run ms (6) | 2.7% | 4.7% | - | - |
| MoE (4) | 2.2% | 2.8% | - | - |

- **Card points**: production builds at 133.33 MHz, from the decode counters
  (`tools/qual/perf.py`), the prompt runs (docs/prefill.md 7), otpu-serve's batched steps,
  and the streamed MoEs' sessions (docs/offload.md 13.7, 14.4).
- **Co-sim points**: the RTL with LiteDRAM's DDR3-1066 controller, including a Qwen3.5 prefill
  run at MCOLS 8 that the model never saw in its fit.
- **Leave-one-out**: refits the point's model without that point. A model with a single
  prefill point predicts it from its decode points alone: Gemma 4 E2B at 6.0% and SmolLM3
  at 4.6% are the largest.
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
| YPCB (production) | 16.6 | 15.4 | 44.0 | 32.0 | 6.88 | 1.00x |
| BCU1525 DDR4-1600, 100 MHz | 24.8 | 22.0 | 59.2 | 43.0 | 11.17 | 1.43x |
| BCU1525 DDR4-1600, 200 MHz | 48.6 | 43.1 | 116.0 | 85.0 | 21.05 | 2.78x |
| BCU1525 DDR4-2400, 100 MHz | 24.8 | 22.0 | 59.2 | 43.0 | 11.17 | 1.43x |
| BCU1525 DDR4-2400, 200 MHz | 49.7 | 43.9 | 116.8 | 85.3 | 21.35 | 2.81x |
| BCU1525 DDR4-2400, 200 MHz, port 256 + MXU x2 | 73.0 | 62.0 | 172.2 | 132.7 | 27.28 | 4.01x |
| F2 (HBM, S 4, 250 MHz) | 124.2 | 101.1 | 244.3 | 178.2 | 43.76 | 5.99x |
| F2, port 256 + MXU x2 | 248.3 | 175.5 | 363.1 | 271.4 | 63.46 | 9.26x |
| F2, port 512 + MXU x4 | 414.0 | 250.1 | 469.6 | 362.3 | 77.89 | 12.20x |

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

summary (geometric means, fp4 weights, context 512): decode 15.4 tok/s, prefill 1024 44.0 tok/s, batched 16 users 32.0 tok/s in all, MoE 6.88 (sqrt of decode x prefill 1024 tok/s)

| model | decode tok/s | MB/token | prefill 256 tok/s | TTFT 256 s | prefill 1024 tok/s | TTFT 1024 s | rows a run |
|---|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | 35.8 | 440 | 138.4 | 1.85 | 135.4 | 7.56 | 8 |
| lfm2 | 97.2 | 165 | 384.9 | 0.67 | 349.5 | 2.93 | 8..4 |
| qwen35 | 27.5 | 571 | 76.9 | 3.33 | 77.2 | 13.26 | 4 |
| qwen35-2b | 12.6 | 1,301 | 41.0 | 6.24 | 41.2 | 24.86 | 4 |
| qwen35-4b | 6.1 | 2,671 | 14.6 | 17.49 | 14.6 | 70.25 | 2 |
| smollm3 | 8.9 | 1,794 | 22.5 | 11.39 | 22.5 | 45.52 | 4 |
| phi4-mini | 6.7 | 2,397 | 17.5 | 14.67 | 17.0 | 60.17 | 2 |
| lfm2-2.6b | 11.1 | 1,446 | 26.7 | 9.59 | 24.4 | 41.90 | 4..2 |
| gemma4 | 10.0 | 1,446 | 30.2 | 8.47 | 30.3 | 33.80 | 4 |

| batched decode, total / each tok/s | 1 users | 4 users | 8 users | 16 users |
|---|---:|---:|---:|---:|
| qwen3 | 35.8 / 35.8 | 96.5 / 24.1 | 96.5 / 12.1 | 96.5 / 6.0 |
| smollm3 | 8.9 / 8.9 | 20.6 / 5.2 | 20.6 / 2.6 | 20.6 / 1.3 |
| phi4-mini | 6.7 / 6.7 | 16.5 / 4.1 | 16.5 / 2.1 | 16.5 / 1.0 |

| MoE | experts on the card | decode tok/s | misses a token | prefill 256 tok/s | TTFT 256 s | prefill 1024 tok/s | TTFT 1024 s |
|---|---:|---:|---:|---:|---:|---:|---:|
| gemma4-26b | 14% (535 slots) | 3.55 | 87 | 8.3 | 31.0 | 8.6 | 119.2 |
| qwen35-35b | 16% (1615 slots) | 5.43 | 104 | 11.9 | 21.5 | 13.5 | 75.8 |

### BCU1525 DDR4-1600, 100 MHz

**bcu1525: S 2, MCOLS 4 (rows a pass 4, MXU x1), TMEM 64K words, VPU x1, 100 MHz, DDR4-1600 x 2 a slice: 24.3 GB/s, 124.2 B a cycle (port-bound), 24.8 GB/s in all; host link 9 GB/s**

summary (geometric means, fp4 weights, context 512): decode 22.0 tok/s, prefill 1024 59.2 tok/s, batched 16 users 43.0 tok/s in all, MoE 11.17 (sqrt of decode x prefill 1024 tok/s)

| model | decode tok/s | MB/token | prefill 256 tok/s | TTFT 256 s | prefill 1024 tok/s | TTFT 1024 s | rows a run |
|---|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | 50.1 | 440 | 175.3 | 1.46 | 172.1 | 5.95 | 8 |
| lfm2 | 137.6 | 165 | 487.5 | 0.53 | 433.2 | 2.36 | 8..4 |
| qwen35 | 38.7 | 571 | 97.4 | 2.63 | 97.7 | 10.48 | 4 |
| qwen35-2b | 18.5 | 1,301 | 59.5 | 4.30 | 59.7 | 17.15 | 4 |
| qwen35-4b | 9.1 | 2,671 | 21.5 | 11.92 | 21.4 | 47.87 | 2 |
| smollm3 | 12.8 | 1,794 | 30.9 | 8.27 | 31.0 | 33.06 | 4 |
| phi4-mini | 9.7 | 2,397 | 24.1 | 10.63 | 23.5 | 43.51 | 2 |
| lfm2-2.6b | 16.2 | 1,446 | 37.7 | 6.80 | 34.5 | 29.71 | 4..2 |
| gemma4 | 13.2 | 1,446 | 38.2 | 6.71 | 38.3 | 26.77 | 4 |

| batched decode, total / each tok/s | 1 users | 4 users | 8 users | 16 users |
|---|---:|---:|---:|---:|
| qwen3 | 50.1 / 50.1 | 120.9 / 30.2 | 120.9 / 15.1 | 120.9 / 7.6 |
| smollm3 | 12.8 / 12.8 | 28.6 / 7.1 | 28.6 / 3.6 | 28.6 / 1.8 |
| phi4-mini | 9.7 / 9.7 | 23.0 / 5.8 | 23.0 / 2.9 | 23.0 / 1.4 |

| MoE | experts on the card | decode tok/s | misses a token | prefill 256 tok/s | TTFT 256 s | prefill 1024 tok/s | TTFT 1024 s |
|---|---:|---:|---:|---:|---:|---:|---:|
| gemma4-26b | 44% (1694 slots) | 6.48 | 16 | 11.5 | 22.3 | 11.4 | 89.9 |
| qwen35-35b | 39% (4005 slots) | 11.33 | 41 | 18.0 | 14.2 | 18.6 | 55.0 |

### BCU1525 DDR4-2400, 200 MHz

**bcu1525: S 2, MCOLS 4 (rows a pass 4, MXU x1), TMEM 64K words, VPU x1, 200 MHz, DDR4-2400 x 2 a slice: 36.5 GB/s, 124.2 B a cycle (port-bound), 49.7 GB/s in all; host link 9 GB/s**

summary (geometric means, fp4 weights, context 512): decode 43.9 tok/s, prefill 1024 116.8 tok/s, batched 16 users 85.3 tok/s in all, MoE 21.35 (sqrt of decode x prefill 1024 tok/s)

| model | decode tok/s | MB/token | prefill 256 tok/s | TTFT 256 s | prefill 1024 tok/s | TTFT 1024 s | rows a run |
|---|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | 100.2 | 440 | 346.0 | 0.74 | 339.8 | 3.01 | 8 |
| lfm2 | 275.2 | 165 | 940.6 | 0.27 | 819.8 | 1.25 | 8..4 |
| qwen35 | 77.5 | 571 | 192.0 | 1.33 | 192.5 | 5.32 | 4 |
| qwen35-2b | 37.1 | 1,301 | 117.9 | 2.17 | 118.4 | 8.65 | 4 |
| qwen35-4b | 18.2 | 2,671 | 42.7 | 6.00 | 42.5 | 24.09 | 2 |
| smollm3 | 25.6 | 1,794 | 61.6 | 4.16 | 61.7 | 16.61 | 4 |
| phi4-mini | 19.3 | 2,397 | 47.8 | 5.35 | 46.7 | 21.91 | 2 |
| lfm2-2.6b | 32.4 | 1,446 | 74.9 | 3.42 | 68.3 | 14.99 | 4..2 |
| gemma4 | 26.4 | 1,446 | 75.9 | 3.37 | 76.1 | 13.46 | 4 |

| batched decode, total / each tok/s | 1 users | 4 users | 8 users | 16 users |
|---|---:|---:|---:|---:|
| qwen3 | 100.2 / 100.2 | 237.4 / 59.4 | 237.4 / 29.7 | 237.4 / 14.8 |
| smollm3 | 25.6 / 25.6 | 56.9 / 14.2 | 56.9 / 7.1 | 56.9 / 3.6 |
| phi4-mini | 19.3 / 19.3 | 45.9 / 11.5 | 45.9 / 5.7 | 45.9 / 2.9 |

| MoE | experts on the card | decode tok/s | misses a token | prefill 256 tok/s | TTFT 256 s | prefill 1024 tok/s | TTFT 1024 s |
|---|---:|---:|---:|---:|---:|---:|---:|
| gemma4-26b | 44% (1694 slots) | 12.96 | 16 | 21.8 | 11.7 | 21.9 | 46.7 |
| qwen35-35b | 39% (4005 slots) | 21.30 | 41 | 32.4 | 7.9 | 34.3 | 29.8 |

### F2 (HBM, S 4, 250 MHz)

**f2: S 4, MCOLS 4 (rows a pass 4, MXU x1), TMEM 64K words, VPU x1, 250 MHz, HBM 115 GB/s a slice: 103.5 GB/s, 124.2 B a cycle (port-bound), 124.2 GB/s in all; host link 20 GB/s**

summary (geometric means, fp4 weights, context 512): decode 101.1 tok/s, prefill 1024 244.3 tok/s, batched 16 users 178.2 tok/s in all, MoE 43.76 (sqrt of decode x prefill 1024 tok/s)

| model | decode tok/s | MB/token | prefill 256 tok/s | TTFT 256 s | prefill 1024 tok/s | TTFT 1024 s | rows a run |
|---|---:|---:|---:|---:|---:|---:|---:|
| qwen3 | 225.2 | 440 | 684.1 | 0.37 | 674.4 | 1.52 | 8 |
| lfm2 | 635.2 | 165 | 1,831.9 | 0.14 | 1,498.1 | 0.68 | 8..4 |
| qwen35 | 174.7 | 571 | 367.1 | 0.70 | 368.0 | 2.78 | 4 |
| qwen35-2b | 90.0 | 1,301 | 276.7 | 0.93 | 277.6 | 3.69 | 4 |
| qwen35-4b | 44.6 | 2,671 | 101.8 | 2.51 | 101.4 | 10.09 | 2 |
| smollm3 | 59.5 | 1,794 | 133.1 | 1.92 | 133.2 | 7.69 | 4 |
| phi4-mini | 45.2 | 2,397 | 102.7 | 2.49 | 100.7 | 10.17 | 2 |
| lfm2-2.6b | 76.5 | 1,446 | 168.1 | 1.52 | 152.0 | 6.74 | 4..2 |
| gemma4 | 53.5 | 1,446 | 145.2 | 1.76 | 145.4 | 7.04 | 4 |

| batched decode, total / each tok/s | 1 users | 4 users | 8 users | 16 users |
|---|---:|---:|---:|---:|
| qwen3 | 225.2 / 225.2 | 451.2 / 112.8 | 451.2 / 56.4 | 451.2 / 28.2 |
| smollm3 | 59.5 / 59.5 | 124.1 / 31.0 | 124.1 / 15.5 | 124.1 / 7.8 |
| phi4-mini | 45.2 / 45.2 | 101.1 / 25.3 | 101.1 / 12.6 | 101.1 / 6.3 |

| MoE | experts on the card | decode tok/s | misses a token | prefill 256 tok/s | TTFT 256 s | prefill 1024 tok/s | TTFT 1024 s |
|---|---:|---:|---:|---:|---:|---:|---:|
| gemma4-26b | all | 27.04 | 0 | 42.4 | 6.0 | 41.7 | 24.5 |
| qwen35-35b | 86% (8786 slots) + 1453 far | 50.81 | 4 | 63.2 | 4.0 | 64.0 | 16.0 |

### MoE on one slice (today's code) against S slices

| board | model | slices | experts on the card | decode tok/s | prefill 1024 tok/s |
|---|---|---:|---:|---:|---:|
| BCU1525 DDR4-2400, 200 MHz | gemma4-26b | 1 | 14% | 7.20 | 12.9 |
| BCU1525 DDR4-2400, 200 MHz | gemma4-26b | 2 | 44% | 12.96 | 21.9 |
| BCU1525 DDR4-2400, 200 MHz | qwen35-35b | 1 | 16% | 10.94 | 20.3 |
| BCU1525 DDR4-2400, 200 MHz | qwen35-35b | 2 | 39% | 21.30 | 34.3 |
| F2 | gemma4-26b | 1 | 14% + 3304 far | 8.54 | 16.0 |
| F2 | gemma4-26b | 4 | all | 27.04 | 41.7 |
| F2 | qwen35-35b | 1 | 16% + 8624 far | 14.22 | 25.0 |
| F2 | qwen35-35b | 4 | 86% + 1453 far | 50.81 | 64.0 |


## 5. What-ifs

`scoreboard.py whatifs [--board B]` gives each what-if's ratio to the board.

**Against ypcb** (S 1, MCOLS 4 (rows a pass 4, MXU x1), TMEM 64K words, VPU x1, 133.33 MHz, DDR3-1066.67 x 2 a slice: 16.6 GB/s, 124.2 B a cycle (port-bound), 16.6 GB/s in all; host link 2.2 GB/s):

| what-if (against ypcb) | decode | prefill | batched | MoE | score |
|---|---:|---:|---:|---:|---:|
| MCOLS 8 | 1.001x | 1.359x | 1.733x | 1.003x | 1.240x |
| TMEM x2 (8 ACT RAM rows) | 1.000x | 1.034x | 1.015x | 1.000x | 1.012x |
| TMEM x4 (16 ACT RAM rows) | 1.000x | 1.033x | 1.018x | 1.000x | 1.013x |
| tiled prefill, 64 rows a run | 1.000x | 1.035x | 1.018x | 1.000x | 1.013x |
| 8 users in one pass (replay) | 1.000x | 0.997x | 1.015x | 1.000x | 1.003x |
| VPU x2 | 1.010x | 1.048x | 1.021x | 1.027x | 1.027x |
| MXU pumped x2 | 1.000x | 1.252x | 1.306x | 1.000x | 1.131x |
| MCOLS 8 + TMEM x4 + VPU x2 | 1.011x | 1.759x | 1.905x | 1.031x | 1.367x |
| MXU x2 + TMEM x4 + VPU x2 (pumped) | 1.010x | 1.793x | 1.905x | 1.027x | 1.372x |
| port 256 B a cycle (memory permitting) | 1.000x | 1.000x | 1.000x | 1.000x | 1.000x |
| port 256 B + MXU x2 | 1.000x | 1.252x | 1.306x | 1.000x | 1.131x |

**Against bcu1525** (S 2, MCOLS 4 (rows a pass 4, MXU x1), TMEM 64K words, VPU x1, 100 MHz, DDR4-1600 x 2 a slice: 24.3 GB/s, 124.2 B a cycle (port-bound), 24.8 GB/s in all; host link 9 GB/s):

| what-if (against bcu1525) | decode | prefill | batched | MoE | score |
|---|---:|---:|---:|---:|---:|
| MCOLS 8 | 1.001x | 1.300x | 1.632x | 1.003x | 1.208x |
| TMEM x2 (8 ACT RAM rows) | 1.000x | 1.032x | 1.017x | 1.000x | 1.012x |
| TMEM x4 (16 ACT RAM rows) | 1.000x | 1.030x | 1.022x | 1.000x | 1.013x |
| tiled prefill, 64 rows a run | 1.000x | 1.041x | 1.022x | 1.000x | 1.015x |
| 8 users in one pass (replay) | 1.000x | 0.990x | 1.017x | 1.000x | 1.002x |
| VPU x2 | 1.019x | 1.060x | 1.038x | 1.042x | 1.040x |
| MXU pumped x2 | 1.000x | 1.216x | 1.275x | 1.000x | 1.116x |
| MCOLS 8 + TMEM x4 + VPU x2 | 1.020x | 1.687x | 1.821x | 1.046x | 1.345x |
| MXU x2 + TMEM x4 + VPU x2 (pumped) | 1.019x | 1.715x | 1.821x | 1.042x | 1.350x |
| port 256 B a cycle (memory permitting) | 1.121x | 1.044x | 1.008x | 1.104x | 1.068x |
| port 256 B + MXU x2 | 1.809x | 1.626x | 1.688x | 1.590x | 1.676x |

**Against f2** (S 4, MCOLS 4 (rows a pass 4, MXU x1), TMEM 64K words, VPU x1, 250 MHz, HBM 115 GB/s a slice: 103.5 GB/s, 124.2 B a cycle (port-bound), 124.2 GB/s in all; host link 20 GB/s):

| what-if (against f2) | decode | prefill | batched | MoE | score |
|---|---:|---:|---:|---:|---:|
| MCOLS 8 | 1.001x | 1.236x | 1.526x | 1.003x | 1.173x |
| TMEM x2 (8 ACT RAM rows) | 1.000x | 1.041x | 1.032x | 1.000x | 1.018x |
| TMEM x4 (16 ACT RAM rows) | 1.000x | 1.041x | 1.045x | 1.000x | 1.021x |
| tiled prefill, 64 rows a run | 1.000x | 1.089x | 1.045x | 1.000x | 1.033x |
| 8 users in one pass (replay) | 1.000x | 0.985x | 1.032x | 1.000x | 1.004x |
| VPU x2 | 1.034x | 1.077x | 1.064x | 1.056x | 1.058x |
| MXU pumped x2 | 1.000x | 1.170x | 1.229x | 1.000x | 1.095x |
| MCOLS 8 + TMEM x4 + VPU x2 | 1.035x | 1.635x | 1.759x | 1.059x | 1.333x |
| MXU x2 + TMEM x4 + VPU x2 (pumped) | 1.034x | 1.660x | 1.759x | 1.056x | 1.336x |
| port 256 B a cycle (memory permitting) | 1.110x | 1.038x | 1.007x | 1.088x | 1.060x |
| port 256 B + MXU x2 | 1.736x | 1.486x | 1.523x | 1.450x | 1.545x |


What each what-if sets:

- **MCOLS 8**: `--mcols 8`, which also means rows a pass 8.
- **TMEM x2 / x4**: `--tmem-words 131072 / 262144` with ACT RAM rows 8 / 16, giving the
  probe's rows a run.
- **Tiled prefill**: `--rows 64 --act-rows 64 --imem 0`.
- **8 users in one pass**: `--batch-rows 8 --act-rows 8`, with the MXU still at MCOLS 4.
- **VPU x2**: `--vpu 2`.
- **MXU pumped x2**: `--mxu-pump 2`.
- **Port 256 B**: `--port-bytes 256`, alone or with an MXU x2.

## 6. What the numbers say

- **The YPCB's MXU is matched to its port, and prefill and batching are MXU-bound.**
  - The MXU takes one D-weight block a cycle for each group of MCOLS rows (two at 4 bits),
    the rate the port streams them. Decode therefore streams at the port's rate.
  - Prefill and batched decode become MXU-bound from 2 rows (4-bit) or 4 rows (int8).
  - So more rows a run (TMEM x2 / x4, 64-row tiling, 8-user passes on MCOLS 4) gain only
    their per-run overheads: 2-4% on prefill, 1-2% on batching.
  - More MACs a cycle gain directly: MCOLS 8 gives prefill 1.36x and batched 1.73x; an MXU
    pumped x2 gives 1.25x and 1.31x.
  - Decode moves with neither.
- **More memory bandwidth needs a wider weight path.**
  - A wider port alone gives at most ~12%, the 4-bit stream's scale bytes, because the MXU
    then bounds decode.
  - A 256-byte port with an MXU x2 gives the BCU1525 1.8x decode and 1.6-1.7x on the other
    workloads.
  - At DDR4-2400 and 200 MHz that is 4.0x the YPCB's score.
- **On a fast board, the fixed costs dominate.**
  - F2 at 512 B a cycle with an MXU x4 streams 3.3x F2's bytes but decodes only 2.5x faster.
  - The vector unit is worth 3.4% decode on F2 against 1% on the YPCB.
  - An MXU x4 needs 4 x 716 DSPs a slice (11.5K for S = 4) against the VU47P's 9,024, so it
    has to be pumped (2x clock) rather than replicated.
- **MoE on the YPCB is link-bound.**
  - Only 14-16% of the experts fit, so a token misses 87-104 experts over PCIe Gen2.
  - The BCU1525's two 4 GiB slices hold 39-44%.
  - F2 holds the whole 26B and 86% of the 35B, with the rest in its DDR4.
  - MCOLS, TMEM and VPU what-ifs barely move MoE (1.00-1.06x). Memory, the host link and
    S do.

## 7. Caveats and decisions for the user

- **4 GiB a slice.** Addresses are 32 bits, so a slice holds 4 GiB of model (`--fast-gb`).
  The BCU1525's 2 x 16 GB a slice and F2's DDR4 tier need address windows or far-memory
  DMA, which is an ISA decision.
- **Gemma 4 and the MoE models run on one slice in today's code.** Their S > 1 rows assume
  tensor-parallel slices like the dense models. Section 4 gives MoE at S = 1 for comparison.
- **TMEM above 64K words** (from tmem-pump): STREAM descriptors hold a 16-bit address, so
  the STREAM area must stay below word 65536.
- **MoE layer runs are capped at R = 2** by the request line (R x k <= 16 expert ids). A
  wider line is `CAL["moe_ids"]`.
- **A single pass of more than MCOLS rows** (`--act-rows` above MCOLS, MM replay) costs every
  row's overhead in the model. The card's two-pass runs hide the second pass's. This is why
  the TMEM x4 what-if with 16 ACT RAM rows can lose a little on Qwen3 against two passes of
  4. The co-sim campaign's TMEM x4 runs settle it (section 8).
- **The vector unit's share** of the overhead (0.5) is provisional until the LANES = 16
  co-sim.
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

The co-sim campaign (`build/sbcosim.sh` on omarchy, one slot job) runs a model's first one
and two layers. The difference between them is a layer's cycles, with the run's own cost
removed. In order, it covers:

1. TMEM x4 with 16 ACT RAM rows (one pass of 8 and 16 rows) against two passes of 4.
2. LANES 16 (decode and R 4, Qwen3 and Qwen3.5).
3. MCOLS 8 (prefill R 8, an 8-row int8 batch).
4. S = 2 and S = 1 at the port's memory rate.
5. The rest of the YPCB baseline.
