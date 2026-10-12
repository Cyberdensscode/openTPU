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

## Status

This is the first version, 2026-10-11. Its validation against the card is good to about 1%
on dense decode, prefill and batched decode, and about 3% on MoE. The baselines for the
other boards, the what-ifs and the full method follow in the sections below.
