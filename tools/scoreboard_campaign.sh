#!/bin/bash
# The scoreboard's co-simulation campaign, in two slot jobs (PART 1: the YPCB's configuration
# and one pass of 8 rows on an 8-row ACT RAM; PART 2: LANES 16, MCOLS 8, the port-rate memory
# at S = 1 and 2, then the rest of the YPCB baseline). Each run's prefixes 1:2 give a layer's
# cycles (2 layers - 1 layer).
# Results: one JSON line a run in build/sbcosim-PART.jsonl (tools/scoreboard_cosim_runs.jsonl:
# both parts of 2026-10-12). Copy the tree to the host's ~/otpu-build/workloads/ and run each
# part through ~/otpu-build/slot.sh 8 LOG -- tools/scoreboard_campaign.sh PART.
cd "$(dirname "$0")/.."
PART=${1:?part 1 or 2}
export PYTHONPATH=$PWD OTPU_SIM_TMP=${OTPU_SIM_TMP:-$HOME/otpu-build/tmp}
mkdir -p "$OTPU_SIM_TMP"
M=${MODELS:-$HOME/otpu-models}
mkdir -p build; OUT=$PWD/build/sbcosim-$PART.jsonl
run() { echo "== $(date +%T) $*"; python3 tools/scoreboard_cosim.py run "$@" --out "$OUT" || echo "{\"failed\": \"$*\"}" >> "$OUT"; }
date
if [ "$PART" = 1 ]; then
  # E. one pass of 8 rows (ACT RAM 8, MM replay) against two passes of 4, both TMEM 64K
  # (TMEM above 64K serializes otpu_seq: its range tracking is 16 bits, the tiling agent)
  run $M/Qwen3-0.6B --layers 1:2 --rows 8 --pos 512
  run $M/Qwen3-0.6B --layers 1:2 --rows 8 --pos 512 --act-rows 8
  # the baselines of B and C
  run $M/Qwen3-0.6B --layers 1:2 --rows 0,4 --pos 544
  run $M/Qwen3.5-0.8B --layers 1:2 --rows 0,4 --pos 544
  run $M/Qwen3-0.6B --layers 1:2 --rows 4 --pos 544 --batch --wformat int8 --head-format int8
  # E'. Qwen3.5 (DeltaNet) prefill R 4
  run $M/Qwen3.5-0.8B --layers 1:2 --rows 4 --pos 512
else
  # B. the vector unit x 2 (16 lanes)
  run $M/Qwen3-0.6B --layers 1:2 --rows 0,4 --pos 544 --lanes 16
  run $M/Qwen3.5-0.8B --layers 1:2 --rows 0,4 --pos 544 --lanes 16
  # C. MCOLS 8
  run $M/Qwen3-0.6B --layers 1:2 --rows 8 --pos 512 --mcols 8
  run $M/Qwen3-0.6B --layers 1:2 --rows 8 --pos 544 --batch --mcols 8 --wformat int8 --head-format int8
  # F. memory at the port's rate (100% of a beat a cycle a channel, 40 cycles), S = 1 and 2
  run $M/Qwen3-0.6B --layers 1:2 --rows 0,4 --pos 544 --mem port --bw 100 --lat 40
  run $M/Qwen3-0.6B --layers 1:2 --rows 0,4 --pos 544 --mem port --bw 100 --lat 40 -S 2
  # A. the rest of the YPCB baseline
  run $M/LFM2.5-230M --layers 1:2:3 --rows 0 --pos 544
  run $M/LFM2.5-230M --layers 1:2:3 --rows 4,8 --pos 512
  run $M/Qwen3.5-0.8B --layers 1:2:3:4 --rows 0 --pos 544
  run $M/Qwen3-0.6B --layers 1:2 --rows 2 --pos 512
fi
date
echo CAMPAIGN DONE
