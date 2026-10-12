#!/bin/bash
# The scoreboard's co-simulation campaign, in slot jobs (PART 1: the YPCB's configuration
# and one pass of 8 rows on an 8-row ACT RAM; PART 2: LANES 16, MCOLS 8, the port-rate memory
# at S = 1 and 2, then the rest of the YPCB baseline; PARTS 3 and 4: Qwen3.5's DeltaNet layers
# at MCOLS 4 and 8, the 0.8B and 2B, then the 4B). Each run's prefixes 1:2 give a layer's
# cycles (2 layers - 1 layer).
# Results: one JSON line a run in build/sbcosim-PART.jsonl (tools/scoreboard_cosim_runs.jsonl:
# the parts of 2026-10-12). Copy the tree to the host's ~/otpu-build/workloads/ and run each
# part through ~/otpu-build/slot.sh 8 LOG -- tools/scoreboard_campaign.sh PART.
cd "$(dirname "$0")/.."
PART=${1:?part 1, 2, 3 or 4}
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
elif [ "$PART" = 2 ]; then
  # B. the vector unit x 2 (16 lanes). The stream engine exists at 8 lanes only: on
  # 2026-10-12 the Qwen3.5 run had DSTEP on and stopped at its first one (error 1);
  # scoreboard_cosim.py now builds 16 lanes without DSTEP / STREAM (DeltaNet on VOPs)
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
elif [ "$PART" = 3 ]; then
  # G. Qwen3.5's DeltaNet layer at MCOLS 4 and 8 (the rows a prompt run takes at each); the
  # 0.8B at MCOLS 4 is E'
  run $M/Qwen3.5-0.8B --layers 1:2 --rows 4 --pos 512 --mcols 8
  run $M/Qwen3.5-2B --layers 1:2 --rows 4 --pos 512
  run $M/Qwen3.5-2B --layers 1:2 --rows 4 --pos 512 --mcols 8
else
  run $M/Qwen3.5-4B --layers 1:2 --rows 2 --pos 512
  run $M/Qwen3.5-4B --layers 1:2 --rows 3 --pos 512 --mcols 8
fi
date
echo CAMPAIGN DONE
