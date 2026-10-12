#!/bin/bash
# EXPERIMENTAL (docs/vpu_pump.md): build and run tb_vpu_pump with Verilator. Usage, from anywhere:
#   rtl/exp/vpu_pump/run_tb.sh [build dir]
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
B=${1:-$ROOT/build/vpu_pump_tb}
mkdir -p "$B"
verilator --binary --timing -Wno-fatal -Wno-WIDTHEXPAND -Wno-WIDTHTRUNC -Wno-UNUSEDSIGNAL \
  --top-module tb_vpu_pump -Mdir "$B" "$ROOT/rtl/vpu/otpu_fp.sv" "$ROOT/rtl/vpu/otpu_fpipe.sv" \
  "$ROOT/rtl/exp/vpu_pump/vpu_pump_lane.sv" "$ROOT/rtl/exp/vpu_pump/tb_vpu_pump.sv" \
  > "$B/build.log" 2>&1 || { tail -30 "$B/build.log"; exit 1; }
"$B/Vtb_vpu_pump"
