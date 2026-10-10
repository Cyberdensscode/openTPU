#!/usr/bin/env bash
# Build the openTPU bitstream for the BCU1525 (xcvu9p-fsgd2104-2L-e) in Vivado batch mode
# (docs/bcu1525.md). The knobs, as boards/ypcb-00338/run_vivado.sh's:
#   S=2 MCOLS=4 VPU_CL=2 LANES=8 ACT_ROWS=MCOLS DSTEP=1   the accelerator (S slices, two DDR4 channels each)
#   CORE_MHZ=100 (the default: development builds at 100 MHz until the design runs; FAST=1: Vivado's
#   default strategies)   PCIE_LANES=16|8 (Gen3)   BUILD_ID   OUT_DIR   JOBS   STEP=impl   IMPL_STRATEGY
# The VU9P is outside Vivado's free tier: the license must cover it, and the Virtex UltraScale+
# device files must be installed (Vivado's installer, "Virtex UltraScale+").
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"
out="${OUT_DIR:-$root/build/vivado_bcu1525}"
jobs="${JOBS:-8}"
s="${S:-2}"
mcols="${MCOLS:-4}"
vpu_cl="${VPU_CL:-2}"
lanes="${LANES:-8}"
act_rows="${ACT_ROWS:-$mcols}"
dstep="${DSTEP:-1}"
core_mhz="${CORE_MHZ:-100}"
build_id="${BUILD_ID:-$(git -C "$root" rev-parse HEAD 2>/dev/null | cut -c1-8)}"
build_id="${build_id:-0}"
mkdir -p "$out"
export OTPU_IP_CACHE="${OTPU_IP_CACHE-$HOME/.cache/otpu-vivado-ip}"
export OTPU_FAST="${FAST:-0}"
export PCIE_LANES="${PCIE_LANES:-16}"
run() { local script="$1"; shift
  vivado -mode batch -nojournal -log "$out/$(basename "$script" .tcl).log" -source "$script" -tclargs "$@"; }
if [[ "${STEP:-}" == impl ]]; then
  run "$here/vivado/build.tcl" "$out" "$jobs" impl "${IMPL_STRATEGY:-}"
else
  run "$here/vivado/create_project.tcl" "$out" "$mcols" "$core_mhz" "$build_id" "$vpu_cl" "$lanes" "$act_rows" "$dstep" "$s"
  run "$here/vivado/build.tcl" "$out" "$jobs" full "${IMPL_STRATEGY:-}"
fi
echo "done: $out/otpu.bit"
