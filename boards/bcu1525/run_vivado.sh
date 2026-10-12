#!/usr/bin/env bash
# Build the openTPU bitstream for the BCU1525 (xcvu9p-fsgd2104-2L-e) in Vivado batch mode
# (docs/bcu1525.md). The knobs, as boards/ypcb-00338/run_vivado.sh's:
#   S=2 MCOLS=4 VPU_CL=2 LANES=8 ACT_ROWS=MCOLS DSTEP=1   the accelerator (S slices, two DDR4 channels each)
#   CORE_MHZ=100 (the default: development builds at 100 MHz until the design runs; FAST=1: Vivado's
#   default strategies)   PCIE_LANES=16|8 (Gen3)   TMEM_URAM=0|1 (the TMEM in UltraRAM)   BUILD_ID
#   OUT_DIR   JOBS   STEP=impl   IMPL_STRATEGY   PREFLIGHT=1|0
# The VU9P is outside Vivado's free tier: the license must cover it, and the Virtex UltraScale+
# device files must be installed (Vivado's installer, "Virtex UltraScale+"). The preflight
# (scripts/preflight.tcl, about a minute) checks both first and stops with what is missing.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"
out="${OUT_DIR:-$root/build/vivado_bcu1525}"
jobs="${JOBS:-8}"
# The target (targets/bcu1525.toml through opentpu/target.py): the knobs' defaults, the platform
# id, the DDR4 rate and the arch hash the CFG_* registers report (without python3: the same
# built-in values, CFG_HASH 0)
if tenv="$(PYTHONPATH="$root" python3 -m opentpu.target env bcu1525 2>/dev/null)"; then
  eval "$tenv"
else
  echo "run_vivado: python3 -m opentpu.target env bcu1525 failed: built-in defaults, CFG_HASH 0" >&2
fi
s="${S:-${T_S:-2}}"
mcols="${MCOLS:-${T_MCOLS:-4}}"
vpu_cl="${VPU_CL:-${T_VPU_CL:-2}}"
lanes="${LANES:-${T_LANES:-8}}"
act_rows="${ACT_ROWS:-$mcols}"
dstep="${DSTEP:-${T_DSTEP:-1}}"
# otpu_seq WAITC (EXPERIMENTAL, docs/seq_vpu.md; CFG_UARCH bit24, in CFG_HASH when on): the
# target's [arch] SEQ_WC, or SEQ_WC=1. The host writes each program's wait entries
seq_wc="${SEQ_WC:-${T_SEQ_WC:-0}}"
core_mhz="${CORE_MHZ:-${T_CORE_MHZ:-100}}"
platform="${T_PLATFORM:-2}"
ddr_mts="${T_DDR_MTS:-1600}"
arch_hash="$(PYTHONPATH="$root" python3 -m opentpu.target hash bcu1525 MCOLS="$mcols" \
  LANES="$lanes" ACT_ROWS="$act_rows" DSTEP="$dstep" SEQ_WC="$seq_wc" 2>/dev/null || echo 0)"
build_id="${BUILD_ID:-$(git -C "$root" rev-parse HEAD 2>/dev/null | cut -c1-8)}"
build_id="${build_id:-0}"
mkdir -p "$out"
export OTPU_IP_CACHE="${OTPU_IP_CACHE-$HOME/.cache/otpu-vivado-ip}"
export OTPU_FAST="${FAST:-0}"
export OTPU_SEQ_WC_GEN="$seq_wc"
export PCIE_LANES="${PCIE_LANES:-16}"
export TMEM_URAM="${TMEM_URAM:-0}"     # 1: the TMEM copies in UltraRAM (vivado/tmem_uram.xdc)
run() { local script="$1"; shift
  vivado -mode batch -nojournal -log "$out/$(basename "$script" .tcl).log" -source "$script" -tclargs "$@"; }
if [[ "${PREFLIGHT:-1}" != 0 ]]; then
  part="${T_PART:-xcvu9p-fsgd2104-2L-e}"
  if ! command -v vivado > /dev/null; then
    echo "run_vivado: no vivado on PATH (source <Vivado>/settings64.sh)" >&2; exit 2
  fi
  if ! (cd "$out" && vivado -mode batch -nojournal -nolog -source "$here/scripts/preflight.tcl" \
          -tclargs "$part") > "$out/preflight.log" 2>&1; then
    grep -E "PREFLIGHT|ERROR" "$out/preflight.log" >&2 || tail -20 "$out/preflight.log" >&2
    echo "run_vivado: preflight failed ($out/preflight.log); PREFLIGHT=0 skips it" >&2
    exit 2
  fi
  grep "PREFLIGHT ok" "$out/preflight.log" >&2
fi
if [[ "${STEP:-}" == impl ]]; then
  run "$here/vivado/build.tcl" "$out" "$jobs" impl "${IMPL_STRATEGY:-}"
else
  run "$here/vivado/create_project.tcl" "$out" "$mcols" "$core_mhz" "$build_id" "$vpu_cl" "$lanes" \
    "$act_rows" "$dstep" "$s" "$platform" "$arch_hash" "$ddr_mts"
  run "$here/vivado/build.tcl" "$out" "$jobs" full "${IMPL_STRATEGY:-}"
fi
echo "done: $out/otpu.bit"
