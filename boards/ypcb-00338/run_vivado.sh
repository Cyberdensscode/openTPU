#!/usr/bin/env bash
# Build the openTPU bitstream for the YPCB-00338 in Vivado batch mode, natively or in Docker.
#
#   ./run_vivado.sh [1066]                # native: `vivado` on PATH (x86-64 Linux / Windows WSL)
#                                         # the LiteDRAM core (DDR3-1066, the one speed it is generated
#                                         # for; host calibration), the target's default variant, the
#                                         # systolic MXU
#   TARGET=ypcb-00338:m4 ./run_vivado.sh  # the target's variant m4 (4 MXU columns at E 4, 64K TMEM words;
#                                         # m8: 8 at E 8, 128K): targets/ypcb-00338.toml [variants]
#   MCOLS=2 ./run_vivado.sh               # 2 MXU columns (over the target's)
#   VPU_CL=4 ./run_vivado.sh              # 4 VPU lanes with exp2/recip/rsqrt (faster softmax)
#   MXU_E=2 ./run_vivado.sh               # 2 MXU epilogue lanes (the target's: MCOLS, full rate)
#   LANES=16 ./run_vivado.sh              # 16 VPU lanes / TMEM banks
#   TMEM_IMPL=1 ./run_vivado.sh           # the banked TMEM (three copies, docs/tmem_pump.md; 0 the default)
#   TMEM_IMPL=1 TMEM_RQ=1 ./run_vivado.sh # the same with request-priority bank addresses
#   TMEM_IMPL=3 ./run_vivado.sh           # one copy (docs/tmem_nocopy.md): 2 RAMs a bank pumped at 2x on
#                                         # the MMCM's core_clk2x (core_clk's divider even: 133.33, 100)
#   TMEM_IMPL=3 TMEM_X=NSB=1 ./run_vivado.sh  # its knobs, NAME=value,... (vivado/tmem_knobs.tcl)
#   CORE_MHZ=100 ./run_vivado.sh          # another core clock (default 133.33, the production image's)
#   FAST=1 ./run_vivado.sh                # development build: default strategies, no retiming or post-route
#                                         # phys_opt (its default CORE_MHZ is 100); about half the time
#   PCIE_GEN=1 ./run_vivado.sh            # PCIe Gen1 x8 (xdma_aclk 125 MHz); the default is 2, Gen2 x8
#                                         # (250 MHz; vivado/bd_native.tcl, docs/offload.md section 6)
#   TRACE_DEPTH=0 ./run_vivado.sh         # no hardware trace buffer (CAPS trace bit 0; the default is the
#                                         # target's 16384 records: 32 RAMB36, docs/infra_hunt.md)
#   COLL_GATHER=1 ./run_vivado.sh         # the collective unit with GATHER (CAPS2 bit0 = 1). The default
#                                         # since release r2 is 0, BAR alone: the card has one slice, where
#                                         # no compiled program has a GATHER and the host refuses one, and
#                                         # the GATHER unit's read select and write-bank mask fed the MXU's
#                                         # TMEM grant (r2m4: -0.261 ns, 12% of core_clk's TNS;
#                                         # docs/infra_hunt.md 5)
#   TMEM_WORDS=131072 ./run_vivado.sh     # a 128K-word TMEM (the target's variant's; in CFG_HASH)
#   VIVADO_DOCKER=image ./run_vivado.sh   # Docker (e.g. Apple Silicon with Rosetta), see docs/board.md
#   VIVADO_AS_USER=1                      # Docker on Linux: run as the calling user (see run())
#   STEP=impl ./run_vivado.sh             # rerun implementation only (keeps project and synthesis)
#   IMPL_STRATEGY=Performance_Explore     # a stronger implementation strategy (with bit or impl)
#   The xc7k480t needs a paid or 30-day evaluation license, node-locked to a MAC address. In Docker
#   set VIVADO_MAC (the MAC the license was issued for) and XILINXD_LICENSE_FILE (path to the .lic).
#
# Output: build/vivado/otpu.bit, build/vivado/otpu.mcs, build/vivado/reports/.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"
speed="${1:-1066}"
out="${OUT_DIR:-$root/build/vivado}"
jobs="${JOBS:-8}"
# The target (targets/ypcb-00338.toml through opentpu/target.py): the defaults of the knobs below,
# the platform id and the arch hash the CFG_* registers report. Without a working python3 the
# built-in defaults stay (the same values) and CFG_HASH reads 0.
target="${TARGET:-ypcb-00338}"
if tenv="$(PYTHONPATH="$root" python3 -m opentpu.target env "$target" 2>/dev/null)"; then
  eval "$tenv"
else
  echo "run_vivado: python3 -m opentpu.target env $target failed: built-in defaults, CFG_HASH 0" >&2
fi
mcols="${MCOLS:-${T_MCOLS:-4}}"     # MXU columns (the host reads them from the VERSION register)
vpu_cl="${VPU_CL:-${T_VPU_CL:-2}}"  # VPU lanes with the composite functions (timing only)
lanes="${LANES:-${T_LANES:-8}}"     # VPU lanes / TMEM banks (likewise in VERSION)
act_rows="${ACT_ROWS:-${T_ACT_ROWS:-$mcols}}"  # ACT RAM rows (> MCOLS: MM replay; the ACT_ROWS register)
mxu_e="${MXU_E:-${T_MXU_E:-4}}"   # MXU epilogue lanes (timing only; 0: MCOLS / 2)
dstep="${DSTEP:-${T_DSTEP:-1}}"     # 0: no DSTEP datapath in the DMA (CAPS bit6 = 0)
# otpu_seq WAITC (EXPERIMENTAL, docs/seq_vpu.md; CFG_UARCH bit24, in CFG_HASH when on): the
# target's [arch] SEQ_WC, or SEQ_WC=1. The host writes each program's wait entries
seq_wc="${SEQ_WC:-${T_SEQ_WC:-1}}"
tmem_impl="${TMEM_IMPL:-${T_TMEM_IMPL:-3}}"   # TMEM: 0 a copy per read port, 1 banked, 3 one copy (timing only)
tmem_rq="${TMEM_RQ:-0}"                      # TMEM_IMPL 1: request-priority bank addresses
tmem_x="${TMEM_X:-}"                         # TMEM_IMPL 3: the one copy's knobs (tmem_knobs.tcl)
if [[ -z "${TMEM_X:-}" && "$tmem_impl" == 3 ]]; then   # unset: the target's (T_TMEM_NSB, ...)
  for k in NSB SBP SBH RPL TDP GST PMP PRK SDM; do
    v="T_TMEM_$k"
    if [[ -n "${!v:-}" ]]; then tmem_x="${tmem_x:+$tmem_x,}$k=${!v}"; fi
  done
fi
mxu="$([[ "${T_MXU_IMPL:-2}" == 0 ]] && echo tree || echo systolic)"   # the MXU's dot product
core_mhz="${CORE_MHZ:-$([[ "${FAST:-0}" == 1 ]] && echo 100 || echo "${T_CORE_MHZ:-133.33}")}"  # accelerator clock (FAST=1: 100)
platform="${T_PLATFORM:-1}"
# TMEM words (the CFG_TMEM register, in CFG_HASH; read by vivado/create_project.tcl): the target's
# [arch] TMEM_WORDS (its variant's: m4 65536, m8 131072), or TMEM_WORDS=
export TMEM_WORDS="${TMEM_WORDS:-${T_TMEM_WORDS:-65536}}"
# CFG_HASH: the arch hash of what this build is (the knobs applied to the target's arch)
arch_hash="$(PYTHONPATH="$root" python3 -m opentpu.target hash "$target" MCOLS="$mcols" \
  LANES="$lanes" ACT_ROWS="$act_rows" TMEM_WORDS="$TMEM_WORDS" DSTEP="$dstep" SEQ_WC="$seq_wc" \
  2>/dev/null || echo 0)"
# BUILD_ID register: the git commit's first 8 hex digits, taken here (Vivado may run in Docker)
build_id="${BUILD_ID:-$(git -C "$root" rev-parse HEAD 2>/dev/null | cut -c1-8)}"
build_id="${build_id:-0}"

if [[ "$speed" != 1066 ]]; then
  echo "DDR3-$speed: the LiteDRAM core is generated for DDR3-1066 (tools/litedram/gen_core.py)" >&2
  exit 2
fi

run() {  # run a Vivado Tcl script with arguments
  local script="$1"; shift
  if [[ -n "${VIVADO_DOCKER:-}" ]]; then
    # VIVADO_AS_USER=1: run as the calling user (a Linux host whose files Docker does not remap:
    # the .lic is often mode 600), with the image's entrypoint bypassed
    local as_user=() shell=(bash)
    if [[ -n "${VIVADO_AS_USER:-}" ]]; then
      as_user=(--user "$(id -u):$(id -g)" -e "HOME=$out/.home" --entrypoint bash); shell=()
      mkdir -p "$out/.home"
    fi
    docker run --rm \
      -v "$root:$root" -w "$root" \
      ${VIVADO_MAC:+--mac-address "$VIVADO_MAC"} \
      ${XILINXD_LICENSE_FILE:+-v "$XILINXD_LICENSE_FILE:$XILINXD_LICENSE_FILE:ro" -e "XILINXD_LICENSE_FILE=$XILINXD_LICENSE_FILE"} \
      ${VIVADO_MOUNT:+-v "$VIVADO_MOUNT"} \
      ${as_user[@]+"${as_user[@]}"} \
      "$VIVADO_DOCKER" \
      ${shell[@]+"${shell[@]}"} -lc "source ${VIVADO_SETTINGS:-/tools/Xilinx/Vivado/2026.1/settings64.sh} && \
                vivado -mode batch -nojournal -log $out/$(basename "$script" .tcl).log \
                -source $script -tclargs $*"
  else
    vivado -mode batch -nojournal -log "$out/$(basename "$script" .tcl).log" \
      -source "$script" -tclargs "$@"
  fi
}

mkdir -p "$out"
# the IP synthesis cache shared by this host's builds (create_project.tcl); OTPU_IP_CACHE= turns it off
export OTPU_IP_CACHE="${OTPU_IP_CACHE-$HOME/.cache/otpu-vivado-ip}"
export OTPU_FAST="${FAST:-0}"
export OTPU_SEQ_WC_GEN="$seq_wc"
export PCIE_GEN="${PCIE_GEN:-2}"    # read by vivado/bd_native.tcl and userclk1_reroute.tcl
export TRACE_DEPTH="${TRACE_DEPTH:-${T_TRACE_DEPTH:-16384}}"   # read by vivado/create_project.tcl
export COLL_GATHER="${COLL_GATHER:-0}"                          # read by vivado/create_project.tcl
if [[ "${STEP:-}" == impl ]]; then
  run "$here/vivado/build.tcl" "$out" "$jobs" impl "${IMPL_STRATEGY:-}"
else
  run "$here/vivado/create_project.tcl" "$speed" "$out" "$mcols" "$core_mhz" "$build_id" "$vpu_cl" "$lanes" "$act_rows" "$dstep" "$mxu" "$platform" "$arch_hash" "$mxu_e" "$tmem_impl" "$tmem_rq" "$tmem_x"
  run "$here/vivado/build.tcl" "$out" "$jobs" full "${IMPL_STRATEGY:-}"
fi
echo "done: $out/otpu.bit"
