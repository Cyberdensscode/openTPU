#!/bin/bash
# EXPERIMENTAL (docs/seq_vpu.md): co-simulations of sequencer / VPU candidates on a workload list,
# P at a time. Every run is the board's configuration (PAIR, DSTEP, STREAM; LiteDRAM's controller
# at DDR3-1066, the core at 133.33 MHz).
#   workloads: lines "TAG perf_qwen-arguments", "TAG moe: moe_prefill_cosim-arguments" or
#     "TAG zero: vpu_hack.py cosim-arguments" (vpu_pump's lists' format)
#   candidates: lines "NAME K=V ..." (the run's environment). OTPU_SIM_PLUSARGS=a,b adds
#     simulator plusargs (otpu_seq's +hz_*, otpu_vpu's +vt_*); HZLOG=1 writes otpu_seq's hazard
#     statistics to <outdir>/<tag>.<n>.hz; KOUT=1 writes the wait-count oracle to
#     <outdir>/<wl>.<n>.k; WC=1 runs on the wait counts of <outdir>/<wl>.<n>.k (a KOUT
#     candidate's, run first); KOUT=x / WC=x: <outdir>/<wl>.x.<n>.k; SYNC=1 / SYNC=x: the same
#     file as otpu_seq WAITC's wait counts (OTPU_SEQ_WC=1); SYNCFROM=w: workload w's file (the
#     same program at other run-time arguments). <n> counts a tool's simulations (the
#     MoE tool runs one per prefix).
# Results: <outdir>/<workload>__<candidate>.log (ends with rc=N and the wall time). A run whose
# log ends in rc=0 is skipped. Usage, from anywhere:
#   rtl/exp/seq_vpu/run_cosim.sh <outdir> <P> <workloads> <candidates> NAME...
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
O=${1:?outdir}; P=${2:?parallel}; W=${3:?workloads}; C=${4:?candidates}; shift 4
mkdir -p "$O"
O=$(cd "$O" && pwd)
W=$(cd "$(dirname "$W")" && pwd)/$(basename "$W")
C=$(cd "$(dirname "$C")" && pwd)/$(basename "$C")
export OTPU_PAIR=1 OTPU_DSTEP=1 OTPU_STREAM=1 PYTHONPATH=$ROOT ROOT O C
one() {   # workload candidate args...
  local wl=$1 name=$2; shift 2
  local tag=${wl}__$name
  grep -q "^rc=0" "$O/$tag.log" 2>/dev/null && return 0
  local cand
  cand=$(grep -E "^$name( |$)" "$C" | head -1)
  [ -n "$cand" ] || { echo "no candidate $name in $C" >> "$O/progress.txt"; return 0; }
  local -a cenv env
  read -r -a cenv <<< "${cand#"$name"}"
  local pa=""
  for kv in "${cenv[@]}"; do
    case $kv in
      HZLOG=1) pa="$pa,+hz_log=$O/$tag.%n.hz" ;;
      KOUT=1) pa="$pa,+hz_kout=$O/$wl.%n.k" ;;
      KOUT=*) pa="$pa,+hz_kout=$O/$wl.${kv#KOUT=}.%n.k" ;;
      WC=1) pa="$pa,+hz_wc=$O/$wl.%n.k" ;;
      WC=*) pa="$pa,+hz_wc=$O/$wl.${kv#WC=}.%n.k" ;;
      SYNC=1) pa="$pa,+hz_sync=$O/$wl.%n.k" ;;
      SYNCFROM=*) pa="$pa,+hz_sync=$O/${kv#SYNCFROM=}.%n.k" ;;
      SYNC=*) pa="$pa,+hz_sync=$O/$wl.${kv#SYNC=}.%n.k" ;;
      OTPU_SIM_PLUSARGS=*) pa="$pa,${kv#OTPU_SIM_PLUSARGS=}" ;;
      *) env+=("$kv") ;;
    esac
  done
  [ -n "$pa" ] && env+=(OTPU_SIM_PLUSARGS="${pa#,}")
  local t0
  t0=$(date +%s)
  cd "$ROOT" || return 1
  local -a av
  read -r -a av <<< "$*"
  echo "# $(date '+%F %T') ${env[*]} :: ${av[*]}" > "$O/$tag.log"
  if [ "${av[0]}" = "moe:" ]; then
    env "${env[@]}" python3 tools/moe_prefill_cosim.py "${av[@]:1}" >> "$O/$tag.log" 2>&1
  elif [ "${av[0]}" = "zero:" ]; then
    env "${env[@]}" python3 tools/vpu_hack.py cosim "${av[@]:1}" >> "$O/$tag.log" 2>&1
  else
    env "${env[@]}" python3 tools/perf_qwen.py --ddr 1066 --mhz 133.33 --ldc "${av[@]}" \
      >> "$O/$tag.log" 2>&1
  fi
  local rc=$?
  echo "rc=$rc $(( $(date +%s) - t0 ))s" >> "$O/$tag.log"
  echo "$(date '+%T') $tag rc=$rc $(( $(date +%s) - t0 ))s" >> "$O/progress.txt"
}
export -f one
for name in "$@"; do
  grep -v '^#' "$W" | grep -v '^ *$' | while read -r wl args; do echo "$wl $name $args"; done
done | xargs -P "$P" -L 1 bash -c 'one "$@"' _
echo "EXIT $* $(date '+%F %T')" >> "$O/progress.txt"
