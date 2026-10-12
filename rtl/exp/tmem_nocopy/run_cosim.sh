#!/bin/bash
# EXPERIMENTAL (docs/tmem_nocopy.md): co-simulations of TMEM candidates on a workload list,
# one at a time. Every run is the board's configuration (PAIR, DSTEP, STREAM; LiteDRAM's
# controller at DDR3-1066, the core at 133.33 MHz).
#   workloads: lines "TAG perf_qwen-arguments", "TAG moe: moe_prefill_cosim-arguments" or
#     "TAG zero: vpu_hack.py cosim-arguments" (vpu_pump's lists' format)
#   candidates: lines "NAME K=V ..." (the run's environment; OTPU_SIM_PLUSARGS takes the
#     otpu_tmem_model / handshake plusargs, comma-separated). TRACE=1 in a candidate's line
#     writes the TMEM trace (+tmtrace) of each of its runs to <outdir>/<tag>.tt.
# Results: <outdir>/<workload>__<candidate>.log (ends with rc=N and the wall time), the TMEMSTAT
# and TMMODEL lines in .stat. A run whose log ends in rc=0 is skipped. Usage, from anywhere:
#   rtl/exp/tmem_nocopy/run_cosim.sh <outdir> <workloads> <candidates> NAME...
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
O=${1:?outdir}; W=${2:?workloads}; C=${3:?candidates}; shift 3
mkdir -p "$O"
O=$(cd "$O" && pwd)
W=$(cd "$(dirname "$W")" && pwd)/$(basename "$W")
C=$(cd "$(dirname "$C")" && pwd)/$(basename "$C")
export OTPU_PAIR=1 OTPU_DSTEP=1 OTPU_STREAM=1 PYTHONPATH=$ROOT
for name in "$@"; do
  cand=$(grep -E "^$name( |$)" "$C" | head -1)
  [ -n "$cand" ] || { echo "no candidate $name in $C" >> "$O/progress.txt"; continue; }
  read -r -a cenv <<< "${cand#"$name"}"
  grep -v '^#' "$W" | grep -v '^ *$' | while read -r wl args; do
    tag=${wl}__$name
    grep -q "^rc=0" "$O/$tag.log" 2>/dev/null && continue
    rm -f "$O/$tag.stat" "$O/$tag.tt"
    env=(OTPU_TMEMSTAT_LOG="$O/$tag.stat")
    pa=""
    for kv in "${cenv[@]}"; do
      case $kv in
        TRACE=1) pa="$pa,+tmtrace=$O/$tag.tt" ;;
        OTPU_SIM_PLUSARGS=*) pa="$pa,${kv#OTPU_SIM_PLUSARGS=}" ;;
        *) env+=("$kv") ;;
      esac
    done
    [ -n "$pa" ] && env+=(OTPU_SIM_PLUSARGS="${pa#,}")
    t0=$(date +%s)
    cd "$ROOT" || exit 1
    read -r -a av <<< "$args"
    echo "# $(date '+%F %T') ${env[*]} :: ${av[*]}" > "$O/$tag.log"
    if [ "${av[0]}" = "moe:" ]; then
      env "${env[@]}" python3 tools/moe_prefill_cosim.py "${av[@]:1}" >> "$O/$tag.log" 2>&1
    elif [ "${av[0]}" = "zero:" ]; then
      env "${env[@]}" python3 tools/vpu_hack.py cosim "${av[@]:1}" >> "$O/$tag.log" 2>&1
    else
      env "${env[@]}" python3 tools/perf_qwen.py --ddr 1066 --mhz 133.33 --ldc "${av[@]}" \
        >> "$O/$tag.log" 2>&1
    fi
    rc=$?
    echo "rc=$rc $(( $(date +%s) - t0 ))s" >> "$O/$tag.log"
    echo "$(date '+%T') $tag rc=$rc $(( $(date +%s) - t0 ))s" >> "$O/progress.txt"
  done
done
echo "EXIT $* $(date '+%F %T')" >> "$O/progress.txt"
