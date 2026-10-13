#!/bin/bash
# EXPERIMENTAL (docs/seq_vpu.md): the study's Vivado out-of-context runs, one after another
# (through a host's ~/otpu-build/slot.sh). Usage, from the tree:
#   rtl/exp/seq_vpu/run_ooc.sh <outdir> [step ...]
# Steps (on xc7k480tffg1156-2 at 7.5 ns, the card's parameters; OTPU_PART / OTPU_PERIOD: another
# part / period in ns):
#   seq          otpu_seq as built, placed and routed
#   seq_s        otpu_seq as built, synthesis only (the ablations' reference)
#   seq_nohaz    ablation: no hazard check (footprints, their storage and the compares pruned)
#   seq_nocmp    ablation: the footprints computed, no storage or compares
#   seq_nodram   ablation: no DRAM range compares
#   seq_notmem   ablation: no TMEM range compares
#   seq_win8     otpu_seq with an 8-slot window, synthesis only
#   vpu          otpu_vpu as built (LANES 8, the stream engine, WBUF), placed and routed
#   <name>:<top>:<impl>:<generics>:<defines>:<sources>   any other run (fields as ooc.tcl's)
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
O=${1:?outdir}; shift
mkdir -p "$O"
export PATH=$HOME/.local/bin:$PATH
T=$ROOT/rtl/exp/seq_vpu/ooc.tcl
SEQ=IMEM_WORDS=32768,TMEM_WORDS=65536,SID=0,S=1,D=128,WIN=16,STREAMS=1
ABL=rtl/exp/seq_vpu/abl/otpu_seq.sv
viv() {   # name top impl generics defines [sources]
  local n=$1; shift
  echo "START $n $(date '+%F %T')" >> "$O/progress.txt"
  local t0=$(date +%s)
  (cd "$O" && timeout 7200 vivado -mode batch -nojournal -log "$O/$n.log" -source "$T" \
     -tclargs "$O/$n" "$1" "${OTPU_PERIOD:-7.5}" "$2" "$3" "$4" "${5:-}" > "$O/$n.out" 2>&1)
  echo "DONE $n rc=$? $(( $(date +%s) - t0 ))s" >> "$O/progress.txt"
  grep -h "^OTPU_" "$O/$n.out" | grep -v OTPU_GROUP | sed "s/^/$n /" >> "$O/progress.txt"
}
for s in "$@"; do
  case $s in
    seq)        viv seq otpu_seq 1 $SEQ - ;;
    seq_s)      viv seq_s otpu_seq 0 $SEQ - ;;
    seq_nohaz)  viv seq_nohaz otpu_seq 0 $SEQ ABL_NOHAZ $ABL ;;
    seq_nocmp)  viv seq_nocmp otpu_seq 0 $SEQ ABL_NOCMP $ABL ;;
    seq_nodram) viv seq_nodram otpu_seq 0 $SEQ ABL_NODRAM $ABL ;;
    seq_notmem) viv seq_notmem otpu_seq 0 $SEQ ABL_NOTMEM $ABL ;;
    seq_win8)   viv seq_win8 otpu_seq 0 ${SEQ/WIN=16/WIN=8} - ;;
    vpu)        viv vpu otpu_vpu 1 LANES=8,CL=2,HAS_SE=1,WBUF=1,SID=0 - ;;
    *:*)        IFS=: read -r n top im g d src <<< "$s"; viv "$n" "$top" "$im" "$g" "$d" "$src" ;;
    *) echo "unknown step $s" >> "$O/progress.txt" ;;
  esac
done
echo "EXIT $(date '+%F %T')" >> "$O/progress.txt"
