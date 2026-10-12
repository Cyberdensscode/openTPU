#!/bin/bash
# EXPERIMENTAL (docs/tmem_pump.md): the experiment's Vivado runs, one at a time (omarchy only,
# through ~/otpu-build/slot.sh). Usage, from the tree:
#   rtl/exp/tmem_pump/run_ooc.sh <outdir> [step ...]
# Steps (xc7k480tffg1156-2 at 133.33 MHz, the pumped copies at 266.67):
#   k7-v0  otpu_tmem IMPL 0 (6 copies)          k7-v1  otpu_tmem IMPL 1 (3 copies)
#   k7-v2  pumped, 3 copies (IMPL 0's slots)    k7-v3  pumped, 2 copies (4 banked slots)
#   k7-v4  pumped, 1 copy (2 banked slots)      k7-v2x2  k7-v2 at 2x the words (128K)
#   k7-v0x2  otpu_tmem IMPL 0 at 128K words (the BRAM count it would take)
#   k7-v2-150  k7-v2 with the core at 150 MHz, the pumped copies at 300 (the margin)
# xcku5p-ffvb676-2-e (the UltraScale+ stand-in), the core at 250 MHz, the pumped copies at 500:
#   ku-v2b  pumped 3 copies, block RAM          ku-v2u  the same in UltraRAM
#   ku-v1b  otpu_tmem IMPL 1 (3 copies of block RAM at 250 MHz)
#   ku-v2u300  pumped UltraRAM at 300 / 600 MHz
# The boards' own parts at 250 / 500 MHz (omarchy has them; opentpu has no UltraScale+ part):
#   vu9-v2b, vu9-v2u    pumped 3 copies, block RAM / UltraRAM, xcvu9p-fsgd2104-2L-e (BCU1525)
#   vu47-v2b, vu47-v2u  the same on xcvu47p-fsvh2892-2-e (AWS F2)
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
O=${1:?outdir}; shift
mkdir -p "$O"
viv() {   # name args...
  local n=$1; shift
  echo "START $n $(date '+%F %T')" >> "$O/progress.txt"
  local t0=$(date +%s)
  (cd "$O" && timeout 7200 vivado -mode batch -nojournal -log "$O/$n.log" \
     -source "$ROOT/rtl/exp/tmem_pump/ooc.tcl" -tclargs "$O/$n" "$@" > "$O/$n.out" 2>&1)
  echo "DONE $n rc=$? $(( $(date +%s) - t0 ))s" >> "$O/progress.txt"
  grep -h "^OTPU_" "$O/$n.out" | sed "s/^/$n /" >> "$O/progress.txt"
}
K7="xc7k480tffg1156-2 0"
KU="xcku5p-ffvb676-2-e 1"
VU9="xcvu9p-fsgd2104-2L-e 1"
VU47="xcvu47p-fsvh2892-2-e 1"
for s in "$@"; do
  case $s in
    k7-v0)   viv $s 0 65536  $K7 block 7.5 8.0 8.0 4 ;;
    k7-v1)   viv $s 1 65536  $K7 block 7.5 8.0 8.0 4 ;;
    k7-v2)   viv $s 2 65536  $K7 block 7.5 8.0 8.0 4 ;;
    k7-v3)   viv $s 3 65536  $K7 block 7.5 8.0 8.0 4 ;;
    k7-v4)   viv $s 4 65536  $K7 block 7.5 8.0 8.0 4 ;;
    k7-v2x2) viv $s 2 131072 $K7 block 7.5 8.0 8.0 4 ;;
    k7-v0x2) viv $s 0 131072 $K7 block 7.5 8.0 8.0 4 ;;
    k7-v2-150) viv $s 2 65536 $K7 block 7.5 9.0 8.0 4 ;;
    ku-v2b)  viv $s 2 65536  $KU block 4.0 4.0 4.0 2 ;;
    ku-v2u)  viv $s 2 65536  $KU ultra 4.0 4.0 4.0 2 ;;
    ku-v1b)  viv $s 1 65536  $KU block 4.0 4.0 4.0 2 ;;
    ku-v2u300) viv $s 2 65536 $KU ultra 3.333 4.0 4.0 2 ;;
    vu9-v2b)  viv $s 2 65536  $VU9 block 4.0 4.0 4.0 2 ;;
    vu9-v2u)  viv $s 2 65536  $VU9 ultra 4.0 4.0 4.0 2 ;;
    vu47-v2b) viv $s 2 65536  $VU47 block 4.0 4.0 4.0 2 ;;
    vu47-v2u) viv $s 2 65536  $VU47 ultra 4.0 4.0 4.0 2 ;;
    *) echo "unknown step $s" >> "$O/progress.txt" ;;
  esac
done
echo "EXIT $(date '+%F %T')" >> "$O/progress.txt"
