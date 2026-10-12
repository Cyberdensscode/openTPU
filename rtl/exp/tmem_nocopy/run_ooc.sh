#!/bin/bash
# EXPERIMENTAL (docs/tmem_nocopy.md): the one-copy TMEM's Vivado runs, one at a time (a Vivado
# host's slot). Usage, from the tree:
#   rtl/exp/tmem_nocopy/run_ooc.sh <outdir> [step ...]
# Steps (tmem_nc_top: the TMEM and the slice's arbiter for it, registered around):
#   k7-v0      otpu_tmem IMPL 0 (6 copies) + the board's arbiter, xc7k480t-2 at 133.33 MHz
#   k7-r8i     IMPL 3, 8 RAMs a bank, interleaved 8-word groups, at 133.33 MHz
#   k7-r16i    IMPL 3, 16 RAMs a bank (RAMB18), at 133.33 MHz
#   k7-r16i-128k  IMPL 3, 16 RAMs a bank at 128K words (RAMB36), at 133.33 MHz
#   ku-v0, ku-r8i, ku-r16i   the same on xcku5p-2 (the UltraScale+ stand-in) at 200 MHz
#   k7-r8iR    IMPL 3 with replays (RPL) and the board's arbiter, 8 RAMs a bank, at 133.33 MHz
#   k7-r16hR   the same, 16 RAMs a bank, the sub-bank hashed (SBH 2)
#   k7-r16hR-128k  k7-r16hR at 128K words
#   ku-r8iR, ku-r16hR   the replays on xcku5p-2 at 200 MHz
#   k7-r8xR    the replays on true dual-port RAMs (TMEM_TDP, VAR 5), 8 RAMs a bank; -128k, -256k
#   k7-r8xhR   the same, hashed (SBH 2); ku-r8xR on xcku5p-2 at 200 MHz
#   k7-r8iR-256k, k7-r16hR-256k   the replays at 256K words
#   k7-r4xR, k7-r2xR   the replays on true dual-port RAMs, 4 / 2 RAMs a bank
#   k7-p3      P3 (the copies pumped at 2x, VAR 6) with the board's arbiter, the clocks from an
#              MMCM (133.33 / 266.67 MHz); k7-v0m the card's TMEM on the same MMCM clocking
#   s-<VAR>-<ABL>  synthesis + opt only (LUTs), xc7k480t-2: VAR v0, p3, r8iR, r8xR, r4xR; ABL
#              full, DSEL, CMP, RPL, AMUX (otpu_tmem OTPU_TMEM_ABL_*: the LUT breakdown)
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
O=${1:?outdir}; shift
mkdir -p "$O"
viv() {   # name args...
  local n=$1; shift
  echo "START $n $(date '+%F %T')" >> "$O/progress.txt"
  local t0=$(date +%s)
  (cd "$O" && timeout 7200 vivado -mode batch -nojournal -log "$O/$n.log" \
     -source "$ROOT/rtl/exp/tmem_nocopy/ooc.tcl" -tclargs "$O/$n" "$@" > "$O/$n.out" 2>&1)
  echo "DONE $n rc=$? $(( $(date +%s) - t0 ))s" >> "$O/progress.txt"
  grep -h "^OTPU_" "$O/$n.out" | sed "s/^/$n /" >> "$O/progress.txt"
}
K7=xc7k480tffg1156-2
KU=xcku5p-ffvb676-2-e
for s in "$@"; do
  case $s in
    k7-v0)    viv $s 0 65536 8 3 $K7 7.5 ;;
    k7-r8i)   viv $s 3 65536 8 3 $K7 7.5 ;;
    k7-r16i)  viv $s 3 65536 16 3 $K7 7.5 ;;
    k7-r16i-128k) viv $s 3 131072 16 3 $K7 7.5 ;;
    k7-r8iR)  viv $s 4 65536 8 3 $K7 7.5 0 ;;
    k7-r16hR) viv $s 4 65536 16 3 $K7 7.5 2 ;;
    k7-r16hR-128k) viv $s 4 131072 16 3 $K7 7.5 2 ;;
    ku-r8iR)  viv $s 4 65536 8 3 $KU 5.0 0 ;;
    k7-r8xR)  viv $s 5 65536 8 3 $K7 7.5 0 ;;
    k7-r8xhR) viv $s 5 65536 8 3 $K7 7.5 2 ;;
    k7-r8xR-128k) viv $s 5 131072 8 3 $K7 7.5 0 ;;
    k7-r8xR-256k) viv $s 5 262144 8 3 $K7 7.5 0 ;;
    ku-r8xR)  viv $s 5 65536 8 3 $KU 5.0 0 ;;
    k7-r8iR-256k) viv $s 4 262144 8 3 $K7 7.5 0 ;;
    k7-r16hR-256k) viv $s 4 262144 16 3 $K7 7.5 2 ;;
    ku-r16hR) viv $s 4 65536 16 3 $KU 5.0 2 ;;
    ku-v0)    viv $s 0 65536 8 3 $KU 5.0 ;;
    ku-r8i)   viv $s 3 65536 8 3 $KU 5.0 ;;
    ku-r16i)  viv $s 3 65536 16 3 $KU 5.0 ;;
    k7-r4xR)  viv $s 5 65536 4 3 $K7 7.5 0 ;;
    k7-r2xR)  viv $s 5 65536 2 3 $K7 7.5 0 ;;
    k7-p3)    viv $s 6 65536 8 3 $K7 7.5 0 1 0 ;;
    k7-v0m)   viv $s 0 65536 8 3 $K7 7.5 0 1 0 ;;
    s-*)
      v=${s#s-}; a=${v#*-}; v=${v%%-*}; [ "$a" = full ] && a=-
      case $v in
        v0)   viv $s 0 65536 8 3 $K7 7.5 0 0 0 $a 1 ;;
        p3)   viv $s 6 65536 8 3 $K7 7.5 0 1 0 $a 1 ;;
        r8iR) viv $s 4 65536 8 3 $K7 7.5 0 0 0 $a 1 ;;
        r8xR) viv $s 5 65536 8 3 $K7 7.5 0 0 0 $a 1 ;;
        r4xR) viv $s 5 65536 4 3 $K7 7.5 0 0 0 $a 1 ;;
        r2xR) viv $s 5 65536 2 3 $K7 7.5 0 0 0 $a 1 ;;
        *) echo "unknown step $s" >> "$O/progress.txt" ;;
      esac ;;
    *) echo "unknown step $s" >> "$O/progress.txt" ;;
  esac
done
echo "EXIT $(date '+%F %T')" >> "$O/progress.txt"
