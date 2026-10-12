#!/bin/bash
# EXPERIMENTAL (docs/tmem_pump.md): otpu_tmem_pump against otpu_tmem in Verilator (two clocks,
# --timing), for the configurations of the experiment. Usage, from the tree's root:
#   rtl/exp/tmem_pump/run_tb.sh <workdir> [cycles]
#   i0_4k, i0_64k  IMPL 0: the card's TMEM (6 copies) against 3 pumped copies (6 slots)
#   i0_byp         the same with the pumped TMEM's (redundant) bypass kept
#   i1n3           IMPL 1, NCP 3: the ASIC's banked TMEM (3 copies, the slice's CPY) against 2
#                  pumped copies (3 of their 4 slots)
#   i1n4           IMPL 1, NCP 4: 4 banked copies against 2 pumped copies (4 slots)
#   i1n3q, i1n4q   the same with request-priority addresses (otpu_tmem RQ, the slice's gnt_q)
#   k3n5, k4n7     one copy pumped at 3x / 4x (K, reads a cycle later): 5 / 7 banked slots against
#                  otpu_tmem's 5 / 7 banked copies a cycle earlier; k3n5q, k4n7q with RQ
# WORDS 4096 (a hot window of 256 words) and 65536 (the card's size, a window of 1024).
set -e
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
W=${1:?workdir}
N=${2:-200000}
mkdir -p "$W"
SRC="$ROOT/rtl/tech/generic/otpu_tech_ram.sv $ROOT/rtl/mem/otpu_tmem.sv $ROOT/rtl/exp/tmem_pump/otpu_tmem_pump.sv $ROOT/rtl/exp/tmem_pump/tb_tmem_pump.sv"
one() {   # name -G...
  local n=$1; shift
  rm -rf "${W:?}/$n"
  verilator --binary --timing -Wno-fatal -Wno-WIDTHEXPAND -Wno-WIDTHTRUNC -Wno-UNOPTFLAT \
    -Wno-UNUSEDSIGNAL -Wno-UNUSEDPARAM -Wno-MULTIDRIVEN -Wno-TIMESCALEMOD --build-jobs 2 --x-assign 0 --x-initial 0 -O2 \
    --top-module tb_tmem_pump -GCYCLES="$N" "$@" --Mdir "$W/$n" -o sim $SRC > "$W/$n.build.log" 2>&1 \
    || { echo "BUILD FAILED $n (see $W/$n.build.log)"; tail -20 "$W/$n.build.log"; return 1; }
  "$W/$n/sim" > "$W/$n.log" 2>&1 || true
  grep -E "TB_RESULT|MISMATCH|Fatal|fatal|Error" "$W/$n.log" | head -25
}
#                    COLL Q2  Q3  VB  VA  Q   DMA MXU
CPY3="24'b010001010001000000010000"   # the slice's (otpu_slice CPY): 2 1 2 1 0 0 2 0
CPY4="24'b011010001001000011010000"   # 4 slots: MXU 0, DMA 2, Q 3, VA 0, VB 1, Q3 1, Q2 2, COLL 3
one i0_4k  -GIMPL=0 -GWORDS=4096 -GSEED=1
one i0_byp -GIMPL=0 -GWORDS=4096 -GSEED=5 -GBYP=1
one i0_64k -GIMPL=0 -GWORDS=65536 -GHOT=1024 -GSEED=2
one i1n3   -GIMPL=1 -GNCP=3 "-GCPYV=$CPY3" -GWORDS=4096 -GSEED=3
one i1n4   -GIMPL=1 -GNCP=4 "-GCPYV=$CPY4" -GWORDS=4096 -GSEED=4
one i1n3q  -GIMPL=1 -GNCP=3 "-GCPYV=$CPY3" -GWORDS=4096 -GSEED=6 -GRQ=1
one i1n4q  -GIMPL=1 -GNCP=4 "-GCPYV=$CPY4" -GWORDS=4096 -GSEED=7 -GRQ=1
CPY5="24'b001100001100011010001000"   # 5 slots: MXU 0, DMA 1, Q 2, VA 3, VB 4, Q3 1, Q2 4, COLL 1
CPY7="24'b001110101100011010001000"   # 7 slots: MXU 0, DMA 1, Q 2, VA 3, VB 4, Q3 5, Q2 6, COLL 1
one k3n5   -GIMPL=1 -GNCP=5 "-GCPYV=$CPY5" -GWORDS=4096 -GSEED=8 -GK=3
one k4n7   -GIMPL=1 -GNCP=7 "-GCPYV=$CPY7" -GWORDS=4096 -GSEED=9 -GK=4
one k3n5q  -GIMPL=1 -GNCP=5 "-GCPYV=$CPY5" -GWORDS=4096 -GSEED=10 -GK=3 -GRQ=1
one k4n7q  -GIMPL=1 -GNCP=7 "-GCPYV=$CPY7" -GWORDS=65536 -GHOT=1024 -GSEED=11 -GK=4 -GRQ=1
