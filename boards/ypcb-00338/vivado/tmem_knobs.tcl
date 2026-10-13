# The TMEM's knobs of a board build, sourced by create_project.tcl (the YPCB-00338's and the
# BCU1525's) after it has TMEM_IMPL and TMEM_X:
#   TMEM_IMPL  0 a copy per read port, 1 banked (three copies, docs/tmem_pump.md), 3 one copy
#              (docs/tmem_nocopy.md)
#   TMEM_X     TMEM_IMPL 3: the one copy's knobs, comma-separated NAME=value of NSB SBP SBH RPL TDP
#              GST PMP PRK SDM (otpu_slice's TMEM_<NAME>), over the card's pick: 2 RAMs a bank
#              (NSB 2, SBP 3), replays (RPL), true dual-port RAMs (TDP) pumped at 2x on core_clk2x
#              (PMP 1), QUANT 3 and COLL the DMA's guests (GST), the default ranks and RAM sides
#              (PRK 0, SDM -1). "" keeps the pick; e.g. NSB=1 one RAM a bank.
# Sets TMEM_GEN, the generics to append (" TMEM_NSB=2 ..."; "" unless TMEM_IMPL 3, so the other
# builds' netlists stay as they were), and CLK2X: 1 with the pumped RAMs (PMP 1), or with CLK2X=1
# in the environment; the block design then makes core_clk2x (bd_native.tcl / bd.tcl) and
# constraints/otpu_clk2x.tcl times it.
if {$TMEM_IMPL ni {0 1 3}} { error "TMEM_IMPL $TMEM_IMPL: 0, 1 or 3" }
array unset TX
array set TX {NSB 8 SBP 3 SBH 0 RPL 0 TDP 0 GST 0 PMP 0 PRK 0 SDM -1}
if {$TMEM_IMPL == 3} { array set TX {NSB 2 RPL 1 TDP 1 GST 1 PMP 1} }
if {$TMEM_X ne "" && $TMEM_IMPL != 3} { error "TMEM_X '$TMEM_X': with TMEM_IMPL 3 only" }
foreach kv [split $TMEM_X ,] {
  if {$kv eq ""} { continue }
  if {![regexp {^([A-Z]+)=(-?[0-9]+)$} $kv -> k v] || ![info exists TX($k)]} {
    error "TMEM_X: '$kv' is not NAME=integer with NAME one of [lsort [array names TX]]"
  }
  set TX($k) $v
}
if {$TX(PMP) ni {0 1}} { error "TMEM_X PMP $TX(PMP): 0 or 1 on a board (2 is the simulation's model)" }
if {$TX(PMP) == 1 && !$TX(TDP)} { error "TMEM_X PMP 1 needs TDP 1 (the pumped RAMs are true dual-port)" }
set TMEM_GEN ""
if {$TMEM_IMPL == 3} {
  foreach k {NSB SBP SBH PMP PRK} { append TMEM_GEN " TMEM_$k=$TX($k)" }
  foreach k {RPL TDP GST} { append TMEM_GEN " TMEM_$k=1'b$TX($k)" }
  if {$TX(SDM) >= 0} { append TMEM_GEN " TMEM_SDM=$TX(SDM)" }
}
set CLK2X [expr {$TX(PMP) == 1 || ([info exists ::env(CLK2X)] && $::env(CLK2X) eq "1") ? 1 : 0}]
puts "TMEM_IMPL $TMEM_IMPL:$TMEM_GEN; CLK2X $CLK2X"
