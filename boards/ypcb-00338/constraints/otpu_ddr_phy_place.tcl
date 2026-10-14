# The memory PHY's output flops beside their I/O (LiteDRAM: every OSERDESE2 / OSERDESE3 under the
# design, both channels of the YPCB-00338, the BCU1525's four). Each flop that drives an OSERDES
# data or tristate input (D1..D8, T1..T4; D[7:0], T) goes in a pblock of the slices nearest its
# I/O: per clock region of the pads, the OTPU_DDR_PHY_COLS slice columns (default 4) closest to
# that region's I/O column, the region's height. The flops' own placement within it is the
# placer's. The pblocks are hard (IS_SOFT false, set here): r2.1's m8 4f8be3d2 reported IS_SOFT 1
# on all six without it, and placed channel 1's address flops u_ld/FDRE_250..253 (in
# otpu_ddr_phy_X0Y6, driving OSERDESE2_114 at OLOGIC_X0Y348, ddram1_a[4]) at SLICE_X32Y348: the
# sys -> writeclocks1_outs0 paths into its D1..D4 met at +0.788 ns with 1.02 ns of route.
#
# Why. r2's m4 candidate (b10245f5, 133.33 MHz) placed channel 0's u_ld/FDRE_51..54 at SLICE_X29Y68,
# 29 columns in from their OSERDES at OLOGIC_X0Y68: the sys clock -> writeclocks0_outs0 paths
# into OSERDESE2_7's D3..D6 met at +0.814 to +0.894 ns with 1.06 ns of route, below production's
# +0.939 (8b33206e: the worst source at SLICE_X0..X1, 0.80 ns). Those paths cross from the sys
# clock to the OSERDES's write clock; their slack is the DDR3 write data's margin. No logic or
# cycle changes; pblocks only (not EXCLUDE_PLACEMENT: other cells may share the slices).
#
# - OTPU_DDR_PHY_PLACE=0 skips the whole file; OTPU_DDR_PHY_COLS sets the columns.
# - The sys reset's first register (crg_rst_t0) goes midway between the channels (rst_mid, below).
# It stops the run (error) when no OSERDES has a flop driver or a pad. Summary line:
# "otpu_ddr_phy_place: N flops of M OSERDES in K pblocks (...)" (or "off").
#
#   project:      constrs_1, PROCESSING_ORDER LATE, USED_IN_SYNTHESIS false (create_project.tcl)
#   non-project:  source otpu_ddr_phy_place.tcl on the linked netlist, before place_design

namespace eval ::otpu_ddr_phy_place {}

# the pad an OSERDES drives (through an ODELAY or an output buffer)
proc ::otpu_ddr_phy_place::pad {o} {
  foreach pin [list $o/OQ $o/TQ $o/T_OUT] {
    set p [get_pins -quiet $pin]
    if {$p eq ""} { continue }
    set ends [all_fanout -quiet -flat -endpoints_only -from $p]
    set ports [get_ports -quiet $ends]
    if {[llength $ports]} { return [lindex $ports 0] }
  }
  return ""
}

proc ::otpu_ddr_phy_place::run {} {
  if {[info exists ::env(OTPU_DDR_PHY_PLACE)] && $::env(OTPU_DDR_PHY_PLACE) eq "0"} {
    puts "otpu_ddr_phy_place: off (OTPU_DDR_PHY_PLACE=0)"
    return
  }
  set ncol [expr {[info exists ::env(OTPU_DDR_PHY_COLS)] ? $::env(OTPU_DDR_PHY_COLS) : 4}]
  set osers [get_cells -quiet -hier -filter {REF_NAME =~ OSERDESE* && IS_PRIMITIVE}]
  if {![llength $osers]} { puts "otpu_ddr_phy_place: no OSERDES"; return }
  # flops per clock region of their pads; a flop with OSERDES in two regions takes the first
  array set ffs {}; array set iox {}; array set seen {}
  set nos 0
  foreach o $osers {
    set port [pad $o]
    if {$port eq ""} { continue }
    set pp [get_property PACKAGE_PIN $port]
    set site [get_sites -quiet -of_objects [get_package_pins -quiet $pp]]
    if {$site eq ""} { continue }
    set cr [get_property CLOCK_REGION $site]
    set drv {}
    foreach pin [get_pins -quiet -of_objects $o -filter {DIRECTION == IN}] {
      if {![regexp {/(D[0-9]+|D\[[0-9]+\]|T[0-9]*)$} $pin]} { continue }
      set net [get_nets -quiet -of_objects $pin]
      if {$net eq ""} { continue }
      set src [get_cells -quiet -of_objects [get_pins -quiet -leaf -of_objects $net -filter {DIRECTION == OUT}]]
      if {$src ne "" && [string match FD* [get_property REF_NAME $src]]} { lappend drv $src }
    }
    if {![llength $drv]} { continue }
    incr nos
    lappend iox($cr) [get_property RPM_X $site]
    foreach f $drv {
      if {[info exists seen($f)]} { continue }
      set seen($f) 1
      lappend ffs($cr) $f
    }
  }
  if {![array size ffs]} { error "otpu_ddr_phy_place: no OSERDES with a flop driver and a pad" }
  set n 0; set desc {}; set soft {}
  foreach cr [lsort [array names ffs]] {
    # the region's slice columns by their distance from its I/O column (RPM_X: one grid for all
    # site types)
    set x0 [lindex [lsort -integer $iox($cr)] 0]
    array unset col
    set ymin ""; set ymax ""
    foreach s [get_sites -of_objects [get_clock_regions $cr] -filter {SITE_TYPE =~ SLICE*}] {
      if {![regexp {SLICE_X(\d+)Y(\d+)} $s -> sx sy]} { continue }
      set col($sx) [expr {abs([get_property RPM_X $s] - $x0)}]
      if {$ymin eq "" || $sy < $ymin} { set ymin $sy }
      if {$ymax eq "" || $sy > $ymax} { set ymax $sy }
    }
    set byd {}
    foreach sx [array names col] { lappend byd [list $col($sx) $sx] }
    set cols {}
    foreach e [lrange [lsort -integer -index 0 $byd] 0 [expr {$ncol - 1}]] { lappend cols [lindex $e 1] }
    set cols [lsort -integer $cols]
    set pb [create_pblock otpu_ddr_phy_$cr]
    foreach c $cols { resize_pblock $pb -add "SLICE_X${c}Y${ymin}:SLICE_X${c}Y${ymax}" }
    set_property IS_SOFT false $pb
    add_cells_to_pblock $pb [get_cells $ffs($cr)]
    incr n [llength $ffs($cr)]
    lappend desc "$cr: [llength $ffs($cr)] on SLICE_X[join $cols ,]"
    lappend soft [get_property IS_SOFT $pb]
  }
  puts "otpu_ddr_phy_place: $n flops of $nos OSERDES in [array size ffs] pblocks ([join $desc {; }]),\
        IS_SOFT [join [lsort -unique $soft] /]"
  rst_mid [lsort [array names ffs]]
}

# The sys reset's plain register (u_ld/crg_rst_t0, WLCRG rst_reg 2: 32143b30) midway between the two
# channels. It feeds crg_rst1's replicas, which sit by both channels' controllers, and the placer put it
# at one end: r2.1's m8 2343d61e had it at SLICE_X43Y334 and the replicas by channel 0 at Y76-82, 7.07 ns
# of route, 0 levels, the sys clock's worst path (+0.058 ns; c819923f the same at X1Y333; production
# b10245f5's FDPE_1 at Y345, +0.184). A hard pblock of the rows midway between the channels' PHY
# regions (OTPU_DDR_RST_ROWS, default 20, about the middle row), the slice columns 8..31 from the I/O
# column. Two channels only (the regions of the PHY pblocks in two groups by row); anything else: none.
proc ::otpu_ddr_phy_place::rst_mid {crs} {
  set t0 [get_cells -quiet -hier -filter {NAME =~ *crg_rst_t0_reg && IS_PRIMITIVE}]
  if {[llength $t0] != 1} { puts "otpu_ddr_phy_place: crg_rst_t0: [llength $t0] cells, not placed"; return }
  set rows {}
  foreach cr $crs {
    set ys {}
    foreach s [get_sites -of_objects [get_clock_regions $cr] -filter {SITE_TYPE =~ SLICE*}] {
      if {[regexp {SLICE_X\d+Y(\d+)} $s -> sy]} { lappend ys $sy }
    }
    set ys [lsort -integer $ys]
    lappend rows [list [lindex $ys 0] [lindex $ys end]]
  }
  set rows [lsort -integer -index 0 $rows]
  # the largest gap between consecutive PHY regions splits the two channels
  set gap 0; set lo ""; set hi ""
  for {set i 1} {$i < [llength $rows]} {incr i} {
    set a [lindex $rows [expr {$i - 1}] 1]; set b [lindex $rows $i 0]
    if {$b - $a > $gap} { set gap [expr {$b - $a}]; set lo $a; set hi $b }
  }
  if {$gap < 2} { puts "otpu_ddr_phy_place: crg_rst_t0: no gap between the PHY regions, not placed"; return }
  set nr [expr {[info exists ::env(OTPU_DDR_RST_ROWS)] ? $::env(OTPU_DDR_RST_ROWS) : 20}]
  set y0 [expr {($lo + $hi) / 2 - $nr / 2}]; set y1 [expr {$y0 + $nr - 1}]
  # the slice columns by their distance from the I/O column of the first PHY region (RPM_X)
  set io [get_sites -quiet -of_objects [get_clock_regions [lindex $crs 0]] -filter {SITE_TYPE =~ ILOGIC*}]
  set x0 [lindex [lsort -integer [get_property RPM_X $io]] 0]
  array set col {}
  foreach s [get_sites -of_objects [get_clock_regions [lindex $crs 0]] -filter {SITE_TYPE =~ SLICE*}] {
    if {[regexp {SLICE_X(\d+)Y} $s -> sx]} { set col($sx) [expr {abs([get_property RPM_X $s] - $x0)}] }
  }
  set byd {}
  foreach sx [array names col] { lappend byd [list $col($sx) $sx] }
  set cols [lsort -integer [lmap e [lrange [lsort -integer -index 0 $byd] 8 31] {lindex $e 1}]]
  set pb [create_pblock otpu_ddr_rst]
  resize_pblock $pb -add "SLICE_X[lindex $cols 0]Y${y0}:SLICE_X[lindex $cols end]Y${y1}"
  set_property IS_SOFT false $pb
  add_cells_to_pblock $pb $t0
  puts "otpu_ddr_phy_place: crg_rst_t0 in SLICE_X[lindex $cols 0]Y${y0}:SLICE_X[lindex $cols end]Y${y1}\
        (between the PHY rows $lo and $hi)"
}

::otpu_ddr_phy_place::run
