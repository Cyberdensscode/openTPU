# The MXU's systolic array placed as a grid (docs/mxu_systolic.md, "Placement on the YPCB").
# Array column j goes in one DSP column, its positions in order up the column, so position i sits
# at the same height in every column; its column end (otpu_mxce) goes above them (OTPU_MXU_CE=mid:
# in the middle, between the lower and the upper half of the chains). The ACT RAM's memory
# of column j (otpu_actram g_row[j]) goes in the block RAM column beside it, each block RAM at the
# height of the positions it feeds. The weight RAMs (otpu_mxu g_fd, u_fs0, u_fs1) go in the block
# RAM column beside column 0.
#
# Why. Left to the placer, r2's MCOLS 8 dev build (f55949f9) spread each array column's 16-DSP
# cascade chains over 5-7 DSP columns across the whole die. Each column's weight hop (2 x 8 bits
# per position, wc[j] -> column j+1) and its activation feed (ACT RAM -> u_a skews -> the PEs)
# crossed the die (the hop bits' pin boxes ~220 tiles), and routing failed at global congestion
# level 6-7. These are LOCs only: no logic and no cycle changes.
#
# Inputs:
# - MCOLS and D: the netlist's g_ask[j].g_p[i] PEs.
# - The DSP site columns (DSP48_X<c>) and block RAM site columns (RAMB36_X<c>) of array columns
#   0..MCOLS-1, and the weight RAMs' block RAM column: the part's table below. Environment
#   overrides: OTPU_MXU_DSP_COLS="1 2 ...", OTPU_MXU_BRAM_COLS="1 2 ...", OTPU_MXU_WRAM_COL=0.
# - The first PE row, OTPU_MXU_BASE (DSP site row). Default: the array and its column ends
#   centred in the DSP column.
# - On at MCOLS 8 and up, off below (the placer's), from the netlist's MCOLS; OTPU_MXU_PLACE=1
#   places at any MCOLS, OTPU_MXU_PLACE=0 skips the whole file.
# - The column ends: OTPU_MXU_CE=top (the default) above the array; mid puts a column's end
#   between its chains 0..n/2-1 and n/2..n-1. Each chain end feeds its own DSP of the column end
#   (otpu_mxce g_m[g]) from the chain's last DSP's P register through fabric: above the array the
#   lowest chain's end crosses 113 DSP rows, at 0 levels with 5.3-5.8 ns of route (r2.1's m8
#   163ab46d: g_ask[0].g_p[15] at DSP48_X1Y25 -> g_col_end[0] g_m[0] at Y138, -0.005 ns before
#   userclk1's reroute; 76 such endpoints under +0.2 ns); in the middle it is 65 at most. But mid
#   moved the congestion onto the column ends: r2.1's m8 4f8be3d2 (mid) had three placer windows
#   at level 5 (101-103%, X27-58 Y200-238: u_mxu, u_mem, u_seq, u_ch1) against 163ab46d's one,
#   the weight feed's endpoints under +0.2 ns went from 88 to 455, and core_clk from -0.003 to
#   -0.156 ns. top places exactly as before mid existed (the scale RAMs above the array too).
# It stops the run (error) when a name pattern matches no cells, when the PEs are not MCOLS x D,
# or when a cascade chain is not in position order. The LOCs it sets are also written to
# otpu_mxu_place.xdc in the implementation run's directory (or to OTPU_MXU_PLACE_XDC).
#
#   project:      constrs_1, PROCESSING_ORDER LATE, USED_IN_SYNTHESIS false (create_project.tcl)
#   non-project:  source otpu_mxu_place.tcl on the linked netlist, before place_design
#
# Device facts (xc7k480t, checked on r2's netlist):
# - DSP48_X0..X11 sit at tile X 9 14 29 34 49 54 69 74 89 94 105 117, 160 sites each.
# - RAMB36_X0..X11 sit at tile X 6 17 26 37 46 57 66 77 86 97 108 114, 80 sites each.
# - RAMB36 row r is level with DSP rows 2r and 2r+1.

namespace eval ::otpu_mxu_place {
  # part prefix, MCOLS, DSP site columns, block RAM site columns, weight RAMs' block RAM column.
  # MCOLS 8 on the xc7k480t: array columns on DSP X1..X8 (tile X 14..89), each beside its block
  # RAM column X1..X8 (tile X 17 26 37 46 57 66 77 86); DSP X0, X9..X11 (640 sites) stay for the
  # VPU, QUANT and the MXU's epilogue lanes; the weight RAMs in block RAM X0 (tile X 6), beside
  # column 0.
  variable table {
    xc7k480t 8 {1 2 3 4 5 6 7 8} {1 2 3 4 5 6 7 8} 0
    xc7k480t 4 {2 3 4 5}         {2 3 4 5}         1
    xc7k480t 2 {3 4}             {3 4}             2
  }
  variable lines {}
  variable used
  variable all
}

# the DSPs (kind DSP48) or block RAMs (RAMB) whose names match a regular expression (the names'
# brackets rule out glob patterns)
proc ::otpu_mxu_place::cells {kind re what} {
  variable all
  if {![info exists all($kind)]} {
    set all($kind) [get_cells -quiet -hierarchical -filter "REF_NAME =~ $kind*"]
  }
  set c [lsearch -all -inline -regexp $all($kind) $re]
  if {![llength $c]} { error "otpu_mxu_place: no cells for $what ($kind, /$re/)" }
  return $c
}

# a name as a regular expression
proc ::otpu_mxu_place::q {s} {
  set o ""
  foreach ch [split $s ""] {
    if {[string first $ch {[]().*+?^$|{}\\}] >= 0} { append o "\\" }
    append o $ch
  }
  return $o
}

# the DSP whose cascade output drives this DSP's PCIN / ACIN / BCIN, or ""
proc ::otpu_mxu_place::pred {c} {
  foreach pn {PCIN[0] ACIN[0] BCIN[0]} {
    set p [get_pins -quiet $c/$pn]
    if {$p eq ""} { continue }
    set n [get_nets -quiet -of_objects $p]
    if {$n eq "" || [get_property TYPE $n] ne "SIGNAL"} { continue }
    foreach d [get_pins -quiet -leaf -filter {DIRECTION == OUT} -of_objects $n] {
      set dc [get_cells -quiet -of_objects $d]
      if {[string match DSP48* [get_property REF_NAME $dc]]} { return $dc }
    }
  }
  return ""
}

# a set of DSPs in cascade order: each chain from its head, the chains in name order
proc ::otpu_mxu_place::chain_order {cs} {
  array set P {}; array set S {}
  foreach c $cs { set P($c) [pred $c] }
  foreach c $cs { if {$P($c) ne "" && [info exists P($P($c))]} { set S($P($c)) $c } }
  set out {}
  foreach c [lsort -dictionary $cs] {
    if {$P($c) ne "" && [info exists P($P($c))]} { continue }
    for {set x $c} {1} {set x $S($x)} { lappend out $x; if {![info exists S($x)]} { break } }
  }
  if {[llength $out] != [llength $cs]} { error "otpu_mxu_place: a cascade loop among [llength $cs] DSPs" }
  return $out
}

# a position's (or a fractional position's) DSP row above the base: the positions from the split
# up sit above the column ends
proc ::otpu_mxu_place::prow {i} {
  variable split_at; variable n_ce
  return [expr {$i >= $split_at ? $i + $n_ce : $i}]
}

proc ::otpu_mxu_place::loc {cell site} {
  variable lines; variable used
  if {[info exists used($site)]} { error "otpu_mxu_place: $site taken twice ($used($site), $cell)" }
  if {[get_sites -quiet $site] eq ""} { error "otpu_mxu_place: no site $site for $cell" }
  set used($site) $cell
  set_property LOC $site [get_cells $cell]
  lappend lines "set_property LOC $site \[get_cells {$cell}\]"
}

# a block RAM at row r of block RAM column x, or the free row nearest r
proc ::otpu_mxu_place::ram {cell x r nrow} {
  variable used
  set t [get_property REF_NAME [get_cells $cell]]
  for {set k 0} {$k < $nrow} {incr k} {
    foreach rr [list [expr {$r + $k}] [expr {$r - $k}]] {
      if {$rr < 0 || $rr >= $nrow} { continue }
      set s36 RAMB36_X${x}Y$rr
      set s18 [list RAMB18_X${x}Y[expr {2 * $rr}] RAMB18_X${x}Y[expr {2 * $rr + 1}]]
      if {$t eq "RAMB36E1"} {
        if {![info exists used($s36)] && ![info exists used([lindex $s18 0])] && ![info exists used([lindex $s18 1])]} {
          loc $cell $s36; return
        }
      } else {
        foreach s $s18 {
          if {![info exists used($s)] && ![info exists used($s36)]} { loc $cell $s; return }
        }
      }
    }
  }
  error "otpu_mxu_place: no free block RAM row in column $x for $cell"
}

proc ::otpu_mxu_place::run {} {
  variable table; variable lines; variable used; variable all
  set lines {}; array unset used; array unset all
  set on [expr {[info exists ::env(OTPU_MXU_PLACE)] ? $::env(OTPU_MXU_PLACE) : ""}]
  if {$on eq "0"} {
    puts "otpu_mxu_place: off (OTPU_MXU_PLACE=0)"
    return
  }
  # ---- the PE grid
  array set PE {}
  set pre ""; set mc 0; set d 0
  foreach c [cells DSP48 {/u_mxu/g_ask\[\d+\]\.g_p\[\d+\]\.u_pe/} \
               "the MXU's PE DSPs (otpu_mxu g_ask\[j\].g_p\[i\].u_pe)"] {
    if {![regexp {^(.*)/g_ask\[(\d+)\]\.g_p\[(\d+)\]\.u_pe/} $c -> p j i]} { error "otpu_mxu_place: PE name $c" }
    if {$pre eq ""} { set pre $p } elseif {$p ne $pre} { error "otpu_mxu_place: two MXUs ($pre, $p): one slice only" }
    if {[info exists PE($j,$i)]} { error "otpu_mxu_place: two DSPs for PE $j.$i ($PE($j,$i), $c)" }
    set PE($j,$i) $c
    set mc [expr {max($mc, $j + 1)}]; set d [expr {max($d, $i + 1)}]
  }
  if {[array size PE] != $mc * $d} { error "otpu_mxu_place: [array size PE] PE DSPs, not MCOLS $mc x D $d" }
  # the default: the grid at MCOLS 8 and up, the placer's below. MCOLS 8 does not route without it
  # (f55949f9); at MCOLS 4 it costs (d352e53d at 133.33 MHz, r2m4 with it against r2m4ng without:
  # core_clk WNS -0.536 / -0.281 ns, TNS -236.1 / -33.9 ns, crg sys +0.019 / +0.098, 57.5K / 54.2K
  # slices; docs/mxu_systolic.md)
  if {$on eq "" && $mc < 8} {
    puts "otpu_mxu_place: off (MCOLS $mc, below 8: the placer's; OTPU_MXU_PLACE=1 places it)"
    return
  }
  # the chains: PE i's cascade input is PE i-1's, or none (a chain's first)
  set heads 0
  for {set j 0} {$j < $mc} {incr j} {
    for {set i 0} {$i < $d} {incr i} {
      set p [pred $PE($j,$i)]
      if {$p eq ""} { incr heads; continue }
      if {$i == 0 || $p ne $PE($j,[expr {$i - 1}])} {
        error "otpu_mxu_place: PE $j.$i's cascade input is $p, not PE $j.[expr {$i - 1}]"
      }
    }
  }
  # ---- columns
  set part [get_property PART [current_design]]
  set dcols ""; set bcols ""; set wcol ""
  foreach {pp m dl bl w} $table {
    if {[string match $pp* $part] && $m == $mc} { set dcols $dl; set bcols $bl; set wcol $w }
  }
  if {[info exists ::env(OTPU_MXU_DSP_COLS)]} { set dcols $::env(OTPU_MXU_DSP_COLS) }
  if {[info exists ::env(OTPU_MXU_BRAM_COLS)]} { set bcols $::env(OTPU_MXU_BRAM_COLS) }
  if {[info exists ::env(OTPU_MXU_WRAM_COL)]} { set wcol $::env(OTPU_MXU_WRAM_COL) }
  if {[llength $dcols] != $mc || [llength $bcols] != $mc || $wcol eq ""} {
    error "otpu_mxu_place: no columns for $part at MCOLS $mc (OTPU_MXU_DSP_COLS / OTPU_MXU_BRAM_COLS /\
           OTPU_MXU_WRAM_COL, or OTPU_MXU_PLACE=0)"
  }
  set nd [llength [get_sites -quiet DSP48_X[lindex $dcols 0]Y*]]
  set nb [llength [get_sites -quiet RAMB36_X[lindex $bcols 0]Y*]]
  if {!$nd || !$nb} { error "otpu_mxu_place: no DSP48_X[lindex $dcols 0] / RAMB36_X[lindex $bcols 0] sites on $part" }
  # ---- the column ends, in cascade order
  for {set j 0} {$j < $mc} {incr j} {
    set ce($j) [chain_order [cells DSP48 [format {^%s/g_col_end\[%d\]\.u_ce/} [q $pre] $j] \
                               "column $j's end (otpu_mxu g_col_end\[$j\].u_ce)"]]
  }
  set nce [llength $ce(0)]
  set h [expr {$d + $nce}]
  set base [expr {2 * (($nd - $h) / 4)}]
  if {[info exists ::env(OTPU_MXU_BASE)]} { set base $::env(OTPU_MXU_BASE) }
  if {$base < 0 || $base + $h > $nd} { error "otpu_mxu_place: rows $base..[expr {$base + $h - 1}] outside the $nd DSP rows" }
  # the column ends' first row above the base: after the lower half of the chains (mid), or after
  # all of them (top); a position's row skips them (prow)
  set cem [expr {[info exists ::env(OTPU_MXU_CE)] ? $::env(OTPU_MXU_CE) : "top"}]
  if {$cem ni {mid top}} { error "otpu_mxu_place: OTPU_MXU_CE '$cem': mid or top" }
  set nch [expr {$heads / $mc}]
  variable split_at [expr {$cem eq "top" ? $d : ($nch / 2) * ($d / $nch)}]
  variable n_ce $nce
  set split $split_at
  # ---- PEs and column ends
  for {set j 0} {$j < $mc} {incr j} {
    set x [lindex $dcols $j]
    for {set i 0} {$i < $d} {incr i} { loc $PE($j,$i) DSP48_X${x}Y[expr {$base + [prow $i]}] }
    set k $split
    foreach c $ce($j) { loc $c DSP48_X${x}Y[expr {$base + $k}]; incr k }
  }
  # ---- the ACT RAM: column j's memory beside it, piece p (bytes p*D/NP ..) level with its PEs
  set act [file dirname $pre]/u_act
  set nact 0
  for {set j 0} {$j < $mc} {incr j} {
    set x [lindex $bcols $j]
    array unset PC
    foreach c [cells RAMB [format {^%s/g_row\[%d\]\.g_pc\[\d+\]} [q $act] $j] \
                 "column $j's ACT RAM pieces (otpu_actram g_row\[$j\].g_pc)"] {
      regexp {g_pc\[(\d+)\]} $c -> p
      lappend PC($p) $c
    }
    set np [array size PC]
    set pb [expr {double($d) / $np}]
    foreach p [lsort -integer [array names PC]] {
      set rs [lsort -dictionary $PC($p)]
      set nr [llength $rs]
      set k 0
      foreach c $rs {
        # the DSP row in the middle of the positions this RAM's bytes feed, as a block RAM row
        set dr [expr {$base + [prow [expr {$p * $pb + ($k + 0.5) * $pb / $nr}]]}]
        ram $c $x [expr {int($dr / 2)}] $nb
        incr k; incr nact
      }
    }
    set k 0
    foreach c [lsort -dictionary [cells RAMB [format {^%s/g_row\[%d\]\.u_asc/} [q $act] $j] \
                                    "column $j's ACT scale RAM (otpu_actram g_row\[$j\].u_asc)"]] {
      ram $c $x [expr {($cem eq "top" ? $base + $d : $base + $split + $nce / 2) / 2 + $k}] $nb
      incr k; incr nact
    }
  }
  # ---- the weight RAMs, spread level with the PEs beside column 0
  set wr [lsort -dictionary [cells RAMB [format {^%s/(g_fd\[\d+\]\.|u_fs0/|u_fs1/)} [q $pre]] \
                               "the MXU's weight RAMs (otpu_mxu g_fd, u_fs0, u_fs1)"]]
  set n [llength $wr]
  set k 0
  foreach c $wr {
    ram $c $wcol [expr {int(($base + [prow [expr {($k + 0.5) * $d / $n}]]) / 2)}] $nb
    incr k
  }
  # ---- the record: in an implementation run's directory, or OTPU_MXU_PLACE_XDC; not where
  # build.tcl opens the synthesized design for its reports (the build's working directory)
  set rec ""
  if {[info exists ::env(OTPU_MXU_PLACE_XDC)]} {
    set rec $::env(OTPU_MXU_PLACE_XDC)
  } elseif {[string match *.runs/impl_* [pwd]]} {
    set rec [file join [pwd] otpu_mxu_place.xdc]
  }
  if {$rec ne ""} {
    set f [open $rec w]
    puts $f "# otpu_mxu_place.tcl on $part: MCOLS $mc, D $d ($heads cascade chains), rows $base..[expr {$base + $h - 1}],\
             column ends ($cem) at rows [expr {$base + $split}]..[expr {$base + $split + $nce - 1}];\
             DSP columns $dcols, ACT block RAM columns $bcols, weight RAMs column $wcol"
    puts $f [join $lines \n]
    close $f
  }
  puts "otpu_mxu_place: MCOLS $mc x D $d PEs ($heads chains) and [llength $ce(0)] column-end DSPs a column on\
        DSP48_X{$dcols} rows $base..[expr {$base + $h - 1}], the column ends ($cem) from row [expr {$base + $split}];\
        $nact ACT RAMs on RAMB36_X{$bcols};\
        $n weight RAMs on RAMB36_X$wcol ([llength $lines] LOCs[expr {$rec ne "" ? ", $rec" : ""}])"
}

::otpu_mxu_place::run
