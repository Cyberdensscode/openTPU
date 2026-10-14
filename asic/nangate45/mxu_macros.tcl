# The MXU's macros by the array's column structure (ORFS MACRO_PLACEMENT_TCL; the macro placer
# places the others around them). The ACT RAM is the MXU's stationary operand: macro (j, k), column
# j's piece of bytes 16k .. 16k+15 (otpu_actram's g_row[j].g_pc[p] tile g_c[c], k = p * tiles + c),
# feeds only PE chain k of column j (otpu_mxu: chains of 16 products), so each macro sits next to
# its chain. Two stacks of abutted macros face a strip between them, where the placer puts the
# array: columns j < MCOLS/2 on the left (pins to the right), the others on the right (pins to the
# left), chain-major (k, then j), so the weights move from column to column inside the strip and no
# column's 1024-bit read crosses another unit. The columns' scale RAMs (u_asc) top their side's
# stack. The weight FIFO (u_fd: the chunk's bits in order, u_fs0 / u_fs1: its scales between them)
# is a third stack left of the left one, pins to the right, the chunk crossing the left stack in the
# upper metals. All three are centred on the core's height at its left edge (the block's port B
# pins come to that edge); the macro placer's fence keeps the other macros to the block's right.
# MXU_STRIP_UM (default 800): the strip's width; MXU_FIFO_GAP_UM (100): the channel between the
# FIFO's stack and the left stack. With TMEM_GAP_UM the TMEM's macros (one copy: bank b's RAMs
# g_sub[0], g_sub[1]) are a block too, that far right of the MXU's: two stacks of abutted macros,
# banks 0 .. 3 on the left (pins to the right) and 4 .. 7 on the right (pins to the left), facing a
# strip of TMEM_STRIP_UM (200) for the banks' muxes and the slice's arbiter, centred on the core's
# height, its readers (VPU, quantizer, DMA, the MXU's drain) placed around it; the fence then
# starts right of the TMEM's block.
set block [ord::get_db_block]
set dbu [expr {double([$block getDbUnitsPerMicron])}]
set core [$block getCoreArea]
set cx0 [expr {[$core xMin] / $dbu}]
set cy0 [expr {[$core yMin] / $dbu}]
set ch [expr {([$core yMax] - [$core yMin]) / $dbu}]
set strip [expr {[info exists ::env(MXU_STRIP_UM)] ? $::env(MXU_STRIP_UM) : 800.0}]
set gap [expr {[info exists ::env(MXU_FIFO_GAP_UM)] ? $::env(MXU_FIFO_GAP_UM) : 100.0}]
set tstrip [expr {[info exists ::env(TMEM_STRIP_UM)] ? $::env(TMEM_STRIP_UM) : 200.0}]

# the macros: ACT (j k), the columns' scale RAMs (j), the FIFO's (an order key); name, width, height
set act [dict create]
set asc [dict create]
set fifo {}
set tmem {}
set nj 0
set nk 0
foreach inst [$block getInsts] {
  if {![$inst isBlock]} { continue }
  set n [$inst getName]
  set s [string map {\\ ""} $n]
  set m [$inst getMaster]
  set rec [list $n [expr {[$m getWidth] / $dbu}] [expr {[$m getHeight] / $dbu}]]
  if {[regexp {^u_slice/u_act/g_row\[(\d+)\]\.g_pc\[(\d+)\]\..*\.g_c\[(\d+)\]} $s -> j p c]} {
    if {[dict exists $act $j [list $p $c]]} { utl::error FLW 903 "mxu_macros.tcl: $s: one macro per tile expected" }
    dict set act $j [list $p $c] $rec
    set nj [expr {max($nj, $j + 1)}]
  } elseif {[regexp {^u_slice/u_act/g_row\[(\d+)\]\.u_asc} $s -> j]} {
    dict set asc $j $rec
  } elseif {[regexp {^u_slice/u_mxu/g_fd\[(\d+)\]\..*\.g_c\[(\d+)\]} $s -> g c]} {
    lappend fifo [list [expr {$g * 100 + $c}] {*}$rec]
  } elseif {[regexp {^u_slice/u_mxu/u_fs(\d)} $s -> i]} {
    lappend fifo [list [expr {150 + $i}] {*}$rec]
  } elseif {[regexp {^u_slice/u_tmem/g_one\.g_bank\[(\d+)\]\.g_sub\[(\d+)\]} $s -> b u]} {
    lappend tmem [list [expr {$b * 2 + $u}] $b {*}$rec]
  }
}
if {$nj < 2 || $nj % 2} { utl::error FLW 901 "mxu_macros.tcl: $nj ACT RAM columns (an even count expected)" }
# the chain of a (piece, tile): pieces in order, tiles within a piece in order
set nc 0
dict for {j pcs} $act { dict for {pc rec} $pcs { set nc [expr {max($nc, [lindex $pc 1] + 1)}] } }
set half [expr {$nj / 2}]
set ha 0.0
set wa 0.0
dict for {j pcs} $act {
  dict for {pc rec} $pcs {
    set ha [expr {max($ha, [lindex $rec 2])}]
    set wa [expr {max($wa, [lindex $rec 1])}]
    set nk [expr {max($nk, [lindex $pc 0] * $nc + [lindex $pc 1] + 1)}]
  }
}
set hs 0.0
dict for {j rec} $asc { set hs [expr {max($hs, [lindex $rec 2])}] }
set fifo [lsort -integer -index 0 $fifo]
set wf 0.0
set hf 0.0
foreach f $fifo { set wf [expr {max($wf, [lindex $f 2])}]; set hf [expr {$hf + [lindex $f 3]}] }

# heights on the row grid (1.4 um): the stacks abut
proc rows {y} { return [expr {floor($y / 1.4) * 1.4}] }
set hact [expr {$nk * $half * $ha}]
set htot [expr {$hact + $half * $hs}]
if {$htot > $ch} { utl::error FLW 902 "mxu_macros.tcl: the ACT stacks ($htot um) exceed the core ($ch um)" }
set y0 [expr {$cy0 + [rows [expr {($ch - $htot) / 2}]]}]
set xf [expr {$cx0 + $wf}]
set xl [expr {$xf + $gap}]
set xr [expr {$xl + $wa + $strip}]
puts "mxu_macros: $nj columns x $nk chains of ACT RAM macros ($wa x $ha um), [llength $fifo] FIFO macros;\
  stacks x [format %.1f $cx0] / [format %.1f $xl] / [format %.1f $xr], y [format %.1f $y0] .. [format %.1f [expr {$y0 + $htot}]]"

# each stack bottom up, every macro on the previous one as placed (place_macro moves a macro up to
# where its pins meet the tracks; the macros' heights are whole rows, so the next one is there)
set left {}
set right {}
dict for {j pcs} $act {
  dict for {pc rec} $pcs {
    set k [expr {[lindex $pc 0] * $nc + [lindex $pc 1]}]
    set e [list [expr {$k * $half + $j % $half}] {*}$rec]
    if {$j < $half} { lappend left $e } else { lappend right $e }
  }
}
dict for {j rec} $asc {
  set e [list [expr {$nk * $half + $j % $half}] {*}$rec]
  if {$j < $half} { lappend left $e } else { lappend right $e }
}
proc stack {list y0 xedge pins_right} {
  set y $y0
  foreach e [lsort -integer -index 0 $list] {
    lassign $e key n w h
    if {$pins_right} {
      place_macro -macro_name $n -location [list [expr {$xedge - $w}] $y] -orientation MY
    } else {
      place_macro -macro_name $n -location [list $xedge $y] -orientation R0
    }
    set y [expr {[[[[ord::get_db_block] findInst $n] getBBox] yMax] / $::dbu}]
  }
  return $y
}
set yl [stack $left $y0 [expr {$xl + $wa}] 1]
set yr [stack $right $y0 $xr 0]
set yf [stack $fifo [expr {$y0 + [rows [expr {($hact - $hf) / 2}]]}] $xf 1]
# the TMEM's block: the first half of the banks on the left, the rest on the right
set ux [expr {$xr + $wa}]
if {[info exists ::env(TMEM_GAP_UM)] && [llength $tmem]} {
  set nb 0; set wt 0.0; set ht 0.0
  foreach e $tmem {
    set nb [expr {max($nb, [lindex $e 1] + 1)}]; set wt [expr {max($wt, [lindex $e 3])}]
    set ht [expr {$ht + [lindex $e 4]}]
  }
  set tl {}; set tr {}
  foreach e $tmem {
    if {[lindex $e 1] < $nb / 2} { lappend tl [lreplace $e 1 1] } else { lappend tr [lreplace $e 1 1] }
  }
  set yt [expr {$cy0 + [rows [expr {($ch - $ht / 2) / 2}]]}]
  set xtl [expr {$ux + $::env(TMEM_GAP_UM)}]
  stack $tl $yt [expr {$xtl + $wt}] 1
  stack $tr $yt [expr {$xtl + $wt + $tstrip}] 0
  set ux [expr {$xtl + 2 * $wt + $tstrip}]
  puts "mxu_macros: the TMEM's [llength $tmem] macros at x [format %.1f $xtl] .. [format %.1f $ux],\
    y [format %.1f $yt] .. [format %.1f [expr {$yt + $ht / 2}]]"
}
# the macro placer keeps the other macros right of the blocks (one channel away), out of the strips
set ux [expr {$ux + 60.0}]
# (ORFS's defaults are 0: no fence)
if {![info exists ::env(RTLMP_FENCE_UX)] || $::env(RTLMP_FENCE_UX) == 0} {
  set ::env(RTLMP_FENCE_LX) $ux
  set ::env(RTLMP_FENCE_LY) $cy0
  set ::env(RTLMP_FENCE_UX) [expr {[$core xMax] / $dbu}]
  set ::env(RTLMP_FENCE_UY) [expr {[$core yMax] / $dbu}]
}
puts "mxu_macros: the stacks end at y [format %.1f $yl] / [format %.1f $yr] (FIFO [format %.1f $yf])"
