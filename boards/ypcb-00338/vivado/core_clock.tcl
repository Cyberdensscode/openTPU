# The core clocks' MMCM (bd_native.tcl's clk_wiz_0, from sys_clk_50), shared by bd_native.tcl and
# create_project.tcl (its CORE_KHZ register). core_mmcm CORE_MHZ CLK2X -> {VCO CORE_DIV}:
# the VCO at 800 MHz (50 MHz x 16) and core_clk's divider rounded to the MMCM's 1/8 steps, as the
# qualified images are built (133.33: 6, 100: 8). With CLK2X core_clk2x divides the same VCO by
# half of core_clk's divider, so that one must be an even integer; where 800 MHz has none for
# CORE_MHZ (150: 5.333), the VCO is the one of 600 to 1200 MHz (the multiplier in its 1/8 steps)
# that an even divider takes to CORE_MHZ exactly, 6 first (150: 900 / 6, core_clk2x 300; 141.67:
# 850 / 6; 162.5: 975 / 6), else the even divider and VCO nearest to it.
proc core_mmcm {mhz clk2x} {
  set vco 800.0
  set div [expr {round($vco / $mhz * 8) / 8.0}]
  if {!$clk2x || ($div == floor($div) && int($div) % 2 == 0)} { return [list 800 $div] }
  foreach d {6 4 8 10 12 14 16} {
    set v [expr {$mhz * $d}]
    set m8 [expr {$v * 8 / 50.0}]
    if {$v >= 600 && $v <= 1200 && abs($m8 - round($m8)) < 0.01} {
      return [list [expr {round($m8) * 50 / 8.0}] [expr {double($d)}]]
    }
  }
  set best {}
  set berr 1e9
  for {set m8 96} {$m8 <= 192} {incr m8} {
    set v [expr {50.0 * $m8 / 8}]
    for {set d 2} {$d <= 32} {incr d 2} {
      set err [expr {abs($v / $d - $mhz)}]
      if {$err < $berr - 1e-9} { set berr $err; set best [list $v [expr {double($d)}]] }
    }
  }
  return $best
}
