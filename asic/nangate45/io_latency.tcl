# The ports' virtual clock takes the core clock's insertion delay after CTS (ORFS POST_CTS_TCL,
# every point: tools/asic/flow.py). The SDC times the ports against vclk, an ideal clock, with 20%
# of the period outside the block; once the core clock is propagated its registers clock 1-2 ns
# after the clock pin, which the ideal vclk charges to every output path and credits to every
# input path (on r2's card after CTS: port B's write data -2.20 ns so, -0.33 with the latency). A
# source latency on vclk equal to the registers' median clock arrival makes the ports' slacks
# latency-neutral, as if the logic around the block clocked its registers from the same tree.
# Before CTS both clocks are ideal and the 20% budget is as it was; ORFS writes the latency into
# 4_cts.sdc, so global routing, routing and the final report keep it (a source latency: it holds
# whatever a later script propagates).
set_propagated_clock [get_clocks core_clock]
set rl {}
foreach p [all_registers -clock_pins] { lappend rl [get_property $p arrival_max_rise] }
if {[llength $rl]} {
  set rl [lsort -real $rl]
  set l [lindex $rl [expr {[llength $rl] / 2}]]
  puts [format "io_latency.tcl: vclk source latency %.3f ns (the median of %d register clock pins)" \
    $l [llength $rl]]
  set_clock_latency -source $l [get_clocks vclk]
}
