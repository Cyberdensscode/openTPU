# The ports latency-neutral before the CTS step's timing repair, not only after it (ORFS
# PRE_CTS_TCL, every point: tools/asic/flow.py). ORFS's cts.tcl builds the tree, runs repair_timing,
# then POST_CTS_TCL (io_latency.tcl: vclk takes the registers' median clock arrival). Before that
# the ideal vclk charges the insertion delay to every output path, so the repair sees ports only
# and gives up: main's card (5873d298, r3's floorplan) 1232 violating endpoints, every one a port
# (b_addr[30] -0.95 ns), 125 s, 1488 cells resized and 355 buffers inserted on those paths, the
# registers' paths untouched. Here the helper sources io_latency.tcl first (the POST hook sources it
# again: the same latency): one violating endpoint (the sequencer's ir_sync, -0.12 ns), 61 s, the
# CTS result's register-to-register period 2.62 -> 2.59 ns.
# (ORFS sources the hook inside a proc: the path a global, the helper a global proc)
set ::otpu_io_latency [file join [file dirname [file normalize [info script]]] io_latency.tcl]
if {[llength [info procs repair_timing_helper]]} {
  rename ::repair_timing_helper ::orfs_repair_timing_helper
  proc ::repair_timing_helper { args } {
    source $::otpu_io_latency
    orfs_repair_timing_helper {*}$args
  }
} else {
  puts "WARNING: cts_latency.tcl: no repair_timing_helper; the CTS repair runs before io_latency.tcl"
}
