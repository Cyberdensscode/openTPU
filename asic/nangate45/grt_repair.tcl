# The global route's incremental repair as repair_timing only (ORFS PRE_GLOBAL_ROUTE_TCL, the card
# point: tools/asic/flow.py's grt_repair). ORFS's GRT step runs repair_design (every net's slew,
# capacitance and fanout, single-threaded) and then repair_timing on the route's parasitics; on the
# card repair_design does not finish (r3: 1.52 M of 2.53 M nets in 25 min, then ~2,000 a minute,
# stopped after 8 h; docs/asic.md), so the card's runs skipped the whole repair. The placement
# stage has already run repair_design (3_4_place_resized); here the GRT step's is a no-op and its
# repair_timing works on the failing paths only: on main's card (5873d298, r3's floorplan) with a
# 0.5 ns setup margin, 100 s, 41 buffers and 295 cells upsized, register to register 2.497 ->
# 2.000 ns; with a 1.0 ns margin it stalls at 1.99 ns, the VPU's floating-point stages
# (docs/timing_ffs.md section 4). The nets over their maximum capacitance or slew that no failing
# path crosses stay as placement left them.
proc ::repair_design_helper { } {
  puts "grt_repair.tcl: the global route's repair_design skipped (repair_timing only)"
}
