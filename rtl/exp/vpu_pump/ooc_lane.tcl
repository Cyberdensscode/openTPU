# EXPERIMENTAL (docs/vpu_pump.md): out-of-context synthesis + implementation of vpu_pump_top (the
# pumped multiply-add pairs) on the YPCB's part, its clocks from an MMCM. On a Vivado host:
#   vivado -mode batch -nojournal -source rtl/exp/vpu_pump/ooc_lane.tcl -tclargs <out> <VAR> \
#     [XR] [RETIME] [MULT DIV_S DIV_F] [PART]
# VAR 0: 8 lanes of today's otpu_fmma at clk_s; 1: 4 pumped units (today's 6 stages) at clk_f;
# 2: 4 pumped deep units (6 x (1 + XR) stages); 3: 4 pumped split units (10 stages). RETIME 1:
# synth_design -retiming. Defaults: VCO 1066.67 (MULT 8), clk_s 133.33 (DIV_S 8), clk_f 266.67
# (DIV_F 4). Prints OTPU_* lines.
set out [file normalize [lindex $argv 0]]
set var [lindex $argv 1]
set xr     [expr {[llength $argv] > 2 ? [lindex $argv 2] : 1}]
set retime [expr {[llength $argv] > 3 ? [lindex $argv 3] : 0}]
set mult   [expr {[llength $argv] > 4 ? [lindex $argv 4] : 8}]
set div_s  [expr {[llength $argv] > 5 ? [lindex $argv 5] : 8}]
set div_f  [expr {[llength $argv] > 6 ? [lindex $argv 6] : 4}]
set part   [expr {[llength $argv] > 7 ? [lindex $argv 7] : "xc7k480tffg1156-2"}]
set root [file normalize [file join [file dirname [info script]] ../../..]]
file mkdir $out

create_project -in_memory -part $part
set_property source_mgmt_mode None [current_project]
read_verilog -sv [list $root/rtl/vpu/otpu_fp.sv $root/rtl/vpu/otpu_fpipe.sv \
  $root/rtl/exp/vpu_pump/vpu_pump_lane.sv]
set opts [list -top vpu_pump_top -part $part -mode out_of_context -flatten_hierarchy rebuilt \
  -verilog_define SYNTHESIS -generic VAR=$var -generic XR=$xr -generic MULT=$mult \
  -generic DIV_S=$div_s -generic DIV_F=$div_f]
if {$retime} { lappend opts -retiming }
synth_design {*}$opts
create_clock -period 7.5 -name clk_in [get_ports clk_in]
set cs [get_clocks -quiet -of_objects [get_pins u_mmcm/CLKOUT0]]
set cf [get_clocks -quiet -of_objects [get_pins u_mmcm/CLKOUT1]]
foreach {t c} [list slow $cs fast $cf] {
  if {[llength $c]} { puts "OTPU_CLOCK $t [get_property NAME $c] [get_property PERIOD $c]" }
}
report_utilization -file $out/util_synth.txt
opt_design
place_design
phys_opt_design
route_design
report_utilization -file $out/util.txt
report_utilization -hierarchical -hierarchical_depth 4 -file $out/util_hier.txt
report_timing_summary -max_paths 10 -file $out/timing_summary.txt
report_clock_interaction -file $out/clock_interaction.txt
report_power -file $out/power.txt
foreach {tag from to} [list SS $cs $cs FF $cf $cf SF $cs $cf FS $cf $cs] {
  if {![llength $from] || ![llength $to]} { puts "OTPU_PATH $tag none"; continue }
  set q [lindex [get_timing_paths -quiet -from $from -to $to -setup -max_paths 1 -nworst 1] 0]
  set h [lindex [get_timing_paths -quiet -from $from -to $to -hold -max_paths 1 -nworst 1] 0]
  if {[llength $q]} {
    puts "OTPU_PATH $tag setup [get_property SLACK $q] req [get_property REQUIREMENT $q] levels [get_property LOGIC_LEVELS $q] [get_property STARTPOINT_PIN $q] -> [get_property ENDPOINT_PIN $q] | hold [get_property SLACK $h]"
    report_timing -from $from -to $to -max_paths 10 -nworst 1 -file $out/timing_$tag.txt
  } else {
    puts "OTPU_PATH $tag none"
  }
}
foreach {k pat} {LUT "LUT as Logic" LUTRAM "LUT as Memory" FF "Slice Registers" DSP "DSPs" SLICE "Slice"} {
  set f [open $out/util.txt]; set t [read $f]; close $f
  if {[regexp -line "^\\|\\s*$pat\\s*\\|\\s*(\[0-9.\]+)" $t -> v]} { puts "OTPU_UTIL $k $v" }
}
set p [lindex [get_timing_paths -setup -max_paths 1 -nworst 1] 0]
set h [lindex [get_timing_paths -hold -max_paths 1 -nworst 1] 0]
puts "OTPU_WNS [get_property SLACK $p] OTPU_WHS [get_property SLACK $h]"
puts "OTPU_DONE"
