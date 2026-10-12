# EXPERIMENTAL (docs/tmem_nocopy.md): out-of-context synthesis + implementation of tmem_nc_top
# (the TMEM and the slice's arbiter for it, every input and output registered), one clock:
#   vivado -mode batch -nojournal -source rtl/exp/tmem_nocopy/ooc.tcl -tclargs <out> <VAR> \
#     <WORDS> <NSB> <SBP> <PART> <PERIOD_NS> [SBH]
# Prints OTPU_* summary lines; reports go to <out>.
set out [file normalize [lindex $argv 0]]
lassign [lrange $argv 1 end] var words nsb sbp part period sbh
if {$sbh eq ""} { set sbh 0 }
set root [file normalize [file join [file dirname [info script]] ../../..]]
file mkdir $out

create_project -in_memory -part $part
set_property source_mgmt_mode None [current_project]
read_verilog -sv [list $root/rtl/tech/generic/otpu_tech_ram.sv $root/rtl/mem/otpu_tmem.sv \
  $root/rtl/exp/tmem_nocopy/tmem_nc_top.sv]
synth_design -top tmem_nc_top -part $part -mode out_of_context -flatten_hierarchy rebuilt \
  -verilog_define SYNTHESIS -generic VAR=$var -generic WORDS=$words -generic NSB=$nsb \
  -generic SBP=$sbp -generic SBH=$sbh
create_clock -period $period -name clk [get_ports clk]
report_utilization -file $out/util_synth.txt
opt_design
place_design
phys_opt_design
route_design
report_utilization -file $out/util.txt
report_utilization -hierarchical -hierarchical_depth 4 -file $out/util_hier.txt
report_timing_summary -max_paths 20 -file $out/timing_summary.txt
report_timing -max_paths 30 -nworst 1 -file $out/timing_worst.txt
report_design_analysis -logic_level_distribution -file $out/levels.txt
set p [lindex [get_timing_paths -setup -max_paths 1 -nworst 1] 0]
set h [lindex [get_timing_paths -hold -max_paths 1 -nworst 1] 0]
puts "OTPU_PATH setup [get_property SLACK $p] levels [get_property LOGIC_LEVELS $p] [get_property STARTPOINT_PIN $p] -> [get_property ENDPOINT_PIN $p]"
# the worst path ending at a grant output register and at a RAM's address pin
foreach {tag pat} [list GNT {NAME =~ gnt_out_reg*} ] {
  set c [get_cells -quiet -hierarchical -filter $pat]
  if {[llength $c]} {
    set q [lindex [get_timing_paths -quiet -to $c -setup -max_paths 1 -nworst 1] 0]
    if {[llength $q]} {
      puts "OTPU_PATH $tag [get_property SLACK $q] levels [get_property LOGIC_LEVELS $q] [get_property STARTPOINT_PIN $q]"
    }
  }
}
set rams [get_cells -quiet -hierarchical -filter {REF_NAME =~ RAMB* || REF_NAME =~ URAM*}]
if {[llength $rams]} {
  set q [lindex [get_timing_paths -quiet -to $rams -setup -max_paths 1 -nworst 1] 0]
  if {[llength $q]} {
    puts "OTPU_PATH RAM [get_property SLACK $q] levels [get_property LOGIC_LEVELS $q] [get_property STARTPOINT_PIN $q] -> [get_property ENDPOINT_PIN $q]"
  }
  set q [lindex [get_timing_paths -quiet -from $rams -setup -max_paths 1 -nworst 1] 0]
  if {[llength $q]} {
    puts "OTPU_PATH RAMOUT [get_property SLACK $q] levels [get_property LOGIC_LEVELS $q] -> [get_property ENDPOINT_PIN $q]"
  }
}
set u [report_utilization -return_string]
foreach k {{Slice LUTs} {Slice Registers} {Block RAM Tile} {RAMB36} {RAMB18} {CLB LUTs} {CLB Registers} {Slice}} {
  foreach l [split $u \n] {
    if {[regexp "^\\|\\s*${k}\\*?\\s*\\|\\s*(\[0-9.\]+)" $l -> n]} { puts "OTPU_UTIL $k $n"; break }
  }
}
puts "OTPU_WNS [get_property SLACK $p] OTPU_WHS [get_property SLACK $h]"
puts "OTPU_DONE"
