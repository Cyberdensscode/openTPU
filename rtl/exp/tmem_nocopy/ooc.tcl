# EXPERIMENTAL (docs/tmem_nocopy.md): out-of-context synthesis + implementation of tmem_nc_top
# (the TMEM and the slice's arbiter for it, every input and output registered):
#   vivado -mode batch -nojournal -source rtl/exp/tmem_nocopy/ooc.tcl -tclargs <out> <VAR> \
#     <WORDS> <NSB> <SBP> <PART> <PERIOD_NS> [SBH] [MMCM] [FAMILY] [ABL] [SYNTHONLY] [GST]
# MMCM 1 (VAR 6, P3, always): the core clock (and 2x) from an MMCM on clk (FAMILY 0 7-series, 1
# UltraScale+), with the pumped copies' multicycle paths (rtl/exp/tmem_pump/ooc.tcl's). ABL: an
# otpu_tmem ablation (DSEL, CMP, RPL, AMUX: OTPU_TMEM_ABL_*; a LUT breakdown only, the TMEM is
# wrong with one), "-" none. SYNTHONLY 1: stop after opt_design (utilization only). GST 1: QUANT 3
# and COLL the DMA's guests (VAR 4 / 5).
# Prints OTPU_* summary lines; reports go to <out>.
set out [file normalize [lindex $argv 0]]
lassign [lrange $argv 1 end] var words nsb sbp part period sbh mmcm family abl synthonly gst
if {$gst eq ""} { set gst 0 }
if {$sbh eq ""} { set sbh 0 }
if {$mmcm eq ""} { set mmcm 0 }
if {$var == 6} { set mmcm 1 }
if {$family eq ""} { set family 0 }
if {$synthonly eq ""} { set synthonly 0 }
set defs [list SYNTHESIS]
if {$abl ne "" && $abl ne "-"} { lappend defs OTPU_TMEM_ABL_$abl }
set root [file normalize [file join [file dirname [info script]] ../../..]]
file mkdir $out

create_project -in_memory -part $part
set_property source_mgmt_mode None [current_project]
read_verilog -sv [list $root/rtl/tech/generic/otpu_tech_ram.sv $root/rtl/mem/otpu_tmem.sv \
  $root/rtl/exp/tmem_pump/otpu_tmem_pump.sv $root/rtl/exp/tmem_nocopy/tmem_nc_top.sv]
synth_design -top tmem_nc_top -part $part -mode out_of_context -flatten_hierarchy rebuilt \
  -verilog_define $defs -generic VAR=$var -generic WORDS=$words -generic NSB=$nsb \
  -generic SBP=$sbp -generic SBH=$sbh -generic MMCM=$mmcm -generic FAMILY=$family \
  -generic CLKIN_NS=$period -generic GST=$gst
create_clock -period $period -name clk [get_ports clk]
if {$mmcm} {
  set cs [get_clocks -quiet -of_objects [get_pins -hier -filter {NAME =~ *u_mmcm/CLKOUT0}]]
  set cf [get_clocks -quiet -of_objects [get_pins -hier -filter {NAME =~ *u_mmcm/CLKOUT1}]]
  foreach {t c} [list slow $cs fast $cf] {
    if {[llength $c]} { puts "OTPU_CLOCK $t [get_property NAME $c] [get_property PERIOD $c]" }
  }
  # the pumped copies' contract (otpu_tmem_pump; as rtl/exp/tmem_pump/ooc.tcl)
  set rams [get_cells -quiet -hierarchical -filter {REF_NAME =~ RAMB* || REF_NAME =~ URAM*}]
  set rp [get_pins -quiet -hierarchical -filter {NAME =~ *u_pb/ra[*] || NAME =~ *u_pb/rb[*] || NAME =~ *u_pb/rena || NAME =~ *u_pb/renb}]
  set pwf {}
  foreach c [get_cells -quiet -hierarchical -filter {NAME =~ *u_tmem/*pw_* && IS_SEQUENTIAL}] {
    set ck [get_clocks -quiet -of_objects [get_pins -quiet -of_objects $c -filter {IS_CLOCK}]]
    if {[llength $cf] && $ck eq [get_property NAME $cf]} { lappend pwf $c }
  }
  puts "OTPU_RAMS [llength $rams] OTPU_MCP_PINS [llength $rp] OTPU_MCP_PW [llength $pwf]"
  if {[llength $rp]} {
    set_multicycle_path 2 -setup -end -from $cs -through $rp
    set_multicycle_path 1 -hold -end -from $cs -through $rp
    set_multicycle_path 2 -setup -start -from $rams -to $cs
    set_multicycle_path 1 -hold -start -from $rams -to $cs
  }
  if {[llength $pwf]} {
    set_multicycle_path 2 -setup -end -from $cs -to $pwf
    set_multicycle_path 1 -hold -end -from $cs -to $pwf
  }
}
report_utilization -file $out/util_synth.txt
opt_design
if {$synthonly} {
  report_utilization -file $out/util_opt.txt
  set u [report_utilization -return_string]
  foreach k {{Slice LUTs} {Slice Registers} {Block RAM Tile} {CLB LUTs} {CLB Registers}} {
    foreach l [split $u \n] {
      if {[regexp "^\\|\\s*${k}\\*?\\s*\\|\\s*(\[0-9.\]+)" $l -> n]} { puts "OTPU_UTIL $k $n"; break }
    }
  }
  puts "OTPU_DONE"
  return
}
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
if {$mmcm} {
  foreach {tag from to} [list SS $cs $cs FF $cf $cf SF $cs $cf FS $cf $cs] {
    if {![llength $from] || ![llength $to]} { continue }
    set q [lindex [get_timing_paths -quiet -from $from -to $to -setup -max_paths 1 -nworst 1] 0]
    if {[llength $q]} {
      puts "OTPU_PATH $tag setup [get_property SLACK $q] levels [get_property LOGIC_LEVELS $q] [get_property STARTPOINT_PIN $q] -> [get_property ENDPOINT_PIN $q]"
    }
  }
}
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
