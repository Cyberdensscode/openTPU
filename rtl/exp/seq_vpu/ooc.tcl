# EXPERIMENTAL (docs/seq_vpu.md): out-of-context synthesis (and optionally implementation) of one
# module on the YPCB's part, with an area breakdown: Vivado's hierarchy, and the flat cells grouped
# by the RTL name they carry (registers by their signal, LUTs and carries by the net they drive).
#   vivado -mode batch -nojournal -source rtl/exp/seq_vpu/ooc.tcl -tclargs \
#     <out> <top> <PERIOD_NS> <impl 0|1> <generics k=v,...|-> <defines D1,D2|-> [extra sources ,]
# Prints OTPU_UTIL / OTPU_WNS / OTPU_PATH / OTPU_GROUP lines (OTPU_GROUP name lut ff carry dsp).
set out [file normalize [lindex $argv 0]]
set top [lindex $argv 1]
set period [lindex $argv 2]
set impl [lindex $argv 3]
set gens [lindex $argv 4]
set defs [lindex $argv 5]
set extra [expr {[llength $argv] > 6 ? [lindex $argv 6] : ""}]
set part "xc7k480tffg1156-2"
set root [file normalize [file join [file dirname [info script]] ../../..]]
file mkdir $out

create_project -in_memory -part $part
set_property source_mgmt_mode None [current_project]
set srcs [list $root/rtl/vpu/otpu_fp.sv $root/rtl/vpu/otpu_fpipe.sv $root/rtl/top/otpu_pkg.sv \
  $root/rtl/tech/generic/otpu_tech_ram.sv $root/rtl/vpu/otpu_vtree.sv $root/rtl/vpu/otpu_se_comp.sv \
  $root/rtl/vpu/otpu_se_tail.sv $root/rtl/vpu/otpu_vpu.sv $root/rtl/seq/otpu_seq.sv]
# extra sources replace the default of the same file name (an experiment's copy of a module), or
# are added
foreach e [split $extra ,] {
  if {$e eq ""} { continue }
  set e [file normalize $root/$e]
  set i [lsearch -glob $srcs "*/[file tail $e]"]
  if {$i >= 0} { set srcs [lreplace $srcs $i $i $e] } else { lappend srcs $e }
}
read_verilog -sv $srcs
set ga {}
if {$gens ne "-"} { foreach g [split $gens ,] { lappend ga -generic $g } }
set da [list -verilog_define SYNTHESIS]
if {$defs ne "-"} { foreach d [split $defs ,] { lappend da -verilog_define $d } }
synth_design -top $top -part $part -mode out_of_context -flatten_hierarchy rebuilt {*}$ga {*}$da
create_clock -period $period -name clk [get_ports clk]
report_utilization -file $out/util_synth.txt
report_utilization -hierarchical -hierarchical_depth 6 -file $out/util_hier_synth.txt
report_timing_summary -max_paths 20 -file $out/timing_synth.txt

# the flat cells by name: a cell's group is the first component of its name below the top that is
# not a generate block (g_*) or an instance index, with indices and Vivado's suffixes dropped
proc grp {c} {
  set n [get_property NAME $c]
  set parts [split $n /]
  set out {}
  foreach p $parts {
    lappend out $p
    if {![regexp {^g_} $p] && ![regexp {^gen} $p]} { break }
  }
  set g [join $out /]
  regsub -all {\[[^\]]*\]} $g "" g
  regsub {(_reg|_i_[0-9]+|_carry.*|__[0-9]+|_rep.*|_psdsp.*|_inv)$} $g "" g
  regsub {(_reg|_i_[0-9]+|_carry.*|__[0-9]+|_rep.*|_psdsp.*)$} $g "" g
  return $g
}
proc groups {fn} {
  array set L {}; array set F {}; array set C {}; array set D {}; array set M {}
  foreach c [get_cells -hier -filter {IS_PRIMITIVE}] {
    set t [get_property REF_NAME $c]
    set g [grp $c]
    foreach a {L F C D M} { if {![info exists ${a}($g)]} { set ${a}($g) 0 } }
    if {[regexp {^LUT[1-6]} $t]} { incr L($g) }
    if {[regexp {^FD} $t]} { incr F($g) }
    if {$t eq "CARRY4"} { incr C($g) }
    if {[regexp {^DSP} $t]} { incr D($g) }
    if {[regexp {^(RAM|SRL)} $t]} { incr M($g) }
  }
  set f [open $fn w]
  foreach g [lsort [array names L]] {
    puts $f "OTPU_GROUP $g $L($g) $F($g) $C($g) $D($g) $M($g)"
  }
  close $f
}
groups $out/groups_synth.txt

proc util_lines {fn} {
  set f [open $fn]; set t [read $f]; close $f
  foreach {k pat} {LUT "LUT as Logic" LUTRAM "LUT as Memory" FF "(?:Slice|CLB) Registers" DSP "DSPs" SLICE "Slice" BRAM "Block RAM Tile" CARRY "CARRY4"} {
    if {[regexp -line "^\\|\\s*$pat\\s*\\|\\s*(\[0-9.\]+)" $t -> v]} { puts "OTPU_UTIL $k $v" }
  }
}
if {$impl} {
  opt_design
  place_design
  phys_opt_design
  route_design
  report_utilization -file $out/util.txt
  report_utilization -hierarchical -hierarchical_depth 6 -file $out/util_hier.txt
  report_timing_summary -max_paths 50 -file $out/timing_summary.txt
  groups $out/groups_impl.txt
  util_lines $out/util.txt
  set n 0
  foreach q [get_timing_paths -setup -max_paths 20 -nworst 1] {
    if {$n < 8} {
      puts "OTPU_PATH [get_property SLACK $q] levels [get_property LOGIC_LEVELS $q] [get_property STARTPOINT_PIN $q] -> [get_property ENDPOINT_PIN $q]"
    }
    incr n
  }
  set p [lindex [get_timing_paths -setup -max_paths 1 -nworst 1] 0]
  set h [lindex [get_timing_paths -hold -max_paths 1 -nworst 1] 0]
  puts "OTPU_WNS [get_property SLACK $p] OTPU_WHS [get_property SLACK $h]"
} else {
  util_lines $out/util_synth.txt
  set p [lindex [get_timing_paths -setup -max_paths 1 -nworst 1] 0]
  puts "OTPU_WNS_SYNTH [get_property SLACK $p] [get_property STARTPOINT_PIN $p] -> [get_property ENDPOINT_PIN $p]"
}
puts "OTPU_DONE"
