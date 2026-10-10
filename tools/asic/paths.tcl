# The worst setup paths of an ORFS run, one per endpoint (tools/asic/paths.py runs it with ORFS's
# `make run`): PATHS_N paths into PATHS_OUT, from the stage PATHS_STAGE (6_final, or an earlier
# result: 4_cts, 5_route). Parasitics: the final SPEF when the stage is 6_final and the run wrote
# one, else estimated from the placement.
source $::env(SCRIPTS_DIR)/load.tcl
set stage $::env(PATHS_STAGE)
load_design $stage.odb $stage.sdc
if {$stage == "6_final" && [file exists $::env(RESULTS_DIR)/6_final.spef]} {
  read_spef $::env(RESULTS_DIR)/6_final.spef
} else {
  estimate_parasitics -placement
}
report_checks -path_delay max -group_path_count $::env(PATHS_N) -endpoint_path_count 1 \
  -format full -digits 3 > $::env(PATHS_OUT)
# every endpoint's worst path, one line each (start, end, slack): the distribution by unit and stage
report_checks -path_delay max -group_path_count $::env(PATHS_ALL) -endpoint_path_count 1 \
  -format summary -digits 3 > $::env(PATHS_OUT).all
report_wns > $::env(PATHS_OUT).wns
# the fp pipelines' stage registers (otpu_fmul m / r, otpu_fadd s1 / s2 / s3 / r, the delay pads),
# the systolic cells' and the sequencer's: the worst path from and to each class, so a stage's own
# logic shows apart from the enables and operands that reach its registers
set classes {
  fmul.s1 {[/.](u_m|u_c|u_m1|u_m2|u_sq|u_q|u_r|u_c2|u_inv|u_sc|u_y)[/.]m(\[\d+\])*\$}
  fmul.s2 {[/.](u_m|u_c|u_m1|u_m2|u_sq|u_q|u_r|u_c2|u_inv|u_sc|u_y)[/.]r(\[\d+\])*\$}
  fadd.s1 {[/.](u_a|u_acc)[/.]s1(\[\d+\])*\$}
  fadd.s2 {[/.](u_a|u_acc)[/.]s2(\[\d+\])*\$}
  fadd.s3 {[/.](u_a|u_acc)[/.]s3(\[\d+\])*\$}
  fadd.s4 {[/.](u_a|u_acc)[/.]r(\[\d+\])*\$}
  fpipe.pad {[/.]u_pad[/.]}
  pe {[/.]u_pe[/.]}
  colend {[/.]u_ce[/.]}
  seq {^u_slice/u_seq/}
}
set out $::env(PATHS_OUT).stages
close [open $out w]
proc note {out text} { set f [open $out a]; puts $f $text; close $f }
set regs {}
foreach c [all_registers -cells] { lappend regs [list $c [get_full_name $c]] }
foreach {cls re} $classes {
  set cells {}
  foreach r $regs {
    if {[regexp $re [lindex $r 1]]} { lappend cells [lindex $r 0] }
  }
  note $out "### $cls [llength $cells] registers"
  if {[llength $cells] == 0} { continue }
  note $out "### from $cls"
  report_checks -path_delay max -from $cells -format full -digits 3 >> $out
  # into the class: the worst path, then the worst from other start registers (the first is
  # often an enable's fanout, which hides the stage's own logic), three distinct sources
  set skip {}
  set from {}
  foreach r $regs { lappend from [lindex $r 0] }
  for {set k 0} {$k < 3} {incr k} {
    # (the path's start before report_checks: a report frees find_timing_paths' paths)
    set ps [find_timing_paths -path_delay max -from $from -to $cells -group_path_count 1]
    if {[llength $ps] == 0} { break }
    set sp [get_full_name [get_property [lindex $ps 0] startpoint]]
    note $out "### to $cls"
    report_checks -path_delay max -from $from -to $cells -format full -digits 3 >> $out
    regsub {/[A-Za-z0-9_]+$} $sp "" sp
    regsub {(\[\d+\])*(\$_\w+)?$} $sp "" sp
    lappend skip $sp
    set from {}
    foreach r $regs {
      set n [lindex $r 1]
      regsub {(\[\d+\])*(\$_\w+)?$} $n "" n
      if {[lsearch -exact $skip $n] < 0} { lappend from [lindex $r 0] }
    }
  }
}
