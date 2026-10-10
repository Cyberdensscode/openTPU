# Create the Vivado project for openTPU on the BCU1525 (xcvu9p-fsgd2104-2L-e; docs/bcu1525.md): the
# four DDR4 channels' LiteDRAM cores (boards/bcu1525/litedram/otpu_ld4.v, one instance per channel),
# the block design (bd.tcl: XDMA Gen3, the core clock, the SYSMON) and the board top
# (rtl/boards/bcu1525/otpu_fpga_top_bcu.sv: S slices, two DDR4 channels each).
#   vivado -mode batch -source create_project.tcl -tclargs OUT_DIR MCOLS CORE_MHZ BUILD_ID VPU_CL \
#     LANES ACT_ROWS DSTEP S PLATFORM ARCH_HASH DDR_MTS
# (run_vivado.sh takes them from the target, targets/bcu1525.toml; the RTL list below is
# opentpu.target sources("bcu1525"): tests/test_target.py)
# Environment: PCIE_LANES (bd.tcl), OTPU_FAST (development strategies), OTPU_IP_CACHE.
# The VU9P needs a Vivado license that covers it (not the free tier) and the Virtex UltraScale+
# device files installed.

set here [file normalize [file dirname [info script]]]
set root [file normalize $here/../../..]
set out [expr {[llength $argv] > 0 ? [file normalize [lindex $argv 0]] : "$root/build/vivado_bcu1525"}]
set MCOLS [expr {[llength $argv] > 1 ? [lindex $argv 1] : 4}]
set CORE_MHZ [expr {[llength $argv] > 2 ? [lindex $argv 2] : 100}]
set BUILD_ID [expr {[llength $argv] > 3 ? [lindex $argv 3] : ""}]
set VPU_CL [expr {[llength $argv] > 4 ? [lindex $argv 4] : 2}]
set LANES [expr {[llength $argv] > 5 ? [lindex $argv 5] : 8}]
set ACT_ROWS [expr {[llength $argv] > 6 ? [lindex $argv 6] : $MCOLS}]
set DSTEP [expr {[llength $argv] > 7 ? [lindex $argv 7] : 1}]
set S [expr {[llength $argv] > 8 ? [lindex $argv 8] : 2}]
# PLATFORM, ARCH_HASH: the CFG_PLATFORM / CFG_HASH registers (run_vivado.sh: targets/bcu1525.toml
# through opentpu/target.py; ARCH_HASH 8 hex digits); DDR_MTS: the cores' data rate (mem_mts)
set PLATFORM [expr {[llength $argv] > 9 ? [lindex $argv 9] : 2}]
set ARCH_HASH [expr {[llength $argv] > 10 ? [lindex $argv 10] : 0}]
if {![regexp {^[0-9a-fA-F]{1,8}$} $ARCH_HASH]} { set ARCH_HASH 0 }
set DDR_MTS [expr {[llength $argv] > 11 ? [lindex $argv 11] : 1600}]
set PART xcvu9p-fsgd2104-2L-e
if {$BUILD_ID eq ""} {
  if {[catch {exec git -C $root rev-parse HEAD} BUILD_ID]} { set BUILD_ID 0 }
  set BUILD_ID [string range $BUILD_ID 0 7]
}
if {![regexp {^[0-9a-fA-F]{1,8}$} $BUILD_ID]} { set BUILD_ID 0 }
# the core clock as bd.tcl makes it (1/8 steps of the MMCM's 1200 MHz VCO), for CORE_KHZ
set VCO 1200
set CORE_KHZ [expr {round($VCO * 1000.0 / (round(double($VCO) / $CORE_MHZ * 8) / 8.0))}]
puts "CORE_KHZ $CORE_KHZ, BUILD_ID $BUILD_ID, S $S, PLATFORM $PLATFORM, ARCH_HASH $ARCH_HASH, DDR4-$DDR_MTS"

create_project -force otpu $out -part $PART
if {[info exists ::env(OTPU_IP_CACHE)] && $::env(OTPU_IP_CACHE) ne ""} {
  file mkdir $::env(OTPU_IP_CACHE)
  config_ip_cache -use_cache_location $::env(OTPU_IP_CACHE)
}
set_property target_language Verilog [current_project]

# ---- RTL: the accelerator's core and tech cells (UltraScale+: DSP48E2), the shell shared with the
# YPCB, the board top, the DDR4 cores
set rtl [list \
  rtl/vpu/otpu_fp.sv rtl/vpu/otpu_fpipe.sv rtl/top/otpu_pkg.sv \
  rtl/mem/otpu_tmem.sv rtl/mem/otpu_actram.sv \
  rtl/seq/otpu_seq.sv rtl/vpu/otpu_vtree.sv rtl/dma/otpu_dma.sv \
  rtl/mxu/otpu_mxu.sv rtl/tech/xilinxus/otpu_tech_dsp.sv \
  rtl/vpu/otpu_quant.sv rtl/vpu/otpu_se_comp.sv rtl/vpu/otpu_se_tail.sv rtl/vpu/otpu_vpu.sv \
  rtl/top/otpu_coll.sv rtl/top/otpu_slice.sv \
  rtl/shell/otpu_ctrl.sv rtl/shell/otpu_trace.sv \
  rtl/shell/otpu_board.sv rtl/mem/otpu_native_dram.sv \
  rtl/shell/otpu_afifo.sv rtl/shell/otpu_mem_ch.sv \
  rtl/shell/otpu_axi_split2.sv rtl/boards/bcu1525/otpu_bcu_sys.sv \
  rtl/boards/bcu1525/otpu_fpga_top_bcu.sv]
foreach f $rtl {
  add_files -norecurse $root/$f
  set_property file_type SystemVerilog [get_files $root/$f]
}
set ld $root/boards/bcu1525/litedram
add_files -norecurse $ld/otpu_ld4.v
foreach f [glob $ld/otpu_ld4_*.init] {
  add_files -norecurse $f
  set_property FILE_TYPE {Memory Initialization Files} [get_files $f]
}
set_property verilog_define {SYNTHESIS} [get_filesets sources_1]

# ---- block design
source $here/bd.tcl
make_wrapper -files [get_files otpu_bd.bd] -top
add_files -norecurse [glob $out/otpu.gen/sources_1/bd/otpu_bd/hdl/otpu_bd_wrapper.v]
set_property top otpu_fpga_top_bcu [current_fileset]
set_property generic "S=$S MCOLS=$MCOLS ACT_ROWS=$ACT_ROWS VPU_CL=$VPU_CL LANES=$LANES CORE_KHZ=$CORE_KHZ BUILD_ID=32'h$BUILD_ID DDR_MTS=$DDR_MTS DSTEP=1'b$DSTEP PCIE_LANES=$PCIE_LANES PLATFORM=$PLATFORM ARCH_HASH=32'h$ARCH_HASH" [current_fileset]

# ---- constraints: the pins (scripts/gen_pins.py), the top's clocks and crossings; each DDR4
# core's own (LiteX's false paths, by names inside the core: scoped to otpu_ld4)
set cons $root/boards/bcu1525/constraints
add_files -fileset constrs_1 -norecurse [concat [glob $cons/*.xdc] [list $ld/otpu_ld4.xdc]]
set_property SCOPED_TO_REF otpu_ld4 [get_files $ld/otpu_ld4.xdc]

if {[info exists ::env(OTPU_FAST)] && $::env(OTPU_FAST) eq "1"} {
  set_property strategy {Vivado Synthesis Defaults} [get_runs synth_1]
  set_property strategy {Vivado Implementation Defaults} [get_runs impl_1]
} else {
  set_property strategy Flow_PerfOptimized_high [get_runs synth_1]
  set_property STEPS.SYNTH_DESIGN.ARGS.RETIMING true [get_runs synth_1]
  set_property strategy Performance_ExplorePostRoutePhysOpt [get_runs impl_1]
}
set_property report_strategy {No Reports} [get_runs synth_1]
set_property report_strategy {No Reports} [get_runs impl_1]
puts "project created in $out (BCU1525, S=$S, Gen3 x$PCIE_LANES, DDR4-$DDR_MTS)"
