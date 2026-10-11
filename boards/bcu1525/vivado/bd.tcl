# Block design "otpu_bd" for the BCU1525 (create_project.tcl; boards/ypcb-00338/vivado/bd_native.tcl
# is its model): PCIe (XDMA, Gen3), the core clock, resets, the SYSMON, the control interconnect and
# XDMA's DMA split onto the channels. No memory controllers: the accelerator and XDMA's DMA meet on
# each DDR4 channel in front of its LiteDRAM core (otpu_ld4, one per channel), outside the block
# design (rtl/boards/bcu1525/otpu_fpga_top_bcu.sv). External interfaces:
#   M_AXI_CTL       AXI4-Lite master -> the accelerator's control registers (BAR0 0x00000, core_clk)
#   M_AXI_MEMCAL<d> AXI4-Lite masters -> DIMM d's LiteDRAM core's CSRs (BAR0 0x40000 + d * 0x10000,
#                   64 KB each, core_clk)
#   M_AXI_DMA<j>    AXI4 masters, 128 bits, no IDs, xdma_aclk -> board channel j = 2s + c's bridge
#                   (otpu_mem_ch): XDMA's DMA (512 bits at x16, 256 at x8) through the SmartConnect
#                   sc_dma, which decodes card address bits 32:31 (channel j at j * 2 GiB: what
#                   slice s addresses of its channel c; opentpu.host.board.channel_map) and
#                   converts the width
# and device_temp (the SYSMON's die temperature, 10 bits, core_clk; BAR0 0x30000: its registers).
# The BAR0 layout keeps the YPCB's (control at 0x0, the system monitor at 0x30000, 1 MB), so the
# host's register access is the same; 0x10000 (the YPCB core's CSRs, R_MEMCAL) is left unmapped.
#
# Clocks: the core clock's MMCM (the clock wizard) runs from clk300_in, a 300 MHz board clock already
# on a global buffer (the top: DDR4 channel C1's, SLR1, which also feeds that channel's core).
# xdma_aclk: 250 MHz.
#
# Variables (set before sourcing): CORE_MHZ (default 100, the development builds' clock). Environment:
# PCIE_LANES, 16 (Gen3 x16, 512-bit AXI, the default) or 8 (Gen3 x8, 256 bits).
# Checked (vivado/ip_check.tcl) on a Kintex UltraScale+ part until a VU9P license exists: the
# xcvu9p settings that differ (the GT quad) are applied only on the VU9P.

proc ip_vlnv {name} {
  set defs [lsort -decreasing [get_ipdefs -all "xilinx.com:ip:${name}:*"]]
  if {[llength $defs] == 0} { error "IP $name not found in this Vivado installation" }
  return [lindex $defs 0]
}

create_bd_design otpu_bd
current_bd_design otpu_bd

# core clock: CORE_MHZ rounded to the MMCM's 1/8 divider steps of its 1200 MHz VCO (300 MHz x 4)
if {![info exists CORE_MHZ]} { set CORE_MHZ 100 }
set VCO 1200
set CORE_DIV [expr {round(double($VCO) / $CORE_MHZ * 8) / 8.0}]
set CORE_MHZ_ACT [format %.3f [expr {double($VCO) / $CORE_DIV}]]
set CORE_HZ [expr {int(floor($VCO * 1.0e6 / $CORE_DIV))}]
puts "core_clk: $CORE_MHZ_ACT MHz (MMCM divide $CORE_DIV)"
set PCIE_LANES [expr {[info exists ::env(PCIE_LANES)] && $::env(PCIE_LANES) ne "" ? $::env(PCIE_LANES) : 16}]
if {$PCIE_LANES ni {8 16}} { error "bd.tcl: PCIE_LANES $PCIE_LANES (8 or 16)" }
set AXI_BITS [expr {$PCIE_LANES == 16 ? 512 : 256}]
set XDMA_MHZ 250
set XDMA_HZ [expr {$XDMA_MHZ * 1000000}]
set VU9P [string match -nocase xcvu9p* [get_property PART [current_project]]]
puts "PCIe: Gen3 x$PCIE_LANES, $AXI_BITS-bit AXI, xdma_aclk $XDMA_MHZ MHz (VU9P: $VU9P)"

# ------------------------------------------------------------------ external ports
create_bd_port -dir I -type clk -freq_hz 300000000 clk300_in
set pcie_refclk [create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:diff_clock_rtl:1.0 pcie_refclk]
set_property CONFIG.FREQ_HZ 100000000 $pcie_refclk
create_bd_port -dir I -type rst pcie_perstn
set_property CONFIG.POLARITY ACTIVE_LOW [get_bd_ports pcie_perstn]
create_bd_port -dir O -type clk core_clk
create_bd_port -dir O -type rst core_rstn
create_bd_port -dir O -type clk xdma_aclk
create_bd_port -dir O -type rst xdma_aresetn
create_bd_port -dir O -from 9 -to 0 device_temp
create_bd_port -dir O pcie_link_up

set lites {M_AXI_CTL M_AXI_MEMCAL0 M_AXI_MEMCAL1 M_AXI_MEMCAL2 M_AXI_MEMCAL3}
foreach p $lites {
  set m [create_bd_intf_port -mode Master -vlnv xilinx.com:interface:aximm_rtl:1.0 $p]
  set_property -dict [list CONFIG.PROTOCOL AXI4LITE CONFIG.DATA_WIDTH 32 CONFIG.ADDR_WIDTH 32 \
    CONFIG.HAS_BURST 0 CONFIG.HAS_LOCK 0 CONFIG.HAS_PROT 0 CONFIG.HAS_CACHE 0 CONFIG.HAS_QOS 0 \
    CONFIG.HAS_REGION 0 CONFIG.FREQ_HZ $CORE_HZ] $m
}
set_property CONFIG.ASSOCIATED_BUSIF [join $lites :] [get_bd_ports core_clk]
set_property CONFIG.ASSOCIATED_RESET {core_rstn} [get_bd_ports core_clk]
set_property CONFIG.FREQ_HZ $CORE_HZ [get_bd_ports core_clk]

# ------------------------------------------------------------------ clocks and resets
set clk [create_bd_cell -type ip -vlnv [ip_vlnv clk_wiz] clk_wiz_0]
set_property -dict [list \
  CONFIG.PRIM_IN_FREQ {300.000} CONFIG.PRIM_SOURCE {Global_buffer} \
  CONFIG.USE_RESET {false} CONFIG.USE_LOCKED {true} \
  CONFIG.CLKOUT1_REQUESTED_OUT_FREQ $CORE_MHZ_ACT CONFIG.NUM_OUT_CLKS {1} \
  CONFIG.CLK_OUT1_PORT {core_clk} \
] $clk
connect_bd_net [get_bd_ports clk300_in] [get_bd_pins clk_wiz_0/clk_in1]
connect_bd_net [get_bd_pins clk_wiz_0/core_clk] [get_bd_ports core_clk]
# the wizard's own M / D / O unless they miss CORE_HZ: then VCO / CORE_DIV, as bd_native.tcl
if {[get_property CONFIG.FREQ_HZ [get_bd_pins clk_wiz_0/core_clk]] != $CORE_HZ} {
  set_property -dict [list CONFIG.OVERRIDE_MMCM {true} CONFIG.MMCM_DIVCLK_DIVIDE {1} \
    CONFIG.MMCM_CLKFBOUT_MULT_F [format %.3f [expr {$VCO / 300.0}]] \
    CONFIG.MMCM_CLKOUT0_DIVIDE_F $CORE_DIV] $clk
}
set f [get_property CONFIG.FREQ_HZ [get_bd_pins clk_wiz_0/core_clk]]
if {$f != $CORE_HZ} { error "bd.tcl: the clock wizard makes core_clk at $f Hz, not $CORE_HZ" }
puts "core_clk: $f Hz"

# core reset: until the MMCM locks, and while the host asserts PCIe PERST#
set rst_core [create_bd_cell -type ip -vlnv [ip_vlnv proc_sys_reset] rst_core]
connect_bd_net [get_bd_pins clk_wiz_0/core_clk] [get_bd_pins rst_core/slowest_sync_clk]
connect_bd_net [get_bd_pins clk_wiz_0/locked] [get_bd_pins rst_core/dcm_locked]
connect_bd_net [get_bd_ports pcie_perstn] [get_bd_pins rst_core/ext_reset_in]
connect_bd_net [get_bd_pins rst_core/peripheral_aresetn] [get_bd_ports core_rstn]

# ------------------------------------------------------------------ PCIe: XDMA, Gen3 x16 (x8)
# The YPCB's identity (device 7028, class 1200, subsystem 10EE:4F54: the host's driver binding and
# opentpu/host find the card the same way) and its H2C setting (xdma_rnum_rids 8: docs/host.md,
# "XDMA's H2C overrun"). The x16 lanes are GTY quads 224-227 (SLR1); the VCU1525 board file's
# select_quad is GTY_Quad_227 (the PCIe reference clock's quad 226 sits inside the four).
set ibuf [create_bd_cell -type ip -vlnv [ip_vlnv util_ds_buf] refclk_buf]
set_property CONFIG.C_BUF_TYPE {IBUFDSGTE} $ibuf
connect_bd_intf_net $pcie_refclk [get_bd_intf_pins refclk_buf/CLK_IN_D]

set xdma [create_bd_cell -type ip -vlnv [ip_vlnv xdma] xdma_0]
set cfg [list \
  CONFIG.mode_selection {Advanced} \
  CONFIG.pl_link_cap_max_link_width "X$PCIE_LANES" \
  CONFIG.pl_link_cap_max_link_speed {8.0_GT/s} \
  CONFIG.axi_data_width "${AXI_BITS}_bit" \
  CONFIG.axisten_freq $XDMA_MHZ \
  CONFIG.ref_clk_freq {100_MHz} \
  CONFIG.pf0_device_id {7028} \
  CONFIG.pf0_class_code_base {12} CONFIG.pf0_class_code_sub {00} \
  CONFIG.pf0_class_code_interface {00} \
  CONFIG.pf0_subsystem_vendor_id {10EE} CONFIG.pf0_subsystem_id {4F54} \
  CONFIG.pf0_revision_id {01} \
  CONFIG.xdma_rnum_chnl {1} CONFIG.xdma_wnum_chnl {1} CONFIG.xdma_rnum_rids {8} \
  CONFIG.axilite_master_en {true} CONFIG.axilite_master_size {1} \
  CONFIG.axilite_master_scale {Megabytes} \
  CONFIG.pciebar2axibar_axil_master {0x00000000} \
  CONFIG.xdma_axi_intf_mm {AXI_Memory_Mapped} \
  CONFIG.plltype {QPLL1} \
]
if {$VU9P} { lappend cfg CONFIG.en_gt_selection {true} CONFIG.select_quad {GTY_Quad_227} }
set_property -dict $cfg $xdma
connect_bd_net [get_bd_pins refclk_buf/IBUF_DS_ODIV2] [get_bd_pins xdma_0/sys_clk]
connect_bd_net [get_bd_pins refclk_buf/IBUF_OUT] [get_bd_pins xdma_0/sys_clk_gt]
connect_bd_net [get_bd_ports pcie_perstn] [get_bd_pins xdma_0/sys_rst_n]
make_bd_intf_pins_external [get_bd_intf_pins xdma_0/pcie_mgt]
set_property NAME pcie_mgt [get_bd_intf_ports -of [get_bd_intf_nets -of [get_bd_intf_pins xdma_0/pcie_mgt]]]
connect_bd_net [get_bd_pins xdma_0/user_lnk_up] [get_bd_ports pcie_link_up]
connect_bd_net [get_bd_pins xdma_0/axi_aclk] [get_bd_ports xdma_aclk]
connect_bd_net [get_bd_pins xdma_0/axi_aresetn] [get_bd_ports xdma_aresetn]

# XDMA's DMA onto the four channels: one 128-bit AXI4 master each (otpu_mem_ch's XDMA side), in
# xdma_aclk; INCR bursts of full-width beats (otpu_mem_ch relies on it: no burst type port), in
# order per master (no IDs: otpu_mem_ch answers in order)
set dmas {M_AXI_DMA0 M_AXI_DMA1 M_AXI_DMA2 M_AXI_DMA3}
set scd [create_bd_cell -type ip -vlnv [ip_vlnv smartconnect] sc_dma]
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {4} CONFIG.NUM_CLKS {1}] $scd
connect_bd_net [get_bd_pins xdma_0/axi_aclk] [get_bd_pins sc_dma/aclk]
connect_bd_net [get_bd_pins xdma_0/axi_aresetn] [get_bd_pins sc_dma/aresetn]
connect_bd_intf_net [get_bd_intf_pins xdma_0/M_AXI] [get_bd_intf_pins sc_dma/S00_AXI]
foreach p $dmas {
  set m [create_bd_intf_port -mode Master -vlnv xilinx.com:interface:aximm_rtl:1.0 $p]
  set_property -dict [list CONFIG.PROTOCOL AXI4 CONFIG.DATA_WIDTH 128 CONFIG.ADDR_WIDTH 33 \
    CONFIG.HAS_BURST 0 CONFIG.HAS_LOCK 0 CONFIG.HAS_PROT 0 CONFIG.HAS_CACHE 0 \
    CONFIG.HAS_QOS 0 CONFIG.HAS_REGION 0 CONFIG.FREQ_HZ $XDMA_HZ] $m
  connect_bd_intf_net [get_bd_intf_pins sc_dma/M0[string index $p end]_AXI] $m
}
set_property CONFIG.ASSOCIATED_BUSIF [join $dmas :] [get_bd_ports xdma_aclk]
set_property CONFIG.ASSOCIATED_RESET {xdma_aresetn} [get_bd_ports xdma_aclk]
set_property CONFIG.FREQ_HZ $XDMA_HZ [get_bd_ports xdma_aclk]

# ------------------------------------------------------------------ SYSMON (die temperature)
# UltraScale+: the System Management Wizard (SYSMONE4; the XADC's successor). temp_out is its 10-bit
# code (UG580: T = code * 509.3140064 / 1024 - 280.23087870 with the internal reference), not the
# XADC's 12-bit one: the host takes the formula from the platform (CFG_PLATFORM).
set sysmon [create_bd_cell -type ip -vlnv [ip_vlnv system_management_wiz] sysmon]
set_property -dict [list CONFIG.INTERFACE_SELECTION {Enable_AXI} \
  CONFIG.DCLK_FREQUENCY [expr {int(ceil($CORE_HZ / 1.0e6))}] \
  CONFIG.SEQUENCER_MODE {Default} CONFIG.CHANNEL_ENABLE_VP_VN {false} \
  CONFIG.ENABLE_TEMP_BUS {true} CONFIG.OT_ALARM {false} CONFIG.USER_TEMP_ALARM {false} \
  CONFIG.VCCINT_ALARM {false} CONFIG.VCCAUX_ALARM {false}] $sysmon
connect_bd_net [get_bd_pins clk_wiz_0/core_clk] [get_bd_pins sysmon/s_axi_aclk]
connect_bd_net [get_bd_pins rst_core/peripheral_aresetn] [get_bd_pins sysmon/s_axi_aresetn]
connect_bd_net [get_bd_pins sysmon/temp_out] [get_bd_ports device_temp]

# ------------------------------------------------------------------ control interconnect
set scl [create_bd_cell -type ip -vlnv [ip_vlnv smartconnect] sc_ctl]
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {6} CONFIG.NUM_CLKS {2}] $scl
connect_bd_net [get_bd_pins xdma_0/axi_aclk] [get_bd_pins sc_ctl/aclk]
connect_bd_net [get_bd_pins clk_wiz_0/core_clk] [get_bd_pins sc_ctl/aclk1]
connect_bd_net [get_bd_pins xdma_0/axi_aresetn] [get_bd_pins sc_ctl/aresetn]
connect_bd_intf_net [get_bd_intf_pins xdma_0/M_AXI_LITE] [get_bd_intf_pins sc_ctl/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins sc_ctl/M00_AXI] [get_bd_intf_ports M_AXI_CTL]
connect_bd_intf_net [get_bd_intf_pins sc_ctl/M01_AXI] [get_bd_intf_pins sysmon/S_AXI_LITE]
foreach d {0 1 2 3} {
  connect_bd_intf_net [get_bd_intf_pins sc_ctl/M0[expr {$d + 2}]_AXI] [get_bd_intf_ports M_AXI_MEMCAL$d]
}

# ------------------------------------------------------------------ address map
# BAR0: 0x00000 the control registers (64 KB), 0x30000 the SYSMON, 0x40000-0x7FFFF the LiteDRAM
# cores' CSRs (DIMM d at 0x40000 + d * 0x10000). XDMA's DMA: 8 GiB, M_AXI_DMA<j> at j * 2 GiB (the
# 2 GiB of each channel the slices address; address bits 32:31)
set lite [get_bd_addr_spaces xdma_0/M_AXI_LITE]
assign_bd_address -offset 0x00000000 -range 64K -target_address_space $lite [get_bd_addr_segs M_AXI_CTL/Reg] -force
assign_bd_address -offset 0x00030000 -range 64K -target_address_space $lite \
  [get_bd_addr_segs -of [get_bd_intf_pins sysmon/S_AXI_LITE]] -force
foreach d {0 1 2 3} {
  assign_bd_address -offset [format 0x%08X [expr {0x40000 + $d * 0x10000}]] -range 64K \
    -target_address_space $lite [get_bd_addr_segs M_AXI_MEMCAL$d/Reg] -force
}
foreach j {0 1 2 3} {
  assign_bd_address -offset [format 0x%09X [expr {$j * 0x80000000}]] -range 2G \
    -target_address_space [get_bd_addr_spaces xdma_0/M_AXI] [get_bd_addr_segs M_AXI_DMA$j/Reg] -force
}

validate_bd_design

foreach p [concat {pcie_mgt pcie_refclk} $dmas $lites] {
  if {[get_bd_intf_ports $p] eq ""} { error "block design port $p missing: [get_bd_intf_ports]" }
}
foreach {k want} [list pf0_device_id 7028 pf0_class_code 120000 pf0_subsystem_id 4F54 \
                       pf0_revision_id 01 axisten_freq $XDMA_MHZ \
                       pl_link_cap_max_link_width X$PCIE_LANES axi_data_width ${AXI_BITS}_bit] {
  set got [get_property CONFIG.$k [get_bd_cells xdma_0]]
  if {$got ne $want} { error "xdma_0: $k is $got, expected $want" }
}
foreach p $dmas {
  set w [get_property CONFIG.DATA_WIDTH [get_bd_intf_ports $p]]
  if {$w != 128} { error "$p is $w bits, expected 128 (otpu_mem_ch's XDMA side)" }
  # read-only on a master port (the SmartConnect sets it): the top has no ID signals, and
  # otpu_mem_ch answers in order
  set id [get_property CONFIG.ID_WIDTH [get_bd_intf_ports $p]]
  if {$id != 0} { error "$p has $id ID bits, expected none (otpu_fpga_top_bcu wires no IDs)" }
}
puts "M_AXI_DMA0..3: 128 bits from XDMA's $AXI_BITS"
save_bd_design
