# A first SLR floorplan of the BCU1525 build (docs/bcu1525.md section 10). UNVERIFIED: written
# before a Vivado with UltraScale+ parts was available; the first build's placement and timing
# reports decide what stays. Unmanaged Tcl (cell lookups with checks), read in implementation
# only and late (create_project.tcl).
#
# The VU9P's three SLRs: the PCIe block (GTY quads 224-227) and DIMMs C1 and C2 in SLR1, C0 in
# SLR0, C3 in SLR2. One slice per outer SLR, on the DIMM there and on one of SLR1's (board
# channel j = 2s + c on DIMM C0, C1, C3, C2: otpu_fpga_top_bcu), the shell (XDMA, the control
# block, the collective unit, the trace) in SLR1 with the PCIe block. Each slice's own block RAM
# (its TMEM copies: 78% of an SLR's, docs/bcu1525.md section 7) fills most of its SLR; SLR1 keeps
# the block design, two DDR4 cores and the shell.
#
# What crosses an SLR:
#   - each slice's second channel (board channels 1 and 3): the native port, ~1,150 signals, in
#     core_clk, through two otpu_native_reg (otpu_bcu_sys SLL): u_tx in the slice's SLR, u_rx in
#     SLR1, flip-flops on both sides of the crossing; the read data, write count and error flags
#     leave u_rx from flip-flops into u_tx's (flop to flop: USER_SLL_REG below puts them in the
#     Laguna sites), the commands and write data leave u_tx from its skid buffers' output
#     registers (Laguna TX) into u_rx's skid entry;
#   - the control block's broadcast to the slices (RUN, LOAD, the arguments, the reset; slice 1 has
#     a reset register of its own, g_slice[1].core_rst, in its SLR) and their status back;
#   - the collective unit's TMEM read and write buses (LANES x 32 bits each way per slice).
# At 100 MHz (the development builds) one SLL hop (~1-1.5 ns) fits without more registers; at
# the YPCB's 133 MHz the collective's buses and the control broadcast are the next to register.
# The DDR4 cores sit where their pins are; each otpu_mem_ch is with its DIMM's core (its
# controller-clock side at 200 MHz stays inside the SLR), its core_clk side meets the slice there
# or across the register slices.
proc otpu_pb {name slr cells} {
  set found {}
  foreach c $cells {
    set x [get_cells -quiet $c]
    if {[llength $x]} { lappend found $x } else {
      puts "CRITICAL WARNING: \[bcu1525_floorplan.tcl\] $name: no cell $c"
    }
  }
  set pb [create_pblock $name]
  resize_pblock $pb -add $slr
  if {[llength $found]} { add_cells_to_pblock $pb [join $found] }
  puts "bcu1525_floorplan.tcl: $name in $slr, [llength $found] of [llength $cells] cells"
}
otpu_pb pb_slice0 SLR0 [list u_sys/u_board/u_slice u_sys/u_board/u_mem \
  {u_sys/g_ch[0].u_ch} {u_sys/g_ch[1].g_sll.u_tx} u_ld0 {g_csr[0].u_iso}]
otpu_pb pb_slice1 SLR2 [list {u_sys/u_board/g_slice[1].u_slice} {u_sys/u_board/g_slice[1].u_mem} \
  {u_sys/u_board/g_slice[1].core_rst_reg} {u_sys/g_ch[2].u_ch} {u_sys/g_ch[3].g_sll.u_tx} u_ld3 \
  {g_csr[3].u_iso}]
otpu_pb pb_shell SLR1 [list u_bd u_sys/u_board/u_ctrl u_sys/u_board/u_coll \
  u_sys/u_board/g_trace.u_trace u_sys/u_board/u_iso {u_sys/g_ch[1].u_ch} {u_sys/g_ch[1].g_sll.u_rx} \
  {u_sys/g_ch[3].u_ch} {u_sys/g_ch[3].g_sll.u_rx} u_ld1 u_ld2 {g_csr[1].u_iso} {g_csr[2].u_iso}]

# Laguna registers on the flop-to-flop crossings: the read data, write count and error flags
# (u_rx's s_* registers drive u_tx's directly) and the commands' / write data's output registers
# (u_tx's m_* registers)
foreach j {1 3} {
  set sll {}
  foreach pat [list "u_sys/g_ch\[$j\].g_sll.u_rx/s_rdata_reg*" "u_sys/g_ch\[$j\].g_sll.u_rx/s_wdone_reg*" \
               "u_sys/g_ch\[$j\].g_sll.u_rx/s_err_reg" "u_sys/g_ch\[$j\].g_sll.u_tx/s_rdata_reg*" \
               "u_sys/g_ch\[$j\].g_sll.u_tx/s_wdone_reg*" "u_sys/g_ch\[$j\].g_sll.u_tx/s_err_reg" \
               "u_sys/g_ch\[$j\].g_sll.u_tx/u_cmd/m_data_reg*" "u_sys/g_ch\[$j\].g_sll.u_tx/u_wd/m_data_reg*"] {
    set sll [concat $sll [get_cells -quiet $pat]]
  }
  if {[llength $sll]} {
    set_property USER_SLL_REG TRUE $sll
    puts "bcu1525_floorplan.tcl: channel $j, [llength $sll] Laguna registers"
  } else {
    puts "CRITICAL WARNING: \[bcu1525_floorplan.tcl\] channel $j: no crossing registers found"
  }
}
