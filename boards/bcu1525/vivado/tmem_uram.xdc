# TMEM in UltraRAM (create_project.tcl with TMEM_URAM=1; run_vivado.sh TMEM_URAM=1): every TMEM
# copy's banks (otpu_tmem g_port[p].g_copy.g_bank[b].mem, 8K x 32 each) as URAM288 instead of block
# RAM. The RTL is unchanged (a synthesis property only), so the simulations' bit-exactness holds
# as it is; the read stays one registered cycle (the bank's q register, URAM's own output
# register off). A slice's block RAM goes from ~561 BRAM36 to ~50 (docs/bcu1525.md section 7, the
# yosys estimate with the same marking). UNVERIFIED in Vivado (no UltraScale+ part installed): its
# timing at the core clock, and that every bank maps (the CRITICAL WARNING below otherwise).
set tmem [get_cells -quiet -hierarchical -filter {NAME =~ *u_tmem/g_port*.g_bank*.mem_reg*}]
if {[llength $tmem]} {
  set_property RAM_STYLE ULTRA $tmem
} else {
  puts "CRITICAL WARNING: tmem_uram.xdc: no TMEM bank memories found"
}
