# TMEM in UltraRAM (create_project.tcl with TMEM_URAM=1; run_vivado.sh TMEM_URAM=1): every TMEM
# copy's banks (otpu_tmem g_port[p].g_copy.g_bank[b].mem, 8K x 32 each) as URAM288 instead of block
# RAM. The RTL is unchanged (a synthesis property only), so the simulations' bit-exactness holds
# as it is; the read stays one registered cycle (the bank's q register, URAM's own output
# register off). The yosys estimate (docs/bcu1525.md section 7): a slice's block RAM from 561.5
# BRAM36 tiles to 177.5 (the TMEM's 384 RAMB36E2 freed), 48 URAM288 (yosys packs a bank's two
# 32-bit words per 72-bit row; a depth cascade, two per bank, would be 96).
# UNVERIFIED in Vivado (no UltraScale+ part installed): its timing at the core clock, and that
# every bank maps (the CRITICAL WARNING below otherwise).
set tmem [get_cells -quiet -hierarchical -filter {NAME =~ *u_tmem/g_port*.g_bank*.mem_reg*}]
if {[llength $tmem]} {
  set_property RAM_STYLE ULTRA $tmem
} else {
  puts "CRITICAL WARNING: tmem_uram.xdc: no TMEM bank memories found"
}
