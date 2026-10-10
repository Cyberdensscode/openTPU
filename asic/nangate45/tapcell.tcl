# The platform's tap cells (ORFS nangate45 tapcell.tcl), with rows narrower than 60 um dropped:
# a row segment boxed in by macros and narrower than the M4 straps' pitch (56 um) gets no strap,
# its rails float and the final power-grid check fails (PSM-0069: nangate45-small had one, 26 um
# between two ACT RAM scale macros under an MXU FIFO macro).
tapcell \
  -distance 120 \
  -tapcell_master "$::env(TAP_CELL_NAME)" \
  -endcap_master "$::env(TAP_CELL_NAME)" \
  -row_min_width 60
