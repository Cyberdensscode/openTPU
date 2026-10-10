# The worst setup paths of a finished ORFS run, one per endpoint (tools/asic/paths.py runs it with
# ORFS's `make run`): PATHS_N paths into PATHS_OUT. Parasitics: the final SPEF when the run wrote
# one, else estimated from the placement.
source $::env(SCRIPTS_DIR)/load.tcl
load_design 6_final.odb 6_final.sdc
if {[file exists $::env(RESULTS_DIR)/6_final.spef]} {
  read_spef $::env(RESULTS_DIR)/6_final.spef
} else {
  estimate_parasitics -placement
}
report_checks -path_delay max -group_path_count $::env(PATHS_N) -endpoint_path_count 1 \
  -format full -digits 3 > $::env(PATHS_OUT)
report_wns > $::env(PATHS_OUT).wns
