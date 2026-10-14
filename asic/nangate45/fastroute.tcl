# The card's global-routing layers (ORFS FASTROUTE_TCL, read at the floorplan and kept in the
# design): the platform's fastroute.tcl with metal4 and up at 7% instead of 25%. At 25% the card's
# global route ended with 426 overflowed edges at 31% usage, and the router's own hint (GRT-0704)
# was 7%; at 7% it routes clean (docs/asic.md, "The card's global route"). metal7-10 stay reduced
# by the power grid's straps, so the change is metal4-6's.
set_global_routing_layer_adjustment metal2-metal3 0.5
set_global_routing_layer_adjustment metal4-$::env(MAX_ROUTING_LAYER) 0.07

set_routing_layers -clock $::env(MIN_CLK_ROUTING_LAYER)-$::env(MAX_ROUTING_LAYER)
set_routing_layers -signal $::env(MIN_ROUTING_LAYER)-$::env(MAX_ROUTING_LAYER)
