# Timing: the wire, and what flip-flops buy

The question (2026-10-16): the card's clock is limited by route delay, not by logic, so can added
flip-flops fix it, and what else in the flow can improve? This is the exploration's record: where
the delay is (the FPGA card, the ASIC card), what the tools do alone, three batches of
cycle-exact register copies (branch `worktree-timing-ffs`), FPGA builds at 133.33 MHz and at a
150 MHz equivalent, the ASIC flow's timing repair, and (the follow-up question: reduce the
fanout) a census of what the card's nets fan out to and the batches that cut it.

**Short answer.**
- **On the FPGA, flip-flops help where the long wire carries a register that is already a cycle
  old.** A copy per consumer group, each loaded with the same next value, sits by its loads and
  changes no cycle (section 3). With the first batch the card closed 133.33 MHz without r22's ECO.
  But the copies must stay narrow (enables, indices, counters, slots). One wide copy, the ACT
  RAM's write stage per column (+9 k LUTs, +9.5 k flip-flops), congested the 87%-full device and
  made the 150 MHz equivalent worse everywhere (-0.70 -> -1.33 ns), so it is reverted. With the
  narrow copies only (batches 1-3 without it), the 150 MHz-equivalent build goes from main's
  -0.696 ns and 8,473 failing core endpoints to **-0.255 ns and 1,577**: ~129 -> ~140 MHz at the
  clock crossings, ~136 -> ~145 MHz core to core, for 0.3% more LUTs and 2.6% more flip-flops;
  at 133.33 MHz the core then closes without the production strategy too.
- **Flip-flops do not fix a same-cycle decision**: an arbiter's grant into the enables of
  thousands of registers, the TMEM's read arbitration, the sequencer's start selection, the MXU's
  stage enables (`h[acc]`, fanout 11.7 k). Those need another cycle in a protocol (a registered
  grant with a skid, a start a cycle later) or a different structure (section 5). Vivado alone
  (replication, re-placement) reaches ~137 MHz on E3; main's own 150 MHz-equivalent build ~136
  MHz core to core and ~129 MHz at the clk -> clk2x crossing.
- **Most of the card's fanout was LUT RAM addresses, and block RAMs read ahead remove it**
  (section 6): 56% of the loads on nets of 200 or more were LUT RAM address pins. The widest
  memories as block RAMs read at their pointer's next value (a cycle early, so the same cycles)
  and late pushes off LUT RAM write enables: at the 150 MHz equivalent the failing endpoints go
  2,066 -> 820, TNS -177 -> -40 ns, the clock it reaches ~140 -> ~145 MHz, for 7.9 k fewer LUTs
  and 90 more block RAM tiles. A second level of the VPU's enable copies did not help (reverted).
  The MXU's TMEM grant (11.3 k enables) is what is left, and it needs a protocol change.
- **On the ASIC the gain is in the flow, not the RTL.** The card's timing repair never ran: CTS's
  repair saw only ports (it runs before their clock latency is set), and the global route's
  `repair_design` does not finish on 2.5 M nets. With the global route's repair as
  `repair_timing` only (`grt_repair.tcl`), main's card goes from 2.497 to **2.000 ns** register
  to register (400 -> 500 MHz on the route's parasitics) in 100 s of repair; CTS's repair now runs
  after the latency (`cts_latency.tcl`: 1 real violator instead of 1,232 ports). Batch 1 (with
  the ACT RAM copy) cost the ASIC 1.2% of area and 11% of wire for 0.008 ns (section 4).

## 1. Where the delay is

**FPGA (E3, the production m8 build, 039b34d5, 133.33 MHz, WNS +0.007 ns).** The 260 core-clock
paths within 0.2 ns of failing: 0.81 ns of logic and 5.67 ns of route on the median (88% route),
45% through at most one LUT, 55% with a net of fanout 30 or more, 40% crossing units, spanning up
to 246 slice rows. The units alone are not much faster: every recent out-of-context unit run at
7.5 ns had 0.07-0.5 ns of margin (~134-142 MHz).

**ASIC (r4, nangate45, global route without repair, 2.67 ns register to register).** The worst
117 register-to-register paths: 28% of their delay the launching flip-flop's clock-to-Q (weak
flip-flops driving long nets: unrepaired), 37% buffers, 35% logic; 68 of the 117 start at one
register, the MXU's decoded weight scale (`dhi1`) into every column.

The paths by the fix they need:

| class | example | fix |
|---|---|---|
| A: a same-cycle value fanned out | the TMEM pump's write stage, the MXU's decoded scale | a register copy per consumer group (cycle-exact) |
| B1: a registered enable with a huge fanout | the VPU's `en_r` (21 k loads) | a copy per lane, loaded with `en_r`'s next value (cycle-exact) |
| B2: a combinational enable | the MXU's TMEM grant and `h[acc]` -> its stage enables (~11.7 k registers, 8 levels) | a registered grant and a skid buffer (protocol) |
| C: cross-unit arbitration | the MXU's `t_raddr` / `t_ren`, the TMEM's `rpv` (10-13 levels) | a request stage (protocol) |
| D: the sequencer's dispatch and start | `sv`, `sstarted`, `soh` -> `can_start` -> every unit's command register | the write side: a copy (batch 3); the start side: a protocol or a structure |
| E: hard-block hops | a DSP chain's end -> the column end, a block RAM / SRAM macro -> far registers | FPGA: +1 cycle; ASIC: the repair's buffers (section 4) |
| F: IP | XDMA, LiteDRAM, `u_split` at 250 MHz | not ours |

## 2. The tools alone (no RTL change)

On E3's checkpoint before the userclk1 reroute, with the setup checks tightened to a 150 MHz
equivalent (`set_clock_uncertainty -setup` +0.833 ns core to core, +0.417 ns on the clk2x paths;
150 MHz itself has no even MMCM divide for CLK2X at `bd_native.tcl`'s fixed 800 MHz VCO): the routed design at
-0.826 ns (9,485 failing endpoints); post-route phys_opt AggressiveExplore -0.802 (one cell).
Placing again under the tighter checks: placed -0.974, phys_opt -0.706 (~860 flip-flops
replicated), routed -0.665, post-route phys_opt -0.620 ns (13,038 failing): **~137 MHz is what
Vivado's replication and re-placement reach alone.** What fails there: the sequencer (5,787
endpoints), the VPU's `en_r` (1,079-1,164), the MXU's `t_raddr`; 84% with fanout >= 30, 86%
route-dominated. Vivado replicates the driver register but not the logic behind it, and its
replicas are placed by the loads it sees, not by the consumer groups of the design.

## 3. Flip-flops in the RTL: copies per consumer

Every copy is loaded with the same next value as the register it copies (so the design's cycles
are unchanged), kept (`(* keep_hierarchy = "yes" *) otpu_delay`, or a register per generate
slice: identical flip-flops would merge in synthesis), and checked in simulation against the
original where one stays.

**Batch 1 (bc92de2f).**
- The pumped TMEM's write stage a copy per block RAM (`otpu_tmem` `g_pmp` `g_sl`, `otpu_pump_bank`
  W): each RAM's slice of the write data, address and valid.
- The MXU's decoded scale a copy per column (`g_bd`, `dlo1` / `dhi1` / `vlo1` / `vhi1` arrays);
  the result FIFO's write counters a copy per row slice (`g_rfm` `g_s`).
- The VPU's fill index a copy per lane (`g_sfi`; `otpu_se_tail` `fi` per lane).
- The ACT RAM's second hop a copy per column (`g_row` `u_h`). **Reverted (aa3ec024):** the copy
  took the wide write stage with it (the byte enables and bytes `wbe` / `wd`, 1,152 bits, and the
  lanes' placement crossbar, per column): `u_act` 1,603 -> 10,648 LUTs and 1,863 -> 11,404
  flip-flops on the m8 card. Main already gives each RAM piece its own copy of the write address
  and select, which is the narrow part.

**Batch 2 (be935256).** The VPU's pipeline enable a copy per lane (`g_enl`: `en_l`, loaded with
`en_r`'s next value; checked `en_l == {LANES{en_r}}`): the lanes' stages, the SE tail's units
(`cen` per lane) and the composites' lanes (`otpu_se_comp` `en_l`) take their lane's.

**Batch 3 (ec27459e).** With wait counts (`SEQ_WC`, the card and the nangate45 card), the
sequencer's command store is written a cycle after the dispatch, from flip-flops: the enable and
the slot a kept copy per unit, the command one register (`otpu_seq` `g_cmd` `g_wq`). A slot starts
two cycles after its dispatch at the earliest (its ready flags `w_all_q` / `w_dram_q` clear in the
dispatch cycle), so its command is in the store by then; a simulation check stops any start of a
slot whose write is pending. At the 150 MHz equivalent (section 2's run) these were 3,036 LUT RAM
write-address pins (`sv` -> the free slot, -0.49 ns) and 868 write enables (`vout` -> `c_go`,
-0.34 ns) of the sequencer's 5,787 failing endpoints.

**Verification.** Every batch and the final tree (aa3ec024: batches 1-3 without the ACT RAM copy):
`tests/test_rtl.py` 176 passed, 1 skipped (as main); the SE, SE composite, stream, quantizer, fp,
wait-count, models-on-board, target and ASIC flow tests all passed; the m8 card co-simulations
(RTL with LiteDRAM, 133.33 MHz) cycle-exact against main itself: the tiled P run 950,732 cycles
with every busy counter equal, Qwen3.5-0.8B's decode at 544 2,539,392, Qwen3-0.6B's at 544
1,617,739 and at 1800 1,708,259, the MoE decode step (Qwen3.5-35B-A3B) 4,489,345.
`tests/test_target.py` `test_the_long_wire_drivers_are_copies_per_consumer` keeps the copies (and
the ACT RAM's write stage one register).

**FPGA at 133.33 MHz (b_t1: batch 1 with the ACT RAM copy, the production recipe).** Closed without
the ECO: WNS +0.002 ns, WHS +0.033, the clk -> clk2x crossing +0.067 and back +0.050 (r22, main's
RTL, failed clk -> clk2x at -0.202 and needed the block-RAM swap ECO). 195,926 LUTs (65.6%),
174,604 flip-flops, slices 89.8%. Under +0.2 ns: 730 core endpoints (r22: 713); the worst the
sequencer's `soh` into the MXU (+0.002), the DMA's `t_re_q` into the VPU (+0.007, fanout 464),
XDMA, the MXU's DSP chain to the column end (+0.026).

**FPGA at a 150 MHz equivalent** (the production recipe at 133.33 MHz with the uncertainty above
appended to `otpu_clk2x.tcl`; core to core the period it reaches is 6.667 ns - WNS, at the clk2x
crossings 2 x (3.333 ns - WNS)):

| build | main (b_t0_150, 5873d298) | batch 1 with the ACT RAM copy (b_t1_150, bc92de2f) | batches 1-3, narrow (b_t4_150, aa3ec024) |
|---|---|---|---|
| core -> core WNS, failing | -0.696 ns, 8,473 | -1.328 ns, 19,055 | **-0.255 ns, 1,577** |
| core TNS | -1,911 ns | -6,829 ns | -142 ns |
| core -> clk2x / clk2x -> core | -0.531 / -0.446 | -0.529 / -0.528 | -0.241 / -0.225 |
| clk2x -> clk2x | -0.369 | -0.418 | -0.150 |
| core to core reaches | 7.363 ns, 135.8 MHz | 7.995 ns, 125.1 MHz | **6.922 ns, 144.5 MHz** |
| with the crossings | 7.728 ns, 129.4 MHz | 7.724 ns, 129.5 MHz | **7.148 ns, 139.9 MHz** |
| LUTs / flip-flops | 185,403 / 161,219 | 196,565 / 174,503 | 186,027 / 165,387 |
| slices | 87.1% | 89.6% | 88.3% |
| level-5 congestion windows | none | North 111% | North 104% (u_mem 60%, u_mxu 21%) |

The narrow copies take main's 150 MHz-equivalent failures from 8,473 core endpoints to 1,577
and the clock it reaches from ~129 to ~140 MHz (the clk2x crossings, where the pumped TMEM's
write stage per RAM sits), for 624 LUTs and 4,168 flip-flops. One build per configuration, so a
difference of a tenth of a nanosecond could be the placement's; these are several times that.
What is left is flat (-0.15 to
-0.26 ns, every unit): the per-lane enable copies themselves (`g_enl`, 2,553 loads each, 6.2 ns of
route: a second level of copies), the pump RAMs' outputs (4 levels), and the protocol classes of
section 5 (the TMEM's `rpv`, the MXU's `lvm` with fanout 11,778, the sequencer's `w_all_q`).

**Fewer flow crutches (b_t4_def: aa3ec024 at 133.33 MHz, `impl_directives.tcl` empty).** Without
the production strategy (`Performance_ExtraTimingOpt`; build.tcl's own `Explore` directives for
opt, place, phys_opt and route, and its post-route phys_opt, stay) and without r22's ECO, the
core domain closes: core +0.008 ns, clk2x +0.065, the crossings +0.005 / +0.001, WHS +0.012;
186,027 -> 187,606 LUTs, slices 88.3 -> 81.4% (the strategy spreads the placement). The one
failing endpoint is the PCIe block's own 500 MHz path into its TX block RAM (`userclk1`, -0.004
ns): `userclk1_reroute.tcl` took it to +0.029 but another clock to -0.053, so it reopened the
routed design. The strategy is now a margin, not the closure: at 133.33 MHz the copies close the
core either way, and the PCIe IP's path is the reroute hook's (its acceptance rule could weigh a
-0.053 core slack against a failing PCIe clock).

## 4. The ASIC flow

**Main against batch 1 on the same floorplan** (r3's macro placement and blockages, the die
pinned at 4530 um so main's 0.06% smaller synthesis does not move a macro row off the core,
global route without repair, the route's parasitics, 2.5 ns):

| | r4 (archive, 1aad9195) | main (openTPU-t0, 5873d298) | batch 1 (openTPU-t1, bc92de2f) |
|---|---|---|---|
| synthesis | | 2115 s, 9.27 mm² | 3355 s, 9.36 mm² |
| after CTS (design area, macros in) | | 9.47 mm² (46%) | 9.58 mm² (47%) |
| global route wire | | 106.6 m | 118.4 m |
| endpoints | 156,535 | 156,410 | 176,498 |
| register to register | 2.666 ns, 32 failing (all `dhi1` -> `bm2`) | **2.497 ns**, none failing | **2.489 ns**, none failing |
| within 0.1 / 0.2 / 0.3 ns | 62 / 84 / 120 | 4 / 9 / 30 | 2 / 6 / 48 |
| the worst | `u_mxu:dhi1` -> `u_mxu:bm2` | the slice's SRAM -> the DMA's chunk buffer (+0.003) | the slice's SRAM -> a TMEM bank (+0.011) |
| ports | +1.244 | +1.169 | +1.094 |

Main already routes at the clock: the r2.2 cuts and the pinned floorplan took `dhi1` off the
list (r4's limit was one weak, unrepaired flip-flop of it), and what is left near 2.5 ns are SRAM
macro to SRAM macro hops. Batch 1 there (with the ACT RAM copy, since reverted; the narrow
copies alone were not run on the ASIC) costs 1.2% of area, 11% of wire and 20 k flip-flop
endpoints for 0.008 ns: the copies are the FPGA's fix (a register next to its loads where route,
not drive, is the delay); on the ASIC that wire is the repair's (buffering and sizing). Before
they reach the ASIC the copies want a target switch: a parameter, on for the FPGA targets.

**CTS's repair saw only ports.** ORFS's CTS step runs `repair_timing` and then `POST_CTS_TCL`
(`io_latency.tcl`: the ports' virtual clock takes the registers' median clock arrival, 1.95 ns).
Before that the ideal virtual clock charges the insertion delay to every output path: on main's
card the repair found 1,232 violating endpoints, every one a port (worst `b_addr[30]` -0.95 ns),
worked 125 s, resized 1,488 cells and inserted 355 buffers on those paths and stopped (RSZ-0062,
-0.788 ns); r4's did the same. With `io_latency.tcl` sourced first (now `asic/nangate45/cts_latency.tcl`,
`PRE_CTS_TCL`, every point): one violating endpoint (the sequencer's `ir_sync`, -0.12 ns), 61 s, 6
changes, the CTS result's register-to-register period 2.62 -> 2.59 ns (the placement's
parasitics), the same hold repair (786 buffers).

**The global route's repair, as `repair_timing` only.** ORFS's GRT step runs `repair_design`
(every net's slew, capacitance and fanout, single-threaded) before `repair_timing`; on r3 (2.5 M
nets) it ran 8 h before it was stopped, so the card's numbers were unrepaired. With the step's
`repair_design` a no-op (`asic/nangate45/grt_repair.tcl`, `PRE_GLOBAL_ROUTE_TCL`, the card point:
placement has run it) the repair works on the failing paths only. On main's card (t0's CTS
result) with ORFS's `SETUP_SLACK_MARGIN` 0.5 (the repair's target: 2.0 ns): 100 s of repair, 41
buffers, 295 cells upsized, and on the route's parasitics register to register **2.497 -> 2.000
ns** (every register-to-register slack at least +0.500 ns at 2.5), the ports unchanged
(+1.169). The SRAM hops that limited it (+0.003 ns) were unbuffered macro outputs, +0.501 after.
The whole step ~2,380 s (2,176 without the repair; its wall time less the 13 min it was paused
while t1's global route ran), peak 33.9 GB (27.5). With a 1.0 ns margin the repair reached
1.99 ns within ~3,000 moves and then stalled (+0.512-0.514 ns over the next 49,000; the VPU's
floating-point stages, the SE tail's `u_ma` / `u_c`; stopped after 2 h): the floor of this
placement without a pipeline change. These are global route numbers (the guides' parasitics);
detailed routing has not been run on the card.

## 5. What flip-flops do not fix

- **A combinational enable (B2).** The MXU's TMEM grant and its `h[acc]` reach ~11.7 k registers'
  enables in the cycle they are decided (8 levels, the worst core path of main's 150 MHz
  equivalent, -0.70 ns). A registered grant moves the decision a cycle ahead of its use: the unit
  must hold its request's data a cycle longer or take a skid buffer of the cycle's beats.
- **Cross-unit arbitration (C).** The TMEM's read selection takes every reader's address and
  enable in the cycle they are formed (10-13 levels; the MXU's `t_raddr` / `t_ren`, the TMEM's
  `rpv`): a request register in each reader costs a cycle of read latency and the arbiter's view
  of the requests.
- **The sequencer's start (D).** `can_start` into every unit's command register's enable and the
  start slot's read into its data (1,625 endpoints, -0.60 / -0.62 ns at the 150 MHz equivalent)
  are the start decision itself. A per-unit dispatch-order queue would give the in-order units
  (DMA, MXU, quantizer, collective) their next slot from a register (its head), so their command
  read starts at flip-flops; the VPU starts its oldest ready instruction and keeps the age-matrix
  selection.
- **Hard blocks on the FPGA (E).** A DSP chain's end or a block RAM's output into far registers
  takes a cycle of pipelining and its latency in the schedule.
- **Wide data.** A copy of a wide bus per consumer multiplies its registers and the logic that
  feeds them; on a device at 87% of its slices that is congestion, and congestion lengthens every
  route (section 3, the ACT RAM).

## 6. Fanout: what the nets reach, and what was cut

**The census** (`b_t4_def`'s opt_design checkpoint: aa3ec024 at 133.33 MHz; every signal net with 200
or more loads, its loads by pin kind). 562 nets carry 278,123 loads: **LUT RAM address pins
156,954 (56%)**, flip-flop enables 53,987 (19%), LUT inputs 25,723, resets 20,477 (most of them
XDMA's own), LUT RAM write enables 18,024 (6.5%). Over every LUT RAM of the design: 1,535 memories,
357,960 address pins, 39,880 write enables. A LUT RAM takes the whole address on every cell (a
RAM32M holds 6 bits of a 32-deep memory), so a 512-bit queue is a fanout of ~86 per address bit
from its pointer register, and a wide memory's pointer was the widest net of its unit: the DMA's
gather (39,424 address pins), the native adapter's B response FIFO (38,528), the MXU's result FIFO
(37,632), the native adapter's A response RAM (19,152), its B queue (12,960), its SW queue's data
and fill data (11,980 each), the channel bridges' output FIFOs (11,980 each), the sequencer's
command store (10,820). The enables: the MXU's TMEM grant (11,254 enables: its whole epilogue,
held while TMEM does not grant), the quantizer's pipeline enable (8,254), the VPU's per-lane
enable copies (~2,500 each).

**Batch 5 (0a2750a4): the widest LUT RAMs as block RAMs read ahead.** `otpu_ram_1r1w_ra`
(`rtl/tech`, both families) is a LUT RAM's behaviour on a block RAM: `rd` is the word at the read
address register, every write to the last edge in it, from the address's next value (`ra_n`) a
cycle early; generic: a block RAM read at `ra_n` and a registered bypass of the word written at
that edge; asic: flip-flops and a multiplexer at the registered address. `tb_tech_ram` kind 4
checks the two families against each other (a third of its reads the word written at that
edge). The DMA's gather (`g_gb`, at `gh_n`), the B response FIFO (`g_rb`, at `rb_hn`; its tail
copies `rb_tq` gone) and the MXU's result FIFO (`g_rfm`, three copies at `rf_qn + c`) use it; each
next value is checked in simulation against its register. And two enables that held for nothing:
port B's input register in the slice (its readers take it only under `b_rvalid_i`) and TMEM's
input stage address and data with one write port per bank (taken only under `wi_gnt`).

**Batch 6 (8bf19b12): late pushes off LUT RAM write enables.** A queue written on its push puts
the push, decided late, on every cell's write enable. The in-order channel FIFOs (`otpu_afifo` OOO
0: `u_aq`, `u_ad`, `u_xq`, `u_xd`) write the tail slot every cycle there is room (`wready`, the
pointers' registers), and the native adapter's A queue (`qam`, 2 QD slots) its tail every cycle,
as its B queue already did; a push only moves the tail. (A first version took the room a cycle
old: the push the synchronized read pointer had just made room for was then not written; the
board model's turnaround test caught it.) The SW queue's data and fill data (`u_qwd`, `u_wrm`) are
block RAMs read ahead at `qw_fn`. The MXU's ASCALE factors (`alpha`) load every cycle the words are
there, not only at the grant (TMEM holds them until it; checked against a grant-only copy at
every read, and a mutation loading early is caught): 90 of the grant net's 127 failing endpoints
in b_t4_150. `tools/memch_test.py` (the bridge's own bench, func and mutations) passes as before.

**Batch 7 (0c0b16c8), reverted (402bb3bd): the VPU's enable copies per lane and per consumer.**
`g_enl` kept four copies per lane, each loaded with `en_r`'s next value: the lanes' own stages,
the composites, the tail's units, and the tree's lanes (`otpu_vtree` took `en_l` for its row
levels). Each copy's loads went from ~2,700 to ~990, but their routes did not shorten (0 levels:
-0.186 ns in b_t6_150, -0.424 in b_t7_150): a lane's consumers spread as far as the lane does,
so a copy per consumer still crosses the same distance. And the build was worse everywhere, in
units the change does not touch too (core -0.517 ns, 3,794 failing, TNS -575; the sequencer
-0.150 -> -0.517). One build each, so part of that may be the placement's, but nothing in it
shows the second level helping. A copy helps where its consumer group is compact (the MXU's
columns, the pump's RAMs); the VPU's next step is its placement (a pblock per lane), not more
copies.

Every batch: `tests/test_rtl.py` 176 passed, 1 skipped; the SE, composite, tail, stream, quantizer,
fp, wait-count, vops, models-on-board, board, target, tech RAM and ASIC flow tests pass; the five
co-simulations are cycle-exact (950,732 / 2,539,392 / 1,617,739 / 1,708,259 / 4,489,345 cycles, as
main).

**FPGA at the 150 MHz equivalent** (the same recipe and constraints as section 3's table):

| build | b_t4_150 (aa3ec024) | + batch 5 (b_t5_150) | + batch 6 (b_t6_150) | + batch 7 (b_t7_150, reverted) |
|---|---|---|---|---|
| core -> core WNS, failing, TNS | -0.255 ns, 1,577, -142 | -0.236, 1,731, -124 | **-0.202, 732, -36** | -0.517, 3,794, -575 |
| core -> clk2x / clk2x -> core | -0.241 / -0.225 | -0.006 / -0.029 | -0.122 / -0.003 | -0.145 / -0.247 |
| clk2x -> clk2x | -0.150 | +0.075 | -0.050 | -0.106 |
| failing endpoints, all clocks | 2,066 | 1,738 | **820** | 4,136 |
| reaches | 7.148 ns, 139.9 MHz | 6.903 ns, 144.9 MHz | 6.910 ns, 144.7 MHz | 7.184 ns, 139.2 MHz |
| LUTs (as memory) | 186,027 (30,535) | 178,435 (22,204) | 178,168 (20,949) | 178,384 (20,947) |
| flip-flops | 165,387 | 169,067 | 171,043 | 171,105 |
| block RAM tiles | 436 (45.7%) | 496.5 (52.0%) | 526.5 (55.1%) | 526.5 |
| slices | 88.3% | 87.0% | 87.0% | 87.1% |

Batch 5 takes the clk2x crossings, which bounded b_t4_150, from -0.24 to -0.03 ns (~140 -> ~145
MHz), and 8.3 k LUTs of memory out (-27%), for 3.7 k flip-flops (the bypass registers) and 60
block RAM tiles. Batch 6 halves the failing endpoints again and cuts TNS by two thirds; the
worst path moves. In b_t6_150 no failing path touches a LUT RAM. What fails there: the VPU's
per-lane enable copies (1,803 endpoints within 0.2 ns, 0 levels; batch 7 did not help), TMEM's `rpv` into the
VPU (12 levels), the sequencer's dispatch (`sv`, `vout`), the MXU's TMEM grant (`u_rw` `E`, 11,301
loads: its epilogue's stage enables) and the ACT RAM's write stage.

**What fanout is left**: the MXU's grant reaches its whole epilogue (the lanes, `r0` / `r1`, the
fmadd, the delay lines: 11.3 k enables) in the cycle TMEM decides it. It needs the protocol change
of section 5: the epilogue running free into a skid of its depth (a pipeline's worth of entries,
popped one per granted cycle, so the requests TMEM sees stay the same cycles), or a registered
grant. The quantizer's enable (8,254) is on no failing path. The remaining LUT RAMs are the
channel bridges' FIFOs (asynchronous: their reads cross clocks, so a block RAM there is a CDC
change) and the native adapter's A response RAM (read at `ra_h + drop`, two registers' sum).

## 7. Next

- The FPGA: a real build above 133.33 MHz needs an even MMCM divide for CLK2X, which
  `bd_native.tcl`'s fixed 800 MHz VCO (50 MHz x 16) does not have; a VCO parameter would (850
  MHz, M 17: core 141.7 / clk2x 283.3 MHz; 900 MHz, M 18: 150 / 300); then the protocol changes
  below. Land the narrow copies and the fanout batches (through 8bf19b12; 0c0b16c8 is reverted)
  on main for the card, and drop r22's ECO from its recipe.
- The ASIC: run the card's flow (both hooks are its default now) through detailed routing, with
  the clock at 2.0 ns rather than a margin (the global route's 2.0 ns is on the guides'
  parasitics); put the copies behind a target parameter (FPGA on, ASIC off); at 2.0 ns the VPU's
  floating-point stages are next (a deeper `LM` / `LA` moves no logic without retiming,
  docs/asic.md).
- The protocol changes of section 5, in the order of their endpoints at the 150 MHz equivalent:
  the MXU's registered grant / stage enable, the TMEM read request stage, the sequencer's
  dispatch-order queue.
