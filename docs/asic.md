# openTPU as an ASIC

The ASIC target of the multi-target work (`docs/superpowers/specs/2026-10-10-multi-target-design.md`,
D5 and D6): the same core RTL as the FPGA card, built from SRAM macros and standard cells, through
OpenROAD on NanGate45 for PPA numbers and as a rehearsal for a manufacturable PDK. How to run it:
`asic/nangate45/README.md`.

## The block

`rtl/asic/otpu_asic_top.sv`: one slice (`otpu_slice`) and its collective unit, with the slice's own
memory ports at the boundary (port A words, the SW byte writes, port B chunks, the ready / valid /
idle handshakes), the run control (resets, the loader, the run's arguments) and the status. The
DRAM side (PHY, controller, the channel interleave of `otpu_native_dram`) is the platform's, as
LiteDRAM is the YPCB card's; the block is what every openTPU chip shares.

## Memories

Every large memory of the core goes through a technology RAM (`rtl/tech/<family>/otpu_tech_ram.sv`:
`otpu_ram_1r1w`, `otpu_ram_1r1w_be`, `otpu_ram_1rw`; registered reads with an enable): IMEM, the ACT
RAM's pieces and scales, the DMA's chunk buffer, the MXU's chunk and scale FIFOs, the stream
engine's delay buffer, and the banked TMEM's banks. `rtl/tech/generic` infers them (Verilator,
yosys, the FPGA, with the `ram_style` they had); `rtl/tech/asic` tiles them onto macros of at most
128 bits x 4096 words (depth tiles with a registered read select). Small asynchronous-read RAMs
(the dispatch window, the MXU's drain rows, the VPU's row sums, the quantizer's block buffer) stay
flip-flops.

**TMEM.** The FPGA keeps a copy of TMEM per read port (block RAM is cheap there; reads never
conflict). The ASIC's TMEM is banked (`otpu_tmem` IMPL 1, `otpu_slice` TMEM_IMPL 1): three copies of
LANES 1R1W bank RAMs, each bank one read and one write per cycle (RPB 1, WPB 1), writes to every
copy. A read port sits on one copy (`CPY`): copy 0 MXU, QUANT, VPU A; copy 1 VPU B, QUANT 2; copy 2
DMA, QUANT 3, COLL. The ports of one arbitration group are on different copies, so a group alone
always fits (the quantizer reads three ports at once, the VPU two; with fewer copies they could
never be granted). The slice's arbiter counts per copy and bank (`gnt_k`), and a bank's address
comes from the port granted in it. Results are the same as the FPGA's (the ISA simulator's); cycles
differ where ports meet in a bank: a Qwen3-0.6B decode token on the RTL (`tools/asic/tmem_cycles.py`,
the board's micro-architecture otherwise) takes 4,819,205 cycles banked against 4,815,735 with a
copy per port, +0.07%, with identical results. Three copies is 3x the TMEM's bits (the FPGA build
has six); a two-copy TMEM needs the quantizer's row-factor read off the TMEM ports.

Verified: the RTL suite (`tests/test_rtl.py`, 133) and the model tokens on the RTL (Qwen3-0.6B
int8, fp4 with an int8 head, fp4 PAIR; Qwen3.5-0.8B with and without DSTEP) bit-exact against the
ISA simulator with the ASIC's TMEM (`OTPU_TARGET=nangate45 OTPU_UARCH=board`: 138 passed) and
with the card's (138 passed).

**Macros.** NanGate45 has no memory compiler. `tools/asic/fakeram.py` writes abstract macros in
bsg_fakeram's form (the pins, layers and straps of ORFS's fakeram45 macros) for 1RW and 1R1W
(bsg_fakeram makes 1RW only), any size, with area, clock-to-q, minimum period and clock power fitted
to ORFS's 24 fakeram45 macros (CACTI); a 1R1W macro is 1.5x the 1RW's area. IHP SG13G2 and
commercial nodes have 2-port (1R1W) macros; GF180 has 1RW only, so a 1RW TMEM is a later variant.

## Results

NanGate45 with `tools/asic/fakeram.py`'s macros, ORFS `openroad/orfs:latest` (pulled 2026-10-10),
2.5 ns asked (the targets' `core_mhz` 400), utilisation 35%, the ports with 20% of the period
outside. Two clock figures: **register to register** (`report_clock_min_period`: the core's own),
and **with the ports** (the worst slack including the paths into and out of the block's ports,
whose budget is a guess until the platform around the block exists). Power is OpenSTA's vectorless
estimate (default activities) at 400 MHz. `tools/asic/flow.py report` and `tools/asic/paths.py`
make these numbers from a run. nangate45-small is the reference point: the same run with a native
ORFS of the same date (f395e4b6, OpenROAD 26Q4-110-g91e5f5ff2f) gives the same within the tools'
noise (422 / 341 MHz, 0.79 W, 0 DRC), and its reports are in `asic/nangate45/results/small/`.

| | nangate45-small | nangate45 (the card's arch) |
|---|---|---|
| stage | routed, final report (signoff extraction) | CTS; global routing ends congested (below) |
| die | 2.65 x 2.65 mm = 7.02 mm² | 5.53 x 5.53 mm = 30.6 mm² |
| standard cells | 1.18 mm², 590 K instances (20.8% of the core): 94 K flip-flops 0.43 mm², 373 K gates 0.66 mm² | 4.36 mm², 2.02 M instances (18.3% of the core; 1.76 M after synthesis) |
| macros | 27, 233 KiB, 1.31 mm² | 113, 1126 KiB, 6.62 mm² |
| register to register | 2.39 ns: **419 MHz** (no violation at 2.5 ns) | 5.60 ns: 178 MHz (clock skew 2.7 ns, see below) |
| with the ports | 2.95 ns: 339 MHz (one endpoint: `sw_req`) | 6.40 ns: 156 MHz |
| power at 400 MHz | 0.79 W: sequential 0.28, clock 0.20, combinational 0.15, macros 0.16; leakage 21 mW | 3.34 W: sequential 1.09, clock 0.93, macros 0.87, combinational 0.46; leakage 82 mW |
| route | 16.2 m of wire, 0 DRC violations, 0 antenna violations | not routed |

The card's clock is a physical-design result, not the RTL's: CTS left the macros' clock pins at
4.9-5.0 ns of insertion delay against 2.3 ns for the flip-flops (2.7 ns of skew on a 30 mm² die
with 113 macros; nangate45-small's skew is 0.13 ns), and the worst path pays it: a TMEM bank
macro (launched 2.6 ns late) -> a 1.1 ns wire across the die -> the TMEM read mux -> the DMA's
`dsw` (-3.10 ns). By kind, after CTS (`tools/asic/paths.py card 300 4_cts`, the worst 100 K
endpoints):

| kind | failing endpoints | worst slack | the worst path |
|---|---|---|---|
| io | 1227 | -3.90 ns | DMA `fs_k` -> port `sw_req` |
| sram | 4815 | -3.10 ns | TMEM bank macro -> DMA `dsw`; the TMEM's reads into every unit -2.2 to -2.7 (the macros' late clock) |
| dma | 73.6 K | -1.81 ns | DMA `fs_k` -> the arbitration -> the quantizer's block buffer (the enable fanout, now across a 5.5 mm die) |
| tmem | 491 | -1.77 ns | DMA `fs_k` -> TMEM bypass hit |
| control | 2572 | -1.69 ns | MXU `h` -> the quantizer's block buffer |
| fpipe | 13.1 K | -1.38 ns | DMA `fs_k` -> the quantizer's fadd s2 (an enable) |
| systolic | 0 | +0.41 ns | MXU `hi0` -> PE |

Without the skew the macro paths are about -0.5 ns; the enable fanout (-1.8 ns, no macro in it)
is then the card's limit, as it is the small point's second: on a 30 mm² die it is wire. The fp
stages' own logic still meets 2.5 ns on the card (fmul s1 -> s2 +0.76 ns, s2 -> the next s1
+0.46, fadd s1 -> s2 +0.95, s2 -> s3 +0.92), while every path into them fails through the
enable (-1.31 to -1.38 ns, 57-65 levels from the DMA's `fs_k`, `ds_fill`, `pl`); the stage-class
report stopped there (the container's memory).

**The card's global routing** (native ORFS, 62 GB) ends congested: GRT-0116 after its 30
congestion iterations, 176 K overflow left and not shrinking, 4 h 22 min at a 31.7 GB peak; the
whole run took 7 hours to there. The macro placer packs the ACT RAM's 32 macros (128 x 128,
136 x 137 um, about 270 pins on one edge each) into the die's bottom-left corner, 0.6 x 1.1 mm with
60 um channels. The router's congestion report puts nearly all the overflow along that block's
inner edge and just past it (x 0.4-1.0 mm, y 0-1.4 mm); the nets in its markers are the ACT RAM's
own (141 K), its 4096-bit read data to the MXU (`act_rdata`, 100 K), the MXU's (39 K) and the
block's port B data (16 K). Global placement saw it coming (routability mode: "could not reach
target", weighted congestion 1.60). Spacing the macros does not help: 40 um halos and 150 um
channels give 1.69, and with the macro placer's boundary weight off and the channels on every side
1.67. A lower placement density does: the target is 0.28 while the cells fill under a fifth of the
core, and with `PLACE_DENSITY_LB_ADDON` 0.02 (target 0.22, the macros where they were) global
placement reaches its routability target (weighted congestion under 1.01 with 0.24% of the tiles
overflowed; 1.18 at its end) at the same wire length. Global routing has not been run on it. Next
for the card: route that placement (then the RTL between the ACT RAM and the MXU, the 4096-bit
bus and the ACT RAM's read muxing, if it still congests); balance the macro clock tree (CTS
options, the macros' placement next to their muxes); and a smaller die (35% core utilization
leaves 5.5 mm of die for 10.7 mm² of cells and macros). For
now nangate45-small is the ASIC reference; the card's point is the stress case.

The final power-grid check failed on one row segment (PSM-0069: 26 um boxed in by macros,
narrower than the M4 straps' pitch, its rails unconnected; 14 cells); the run finished with the
IR-drop analysis off. The flow now drops such rows (`asic/nangate45/tapcell.tcl`, checked on this
floorplan: no row narrower than 60 um is left) and no longer fixes hold at the ports (2188 input
endpoints, 12 K hold buffers here); the card's run has both.

Area per unit after synthesis (before placement's buffers), nangate45-small, mm²:

| unit | total | macros | logic |
|---|---|---|---|
| TMEM (banked, 12 x 4096x32) | 1.019 | 1.000 | 0.019 |
| VPU | 0.461 | | 0.461 |
| sequencer (IMEM 2 x 1024x128) | 0.438 | 0.214 | 0.224 |
| MXU (systolic, 32 x 2) | 0.267 | 0.030 | 0.237 |
| quantizer | 0.195 | | 0.195 |
| DMA | 0.072 | 0.037 | 0.035 |
| ACT RAM | 0.031 | 0.026 | 0.005 |
| collective unit | 0.010 | | 0.010 |

nangate45 (the card's arch), mm²: TMEM 4.053 (macros 4.000: 48 x 4096x32), MXU 1.993 (0.951; 128 x 4
systolic, logic 1.042), DMA with the stream engine 1.420 (0.150), VPU 1.161 (0.040), sequencer
1.082 (IMEM 0.858), ACT RAM 0.645 (0.617), quantizer 0.364, collective unit 0.014.

The banked TMEM's logic (bank muxes, copy selection, the write stage) is 19 K um² (small) and 53 K
um² (card): its cost is the three copies' macros, 38% of the card's synthesized area.

### Critical paths

nangate45-small, final (`tools/asic/paths.py small 300 6_final`: 91 K endpoints, signoff
parasitics). The worst path of each kind:

| kind | endpoints | worst slack | the worst path |
|---|---|---|---|
| io | 435 | -0.452 ns | DMA stream state `fs_k` -> the slice's SW port arbitration -> port `sw_req` (27 levels) |
| control | 19 K | 0.114 ns | sequencer `q_pr` -> `c_fp` (50 levels): the dispatch footprint of the queued command, `fp_ranges` then `fp_seg` |
| dma | 52 K | 0.398 ns | DMA stream state `fs_k` -> the TMEM arbitration -> the quantizer's grant -> its enables, into its block buffer (39 levels); the same source reaches the VPU's registers (8.6 K endpoints at 0.57 ns) |
| tmem | 4.1 K | 0.533 ns | DMA `fs_k` -> TMEM bypass hit (`bh`) |
| fpipe | 8.1 K | 0.544 ns | DMA `fs_k` -> the quantizer's fadd s1 (37 levels): an enable again |
| sram | 3.9 K | 0.593 ns | DMA chunk buffer macro -> TMEM bank macro (write data) |
| systolic | 3.6 K | 0.765 ns | MXU PAIR column select (`hi0`) -> PE |

Per stage class, the worst path out of it (its own logic) and into it from its operands:

| stage class | out of it | into it, from its operands |
|---|---|---|
| otpu_fmul s1 (the significand product) | | 0.657 ns from an fmul's result, 0.692 from an fadd's (19-21 levels): about 1.8 ns |
| otpu_fmul s2 (normalise, round) | 1.353 ns (13 levels) | |
| otpu_fadd s1 -> s2 | 1.055 ns (29 levels) | |
| otpu_fadd s2 -> s3 | 0.831 ns (26 levels) | |
| otpu_fadd s3 -> s4 | 1.267 ns (21 levels) | |
| systolic PE -> PE | 1.102 ns (19 levels) | 1.017 from the weight data |
| systolic column end | 1.106 ns (25 levels) | |
| sequencer | 0.114 ns (50 levels) | |

Into every fp stage register the worst path is an enable (0.54-0.59 ns: the DMA's stream state, the
MXU's `al_i` or the VPU's issue test through the arbiter), not the arithmetic: the units stall by
enable, and the grant reaches thousands of registers. After CTS the order was the same except that
the VPU's issue test (`ic` -> the arbiter -> the quantizer's block buffer, 0.29 ns) came second;
routing's repair moved it behind the DMA's.

**LM / LA.** `otpu_fmul` is two stages and `otpu_fadd` four whatever LM / LA say: a larger LM or LA
appends delay registers after the last stage (`otpu_delay u_pad`), it moves no logic. Raising them
shortens no path unless synthesis retimes (ORFS's `SYNTH_RETIME_MODULES`, experimental), so a
lm3la6 run would time the same as the default and was not run. The fp stages are not the limit
either: their own logic has 0.66-1.35 ns of slack at 2.5 ns (the product stage about 1.8 ns,
fadd's second stage 1.67 ns), against 0.11 ns for the sequencer and 0.40-0.60 ns for the
arbitration and enable paths. A faster fp datapath (a split multiplier: the XLS pilot's +27..48%
clock on the operator) pays once those are fixed, from about 540 MHz.

### Open problems

- **The ports.** The SW and port B requests (`sw_req`, `sw_addr`, `sw_wdata`, `b_wmask`, ...) left
  the block combinationally from the DMA's and quantizer's state through the slice's arbitration
  (12-27 levels): with 20% of the period outside, 339 MHz. A register stage at the boundary (as the
  platform's adapter would add) frees them, and every port has one now, on every target (the
  card's too: one design):
  - The SW port's register took the small block at 3.0 ns from 21 routed port violators to none
    ([asic_timing_loop.md](asic_timing_loop.md)).
  - Port B's output register (`b_req` ... `b_par`) and input register (`b_rvalid`, `b_rtag`,
    `b_rdata`). At the card point (a8, after CTS, 2.5 ns) every one of port B's 1,092 output
    endpoints missed, worst `b_wdata` -2.50 ns from the DMA's write-data rotator (18 buffers and
    six MUX2 levels from one select flip-flop), `b_wmask` -1.28, `b_par` -1.25, `b_addr` -0.97,
    `b_req` -0.85. Those figures charge the core clock's 1.9 ns insertion delay to the outputs
    against the ideal virtual I/O clock; with it credited on both sides, `b_wdata` is -0.59 and
    the rest meet (+0.62 ... +1.16), and the inputs read `b_rdata` -> the DMA's line-buffer macros
    -0.04, `b_rvalid` +0.25, `b_rtag` +0.27. The routed small block met at 2.5 ns (`b_wmask`
    +0.09 worst): port B is a card-size problem. The a8 point's global routing overflowed only at
    the DMA's line-buffer macros on the top edge, where 967 `b_wdata` pins and the buffers' read
    data meet; the register also lets the pins' drivers sit by the pins, away from the DMA.
  - Port A's output register (`a_req`, `a_addr`: the MXU's scale reads), the register the card's
    DRAM adapter (`otpu_native_dram`) took them into, moved into the slice: the card has one
    register between the MXU and the adapter's queues as before. At the card point `a_req` was
    -1.62 ns against the ideal I/O clock (+0.25 credited), from the DMA's request through port
    B's grant and the MXU's issue, `a_addr` -0.93 (+0.94), and `a_rdy` and `b_rdy` crossed the
    block to `a_req` (input to output, +0.11 and -0.02); the routed small block met (+0.15). Its
    read data needs no register: the MXU registers `a_rdata` at the port (`a_rvalid` +0.41,
    `a_rdata` +1.73, credited).
  - The registers' valid bits are the contract: a request waits in a register while `sw_req` /
    `a_req` / `b_req` is high, and port B's read data is the block's a cycle after its
    `b_rvalid`. So a platform's "every write the block gave is taken" is `wr_idle && !sw_req &&
    !(b_req && b_we)`, and "no read of the block's in flight" is `rd_idle && !a_req && !(b_req &&
    !b_we)` with no `b_rvalid` the cycle before (as `otpu_board`'s STATUS WR_IDLE and run start;
    port A's requests are reads).
  - `a_rdy` and `b_rdy` still reach the slice's grant logic combinationally (-> the MXU's tag
    count: `b_rdy` -0.30 ns, `a_rdy` -0.18, with the insertion delay credited): a registered
    ready (a credit: the memory's ready a cycle ahead, or a two-entry skid buffer) waits for the
    card's routed timing with the TMEM placed by script.
- **Sequencer dispatch.** `q_pr -> c_fp` (the footprint of the queued command, 50 levels) is the
  worst register-to-register path; `fp_ranges` and `fp_seg` in two cycles would give it a full one.
- **Grant fanout.** The DMA's stream state -> the arbitration -> the quantizer's and the VPU's
  enables (39 levels), a unit's request -> the banked arbiter -> another unit's grant -> its
  enables (43-45): the stall-by-enable design puts a high-fanout enable behind the arbitration. A
  registered grant (a cycle from request to grant) or per-unit clock gating is the lever.
- The macros are CACTI-fitted estimates (`tools/asic/fakeram.py`), not a compiler's; a real PDK's
  1R1W macros (IHP SG13G2, commercial) replace them through `rtl/tech/asic`'s leaves.
- Power is vectorless; a VCD from the Verilator RTL (a decode token) would give real activities.
- **The card's floorplan**: its global routing ends congested next to the ACT RAM's macros; a lower
  placement density clears global placement's estimate, not yet routed (above).
