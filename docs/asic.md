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
make these numbers from a run.

| | nangate45-small | nangate45 (the card's arch) |
|---|---|---|
| stage | routed, final report (signoff extraction) | CTS (global routing ran out of memory: 24 GB peak, Docker's VM has 23.4) |
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
is then the card's limit, as it is the small point's second: on a 30 mm² die it is wire. Next for
the card: balance the macro clock tree (CTS options, the macros'
placement next to their muxes), a denser floorplan (35% leaves 5.5 mm of die for 10.7 mm² of
cells and macros), and Docker with more than 24 GB for global routing.

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

- **The ports.** The SW and port B requests (`sw_req`, `sw_addr`, `sw_wdata`, `b_wmask`, ...) leave
  the block combinationally from the DMA's and quantizer's state through the slice's arbitration
  (12-27 levels): with 20% of the period outside, 339 MHz. A register stage at the boundary (as the
  platform's adapter would add) frees them.
- **Sequencer dispatch.** `q_pr -> c_fp` (the footprint of the queued command, 50 levels) is the
  worst register-to-register path; `fp_ranges` and `fp_seg` in two cycles would give it a full one.
- **Grant fanout.** The DMA's stream state -> the arbitration -> the quantizer's and the VPU's
  enables (39 levels), a unit's request -> the banked arbiter -> another unit's grant -> its
  enables (43-45): the stall-by-enable design puts a high-fanout enable behind the arbitration. A
  registered grant (a cycle from request to grant) or per-unit clock gating is the lever.
- The macros are CACTI-fitted estimates (`tools/asic/fakeram.py`), not a compiler's; a real PDK's
  1R1W macros (IHP SG13G2, commercial) replace them through `rtl/tech/asic`'s leaves.
- Power is vectorless; a VCD from the Verilator RTL (a decode token) would give real activities.
