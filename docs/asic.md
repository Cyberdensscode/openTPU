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
differ where ports meet in a bank. Three copies is 3x the TMEM's bits (the FPGA build has six); a
two-copy TMEM needs the quantizer's row-factor read off the TMEM ports.

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
| stage | global route (detailed route running) | (running) |
| die | 2.65 x 2.65 mm = 7.02 mm² | |
| standard cells | 1.18 mm², 590 K instances (20.8% of the core) | |
| macros | 27, 233 KiB, 1.31 mm² | 113, 1126 KiB, 6.62 mm² |
| register to register | 2.40 ns: **416 MHz** (CTS: 2.37 ns, 422 MHz) | |
| with the ports | 3.12 ns: 320 MHz (worst: `sw_req`) | |
| power at 400 MHz | 0.81 W: sequential 0.28, clock 0.21, combinational 0.16, macros 0.16; leakage 21 mW | |

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

The banked TMEM's logic (bank muxes, copy selection, the write stage) is 19 K um²: its cost is the
three copies' macros.

### Critical paths

nangate45-small after CTS (`tools/asic/paths.py small 300 4_cts`; 91 K endpoints). The worst path
of each kind, and per stage class the worst path out of it (its own logic) and into it:

| kind | endpoints | worst slack | the worst path |
|---|---|---|---|
| io | 435 | -0.749 ns | DMA `is_st` -> the slice's SW port arbitration -> port `sw_req` (31 levels) |
| control | 25 K | 0.133 ns | sequencer `q_pr` -> `c_fp` (50 levels): the dispatch footprint, `fp_ranges` then `fp_seg` |
| control | | 0.293 ns | VPU `ic` -> quantizer block buffer (45 levels): the VPU's issue test (`ic + l < cols`) -> its TMEM request -> the slice's banked arbiter (`gnt_k`) -> the quantizer's grant -> its enables |
| dma / fpipe | 43 K / 8 K | 0.401 / 0.409 ns | DMA stream state (`fs_k`, `is_st`, `busy`) -> the stream engine's handshake -> the VPU's enables (30 levels; it reaches the fp pipes' stage registers, the logic is the enable's) |
| tmem | 7.6 K | 0.426 ns | DMA `is_st` -> TMEM bypass hit (`bh`) |
| sram | 3.9 K | 0.455 ns | DMA chunk buffer macro -> TMEM bank macro (write data) |
| systolic | 3.6 K | 0.690 ns | MXU PAIR column select (`hi0`) -> PE |

| stage class | out of it (its own logic) | into it, from its operands |
|---|---|---|
| otpu_fmul s1 (the significand product) | | 0.619 ns from an fmul's result, 0.683 from an fadd's (19-21 levels): about 1.85 ns |
| otpu_fmul s2 (normalise, round) | 1.262 ns (17 levels) | |
| otpu_fadd s1 -> s2 | 1.052 ns (29 levels) | |
| otpu_fadd s2 -> s3 | 0.829 ns (26 levels) | |
| otpu_fadd s3 -> s4 | 1.217 ns (19 levels) | |
| systolic PE -> PE | 1.142 ns (19 levels) | 1.063 from the weight data |
| systolic column end | 1.086 ns (25 levels) | |
| sequencer | 0.133 ns (50 levels) | |

Into every fp stage register the worst path is an enable (0.41-0.50 ns: the DMA's stream state or
the VPU's issue test through the arbiter), not the arithmetic: the units stall by enable, and the
grant reaches thousands of registers.

**LM / LA.** `otpu_fmul` is two stages and `otpu_fadd` four whatever LM / LA say: a larger LM or LA
appends delay registers after the last stage (`otpu_delay u_pad`), it moves no logic. Raising them
shortens no path unless synthesis retimes (ORFS's `SYNTH_RETIME_MODULES`, experimental), so a
lm3la6 run would time the same as the default and was not run. The fp stages are not the limit
either: their own logic has 0.6-1.3 ns of slack at 2.5 ns (the product stage about 1.85 ns, the
next 1.67 ns), against 0.13 ns for the sequencer and 0.29-0.41 ns for the arbitration and enable
paths. A faster fp datapath (a split multiplier: the XLS pilot's +27..48% clock on the operator)
pays once those are fixed, from about 530 MHz.

### Open problems

- **The ports.** The SW and port B requests (`sw_req`, `sw_addr`, `sw_wdata`, `b_wmask`, ...) leave
  the block combinationally from the DMA's and quantizer's state through the slice's arbitration
  (12-31 levels): with 20% of the period outside, 320 MHz. A register stage at the boundary (as the
  platform's adapter would add) frees them.
- **Sequencer dispatch.** `q_pr -> c_fp` (the footprint of the queued command, 50 levels) is the
  worst register-to-register path; `fp_ranges` and `fp_seg` in two cycles would give it a full one.
- **Grant fanout.** A unit's request -> the banked arbiter -> another unit's grant -> its enables
  (45 levels), and the DMA's stream state -> the VPU's enables: the stall-by-enable design puts a
  high-fanout enable behind the arbitration. A registered grant (a cycle from request to grant) or
  per-unit clock gating is the lever.
- The macros are CACTI-fitted estimates (`tools/asic/fakeram.py`), not a compiler's; a real PDK's
  1R1W macros (IHP SG13G2, commercial) replace them through `rtl/tech/asic`'s leaves.
- Power is vectorless; a VCD from the Verilator RTL (a decode token) would give real activities.
