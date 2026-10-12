# Targets: one design, many boards and chips

openTPU builds for several targets from one source tree: the YPCB-00338 card in production, the
BCU1525 (Virtex UltraScale+ VU9P, four DDR4 DIMMs), ASICs (NanGate45 with OpenROAD now), and
whatever comes next. A target is one file, `targets/<name>.toml`; everything that depends on it
is derived from that file by `opentpu/target.py`. The design rationale is in
`docs/superpowers/specs/2026-10-10-multi-target-design.md`.

```
python3 -m opentpu.target list                  the targets
python3 -m opentpu.target show bcu1525          one, resolved
OTPU_TARGET=bcu1525 python3 ...                 simulators and tools use that target
```

## The four layers of a target

| Section | What | Who depends on it |
|---|---|---|
| `[arch]` | the `opentpu.isasim.Config` fields: S, D, MCOLS, ACT_BLOCKS, ACT_ROWS, TMEM / IMEM words, DRAM bytes per slice, LANES, PAIR / DSTEP / STREAM | the compiler and the programs. Two targets with the same arch run the same programs |
| `[uarch]` | timing-only RTL parameters: MXU_IMPL, VPU_CL, ULANES, TMEM_IMPL, RPB / WPB, WIN, FIFO_DEPTH, the trace buffer | the RTL, the co-simulation |
| `[tech]` | the cell family (`rtl/tech/<family>`: generic, xilinx7, xilinxus, asic), the part or the PDK | the RTL file list |
| `[platform]` | the board: memory type and rate, channels per slice, bytes a slice addresses per channel, the host link, the core clock | the board flow, the host runtime |

Every `[arch]` field is stated: the file is the hardware's truth. `board_config()`, which the
simulators and tests use, takes the target's sizes but keeps its optional features (PAIR,
DSTEP, STREAM) off unless `OTPU_PAIR` / `OTPU_DSTEP` / `OTPU_STREAM` turn them on, as it did
before targets; `opentpu.target.config(name)` is the full truth.

## What is derived

- `target.config(t)`: the `Config`.
- `target.sim_uarch(t)`: the Verilator builds' micro-architecture (`rtlsim.BOARD_UARCH`).
- `target.rtl_params(t)`: the parameters of a board or ASIC top, including the identity the
  control block reports.
- `target.build_config(t, MCOLS=, LANES=, ACT_ROWS=, DSTEP=)`: the Config of an FPGA build
  with the board flow's knobs applied.
- `target.arch_hash(cfg)`: a 32-bit hash of a Config, in the bitstream's CFG_HASH register.
- `python3 -m opentpu.target env NAME`: shell assignments the board flows read
  (`boards/ypcb-00338/run_vivado.sh` takes its knobs' defaults, the platform id and the arch
  hash from them).

## The device reports its configuration

Register map 4 (`docs/observability.md`) adds the CFG_* registers at 0x80..0x9C: slices, TMEM,
IMEM, ACT RAM, DRAM per slice, channels, the micro-architecture, the platform id and the arch
hash. `Board.info()` reads them; `device_config(info)` builds the host's Config from them (map
3 and older bitstreams: the YPCB-00338's values, as before) and refuses a CFG_HASH that is not
the hash of the Config the registers describe. `check_fits(cfg, info)` (BoardBackend) refuses
programs compiled for another MXU, lanes or ACT RAM rows, or needing more slices, TMEM, IMEM,
ACT RAM or DRAM than the device has.

## The RTL's layers

- The core, `rtl/top`, `rtl/mxu`, `rtl/vpu`, `rtl/mem`, `rtl/seq`, `rtl/dma`: no vendor cell.
- `rtl/tech/<family>/`: the same modules in each family, the file list picks one. Today the
  systolic MXU's cells `otpu_pe` and the MX column end `otpu_mxce` (`otpu_tech_dsp.sv`; docs/
  mx_mxu.md section 9): `generic` (behavioural:
  the simulators and the ASIC flows), `xilinx7` (DSP48E1: the YPCB card), `xilinxus` (DSP48E2:
  UltraScale+). `tests/test_tech.py` checks that the xilinx7 cells equal the generic ones cycle
  for cycle on yosys's DSP48E1 model.
- `rtl/shell/`: what the PCIe + native-memory FPGA boards share (control block, trace, the
  channel bridges, XDMA's split, `otpu_board`). The native DRAM adapter is `rtl/mem/otpu_native_dram.sv`.
- `rtl/boards/<board>/`: the board's top and its clock / memory / host wiring.

`opentpu.target.sources(t)` is the file list (`python3 -m opentpu.target files NAME`); the board
Makefile, `create_project.tcl` and `rtlsim.RTL_SOURCES` hold the same lists, which
`tests/test_target.py` checks.

## Adding a target

1. Copy the closest `targets/*.toml`, give it a new name and a new `platform_id`.
2. A new board: its top under `rtl/boards/<board>/`, its flow under `boards/<board>/`
   (constraints, the memory and host IP), reading `python3 -m opentpu.target env <name>`.
3. A new cell family: `rtl/tech/<family>/` with the same module names as `rtl/tech/generic/`.
4. Run the target's arch through the config matrix (`tests/test_config_matrix.py`) and its RTL
   through the token-exact tests (`OTPU_TARGET=<name>`).

## Limits

DRAM addresses are 32 bits per slice, so a slice addresses 4 GiB. A platform with more memory
(the BCU1525's 64 GB) uses it through more slices (each on its own channels); offsets 0xA0 + 4s
are reserved for per-slice DRAM windows the host would set, and far-memory DMA for MoE expert
pools held on the card is a later ISA decision.
