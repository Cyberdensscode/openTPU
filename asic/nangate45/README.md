# openTPU on NanGate45 (OpenROAD)

The ASIC block (`rtl/asic/otpu_asic_top.sv`: one slice and its collective unit, the slice's memory
ports at the boundary) through OpenROAD-flow-scripts on the NanGate45 library (FreePDK45, a
predictive 45 nm kit: not manufacturable, the usual open PPA reference). `docs/asic.md` has the
design choices and the results.

## Tools

Nothing to install on the host beyond Docker and yosys:

- **OpenROAD-flow-scripts** in Docker: `docker pull --platform linux/amd64 openroad/orfs:latest`
  (6.5 GB; the image is amd64 only, Docker Desktop on Apple silicon runs it under Rosetta: ORFS's
  `gcd` example takes 1.5 minutes). It holds OpenROAD, OpenSTA, yosys with slang and the
  `nangate45` platform (standard cells, the fakeram45 macros).
- **yosys with the slang plugin** on the host (`yosys -m slang`; the oss-cad-suite has it; `YOSYS`
  names another binary): `prep` elaborates the design once to list its RAM tiles.

omarchy has Docker, but its user is not in the `docker` group (adding it needs sudo there:
`sudo usermod -aG docker $USER`, the user's call), so the flow runs on the Mac. Give Docker
Desktop at least 16 GB (the small point peaks near 10 GB in global routing).

## Run

```
python3 tools/asic/flow.py points                  # the design points, their parameters, clock
python3 tools/asic/flow.py all small               # prep + ORFS + report  (build/asic/small/)
python3 tools/asic/flow.py all card                # the card's arch
ORFS_TARGET=cts python3 tools/asic/flow.py run card            # stop after CTS (make target)
ORFS_CLEAN=floorplan python3 tools/asic/flow.py run small      # redo from the floorplan on
python3 tools/asic/flow.py report small            # report.json from the latest stage's reports
python3 tools/asic/paths.py small 300 6_final      # the critical paths by unit and stage
python3 tools/asic/paths.py small 300 4_cts        # the same, after CTS (placement parasitics)
```

`ORFS_CPUS` (default 6) is the container's CPUs and ORFS's NUM_CORES; `ORFS_IMAGE` the image.
`make` resumes from the stage files that exist: an interrupted `run` continues where it stopped
(after a config change, clean from the stage it affects). The sources are the worktree's own: an
RTL file newer than the run's synthesis makes `make` start over from synthesis; to run one stage
on the results there are, name its `do-` target (`ORFS_TARGET=do-finish`, `do-grt`, ...).
`ORFS_VARS` adds ORFS variables to a run (`ORFS_VARS="SKIP_INCREMENTAL_REPAIR=1"`: no repair in
global routing, an hour less on nangate45-small).

Design points (`tools/asic/flow.py` POINTS, from the targets: arch, uarch, `core_mhz`):

| point | target | D | MCOLS | LANES | TMEM words | IMEM words | ACT blocks | FIFO | streams | clock asked |
|---|---|---|---|---|---|---|---|---|---|---|
| small | nangate45-small | 32 | 2 | 4 | 16K | 8K | 32 (2 rows) | 64 | no | 2.5 ns |
| card | nangate45 | 128 | 4 | 8 | 64K | 32K | 128 (4 rows) | 1024 | yes | 2.5 ns |

Both: TMEM_IMPL 1 (banked: three copies of 1R1W bank RAMs), RPB 1, WPB 1, the systolic MXU
(MXU_IMPL 2) with the generic cells (`rtl/tech/generic/otpu_tech_dsp.sv`), WIN 16. `*-lm3la6`:
the same with otpu_fpipe's LM 3 and LA 6 (see `docs/asic.md`: without retiming they move no logic).

## What prep makes (build/asic/POINT/)

- `src/tiles.json`: the RAM tiles (`rtl/tech/asic/otpu_tech_ram.sv` tiles every technology RAM
  onto macros of at most 128 bits x 4096 words).
- `macros/`: one 1R1W (or 1RW) macro per tile size (`tools/asic/fakeram.py`: LEF, Liberty,
  behavioural model, black box).
- `src/otpu_sram_map.sv`: the leaves (`otpu_sram_1r1w`, `otpu_sram_1rw`), one branch per macro
  size.
- `config.mk`, `constraint.sdc`: the ORFS design: the slang front end, hierarchical synthesis
  (the macro placer groups by it), utilisation 35%, one clock (`core_clock`), the ports timed
  against a virtual clock with 20% of the period outside.

## Settings that matter (and why)

- `SYNTH_MEMORY_MAX_BITS 16384`: the remaining flip-flop memories (the quantizer's block buffer,
  the dispatch window, the MXU's drain rows) are above ORFS's 4096-bit default; they stay flops.
- `RTLMP_MIN_CHANNEL_SIZE 60 60`: the channels between macros must be wider than the M4 power
  straps' pitch (56 um), or PDN fails (PDN-0179, "Unable to repair all channels").
- `MACRO_PLACE_HALO 10 10`, `PLACE_DENSITY_LB_ADDON 0.10`, `TNS_END_PERCENT 100`.
- The macros' pins sit on the 0.14 um track grid (tools/asic/fakeram.py `GAP`): an off-grid pin
  stops the macro placer (MPL-0002).

## Outputs

`build/asic/POINT/report.json` (`flow.py report`): the stage reported, `wns_ns` and `fmax_io_mhz`
(every path, the ports' budgets included), `period_min_ns` and `fmax_mhz` (register to register:
`report_clock_min_period`), `power_w` (vectorless, OpenSTA's default activities; by group),
`units` (area per unit after synthesis, with its macros' share), the macros and their area, and
ORFS's metrics per stage. `paths_<stage>.txt*` / `.json` (`paths.py`): the worst paths, every
endpoint's worst slack, the stage classes. ORFS's own logs, reports and results are in `work/`.
