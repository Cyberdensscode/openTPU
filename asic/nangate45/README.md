# openTPU on NanGate45 (OpenROAD)

The ASIC block (`rtl/asic/otpu_asic_top.sv`: one slice and its collective unit, the slice's memory
ports at the boundary) through OpenROAD-flow-scripts on the NanGate45 library (FreePDK45, a
predictive 45 nm kit: not manufacturable, the usual open PPA reference). See `docs/asic.md` for the
design choices and the results.

## Tools

No install on the host beyond Docker and yosys:

- **OpenROAD-flow-scripts** in Docker: `docker pull --platform linux/amd64 openroad/orfs:latest`
  (6.5 GB; the image is amd64 only, Docker Desktop on Apple silicon runs it under Rosetta: the
  `gcd` example takes 1.5 minutes). It holds OpenROAD, OpenSTA, yosys with slang and the
  `nangate45` platform (standard cells, the fakeram45 macros).
- **yosys with the slang plugin** on the host (`yosys -m slang`; the oss-cad-suite has it): `prep`
  elaborates the design once to list its RAM tiles.

omarchy has Docker but its user is not in the `docker` group (no sudo there for this work), so the
flow runs on the Mac.

## Run

```
python3 tools/asic/flow.py all small          # prep + ORFS + report  (build/asic/small/)
python3 tools/asic/flow.py all card           # the card's arch
python3 tools/asic/paths.py small             # the critical paths, by unit and kind
ORFS_CPUS=8 ORFS_TARGET=route python3 tools/asic/flow.py run card   # stop after routing
```

Design points (`tools/asic/flow.py` POINTS; the lead's `targets/nangate45*.toml` carry the same
values):

| point | D | MCOLS | LANES | TMEM words | IMEM words | ACT blocks | FIFO | streams | clock asked |
|---|---|---|---|---|---|---|---|---|---|
| small | 32 | 2 | 4 | 16K | 8K | 32 | 64 | no | 2.5 ns |
| card | 128 | 4 | 8 | 64K | 32K | 128 (4 rows) | 1024 | yes | 2.5 ns |

Both: TMEM_IMPL 1 (banked: three copies of 1R1W bank RAMs), RPB 1, WPB 1, the systolic MXU
(MXU_IMPL 2) with behavioural cells, WIN 16. `*-lm3la6`: the same with otpu_fpipe's multiplier and
adder one and two stages deeper (LM 3, LA 6).

## What prep makes (build/asic/POINT/)

- `src/tiles.json`: the RAM tiles (`rtl/tech/asic/otpu_tech_ram.sv` tiles every technology RAM
  onto macros of at most 128 bits x 4096 words).
- `macros/`: one 1R1W (or 1RW) macro per tile size (`tools/asic/fakeram.py`: LEF, Liberty,
  model, black box).
- `src/otpu_sram_map.sv`: the leaves, one branch per macro size.
- `src/otpu_mxu_asic.sv`: `otpu_mxu.sv` with the systolic cells' behavioural models (under
  `SYNTHESIS` the file instantiates Xilinx DSP48E1s; once the cells live in
  `rtl/tech/generic/otpu_tech_dsp.sv` the copy is unchanged and that file is used).
- `config.mk`, `constraint.sdc`: the ORFS design (slang front end, hierarchical synthesis for the
  macro placer, utilisation 35%, one clock, 20% of the period for the ports).
