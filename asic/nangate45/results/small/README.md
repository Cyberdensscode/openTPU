# nangate45-small, routed: the reference run

`python3 tools/asic/flow.py all small` on main 596d62fa (2026-10-11), with native
OpenROAD-flow-scripts at f395e4b6 (`ORFS_HOME`; OpenROAD 26Q4-110-g91e5f5ff2f, yosys 0.68+,
KLayout 0.30.0) on 20 x86-64 cores, with `flow.py`'s `config.mk` and no `ORFS_VARS`; 81
minutes in all (global routing 29, detailed routing 14.5), 15 GB of memory at the peak (the final
report).

| | |
|---|---|
| die | 7.02 mm² (2650 x 2650 um), 35% utilization, 27 SRAM macros (233 KiB, 1.31 mm²) |
| instances | 571 K after global routing |
| clock | 2.5 ns asked: register to register 2.37 ns (422 MHz); with the ports' 20% budget 341 MHz (WNS -0.43 ns) |
| power | 0.79 W at 400 MHz, vectorless (internal 0.58, switching 0.19, leakage 0.02; clock 0.20, macros 0.16) |
| DRC | 0 violations after detailed routing |

The same as the run in `docs/asic.md` (ORFS's Docker image of 2026-10-10: 419 / 339 MHz, 0.79 W,
0 DRC) within the tools' run-to-run noise.

- `report.json`: `flow.py report` (area per unit, power, timing, every stage's ORFS metrics)
- `6_finish.rpt`: ORFS's final timing / power / design report
- `final_all.png`: the routed layout

The GDS (585 MB) and the database are not kept here: `flow.py all small` makes them in
`build/asic/small/work/results/`.
