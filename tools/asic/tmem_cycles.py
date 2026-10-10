"""What the banked TMEM (otpu_tmem IMPL 1, the ASIC's) costs in cycles against the FPGA's TMEM
(a copy per read port), on the same programs: one decode token of a real model (the Verilator
RTL, from the same DRAM state; the results are checked bit for bit against each other).

    PYTHONPATH=. python3 tools/asic/tmem_cycles.py [models/Qwen3-0.6B]

Both runs use the board's micro-architecture (BOARD_UARCH: WIN 16, the 1024-chunk MXU FIFO); the
FPGA's TMEM with RPB covering every read lane and WPB 1, the ASIC's banked TMEM with RPB 1, WPB 1.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from opentpu import rtlsim
from opentpu.isasim import board_config
from opentpu.llm.qwen3 import Engine, Spec, load_weights
from opentpu.llm.rtl_backend import RtlBackend

FPGA = dict(rtlsim.BOARD_UARCH, TMEM_IMPL=0)
ASIC = dict(rtlsim.BOARD_UARCH, TMEM_IMPL=1, RPB=1, WPB=1)


def main(argv: list[str]) -> None:
    path = Path(argv[0] if argv else "models/Qwen3-0.6B")
    spec = Spec.from_hf(path)
    eng = Engine(spec, load_weights(path), cap=256, cfg=board_config(DRAM_BYTES=1 << 30))
    prompt = [151644, 872, 198, 3838, 374, 279, 6722, 315, 9625, 30]
    for t in prompt[:-1]:
        eng.step(t)
    n = eng.image.nbytes
    state = [s.dram[:n].copy() for s in eng.backend.machine.slices]
    out = {}
    for name, ua in (("fpga", FPGA), ("asic", ASIC)):
        rtl = RtlBackend(eng.cfg, [d.copy() for d in state], uarch=ua)
        eng.backend, eng.pos = rtl, len(prompt) - 1
        stats = []
        orig = rtl.run

        def run(programs, args=None, _orig=orig, _stats=stats):
            st = _orig(programs, args)
            _stats.append(st["cycles"])
            return st
        rtl.run = run
        logits = eng.step(prompt[-1])
        out[name] = (sum(stats), logits.view(np.uint32).copy(), [d.copy() for d in rtl.drams])
        print(f"{name}: {sum(stats)} cycles over {len(stats)} runs", flush=True)
    same = np.array_equal(out["fpga"][1], out["asic"][1]) and all(
        np.array_equal(a, b) for a, b in zip(out["fpga"][2], out["asic"][2]))
    f, a = out["fpga"][0], out["asic"][0]
    print(f"{path.name}: banked TMEM {a / f - 1:+.1%} "
          f"cycles ({a} vs {f}); results identical: {same}")
    if not same:
        sys.exit(1)


if __name__ == "__main__":
    main(sys.argv[1:])
