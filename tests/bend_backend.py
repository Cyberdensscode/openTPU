"""pytest -p bend_backend (with tests/ on the path): every Engine on the default backend (the ISA
simulator) runs on the Bend machine instead (opentpu.bend_sim.BendBackend), host hooks included:
the MoE models' expert serving and Gemma 4's PLE rows, which bend_shadow cannot compare (the hook
moves the host's state, so a run cannot be replayed). The tests' own checks then hold the Bend
machine to their references. A test that reads isasim's own state (eng.backend.machine.slices)
fails with an AttributeError: a harness limit, not a mismatch.

OTPU_BEND_THREADS: the Bend runtime's threads."""
from __future__ import annotations

from opentpu.bend_sim import bend_backend
from opentpu.llm import qwen3

_init = qwen3.Engine.__init__


def _engine_init(self, *a, backend="isa", **kw):
    if isinstance(backend, str) and backend == "isa":
        backend = bend_backend()
    _init(self, *a, backend=backend, **kw)


qwen3.Engine.__init__ = _engine_init
