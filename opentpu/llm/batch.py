"""Batched decode steps on the device: one run feeds a token to each of several sequences (an
Engine(batch=B) image's KV slots), every weight streamed once for all of them, and picks each
row's next token on the device, greedy or with the row's own sampling (the decode loop's
sampler, opentpu/llm/generate.py), so the host writes the rows' tokens and reads back one
word per row: no logits leave the card and the host computes nothing of the model (the server's
continuous batching, opentpu/serve/scheduler.py).

The batch area (DRAM past the image, Engine(batch_area=True)):

    tin[B]    the rows' input token ids (fp32 words, the host's)
    tout[B]   the rows' picked token ids (the device's)
    uni[B]    the rows' uniforms (sampled rows; the host's generator of each request)
    per slot: st[STATE_WORDS] (the sampler's words: top-k, log2(e) / T, top-p, the penalty R
              and 1 / R), lg[V] (the sampler's logits), pa[V], pb[V] (the repetition
              penalty's factors over the sequence's context, generate.py's)

A step's program (compile_step) takes compile-time rows (slot, position) and per row a kind:
None (greedy: ARGMAX) or a Sampling key (kmax, penalty); the tokens come from tin (the
embedding rows gathered or loaded on the device from the image's tables, the RoPE rows from
its tables), every layer is qwen3's rows kernel (each row appends its K / V to its slot's cache
and attends over its own positions), and the LM head hands each chunk to every row's sink. The
row's pick goes to tout[r]; a sampled row with a penalty adds it to its slot's context. Per row
the arithmetic is the decode step's (qwen3_rows: bit-identical to step()), and the sampler is
the decode loop's (generate.reference_pick on the logits with the same uniform).
"""
from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np

from .. import isa as I
from .. import language as ol
from ..compiler import CompileError, Tensor, current
from . import generate as G

GREEDY = None


def alloc(image, cfg, B: int, base: int) -> dict:
    """The batch area's addresses for B slots from `base` (bytes; page-aligned end)."""
    V = G._vpad(image.spec)
    a = {"B": B, "base": base, "V": V}
    o = base

    def take(n):
        nonlocal o
        x = o
        o = -(-(o + n) // 128) * 128
        return x
    a["tin"], a["tout"], a["uni"] = take(4 * B), take(4 * B), take(4 * B)
    a["st"] = [take(4 * G.STATE_WORDS) for _ in range(B)]
    a["lg"] = [take(4 * V) for _ in range(B)]
    a["pa"] = [take(4 * V) for _ in range(B)]
    a["pb"] = [take(4 * V) for _ in range(B)]
    a["end"] = -(-o // 4096) * 4096
    return a


def build(images, a: dict, spec, cfg) -> None:
    """The area's first contents in the image arrays: penalty factors 1, the logits buffers
    -inf past this slice's rows (as the decode loop's)."""
    V, v_loc = a["V"], spec.vocab // cfg.S
    ones = np.ones(V, np.float32).view(np.uint8)
    lg = np.full(V, -np.inf, np.float32)
    lg = lg[:-(-v_loc // G.BLK) * G.BLK].view(np.uint8)
    for img in images:
        for s in range(a["B"]):
            img[a["pa"][s]:a["pa"][s] + len(ones)] = ones
            img[a["pb"][s]:a["pb"][s] + len(ones)] = ones
            img[a["lg"][s]:a["lg"][s] + len(lg)] = lg


def kind_of(samp) -> tuple | None:
    """A row's compile-time kind: None (greedy, no penalty) or the Sampling's key."""
    return None if samp is None else samp.key


@ol.jit
def batch_step(m, rows, kinds, block: int, area: dict):
    from .qwen3 import _attention_rows, _embed_word, _layers, _mlp, head_rows_chunk
    b = current()
    spec, R = m.spec, len(rows)
    if R > area["B"]:
        raise CompileError(f"{R} rows for {area['B']} slots")
    if len({s for s, _ in rows}) != R:
        raise CompileError("a step feeds each slot one row")
    x = ol.empty([R, m.xr.shape[1]], dense=True)
    c = ol.empty([R, m.cosr.shape[1]], dense=True)
    s_ = ol.empty([R, m.sinr.shape[1]], dense=True)
    for r, (_, p) in enumerate(rows):
        ol.load(m.cos_t[p:p + 1, :], out=c[r:r + 1, :])
        ol.load(m.sin_t[p:p + 1, :], out=s_[r:r + 1, :])
    tk = ol.load(Tensor(area["tin"], (R,), (1,)))
    for r in range(R):
        x[r:r + 1, :].set(_embed_word(m, tk[r:r + 1]))
    del tk
    for lw in _layers(m):
        x.set(_attention_rows(x, lw, c, s_, rows, spec, block))
        x.set(_mlp(x, lw, spec))
    del c, s_
    # the sinks: one per row
    chunk = area.get("chunk") or head_rows_chunk(R)
    g0 = m.gen
    uni = Tensor(area["uni"], (area["B"],), (1,))
    consts, sts, sinks = {}, [], []
    for r, ((slot, _), kind) in enumerate(zip(rows, kinds)):
        if kind is GREEDY:
            sinks.append(G.Greedy(b, m.v_loc, chunk))
            sts.append(None)
            continue
        if b.S != 1:
            raise CompileError("sampled rows run on one slice")
        samp = _Key(*kind)
        if kind not in consts:
            consts[kind] = G.Sampler.constants(b, g0, samp)
        st = ol.load(Tensor(area["st"][slot], (G.STATE_WORDS,), (1,)))
        g = SimpleNamespace(addr={"lg": area["lg"][slot], "pa": area["pa"][slot],
                                  "pb": area["pb"][slot]}, uni=uni, iota=g0.iota)
        sinks.append(G.Sampler(b, m, g, samp, st, None, consts[kind], u_at=r))
        sts.append(st)
    _head(x, m, spec, sinks, chunk)
    del x
    for r, sk in enumerate(sinks):
        t = sk.token()
        ol.store(Tensor(area["tout"] + 4 * r, (1,), (1,)), t)
        if kinds[r] is not GREEDY:
            sk.after(t)
        del t


def _head(x, m, spec, sinks, chunk: int) -> None:
    """The final norm and this slice's LM head rows for every row of x, each chunk's logits
    handed to every row's sink (qwen3._lm_head_rows with the chunk given)."""
    from ..kernels.lib import rmsnorm
    sid = ol.program_id()
    xs = ol.quantize(rmsnorm(x, ol.load(m.g_final), spec.eps))
    for c0 in range(0, m.v_loc, chunk):
        n = min(chunk, m.v_loc - c0)
        y = ol.dot(xs, m.head[c0:c0 + n, :])
        for r, sk in enumerate(sinks):
            sk(y[r:r + 1, :], sid * m.v_loc + c0)
        del y


class _Key:
    """A Sampling as its compile-time key: kmax and the penalty (the rest are state words)."""

    def __init__(self, kmax: int, pen: bool):
        self.kmax, self.pen = kmax, pen
        self.key = (kmax, pen)


def compile_step(image, area: dict, rows, kinds, block: int) -> list:
    """One program per slice: the batched step for rows [(slot, position)] of kinds."""
    rows, kinds = [(int(s), int(p)) for s, p in rows], list(kinds)
    return [batch_step.trace(image.cfg, s, {"m": image.descriptors(s), "rows": rows,
                                            "kinds": kinds, "block": block, "area": area})
            .finish() for s in range(image.cfg.S)]


def _worker_step(rows, kinds, area):
    """The engine's compile worker process: compile_step, assembled (one slice)."""
    from .qwen3 import _WORKER
    image, block = _WORKER
    return np.asarray(I.assemble(compile_step(image, area, rows, kinds, block)[0]), np.uint32)


class Steps:
    """An Engine's batched steps (Engine(batch=B, batch_area=True)): each slot's sampling
    (set_sampling), the step (run). A step's rows run in groups whose program fits IMEM (the
    LM head's sinks grow it per row: 4 rows of Qwen3-0.6B with sampling and a penalty fit, 8
    greedy ones do not), found by compiling and kept per (kinds, attention buckets); the next
    group's program is compiled while the device runs the current one (on the engine's
    compile worker, or a thread)."""

    def __init__(self, eng):
        if getattr(eng, "bat", None) is None:
            raise ValueError("batched steps need Engine(batch=B > 1, batch_area=True)")
        self.eng, self.a = eng, eng.bat
        self.kinds: dict = {}             # slot -> its kind
        self.limit = eng.cfg.IMEM_WORDS // 8
        self._fit: dict = {}              # (kinds, buckets) -> the program fits IMEM
        self._ready: dict = {}            # key -> programs (compiled for a fit check)
        self._next = None                 # (key, Future)
        self._pool = None
        self.runs = 0
        # where a step's time goes: device cycles, the host's waits for programs (compiles
        # not ready), the whole runs
        self.timing = {"cycles": 0, "compile_wait_s": 0.0, "compiled_inline": 0, "run_s": 0.0}

    def set_sampling(self, slot: int, samp, context=()) -> None:
        """Slot `slot` picks greedily (samp None) or with samp (generate.Sampling): its state
        words, and with a penalty its factors over `context` (the ids so far, the token it
        feeds next among them)."""
        eng, a = self.eng, self.a
        self.kinds[slot] = kind_of(samp)
        if samp is None:
            return
        st = G.state_words(eng.spec, 0, 0, 0, [], eng.block, samp)
        for s in range(eng.cfg.S):
            eng.backend.write(s, a["st"][slot], st)
        if samp.pen:
            V = a["V"]
            pa, pb = np.ones(V, np.float32), np.ones(V, np.float32)
            ix = np.unique(np.asarray(list(context), np.int64))
            pa[ix] = np.float32(1.0) / np.float32(samp.penalty)
            pb[ix] = samp.penalty
            for s in range(eng.cfg.S):
                eng.backend.write(s, a["pa"][slot], pa)
                eng.backend.write(s, a["pb"][slot], pb)

    # ---- programs
    def _key(self, rows):
        return tuple((int(s), int(p)) for s, p in rows), tuple(self.kinds.get(s) for s, _ in rows)

    def _sig(self, key):
        rows, kinds = key
        blk = self.eng.block
        return tuple(sorted(zip(map(repr, kinds), (p // blk for _, p in rows))))

    def _compile(self, key):
        rows, kinds = key
        progs = compile_step(self.eng.image, self.a, rows, kinds, self.eng.block)
        prep = getattr(self.eng.backend, "prepare", None)
        if prep is not None:
            prep(progs)
        return progs

    def plan(self, rows) -> list[list]:
        """The step's rows in groups whose programs fit IMEM."""
        rows = list(rows)
        key = self._key(rows)
        sig = self._sig(key)
        if sig not in self._fit:
            if len(self._ready) > 8:
                self._ready.clear()
            progs = self._compile(key)
            n = len(progs[0]) if isinstance(progs[0], list) else len(progs[0]) // 8
            self._fit[sig] = n <= self.limit
            if self._fit[sig]:
                self._ready[key] = progs
        if self._fit[sig] or len(rows) == 1:
            if not self._fit[sig]:
                raise CompileError(f"one row's step does not fit IMEM ({self.limit})")
            return [rows]
        h = -(-len(rows) // 2)
        return self.plan(rows[:h]) + self.plan(rows[h:])

    def _program(self, key):
        if key in self._ready:
            return self._ready.pop(key)
        t0 = time.perf_counter()
        nxt, self._next = self._next, None
        try:
            if nxt is not None:
                k, fut = nxt
                res = fut.result()
                if k == key:
                    return res
            self.timing["compiled_inline"] += 1
            return self._compile(key)
        finally:
            self.timing["compile_wait_s"] += time.perf_counter() - t0

    def _prefetch(self, key) -> None:
        eng = self.eng
        if key is None or key in self._ready or (self._next is not None and self._next[0] == key):
            return
        if self._next is not None:
            self._next[1].result()
            self._next = None
        if eng._procs and eng._ready is not None and eng._ready.done():
            self._next = (key, eng._pool.submit(_worker_step, key[0], key[1], self.a))
            return
        if not eng.pipeline:
            return
        if self._pool is None:
            from concurrent.futures import ThreadPoolExecutor
            self._pool = ThreadPoolExecutor(1, thread_name_prefix="otpu-batch-compile")
        self._next = (key, self._pool.submit(self._compile, key))

    # ---- the step
    def run(self, rows, tokens, uniforms=None, ahead=True) -> list[int]:
        """Feed tokens[r] to slot rows[r][0] at its position rows[r][1] (the slot's next);
        returns each row's picked token. uniforms: the sampled rows' (one per row). ahead:
        while a group runs, compile the next one (after the last, the next step's first, each
        row one position on)."""
        eng, a = self.eng, self.a
        rows = [(int(s), int(p)) for s, p in rows]
        for s, p in rows:
            if p != eng.poss[s]:
                raise ValueError(f"slot {s} is at {eng.poss[s]}, not {p}")
            if p >= eng.cap:
                raise RuntimeError("KV cache full")
        tok = {s: int(t) for (s, _), t in zip(rows, tokens)}
        uni = None if uniforms is None else {s: u for (s, _), u in zip(rows, uniforms)}
        groups = self.plan(rows)
        nxt_rows = [(s, p + 1) for s, p in rows if p + 1 < eng.cap]
        out = {}
        for gi, grp in enumerate(groups):
            key = self._key(grp)
            progs = self._program(key)
            tin = np.asarray([tok[s] for s, _ in grp], np.float32)
            for s in range(eng.cfg.S):
                eng.backend.write(s, a["tin"], tin)
                if uni is not None:
                    u = np.minimum(np.asarray([uni[x] for x, _ in grp], np.float32),
                                   np.float32(1 - 2.0 ** -24))
                    eng.backend.write(s, a["uni"], u)
            ahead_key = None
            if ahead:
                if gi + 1 < len(groups):
                    ahead_key = self._key(groups[gi + 1])
                elif nxt_rows:
                    k1 = self._key(nxt_rows)
                    if self._fit.get(self._sig(k1)):
                        ahead_key = k1
                    elif self._fit.get(self._sig(k1)) is False:
                        ahead_key = self._key(self.plan(nxt_rows)[0])
            t0 = time.perf_counter()
            start = getattr(eng.backend, "start", None)
            if start is not None:
                start(progs)
                self._prefetch(ahead_key)
                st = eng.backend.wait()
            else:
                st = eng.backend.run(progs)
                self._prefetch(ahead_key)
            self.timing["run_s"] += time.perf_counter() - t0
            self.timing["cycles"] += int((st or {}).get("cycles") or 0)
            eng.stats.append(dict(st or {}, rows=len(grp)))
            self.runs += 1
            got = eng.backend.read(0, a["tout"], 4 * len(grp)).view(np.float32)
            for (s, _), t in zip(grp, got):
                out[s] = int(t)
                eng.poss[s] += 1
        return [out[s] for s, _ in rows]

    def drop(self) -> None:
        """Forget the program compiled ahead."""
        if self._next is not None:
            self._next[1].result()
            self._next = None
