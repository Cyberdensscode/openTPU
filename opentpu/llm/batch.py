"""Batched decode steps on the device: one run feeds a token to each of several sequences (an
Engine(batch=B) image's KV slots), every weight streamed once for all of them, and picks each
row's next token on the device, greedy or with the row's own sampling (the decode loop's
sampler, opentpu/llm/generate.py), so the host writes the rows' tokens and reads back one
word per row: no logits leave the card and the host computes nothing of the model (the server's
continuous batching, opentpu/serve/scheduler.py).

The batch area (DRAM past the image, Engine(batch_area=True)):

    io        tin[B] the rows' input token ids (fp32 words), uni[B] the rows' uniforms (sampled
              rows; the host's generator of each request), step (the run's step, below): the
              host's, one write per run
    tout[B]   the rows' picked token ids (the device's)
    per slot: st[STATE_WORDS] (the sampler's words: top-k, log2(e) / T, top-p, the penalty R
              and 1 / R), lg[V] (the sampler's logits), pa[V], pb[V] (the repetition
              penalty's factors over the sequence's context, generate.py's)

A step's program (compile_step) takes rows (slot, position) and per row a kind: None (greedy:
ARGMAX) or a Sampling key (kmax, penalty); the tokens come from tin (the embedding rows gathered
or loaded on the device from the image's tables, the RoPE rows from its tables), every layer is
qwen3's rows kernel (each row appends its K / V to its slot's cache and attends over its own
positions), and the LM head hands each chunk to every row's sink. The row's pick goes to
tout[r]; a sampled row with a penalty adds it to its slot's context. Per row the arithmetic is
the decode step's (qwen3_rows: bit-identical to step()), and the sampler is the decode loop's
(generate.reference_pick on the logits with the same uniform).

With a span n, the program serves n steps of its rows: row r is at position p_r + t, t the
run-time step (0 <= t < n) in the io block's step word, which the program loads and multiplies
into its argument and address registers (RLD; qwen3.RunWords' way). Every row's position adds
the same t, so the rows share those registers. n keeps every row in its attention block and
V^T tile (Steps.span), so a program serves ~block / (R + 1) steps on average: the host
compiles one per span, ahead, instead of one per step.
"""
from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np

from .. import isa as I
from .. import language as ol
from ..compiler import Affine, CompileError, RunVar, Tensor, current, vt_tile
from . import generate as G

GREEDY = None
FEED = "feed"                   # a row that only feeds its token (a prompt's): no LM head, no pick


def alloc(image, cfg, B: int, base: int) -> dict:
    """The batch area's addresses for B slots from `base` (bytes; page-aligned end)."""
    V = G._vpad(image.spec)
    R = max(B, cfg.MCOLS)                   # rows a run takes (a prompt's feed: MCOLS of a slot)
    a = {"B": B, "R": R, "base": base, "V": V, "cap": image.cap,
         "zmask": (image.lookup or {}).get("zmask", 0)}
    o = base

    def take(n):
        nonlocal o
        x = o
        o = -(-(o + n) // 128) * 128
        return x
    io = take(4 * (2 * R + 1))              # the host's words: one write per run
    a["io"], a["tin"], a["uni"], a["step"] = io, io, io + 4 * R, io + 8 * R
    a["tout"] = take(4 * R)
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


def step_rows(rows, span: int, block: int, zmask: int, cap: int) -> list:
    """rows [(slot, position)] at step 0 -> [(slot, qwen3.RunPos)] at position + t, t the
    run-time step below span, one RunVar for every row (each row's bucket: the blocks up to
    its position's, the last one masked at position + t, as RunPos's)."""
    from ..kernels.attention import Bucket
    from .qwen3 import RunPos
    t = RunVar("tpos", bound=span)
    out = []
    for s, p in rows:
        if (p % block) + span > block:
            raise CompileError(f"position {p} + {span} steps crosses an attention block")
        rp = object.__new__(RunPos)
        rp.blocks, rp.block, rp.lo = p // block + 1, block, p
        rp.t0 = (rp.blocks - 1) * block
        rp.tpos, rp.pos = t + (p - rp.t0), t + p
        rp.bucket = Bucket(rp.blocks, Affine(zmask + 4 * (cap - 1 - p)) + t * -4)
        out.append((s, rp))
    return out


@ol.jit
def batch_step(m, rows, kinds, block: int, area: dict, span: int = 0):
    from .qwen3 import _attention_rows, _embed_word, _layers, _mlp
    b = current()
    spec, R = m.spec, len(rows)
    if R > area["R"]:
        raise CompileError(f"{R} rows for a run of {area['R']}")
    last = {}
    for (s, p), kind in zip(rows, kinds):      # a slot's rows: consecutive positions, all but
        if s in last and (last[s][0] != p - 1 or last[s][1] is not FEED):    # the last feed
            raise CompileError(f"slot {s}: rows at consecutive positions, only its last picks")
        last[s] = (p, kind)
    if span:                                    # the run's step: a TMEM word for the RLDs
        rows = step_rows(rows, span, block, area["zmask"], area["cap"])
        stw = ol.load(Tensor(area["step"], (1,), (1,)))
        at = b.stack[-1][-1]
        b.run_words = {"tpos": stw.base}
    x = ol.empty([R, m.xr.shape[1]], dense=True)
    c = ol.empty([R, m.cosr.shape[1]], dense=True)
    s_ = ol.empty([R, m.sinr.shape[1]], dense=True)
    for r, (_, p) in enumerate(rows):
        q = p.pos if span else p
        ol.load(m.cos_t[q:q + 1, :], out=c[r:r + 1, :])
        ol.load(m.sin_t[q:q + 1, :], out=s_[r:r + 1, :])
    tk = ol.load(Tensor(area["tin"], (R,), (1,)))
    for r in range(R):
        x[r:r + 1, :].set(_embed_word(m, tk[r:r + 1]))
    del tk
    for lw in _layers(m):
        x.set(_attention_rows(x, lw, c, s_, rows, spec, block))
        x.set(_mlp(x, lw, spec))
    del c, s_
    # the sinks: one per row that picks (the others only fed their tokens)
    pick = [r for r, k in enumerate(kinds) if k != FEED]
    if pick:
        _sinks(b, m, spec, x, rows, kinds, pick, area)
    del x
    if span:                                    # the arguments from the step word, at the start
        body = b.stack[-1]
        i = next(j for j, x_ in enumerate(body) if x_ is at) + 1
        body[i:i] = [I.rld(15 - k, b.run_words[v.name], mul=int(c_),
                           comment=f"argument {c_}*{v.name}")
                     for k, (v, c_) in enumerate(b.run_args)]
        b.run_words = None
        del stw


def _sinks(b, m, spec, x, rows, kinds, pick, area) -> None:
    """The LM head for the rows `pick` of x, each chunk to every such row's sink (Greedy, or
    the slot's Sampler with the row's uniform); row r's token to tout[r]."""
    from .qwen3 import head_rows_chunk
    if len(pick) < len(rows):
        xs = ol.empty([len(pick), x.cols], dense=True)
        for i, r in enumerate(pick):
            xs[i:i + 1, :].set(x[r:r + 1, :])
        x = xs
    chunk = area.get("chunk") or head_rows_chunk(len(pick))
    g0 = m.gen
    uni = Tensor(area["uni"], (area["R"],), (1,))
    consts, sts, sinks = {}, [], []
    for r in pick:
        (slot, _), kind = rows[r], kinds[r]
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
    if len(pick) < len(rows):
        del x
    for r, sk in zip(pick, sinks):
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


def compile_step(image, area: dict, rows, kinds, block: int, span: int = 0) -> list:
    """One program per slice: the batched step for rows [(slot, position)] of kinds; with a
    span, for the steps t < span at positions + t (the step word: t)."""
    rows, kinds = [(int(s), int(p)) for s, p in rows], list(kinds)
    return [batch_step.trace(image.cfg, s, {"m": image.descriptors(s), "rows": rows,
                                            "kinds": kinds, "block": block, "area": area,
                                            "span": span})
            .finish() for s in range(image.cfg.S)]


def _worker_step(rows, kinds, area, span=0):
    """The engine's compile worker process: compile_step, assembled (one slice)."""
    from .qwen3 import _WORKER
    image, block = _WORKER
    return np.asarray(I.assemble(compile_step(image, area, rows, kinds, block, span)[0]),
                      np.uint32)


class Steps:
    """An Engine's batched steps (Engine(batch=B, batch_area=True)): each slot's sampling
    (set_sampling), the step (run). A step's rows run in groups whose program fits IMEM (the
    LM head's sinks grow it per row: 4 rows of Qwen3-0.6B with sampling and a penalty fit, 8
    greedy ones do not), found by compiling and kept per (kinds, attention buckets).

    span (default): a program serves its rows for `span(rows)` steps (the run's step in the io
    block), so the same program runs again, loaded in IMEM already, until a row reaches the
    end of its attention block; the next one (each row there) is compiled ahead as soon as a
    program starts serving, on the engine's compile worker or a thread. span=False: one
    program per step (positions at compile time), the next step's compiled while a step runs."""

    def __init__(self, eng, span: bool = True):
        if getattr(eng, "bat", None) is None:
            raise ValueError("batched steps need Engine(batch=B > 1, batch_area=True)")
        self.eng, self.a = eng, eng.bat
        self.span_steps = bool(span)
        self.kinds: dict = {}             # slot -> its kind
        self.limit = eng.cfg.IMEM_WORDS // 8
        self._fit: dict = {}              # signature (kinds, buckets) -> the program fits IMEM
        self._ready: dict = {}            # key -> programs (compiled, not run yet)
        self._next: dict = {}             # key -> Future of its programs
        self._live: dict = {}             # (slots, kinds) -> (positions at step 0, span, programs)
        self._pool = None
        self._words = bool(getattr(eng.backend, "runs_words", False)) and eng.cfg.S == 1
        self.runs = 0
        # where a step's time goes: device cycles, the host's waits for programs (compiles
        # not ready), the whole runs; programs compiled, runs of a program already loaded
        self.timing = {"cycles": 0, "compile_wait_s": 0.0, "compiled_inline": 0, "run_s": 0.0,
                       "programs": 0, "reused": 0}

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
    def span(self, rows) -> int:
        """The steps one program serves from rows' positions: each row stays in its attention
        block and its V^T tile, below the capacity (0: span off)."""
        if not self.span_steps:
            return 0
        eng = self.eng
        blk, tb = eng.block, vt_tile(eng.cap)
        return max(1, min(min(blk - p % blk, tb - p % tb, eng.cap - p) for _, p in rows))

    def _key(self, rows, kinds=None):
        rows = tuple((int(s), int(p)) for s, p in rows)
        kinds = tuple(self.kinds.get(s) for s, _ in rows) if kinds is None else tuple(kinds)
        return rows, kinds, self.span(rows)

    def _sig(self, key):
        rows, kinds, n = key
        blk = self.eng.block
        return tuple(sorted(zip(map(repr, kinds), (p // blk for _, p in rows)))), bool(n)

    def _compile(self, key):
        rows, kinds, n = key
        progs = compile_step(self.eng.image, self.a, rows, kinds, self.eng.block, n)
        if self._words:                 # assembled once: the backend loads the words as they are
            return np.asarray(I.assemble(progs[0]), np.uint32)
        return progs

    def _size(self, progs) -> int:
        return len(progs) // 8 if isinstance(progs, np.ndarray) else len(progs[0])

    def _get(self, key):
        """key's programs: compiled ahead, or now."""
        if key in self._ready:
            return self._ready.pop(key)
        t0 = time.perf_counter()
        try:
            fut = self._next.pop(key, None)
            if fut is not None:
                return fut.result()
            self.timing["compiled_inline"] += 1
            return self._compile(key)
        finally:
            self.timing["compile_wait_s"] += time.perf_counter() - t0

    def plan(self, rows) -> list[list]:
        """The step's rows in groups whose programs fit IMEM."""
        rows = list(rows)
        key = self._key(rows)
        sig = self._sig(key)
        if sig not in self._fit:
            if len(self._ready) > 8:
                self._ready.clear()
            progs = self._get(key)
            self._fit[sig] = self._size(progs) <= self.limit
            if self._fit[sig]:
                self._ready[key] = progs
        if self._fit[sig] or len(rows) == 1:
            if not self._fit[sig]:
                raise CompileError(f"one row's step does not fit IMEM ({self.limit})")
            return [rows]
        h = -(-len(rows) // 2)
        return self.plan(rows[:h]) + self.plan(rows[h:])

    def _prefetch(self, key) -> None:
        """Compile key's programs in the background (if not there or on the way already)."""
        eng = self.eng
        if key is None or key in self._ready or key in self._next:
            return
        if self._fit.get(self._sig(key)) is False:
            return
        while len(self._next) > 4:               # (keys that never came)
            self._next.pop(next(iter(self._next))).result()
        rows, kinds, n = key
        if eng._procs and eng._ready is not None and eng._ready.done():
            self._next[key] = eng._pool.submit(_worker_step, rows, kinds, self.a, n)
            return
        if not eng.pipeline:
            return
        if self._pool is None:
            from concurrent.futures import ThreadPoolExecutor
            self._pool = ThreadPoolExecutor(1, thread_name_prefix="otpu-batch-compile")
        self._next[key] = self._pool.submit(self._compile, key)

    def _serving(self, grp, kinds=None):
        """(programs, t) of a live program that serves grp at its step t, or None."""
        rows = tuple(grp)
        slots = tuple(s for s, _ in rows)
        kinds = tuple(self.kinds.get(s) for s in slots) if kinds is None else tuple(kinds)
        live = self._live.get((slots, kinds))
        if live is None:
            return None
        base, n, progs = live
        d = {p - b for (_, p), b in zip(rows, base)}
        if len(d) != 1:
            return None
        t = d.pop()
        return (progs, t) if 0 <= t < n else None

    def _take(self, grp, kinds=None, adv: int = 1):
        """grp's programs and its step word: the live program's, or a new one made live (and
        its successor compiled ahead: each row where the program's span ends, the rows
        advancing `adv` positions a run)."""
        got = self._serving(grp, kinds)
        if got is not None:
            self.timing["reused"] += 1
            return got
        key = self._key(grp, kinds)
        progs = self._get(key)
        self.timing["programs"] += 1
        rows, kinds, n = key
        slots = tuple(s for s, _ in rows)
        if len(self._live) > 16:
            self._live.clear()
        self._live[(slots, kinds)] = (tuple(p for _, p in rows), n, progs)
        if n:
            d = -(-n // adv) * adv
            nxt = [(s, p + d) for s, p in rows]
            if all(p < self.eng.cap for _, p in nxt):
                self._prefetch(self._key(nxt, kinds))
        return progs, 0

    def feed_rows(self) -> int:
        """Tokens a feed run takes: the MXU's columns (one pass of every weight)."""
        return min(self.eng.cfg.MCOLS, self.a["R"])

    def feed(self, slot: int, tokens) -> None:
        """Feed a prompt's tokens to `slot` at its position, in runs of up to feed_rows() (from a
        multiple of it on, so that a span program serves an attention block's runs): the KV
        cache only, no LM head."""
        eng, k = self.eng, self.feed_rows()
        tokens = [int(t) for t in tokens]
        i = 0
        while i < len(tokens):
            p = eng.poss[slot]
            n = min(len(tokens) - i, k - p % k)
            if p + n > eng.cap:
                raise RuntimeError("KV cache full")
            grp = [(slot, p + j) for j in range(n)]
            self._go(grp, (FEED,) * n, tokens[i:i + n], None, n)
            i += n

    def prefetch_feed(self, slot: int, p: int, n: int) -> None:
        """Compile ahead the feed run of n tokens (< feed_rows()) at p: a prompt's last run."""
        if 0 < n:
            self._prefetch(self._key([(slot, p + j) for j in range(n)], (FEED,) * n))

    def _go(self, grp, kinds, toks, uni, adv, ahead_key=None):
        """One run of grp (its live program or a new one) with tokens toks, uniforms uni (or
        None); while it runs, ahead_key's programs compile. Returns tout's words."""
        eng, a = self.eng, self.a
        progs, t = self._take(grp, kinds, adv)
        Rm = a["R"]
        io = np.zeros(2 * Rm + 1, np.float32)
        io[:len(grp)] = toks
        if uni is not None:
            io[Rm:Rm + len(grp)] = np.minimum(np.asarray(uni, np.float32),
                                              np.float32(1 - 2.0 ** -24))
        io[2 * Rm] = t
        for s in range(eng.cfg.S):
            eng.backend.write(s, a["io"], io)
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
        for s, _ in grp:
            eng.poss[s] += 1
        if kinds is not None and all(k == FEED for k in kinds):
            return None
        return eng.backend.read(0, a["tout"], 4 * len(grp)).view(np.float32)

    # ---- the step
    def run(self, rows, tokens, uniforms=None, ahead=True) -> list[int]:
        """Feed tokens[r] to slot rows[r][0] at its position rows[r][1] (the slot's next);
        returns each row's picked token. uniforms: the sampled rows' (one per row). ahead:
        while a group runs, compile the next group's program (span off: after the last, the
        next step's first, each row one position on)."""
        eng = self.eng
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
            ahead_key = None
            if ahead:
                if gi + 1 < len(groups):
                    if self._serving(groups[gi + 1]) is None:
                        ahead_key = self._key(groups[gi + 1])
                elif nxt_rows and not self.span_steps:
                    k1 = self._key(nxt_rows)
                    if self._fit.get(self._sig(k1)):
                        ahead_key = k1
                    elif self._fit.get(self._sig(k1)) is False:
                        ahead_key = self._key(self.plan(nxt_rows)[0])
            got = self._go(grp, None, [tok[s] for s, _ in grp],
                           None if uni is None else [uni[x] for x, _ in grp], 1, ahead_key)
            for (s, _), t_ in zip(grp, got):
                out[s] = int(t_)
        return [out[s] for s, _ in rows]

    def drop(self) -> None:
        """Forget the programs compiled ahead and the live ones."""
        for fut in self._next.values():
            fut.result()
        self._next.clear()
        self._live.clear()
