"""A mixture-of-experts model bigger than the card's DRAM, its experts streamed from a pool file
on the host (docs/offload.md, path (a)): the Engine, its pool and its expert server's settings,
as otpu-chat and tools/offload/moe_card.py build them.

The card routes every token and computes every expert; the experts it lacks stream from the
pool file (each expert packed once in the card's slot format, tools/offload/pack_pool.py)
into per-layer slots in its DRAM. The host only reads the file and writes DRAM
(opentpu/host/offload.py's ExpertServer, polled while every run is in flight: the prompt's
layer runs or steps, the card's decode loop, a host-sampled step).

    eng = engine("models/Qwen3.5-35B-A3B", cfg, backend, cap=2048,
                 pool="models/Qwen3.5-35B-A3B/pool-fp4.bin")

The settings default to the ones measured best on the card (docs/offload.md 10-13): the slots
filling the DRAM, replaced by decayed use; the prompt layer by layer in runs of two rows with
every slot serving the running layer (Qwen3.5-MoE, Gemma 4 MoE; LFM2-MoE's prompt runs token
by token), and from EM_ROWS new rows a call on each MoE layer's experts in one expert run,
streamed as needs in whole-expert idle parts (expert-major, 13.15, 13.16); a request's disk
reads queued at once; the idle parts read ahead and halt-aware; the checkpoint's pages given
back once the image is written. `defaults(spec)` holds each family's weight formats.

A pool is checked before the image is built (check_pool): it must exist, have this image's
layout and key (moe.pool_key: the quantizer's code, the format, D, the layout, the checkpoint),
and hold every expert unless the checkpoint has the experts to pack the rest. `create=True`
(moe_card's) makes a missing pool and packs each expert on its first use instead.
"""
from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

LAYER_MAJOR = 2         # the prompt's rows a layer run (docs/offload.md 13.7: R = 2, pooled)
NEED_PART = 4 << 20     # expert-major's idle-poll part: a whole expert (13.15's A/B ran 4 MiB)
DEFAULT = "default"     # engine's pool: the checkpoint's own (default_pool)
IDLE_PARTS = ("v1", "ra", "halt", "v2")      # ExpertServer's idle parts (13.12)
# each family's weight formats on the card (wformat, head_format, formats): 4-bit experts with
# an int8 head (docs/offload.md 10); Gemma 4 26B-A4B int8 layers, fp4 experts and head (11.5)
FORMATS = {"lfm2": ("fp4", "int8", None), "qwen35": ("fp4", "int8", None),
           "gemma4": ("int8", "fp4", "experts=fp4")}
# each family's least new prompt rows a call for expert-major (docs/offload.md 13.16): below
# them the layer-major prompt (pooled, a request a run) is as fast on the card or faster. A
# short call has few mixer runs to hide its needs behind: they land mostly while its expert
# runs wait, an idle part an expert (35B 1.06 ms, 26B 1.89 ms) against a request's 0.82 /
# 1.59 ms a miss (session emdef: Qwen3.5-35B-A3B's turns of 24 and 31 new tokens +0.50 /
# +0.39 s TTFT, each a call and the 4 rows after its snapshot; gemma-4-26B-A4B-it's 25 even,
# 31 -0.36 s). Session emdef2 measures the crossover (tools/offload/sessions/emdef2.sh)
EM_ROWS = {"qwen35": 48, "gemma4": 24}


class PoolError(ValueError):
    """An expert pool that cannot serve this image: missing, another image's, or not whole
    beside a checkpoint without the experts."""


def streams(spec) -> bool:
    """A MoE model: its experts stream from a pool into slots."""
    return getattr(spec, "moe", None) is not None


def _family(spec) -> str:
    return type(spec).__module__.rsplit(".", 1)[-1]


def defaults(spec) -> tuple:
    """(wformat, head_format, formats) of a MoE model's family on the card (FORMATS)."""
    return FORMATS.get(_family(spec), ("fp4", "int8", None))


def em_rows_of(spec) -> int:
    """The least new prompt rows a call of a MoE model's family runs expert-major by default
    (EM_ROWS; 0: every call)."""
    return EM_ROWS.get(_family(spec), 0)


def probe(spec, cfg, cap: int, experts: int | None = None, **kw):
    """The image's layout alone (a DRAM too large to fill, no weights): its offload layout and
    expert format, the pool's key, the slots that fit."""
    return spec.image(replace(cfg, DRAM_BYTES=1 << 40), cap, 1, 1,
                      experts=spec.moe.k if experts is None else experts, **kw)


def fit_experts(spec, cfg, cap: int, image=None, **kw) -> int:
    """The expert slots per MoE layer that fill the card's DRAM beside the rest of the image
    (image: its probe, if made already)."""
    L = (image or probe(spec, cfg, cap, **kw)).offload
    return min(spec.moe.E, (cfg.DRAM_BYTES - L.slots[0][0]) // (L.layers * L.pitch))


def expert_major_fits(spec, image, experts: int) -> bool:
    """Whether expert-major's needs always find a slot (docs/offload.md 13.14, 13.15): while
    layer j runs, the server takes no victim from layers j - 1, j and j + 1, whose experts hold
    at most 3 E slots, so the pooled slots less the scratch of the largest chunk (the image's
    prefill rows, a record each) must be at least 3 E. The card's: 35B 1720 - 26 at chat's KV
    capacity (1680 - 26 at moe_card's) against 768, 26B 540 - 17 against 384; a tiny test
    model's 2 slots a layer are not (12 - 20 against 24)."""
    from opentpu.llm.moe import em_record
    L = image.offload
    scratch = -(-image.prefill_rows * em_record(spec.hidden, spec.moe.k) // L.pitch)
    return L.layers * experts - scratch >= 3 * L.E


def default_pool(model_dir, efmt: str) -> Path:
    """Where a checkpoint's pool goes by default: in its directory, pool-<the experts'
    format>.bin (with its .packed, .format and .key beside it)."""
    return Path(model_dir) / f"pool-{efmt}.bin"


def _pack_hint(model, pool, wformat: str, formats: str | None) -> str:
    env = f"OTPU_FORMATS={formats} " if formats else ""
    return (f"pack it from the whole checkpoint: {env}python3 tools/offload/pack_pool.py "
            f"{model} {pool} init --wformat {wformat}, then {env}python3 "
            f"tools/offload/pack_pool.py {model} {pool} pack I N --wformat {wformat} for I in "
            f"0 .. N-1 (N workers; docs/offload.md 14)")


def check_pool(pool, image, W, wformat: str = "fp4", formats: str | None = None) -> Path:
    """The pool file for `image` (a probe's will do), or PoolError: missing; of another size
    (another model or format: open_pool would take it for a new pool and truncate it); with
    another key (moe.check_pool_key); or with experts not packed while the checkpoint `W` has
    none to pack them from (strip_experts.py's). Returns the pool's path: a symbolic link with
    no .packed file beside it is followed (its .packed, .format and .key sit beside the file it
    names)."""
    from opentpu.llm import moe as MO
    model = getattr(W, "model_dir", "MODEL")
    path = Path(pool).expanduser()
    if path.is_symlink() and not Path(str(path) + ".packed").exists():
        path = path.resolve()
    if not path.is_file():
        raise PoolError(f"no expert pool at {pool}: {_pack_hint(model, pool, wformat, formats)}")
    L = image.offload
    n = L.layers * L.E
    if path.stat().st_size != n * L.slot_bytes:
        raise PoolError(f"{pool} is {path.stat().st_size} bytes, not this image's pool ({n} "
                        f"experts of {L.slot_bytes} bytes, {image.fmt.wformat}): another model "
                        f"or format (pack_pool.py MODEL {pool} info)")
    try:
        MO.check_pool_key(path, MO.pool_key(image, getattr(W, "model_dir", None)))
    except ValueError as e:
        raise PoolError(str(e)) from None
    done = Path(str(path) + ".packed")
    have = n if not done.exists() else int(sum(1 for b in done.read_bytes() if b))
    if have < n and not any(".experts." in k for k in W.keys()):
        raise PoolError(f"{pool} holds {have} of {n} experts and {model} has none to pack the "
                        f"rest (a checkpoint without its experts): fill it from the whole "
                        f"checkpoint (pack_pool.py ... pack, or send | recv)")
    return path


def engine(model, cfg, backend="isa", cap: int = 4096, pool=None, experts: int = 0,
           policy: str = "lfu", wformat: str = "fp4", head_format: str | None = "int8",
           formats: str | None = None, embed: str | None = None,
           embed_host: bool | None = None, hints: bool | None = None,
           layer_major: int | None = None, pooled: bool = True, embed_runs: bool = False,
           layer_ahead=None, expert_major: bool | None = None, em_rows: int | None = None,
           ahead_part: int | None = None,
           idle_parts: str = "v2", willneed: bool = True, pool_map: bool = True,
           release_weights: bool = True, create: bool = False, W=None, **kw):
    """The Engine of the MoE checkpoint at `model` on `backend` ("isa", or a backend factory
    as Engine takes: the card's BoardBackend) with configuration `cfg` (the card's 4 GiB), its
    experts streamed from `pool` (a path; DEFAULT: the checkpoint's, default_pool; None: packed
    in host RAM as each is first asked for; the engine's `pool_path`). Resident decode, one row
    a step (the card's generate loop picks the tokens).

    experts: slots per MoE layer, 0 filling the DRAM (fit_experts); policy: the slots'
    replacement (ExpertServer: "lfu" least decayed use, or "lru"); formats: the weights' per
    kind over `wformat` (Spec.formats; OTPU_FORMATS overrides it); embed: Spec.embed; embed_host:
    an untied embedding table on the host (None: the image's choice, a MoE's own table there);
    hints: the router's prefetch hints (None: the Spec's, off); layer_major: the prompt layer
    by layer in runs of that many rows (None: LAYER_MAJOR where the image has layer runs, else
    0, token by token), pooled / embed_runs / layer_ahead / expert_major / ahead_part: Engine's
    (expert_major None: on wherever the prompt is layer-major with pooled slots, no
    layer_ahead and room for its needs, expert_major_fits, docs/offload.md 13.15, for the
    prefill_layers calls of at least em_rows new rows (None: the family's EM_ROWS, 13.16;
    gate_expert_major); True: on for every call unless em_rows says otherwise; ahead_part
    None: NEED_PART with it, else Engine's);
    idle_parts: ExpertServer's read ahead and halt-aware parts (IDLE_PARTS); willneed: a
    request's disk reads queued at once (PoolFile.willneed); pool_map, release_weights:
    Engine's. create: a missing pool is made and each expert packed on its first use (else
    check_pool). W: the weights (default LazyWeights of `model`). kw: Engine's own."""
    from opentpu.llm import load_spec
    from opentpu.llm.qwen3 import Engine, LazyWeights
    if idle_parts not in IDLE_PARTS:
        raise ValueError(f"idle parts {idle_parts!r}: one of {', '.join(IDLE_PARTS)}")
    spec = load_spec(model)                     # a MoE's Spec.embed: "int8" (from_hf)
    if not streams(spec):
        raise ValueError(f"{model}: not a mixture-of-experts model")
    if formats is not None:
        spec = replace(spec, formats=formats)
    if embed is not None:
        spec = replace(spec, embed=embed)
    if hints is not None:                       # (default: the Spec's)
        spec = replace(spec, moe=replace(spec.moe, hint=hints))
    W = LazyWeights(model) if W is None else W
    ekw = {} if embed_host is None else {"embed_host": embed_host}   # (default: the image's)
    img = probe(spec, cfg, cap, wformat=wformat, head_format=head_format, lookup=True, **ekw)
    if not experts:
        experts = fit_experts(spec, cfg, cap, img)
    if pool == DEFAULT:
        pool = default_pool(model, img.fmt.wformat)
    if pool is not None and not (create and not Path(pool).exists()):
        pool = check_pool(pool, img, W, wformat, formats)
    if layer_major is None:                     # (the layer runs: Qwen3.5-MoE's, Gemma 4's)
        layer_major = LAYER_MAJOR if getattr(img, "prefill_rows", 0) else 0
    if layer_major:                             # the prompt a layer at a time, runs of R rows
        ekw.update(layer_major=layer_major, pooled=pooled, embed_runs=embed_runs)
        if layer_ahead:                         # the next layer's experts during a layer's runs
            ekw.update(layer_ahead=layer_ahead)
        if expert_major is None:                # (the default since session pfB, 13.15;
            expert_major = pooled and not layer_ahead and expert_major_fits(spec, img, experts)
            em_rows = em_rows_of(spec) if em_rows is None else em_rows  # from EM_ROWS, 13.16)
        if expert_major:                        # a layer's experts in one run a chunk
            ekw.update(expert_major=True)
    kw.setdefault("rows", 1)
    eng = Engine(spec, W, cap=cap, cfg=cfg, wformat=wformat, head_format=head_format,
                 resident=True, experts=experts, pool_file=pool, backend=backend,
                 release_weights=release_weights, pool_map=pool_map, **ekw, **kw)
    eng.pool_path = pool
    if eng.expert_major and em_rows:            # (a short call layer-major: 13.16)
        gate_expert_major(eng, em_rows)
    srv = eng.server
    if eng.layer_ahead and not (pooled and hasattr(srv, "ahead_layer")):
        raise ValueError("layer_ahead needs pooled slots and a server that streams ahead "
                         "(ExpertServer.ahead_layer)")
    if ahead_part is None and eng.expert_major:     # (the needs: a whole expert a part)
        ahead_part = NEED_PART
    if ahead_part:                              # (the hints' and needs' idle-poll part)
        eng.ahead_part = ahead_part
    srv.policy = policy                         # the slots' replacement
    # idle parts (docs/offload.md 13.12): read ahead, one call; halt-aware
    srv.read_ahead, srv.halt_aware = idle_parts in ("ra", "v2"), idle_parts in ("halt", "v2")
    pf = getattr(srv, "pool_file", None)
    if willneed and pf is not None:             # a request's misses read from the disk at once
        srv.ahead = pf.willneed
    return eng


def gate_expert_major(eng, rows: int) -> None:
    """An expert-major Engine's prompts expert-major only in its prefill_layers calls of at
    least `rows` new rows, layer-major (pooled, a request a run) in the shorter ones (docs/
    offload.md 13.16): a chat turn of a model with state (Qwen3.5's, Gemma 4's sliding rings)
    is two calls, its part up to the snapshot at the reply's start and the few rows after it.
    Each call sets the Engine's expert_major for its runs and takes that mode's layer runs
    (their compiled programs: a key does not hold the mode, an expert-major run posts needs
    where the other requests its experts); after it the Engine is as before (expert_major on).
    eng.em_rows: the threshold; eng.em_last: the last call's (rows, expert-major);
    eng.em_calls: the calls [layer-major, expert-major]."""
    base = eng.prefill_layers
    runs = {True: (eng._layer_runs, eng._layer_next), False: ({}, {})}
    eng.em_rows, eng.em_last, eng.em_calls = rows, None, [0, 0]

    def prefill_layers(tokens, logits: bool = True):
        on = len(tokens) >= rows
        eng.expert_major = on
        eng._layer_runs, eng._layer_next = runs[on]
        try:
            return base(tokens, logits)
        finally:
            eng.expert_major = True
            eng._layer_runs, eng._layer_next = runs[True]
            eng.em_last = (len(tokens), on)
            eng.em_calls[on] += 1
    eng.prefill_layers = prefill_layers


def poll_idle(eng, mode: str | None):
    """The card's polls that find nothing while a run is in flight: None or "spin" (the card's
    transport: none), "predict" (PollPacer: sleep most of each gap's expected length, then
    spin; docs/offload.md 10.10) or a sleep of that many seconds. Returns the pacer, or None."""
    t = getattr(getattr(eng.backend, "board", None), "t", None)
    if mode in (None, "spin"):
        return None
    if mode == "predict":
        if not callable(getattr(eng.backend, "host", None)):
            return None
        from opentpu.host.offload import PollPacer
        pacer = PollPacer([x for x in (eng.server, eng.row_server) if x is not None])
        eng.backend.host = pacer.poll
        return pacer
    if t is not None:
        t.host_idle = float(mode)
    return None


def counters(eng) -> dict | None:
    """The expert server's counts so far (None without one): the requests' hits and misses,
    the bytes streamed."""
    srv = getattr(eng, "server", None)
    if srv is None:
        return None
    return dict(hits=srv.hits, misses=srv.misses, bytes=srv.bytes)


def describe(eng, pool=None) -> str:
    """The streaming in words, as otpu-chat shows it: "1680 slots of 10240 experts (16%),
    1.67 MB each; pool <path>"."""
    L = eng.image.offload
    n, slots = L.layers * L.E, sum(s for _, s in L.slots)
    s = f"{slots} slots of {n} experts ({100 * slots / n:.0f}%), {L.slot_bytes / 1e6:.2f} MB each"
    return s + (f"; pool {pool}" if pool is not None else "; pool in host RAM")


def host_mem() -> dict | None:
    """The host's memory now, GB (Linux /proc): this process's resident (rss, its peak hwm;
    rss_file the mapped files' pages, which the page cache holds and may drop) and swapped, its
    children's resident and swapped, the system's available, page cache, anonymous and swap
    used."""
    def kb(path, keys):
        try:
            with open(path) as f:
                return {k: int(v.split()[0]) for k, _, v in (ln.partition(":") for ln in f)
                        if k in keys}
        except OSError:
            return {}
    me = kb("/proc/self/status", ("VmRSS", "VmHWM", "VmSwap", "RssFile"))
    if not me:
        return None
    kids = {"VmRSS": 0, "VmSwap": 0}
    for d in Path("/proc").iterdir():           # (the Engine's compile workers)
        try:
            if d.name.isdigit() and int((d / "stat").read_text().rsplit(")", 1)[1].split()[1]) \
                    == os.getpid():
                for k, v in kb(d / "status", ("VmRSS", "VmSwap")).items():
                    kids[k] += v
        except (OSError, ValueError, IndexError):
            continue
    m = kb("/proc/meminfo", ("MemTotal", "MemAvailable", "Cached", "AnonPages", "SwapTotal",
                             "SwapFree"))
    gb = lambda v: round(v / 1e6, 2)            # noqa: E731 (kB)
    return dict(rss=gb(me.get("VmRSS", 0)), hwm=gb(me.get("VmHWM", 0)),
                rss_file=gb(me.get("RssFile", 0)), swap=gb(me.get("VmSwap", 0)),
                children_rss=gb(kids["VmRSS"]),
                children_swap=gb(kids["VmSwap"]), total=gb(m.get("MemTotal", 0)),
                available=gb(m.get("MemAvailable", 0)), cached=gb(m.get("Cached", 0)),
                anon=gb(m.get("AnonPages", 0)),
                swap_used=gb(m.get("SwapTotal", 0) - m.get("SwapFree", 0)))
