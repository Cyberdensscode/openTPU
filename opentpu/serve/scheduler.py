"""The device's worker (otpu-serve): requests from any number of clients, one model on the card,
each request's context switched in and out of the device's KV cache.

One thread owns the Engine. A request is a prompt's token ids (the API renders the chat), its
sampling and its stop conditions; the worker admits up to `max_running` at a time and runs them
in turns:

- a new request's prompt continues the longest prefix the device can take back: what the
  device holds (a common prefix where the KV caches hold the whole context; where there is
  state, a context the prompt continues), or a context saved in the KV store
  (opentpu/serve/store.py: host RAM, then disk), whichever is longer; only the rest is fed
  (prefill), then the first token is picked from the last row's logits;
- a running request decodes a quantum of tokens per turn, on the card's decode loop
  (Engine.generate_card: greedy or with the request's sampling, its uniforms from the request's
  own generator) or, for sampling the device does not take, step by step on the host;
- before the device's context is overwritten it is saved to the store (its new pages and its
  tail; an active request's pinned until it ends), so that the request it belongs to, or a
  later one sharing its prefix, continues it instead of feeding it again.

New requests come first (their prefill), then the running ones in turn. While one request runs
alone, it runs to its end; the card's loop is halted after the token in flight as soon as
another request arrives (stop(), polled while the card runs).
"""
from __future__ import annotations

import itertools
import queue
import threading
import time
import traceback
from dataclasses import dataclass, field

import numpy as np

from ..host.chat import Detok, sampler as host_sampler
from ..llm import generate as G
from . import kvio as KV
from . import store as ST

QUANTUM = 32            # tokens a running request decodes per turn while others wait
CHUNK = 64              # tokens per decode run on a backend that does not poll stop() mid-run


@dataclass
class Sampling:
    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    seed: int | None = None

    @property
    def greedy(self) -> bool:
        return self.temperature <= 0 and self.repetition_penalty == 1.0


@dataclass
class Request:
    """One generation. prompt: the token ids fed; at: where a model with state keeps a saved
    context in the prompt (a chat's reply start: the next turn's rendering parts there), or
    None; stop_ids end it (not part of the output); stop_strs end it at the text's first match
    (cut there)."""
    prompt: list
    max_new: int = 256
    sampling: Sampling = field(default_factory=Sampling)
    stop_ids: frozenset = frozenset()
    stop_strs: tuple = ()
    at: int | None = None
    rid: str = ""
    skip_special: bool = True         # the text without the tokenizer's special tokens
    # progress (the worker's)
    out: list = field(default_factory=list)
    text: str = ""
    finish: str | None = None         # "stop", "length", "abort", "error"
    error: str | None = None
    cached: int = 0                   # prompt tokens the device did not have to feed
    prefill_tokens: int = 0
    t_submit: float = 0.0
    t_first: float | None = None
    t_done: float | None = None
    turns: int = 0
    swaps: int = 0
    events: queue.Queue = field(default_factory=queue.Queue)
    cancelled: threading.Event = field(default_factory=threading.Event)
    # the worker's state of it
    fed: list = field(default_factory=list)      # tokens in its context (the device's or saved)
    next: int | None = None           # its next token: picked, not fed
    entry: object = None              # its saved context while it is off the device (pinned)
    _shown: int = 0                   # chars of text sent
    _detok: object = None
    _pick: object = None

    def cancel(self) -> None:
        self.cancelled.set()

    def stream(self, timeout: float | None = None):
        """The text deltas as they come; ends when the request does."""
        while True:
            kind, val = self.events.get(timeout=timeout)
            if kind == "delta":
                yield val
            else:
                return

    def wait(self, timeout: float | None = None) -> "Request":
        for _ in self.stream(timeout):
            pass
        return self


class Worker:
    """The device's thread: Worker(engine, tokenizer, ...).start(); submit(request). store: a
    store.BlobStore for the saved contexts (None: host RAM of `ram_bytes`); model_key: what
    makes the contexts' bytes (store.model_key)."""

    def __init__(self, eng, tok, store=None, model_key: str = "model",
                 max_running: int = 8, quantum: int = QUANTUM, ram_bytes: int = 4 << 30,
                 area=False):
        self.eng, self.tok = eng, tok
        self.tok_lock = threading.Lock()
        self.io = KV.ContextIO(eng, area=area)      # (False: kvio.staging_area's)
        self.store = store if store is not None else ST.BlobStore(ram_bytes)
        self.cache = ST.PrefixCache(self.store, self.io.layout, model_key)
        self.max_running, self.quantum = max_running, quantum
        self.stateful = self.io.layout.stateful
        self.softcap = getattr(eng.spec, "softcap", None)
        self.waiting: list[Request] = []
        self.running: list[Request] = []
        self.cv = threading.Condition()
        self.dev: list = []               # the tokens the device's context holds
        self.dirty = False                # ... not saved in the store yet
        self.owner: Request | None = None
        self._rr = 0
        self._ids = itertools.count(1)
        self._stop = False
        self.thread = None
        self.stats = {"requests": 0, "finished": 0, "aborted": 0, "errors": 0,
                      "prompt_tokens": 0, "cached_tokens": 0, "prefill_tokens": 0,
                      "generated": 0, "decode_s": 0.0, "prefill_s": 0.0, "switch_s": 0.0,
                      "saves": 0, "restores": 0, "device_reuses": 0, "turns": 0}

    # ---- the clients' side
    def start(self) -> "Worker":
        self.thread = threading.Thread(target=self._loop, daemon=True, name="otpu-serve-worker")
        self.thread.start()
        return self

    def stop(self, persist: bool = True) -> None:
        """End the worker (requests in flight aborted); persist: the device's context and the
        store's RAM onto its disk, for a later server."""
        with self.cv:
            self._stop = True
            self.cv.notify_all()
        if self.thread is not None:
            self.thread.join()
            self.thread = None
        if persist and self.store.dir is not None:
            try:
                self._save_device()
            except BaseException:
                traceback.print_exc()
            self.store.persist()

    def submit(self, r: Request) -> Request:
        if not r.prompt:
            raise ValueError("an empty prompt")
        if len(r.prompt) >= self.eng.cap:
            raise ValueError(f"the prompt's {len(r.prompt)} tokens leave no room in the "
                             f"context of {self.eng.cap}")
        r.rid = r.rid or f"r{next(self._ids)}"
        r.stop_ids = frozenset(r.stop_ids) or frozenset(self.eng.spec.eos)
        if len(r.stop_ids) > G.N_STOP:
            raise ValueError(f"{len(r.stop_ids)} stop token ids: the card's loop takes "
                             f"{G.N_STOP}")
        r.t_submit = time.perf_counter()
        r.max_new = max(0, min(r.max_new, self.eng.cap - len(r.prompt)))
        with self.cv:
            self.waiting.append(r)
            self.stats["requests"] += 1
            self.cv.notify_all()
        return r

    def busy(self) -> int:
        with self.cv:
            return len(self.waiting) + len(self.running)

    # ---- the loop
    def _loop(self) -> None:
        while True:
            with self.cv:
                while not self._stop and not self.waiting and not self.running:
                    self.cv.wait()
                if self._stop:
                    for r in self.waiting + self.running:
                        self._end(r, "abort")
                    return
                r = self._pick()
            if r is None:
                continue
            try:
                self._turn(r)
            except BaseException as e:      # the request fails, the server goes on
                r.error = f"{type(e).__name__}: {e}"
                traceback.print_exc()
                self.stats["errors"] += 1
                self.owner, self.dev, self.dirty = None, [], False
                try:
                    self.eng.reset()
                except BaseException:
                    pass
                self._end(r, "error")

    def _pick(self) -> Request | None:
        """New requests first (up to max_running running), then the running ones in turn."""
        for r in [r for r in self.running if r.cancelled.is_set()]:
            self._end(r, "abort")
        for r in [r for r in self.waiting if r.cancelled.is_set()]:
            self._end(r, "abort")
        if self.waiting and len(self.running) < self.max_running:
            r = self.waiting.pop(0)
            self.running.append(r)
            return r
        if not self.running:
            return None
        self._rr = (self._rr + 1) % len(self.running)
        if self.owner in self.running and len(self.running) == 1:
            return self.owner
        return self.running[self._rr]

    def _others(self, r: Request) -> bool:
        """Another request waits for the device (no lock: polled while the card runs)."""
        return bool(self.waiting) or len(self.running) > 1

    def _turn(self, r: Request) -> None:
        r.turns += 1
        self.stats["turns"] += 1
        t0 = time.perf_counter()
        self._switch_to(r)
        self.stats["switch_s"] += time.perf_counter() - t0
        if r.next is None and r.finish is None:
            self._prefill(r)
        if r.finish is None and r.next is not None:
            self._decode(r)
        if r.finish is not None:
            self._end(r, r.finish)

    # ---- contexts
    @staticmethod
    def _common(a, b) -> int:
        n = min(len(a), len(b))
        if not n:
            return 0
        eq = np.asarray(a[:n], np.int64) == np.asarray(b[:n], np.int64)
        return n if eq.all() else int(np.argmin(eq))

    def _save_device(self, pin_for: Request | None = None) -> None:
        """The device's context into the store (what the store lacks of it)."""
        if not self.dev or (not self.dirty and pin_for is None):
            return
        e = self.cache.save(self.io, list(self.dev), pin=pin_for is not None)
        self.stats["saves"] += 1
        self.dirty = False
        if pin_for is not None:
            pin_for.entry = e

    def _switch_to(self, r: Request) -> None:
        """Make the device's context r's: a running request's own (as it left it), or for a new
        one the longest prefix of its prompt the device or the store has."""
        if self.owner is r:
            return
        old = self.owner
        if old is not None and old.finish is None and old.fed:
            old.swaps += 1
            self._save_device(pin_for=old)      # it comes back to this
        self.owner = None
        c = self._common(self.dev, r.fed or r.prompt)
        if r.fed:                               # r ran before: exactly its context
            want, m = r.fed, ST.Match(len(r.fed), r.entry)
            exact = len(want)
            dev_k = exact if c == exact and (not self.stateful or len(self.dev) == exact) \
                else 0
        else:
            want = r.prompt
            limit = len(want) - 1               # (a token to feed: its logits)
            m = self.cache.match(want, limit=limit)
            if self.stateful:                   # the device's context, continued
                dev_k = len(self.dev) if c == len(self.dev) <= limit else 0
            else:                               # any common prefix
                dev_k = min(c, limit)
        if dev_k > 0 and dev_k >= m.k:          # the device holds it
            if dev_k < len(self.dev):
                self._save_device()             # (what lies past it is overwritten)
            self.eng.adopt(dev_k)
            k = dev_k
            self.stats["device_reuses"] += 1
        else:
            self._save_device()
            k = self.cache.restore(self.io, m, have=c) if m.k else 0
            if k:
                self.eng.adopt(k)
                self.stats["restores"] += 1
            else:
                self.eng.reset()
        if r.fed:
            if r.entry is not None:
                self.cache.unpin(r.entry)
                r.entry = None
            if k != len(r.fed):                 # its saved context is gone (dropped): again
                r.fed, r.next = [], None
                self.eng.reset()
                k = 0
        else:
            r.fed = list(want[:k])
            r.cached = k
            self.stats["cached_tokens"] += k
        self.dev = r.fed                        # (the same list: it grows as r runs)
        self.dirty = True
        self.owner = r
        if self.eng.pos != len(r.fed):
            raise RuntimeError(f"the device is at {self.eng.pos}, the context has {len(r.fed)}")

    # ---- prefill and decode
    def _sampler(self, r: Request):
        if r._pick is None:
            s = r.sampling
            r._pick = host_sampler(0 if s.greedy else s.temperature, s.top_k, s.top_p,
                                   s.seed, s.repetition_penalty, self.softcap)
        return r._pick

    def _prefill(self, r: Request) -> None:
        t0 = time.perf_counter()
        ids = r.prompt
        n = len(r.fed)
        at = r.at if self.stateful and r.at is not None and n < r.at < len(ids) else None
        cut = [at] if at is not None else []
        logits = None
        self.stats["prompt_tokens"] += len(ids)
        for a, b in zip([n] + cut, cut + [len(ids)]):
            kw = {} if b == len(ids) else {"logits": False}
            for part, logits in self.eng.prefill_chunks(ids[a:b], **kw):
                r.fed += part
                r.prefill_tokens += len(part)
                self.stats["prefill_tokens"] += len(part)
                if r.cancelled.is_set():
                    r.finish = "abort"
                    self.stats["prefill_s"] += time.perf_counter() - t0
                    return
            if b != len(ids):                   # a saved context where the next turn parts
                self.cache.save(self.io, list(r.fed))
                self.stats["saves"] += 1
        self.stats["prefill_s"] += time.perf_counter() - t0
        t = int(np.argmax(logits)) if r.sampling.greedy else \
            int(self._sampler(r)(logits, r.fed))
        self._emit(r, t)

    def _emit(self, r: Request, t: int) -> None:
        """Token t generated: the request's output and its text, its end when it stops."""
        if r.finish is not None:
            return
        if t in r.stop_ids:
            r.next = None
            r.finish = "stop"
            return
        if r.t_first is None:
            r.t_first = time.perf_counter()
        r.out.append(t)
        r.next = t
        self.stats["generated"] += 1
        if r._detok is None:
            r._detok = _Detok(_Locked(self.tok, self.tok_lock), r.skip_special)
        r.text += r._detok.add(t)
        cut = None
        for s in r.stop_strs:
            i = r.text.find(s, max(0, r._shown - len(s)))
            if i >= 0 and (cut is None or i < cut):
                cut = i
        if cut is not None:
            r.text = r.text[:cut]
            r.finish = "stop"
        elif len(r.out) >= r.max_new:
            r.finish = "length"
        hold = 0                                # what may start a stop string: held back
        if r.finish is None:
            for s in r.stop_strs:
                for j in range(min(len(s) - 1, len(r.text)), 0, -1):
                    if r.text.endswith(s[:j]):
                        hold = max(hold, j)
                        break
        end = len(r.text) - hold
        if end > r._shown:
            r.events.put(("delta", r.text[r._shown:end]))
            r._shown = end

    def _decode(self, r: Request) -> None:
        """Up to the request's end, or a quantum when others wait. Invariant: the device has
        fed r.fed (eng.pos == len(r.fed)); r.next is picked, not fed."""
        t0 = time.perf_counter()
        eng = self.eng
        k0 = len(r.out)
        while r.finish is None and r.next is not None:
            if len(r.fed) >= eng.cap:           # no room to feed the next token
                r.finish = "length"
                break
            if r.cancelled.is_set():
                r.finish = "abort"
                break
            left = r.max_new - len(r.out)
            n = left if not self._others(r) else min(left, self.quantum)
            if not getattr(eng.backend, "run_generate", None):
                n = min(n, CHUNK)               # (no stop() mid-run: a cancel waits for it)
            pick = self._sampler(r)
            card = getattr(eng, "can_generate", False)
            samp = None
            if card and not r.sampling.greedy:
                card = G.Sampling.fits(**pick.params)
                samp = G.Sampling(**pick.params) if card else None
            t = r.next
            if card:
                def stop() -> bool:
                    return (r.finish is not None or r.cancelled.is_set()
                            or (self._others(r) and len(r.out) - k0 >= self.quantum))
                r.fed.append(t)
                got = eng.generate_card(t, n, stop_ids=sorted(r.stop_ids),
                                        on_token=lambda x: self._emit(r, x), stop=stop,
                                        sampling=samp, context=list(r.fed),
                                        rng=pick.rng if samp is not None else None)
                r.fed += got[:-1]
                if not got:
                    r.fed.pop()
                    break
                if got[-1] in r.stop_ids:
                    r.next = None
                    r.finish = r.finish or "stop"
            else:                               # the host picks each token
                for _ in range(n):
                    r.fed.append(t)
                    lg = eng.step(t)
                    t = int(np.argmax(lg)) if r.sampling.greedy else int(pick(lg, r.fed))
                    self._emit(r, t)
                    if r.finish is not None or len(r.fed) >= eng.cap:
                        break
            if eng.pos != len(r.fed):
                raise RuntimeError(f"the device is at {eng.pos}, the context has {len(r.fed)}")
            if self._others(r) and len(r.out) - k0 >= self.quantum:
                break
        self.stats["decode_s"] += time.perf_counter() - t0

    def _end(self, r: Request, why: str) -> None:
        with self.cv:
            if r in self.running:
                self.running.remove(r)
            if r in self.waiting:
                self.waiting.remove(r)
        if r.entry is not None:
            self.cache.unpin(r.entry)
            r.entry = None
        r.finish = r.finish or why
        if r.finish == "abort":
            self.stats["aborted"] += 1
        else:
            self.stats["finished"] += 1
        r.t_done = time.perf_counter()
        if r._shown < len(r.text):
            r.events.put(("delta", r.text[r._shown:]))
            r._shown = len(r.text)
        r.events.put(("done", r.finish))
        if self.owner is r:                     # the device keeps its context (unsaved)
            self.owner = None

    def usage(self) -> dict:
        with self.cv:
            q = {"waiting": len(self.waiting), "running": len(self.running)}
        return {**q, **self.stats, "device_context": len(self.dev),
                "kv_store": self.store.usage(), "prefix_cache": self.cache.usage(),
                "kvio": dict(self.io.stats)}


class _Detok(Detok):
    """chat.Detok, the special tokens' text kept or skipped."""

    def __init__(self, tok, skip: bool):
        super().__init__(tok)
        self.skip = skip

    def _dec(self, ids) -> str:
        return self.tok.decode(ids, skip_special_tokens=self.skip)


class _Locked:
    """A tokenizer whose decode holds a lock (the API's threads encode with it meanwhile)."""

    def __init__(self, tok, lock):
        self.tok, self.lock = tok, lock

    def decode(self, ids, **kw):
        with self.lock:
            return self.tok.decode(ids, **kw)
