"""Benchmark and check an otpu-serve server (opentpu/serve/api.py) from clients over HTTP (the
standard library only, so it runs from the card host's venv).

    python tools/serve_bench.py --port 8000 [--max-tokens 128] [--levels 1,2,4,8] [--out x.json]

1. Correctness: each of the prompts alone, greedy, one after the other; then all of them at once
   (their contexts switched in and out of the device between turns): every reply must equal its
   solo one, and a repeat of the first (its prompt from the KV store or the device) too.
2. Throughput: `level` clients at once, each a streamed greedy chat of max_tokens: the wall
   time, the aggregate generated tokens/s, each client's time to first token and decode rate.
3. A conversation's turns: each turn re-sends the history; its prompt tokens the server already
   held (usage.prompt_tokens_details.cached_tokens) and its TTFT against the same prompt with
   the cache flushed.
4. Arrivals (--arrivals N): clients starting one every --arrival-interval s: their TTFTs and
   the gaps between their tokens (how long a newcomer's prompt held the others).
5. Users (--users N, more than the slots): every user's first turn at once, then every user's
   second turn at once: the tokens the server held for each (its context from a slot, host RAM
   or disk), the KV store's tiers; then a few second turns again with the cache flushed: the
   same replies (greedy), so the contexts came back bit for bit.
6. Images (--image PATH, repeated; a model with a vision side, docs/serve.md Images): the first
   image with a question, greedy; again (the context reused past the image: cached tokens); its
   next turn (the history reused); each other image with the same question (no context past
   the other image reused); a text-only question. With --image-ref (tools/qual/chatmoe.py's
   JSON of the same image, question and slices) the first reply must be its first turn's.
"""
from __future__ import annotations

import argparse
import http.client
import json
import statistics
import sys
import threading
import time

QUESTIONS = ["Why is the sky blue?", "Write a haiku about the sea.",
             "What is the capital of France, and why is it famous?",
             "Explain recursion to a five-year-old.", "List three uses of a paperclip.",
             "What does a CPU do?", "Name a prime number between 20 and 30.",
             "How do airplanes stay in the air?"]


class Client:
    def __init__(self, host: str, port: int, key: str | None = None):
        self.host, self.port, self.key = host, port, key

    def _conn(self):
        return http.client.HTTPConnection(self.host, self.port, timeout=3600)

    def _headers(self):
        h = {"Content-Type": "application/json"}
        if self.key:
            h["Authorization"] = f"Bearer {self.key}"
        return h

    def get(self, path: str):
        c = self._conn()
        c.request("GET", path, headers=self._headers())
        r = c.getresponse()
        return r.status, json.loads(r.read() or b"null")

    def post(self, path: str, body: dict):
        c = self._conn()
        c.request("POST", path, json.dumps(body), self._headers())
        r = c.getresponse()
        return r.status, json.loads(r.read() or b"null")

    def chat_stream(self, messages, **kw) -> dict:
        """A streamed chat: its text, the times of its first and last token, usage."""
        c = self._conn()
        body = {"messages": messages, "stream": True,
                "stream_options": {"include_usage": True}, **kw}
        t0 = time.perf_counter()
        c.request("POST", "/v1/chat/completions", json.dumps(body), self._headers())
        r = c.getresponse()
        if r.status != 200:
            raise RuntimeError(f"HTTP {r.status}: {r.read()[:300]!r}")
        text, first, last, usage, n, fin = "", None, None, None, 0, None
        times = []
        buf = b""
        while True:
            line = r.fp.readline()
            if not line:
                break
            line = line.strip()
            if not line.startswith(b"data: "):
                continue
            x = line[6:]
            if x == b"[DONE]":
                break
            ev = json.loads(x)
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices", []):
                d = ch.get("delta", {}).get("content")
                if d:
                    now = time.perf_counter()
                    first = first or now
                    last = now
                    times.append(now)
                    text += d
                    n += 1
                fin = ch.get("finish_reason") or fin
        del buf
        return {"text": text, "ttft": (first - t0) if first else None,
                "wall": time.perf_counter() - t0, "decode_s": (last - first) if first else 0,
                "deltas": n, "usage": usage, "finish": fin,
                "gaps": [b - a for a, b in zip(times, times[1:])]}


def _msgs(q):
    return [{"role": "user", "content": q}]


def correctness(cl: Client, n: int, max_tokens: int) -> dict:
    qs = QUESTIONS[:n]
    solo = [cl.chat_stream(_msgs(q), max_tokens=max_tokens, temperature=0)["text"] for q in qs]
    res = [None] * n

    def run(i):
        res[i] = cl.chat_stream(_msgs(qs[i]), max_tokens=max_tokens, temperature=0)["text"]
    ts = [threading.Thread(target=run, args=(i,)) for i in range(n)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    again = cl.chat_stream(_msgs(qs[0]), max_tokens=max_tokens, temperature=0)
    same = [a == b for a, b in zip(solo, res)]
    out = {"prompts": n, "together_equal_solo": same, "repeat_equal": again["text"] == solo[0],
           "repeat_cached": (again["usage"] or {}).get("prompt_tokens_details", {}).get(
               "cached_tokens"), "solo": solo}
    out["pass"] = all(same) and out["repeat_equal"]
    return out


def throughput(cl: Client, level: int, max_tokens: int, sampled: bool = False) -> dict:
    """level clients at once; greedy, or sampled with the model's defaults (seeded)."""
    res = [None] * level

    def run(i):
        q = QUESTIONS[i % len(QUESTIONS)] + f" (client {i})"
        kw = {"seed": i} if sampled else {"temperature": 0}
        res[i] = cl.chat_stream(_msgs(q), max_tokens=max_tokens, ignore_eos=True, **kw)
    t0 = time.perf_counter()
    ts = [threading.Thread(target=run, args=(i,)) for i in range(level)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    wall = time.perf_counter() - t0
    gen = sum((r["usage"] or {}).get("completion_tokens", 0) for r in res)
    ttft = [r["ttft"] for r in res if r["ttft"] is not None]
    rates = [((r["usage"] or {}).get("completion_tokens", 1) - 1) / r["decode_s"]
             for r in res if r["decode_s"] > 0]
    return {"clients": level, "sampled": sampled, "wall_s": round(wall, 2), "generated": gen,
            "aggregate_tok_s": round(gen / wall, 2),
            "ttft_s": {"min": round(min(ttft), 3), "median": round(statistics.median(ttft), 3),
                       "max": round(max(ttft), 3)},
            "client_tok_s_median": round(statistics.median(rates), 2) if rates else None}


def conversation(cl: Client, turns: int, max_tokens: int) -> list:
    msgs, out = [{"role": "system", "content": "You are a concise assistant."}], []
    qs = ["Tell me about the moon.", "How far is it?", "Could people live there?",
          "What would they eat?", "Thanks! Summarize in one line."][:turns]
    for q in qs:
        msgs.append({"role": "user", "content": q})
        r = cl.chat_stream(msgs, max_tokens=max_tokens, temperature=0)
        cl.post("/flush_cache", {})
        # the same prompt with nothing saved: the device's context is another's first
        cl.chat_stream(_msgs("Say one word."), max_tokens=1, temperature=0)
        cold = cl.chat_stream(msgs, max_tokens=1, temperature=0)
        u = r["usage"] or {}
        out.append({"prompt_tokens": u.get("prompt_tokens"),
                    "cached": u.get("prompt_tokens_details", {}).get("cached_tokens"),
                    "ttft_s": round(r["ttft"], 3), "cold_ttft_s": round(cold["ttft"], 3)})
        msgs.append({"role": "assistant", "content": r["text"]})
        # put the conversation's context back for the next turn (the cold run replaced it)
        cl.chat_stream(msgs[:-1], max_tokens=max_tokens, temperature=0)
    return out


def arrivals(cl: Client, n: int, interval: float, max_tokens: int) -> dict:
    """n clients, the i-th starting i * interval s after the first, each a streamed greedy chat
    of max_tokens (ignore_eos): their times to first token and the gaps between their tokens
    (the longest: how long a newcomer's prompt held the others)."""
    def run(i):
        time.sleep(i * interval)
        return cl.chat_stream(_msgs(QUESTIONS[i % len(QUESTIONS)] + f" (arrival {i})"),
                              max_tokens=max_tokens, ignore_eos=True, temperature=0)
    s0 = cl.get("/v1/stats")[1]
    t0 = time.perf_counter()
    res = _all(n, run)
    wall = time.perf_counter() - t0
    s1 = cl.get("/v1/stats")[1]
    gaps = sorted(g for r in res for g in r["gaps"])
    pick = (lambda q: round(1e3 * gaps[min(len(gaps) - 1, int(q * len(gaps)))], 1)) \
        if gaps else (lambda q: None)
    return {"clients": n, "interval_s": interval, "wall_s": round(wall, 2),
            "generated": sum((r["usage"] or {}).get("completion_tokens", 0) for r in res),
            "ttft_s": [round(r["ttft"], 3) for r in res],
            "gap_ms": {"p50": pick(0.5), "p90": pick(0.9), "p99": pick(0.99),
                       "max": round(1e3 * gaps[-1], 1) if gaps else None},
            "max_gap_ms_per_client": [round(1e3 * max(r["gaps"]), 1) if r["gaps"] else None
                                      for r in res],
            "server": _server(s0, s1)}


def _server(s0: dict, s1: dict) -> dict:
    """The batching server's counts between two /v1/stats: decode steps, their runs (a step
    runs a block of slots a run), mixed runs, inline compiles."""
    d = {k: s1.get(k, 0) - s0.get(k, 0) for k in ("steps", "step_runs", "mixed_runs",
                                                   "prefill_tokens")}
    t0, t1 = s0.get("step_timing") or {}, s1.get("step_timing") or {}
    d["compiled_inline"] = t1.get("compiled_inline", 0) - t0.get("compiled_inline", 0)
    return d


def _all(n, fn) -> list:
    """fn(i) for i < n, each in its own thread at once."""
    res = [None] * n

    def run(i):
        res[i] = fn(i)
    ts = [threading.Thread(target=run, args=(i,)) for i in range(n)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return res


def users(cl: Client, n: int, max_tokens: int, checks: int = 3) -> dict:
    hist = [[{"role": "user", "content": f"I am user {i}. " + QUESTIONS[i % len(QUESTIONS)]}]
            for i in range(n)]
    t1 = _all(n, lambda i: cl.chat_stream(hist[i], max_tokens=max_tokens, temperature=0))
    for i in range(n):
        hist[i] += [{"role": "assistant", "content": t1[i]["text"]},
                    {"role": "user", "content": "Tell me more, briefly."}]
    s0 = cl.get("/v1/stats")[1]
    t2 = _all(n, lambda i: cl.chat_stream(hist[i], max_tokens=max_tokens, temperature=0))
    s1 = cl.get("/v1/stats")[1]
    cl.post("/flush_cache", {})
    cold = [cl.chat_stream(hist[i], max_tokens=max_tokens, temperature=0) for i in range(checks)]
    p1 = [(r["usage"] or {}).get("prompt_tokens", 0) for r in t1]
    cached = [(r["usage"] or {}).get("prompt_tokens_details", {}).get("cached_tokens", 0)
              for r in t2]
    equal = [c["text"] == r["text"] for c, r in zip(cold, t2)]
    ks = s1.get("kv_store", {})
    k0 = s0.get("kv_store", {})
    out = {"users": n, "turn1_prompt_median": statistics.median(p1),
           "turn2_prompt_median": statistics.median(
               (r["usage"] or {}).get("prompt_tokens", 0) for r in t2),
           "turn2_cached": {"min": min(cached), "median": statistics.median(cached)},
           "turn2_ttft_median": round(statistics.median(r["ttft"] for r in t2), 3),
           "cold_ttft_median": round(statistics.median(c["ttft"] for c in cold), 3),
           "cold_equal": equal,
           "store": {k: ks.get(k) for k in ("ram_bytes", "ram_blobs", "disk_bytes",
                                            "disk_blobs", "spilled", "dropped")},
           "turn2_store_hits": {"ram": ks.get("ram_hits", 0) - k0.get("ram_hits", 0),
                                "disk": ks.get("disk_hits", 0) - k0.get("disk_hits", 0)}}
    out["pass"] = all(equal) and min(cached) >= min(p1) // 2
    return out


def _image_part(path: str) -> dict:
    """An image_url part: the file as a data: URL."""
    import base64
    import mimetypes
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    return {"type": "image_url", "image_url": {
        "url": f"data:{mimetypes.guess_type(path)[0] or 'image/png'};base64,{data}"}}


def images(cl: Client, paths: list, max_tokens: int, max_slices: int | None = None,
           question: str = "Describe this image in one sentence.",
           follow: str = "What colors stand out? One sentence.", ref: dict | None = None) -> dict:
    """Part 6 (docs/serve.md Images): each request's reply, prompt and cached tokens and TTFT.
    Passes when the repeat gives the first reply with most of its prompt cached, the next turn
    has half the first prompt cached or more, every other image's request less (a context is
    reused past an image only for the same image), and (ref) the first reply is ref's first
    turn's."""
    kw = {"max_tokens": max_tokens, "temperature": 0}
    if max_slices is not None:
        kw["max_slices"] = max_slices

    def ask(msgs):
        r = cl.chat_stream(msgs, **kw)
        u = r["usage"] or {}
        return {"text": r["text"], "prompt_tokens": u.get("prompt_tokens", 0),
                "cached": u.get("prompt_tokens_details", {}).get("cached_tokens", 0),
                "ttft_s": round(r["ttft"], 3) if r["ttft"] else None, "finish": r["finish"]}

    def asked(path):
        return [{"role": "user", "content": [_image_part(path),
                                             {"type": "text", "text": question}]}]
    m1 = asked(paths[0])
    first = ask(m1)
    again = ask(m1)
    turn2 = ask(m1 + [{"role": "assistant", "content": first["text"]},
                      {"role": "user", "content": follow}])
    others = [ask(asked(p)) for p in paths[1:]]
    text = ask(_msgs("Why is the sky blue? One sentence."))
    n = first["prompt_tokens"]
    out = {"images": paths, "max_slices": max_slices, "first": first, "again": again,
           "turn2": turn2, "others": others, "text": text,
           "again_equal": again["text"] == first["text"],
           "again_cached": again["cached"] >= n // 2,
           "turn2_cached": turn2["cached"] >= n // 2,
           "others_uncached": all(o["cached"] < n // 2 for o in others)}
    if ref is not None:
        want = ref["turns"][0]["reply"]
        out["ref_reply"] = want
        out["ref_equal"] = first["text"].strip() == want.strip()
    out["pass"] = all(out[k] for k in ("again_equal", "again_cached", "turn2_cached",
                                       "others_uncached")) and out.get("ref_equal", True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--api-key")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--check-prompts", type=int, default=4)
    ap.add_argument("--check-tokens", type=int, default=48)
    ap.add_argument("--levels", default="1,2,4,8")
    ap.add_argument("--sampled-levels", default="8",
                    help="levels run again with the model's sampling (seeded)")
    ap.add_argument("--turns", type=int, default=4)
    ap.add_argument("--users", type=int, default=16, help="users of the users part (0: none)")
    ap.add_argument("--arrivals", type=int, default=6,
                    help="clients of the arrivals part, one every --arrival-interval s (0: none)")
    ap.add_argument("--arrival-interval", type=float, default=1.5)
    ap.add_argument("--user-tokens", type=int, default=64, help="max tokens of a user's reply")
    ap.add_argument("--wait", type=float, default=900, help="seconds to wait for /health")
    ap.add_argument("--image", action="append", default=[],
                    help="an image of the images part (repeated: the first is asked about twice "
                         "and followed up; none: no images part)")
    ap.add_argument("--image-slices", type=int, help="the images' max_slices (default the "
                                                     "server's)")
    ap.add_argument("--image-tokens", type=int, default=16, help="max tokens of an image's reply")
    ap.add_argument("--image-ref", help="tools/qual/chatmoe.py's JSON of the first image (its "
                                        "first turn's reply: the first request's)")
    ap.add_argument("--out")
    ap.add_argument("--skip", default="",
                    help="parts to skip: check,throughput,conversation,users,arrivals")
    a = ap.parse_args(argv)
    cl = Client(a.host, a.port, a.api_key)
    t0 = time.time()
    while True:
        try:
            if cl.get("/health")[0] == 200:
                break
        except OSError:
            pass
        if time.time() - t0 > a.wait:
            raise SystemExit("serve_bench: the server did not come up")
        time.sleep(1)
    rep = {"models": cl.get("/v1/models")[1]}
    skip = set(a.skip.split(","))
    if "check" not in skip:
        rep["check"] = correctness(cl, a.check_prompts, a.check_tokens)
        print("check:", "PASS" if rep["check"]["pass"] else "FAIL",
              {k: v for k, v in rep["check"].items() if k != "solo"}, flush=True)
    if "throughput" not in skip:
        rep["throughput"] = []
        for lv, smp in [(int(x), False) for x in a.levels.split(",") if x] + \
                [(int(x), True) for x in a.sampled_levels.split(",") if x]:
            x = throughput(cl, lv, a.max_tokens, smp)
            rep["throughput"].append(x)
            print("throughput:", x, flush=True)
    if "conversation" not in skip:
        rep["conversation"] = conversation(cl, a.turns, a.max_tokens)
        for i, x in enumerate(rep["conversation"]):
            print(f"turn {i + 1}:", x, flush=True)
    if "arrivals" not in skip and a.arrivals:
        rep["arrivals"] = arrivals(cl, a.arrivals, a.arrival_interval, a.max_tokens)
        print("arrivals:", rep["arrivals"], flush=True)
    if "users" not in skip and a.users:
        rep["users"] = users(cl, a.users, a.user_tokens)
        print("users:", "PASS" if rep["users"]["pass"] else "FAIL", rep["users"], flush=True)
    if a.image:
        ref = None
        if a.image_ref:
            with open(a.image_ref) as f:
                ref = json.load(f)
        rep["images"] = images(cl, a.image, a.image_tokens, a.image_slices, ref=ref)
        print("images:", "PASS" if rep["images"]["pass"] else "FAIL", rep["images"], flush=True)
    rep["stats"] = cl.get("/v1/stats")[1]
    print("stats:", json.dumps(rep["stats"])[:2000], flush=True)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(rep, f, indent=1)
    ok = all(rep.get(k, {}).get("pass", True) for k in ("check", "users", "images"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
