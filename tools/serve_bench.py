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
                    text += d
                    n += 1
                fin = ch.get("finish_reason") or fin
        del buf
        return {"text": text, "ttft": (first - t0) if first else None,
                "wall": time.perf_counter() - t0, "decode_s": (last - first) if first else 0,
                "deltas": n, "usage": usage, "finish": fin}


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


def throughput(cl: Client, level: int, max_tokens: int) -> dict:
    res = [None] * level

    def run(i):
        q = QUESTIONS[i % len(QUESTIONS)] + f" (client {i})"
        res[i] = cl.chat_stream(_msgs(q), max_tokens=max_tokens, temperature=0,
                                ignore_eos=True)
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
    return {"clients": level, "wall_s": round(wall, 2), "generated": gen,
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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--api-key")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--check-prompts", type=int, default=4)
    ap.add_argument("--check-tokens", type=int, default=48)
    ap.add_argument("--levels", default="1,2,4,8")
    ap.add_argument("--turns", type=int, default=4)
    ap.add_argument("--wait", type=float, default=900, help="seconds to wait for /health")
    ap.add_argument("--out")
    ap.add_argument("--skip", default="", help="parts to skip: check,throughput,conversation")
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
        for lv in [int(x) for x in a.levels.split(",") if x]:
            x = throughput(cl, lv, a.max_tokens)
            rep["throughput"].append(x)
            print("throughput:", x, flush=True)
    if "conversation" not in skip:
        rep["conversation"] = conversation(cl, a.turns, a.max_tokens)
        for i, x in enumerate(rep["conversation"]):
            print(f"turn {i + 1}:", x, flush=True)
    rep["stats"] = cl.get("/v1/stats")[1]
    print("stats:", json.dumps(rep["stats"])[:2000], flush=True)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(rep, f, indent=1)
    return 0 if rep.get("check", {}).get("pass", True) else 1


if __name__ == "__main__":
    sys.exit(main())
