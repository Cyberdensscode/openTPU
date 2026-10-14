"""otpu-serve's HTTP API (opentpu/serve/api.py) end to end: a tiny Qwen3 on the ISA simulator
behind the server, clients over HTTP. Chat and text completions (plain and streamed, the
streamed text the plain one's), SGLang's /generate, greedy replies equal to a fresh engine's on the rendered
prompt, several clients' conversations at once (their second turns continuing the first's
context: cached tokens), stop strings, n choices, usage, the models and stats routes, errors
(a prompt over the context, a bad body, an API key), and the reasoning split."""
import http.client
import json
import threading

import numpy as np
import pytest

from opentpu.llm.qwen3 import Engine
from opentpu.serve import api as API
from opentpu.serve import scheduler as SC

from test_serve_kvio import _model

torch = pytest.importorskip("torch")

CAP = 512


class _Tok:
    """A character tokenizer for the tiny model's 1000 ids: id = code point mod 1000 (decoded
    as the code point 0x100 + id), and a Qwen-shaped chat template."""
    chat_template = "qwen-like"
    S, E = "\x01", "\x02"

    def __call__(self, text):
        return {"input_ids": [ord(c) % 1000 for c in text]}

    def apply_chat_template(self, msgs, add_generation_prompt, enable_thinking, tokenize):
        out = "".join(f"{self.S}{m['role']}\n{m['content']}{self.E}\n" for m in msgs)
        if add_generation_prompt:
            out += f"{self.S}assistant\n"
        return self(out)["input_ids"]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(0x100 + int(i)) for i in ids)


@pytest.fixture(scope="module")
def server():
    W, spec, cfg = _model("qwen3")
    eng = Engine(spec, W, cap=CAP, cfg=cfg, resident=True, prompt_runs=True)
    tok = _Tok()
    w = SC.Worker(eng, tok, quantum=4).start()
    srv = API.Server(w, tok, spec, "tiny-qwen3", dict(temperature=0.0, top_k=20, top_p=1.0,
                                                       repetition_penalty=1.0))
    httpd = API.serve(srv, "127.0.0.1", 0)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield srv, httpd.server_address[1], (W, spec, cfg)
    httpd.shutdown()
    httpd.server_close()
    w.stop()


def _call(port, method, path, body=None, headers=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=600)
    c.request(method, path, None if body is None else json.dumps(body),
              {"Content-Type": "application/json", **(headers or {})})
    r = c.getresponse()
    data = r.read()
    c.close()
    return r.status, (json.loads(data) if data and r.getheader("Content-Type", "").startswith(
        "application/json") else data)


def _stream(port, path, body):
    """The SSE events of a streamed call (the JSON objects, then 'DONE')."""
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=600)
    c.request("POST", path, json.dumps({**body, "stream": True}),
              {"Content-Type": "application/json"})
    r = c.getresponse()
    assert r.status == 200 and r.getheader("Content-Type") == "text/event-stream"
    out = []
    for line in r.read().decode().split("\n"):
        if line.startswith("data: "):
            x = line[6:]
            out.append("DONE" if x == "[DONE]" else json.loads(x))
    c.close()
    return out


def _ref(models, ids, n):
    W, spec, cfg = models
    e = Engine(spec, W, cap=CAP, cfg=cfg)
    lg, out = e.prefill(ids), []
    while len(out) < n:
        t = int(np.argmax(lg))
        if t in spec.eos:
            break
        out.append(t)
        if len(out) < n:
            lg = e.step(t)
    return out


def test_models_health_and_stats(server):
    srv, port, _ = server
    assert _call(port, "GET", "/health") == (200, {"status": "ok"})
    st, m = _call(port, "GET", "/v1/models")
    assert st == 200 and m["data"][0]["id"] == "tiny-qwen3" and m["data"][0]["max_model_len"] == CAP
    assert _call(port, "GET", "/v1/models/tiny-qwen3")[0] == 200
    assert _call(port, "GET", "/v1/models/other")[0] == 404
    st, s = _call(port, "GET", "/v1/stats")
    assert st == 200 and "kv_store" in s and "prefix_cache" in s


def test_many_connections_at_once(server):
    """64 clients connecting at the same moment are all served (the listen backlog)."""
    srv, port, _ = server
    go, res = threading.Barrier(64), []

    def client():
        go.wait()
        res.append(_call(port, "GET", "/health")[0])
    ts = [threading.Thread(target=client) for _ in range(64)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert res == [200] * 64


def test_chat_is_the_engines_greedy_reply(server):
    srv, port, models = server
    msgs = [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hello!"}]
    st, r = _call(port, "POST", "/v1/chat/completions",
                  {"model": "tiny-qwen3", "messages": msgs, "max_tokens": 7, "temperature": 0})
    assert st == 200, r
    ids = _Tok().apply_chat_template(msgs, True, False, True)
    want = _Tok().decode(_ref(models, ids, 7))
    c = r["choices"][0]
    assert c["message"] == {"role": "assistant", "content": want}
    assert c["finish_reason"] == "length" and r["object"] == "chat.completion"
    assert r["usage"]["prompt_tokens"] == len(ids) and r["usage"]["completion_tokens"] == 7
    # streamed: the same text in deltas, a role first, the finish, the usage, [DONE]
    ev = _stream(port, "/v1/chat/completions", {"messages": msgs, "max_tokens": 7,
                                                "temperature": 0,
                                                "stream_options": {"include_usage": True}})
    assert ev[-1] == "DONE" and ev[0]["choices"][0]["delta"]["role"] == "assistant"
    text = "".join(e["choices"][0]["delta"].get("content", "") for e in ev[:-1] if e["choices"])
    assert text == want
    assert [e["choices"][0]["finish_reason"] for e in ev[:-2]][-1] == "length"
    assert ev[-2]["usage"]["completion_tokens"] == 7 and ev[-2]["choices"] == []
    assert ev[-2]["usage"]["prompt_tokens_details"]["cached_tokens"] == len(ids) - 1


def test_clients_at_once_and_their_next_turns(server):
    """Four clients' two-turn conversations at once: every reply is the engine's greedy reply on
    its rendering; each second turn continues its first's context (cached tokens)."""
    srv, port, models = server
    tok = _Tok()
    res, errs = {}, []

    def client(i):
        try:
            msgs = [{"role": "user", "content": f"question {i}: " + "why " * (3 + i)}]
            st, r = _call(port, "POST", "/v1/chat/completions",
                          {"messages": msgs, "max_tokens": 6, "temperature": 0})
            a1 = r["choices"][0]["message"]["content"]
            msgs2 = msgs + [{"role": "assistant", "content": a1},
                            {"role": "user", "content": f"and {i}?"}]
            st2, r2 = _call(port, "POST", "/v1/chat/completions",
                            {"messages": msgs2, "max_tokens": 6, "temperature": 0})
            res[i] = (msgs, r, msgs2, r2)
        except Exception as e:      # noqa: BLE001
            errs.append(e)
    ts = [threading.Thread(target=client, args=(i,)) for i in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert not errs, errs
    for i, (msgs, r, msgs2, r2) in res.items():
        for m, x in ((msgs, r), (msgs2, r2)):
            ids = tok.apply_chat_template(m, True, False, True)
            assert x["choices"][0]["message"]["content"] == tok.decode(_ref(models, ids, 6)), i
        # turn 2's prompt shares turn 1's prompt (the reply's text re-encoded may differ)
        p1 = len(tok.apply_chat_template(msgs, True, False, True))
        assert r2["usage"]["prompt_tokens_details"]["cached_tokens"] >= p1, i


def test_completions_stop_and_n(server):
    srv, port, models = server
    ids = [5, 17, 300, 42, 8, 99, 123]
    want = _Tok().decode(_ref(models, ids, 10))
    st, r = _call(port, "POST", "/v1/completions", {"prompt": ids, "max_tokens": 10,
                                                    "temperature": 0})
    assert st == 200 and r["choices"][0]["text"] == want and r["object"] == "text_completion"
    stop = want[4:6]
    st, r = _call(port, "POST", "/v1/completions", {"prompt": ids, "max_tokens": 10,
                                                    "temperature": 0, "stop": [stop]})
    assert r["choices"][0]["text"] == want[:want.find(stop)]
    assert r["choices"][0]["finish_reason"] == "stop"
    ev = _stream(port, "/v1/completions", {"prompt": ids, "max_tokens": 10, "temperature": 0})
    assert "".join(e["choices"][0]["text"] for e in ev[:-1]) == want
    st, r = _call(port, "POST", "/v1/completions", {"prompt": "abc", "max_tokens": 3,
                                                    "temperature": 0, "n": 2, "echo": True})
    assert st == 200 and len(r["choices"]) == 2
    assert all(c["text"].startswith("abc") for c in r["choices"])
    assert r["choices"][0]["text"] == r["choices"][1]["text"]      # greedy: the same
    # sampled, seeded: the same seed the same text
    body = {"prompt": ids, "max_tokens": 6, "temperature": 1.0, "top_k": 20, "seed": 7}
    a = _call(port, "POST", "/v1/completions", body)[1]["choices"][0]["text"]
    b = _call(port, "POST", "/v1/completions", body)[1]["choices"][0]["text"]
    assert a == b and len(a) == 6


def test_sglang_generate(server):
    """SGLang's /generate: input_ids or text (one, or a list), sampling_params, meta_info; streamed:
    each event the whole text so far, then [DONE]."""
    srv, port, models = server
    ids = [5, 17, 300, 42, 8, 99, 123]
    want_ids = _ref(models, ids, 8)
    st, r = _call(port, "POST", "/generate", {"input_ids": ids, "sampling_params": {
        "max_new_tokens": 8, "temperature": 0}})
    assert st == 200, r
    assert r["output_ids"] == want_ids and r["text"] == _Tok().decode(want_ids)
    m = r["meta_info"]
    assert m["prompt_tokens"] == len(ids) and m["completion_tokens"] == 8
    assert m["finish_reason"] == {"type": "length", "length": 8} and m["e2e_latency"] > 0
    st, rs = _call(port, "POST", "/generate", {"text": ["ab", "cd"], "sampling_params": {
        "max_new_tokens": 3, "temperature": 0, "ignore_eos": True}})
    assert st == 200 and isinstance(rs, list) and len(rs) == 2
    for x, t in zip(rs, ["ab", "cd"]):
        assert x["output_ids"] == _ref(models, _Tok()(t)["input_ids"], 3)
    ev = _stream(port, "/generate", {"input_ids": ids, "sampling_params": {
        "max_new_tokens": 8, "temperature": 0}})
    assert ev[-1] == "DONE" and ev[-2]["text"] == r["text"]
    assert ev[-2]["meta_info"]["finish_reason"]["type"] == "length"
    texts = [e["text"] for e in ev[:-1]]
    assert all(b.startswith(a) for a, b in zip(texts, texts[1:]))
    assert _call(port, "POST", "/generate", {"sampling_params": {}})[0] == 400


def test_errors(server):
    srv, port, _ = server
    st, r = _call(port, "POST", "/v1/completions", {"prompt": list(range(CAP)), "max_tokens": 1})
    assert st == 400 and "context length" in r["error"]["message"]
    st, r = _call(port, "POST", "/v1/chat/completions", {"messages": []})
    assert st == 400
    st, r = _call(port, "POST", "/v1/chat/completions",
                  {"messages": [{"role": "user", "content": [{"type": "image_url"}]}]})
    assert st == 400 and "text only" in r["error"]["message"]
    assert _call(port, "POST", "/v1/nothing", {})[0] == 404
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
    c.request("POST", "/v1/completions", "{not json", {"Content-Length": "9"})
    assert c.getresponse().status == 400
    srv.api_key = "k"
    try:
        assert _call(port, "GET", "/v1/models")[0] == 401
        assert _call(port, "GET", "/v1/models", headers={"Authorization": "Bearer k"})[0] == 200
    finally:
        srv.api_key = None


def test_think_split():
    for inside, text, want in [(False, "<think>\nhm\n</think>\n\nIt is.", ("\nhm\n", "It is.")),
                               (True, "hm</think>\nok", ("hm", "ok")),
                               (False, "no block", ("", "no block"))]:
        sp, rs, cs = API.ThinkSplit(inside=inside), "", ""
        for ch in text:
            a, b = sp.feed(ch)
            rs, cs = rs + a, cs + b
        a, b = sp.feed("", final=True)
        assert (rs + a, cs + b) == want


class _HeldReads:
    """PoolFile's reads held until released (a slow disk); the read of expert `fail` raises."""

    def __init__(self, monkeypatch, slot, fail=None):
        from opentpu.host import offload as OF
        self.sem, real = threading.Semaphore(0), OF.preadv

        def read(fd, bufs, off):
            assert self.sem.acquire(timeout=60)
            if fail is not None and off // slot == fail:
                raise OSError(5, "Input/output error")
            real(fd, bufs, off)
        monkeypatch.setattr(OF, "preadv", read)

    def release(self, k):
        for _ in range(k):
            self.sem.release()


def _until(cond, s=30.0):
    import time
    t = time.time()
    while not cond():
        assert time.time() - t < s
        time.sleep(0.01)


def _pool_server(tmp_path, cap, n=8, slot=4096 * 2):
    """A Server over a streamed MoE's pool file alone (a stand-in worker and engine), on HTTP."""
    from types import SimpleNamespace

    from opentpu.host.offload import PoolFile
    f = tmp_path / "pool.bin"
    f.write_bytes(np.random.default_rng(0).integers(0, 256, n * slot, dtype=np.uint8).tobytes())
    pf = PoolFile(f, slot, split=False)
    pf.ids = np.arange(n)
    w = SimpleNamespace(tok_lock=threading.Lock(), usage=lambda: {"decode_s": 0, "generated": 0},
                        eng=SimpleNamespace(cap=CAP, server=SimpleNamespace(pool_file=pf)))
    srv = API.Server(w, _Tok(), None, "pool", {}, warm_cap=cap)
    httpd = API.serve(srv, "127.0.0.1", 0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return pf, srv, httpd


def test_health_is_warming_while_the_pool_is_read(tmp_path, monkeypatch):
    """A streamed MoE's /health (moe.rewarm's read of its pool into the page cache): 503
    "warming" with the share read while the warm thread runs, ok once it has read all of it;
    /v1/stats keeps the read's record. A model without a pool: ok (test_models_health_and_stats)."""
    from opentpu.host.offload import warm_report
    pf, srv, httpd = _pool_server(tmp_path, cap=60.0)
    port = httpd.server_address[1]
    reads = _HeldReads(monkeypatch, pf.slot)
    try:
        t = pf.warm(pf.ids)
        assert _call(port, "GET", "/health") == (503, {"status": "warming", "progress": 0.0})
        reads.release(4)
        _until(lambda: t.bytes == 4 * pf.slot)
        assert _call(port, "GET", "/health") == (503, {"status": "warming", "progress": 0.5})
        assert _call(port, "GET", "/health_generate")[0] == 503
        reads.release(4)
        t.join(60)
        assert _call(port, "GET", "/health") == (200, {"status": "ok"})
        st, s = _call(port, "GET", "/v1/stats")
        r = s["pool_warm"]
        assert st == 200 and r == warm_report(t)
        assert r["of_gb"] == r["gb"] == round(8 * pf.slot / 1e9, 2) and r["end"] >= r["start"]
        assert r["error"] is None and not r["stopped"] and r["resident"] in (None, 1.0)
    finally:
        reads.release(64)
        httpd.shutdown()
        httpd.server_close()


def test_health_is_ok_past_the_warm_cap_and_when_the_read_fails(tmp_path, monkeypatch):
    """/health's 503 lasts at most warm_cap seconds from the read's start (then ok, "warm":
    "incomplete", the cap in its note; warm_cap 0: never 503). A read that raises ends the warm
    thread with its error, and /health says ok, incomplete, with the error; so does a thread
    gone without its end recorded."""
    import time
    from types import SimpleNamespace
    pf, srv, httpd = _pool_server(tmp_path, cap=0.3)
    port = httpd.server_address[1]
    reads = _HeldReads(monkeypatch, pf.slot, fail=5)
    try:
        t = pf.warm(pf.ids)
        assert srv.health() == (503, {"status": "warming", "progress": 0.0})
        _until(lambda: time.time() - t.t0 > 0.3)
        assert srv.health() == (200, {"status": "ok", "warm": "incomplete", "progress": 0.0,
                                      "note": "the pool's warm runs past the 0.3 s cap"})
        srv.warm_cap = 0
        assert srv.health()[0] == 200
        srv.warm_cap = 60.0
        assert srv.health()[0] == 503
        reads.release(8)
        t.join(60)
        assert t.error == "OSError: [Errno 5] Input/output error" and t.bytes == 5 * pf.slot
        assert _call(port, "GET", "/health") == (200, {"status": "ok", "warm": "incomplete",
                                                       "progress": 0.625, "note": t.error})
        pf.warm_t = SimpleNamespace(is_alive=lambda: False, bytes=0, total=pf.slot,
                                    t0=time.time(), t1=None, error=None)
        assert srv.health() == (200, {"status": "ok", "warm": "incomplete", "progress": 0.0,
                                      "note": "the pool's warm thread died"})
    finally:
        reads.release(64)
        httpd.shutdown()
        httpd.server_close()


def test_the_warm_log_says_when_the_read_ends(tmp_path, monkeypatch):
    """log_warm (otpu-serve's start): a line as the server starts, and the read's record when
    it ends (start, end, GB, MB/s, the pool's share in the page cache), or why it did not."""
    import io

    from opentpu.host.offload import warm_report
    pf, srv, httpd = _pool_server(tmp_path, cap=60.0)
    reads = _HeldReads(monkeypatch, pf.slot, fail=7)
    try:
        out = io.StringIO()
        t = pf.warm(pf.ids)
        API.log_warm(t, out)
        reads.release(8)
        t.join(60)
        _until(lambda: out.getvalue().count("\n") == 2)
        first, last = out.getvalue().splitlines()
        r = warm_report(t)
        assert first.startswith("pool warm: started ") and first.endswith(" GB read")
        assert last.startswith("pool warm: incomplete (OSError: [Errno 5] Input/output error): ")
        assert f", {r['gb']:g} GB, {r['mb_s']} MB/s; " in last
        assert last.endswith("of the pool in the page cache")
    finally:
        reads.release(64)
        httpd.shutdown()
        httpd.server_close()
