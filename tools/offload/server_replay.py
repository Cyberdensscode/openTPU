"""The expert server's slot decisions replayed offline (docs/offload.md 15.4): a chat's server calls
recorded once on the ISA simulator, then replayed through this tree's ExpertServer, or a variant of
its slot order, on a stand-in memory with the record's layout.

    python3 tools/offload/server_replay.py record OUT.json -- CHATMOE_ARGS...
    python3 tools/offload/server_replay.py replay REC.json [--variant NAME ...] [--json OUT]

record runs tools/qual/chatmoe.py (its arguments after --) with the server's calls logged in order:
the load's warm set, each prefill's begin (expert-major or not, its scratch slots) and end, the
requests (the ids a poll serves) and the need lines, and each turn's start. The routing is the
card's (the programs are the image's) and does not depend on the server: the outputs are the
same bit for bit whatever it keeps, so one record serves every slot policy.

replay feeds the record to a fresh ExpertServer: requests through serve, need lines through need,
the needs landing on idle polls at each layer switch (the expert run of a layer waits for all of
its needs before the next layer's lines come), prefills through begin_prefill / end_prefill. Per
turn: the prompt's misses and needs queued (what the prompt streams), and the decode's hits, hit
rate, misses a token and MB a token (MB: misses x the record's slot bytes). A replay of the
card's server reproduces the card's hit rates (the card's timing changes no slot decision but
the needs' landing order within a layer).

The variants (--variant, default "base": this tree's server as it is):
- count: a need line counts as a request does (a use of each id, one more of its layer's), as
  les's slot order (O, ee08e88c) does;
- order: at the layer switch the finished layer's experts the need lines named go to the end of
  the pooled order, by the last line that named each (O's other half);
- O: both (les's server);
- keep: the need lines' uses count only for end_prefill's choice of what each layer keeps;
  decode's use counts are then as without them;
- nodecay: a need line adds a use of each id at its layer's time without a step of it (the
  earlier uses do not decay over the prompt);
- long: count, but an expert-major call of fewer than LONG_LINES need lines in its first layer
  (about 2 x LONG_LINES rows) leaves the uses as they were before it.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def record(out: str, args: list) -> None:
    sys.path.insert(0, str(ROOT / "tools" / "qual"))
    import chatmoe
    from opentpu.host import chat as C
    from opentpu.host.offload import ExpertServer

    ev: list = []
    info: dict = {}
    srv_cls = ExpertServer
    base = {k: getattr(srv_cls, k) for k in ("load", "serve", "need", "begin_prefill",
                                              "end_prefill")}

    def load(self, warm=()):
        L = self.L
        info.update(layers=L.layers, E=L.E, k=L.k, slots=[n for _, n in L.slots],
                    slot_bytes=L.slot_bytes, half=self.half)
        ev.append(["load", [int(g) for g in warm]])
        return base["load"](self, warm)

    def serve(self, ids, pos=None):
        info["policy"] = self.policy        # (streamed.engine's, set after the load)
        ev.append(["serve", [int(g) for g in ids]])
        return base["serve"](self, ids, pos)

    def need(self, ids):
        ev.append(["need", [int(g) for g in ids]])
        return base["need"](self, ids)

    def begin_prefill(self, ahead=False, part=None, expert_major=False, scratch=0):
        r = base["begin_prefill"](self, ahead, part, expert_major, scratch)
        ev.append(["begin", bool(expert_major), len(self.scratch_slots)])
        return r

    def end_prefill(self, restore="lazy"):
        ev.append(["end", restore])
        return base["end_prefill"](self, restore)
    for k, f in (("load", load), ("serve", serve), ("need", need),
                 ("begin_prefill", begin_prefill), ("end_prefill", end_prefill)):
        setattr(srv_cls, k, f)
    ask = C.Chat.ask

    def asked(self, text, *a, **kw):
        ev.append(["turn", text[:60]])
        r = ask(self, text, *a, **kw)
        ev.append(["reply", len(self._reply)])
        return r
    C.Chat.ask = asked
    rc = chatmoe.main(args)
    Path(out).write_text(json.dumps(dict(info=info, events=ev)))
    print(f"record: {len(ev)} events, {sum(e[0] == 'serve' for e in ev)} requests, "
          f"{sum(e[0] == 'need' for e in ev)} need lines -> {out}", flush=True)
    sys.exit(rc)


VARIANTS = ("base", "count", "order", "O", "keep", "nodecay", "long")
LONG_LINES = 50         # "long": an expert-major call's need lines in its first layer (~2 rows each)


def _variant(srv, name: str) -> None:
    """The slot order `name` on server srv (this tree's ExpertServer: production's)."""
    if name == "base":
        return
    if name not in VARIANTS:
        raise SystemExit(f"variant {name!r}: one of {', '.join(VARIANTS)}")
    need0, at0, end0, begin0 = srv.need, srv._at, srv.end_prefill, srv.begin_prefill
    srv.last_line, srv.lines, srv.saved, srv.first, srv.first_lines = {}, 0, None, None, 0
    counted = name in ("count", "O", "keep", "nodecay", "long")

    def begin_prefill(*a, **kw):
        r = begin0(*a, **kw)
        if name in ("keep", "long") and srv.expert_major:     # (the uses before the call)
            srv.saved = ([dict(u) for u in srv.use], list(srv.t))
        srv.first, srv.first_lines = None, 0
        return r

    def need(ids):
        j = srv._layer(ids)
        need0(ids)
        srv.lines += 1
        if srv.first is None:
            srv.first = j
        srv.first_lines += j == srv.first
        for g in ids:
            srv.last_line[g] = srv.lines
        if not counted:
            return
        if name != "nodecay":               # (nodecay: a use at the layer's time, no step)
            srv.t[j] += 1
        use, t = srv.use[j], srv.t[j] / srv.half
        for g in ids:
            use[g] = math.log2(2.0 ** (use[g] - t) + 1.0) + t if g in use else t

    def at(j):
        if srv.expert_major and j != srv._cur and name in ("order", "O") and \
                srv._cur is not None:
            E = srv.L.E
            want = [g for g in srv.last_line if g in srv.order and g // E == srv._cur]
            for g in sorted(want, key=srv.last_line.get):
                srv.order.move_to_end(g)
        if srv.expert_major and j != srv._cur:
            srv.last_line = {g: v for g, v in srv.last_line.items() if g // srv.L.E == j}
        at0(j)

    def end_prefill(restore="lazy"):
        saved, srv.saved = srv.saved, None
        if saved is not None and name == "keep":
            end0(restore)                   # what each layer keeps: by the prompt's uses
            srv.use, srv.t = saved          # decode's counts: as without them
            return
        if saved is not None and name == "long" and srv.first_lines < LONG_LINES:
            srv.use, srv.t = saved          # a short prompt's uses dropped
        end0(restore)
    srv.need, srv._at, srv.end_prefill, srv.begin_prefill = need, at, end_prefill, begin_prefill


def replay(rec: dict, variant: str = "base") -> list:
    from opentpu.host.offload import ExpertServer, Layout, SimDram
    info = rec["info"]
    SLOT = 4096
    lay = Layout.build(4096, info["E"], info["k"], info["slots"], SLOT)
    mem = SimDram(np.zeros(lay.end + 4096, np.uint8))
    data = bytes(SLOT)
    srv = ExpertServer(mem, lay, lambda g: data)
    srv.policy = info.get("policy", "lfu")
    srv.half = info["half"]
    srv.part = SLOT
    srv.read_ahead = srv.halt_aware = False
    _variant(srv, variant)
    mb = info["slot_bytes"] / 1e6
    turns, cur, in_prefill, need_layer = [], None, False, None

    def idle():                             # the needs land on idle polls, in order
        while srv.pending or srv._next_ahead():
            srv.step()
    for e in rec["events"]:
        k = e[0]
        if k == "load":
            srv.load(e[1])
        elif k == "turn":
            cur = dict(text=e[1], p_misses=0, p_queued=0, d_hits=0, d_misses=0, tokens=0)
            turns.append(cur)
            c0 = (srv.hits, srv.misses, srv.needs_queued)
        elif k == "begin":
            m = srv.misses
            sc = e[2] * lay.pitch
            srv.begin_prefill(expert_major=e[1], scratch=sc)
            in_prefill, need_layer = True, None
        elif k == "need":
            j = e[1][0] // lay.E
            if need_layer is not None and j != need_layer:
                idle()
            need_layer = j
            srv.need(e[1])
        elif k == "serve":
            if in_prefill:
                srv.serve(e[1])
            else:
                h, m = srv.hits, srv.misses
                srv.serve(e[1])
                if cur is not None:
                    cur["d_hits"] += srv.hits - h
                    cur["d_misses"] += srv.misses - m
        elif k == "end":
            idle()
            srv.end_prefill(e[1])
            in_prefill = False
            if cur is not None:
                cur["p_misses"] = srv.misses - c0[1] - cur["d_misses"]
                cur["p_queued"] = srv.needs_queued - c0[2]
        elif k == "reply":
            if cur is not None:
                cur["tokens"] = e[1]
    for t in turns:
        n = t["d_hits"] + t["d_misses"]
        steps = max(1, t["tokens"] - 1)     # (the decode's steps: the last token not fed)
        t.update(hit_rate=round(t["d_hits"] / max(1, n), 4),
                 misses_tok=round(t["d_misses"] / steps, 1),
                 mb_tok=round(t["d_misses"] * mb / steps, 1))
    return turns


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("out")
    r.add_argument("args", nargs=argparse.REMAINDER)
    p = sub.add_parser("replay")
    p.add_argument("rec")
    p.add_argument("--variant", action="append")
    p.add_argument("--json")
    a = ap.parse_args(argv)
    if a.cmd == "record":
        args = a.args[1:] if a.args[:1] == ["--"] else a.args
        return record(a.out, args)
    rec = json.loads(Path(a.rec).read_text())
    out = {}
    for v in a.variant or ["base"]:
        out[v] = turns = replay(rec, v)
        for i, t in enumerate(turns):
            print(f"{v:6s} turn {i + 1}: prompt misses {t['p_misses']:5d} needs queued "
                  f"{t['p_queued']:5d} | decode hits {t['hit_rate']:.4f}, {t['misses_tok']:5.1f}"
                  f" misses / {t['mb_tok']:6.1f} MB a token ({t['tokens']} tokens)", flush=True)
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
