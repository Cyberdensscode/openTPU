"""otpu-chat with a mixture-of-experts model streamed from its pool (docs/offload.md 14), scripted:
a few turns of one conversation through otpu-chat's own Chat, engine and sampler, on the ISA
simulator or the card, each turn's tokens and numbers as JSON. The run on the simulator with the
card's configuration (tools/qual/refs.py cfg) is the card's reference bit for bit: --check
compares every turn's tokens with it.

    python3 tools/qual/chatmoe.py MODEL --cfg cfg.pkl --out ref.json              # ISA simulator
    python3 tools/qual/chatmoe.py MODEL --backend board --check ref.json --out card.json

MODEL: a short name of opentpu.llm.MODELS (lfm2-8b, qwen35-35b, gemma4-26b) or a checkpoint
directory. The turns default to TURNS (two: a question, then one about the answer), greedy
(--sample: the model's sampling defaults, the uniforms from --seed's generator: the card's loop
picks the same tokens as the simulator's), --max-new tokens each. The pool, slots and prompt
settings are otpu-chat's (opentpu/host/streamed.py). Exit 1 when --check finds another token.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

TURNS = ("What is the capital of France? Answer in one sentence.",
         "And what is it famous for? Two sentences.")
DATE = (2026, 9, 29)        # "today" for a chat template that states it (refs.py's DATE)


class _Pinned:
    """The tokenizer with its chat template's date pinned (a reference must not expire)."""

    def __init__(self, tok):
        self.tok = tok

    def __getattr__(self, k):
        return getattr(self.tok, k)

    def __call__(self, *a, **kw):
        return self.tok(*a, **kw)

    def apply_chat_template(self, *a, **kw):
        import datetime
        day = datetime.date(*DATE)
        return self.tok.apply_chat_template(*a, strftime_now=lambda fmt: day.strftime(fmt), **kw)


def run(model, backend: str = "isa", cfg_file: str | None = None, dev: str = "/dev/xdma0",
        turns=TURNS, max_new: int = 16, cap: int = 2048, pool: str | None = None,
        experts: int = 0, layer_major: int | None = None, sample: bool = False,
        seed: int = 0, create: bool = False, tok=None) -> dict:
    from opentpu.host import chat as C
    from opentpu.host import streamed as ST
    from opentpu.isasim import board_config
    from opentpu.llm import load_spec, model_dir
    path = model_dir(model)
    if tok is None:
        from transformers import AutoTokenizer
        tok = _Pinned(AutoTokenizer.from_pretrained(path))
    spec = load_spec(path)
    wformat, head, formats = ST.defaults(spec)
    be, cfg = C.make_backend(backend, spec, cap, dev, path.name)
    if cfg_file is not None:                    # (the card's: refs.py cfg)
        want = pickle.loads(Path(cfg_file).read_bytes())
        if cfg is not None and cfg != want:
            raise SystemExit(f"the device's configuration is not {cfg_file}'s: {cfg} / {want}")
        cfg = want
    t = time.time()
    eng = ST.engine(path, cfg or board_config(), be, cap, pool or ST.DEFAULT, experts=experts,
                    wformat=wformat, head_format=head, formats=formats, layer_major=layer_major,
                    create=create, snapshots=C.SNAPSHOTS)     # (otpu-chat's: its turns roll back)
    load_s = time.time() - t
    sp = C.sampling(spec, argparse.Namespace(), path)
    pick = C.sampler(sp["temperature"] if sample else 0, sp["top_k"], sp["top_p"], seed,
                     sp["repetition_penalty"], getattr(spec, "softcap", None))
    khz = (getattr(eng.backend, "info", None) or {}).get("core_khz")
    chat = C.Chat(eng, tok, False, pick, max_new, clock_mhz=khz / 1e3 if khz else 0.0)
    out = dict(model=path.name, backend=backend, cfg=cfg_file, sample=sample, seed=seed,
               weights=C.weights_label(eng.image, wformat),
               experts=ST.describe(eng, eng.pool_path), layer_major=eng.layer_major,
               on_card=chat.on_card, load_s=round(load_s, 1), turns=[])
    for text in turns:
        n0 = len(chat.fed)
        reply, tr = chat.ask(text)
        r = lambda x, k=3: None if x is None else round(x, k)      # noqa: E731
        out["turns"].append(dict(
            text=text, reply=reply, tokens=list(chat._reply), end=tr.end, fed_before=n0,
            fed=[int(t) for t in chat.fed],
            prompt_tokens=tr.prefill_tokens, reused=tr.reused, restarted=tr.restarted,
            context=tr.context, ttft_s=r(tr.ttft_s),
            prefill_s=r(tr.prefill_s), prefill_tok_s=r(tr.prefill_tok_s, 2),
            prefill_dev_tok_s=r(tr.prefill_dev_tok_s, 2), gen_tokens=tr.gen_tokens,
            decode_tok_s=r(tr.decode_tok_s, 2), decode_dev_tok_s=r(tr.decode_dev_tok_s, 2),
            hits=tr.expert_hits, misses=tr.expert_misses,
            hit_rate=r(tr.expert_hit_rate, 4), mb_per_token=r(tr.streamed_mb_per_token, 2),
            prefill_misses=tr.prefill_misses, line=tr.line()))
    out["host_mem"] = ST.host_mem()
    return out


def check(got: dict, ref: dict) -> list:
    """The turns whose tokens differ from the reference's: (turn, first different index)."""
    bad = []
    for i, (a, b) in enumerate(zip(got["turns"], ref["turns"])):
        if a["tokens"] != b["tokens"]:
            bad.append((i, next((j for j, (x, y) in enumerate(zip(a["tokens"], b["tokens"]))
                                 if x != y), min(len(a["tokens"]), len(b["tokens"])))))
    if len(got["turns"]) != len(ref["turns"]):
        bad.append((min(len(got["turns"]), len(ref["turns"])), None))
    return bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("model")
    ap.add_argument("--backend", choices=("isa", "board"), default="isa")
    ap.add_argument("--dev", default="/dev/xdma0")
    ap.add_argument("--cfg", help="the card's configuration, pickled (tools/qual/refs.py cfg); "
                                  "on the card it must be the device's")
    ap.add_argument("--turn", action="append", help="a user turn (repeated; default TURNS)")
    ap.add_argument("--max-new", type=int, default=16, help="tokens a reply at most")
    ap.add_argument("--cap", type=int, default=2048)
    ap.add_argument("--pool", help="the expert pool (default: the checkpoint's, as otpu-chat)")
    ap.add_argument("--pool-create", action="store_true",
                    help="a missing pool is made, each expert packed on its first use (a "
                         "reference on a host with the whole checkpoint and no pool)")
    ap.add_argument("--experts", type=int, default=0, help="slots a MoE layer (0: fill)")
    ap.add_argument("--layer-major", type=int, metavar="R")
    ap.add_argument("--sample", action="store_true", help="the model's sampling defaults")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check", help="a reference run's JSON: its tokens, turn by turn")
    ap.add_argument("--out", help="this run's JSON")
    a = ap.parse_args(argv)
    r = run(a.model, a.backend, a.cfg, a.dev, a.turn or TURNS, a.max_new, a.cap, a.pool,
            a.experts, a.layer_major, a.sample, a.seed, a.pool_create)
    for i, t in enumerate(r["turns"]):
        print(f"turn {i + 1}: {t['reply']!r}\n  {t['line']}", flush=True)
    rc = 0
    if a.check:
        ref = json.loads(Path(a.check).read_text())
        bad = check(r, ref)
        r["check"] = dict(ref=a.check, ok=not bad, differ=bad)
        print(f"[{'FAIL' if bad else 'PASS'}] {r['model']} against {a.check}: " +
              (", ".join(f"turn {i + 1} from token {j}" for i, j in bad) if bad else
               f"{len(r['turns'])} turns' tokens the same"), flush=True)
        rc = 1 if bad else 0
    if a.out:
        Path(a.out).write_text(json.dumps(r, indent=1))
    return rc


if __name__ == "__main__":
    sys.exit(main())
