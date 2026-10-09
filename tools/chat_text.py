"""A chat as a text for tools/formats_scan.py (ppl --ids): an instruct model's own replies to
QUESTIONS, asked one after another in one conversation, sampled by Hugging Face in bf16 on the
CPU with the checkpoint's generation_config.json (seeded), then the whole conversation rendered
by its chat template. The weight formats of a chat model are then scored on what a chat feeds
it, where the float model is calibrated: its own replies (an instruct model can be far from its
base on plain prose: gemma-4-E2B-it's perplexity on docs/isa.md is 142, the base's 18.6).

    python3 tools/chat_text.py MODEL OUT.json [--max-new 160] [--seed 0]

OUT.json: {"model", "seed", "max_new", "messages", "ids"} (ids: the rendering's token ids).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

QUESTIONS = [
    "What causes the seasons on Earth?",
    "Write a Python function that checks whether a string is a palindrome, and explain it "
    "briefly.",
    "Give me three tips for writing a clear email to a colleague.",
    "What is the difference between a virus and a bacterium?",
    "Summarize the plot of Romeo and Juliet in a few sentences.",
    "How do I compute the area of a circle with a radius of 7 cm? Show the steps.",
    "Suggest a vegetarian dinner I can cook in 30 minutes.",
    "Explain what a hash table is to a beginner.",
    "Translate 'Where is the train station?' into French, Spanish and German.",
    "Why is the sky blue?",
    "What are the pros and cons of working from home?",
    "Write a haiku about autumn rain.",
    "How does compound interest work? Give a small example.",
    "What should I pack for a three-day hiking trip?",
    "Explain the difference between TCP and UDP.",
    "Recommend two classic science fiction novels and say why.",
]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("model", help="a short name of opentpu.llm.MODELS or a checkpoint directory")
    ap.add_argument("out")
    ap.add_argument("--max-new", type=int, default=160, help="tokens a reply at most")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    from opentpu.llm import model_dir
    path = model_dir(a.model)
    tok = AutoTokenizer.from_pretrained(path)
    m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).eval()
    gen = GenerationConfig.from_pretrained(path)
    gen.max_new_tokens = a.max_new
    stop = set(gen.eos_token_id if isinstance(gen.eos_token_id, list) else [gen.eos_token_id])
    torch.manual_seed(a.seed)
    msgs, t0 = [], time.time()
    for q in QUESTIONS:
        msgs.append({"role": "user", "content": q})
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
        ids = list(ids["input_ids"] if hasattr(ids, "keys") else ids)
        with torch.no_grad():
            out = m.generate(torch.tensor([ids]), generation_config=gen)[0, len(ids):].tolist()
        out = out[:next((i for i, t in enumerate(out) if t in stop), len(out))]
        msgs.append({"role": "assistant", "content": tok.decode(out, skip_special_tokens=False)})
        print(f"{len(msgs) // 2}: {len(out)} tokens, {time.time() - t0:.0f} s", flush=True)
    ids = tok.apply_chat_template(msgs, tokenize=True)
    ids = [int(t) for t in (ids["input_ids"] if hasattr(ids, "keys") else ids)]
    Path(a.out).write_text(json.dumps(dict(model=path.name, seed=a.seed, max_new=a.max_new,
                                           messages=msgs, ids=ids), indent=1))
    print(f"{len(ids)} tokens in {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
