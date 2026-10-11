# otpu-serve: an OpenAI-style server on the card

`otpu-serve` serves one model on the card to any number of clients over an OpenAI-compatible
HTTP API. It keeps every user's context, on the card while it runs and in the host's RAM and
on disk after that, and it decodes several users' requests together in one run of the card.
In that it follows SGLang: a prefix cache of contexts (SGLang's RadixAttention), a hierarchy of
memory under it (its HiCache: GPU, host RAM, disk), and continuous batching.

```sh
otpu-serve --model qwen3 --backend board                     # http://127.0.0.1:8000/v1
otpu-serve --model qwen3 --backend board --kv-ram-gb 8 --kv-disk-gb 32 --kv-dir /data/kv
otpu-serve --model lfm2 --backend board --port 8001          # one sequence: requests in turns
```

Any OpenAI client works against it (the `openai` Python package, checked: models, chat plain and
streamed with usage, completions):

```python
from openai import OpenAI
c = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="none")
r = c.chat.completions.create(model="Qwen3-0.6B", messages=[{"role": "user", "content": "Hi!"}])
```

## The API

| Route | |
|---|---|
| `POST /v1/chat/completions` | the chat template as otpu-chat renders it; `stream` (SSE), `n`, `max_tokens` / `max_completion_tokens`, `temperature`, `top_p`, `top_k`, `repetition_penalty`, `seed`, `stop` (strings), `stop_token_ids`, `ignore_eos`, `stream_options.include_usage`, `chat_template_kwargs.enable_thinking`, `skip_special_tokens`; content as a string or text parts |
| `POST /v1/completions` | a prompt string or token ids, `echo`, the same sampling and stops |
| `POST /generate` | SGLang's native call: `text` or `input_ids` (one, or a list), `sampling_params` (`max_new_tokens` (default 128), `temperature`, `top_p`, `top_k` (-1: all), `repetition_penalty`, `stop`, `stop_token_ids`, `ignore_eos`, `n`), `stream` (each event the whole text so far); `output_ids` and `meta_info` (prompt, completion and cached tokens, `finish_reason`, `e2e_latency`) |
| `GET /v1/models` | the model, its context length (`max_model_len`) |
| `GET /v1/stats` | the queue, throughput, the slots, the KV store and prefix cache, the steps' timing |
| `GET /health`, `/health_generate`, `/get_model_info`, `/get_server_info` | |
| `POST /flush_cache` | forget every saved context and what idle slots hold |

`usage.prompt_tokens_details.cached_tokens` counts the prompt tokens the server did not feed
(the card or the store held them). Sampling defaults are the model's (otpu-chat's: the
checkpoint's generation_config.json, or its family's), and a request's fields override them. A
top-k above 64 (or none) is 64, the most the card's sampler takes. `--reasoning-parser qwen3`
moves a reply's `<think> ... </think>` to `reasoning_content`, also while it streams. `--api-key`
(or `OTPU_API_KEY`) requires `Authorization: Bearer <key>`. Not supported: tools, images,
logprobs, `suffix`; `presence_penalty` and `frequency_penalty` are ignored.

## The KV cache: pages, the store, the prefix cache

**A context's bytes** (`opentpu/serve/kvio.py`). Every positional KV cache keeps the positions
`[0, n)` in contiguous byte ranges:

- K's first n rows, and its scales;
- V^T's first `ceil(n / 256)` tiles, since V^T is stored in tiles of 256 tokens, each `[d, 256]`
  contiguous (`compiler.KVDesc`);
- V's first n scales.

A **page** is 256 positions of every cache of every layer: 15.1 MB for Qwen3-0.6B. Page i
holds what the tokens `[0, 256 (i + 1))` put there and nothing else, so contexts with a common
prefix share its whole pages. A context of n tokens is:

- its whole pages;
- its last partial page;
- the state the KV caches do not hold (`Image.state_regions`: Qwen3.5's DeltaNet states, LFM2's
  convolution rings, Gemma 4's sliding rings).

Writing the pages back and setting the position (`Engine.adopt`) gives the context bit for bit,
as `Engine.rollback` does. On the card, a save or restore is staged: LD / ST copy programs
gather a page's ~900 regions into an area of the DRAM past the program area, and one DMA moves
it. Each region's own DMA would cost ~30 us of set-up.

**The store** (`opentpu/serve/store.py`, `BlobStore`). Pages and tails live in host RAM up to
`--kv-ram-gb`. Past that, the least recently used go to files under `--kv-dir`, written by a
thread, up to `--kv-disk-gb`; past that they are dropped. A blob read from disk comes back into
RAM. An active request's blobs are pinned. The entries are written through to disk, and at
shutdown the RAM goes too, so a server started on the same directory continues the contexts
saved before. The directory is keyed by the model, its formats and layout.

**The prefix cache** (`PrefixCache`). Entries are the contexts saved, by their tokens; a page's
key hashes the tokens up to its end. `match(prompt)` finds where the card can continue a
prompt:

- for a model whose KV caches hold the whole context, any common prefix of a saved context (its
  pages up to there; what lies past is overwritten);
- for a model with state, the end of a saved context that the prompt continues.

A prompt always feeds its last token, whose logits pick the first one of the reply.

## Requests at once

**Batched, in KV slots** (`opentpu/serve/batching.py`, `BatchWorker`: Qwen3 and the Llama-likes,
`--slots`, default 8). The image holds B sequences' KV caches (`Engine(batch=B,
batch_area=True)`). Each iteration of the worker:

1. Waiting requests take free slots: the one holding the longest prefix of the prompt, else
   slot 0, else the least recently used. The prompt continues the longest prefix found in that
   slot, in another slot (its pages copied on the card, no PCIe) or in the store. A slot's
   context is saved to the store before it is overwritten.
2. Feed runs of the prompts being fed, oldest first, up to `--chunked-prefill-size` tokens
   (default 64, SGLang's chunked prefill; more: a shorter time to first token, longer pauses
   of the decoding requests). A feed run takes up to MCOLS (4) tokens of a prompt, every
   weight streamed once, no LM head (`Steps.feed`). A prompt feeds all but its last token
   this way.
3. One batched step for every decoding request (`opentpu/llm/batch.py`):
   - a run feeds each one its token from DRAM; the embedding rows are gathered and the RoPE
     rows loaded from the image's tables;
   - every weight streams once for all of them;
   - the LM head hands each chunk to every row's sink: the decode loop's `Greedy`, or its
     `Sampler` with the slot's words (top-k, temperature, top-p, repetition penalty) and the
     request's uniform;
   - the picks go to DRAM.

   A request's first step feeds its prompt's last token, so its first token is picked on the
   card too. The host writes the tokens and reads one word per request: no logits leave the
   card, and the host computes nothing of the model.

A request decoding alone in slot 0 runs the card's decode loop instead (`generate_card`),
halted after the token in flight when another arrives. A step's rows run in groups whose program
fits IMEM. The LM head's sinks and each row's attention grow it, so four rows of Qwen3-0.6B fit
early in the context and fewer late in it.

**Span programs.** A step's program would take its rows' positions at compile time, and the
host's compile of a 4-row step (~50-130 ms on the card host's i7-4790, growing with the context)
took longer than the card's run of it (44-59 ms). Instead, a program serves its rows for a span
of steps: row r is at position p_r + t, where t is the run's step in a DRAM word that the
program loads and multiplies into its argument and address registers (RLD, as the prompt runs
do). All rows add the same t, so they share those registers. The span ends where a row reaches
the end of its attention block (256 tokens); until then the same program runs again, already
in IMEM. The next one is compiled ahead as soon as a program starts serving, and so are the
steps whose rows the worker can tell ahead (a prompt the feed budget finishes, a request
reaching max_tokens). A feed run's
program serves its slot's runs through an attention block the same way, its rows advancing by
MCOLS a run.

**In turns** (`opentpu/serve/scheduler.py`, `Worker`: LFM2, Qwen3.5, Gemma 4, the streamed
MoEs, whose images hold one sequence). The requests (`--max-running` at once) take the card in
turns of `--quantum` tokens. Each switch saves the outgoing context's new pages and tail and
restores the incoming one. A request running alone runs to its end. The prompt is fed but its
last token, which the card's decode loop feeds and picks from (when the loop takes the request's
sampling; else the host picks the first token from the logits).

## Measured on the card

Run on 2026-10-09 on opentpu with build 1025907f (133.33 MHz, Gen2 x8), Qwen3-0.6B int8, cap
2048, using `tools/serve_bench.py`. Each client streams a greedy chat of 128 tokens
(`ignore_eos`) unless the row says sampled. Three trees: requests in turns (70ec5139), batched
with a program compiled per step (f99acce6), batched with span programs and feed runs
(6c3a1c75, with the feed-kind fix of 40fa0062).

| clients | in turns | batched, per-step programs | batched, span programs | TTFT median / max (span) |
|---|---|---|---|---|
| 1 | 25.0 tok/s | 23.9 tok/s | 24.6 tok/s | 0.24 / 0.24 s |
| 2 | 24.2 | 41.9 | 44.4 | 0.23 / 0.37 s |
| 4 | 24.0 | 53.3 | 77.7 | 0.56 / 0.56 s |
| 8 | 23.8 | 50.1 | 75.7 | 1.19 / 1.78 s |
| 8, sampled (temperature 0.7, top-k 20, top-p 0.8) | | 46.3 | 78.5 | 0.60 / 0.60 s |

- **Correctness.** Four prompts alone, then together. Every reply is equal to its own, and a
  repeat too (`check: PASS` in every run).
- **In turns.** About 49 ms per context switch.
- **Batched steps** (`tools/batch_bench.py`, 8 slots, 16 steps per case): the card's run of a
  step against its wall time. Per-step programs left the card idle while the host compiled
  the next one (a 4-row step: 44 ms on the card, 71.5 ms wall at context 64; 58.5 and 136 ms
  at 1024). With span programs:

  | rows | context 64: tok/s | card / wall per run | context 1024: tok/s | card / wall per run |
  |---|---|---|---|---|
  | 1 | 24.1 | 39.5 / 41.5 ms | 21.8 | 43.1 / 46.0 ms |
  | 2 | 45.1 | 41.6 / 44.3 ms | 37.4 | 48.8 / 53.4 ms |
  | 4 | 79.1 | 46.3 / 50.6 ms | 56.7 | 60.6 / 70.5 ms |
  | 8 | 79.2 | 91.3 / 101 ms (one run of 8) | 54.7 | 121 / 146 ms (two runs of 4) |
  | 4, sampled | 77.1 | 46.8 / 51.9 ms | 57.1 | 61.2 / 70.1 ms |

  The decode loop alone (`generate_card`) makes 25.3 tok/s at context 64. Eight rows in one
  run cost two runs of four on the card (at context 1024 their program no longer fits IMEM,
  and they run as two): the MXU takes MCOLS = 4 rows per pass of the
  weights, so four rows a run is the most a pass gives (~1.45 M cycles a token at a short
  context). A trace of a 4-row step's first 250 k cycles shows the MXU busy 95%. The wall
  time over the card's at context 1024 is mostly each case's first program, compiled inline
  and paid over only 16 steps (a program serves up to 256).
- **Prefill.** Feed runs fed 84 tok/s of 64-token prompts and 93 tok/s of 1024-token ones,
  on any slot (slot 0's prompt runs: 47.5 and 65 tok/s).
- **A conversation's turns** (each re-sends the history, max 128 tokens a reply), against the
  same prompt with nothing saved:

  | turn | prompt tokens | cached | TTFT | TTFT cold |
  |---|---|---|---|---|
  | 3 | 157 | 121 | 0.46 s | 1.56 s |
  | 4 | 196 | 153 | 0.50 s | 2.02 s |

  Qwen3's template renders a past reply without the empty thinking block its generation prompt
  had, so each turn feeds the last reply again (as otpu-chat).
- **A KV store of 50 MB of RAM** (`--kv-ram-gb 0.05`, 1 GB of disk): `check: PASS`, the same
  turns, and `/generate` replied.

## Tests

All on the ISA simulator with tiny models.

- `tests/test_serve_kvio.py`: save and restore bit for bit (Qwen3, LFM2, Qwen3.5 on one and two
  slices, direct and staged), partial restores, shared pages.
- `test_serve_store.py`: the tiers, pins, restart, the prefix cache.
- `test_serve_scheduler.py`: requests in turns equal their fresh runs, prefixes from the device
  and the store, stop strings, cancels, a store on disk across workers.
- `test_batch_steps.py`: greedy rows each sequence's decode, sampled rows the decode loop's pick,
  IMEM groups, span programs across attention blocks, feed runs then the last token's pick.
- `test_serve_batching.py`: requests in slots equal their fresh runs, a prefix copied between
  slots and restored from the store, a seeded sampled request the same alone and batched.
- `test_serve_api.py`: the HTTP API end to end with concurrent clients.
