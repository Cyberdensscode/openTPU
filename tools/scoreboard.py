"""The workload scoreboard: what a hardware configuration gives on the four workloads the user
weighs equally (2026-10-11): single-user decode, prefill, multi-user batched decode, MoE.

    python3 tools/scoreboard.py [report] [--board ypcb|bcu1525|f2] [what-if options]
    python3 tools/scoreboard.py compare --mcols 8            # a what-if against its board
    python3 tools/scoreboard.py whatifs [--board B]           # the standard what-if set
    python3 tools/scoreboard.py validate                      # the model against card / co-sim
    python3 tools/scoreboard.py boards | models | fit

What-if options (each overrides the board preset):
  --mcols N        MXU columns: MCOLS x D MACs a cycle, MCOLS rows a pass (int8; MCOLS / 2 PAIRed
                   4-bit)
  --act-rows N     rows a weight pass (MM replay, "virtual MCOLS"): a run of R rows streams the
                   weights ceil(R / N) times; the MXU still takes ceil(R / MCOLS) cycles a block
  --mxu-pump X     MACs a cycle x X at the same rows a pass (a double-pumped MXU)
  --tmem-words N   TMEM words: the prefill rows a run (tools/scoreboard_rmax.json, the
                   compiler's fit)
  --rows N         prefill rows a run, forced (tiling); --imem 0: no IMEM limit (looped
                   attention), else instructions
  --batch-rows N   rows of one batched-decode run (default: rows a pass)
  --vpu X          vector unit and DSTEP throughput x X
  -S N             slices (tensor-parallel: weights, KV heads and experts split)
  --mhz F          core clock
  --mem ddr3|ddr4|hbm --mts N --channels N --mem-eff E --gbps G (a slice's memory GB/s)
  --port-bytes N   core port bytes a cycle a slice (128: two 64-byte channel beats)
  --pcie-gbps G    host link for offloaded experts; --fast-gb G memory a slice holds
  --moe-rows all|gather   a MoE layer run's union experts on all R rows, or each on its own
  --wformat fp4|int8|auto  --ctx N  --models a,b  --json FILE  --markdown

Python (the other experiments):

    sys.path.insert(0, "tools"); import scoreboard as SB
    hw = SB.board("ypcb", mcols=8, tmem_words=262144)
    sc = SB.scoreboard(hw)            # {"decode": {...}, "prefill", "batch", "moe"}
    SB.compare(SB.board("ypcb"), hw)  # ratio a workload and their geometric mean ("score")

The model (docs/scoreboard.md): per layer and run of R rows, max(DRAM bytes / bandwidth, MXU
cycles) + the DeltaNet recurrence + a calibrated overhead a row; the bytes and MACs from the
real layer shapes (tools/scoreboard_models.json: each checkpoint's config.json and safetensors
headers), the MXU's row groups, the KV cache read in whole 256-token blocks, DSTEP at 8 state
words a cycle; the constants fitted to the YPCB card's counters, checked against the RTL
co-simulation (tools/scoreboard_cosim.py). Pure Python, no numpy.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS_FILE = HERE / "scoreboard_models.json"
RMAX_FILE = HERE / "scoreboard_rmax.json"
D = 128                         # the MXU's depth, one weight block; the port's chunk is D bytes
BLOCK = 256                     # the attention block: decode and prompt runs read whole blocks
GB = 1e9
MB = 1e6

# ------------------------------------------------------------------------------- the models
NAMES = {"qwen3": "Qwen3-0.6B", "lfm2": "LFM2.5-230M", "qwen35": "Qwen3.5-0.8B",
         "qwen35-2b": "Qwen3.5-2B", "qwen35-4b": "Qwen3.5-4B", "smollm3": "SmolLM3-3B",
         "phi4-mini": "Phi-4-mini-instruct", "lfm2-2.6b": "LFM2-2.6B",
         "gemma4": "gemma-4-E2B-it", "gemma4-e4b": "gemma-4-E4B-it",
         "lfm2-8b": "LFM2.5-8B-A1B", "qwen35-35b": "Qwen3.5-35B-A3B",
         "gemma4-26b": "gemma-4-26B-A4B-it"}
SHORT = {v: k for k, v in NAMES.items()}
DENSE = ["qwen3", "lfm2", "qwen35", "qwen35-2b", "qwen35-4b", "smollm3", "phi4-mini",
         "lfm2-2.6b", "gemma4"]
BATCHED = ["qwen3", "smollm3", "phi4-mini"]     # batched-decode kernels today: Qwen3, Llama-likes
MOE = ["gemma4-26b", "qwen35-35b"]
MIXES = {"phi4-mini": "mlp@4-29=fp4", "qwen35-4b": "delta=fp4,mlp=fp4",
         "smollm3": "gateup@9-35=fp4", "lfm2-2.6b": "conv=fp4,mlp=fp4"}   # formats.MIXES
MOE_FORMATS = {"gemma4-26b": ("int8", "fp4", "experts=fp4"),            # layers, head, formats
               "qwen35-35b": ("fp4", "int8", ""), "lfm2-8b": ("fp4", "int8", "")}
BITS = {"int8": 8.25, "fp4": 4.25, "int4": 4.25}   # storage bits a weight, block scales included

_MODELS: dict | None = None


def models() -> dict:
    global _MODELS
    if _MODELS is None:
        _MODELS = json.loads(MODELS_FILE.read_text())
    return _MODELS


def model(name: str) -> dict:
    key = NAMES.get(name, name)
    if key not in models():
        raise KeyError(f"unknown model {name!r}: {', '.join(NAMES)}")
    return models()[key]


def short(m: dict) -> str:
    return SHORT.get(m["name"], m["name"])


def layers(m: dict):
    """The model's layers in order: (index, layer)."""
    i = 0
    for n, lay in m["runs"]:
        for _ in range(n):
            yield i, lay
            i += 1


def _pad(n: float, m: int = D) -> int:
    return -(-int(n) // m) * m


SKIP = re.compile(r"(^|\.)(visual|vision_tower|vision_model|audio_tower|audio_model|"
                  r"embed_vision|embed_audio|mtp)(\.|_)")


def shapes(name: str, cfg: dict, tensors: dict) -> dict:
    """A model's shapes from its config.json and safetensors headers ({name: [dtype, shape]}):
    per layer its kind, the parameters of each weight group (attn, gateup, down, delta, conv,
    router, shared, ple; padded to D, Gemma 4's FFN widths to 2D as its image pads them), its
    attention / DeltaNet / MoE dimensions; the LM head and Gemma 4's PLE projection."""
    t = cfg.get("text_config", cfg)
    mt = t.get("model_type")
    fam = {"qwen3": "qwen3", "lfm2": "lfm2", "lfm2_moe": "lfm2", "qwen3_5_text": "qwen35",
           "qwen3_5_moe_text": "qwen35", "gemma4_text": "gemma4", "phi3": "llama",
           "smollm3": "llama", "llama": "llama"}[mt]
    H, L, V = t["hidden_size"], t["num_hidden_layers"], t["vocab_size"]
    kinds = t.get("layer_types") or ["full_attention"] * L
    ffn_pad = 2 * D if fam == "gemma4" else D
    n_shared = t.get("num_kv_shared_layers") or 0
    per, head, embed, ple = {}, 0, 0, {}
    for k, (_, sh) in tensors.items():
        if SKIP.search(k):
            continue
        mm = re.search(r"layers\.(\d+)\.(.*)$", k)
        if not mm:
            if k.endswith("lm_head.weight"):
                head = _pad(sh[0]) * _pad(sh[1])
            elif k.endswith("embed_tokens.weight"):
                embed = sh[0] * sh[1]
            elif k.endswith("embed_tokens_per_layer.weight"):
                ple["table"] = sh[0] * sh[1]
            elif k.endswith("per_layer_model_projection.weight"):
                ple["proj"] = _pad(sh[0]) * _pad(sh[1])
            continue
        per.setdefault(int(mm.group(1)), {})[mm.group(2)] = sh
    head = head or _pad(V) * _pad(H)               # tied: the head streams the embedding
    out = []
    for i in range(L):
        kind = {"full_attention": "attn", "sliding_attention": "sliding",
                "linear_attention": "delta", "conv": "conv"}[kinds[i]]
        shared = fam == "gemma4" and i >= L - n_shared
        g = {}
        for nm, sh in per[i].items():
            if len(sh) != 2 or "conv1d" in nm or re.search(r"experts\.\d+\.", nm):
                continue
            if shared and re.search(r"self_attn\.(k|v)_proj", nm):
                continue                          # a KV-shared layer reads an earlier one's
            n, kd = sh
            if "shared" not in nm and "router" not in nm and \
                    re.search(r"(gate_proj|up_proj|\.w1\.|\.w3\.|gate_up_proj)", nm):
                n = _pad(n, ffn_pad)
            if "shared" not in nm and re.search(r"(down_proj|\.w2\.)", nm):
                kd = _pad(kd, ffn_pad)
            grp = ("attn" if "self_attn" in nm else "delta" if "linear_attn" in nm else
                   "conv" if re.search(r"conv\.(in|out)_proj", nm) else
                   "shared" if "shared_expert" in nm else
                   "router" if re.search(r"(router\.proj|mlp\.gate\.weight|"
                                         r"feed_forward\.gate\.weight)", nm) else
                   "gateup" if re.search(r"(gate_proj|up_proj|\.w1\.|\.w3\.|gate_up)", nm)
                   else "down" if re.search(r"(down_proj|\.w2\.)", nm) else
                   "ple" if "per_layer" in nm else "other")
            g[grp] = g.get(grp, 0) + _pad(n) * _pad(kd)
        lay = {"kind": kind, "groups": dict(sorted(g.items()))}
        if kind in ("attn", "sliding"):
            if fam == "gemma4" and kinds[i] == "full_attention":
                lay["kind"] = "global"
                d = t.get("global_head_dim") or t["head_dim"]
                kvh = t.get("num_global_key_value_heads") or t["num_key_value_heads"]
            else:
                d = t.get("head_dim") or H // t["num_attention_heads"]
                kvh = t["num_key_value_heads"]
            lay.update(q_heads=t["num_attention_heads"], kv_heads=kvh, head_dim=d,
                       window=(t.get("sliding_window") or 0) if kind == "sliding" else 0,
                       kv_shared=shared)
        elif kind == "delta":
            lay["deltanet"] = dict(v_heads=t["linear_num_value_heads"],
                                   k_heads=t["linear_num_key_heads"],
                                   dk=t["linear_key_head_dim"], dv=t["linear_value_head_dim"],
                                   conv_k=t.get("linear_conv_kernel_dim", 4))
        if any("experts" in nm for nm in per[i]):
            E = t.get("num_experts") or t.get("num_local_experts")
            F = t.get("moe_intermediate_size") or t.get("expert_intermediate_size")
            lay["moe"] = dict(experts=E, top_k=t.get("num_experts_per_tok") or
                              t.get("top_k_experts"),
                              expert_params=3 * _pad(F, ffn_pad) * _pad(H))
        out.append(lay)
    runs = []
    for lay in out:
        if runs and runs[-1][1] == lay:
            runs[-1][0] += 1
        else:
            runs.append([1, lay])
    res = dict(name=name, family=fam, model_type=mt, hidden=H, vocab=V, n_layers=L,
               head=head, embed=embed, untied=bool(head != _pad(V) * _pad(H) or
                                                   not t.get("tie_word_embeddings", True)),
               runs=runs)
    if ple.get("proj"):
        res["ple"] = dict(proj=ple["proj"], table=ple.get("table", 0),
                          row=t["hidden_size_per_layer_input"] * L)
    return res


def safetensors_headers(d) -> dict:
    import struct
    out = {}
    for f in sorted(Path(d).glob("*.safetensors")):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            h = json.loads(fh.read(n))
        h.pop("__metadata__", None)
        out.update({k: [v["dtype"], v["shape"]] for k, v in h.items()})
    return out


# ------------------------------------------------------------------------------- formats
def formats_of(name: str, wformat: str) -> tuple:
    """(layer format, head format, formats string) of a weight choice: fp4 (int8 head), int8,
    mix / auto (the recommended mix, else int8: otpu-chat's default); a MoE model takes its
    production formats."""
    if name in MOE_FORMATS:
        return MOE_FORMATS[name]
    if wformat == "fp4":
        return "fp4", "int8", ""
    if wformat in ("mix", "auto"):
        return "int8", "int8", MIXES.get(name, "")
    return wformat, wformat, ""


def _rules(formats: str):
    out = []
    for item in (formats or "").replace(" ", "").split(","):
        if item:
            key, _, fmt = item.partition("=")
            kind, _, span = key.partition("@")
            lo, dash, hi = span.partition("-")
            out.append((kind, int(lo) if lo else None,
                        (int(hi) if dash else int(lo)) if lo else None, fmt))
    return out


def group_format(group: str, i: int, base: str, formats: str) -> str:
    """A weight group's format in layer i: a ranged rule over an unranged one, the first of
    equals ("mlp" is gateup and down; "experts" the routed experts; "head" unranged)."""
    best = None
    for kind, lo, hi, fmt in _rules(formats):
        if kind == group or (kind == "mlp" and group in ("gateup", "down")):
            if lo is None:
                best = best or (1, fmt)
            elif lo <= i <= hi and (best is None or best[0] == 1):
                best = (0, fmt)
    return best[1] if best else base


# ------------------------------------------------------------------------------- hardware
@dataclasses.dataclass(frozen=True)
class HW:
    """A hardware configuration. A slice's port moves port_bytes a core cycle (two 64-byte
    channel beats on the YPCB); its memory delivers channels x mts x 8 bytes a second (or
    gbps) at mem_eff while streaming; the slice streams at the lower of the two."""
    name: str = "ypcb"
    S: int = 1                      # slices (tensor-parallel: weights, KV heads, experts split)
    mcols: int = 4                  # MXU columns: MCOLS x D MACs a cycle, MCOLS rows a pass
    act_rows: int = 0               # rows a weight pass (MM replay; 0: mcols)
    mxu_pump: float = 1.0           # MACs a cycle x pump at the same rows a pass
    pair: bool = True               # 4-bit MMs of <= MCOLS / 2 rows take two blocks a cycle
    tmem_words: int = 65536
    imem: int = 4096                # instructions a program (0: no limit, looped attention)
    vpu: float = 1.0                # vector unit and DSTEP throughput x vpu
    mhz: float = 133.33
    mem: str = "ddr3"
    mts: float = 1066.67
    channels: int = 2               # memory channels a slice
    mem_eff: float = 0.97           # data share while streaming (the card: 8.04 K cycles a MB)
    port_bytes: int = 128           # core port bytes a cycle a slice
    port_eff: float = 0.97
    gbps: float = 0.0               # a slice's peak memory GB/s (0: channels x mts x 8)
    fast_gb: float = 4.0            # GiB a slice holds for the model (32-bit addresses: 4)
    far_gbps: float = 0.0           # a second memory tier on the card (F2's DDR4)
    far_gb: float = 0.0
    pcie_gbps: float = 2.2          # host -> card for offloaded experts (Gen2 x8, a miss's rate)
    rows: int = 0                   # prefill rows a run (0: the TMEM / IMEM fit)
    batch_rows: int = 0             # rows of a batched decode run (0: rows a pass)
    moe_rows: str = "all"           # a layer run's union experts on all R rows, or "gather"
    host_run_ms: float = 0.6        # the host's time a run (prompt runs, batched steps)
    note: str = ""

    def replace(self, **kw) -> "HW":
        return dataclasses.replace(self, **kw)

    @property
    def rows_pass(self) -> int:
        return self.act_rows or self.mcols

    def mem_gbps(self) -> float:
        """A slice's memory bandwidth while streaming, GB/s."""
        return (self.gbps or self.channels * self.mts * 8 / 1e3) * self.mem_eff

    def bw(self) -> float:
        """A slice's streaming bandwidth in bytes a core cycle: the port or the memory."""
        return min(self.port_bytes * self.port_eff, self.mem_gbps() * GB / (self.mhz * 1e6))

    def bound(self) -> str:
        return "port" if self.port_bytes * self.port_eff <= self.mem_gbps() * GB / (
            self.mhz * 1e6) else "memory"

    def describe(self) -> str:
        mem = (f"{self.mem.upper()} {self.gbps:g} GB/s" if self.gbps else
               f"{self.mem.upper()}-{self.mts:g} x {self.channels}")
        return (f"{self.name}: S {self.S}, MCOLS {self.mcols} (rows a pass {self.rows_pass}, "
                f"MXU x{self.mxu_pump:g}), TMEM {self.tmem_words // 1024}K words, VPU "
                f"x{self.vpu:g}, {self.mhz:g} MHz, {mem} a slice: {self.mem_gbps():.1f} "
                f"GB/s, {self.bw():.1f} B a "
                f"cycle ({self.bound()}-bound), {self.S * self.bw() * self.mhz / 1e3:.1f} "
                f"GB/s in all; host link {self.pcie_gbps:g} GB/s")


BOARDS = {
    "ypcb": HW(note="YPCB-00338 production: xc7k480t, 2 x DDR3-1066 (LiteDRAM), 133.33 MHz, "
                    "MCOLS 4, TMEM 64K words, PCIe Gen2 x8"),
    "bcu1525": HW(name="bcu1525", S=2, mhz=100.0, mem="ddr4", mts=1600, mem_eff=0.95,
                  pcie_gbps=9.0,
                  note="BCU1525: VU9P, S = 2 slices on 2 DDR4 channels each (4 GiB a slice "
                       "addressed of 2 x 16 GB), PCIe Gen3 x16; bring-up: DDR4-1600, 100 MHz"),
    "f2": HW(name="f2", S=4, mhz=250.0, mem="hbm", mts=0, channels=0, gbps=115.0,
             mem_eff=0.90, far_gbps=19.2, far_gb=64.0, pcie_gbps=20.0,
             note="AWS F2: VU47P, 16 GB HBM (~460 GB/s: 115 a slice), S = 4 at 250 MHz, "
                  "64 GB DDR4 as a second tier (one channel assumed)"),
}


def board(b: str = "ypcb", **kw) -> HW:
    """A board preset with what-if overrides: board("bcu1525", mhz=200, mts=2400)."""
    if b not in BOARDS:
        raise KeyError(f"unknown board {b!r}: {', '.join(BOARDS)}")
    return BOARDS[b].replace(**kw) if kw else BOARDS[b]


# ------------------------------------------------------------------------------- calibration
# Fitted on the YPCB card (docs/scoreboard.md; `validate` shows the agreement, `fit` refits
# kappa):
# - a layer's overhead outside the bandwidth and MXU bounds (the vector unit's work and the
#   dependency chains between a layer's phases), in units of U = ROW_CYCLES x hidden / 1024:
#   kappa_pass a weight pass + kappa_row a row of the pass (rows of a further pass overlap
#   the stream of the next: h of a row), (kappa_pass, kappa_row) by model, least squares over
#   its card decode, prefill and batched points (else the family's mean);
# - vpu_share: the share of it that scales with the vector unit (the RTL at 16 lanes);
# - a DeltaNet layer's recurrence: DSTEP at 8 state words a cycle plus a fill a head; one row's
#   hides behind the layer's weight stream, every further row's adds (the RTL's Qwen3.5-0.8B
#   prefill at R = 4: 1.95 M of 6.82 M cycles are DSTEP's);
# - attention streams a row's KV once a group of MCOLS query heads of a KV head (the code's
#   parts: gemma4._attention, qwen3._attend_heads);
# - run: cycles a device run; head_row: a logits row's sink (argmax / sampling);
# - coll: S > 1, cycles a layer and row for the all-gathers (an estimate);
# - moe_layer / moe_expert: a MoE layer's routing a row, an expert's start (the card's
#   zero-miss layer runs); link_lead_ms: the host's time to see a decode layer's misses and
#   start their DMA; a layer's link time past what the card hides (the dense MLP / shared
#   expert and the hit experts) is exposed; prefill_exposed: the share of a prompt's
#   offloaded experts' link time the card does not hide; moe_run_ms: the host's time a
#   layer run (the card's 35B and 26B); miss_scale: the card's misses against the router
#   traces' cache model.
ROW_CYCLES = 7000.0
CAL = {
    "kappa": {},                    # {model: (kappa_pass, kappa_row)}: KAPPA below
    "kappa_family": {},             # (the family's mean; set on first use)
    "h": 0.0,                       # a row's overhead in a further pass, hidden but h
    "vpu_share": 0.5,
    "run": 5000.0,
    "head_row": 2000.0,
    "dstep_fill": 200.0,
    "dstep_words": 8.0,
    "moe_alpha": 0.6,               # a run's union of R rows' experts: k R^alpha (consecutive
                                    # tokens share experts), at most the prompt's curve
    "coll": 1500.0,
    "moe_layer": 6000.0,
    "moe_expert": 600.0,
    "link_lead_ms": 0.15,           # the host's lead a request (it sees it, starts the DMA)
    "moe_run_ms": 0.2,
    "moe_ids": 16,                  # expert ids a MoE layer run's request line holds (R k <= 16)              # the host's time a layer run (start, HALTED, between)
    "prefill_exposed": 0.85,
    "miss_scale": 1.2,
}


KAPPA = {                           # `scoreboard.py fit` (2026-10-11)
    "gemma4": (3.6074, 1.0332), "lfm2": (0.0, 0.3303), "lfm2-2.6b": (0.2927, 0.4359),
    "phi4-mini": (0.3225, 0.6516), "qwen3": (0.0, 0.8956), "qwen35": (0.0, 1.4153),
    "qwen35-2b": (0.3459, 0.0), "qwen35-4b": (0.2989, 0.0), "smollm3": (0.303, 0.7391)}
CAL["kappa"].update(KAPPA)


def _family_kappa():
    fam = {}
    for n, k in CAL["kappa"].items():
        fam.setdefault(model(n)["family"], []).append(k)
    CAL["kappa_family"] = {f: tuple(sum(x[j] for x in v) / len(v) for j in (0, 1))
                           for f, v in fam.items()}


# ------------------------------------------------------------------------------- the time model
def att_tokens(lay: dict, ctx: int) -> int:
    """Positions a row at context ctx (its own included) reads from the KV cache: whole
    256-token blocks (the decode programs' buckets, the prompt runs' masked block); a sliding
    layer at most its window and a block."""
    n = _pad(ctx + 1, BLOCK)
    if lay.get("window"):
        n = min(n, _pad(lay["window"], BLOCK) + BLOCK)
    return n


def kv_token_bytes(d: int) -> float:
    """KV bytes a token and KV head: K int8 + an fp32 scale a D block, V int8 + its scale."""
    d = _pad(d)
    return 2 * d + 4 * (d // D) + 4


def mxu_cycles(params: float, fmt: str, R: int, hw: HW) -> float:
    """Cycles the MXU takes for `params` weights on R rows: int8 one block (D weights) a cycle
    a group of MCOLS rows; 4-bit (PAIR) two blocks a cycle a group of MCOLS / 2 rows."""
    if fmt == "int8" or not hw.pair:
        c = params / D * math.ceil(R / hw.mcols)
    else:
        c = params / (2 * D) * math.ceil(2 * R / hw.mcols)
    return c / hw.mxu_pump


def kappa(m: dict) -> tuple:
    """(kappa_pass, kappa_row) of a model."""
    if not CAL["kappa_family"] and CAL["kappa"]:
        _family_kappa()
    return CAL["kappa"].get(short(m), CAL["kappa_family"].get(m["family"], (0.3, 0.7)))


def over_cycles(m: dict, hw: HW, R: int) -> float:
    """A layer's overhead for a run of R rows at the configuration's vector unit."""
    kp, kr = kappa(m)
    p = hw.rows_pass
    reff = min(R, p) + CAL["h"] * max(0, R - p)
    s = CAL["vpu_share"]
    return (kp * math.ceil(R / p) + kr * reff) * ROW_CYCLES * m["hidden"] / 1024 * \
        (s / hw.vpu + 1 - s)


def dstep_cycles(lay: dict, hw: HW) -> float:
    dn = lay["deltanet"]
    return dn["dk"] * dn["dv"] / (CAL["dstep_words"] * hw.vpu) + CAL["dstep_fill"]


def layer_cycles(m: dict, i: int, lay: dict, hw: HW, fmt: tuple, ctxs: list,
                 experts: float | None = None, split: bool = False):
    """Cycles of layer i (on each slice) for a run of len(ctxs) rows, row r at context
    ctxs[r]. experts: the distinct routed experts the run streams (MoE; default the rows'
    expected union). split: (cycles, {bytes, mem, mxu, rec, over})."""
    base, _, formats = fmt
    R, S = len(ctxs), hw.S
    wbytes = mxu = 0.0
    for g, p in lay["groups"].items():
        f = group_format(g, i, base, formats)
        wbytes += p * BITS[f] / 8
        mxu += mxu_cycles(p, f, R, hw)
    byts = math.ceil(R / hw.rows_pass) * wbytes
    over = over_cycles(m, hw, R)
    rec = 0.0
    if lay["kind"] in ("attn", "sliding", "global"):
        kvb = kv_token_bytes(lay["head_dim"])
        G = max(1, lay["q_heads"] // lay["kv_heads"])
        passes = math.ceil(G / hw.mcols)
        for c in ctxs:
            byts += lay["kv_heads"] * kvb * (att_tokens(lay, c) * passes +
                                             (0 if lay.get("kv_shared") else 1))
    elif lay["kind"] == "delta":
        dn = lay["deltanet"]
        # the recurrence: the first row's DSTEPs hide behind the layer's weight stream (their
        # state traffic shares the port with it); every further row's run after, bound by
        # DSTEP's datapath, which covers their own state traffic
        byts += 2 * dn["v_heads"] * dn["dk"] * dn["dv"] * 4
        rec = (R - 1) * dn["v_heads"] / S * dstep_cycles(lay, hw)
    elif lay["kind"] == "conv":
        byts += R * 2 * 3 * m["hidden"] * 4                             # the conv window
    if "moe" in lay:
        mo = lay["moe"]
        f = group_format("experts", i, base, formats)
        if experts is None:
            experts = union(mo["experts"], mo["top_k"], R, MOE_EEFF.get(short(m)))
        byts += experts * mo["expert_params"] * BITS[f] / 8
        rows_e = R if hw.moe_rows == "all" else max(1, round(R * mo["top_k"] / experts))
        mxu += experts * mxu_cycles(mo["expert_params"], f, rows_e, hw)
        over += R * CAL["moe_layer"] + experts * CAL["moe_expert"]
    if S > 1:
        over += R * CAL["coll"]
    mem = byts / S / hw.bw()
    cyc = max(mem, mxu / S) + rec + over
    if split:
        return cyc, dict(bytes=byts, mem=mem, mxu=mxu / S, rec=rec, over=over)
    return cyc


def union(E: int, k: int, R: int, E_eff: float | None = None) -> float:
    """Expected distinct experts among R consecutive tokens' top-k picks: k R^alpha
    (consecutive tokens share experts: the card's zero-miss layer runs at R = 1 and 2), at
    most as from E_eff equally likely experts (E_eff < E: the router's skew over a prompt,
    MOE_EEFF)."""
    e = min(E, E_eff or E)
    return min(float(E), k * R ** CAL["moe_alpha"], e * (1 - (1 - k / e) ** R))


def head_cycles(m: dict, hw: HW, fmt: tuple, R: int) -> float:
    """The LM head for R logits rows: it streams once a pass of rows; each row's sink."""
    _, hf, formats = fmt
    hf = group_format("head", 0, hf, formats)
    p = m["head"]
    byts = p * BITS[hf] / 8 * math.ceil(R / hw.rows_pass)
    return max(byts / hw.S / hw.bw(), mxu_cycles(p, hf, R, hw) / hw.S) + R * CAL["head_row"]


def ple_cycles(m: dict, hw: HW, fmt: tuple, R: int) -> float:
    """Gemma 4's PLE model projection, a row's."""
    if "ple" not in m:
        return 0.0
    f = group_format("ple", 0, fmt[0], fmt[2])
    p = m["ple"]["proj"]
    return max(p * BITS[f] / 8 * math.ceil(R / hw.rows_pass) / hw.S / hw.bw(),
               mxu_cycles(p, f, R, hw) / hw.S)


def run_cycles(m: dict, hw: HW, fmt: tuple, ctxs: list, logits: int = 1,
               experts: float | None = None, split: bool = False):
    """Cycles of one device run: len(ctxs) rows through every layer, `logits` of them with
    logits (the LM head streams once for them)."""
    tot = CAL["run"] + ple_cycles(m, hw, fmt, len(ctxs))
    parts = dict(bytes=0.0, mem=0.0, mxu=0.0, rec=0.0, over=0.0)
    for i, lay in layers(m):
        c, p = layer_cycles(m, i, lay, hw, fmt, ctxs, experts, split=True)
        tot += c
        for k in parts:
            parts[k] += p[k]
    if logits:
        tot += head_cycles(m, hw, fmt, logits)
    return (tot, parts) if split else tot


def bytes_per_token(m: dict, fmt: tuple, ctx: int) -> float:
    """A decode token's DRAM bytes at context ctx (all slices)."""
    hw = BOARDS["ypcb"]
    tot = sum(layer_cycles(m, i, lay, hw, fmt, [ctx], split=True)[1]["bytes"]
              for i, lay in layers(m))
    tot += m["head"] * BITS[group_format("head", 0, fmt[1], fmt[2])] / 8
    if "ple" in m:
        tot += m["ple"]["proj"] * BITS[group_format("ple", 0, fmt[0], fmt[2])] / 8
    return tot


# ------------------------------------------------------------------------------- workloads
def decode(name: str, hw: HW, wformat: str = "fp4", ctx: int = 512) -> dict:
    """Single-user decode at context ctx, device tokens a second (the card's decode loop picks
    every token: the host adds nothing a token)."""
    m = model(name)
    fmt = formats_of(name, wformat)
    cyc = run_cycles(m, hw, fmt, [ctx])
    return dict(tok_s=hw.mhz * 1e6 / cyc, mcycles=cyc / 1e6,
                mb_token=bytes_per_token(m, fmt, ctx) / MB)


_RMAX: dict | None = None


def rmax_table() -> dict:
    """The compiler's prefill rows (tools/scoreboard_cosim.py probe): {"model|tmem|mcols|
    act_rows": {bucket: [TMEM / ACT RAM rows, also IMEM rows]}}."""
    global _RMAX
    if _RMAX is None:
        _RMAX = json.loads(RMAX_FILE.read_text()) if RMAX_FILE.exists() else {}
    return _RMAX


RMAX_CARD = {"qwen3": 9, "lfm2": 8, "qwen35": 4, "qwen35-2b": 4, "qwen35-4b": 3,
             "smollm3": 4, "phi4-mini": 3, "lfm2-2.6b": 4, "gemma4": 4}


def _fit_rows(name: str, hw: HW, bucket: int) -> tuple:
    """(TMEM rows, IMEM rows) of a prefill run in a bucket: the probe's, interpolated between
    its buckets (1 and 4), else the card's R_max scaled by the TMEM words."""
    t = rmax_table().get(f"{NAMES.get(name, name)}|{hw.tmem_words}|{hw.mcols}|{hw.act_rows}")
    if not t:
        cands = [(k, v) for k, v in rmax_table().items()
                 if k.startswith(NAMES.get(name, name) + "|") and
                 int(k.split("|")[1]) == hw.tmem_words]
        t = cands[0][1] if cands else None
    if not t:
        r = max(1, int(RMAX_CARD.get(name, 4) * hw.tmem_words / 65536))
        return r, r
    bs = sorted(int(b) for b in t)
    lo = max([b for b in bs if b <= bucket] or bs[:1])
    hi = min([b for b in bs if b >= bucket] or bs[-1:])
    a, b = t[str(lo)], t[str(hi)]
    w = 0 if hi == lo else (bucket - lo) / (hi - lo)
    return (int(a[0] + (b[0] - a[0]) * w), int(a[1] + (b[1] - a[1]) * w))


def _rule_rows(r: int, hw: HW, fmt: str) -> int:
    p = hw.rows_pass
    if r > p:
        return (r // p) * p
    if fmt == "fp4" and hw.pair and r > 1 and r % 2 and r > hw.mcols // 2:
        return r - 1
    return r


def prefill_rows(name: str, hw: HW, bucket: int = 1, fmt: str = "fp4") -> int:
    """Rows a prefill run in a bucket: --rows, else the most the TMEM fit allows (and IMEM's
    unless imem 0) as the host's rule takes them: whole passes above one pass, and for 4-bit
    an odd R's R - 1 (PAIR: prefer_rows); of those, and of the fewer whole passes and MXU
    groups, the R the model finds fastest a row (the host's rule weighs its cost a row)."""
    if hw.rows:
        return hw.rows
    tm, im = _fit_rows(name, hw, bucket)
    top = _rule_rows(max(1, min(tm, im) if hw.imem else tm), hw, fmt)
    cands = {top} | {r for r in range(hw.mcols, top, hw.mcols)}
    if len(cands) == 1:
        return top
    m = model(name)
    f = formats_of(name, "fp4" if fmt == "fp4" else "auto" if fmt == "mix" else fmt)
    c0 = (bucket - 1) * BLOCK + BLOCK // 2
    cost = {r: run_cycles(m, hw, f, [c0 + i for i in range(r)], logits=0) / r for r in cands}
    best = min(cost.values())
    return max(r for r in cands if cost[r] <= best * 1.001)


def prefill(name: str, hw: HW, wformat: str = "fp4", P: int = 1024, p0: int = 0) -> dict:
    """A P-token prompt from position p0 in prompt runs (a run stays in its 256-token bucket
    when R < 256; the last run's last row takes the logits). TTFT = the runs' device time
    plus the host's time a run (warm program cache)."""
    m = model(name)
    fmt = formats_of(name, wformat)
    cyc, runs, p, rows = 0.0, 0, p0, []
    while p < p0 + P:
        R = prefill_rows(name, hw, p // BLOCK + 1, fmt[0] if not fmt[2] else "mix")
        n = min(R, p0 + P - p, BLOCK - p % BLOCK if R < BLOCK else R)
        last = p + n >= p0 + P
        cyc += run_cycles(m, hw, fmt, [p + r for r in range(n)], logits=1 if last else 0)
        runs += 1
        rows.append(R)
        p += n
    t = cyc / (hw.mhz * 1e6) + runs * hw.host_run_ms / 1e3
    return dict(rows=rows[0], rows_last=rows[-1], runs=runs, ttft_s=t, tok_s=P / t,
                device_s=cyc / (hw.mhz * 1e6))


def batch(name: str, hw: HW, users: int, wformat: str = "fp4", ctx: int = 512) -> dict:
    """Batched decode of `users` sequences at context ctx: a step runs them in blocks of
    batch_rows (rows a pass), one run a block, every weight streamed once a run, each row's
    logits picked on the card. Total and per-user tokens a second (device and host)."""
    m = model(name)
    fmt = formats_of(name, wformat)
    rb = hw.batch_rows or hw.rows_pass
    cyc, runs, left = 0.0, 0, users
    while left > 0:
        n = min(rb, left)
        cyc += run_cycles(m, hw, fmt, [ctx] * n, logits=n)
        runs += 1
        left -= n
    t = cyc / (hw.mhz * 1e6) + (runs * hw.host_run_ms / 1e3 if users > 1 else 0.0)
    return dict(users=users, total_tok_s=users / t, user_tok_s=1 / t, runs=runs,
                device_ms=cyc / (hw.mhz * 1e3))


# ------------------------------------------------------------------------------- MoE
# Misses a decode token (of L x k) against the share of the pool in slots: the per-layer LRU
# over four 2048-token router traces (docs/offload.md section 3); decayed use (the default)
# misses ~12% less (LFU); the card's chats miss more than the traces (miss_scale)
CACHE = {"gemma4-26b": [(0.05, 129.8), (0.10, 102.4), (0.19, 57.3), (0.20, 55.3),
                        (0.30, 31.4), (0.50, 8.9), (0.75, 1.2), (1.0, 0.0)],
         "qwen35-35b": [(0.05, 169.4), (0.10, 125.4), (0.13, 110.2), (0.20, 80.8),
                        (0.30, 52.8), (0.50, 22.6), (0.75, 5.9), (1.0, 0.0)]}
LFU = 0.88
MOE_EEFF = {"gemma4-26b": 75.0, "qwen35-35b": 145.0}   # a layer's distinct experts over a prompt
RESERVE_GB = 0.3                                        # KV, I/O, programs beside the slots


def _interp(tab, x):
    if x <= tab[0][0]:
        return tab[0][1]
    for (x0, y0), (x1, y1) in zip(tab, tab[1:]):
        if x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return 0.0


def moe_fit(name: str, hw: HW) -> dict:
    """Where a MoE model's experts live: slots in the fast memory beside the rest of the image
    (the S slices' memory, each expert split across them), then the far tier's, the rest on
    the host. The untied embedding table stays on the host (the 35B's)."""
    m = model(name)
    base, hf, formats = formats_of(name, "auto")
    dense = sum(p * BITS[group_format(g, i, base, formats)] / 8
                for i, lay in layers(m) for g, p in lay["groups"].items())
    dense += m["head"] * BITS[hf] / 8
    lay = next(x for _, x in layers(m) if "moe" in x)
    mo = lay["moe"]
    nmoe = sum(1 for _, x in layers(m) if "moe" in x)
    eb = mo["expert_params"] * BITS[group_format("experts", 0, base, formats)] / 8
    pool = nmoe * mo["experts"] * eb
    fast = hw.S * (hw.fast_gb * 2**30 - RESERVE_GB * GB) - dense
    slots = max(0.0, min(pool, fast)) / eb
    far = min(pool - slots * eb, hw.far_gb * GB) / eb if hw.far_gbps else 0.0
    return dict(dense_gb=dense / GB, pool_gb=pool / GB, expert_mb=eb / MB, slots=int(slots),
                frac=slots * eb / pool, far_slots=int(far), experts=nmoe * mo["experts"],
                layers=nmoe, top_k=mo["top_k"], E=mo["experts"],
                resident=slots * eb >= pool * 0.999)


def _misses(name: str, frac: float) -> float:
    return 0.0 if frac >= 0.999 else _interp(CACHE[name], frac) * LFU * CAL["miss_scale"]


def moe_decode(name: str, hw: HW, ctx: int = 512) -> dict:
    """MoE decode: the card's time with the routed experts present, plus the exposed link
    time of the misses (from the host over PCIe, or from the far tier)."""
    m = model(name)
    base, _, formats = fmt = formats_of(name, "auto")
    fit = moe_fit(name, hw)
    f = hw.mhz * 1e6
    card = run_cycles(m, hw, fmt, [ctx]) / f
    misses = _misses(name, fit["frac"])                 # a token's, beyond the fast memory
    host = _misses(name, min(1.0, (fit["slots"] + fit["far_slots"]) / fit["experts"])) \
        if fit["far_slots"] else misses                 # ... and beyond the far tier too
    far = misses - host
    eb = fit["expert_mb"] * MB
    fast = eb / hw.S / (hw.bw() * f)                    # an expert from the fast memory, s
    exposed = link = 0.0
    k = fit["top_k"]
    for i, lay in layers(m):
        if "moe" not in lay:
            continue
        mh, mf = host / fit["layers"], far / fit["layers"]
        # what the card computes while the misses stream: the part beside the request (the
        # dense MLP, the shared expert) and the experts it has
        beside = sum(p * BITS[group_format(g, i, base, formats)] / 8
                     for g, p in lay["groups"].items() if g in ("gateup", "down", "shared"))
        hide = beside / hw.S / (hw.bw() * f) + max(0.0, k - mh - mf) * fast
        lk = CAL["link_lead_ms"] / 1e3 * (mh > 0) + mh * eb / (hw.pcie_gbps * GB)
        link += lk
        exposed += max(0.0, lk - hide)
        if mf:                                          # read from the far tier instead
            exposed += mf * (eb / (hw.far_gbps * GB) - fast)
    t = card + exposed
    return dict(tok_s=1 / t, card_tok_s=1 / card, misses=misses, far_misses=far,
                link_s=link, exposed_s=exposed, resident=fit["resident"],
                slots=fit["slots"], frac=fit["frac"])


def moe_prefill(name: str, hw: HW, P: int = 1024, R: int | None = None) -> dict:
    """MoE prefill, layer-major (docs/offload.md 13): runs of R rows a layer (default 2, the
    card's; --rows, at most the request line's ids / top-k: CAL moe_ids), every slot serving
    the running layer, so a layer's misses are its prompt's distinct experts; each run
    streams its rows' union of experts. A DeltaNet model's first conv_k - 1 rows run as
    decode steps (docs/offload.md 13.6)."""
    m = model(name)
    fmt = formats_of(name, "auto")
    fit = moe_fit(name, hw)
    R = R or min(hw.rows or 2, max(1, int(CAL["moe_ids"]) // fit["top_k"]))
    eeff = MOE_EEFF[name]
    steps = 0
    dn = next((x["deltanet"] for _, x in layers(m) if "deltanet" in x), None)
    if dn and P > 1:
        steps = min(P - 1, dn.get("conv_k", 4) - 1)
    cyc, runs = 0.0, 0
    for i, lay in layers(m):
        p = steps
        while p < P:
            n = min(R, P - p)
            ex = union(lay["moe"]["experts"], lay["moe"]["top_k"], n, eeff) \
                if "moe" in lay else None
            cyc += layer_cycles(m, i, lay, hw, fmt, [p + r for r in range(n)], ex) + \
                CAL["run"]
            runs += 1
            p += n
    cyc += head_cycles(m, hw, fmt, 1)
    device = cyc / (hw.mhz * 1e6)
    link = 0.0
    if not fit["resident"]:
        # a layer's distinct experts over the prompt; the slots hold the decode's share of
        # them, the rest comes from the far tier or the host, once each (pooled slots)
        U = union(fit["E"], fit["top_k"], P, eeff) * fit["layers"]
        far = fit["far_slots"] / fit["experts"]
        eb = fit["expert_mb"] * MB
        link = U * (1 - fit["frac"] - far) * eb / (hw.pcie_gbps * GB)
        if far:
            link += U * far * eb / (hw.far_gbps * GB)
    t = device + CAL["prefill_exposed"] * link + runs * CAL["moe_run_ms"] / 1e3
    if steps:
        t += steps / moe_decode(name, hw, 0)["tok_s"]
    return dict(ttft_s=t, tok_s=P / t, device_s=device, link_s=link, rows=R, runs=runs)


# ------------------------------------------------------------------------------- the scoreboard
def scoreboard(hw: HW, wformat: str = "fp4", ctx: int = 512, names=None,
               users=(1, 4, 8, 16), prompts=(256, 1024)) -> dict:
    """Every workload on a configuration: decode and prefill of the dense models, batched
    decode of the batched ones, decode and prefill of the MoEs."""
    names = names or DENSE
    out = {"hw": dataclasses.asdict(hw), "wformat": wformat, "ctx": ctx,
           "decode": {}, "prefill": {}, "batch": {}, "moe": {}}
    for n in names:
        if n not in MOE:
            out["decode"][n] = decode(n, hw, wformat, ctx)
            out["prefill"][n] = {P: prefill(n, hw, wformat, P) for P in prompts}
    for n in [x for x in names if x in BATCHED] or BATCHED:
        out["batch"][n] = {u: batch(n, hw, u, wformat, ctx) for u in users}
    for n in MOE:
        out["moe"][n] = dict(decode=moe_decode(n, hw, ctx), fit=moe_fit(n, hw),
                             prefill={P: moe_prefill(n, hw, P) for P in prompts})
    return out


def _gmean(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def summary(sc: dict) -> dict:
    """One number a workload, geometric means over the models: decode tok/s; prefill tok/s at
    the longest prompt; batched total tok/s at the most users; MoE sqrt(decode x prefill)."""
    P = max(next(iter(sc["prefill"].values())))
    U = max(next(iter(sc["batch"].values())))
    return dict(decode=_gmean([v["tok_s"] for v in sc["decode"].values()]),
                prefill=_gmean([v[P]["tok_s"] for v in sc["prefill"].values()]),
                batch=_gmean([v[U]["total_tok_s"] for v in sc["batch"].values()]),
                moe=_gmean([math.sqrt(v["decode"]["tok_s"] * v["prefill"][P]["tok_s"])
                            for v in sc["moe"].values()]))


def compare(base: HW, new: HW, **kw) -> dict:
    """Each workload's ratio new / base and their geometric mean, "score" (the four weighed
    equally)."""
    a, b = summary(scoreboard(base, **kw)), summary(scoreboard(new, **kw))
    r = {k: b[k] / a[k] for k in a}
    r["score"] = _gmean(list(r.values()))
    return r


WHATIFS = [
    ("MCOLS 8", dict(mcols=8)),
    ("TMEM x2 (8 ACT RAM rows)", dict(tmem_words=131072, act_rows=8)),
    ("TMEM x4 (16 ACT RAM rows)", dict(tmem_words=262144, act_rows=16)),
    ("tiled prefill, 64 rows a run", dict(rows=64, act_rows=64, imem=0)),
    ("8 users in one pass (replay)", dict(batch_rows=8, act_rows=8)),
    ("VPU x2", dict(vpu=2.0)),
    ("MXU pumped x2", dict(mxu_pump=2.0)),
    ("MCOLS 8 + TMEM x4 + VPU x2", dict(mcols=8, tmem_words=262144, act_rows=16, vpu=2.0)),
    ("MXU x2 + TMEM x4 + VPU x2 (pumped)", dict(mxu_pump=2.0, tmem_words=262144, act_rows=16,
                                                 vpu=2.0)),
    ("port 256 B a cycle (memory permitting)", dict(port_bytes=256)),
    ("port 256 B + MXU x2", dict(port_bytes=256, mxu_pump=2.0)),
]


def whatifs(b: str = "ypcb", wformat: str = "fp4", items=None) -> list:
    base = board(b)
    out = []
    for label, kw in items or WHATIFS:
        out.append((label, kw, compare(base, base.replace(**kw), wformat=wformat)))
    return out


# ------------------------------------------------------------------------------- card / co-sim
# What the model is checked against. Card: production builds at 133.33 MHz (pa e4db91c9,
# fmvf 542fc43a, fixmath 1025907f), tools/qual/perf.py's decode counters (positions 512-575,
# int8 head) and prompt runs (docs/prefill.md 7), otpu-serve's batched steps (docs/serve.md),
# the streamed MoEs (docs/offload.md 13.7, 14.4). Co-sim: the RTL with LiteDRAM's controller
# at DDR3-1066 (docs/litedram.md, docs/prefill.md 8; build/sbcosim.jsonl: this tool's runs).
CARD_DECODE = [   # (model, (layers, head, formats), Mcycles a token, MB a token measured, src)
    ("qwen3", ("fp4", "int8", ""), 3.730, 444.5, "fmvf qual"),
    ("qwen3", ("int8", "int8", ""), 5.506, 664.5, "fmvf qual"),
    ("lfm2", ("fp4", "int8", ""), 1.369, 164.5, "fmvf qual"),
    ("lfm2", ("int8", "int8", ""), 2.025, 245.5, "fmvf qual"),
    ("qwen35", ("fp4", "int8", ""), 4.817, 582.4, "fmvf qual"),
    ("qwen35", ("int8", "int8", ""), 6.818, 831.4, "fmvf qual"),
    ("phi4-mini", ("int8", "int8", ""), 32.936, 4011, "pa mix1"),
    ("phi4-mini", ("int8", "int8", "mlp@4-27=fp4"), 25.640, 3105, "pa mix1"),
    ("phi4-mini", ("fp4", "int8", ""), 19.971, 2401, "pa mix1"),
    ("smollm3", ("int8", "int8", ""), 26.309, 3203, "pa mix1"),
    ("smollm3", ("int8", "int8", "gateup@9-35=fp4"), 21.409, 2594, "pa mix1"),
    ("smollm3", ("fp4", "int8", ""), 14.980, 1796, "pa mix1"),
    ("qwen35-4b", ("int8", "int8", "delta=fp4,mlp=fp4"), 23.384, 2770, "pa mix1"),
    ("qwen35-4b", ("fp4", "int8", ""), 22.206, 2623, "pa mix1"),
    ("qwen35-2b", ("fp4", "int8", ""), 10.800, 1282, "pa mix1"),
    ("lfm2-2.6b", ("int8", "int8", ""), 21.764, 2661, "pa mix2"),
    ("lfm2-2.6b", ("int8", "int8", "conv=fp4,mlp=fp4"), 12.298, 1486, "pa mix2"),
    ("lfm2-2.6b", ("fp4", "int8", ""), 11.958, 1444, "pa mix2"),
    ("gemma4", ("int8", "int8", ""), 20.875, 2496, "fmvf mix4"),
    ("gemma4", ("int8", "int8", "attn@15-34=fp4,mlp@15-34=fp4"), 15.751, 1855, "fmvf mix4"),
    ("gemma4", ("fp4", "int8", ""), 13.375, 1558, "fmvf mix4"),
]
CARD_PREFILL = [  # (model, formats, prompt P, rows a run, device Mcycles, source)
    ("qwen3", ("int8", "int8", ""), 236, 4, 288.0, "fmvf prompt runs, 59 runs"),
    ("qwen3", ("int8", "int8", ""), 236, 8, 268.7, "fmvf, 30 runs of 8 (2 passes)"),
    ("lfm2", ("int8", "int8", ""), 238, 4, 96.5, "fmvf prompt runs, 60 runs"),
    ("lfm2", ("int8", "int8", ""), 238, 8, 92.7, "fmvf, 31 runs"),
    ("lfm2-2.6b", ("int8", "int8", "conv=fp4,mlp=fp4"), 238, 4, 1185.5, "fmvf prompt runs"),
    ("smollm3", ("int8", "int8", "gateup@9-35=fp4"), 291, 4, 1757.7,
     "fmvf prompt runs, 73 runs"),
    ("phi4-mini", ("int8", "int8", "mlp@4-27=fp4"), 227, 3, 2066.2,
     "fmvf prompt runs, 76 runs"),
    ("gemma4", ("int8", "int8", ""), 224, 4, 1056.2, "fmvf prompt runs, 56 runs"),
    ("qwen35-4b", ("fp4", "int8", ""), 237, 2, 2113.1, "fmvf, 119 runs"),
    ("qwen35-4b", ("fp4", "int8", ""), 237, 3, 2441.7, "fmvf, 79 runs"),
    ("qwen35", ("fp4", "int8", ""), 237, 4, 412.8, "pa, 3.096 s, 60 runs"),
    ("qwen35-2b", ("fp4", "int8", ""), 237, 4, 752.6, "pa, 5.645 s, 60 runs"),
]
CARD_BATCH = [    # (model, formats, rows, context, device ms a run, source)
    ("qwen3", ("int8", "int8", ""), 1, 64, 39.5, "1025907f serve"),
    ("qwen3", ("int8", "int8", ""), 2, 64, 41.6, "1025907f serve"),
    ("qwen3", ("int8", "int8", ""), 4, 64, 46.3, "1025907f serve"),
    ("qwen3", ("int8", "int8", ""), 1, 1024, 43.1, "1025907f serve"),
    ("qwen3", ("int8", "int8", ""), 2, 1024, 48.8, "1025907f serve"),
    ("qwen3", ("int8", "int8", ""), 4, 1024, 60.6, "1025907f serve"),
]
COSIM = [         # (model, formats, rows, position, cycles, what, hw overrides)
    ("qwen35", ("fp4", "int8", ""), 4, 544, 6820714, "prefill run, R 4 (prefill.md 8)", {}),
    ("qwen35", ("fp4", "int8", ""), 4, 544, 4 * 1413000, "R 4, MCOLS 8 (prefill.md 8)",
     {"mcols": 8}),
    ("qwen3", ("fp4", "int8", ""), 0, 544, 3696899, "decode (litedram.md)", {}),
    ("lfm2", ("fp4", "int8", ""), 0, 544, 1355035, "decode (litedram.md)", {}),
    ("qwen35", ("fp4", "int8", ""), 0, 544, 4765281, "decode (litedram.md)", {}),
    ("qwen3", ("int8", "int8", ""), 0, 544, 5471471, "decode (litedram.md)", {}),
    ("lfm2", ("int8", "int8", ""), 0, 544, 2012426, "decode (litedram.md)", {}),
    ("qwen35", ("int8", "int8", ""), 0, 544, 6764651, "decode (litedram.md)", {}),
]
CARD_MOE = [      # (model, quantity, measured, source)
    ("qwen35-35b", "decode tok/s", 5.30, "s17, 128 tokens, 103 misses a token"),
    ("gemma4-26b", "decode tok/s", 3.70, "14.4, 128 tokens, 88 misses a token"),
    ("qwen35-35b", "prefill s (134 tokens)", 13.32, "pfcomp, R 2 pooled"),
    ("gemma4-26b", "prefill s (124 tokens)", 16.25, "pfcomp2, R 2 pooled"),
]


MOE_LAYER_RUNS = [  # (model, layer kind, rows, ms a zero-miss layer run, source: offload.md 13.7)
    ("qwen35-35b", "delta", 1, 2.48, "pftime"), ("qwen35-35b", "delta", 2, 3.38, "pfpool"),
    ("qwen35-35b", "attn", 1, 1.98, "pftime"), ("qwen35-35b", "attn", 2, 2.61, "pfpool"),
    ("gemma4-26b", "sliding", 2, 7.29, "pfpool"), ("gemma4-26b", "global", 2, 7.81, "pfpool"),
]


def _prompt_runs(P: int, R: int, p0: int = 0) -> list:
    """A prompt's runs: (rows' contexts, logits rows), R rows a run inside a bucket."""
    out, p = [], p0
    while p < p0 + P:
        n = min(R, p0 + P - p, BLOCK - p % BLOCK if R < BLOCK else R)
        out.append(([p + r for r in range(n)], int(p + n >= p0 + P)))
        p += n
    return out


def points() -> list:
    """Every card and co-sim point the dense model is fitted or checked on: dicts of group,
    model, fmt, runs [(contexts, logits rows)], hw overrides, measured cycles, label."""
    out = []
    for name, fmt, mc, _, src in CARD_DECODE:
        out.append(dict(group="decode", model=name, fmt=fmt, runs=[([544], 1)], hw={},
                        y=mc * 1e6, label=f"{name} {'/'.join(x for x in fmt if x)} ({src})"))
    for name, fmt, P, R, mc, src in CARD_PREFILL:
        out.append(dict(group="prefill", model=name, fmt=fmt, runs=_prompt_runs(P, R), hw={},
                        y=mc * 1e6, label=f"{name} {P} tokens, R {R} ({src})"))
    for name, fmt, R, ctx, ms, src in CARD_BATCH:
        out.append(dict(group="batched", model=name, fmt=fmt, runs=[([ctx] * R, R)], hw={},
                        y=ms * 133.33e3, label=f"{name} {R} rows at {ctx} ({src})"))
    for name, fmt, R, pos, cyc, what, kw in COSIM:
        runs = [([pos], 1)] if R == 0 else [([pos + r for r in range(R)], 0)]
        out.append(dict(group="co-sim", model=name, fmt=fmt, runs=runs, hw=kw, y=cyc,
                        label=f"{name} {fmt[0]} {what}"))
    return out


def point_cycles(pt: dict) -> float:
    hw = BOARDS["ypcb"].replace(**pt["hw"])
    m = model(pt["model"])
    return sum(run_cycles(m, hw, pt["fmt"], ctxs, logits=lg) for ctxs, lg in pt["runs"])


def _with_kappa(name, k, f):
    save = CAL["kappa"].get(name)
    CAL["kappa"][name] = k
    try:
        return f()
    finally:
        if save is None:
            CAL["kappa"].pop(name)
        else:
            CAL["kappa"][name] = save


def _lsq(rows):
    """min sum ((b + x a1 + y a2 - t) / t)^2 over x, y >= 0; rows (b, a1, a2, t)."""
    def solve(use):
        S = [[0.0] * 2 for _ in range(2)]
        v = [0.0, 0.0]
        for b, a1, a2, t in rows:
            a = (a1 / t, a2 / t)
            r = (t - b) / t
            for i in range(2):
                v[i] += a[i] * r * use[i]
                for j in range(2):
                    S[i][j] += a[i] * a[j] * use[i] * use[j]
        if use == (1, 1):
            det = S[0][0] * S[1][1] - S[0][1] * S[1][0]
            if abs(det) < 1e-18:
                return None
            return ((v[0] * S[1][1] - v[1] * S[0][1]) / det,
                    (S[0][0] * v[1] - S[1][0] * v[0]) / det)
        i = use.index(1)
        x = v[i] / S[i][i] if S[i][i] else 0.0
        return (x, 0.0) if i == 0 else (0.0, x)

    def err(k):
        return sum(((b + k[0] * a1 + k[1] * a2 - t) / t) ** 2 for b, a1, a2, t in rows)
    cands = [k for k in (solve((1, 1)), solve((1, 0)), solve((0, 1))) if k is not None and
             min(k) >= 0]
    return min(cands or [(0.0, 0.0)], key=err)


def fit_model(name: str, pts: list) -> tuple:
    """(kappa_pass, kappa_row) of a model: least squares (relative) over its points."""
    rows = []
    for pt in pts:
        b = _with_kappa(name, (0.0, 0.0), lambda: point_cycles(pt))
        a1 = _with_kappa(name, (1.0, 0.0), lambda: point_cycles(pt)) - b
        a2 = _with_kappa(name, (0.0, 1.0), lambda: point_cycles(pt)) - b
        rows.append((b, a1, a2, pt["y"]))
    k = _lsq(rows)
    return round(k[0], 4), round(k[1], 4)


def fit(pts=None) -> dict:
    """kappa of every model with card points (decode, prefill, batched; the co-sim's
    prefill run where the card has none)."""
    pts = pts if pts is not None else [p for p in points() if not p["hw"]]
    out = {}
    for name in sorted({p["model"] for p in pts}):
        out[name] = fit_model(name, [p for p in pts if p["model"] == name])
    return out


def validate(loo: bool = True) -> list:
    """(group, point, measured, model, error %, leave-one-out error %) for every point; the
    leave-one-out model refits the point's model without it (decode points: its other
    formats)."""
    out = []
    pts = points()
    fitted = [p for p in pts if not p["hw"]]
    for pt in pts:
        got = point_cycles(pt)
        e = 100 * (got / pt["y"] - 1)
        el = None
        if loo:
            rest = [p for p in fitted if p is not pt and p["model"] == pt["model"]]
            if rest:
                k = fit_model(pt["model"], rest)
                el = 100 * (_with_kappa(pt["model"], k, lambda: point_cycles(pt)) / pt["y"] - 1)
        out.append((pt["group"], pt["label"], pt["y"] / 1e6, got / 1e6, e, el))
    hw = BOARDS["ypcb"]
    for name, kind, R, ms, src in MOE_LAYER_RUNS:
        m = model(name)
        fmt = formats_of(name, "auto")
        i, lay = next((i, x) for i, x in layers(m) if x["kind"] == kind and "moe" in x)
        ex = union(lay["moe"]["experts"], lay["moe"]["top_k"], R, MOE_EEFF[name])
        got = (layer_cycles(m, i, lay, hw, fmt, [256 + r for r in range(R)], ex) +
               CAL["run"]) / (hw.mhz * 1e3)
        out.append(("MoE layer run ms", f"{name} {kind} R {R} zero-miss ({src})", ms, got,
                    100 * (got / ms - 1), None))
    for name, q, val, src in CARD_MOE:
        if q.startswith("decode"):
            got = moe_decode(name, hw, 128)["tok_s"]
        else:
            got = moe_prefill(name, hw, int(re.search(r"\((\d+)", q).group(1)))["ttft_s"]
        out.append(("MoE", f"{name} {q} ({src})", val, got, 100 * (got / val - 1), None))
    return out


def validate_report(md: bool = False) -> str:
    rows = validate()
    L = []
    if md:
        L += ["| group | point | measured | model | error | leave-one-out |",
              "|---|---|---:|---:|---:|---:|"]
    for g, w, a, b, e, el in rows:
        loo = "-" if el is None else f"{el:+.1f}%"
        L.append(f"| {g} | {w} | {a:,.3f} | {b:,.3f} | {e:+.1f}% | {loo} |" if md else
                 f"{g:17s} {w:64s} {a:9.3f} {b:9.3f} {e:+6.1f}% {loo:>7s}")
    groups = {}
    for g, _, _, _, e, el in rows:
        groups.setdefault(g, []).append((abs(e), None if el is None else abs(el)))
    L.append("")
    for g, v in groups.items():
        lo = [x for _, x in v if x is not None]
        L.append(f"{g}: mean |error| {sum(x for x, _ in v) / len(v):.1f}% (max "
                 f"{max(x for x, _ in v):.1f}%)" +
                 (f", leave-one-out {sum(lo) / len(lo):.1f}% (max {max(lo):.1f}%)" if lo else "")
                 + f", {len(v)} points")
    return "\n".join(L)


# ------------------------------------------------------------------------------- output
def _f(x, nd=1):
    return "-" if x is None else f"{x:,.{nd}f}"


def render(sc: dict, md: bool = False) -> str:
    hw = HW(**sc["hw"])
    L = []

    def table(head, rows):
        if md:
            L.append("| " + " | ".join(head) + " |")
            L.append("|" + "|".join("---" if j == 0 else "---:" for j in range(len(head))) + "|")
            L.extend("| " + " | ".join(str(c) for c in r) + " |" for r in rows)
        else:
            w = [max(len(str(c)) for c in col) for col in zip(head, *rows)]
            for r in [head] + rows:
                L.append("  ".join(str(c).ljust(w[0]) if j == 0 else str(c).rjust(w[j])
                                   for j, c in enumerate(r)))
        L.append("")
    L += [("**" if md else "") + hw.describe() + ("**" if md else ""), ""]
    s = summary(sc)
    P = max(next(iter(sc["prefill"].values())))
    U = max(next(iter(sc["batch"].values())))
    L += [f"summary (geometric means, {sc['wformat']} weights, context {sc['ctx']}): decode "
          f"{s['decode']:.1f} tok/s, prefill {P} {s['prefill']:.1f} tok/s, batched {U} users "
          f"{s['batch']:.1f} tok/s in all, MoE {s['moe']:.2f} (sqrt of decode x prefill {P} "
          f"tok/s)", ""]
    Ps = list(next(iter(sc["prefill"].values())))
    head = ["model", "decode tok/s", "MB/token"] + \
        [h for P in Ps for h in (f"prefill {P} tok/s", f"TTFT {P} s")] + ["rows a run"]
    rows = []
    for n, d in sc["decode"].items():
        pf = sc["prefill"][n]
        rr = pf[Ps[-1]]
        rows.append([n, _f(d["tok_s"]), _f(d["mb_token"], 0)] +
                    [x for P in Ps for x in (_f(pf[P]["tok_s"]), _f(pf[P]["ttft_s"], 2))] +
                    [str(rr["rows"]) + ("" if rr["rows_last"] == rr["rows"] else
                                        f"..{rr['rows_last']}")])
    table(head, rows)
    Us = list(next(iter(sc["batch"].values())))
    table(["batched decode, total / each tok/s"] + [f"{u} users" for u in Us],
          [[n] + [f"{v[u]['total_tok_s']:.1f} / {v[u]['user_tok_s']:.1f}" for u in Us]
           for n, v in sc["batch"].items()])
    head = ["MoE", "experts on the card", "decode tok/s", "misses a token"] + \
        [h for P in Ps for h in (f"prefill {P} tok/s", f"TTFT {P} s")]
    rows = []
    for n, v in sc["moe"].items():
        f, d = v["fit"], v["decode"]
        where = "all" if d["resident"] else f"{100 * f['frac']:.0f}% ({f['slots']} slots)" + \
            (f" + {f['far_slots']} far" if f["far_slots"] else "")
        rows.append([n, where, _f(d["tok_s"], 2), _f(d["misses"], 0)] +
                    [x for P in Ps for x in (_f(v["prefill"][P]["tok_s"]),
                                             _f(v["prefill"][P]["ttft_s"], 1))])
    table(head, rows)
    return "\n".join(L)


def whatifs_report(b: str = "ypcb", md: bool = False, wformat: str = "fp4") -> str:
    res = whatifs(b, wformat)
    head = ["what-if (against " + b + ")", "decode", "prefill", "batched", "MoE", "score"]
    L = []
    if md:
        L += ["| " + " | ".join(head) + " |", "|---|---:|---:|---:|---:|---:|"]
    else:
        L.append(f"{head[0]:44s} {'decode':>8s} {'prefill':>8s} {'batched':>8s} {'MoE':>8s} "
                 f"{'score':>8s}")
    for label, kw, r in res:
        cells = [f"{r[k]:.3f}x" for k in ("decode", "prefill", "batch", "moe", "score")]
        L.append("| " + " | ".join([label] + cells) + " |" if md else
                 f"{label:44s} " + " ".join(f"{c:>8s}" for c in cells))
    return "\n".join(L)


# ------------------------------------------------------------------------------- CLI
OVERRIDES = ("S", "mcols", "act_rows", "mxu_pump", "tmem_words", "rows", "batch_rows",
             "moe_rows", "vpu", "mhz", "mem", "mts", "channels", "mem_eff", "port_bytes", "gbps",
             "pcie_gbps", "fast_gb", "imem")


def hw_from_args(a) -> HW:
    kw = {k: getattr(a, k) for k in OVERRIDES if getattr(a, k, None) is not None}
    if kw:
        kw["name"] = a.board + " + " + ", ".join(f"{k} {v}" for k, v in kw.items())
    return board(a.board, **kw)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n\n", 2)[2].split("Python")[0])
    ap.add_argument("cmd", nargs="?", default="report",
                    choices=["report", "compare", "whatifs", "validate", "fit", "boards",
                             "models", "gen-models"])
    ap.add_argument("paths", nargs="*", help="gen-models: checkpoint directories")
    ap.add_argument("--board", default="ypcb", choices=list(BOARDS))
    ap.add_argument("-S", type=int)
    for k, t in (("mcols", int), ("act-rows", int), ("mxu-pump", float), ("tmem-words", int),
                 ("rows", int), ("batch-rows", int), ("vpu", float), ("mhz", float),
                 ("mts", float), ("channels", int), ("mem-eff", float), ("port-bytes", int),
                 ("gbps", float), ("pcie-gbps", float), ("fast-gb", float), ("imem", int)):
        ap.add_argument("--" + k, type=t)
    ap.add_argument("--mem", choices=["ddr3", "ddr4", "hbm"])
    ap.add_argument("--moe-rows", choices=["all", "gather"])
    ap.add_argument("--wformat", default="fp4", choices=["fp4", "int8", "auto", "mix"])
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--models", help="comma list of short names (default: " +
                    ",".join(DENSE) + ")")
    ap.add_argument("--json")
    ap.add_argument("--markdown", action="store_true")
    a = ap.parse_args(argv)
    names = a.models.split(",") if a.models else None
    if a.cmd == "boards":
        for k, v in BOARDS.items():
            print(f"{k:8s} {v.note}\n         {v.describe()}")
        return
    if a.cmd == "models":
        for k, v in NAMES.items():
            m = model(k)
            print(f"{k:12s} {v:22s} {m['family']:7s} {m['n_layers']:3d} layers, hidden "
                  f"{m['hidden']:5d}, kappa {kappa(m)}")
        return
    if a.cmd == "gen-models":
        res = json.loads(MODELS_FILE.read_text()) if MODELS_FILE.exists() else {}
        for p in map(Path, a.paths):
            res[p.name] = shapes(p.name, json.loads((p / "config.json").read_text()),
                                 safetensors_headers(p))
            print(f"{p.name}: {res[p.name]['family']}, {res[p.name]['n_layers']} layers")
        MODELS_FILE.write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
        return
    if a.cmd == "fit":
        print(json.dumps(fit(), indent=1))
        return
    if a.cmd == "validate":
        print(validate_report(md=a.markdown))
        return
    if a.cmd == "whatifs":
        print(whatifs_report(a.board, md=a.markdown, wformat=a.wformat))
        return
    hw = hw_from_args(a)
    sc = scoreboard(hw, a.wformat, a.ctx, names)
    if a.cmd == "compare":
        base = board(a.board)
        print(render(scoreboard(base, a.wformat, a.ctx, names), a.markdown))
        print(render(sc, a.markdown))
        r = compare(base, hw, wformat=a.wformat, ctx=a.ctx, names=names)
        print("what-if / board: " + ", ".join(f"{k} {v:.3f}x" for k, v in r.items()))
    else:
        print(render(sc, a.markdown))
    if a.json:
        Path(a.json).write_text(json.dumps(sc, indent=1, default=str))


if __name__ == "__main__":
    main()
