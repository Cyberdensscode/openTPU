#!/bin/bash
# otpu-chat's streamed MoE models on the card (docs/offload.md 14): for each model a scripted chat
# of two greedy turns (tools/qual/chatmoe.py: otpu-chat's Chat, engine and settings, the card's
# decode loop picking every token), checked turn by turn against the ISA simulator's run with
# the card's configuration, with each turn's TTFT, prefill and decode tok/s and the slots' hits.
#   lfm2-8b      LFM2.5-8B-A1B: fp4 with an int8 head, its prompt token by token
#   qwen35-35b   Qwen3.5-35B-A3B: fp4, int8 head, the embedding table on the host, its prompt
#                layer by layer (R = 2, pooled slots)
#   gemma4-26b   gemma-4-26B-A4B: int8 layers, fp4 experts and head, its prompt layer by layer
# The checkpoints and pools are the offload sessions' (tools/offload/sessions/env.sh: O), each
# pool stamped with its key (pack_pool.py MODEL POOL stamp); the references are
# RF/chatmoe-<model>.json. A model whose files are not staged is skipped. Before each model
# every other file of 100 MB or more under O and ~/openTPU/models leaves the page cache (as
# card_moe.sh's DROPOTHER; a link out of ~/openTPU and ~/otpu-build is skipped, never opened)
# and its pool is read into it. Selftest before and after.
# The references, on a host with the whole checkpoints (CFG: the card's refs.py cfg, REFCFG):
#   python3 tools/qual/chatmoe.py CHECKPOINT --cfg CFG --pool POOL --out chatmoe-<model>.json
#   (--pool-create where the host has no pool: the experts packed as they are asked for)
# Run (the card's lock for the whole session; no pipe):
#   setsid -f otpu-lock --wait 3600 -- bash tools/qual/chatmoe.sh > /dev/null 2>&1
#   (the log: R/session.log, R = O/SESSION; RUNS="lfm2-8b qwen35-35b gemma4-26b"; env.sh's T,
#   O, RF, REFCFG; G26POOL relative to O)
set -u
SESSION=${SESSION:-chatmoe-$(date +%m%d-%H%M)}
source "$(dirname "$0")/../offload/sessions/env.sh"
exec > >(tee -a $R/session.log) 2>&1
G26POOL=${G26POOL:-../g26/pool-g26-fp4.split.bin}
declare -A RUN=(        # checkpoint, pool (in O)
  [lfm2-8b]="LFM2.5-8B-A1B pool-lfm25-8b-fp4.split.bin"
  [qwen35-35b]="Qwen3.5-35B-A3B pool-q35-fp4.split.bin"
  [gemma4-26b]="gemma-4-26B-A4B $G26POOL")
echo "chatmoe start $(date +%T) tree $rev mem $(mem) GB"
timeout 1800 python -m opentpu.host.selftest 2>&1 | grep -E "\[(PASS|FAIL)\]|config" | tail -12
timeout 300 python tools/qual/refs.py cfg $R/cfg-dev.pkl > /dev/null 2>&1; echo "cfg exit $?"
python - $R/cfg-dev.pkl $REFCFG <<'PY'
import dataclasses, pickle, sys
a, b = (pickle.load(open(f, "rb")) for f in sys.argv[1:])
d = {f.name: (getattr(a, f.name), getattr(b, f.name)) for f in dataclasses.fields(a)
     if getattr(a, f.name) != getattr(b, f.name)}
print("  cfg: the references' configuration" if not d else f"  cfg DIFFERS from the references' (device, ref): {d}")
sys.exit(1 if d else 0)
PY
[ $? = 0 ] || { echo "STOP: the device configuration is not the references'"; exit 2; }
for m in ${RUNS:-lfm2-8b qwen35-35b gemma4-26b}; do
  read -r md pool <<< "${RUN[$m]}"
  ref=$RF/chatmoe-$m.json
  if [ ! -f $O/$md/config.json ] || [ ! -f $O/$pool ] || [ ! -f $ref ]; then
    echo "  [SKIP] $m: not staged ($O/$md, $O/$pool, $ref)"; continue; fi
  echo "=== $m $(date +%T)"
  python - $O/$md $O/$pool $O:$HOME/openTPU/models <<'PY'
import os, sys                                # every other file of 100 MB or more out of the
keep = {os.path.realpath(sys.argv[1]), os.path.realpath(sys.argv[2])}    # page cache
ours = tuple(os.path.realpath(os.path.expanduser(d)) + os.sep for d in ("~/openTPU", "~/otpu-build"))
gb = n = 0
seen = set()
for d in sys.argv[3].split(":"):
    for root, _, files in os.walk(d):
        for f in files:
            r = os.path.realpath(os.path.join(root, f))
            if r in seen or not r.startswith(ours) or not os.path.isfile(r) or \
                    os.path.getsize(r) < 100 << 20 or r in keep or os.path.dirname(r) in keep:
                seen.add(r)
                continue
            seen.add(r)
            fd = os.open(r, os.O_RDONLY)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(fd)
            gb, n = gb + os.path.getsize(r) / 1e9, n + 1
print(f"  dropped from the page cache: {n} other files of {gb:.1f} GB")
PY
  t0=$SECONDS; cat $O/$pool > /dev/null       # the pool in the page cache (the RAM tier)
  python - $O/$pool <<'PY'
import os, sys
from opentpu.host.offload import PoolFile
p = os.path.realpath(sys.argv[1])
n = os.path.getsize(p + ".packed")
print(f"  pool: {PoolFile(p, os.path.getsize(p) // n, True, False).resident(range(n)) / 1e9:.2f} of "
      f"{os.path.getsize(p) / 1e9:.2f} GB in the page cache, {sum(open(p + '.packed', 'rb').read())} "
      f"of {n} experts packed, key {'yes' if os.path.exists(p + '.key') else 'NO (not stamped)'}")
PY
  echo "  (read in $((SECONDS - t0)) s, mem $(mem) GB)"
  timeout 3600 python tools/qual/chatmoe.py $O/$md --pool $O/$pool --backend board \
    --cfg $R/cfg-dev.pkl --check $ref --out $R/chatmoe-$m.json > $R/chatmoe-$m.log 2>&1
  echo "  exit $? $(date +%T)"; grep -E "\[(PASS|FAIL)\]|Error|Traceback" $R/chatmoe-$m.log | tail -3
  [ -f $R/chatmoe-$m.json ] || continue
  python - $R/chatmoe-$m.json <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
print(f"  {r['weights']}; {r['experts']}; layer-major {r['layer_major']}, loop on the card "
      f"{r['on_card']}, load {r['load_s']} s, host {r['host_mem']}")
for i, t in enumerate(r["turns"]):
    print(f"  turn {i + 1}: prompt {t['prompt_tokens']} tokens (reused {t['reused']}"
          f"{', restarted' if t['restarted'] else ''}) in {t['prefill_s']} s "
          f"({t['prefill_tok_s']} tok/s, device {t['prefill_dev_tok_s']}), {t['prefill_misses']} "
          f"misses; TTFT {t['ttft_s']} s; decode {t['gen_tokens']} tokens at {t['decode_tok_s']} "
          f"tok/s (device {t['decode_dev_tok_s']}), hits {t['hit_rate']}, "
          f"{t['mb_per_token']} MB/token; end {t['end']}")
PY
done
echo "=== final selftest $(date +%T)"
timeout 1800 python -m opentpu.host.selftest 2>&1 | grep -E "\[(PASS|FAIL)\]|config" | tail -12
echo "CHATMOE DONE $(date +%T)"
