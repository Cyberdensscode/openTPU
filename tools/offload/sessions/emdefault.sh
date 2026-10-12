#!/bin/bash
# Session emdef (docs/offload.md 13.15): the expert-major default on the card, before its host swap.
# Trees: the candidate deploy's host, ~/otpu-build/deploy_em_91a9c2d3/host (T: main ee8f118d + pfB b6cfa9db + the
# default e42f5eb2); production's host for the A arm, read-only, no bytecode written (PH: deploy_mt4_91a9c2d3/host,
# main 23b68089). Card: production deploy_mt4_91a9c2d3 as loaded (EXPECT 91a9c2d3, register map 4). One otpu-lock:
# 1. selftest (T); the device configuration must be production's qual/cfg.pkl;
# 2. the production qual's chatmoe step from T: tools/qual/chatmoe.sh, RUNS="qwen35-35b gemma4-26b" (two greedy turns
#    each, the second continuing the context), token-exact against the ISA references (O/chatmoe-<model>.json);
# 3. otpu-chat by name from T, one greedy answer each (qual_run.sh's), its reply against production's mt4 chat-<model>.txt;
# 4. per model, production's host (P) against T (N): a discarded warm-up run (N), then P N N P, each a chatmoe.py chat of
#    three turns: TURNS' two, then a long passage (lmtime's text and a question, about 150 tokens continuing the
#    context). Every run's tokens must equal the warm-up's, and its first two turns the ISA references';
# 5. selftest.
# A build other than EXPECT, a failed selftest, another configuration, a FAIL, a failed or timed-out run or a mismatch
# stops it (then the final selftest). About 30 min.
# Run (no pipe): setsid nohup ~/.local/bin/otpu-lock --wait 600 -- bash $T/tools/offload/sessions/emdefault.sh
#   > ~/otpu-build/moe-ab/emdef.launch.log 2>&1 < /dev/null &
set -u
for v in $(env | grep -o '^OTPU_[A-Z0-9_]*' | grep -vE '^OTPU_LOCK_(HELD|PID|WAIT)$'); do unset $v; done
export OTPU_IMG_CACHE_GB=30 OTPU_IMG_CACHE_FREE_GB=6      # (the card host's, from its ~/.bashrc)
MA=${MA:-$HOME/otpu-build/moe-ab}
T=${T:-$HOME/otpu-build/deploy_em_91a9c2d3/host}
PH=${PH:-$HOME/otpu-build/deploy_mt4_91a9c2d3/host}
R=${R:-$MA/emdef}
source $T/tools/offload/sessions/env.sh       # (O: card2; OTPU_QUIET claimed, removed at exit)
exec > >(tee -a $R/session.log) 2>&1
EXPECT=${EXPECT:-91a9c2d3}
PD=$HOME/otpu-build/deploy_mt4_91a9c2d3        # (production's qual/cfg.pkl and chat-<model>.txt)
DROPOTHER=$HOME/openTPU/models:$O:$HOME/otpu-build/offload/g26it
T1="What is the capital of France? Answer in one sentence."
T2="And what is it famous for? Two sentences."
T3="$(python -c 'import json, sys; print(json.load(open(sys.argv[1]))["prompt"])' \
      $HOME/otpu-build/moepf/lmtime/q35w-hf.json) Summarize this passage in one sentence."
declare -A M=([q]="qwen35-35b Qwen3.5-35B-A3B pool-q35-fp4.split.bin"
              [g]="gemma4-26b gemma-4-26B-A4B-it ../g26it/pool-g26it-fp4.split.bin")
fail=0
echo "emdef start $(date +%T) T $T ($(cat $T/COMMIT)) PH $PH ($(cat $PH/COMMIT)) mem $(mem) GB, /home $(df -BG --output=avail /home | tail -1 | tr -d ' ') free; $(uptime)"
st=$(timeout 600 python -m opentpu.host.selftest 2>&1)
echo "$st" | grep -E "\[(PASS|FAIL)\]|ALL|stopped" | tail -14
echo "$st" | grep -q "build $EXPECT" || { echo "STOP: the card's build is not $EXPECT"; exit 2; }
echo "$st" | grep -q "register map 4" || { echo "STOP: not register map 4"; exit 2; }
echo "$st" | grep -q "ALL PASS" || { echo "STOP: selftest"; exit 2; }
timeout 300 python tools/qual/refs.py cfg $R/cfg-dev.pkl > /dev/null 2>&1; echo "cfg exit $?"
python - $R/cfg-dev.pkl $PD/qual/cfg.pkl <<'PY' || { echo "STOP: the device configuration"; exit 2; }
import dataclasses, pickle, sys
a, b = (pickle.load(open(f, "rb")) for f in sys.argv[1:])
d = {f.name: (getattr(a, f.name), getattr(b, f.name)) for f in dataclasses.fields(a)
     if getattr(a, f.name) != getattr(b, f.name)}
print("  cfg: production's (qual/cfg.pkl)" if not d else f"  cfg DIFFERS from production's: {d}")
sys.exit(1 if d else 0)
PY
warm() {   # checkpoint, pool (in O): every other big file out of the page cache, the pool read into it
  python - $O/$1 $O/$2 $DROPOTHER <<'PY'
import os, sys                      # every other file of 100 MB or more out of the page cache
keep = {os.path.realpath(sys.argv[1]), os.path.realpath(sys.argv[2])}  # (card_moe.sh's DROPOTHER)
ours = tuple(os.path.realpath(os.path.expanduser(d)) + os.sep for d in ("~/openTPU", "~/otpu-build"))
gb = n = 0
seen = set()
for d in sys.argv[3].split(":"):
    for root, _, files in os.walk(os.path.expanduser(d)):
        for f in files:
            r = os.path.realpath(os.path.join(root, f))
            if r in seen or not r.startswith(ours) or not os.path.isfile(r) \
                    or os.path.getsize(r) < 100 << 20 or r in keep or os.path.dirname(r) in keep:
                continue
            seen.add(r)
            fd = os.open(r, os.O_RDONLY)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(fd)
            gb, n = gb + os.path.getsize(r) / 1e9, n + 1
print(f"  dropped from the page cache: {n} other files of {gb:.1f} GB")
PY
  local t0=$SECONDS; cat $O/$2 > /dev/null; echo "  pool read in $((SECONDS - t0)) s, mem $(mem) GB"
}
# 2. the production qual's chatmoe step, from T
echo "=== chatmoe.sh $(date +%T)"
env RUNS="qwen35-35b gemma4-26b" SESSION=chatmoe R=$R/chatmoe timeout 3600 bash $T/tools/qual/chatmoe.sh \
  > $R/chatmoe.log 2>&1
echo "  CHATMOE EXIT $? $(date +%T)"
grep -E "\[(PASS|FAIL|SKIP)\] [^ ]+ against|  turn [0-9]|Error|Traceback" $R/chatmoe.log | cut -c1-260
[ "$(grep -c "turns' tokens the same" $R/chatmoe.log)" = 2 ] && ! grep -q "\[FAIL\]" $R/chatmoe.log \
  || { echo "STOP: chatmoe"; fail=1; }
# 3. otpu-chat by name, from T
for m in q g; do
  [ $fail = 0 ] || break
  read -r name md pool <<< "${M[$m]}"
  echo "=== otpu-chat $name $(date +%T)"
  warm $md $pool
  (cd /tmp && timeout 1200 python -m opentpu.host.chat --backend board --model $name --greedy --max-new 24 \
     --prompt "$T1" > $R/chat-$name.txt 2>&1)
  echo "  CHAT $name EXIT $? $(date +%T)"
  python - $R/chat-$name.txt $PD/chat-$name.txt <<'PY' || { echo "STOP: otpu-chat $name"; fail=1; }
import sys
def part(f):
    ln = [x for x in open(f).read().splitlines() if x.strip()]
    reply = [x for x in ln if not x.startswith(("loading ", "experts:", "weights:", "loaded in", "[TTFT"))]
    return reply, [x for x in ln if x.startswith(("loaded in", "[TTFT"))]
(a, sa), (b, sb) = part(sys.argv[1]), part(sys.argv[2])
print(f"  [{'PASS' if a == b and a else 'FAIL'}] reply {a!r} {'=' if a == b else '!='} production's (mt4)")
for x, y in zip(sa, sb):
    print(f"    {x[:200]}\n    (mt4) {y[:200]}")
sys.exit(0 if a == b and a else 1)
PY
done
# 4. production's host (P) against T (N), per model: W, then P N N P
for m in q g; do
  [ $fail = 0 ] || break
  read -r name md pool <<< "${M[$m]}"
  for x in W P1 N1 N2 P2; do
    tree=$T; [ ${x:0:1} = P ] && tree=$PH
    echo "=== $m$x $(date +%T) ($name, host $(cat $tree/COMMIT))"
    warm $md $pool
    (cd $tree && env PYTHONPATH=$tree PYTHONDONTWRITEBYTECODE=1 timeout 1200 python tools/qual/chatmoe.py \
       $O/$md --pool $O/$pool --backend board --cfg $R/cfg-dev.pkl --turn "$T1" --turn "$T2" --turn "$T3" \
       --out $R/$m$x.json > $R/$m$x.log 2>&1)
    rc=$?; echo "  exit $rc $(date +%T)"; grep -E "Error|Traceback" $R/$m$x.log | tail -3
    [ $rc = 0 ] && [ -f $R/$m$x.json ] || { echo "STOP: $m$x"; fail=1; break; }
    python - $R/$m$x.json $R/${m}W.json $RF/chatmoe-$name.json <<'PY' || { echo "STOP: $m$x differs"; fail=1; break; }
import json, sys
d, w, ref = (json.load(open(f)) for f in sys.argv[1:])
tok = [t["tokens"] for t in d["turns"]]
ok_w = tok == [t["tokens"] for t in w["turns"]]
ok_r = tok[:2] == [t["tokens"] for t in ref["turns"]]
print(f"  [{'PASS' if ok_w and ok_r else 'FAIL'}] tokens {'=' if ok_w else '!='} W's, first two turns "
      f"{'=' if ok_r else '!='} the ISA reference's; layer-major {d['layer_major']}, load {d['load_s']} s")
for i, t in enumerate(d["turns"]):
    print(f"  turn {i + 1}: prompt {t['prompt_tokens']} (reused {t['reused']}) TTFT {t['ttft_s']} s, prefill "
          f"{t['prefill_tok_s']} tok/s (device {t['prefill_dev_tok_s']}), {t['prefill_misses']} misses; decode "
          f"{t['gen_tokens']} at {t['decode_tok_s']} tok/s (device {t['decode_dev_tok_s']}); end {t['end']}")
sys.exit(0 if ok_w and ok_r else 1)
PY
  done
done
echo "=== final selftest $(date +%T)"
timeout 600 python -m opentpu.host.selftest 2>&1 | grep -E "\[(PASS|FAIL)\]|ALL|stopped" | tail -14
echo "emdef end $(date +%T) fail $fail"
python - $R <<'PY'
import json, os, sys
r = sys.argv[1]
def get(run):
    f = f"{r}/{run}.json"
    return json.load(open(f)) if os.path.exists(f) else None
for name, m in (("35B", "q"), ("26B", "g")):
    p1, n1, n2, p2 = (get(m + x) for x in ("P1", "N1", "N2", "P2"))
    if None in (p1, n1, n2, p2):
        print(f"  {name}: incomplete"); continue
    for i in range(len(p1["turns"])):
        for key in ("ttft_s", "decode_tok_s", "prefill_dev_tok_s"):
            v = lambda c: c["turns"][i][key] or 0.0     # noqa: E731
            d1, d2 = v(n1) - v(p1), v(n2) - v(p2)
            mp, mn = (v(p1) + v(p2)) / 2, (v(n1) + v(n2)) / 2
            print(f"  {name} turn {i + 1} ({p1['turns'][i]['prompt_tokens']} tokens) {key}: P {v(p1):.3f} {v(p2):.3f}, "
                  f"N {v(n1):.3f} {v(n2):.3f}; pairs N-P {d1:+.3f} {d2:+.3f}; mean N-P {mn - mp:+.3f}")
PY
date > $R/DONE
echo "EMDEF EXIT $fail $(date +%T)"
