#!/usr/bin/env bash
# git bisect run step: does Loom's Qwen3-Embedding NPU program register?
# Exit 0: ready (good). 1: XDNA tile-memory failure (bad). 125: skip.
SP=/tmp/claude-1000/-home-mixer-gufo/8e84d6e4-1c1c-4d14-ba0b-69045e8d2aca/scratchpad
B=$SP/hrx-pin-build
LOG=$SP/bisect-npu.log
rev=$(git rev-parse --short HEAD)
echo "== $rev $(git log -1 --format=%s)" >> "$LOG"
if ! systemd-run --user --scope -q -p MemoryMax=40G -p TasksMax=512 -- \
     nix shell nixpkgs#ninja nixpkgs#cmake nixpkgs#python3 -c \
     ninja -C "$B" -j24 loom/src/loom/tools/loom-serve/loom-serve > "$SP/bisect-build-$rev.log" 2>&1; then
  echo "   build failed: skip" >> "$LOG"; exit 125
fi
unit=loom-bisect-$rev
( cd "$SP/loom-emb" && systemd-run --user --scope -q --unit="$unit" -p MemoryMax=24G -p TasksMax=512 -- \
    "$B/loom/src/loom/tools/loom-serve/loom-serve" --device='xdna://0?columns=8' --pool_backing=fixed \
    --model="$SP/loom-emb/model.json" --listen=127.0.0.1:8097 > "$SP/bisect-serve-$rev.log" 2>&1 ) &
result=$(python3 - <<'EOF'
import re, time, urllib.request
t = time.time()
while time.time() - t < 600:
    try:
        b = urllib.request.urlopen('http://127.0.0.1:8097/v1/status', timeout=2).read().decode()
        m = re.search(r'"registration":"(\w+)"', b)
        if m and m.group(1) == 'ready':
            print('good'); break
        if m and m.group(1) == 'failed':
            print('bad' if 'XDNA/049' in b else 'other:' + b[b.find('"message"'):][:200]); break
    except Exception:
        pass
    time.sleep(2)
else:
    print('timeout')
EOF
)
systemctl --user stop "$unit.scope" 2>/dev/null
for _ in $(seq 1 30); do pgrep -f "listen=127.0.0.1:8097" > /dev/null || break; sleep 1; done
echo "   $result" >> "$LOG"
case "$result" in
  good) exit 0 ;;
  bad) exit 1 ;;
  *) exit 125 ;;
esac
