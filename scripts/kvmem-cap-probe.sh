#!/usr/bin/env bash
# KVMem 长上下文容量曲线压测：单次长 prefill + 高频内存采样，带 anon 上限自动中止。
#
# ⚠️ 该服务 total_slots=1 —— 压测会「独占服务」，跑之前先确认空闲：
#      curl -s --noproxy '*' http://127.0.0.1:8095/slots -H "Authorization: Bearer $API_KEY" \
#        | python3 -c 'import json,sys; print(json.load(sys.stdin)[0]["is_processing"])'
#
# 用法: bash kvmem-cap-probe.sh [sudo密码] <prompt字符数> [anon上限MiB]
#   例: bash kvmem-cap-probe.sh "" 300000 10500
#
# 环境变量: NAME（默认 kvmem-test）、PORT（默认 8095）、API_KEY（默认 changeme）
#
# 输出: CSV 表头 elapsed_s,anon_MiB,file_MiB,shmem_MiB,avail_MiB,swap_MiB,gpu_MiB
#       结尾打印 peak_anon_MiB / wall_s / abort，以及响应的 usage 与 timings。

PW="${1:-}"
CHARS="${2:?需要 prompt 字符数，例如 300000}"
LIMIT="${3:-10500}"
NAME="${NAME:-kvmem-test}"
PORT="${PORT:-8095}"
API_KEY="${API_KEY:-changeme}"

if [ -n "$PW" ]; then
  SUDO() { printf '%s\n' "$PW" | sudo -S -p '' "$@"; }
else
  SUDO() { "$@"; }
fi

CID=$(SUDO docker inspect -f '{{.Id}}' "$NAME" 2>/dev/null)
if [ -z "$CID" ]; then
  echo "找不到容器 $NAME（或没有权限）" >&2
  exit 1
fi
CG="/sys/fs/cgroup/system.slice/docker-${CID}.scope"
STAT() { SUDO cat "$CG/memory.stat" 2>/dev/null; }

# 构造一个纯填充的长 prompt（不含特殊字符，避免模板干扰）
python3 - "$CHARS" <<'PY' > /tmp/kvprobe.json
import json, sys
n = int(sys.argv[1])
unit = "The quick brown fox jumps over the lazy dog. "
filler = (unit * (n // len(unit) + 1))[:n]
payload = {"model": "model.gguf",
           "messages": [{"role": "user", "content": filler + "\n\nReply with the single word: OK"}],
           "max_tokens": 1, "temperature": 0.0, "stream": False}
print(json.dumps(payload))
PY

echo "target_chars=$CHARS  anon_limit_MiB=$LIMIT"
echo "elapsed_s,anon_MiB,file_MiB,shmem_MiB,avail_MiB,swap_MiB,gpu_MiB"

BASE=$(STAT)
echo "0,$(printf '%s\n' "$BASE" | awk '/^anon /{printf "%.0f",$2/1048576}'),$(printf '%s\n' "$BASE" | awk '/^file /{printf "%.0f",$2/1048576}'),$(printf '%s\n' "$BASE" | awk '/^shmem /{printf "%.0f",$2/1048576}'),$(awk '/^MemAvailable/{printf "%.0f",$2/1024}' /proc/meminfo),$(awk '/^SwapTotal/{t=$2}/^SwapFree/{f=$2}END{printf "%.0f",(t-f)/1024}' /proc/meminfo),$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)"

START=$(date +%s)
( curl -s --noproxy '*' -m 1800 -X POST "http://127.0.0.1:${PORT}/v1/chat/completions" \
    -H "Authorization: Bearer ${API_KEY}" -H 'Content-Type: application/json' \
    --data-binary @/tmp/kvprobe.json -o /tmp/kvprobe.out ) &
CURL=$!

PEAK=0; ABORT=0
while kill -0 "$CURL" 2>/dev/null; do
  MS=$(STAT)
  A=$(printf '%s\n' "$MS" | awk '/^anon /{printf "%.0f",$2/1048576}')
  [ -z "$A" ] && A=0
  [ "$A" -gt "$PEAK" ] && PEAK=$A
  printf '%s,%s,%s,%s,%s,%s,%s\n' \
    "$(( $(date +%s) - START ))" "$A" \
    "$(printf '%s\n' "$MS" | awk '/^file /{printf "%.0f",$2/1048576}')" \
    "$(printf '%s\n' "$MS" | awk '/^shmem /{printf "%.0f",$2/1048576}')" \
    "$(awk '/^MemAvailable/{printf "%.0f",$2/1024}' /proc/meminfo)" \
    "$(awk '/^SwapTotal/{t=$2}/^SwapFree/{f=$2}END{printf "%.0f",(t-f)/1024}' /proc/meminfo)" \
    "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)"
  if [ "$A" -gt "$LIMIT" ]; then
    echo "ABORT: anon>${LIMIT}MiB"
    kill "$CURL" 2>/dev/null
    ABORT=1
    break
  fi
  sleep 5
done
wait "$CURL" 2>/dev/null

echo "peak_anon_MiB=$PEAK  wall_s=$(( $(date +%s) - START ))  abort=$ABORT"
echo "--- response ---"
python3 - <<'PY'
import json
try:
    d = json.load(open("/tmp/kvprobe.out"))
    u = d.get("usage", {})
    t = d.get("timings", {})
    print("prompt_tokens =", u.get("prompt_tokens"), "completion_tokens =", u.get("completion_tokens"))
    for k in ("prompt_n", "prompt_ms", "prompt_per_second",
              "predicted_n", "predicted_ms", "predicted_per_second"):
        if k in t:
            print(f"  {k} = {t[k]}")
    print()
    print("注意: 若 prompt 超过 budget+gen_reserve，prefill 末尾会有数十秒『停摆』，")
    print("      这段不计入任何进度行 —— 评估端到端耗时请用 prompt_ms，不要用进度行 t/s 反推。")
except Exception as e:
    print("parse fail:", e)
    print(open("/tmp/kvprobe.out", errors="replace").read()[:500])
PY
