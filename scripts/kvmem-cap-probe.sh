#!/usr/bin/env bash
# KVMem 长上下文容量曲线压测：单次长 prefill + 高频内存采样，带 anon 上限自动中止。
#
# ☠️ 危险：本脚本的设计意图就是**把宿主内存推到接近 OOM 击杀线**。
#    - 在共享 / 生产宿主上跑，会**连带杀掉宿主上的其它服务**（OOM 杀进程，不挑对象）。
#      这正是本项目 FINDINGS 里那条「客户端反复掉线」故障的成因机制。
#    - 因此必须显式确认后才运行：CONFIRM_OOM_RISK=1
#    - 中止线 = 第 3 个参数（anon 上限 MiB），默认 8000。
#      宿主 15.5 GiB 时击杀线约 13.6 GiB，8000 留出 ~5.6 GiB 给宿主其它进程。
#      **调高这个值 = 主动缩小安全余量**，别为了多测几个 token 把宿主搭进去。
#
# ⚠️ 该服务 total_slots=1 —— 压测会「独占服务」，跑之前先确认空闲：
#      curl -s --noproxy '*' http://127.0.0.1:8095/slots -H "Authorization: Bearer $API_KEY" \
#        | python3 -c 'import json,sys; print(json.load(sys.stdin)[0]["is_processing"])'
#
# 用法: CONFIRM_OOM_RISK=1 bash kvmem-cap-probe.sh [sudo密码] <prompt字符数> [anon上限MiB]
#   例: CONFIRM_OOM_RISK=1 bash kvmem-cap-probe.sh "" 300000 8000
#
# 环境变量: NAME（默认 kvmem-test）、PORT（默认 8095）、API_KEY、CONFIRM_OOM_RISK
#
# 输出: CSV 表头 elapsed_s,anon_MiB,file_MiB,shmem_MiB,avail_MiB,swap_MiB,gpu_MiB
#       结尾打印 peak_anon_MiB / wall_s / abort，以及响应的 usage 与 timings。

PW="${1:-}"
CHARS="${2:?需要 prompt 字符数，例如 300000}"
LIMIT="${3:-8000}"
NAME="${NAME:-kvmem-test}"
PORT="${PORT:-8095}"
API_KEY="${API_KEY:-changeme}"

# 本脚本是**客户端**（只压服务，不启动任何东西），所以不像 run-8095-exact.sh
# 那样对占位密钥硬失败。但占位密钥会让请求 401，压测结果无意义 —— 明确提示。
if [ "$API_KEY" = "changeme" ]; then
  echo "[warn] API_KEY 未设置（或仍是占位符 changeme）—— 请求会返回 401，压测无效。" >&2
  echo "[warn]   export API_KEY=<服务端 --api-key 设的那个值>" >&2
fi

# ---- 安全门：OOM 风险必须显式确认 ----
if [ "${CONFIRM_OOM_RISK:-0}" != "1" ]; then
  cat >&2 <<'EOF'
[error] 本脚本会主动把宿主内存推到接近 OOM 击杀线，可能连带杀掉宿主上的其它服务。
        确认当前宿主上「没有别的服务会被波及」之后，加 CONFIRM_OOM_RISK=1 再跑：

    CONFIRM_OOM_RISK=1 bash kvmem-cap-probe.sh "" 300000 8000

        中止线（第 3 个参数）默认 8000 MiB；宿主 15.5 GiB 时击杀线约 13.6 GiB。
        调高中止线等于主动缩小安全余量。
EOF
  exit 1
fi

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
