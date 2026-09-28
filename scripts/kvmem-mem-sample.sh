#!/usr/bin/env bash
# 被动采样 8095 容器的宿主内存 / 显存 / 当前上下文长度。
# 不打扰服务（只读 + 一次 /slots 查询）。
#
# 用法: bash kvmem-mem-sample.sh [sudo密码] [轮数] [间隔秒]
#   密码留空（""）则直接执行（若当前用户可读 cgroup）
#   例: bash kvmem-mem-sample.sh "" 24 15
#
# 环境变量: NAME（容器名，默认 kvmem-test）、PORT（默认 8095）、
#           API_KEY（服务端 --api-key 设的那个值）
#
# 输出列:
#   anon_MiB   KV 池所在，真正要盯的（顶到击杀线就 OOM）
#   file_MiB   page cache（权重 mmap），下降是正常的
#   avail_MiB  宿主可用内存
#   swap_MiB   swap 已用（持续增长 = 危险信号）
#   gpu_MiB    整卡显存占用
#   n_prompt / n_decoded  当前上下文长度 / 已生成数

PW="${1:-}"
N="${2:-24}"
IV="${3:-15}"
NAME="${NAME:-kvmem-test}"
PORT="${PORT:-8095}"
API_KEY="${API_KEY:-changeme}"

# 本脚本是**客户端**（只查询服务，不启动任何东西），所以不像 run-8095-exact.sh
# 那样对占位密钥硬失败。但用占位密钥只会拿到 401，采样表里 n_prompt/n_decoded
# 会全是 "-" —— 这里明确提示，避免"表出来了但不知道数据为什么是空的"。
if [ "$API_KEY" = "changeme" ]; then
  echo "[warn] API_KEY 未设置（或仍是占位符 changeme）—— /slots 会返回 401，" >&2
  echo "[warn] n_prompt / n_decoded 两列会显示 '-'。正确用法：" >&2
  echo "[warn]   export API_KEY=<服务端 --api-key 设的那个值>" >&2
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

printf '%-10s %10s %10s %9s %9s %9s %7s %9s %9s\n' \
  TS anon_MiB file_MiB shmem_MiB avail_MiB swap_MiB gpu_MiB n_prompt n_decoded

for _ in $(seq 1 "$N"); do
  TS=$(date -u +%H:%M:%S)
  MS=$(SUDO cat "$CG/memory.stat" 2>/dev/null)
  ANON=$(printf '%s\n' "$MS" | awk '/^anon /{print $2}')
  FILE=$(printf '%s\n' "$MS" | awk '/^file /{print $2}')
  SHM=$(printf '%s\n' "$MS" | awk '/^shmem /{print $2}')
  AVAIL=$(awk '/^MemAvailable/{print $2}' /proc/meminfo)
  SWAP=$(awk '/^SwapTotal/{t=$2} /^SwapFree/{f=$2} END{print t-f}' /proc/meminfo)
  GPU=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  SL=$(curl -s --noproxy '*' -m 5 "http://127.0.0.1:${PORT}/slots" -H "Authorization: Bearer ${API_KEY}")
  NP=$(printf '%s' "$SL" | grep -o '"n_prompt_tokens":[0-9]*' | head -1 | cut -d: -f2)
  ND=$(printf '%s' "$SL" | grep -o '"n_decoded":[0-9]*' | head -1 | cut -d: -f2)
  printf '%-10s %10.0f %10.0f %9.0f %9.0f %9.0f %7s %9s %9s\n' \
    "$TS" \
    "$(awk -v a="${ANON:-0}" 'BEGIN{print a/1048576}')" \
    "$(awk -v a="${FILE:-0}" 'BEGIN{print a/1048576}')" \
    "$(awk -v a="${SHM:-0}"  'BEGIN{print a/1048576}')" \
    "$(awk -v a="${AVAIL:-0}" 'BEGIN{print a/1024}')" \
    "$(awk -v a="${SWAP:-0}" 'BEGIN{print a/1024}')" \
    "$GPU" "${NP:- -}" "${ND:- -}"
  sleep "$IV"
done
