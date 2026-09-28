#!/usr/bin/env bash
# 精确复刻 8095 容器（kvmem-test）。
#
# 本脚本由 `docker inspect kvmem-test` 反推而来。
# 与 start-8095.sh 的差异（start-8095.sh 是「带 Web UI」的变体，不是实际在跑的配置）：
#   start-8095.sh 多出: --shm-size 2g、-e LD_LIBRARY_PATH=/src/build/bin、-w /src、
#                       -v <repo>/kvmem-ui-dist:/ui:ro、--ui-dir /ui
#   start-8095.sh 缺少: --api-key
# 重建/改参时以本脚本或 `docker inspect` 为准，不要直接跑 start-8095.sh。
#
# 用法:
#   bash run-8095-exact.sh [context] [budget]
#     context 默认 131072（128K 逻辑工作区）
#     budget  默认 24576 （检索可留在 GPU 的历史 token 数 = 有效注意力窗口）
#   例: bash run-8095-exact.sh 131072 36864    # 对齐官方 IQ3 配方
#
# 环境变量（按需覆盖）:
#   KVREPO   fork 源码目录（同时挂载为容器 /src）  默认 /home/$USER/kvmem-llama.cpp
#   MODEL    模型 GGUF 路径                        默认 /llama/models/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf
#   API_KEY  服务端鉴权 key                        无默认值，必须显式设置
#   KVDTYPE  KV 缓存精度                           默认 q5_0
#   PORT     宿主端口                              默认 8095
#   BIND_IP  宿主绑定网卡                          默认 127.0.0.1（仅本机）
#   NAME     容器名                                默认 kvmem-test
#
# ⚠️ 安全默认值（v2 起，改这三处的理由）:
#   1. 端口默认只绑 127.0.0.1。要让局域网访问必须显式 BIND_IP=0.0.0.0
#      —— 注意 /health 是**免鉴权**的，暴露到网络等于连存活信息一起公开。
#   2. API_KEY 未设置或仍是 changeme 时**直接退出**（以前只警告然后照跑）。
#      仅本机临时自测可 ALLOW_INSECURE_KEY=1 显式放行。
#   3. 校验段打印 Cmd 时遮蔽 --api-key 的值，避免密钥进入终端回滚 / CI 日志 / 截图。
#      更彻底的做法是改用 `--api-key-file`（密钥不进 docker inspect），
#      见 docs/parameters.md 的「--api-key / --api-key-file」一节。
set -eu

# $USER 并非所有环境都设置（例如 Windows Git Bash），而 set -u 会让它直接报错
: "${USER:=$(id -un 2>/dev/null || echo user)}"

CTX="${1:-131072}"
BUDGET="${2:-24576}"

KVREPO="${KVREPO:-/home/$USER/kvmem-llama.cpp}"
MODEL="${MODEL:-/llama/models/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf}"
KVDTYPE="${KVDTYPE:-q5_0}"
BIND_IP="${BIND_IP:-127.0.0.1}"

# ---- 安全门 1：拒绝占位密钥 ----
API_KEY="${API_KEY:-}"
if [ -z "$API_KEY" ] || [ "$API_KEY" = "changeme" ]; then
  if [ "${ALLOW_INSECURE_KEY:-0}" = "1" ]; then
    echo "[warn] 正在使用占位密钥 changeme（已由 ALLOW_INSECURE_KEY=1 显式放行）" >&2
    echo "[warn] 仅限本机自测；对外提供服务前务必更换" >&2
    API_KEY="${API_KEY:-changeme}"
  else
    cat >&2 <<'EOF'
[error] 未设置 API_KEY（或仍是占位符 changeme），已中止。

  设置一个真实密钥再跑：
      export API_KEY="$(openssl rand -hex 24)"

  仅本机临时自测、且明确接受风险时，可以显式放行：
      ALLOW_INSECURE_KEY=1 bash run-8095-exact.sh
EOF
    exit 1
  fi
fi

# ---- 安全门 2：绑定到全网卡时给出明确警告 ----
case "$BIND_IP" in
  0.0.0.0|::|"*")
    echo "[warn] BIND_IP=$BIND_IP —— 服务将对整个网络开放。" >&2
    echo "[warn] /health 免鉴权；/v1/* 依赖 --api-key，请确认密钥足够强且已换掉默认值。" >&2
    ;;
esac

PORT="${PORT:-8095}"
NAME="${NAME:-kvmem-test}"
IMAGE="${IMAGE:-nvidia/cuda:12.8.1-devel-ubuntu24.04}"

# ---- 回滚留档（改生产服务的最低要求）----
if docker inspect "$NAME" >/dev/null 2>&1; then
  ROLLBACK="$KVREPO/kvmem-cmd-rollback-$(date +%Y%m%d-%H%M%S).json"
  docker inspect "$NAME" --format '{{json .Config.Cmd}}' > "$ROLLBACK"
  echo "[rollback] 旧 Cmd 已留档: $ROLLBACK"
fi

# ---- 重建（docker restart 改不了命令行参数，必须 rm + run）----
docker rm -f "$NAME" >/dev/null 2>&1 || true

docker run -d --name "$NAME" \
  --gpus all \
  -p "${BIND_IP}:${PORT}:8080" \
  --restart unless-stopped \
  -v "${KVREPO}:/src" \
  -v "${MODEL}:/models/model.gguf:ro" \
  "$IMAGE" \
  /src/build/bin/llama-kvmem-server -m /models/model.gguf \
  --host 0.0.0.0 --port 8080 -c "$CTX" -n 16384 \
  --kvmem-budget "$BUDGET" --kvmem-gen-reserve 16384 --kv-dtype "$KVDTYPE" \
  --spec-type draft-mtp --spec-draft-n-max 2 --kvmem-mtp-state replay \
  --api-key "$API_KEY"

echo "容器已重建: -c=$CTX  --kvmem-budget=$BUDGET  --kv-dtype=$KVDTYPE"
echo "等待就绪..."

for i in $(seq 1 40); do
  sleep 3
  if curl -s --noproxy '*' -m 3 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    echo "就绪（$((i*3))s）"
    break
  fi
done

# ---- 校验 ----
docker inspect "$NAME" --format 'RestartCount={{.RestartCount}} Status={{.State.Status}}'
# ⚠️ Cmd 里含 --api-key 的值，必须遮蔽后再打印
#    （否则密钥会进入终端回滚、CI 日志、截图 —— 这是本脚本早期版本的漏洞）
echo -n "Cmd:   "; docker inspect "$NAME" --format '{{json .Config.Cmd}}' | python3 -c '
import json,sys
c=json.load(sys.stdin)
for i,a in enumerate(c):
    if a=="--api-key" and i+1<len(c): c[i+1]="***"
    elif isinstance(a,str) and a.startswith("--api-key="): c[i]="--api-key=***"
print(json.dumps(c,ensure_ascii=False))'
echo -n "Binds: "; docker inspect "$NAME" --format '{{json .HostConfig.Binds}}'

echo -n "n_ctx: "
curl -s --noproxy '*' -m 8 "http://127.0.0.1:${PORT}/slots" \
  -H "Authorization: Bearer ${API_KEY}" \
  | python3 -c 'import json,sys; s=json.load(sys.stdin)[0]; print(s["n_ctx"], "| busy =", s["is_processing"])' 2>/dev/null \
  || echo "(查询失败)"

echo "--- free ---"; free -m | head -3
echo "--- gpu  ---"; nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
echo "--- log  ---"; docker logs --tail 6 "$NAME" 2>&1 | tail -6
