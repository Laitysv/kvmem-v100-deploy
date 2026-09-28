# 端到端部署教程

> 从一台裸机到跑通长上下文服务，逐步可复制粘贴。
> 每一步都有**预期输出**与**验证方法**——对不上就停下来查，别往下走。
>
> 参数细节见 [parameters.md](parameters.md)；关键结论见 [FINDINGS.md](../FINDINGS.md)。

---

## 0. 你将得到什么

一条能用的 OpenAI 兼容推理服务：

```
客户端 (Codex / 任意 OpenAI SDK)
        │  POST /v1/chat/completions
        ▼
宿主 :8095  →  容器 :8080  →  llama-kvmem-server
                                 │
                                 └─ Qwen3.8-27B (IQ3_S, 11.29 GiB)
                                    128K 逻辑工作区 / 24K 有效窗口
```

---

## 1. 前置条件

### 1.1 硬件与系统

| 项 | 要求 | 本仓库实测环境 |
|---|---|---|
| GPU | **Tesla V100（`sm_70`）** 16 GB | V100-SXM2-16GB |
| **宿主内存** | **≥ 16 GiB 强烈建议**（本机 15.5 GiB 是 OOM 根因） | ⚠️ 15.5 GiB |
| 系统 | 64-bit Linux | Debian 12 系 |
| NVIDIA 驱动 | 支持 CUDA 12.8 | 580.159.04 |

> ⚠️ **宿主内存是本项目最容易踩的坑**。官方基准机是 32 GiB，本机 15.5 GiB
> 直接导致了 OOM 事故（见 [FINDINGS.md](../FINDINGS.md) 的 A6）。
> **如果你的宿主内存 < 20 GiB，请务必读第 8 节再动手。**

### 1.2 软件

| 项 | 检查命令 | 期望 |
|---|---|---|
| Docker | `docker --version` | 任意近期版本 |
| NVIDIA Container Toolkit | `docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi` | 能打印 GPU 信息 |
| 免密 docker（推荐） | `docker ps` | 无需 sudo |
| GPU 空闲 | `nvidia-smi` | 显存占用接近 0 |

> **显存互斥**：模型权重就占 11.29 GiB，16 GB 卡上**同一时刻只能跑一个服务**。
> 若已有其它推理容器在跑，先 `docker stop` 它。

---

## 2. 获取源码

```bash
# KVMem fork 源码（自行替换为你的来源）
export KVREPO=/home/$USER/kvmem-llama.cpp
git clone <你的 fork 地址> "$KVREPO"
cd "$KVREPO"
```

**确认关键文件存在**：

```bash
ls kvmem/include/kvmem/kvmem_store.hpp    # KVMem 核心实现
ls tools/llama-kvmem-server.cpp           # server 入口
```

---

## 3. 编译

**必须在 CUDA 12.8 的容器里编译**（宿主工具链通常不满足）。

```bash
docker run --rm --gpus all \
  -v "$KVREPO":/src -w /src \
  nvidia/cuda:12.8.1-devel-ubuntu24.04 \
  bash -c '
    apt-get update -qq && apt-get install -y -qq cmake ninja-build git libcurl4-openssl-dev &&
    cmake -S . -B build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DGGML_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES=70 \
      -DGGML_CUDA_FA_ALL_QUANTS=ON &&
    cmake --build build -j4 --target llama-kvmem-server
  '
```

### ⚠️ 三个必须注意的点

| 点 | 说明 |
|---|---|
| **`-DCMAKE_CUDA_ARCHITECTURES=70`** | 只编 `sm_70`。**别漏**——默认架构表在 CUDA 13 起会丢掉 sm_70 |
| **`-DGGML_CUDA_FA_ALL_QUANTS=ON`** | 换 KV 精度（`q5_0` 等）的**前提**；不开会导致 FA 弃权退回 CPU |
| **`-j4`（不是 `-j$(nproc)`）** | nvcc 每进程吃 1–3 GB 内存；宿主 15.5 GiB 上开满并发会 OOM |

> 以上编译选项由本机 `build/CMakeCache.txt` 反推，实测值为：
> `CMAKE_BUILD_TYPE=Release`、`CMAKE_CUDA_ARCHITECTURES=70`、
> `CMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc`、
> `GGML_CUDA=ON`、`GGML_CUDA_FA=ON`、**`GGML_CUDA_FA_ALL_QUANTS=ON`**、`GGML_NATIVE=ON`。

### 验证编译产物

```bash
ls -la "$KVREPO/build/bin/llama-kvmem-server"
# 顺便导出参数表存档（版本间可能有差异）
"$KVREPO/build/bin/llama-kvmem-server" --help > "$KVREPO/help-$(date +%Y%m%d).txt" 2>&1 || true
```

---

## 4. 准备模型

把 GGUF 放到固定路径（**建议放在系统分区，别放 `/vol1` 这类数据盘**）：

```bash
export MODEL=/llama/models/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf
ls -la "$MODEL"
```

**用脚本核对模型几何**（决定 KV 内存开销）：

```bash
python3 scripts/gguf_kv_math.py "$MODEL"
```

预期输出（本机实测的模型）：

```
--- 模型几何 ---
  n_layer=65  n_head=40  n_head_kv=4  head_dim=256  v_dim=256  n_embd=5120
  full_attention_interval=4  → 全注意力层 ≈ 16 层（其余为循环层，KV 与上下文无关）

--- KV 每 token 成本 ---
  每 token 元素数 = 16 层 × 4 kv_head × (256+256) = 32768
  f16   =    64.0 KiB/token
  q8_0  =    34.0 KiB/token
  q5_0  =    22.0 KiB/token
  q4_0  =    18.0 KiB/token
```

> **拿到这张表，你就能提前算出"我这个上下文要多少内存"**，
> 而不是等 OOM 了再猜。

---

## 5. 启动服务

### 方式 A：`docker run`（照抄即可）

```bash
export API_KEY=changeme          # 务必换掉
export KVDTYPE=q5_0              # q8_0 更准但更吃内存；q5_0 是本仓库推荐
export BUDGET=24576              # 有效注意力窗口（token 数）

docker rm -f kvmem-test 2>/dev/null

docker run -d --name kvmem-test \
  --gpus all \
  -p 8095:8080 \
  --restart unless-stopped \
  -v "$KVREPO":/src \
  -v "$MODEL":/models/model.gguf:ro \
  nvidia/cuda:12.8.1-devel-ubuntu24.04 \
  /src/build/bin/llama-kvmem-server -m /models/model.gguf \
  --host 0.0.0.0 --port 8080 \
  -c 131072 -n 16384 \
  --kvmem-budget "$BUDGET" --kvmem-gen-reserve 16384 --kv-dtype "$KVDTYPE" \
  --spec-type draft-mtp --spec-draft-n-max 2 --kvmem-mtp-state replay \
  --api-key "$API_KEY"
```

> 也可以直接用仓库脚本，它会自动留回滚记录：
> ```bash
> bash scripts/run-8095-exact.sh 131072 24576
> ```

### 方式 B：`docker compose`

见 [`deploy/docker-compose.yml`](../deploy/docker-compose.yml)：

```bash
cd deploy
export KVREPO=/home/$USER/kvmem-llama.cpp MODEL=/llama/models/xxx.gguf API_KEY=changeme
docker compose up -d
```

**compose 的好处**：参数集中在 `environment` 段，改参数只需 `docker compose up -d`，
不用记一长串 `docker run`。

---

## 6. 验证（4 个检查点）

### 检查点 1 · 进程起来了

```bash
docker ps --filter name=kvmem-test --format '{{.Status}}'
# 期望: Up 30 seconds
```

### 检查点 2 · 健康检查通过

```bash
# 注意：必须看 HTTP 状态码，加载中会返回 503
curl -s -o /dev/null -w '%{http_code}\n' --noproxy '*' \
  http://127.0.0.1:8095/health
# 期望: 200
```

> ⚠️ `curl -s` 不看状态码会把"503 加载中"误判成"已就绪"。

### 检查点 3 · 参数真的生效了

```bash
curl -s --noproxy '*' http://127.0.0.1:8095/slots \
  -H "Authorization: Bearer $API_KEY" | python3 -m json.tool | head -20
```

**真实响应示例**：

```json
[{
  "id": 0, "is_processing": false,
  "n_ctx": 131072,
  "n_prompt_tokens": 0,
  "n_decoded": 0,
  "params": { "max_tokens": 4096, "temperature": 0.7, "top_p": 0.8, "top_k": 20 },
  "speculative": true
}]
```

| 字段 | 该看什么 |
|---|---|
| `n_ctx` | 是否等于你设的 `-c` |
| `is_processing` | **压测前必须为 `false`**（服务 `-np 1`，会独占） |
| `speculative` | MTP 是否在跑 |

### 检查点 4 · ⭐️ 真的出词了，且 FA 没弃权

```bash
curl -s --noproxy '*' -X POST http://127.0.0.1:8095/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" -H 'Content-Type: application/json' \
  -d '{"model":"model.gguf","messages":[{"role":"user","content":"Reply with the single word: OK"}],"max_tokens":8,"temperature":0}' \
  | python3 -m json.tool | head -30
```

**同时**（关键）检查 FA 是否真的在工作：

```bash
docker logs kvmem-test 2>&1 | grep -i "prompt processing" | tail -3
```

| prefill 速度 | 判定 |
|---|---|
| **400–560 t/s** | ✅ FA 正常工作 |
| **< 80 t/s** | ❌ **FA 已弃权退回 CPU** → 检查 `GGML_CUDA_FA_ALL_QUANTS` 与 K/V 类型 |

> ⚠️ **启动日志不会有任何 warning**。只看"没报错"会误判成 FA 正常。
> **量 prefill 是唯一的验收手段。**

---

## 7. 客户端接入

### 7.1 通用 OpenAI 客户端

```python
from openai import OpenAI

client = OpenAI(base_url="http://<HOST_IP>:8095/v1", api_key="changeme")

resp = client.chat.completions.create(
    model="model.gguf",                      # 默认取 -m 的文件名
    messages=[{"role": "user", "content": "你好"}],
    max_tokens=512,
)
print(resp.choices[0].message.content)
```

### 7.2 Codex / cc-switch

| 配置项 | 值 |
|---|---|
| base_url / endpoint | `http://<HOST_IP>:8095/v1` |
| api_key | 你设的 `--api-key` |
| model | `model.gguf`（= 容器内 `-m` 的文件名） |

> ⚠️ **务必确认配置与实际一致**。本机曾出现 cc-switch 里 endpoint 写 8092、
> 实际流量却打在 8095 的情况——**排查时按日志里的"请求目标"行为准**。

### 7.3 端口与鉴权约定

| 路径 | 是否需要 key |
|---|---|
| `/health` | ❌ 免鉴权 |
| `/slots` | ✅ 要 key |
| `/v1/*` | ✅ 要 key |

---

## 8. 日常运维

### 8.1 改参数

> ⚠️ **`docker restart` 改不了命令行参数**，必须重建：

```bash
bash scripts/run-8095-exact.sh 131072 98304     # 例：把有效窗口提到 98K
```

脚本会自动把旧 Cmd 留档到 `$KVREPO/kvmem-cmd-rollback-<时间戳>.json`。

**回滚**：

```bash
# 从留档里恢复旧参数
cat $KVREPO/kvmem-cmd-rollback-20260928-202704.json
# 然后按它重建
```

### 8.2 监控内存（本项目的命门）

```bash
# 被动采样：24 轮，每 15 秒一次
bash scripts/kvmem-mem-sample.sh "" 24 15
```

| 列 | 该看什么 |
|---|---|
| **`anon_MiB`** | ⭐ **KV 池所在，顶到击杀线就 OOM** |
| `file_MiB` | page cache，**下降是正常的** |
| `swap_MiB` | **持续增长 = 危险信号** |
| `avail_MiB` | 宿主可用内存 |

**击杀线**：容器 anon ≈ **13.6 GiB**（宿主 15.5 GiB 时）。

### 8.3 判断是否发生过 OOM

```bash
# ① 容器重启了几次
docker inspect kvmem-test --format '{{.RestartCount}}'

# ② 每次启动的时刻（启动指纹）
docker logs kvmem-test 2>&1 | grep -c "== CUDA =="

# ③ 系统级 OOM 计数
cat /sys/fs/cgroup/system.slice/memory.events
# ⚠️ 不要看叶子 cgroup（未设 --memory 时它恒为 0）
```

### 8.4 内存不够时的处置优先级

```
① 重启容器清池            ← 最直接（KV 池是高水位语义，涨上去不降）
② --kv-dtype q8_0 → q5_0  ← 每 token 成本 −35% ⭐
③ --spec-kv-dtype q8_0    ← 把 MTP 的 f16 KV 降下来
④ 降 --kvmem-gen-reserve  ← 本机设的 16384 是默认值 256 的 64 倍
⑤ --kvmem-cpu-gb 设上限   ← ⚠️ 未实测，见 [parameters.md](parameters.md)
⑥ 给容器加 --memory 上限   ← 让 OOM 只杀容器，不连带杀宿主其它服务
```

---

## 9. 常见故障速查

| 症状 | 最可能的原因 | 处理 |
|---|---|---|
| 客户端随机 502，服务日志**无报错** | **宿主 OOM 击杀**（静默 SIGKILL） | 见 §8.3、§8.4 |
| 启动即失败，报 `invalid ggml type NNN` | 模型格式是 fork 私有格式（如 PQ2_0 = type 142） | 换对应的 fork 二进制 |
| prefill 只有几十 t/s | **FA 弃权退回 CPU** | 检查 `GGML_CUDA_FA_ALL_QUANTS` + K/V 同类型 |
| `/health` 一直 503 | 模型还在加载（大模型 8–10 s 正常） | 等；用 `docker logs` 看进度 |
| 改完参数没生效 | 只改了脚本文件，**容器没重建** | `docker inspect` 确认实际 Cmd |
| 首请求卡 80–90 s | 无 warmup + 无 prompt cache，重建 KV | 正常现象；调大客户端超时 |
| 容器能起但外部连不上 | `--host` 不是 `0.0.0.0` | 改 `--host 0.0.0.0` |
| 请求返回 401 | `/v1/*` 需要 key | 加 `Authorization: Bearer <key>` |
| 压测时其它客户端全部失败 | `-np 1`，**服务独占** | 压测前确认 `is_processing=false` |

---

## 10. 快速检查清单

**部署前**

- [ ] GPU 空闲（`nvidia-smi` 显存接近 0）
- [ ] **宿主内存 ≥ 16 GiB**（< 20 GiB 要读第 8 节）
- [ ] 模型 GGUF 就位，几何已用 `gguf_kv_math.py` 核对
- [ ] 编译时带了 `CMAKE_CUDA_ARCHITECTURES=70` 与 `GGML_CUDA_FA_ALL_QUANTS=ON`

**部署后**

- [ ] `/health` 返回 **200**（不是 503）
- [ ] `/slots` 的 `n_ctx` 等于设定的 `-c`
- [ ] **prefill 速度 400–560 t/s**（< 80 = FA 弃权）
- [ ] 出词正常，语义合理

**观察期**

- [ ] `RestartCount` 不增长
- [ ] `system.slice/memory.events` 的 `oom_kill` 不增长
- [ ] swap 不持续增长
- [ ] 长会话下 anon 斜率可控

---

## 下一步

- 参数细节与三档推荐配置 → [parameters.md](parameters.md)
- 为什么某些参数会把人带偏 → [pitfalls.md](pitfalls.md)
- 内存/速度的完整实测数据 → [benchmarks.md](benchmarks.md)
