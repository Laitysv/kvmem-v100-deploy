# KVMem on a Single V100

> 在**单张 Tesla V100-SXM2-16GB**、宿主内存仅 **15.5 GiB** 的机器上部署并调优
> **KVMem**（块稀疏 KV 缓存 + 分层存储的 llama.cpp fork）。
>
> 本文记录一次完整的真实运维事故：从「客户端反复掉线」一路挖到「宿主 OOM 击杀进程」，
> 再到 KV 精度调优、128K 工作区恢复与全量性能实测。
>
> **全部数据来自真机实测**，所有命令与脚本均可复现。

**仓库分三块**：

| 想干什么 | 去哪 |
|---|---|
| 🚀 **跑起来** | [快速开始](#快速开始) · [`docs/08-tutorial.md`](docs/08-tutorial.md) |
| ⚙️ **知道每个参数该设多少** | [`docs/07-parameters.md`](docs/07-parameters.md) |
| 🔍 **搞清楚为什么** | [`docs/02-incident-report.md`](docs/02-incident-report.md) · [`docs/01-architecture.md`](docs/01-architecture.md) |

---

## 一句话结论

| 问题 | 根因 | 处置 | 效果 |
|---|---|---|---|
| Codex 每小时掉线 2~25 次（502） | 15.5 GiB 宿主内存被 128K 上下文的**宿主侧 KV 池**挤爆 → OOM killer SIGKILL → 容器自动重启（8~13s 空窗）→ 代理 5s 超时 | `--kv-dtype q8_0` → **`q5_0`**（每 token 宿主成本 34 KiB → 22 KiB） | 容器 anon **13.35 → 4.36 GiB**；宿主 available **609 MiB → 8.9 GiB**；新增 OOM **0** |

**关键反直觉点（本次最值钱的三条）**：

1. **`-c`（上下文长度）不是 OOM 的杠杆**。实测 `-c 131072` 空载容器 anon 仅 **245 MiB**；
   宿主占用正比于**客户端实际发送的历史长度**，不是 `-c` 的预分配。降 `-c` 只封顶、不减存量。
2. **真正的杠杆是 `--kv-dtype`**。KV 精度直接线性缩放宿主侧 KV 池大小，改一行参数即可。
3. **`--kvmem-budget` 不是显存预算，而是"有效注意力窗口"**。它是检索允许留在 GPU 的历史 token 数，
   默认 131072。设为 24576 就意味着**模型实际只看得到 2.4 万 token 历史**——`-c` 再大也不改变这一点。

---

## 环境

| 项 | 值 |
|---|---|
| 主机 | `gpu-host`（NEC 工作站），Debian 12，内核 6.18 |
| GPU | Tesla **V100-SXM2-16GB**（`sm_70` / Volta），驱动 580.159.04 |
| **宿主内存** | **15.5 GiB** ← 本次事故的主角 |
| 容器基础镜像 | `nvidia/cuda:12.8.1-devel-ubuntu24.04` |
| 模型 | `Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf`（IQ3_S，866 张量，11.29 GiB） |
| 服务 | 宿主 `:8095` → 容器 `:8080`，`llama-kvmem-server` |
| 对照基准机 | 官方文档基准机宿主 **32 GiB**（WSL2 可见 19.53 GiB）→ **差 4 GiB 是根因** |

> **脱敏说明**：本仓库为公开文档，已做统一脱敏——主机 IP 记为 `<HOST_IP>`，主机名记为 `gpu-host`，
> 系统用户名记为 `<USER>`，API Key 记为 `<API_KEY>`。脚本均改用环境变量读取这些值。
> 文中所有性能与内存数字**未做任何改动**，为真机原始实测值。

---

## 快速开始

> 完整版（编译细节、客户端接入、故障速查）见 [`docs/08-tutorial.md`](docs/08-tutorial.md)。
> 参数含义与推荐值见 [`docs/07-parameters.md`](docs/07-parameters.md)。

### ① 编译（在 CUDA 12.8 容器内，只编 `sm_70`）

```bash
export KVREPO=/home/$USER/kvmem-llama.cpp
export MODEL=/llama/models/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf

docker run --rm --gpus all -v "$KVREPO":/src -w /src \
  nvidia/cuda:12.8.1-devel-ubuntu24.04 \
  bash -c 'apt-get update -qq && apt-get install -y -qq cmake ninja-build git libcurl4-openssl-dev &&
    cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES=70 -DGGML_CUDA_FA_ALL_QUANTS=ON &&
    cmake --build build -j4 --target llama-kvmem-server'
```

⚠️ **三个必须**：`-DCMAKE_CUDA_ARCHITECTURES=70`、`-DGGML_CUDA_FA_ALL_QUANTS=ON`、
以及 `-j4`（**不要** `-j$(nproc)`——nvcc 每进程吃 1–3 GB 内存，宿主小时会 OOM）。

### ② 启动

```bash
export API_KEY=changeme        # ⚠️ 务必换掉

# 方式 A：脚本（会自动留回滚档）
bash scripts/run-8095-exact.sh 131072 24576

# 方式 B：compose（参数集中在 .env，推荐长期使用）
cd deploy && cp .env.example .env && vi .env && docker compose up -d
```

### ③ 验证（4 个检查点）

```bash
# 1) 进程起来了
docker ps --filter name=kvmem-test --format '{{.Status}}'

# 2) 健康检查 —— 必须看 HTTP 状态码（加载中是 503）
curl -s -o /dev/null -w '%{http_code}\n' --noproxy '*' http://127.0.0.1:8095/health

# 3) 参数真的生效了
curl -s --noproxy '*' http://127.0.0.1:8095/slots -H "Authorization: Bearer $API_KEY" \
  | python3 -c 'import json,sys; s=json.load(sys.stdin)[0]; print("n_ctx =", s["n_ctx"], "| busy =", s["is_processing"])'

# 4) 出词 + ⭐ FA 没有弃权
curl -s --noproxy '*' -X POST http://127.0.0.1:8095/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" -H 'Content-Type: application/json' \
  -d '{"model":"model.gguf","messages":[{"role":"user","content":"Reply with the single word: OK"}],"max_tokens":8,"temperature":0}'
docker logs kvmem-test 2>&1 | grep -i "prompt processing" | tail -3
```

> ⭐ 第 4 步的 `prompt processing` 速度是**唯一**能确认 FlashAttention 是否工作的手段：
> **400–560 t/s = 正常**；**< 80 t/s = FA 已弃权退回 CPU**，必须换回原 KV 精度。
> 启动日志**不会有任何 warning**。

### ④ 日常：改参数 / 看内存

```bash
bash scripts/run-8095-exact.sh 131072 98304   # 改参数（自动留回滚档）
bash scripts/kvmem-mem-sample.sh "" 24 15     # 被动采样内存 / 显存 / 上下文
python3 scripts/gguf_kv_math.py "$MODEL"      # 算这个模型的 KV 每 token 成本
```

> ⚠️ **`docker restart` 改不了命令行参数**，必须 `rm` + `run`（脚本已处理）。

---

## 目录

### 文档

| 文档 | 内容 |
|---|---|
| [`docs/01-architecture.md`](docs/01-architecture.md) | KVMem 是什么：块稀疏 KV 注意力、re-RoPE、分层存储；**参数语义逐个澄清**（最易搞反的部分） |
| [`docs/02-incident-report.md`](docs/02-incident-report.md) | 事故报告：完整证据链，如何从 502 反推到 OOM SIGKILL |
| [`docs/03-tuning-log.md`](docs/03-tuning-log.md) | 调优日志：三次改动（`-c` 无效 → `q5_0` 有效 → 恢复 128K）的完整过程与验证方法 |
| [`docs/04-benchmarks.md`](docs/04-benchmarks.md) | 性能与容量实测：速度表、A/B 复测、官方基准对照、128K 可行性推算 |
| [`docs/05-pitfalls-and-methodology.md`](docs/05-pitfalls-and-methodology.md) | 踩坑清单（15 条）+ 可复用的排查方法学 + 检查清单 |
| [`docs/06-v100-multi-service.md`](docs/06-v100-multi-service.md) | V100 部署全景：8089 / 8092 / 8095 三个服务如何在同一张卡上共存与切换 |
| [`docs/07-parameters.md`](docs/07-parameters.md) | ⚙️ **参数配置手册**：逐参数含义/取值/默认/推荐/陷阱 + 一页速查表 + 三档推荐配置 + 调参决策树 |
| [`docs/08-tutorial.md`](docs/08-tutorial.md) | 🚀 **端到端部署教程**：环境检查 → 编译 → 模型 → 启动 → 4 个验证检查点 → 客户端接入 → 运维 → 故障速查 |

### 脚本

| 脚本 | 用途 |
|---|---|
| [`scripts/run-8095-exact.sh`](scripts/run-8095-exact.sh) | **精确复刻**当前运行的容器（按 `docker inspect` 反推，支持 `[context] [budget]` 两个参数，自动留回滚记录） |
| [`scripts/kvmem-cap-probe.sh`](scripts/kvmem-cap-probe.sh) | 长上下文容量曲线压测：单次长 prefill + 每 5s 采样 anon/file/avail/swap/gpu，带 anon 上限自动中止 |
| [`scripts/kvmem-mem-sample.sh`](scripts/kvmem-mem-sample.sh) | 纯被动采样（不打扰服务）：内存 / 显存 / 当前上下文长度 |
| [`scripts/gguf_kv_math.py`](scripts/gguf_kv_math.py) | 解析 GGUF 头，按层数/头数/head_dim 推算 KV 缓存**每 token 成本**与各上下文长度下的总量 |

### 部署

| 文件 | 用途 |
|---|---|
| [`deploy/docker-compose.yml`](deploy/docker-compose.yml) | compose 版部署配置（参数集中在 `.env`，改参数只需 `docker compose up -d`） |
| [`deploy/.env.example`](deploy/.env.example) | 参数模板，含每个参数的含义与推荐值注释 |

---

## 时间线（2026-09-28）

| 时刻 | 事件 | 结论 |
|---|---|---|
| 全天 | Codex 反复掉线，502 按小时分布 04(23) 05(25) 08(5) 09(25) 10(2) 15(1) 16(8) 17(2) 18(2) | 需要定位 |
| 18:0x | 证据链收口：`RestartCount=6`、7 次 `== CUDA ==` 横幅与 502 时间戳**逐一对上** | **定案：宿主 OOM** |
| 18:44 | 改 `-c 131072 → 98304` | **无效**（后经定案证实） |
| 18:52 | 停用并禁用 root 的 `obsidian-vnc.service`，腾出内存 | 释放约 70 MiB RSS |
| 19:15 | `--kv-dtype q8_0 → q5_0`（前置验证 FA 配置 + K/V 同类型） | **OOM 止住** |
| 20:10 | 受控压测 + 官方基准对照 | 建立每 token 成本模型 |
| 20:26 | 恢复 `-c 131072`（128K 工作区），按 `docker inspect` 精确复刻 | 空载 anon 仅 245 MiB，推翻"`-c` 预分配"假设 |
| 20:30 | 同一 prompt（66,686 token）A/B 复测 | 斜率一致，模型验证通过 |

---

## 适用与不适用

**适用**：在显存/内存受限的卡（尤其 Volta `sm_70`）上部署长上下文推理服务；
排查「服务无报错但客户端反复掉线」这类 OOM 型故障；评估 KV 量化对内存与速度的实际影响。

**不适用**：多卡 / 张量并行场景；非 llama.cpp 系引擎；云端托管服务的调优。

---

## License

[MIT](LICENSE) —— 文档与脚本均可自由使用、修改、分发。
