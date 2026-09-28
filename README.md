# KVMem on a Single V100

> 在**单张 Tesla V100-SXM2-16GB**、宿主内存仅 **15.5 GiB** 的机器上部署并调优
> **KVMem**（块稀疏 KV 缓存 + 分层存储的 llama.cpp fork）。
>
> 本仓库**不做过程复盘**，只留这次实战中的**关键发现、结论与可复现的配置**。
> 全部数据来自真机实测。

---

## 三条最值钱的结论

| # | 结论 | 依据 |
|---|---|---|
| 1 | **`-c` 不是内存杠杆** | `-c 131072` 空载容器 anon **仅 245 MiB**；宿主占用正比于**实际发送的历史长度**。降 `-c` 只封顶、不减存量 |
| 2 | **真正的杠杆是 `--kv-dtype`** | `q8_0 → q5_0`：每 token 成本 34 → 22 KiB，容器 anon **13.35 → 4.36 GiB**，新增 OOM **归零** |
| 3 | **`--kvmem-budget` 不是显存预算**，而是**有效注意力窗口** | 设 24576 就意味着模型**实际只看得到 2.4 万 token**；`-c` 再大也不改变这一点 |

> 完整清单（含内存模型、排查方法、V100 硬件约束、操作纪律）→ **[`FINDINGS.md`](FINDINGS.md)**

---

## 三个最易搞反的参数

```
-c                   逻辑工作区（客户端能发多长）    默认 2048     便宜，别拿来治 OOM
--kvmem-budget       有效注意力窗口（能看到多少）    默认 131072   调大 = 变慢
--kv-dtype           每 token 成本（真正的杠杆）     默认 q8_0     34K → q5_0 22K
--kvmem-gen-reserve  decode slack                   默认 256      ⚠️ 本机设 16384（64 倍）
```

> ⚠️ `--kvmem-gen-reserve` 的默认值是 **256**，不是 16384 ——
> **16384 是本机特意调大的**。这类"把惯用值当默认值"的错误很难自查。

---

## 快速开始

> 完整版见 [`docs/tutorial.md`](docs/tutorial.md)；参数含义见 [`docs/parameters.md`](docs/parameters.md)。

### ① 编译（CUDA 12.8 容器内，只编 `sm_70`）

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
export API_KEY="$(openssl rand -hex 24)"   # ⚠️ 别用 changeme：脚本会直接拒绝启动

bash scripts/run-8095-exact.sh 131072 24576          # 方式 A：脚本（自动留回滚档）
cd deploy && cp .env.example .env && vi .env && docker compose up -d   # 方式 B：compose
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
> **400–560 t/s = 正常**；**< 80 t/s = FA 已弃权退回 CPU**。启动日志**不会有任何 warning**。

### ④ 日常

```bash
bash scripts/run-8095-exact.sh 131072 98304   # 改参数（自动留回滚档）
bash scripts/kvmem-mem-sample.sh "" 24 15     # 采样内存 / 显存 / 上下文
python3 scripts/gguf_kv_math.py "$MODEL"      # 算这个模型的 KV 每 token 成本
```

> ⚠️ **`docker restart` 改不了命令行参数**，必须 `rm` + `run`（脚本已处理）。

---

## 安全基线

本仓库的脚本与 compose 都按下面这套默认值走。**手抄命令时请照做** ——
它们对应的是四个真实存在的坑，不是洁癖。

| # | 默认值 | 为什么 |
|---|---|---|
| 1 | 端口只绑 **`127.0.0.1`**（`-p 127.0.0.1:8095:8080`） | 裸写 `-p 8095:8080` 等于绑 `0.0.0.0`，服务对**整个局域网**开放。而 **`/health` 是免鉴权的**，暴露出去连存活信息都公开。要开局域网得显式设 `BIND_IP=0.0.0.0`（会打警告） |
| 2 | **拒绝占位密钥** | `run-8095-exact.sh` 在 `API_KEY` 为空或 `changeme` 时**直接退出**（要临时放行得显式 `ALLOW_INSECURE_KEY=1`）；compose 用 `${API_KEY:?}` 硬失败；`.env.example` 的 `API_KEY` 留空 |
| 3 | **打印容器 Cmd 时遮蔽 `--api-key`** | 密钥是命令行参数，会进 `docker inspect`。脚本校验段已遮蔽，避免泄漏到终端回滚 / CI 日志 / 截图。**更彻底的做法是改用 `--api-key-file`**（密钥不进 `docker inspect`），见 [`docs/parameters.md`](docs/parameters.md) |
| 4 | 压测脚本需 **`CONFIRM_OOM_RISK=1`** 才跑，中止线默认 **8000 MiB** | `kvmem-cap-probe.sh` 的设计意图就是**把宿主内存推到接近 OOM 击杀线** —— 在共享宿主上跑会**连带杀掉其它服务**（正是 FINDINGS 里那条故障的成因）。宿主 15.5 GiB 时击杀线约 13.6 GiB，8000 留出 ~5.6 GiB 余量 |

> 另外：容器**没有挂载 `docker.sock`**，模型以 `:ro` 只读挂载 —— 这两点请保持不变。

---

## 目录

### 核心

| 文档 | 内容 |
|---|---|
| [`FINDINGS.md`](FINDINGS.md) | ⭐ **关键发现与结论**：内存模型、排查方法、V100 硬件约束、操作纪律、**安全基线**、一页速查 |

### 参考

| 文档 | 内容 |
|---|---|
| [`docs/parameters.md`](docs/parameters.md) | **参数配置手册**：逐参数含义/取值/默认/推荐/陷阱 + 速查表 + 三档推荐配置 + 调参决策树 |
| [`docs/tutorial.md`](docs/tutorial.md) | **端到端部署教程**：环境 → 编译 → 模型 → 启动 → 4 个验证检查点 → 客户端接入 → 运维 → 故障速查 |
| [`docs/benchmarks.md`](docs/benchmarks.md) | 完整实测数据：速度表、A/B 复测、官方基准对照、容量推算 |
| [`docs/pitfalls.md`](docs/pitfalls.md) | 踩坑清单（15 条）+ 排查方法学 + 改动前后检查清单 |
| [`docs/v100-hardware.md`](docs/v100-hardware.md) | V100（`sm_70`）硬坑登记 + 同卡多服务共存与切换 |

### 脚本

| 脚本 | 用途 |
|---|---|
| [`scripts/run-8095-exact.sh`](scripts/run-8095-exact.sh) | **精确复刻**容器（按 `docker inspect` 反推，支持 `[context] [budget]`，自动留回滚记录；默认只绑 `127.0.0.1`、拒绝占位密钥、遮蔽 key 输出） |
| [`scripts/kvmem-cap-probe.sh`](scripts/kvmem-cap-probe.sh) | 长上下文容量压测：单次长 prefill + 每 5s 采样，带 anon 上限自动中止（☠️ 需 `CONFIRM_OOM_RISK=1`） |
| [`scripts/kvmem-mem-sample.sh`](scripts/kvmem-mem-sample.sh) | 纯被动采样（不打扰服务）：内存 / 显存 / 上下文长度 |
| [`scripts/gguf_kv_math.py`](scripts/gguf_kv_math.py) | 解析 GGUF 头，推算 KV **每 token 成本**与各上下文长度下的总量 |

### 部署

| 文件 | 用途 |
|---|---|
| [`deploy/docker-compose.yml`](deploy/docker-compose.yml) | compose 版配置（参数集中在 `.env`，改参数只需 `docker compose up -d`） |
| [`deploy/.env.example`](deploy/.env.example) | 参数模板，含每个参数的含义与推荐值注释 |

---

## 环境与脱敏

| 项 | 值 |
|---|---|
| GPU | Tesla **V100-SXM2-16GB**（`sm_70` / Volta） |
| **宿主内存** | **15.5 GiB** ← 本次的命门 |
| 容器基础镜像 | `nvidia/cuda:12.8.1-devel-ubuntu24.04` |
| 模型 | `Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf`（IQ3_S，866 张量，11.29 GiB） |
| 服务 | 宿主 `:8095` → 容器 `:8080`，`llama-kvmem-server` |

> **脱敏说明**：主机 IP 记为 `<HOST_IP>`，主机名记为 `gpu-host`，用户名记为 `<USER>`，
> API Key 记为 `<API_KEY>`。脚本改用环境变量读取。**所有性能与内存数字未做任何改动。**

---

## 适用与不适用

**适用**：在显存/内存受限的卡（尤其 Volta `sm_70`）上部署长上下文推理服务；
排查「服务无报错但客户端反复掉线」这类 OOM 型故障；评估 KV 量化对内存与速度的实际影响。

**不适用**：多卡 / 张量并行；非 llama.cpp 系引擎；云端托管服务的调优。

---

## License

[MIT](LICENSE) —— 文档与脚本均可自由使用、修改、分发。
