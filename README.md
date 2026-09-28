# KVMem on a Single V100

> 在**单张 Tesla V100-SXM2-16GB**、宿主内存仅 **15.5 GiB** 的机器上部署并调优
> **KVMem**（块稀疏 KV 缓存 + 分层存储的 llama.cpp fork）。
>
> 全部数据来自真机实测。**本 README 是自足的** —— 原理、结论、参数、实测数据、故障速查都在这里，
> 不点开任何其它文件也能读完并复现。

**目录**

- [1. KVMem 是什么](#1-kvmem-是什么)
- [2. 三条最值钱的结论](#2-三条最值钱的结论)
- [3. 三个最易搞反的参数](#3-三个最易搞反的参数)
- [4. 快速开始](#4-快速开始)
- [5. 实测数据](#5-实测数据)
- [6. 关键发现与结论](#6-关键发现与结论)（A 内存 / B 性能 / C 排查 / D V100 / E 纪律）
- [7. 安全基线](#7-安全基线)
- [8. 故障速查表](#8-故障速查表)
- [9. 一页速查](#9-一页速查)
- [10. 仓库文件导航](#10-仓库文件导航)
- [11. 环境与脱敏](#11-环境与脱敏)
- [12. 适用与不适用](#12-适用与不适用)

---

## 1. KVMem 是什么

### 1.1 一句话

**KVMem = 让 llama.cpp 在显存/内存装不下整个 KV 缓存时，依然能跑长上下文的改造方案。**

它把「上下文必须整段驻留在显存里」这个前提拆掉，换成两个机制：

| 机制 | 解决什么 |
|---|---|
| **块稀疏 KV 注意力** | 让每一步 decode **不需要**扫过全部历史 KV |
| **分层存储（GPU → CPU RAM → NVMe）** | 让放不下的 KV **有地方可去**，而不是直接 OOM |

对照上游 llama.cpp：上游是「KV 缓存按 `-c` 全量分配、全量驻留、全量参与注意力」。
KVMem 把 KV 变成**可分页、可分级、可按需检索**的资源。

### 1.2 块稀疏 KV 注意力：把上下文切成块，只把一部分搬上 GPU

```
历史 token 流
  │
  ├─ 切成 128-token 的块        ← --kvmem-block-tokens（默认 128）
  │
  ├─ 每步 decode 给每个块打分     ← --kvmem-method（默认 retrieval，按检索分数；
  │                                 也可 recency，只看新旧）
  │      检索 query 来自最近一轮 user 内容
  │      ← --kvmem-query-max-tokens（默认 512）/ --kvmem-query-last（兜底 64）
  │
  ├─ 分数高的块被选进「GPU 窗口」  ← 窗口大小 = --kvmem-budget
  │      窗口外的块留在 CPU / NVMe，本步不参与注意力
  │
  └─ 保底命中：开头和最新内容一定在窗口里
         ← --kvmem-sink-tokens（前缀 / attention sink，默认 0 = 一个块）
         ← --kvmem-recent-tokens（最新后缀，默认 0）
```

**关键推论**：`--kvmem-budget` **不是显存预算（MiB）**，而是
「**允许留在 GPU 上的历史 token 数**」——也就是**模型实际能"看到"多长的上下文**。

设 `--kvmem-budget 24576`，模型就**只看得到 2.4 万 token**；`-c` 设多大都不改变这一点。
（这条是本项目最大的一个认知坑，见 [§3](#3-三个最易搞反的参数)。）

> **代价**：budget 越大 → 每步检索要搬运更多块 → **prefill 变慢**。
> 实测官方首遍 436.7 → 有效 243.2 t/s。

### 1.3 分层存储：GPU → CPU RAM → NVMe

| 层 | 由什么控制 | 默认 | 本机现役 |
|---|---|---|---|
| **GPU** | `--kvmem-budget`（窗口 token 数）+ `--kvmem-gpu-ratio`（槽池占显存比例上限） | 131072 / `0.50` | 24576 / 未设 |
| **CPU RAM** | `--kvmem-cpu-gb GB` | `0`（off） | **未设** |
| **NVMe** | `--kvmem-nvme-gb GB` + `--kvmem-nvme-dir PATH` | `0`（off）/ `/tmp/kvmem_nvme` | **未设** |

被挤出 GPU 窗口的块**不会消失**，而是逐级下沉。需要时再回捞。

> 💡 **`--kvmem-cpu-gb` 是本仓库最值得继续挖的一个参数**：本机 OOM 的直接原因就是
> **宿主 RAM 被 KV 池吃掉**，而它是**显式给 CPU 层设上限**的。
> 如果语义是"上限"而非"预分配"，它可能比换 KV 精度更直接地治 OOM。
> **⚠️ 本仓库未实测过它**，`0 = off` 究竟是"不启用"还是"不限制"也待验证。
> 验证方法：设 `--kvmem-cpu-gb 2`，观察 anon 是否被硬顶在 2 GiB。

### 1.4 这个模型为什么特别：65 层里只有 16 层有 KV

`Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf` 是**混合架构**（数字直接来自 `scripts/gguf_kv_math.py`
解析 GGUF 头的输出）：

| 项 | 值 |
|---|---|
| GGUF `block_count` | **65** |
| 全注意力层 | **16** —— `full_attention_interval = 4` → 层号 3, 7, 11, …, 63 |
| 循环层 | **48** —— Gated DeltaNet（线性注意力，状态 **O(1)**，**与上下文长度无关**） |
| 其余 | **1 层 MTP / nextn 预测头**（模型名里的 `-mtp`） |
| 几何 | `n_head = 40`，`n_head_kv = 4`，`key_length = value_length = 256`，`n_embd = 5120` |

两个直接后果：

1. **只有那 16 层有 KV 缓存**。每 token 元素数 = `16 层 × 4 kv_head × (256 + 256)` = **32768**。
   ⚠️ **别按「65 层全有 KV」估** —— 会**高估 4 倍**，容量规划直接错。
2. 那 48 层 GDN 的状态是 O(1) 的，**KVMem 的块稀疏方案只作用在 16 层上**。

### 1.5 每 token 成本公式

```
宿主 anon ≈ 基线 + 实际 token 数 × 每 token 成本
```

| dtype | 理论每 token | 说明 |
|---|---:|---|
| `f16` | 64 KiB | 基准 |
| `q8_0` | 34 KiB | 默认；官方 IQ3 配方 |
| **`q5_0`** | **22 KiB** | **推荐**；官方 IQ4 配方 |
| `q4_0` | 18 KiB | 更省，精度风险更高 |

实测 27.1 ~ 29.0 KiB/token（`q5_0`），略高于理论，差额来自计算缓冲 / 页表等。

**别手算** —— 用脚本读 GGUF 自动算：

```bash
python3 scripts/gguf_kv_math.py "$MODEL"
```

---

## 2. 三条最值钱的结论

| # | 结论 | 依据 |
|---|---|---|
| 1 | **`-c` 不是内存杠杆** | `-c 131072` 空载容器 anon **仅 245 MiB**；宿主占用正比于**实际发送的历史长度**。降 `-c` 只封顶、不减存量 |
| 2 | **真正的杠杆是 `--kv-dtype`** | `q8_0 → q5_0`：每 token 成本 34 → 22 KiB，容器 anon **13.35 → 4.36 GiB**，新增 OOM **归零** |
| 3 | **`--kvmem-budget` 不是显存预算**，而是**有效注意力窗口** | 设 24576 就意味着模型**实际只看得到 2.4 万 token**；`-c` 再大也不改变这一点 |

---

## 3. 三个最易搞反的参数

```
-c                   逻辑工作区（客户端能发多长）    默认 2048     便宜，别拿来治 OOM
--kvmem-budget       有效注意力窗口（能看到多少）    默认 131072   调大 = 变慢
--kv-dtype           每 token 成本（真正的杠杆）     默认 q8_0     34K → q5_0 22K
--kvmem-gen-reserve  decode slack                   默认 256      ⚠️ 本机设 16384（64 倍）
```

> ⚠️ `--kvmem-gen-reserve` 的默认值是 **256**，不是 16384 ——
> **16384 是本机特意调大的**。这类"把惯用值当默认值"的错误很难自查。
>
> **参数语义只能从 `--help` 或源码读**。本项目最大的方向性错误，
> 就是按字面把 `--kvmem-budget` 理解成"显存预算 MiB"——
> 而源码 `kvmem_store.hpp` 写的是 `semantic window tokens`。

**GPU 侧账**：`池大小 ≈ budget + gen_reserve`。
本机 24576 + 16384 = 实际占 **40960** token 槽位。

---

## 4. 快速开始

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

## 5. 实测数据

> 现役配置：`-c 131072 -n 16384 --kvmem-budget 24576 --kvmem-gen-reserve 16384 --kv-dtype q5_0`

### 5.1 速度

| 场景 | prefill (t/s) | decode (t/s) |
|---|---:|---:|
| 首遍 2K tokens | 574 | — |
| 4.8K tokens | 508 | 36–38 |
| 6.7K tokens | 553–544 | 35.7–37.9 |
| 14.3K tokens | 456–514 | 39.2–41.2 |
| **66.7K（受控压测）** | **424.6** | — |
| **69.9K（真实 Codex 请求）** | 405→411 瞬时 / **337.9 平均** | **40.8–44.4** |
| 官方参照（RTX 5060 Ti，满 256K） | 436.7 首遍 / 243.2 有效 | 29.96 |

**读数**：

- **prefill** 从 2K 的 574 衰减到 66.7K 的 424.6 —— 约 **−26%**。这是 `--kvmem-budget` 的税。
- **decode 36–44 t/s，且不随上下文明显衰减**（块稀疏调度把成本压住了）。
- **V100 的 decode（40–44）明显优于官方参照的 29.96**，prefill 同档。

> ⚠️ **prefill 末尾有"隐形时间"**：上下文超过 `budget + gen_reserve` 后，
> 末尾有数十秒**不计入任何进度行**（68K 请求：进度行 170.9s 走完，`prompt eval time` 报 206.8s）。
> **评估端到端耗时用 `prompt eval time`，别用进度行 t/s 反推。**

### 5.2 内存模型验证：同一 prompt 的 A/B（66,686 token）

这是最干净的对照 —— **同一个 prompt、同一台机器、同一份模型**，只有 `-c` 不同：

| 指标 | `-c 98304`（旧基线） | `-c 131072`（干净基线） |
|---|---:|---:|
| 起始 anon | 6387 MiB | **246 MiB** |
| 峰值 anon | 8153 MiB | **2132 MiB** |
| Δanon | 1766 MiB | 1886 MiB |
| **斜率** | **27.1 KiB/tok** | **29.0 KiB/tok** |
| prefill | 424.6 t/s | **449.4 t/s** |
| file（page cache） | 6423 → 5078（**被驱逐**） | 10228（**不变**） |
| swap | 2312 MiB | 2130 MiB（未动） |

**三条结论**：

1. **斜率两次一致**（27.1 / 29.0 KiB/tok）→ **每 token 成本模型成立**（理论 22，实测略高，同档）。
2. **`-c` 越大 ≠ 越贵**：`-c 131072` 的起始 anon 反而只有 246 MiB。
3. **内存压力会拖慢速度**：旧基线 prefill 慢 5.8%（page cache 被驱逐、swap 在抖动）。
   **内存健康 = 性能健康。**

**池子不随请求缩小**：发一个 **852 token** 的小请求，容器 anon 仍是 **6482 MiB**。
KV 池是**高水位语义** —— 这正是「重启容器清池」成为最直接杠杆的原因。

### 5.3 OOM 前后的 KV 精度对照

| 指标 | `q8_0` | `q5_0` |
|---|---:|---:|
| 容器 anon | 13.35 GiB | **4.36 GiB** |
| 宿主 available | 609 MiB | **8.9 GiB** |
| swap | 4.1 GiB（增长中） | 2.3 GiB（稳定） |
| 新增 OOM | 持续 | **0** |
| prefill | 430–500 t/s | 417–560 t/s |

**改一行参数，每 token 成本降 35%，OOM 归零。**

### 5.4 官方基准对照

官方 fork 的 256K 满填充基准：

| 指标 | IQ3（q8_0 主 KV） | IQ4（q5_0 主 KV） |
|---|---:|---:|
| **运行期 RSS 峰值** | **13,444 MiB**（13.13 GiB） | **11,250 MiB**（10.99 GiB） |
| 整卡显存峰值 | 15,873 MiB | 15,931 MiB |

| 项 | 官方基准机 | 本机 |
|---|---:|---:|
| 宿主内存 | 32 GiB（WSL2 可见 19.53 GiB） | **15.5 GiB** |
| 本机 OOM 击杀线 | — | **≈13.6 GiB** |

> ⚠️ 官方峰值 13.44 GiB **已经越过本机 13.6 GiB 的击杀线** —— **官方 256K 配方在本机根本跑不了。**

### 5.5 容量推算

**宿主侧 —— 128K 能不能撑住**

```
单次 128K：246 MiB + 131072 × 29.0 KiB ≈ 3.9 GiB
余量 = 13.6 − 3.9 ≈ 9.7 GiB            ✅ 安全

但长期会话斜率会涨到 ~90 KiB/tok（约 3 倍）：
128K × 90 KiB ≈ 11.3 GiB
余量 = 13.6 − 11.3 ≈ 2.3 GiB           ⚠️ 仍在线下，但余量收窄
```

**GPU 侧 —— 有效窗口上限**

| 项 | 值 |
|---|---:|
| 模型权重（固定） | 11,560 MiB |
| "其它"（计算缓冲 + MTP + SSM） | ≈ 1,814 MiB |
| **剩余可给 KV** | **≈ 3,010 MiB** |
| q5_0 下 token 上限 | **≈ 140K** |

| 窗口配置 | budget + reserve | 需要显存 | 余量 | 评价 |
|---|---:|---:|---:|---|
| 官方 IQ3 配方 | 36864 + 16384 | 1,144 MiB | ~1.9 GiB | 很宽松 |
| **96K 窗口** | 98304 + 16384 | 2,464 MiB | ~546 MiB | **较稳** |
| **真 128K 窗口** | 114688 + 16384 | 2,816 MiB | **~194 MiB** | ⚠️ **极紧** |

> **区分两个"128K"**：`-c 131072`（逻辑工作区，成本低）≠ **真 128K 有效窗口**（budget 114688，显存只剩 194 MiB）。
> 要扩大有效窗口，**从 96K 档起步更稳妥**。

### 5.6 三档推荐配置

| 档位 | 参数 | 有效窗口 | 适用 |
|---|---|---|---|
| **A · 稳**（本机现役） | `--kvmem-budget 24576 --kv-dtype q5_0 --spec-type none` | 24,576 | 日常 / 长会话 / 内存紧张 |
| **B · 均衡** | `--kvmem-budget 98304 --kv-dtype q5_0` | 98,304 | 需要真正的长上下文理解（显存余 ~546 MiB） |
| **C · 对齐官方** | `--kvmem-budget 36864 --kv-dtype q5_0` | 36,864 | 保守起步，逐步往上试 |

三档都带 `-c 131072 -n 16384 --kvmem-gen-reserve 16384`。

### 5.7 复现方式

```bash
bash scripts/kvmem-mem-sample.sh "" 24 15                      # 被动采样（不打扰服务）
CONFIRM_OOM_RISK=1 bash scripts/kvmem-cap-probe.sh "" 300000 8000   # 长 prefill 压测
python3 scripts/gguf_kv_math.py "$MODEL"                       # GGUF 几何与每 token 成本
```

采样输出示例：

```
TS            anon_MiB  file_MiB shmem_MiB avail_MiB swap_MiB gpu_MiB n_prompt n_decoded
20:13:05          6482     10228         0      9120     2130   13866   131072      1024
```

| 列 | 含义 |
|---|---|
| `anon_MiB` | **KV 池所在，真正要盯的** |
| `file_MiB` | page cache（权重 mmap），**下降是正常的** |
| `avail_MiB` | 宿主可用内存 |
| `swap_MiB` | swap 已用（持续增长 = 危险信号） |
| `gpu_MiB` | 整卡显存占用 |
| `n_prompt` / `n_decoded` | 当前上下文长度 / 已生成数 |

---

## 6. 关键发现与结论

> 标 ⭐️ 的是最值钱、最容易搞反的几条。

### A. 内存与容量（最重要）

#### A1. ⭐️ `-c`（上下文长度）**不是**内存杠杆

| 项 | 值 |
|---|---|
| `-c 131072` 空载容器 anon | **仅 245 MiB** |
| 852-token 小请求后 anon | 仍是 **6482 MiB**（池子不随请求缩小） |

- 宿主占用正比于**客户端实际发送的历史长度**，与 `-c` 的封顶值无关。
- **降 `-c` 只能封顶，不能减少存量。**
- 那条"改完 `-c` 内存降了 0.4 GiB"是**重启清池**的假象，不是 `-c` 的功劳。

**该怎么做**：别用 `-c` 治 OOM。

#### A2. ⭐️ 真正的杠杆是 `--kv-dtype`

每 token 宿主成本（本模型几何：**16** 个全注意力层 × 4 kv_head × 512 —— 65 层里只有这 16 层有 KV，见 [§1.4](#14-这个模型为什么特别65-层里只有-16-层有-kv)）：

| dtype | 每 token | 说明 |
|---|---:|---|
| `f16` | 64 KiB | 基准 |
| `q8_0` | 34 KiB | 默认；官方 IQ3 配方 |
| **`q5_0`** | **22 KiB** | **推荐**；官方 IQ4 配方 |
| `q4_0` | 18 KiB | 更省，精度风险更高 |

实测对照见 [§5.3](#53-oom-前后的-kv-精度对照)。
**该怎么做**：内存不够，先换 `--kv-dtype`。

#### A3. ⭐️ `--kvmem-budget` **不是**显存预算，而是「有效注意力窗口」

| 项 | 值 |
|---|---|
| 真实含义 | 检索允许**留在 GPU 上的历史 token 数** = 模型**实际能看到**的上下文长度 |
| 默认 | 131072（`0` = 等于 `n_ctx`） |
| 本机现役 | **24576** |

**结论**：本机 `--kvmem-budget 24576` 意味着**模型实际只看得到 2.4 万 token 历史**。
`-c 131072` 只是"客户端可以发 12.8 万长度的历史"，两者完全独立。

**代价**：budget 越大 → 每步检索要搬运更多块 → **prefill 变慢**。

#### A4. `--kvmem-gen-reserve` 的默认值只有 **256**

| 项 | 值 |
|---|---|
| 默认 | **256**（`decode slack`） |
| 本机现役 | **16384**（**默认值的 64 倍**） |

> ⚠️ 这类"把惯用值当默认值"的错误极难自查——**默认值只能从 `--help` 或源码读**。

**GPU 侧账**：`池大小 ≈ budget + gen_reserve` → 本机 24576 + 16384 = **40960** 槽位。

#### A5. 每 token 成本公式（可直接用来估内存）

```
宿主 anon ≈ 基线 + 实际 token 数 × 每 token 成本
```

| 来源 | 数值 |
|---|---:|
| 理论（`q5_0`） | 22 KiB/token |
| **实测** | **27.1 ~ 29.0 KiB/token** |

两次独立压测（66,686-token prompt）斜率一致 → **模型成立**。

> ⚠️ **决定风险的是「累积量」而非「单次 prompt 长度」**：
> 长期会话曾在 70K prompt 下累积到 ~6 GiB anon（**约 90 KiB/token，是干净单次请求的 3 倍**）。

#### A6. OOM 击杀线 ≈ **13.6 GiB**（宿主只有 15.5 GiB）

| 项 | 值 |
|---|---|
| 宿主总内存 | **15.5 GiB** |
| **容器 anon 击杀线** | **≈ 13.6 GiB**（9 次 dmesg 的 `anon-rss` 一致） |
| 官方基准机宿主 | **32 GiB**（WSL2 可见 19.53 GiB） |

**结论**：**差 4 GiB 就是根因。** 官方配方按 32 GiB 调，搬到 15.5 GiB 必然爆。
详见 [§5.4](#54-官方基准对照)。

#### A7. 内存不够时的处置优先级

```
① 重启容器清池            ← 最直接（KV 池是高水位语义，涨上去不降）
② --kv-dtype q8_0 → q5_0  ← 每 token 成本 −35% ⭐
③ --spec-kv-dtype q8_0    ← 把 MTP 默认的 f16 KV 降下来
④ 降 --kvmem-gen-reserve  ← 本机 16384 是默认 256 的 64 倍
⑤ --kvmem-cpu-gb 设上限   ← ⚠️ 未验证，见 A8
⑥ 容器加 --memory 上限     ← 让 OOM 只杀容器，不连带杀宿主其它服务
```

#### A8. `--kvmem-cpu-gb` / `--kvmem-nvme-gb`（⚠️ 待验证）

本次才发现的两个参数，**直接对应分层存储的 CPU / NVMe 层**：

| 参数 | 默认 | 含义 |
|---|---|---|
| `--kvmem-cpu-gb GB` | `0`（off） | CPU 溢出区大小 |
| `--kvmem-nvme-gb GB` | `0`（off） | NVMe 溢出文件大小 |
| `--kvmem-nvme-dir PATH` | `/tmp/kvmem_nvme` | NVMe 目录 |
| `--kvmem-gpu-ratio R` | `0.50` | GPU 槽池占显存比例上限 |

**推测**：如果 `--kvmem-cpu-gb` 的语义是"**上限**"而非"预分配"，
它可能比换 KV 精度更直接地治 OOM（OOM 的直接原因就是宿主 RAM 被 KV 池吃掉）。

**⚠️ 未实测，且 `0 = off` 究竟是"不启用"还是"不限制"也不确定。**
**验证方法**：设 `--kvmem-cpu-gb 2`，观察 anon 是否被硬顶在 2 GiB。

### B. 性能

#### B1. 实测速度

完整表见 [§5.1](#51-速度)。要点：

- prefill 从 2K 的 574 衰减到 66.7K 的 424.6（**−26%**）——这是 `--kvmem-budget` 的税。
- **decode 36–44 t/s，且不随上下文明显衰减**。
- **V100 的 decode 优于官方参照的 29.96。**

#### B2. ⚠️ prefill 末尾有「隐形时间」

上下文超过 `budget + gen_reserve` 后，**prefill 末尾有数十秒不计入任何进度行**：

```
68K 请求：进度行 170.9s 走完，prompt eval time 报 206.8s
        → 36s 的差额真实存在，但进度行看不到
```

**结论**：评估端到端耗时**用 `prompt eval time`**，不要用进度行 t/s 反推。

#### B3. 内存压力会拖慢速度

同一 prompt 的 A/B：干净基线 prefill **449.4 t/s**，有内存压力时 **424.6 t/s**（慢 5.8%）。

**结论**：**内存健康 = 性能健康。**

#### B4. GPU 侧有效窗口上限

剩余可给 KV ≈ **3,010 MiB** → q5_0 下上限 **≈ 140K token**。
真 128K 窗口只剩 **~194 MiB** 余量。详见 [§5.5](#55-容量推算)。

#### B5. 128K 可行性

单次 128K 约 **3.9 GiB**（余量 ~9.7 GiB ✅）；
但长期会话按保守斜率（90 KiB/tok）约 **10–11 GiB**（余量收窄到 ~2.3 GiB ⚠️）。

**结论**：**128K 可行，但决定风险的是累积量，长会话要盯着。**

### C. 排查方法（怎么定位这类问题）

#### C1. ⭐️ 崩溃时日志**完全静默** = 外部信号（最可能是 OOM SIGKILL）

正常崩溃（段错误、断言、CUDA error）都会留痕。
**唯一会静默杀死进程的是外部信号。**

| 崩溃前征兆 | 观测 |
|---|---|
| 静默 | 3~100 s 无输出 |
| prefill 崩塌 | 550 t/s → **68 t/s** |

#### C2. ⭐️ 叶子 cgroup 的 `oom_kill` 恒为 0（假象）

```bash
# ❌ 会得出"不是 OOM"的相反结论
cat /sys/fs/cgroup/system.slice/docker-<ID>.scope/memory.events   # oom_kill 0

# ✅ 真相在这里
cat /sys/fs/cgroup/system.slice/memory.events                      # oom_kill 7
```

**原因**：容器没设 `--memory`，内存统计被上推到父 slice 记账。

#### C3. ⭐️ 「三方对表」定位重启型故障

```
容器 RestartCount（重启了几次）
      ×  日志启动指纹的时刻（什么时候重启的）
      ×  客户端报错的时间戳与 latency（用户何时感知到）
```

**三者一一对应 → 因果链闭合。**

本次：`RestartCount=6` × 7 次 `== CUDA ==` 横幅 × 502 时间戳，**逐一对上、无遗漏无多余**。

**关键技巧**：找一个**每次启动都会打印的稳定指纹**（这里是 CUDA 环境横幅），
有了它 `docker logs` 就能当重启时间线用。

#### C4. 重启 1 次 = 客户端掉线 1 次

| 环节 | 数值 |
|---|---|
| 容器重启空窗 | **8~13 s**（端口仍 accept，但没有进程应答） |
| 代理超时 | **5 s**（502 的 `latency_ms` 精确落在 ~5000 ms） |

#### C5. 日志行首的时间戳**不是**墙上时钟

```
11.49.050.299   ← 是「服务启动后经过的时间」，格式 分.秒.毫秒.微秒
```

**正确做法**：一律用 `docker logs --timestamps` 拿绝对时间。

> 搞错这一点，会让"什么时刻发生了什么"整条时间线全错。

#### C6. 盯对内存指标

| 指标 | 行为 | 该不该盯 |
|---|---|---|
| **anon** | 顶到击杀线 → OOM | ✅ **要盯** |
| file（page cache） | **内核优先回收**，下降正常 | ❌ 下降不代表故障 |
| swap | 持续增长 = 危险信号 | ✅ 看趋势 |

### D. V100（`sm_70`）硬件约束

#### D1. ⭐️ 换 KV 精度前必验两条

| # | 前提 | 检查 |
|---|---|---|
| 1 | 编译期 **`GGML_CUDA_FA_ALL_QUANTS=ON`** | `grep GGML_CUDA_FA_ALL_QUANTS build/CMakeCache.txt` |
| 2 | **K / V 必须同类型** | 用 `--kv-dtype` 一次设两个，别分开设 `-ctk` / `-ctv` |

否则 **`fattn.cu` 静默弃权**，整个注意力丢给 CPU 后端。

#### D2. ⭐️ FA 弃权**没有任何 warning**，只能量 prefill

| prefill | 判定 |
|---|---|
| **400–560 t/s** | ✅ FA 正常 |
| **< 80 t/s** | ❌ **已弃权退回 CPU** → 换回原精度 |

**结论**：**量 prefill 是唯一的验收手段**，启动日志"没报错"不代表正常。

#### D3. IQ 系量化在 Volta 上**没有 INT8 张量核**

- 量化矩阵乘**退回 ALU** → **是算力瓶颈，不是带宽瓶颈**。
- **推论：MTP / 投机解码在 V100 上是负收益。**

#### D4. CUDA 版本红线

| CUDA | sm_70 | 后果 |
|---|---|---|
| **13.x** | ❌ | **官方 `cuda-13.4` 包在 V100 上根本起不来** |
| **12.9.x** | ⚠️ | 编 sm_70 有 **OOM bug**（上游 issue #28416） |
| **12.8.x** | ✅ | 安全 |
| **11.8** | ✅ | 现役生产用它 |

**结论**：**官方预编译包在 V100 上只能用 `cuda-12.8` 那一个资产。**

**判断 CUDA 架构只信 `cuobjdump`**：

```bash
cuobjdump --list-ptx <binary>          # ✅ 输出 sm_70.ptx
strings <binary> | grep compute_70     # ❌ 漏报（PTX 在 fatbin 里是压缩的）
```

#### D5. `sm_70` 不在上游 CI 覆盖内

上游 PR 原话：**"CI does not build sm70, so this was not caught"**。

**结论**：**每次升级必须本地编译验证，不能只看 CI 绿。**

#### D6. `--fit` 的丢层在 server 模式日志里完全看不见

只能靠 `nvidia-smi` 占用 + 实测 t/s 反推。

### E. 操作纪律

| # | 结论 |
|---|---|
| E1 | **`docker restart` 改不了命令行参数**——必须 `docker rm -f` + `docker run` |
| E2 | **启动脚本 ≠ 实际运行的容器**（本次差 5 处）——永远以 `docker inspect` 为准 |
| E3 | **「改了脚本文件」≠「改了运行中的服务」**——改完必须验证容器里的真实参数 |
| E4 | **改动前必须留回滚档**（旧 Cmd + 脚本备份）——远程改生产服务的最低要求 |
| E5 | **压测会独占服务**（`-np 1`）——跑之前先确认 `is_processing=false` |
| E6 | **配置与实际可能不一致**——以日志里的"请求目标"行为准 |
| E7 | **要区分「看起来占资源」和「实际占资源」**（某 headless GUI 服务一堆进程，RSS 合计仅 70 MiB；KV 池一个数字，GiB 级） |
| E8 | **要区分「理论值」和「实测值」**——理论定方向，实测做决策，且说明差异来源 |
| E9 | **参数语义回到源码/`--help` 确认**——字面联想 + 经验迁移最容易翻车 |

**改动的三段检查清单**

| 阶段 | 检查项 |
|---|---|
| **改动前** | `docker inspect` 拿真实 Cmd/Binds/端口 · 留档旧 Cmd 与脚本备份 · 确认服务空闲 · 确认 V100 前提（FA 配置 / K-V 同类型） |
| **改动后** | `docker inspect` 确认参数生效 · `/slots` 确认 `n_ctx` · **量 prefill（400–560 = FA 正常）** · 采样 anon/avail/swap · 记录 `RestartCount` 基线 |
| **观察期** | `RestartCount` 是否增长 · `system.slice/memory.events` 的 `oom_kill` 是否增长 · swap 是否持续增长 · 长会话累积斜率是否失控 |

---

## 7. 安全基线

> 这一节回答的是**另一个问题**：「照着这个仓库做，会不会把自己搞出事」。
> 它与「仓库里有没有敏感信息」是两件事 —— 后者是扫字面量，前者是审默认值。

本仓库的脚本与 compose 都按下面这套默认值走。**手抄命令时请照做** ——
它们对应的是四个真实存在的坑，不是洁癖。

| # | 默认值 | 为什么 |
|---|---|---|
| F1 | 端口只绑 **`127.0.0.1`**（`-p 127.0.0.1:8095:8080`） | 裸写 `-p 8095:8080` 等于绑 `0.0.0.0`，服务对**整个局域网**开放。而 **`/health` 是免鉴权的**，暴露出去连存活信息都公开。要开局域网得显式设 `BIND_IP=0.0.0.0`（会打警告） |
| F2 | **拒绝占位密钥** | `run-8095-exact.sh` 在 `API_KEY` 为空或 `changeme` 时**直接退出**（要临时放行得显式 `ALLOW_INSECURE_KEY=1`）；compose 用 `${API_KEY:?}` 硬失败；`.env.example` 的 `API_KEY` 留空 |
| F3 | **打印容器 Cmd 时遮蔽 `--api-key`** | 密钥是命令行参数，会进 `docker inspect`。脚本校验段已遮蔽，避免泄漏到终端回滚 / CI 日志 / 截图。**更彻底的做法是改用 `--api-key-file`**（密钥不进 `docker inspect`，一行一个 key，空行与 `#` 开头忽略） |
| F4 | 压测脚本需 **`CONFIRM_OOM_RISK=1`** 才跑，中止线默认 **8000 MiB** | `kvmem-cap-probe.sh` 的设计意图就是**把宿主内存推到接近 OOM 击杀线** —— 在共享宿主上跑会**连带杀掉其它服务**（正是 C 节那条故障的成因）。宿主 15.5 GiB 时击杀线约 13.6 GiB，8000 留出 ~5.6 GiB 余量 |
| F5 | 容器**不挂 `docker.sock`**、模型 `:ro` 只读挂载 | 保持这样。挂 `docker.sock` ≈ 给容器一份等于 root 的宿主控制权 |

**两条通用教训**：

- **「隐私干净」≠「部署安全」**：前者扫字面量，后者审默认值。只做前者，会漏掉
  「默认密钥 + 全 `0.0.0.0` 绑定」这类**组合**风险 —— 单看每一项都不像漏洞。
- **安全默认值要「硬失败」，不要「只警告」**。警告会被 `2>/dev/null` 吃掉、
  被刷屏淹没、被"先跑起来再说"忽略。

---

## 8. 故障速查表

| 症状 | 最可能的原因 | 处理 |
|---|---|---|
| 客户端随机 502，服务日志**无报错** | **宿主 OOM 击杀**（静默 SIGKILL） | 查 `system.slice/memory.events`（**别查叶子 cgroup**）；按 [A7](#a7-内存不够时的处置优先级) 顺序处置 |
| 服务无报错却"消失"、随后自动重启 | 同上（外部信号） | 静默 = 外部信号；查 `dmesg \| grep -i "killed process"` |
| 重启 1 次就掉线 1 次，`latency_ms`≈5000 | 容器重启空窗 8–13 s > 代理超时 5 s | 给容器加 `--memory` 上限，让 OOM 只杀容器；或调大代理超时 |
| 启动即失败，报 `invalid ggml type NNN` | 模型是 fork 私有格式（如 PQ2_0 = type 142） | 换对应的 fork 二进制 |
| prefill 只有几十 t/s | **FA 弃权退回 CPU** | 查 `GGML_CUDA_FA_ALL_QUANTS` + K/V 同类型；量 prefill 验收 |
| 改完 KV 精度后**速度暴跌** | 同上，FA 静默弃权（无任何 warning） | 换回原精度，先修编译/参数前提 |
| 端到端耗时比进度行算出来的长几十秒 | prefill 末尾「隐形时间」 | 用 `prompt eval time`，别用进度行 t/s 反推 |
| 内存"改了参数却降不下来" | KV 池是**高水位**语义，或只是重启清池 | 重启清池最直接；要真降成本就换 `--kv-dtype` |
| 降了 `-c` 但内存没降 | **`-c` 不是内存杠杆** | 别动 `-c`；换 `--kv-dtype` 或重启 |
| 以为 `--kvmem-budget` 是显存 MiB | 语义是**有效窗口 token 数** | 回源码/`--help` 确认；模型实际只看得到这么多 |
| `/health` 一直 503 | 模型还在加载（大模型 8–10 s 正常） | 等；`docker logs` 看进度 |
| 改完参数没生效 | 只改了脚本文件，**容器没重建** | `docker inspect` 确认实际 Cmd |
| 首请求卡 80–90 s | 无 warmup + 无 prompt cache，重建 KV | 正常现象；调大客户端超时 |
| 容器能起但外部连不上 | `--host` 不是 `0.0.0.0` | 改 `--host 0.0.0.0`（但宿主侧仍建议绑 `127.0.0.1`） |
| 请求返回 401 | `/v1/*` 需要 key | 加 `Authorization: Bearer <key>` |
| 压测时其它客户端全部失败 | `-np 1`，**服务独占** | 压测前确认 `is_processing=false` |
| `curl` 连不上本机服务 | 本地代理污染 | 加 `--noproxy '*'` |
| 时间线怎么也对不上 | 日志行首是**已运行时长**，不是墙上时钟 | 用 `docker logs --timestamps` |
| 显存够但速度掉层 | `--fit` 丢层，server 日志看不见 | `nvidia-smi` 占用 + 实测 t/s 反推 |

---

## 9. 一页速查

```
【内存】
  宿主 anon ≈ 基线 + token 数 × 每 token 成本
  q5_0 实测 27~29 KiB/token（理论 22）
  击杀线 ≈ 13.6 GiB（宿主 15.5 GiB）

【三个最易搞反的参数】
  -c                逻辑工作区（客户端能发多长）   默认 2048   便宜，别拿来治 OOM
  --kvmem-budget    有效注意力窗口（能看到多少）   默认 131072 调大 = 变慢
  --kv-dtype        每 token 成本（真正的杠杆）    默认 q8_0   34K → q5_0 22K
  --kvmem-gen-reserve  decode slack               默认 256    本机设 16384（64 倍）

【模型】
  65 层 = 16 全注意力 + 48 GDN + 1 MTP
  只有 16 层有 KV → 32768 元素/token（别按 65 层估，会高估 4 倍）

【排查】
  崩溃无日志 → 查 OOM（system.slice/memory.events，别查叶子 cgroup）
  三方对表：RestartCount × 启动指纹时刻 × 客户端报错时间戳

【V100】
  换 KV 精度前：FA_ALL_QUANTS=ON + K/V 同类型，然后量 prefill（400-560 正常，<80 弃权）
  只用 CUDA 12.8；IQ 系无 INT8 张量核 → MTP 负收益；sm_70 不在 CI

【操作】
  docker restart 改不了参数 → 必须 rm + run
  以 docker inspect 为准，不信启动脚本
  改动前留回滚档

【安全】
  端口绑 127.0.0.1（/health 免鉴权）；拒绝 changeme；Cmd 打印遮蔽 key
  压测需 CONFIRM_OOM_RISK=1
```

---

## 10. 仓库文件导航

### 核心

| 文档 | 内容 |
|---|---|
| [`FINDINGS.md`](FINDINGS.md) | 本 README 第 6/7/9 节的**编号清单版**（A1–E9 / F1–F5），供交叉引用 |

### 参考

| 文档 | 内容 |
|---|---|
| [`docs/parameters.md`](docs/parameters.md) | **参数配置手册**：逐参数含义/取值/默认/推荐/陷阱 + 速查表 + 三档推荐配置 + 调参决策树 |
| [`docs/tutorial.md`](docs/tutorial.md) | **端到端部署教程**：环境 → 编译 → 模型 → 启动 → 4 个验证检查点 → 客户端接入 → 运维 → 故障速查 |
| [`docs/benchmarks.md`](docs/benchmarks.md) | 完整实测数据（本 README 第 5 节的展开版） |
| [`docs/pitfalls.md`](docs/pitfalls.md) | 踩坑清单（15 条，含「现象 → 误判 → 真相 → 正确做法」）+ 排查方法学 |
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

## 11. 环境与脱敏

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

## 12. 适用与不适用

**适用**：在显存/内存受限的卡（尤其 Volta `sm_70`）上部署长上下文推理服务；
排查「服务无报错但客户端反复掉线」这类 OOM 型故障；评估 KV 量化对内存与速度的实际影响。

**不适用**：多卡 / 张量并行；非 llama.cpp 系引擎；云端托管服务的调优。

---

## License

[MIT](LICENSE) —— 文档与脚本均可自由使用、修改、分发。
