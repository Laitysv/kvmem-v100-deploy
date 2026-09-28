# KVMem on a Single V100

> 在**单张 Tesla V100-SXM2-16GB**、宿主内存仅 **15.5 GiB** 的机器上部署并调优
> **KVMem**（块稀疏 KV 缓存 + 分层存储的 llama.cpp fork）。
>
> 全部数据来自真机实测。**本 README 是自足的** —— 原理、结论、参数、实测数据、关键发现都在这里，
> 不点开任何其它文件也能读完并复现。

**目录**

- [1. KVMem 是什么](#1-kvmem-是什么)
- [2. 三条最值钱的结论](#2-三条最值钱的结论)
- [3. 三个最易搞反的参数](#3-三个最易搞反的参数)
- [4. 快速开始](#4-快速开始)
- [5. 实测数据](#5-实测数据)
- [6. 关键发现与结论](#6-关键发现与结论)（内存 / 排查 / V100 三条）
- [7. 仓库文件导航](#7-仓库文件导航)
- [8. 环境与脱敏](#8-环境与脱敏)
- [9. 适用与不适用](#9-适用与不适用)

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

> 跨维度的**三条**。完整编号清单（A1–E9 / F1–F5，含全部数据表与推理）在 [`FINDINGS.md`](FINDINGS.md)。

### 6.1 内存：只有一个公式，和一个假象

```
宿主 anon ≈ 基线 + 实际 token 数 × 每 token 成本
```

| 要点 | 结论 |
|---|---|
| 每 token 成本 | `q5_0` 实测 **27–29 KiB/token**（理论 22）；两次独立压测斜率一致，模型成立 |
| 宿主击杀线 | **≈13.6 GiB**（宿主只有 15.5 GiB）——官方配方按 32 GiB 调，**差这 4 GiB 就是根因** |
| 真正的杠杆 | **`--kv-dtype`**（34 → 22 KiB/token），**不是** `-c` |
| 假象 | KV 池是**高水位**语义，重启归零。任何"改完参数内存降了"的读数，先怀疑是**重启清池** |

内存不够时的处置顺序 → [`FINDINGS.md`](FINDINGS.md) 的 **A7**。

### 6.2 排查：崩溃时没有日志，才是最有信息量的情况

| 要点 | 结论 |
|---|---|
| 静默消失 | **= 外部信号**（最可能是 OOM SIGKILL）。段错误 / 断言 / CUDA error 都会留痕 |
| 叶子 cgroup 假象 | 没设 `--memory` 时叶子 `oom_kill` **恒为 0**，要查 `system.slice/memory.events` —— 搞错会得出**完全相反**的结论 |
| 三方对表 | `RestartCount` × 日志启动指纹时刻 × 客户端报错时间戳。本次三者逐一对上，因果链闭合 |
| 时间戳陷阱 | 日志行首 `11.49.050.299` 是**已运行时长**，不是墙上时钟 → 一律用 `docker logs --timestamps` |

### 6.3 V100：三条硬约束决定了所有参数选择

| 要点 | 结论 |
|---|---|
| FA 静默弃权 | **没有任何 warning**。唯一验收手段是量 prefill：**400–560 t/s 正常，<80 = 已弃权退回 CPU** |
| 无 INT8 张量核 | IQ 系量化在 Volta 上退回 ALU → 是**算力**瓶颈 → **MTP / 投机解码是负收益** |
| CUDA 红线 | **只用 12.8**（13.x 无 `sm_70`；12.9.x 编 `sm_70` 有 OOM bug）。且 `sm_70` **不在上游 CI 覆盖内**，每次升级必须本地编译验证 |

---

## 7. 仓库文件导航

### 核心

| 文档 | 内容 |
|---|---|
| [`FINDINGS.md`](FINDINGS.md) | ⭐ **完整编号清单**（A1–E9 / F1–F5）：本 README 第 6 节的展开版，含全部数据表与推理过程 |

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

## 8. 环境与脱敏

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

## 9. 适用与不适用

**适用**：在显存/内存受限的卡（尤其 Volta `sm_70`）上部署长上下文推理服务；
排查「服务无报错但客户端反复掉线」这类 OOM 型故障；评估 KV 量化对内存与速度的实际影响。

**不适用**：多卡 / 张量并行；非 llama.cpp 系引擎；云端托管服务的调优。

---

## License

[MIT](LICENSE) —— 文档与脚本均可自由使用、修改、分发。
