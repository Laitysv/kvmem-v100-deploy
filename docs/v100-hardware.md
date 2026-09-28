# V100 部署全景：一张卡上的三个服务

> 本次事故的机器上并非只跑 KVMem。理解**同卡多服务的共存与互斥关系**，
> 才能解释「为什么内存会紧张」和「为什么改配置要先停别的服务」。

---

## 1. 总览

```
              Tesla V100-SXM2-16GB  (sm_70 / Volta)
              ┌──────────────────────────────────────┐
              │  16 GiB 显存  ← 同一时刻只能装一个模型 │
              └──────────────────────────────────────┘
                 ▲            ▲            ▲
                 │            │            │
        ┌────────┴───┐ ┌──────┴─────┐ ┌────┴───────┐
        │   8089     │ │    8092    │ │    8095    │
        │ Qwen-27B   │ │ Bonsai PQ2 │ │  KVMem     │
        │ Q4_0       │ │            │ │            │
        └────────────┘ └────────────┘ └────────────┘
         生产基线        三元模型      长上下文 fork
         33.8–34 t/s    53.9 t/s      40–44 t/s
              └──────── 三者互斥 ────────┘
```

| 服务 | 端口 | 容器 | 模型 | 引擎 |
|---|---:|---|---|---|
| **8089** | 8089 | `Qwen-27B-Q4_0` | `Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf` | llama.cpp（自编译，CUDA 11.8 / 纯 sm_70） |
| **8092** | 8092 | `Ternary-Bonsai-PQ2_0` | `Ternary-Bonsai-2-27B-PQ2_0.gguf` | PrismML-Eng/llama.cpp fork |
| **8095** | 8095 | `kvmem-test` | 同 8089 | **KVMem**（自研 fork） |

**互斥原因**：模型权重就占 11.29 GiB，16 GiB 显存装不下第二个。

---

## 2. 8089 —— 生产基线

| 项 | 值 |
|---|---|
| 容器 | `Qwen-27B-Q4_0`（镜像 `qwen27b-q4_0:latest`） |
| 端口 | 宿主 **8089** → 容器 8080 |
| 策略 | `restart: unless-stopped` |
| 清单 | `/vol1/1000/<USER>/qwen27b-deploy/docker-compose.yml` |
| 模型 | `Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf`（IQ3_S，866 张量，11.29 GiB） |
| **架构** | **混合架构**：全注意力层只有 **16** 层，其余为 Gated DeltaNet 循环层 |
| 参数 | `-c 102400 -ctk q8_0 -ctv q8_0 -fa on -fitt 256 -t 20 -b 2048 -ub 512 -np 1` |
| **速度基线** | **33.8–34.0 t/s**（2026-09-26 实测稳态） |

### 模型架构：为什么 KV 只按 16 层算

（数字直接来自 `scripts/gguf_kv_math.py` 对 GGUF 头的解析）

| 项 | 值 |
|---|---|
| GGUF `block_count` | **65** |
| 全注意力层 | **16** —— `full_attention_interval = 4` → 层号 3, 7, 11, …, 63 |
| 循环层 | **48** —— Gated DeltaNet（线性注意力，状态 **O(1)**，**与上下文长度无关**） |
| 其余 | **1 层 MTP / nextn 预测头**（模型名里的 `-mtp`） |
| 几何 | `n_head = 40`，`n_head_kv = 4`，`key_length = value_length = 256`，`n_embd = 5120` |

> ⭐ **只有那 16 层有 KV 缓存。** 每 token 元素数
> = `16 层 × 4 kv_head × (256 + 256)` = **32768 元素/token**。
>
> ⚠️ **别按「65 层全有 KV」估**——那样会**高估 4 倍**，容量规划直接错。
> 这 48 层 GDN 也是 KVMem 的块稀疏方案只作用在 16 层上的原因。
>
> 这个数**不要手算**：脚本会读 GGUF 里的 `full_attention_interval`
> 自动取 16（`n_attn = nl // fai`）。验证输出见
> [tutorial.md 第 4 节](tutorial.md)。

### 值得记住的行为特征

| 特征 | 说明 |
|---|---|
| **会话间漂移 ±3%** | 同一天：会话开头 32.92 → 重启后 33.85。**比版本差异还大** |
| **首请求冷开销 80–90 s** | `--no-warmup` + `cache_prompt=false` → 每次重建 100K KV + 状态 |
| 就绪探针 | **必须看 `/health` 的 HTTP 状态码**（加载中是 **503**）；`curl -s` 不看状态码会误判"已就绪" |
| 重启后就绪 | 5–10 s（权重在 page cache） |

> ⚠️ **低于 25 t/s 说明掉层/掉速**，改 KV 或上下文后**必须实测**。

### 版本溯源（一个有用的技巧）

从 zip 编译的产物**自报 `build 0, commit unknown`**，查不到版本号。

**解法**：用**源码包内 entry 的时间戳反推**：

```
llama.cpp-master.zip 内 3891 个文件时间戳统一为 2026-09-02 09:19:54
GitHub 归档把 mtime 写成 HEAD 提交的 committer 时间（US Pacific，UTC-7）
→ 09:19:54 + 7h = 2026-09-02T16:19:54Z
→ 精确命中上游提交 e750b887a  →  ≈ b10761
```

---

## 3. 8092 —— 三元量化模型

| 项 | 值 |
|---|---|
| 容器 | `Ternary-Bonsai-PQ2_0` |
| 端口 | 宿主 **8092** |
| 部署目录 | `/vol1/1000/<USER>/bonsai-pq2-deploy/` |
| 模型 | `Ternary-Bonsai-2-27B-PQ2_0.gguf`（7,206,168,928 B） |
| 参数 | 与 8089 **完全一致** |
| 显存 | 11,038 / 16,384 MiB（比 8089 轻约 4 G） |
| 速度 | 稳态 decode **53.9 t/s**（首读数） |

### 关键点：PQ2_0 是私有格式

> **`PQ2_0` = GGML type 142，PrismML 私有格式。**
> **主线 llama.cpp 一律拒载**（`invalid ggml type 142`）。
> 必须用 **PrismML-Eng/llama.cpp fork**——运行时 Walsh-Hadamard 激活变换内核在那个 fork 里。

| 项 | 值 |
|---|---|
| 二进制 | fork release `prism-b10743-adfffbe` 的 `linux-cuda-12.8-x64` 包 |
| sm_70 支持 | **无 sm_70 cubin，但有 sm_70 PTX** → 驱动 JIT 可用（另有 sm_67 cubin） |
| 镜像 | `bonsai-pq2:latest` = `cuda:12.8.1-devel` + `libcurl4` + 复用的 entrypoint.sh |
| JIT 缓存 | 挂载 `./nvcache`（`CUDA_CACHE_PATH`） |

### 模板补丁（值得记的一个坑）

模型内置的 Jinja 模板会对某些请求**直接 500**：

| 报错 | 次数 | 原因 |
|---|---:|---|
| `System message must be at the beginning` | 40 | 模板强制 system 必须是 `messages[0]` |
| `Unexpected reasoning effort high` | 20 | 模板只支持 `xhigh`(默认)/`medium`/`low` |
| `Unexpected reasoning effort max` | 10 | 同上 |

**客户端改不了 → 服务端修**：

```bash
# 1. 从 /props 提取模型内置模板
# 2. 打两个补丁
#    ① 非首位 system 降级渲染为 user 轮（内容不丢）
#    ② reasoning_effort 的 high/max → xhigh
# 3. 挂载 + compose 指定
command: "--chat-template-file /models/chat-template.jinja"
```

> ⚠️ 两个容易踩的点：
> 1. **模板真变量是 `reasoning_effort`**，entrypoint 传的 `thinking_effort` 是**被忽略的死参数**。
> 2. **改模板后必须重建容器**（不热加载）。

---

## 4. 8095 —— KVMem（本仓库主角）

详见 [FINDINGS.md](../FINDINGS.md) 与 [benchmarks.md](benchmarks.md)。

| 项 | 值 |
|---|---|
| 容器 | `kvmem-test` |
| 端口 | 宿主 **8095** → 容器 8080 |
| 源码/挂载 | `/home/<USER>/kvmem-llama.cpp → /src` |
| 二进制 | `build/bin/llama-kvmem-server` |
| 模型 | 同 8089 |
| 现役参数 | `-c 131072 -n 16384 --kvmem-budget 24576 --kvmem-gen-reserve 16384 --kv-dtype q5_0` |
| 鉴权 | `/health` **免鉴权**；`/v1/*` **要 key** |

---

## 5. 互斥与切换

三个服务**共用一张 V100**，同一时刻只能跑一个。

```bash
# 切到 8095（KVMem）
docker stop Ternary-Bonsai-PQ2_0    # 或 Qwen-27B-Q4_0
# ... 启动 kvmem-test ...

# 切回 8089（生产基线）
docker stop Ternary-Bonsai-PQ2_0 && docker start Qwen-27B-Q4_0

# 切到 8092（三元模型）
docker stop Qwen-27B-Q4_0 && docker start Ternary-Bonsai-PQ2_0
```

> ⚠️ **切换前确认 GPU 空闲**：`nvidia-smi` 看显存是否已释放。
> 容器 `stop` 后显存释放有延迟。

---

## 6. V100（`sm_70`）硬坑登记

这一节是**跨服务通用**的，任何在 Volta 上跑量化推理的人都会撞上。

### 坑 1 · IQ 系量化在 Volta 上没有 INT8 张量核

| 项 | 说明 |
|---|---|
| 现象 | 量化矩阵乘**退回 ALU** |
| 后果 | **是算力瓶颈，不是带宽瓶颈** |
| 推论 | **MTP / 投机解码在 V100 上是负收益**（已实测确认关闭更好） |

### 坑 2 · K / V 类型必须相同

混用会让 **`fattn.cu` 直接弃权 FlashAttention**，整个注意力丢给 CPU 后端。

| 解法 | 说明 |
|---|---|
| `--kv-dtype <t>` | 一次同时设 K 和 V（推荐） |
| `GGML_CUDA_FA_QUANTS` | 上游 2026-09-09 / #28079 起，取代 all-or-nothing 的 `GGML_CUDA_FA_ALL_QUANTS`，可以**只编译需要的 KV 组合** → 终于能做 K `q8_0` + V `q4_0` |

### 坑 3 · CUDA 版本红线

`ggml-cuda/CMakeLists.txt` 的默认架构表把 `50/61/70-virtual` 包在
`if (CUDAToolkit_VERSION VERSION_LESS "13")` 里：

| CUDA 版本 | sm_70 | 后果 |
|---|---|---|
| **13.x** | ❌ 无 | **官方 `cuda-13.4` 包在 V100 上根本起不来** |
| **12.9.x** | ⚠️ 有但有问题 | 编 sm_70 有 **OOM bug**（上游 issue #28416）：同源同码在空 32G V100 上申请权重显存稳定失败 |
| **12.8.x** | ✅ | 安全 |
| **11.8**（现役生产） | ✅ | 安全 |

> **官方预编译包在 V100 上只能用 `cuda-12.8` 那一个资产。**

**判断 CUDA 架构只信 `cuobjdump`**：

```bash
# ✅ 正确
cuobjdump --list-ptx <binary>       # 输出文件名是 sm_70.ptx
cuobjdump --list-elf <binary>       # cubin

# ❌ 会漏报 —— PTX 在 fatbin 里是压缩存储的
strings <binary> | grep compute_70
```

> 实测：`cuda-12.8` 包 = cubin sm_86/89/120a + **PTX sm_50/61/70/75/80/90（含 70，V100 靠 JIT 能跑）**；
> `cuda-13.4` 包 = cubin sm_86/89/120/121 + PTX sm_75/80/90，**无 sm_70**。

### 坑 4 · `sm_70` 不在上游 CI 覆盖内

上游 PR 原话：**"CI does not build sm70, so this was not caught"**。

**后果**：破坏会反复出现。实测 2026-09-21 10:58–16:11 的 master **确实编不过 sm_70**。

> **每次升级必须本地编译验证，不能只看 CI 绿。**

### 坑 5 · `--fit` 的丢层在 server 模式日志里完全看不见

只能靠 **`nvidia-smi` 占用 + 实测 t/s 反推**。

---

## 7. 宿主机侧注意

| 项 | 说明 |
|---|---|
| 系统 | 飞牛私有云 fnOS（Debian 12 系，内核 6.18） |
| Web 面板 | 走 `/usr/trim/nginx` 占 **80 / 443** → **不能随便抢这两个端口** |
| 宿主内存 | **15.5 GiB** ← 本次事故根因 |
| 目录权限 | `/usr/local`、`/opt`、`/llama` **只读**；可写 `/vol1`、`/vol2`、`/fs`、`/home/<USER>` |
| sudo | **需要密码** |

---

## 8. 小结

| 问题 | 答案 |
|---|---|
| 为什么三个服务互斥？ | 模型权重 11.29 GiB，16 GiB 显存装不下第二个 |
| 为什么内存会紧张？ | 宿主仅 15.5 GiB，而 KVMem 的宿主侧 KV 池是 GiB 级 |
| 换 KV 精度前要做什么？ | 验 FA 配置 + K/V 同类型 + 量 prefill 速度 |
| V100 上最大的限制？ | **无 INT8 张量核**（IQ 系退回 ALU）+ **sm_70 不在 CI 覆盖内** |
