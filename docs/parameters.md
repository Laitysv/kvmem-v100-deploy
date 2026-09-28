# 参数配置手册

> 本篇是**可直接照抄的参数配置参考**。
> 所有参数名、取值域、默认值均来自 `llama-kvmem-server --help`（真机导出）与源码；
> 标注 `📌 实测` 的是本机验证过的行为，标注 `⚠️` 的是**会把人带偏的陷阱**。

```bash
# 随时导出你手上这一版的权威参数表（版本间可能有差异）
docker exec kvmem-test /src/build/bin/llama-kvmem-server --help
```

---

## 1. 一页速查表

### 决定「内存」的参数（最关心）

| 参数 | 作用 | 默认 | 本机现役 | 调它会发生什么 |
|---|---|---|---|---|
| **`--kv-dtype`** | KV 缓存精度（同时设 K 和 V） | `q8_0` | **`q5_0`** | ⭐ **每 token 宿主成本 34 → 22 KiB（−35%）** |
| `--kvmem-cpu-gb` | CPU 溢出区大小（GiB） | `0`（off） | 未设 | 显式给 CPU 层设上限 |
| `--kvmem-nvme-gb` | NVMe 溢出文件大小（GiB） | `0`（off） | 未设 | 启用第三层存储 |
| `--kvmem-gpu-ratio` | GPU 槽池占显存比例上限 | `0.50` | 未设 | 限制 GPU 侧池 |
| `--spec-kv-dtype` | **MTP draft 的** KV 精度 | **`f16`** | 未设 | 容易被忽略的一块显存 |
| `-c` | 逻辑工作区 | `2048` | `131072` | ⚠️ **很便宜，别拿来治 OOM** |

### 决定「有效上下文」的参数

| 参数 | 作用 | 默认 | 本机现役 |
|---|---|---|---|
| **`--kvmem-budget`** | **GPU 工作集 token 数 = 有效注意力窗口** | `131072` | **`24576`** |
| `--kvmem-gen-reserve` | decode slack（留给新生成 token） | **`256`** | **`16384`** |
| `--kvmem-block-tokens` | 块大小 | `128` | 未设（=128） |
| `--kvmem-sink-tokens` | 永远保留的前缀 | `0`（= 一个块） | 未设 |
| `--kvmem-recent-tokens` | 永远保留的最新后缀 | `0` | 未设 |
| `--kvmem-method` | 选块方法 | `retrieval` | 未设（=retrieval） |

### 决定「速度」的参数

| 参数 | 作用 | 默认 | 本机现役 |
|---|---|---|---|
| `-fa` | FlashAttention | `auto` | 未设（但编译期已开 FA_ALL_QUANTS） |
| `-b` / `-ub` | 逻辑 / 物理 batch | `512` / `= -b` | 未设 |
| `-ngl` | 卸载到 GPU 的层数 | `99` | 未设（=99，全卸） |
| `--spec-type` | 投机解码 | `none` | `draft-mtp` ⚠️ **V100 上是负收益** |

---

## 2. KVMem 核心参数（详细）

### 2.1 `--kvmem-budget N` —— 有效注意力窗口 ⭐️

```
--kvmem-budget N           GPU working-set tokens; 0 = n_ctx
```

| 项 | 值 |
|---|---|
| 默认 | **131072**（源码 `kvmem_store.hpp:233` `select_budget = 131072`） |
| 特殊值 | `0` = 等于 `n_ctx`（即不做限制） |
| 语义 | 检索允许**留在 GPU 上的历史 token 数** = **模型实际能"看到"的上下文长度** |

**⚠️ 名字极具误导性**：它**不是**"显存预算（MiB）"。

> 源码注释原文：`// --kvmem-budget (semantic window tokens)`
> 仓库 README 原文：*"How many historical tokens retrieval may keep on GPU"*

**后果**：本机设 `24576` → **模型实际只看得到 2.4 万 token 历史**。
`-c 131072` 只是"客户端可以发 12.8 万长度的历史"，两者完全独立。

**代价**：budget 越大 → 每步检索要搬运更多块 → **prefill 变慢**。
（官方基准：首遍 436.7 t/s → 有效 243.2 t/s，就是这个税。）

### 2.2 `--kvmem-gen-reserve N` —— decode slack ⭐️

```
--kvmem-gen-reserve N      decode slack (default 256)
```

| 项 | 值 |
|---|---|
| 默认 | **256** |
| 本机现役 | **16384**（是默认值的 **64 倍**） |
| 语义 | 为**本轮新生成**的 token 预留的 GPU 槽位；单轮生成不能超过它 |

> ⚠️ **修正一个常见误解**：默认值只有 **256**，不是 16384。
> 16384 是**本机特意调大的**，代价是占掉 GPU KV 池的一大块。

**它和 `--kvmem-budget` 的关系**（源码 `kvmem_store.hpp:379`）：

```
GPU 侧 KV 池大小 ≈ budget + gen_reserve
```

所以：`--kvmem-gen-reserve 16384` + `--kvmem-budget 24576` = 实际占 40960 token 的槽位。

**调优含义**：如果想让**有效窗口更大**，可以**降 gen_reserve 来换 budget**
（前提是你的单轮生成不需要那么长）。

### 2.3 `--kv-dtype NAME` —— KV 精度（**真正的内存杠杆**）⭐️

```
--kv-dtype NAME            GPU KV cache type for K and V:
                           f16 | f32 | q8_0 | q5_0 | q4_0 (default q8_0)
```

| 取值 | 每元素字节 | 每 token（本模型） | 说明 |
|---|---:|---:|---|
| `f16` | 2 B | 64 KiB | 基准 |
| `f32` | 4 B | 128 KiB | 一般不用 |
| `q8_0` | 1.0625 B | 34 KiB | 默认；官方 IQ3 配方 |
| **`q5_0`** | 0.6875 B | **22 KiB** | **本机现役**；官方 IQ4 配方 |
| `q4_0` | 0.5625 B | 18 KiB | 更省，精度风险更高 |

**`📌 实测效果`**（同机同负载，`q8_0` → `q5_0`）：

| 指标 | `q8_0` | `q5_0` |
|---|---:|---:|
| 容器 anon | 13.35 GiB | **4.36 GiB** |
| 宿主 available | 609 MiB | **8.9 GiB** |
| swap | 4.1 GiB（增长中） | 2.3 GiB（稳定） |
| 新增 OOM | 持续 | **0** |
| prefill | 430–500 t/s | 417–560 t/s |

**⚠️ V100 上换 KV 精度的两个硬前提**：

| # | 前提 | 检查方法 |
|---|---|---|
| 1 | 编译期 **`GGML_CUDA_FA_ALL_QUANTS=ON`** | `grep GGML_CUDA_FA_ALL_QUANTS build/CMakeCache.txt` → 本机为 `ON` ✅ |
| 2 | **K / V 必须同类型** | 用 `--kv-dtype` **一次设两个**，不要分开设 `-ctk` / `-ctv` |

**⚠️ 验证 FA 真的在工作**（启动日志**不会有任何 warning**）：

```bash
# 读容器日志的 prompt processing 行
docker logs kvmem-test 2>&1 | grep -i "prompt processing" | tail -5
```

| prefill | 判定 |
|---|---|
| 400–560 t/s | ✅ FA 正常 |
| **< 80 t/s** | ❌ **FA 已弃权退回 CPU** → 立刻换回原精度 |

### 2.4 `--kvmem-gpu-ratio R` —— GPU 槽池占显存比例

```
--kvmem-gpu-ratio R        cap slot pool at this fraction of GPU VRAM (default 0.50)
```

| 项 | 值 |
|---|---|
| 默认 | **0.50**（显存的 50%） |
| 语义 | GPU 侧槽池的**硬上限比例** |

> 源码 `kvmem_store.hpp:379` 提到：比值 `<= 0` 或 `> 1` 会被当作 `1`（硬顶）。
> 实际池大小取 `min(budget + gen_reserve, ratio × VRAM)` 或高水位。

**用途**：显存紧张时可以直接压这个比例，比逐个调 budget 更粗暴有效。

### 2.5 `--kvmem-cpu-gb GB` / `--kvmem-nvme-gb GB` —— 分层存储的显式开关 ⭐️

```
--kvmem-cpu-gb GB          CPU spill arena in GiB (0 = off)
--kvmem-nvme-gb GB         NVMe file in GiB (0 = off)
--kvmem-nvme-dir PATH      NVMe directory (default /tmp/kvmem_nvme)
```

这两个参数**直接对应分层存储的 CPU / NVMe 层**：

| 层 | 参数 | 默认 | 本机现役 |
|---|---|---|---|
| GPU | `--kvmem-gpu-ratio` / `--kvmem-budget` | 0.50 / 131072 | 未设 / 24576 |
| **CPU** | **`--kvmem-cpu-gb`** | `0`（off） | **未设** |
| **NVMe** | **`--kvmem-nvme-gb`** | `0`（off） | **未设** |

> 💡 **对治 OOM 的潜在新杠杆**：本机 OOM 的直接原因是**宿主 RAM 被 KV 池吃掉**。
> `--kvmem-cpu-gb` 是**显式给 CPU 层设上限**的参数——如果它的语义是"上限"而非"预分配"，
> 那么它比换 KV 精度更直接。
>
> ⚠️ **但本仓库未实测过它**，且 `0 = off` 的确切语义（"不启用"还是"不限制"）**待验证**。
> 验证方法：设一个较小的值（如 `--kvmem-cpu-gb 2`），观察 anon 是否被硬性封顶在 2 GiB。
> 若成立，这会是本仓库最有价值的后续发现。

### 2.6 检索行为相关（进阶）

| 参数 | 默认 | 含义 |
|---|---|---|
| `--kvmem-block-tokens N` | `128` | 块大小（KVMem 把上下文切成 128-token 的块） |
| `--kvmem-sink-tokens N` | `0` = 一个块 | **永远保留的前缀**（attention sink）；向下取整，最少一个块 |
| `--kvmem-recent-tokens N` | `0` | **永远保留的最新后缀**（进 select budget） |
| `--kvmem-method NAME` | `retrieval` | 选块方法：`recency`（只看新旧）\| `retrieval`（按检索分数） |
| `--kvmem-query-last N` | `64` | 找不到 last-user span 时的兜底 query 长度 |
| `--kvmem-query-max-tokens N` | `512` | 用 last-user span 做检索 query 时截断到多少 token |
| `--kvmem-query-replay MODE` | `auto` | `legacy` \| `auto` |
| `--kvmem-query-policy MODE` | `user` | `legacy` \| `user` |

> **实践含义**：`--kvmem-sink-tokens` 和 `--kvmem-recent-tokens` 是"**保底命中**"——
> 让最关键的开头和最新内容**一定**在窗口里，不被检索算法漏掉。
> 长对话场景下，适当设 `--kvmem-recent-tokens` 通常比单纯加大 budget 更划算。

### 2.7 开关与诊断

| 参数 | 默认 | 含义 |
|---|---|---|
| `--kvmem` / `--no-kvmem` | **on** | 总开关（可用来对照 KVMem 开/关的差异） |
| `--kvmem-trace` / `--no-kvmem-trace` | off | 输出原始 `KVMEM_*` 诊断（或 `KVMEM_TRACE=1`） |

---

## 3. 投机解码（MTP）

| 参数 | 默认 | 本机现役 | 说明 |
|---|---|---|---|
| `--spec-type TYPE` | **`none`** | `draft-mtp` | `none` \| `draft-mtp` |
| `--spec-kv-dtype TYPE` | **`f16`** | 未设 | **MTP K/V 精度**，容易被忽略的显存开销 |
| `--spec-draft-n-max N` | **`3`** | `2` | draft token 数 |
| `--spec-draft-p-min P` | `0` | 未设 | 最小 draft 概率 |
| `--kvmem-mtp-state MODE` | `replay`（带 MTP 时） | `replay` | `snapshots` \| `auto` \| `replay` |

> ⚠️ **V100 上 MTP 是负收益**（IQ 系量化在 Volta 无 INT8 张量核 → 退回 ALU → 算力瓶颈）。
> 本机保留 MTP 参数是沿袭既有配置；**在 V100 上关掉它通常更快**。
> 若要省显存，`--spec-kv-dtype q8_0` 可把 MTP 的 KV 从 f16 降下来，
> 且**不影响最终输出质量**（draft token 会被主模型校验）。

---

## 4. 通用服务参数

### 4.1 进程与监听

| 参数 | 默认 | 说明 |
|---|---|---|
| `-m, --model PATH` | — | **必填**，GGUF 路径 |
| `--host HOST` | `127.0.0.1` | 容器内要设 `0.0.0.0` 才能被外部访问 |
| `--port N` | `8080` | 容器内端口 |
| `--api-key KEY[,KEY...]` | 无 | 支持逗号分隔多 key |
| `--api-key-file PATH` | — | 一行一个 key；空行与 `#` 开头忽略 |
| `-a, --alias NAME` | — | API 暴露的模型名 |
| `--timeout, -to N` | `1800` | HTTP 读写超时（秒） |
| `--threads-http N` | 自动 | HTTP worker 线程数 |

### 4.2 上下文与批处理

| 参数 | 默认 | 说明 |
|---|---|---|
| `-c, --ctx-size N` | `2048` | 上下文长度（= 逻辑工作区） |
| `-n, --n-predict N` | `-1` | 默认 max_tokens（`-1` = 不额外限制） |
| `-b, --batch-size N` | `512` | 逻辑 batch |
| `-ub, --ubatch-size N` | `= -b` | 物理 batch |
| `-t, --threads N` | 自动 | CPU 生成线程；`<=0` = 硬件并发数 |
| `-tb, --threads-batch N` | `= -t` | CPU batch 线程 |

### 4.3 设备与显存

| 参数 | 默认 | 说明 |
|---|---|---|
| `-ngl, --n-gpu-layers N` | `99` | 卸载到 GPU 的层数（99 = 全卸） |
| `--device, -dev NAME` | — | 单个卸载设备，如 `CUDA0`；`none` = 纯 CPU |
| `--list-devices` | — | 列出可用设备后退出 |
| `--main-gpu, -mg N` | `0` | 主设备索引 |
| `--split-mode, -sm MODE` | — | `none` \| `layer`；**多 GPU 模式不支持** |
| `--tensor-split, -ts N` | — | 单个设备比例；多个会被拒绝 |
| `-fa, --flash-attn MODE` | `auto` | `on` \| `off` \| `auto` |
| `-lm, --load-mode MODE` | `auto` | `auto` \| `none` \| `mmap` \| `mlock` \| `mmap+mlock` \| `dio` |
| `--mmap` / `--no-mmap` / `--mlock` | — | `load-mode` 的旧别名 |

> `-np, --parallel N`：**当前只支持 1**。这意味着**压测会独占服务**。

### 4.4 多模态与日志

| 参数 | 默认 | 说明 |
|---|---|---|
| `--mmproj PATH` | — | 视觉投影器 GGUF |
| `--mmproj-offload` / `--no-mmproj-offload` | 开 | 视觉编码器放 GPU / CPU |
| `--image-min-tokens` / `--image-max-tokens` | — | 图像 token 数范围 |
| `-lv, --verbosity N` | `3` | 0 silent / 1 error / 2 warn / **3 info** / 4 trace / 5 debug |
| `--ui-dir PATH` / `--no-ui` / `--ui` / `--webui` | UI 默认开 | 内置 chat UI |

---

## 5. 采样与思考（Thinking）

### 5.1 采样默认值

> 该服务对 **Qwen3.8-27B 的 Thinking / non-Thinking 各有一套默认值**，按请求选择。
> 下表括号内为 `(Thinking / non-Thinking)`。

| 参数 | 取值范围 | 默认 |
|---|---|---|
| `--temp, --temperature T` | [0, 2] | `1.0` / `0.7`；**`0` = greedy** |
| `--top-p P` | [0, 1] | `0.95` / `0.80` |
| `--top-k K` | ≥ 0，`0` 关闭 | `20` |
| `--min-p P` | [0, 1] | `0` |
| `--presence-penalty P` | [-2, 2] | `0` / `1.5` |
| `--frequency-penalty P` | [-2, 2] | `0` |
| `--repeat-penalty P` | > 0 | `1` |
| `--seed N` | uint32 | 随机 |

> **请求里的同名字段会覆盖这些进程默认值。**

### 5.2 思考控制

| 参数 | 默认 | 说明 |
|---|---|---|
| `--enable-thinking` | **off** | 打开 Qwen 思考（请求可覆盖） |
| `--no-think` | — | 强制关闭思考 |
| `--reasoning-effort LEVEL` | 模板默认 | 模板 effort；`none` = 关闭思考 |
| `--reasoning-budget N` | `-1` | `-1` 不限；`0` 立刻结束；`N>0` 在 N 个思考 token 后强制 `</think>` |
| `--reasoning-budget-message MSG` | 无 | 强制 `</think>` 前注入的文本 |

### 5.3 模板

| 参数 | 说明 |
|---|---|
| `--jinja` | 原生 Jinja 渲染（**始终启用**） |
| `--chat-template TEMPLATE` | 直接传 Jinja 文本 |
| `--chat-template-file PATH` | **从文件加载模板**（8092 就是用它打的补丁） |
| `--chat-template-kwargs JSON` | 模板默认参数 |

---

## 6. 本机现役配置逐参数解读

```bash
docker rm -f kvmem-test && docker run -d --name kvmem-test \
  --gpus all -p 8095:8080 --restart unless-stopped \
  -v /home/$USER/kvmem-llama.cpp:/src \
  -v /llama/models/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf:/models/model.gguf:ro \
  nvidia/cuda:12.8.1-devel-ubuntu24.04 \
  /src/build/bin/llama-kvmem-server -m /models/model.gguf \
  --host 0.0.0.0 --port 8080 -c 131072 -n 16384 \
  --kvmem-budget 24576 --kvmem-gen-reserve 16384 --kv-dtype q5_0 \
  --spec-type draft-mtp --spec-draft-n-max 2 --kvmem-mtp-state replay \
  --api-key "$API_KEY"
```

| 参数 | 值 | 为什么这么设 | 相对默认 |
|---|---|---|---|
| `--host 0.0.0.0` | — | 容器内必须，否则外部访问不到 | 默认 `127.0.0.1` |
| `-c 131072` | 128K | 允许客户端发 12.8 万长度历史；**成本极低**（空载仅 245 MiB） | 默认 2048 |
| `-n 16384` | — | 默认生成上限 | 默认 `-1` |
| `--kvmem-budget 24576` | 2.4 万 | ⚠️ **有效窗口只有 2.4 万**，远小于 `-c` | 默认 131072 |
| `--kvmem-gen-reserve 16384` | — | 为长生成留足槽位 | 默认 **256**（调大了 64 倍） |
| `--kv-dtype q5_0` | — | ⭐ **止住 OOM 的关键**（34 → 22 KiB/token） | 默认 `q8_0` |
| `--spec-type draft-mtp` | — | ⚠️ 沿袭既有配置；**V100 上其实是负收益** | 默认 `none` |
| `--spec-draft-n-max 2` | — | 比默认 3 保守 | 默认 3 |

**这份配置的两个"隐藏成本"**：

1. `--kvmem-gen-reserve 16384`（默认的 64 倍）占掉 GPU KV 池一大块；
2. `--spec-type draft-mtp` 在 V100 上**大概率是负收益**，还额外占 MTP 的 KV（`--spec-kv-dtype` 默认 f16）。

---

## 7. 三档推荐配置

### 档位 A · 稳（日常 / 长会话 / 内存紧张）—— 本机现役

```bash
-c 131072 -n 16384 \
--kvmem-budget 24576 --kvmem-gen-reserve 16384 \
--kv-dtype q5_0 --spec-type none \
--api-key "$API_KEY"
```

| 项 | 值 |
|---|---|
| 有效窗口 | 24,576 |
| 宿主占用（单次 128K） | ≈3.9 GiB |
| 显存 | 宽松 |

> 相对现役，这里把 `--spec-type` 改成 `none`（V100 上关掉 MTP 更快、更省）。

### 档位 B · 均衡（需要真正的长上下文理解）

```bash
-c 131072 -n 16384 \
--kvmem-budget 98304 --kvmem-gen-reserve 16384 \
--kvmem-recent-tokens 8192 \
--kv-dtype q5_0 --spec-type none \
--api-key "$API_KEY"
```

| 项 | 值 |
|---|---|
| 有效窗口 | 98,304 |
| 显存需求 | ≈2,464 MiB（余 ~546 MiB，较稳） |
| 代价 | prefill 变慢 |

### 档位 C · 对齐官方配方（保守起步）

```bash
-c 131072 -n 16384 \
--kvmem-budget 36864 --kvmem-gen-reserve 16384 \
--kv-dtype q5_0 --api-key "$API_KEY"
```

| 项 | 值 |
|---|---|
| 有效窗口 | 36,864（官方 IQ3 配方值） |
| 显存需求 | ≈1,144 MiB（余 ~1.9 GiB） |

### ⚠️ 不要照抄官方 256K 配方

官方基准机的宿主内存是 **32 GiB**（WSL2 可见 19.53 GiB），本机只有 **15.5 GiB**。
官方 256K 满填充的运行期 RSS 峰值是 **11.25–13.44 GiB**——**在本机已经越过 OOM 击杀线**。

---

## 8. 调参决策树

```
宿主 OOM / 服务反复重启？
├─ 先看是不是 KV 池 → 采样容器 anon（scripts/kvmem-mem-sample.sh）
│
├─ 要立刻缓解
│   ├─ ① 重启容器清池（最直接，KV 池是高水位语义）
│   ├─ ② --kv-dtype q8_0 → q5_0  ⭐ 每 token 成本 −35%
│   └─ ③ --kvmem-cpu-gb 设上限（⚠️ 未实测，见 §2.5）
│
├─ 要长期稳
│   ├─ --spec-kv-dtype q8_0（把 MTP 的 f16 KV 降下来）
│   ├─ 降 --kvmem-gen-reserve（16384 → 更小）
│   └─ 给容器加 --memory 上限（让 OOM 只杀容器，不连带杀宿主其它服务）
│
└─ 想扩大"有效上下文"（不是 `-c`！）
    ├─ 提 --kvmem-budget（代价：prefill 变慢）
    ├─ 或加 --kvmem-recent-tokens / --kvmem-sink-tokens（保底命中，更划算）
    └─ 别忘了 -c 只是"客户端能发多长"，跟模型"看得到多少"无关
```

---

## 附：参数来源说明

| 来源 | 覆盖范围 | 可信度 |
|---|---|---|
| `llama-kvmem-server --help`（真机导出） | 全部参数名 / 取值域 / 大部分默认值 | ✅ 权威 |
| 源码 `kvmem_store.hpp` 等 | 少数 help 未标的默认值（如 `--kvmem-budget` = 131072） | ✅ 权威 |
| `📌 实测` 标注处 | 本机真机读数 | ✅ 实测 |
| `⚠️ 未实测` 标注处 | 语义待验证的推断 | ⚠️ 需自行验证 |

> 不同 fork 版本的参数可能有增删，**以你自己那版的 `--help` 为准**。
