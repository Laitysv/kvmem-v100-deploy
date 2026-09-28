# 03 · 调优日志：三次改动，只有一次真正有效

> 承接 [02-incident-report.md](02-incident-report.md) 的定案（宿主 OOM）。
> 本篇记录处置过程。**最有价值的不是"改了什么"，而是"哪一步是无效功，以及为什么"。**

---

## 0. 处置总览

| # | 时刻 | 改动 | 有效性 | 结论 |
|---|---|---|---|---|
| 1 | 18:44 | `-c 131072 → 98304` | ❌ **无效** | `-c` 不是内存杠杆 |
| — | 18:52 | 停用 Obsidian（root systemd 服务） | ⚠️ 辅助 | 只腾出约 70 MiB |
| 2 | 19:15 | **`--kv-dtype q8_0 → q5_0`** | ✅ **有效** | **真正的杠杆** |
| 3 | 20:26 | `-c 98304 → 131072`（恢复 128K） | ✅ 安全 | 且成本极低 |

**一句话**：有效的那一步只改了**一行参数**。

---

## 1. 第一步：改 `-c`（无效）

### 做了什么

按容器**实际配置**原样重建（**不是**按启动脚本，两者有差异——见第 5 节），
只把 `-c 131072` 改成 `98304`：

```bash
docker rm -f kvmem-test && docker run -d --name kvmem-test \
  --gpus all --restart unless-stopped -p 8095:8080 \
  -v /home/<USER>/kvmem-llama.cpp:/src \
  -v /llama/models/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf:/models/model.gguf:ro \
  nvidia/cuda:12.8.1-devel-ubuntu24.04 \
  /src/build/bin/llama-kvmem-server -m /models/model.gguf \
  --host 0.0.0.0 --port 8080 -c 98304 -n 16384 \
  --kvmem-budget 24576 --kvmem-gen-reserve 16384 --kv-dtype q8_0 \
  --spec-type draft-mtp --spec-draft-n-max 2 --kvmem-mtp-state replay \
  --api-key "$API_KEY"
```

### 即时读数（看起来"有效"）

| 指标 | 改前 | 改后 |
|---|---|---|
| 容器 cgroup anon | 3.44 GiB | 3.04 GiB |
| 宿主 available | ~9.8 GiB | 9.0–10 GiB |

当时判断：「宿主侧 KV 池约按 `n_ctx` 等比缩了 25%，约 −0.4 GiB」。

### 但这是错觉 —— 最终定案

**证据一：改完之后又 OOM 了。**

> 19:25:28 那次 OOM **就发生在改完 `-c` 之后**。
> （18:42:26 也又 OOM 重启了一次，比这次改动还早。）

**证据二：改动根本没生效在运行中的容器上。**

`-c` 改的是**脚本文件**，但容器直到 **19:59:05** 才重建。
所以 19:25 那次崩溃时，跑的**还是旧的 `-c 131072`**。

**证据三（最要紧）：`-c` 压根不触及当前占用。**

- 宿主占用**正比于客户端实际发送的历史长度**，而不是 `-c` 的封顶值。
- Codex 当时实际只发 **~70K** token，**远低于 98K 的封顶**。
- → 降 `-c` 根本不改变"当前已经占用了多少"。

**证据四：那 −0.4 GiB 是"重启后池子清零"的假象。**

KV 池**不随当前请求缩小**。旁证：

> 20:13 时发了一个 **852 token 的小请求**，容器 anon 仍然是 **6482 MiB** ——
> 池子完全没有随请求变小。

所以"改 `-c` 省了 0.4 GiB"其实是「重启一次清了一次池」，与 `-c` 无关。

### 结论

> **`-c` 只能封顶，不能减存量。别再指望调 `-c` 治 OOM。**

---

## 2. 穿插：腾内存（辅助手段）

同一晚上还做了两件"腾地方"的事。

### 2.1 停用 Obsidian

`gpu-host` 上跑着一个 **root 的 headless VNC 版 Obsidian**（systemd 系统服务，不是桌面应用）：

```
/etc/systemd/system/obsidian-vnc.service
  User=root
  ExecStart=/usr/local/bin/obsidian-vnc   # Xvfb :99 + openbox + x11vnc :5900 + Obsidian
  Restart=on-failure / RestartSec=10
```

**权限边界**（记一笔）：普通用户在 `docker` 和 `Administrators` 组，但 `sudo` 要密码；
`systemctl stop` 被 polkit 拒（`Interactive authentication required`），**root 进程也杀不掉**。

> ⚠️ 该单元是 `Restart=on-failure`，**光 kill 进程会被 systemd 拉起来**，必须先 `disable`。

```bash
sudo systemctl stop    obsidian-vnc.service
sudo systemctl disable obsidian-vnc.service obsidian-novnc.service
sudo systemctl reset-failed              # 清掉 failed 残留
```

**效果**：所有 obsidian/Xvfb/x11vnc/openbox 进程消失。

**实际收益很小**：obsidian 相关全部进程 RSS 合计约 **70 MiB** + swap 136 MiB。

> 教训：**它是个"看着碍眼但几乎不占内存"的服务**。真正吃内存的是 KV 池（GiB 级）。
> 排查内存问题时要分清「看起来占资源」和「实际占资源」。

### 2.2 ⚠️ 保留不动的东西

排查自启路径时发现 root crontab 里有两条 obsidian 相关定时任务：

```
7  2 * * *   /usr/local/bin/obsidian-auto-distill   # 每日知识库蒸馏
13 8 * * 0   /usr/local/bin/obsidian-auto-audit     # 每周知识库巡检
```

**这两条不是 Obsidian 应用，是知识库自动化**（只读 vault 文件、不依赖 GUI），
**已保留未动**。关掉 Obsidian GUI 后它们仍能正常工作。

> 顺手确认干净：`/etc/rc.local` 不存在；`/etc/cron.d/`、`/etc/xdg/autostart/`、
> `/root/.config/autostart/` 均无 obsidian 引用。

---

## 3. 第二步：换 KV 精度（✅ 有效）

### 3.1 动手前必须先验的两件事（V100 专用）

**换 KV 精度不是改个参数就完事**，在 Volta 上有两个硬门槛：

| # | 前提 | 为什么 | 本次验证 |
|---|---|---|---|
| 1 | 编译期 **`GGML_CUDA_FA_ALL_QUANTS=ON`** | `q5_0` 在 hybrid 模型上需要它，否则 FlashAttention 直接弃权 | 查 `build/CMakeCache.txt`：`GGML_CUDA_ARCHITECTURES=70`、`GGML_CUDA_FA=ON`、**`GGML_CUDA_FA_ALL_QUANTS=ON`** ✅ |
| 2 | **K / V 必须同类型** | 混用会让 `fattn.cu` 弃权、整个注意力丢给 CPU 后端 | 用 `--kv-dtype <t>` **同时设 K 和 V** ✅ |

### 3.2 ⚠️ 怎么确认 FA 真的在工作（关键）

**启动日志不会有任何 warning。只看"没报错"会误判成 FA 正常。**

唯一可靠的判据是**量 prefill 速度**——读容器日志的 `prompt processing` 行：

| prefill 速度 | 判定 |
|---|---|
| **400–560 t/s** | ✅ FA 正常工作 |
| **< 80 t/s** | ❌ **FA 已弃权，退回 CPU** → 必须换回原精度 |

本次实测 **417–560 t/s**（与 `q8_0` 时的 430–500 一致）→ **FA 正常**。

### 3.3 改动

与第一步的重建命令**完全相同**，只把 `--kv-dtype q8_0` 换成 `--kv-dtype q5_0`：

```bash
# 旧 Cmd 先留档，回滚用
docker inspect kvmem-test --format '{{json .Config.Cmd}}' > /tmp/kvmem-old-cmd.json

docker rm -f kvmem-test && docker run -d ... \
  --kvmem-budget 24576 --kvmem-gen-reserve 16384 --kv-dtype q5_0 \
  ...
```

### 3.4 效果（实测，同机同负载）

| 指标 | `q8_0`（改前） | `q5_0`（改后） |
|---|---:|---:|
| 容器 anon | **13.35 GiB** | **4.36 GiB** |
| 宿主 available | **609 MiB** | **8.9 GiB** |
| swap 使用 | 4.1 GiB（仍在增长） | 2.3 GiB（**稳定不涨**） |
| 新增 OOM | 持续 | **0** |
| prefill | 430–500 t/s | 417–560 t/s |

**改动前的现场（19:14）**：

- 容器 anon 已涨到 **13.35 GiB**，另有 +2.2 GiB 换出到 swap；
- 宿主 `available` 只剩 **609 MiB**，swap 用 4.1 GiB，page cache 被压到 1.4 GiB，正在**剧烈抖动**；
- `oom_kill` 又 +2（host 22→24、system.slice 7→9）——这两次杀的是 **system.slice 里别的服务**，
  不是容器（容器 `RestartCount` 一直是 0），属于**连带伤害**。

**连续 4 分钟采样**：anon 3.03 → 4.21 GiB（约 +30 MiB/min），随上下文增长但**远低于改前**。

> ⚠️ **诚实标注**：部分改善来自"**重启后 KV 池清零**"，不能全部归功于 `q5_0`；
> 长会话仍会缓慢爬升。但 `q8_0 → q5_0` 使每 token 成本从 34 KiB 降到 22 KiB（**−35%**），
> 这是**结构性**的改善，与重启无关。

### 3.5 为什么 `q5_0` 能止住

回到 [01 篇](01-architecture.md) 的公式：

```
宿主 anon ≈ 基线 + 实际 token 数 × 每 token 成本

q8_0: 34 KiB/token  →  13.35 GiB  ← 越过 13.6 GiB 击杀线
q5_0: 22 KiB/token  →   4.36 GiB  ← 余量充足
```

**每 token 成本降 35%，直接把"能撑多长"提升了约 1.5 倍。**

---

## 4. 第三步：恢复 128K 工作区（安全）

用户指令：**「恢复」**（把 `-c` 改回 131072）。

### 4.1 ⚠️ 重建方式：不能跑启动脚本

**实际运行的容器与 `start-8095.sh` 有 4 处差异**（脚本是「带 Web UI」的变体）：

| 差异 | `start-8095.sh` | 实际容器 |
|---|---|---|
| `--shm-size 2g` | ✅ 有 | ❌ 无 |
| `-e LD_LIBRARY_PATH=/src/build/bin` | ✅ 有 | ❌ 无 |
| `-w /src` | ✅ 有 | ❌ 无 |
| `-v .../kvmem-ui-dist:/ui:ro` + `--ui-dir /ui` | ✅ 有 | ❌ 无 |
| **`--api-key`** | ❌ **无** | ✅ **有** |

> **结论：必须按 `docker inspect` 复刻，不要直接跑 `start-8095.sh`。**

为此新建了精确复刻脚本（本仓库 [`scripts/run-8095-exact.sh`](../scripts/run-8095-exact.sh)），
并在 `start-8095.sh` 顶部加了指向它的警告注释。

### 4.2 新容器

```bash
bash scripts/run-8095-exact.sh 131072 24576
```

```
-c 131072 -n 16384 --kvmem-budget 24576 --kvmem-gen-reserve 16384
--kv-dtype q5_0 --spec-type draft-mtp --spec-draft-n-max 2
--kvmem-mtp-state replay --api-key "$API_KEY"
```

`RestartCount=0`、`n_ctx=131072`、模型 8.3s 加载完、端口/挂载逐字复刻。

### 4.3 ⚠️ 重大修正：`-c` 并不驱动大额宿主预分配

重建后**空载 anon 只有 245 MiB**——而此前假设基线是 6.0 GiB，**错了**。

| 项 | 修正后的认识 |
|---|---|
| 那个 6.0–6.7 GiB | 是**长期使用累积的高水位**，不是 `-c` 的预分配 |
| 旁证 | 20:13 一个 852-token 的小请求，anon 仍是 6482 MiB → **池子不随当前请求缩小** |
| 也推翻了一条 | 「官方 `-c 262144` 空载 RSS 11 GiB = 预分配」→ 那条应是「**模型 + 已填充的池**」 |

> **结论：`-c` 本身很便宜**（128K vs 96K 几乎不差），**成本在实际历史/累积量**。

### 4.4 同一 prompt 的 A/B 复测（66,686 token）

| 指标 | 重建前 `-c 98304`（旧基线） | 重建后 `-c 131072`（干净基线） |
|---|---:|---:|
| 起始 anon | 6387 MiB | **246 MiB** |
| 峰值 anon | 8153 MiB | **2132 MiB** |
| Δanon | 1766 MiB | 1886 MiB |
| **斜率** | **27.1 KiB/tok** | **29.0 KiB/tok** |
| prefill | 424.6 t/s | **449.4 t/s** |
| wall | 161 s | 151 s |
| file（page cache） | 6423→5078（被驱逐） | 10228 不变 |
| swap | 2312 MiB | 2130 MiB（未动） |

**两个关键结论**：

1. **斜率两次一致**（27.1 / 29.0 KiB/tok）→ **每 token 成本模型验证通过**。
2. **prefill 快 5.8%** → 干净基线无内存压力，page cache 未被驱逐、无 swap 抖动。

---

## 5. 回滚机制（工程实践）

每次改参数都自动留档旧配置，这是**远程改生产服务的最低要求**：

```bash
# 改动前：留档旧 Cmd
docker inspect kvmem-test --format '{{json .Config.Cmd}}' \
  > "$KVREPO/kvmem-cmd-rollback-$(date +%Y%m%d-%H%M%S).json"

# 改动后：备份启动脚本（如果也改了脚本）
cp -a start-8095.sh "start-8095.sh.bak-$(date +%Y%m%d-%H%M%S)"
```

本次产生的备份序列（`start-8095.sh`）：

| 备份文件 | 对应改动 |
|---|---|
| `start-8095.sh.bak-20260928-184854` | `-c → 98304` |
| `start-8095.sh.bak2-20260928-200830` | `--kv-dtype → q5_0` |
| `start-8095.sh.bak3-20260928-202704` | `-c → 131072` |

> **`docker restart` 改不了命令行参数**——必须 `docker rm -f` + `docker run` 重建。

---

## 6. 处置结论

| 项 | 结论 |
|---|---|
| **有效杠杆** | `--kv-dtype`（每 token 成本 −35%） |
| **无效杠杆** | `-c`（只封顶，不减存量） |
| **辅助手段** | 腾内存（收益小，但要排除干扰项） |
| **长期风险** | 长会话仍会缓慢累积 → 需看门狗或定期重启清池 |
| **复测结果** | 全部 9 次 `llama-kvmem-ser` OOM 都发生在 **19:59:05 之前**；该容器 `StartedAt=19:59:05`、`RestartCount=0`、`system.slice oom_kill` 至今未增 ✅ |

### 还没做的可选下一步

- `--spec-kv-dtype`：MTP draft 的 KV（**默认 f16**）可再降一点；
  不影响最终输出质量（draft token 会被主模型校验）。
- 给容器加 **`--memory` 上限**：让 OOM 只杀容器、不再**连带杀死宿主机其它服务**。
- **挂看门脚本**：`RestartCount` 增加时自动记录 `free` / `oom_kill` / 日志尾部。
