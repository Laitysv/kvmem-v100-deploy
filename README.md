# KVMem on a Single V100

> 在**单张 Tesla V100-SXM2-16GB**、宿主内存仅 **15.5 GiB** 的机器上部署并调优
> **KVMem**（块稀疏 KV 缓存 + 分层存储的 llama.cpp fork）。
>
> 本文记录一次完整的真实运维事故：从「客户端反复掉线」一路挖到「宿主 OOM 击杀进程」，
> 再到 KV 精度调优、128K 工作区恢复与全量性能实测。
>
> **全部数据来自真机实测**，所有命令与脚本均可复现。

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

## 目录

| 文档 | 内容 |
|---|---|
| [`docs/01-architecture.md`](docs/01-architecture.md) | KVMem 是什么：块稀疏 KV 注意力、re-RoPE、分层存储；**参数语义逐个澄清**（最易搞反的部分） |
| [`docs/02-incident-report.md`](docs/02-incident-report.md) | 事故报告：完整证据链，如何从 502 反推到 OOM SIGKILL |
| [`docs/03-tuning-log.md`](docs/03-tuning-log.md) | 调优日志：三次改动（`-c` 无效 → `q5_0` 有效 → 恢复 128K）的完整过程与验证方法 |
| [`docs/04-benchmarks.md`](docs/04-benchmarks.md) | 性能与容量实测：速度表、A/B 复测、官方基准对照、128K 可行性推算 |
| [`docs/05-pitfalls-and-methodology.md`](docs/05-pitfalls-and-methodology.md) | 踩坑清单（11 条）+ 可复用的排查方法学 |
| [`docs/06-v100-multi-service.md`](docs/06-v100-multi-service.md) | V100 部署全景：8089 / 8092 / 8095 三个服务如何在同一张卡上共存与切换 |

## 脚本

| 脚本 | 用途 |
|---|---|
| [`scripts/run-8095-exact.sh`](scripts/run-8095-exact.sh) | **精确复刻**当前运行的容器（按 `docker inspect` 反推，支持 `[context] [budget]` 两个参数，自动留回滚记录） |
| [`scripts/kvmem-cap-probe.sh`](scripts/kvmem-cap-probe.sh) | 长上下文容量曲线压测：单次长 prefill + 每 5s 采样 anon/file/avail/swap/gpu，带 anon 上限自动中止 |
| [`scripts/kvmem-mem-sample.sh`](scripts/kvmem-mem-sample.sh) | 纯被动采样（不打扰服务）：内存 / 显存 / 当前上下文长度 |
| [`scripts/gguf_kv_math.py`](scripts/gguf_kv_math.py) | 解析 GGUF 头，按层数/头数/head_dim 推算 KV 缓存**每 token 成本**与各上下文长度下的总量 |

---

## 快速开始

```bash
# 0. 变量（按需覆盖）
export KVREPO=/home/$USER/kvmem-llama.cpp          # fork 源码目录（同时挂载为容器 /src）
export MODEL=/llama/models/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf
export API_KEY=changeme                            # 服务端 --api-key，务必自行更换

# 1. 看看当前容器的"真实"配置（不要相信启动脚本）
docker inspect kvmem-test --format '{{json .Config.Cmd}}'

# 2. 复刻 / 改参重建（例：128K 工作区 + 24576 有效窗口）
bash scripts/run-8095-exact.sh 131072 24576

# 3. 被动观察内存与上下文
bash scripts/kvmem-mem-sample.sh "" 24 15

# 4. 单次长 prefill 压测（会独占服务，total_slots=1）
bash scripts/kvmem-cap-probe.sh "" 300000 10500

# 5. 推算模型的 KV 每 token 成本
python3 scripts/gguf_kv_math.py "$MODEL"
```

---

## 适用与不适用

**适用**：在显存/内存受限的卡（尤其 Volta `sm_70`）上部署长上下文推理服务；
排查「服务无报错但客户端反复掉线」这类 OOM 型故障；评估 KV 量化对内存与速度的实际影响。

**不适用**：多卡 / 张量并行场景；非 llama.cpp 系引擎；云端托管服务的调优。

---

## License

[MIT](LICENSE) —— 文档与脚本均可自由使用、修改、分发。
