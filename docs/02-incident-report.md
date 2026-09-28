# 02 · 事故报告：客户端反复掉线 → 宿主 OOM

> **报告日期**：2026-09-28
> **严重级别**：生产可用性（用户侧表现为"工作中随机断开"）
> **最终判定**：宿主内存不足触发 OOM killer，SIGKILL 容器主进程，Docker 自动重启
> **状态**：已定案并处置（处置过程见 [03-tuning-log.md](03-tuning-log.md)）

---

## 1. 症状

用户报障原话：**「8095 工作中 Codex 和模型断开 2 次」**。

**调用链路**：

```
Codex (Windows 客户端)
   │
   ▼
cc-switch 代理  127.0.0.1:15721
   │
   ▼  POST http://<HOST_IP>:8095/v1/chat/completions   (model = model.gguf)
   │
   ▼
容器 kvmem-test  →  llama-kvmem-server  →  Qwen3.8-27B (IQ3_S)
```

**关键特征（后来回看，每一条都是线索）**：

- 掉线是**间歇性**的，不是持续不可用。
- 掉线时**服务日志里没有任何报错**——这是本案最大的迷惑点。
- 客户端侧看到的是 **HTTP 502**。
- 掉线后**过一会儿自己就好了**，无需人工干预。

---

## 2. 排除法：先证明"不是软件崩溃"

第一反应通常是「服务挂了」。但逐项排查后，所有"软件崩溃"的特征**都不成立**：

| 假设 | 检查 | 结果 |
|---|---|---|
| 段错误 / 断言失败 | 容器日志搜 `error` / `abort` / `assert` / `signal` | ❌ **崩溃点附近零输出** |
| CUDA 错误 | 搜 `CUDA error` / `out of memory` / `illegal memory` | ❌ 无 |
| 端口冲突 / 绑定失败 | `docker inspect` 端口映射 | ❌ 正常 |
| 模型加载失败 | 启动日志 | ❌ 每次都能正常加载（8.3s） |
| 代理配置错误 | cc-switch 侧日志 | ❌ 代理侧只报超时 |

**「崩溃时日志完全静默」这个特征本身就是最强的信号**——
正常崩溃（段错误、断言、CUDA error）都会留下痕迹；
**唯一会静默杀死进程的是外部信号，最典型的就是 SIGKILL（OOM killer 专用）。**

---

## 3. 证据链：把 502 和进程重启对上表

### 3.1 容器被重启过几次

```bash
$ docker inspect kvmem-test --format '{{.RestartCount}} {{.HostConfig.RestartPolicy.Name}}'
6  unless-stopped
```

**重启 6 次**，且策略是 `unless-stopped` —— 意味着**崩溃后 Docker 会自动拉起**。

这解释了症状里的「过一会儿自己就好了」。

### 3.2 每次重启的时刻

服务每次启动都会打印 CUDA 环境横幅（`== CUDA ==`），可以当作启动指纹：

```bash
$ docker logs kvmem-test 2>&1 | grep -c "== CUDA =="
7                      # 1 次正常启动 + 6 次重启
```

7 次启动的时刻（CST）：

| # | 启动时刻 |
|---|---|
| 1 | 09:04:02 |
| 2 | 10:32:58 |
| 3 | 15:45:59 |
| 4 | 16:10:00 |
| 5 | 16:36:29 |
| 6 | 17:23:12 |
| 7 | 18:05:01 |

### 3.3 与客户端 502 对表

把上表与 cc-switch 记录的 502 时间戳并排：

| 容器重启时刻 | 客户端 502 时刻 | 吻合 |
|---|---|---|
| 10:32:58 | 10:32:58 | ✅ |
| 15:45:59 | 15:46:03 | ✅（+4s） |
| 16:10:00 | 16:10:00 | ✅ |
| 16:36:29 | 16:36:29 | ✅ |
| 17:23:12 | 17:23:17 | ✅（+5s） |
| 18:05:01 | 18:05:06 | ✅（+5s） |

**一一对应，无遗漏、无多余。**

### 3.4 超时时间也吻合

502 记录的 `latency_ms` **精确落在 ~5000 ms**：

```
客户端发出请求 → 容器正在重启（端口仍 accept，但没有进程应答）
              → 代理等满 5s 超时 → 返回 502
```

**8~13 秒的重启空窗** + **5 秒的代理超时** = 客户端必然看到一次 502。

> **结论：容器重启 1 次 = 客户端掉线 1 次。** 数字完全对得上。

### 3.5 崩溃前的征兆

崩溃前日志有可辨认的模式：

| 征兆 | 观测 |
|---|---|
| 崩溃前静默 | **3~100 s 无任何输出** |
| prefill 速度崩塌 | 正常 550 t/s → 崩溃前 **68 t/s** |

这两个都是**内存压力下 swap 抖动**的典型表现：
系统开始换页 → 磁盘 I/O 拖慢一切 → 最终 OOM killer 出手。

---

## 4. 证据链：确认是 OOM killer

### 4.1 系统级计数

```bash
$ cat /proc/vmstat | grep oom_kill
oom_kill 22
```

宿主累计 22 次 OOM 击杀。

### 4.2 定位到容器所在的 slice

```bash
$ cat /sys/fs/cgroup/system.slice/memory.events
oom_kill 7
```

**7 次**——与容器的 7 次启动（1 正常 + 6 重启）**数量一致**。

### 4.3 ⚠ 一个必须记住的坑

```bash
$ cat /sys/fs/cgroup/system.slice/docker-<容器ID>.scope/memory.events
oom_kill 0          # ← 看起来"没被 OOM 过"，是假象！
```

**叶子 cgroup 会显示 `oom_kill=0`**，因为容器**没有设置 `--memory` 上限**——
内存统计被上推到父 slice 记账。

> **教训：查容器 OOM 不要只看叶子 cgroup，要看 `system.slice/memory.events`。**
> 只看叶子会得出「不是 OOM」的相反结论。

### 4.4 现场内存快照

```bash
$ free -m
               total        used        free      shared  buff/cache   available
Mem:           15872        ...         165        ...        ...          ...
Swap:           ...
```

| 指标 | 值 |
|---|---|
| 宿主总内存 | 15,872 MiB（≈15.5 GiB） |
| **free** | **仅 165 MiB** |
| **swap 已用** | **2.9 GiB** |

---

## 5. 根因

```
┌─────────────────────────────────────────────────────────────┐
│  宿主内存 15.5 GiB                                          │
│                                                             │
│  ├─ 128K 上下文的宿主侧 KV 池（anon，不可回收）  ← 主要占用  │
│  ├─ 桌面 / 面板 / 其他常驻服务                              │
│  └─ ...                                                     │
│                                                             │
│  ────────────── 15.5 GiB 总内存 ──────────────              │
│                    ▲                                        │
│                    │ 超出                                    │
│                    ▼                                        │
│            OOM killer 出手                                  │
│                    │                                        │
│                    ▼                                        │
│         SIGKILL llama-kvmem-server                          │
│                    │                                        │
│                    ▼                                        │
│     Docker 自动重启（8~13s 空窗）                            │
│                    │                                        │
│                    ▼                                        │
│      代理 5s 超时 → 客户端看到 502                           │
└─────────────────────────────────────────────────────────────┘
```

**一句话根因**：

> 宿主只有 15.5 GiB RAM，而 128K 上下文的**宿主侧 KV 池**（anon 页，不可回收）
> 加上其它常驻服务把内存挤爆 → OOM killer 杀 `llama-kvmem-server`
> → Docker 自动重启 → 代理超时 → 客户端掉线。

**为什么是"随机的"**：OOM 只在内存**瞬时冲高**时触发，
而内存冲高发生在**长上下文 prefill 的瞬间**——所以表现为「用着用着突然断」。

---

## 6. 影响范围

当天 502 按小时分布：

| 小时 | 502 次数 |
|---|---:|
| 04 | 23 |
| 05 | 25 |
| 08 | 5 |
| 09 | 25 |
| 10 | 2 |
| 15 | 1 |
| 16 | 8 |
| 17 | 2 |
| 18 | 2 |

**规律**：集中在 04–05 时与 09 时（各 20+ 次），正是**长时间会话**的时段——
与「宿主占用正比于实际历史长度」的模型完全一致。

---

## 7. 顺带发现：配置与实际不一致

排查过程中发现一处**独立**的配置问题：

- cc-switch 数据库里当前 provider（名为 `copy`）的 endpoint 写的是 **8092**；
- 但实际流量全部打在 **8095**。

**以日志里的"请求目标"行为准**——也就是说，**配置文件与实际生效的行为不一致**。

> 这类不一致本身不是本次故障的原因，但会严重误导后续排查
> （比如你去 8092 的日志里找证据，什么也找不到）。**建议对齐**。

---

## 8. 判定小结

| 问题 | 答案 |
|---|---|
| 是服务 bug 吗？ | ❌ 不是。服务每次都能正常启动，崩溃点无任何报错 |
| 是显存不足吗？ | ❌ 不是。是**宿主内存**不足 |
| 是 `-c` 设太大吗？ | ❌ 不是（见 [03-tuning-log.md](03-tuning-log.md)，`-c` 只占 245 MiB） |
| **是什么？** | ✅ **宿主 15.5 GiB 内存被宿主侧 KV 池挤爆，OOM killer 击杀进程** |
| 怎么验证？ | `RestartCount` + `== CUDA ==` 横幅时刻 + 502 时间戳三方对表；`system.slice/memory.events` |
| 怎么处置？ | 见 [03-tuning-log.md](03-tuning-log.md) |

---

## 附：本次用到的排查命令

```bash
# 1. 容器重启次数与策略
docker inspect kvmem-test --format '{{.RestartCount}} {{.HostConfig.RestartPolicy.Name}}'

# 2. 每次启动的时刻（启动指纹）
docker logs kvmem-test 2>&1 | grep "== CUDA =="

# 3. 崩溃点是否有报错
docker logs kvmem-test 2>&1 | tail -50

# 4. 系统级 OOM 计数
cat /proc/vmstat | grep oom_kill

# 5. 容器所在 slice 的 OOM 计数（注意不是叶子 cgroup！）
cat /sys/fs/cgroup/system.slice/memory.events

# 6. 现场内存
free -m
awk '/^MemAvailable/{print $2/1024 " MiB available"}' /proc/meminfo
awk '/^SwapTotal/{t=$2}/^SwapFree/{f=$2}END{print (t-f)/1024 " MiB swap used"}' /proc/meminfo

# 7. 容器内 anon 内存（KV 池在这里）
CID=$(docker inspect -f '{{.Id}}' kvmem-test)
cat "/sys/fs/cgroup/system.slice/docker-${CID}.scope/memory.stat" | grep -E '^(anon|file|shmem) '
```
