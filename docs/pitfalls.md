# 踩坑清单与排查方法学

> 本篇不按时间顺序，按**能不能复用**排序。
> 每条坑都标注了「现象 → 根因 → 正确做法」。

---

## 一、最容易误判的五个坑

### 坑 1 · 日志行首的时间戳不是墙上时钟 ⭐️

**现象**：日志行看起来带时间戳：

```
11.49.050.299  slot release: ...
12.26.122.049  prompt processing progress: 100%
```

**误判**：把它当成 `时.分.秒.毫秒`，于是认为服务在 11:49 和 12:26 处理了请求。

**真相**：**这是服务启动后经过的时间**，格式是 `分.秒.毫秒.微秒`。

**验证方式**（当时就是这么确认的）：

```
前缀 12.26.122.049  +  容器启动时刻 11:59:05.678
= 12:11:32.8Z
与 cc-switch 记录的请求完成时间 20:11:33 CST 精确吻合  ✅
```

**正确做法**：

```bash
# 一律用这个拿绝对时间，别信行内前缀
docker logs --timestamps kvmem-test
```

> **代价**：搞错这一点，会让"服务在什么时刻发生了什么"整条时间线全错，
> 进而得出"崩溃时间对不上"的错误结论。

---

### 坑 2 · 叶子 cgroup 的 `oom_kill` 显示 0 ⭐️

**现象**：

```bash
$ cat /sys/fs/cgroup/system.slice/docker-<ID>.scope/memory.events
oom_kill 0          # ← 结论："不是 OOM"
```

**真相**：容器**没有设置 `--memory` 上限**，内存统计被上推到**父 slice** 记账。
叶子 cgroup 因此显示 0。

**正确做法**：

```bash
# 查容器所在 slice，不是叶子 cgroup
cat /sys/fs/cgroup/system.slice/memory.events
# oom_kill 7   ← 真相在这里
```

> **这个坑会直接把你带到完全相反的结论上**：从"不是 OOM"变成"就是 OOM"。
> 排查 OOM 时**两级都要看**。

---

### 坑 3 · 「改 `-c` 有效」是重启清池的假象 ⭐️

**现象**：改完 `-c` 重建容器后，内存读数确实下降了（3.44 → 3.04 GiB），
看起来"改动生效了"。

**真相**：

1. **KV 池是高水位语义**——重启后清零，与 `-c` 无关。
2. 旁证：一个 **852 token** 的小请求后，anon 仍是 **6482 MiB**——池子不随请求缩小。
3. `-c 131072` 空载 anon 仅 **245 MiB**——`-c` 本身极便宜。

**正确做法**：要判断"某个改动是否降低了内存"，必须
**控制住"重启清池"这个混淆变量**——比如用**同一 prompt 做 A/B**（见 [benchmarks.md](benchmarks.md)）。

> **一般化教训**：**任何"重启后指标变好"的观测，都要先怀疑是不是重启本身带来的。**

---

### 坑 4 · 崩溃时日志完全静默，不代表"没问题" ⭐️

**现象**：服务在某个时刻停止输出，随后自动重启，**崩溃点附近一条错误都没有**。

**误判**：「日志没报错 → 不是崩溃 → 是网络问题 / 代理问题」。

**真相**：**唯一会静默杀死进程的是外部信号**，最典型的就是 **SIGKILL（OOM killer 专用）**。
正常崩溃（段错误、断言、CUDA error）都会留痕。

**崩溃前的可辨识征兆**：

| 征兆 | 观测 |
|---|---|
| 静默 3~100 s | 无任何输出 |
| prefill 速度崩塌 | 550 t/s → **68 t/s** |

**正确做法**：遇到"无报错的消失"，立刻去查 OOM / 外部 kill：

```bash
cat /proc/vmstat | grep oom_kill
cat /sys/fs/cgroup/system.slice/memory.events
dmesg | grep -i "killed process"
```

---

### 坑 5 · 只看"没报错"就以为 FA 在正常工作 ⭐️

**现象**：换了 KV 精度（`q5_0`）之后，服务启动**没有任何 warning**，请求也正常返回。

**误判**：「FA 正常工作」。

**真相**：V100 上如果 K/V 类型不同、或编译期缺 `GGML_CUDA_FA_ALL_QUANTS`，
**`fattn.cu` 会静默弃权**，把整个注意力丢给 CPU 后端——
**启动日志不会有任何提示**。

**唯一可靠的判据**：量 **prefill 速度**（读日志的 `prompt processing` 行）。

| prefill | 判定 |
|---|---|
| 400–560 t/s | ✅ FA 正常 |
| **< 80 t/s** | ❌ **FA 已弃权** → 立刻换回原精度 |

**正确做法**：**换 KV 精度前后都要量 prefill**，把它当作验收项，而不是可选项。

---

## 二、工程操作类坑

### 坑 6 · `docker restart` 改不了命令行参数

`docker restart` 只是重启进程，**命令行参数还是老的**。

```bash
# ❌ 改了脚本然后 restart —— 参数根本没变
# ✅ 必须重建
docker rm -f kvmem-test && docker run -d ... <新参数>
```

### 坑 7 · 启动脚本与实际运行的容器不一致

`start-8095.sh` 是「**带 Web UI 的变体**」，与**实际在跑的容器**有 4 处差异：

| 差异 | 脚本 | 实际容器 |
|---|---|---|
| `--shm-size 2g` | ✅ | ❌ |
| `-e LD_LIBRARY_PATH=/src/build/bin` | ✅ | ❌ |
| `-w /src` | ✅ | ❌ |
| `-v .../kvmem-ui-dist:/ui:ro` + `--ui-dir /ui` | ✅ | ❌ |
| **`--api-key`** | ❌ | ✅ |

**正确做法**：

```bash
# 永远以实际容器为准
docker inspect kvmem-test --format '{{json .Config.Cmd}}'
docker inspect kvmem-test --format '{{json .HostConfig.Binds}}'
```

本仓库的 [`scripts/run-8095-exact.sh`](../scripts/run-8095-exact.sh) 就是按 `docker inspect`
反推出来的**精确复刻脚本**。

### 坑 8 · 改了脚本文件，但容器没重建

这次踩得最实在的一个：

> `-c` 改的是**脚本文件**，但容器直到 **19:59:05** 才重建。
> 所以 19:25 那次 OOM 崩溃时，**跑的仍然是旧的 `-c 131072`**。

**正确做法**：改完配置后，**立刻验证容器里的真实参数**：

```bash
docker inspect kvmem-test --format '{{json .Config.Cmd}}'   # 看命令行
curl -s http://127.0.0.1:8095/slots -H "Authorization: Bearer $API_KEY" \
  | python3 -c 'import json,sys; print("n_ctx =", json.load(sys.stdin)[0]["n_ctx"])'
```

**「改了文件」≠「改了运行中的服务」。**

### 坑 9 · 压测会独占服务

该服务 `total_slots = 1` —— **压测期间其他客户端全部排队/失败**。

**正确做法**：跑压测前先确认空闲：

```bash
curl -s http://127.0.0.1:8095/slots -H "Authorization: Bearer $API_KEY" \
  | python3 -c 'import json,sys; print("busy =", json.load(sys.stdin)[0]["is_processing"])'
```

---

## 三、网络 / 工具链类坑

### 坑 10 · 本地代理会污染 curl

本机设置了 `http_proxy`（指向本地代理端口），访问内网服务会被代理拦下。

```bash
# ❌ 会被代理吃掉
curl -s http://127.0.0.1:8095/health

# ✅ 必须显式绕过
curl -s --noproxy '*' http://127.0.0.1:8095/health
```

### 坑 11 · Git Bash 下内联长命令 / 含引号的 `python3 -c` 会出错

Windows Git Bash 会把长 URL 和引号搞乱。

**正确做法**：

```bash
# 把脚本写成文件，再喂给远端
ssh gpu-host 'python3 -' < local_script.py
```

另外：**侦察脚本的段头只用 ASCII** —— 中文经 SSH 回传会变 GBK 乱码。

### 坑 12 · 本地命令超时会杀掉长任务

本地 Bash 工具有超时限制，长循环会被杀。

**正确做法**：**长任务一律后台运行**，不要在前台等。

---

## 四、测量类坑

### 坑 13 · `/tokenize` 端点不存在

该服务**没有** `/tokenize`。token 数只能：

| 方式 | 说明 |
|---|---|
| `usage.prompt_tokens` | 从**响应**里读（唯一准确来源） |
| 字符数 ÷ 4.5 | 英文粗估（仅用于构造测试 prompt） |

### 坑 14 · prefill 末尾有"隐形时间"

上下文超过 `budget + gen_reserve` 后，**prefill 末尾有数十秒不计入任何进度行**：

```
进度行在 170.9s 走完，prompt eval time 报 206.8s
→ 36s 的差额是真实存在的，但进度行看不到
```

**正确做法**：评估端到端耗时**用 `prompt eval time`**，不要用进度行的 t/s 反推。

### 坑 15 · 盯错内存指标

| 指标 | 行为 | 该不该盯 |
|---|---|---|
| **anon** | 顶到击杀线 → OOM | ✅ **要盯** |
| file（page cache） | **内核优先回收**，下降正常 | ❌ 下降不代表故障 |
| swap | 持续增长 = 危险信号 | ✅ 趋势要看 |

---

## 五、方法学总结

### 5.1 「三方对表」定位重启型故障

本次能快速定案，靠的是把三个独立来源对上：

```
容器 RestartCount（重启了几次）
        ×
日志启动指纹 "== CUDA ==" 的时刻（什么时候重启的）
        ×
客户端 502 的时间戳与 latency（用户什么时候感知到的）
```

**三者一一对应 → 因果链闭合。** 任何一环对不上，都要继续查。

> 关键：**要找一个"每次启动都会打印"的稳定指纹**（这里是 CUDA 环境横幅）。
> 有了它，`docker logs` 就能当重启时间线用。

### 5.2 「改动前后必须留回滚档」

```bash
docker inspect <容器> --format '{{json .Config.Cmd}}' \
  > "<repo>/rollback-$(date +%Y%m%d-%H%M%S).json"
cp -a start-8095.sh "start-8095.sh.bak-$(date +%Y%m%d-%H%M%S)"
```

在**远程生产服务**上改参数，这是**最低要求**，不是可选项。

### 5.3 「单一变量 + 同一 prompt」做 A/B

内存类改动的最大混淆变量是「重启清池」。
**只有用同一个 prompt 跑两次、比较斜率**，才能把结构性改善和重启效应分开。

```
斜率（KiB/token）= Δanon ÷ token 数
```

**这个指标不受基线影响**，是判断"每 token 成本"是否真的降了的唯一可靠指标。

### 5.4 「区分看起来占资源 vs 实际占资源」

| 服务 | 看起来 | 实际 |
|---|---|---|
| Obsidian（headless VNC，Xvfb+openbox+x11vnc） | 一堆进程，看着很占 | **RSS 合计仅 ~70 MiB** |
| KV 池（anon） | 一个数字 | **GiB 级，才是主角** |

**排查内存问题要先按量级排序，别按"进程数"直觉。**

### 5.5 「区分理论值与实测值」

| 项 | 理论 | 实测 |
|---|---:|---:|
| q5_0 每 token | 22 KiB | **27.1–29.0 KiB** |
| `-c 131072` 空载 | （曾误以为 ~6 GiB） | **245 MiB** |

**理论值用来定方向，实测值用来做决策。** 两者都要有，且要说明差异来源。

### 5.6 「读源码注释，别猜参数语义」

本次最大的方向性错误，源于**按字面意思理解 `--kvmem-budget`**（以为是"显存预算 MiB"）。

**实际语义写在源码注释里**：

```cpp
// kvmem_store.hpp
select_budget = 131072   // --kvmem-budget (semantic window tokens)
```

> **教训**：面对自研/小众项目，**参数语义必须回到源码或官方文档确认**，
> 字面联想 + 经验迁移是最容易翻车的地方。
>
> **另一类同源错误：把「惯用值」当成「默认值」。**
> 例如 `--kvmem-gen-reserve`，本机长期设 16384，久而久之就被当成默认值——
> 而 `--help` 里写的其实是 **256**（`decode slack (default 256)`）。
> **默认值只能从 `--help` 或源码读，不能靠印象。**

---

## 六、给下次的检查清单

**改动前**

- [ ] `docker inspect` 拿到真实 Cmd / Binds / 端口
- [ ] 留档旧 Cmd 与脚本备份
- [ ] 确认服务空闲（`is_processing = false`）
- [ ] 确认 V100 前提（FA 配置 / K-V 同类型）

**改动后**

- [ ] `docker inspect` 确认参数真的生效了
- [ ] `/slots` 确认 `n_ctx` 与预期一致
- [ ] **量 prefill 速度**（400–560 = FA 正常；<80 = 弃权）
- [ ] 采样 anon / avail / swap 趋势
- [ ] 记录 `RestartCount` 基线

**观察期**

- [ ] `RestartCount` 是否增加
- [ ] `system.slice/memory.events` 的 `oom_kill` 是否增加
- [ ] swap 是否持续增长
- [ ] 长会话下的累积斜率是否失控

---

## 七、故障速查表（症状 → 最可能原因 → 处理）

| 症状 | 最可能的原因 | 处理 |
|---|---|---|
| 客户端随机 502，服务日志**无报错** | **宿主 OOM 击杀**（静默 SIGKILL） | 查 `system.slice/memory.events`（**别查叶子 cgroup**）；按 [`FINDINGS.md`](../FINDINGS.md) 的 A7 顺序处置 |
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
