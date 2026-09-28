#!/usr/bin/env python3
"""解析 GGUF 元数据头，并推算 KV 缓存的「每 token 成本」。

用法:
    python3 gguf_kv_math.py [模型路径]

不加载权重，只读文件头（纯标准库，无需 torch / gguf）。

输出:
    - 全部元数据键值
    - 模型几何（层数 / 头数 / head_dim）
    - 各精度（f16 / q8_0 / q5_0 / q4_0）下的每 token 成本
    - 常见上下文长度（64K / 96K / 128K）下的 KV 总量

推导依据（见本仓库 FINDINGS.md 的 A5 节）:
    每 token 元素数 = n_head_kv × (key_length + value_length) × 全注意力层数
    f16   = 2 B/元素
    q8_0  = 1.0625 B/元素
    q5_0  = 0.6875 B/元素
    q4_0  = 0.5625 B/元素
"""
import struct
import sys

DEFAULT_PATH = "/llama/models/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf"

WANT = ("general.architecture", "general.name", "general.size_label",
        "block_count", "attention.head_count", "attention.head_count_kv",
        "embedding_length", "context_length", "attention.key_length",
        "attention.value_length", "attention.layer_norm_rms_epsilon",
        "rope.freq_base", "expert_count", "expert_used_count",
        "attention.sliding_window", "attention.sliding_window_pattern",
        "full_attention_interval", "nextn_predict_layers", "vocab_size",
        "ssm.state_size", "ssm.inner_size", "ssm.group_count", "head_dim")

# GGUF 值类型 -> (struct 格式, 字节数)
T = {0: ("B", 1), 1: ("b", 1), 2: ("H", 2), 3: ("h", 2), 4: ("I", 4), 5: ("i", 4),
     6: ("f", 4), 7: ("?", 1), 10: ("Q", 8), 11: ("q", 8), 12: ("d", 8)}

BYTES_PER_ELEM = {"f16": 2.0, "q8_0": 1.0625, "q5_0": 0.6875, "q4_0": 0.5625}


def rd(f, fmt, n=1):
    sz = struct.calcsize("<" + fmt)
    d = f.read(sz * n)
    if len(d) < sz * n:
        raise EOFError
    v = struct.unpack("<" + fmt * n, d)
    return v[0] if n == 1 else v


def rdstr(f):
    n = rd(f, "Q")
    return f.read(n).decode("utf-8", "replace")


def rdval(f, t):
    if t == 8:                      # string
        return rdstr(f)
    if t == 9:                      # array
        et = rd(f, "I")
        n = rd(f, "Q")
        if et == 8:
            vals = [rdstr(f) for _ in range(n)]
        else:
            fmt, _ = T[et]
            vals = list(rd(f, fmt, n))
        if len(vals) > 8:
            return f"<array {n} x {et}> {vals[:8]} ..."
        return vals
    if t not in T:
        raise ValueError(f"bad type {t}")
    fmt, _ = T[t]
    return rd(f, fmt)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH

    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"GGUF":
            print(f"不是 GGUF 文件: {magic!r}", file=sys.stderr)
            return 1

        ver = rd(f, "I")
        nten = rd(f, "Q")
        nkv = rd(f, "Q")
        print(f"file        = {path}")
        print(f"GGUF version={ver}  tensors={nten}  kv_pairs={nkv}")

        arch = None
        found = {}
        for _ in range(nkv):
            k = rdstr(f)
            t = rd(f, "I")
            v = rdval(f, t)
            if k == "general.architecture":
                arch = v
            found[k] = v

        print(f"architecture= {arch}")
        print()
        print("--- 关键元数据 ---")
        for k in sorted(found):
            if k.split(".")[-1] in WANT or k.startswith("general."):
                print(f"  {k} = {found[k]}")

        # ---- KV 算术 ----
        a = arch
        nl = found.get(f"{a}.block_count")
        nh = found.get(f"{a}.attention.head_count")
        nhkv = found.get(f"{a}.attention.head_count_kv")
        ne = found.get(f"{a}.embedding_length")
        kl = found.get(f"{a}.attention.key_length")
        vl = found.get(f"{a}.attention.value_length")
        fai = found.get(f"{a}.full_attention_interval")

        if not (nl and nh and ne):
            print("\n[KV] 元数据不足，无法推算（缺 block_count / head_count / embedding_length）")
            return 0

        hd = kl or (ne // nh)
        vl = vl or hd
        nhkv = nhkv or nh

        # 全注意力层数：有 full_attention_interval 时按间隔算，否则视为全部层
        n_attn = (nl // fai) if fai else nl

        print()
        print("--- 模型几何 ---")
        print(f"  n_layer={nl}  n_head={nh}  n_head_kv={nhkv}  "
              f"head_dim={hd}  v_dim={vl}  n_embd={ne}")
        if fai:
            print(f"  full_attention_interval={fai}  → 全注意力层 ≈ {n_attn} 层"
                  f"（其余为循环层，KV 与上下文无关）")

        per_tok_elems = n_attn * nhkv * (hd + vl)
        print()
        print("--- KV 每 token 成本 ---")
        print(f"  每 token 元素数 = {n_attn} 层 × {nhkv} kv_head × ({hd}+{vl}) = {per_tok_elems}")
        for name, bpe in BYTES_PER_ELEM.items():
            kib = per_tok_elems * bpe / 1024
            print(f"  {name:5s} = {kib:7.1f} KiB/token")

        print()
        print("--- 各上下文长度下的 KV 总量（按上表）---")
        for n in (65536, 98304, 131072, 262144):
            parts = "  ".join(
                f"{name}={per_tok_elems * bpe * n / 2**30:6.2f} GiB"
                for name, bpe in BYTES_PER_ELEM.items())
            print(f"  ctx={n:7d}  {parts}")

        print()
        print("提示: 实测值通常略高于理论值（计算缓冲 / 页表等开销）。")
        print("      本仓库实测 q5_0 为 27.1–29.0 KiB/token（理论 22 KiB）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
