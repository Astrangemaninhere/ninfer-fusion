#!/usr/bin/env python3
"""导入器第二轮修正（S54 的只读分析 + 我逐条核实后）：
  1) **阻塞级**：生成物里的 C++ 命名空间带小数点（`spark_x2.5_4b`）编不过 —— 加 cpp_ident 归一化；
  2) `attn:headwise_output_gate` 由 hook 改 **new_op**：`sigmoid_mul` 要求四维同形（实测
     `wrapper/sigmoid_mul.cpp:36-40`），HF 的 `[n_q,T]` 逐头广播没有算子；零算子出路是复制 g_proj 权重（给体积）；
  3) `mlp:act=gelu`（带门控）由 hook 改 **new_op**：`ops::gelu` 是 in-place 单元激活，
     `gelu(gate)*up` 的两输入乘全树无算子（ops 目录只有 silu_mul/sigmoid_mul/gelu/causal_conv1d_silu）；
  4) 新增 `attn:head_geometry` 探测器：**解析引擎几何注册表**（`gqa_attention_geometry.cuh` 的
     `GqaGeometry<...>` 实例）比对 (q,kv,head_dim)，没注册就是 new_op（Spark 16Q/4KV@256 正是如此；
     `wrapper/gqa_attention.cpp` 还会把 16Q 当成 2 KV 再抛错）；
  5) 新增 `attn:qk_norm=absent` 探测器：这些配置没有 qk-norm，而家族无条件做 `rmsnorm(q,k)` ⇒ hook（加配置门）。
"""
import pathlib
import re
import sys

P = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/ninfer-fusion-repo/tools/archkit/adapt.py")
src = P.read_text(encoding="utf-8")

# ---------------- (1) C++ 标识符归一化 ----------------
A1 = "        'namespace ninfer::targets::%s::detail {' % spec['model_id'].replace('-', '_'),"
B1 = ("        # C++ identifiers cannot carry dots: a model id like `spark-x2.5-4b` used to\n"
      "        # emit `namespace ninfer::targets::spark_x2.5_4b::detail` and the header could\n"
      "        # never compile (found by S54 against the real Spark artifact).\n"
      "        'namespace ninfer::targets::%s::detail {' % cpp_ident(spec['model_id']),")
if "cpp_ident(spec['model_id'])" in src:
    print("命名空间归一化已存在")
else:
    if src.count(A1) != 1:
        print("锚点1 不唯一/缺失: %d" % src.count(A1)); sys.exit(2)
    src = src.replace(A1, B1)
    print("已修：命名空间用 cpp_ident()")

# 定义 cpp_ident（放在 extract_spec 之前）
A1b = "def extract_spec(config_path, model_id):"
B1b = ('def cpp_ident(model_id: str) -> str:\n'
       '    """模型 id -> C++ 标识符：非 [A-Za-z0-9_] 一律折成下划线，首字符不能是数字。"""\n'
       '    s = re.sub(r"[^0-9A-Za-z_]", "_", str(model_id))\n'
       '    s = re.sub(r"_{2,}", "_", s).strip("_") or "model"\n'
       '    return s if not s[0].isdigit() else "m_" + s\n'
       '\n'
       '\n'
       'def extract_spec(config_path, model_id):')
if "def cpp_ident(" in src:
    print("cpp_ident 已定义")
else:
    if src.count(A1b) != 1:
        print("锚点1b 不唯一: %d" % src.count(A1b)); sys.exit(2)
    src = src.replace(A1b, B1b)
    print("已加 cpp_ident() 定义")

# ---------------- (2) 逐头输出门：hook -> new_op ----------------
A2 = """    if knobs.get('headwise_attn_output_gate'):
        mode = knobs.get('gate_attn_act_mode') or 'sigmoid'
        out.append(('attn:headwise_output_gate(%s)' % mode,
                    'hook', '逐头注意力输出门控 (linear + sigmoid_gate_mul 已存在，需 leaf 接线)'))"""
B2 = """    if knobs.get('headwise_attn_output_gate'):
        mode = knobs.get('gate_attn_act_mode') or 'sigmoid'
        g0 = spec.get('geometry') or {}
        # 零算子出路：把 g_proj 的每一行复制到该 head 的每个通道上，让元素级 sigmoid_mul 等价于
        # 逐头广播门控；代价是权重体积膨胀（BF16 约 4x，NVFP4 约 8x 于 n_q*hidden）。
        rep_mb = (g0.get('query_heads', 0) * g0.get('hidden', 0) * 2 * 3) / 1e6
        out.append(('attn:headwise_output_gate(%s)' % mode,
                    'new_op',
                    '逐头注意力输出门控: sigmoid_mul 要求四维同形 (wrapper/sigmoid_mul.cpp:36-40), '
                    '逐头广播无算子; 零算子出路=复制 g_proj 权重(约 +%.0f MB/层) 或新内核' % rep_mb))"""
if "'new_op',\n                    '逐头注意力输出门控" in src:
    print("门控分级已修")
else:
    if src.count(A2) != 1:
        print("锚点2 不唯一: %d" % src.count(A2)); sys.exit(2)
    src = src.replace(A2, B2)
    print("已修：逐头输出门 -> new_op")

# ---------------- (3) gated gelu MLP：hook -> new_op ----------------
A3 = """    elif act in ('gelu', 'geglu'):
        out.append(('mlp:act=%s' % act, 'hook', 'ops::gelu 已存在，需 MLP 前向选择激活'))"""
B3 = """    elif act in ('gelu', 'geglu'):
        gated = bool(knobs.get('headwise_attn_output_gate')) or True  # gate/up 结构本身就带门控乘
        out.append(('mlp:act=%s%s' % (act, '(gated)' if gated else ''),
                    'new_op',
                    'ops::gelu 是 in-place 单元激活, gelu(gate)*up 的两输入乘全树无算子 '
                    '(ops 下只有 silu_mul/sigmoid_mul/gelu/causal_conv1d_silu) ⇒ 需新融合内核'))"""
if "两输入乘全树无算子" in src:
    print("MLP 激活分级已修")
else:
    if src.count(A3) != 1:
        print("锚点3 不唯一: %d" % src.count(A3)); sys.exit(2)
    src = src.replace(A3, B3)
    print("已修：gelu MLP -> new_op")

# ---------------- (4)(5) 新探测器：几何注册表 + qk_norm 缺失 ----------------
A4 = """    if knobs.get('attention_qk_norm') or knobs.get('qk_norm'):
        out.append(('attn:qk_norm', 'hook', 'q/k norm leaf'))"""
B4 = """    # 头几何：直接解析引擎的注册表，而不是猜。注册表在
    # src/ops/kernel/gqa_attention_geometry.cuh（GqaGeometry<Q,KV,Split[,D]> 的 using 别名）。
    g0 = spec.get('geometry') or {}
    q, kv, hd = g0.get('query_heads'), g0.get('kv_heads'), g0.get('head_dim')
    reg = engine_geometry_registry()
    if q and kv and hd:
        if (q, kv, hd) in reg:
            out.append(('attn:head_geometry(%dq/%dkv@%d)' % (q, kv, hd),
                        'covered', '已注册: %s' % reg[(q, kv, hd)]))
        else:
            out.append(('attn:head_geometry(%dq/%dkv@%d)' % (q, kv, hd),
                        'new_op',
                        '引擎几何注册表未收 (现有 %s); wrapper 还会按 q_heads 反推 KV 头 '
                        '(wrapper/gqa_attention.cpp:25-30: 16->2) 再抛错'
                        % ', '.join('%d/%d@%d' % k for k in sorted(reg))))
    if knobs.get('attention_qk_norm') or knobs.get('qk_norm'):
        out.append(('attn:qk_norm', 'hook', 'q/k norm leaf'))
    elif q and not (knobs.get('attention_qk_norm') or knobs.get('qk_norm')):
        # 家族无条件 rmsnorm(q,k)，这些配置没有 qk-norm ⇒ 需要配置门才不静默出错
        out.append(('attn:qk_norm=absent', 'hook',
                    '家族无条件 rmsnorm(q,k) (text_context_impl.h:958-959), 本模型无 qk-norm ⇒ 需 qk_norm_enabled() 门'))"""
if "engine_geometry_registry" in src:
    print("几何探测器已存在")
else:
    if src.count(A4) != 1:
        print("锚点4 不唯一: %d" % src.count(A4)); sys.exit(2)
    src = src.replace(A4, B4)
    print("已加：几何注册表 + qk_norm=absent 探测器")

# 注册表解析函数
A5 = "def catalog_gaps(spec):"
B5 = ('def engine_geometry_registry() -> dict:\n'
      '    """解析引擎几何注册表: { (q_heads, kv_heads, head_dim): alias }。\n'
      '    读不到就返回空 dict（探测器会退化为"未注册"），不让 IO 失败打断报告。"""\n'
      '    path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(\n'
      '        os.path.abspath(__file__)))), "src", "ops", "kernel",\n'
      '        "gqa_attention_geometry.cuh")\n'
      '    out = {}\n'
      '    try:\n'
      '        text = open(path, encoding="utf-8", errors="replace").read()\n'
      '    except OSError:\n'
      '        return out\n'
      '    for m in re.finditer(r"using\\s+(\\w+)\\s*=\\s*GqaGeometry<\\s*(\\d+)\\s*,\\s*(\\d+)\\s*,\\s*(\\d+)'
      '(?:\\s*,\\s*(\\d+))?\\s*>", text):\n'
      '        alias, qh, kvh, _split, hd = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4), m.group(5)\n'
      '        out[(qh, kvh, int(hd) if hd else 256)] = alias\n'
      '    return out\n'
      '\n'
      '\n'
      'def catalog_gaps(spec):')
if "def engine_geometry_registry(" in src:
    print("注册表解析器已存在")
else:
    if src.count(A5) != 1:
        print("锚点5 不唯一: %d" % src.count(A5)); sys.exit(2)
    src = src.replace(A5, B5)
    print("已加 engine_geometry_registry()")

P.write_text(src, encoding="utf-8")
print("adapt.py 现在 %d 行" % (src.count("\n") + 1))
