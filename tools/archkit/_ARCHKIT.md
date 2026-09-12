# ARCHKIT — ninfer 通用架构接入框架 (设计)

目标: 让"加一个新架构的模型"(DeepSeek v4 flash、Kimi K3、FlashNext/Qwen4Exp …)
像填表一样完成 —— **先写这套框架, 再用它去接 FlashNext**, 证明通用性的同时
保持专用效率。

## 0. 核心事实: ninfer 已经"通用内核 + 专用薄壳"

qwen3_6 家族 runtime 就是范本:
- `src/targets/qwen3_6/impl/runtime/*` — 共享引擎: paged-KV、swa/softmax 注意力、
  decode/prefill 调度、采样、投机验证、arena、图缓存 —— **与具体模型无关的机器**。
- 每个 exact 变体 (qwen3_6_27b / qwen3_6_35b_a3b / 未来的 qwen4_exp …) 只提供:
  `TextConfig`(constexpr 几何) + `Variant` 叶子函数(哪些层全注意力/哪些 GDN、
  projection 拆分、norm 口味) + 权重绑定表 + 转换 recipe。

所以"支持新架构" = 回答 6 个问题 + 生成 6 块薄代码, 其余全部复用共享内核。
通用性来自共享内核; 效率来自 constexpr 几何 + 叶子函数(编译期闭合, 零虚开销)。

## 1. 架构规格文件 (arch-spec.json) — 填表入口

```jsonc
{
  "model_id": "qwen4-exp",                  // 注册身份 (registry + artifact identity)
  "family": "qwen3_6",                      // 复用哪个共享 runtime (或 "standalone")
  "hf": { "model_type": "qwen4exp", "architectures": ["Qwen4ExpForCausalLM"],
          "revision": "..." },
  "geometry": {                             // -> TextConfig constexpr
    "hidden": 2560, "layers": 48, "vocab": 152064, "max_ctx": 131072,
    "query_heads": 24, "kv_heads": 2, "head_dim": 256
  },
  "layer_types": ["linear_attention", ...], // 每层口味 (full|gdn, 归一化自动)
  "attention": {...}, "mlp": {...}, "moe": {...}, "gdn": {...},
  "ple": {...}, "mtp": {...}, "indexer": {...}, "vision": {...},
  "weights": { "hf 键": "引擎语义名" },      // 权重映射 (bindings + recipe)
  "quirks": [ "no_bias", "qk_norm", "multimodal:vision" ]
}
```

## 2. 六个问题 (调研每个新架构时逐项回答)

| 问题 | 决定什么 | 若共享内核没有 → |
|---|---|---|
| 1 几何/层型 | TextConfig constexpr | 纯填表 (arch_spec.py extract 自动) |
| 2 注意力口味 (GQA/MLA/线性/滑动窗) | 用哪个注意力核 | 新增一个 attention leaf |
| 3 MLP 口味 (稠密/MoE+router) | 是否走 MoE 路径 | MoE 核 + expert SSD offload 已是规划项 |
| 4 权重键映射 | bindings + converter recipe | 纯填表 (recipe 半自动生成) |
| 5 数值/量化 (bf16/nvfp4/groupwise) | 走哪套 quantize 管线 | 已有三套, 选择即可 |
| 6 投机/草稿能力 (mtp/dflash/ngram/ple) | 注册哪些 draft 后端 | MTP 已通用; 查表/ngram 独立实现 |

## 3. 生成器产物 (tools/archkit/gen_target.py)

v1 (2026-09-03 已落地, g++ 语法验证通过):
1. `config.h` — TextConfig/MoE/GDN/PLE/MTP constexpr (spec → 常量直译)
2. `arch_manifest.json` — 机器可读清单 (GUI 白名单/registry/转换器输入)

v2 (待做, 对着真实 runtime API 回填):
3. `impl/variant.h/.cpp` — 叶子函数: 口味枚举 → 共享 ops 调用
4. `impl/load/bindings.cpp` — weights 表展开
5. `impl/package.cpp` + `include/.../package.h` — identity 注册
6. `tools/convert/<variant>/recipe_*.py` — HF 键提取骨架
7. GUI/导入向导自动消费 manifest (白名单改读清单)

## 4. 先用它接 FlashNext (qwen4-exp), 证明方法论

specs/qwen4_exp_spec.json 已从 fn_official.json 提取 (text_config 几何):
176B/6B active MoE、512 experts ×10/tok @640、shared 640、hidden 2560、48 层
(12 full + 36 linear_attention)、24q/2kv/hd256、vocab 248320、ctx 262144、
PLE ngram 3×8 @vocab-base 20M (conv 4, layer 2)、MTP 1、indexer 稀疏头部。

验证顺序: spec → 生成 → 转换跑通(离线 CPU) → 引擎 smoke → 性能后置。

## 5. 通用性与效率如何兼得

- 通用: 共享 runtime 机器 (paged KV/调度/sampling/图缓存) 对一切文本生成架构成立;
  转换 recipe + 权重映射表由 spec 驱动, 新架构不再手写加载器。
- 专用: 几何与口味全部 constexpr/枚举编译期闭合 (variant-template 已在用);
  注意力/MLP 核按需选型, 不存在"为通用而通用"的解释层。
- 边界: 每个新"口味"第一次出现时仍要写一个 leaf 内核 —— 但只写那一个,
  之后同口味架构全部免费。

## 6. 执行顺序 (下一阶段)

1. ✅ gen_target.py + arch_spec.py + qwen4-exp spec + 生成物语法验证
2. v2 stub 生成器 (bindings/package/recipe, 抄 qwen3_6_27b 回填)
3. hf 键核对 (拉 Qwen4Exp 真实 config/safetensors index 验证 weights 表)
4. 共享-expert 版 recipe 转出最小可跑 artifact → serve smoke (性能后置)
5. DeepSeek v4 flash / Kimi K3: 同流程 (config -> spec), 稀疏注意力经验
   见 RESEARCH-FREETOKEN.md
