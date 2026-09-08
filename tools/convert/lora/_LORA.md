# _LORA.md — LoRA 支持设计 (TeichAI/Qwen3.8-27B-Fable-Distill-LoRA 实测锚定)

## 路线: 离线融合 (非运行时 LoRA)
引擎权重 NVFP4/行量化 -> 运行时加 delta 需每层解量化基座, 代价高;
离线融合 = 逐层 解量化 -> W'=W+s(B@A) -> 重量化 -> 流式写回 .ninfer
(patch_dflash2 同款 ArtifactWriter), 运行时零改动, 多 adapter 各出一个 artifact。

## 实测 adapter 结构 (2026-09-03, 头文件分析, 992 张量)
- r=32, alpha 缺省(=r, scale 1), CAUSAL_LM, peft 0.19.1
- 键前缀: base_model.model.model.language_model.layers.N.<module>
- 覆盖 = 64 层全量, 与引擎层型完全对齐:
  * self_attn.{q,k,v,o}_proj    -> 16 全注意力层 (独立投影!)
  * linear_attn.{in_proj_qkv, in_proj_a, in_proj_b, in_proj_z, out_proj}
                                 -> 48 GDN 层
  * mlp.{gate,up,down}_proj      -> 64 层
- A [32, 5120], B [out, 32]; out 样例: in_proj_a -> 48

## 布局映射 (lora_merge.py 分类器已用真实键验证)
kind -> 引擎对象 (对象名 TODO: 对照 bindings.cpp 回填; q/k/v 独立 -> 引擎
如为融合 qkv 需行拼接 q|k|v; GDN in_proj_qkv/a/b/z 对应 gdn_input_projection
的 q/k/v/output_gate 行序与 conv a/b —— 需核对 bindings 行切分)

## 工具
tools/convert/lora/lora_merge.py:
  --adapter-dir <dir> --list-groups | --keys (只读头, 截断文件可用)
  实测: gdn_qkv 48 / qkv o 16 各 / gate up down 64 各 / layers 0-63 ✓
TODO: 完整下载 (~934MB) 后: 解量化基座 (复用 fp8 行 scale 反量化) ->
delta 行拼接 -> 重量化 -> ArtifactWriter 出新 artifact; 数值验收
(ppl/对话冒烟 vs 未融合)。

## GUI
导入向导识别 adapter_config.json (peft_type=LORA) -> kind=lora 提示
"这是 LoRA 微调包: 需要对应基座模型 + 离线融合" (model_import 待加)
