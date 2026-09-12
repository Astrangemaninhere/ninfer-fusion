# _COMPRESS_VRAM.md — 三项落实评估 (2026-09-03, 精度不压)

## 1. ninfer 格式无损压缩 (借鉴 llama.cpp, 精度不压)
实测 (qwen3.8-27b nvfp4-dflash2 artifact, 25.43GB 张量):
- token_embedding/output_head = 248320x5120, 1271.9MB 各:
  bytes = codes(1B/元素) + 每行 bf16 scale (2B/行, 已含在同一对象)
- 采样 64 行零行率 = 0 (本版 vocab 无填充行; 早期 fp8 NaN padding 已清)
- 4bit 码近均匀分布 -> 熵编码收益 ~0; scale 区占比 ~0.04%
结论 (诚实): 无损侧余量很小 (~1-3% 上限, 主要来自对齐/打包浪费与可能的
重复 scale); llama.cpp 的大头收益是 k-quant 有损, 用户禁 -> 本项收敛为:
  a) 布局级: 消除对齐填充 + scale 位宽核实 (e8m3 已是 1B) + 对象级去重
  b) 若未来允许有损: per-group scale / 3.5bit 混合 (k-quant 思路) 再议
工具: tools/convert/compress_probe.py (只读采样)

## 2. 按需显存 (不先占)
稠密 27B: decode 每步全层参与, 稳态无按需空间 -> 收益仅启动/闲置;
真实收益 = MoE 专家页粒度 (FlashNext 176B/6B 路线) + 多模型轮换
(闲置模型整模型 host 换出, byte 级镜像, 精度无损)。引擎侧挂冷机制同族。

## 3. 模型权重 OOM -> host 卸载 (KV 已有, 权重没有)
补法: 层/专家粒度 host 镜像 (byte 级精确副本) + 按需换入/换出,
复用 cold-host/disk 基建语义; 不重量化。实现属引擎 arena/加载层, GPU 批。
