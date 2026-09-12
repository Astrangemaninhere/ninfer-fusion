# FlashNext 契约 vs 真实 checkpoint: 前缀规范化探针

- 真实键 296475, 契约条目 74520
- 前两段分布 Top6: [('model.language_model', 296110), ('model.visual', 333), ('mtp.layers', 24), ('mtp.hyper_connection_mixer', 3), ('lm_head.weight', 1), ('mtp.fc_embedding', 1)]

| 规范化 | 命中条目 | 消费键 |
|---|---|---|
| orig | 1 / 74520 | 1 / 296475 |
| strip_lm | 74174 / 74520 | 74210 / 296475 |
| drop_model | 1 / 74520 | 1 / 296475 |

- 真实 `.weight` 75047 个; NVFP4 四元组 (weight/scale/scale_2/input_scale) 齐全的 73728 个
- 契约条目里含 scale/quant 字样的: 0

## 判读
- 若 `strip_lm` 接近满覆盖 ⇒ 契约只差一层前缀规范化 (`model.language_model.`→`model.`);
  修法首选转换器入口规范化 (不动 74,520 条 alias), 其次才是逐条加前缀变体。
- NVFP4 伴生张量若不在契约里, 转换器必须自行推导/消费, 否则权重会被静默丢弃。
