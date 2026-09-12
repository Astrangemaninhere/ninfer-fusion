# FlashNext 契约审计 (快版, 语义同 flashnext_bindings.audit)

- checkpoint 键: 296475  |  契约条目: 74520
- 含通配的 alias 回退次数: 0 (其余走 set 查表)

## 方向 1: 契约条目 -> 源键 (缺失即引擎张量没有权重来源)
- 命中条目: 1 / 74520
- 未命中条目: 74519
  - token_embd
  - layer.0.attn_norm
  - layer.0.ffn_norm
  - layer.1.attn_norm
  - layer.1.ffn_norm
  - layer.2.attn_norm
  - layer.2.ffn_norm
  - layer.3.attn_norm
  - layer.3.ffn_norm
  - layer.4.attn_norm
  - layer.4.ffn_norm
  - layer.5.attn_norm
  - layer.5.ffn_norm
  - layer.6.attn_norm
  - layer.6.ffn_norm
  - layer.7.attn_norm
  - layer.7.ffn_norm
  - layer.8.attn_norm
  - layer.8.ffn_norm
  - layer.9.attn_norm
  - layer.9.ffn_norm
  - layer.10.attn_norm
  - layer.10.ffn_norm
  - layer.11.attn_norm
  - layer.11.ffn_norm
  ... 另有 74494 条

## 方向 2: 源键 -> 契约 (未消费即被静默丢弃)
- 已消费: 1 / 296475
- 未消费: 296474
  - .weight x75046
  - .weight_scale x73729
  - .input_scale x73728
  - .weight_scale_2 x73728
  - .bias x166
  - .A_log x36
  - .dt_bias x36
  - .layer_multipliers x1
  - .ngram_heads_offsets x1
  - .ngram_heads_vocab_sizes x1
  - .down_proj x1
  - .gate_up_proj x1
  e.g. model.language_model.embed_tokens.weight
  e.g. model.language_model.hyper_connection_mixer.hc_norm.weight
  e.g. model.language_model.hyper_connection_mixer.input_mix_weight_down.weight
  e.g. model.language_model.hyper_connection_mixer.input_mix_weight_up.weight
  e.g. model.language_model.layers.0.attn_hyper_connection.block_inject_weight.weight
  e.g. model.language_model.layers.0.attn_hyper_connection.hc_norm.weight
  e.g. model.language_model.layers.0.attn_hyper_connection.input_mix_weight_down.weight
  e.g. model.language_model.layers.0.attn_hyper_connection.input_mix_weight_up.weight
  e.g. model.language_model.layers.0.linear_attn.A_log
  e.g. model.language_model.layers.0.linear_attn.conv1d.weight
  e.g. model.language_model.layers.0.linear_attn.dt_bias
  e.g. model.language_model.layers.0.linear_attn.in_proj_a.weight
  e.g. model.language_model.layers.0.linear_attn.in_proj_b.weight
  e.g. model.language_model.layers.0.linear_attn.in_proj_qkv.weight
  e.g. model.language_model.layers.0.linear_attn.in_proj_z.weight
