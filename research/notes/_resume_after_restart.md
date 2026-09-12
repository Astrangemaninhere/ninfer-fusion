
## 重启前状态快照（2026-09-11 15:5X，用户要求重启 ZCode 以让子代理模型生效）
- **本轮已落地的引擎改动**（都在 WSL 构建树 `/home/user/ninfer-fusion`，均有备份/可回滚）：
  1. `src/ops/gdn_input_proj/gdn_conv.cuh:116` — UP1_gdn_conv_column（`s2` 回带 BF16 发布值），
     备份 `/home/user/gdnconv_bak/gdn_conv.cuh.143109`，md5 `f9a21c5a9b54 → 3a3b445c9161`；
     5 个 includer 已 touch 并重编通过（binary 14:35）。
  2. `dflash2_selector.cuh` — walk argmax 修复（F4，对比本 lane 自身分值）+ env-gated 探针
     （`NINFER_DF2SEL` / `NINFER_DF2_PAIR_SCALE`，默认不影响算术）。
  3. `speculative_options.h` — 解禁 dflash2 的 `ProposalHead::Optimized`；`dflash2_impl.h` 加两 route；
     三处 config.h 补 `dflash2::draft_head_rows`。
  4. 补丁留档：`_collab/F1_df2_draft_head.diff`、`F2/F3/F4`、`UP1_gdn_conv_column.diff`。
- **重启后的第一件事（按顺序）**：
  1. 定位 spec 解码实际走的 GDN kernel（按权重格式/ token 数分派）：`src/ops/gdn_input_proj/{nvfp4,q4_q5,w8,fp8}/*plan*`；
     已知 mtp 的流未受 `GdnConvEpilogue` 影响 ⇒ 别在错的 kernel 上打补丁。
  2. 判 "chunked(一次 W 列) vs 递推(一 token 一步)" 的数值不等价，并二选一对齐；验收 = spec 流与 plain 逐位一致。
  3. 保留项：selector 分数 FP32 化（参照保真）；接受率上限仍归 ckpt 口径（R12），retrain 后复测。
- **子代理**：`agents-state.json` 的两个 `builtInModelOverrides` 已改为
  `8060ff93-47f4-4298-841f-7561864c0cfe/deepseek-flash`（备份 `dl/agents-state.bak_20260911_091726.json`）；
  重启即为生效。重启后可派子代理做：GDN kernel 分派定位、8 项未落地清单、LABD 调研。
- 报告索引：R6–R13 在 `_collab/`（R12=引擎侧排除、R13=GDN 周期性偏移）。
