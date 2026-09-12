# R7：dflash2 链式塌陷的最终定位（2026-09-11 14:0X）

## 结论（三段，按可信度排序）

### 1. 引擎侧发现并修复了两个真缺陷（都已落地并通过验证）
- **F4（实质缺陷）**：`dflash2_selector_walk_kernel` 的 argmax 是坏的。
  `value` 被 `fmaxf` 归约**覆盖之后**才与 `best` 比较，再以 `min(lane)` 破平 ⇒ lane 0
  归约后必然持有全局最大 ⇒ `equal` 恒真 ⇒ `chosen` **恒为 0**（= unary 的 argmax）。
  后果：契约要求的「coherent path walk」退化为「逐步取 unary argmax」，**训练好的边项被整个丢弃**。
  修复后探针直接可见机制恢复：`chosen == Earg`（修前恒 `Earg≠chosen=0`），
  `pred` 真的在走链（0→2→14→1→2→11→0），草稿不再是同一 token 重复串。
- **F1（契约/上游不一致，非本症根因）**：dflash2 未接 checkpoint 的专用草稿头
  （`text/draft_head [131072,5120]` + `text/draft_head_token_ids`，契约 §6.1:251-254），
  且 CLI 层硬性拒绝（`speculative_options.h:59-61`，上游同处无此限制）。
  已实现两条 route 并实测生效（737 vs 735 张量、43/87 轮草稿变化），但**剖面纹丝不动** ⇒ 不是根因。

### 2. 残差在草稿 checkpoint 自身，不在引擎（本轮最强实证）
- **边项权重扫描**（`NINFER_DF2_PAIR_SCALE`，同一 prompt）：

  | pair 权重 | 接受率 | AL | 位置剖面 |
  |---|---|---|---|
  | 1.0（契约值） | 4.76% | 1.33 | `[6,1,0,0,0,0,0]` |
  | 0.25 | 4.97% | 1.35 | `[8,0,0,0,0,0,0]` |
  | 0.0（整段关掉） | 4.97% | 1.35 | `[8,0,0,0,0,0,0]` |

  ⇒ 边项**不携带可用信息**（全权重略差于关掉），所以"标度修一修就好"被证伪。
- **unary 很平**：探针实测 top-16 的 `uspan` 仅 ~1.5–2.75，而边项 `pairspan` 5–18；
  在可复制文本（数字模式）上 unary 却 6/7 列正确。
- 与 `_collab/A2_draft_ceiling.md` 的既有诊断吻合：**该 ckpt 的训练配方喂真 token，
  推理喂 mask 块**（mask id 在训练语料 0/4468 出现，见 A2）⇒ 模型只会"泛化续写"，
  不会"逐位置具体预测"。这解释了：复制文本可用、散文下 unary 平、边项无信息、位置 ≥1 全灭。
- 权重真伪已排除：artifact 由 `patch_dflash2.py --ckpt data\dflash2_ckpts\step_006000.pt`
  生成，其 verify 闸门要求 `replaced tensors ok: 73 match checkpoint` 且 round-trip
  `BIT-EXACT`（`_collab/C_artifact_manifest.md:15-30`）⇒ 73 张草稿权重是真的、逐字节一致。
  本机 HF 目录 `Qwen3.8-27B-NVFP4-RTX5090` 只含 target（无任何草稿张量），
  所以"用 HF 对照草稿"不可行；对照基准应是**我们自己的训练 ckpt**。

### 3. 目标侧完全健康（本条贯穿始终，未变）
blame 对齐分析（`_df2_blame2.py`）：列 0 target argmax 命中真值 **17/20 = 85%**，
在引擎已接受前缀上列 1 命中 **6/8 = 75%**（参考 ~82%）⇒ verify 块的位置/KV/特征捕获/注意力
全部正常，`valid_columns` 语义、第 6 参、4.x 号差异等嫌疑全部排除。
KV 精度不是自变量：引擎 summary 实测 `kv cache dtype = bf16`，显式 `--kv-dtype int8` 剖面同形。

## 今天被证伪的假设（连同证据）
| 假设 | 结论 | 证据 |
|---|---|---|
| 用错 proposal head（应走草稿头） | **非根因**（但契约缺口真实，已修） | F1 实测两 route 剖面相同；1Cat 参考实现只用 target LM head（`qwen3_dflash2.py:456-509`） |
| 边项标度失配 | **证伪** | scale 扫描 1.0/0.5/0.25/0.1/0.0 全在 4.5–5.0% |
| artifact 草稿权重被转换改坏 | **证伪** | 转换链自带 `73 match checkpoint` + `BIT-EXACT` 闸门 |
| KV int8 污染 | **证伪** | 实测 bf16，且 int8 同形 |
| `verify[c+1]==draft[c]` 是"发现" | **tautology** | 那是块布局本身；真正该看的是 `argmax[c] vs draft[c]` |

## 落地补丁（全部有 diff + 备份，可 `patch -R` 回滚）
| # | 文件 | 内容 | 备注 |
|---|---|---|---|
| F1 | `_collab/F1_df2_draft_head.diff` | 草稿头两 route + CLI 解禁 + selector 的 id 映射 | 8 文件 +86/−32；备份 `/home/user/df2head_bak` |
| F2 | `_collab/F2_df2_scale_probe.diff` | selector walk 的 `E/u/pair/uspan` 探针 | `NINFER_DF2SEL=1` 才生效；输出走 **stdout** |
| F3 | `_collab/F3_df2_pair_spread.diff` | 追加 `pairspan/Earg/Uarg` | 同上 |
| F4 | `_collab/F4_df2_walk_argmax_fix.diff` | **修 walk argmax（对比本 lane 自身分值）** | 本轮实质修复 |
| F5 | `_run_scale_probe.sh` 系列 | `NINFER_DF2_PAIR_SCALE` 运行时权重旋钮 | 默认 1.0，与契约算术一致 |

变异体常量：`DFlash2Config::draft_head_rows = 131072` 已补进 27b/35b/Muse 三处 config.h
（共享头 `dflash2_impl.h` 需要每个 variant 都有）。

## 下一步（唯一有意义的路径）
1. **恢复续训**：`_train_df2_shift0.bat`（mask 输入 + `--target-shift 0`，即 A2 认定正确的配方），
   产出新的 `step_XXXXXX.pt` 后按 `_collab/C_artifact_manifest.md` 的 patch→verify→round-trip
   三闸门出新 artifact。
2. 用现成仪器复测（无需再写代码）：
   - `NINFER_DF2DBG=1` + `_df2_blame2.py` → 目标侧/草稿侧 blame 与列剖面
   - `NINFER_DF2SEL=1` → `pairspan` 是否变为"有信息"（应显著影响 `Earg≠Uarg` 且 token 合理）
   - `_verify_df2head.sh` → 接受率 + G-A（生成 token 流逐位一致）
3. 验收判据：位置剖面出现链式（p1+ 非零）、AL ≥ 3、G-A 保持 IDENTICAL。
   若新 ckpt 下 `pairspan` 仍无信息，则应回头查 mask 块/context K-V 的喂法（当前唯一未证伪的
   引擎侧残余项），并用 F5 旋钮做 A/B。
