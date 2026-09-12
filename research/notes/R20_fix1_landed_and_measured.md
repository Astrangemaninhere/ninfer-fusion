# R20：修① 落地实测（单变量验证 + 交叉验证）（2026-09-11 18:5X）

## 一、已落地
- **修②**：`src/ops/gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_plan.cpp:43` `tokens <= 3` → `<= 16`
  （备份 `/home/user/fix2_bak/`）；单变量实测**无任何变化** ⇒ 疑为 no-op（待 schedule 日志定案）。
- **修①**：`src/ops/gdn_input_proj/gdn_conv.cuh:99`
  `const float p              = __bfloat162float(__float2bfloat16_rn(projected[token]));`
  （备份 `/home/user/fix1_bak/gdn_conv.cuh`）；两路（A/B）逐字一致的共同核心。
- 构建：`export PATH=/home/user/.local/bin:$PATH && make ninfer -j2` rc=0，`apps/ninfer` 15:35。

## 二、单变量实测（同一套基准脚本，修复前 → ①+② 后）
| 指标 | 修复前 | ①+② 后 | 判读 |
|---|---|---|---|
| `zh plain vs dflash2` 首次偏离 | DIFFER at **10** | DIFFER at **19** | **后移 ✓ 修① 在预期路径上生效** |
| `zh plain vs mtp3` 首次偏离 | at 40 | at 40（同 token） | **完全未变 ✓ 与触发谓词一致** |
| chunk 探针（非对齐长 prompt） | DIFFER at 0 | DIFFER at 0 | ③ 未落地，符合预期 |
| chunk 探针（128 对齐） | IDENTICAL | IDENTICAL | 对照项 |
| dflash2 接受率 / 剖面 | 4.51% `[6,1,0,…]` | 3.76% `[5,0,0,…]` | ±1 token 量级（23 轮）⇒ 噪声内，**不能说提升** |
| mtp3 接受率 / 剖面 | 37.31% `[29,16,5]` | 37.31% `[29,16,5]` | 逐位未变 ✓ |
| decode tok/s（长 prompt） | ~70.9 | **71.16** | 无退化 ✓ |
| decode tok/s（zh） | （未留基线） | 44.78 | 作为后续对照值 |

### 交叉验证（值得记的一条）
mtp3 在所有指标上**逐位未变**，而 dflash2 的首次偏离后移——这正好对上修①的**触发谓词**
（fused conv 仅在 batch==1 且 **W∈{2,3,7..10}** 生效；dflash2 默认 W=8 命中，mtp3 的 W=4 不命中）。
⇒ 两路（A/B）给的谓词被独立实测证实 ✓。

## 三、诚实的边界
- 修① 让 spec 与 plain **更接近**（首次偏离 10 → 19），但**没有**让接受率上升；
  接受率上限仍受 ckpt 口径（R12/R18）与残余 ~1e-7 归约差（两路同判）约束。
- 修② 是否有效**未定**：需加一行 schedule 日志打印解析出的 `Nvfp4GdnConvScheduleId`，
  确认 dflash2 verify 的 GDN 输入到底走 snapshot / w4a4 / independent 哪条。
- 非对齐 prompt 的 chunk 不变量仍未达成（DIFFER at 0）：按 B 路判断，残余来自
  attention/MLP/lm_head 的 T 形状 kernel，**修③ 只能消掉 GDN 那一份**，不能单独达成 IDENTICAL。

## 四、下一步
1. 一行 schedule 日志给 ② 定案（决定它算不算收益）。
2. 修③ 评估：若目标是"chunk 无关性"，需一并处理其它 T 形状 kernel（工作量已知但更大）；
   若只求 spec==plain 尽量接近，③ 的边际收益已由 ① 覆盖一部分 ⇒ 先量化再决定。
3. 验收口径按 R19 §一"残余"栏：**同精度档 + 同 conv 语义 + 首次偏离后移/消失 + 性能不退化**。
4. 回到主线：ckpt 口径（retrain 的 `--out-dir` 坑与 R 判定）与 serve/CLI 混淆项。
