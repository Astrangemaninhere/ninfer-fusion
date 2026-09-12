# R15：扰动来源——形状相关的数值不等价（与投机无关，已实证）（2026-09-11 16:4X）

## 决定性实测（零编译、零投机）
同一长 prompt（60 段重复 ≈1200 token），只改 `--prefill-chunk`：
```
--prefill-chunk 128  : 生成流 A = [104980, 99943, ...]  (55 tok)
--prefill-chunk 4096 : 生成流 B = [103735, 95852, ...]  (31 tok)
⇒ **DIFFER at 第 0 个生成 token**
```
⇒ 本引擎**同一数学在不同形状 kernel 上数值不等价**，且差异足以翻转**第一个** token 的 argmax。
这条与投机解码无关，是引擎的既有性质。

## 意义（回答"随机扰动从哪来"）
1. plain 解码走 T=1 形状（GEMV/单列注意力/递推 GDN），verify 走 T=W=8 形状（MMA/分块）——
   与上面 chunk 128 vs 4096 属**同一类形状切换** ⇒ spec≠plain 的漂移有**结构性来源**，
   不能只归因于 ckpt 或纯浮点抖动。
2. 与今天落地的 `UP1_gdn_conv_column` 同族：那个滚动窗口正是**跨 chunk / 跨块的状态携带**机制。
   本次实验是在**已打该补丁之后**跑的、仍然不等价 ⇒ **还有边界状态未精确携带**
   （GDN snapshot、conv 窗口在 chunk 边界的处理、可能的 rope/位置口径）。

## 给 S_A / S_B 的新证据（已同步给他们）
- 判据从"首次翻转点"改为：**同一 (上下文, 位置) 下两条形状路径的 logits 差距**；
- 现成的、无需编译的复现：`--prefill-chunk 128` vs `4096` 同 prompt 首 token 即分叉
  （脚本 `_prefill_chunk_equiv.sh`，日志 `dl/prefill_chunk_equiv.log`）；
- 请重点回答：**哪些状态是"跨形状边界"携带的**（chunk 边界、block 边界），各自是否按
  "已发布（BF16/量化后）的值"还是"更宽的累加器"携带——后者就是 `s2 = p` 那类 bug 的通式。

## 主代理下一步（我的车道）
1. 把 `_prefill_chunk_equiv.sh` 的判据升级为 **logits 级**：同 (上下文, 位置) 下比较
   两条形状路径的 top-8（差距 <1e-2 且只差一位 ⇒ 纯数值；差距大 ⇒ 结构性 bug）。
2. 回退验证：临时 `patch -R` 今天那条 conv 补丁，重跑 chunk 实验，量化该补丁对"形状不等价"的贡献
   （备份 `/home/user/gdnconv_bak/`，可干净回退）。
3. 与 S_A/S_B/S_C/S_D 合并后定"改算术 vs 改状态携带"。
