# R13：周期性偏移（GDN）——已落一条真修复，余因锁定到 chunked 递推（2026-09-11 15:4X）

## 一、已落地：UP1_gdn_conv_column（真缺陷，此前未落）
`src/ops/gdn_input_proj/gdn_conv.cuh:116`
```diff
-            s2 = p;                                                  // 更宽的 FP32 累加器
+            s2 = __bfloat162float(__float2bfloat16_rn(p));            // 该列实际发布出去的 BF16 值
```
- 机理：GDN 短卷积的 3 宽滚动窗口 (s0,s1,s2) 里，**发布的列是 BF16，回带却用了 FP32 累加器**
  ⇒ 下一列回读一个"从未真正发布过"的值 ⇒ 每 token 重注入一次、随窗口周期性循环的累积误差。
- 这正好解释 `M_offset_probe/2` 的签名：**偏离固定在"生成序号"**（短 36/长 222 两档都偏在生成 36；
  英文档 37；中档全同）而**不随 prompt 长度/`max-new` 漂到相同绝对位置** ⇒ 与"解码步数"相关，
  与 KV 位置结构无关 ⇒ 循环状态（GDN）。
- 落地状态：dry-run/apply rc=0；`s2` 行已改；md5 `f9a21c5a9b54 → 3a3b445c9161`；
  备份 `/home/user/gdnconv_bak/gdn_conv.cuh.143109`；includer 5 个 TU 已 touch 并重编通过。
- 旁证：树里有 `gdn_conv.cuh.orig` ⇒ 这条补丁**以前打过又回退过**（值得记账）。

## 二、但偏移没有消失，且证据指向 GDN 的 chunked 递推
修复后同一 prompt（`dl/gdn_fix_verify.log`）：

| 运行 | 修复前 | 修复后 |
|---|---|---|
| `dflash2` vs plain | DIFFER at 29 | **DIFFER at 10**（数值变了 ⇒ 改动确实打到它的路径） |
| `mtp3` vs plain | DIFFER at 40 `[131978,3709,124554,…]` | **同一位置、逐位同值**（⇒ mtp 不走这个 epilogue） |
| 接受率 | dflash2 4.76% / mtp3 37.31% | dflash2 4.51% / mtp3 37.31% |

⇒ 该修复是**真缺陷、该落**，但漂移另有主因。主因签名（三条独立实测合起来）：
1. **与解码步数相关**（固定生成序号，不随 prompt 长度漂到同一绝对位置）⇒ 循环状态；
2. **与每轮吃多少列相关**：K=1→偏@62、K=3→偏@62、K=7→偏@29 ⇒ **每轮喂进去的列数越多，漂移越快**；
3. **两个草稿后端同症状**（mtp 与 dflash2 同位置同值；9-10 记录里也是 36/40 相邻）⇒ 共享的 target 侧状态。

⇒ **GDN 在 verify 里是"一次吃 W 列的分块递推"，在 plain 里是"一 token 一步的递推"**，两者数学等价、
数值不等价（分块 matmul/归约顺序 + BF16），误差逐步累积，直到在 margin 极小处翻 argmax。
这解释了全部签名，也解释了为什么可复制文本（margin 大）复现不出。

## 三、下一步（按性价比）
1. **定位 spec 解码实际走的 GDN kernel**（按权重格式 + token 数选择）：
   查 `src/ops/gdn_input_proj/{w8,nvfp4,q4_q5,fp8}/` 的 plan/dispatch，
   看 T=1（plain decode）与 T=W（verify）是否走同一 kernel、分块边界如何处理；
   若 T=W 走分块而 T=1 走递推 ⇒ 这就是"数值不等价"的落点。
   注意：mtp 的流未变，说明实际生效的不是 `GdnConvEpilogue` 那条 epilogue，别在错的 kernel 上打补丁。
2. **消除方式二选一**（取决于 1 的结论）：
   a) verify 的 GDN 也按 token 逐步推进（与 plain 完全同算术）⇒ 牺牲一点并行换 G-A 一致；
   b) 保持分块但在 BF16 累加处对齐（例如把分块中间量也走 BF16 发布，与 conv 那条同思路）。
3. **验收判据**：spec 流与 plain **逐位一致**（G-A），且接受率不低于修复前。
4. **不变的另一条线**：接受率上限仍是 ckpt 的口径问题（R12），retrain（`_train_df2_shift0.bat`）之后复测。
