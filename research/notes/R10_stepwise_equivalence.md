# R10：逐步（逐列）等价性体检结论 + 下一步探针（2026-09-11 14:5X）

## 一、方法（不依赖任何现有实现，只用本引擎自己的两条路径）
把 dflash2 的 verify 块逐列拿出来，与**同一 prompt 的 plain（无投机）流**按位置比：
- 列 j 在绝对位置 `p` 上预测的是位置 `p+1` 的 token；
- **只有"上下文干净"的列才可比较**：第 j 列的输入必须是真 token、上下文必须是真实已接受前缀
  ⇒ 即 j=0（输入=真 anchor）恒干净；j≥1 仅当第 0..j-1 列都被接受时才干净；
- 对齐必须**按 token 锚定**（`_df2_blame2.py` 的做法：在 plain 流里单调搜索，自校正探针的滞后），
  按 `pos` 直接算术对齐会被探针的一轮滞后带偏（实测会给出 21.8% 这种自相矛盾的数）。

## 二、逐列等价性实测（权威口径，token 锚定）
| 列 | 干净样本 | 与 plain 一致 | 含义 |
|---|---|---|---|
| 0（真 anchor 列） | 20（有草稿的轮） | **17/20 = 85%** | **15% 的轮次 veriy 给出的 bonus/correction 与 plain 不同** |
| 1（前缀已接受） | 8 | **6/8 = 75%** | 参考实现同位置约 82% |
⇒ **verify 块与"顺序解码"不等价**：约 15% 的列给出与 plain 不同的 argmax。
被 licensed 的 token 必然等于该列的 argmax（结构保证），所以**这一不等价直接等于流的偏移**
⇒ 这正是接受率的上限，也正是 9-10 `M_patchA_effect.md` §3 与 `M_verify_equivalence.md`
（plain vs dflash2 与 plain vs **mtp3** 都在第 36 token、同样两个值）反复指向的同一个缺口。

## 三、已排除的成因（都有实测）
| 假设 | 结论 |
|---|---|
| 草稿质量 / 草稿头 / 边项标度 | 排除：K 无关、mtp 同位置同值、边项 scale 扫描无益 |
| 跨列掩码污染（后面的 draft 列被前面看到） | **不足以解释**：K=1 就偏，且 K=1 与 K=3 同位置同 token |
| rope_delta / 位置偏移 | 排除：纯文本时 `rope_delta_=0`，plain 与 verify 的 positions 同为绝对位置 [F..F+K] |
| `apply_final_logit_policy` 漏调 | 排除：qwen3 编译期 no-op |
| KV 精度 | 排除：实测 bf16，int8 同形 |
| 声明型草稿超参 | 排除：与 ckpt 自带 `config.json` 逐项一致（R8/R9） |
| 参照实现的 walk 语义 | 一致（R9）；差异仅"参照全程 FP32，我们 BF16" |

## 四、剩余两个成因（下一步按此顺序单独考察）
**(A) 块 verify 与单步解码的算术不同 → 近似并列的 argmax 被翻**
两路走的是不同核：verify 是 `gqa_attention(q_batch,...,position_batch, valid, kv_table_rows,...)`
（一次 W 列），plain 是 `gqa_attention_cached(qn, last_position, ...)`（一列）；
GDN（线性注意力）层在 verify 里一次吃 W 列、在 plain 里一 token 一步。
⇒ 同一个 (上下文, 位置) 上两路的 logits 可能差在 1e-3 量级，在 margin 极小的位置翻 argmax。
**判别探针**：在某个"不一致列"上把两路的 **top-8 logits + 该列 margin** 打出来：
- 差值 <1e-2 且 top-2 近乎并列 ⇒ 数值路径问题（要"消除偏移"就得让块 verify 走与单步相同的算术）；
- 差值大 / 排名完全不同 ⇒ 结构性（掩码或状态）问题，回去查 GDN replay 与 KV 槽。

**(B) GDN `RecordForReplay` / 被拒列的状态回滚**
判据：把一轮的 verify 拆成"逐 token 单步"重算（可用现有 plain 路径），比较每一步的
GDN 输出与状态；若 chunked 版与逐步版不等价，就是它。

## 五、附带发现（也属"传参"，已记录待改）
参照实现 `speculator.py:904-907` 明确把 selector 分数保持 **FP32**（自注"BF16 会 measurably
改变候选顺序"），而我们 `logits` 是 `DType::BF16`。属保真度对齐项，改动小。
