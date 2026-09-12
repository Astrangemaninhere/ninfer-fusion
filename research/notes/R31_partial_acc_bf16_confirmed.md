# R31：核心嫌疑确认 —— split-K 的 `partial_acc` 是 BF16（修法极小）（2026-09-11 17:3X）

## 一、确认（代码直证）
`src/ops/softmax_attention/dense/context/kernel.cuh:71-101`
```cpp
context_attention_split_partial_kernel(..., __nv_bfloat16* __restrict__ partial_acc,
                                       float* __restrict__ partial_m, float* __restrict__ partial_l,
                                       __nv_bfloat16* __restrict__ out)
context_attention_reduce_kernel(const __nv_bfloat16* partial_acc, const float* partial_m,
                                const float* partial_l, ...)
```
⇒ **split-K 各分块的注意力输出以 BF16 落在 `partial_acc`**，随后由 reduce kernel 归约；
online-softmax 的 `m/l` 是 FP32 ✓。结合 A 路事实（**T≤6 走 SmallT/split-K**，**T∈[7,16] 恒走 Prompt**）：
- 我们的默认 setup：**plain(T=1) = SmallT/split-K + BF16 partial**，**verify(K=7, T=8) = Prompt**（无 split）
  ⇒ 两条不同的归约 ⇒ **同 (上下文,位置) 上的注意力输出不等价** ✓；
- BF16 partial 的粗粒度（分块数随 T 变、每块输出 8-bit 尾数）⇒ 相对误差 ~1e-2 ⇒ near-tie 处翻 argmax
  ⇒ 与"verify 列 0 与 plain 仅 69% 一致"的量级吻合 ✓；
- 也解释了为何此前"对齐分派/累加链"零效果：那些都不是**核心注意力内部**的归约精度 ✓。

## 二、修法（性能优先，代价≈0）
把 split-K 的 `partial_acc` 从 **BF16 提到 FP32**（不改路由、不动 Prompt、不动 KV）：
- 改动面（预计 3–5 处）：该 op 的 plan/launch 里 `partial_acc` 的 dtype、partial kernel 的指针类型、
  reduce kernel 的入参类型；`m/l` 已是 FP32 不动 ✓；
- 代价：`partial_acc` 缓冲 = `head_dim × n_q × splits` 的小张量，FP32 化后 ×2 ⇒ **可忽略**；
  归约本身的算术不变（仍是各块先算 m/l 再合并）⇒ 预计 **<1% tok/s**；
- **性质**：这是"**统一算术**"（与用户定的口径一致）里性价比最高的一项——把**精度档**对齐，
  而不动任何路由与并行度 ✓。

## 三、可证伪判据（落地即测）
1. `verify 列 0 与 plain 一致率`：**69% → 期望 ≥90%**（样本 ≥29）；
2. `zh plain vs dflash2` 首次偏离：期望**后移或消失**；
3. 接受率与 decode tok/s：**不得退化**（tok/s 是硬底线）。

## 四、顺序（两个编译窗口）
1. **先落本项**（改动小、直指核心）→ 四项目基准复测；
2. **再落暂存 TU 拆分**（根治"改一行头重编半小时"）——此后每次实验的成本从半小时级降到分钟级，
   而 §R30/R31 这类"改精度档 + 复测"的迭代正是最需要它的场景。
（若本项判据不达标 ⇒ 嫌疑转向 Prompt 侧或无 split 路径的 post-attention，再按同样方法定点。）
