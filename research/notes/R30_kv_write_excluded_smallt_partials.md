# R30：KV 写路径排除；核心嫌疑收敛到"SmallT 的 BF16 partial"（2026-09-11 17:2X）

## 一、KV 写路径：**逐位一致**（排除）
工具：`NINFER_KVDUMP_DIR` + `NINFER_KVDUMP_KV=0,15`，**必须配 `--no-cuda-graph`**
（否则 dump 内的 `cudaMemcpyAsync`+`cudaStreamSynchronize` 撞 `cudaErrorStreamCaptureUnsupported`，
仓内 `kv_calibration.h:46` 早已注明这类离线模式要配 `--no-cuda-graph`）。
| 指标 | 结果 |
|---|---|
| 两臂 dump 文件 | 各 15 个 |
| 公共 (layer,pos) 条目 | **64**（层 0 与 15）|
| K 不一致 | **0 / 64** |
| V 不一致 | **0 / 64** |
⇒ **plain 与 spec 写入的（rmsnorm+rope 后的）K/V 逐位一致 ⇒ 写路径不是本因** ✓

## 二、必须纠正的一处推断
此前我说"路由已对齐（A 的 373 行）"——**对核心注意力不成立**：A 路自己把 **core attention**
列为"**无法统一**"（改任一侧都要牺牲性能），它那 14 处分派点里**没有核心注意力**，
改的是 *输入投影 / MLP / GDN / lm_head 外围*。⇒ 核心注意力**仍是未验证、未对齐**的一环 ✓

## 三、收敛后的核心嫌疑（**有机制、有量级、有可证伪预测**）
A 路原文：27B 24Q 下 **T≤6 走 `SmallT`（key 轴 split-K、BF16 partial）**，**T∈[7,16] 恒走 `Prompt`**。
- 我们的默认 setup：**plain = T=1 ⇒ SmallT**、**verify(K=7) = T=8 ⇒ Prompt** ⇒ **两条不同归约**；
- **BF16 partial 的粗粒度**：对 32–64 个 key 的归约用 BF16 中间量 ⇒ 相对误差可达 ~1e-2
  ⇒ 在 near-tie 处翻 argmax ⇒ **与实测的 31% 列 0 偏差量级吻合** ✓✓；
- 也解释了为何"对齐分派/累加链"零效果：那些都不是**核心注意力内部**的归约精度 ✓。

### 可证伪预测
若把 **SmallT 的 split-K partial 从 BF16 提到 FP32**（不改路由、不动 Prompt）：
- 同一 (上下文,位置) 上 SmallT 与 Prompt 的输出应当**收敛到同一数值**（至多剩 FP32 结合序差）；
- 判据：`verify 列 0 与 plain 一致率` 应从 **69%** 显著上升（→ 90%+）；
- 代价：仅该归约的 partial 寄存器/带宽 ~2×，属**极小的归约**（key 轴 32–64）⇒ 预计 <1% tok/s。

## 四、下一步
1. **核对 SmallT 是否真的用 BF16 partial**（读 `ops/softmax_attention/.../small_t` 系列，定点）——
   若属实，改 FP32；若本来就是 FP32，则核心嫌疑转向 Prompt 侧或 post-attention 路径。
2. 落地后按 §三的判据复测（列 0 一致率 + 接受率 + tok/s）。
3. 并行：落地暂存 TU 拆分，根治"改一行头重编半小时"（下一步的实验都要靠它提速）。

## 五、副产品
- 该两臂 dump 的 `rc=1`（在 dump 之后报错退出）不影响结论：条目已足量（64 条）。
- 纪律复述：dump 类仪器必须 `--no-cuda-graph`；产物写 `dl/`；`pkill -x` 而非 `-f`。
