# gemma_engine_plan.md — B 层引擎口味改造施工单 (照此执行, 每步 make 验证)

目标: 让 Gemma-4-31B 的 (full + swa) 混合、qk-norm、per-layer scalar、
tied head、NVFP4-QAT 权重在 qwen3_6 family runtime 上可表达。先备份:
cp hybrid_topology.h{,.bak} 等 (WSL 树无 git!)。

## 1. 三层口味基建 (先做, 不动行为)
文件 A: include/ninfer/targets/qwen3_6/hybrid_topology.h (+ export 镜像同步)
追加:
```cpp
enum class LayerKind : std::uint8_t { Full, Swa, Gdn };
[[nodiscard]] constexpr LayerKind layer_kind(std::int32_t layer) noexcept {
    return is_full_attention_layer(layer) ? LayerKind::Full : LayerKind::Gdn;
}
[[nodiscard]] constexpr std::int32_t swa_layers(std::int32_t /*n*/) noexcept { return 0; }
[[nodiscard]] constexpr std::int32_t swa_index(std::int32_t /*l*/) noexcept { return 0; }
```
文件 B: qwen3_6_27b/impl/config.h TextConfig 追加 (行为不变):
```cpp
static constexpr LayerKind layer_kind_at(int layer) {
    return qwen3_6::layer_kind(layer);
}
static constexpr int swa_layers() { return qwen3_6::swa_layers(layers); }
static constexpr int swa_index(int layer) { return qwen3_6::swa_index(layer); }
```
(需要 #include <ninfer/targets/qwen3_6/hybrid_topology.h> 已含)

文件 C: runtime/text_context.h ModelConfig 追加:
```cpp
[[nodiscard]] static constexpr bool is_swa(int layer) {
    return TextConfig::layer_kind_at(layer) == qwen3_6::LayerKind::Swa;
}
[[nodiscard]] static constexpr int n_swa() { return TextConfig::swa_layers(); }
[[nodiscard]] static constexpr int swa_idx(int layer) { return TextConfig::swa_index(layer); }
```
文件 D: runtime/text_context_impl.h run_layers: 三分支 (swa 首版 = full 同路径,
窗口掩码 TODO 下一片; 保证 is_full 语义不变):
```cpp
if (ModelConfig::is_full(layer) || ModelConfig::is_swa(layer)) {
    // swa: 现阶段复用 full mixer; 掩码切片见 TODO-1
```
验证: make ninfer_engine + 冒烟 (serve 27b artifact 正常)。

## 2. Gemma 几何 target (config 层)
- 依据 specs/gemma4_31b_spec.json 的 layer_types 顺序生成 kind 表:
  TextConfig::layer_kind_at 用 if constexpr 表 (从 spec 抄, ~60 项;
  含 swa/full 位置) + hidden 5376/layers 60/heads 32 q16kv hd256/
  vocab 262144/ctx 262144/rotary/eps/rope_theta/中间 21504/
  qk-norm=true/每层 layer_scalar/attention_bias=false。

## 3. qk-norm + per-layer scalar hook
- attn_mix (text_context_impl.h) full/swa 分支: q/k 过 rmsnorm(q_norm/k_norm
  权重) 后再 rope (Gemma 顺序: q_norm->rope); 若 TextConfig 提供
  layer_scalar, hidden 逐层乘之。用 constexpr 开关关掉对 qwen3.6 的影响。

## 4. tied head
- bindings: lm_head 对象缺省 = token_embedding 转置引用 (先物化拷贝最稳:
  loader 里 embed 载完再建 head 拷贝对象, 零运行时改动)。

## 5. NVFP4-QAT 解码转换器 (tools/convert/gemma4_31b/)
- 语义待解码: 每线性模块 4 张量 {weight_packed, weight_scale,
  weight_global_scale, input_global_scale}; config: group16/actorder/
  static_minmax/scale fp8e4m3fn。第一步: 小矩阵手工解码对照 (取
  q_proj 前 16x16, 用 torch 加载 bf16? 无原值对照 -> 与 vLLM 参考
  输出对齐或自洽(重建 scale 数学)); 第二步: 全量 CPU 流式 dequant ->
  bf16 safetensors; 第三步: 走 convert.py groupwise 或引擎 nvfp4
  requant 出 artifact。
- 辅助: 先读 v100-skinny/或 vLLM gemma QAT 解码参考 (社区) 定 packed 位序。

## 6. 验收
- 转换 smoke: serve gemma artifact, 输出连贯 (English prompt);
- ppl/对照: 与 vLLM 同权重 logits 对齐 <=1e-2 (抽样 token)。

## 阻塞确认清单
- [ ] 1 施工+编译 (三层分发; 27b 回归)
- [ ] 2 gemma config+kind 表
- [ ] 3 qk-norm/scalar hooks (编译+27b 无影响)
- [ ] 4 tied head
- [ ] 5 QAT 解码器 (先 16x16 手工对照)
- [ ] 6 smoke+对齐
- TODO-1 (性能后置): SWA 窗口掩码 (softmax over paged KV 加 window mask)。

## 施工进展与修正 (2026-09-03)
- TextConfig 口味词表已入 27b (is_swa_attention/swa_attention_layers/
  swa_attention_index/qk_norm_enabled/per_layer_scalar, 全 false) +
  ninfer_engine 编译通过 (零行为变化)。
- 代码侦察修正 blockers:
  * qk-norm 已覆盖: attn_mix 对 q/k 做 rmsnorm(q_norm/k_norm) 后 rope
    (text_context_impl.h L~880) -> Gemma qk-norm 无需新 hook。
  * attention_projection / attention_output_projection / post_mixer 均为
    Variant 叶子 -> Gemma 变体在叶子层实现 q/k/v 拆分(无 output gate 则
    attention_projection 只出 qkv; sigmoid_mul(gate) 由叶子形状决定)。
  * 因此 run_layers 三层分发可推迟: Gemma 每层都走 full 权重数组,
    is_full 恒真 (milestone-1, 窗口掩码 TODO-1); swa 语义后补。
- 仍缺的引擎点: per-layer scalar 语义待定 (先查 Gemma-4 config/推理参考
  确认乘在哪: 层输入/输出/残差前), 再在 run_layers 加 constexpr hook;
  tied head 走 loader 物化拷贝; 其余全是 Gemma variant 叶子+bindings+
  converter, 按第 2/5 步执行。
