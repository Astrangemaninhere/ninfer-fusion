# Muse-Glimmer-30B 钉死语义与引擎接线契约 (2026-09-02 晚, 自动推导+人工核对)

参考源缓存: tools/archkit/refs/modeling_muse_glimmer.py (HF main)
manifest:  tools/archkit/out/muse-glimmer-30b/manifest.json -> spec.semantics (自动窗口)
config.h / engine_hook.patch: 同目录 (adapt.py 产物, 已含 semantics 注入)

## 已钉死公式 (HF 实际代码, 非猜测)

### 层图 (MuseGlimmerTextDecoderLayer, 全部 52 层同构)
residual = x
h = input_layernorm(x)                       # CenteredRMSNorm eps=rms_norm_eps
a = self_attn(h)                             # 输入是 norm 输出
a = post_attention_layernorm(a)              # eps=post_norm_eps   ← 第二次 norm
x = residual + a                             # ← 残差在 post-norm 之后
m = pre_feedforward_layernorm(x)             # eps=rms_norm_eps
y = mlp(m)                                   # swiglu
y = post_feedforward_layernorm(y)            # eps=post_norm_eps   ← 第二次 norm
x = residual + y

=> 每层 4 个 RMSNorm: input/post_attn (eps=rms_norm_eps), post_attn/post_ff 用 post_norm_eps。
   引擎对照: qwen3 是 norm→attn→残差 单 norm。此为 flavor 级差异(双 norm 包裹 attn/mlp)。
   "Centered" 仅是命名: 实际 = x*rsqrt(mean(x^2)+eps)*(1+w)  无 mean 减除!
   w 初值 0 -> 打包时权重按 w'=1+w 烘焙进引擎 norm 权重即可 (引擎 rmsnorm 内核不变)。

### Attention (MuseGlimmerTextAttention)
q = q_proj(h)  k = k_proj(h)  v = v_proj(h)      # bias=attention_bias
q = qk_norm(q) * qk_scale_factor                 # qk_norm = 无参 RMSNorm(with_scale=False)
k = qk_norm(k)                                   # key 不再乘 factor!
if layer_rope_theta[i] != 0: rope(q,k)           # NoPE(theta=0)层整层无 rope
scores = q@k^T * head_dim^-0.5                   # 额外标准缩放 (eager 走 attention_interface)
mask: causal; 若 layer_types[i]=="sliding_attention" 再加 window 2048 (sliding_window)
a = softmax(scores)@v
a = a * sigmoid(gate_proj(h))                    # ← 门控注意力 (gate 输入 = norm 后的 h!)
x_out = o_proj(a)

=> qk_scale 乘在 qk_norm 后的 q 上; scores 另有 1/sqrt(head_dim)。引擎若自带 head_dim 缩放则
   只需 q*=factor, 别重复。sliding 层窗口=2048 (纯 causal+window), full 层纯 causal。

### 最终 logits (MuseGlimmerTextModel.forward)
logits = lm_head(final_norm(x))                  # final_norm = CenteredRMSNorm
logits = logits * output_multiplier              # 0.196116
logits = logits / final_logit_softcapping        # /20
logits = tanh(logits)
logits = logits * final_logit_softcapping        # => 20*tanh(mult*x/20), mult 在 tanh 内!
softcap = T*tanh(x*mult/T) 与 mult*softcap(x) 不等价, 顺序必须: 先 mult 后 tanh。

### 其它
layer_rope_theta: 52 长, 13 个 0 恰在 full-attention 层位 (自动发现);
full 层 = NoPE 无 rope, sliding 层有 rope(各自 theta)。
num 结构: heads 32 / kv_heads 2 / hd 128 / hidden 6656 / 52 层 (39 sliding+13 full)。
attention_bias: config False (确认过 config.json)。
tie_word_embeddings: false -> 独立 lm_head。
qk_norm 无缩放参数 -> 引擎 rmsnorm weight=1 或 None。
post_norm_eps != rms_norm_eps -> 两层 norm 的 eps 不同, 引擎按层选 eps。

## 引擎接线状态 (2026-09-02 晚)

已完成:
- ops::logit_policy (新 op, in-place x*mult + cap*tanh(x/cap), BF16, bf16x8/x2/scalar
  路由, GPU 数值测试 OK: tests/ops/test_logit_policy.cpp 全绿含 identity/edge)
- ModelConfig::apply_final_logit_policy constexpr-if 助手 (text_context.h)
  qwen3 系零开销 (mult==1 && cap==0 -> 编译期空)
- 接线 8 个 lm_head logits 产出点: proposal_argmax / verify x2 / target_verify_batch /
  prefill last-token / sample_from_hidden / HS-dump TOPK 真值 / dflash2 proposal
  (text_context_impl.h + text_prefill_impl.h + dflash2_impl.h)
- ninfer_engine + ninfer_serve 编译干净; 文件已同步回 Windows 镜像

### 主层结构实测结论 (attn_mix / mlp_tail / run_layers, text_context_impl.h)
[已钉死; 本文件上段 = 2026-09-02 晚结论]

### 引擎特性化落地状态 (2026-09-03)
- E1 DONE: attn_mix 逐层 rope theta (rope_theta_at(layer_of_full(fidx))) +
  theta<=0 整层跳过 rope (NoPE); qwen 编译期常量, 行为/性能不变;
  ninfer_engine+serve 编译干净, 已同步镜像 (_TODO.md 9c)
- E2 定案: qk_scale_factor 折进变体 attention_scale (= qk_scale/sqrt(hd));
  rope 线性 => q 乘因子 ≡ scores 缩放, 零内核/零运行时开销; 引擎
  kAttentionScale=Variant::attention_scale (instance.h:58) 已存在; qk_norm
  无参(无权重)留 bindings 层 (weight 槽空)
- E3 查证: decode gqa 核 (gqa_attention_decode_nvfp4.cuh:134/265) 已有
  sliding_window 参数; prefill 有 src/ops/softmax_attention/sliding_window/
  sliding_window_attention.cpp op (现仅 dflash 草稿侧用) -> 主路径接线待做
- E4 (D1) 待: FullLayerW 新槽 post_attn_out_norm + attn_mix/mlp_tail 分支
- E5 待验: n_gdn=0 时 bind/load/artifact 空 gdn 布局可跑

### 下一步 (todo#3/#4)
qwen3.6 full 层已含: input_norm -> (Variant::attention_projection 出 q/gate/k/v) ->
  rmsnorm(q/k) -> rope(统一 kCfg.rope_theta) -> gqa_attention(kAttnScale, 无 window) ->
  sigmoid_mul(gate, a) -> Variant::attention_output_projection(a, o_proj, x=残差流,
  leaf 内 fused o_proj+residual) -> mlp_tail(post_attn_norm, mlp) (norm 残差流 -> mlp
  -> leaf 内 residual)
=> qwen post_attn_norm 位置 == Muse pre_feedforward_layernorm 位置。
Muse 与 qwen3.6 full 层差异 = 恰好四处:
  D1. o_proj 后先 post_attention_layernorm 再残差 (qwen: leaf fused 残差) -> driver
      需给 Muse 单独 o_buf: leaf 写 o_buf, rmsnorm(o_buf, 新权重槽, post_norm_eps),
      residual_add。FullLayerW 需新槽 post_attn_out_norm。
  D2. 逐层 rope theta + NoPE 整层跳过 (attn_mix 内 rope 调用点, 按 layer 而非 fidx;
      fidx<->layer 已有 layer_of_full 反查)
  D3. 39 个 sliding 层 = causal+window(2048): 主 gqa_attention 签名无 window ->
      需 windowed 变体或 swa/sliding_window_attention op 路由 (schedule.h 已 include
      SWA op, 查其现用方/参数)
  D4. mlp 前 norm = 4 号 norm (qwen post_attn_norm 槽复用, 权重不同 -> bind 时指到
      Muse pre_feedforward_layernorm; post_norm_eps 独立 -> rmsnorm eps 按层/槽)
qwen 侧 D1-D4 全部 constexpr/分支编译期消除 (perf 不变)。gate 机已在 (qwen3.6 full
层本就有 sigmoid_mul); qk_norm 机已在; kAttnScale 额外 head_dim^-0.5 已含 -> q 侧
qk_scale 只是乘 3.87 (查 kAttentionScale 定义确认没有二次)。

### 下一步 (todo #3/#4)
1. FullLayerW + bind + attn_mix/mlp_tail/run_layers 的 Muse 味开关 (config 声明式):
   D1 o_buf+post-attn-out-norm+后残差; D2 逐层 theta/NoPE; D3 windowed 路由;
   D4 四 norm eps 表 -> 收进 flavors.py 词表 (sliding_double_norm / rope_no_pe ...)
2. 主层驱动: 逐层 theta + NoPE 跳过 + q 侧 qk_scale (MTP tail 是现成范例)
3. Muse converter + serve 对拍 (权重下载中: data/muse_nvfp4 仅元数据)
4. Gemma milestone-1 施工单继续; v3 qwen38 golden 对拍
5. Muse NVFP4 safetensors 下载完成度检查 (dl_muse_nvfp4.py 后台)

## 旧备注(存档)
