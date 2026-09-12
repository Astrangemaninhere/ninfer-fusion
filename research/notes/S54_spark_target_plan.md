# S54 · Spark-X2.5-4B → 引擎新目标接入：只读分析与实施计划

- 日期：2026-09-10
- 范围：**只读分析**（未改任何 `src/**`，未编译，未开 nvcc/ptxas，未用 GPU）
- 行号口径：全部取自镜像树 `C:\Users\User\Documents\ziqinzhang\ninfer-fusion-repo`（相对仓库根），
  与 `/home/user/ninfer-fusion` 的行号可能有偏移。**没有行号证据的判断一律标 `待证实`。**
- 上游参考（**唯一事实**）：`models\Spark-X2.5-4B\modeling_spark.py` + `configuration_spark.py` +
  `model.safetensors.index.json`（真索引 290 张量，已读）。

---

## 0. 结论摘要（先看这 6 条）

1. **缺口的真实规模比 manifest 大**：manifest 报 `hook 7 / new_op 0 / post 1`，但只读实证后有
   **3 个 manifest 未列 + 2 个被降级的 new_op 级缺口**（§3.0 表）。其中"主注意力头几何 16Q/4KV@D256
   未注册"和"逐头门控 / GELU 双输入乘缺算子"是**写代码+测试**级别，不是接线级别。
2. **`spark-x2.5-4b` 当前生成的 config.h 编译不过**：`namespace ninfer::targets::spark_x2.5_4b::detail`
   含小数点（`config.h:3`，来自 `adapt.py:343` / `gen_full_target.py:86,203` 的 `replace('-','_')`）。
   **第 0 步必须先定 target id**（建议 `spark_x2_5_4b`，artifact 的 `model_id` 字符串仍保留
   `spark-x2.5-4b`）。
3. **家族 runtime 有 3 条"默认假设"对 Spark 不成立**，且都**静默算错**不是报错：
   ① 逐头门 vs 逐通道门（`gate` 形状）；② 无条件 q/k RMSNorm（Spark 无 qk-norm）；
   ③ `rope_theta`/`rotary_dim` 单值（Spark 按层型二值）。
4. **`ops::rope` 的语义与 HF 完全对得上**（前 R 维旋转 + split-half + `θ^(-2i/R)`），
   R=256 在通用核里恰好落在 `kRopeMaxHalf=128` 边界内 → **rope 不需要新内核，只需接线**
   （这与 manifest 的 `hook` 判定一致，是 7 条里唯一"判定=现实"的一条）。
5. **滑窗（W=512）是最大不确定项**：引擎主模型路径**完全没有窗口通路**（`sliding_window_tokens`
   从未被赋值，且只有 iso3/NVFP4 两个 KV 档读它）；草稿侧 `ops::swa` / `sliding_window_attention`
   是 D128/32/8/W4096 的固定域，**不可复用**。
6. **分两步走能最快拿到"能出 token"**：先 BF16 权重 + 无窗（T<512 时与 HF 等价）跑通数值，
   再上窗口与新头几何。KV 预算必须提前算（36 层全量 KV = 144 KB/token）。

---

## 1. 基线：引擎"一个 exact target"的机制（27b / muse 双样本）

### 1.1 家族 runtime 的编译期闭合结构

| 机制 | 位置 | 说明 |
|---|---|---|
| 宏实例化 | `src/targets/qwen3_6/impl/runtime/instantiate.h:1-29` | target 的 `variant.cpp` 定义 `NINFER_QWEN36_VARIANT` / `NINFER_QWEN36_RUNTIME_NS` 后 include，整个家族 runtime 编进该 TU |
| 名字绑定 | `.../runtime/instance.h:14-30` | `Variant::TextConfig/VisionConfig/DFlashConfig/DFlash2Config/ModelView/...→` 家族内部别名 |
| 架构旋钮 | `.../runtime/text_context.h:31-165` | `ModelConfig` 把 `TextConfig` 逐字段映射；**检测惯用法**（`requires { ... }`）已内建 7 个：`rope_theta_at(layer)`(75-82)、`layer_of_full_index`(83-91)、`output_multiplier()`(92-99)、`final_logit_softcapping`(100-107)、`attn_out_post_norm()`(135-141)、`mlp_out_post_norm()`(142-148)、`post_norm_eps`(149-155) |
| 常量 | `.../runtime/text_context.h:167-169` | `kCfg` / `kAttnScale = kAttentionScale`（来自 `instance.h:58` = `Variant::attention_scale`） |
| 层数组 | `.../runtime/text_context.h:445` | `std::array<FullLayerW, TextConfig::full_attention_layers()> full_` |
| 层循环 | `.../runtime/text_context_impl.h:1144-1203` | `run_layers` 只有 **full / gdn 二分**（`ModelConfig::is_full(layer)`），没有第三种注意力 |
| 全注意力实现 | `.../runtime/text_context_impl.h:922-1024` | 唯一的 `attn_mix`：rmsnorm(q,k) → rope → gqa_attention → sigmoid_mul → o_proj |

### 1.2 上游参考的 4 条硬语义（Spark）

| 语义 | HF 代码 | 引擎现状 |
|---|---|---|
| scale = 1/√head_dim（无 qk_scale、无 softcap） | `modeling_spark.py:146` `self.scaling = 1.0/math.sqrt(self.head_dim)`；`:191-196` 传入 eager attention | `kAttentionScale = qk_scale_factor * 1/√hd`（`gen_full_target.py:200`）→ 1.0×1/16 = **0.0625** ✓ 与 27b 同值 |
| 逐头门控：`g_proj: hidden→num_heads`，sigmoid 后乘到每个 head | `modeling_spark.py:154`、`:174,179-180`、`:198-206` | `sigmoid_mul` 是**逐元素同形**门（§3.4） |
| MLP = `down(gelu(gate(x)) * up(x))`，gelu = **exact erf** | `modeling_spark.py:116-132`；ACT2FN["gelu"] | `ops::gelu(Exact)` 存在，但**没有两输入逐元素乘**（§3.5） |
| rope 按层型：full(θ=5e6, prf=0.25→R=64) / sliding(θ=1e4, prf=1→R=256) | `modeling_spark.py:31-40,43-57,368-378`；`configuration_spark.py:109-115` | 单值 `rope_theta`/`rotary_dim`（§3.2/§3.3） |

补充实证（真索引，`models\Spark-X2.5-4B\model.safetensors.index.json`）：每层 **恰好 8 张**权重
`input_layernorm.weight`、`post_attention_layernorm.weight`、`mlp.{gate,up,down}_proj.weight`、
`self_attn.{q_k_v_proj,g_proj,out_proj}.weight`（8×36 = 288）+ `model.embedding.weight`、
`model.norm.weight` = **290**。→ **没有 q_norm/k_norm**（§3.7）、没有 bias（config `attention_bias=false`,
`mlp_bias=false`）、**没有 MTP/草稿头**（故 `--spec` 只能 None）。

### 1.3 适配产物现状

| 产物 | 内容/问题 |
|---|---|
| `tools/archkit/out/spark-x2.5-4b/manifest.json` | `gaps` 9 条（hook 7 / covered 1 / post 1）；JSON 自带 `rope_by_kind`/`partial_rotary_by_kind`/`layer_kind_order`/`knobs` —— 生成器需要的信息**都在** |
| `.../config.h`（946 B，`:1-22`） | **不能直接用**：命名空间非法（`:3`）；缺家族必需字段 `output_rows/token_domain/rotary_dim/key_dim/value_dim/convolution_dim/query_size/kv_size/mtp_*/rope_theta/mtp_layers/is_full_attention()/full_attention_index()/gdn_index()`；`full_attention_layers()=9` 而 `gdn_layers()=0`（两者相加≠36，家族 `run_layers` 会漏 27 层） |
| `.../engine_hook.patch`（124 B，`:1-3`） | 只有一行 TODO 注释，无实际补丁 |
| 参考模板 | `src/targets/muse_glimmer_30b/impl/config.h`：**27b 同形完整表面 + 逐层 kind/theta 表**（`:12-100`），由 `tools/archkit/gen_full_target.py` 生成 —— 但它有 **NoPE 硬断言**（`gen_full_target.py:45-50,191-198`），Spark 两种层型 θ 都 >0，**直接 assert 失败**；`rotary_dim` 也是单值（`:63,108`） |

---

## 2. 新增一个 target 要落哪些文件（逐文件清单）

以 `qwen3_6_27b`（混合族：full+gdn+mtp+dflash2）与 `muse_glimmer_30b`（**纯 softmax 族：gdn=0、
无 mtp、无草稿** —— 与 Spark 同形）为样本。**Spark 应照抄 muse 的骨架**。

### 2.1 引擎侧（`ninfer-fusion-repo/src/**`）

| # | 文件 | 作用 | 来源 | 关键结构体/函数 + 证据行号 |
|---|---|---|---|---|
| 1 | `src/targets/<id>/CMakeLists.txt` | 3 个 TU 进 `ninfer_engine` + 2 个 include 目录 | 手写（4 行，抄 `qwen3_6_27b/CMakeLists.txt:1-9`） | `target_sources/target_include_directories` |
| 2 | `src/targets/<id>/impl/config.h` | **TextConfig**：几何常量 + 逐层 kind/θ/(新)rotary 表 + 访问器 | **生成器**（`gen_full_target.py` 升级版），形态见 `muse_glimmer_30b/impl/config.h:12-124` | `TextConfig::{hidden,layers,intermediate,output_rows,token_domain,query_heads,kv_heads,head_dim,rotary_dim,gdn_*,query_size,kv_size,mtp_*,layer_kind,layer_rope_theta,kFullLayers,is_full_attention,is_swa_attention,full_attention_index,layer_of_full_index,rope_theta_at}` |
| 3 | `src/targets/<id>/impl/variant.h` | **Variant 表面**：~15 个叶子声明 + payload/Profile/ModelView 别名 + constexpr（attention_scale 等） | **抄 muse**（`muse_glimmer_30b/impl/variant.h:17-148`，与 27b `variant.h:17-151` 同形） | `struct Variant`、`default_layer_kv_dtypes`、`supports_per_layer_kv_defaults`、`ordinary_graph_profiles` 等 4 个 profile |
| 4 | `src/targets/<id>/impl/variant.cpp` | **叶子实现**：ops 组合 + `instantiate.h` 宏实例化 + graph profiles | 半抄 muse（`muse_glimmer_30b/impl/variant.cpp:1-32` 宏、`:227-247` attention_projection、`:365-379` post_mixer 必须重写） | `NINFER_QWEN36_VARIANT` / `NINFER_QWEN36_RUNTIME_NS`（`variant.cpp:17-19`） |
| 5 | `src/targets/<id>/impl/load/bindings.h` | **artifact 对象计划**（`BindingPlan`/`TextLayerPlan`/`WeightPlan`）+ **运行时 payload** + `RuntimeModelView` 别名 | 抄 muse（`muse_glimmer_30b/impl/load/bindings.h:22-265`），payload 需改造（§3.4） | `kTextLayers/kFullAttentionLayers/kGdnLayers`、`*Plan`、`*Payload`、`RuntimeModelView`、`LoadedModelData` |
| 6 | `src/targets/<id>/impl/load/bindings.cpp` | binder 键 → 对象计划展开（每条线性的 format/divisor/形状） | **手写**（27b `bindings.cpp` 48 KB；muse 16 KB） | `bind_artifact(binder, profile, features)` |
| 7 | `src/targets/<id>/impl/package.cpp` | `LoadPlan/LoadedModel/Impl` + **identity 注册** + frontend/planner/program 装配 | 抄 muse `package.cpp`（`qwen3_6_27b/impl/package.cpp:19-207` 同构） | `Package::{sampling_defaults,resolve_weights,resolved_auto_speculative,plan_load,construct_loaded_model,make_frontend,make_sequence_planner,create_program,export_head_weights}` |
| 8 | `src/targets/<id>/export/ninfer/targets/<id>/package.h` | 公共身份头（`model_id`/`target_key`/`WeightsProfile` 枚举/一串 using） | 抄 muse `export/.../package.h`（27b 版 `:83-146`） | `Package::model_id="spark-x2.5-4b"`、`target_key` |
| 9 | `src/targets/registry.h` | `Loaded<T>` / `<T>Instance` / `ActiveTarget` variant | 手写 3 处（`registry.h:5-20,86-130`） | `ActiveTarget`、`ConstructedTarget` |
| 10 | `src/targets/registry.cpp` | `construct_target` 的 model_id 分派 | 手写 1 处（`registry.cpp:319-343`） | `if (identity.model_id == SparkX2_5_4B::model_id) ...` |
| 11 | `src/CMakeLists.txt` | `add_subdirectory(targets/<id>)` | 手写 1 行（`:354-357` 现为 4 行） | — |
| 12 | `tests/targets/<id>/test_load_plan.cpp` + `tests/CMakeLists.txt` | **结构门**：用真 artifact 校验 materialization 计数/几何 | 抄 27b（`tests/targets/qwen3_6_27b/test_load_plan.cpp:1-60`；注册 `tests/CMakeLists.txt:152-160`） | `SKIP_RETURN_CODE 77` |

### 2.2 config.h 里"能直接得到" vs "必须手写"

- **生成器可得（前提是先修生成器）**：`hidden=2560 layers=36 intermediate=10240 query_heads=16
  kv_heads=4 head_dim=256 rms_eps=1e-6`（manifest `geometry`）、`layer_kind[36]`（`layer_kind_order`）、
  θ 表（`rope_by_kind`）、窗口 512、`kAttentionScale=0.0625`、`kNativeContext`。
- **必须手写/生成器需新增**：
  - `output_rows` / `token_domain` = **131072**（Spark 无 padding，131072 % 128 == 0 ✓）；
  - `gdn_*` 槽位：生成器已给 0（`gen_full_target.py:98-103`），但 muse 落地时改成 1（`muse config.h:24-29`，
    理由"池规划要求非零几何"）—— 引擎侧已用 `max(1, ...)` 兜住（`layouts_impl.h:159-166`），两者都可编译，
    照 muse 填 1 更稳；
  - 访问器补齐：`is_full_attention()` 必须**对所有 36 层返回 true**（家族只有 full/gdn 二分，
    `full_attention_layers()` 同时决定 KV 层数 `decoder_state.cpp:164`），swa 语义只能靠新钩子（§3.1）；
  - **新增** per-layer 表：`layer_rotary_dim[]`（或 `rotary_dim_at(layer)`）—— 家族当前只有单值
    `rotary_dim`（`text_context.h:44`）。
- **必须手写（生成器不做）**：`variant.{h,cpp}`、`load/bindings.{h,cpp}`、`package.{h,cpp}`、CMake、
  registry、测试、转换器。

### 2.3 转换器侧（不在 `src/`，但属于"新增一个 model"的必要组成）

| 文件 | 作用 | 先例 |
|---|---|---|
| `tools/convert/<id>/convert.py` | 写 `.ninfer`：`ArtifactIdentity(MODEL_ID, WEIGHTS_ID)` | `muse_glimmer_30b/convert.py:35-36,396` |
| `tools/convert/<id>/recipe.py` + `inventory.py` | HF 键 → 对象名/形状/量化格式 | `qwen3_6_27b/{recipe,inventory}.py`（S53 已把 tied 物化链路做通：`MODEL_ID="qwen3.6-27b"` `inventory.py:35-36`） |
| `tools/convert/<id>/verify.py` | 结构+代表源校验（对象数/几何/量化） | `qwen3_6_27b/verify.py:1-60` |
| `tools/convert/<id>/qwen_chat_template.jinja` + 2 个 preprocessor json | **必须**：前端只认两个 sha256 白名单模板 | `muse_glimmer_30b/convert.py:220-260`（`:242-253` 嵌 qwen 模板 + `tokenizer_config` 补丁） |

反向依赖（**顺序要求**）：`config.h` 的几何 / `bindings.cpp` 的对象名 / 转换器 recipe 三者是
**同一份布局 schema 的三个投影**，必须一起改（`_AUTOADAPT.md §2b` 的同一结论）。

---

## 3. 七个钩子（+5 个漏项）的逐条接入点

### 3.0 判定总表（左=manifest，右=只读实证）

| # | manifest 条目 | manifest 判定 | 实证判定 | 差在哪 |
|---|---|---|---|---|
| H1 | `attention:sliding_window(512)` | hook | **new_op 级 + runtime 双缺口** | 主模型无窗口通路；只有 iso3/NVFP4 KV 档读窗口（§3.1） |
| H2 | `rope:per_type_theta` | hook | **hook（判定正确）** | 钩子已在 `text_context.h:75-82`，调用点未用（§3.2） |
| H3 | `rope:partial_rotary` | hook | **hook（判定正确）** | `ops::rope` 已收 rotary_dim；缺 per-layer rotary 钩子 + 走通用核（§3.3） |
| H4 | `attn:headwise_output_gate(sigmoid)` | hook | **new_op 级** | `sigmoid_mul` 是逐元素同形；逐头需 broadcast 算子或复制权重（§3.4） |
| H5 | `mlp:act=gelu` | hook | **new_op 级** | `ops::gelu` 有，但 `gelu(gate)*up` 的双输入乘**没有算子**（§3.5） |
| H6 | `token_domain:vocab=131072` | hook | **hook（判定正确，但面更大）** | 还要关官方特殊 token 校验、过 thinking-control 圆整、模板白名单（§3.6） |
| H7 | `layers:36>16` | hook | **covered（家族已 64）** | 无缺口，只需审计确认（§3.8） |
| **X1** | *(未列)* 主注意力头几何 16Q/4KV@D256 | — | **new_op 级（最硬）** | 全链路精确几何注册制，硬 throw；且 `q_heads=16` 会命中 35B 分支（§3.9） |
| **X2** | *(未列)* 无 qk-norm | — | **hook（静默错）** | 家族无条件 rmsnorm(q,k)（§3.7） |
| **X3** | *(未列)* `q_k_v` 融合投影 vs 家族 `attn_input_proj` 固定几何 | — | **new_op 级** | `attn_input_proj` 硬编码 6144/1024 行；须走 muse 式独立 linear（§3.10） |
| **X4** | *(未列)* 131072 行的 head/embed 走 `ops::linear` | — | **new_op 级**（post 项的一部分） | 所有 linear 档位都是精确 (n,k) 表（§3.11） |
| **X5** | *(未列)* `spark-x2.5-4b` → 非法 C++ 命名空间 | — | **阻塞级（1 行修复）** | `adapt.py:343` / `gen_full_target.py:86,203`（§3.12） |

---

### 3.1 H1 滑窗（W=512）

**现状（证据）**

- 主模型 KV 视图有字段但**从没人写**：`PagedKVLayerView.sliding_window_tokens`（`src/core/paged_kv_cache.h:60`）
  与 `PagedKVBatchLayerView` 版（`:87`）；构造点 `PagedKVCache::layer_view`（`decoder_state.cpp:249-299`）
  与 `batch_layer_view`（`:301-...`）的聚合初始化里**没有这个字段** → 恒 0。
- 只有两个 KV 档读它：iso3 解码核（`src/ops/kernel/gqa_attention_decode_iso3.cuh:31,132`）、
  NVFP4 解码核（`...decode_nvfp4.cuh:134,273`）、NVFP4 预填核（`...prefill_nvfp4.cuh:998,1160`，
  且 `:1160` 显式 `sliding_window > 0 && KVDType == DType::NVFP4`）。
- **BF16 解码核把 `window` 定义为可见键数**：`src/ops/kernel/gqa_attention_decode_bf16.cuh:124`
  `const int window = last_pos + 1;` → 与 `sliding_window_tokens` 无关。I8/FP8 同理（同族核）。
- 草稿侧窗口 **不可复用**：`ops::swa`（`include/ninfer/ops/swa.h:29-52`）与
  `sliding_window_attention`（`include/ninfer/ops/sliding_window_attention.h:21-56`；
  `.../sliding_window/sliding_window_attention.cpp:16-30` 校验 profile）都是
  **D=128 / Hq=32 / Hkv=8 / W∈{2048,4096} / cyclic cache / T≤16 / B≤8** 的固定域，Spark 需要
  D=256/Hq=16/Hkv=4/W=512。`Config::local_window`（`qwen3_6_27b/config.h:97,128`）是 DFlash/DFlash2
  草稿配置，非主模型。
- 家族里已声明但**无人消费**的 SWA 访问器：`is_swa_attention()` / `swa_attention_index()`
  （`muse_glimmer_30b/impl/config.h:71-73,83-87`），`qk_norm_enabled()`（`:74`）同理 —— grep 全树无调用点。

**接入点（要做的事）**

1. `PagedKVCacheLayout` 加 per-layer 窗口表（照抄已有的 64 宽数组风格：
   `src/targets/qwen3_6/export/ninfer/targets/qwen3_6/decoder_state.h:57-73`），
   `PagedKVCache` 加 `std::array<std::uint32_t,64> layer_window_`（`:149-153` 同款），
   在 `layer_view`/`batch_layer_view`（`decoder_state.cpp:262-298` / `:314-...`）填
   `.sliding_window_tokens = layer_window_[layer]`。
2. 窗口值来源：`DecoderStateSpec` 加窗口表（`decoder_state.h:27-46`）+ 规划侧
   `layouts_impl.h:136-155` 从 `TextConfig` 生成（新访问器 `window_at(layer)`，0=全注意力）。
3. **BF16/I8/FP8 KV 档必须新增窗口支持**（否则首版接线等于没接）：两条路
   a) 首版强制 `--kv-dtype nvfp4`（已有窗口支持，但要 `layer_views` 之外确认 NVFP4 KV 与 4 KV 头
      组合可跑）；b) 给 `gqa_attention_decode_bf16.cuh:124` 的 `window` 加"取 min(last_pos+1, sw)"逻辑
      （1 行级，但**要重测所有历史 profile**）。
4. 窗口**语义**判定：引擎注释写"distance window-1 含、window 不含"（`swa.h:36-38`），
   HF 用 `create_sliding_window_causal_mask`（`modeling_spark.py:359`）——**差 1 的边界必须实测对齐**
   （`待证实`）。

**验收判据**：prompt 长度 >512 时，开窗/关窗输出**必须不同**（否则视为未接线）；且开窗后与 HF
greedy 逐 token 一致。

### 3.2 H2 按层型的 rope theta

**现状（证据）**：`ModelConfig::rope_theta_at(int layer)` 早已存在，且是**检测惯用法**
（`text_context.h:75-82`：`TC::rope_theta_at(0)` 不存在就回落 `TC::rope_theta`）；
配套的层号映射钩子 `layer_of_full_index`（`:83-91`）与 `layer_of_full(fidx)`（`:111-113`）也在。
但**唯一的调用点用的是单值**：

```cpp
// text_context_impl.h:960-968（attn_mix 内，唯一主模型 rope 调用点）
const Tensor& rope_positions = ... ;
if (ctx_.yarn_enabled) { ops::rope_yarn4(rope_for_op, kCfg.rotary_dim, kCfg.rope_theta, qn, kn, s); }
else                    { ops::rope    (rope_for_op, kCfg.rotary_dim, kCfg.rope_theta, qn, kn, s); }
```

MTP 侧另有 3 处同形（`:476-478,590-592,634-636`），Spark 无 MTP → 首版不用改。

**改成什么形状**：调用点换成"**取层号 → 查表**"两段：

```cpp
const int layer = ModelConfig::layer_of_full(fidx);          // attn_mix 收到的是 fidx（full 序号）
ops::rope(rope_for_op, kCfg.rotary_dim_at(layer), kCfg.rope_theta_at(layer), qn, kn, s);
```

对既有目标 **零行为变化**（27b/35b 的 `TextConfig` 不声明这两个钩子 → 回落单值常量；
muse 已声明 `rope_theta_at` 且其 `layer_rope_theta` 是逐层值 → **顺带把 Muse 的 E1 也接上了**，
但 muse 的 `layer_of_full_index` 是恒等映射且 52 层全 full，故行为不变）。
**这是本次唯一"改家族代码但零回归风险"的一步**（建议单独成 step，便于回归）。

### 3.3 H3 按层型的 rotary_dim

- `ops::rope` 的公共契约已收 `rotary_dim`（`include/ninfer/ops/rope.h:16-34`；校验
  `src/ops/wrapper/rope.cpp:83-97`：1-D positions 时 **head_dim 必须 =256**、`0<rotary_dim<=256 且偶数`，
  或 DFlash 的 D=R=128 特例）。R=256 与 R=64 都合法。
- 通用核能承载 R=256：`src/ops/kernel/rope.cuh:24` `kRopeMaxHalf = 128`，
  `rope_generic_kernel`（`:278-327`）的 `__shared__ float cos_cache[128]` 恰好装下 `rotary_dim/2=128`；
  launcher block=128（`src/ops/launcher/rope.cu:173`），`threadIdx.x < half` 正好全覆盖。
- **固定快路径只剩 `rotary_dim==64 && theta==1e7`**（`launcher/rope.cu:94-115`，heads 24/4 或 16/2）
  → Spark 的 full 层（θ=5e6）与 sliding 层（R=256）**全部落到 generic 核**（`rope.cu:186-200`）。
- 语义核对（同 HF）：核里 `phi = positions * theta^(-2*pair/rotary_dim)`、只改 `[0,rotary_dim)`
  （`rope.h:9-21`），与 HF `compute_rope_cos_sin`/`apply_rotary_pos_emb`
  （`modeling_spark.py:31-40,43-57`，"前 R 维旋转 + 后 HeadDim-R 直通"）**逐条对上**。

**接入点**：家族侧新增检测惯用法 `rotary_dim_at(layer)`（照抄 `text_context.h:75-82` 的写法，
回落 `TC::rotary_dim`），调用点同 §3.2。**不需要新内核**。

### 3.4 H4 逐头输出门（sigmoid）

**现状（证据）**：

```cpp
// text_context_impl.h:933-941
Tensor q    = projection.query.view({kCfg.head_dim, kCfg.n_q, T});
Tensor gate = projection.gate .view({kCfg.head_dim, kCfg.n_q, T});   // ← 与 q 同形（逐通道门）
// :1019
ops::sigmoid_mul(gate, a, s);                                        // ← 要求 gate/a 同形
// :1022
Variant::attention_output_projection(a.view({kCfg.q_size, T}), *w.o_proj, x, ph, work_, s);
```

- `ops::sigmoid_mul` 契约：`gate` 与 `x` **同形、同 dtype、都连续**
  （`include/ninfer/ops/sigmoid_mul.h:9-20`；wrapper `src/ops/wrapper/sigmoid_mul.cpp:36-47`）。
- 缓冲区由 recipe 按 `Config::query_size` 分配（`.../runtime/workspace_recipe.h:58-67`，
  `gate` 是第 3 个 `query_size` 行矩阵）。
- HF 是**逐头标量**：`g_proj: hidden→num_heads`（`modeling_spark.py:154`），
  `gate_score.view(bsz, seq, num_heads, 1)`（`:180`），sigmoid 在 fp32 算再转 dtype，
  `attn_output = attn_output * gate`（`:198-206`）。

**差在哪**：形状从 `[head_dim, n_q, T]` 变 `[n_q, T]` 并广播到 head_dim。三条路：

| 方案 | 改动 | 代价 | 评价 |
|---|---|---|---|
| A. 转换期把 `g_proj` 复制成 `[query_size, hidden]`（行 h×256+c = g_proj 第 h 行） | **引擎零改动**（family attn_mix/recipe/op 全不动，只多一个新 (n,k) 线性几何） | BF16 21 MB/层×36 = **755 MB**；NVFP4 约 5.8 MB/层 ≈ **210 MB** | **首版首选**（数值与 HF 等价：同一 BF16 权重同一 GEMM，只是行重复） |
| B. 新算子 `headwise_gate_mul(gate[n_q,T], x[head_dim,n_q,T])` | 新 op + oracle + `tests/ops/test_*` + 3 处 CMake | 0 额外权重 | 正解，工时 0.5-1 d |
| C. 新算子 `broadcast_rows` + 复用 `sigmoid_mul` | 同上但要两个 op 语义 | 0 额外权重 | 不推荐（多一个中间张量） |

**接入点（若走 B）**：`workspace_recipe.h:58-67` 的 `gate` 行数参数化（新钩子 `gate_rows`，回落
`query_size`）+ `text_context_impl.h:934` 的视图 + `:1019` 换算子。

### 3.5 H5 GELU MLP

**现状（证据）**：

- 27b 的 MLP 是**融合 swiglu**：`ops::linear_swiglu(hidden, weights.gate_up, activation, policy, ws, stream)`
  （`qwen3_6_27b/impl/variant.cpp:369-377`）；该 op 语义**硬编码 SiLU**（`silu_mul.h:9-21`；
  `linear_swiglu.h:38-53` 且是精确几何注册制：`[34816,5120]/[12288,2048]` 等）。
- muse 因为要拆开（FP8 无 workspace 路线）改成：`linear(gate)→linear(up)→silu_mul(gate,up)→linear(down)
  →residual_add`（`muse_glimmer_30b/impl/variant.cpp:365-379`）。
- `ops::gelu(x, GeluMode::Exact|Tanh)` 存在（`include/ninfer/ops/gelu.h:9-26`），但是 **in-place 单元激活**。
- **没有任何"两输入逐元素乘"算子**：全 ops 清单只有 `silu_mul`（silu(g)*u）、`sigmoid_mul`（sigmoid(g)*x）、
  `residual_add`、`linear_add`（`include/ninfer/ops/` 目录 + `src/ops/launcher/` 目录逐个核对）。

**缺什么**：`gelu(gate) * up` 需要一个新算子 `gelu_mul`（照 `silu_mul` 的
`wrapper+launcher+kernel+header+测试` 五件套；注册点 `src/CMakeLists.txt:89/103/110`(launcher) 与
`:282/297/304`(wrapper) 附近、测试 `tests/CMakeLists.txt:234-249`）。
**接入点**：Spark 的 `Variant::post_mixer` 叶子（照 muse `variant.cpp:365-379` 改写第三行）。

> 注：`ops::gelu` 不能"配合"出乘法；`logit_policy` 的 mult-only 是标量乘（`text_context.h:125-129`），
> 也不可用。所以 manifest 那句"ops::gelu 已存在，需 MLP 前向选择激活"**低估了工作量**。

### 3.6 H6 token_domain / 前端（131072 ≠ 248077）

- 家族常量 `qwen3_6::kTokenDomain = 248077`（`src/targets/qwen3_6/export/ninfer/targets/qwen3_6/frontend.h:16`），
  27b 的 `TextConfig::token_domain` 直接取它（`qwen3_6_27b/impl/config.h:20`），
  运行期用在 argmax/sample 域（`text_context_impl.h:676,853,1363,1366`）。
- **运行时侧**：新 target 的 `config.h` 直接写 `output_rows = token_domain = 131072`（照 muse
  `config.h:20-21`），`frontend.h` 的 `kTokenDomain` 不动。✓ 纯填表。
- **前端侧**（比 manifest 描述的更大）：`Package::make_frontend` 传
  `FrontendOptions{.token_domain=131072, .validate_official_special_ids=false}`
  （对应 `frontend.h:18-29`；先例 `27b package.cpp:154-164` 目前**没传** token_domain，走默认 248077）：
  - `validate_registered_tokenizer`（`frontend.cpp:249-271`）要求
    `has_exact_token_domain(131072)`（`tokenizer.cpp:941-945`：`valid_token_ids_` 全 true 且长度相等）
    → 必须显式传 131072；
  - 官方 vision/config 特殊 token 表（`frontend.cpp:54-69`，248053-248076）→ 必须
    `validate_official_special_ids=false` **且转换器不得往词表塞这些 id**；
  - **无条件**的 thinking-control 圆整校验：`frontend.cpp:883-895` 要求
    `tokenizer->encode("\n\n Considering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>\n\n")`
    （`frontend.cpp:42-44`）能**精确往返且 decode 含 `</think>`** —— **没有开关可关**（`待证实`：
    Spark 的 tokenizer 是否圆整成功；失败则前端起不来，只能改家族前端加门或换前端）；
  - chat_template **sha256 白名单只有 2 个**（`chat_template.cpp:414-424`）→ 转换器必须嵌 qwen 模板
    （先例 muse `convert.py:242-253`），Spark 自己的模板会被拒；
  - stop token 取 tokenizer 的 `default_stop_token_ids`（`frontend.cpp:876-882`）。

### 3.7 X2 无 qk-norm（manifest 未列，静默错）

`attn_mix` **无条件**归一化 q/k：

```cpp
// text_context_impl.h:955-959
const auto results = workspace_recipe::text_attention_results<TextConfig>(work_, T);
ops::rmsnorm(q, *w.q_norm, kCfg.rms_eps, true, qn, s);
ops::rmsnorm(k, *w.k_norm, kCfg.rms_eps, true, kn, s);   // ← Spark 不应执行
```

- Muse 也"没有 qk_norm 权重"，但它的解法是**转换期合成全 1 权重**（`with_scale=False`）——
  对 muse 是"归一化但无缩放"，**对 Spark 是错的**：Spark 的 HF 里 q/k **完全不做 RMSNorm**
  （`modeling_spark.py:170-184` 直接 rope）。
- 加法：家族已有 `qk_norm_enabled()` 访问器（`muse config.h:74`）但**全树无消费点** → 需要新增
  `if constexpr (ModelConfig::qk_norm_enabled())` 门（回落 true，27b/muse 行为不变），
  Spark 的 config 声明 false。
- 判定依据（真索引）：每层 8 张权重里**没有** `q_norm/k_norm`（§1.2）。

### 3.8 H7 layers>16（判定：无缺口）

- 家族上限 `kPagedKVCacheMaxLayers = 64`（`decoder_state.h:25`），per-layer 数组一律 64 宽：
  `PagedKVCacheLayout.cold_slots/cold_slot_valid/layer_dtypes/layer_residual/layer_plane_base`
  （`decoder_state.h:57-73`）、`DecoderStateSpec.layer_kv_dtypes/layer_residual`（`:37-39`）、
  `Variant::default_layer_kv_dtypes` 返回 `std::array<DType,64>`（`27b variant.cpp:23`）、
  `PagedKVLayerView.layer_dtypes`（`paged_kv_cache.h:61`）。
- 冷槽按 full-attention 序号索引，容量数组同为 64（`decoder_state.cpp:184-192`）。
- **36 层 < 64 → 无需扩容量**；但要确认两件事：
  ① `TextConfig::full_attention_layers()` 必须返回 **36**（不是 9），否则 `plan_cache` 只布 9 层 KV
  而 `run_layers` 走 36 层 → 越界（`decoder_state.cpp:164` vs `text_context.h:445`；
  **生成的 config.h 现在正是 9**，`config.h:18`：`full_attention_layers() { return 9; }`）；
  ② 每层的 plane 数 × 36（BF16=2 → 72 个 plane；NVFP4=4 → 144）在 page-group 几何里无隐藏上限
  （`plan_device_kv_page_pool` 未见宽度上限，`待证实`）。

### 3.9 X1 主注意力头几何 16Q/4KV@D256（最硬的缺口）

**证据（全链路都是精确注册制，且会硬 throw）**：

```cpp
// src/ops/wrapper/gqa_attention.cpp:25-36
std::int32_t kv_heads_for_q_heads(std::int32_t q_heads, const char* op) {
    if (q_heads == 24) { return 4; }
    if (q_heads == 16) { return 2; }     // ← Spark 16Q/4KV 在这里被判成 2
    if (q_heads == 32) { return 2; }     // Muse-Glimmer
    throw ... "unsupported Q/KV head geometry";
}
// :253-264
const std::int32_t kv_heads = kv_heads_for_q_heads(q_heads, op);
if (cache.num_kv_heads != kv_heads) throw ... "invalid KV cache head geometry";  // ← 4 != 2 → 抛
```

- 注册几何只有 3 个：`Gqa27Geometry=GqaGeometry<24,4,1>`、`Gqa35Geometry=<16,2,2>`、
  `GqaMuseGeometry=<32,2,1,128>`（`src/ops/kernel/gqa_attention_geometry.cuh:25-27`；
  `GroupSize = QHeads/KVHeads`）。
- 解码分派按 `q.ne[1]`/`q.ne[0]`：`src/ops/launcher/gqa_attention_decode.cu:52-76`；
  **注意 `:64` 的 `q.ne[1] == Gqa35Geometry::QHeads(16) && q.ne[0]==256` 对 Spark 也为真** ——
  若上游校验被放过，会**静默按 KVHeads=2/GroupSize=8 算**（少算一半 KV 头 + head 映射错），
  这是比 throw 更危险的失败模式。
- 预填分派同形：`src/ops/launcher/gqa_attention_prefill.cu:372-390`（+:396-408 的 KV append 按
  `k.ne[1]` 分派）；旧因果路径 `causal_softmax_attention` 也写死
  `(24,4) || (16,2)`（`src/ops/softmax_attention/dense/causal_cache/causal_softmax_attention.cpp:26-31`）。
- kernel 侧 group 分支只有 `GroupSize==16` 与 `==6`（`...decode_impl.cuh:283-335` iso3、
  `:415-436` nvfp4），其余落到"35B(group 8)"分支 → **group=4 的 RowTiles/Wc 数学需重新推导+重测**
  （`TokenTile*GroupSize` 相关的行 tile 数见 `:371-375`）。

**要落**：① 新 `Gqa16x4Geometry`（或参数化 geometry 表）+ 4 处分派（decode/prefill/kv_append/
wrapper 校验，KV 头数要从 `cache.num_kv_heads` 推而不是从 q 头数反推）；
② group=4 的调度分支（或把 35B 分支的推导在 group 4 下重算）；
③ 一致性测试（新几何的 oracle 对拍，参照 `include/ninfer/ops/gqa_attention.h:20-47` 的 FP64 ideal）。

### 3.10 X3 融合 QKV 投影不可用 `ops::attn_input_proj`

- 该 op 的契约把行数写死：`[14336,5120]` 父权重、输出 `q/gate [6144,T]`、`k/v [1024,T]`
  （`include/ninfer/ops/attn_input_proj.h:18-50`），NVFP4 的 A16 路径还把
  `kQRows=6144 / kKvRows=1024` 写成常量（`src/ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.cpp:30-31`）。
- Spark 是 `q_k_v_proj: 2560→6144`（=4096q+1024k+1024v）+ **独立的** `g_proj: 2560→16`
  （`modeling_spark.py:152-154`）→ 行数/拆分方式都不同。
- 先例解法：muse 直接把 4 个投影拆成独立 `ops::linear`（`muse load/bindings.h:51-65` 注释 +
  `MuseAttentionProjectionPayload{query,key,gate,value}` `:206-213`）。
  **Spark 建议**：转换期把 `q_k_v_proj` 切成 3 个对象（q 4096×2560 / k 1024×2560 / v 1024×2560），
  payload 用 `{Weight query; Weight key; Weight value; Weight gate;}`，叶子用 4 次 `ops::linear`
  （或 `ops::linear_pair` 一次出两路，见 `include/ninfer/ops/linear_pair.h`），
  避免再撞 `attn_input_proj` 的固定几何。

### 3.11 X4 所有 linear 档位都是精确 (n,k) 表（含 BF16、含 A16）

| 档位 | 注册表 | 证据 |
|---|---|---|
| BF16 dispatch 门 | `legacy/dspark/dflash2` 三张白名单 | `src/ops/linear/bf16/bf16_dispatch.cpp:11-35`（不在表内 `throw "unsupported shape or T"`） |
| BF16 decode(T=1) | 只有 `(14336,5120)`、`(5120,6144)` | `src/ops/linear/bf16/bf16_gemv.cu:27-37` |
| BF16 small-T | 同上两个问题 | `src/ops/linear/bf16/bf16_small_t.cu:37-57` |
| BF16 MMA | 15 个固定 (n,k) | `src/ops/linear/bf16/bf16_gemm_mma.cu:51-113` |
| NVFP4 | 8 个 problem，**A16 也要过 `is_nvfp4_linear_problem`** | `src/ops/linear/nvfp4/nvfp4_config.h:108-151`；`nvfp4_dispatch.cpp:22,85` |
| FP8 | 11 个 problem，**A16 同样要过** | `src/ops/linear/fp8/fp8_config.h:174-197`；`fp8_dispatch.cpp:23` |

Spark 需要的形状（全未注册）：`(6144,2560)`、`(4096,2560)`、`(1024,2560)`、`(16,2560)`、
`(2560,4096)`、`(10240,2560)`/`(20480,2560)`、`(2560,10240)`、`(131072,2560)`。
manifest 的 `post: quant_geometry` 只说"**转换后校验**形状对照注册表（A16 起步）"——
**A16 并不能绕过注册**，所以这是引擎侧要写代码的活（不是校验）。

**首版最省力路径**：BF16 权重 + 在 `select_bf16_a16_launch`（`bf16_dispatch.cpp:11-35`）加
"spark 分支恒返回 `launch_bf16_mma`"（该函数现有注释已说明 MMA core 能 tile 任意 n/k，
`:25-27`），并在 `launch_bf16_mma`（`bf16_gemm_mma.cu:51-113`）补 7-8 个
`Bf16GemvGeometry<n,k>` 实例。**注意 MMA 的 static_assert 要求 `n % kBlockRows == 0`、
`k % kBlockK == 0`（`bf16_gemm_mma.cu:17-18`）→ `n=16` 的 `g_proj` 必须 padding 到 64/128
（或给它单独一个小 kernel）**。

### 3.12 X5 target id / 命名空间（阻塞级，1 行修）

- `tools/archkit/adapt.py:343` 与 `tools/archkit/gen_full_target.py:86,203` 只做 `replace('-','_')`；
  `spark-x2.5-4b` 的 `.` 被留在了标识符里：
  `tools/archkit/out/spark-x2.5-4b/config.h:3` = `namespace ninfer::targets::spark_x2.5_4b::detail {`
  → **不是合法 C++**（其它生成物 `muse_glimmer_30b/gemma4_31b/minicpm5_1b/qwen4_exp` 都没这个问题）。
- artifact 的 `model_id` 是字符串（`ArtifactIdentity(model_id, weights_id)`，
  `tools/artifact/container.py:35-36,184-185`；先例 `inventory.py:35` = `"qwen3.6-27b"` 带点），
  所以 **`Package::model_id` 保持 `"spark-x2.5-4b"`，只把 C++ 命名空间/目录名规范化**（建议 `spark_x2_5_4b`）。

---

## 4. 分步实施顺序与验收判据

> 原则：**每一列"可独立验证"的步骤都能单独编译/单独跑**；打 ★ 的步骤必须与相邻步骤同批落地
> （否则既编不过也测不了）。

| 步 | 内容 | 可独立验证？ | 验收判据（客观、可复跑） |
|---|---|---|---|
| **S54-0** | id/命名规范化：`adapt.py:343`+`gen_full_target.py:86,203` 加 `.`→`_`；生成器支持 per-kind θ（去掉 NoPE 断言 `:45-50,191-198`）与 per-kind `rotary_dim` | ✅（纯 Python + 桩 g++） | 生成物 `namespace` 无 `.`；`g++ -fsyntax-only` 过；`layer_rope_theta` 与 `rope_by_kind` 逐层一致；新增 `layer_rotary_dim` 表长度=36 |
| **S54-1** | 目标骨架：`config.h`（§2.2 补齐）+ `variant.{h,cpp}`（叶子先 `throw std::logic_error`）+ `bindings.{h,cpp}`（对象计划）+ `package.{h,cpp}` + CMake + registry，全 36 层 full、gdn=0、mtp=0 | ✅（编过即可） | `ninfer_engine` 编过；`registry.cpp` 命中新 model_id（构造失败信息是"叶子未实现"而非"no registered target"）；27b/35b/muse 行为不变（同一批叶子未改） |
| **S54-2** ★ | 转换器首版（BF16 权重）+ BF16 linear 几何注册（§3.11）；`q_k_v` 切成 q/k/v 三对象；`g_proj` 先走"复制权重"方案 A | ✅（结构可验） | `tools/convert/<id>/verify.py` 全绿（对象数/形状/identity）；`tests/targets/<id>/test_load_plan.cpp` 通过（materialization 计数 + `device_capacity_bytes>0`）；`NINFER_EXPORT_HEAD_DIR` dump 出 `[131072,2560]` |
| **S54-3** | 数值接线：per-layer theta + per-layer rotary（§3.2/3.3）；`qk_norm_enabled()` 门（§3.7）；`gelu_mul` 新算子（§3.5）；headwise gate（先方案 A） | ⚠️ 要与 S54-2 同批才能出数 | **短 prompt（T≤512，滑窗尚未生效区）与 HF logits 最大绝对误差 ≤1e-2 且 top-1 一致**；逐层 hidden cos ≥0.999（`NINFER_HS_DUMP_DIR`，§4.1） |
| **S54-4a** | 主注意力新几何 16Q/4KV@D256（§3.9）：geometry + 4 处分派 + group=4 调度 | ⚠️ 与 4b 同批 | 新几何的 oracle 对拍通过（FP64 ideal，`gqa_attention.h:20-47` 口径）；单层 GQA smoke（随机 q/k/v vs FP64）误差 ≤1e-2 |
| **S54-4b** | SWA 窗口（§3.1）：per-layer 窗口表 + `layer_view` 接线 + BF16 档窗口支持（或锁 NVFP4 KV） | ❌（必须与 4a/4a 的 KV 路径一起） | **长度 >512 的 prompt：开窗 ≠ 关窗**（接线有效性）；开窗后与 HF greedy 逐 token 一致；`window=512` 与 `511` 的输出差异符合"distance<window 含/不含"的口径判定 |
| **S54-5** | 前端/词表（§3.6）+ sampling 默认值（`temperature=1.0, top_p=0.95, top_k=-1` → 引擎 `top_k=0`） | ✅ | `--serve` 起得来 + 一轮 chat 出 token；`token_domain=131072` 生效（argmax 域）；stop token 正确 |
| **S54-6** | NVFP4/FP8 权重几何注册 + KV 量化（`--kv-dtype nvfp4`）+ rope 固定表性能 + （可选）MTP | ✅ | 量化档与 BF16 档 logits ≤1e-2；`decode tok/s` 基线 |

**必须一起落的组合**（否则出现"编不过"或"测不出"）：
`config.h` 几何 ↔ `bindings.cpp` 对象名 ↔ 转换器 recipe（同一 schema）；
`variant.h` payload 类型 ↔ `variant.cpp` 叶子 ↔ family runtime 的形状假设（gate 行数/rotary 钩子）；
S54-2/S54-3（无 artifact 就没有可对拍的数）；S54-4a/4b（窗口只有在新几何下才可测）。

### 4.1 与 HF 的数值对齐怎么做（Muse 同款，工具已在位）

1. **引擎侧 dump**：`NINFER_HS_DUMP_DIR`（`text_prefill_impl.h:92,140,300`：逐层 hidden + top-k）、
   `NINFER_HEADDBG=1`（`text_context_impl.h:56,944-953`：每层每阶段 probe）、
   `NINFER_KVDUMP_DIR`（`:88,975-993`：post-rope 的 K/V 与 positions）。
2. **HF 侧**：Spark 自带 `modeling_spark.py` + `configuration_spark.py` + `auto_map`
   （`config.json` 里 `auto_map.AutoModel=modeling_spark.Spark2_5Model`）→ 可直接
   `AutoModelForCausalLM.from_pretrained(dir, trust_remote_code=True, torch_dtype=bfloat16)`，
   取 `outputs.hidden_states[L]` 与 `logits`。
3. **判定**：logits 最大绝对误差 ≤1e-2（Muse 的验收线，`_AUTOADAPT.md` S7）；逐层 hidden
   `cos ≥ 0.999`（DFlash/H1 那条线的同款判据也出现在 board 的 E5/E-2）。首轮用
   **T<512 的短 prompt**（窗口不裁剪区），把 SWA 变量隔离掉。

---

## 5. 风险与未知（按严重度）

| # | 风险 | 严重度 | 证据/判据 | 处置 |
|---|---|---|---|---|
| R1 | **16Q/4KV@D256 未注册 + `q_heads==16` 会命中 35B 分支** | 🔴 高（可静默错） | `wrapper/gqa_attention.cpp:25-30,263-264`；`launcher/gqa_attention_decode.cu:52-76`（尤其 `:64`） | S54-4a；先加显式断言（KV 头数不一致即抛），再上新几何 |
| R2 | group=4 内核调度未验证（现有分支只覆盖 6/8/16） | 🟠 中高 | `launcher/...decode_impl.cuh:283-335,415-436`；`geometry.cuh:19` | 需按 `TokenTile*GroupSize` 重算 RowTiles/Wc 并单测（`待证实`） |
| R3 | **SWA 在 BF16/I8/FP8 KV 档完全无效**（只有 iso3/NVFP4 读窗口） | 🔴 高 | `kernel/..._iso3.cuh:31,132`、`..._nvfp4.cuh:134,273`、`...prefill_nvfp4.cuh:1160`；对比 `...decode_bf16.cuh:124` | S54-4b；或首版锁 NVFP4 KV |
| R4 | 窗口语义差 1（含/不含 `distance=window`） | 🟠 中 | 引擎注释 `swa.h:36-38`；HF `create_sliding_window_causal_mask`（`modeling_spark.py:359`） | 512 vs 511 对照实验钉死（`待证实`） |
| R5 | **KV 显存：36 层全量 KV = 144 KB/token**（BF16，4×256×2×2×36）；32k≈4.7 GB、128k≈18.9 GB；且主 KV 池**没有 per-layer 容量**（SWA 层的省显存收益不会自动出现） | 🔴 高 | `decoder_state.cpp:164`（单 capacity 覆盖全部层）；`plan_cache` 无 per-layer 容量参数 | 首版限短上下文；中期要 per-layer 窗口容量（这是新功能，不是"接线"）；`--kv-dtype nvfp4` 可降到 ~1/3-1/4 |
| R6 | rope 全层走 generic 核（θ≠1e7、R=256、heads 16/4 都不在快路径） | 🟡 中（性能） | `launcher/rope.cu:94-115,171-200`；`kernel/rope.cuh:278-327` | 先测后优化；可加 `R=256/θ=1e4` 与 `θ=5e6` 的固定表（表结构已就绪，`rope.cuh:26-34`） |
| R7 | 无 qk-norm 但家族无条件归一化 q/k（**静默**） | 🔴 高 | `text_context_impl.h:958-959`；真索引无 `q_norm/k_norm`；`qk_norm_enabled()` 无消费点 | S54-3 必须含，短上下文对拍能抓到 |
| R8 | 逐头门控缺算子（方案 A 需 +755 MB BF16 / +210 MB NVFP4 的复制权重） | 🟠 中高 | `sigmoid_mul` 契约 `include/ninfer/ops/sigmoid_mul.h:9-20`、`wrapper/sigmoid_mul.cpp:36-40`；`workspace_recipe.h:58-67` | 方案 A 先跑通；方案 B（新算子）排期 |
| R9 | GELU MLP 的两输入乘缺算子 | 🟠 中高 | `include/ninfer/ops/gelu.h:9-26`；ops 清单无 mul；`silu_mul` 硬编码 SiLU | S54-3 加 `gelu_mul`（含 oracle + 测试） |
| R10 | 前端 hard-code：thinking-control 圆整**不可关**、模板 sha256 白名单、`added_tokens_decoder` 必填 | 🟠 中高 | `frontend.cpp:42-44,883-895`；`chat_template.cpp:414-424`；`muse convert.py:250` | 转换器照 muse 打补丁；圆整失败则必须改家族前端（`待证实`） |
| R11 | `fi::Tokenizer` 能否吃 Spark 的 `tokenizer.json`（10 MB，`PreTrainedTokenizerFast`） | 🟡 中 | `frontend.cpp:847-850`；Spark `tokenizer_config.json` 的 `tokenizer_class` | 转换后做一次前端构造 smoke（`待证实`） |
| R12 | 131072 词表与家族 248077 特殊 id 的冲突面 | 🟢 低-中 | `frontend.h:16`；`frontend.cpp:54-69`；argmax/sample 域 `text_context_impl.h:676,1363-1366` | 只需 `token_domain=131072` + `validate_official_special_ids=false` + **不往词表插 qwen 特殊 token**；`131072%128==0` → NVFP4 头行数 padding 无需求 ✓ |
| R13 | 1M 声明上下文（`max_position_embeddings=1048576`）与引擎可见键上限 | 🟡 中（远期） | `gqa_attention.h:13` `kGqaAttentionMaximumVisibleKeys=1'010'000`；`softmax_attention.h:17-18` | 首版不追 1M；`kNativeContext` 先取试验值（如 32768） |
| R14 | page-group 里 plane 数随层数增长（36×2=72 或 36×4=144）是否有隐藏上限 | 🟡 中 | `decoder_state.cpp:62-136`（plane_cursor 线性增长）；未见宽度上限 | 加载/规划 smoke 时观察（`待证实`） |

---

## 6. 需要 M 裁决的 3 件事

1. **target id**：确认 `spark_x2_5_4b`（目录/命名空间）vs artifact `model_id="spark-x2.5-4b"` 的分离写法。
2. **首版权重档**：BF16（要写 7-8 个新 BF16 linear 几何，但无量化表要测）vs NVFP4（几何注册 + 量化表 + 窗口支持天生可用）。
3. **SWA 范围**：首版"锁 NVFP4 KV + 只在 >512 时对拍"是否可接受；还是必须先落 §3.1 的 BF16 窗口支持。

---

## 附：本次分析用到的关键命令（只读，均可复跑）

```bash
# 上游真索引张量清单（290 张，8 类/层）
python -c "import json,re;d=json.load(open(r'models/Spark-X2.5-4B/model.safetensors.index.json',encoding='utf-8'));print(len(d['weight_map']));[print(s) for s in sorted(set(re.sub(r'^model\.layers\.\d+\.','',k) for k in d['weight_map']))]"
# 生成物命名空间合法性
grep -n "namespace ninfer::targets" tools/archkit/out/*/config.h
# 主注意力几何注册表（3 个）
sed -n '20,30p' src/ops/kernel/gqa_attention_geometry.cuh
# 窗口是否有主模型通路
grep -rn "sliding_window_tokens" src/ | grep -v "^src/ops/kernel/gqa_attention_decode_\(iso3\|nvfp4\)" 
# 家族 auto-adaptation 钩子清单
grep -rn "requires {" src/targets/qwen3_6/impl/runtime/*.h
```


---

## §7 协调方（M）裁决 — 2026-09-10 17:31

1. **target id**：C++ 标识符统一走 `cpp_ident()` ⇒ `spark_x2_5_4b`（小数点/连字符一律折成下划线，
   已在 `tools/archkit/adapt.py` 落地并验证：生成物第 4 行是 `namespace ninfer::targets::spark_x2_5_4b::detail`）。
   对外 model-id 仍保留 `spark-x2.5-4b`（HF 侧名字），**不要再造第二套命名**。
2. **首版权重档 = BF16**。理由：Spark 无任何量化产物、且新架构没有几何注册，NVFP4/FP8 都要先有自己的量化与
   几何校验；先用 BF16 把**数值对齐**做成（与 `modeling_spark.py` 逐层比），量化作为后续性能项。
3. **SWA 首版范围 = 上下文 ≤ 512 token**。理由（实证）：`W=512` 时滑窗注意力在语义上等价于全注意力，
   而 36 层全量 KV = 144 KB/token（32k ≈ 4.7 GB），主 KV 池又没有 per-layer 容量，SWA 的省显存收益
   不会自动出现；同时 BF16 解码核的 `window` 语义是"可见键数"、`sliding_window_tokens` 从不被赋值、
   草稿侧 `ops::swa` 是 D128/32/8/W4096 固定域不可复用。⇒ **v1 验收 = 短上下文（≤512）与 HF 对齐**；
   长上下文必须等 SWA 通路（per-layer 窗口表 + 主通路赋值点 + BF16 核语义）真正落地。
