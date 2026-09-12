# _AUTOADAPT.md — 一键全自动适配管线 (2026-09-03 定调)

用户目标 (原话转述):
> 连引擎的适配也要全自动。笨蛋用户把纯粹全新的模型拖进来, 程序跑一会,
> 最后显示"适配完成", 用户直接开 serve 使用。

即: 引擎适配从"archkit 半自动(产物+人工 apply/接线)"升级为无人值守管线,
GUI 导入向导 = 唯一入口。本文 = 管线规格 + 现状清单 + 分阶段执行。

## 0. 端到端状态机 (GUI 导入向导视角)

```
[拖入模型目录/文件]
   │
   ▼ S1 检测与指纹       config.json / GGUF 头 / .ninfer → 模型身份+源格式
   │                    (已就位: model_import.py / gguf_extract.py / convert_runner.py)
   ▼ S2 规格提取+语义    arch_spec → spec; semantics_v5 → 公式窗口 (manifest)
   │                    (已就位: adapt.py, 2026-09-02 注入 semantics)
   ▼ S3 口味/缺口判定    每需求 → covered | engine_hook | new_op
   │                    (已就位: flavors.py + adapt.py 目录; manifest.gaps)
   │
   ▼ S4 代码生成         目标目录自动生成: config.h + variant 叶子 + bindings/
   │                    package/load + CMake 注册   ← v2 stub 生成器 (半就位)
   │  + engine hook 落树  检测到的 hook 以"可编译代码"落进引擎 (非补丁文本)
   │  + new_op? → 精确失败信息 + 指引 (进化点, 不假装成功)
   ▼ S5 权重转换         源权重 (hf safetensors nvfp4/bf16 | gguf | ninfer)
   │                    → .ninfer artifact (键映射 + QAT 解包/重包 + norm 烘焙)
   │                    (半就位: convert.py/gguf_extract 现有; per-arch recipe 待生成)
   ▼ S6 目标编译+冒烟    引擎树内独立目标构建 (build_arch.sh 已有雏形) + serve 自检
   ▼ S7 自动验收         与 HF 参考对拍: logits ≤1e-2 (引擎内 dump 机已就位)
   │                    → 通过: manifest 状态 adapted+verified
   ▼ S8 就绪            服务卡解锁: 直接 --serve / GUI 一键开
   [显示: 适配完成]
```

失败路径: S3 new_op / S5 键缺 / S7 对拍失败 → 停在对应步, 给出
"缺什么、参考哪、重试方式"; 不产生半成品 artifact; 可回滚 (生成物在 out/ 下,
引擎树改动 = 新增目标目录, 不触碰现有目标)。

## 1. 全自动的引擎侧前提 (当前缺口 → 分阶段消解)

引擎共享主干 (qwen3_6 runtime) 目前假设 qwen 混合架构路径全在
(GDN 状态/conv、MTP stem/tail、dflash sinks)。全新 softmax 家族模型
(Gemma-4-31B / Muse-Glimmer-30B) 跑它需要主干"特性化":

| # | 引擎特性 | 用途 | 现状 |
|---|---|---|---|
| E1 | 逐层 rope theta + NoPE 整层跳过 | Muse 13 NoPE full / 39 theta sliding | 待接 (attn_mix, D2) |
| E2 | q 侧 qk_scale_factor (qk_norm 后) | Muse 3.87 / Gemma (qk-norm 已有) | 待接 (D2b) |
| E3 | sliding-window 主模型层 (非草稿) | Muse 39 层 / Gemma 50 层 | 引擎全注意路径查证 (解码核有窗口参数) |
| E4 | 双 norm 层图 (o_proj 后先 norm 再残差) | Muse 四 norm/层 | 待接 (D1) |
| E5 | n_gdn=0 / 全 softmax 层图 | 纯 softmax 族 | run_layers 二分判定, gdn 数组空则天然跳过, 待验证 |
| E6 | tied head (lm_head = embed^T) | Gemma | bindings/load 层处理, 待查 |
| E7 | 逐层 scalar (每层乘系数) | Gemma-4 | semantics 待钉 (拉 modeling_gemma4) |
| E8 | qk_norm 无参 (无 scale 权重) | Muse qk_norm with_scale=False | load 时 weight=None 即可 |
| E9 | 变体目标自动装配 | 所有新模型 | v2 stub 生成器 + CMake 注册 + build_arch.sh |

引擎活全部加 constexpr/特性开关, qwen 现目标行为与性能不变 (回归门:
ninfer_engine + ninfer_serve + 27b 编译 + logit/机制测试)。

## 2. 分阶段执行 (每阶段可验收)

- 阶段 A (本轮): todo 固化 + _AUTOADAPT.md; E1+E2 接进 attn_mix (逐层 theta/
  NoPE 跳过/q 侧 scale, 复用 ops::logit_policy mult-only 作为 x*scale 无新内核);
  编译回归。Muse manifest/config.h 重生成核对 E1/E2 常量。
- 阶段 B: E4 双 norm (FullLayerW 新槽 + attn_mix/mlp_tail 特性分支) +
  E3 window 查证接线。同族 (含 Gemma 如无 scalar) 收进 flavors 词表。
- 阶段 C: E6/E7/E8 钉死 Gemma (semantics_v5 拉 modeling_gemma4 + 键表核对),
  Gemma milestone-1 引擎活收尾 (gemma_engine_plan.md 施工单)。
- 阶段 D: v2 stub 生成器升级 → manifest 一次性产出完整目标目录 (config.h/
  variant/load/package/CMake 注册/build 脚本); adapt.py 改为可直接驱动的
  编排入口 (--to-engine-tree); GUI 导入向导接 S4-S8 状态机 (轮询进度 +
  阶段日志 + 失败指引 + 完成解锁服务卡)。
- 阶段 E: 权重转换 recipe 自动生成 (从 spec.weights + 量化键普查表 + QAT
  包语义) → Muse/Gemma artifact 产出 → S6/S7 serve 对拍自动验收 (logits
  ≤1e-2 vs HF) → 两模型"适配完成"全流程首验 (用户给的真模型案例)。
- 阶段 F: 新架构盲测协议 (拖一个完全没见过的模型跑全流程, 记录每个缺口
  是否给出可执行指引)。

## 2b. 目标装配清单 (2026-09-03 深夜实测枚举, stage D 生成器依据)

一个 exact 目标 = 下列文件 (27b 与 35b 双样本对照, 差异=各变体手写区):
- <id>/CMakeLists.txt: 3 源文件进 ninfer_engine + 2 include dirs
- <id>/impl/config.h: 几何+旋钮+表 (gen_full_target.py 已自动 ✅)
- <id>/impl/variant.h (~279 行): Variant 全表面 = 叶方法 ~15 个 +
  graph profiles 4 个 + 别名 (payload/WeightsProfile/ModelView/GraphExecutionProfile/
  TextConfig 等) + constexpr 常量 (attention_scale/gdn_scale/prefill_chunk_alignment/
  maximum_*_draft_tokens/maximum_context/supports_dflash[2]/draft_head_rows/
  default_layer_kv_dtypes/supports_per_layer_kv_defaults) —— 面已全枚举
- <id>/impl/variant.cpp (~598): 叶子实现 (ops 组合) + instantiate.h 宏实例化
  (NINFER_QWEN36_VARIANT/RUNTIME_NS) + graph profiles 生成 + 校验
- <id>/impl/load/bindings.h (~279): artifact 对象计划 (每层 WeightPlan/对象句柄)
  + 运行时 payload 结构 (FullAttentionProjectionPayload 等, 别名来源)
- <id>/impl/load/bindings.cpp (~788): artifact binder/materializer 键→计划展开
  (27b/35b 全手写且互异: 层公式/is_full/格式表/量化档)
- <id>/impl/package.cpp (~207): LoadPlan/LoadedModel/identity 注册
- <id>/export/ + include/ninfer/targets/<id>/package.h: 公共身份头
- src/targets/registry.cpp + src/CMakeLists add_subdirectory: 注册

结论: bindings/package 深度耦合 artifact 容器 API + 各变体布局, 生成器必须
声明式输入 = "布局 schema"(HF 键→对象计划 + 口味→融合拆分 + 量化档), 即
converter 的同一份 schema 反用; 全自动 bindings 生成与 converter recipe 生成
应共享同一布局描述 (阶段 E 的 recipe 自动生成是 bindings 自动化的前置)。

编译门策略 (给 E3/E4/E5 分支编译实例, 不先做 artifact):
- variant.h 链 (含 bindings.h → artifact/binder.h + qwen3_6 公共头) 为 host C++
  (无 CUDA 内核) -> 可 g++ -fsyntax-only 门 (verify_v3_gen 同款桩环境)
- variant.cpp 需要 nvcc+引擎全 include -> 作为 ninfer_engine 内目标 TU 编译,
  错误驱动枚举剩余表面 (Variant stub 先行, 运行时体 throw)

Muse NVFP4 权重普查 (converter 输入, 2026-09-03 实测):
- 前缀 model.language_model. 需剥离; layers.N. 下 4 个 norm.weight/层
  (input/post_attention/pre_feedforward/post_feedforward) = 双 norm 层图实证
- 无 q_norm/k_norm 权重 = qk_norm with_scale=False 实证 (引擎 weight 槽空)
- mlp: gate/up/down_proj.{weight,weight_scale,weight_scale_2} (双级 scale)
- self_attn: q/k/v/o/gate_proj.{input_scale,weight,weight_scale}
- vision_tower.* 存在 (vision new_op); kv_cache_quant_algo 键存在
- 分片: 3 × safetensors (2.79G+6.19G+2.15G) + index.json + tokenizer



## 3. 参考索引

- 架构目录/缺口判定/口味: _ARCHKIT.md v1-v5 段
- Muse 公式与施工契约: _MUSE_SEMANTICS.md
- Gemma 键表普查与施工: _ARCHKIT.md Gemma 段 + gemma_engine_plan.md
- GUI 导入向导现状: tools/gui/model_import.py + convert_runner.py + ninfer-gui.py
- 引擎侧施工单总表: _TODO.md (9b + 新增条目)


## 4. Muse 破墙揭示的家族不变量清单 (2026-09-04, 适配器下一版需求输入)

Muse-Glimmer-30B serve 破墙链 (9 项修复) 按性质归类, 每类都是适配器要自动化的
"新架构=家族不同" 的缺口:

### A. 转换器侧自动合成 (artifact 资源, 无需引擎改动)
- chat_template.jinja: 引擎前端按 sha256 白名单映射语义 (ThinkingToggle e84f /
  ReasoningEffort c3cf); 转换器需从白名单模板库选一份嵌入 (非目标家族原生模板)
- tokenizer_config.json 补丁: add_bos/add_prefix false + pad_token 家族语义 +
  added_tokens_decoder 字段必须存在 (引擎 require_object_field; 可空对象)
- preprocessor_config/video_preprocessor_config: 引擎按编译期视觉几何无条件校验
  (patch_size 16 / temporal 2 / merge 2 / mean-std 0.5 / rescale 1/255 / size 字段);
  文本模型也必须嵌家族同款资源 (hash 白名单 pinned)
=> 适配器: 转换配方模板携带"家族资源补丁集", spec 只需选家族

### B. 家族编译期不变量 (适配器需生成 per-variant 覆盖, 本次全部手改)
1. token_domain + official specials: qwen3_6 前端原硬编码 248077 + 官方 vision/audio
   特殊 token id 表 (248053-248076); 参数化为 FrontendOptions.token_domain /
   validate_official_special_ids; Muse=202048/false
   => 适配器: spec.vocab_size 自动生成 FrontendOptions 覆盖
2. per-layer 数组容量: kKvLayerStorageSlots=64 是家族上限; cold_slots/cold_slot_valid
   原 16 (恰好 qwen27 full_attn=16) -> Muse 52 越界写/读 -> 数组一律按家族 max 层数
   => 适配器: 生成器审计所有 per-layer std::array 容量 >= spec 最大层数
3. 量化 linear 几何白名单: fp8/nvfp4 linear 是 exact-geometry 注册制 (measured schedule);
   新 (n,k) 必须注册 (geometry alias + problem enum + resolve/is + decode/small_t 开关 +
   调度特化/继承) ; Muse 形状 fp8: 4096x6656/256x6656/6656x4096/19968x6656/6656x19968;
   nvfp4: 19968x6656/6656x19968/202112x6656; A16-only 起步, A8/W4A4 待测量
   => 适配器: spec 的权重形状表 -> 自动展开注册 + 按角色继承调度
4. embedding FP8 gate / gqa wrapper head_dim / scale 期望: 全是 27b 硬编码
   (248320x5120 / 256 / 1/sqrt(256)); 全部改为按权重形状 / cache.head_dim 推导
   => 适配器: 审计 ops wrapper 硬编码几何, 迁移为运行时推导

### C. 环境/基建 (与架构无关但决定破墙速度)
- 宿主 32GB RAM: WSL .wslconfig 内存切换 (构建 26GB / serve 14GB) = wslmem.bat
- 大块 cudaMalloc 瞬时失败/VM 崩溃 = GPU-PV 宿主内存背书不足 -> DeviceBuffer/DeviceArena
  重试(6x2s) + 错误带 free mem
- 构建: 影响面控制在 .cpp / 避免公共头签名变更 (gqa_attention.h 一次变更 = ops 全量 CUDA 重编)
- 参考侧 logits: transformers main 已原生支持 muse_glimmer (modular 生成)
