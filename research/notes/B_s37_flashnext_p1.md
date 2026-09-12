# B_s37_flashnext_p1.md — S37: FlashNext 文本主线 (W7/P1) 移植清单 + 分阶段计划 + 阶段 (a) 补丁

日期 2026-09-09 · B · 产物: `_collab/B_s37_stage_a.diff` (6 文件/7 hunks/219 行,
`patch -p1 --dry-run` **S37_DRYRUN_OK**; 生成器 `_collab/_b_tmp/mk_s37.sh`)。
**流程**: 骨架先行 → M checkpoint 质疑 → M 实读 reader.cpp 定谳 → 三项折叠共识 → 本版。
未落树 (window G 在编 src/), 无 GPU, 无构建。

## 0. M 共识三项折叠 (本版相对骨架版的增量)

1. **几何交叉核验 (M 提议, 已入补丁)**: artifact manifest **无语义几何** (M 实读
   reader.cpp parse_tensor :112-134 — 仅 name/kind/shape/format/layout/offset/bytes;
   tools/artifact/layouts.py 的 geometry 是打包布局非模型几何)。解法: spec 派生常量
   =期望值, **manifest 张量形状**=artifact 侧唯一几何源, 逐量断言相等:
   `hidden/vocab`←token_embd 形状, `q_dim/kv_dim`←layer.3.qsa.{q,k} 形状,
   `intermediate`←layer.0.moe.e0.down 的 K 维, `layers`←layer.<N>. 名字最大序号+1。
   不等或不可推导 → 抛错报双侧值; 张量缺失 → 抛错指名 S28 writer 须产出。
   **从不静默信任常量。** 实现为 `qwen4_exp::validate_stage_a_geometry(reader)`
   (export 头内联, host 可编译 — reader.h 纯 std 头), registry 分支调用并把
   PASS 判词拼进拒绝消息。
2. **接缝清单扩展 (gap 1)**: grep 全 src/include/apps 的 model_id 值级消费者:
   唯一非 targets 值开关 = **context_cost 预设匹配** (context_cost.cpp:210/:245,
   按 model_id+weights_id; 默认表 context_cost_defaults.cpp:55/:67 仅 qwen3.6-27b/
   qwen3.8-27b 两行) — qwen4-exp 走通用回退, 非阶段 (a) 阻塞, 记为阶段 (b) 接缝
   (补默认行或接受通用曲线)。serve/* (http_server/openai_*/request_log/
   serve_options) 均为字符串透传, 无值分支。
3. **TextConfig 普查扩展 (gap 2)**: 扩到 src/ops+apps+include+serve 后**零新增直接
   使用** (ops 对 Variant 泛型); 全树聚合普查确认字段集并多出
   `is_full_attention/layers/rope_theta_at/layer_rope_theta/swa_attention_layers`
   — 合计 ~30 个被读字段 vs stub 13 个 → 阶段 (b) 补齐清单 (昂贵学费已避)。

## 1. 移植清单 — 新目标运行时必须提供的接缝 (load 顺序; qwen3_6 实证 file:line)

| # | 接缝 | qwen3_6 定义处 | qwen4_exp 现状 |
|---|---|---|---|
| 1 | 构建注册 | src/CMakeLists.txt:349-352 add_subdirectory; 目标 CMakeLists 挂 sources+include(impl,export) (qwen3_6_27b/CMakeLists.txt:1-10) | 无 (阶段 a 补) |
| 2 | 公共身份头 `<ninfer/targets/<id>/package.h>` | 27b export package.h (Package/WeightsProfile/Frontend) | 无 export 目录; 旧 stub 有 `{{` 缺陷 (阶段 a 补) |
| 3 | `construct_target` 分派 | registry.cpp:280-302 (identity.model_id) | 无 (阶段 a 补识别分支+几何核验+响亮拒绝) |
| 4 | `construct_registered<T,L,I>` 模板面 | registry.cpp:112-214: resolve_weights(:116)/resolved_auto_speculative(:122)/sampling_defaults(:126)/plan_load(:137)/make_sequence_planner(:157)/materialize(:180)/construct_loaded_model(:183)/make_frontend(:247)/create_program(:197) | 无 (阶段 b 起) |
| 5 | registry.h 包装类型 | registry.h:21-56 Loaded/Instance; 第二消费者 replan_target_kv (registry.cpp:218, InstanceType::Package) | 无 |
| 6 | Variant/bindings/package.cpp | 27b impl/variant.{h,cpp}+load/bindings.{h,cpp}+package.cpp | 无 (键表已备: S26 契约 74,804 行) |
| 7 | TextConfig 被读字段 (~30) | 全树普查: hidden(100)/intermediate(37)/value_dim(28)/key_dim(23)/head_dim(22)/query_size(19)/token_domain(18)/output_rows(12)/kv_size(12)/query_heads(11)/gdn_layers(10)/kv_heads(8)/full_attention_layers(7)/gdn_value_heads(6)/mtp_*(8)/rms_epsilon(4)/rope_theta(3)/convolution_dim(3)/layers(2)/is_full_attention(2)/swa/rotary_dim/rope_theta_at/layer_rope_theta | stub 13 个且 intermediate=None 不可编译 (阶段 a 修 1 个; 其余阶段 b) |
| 8 | 共享家族复用 | src/targets/qwen3_6/ (frontend/runtime/state + export: hybrid_topology/model_view/...) | 同法复用; MoE/PLE/hyper-connection/MTP 层体走新算子 |
| 9 | context-cost 预设行 | context_cost.cpp:210/:245 匹配; defaults.cpp:55/:67 两行现有 | qwen4-exp 缺行 → 通用回退 (阶段 b 决策) |

## 2. 分阶段计划 (M 已接受)

(a) **本轮** 身份+config 可编译+几何核验+响亮拒绝 → 可编树、artifact 被识别且绝不静默装载。
(b) BF16 单 QSA 层前向+embed/lm_head → 首个真实数字; 需 **S28 writer** (emit 本补丁核验的
    张量名/形状)、TextConfig 补 ~17 字段、最小 variant/bindings。测试: 定长 prompt 单层
    logits vs HF (GPU 窗口)。(c) MoE 512×top-10 → 全层 MLP; 分片已在盘、writer 已证;
    **缺 MoE 路由内核 (最大缺口)**。(d) GDN 36 层+hyper-connection 块 (GDN 主路径已有,
    hc 算子无; in_proj_a/b 槽语义待引擎定)。(e) PLE — **FP8 分片阻塞**。(f) MTP 草稿层。

## 3. 阶段 (a) 补丁 (最终版) 与证据

```
config.h        @@ -11,7 +11,14 @@  intermediate: None→640 + M 共识推源注释 (期望值;
                                   交叉核验从 layer.0.moe.e0.down 形状取 artifact 侧值)
package.h       整文件→export 头薄别名 (修 {{ 缺陷)
registry.cpp    +include <ninfer/targets/qwen4_exp/package.h>
                +分支 (通用 throw 前): identity==qwen4_exp::kModelId →
                   verdict = qwen4_exp::validate_stage_a_geometry(reader);
                   throw "... recognized (" + verdict + "), but its runtime is not
                   implemented yet (W7/P1 staged plan, _collab/B_s37_flashnext_p1.md);
                   refusing to load a stub target"
src/CMakeLists  @@ -350,6 +350,9 @@ add_subdirectory(targets/qwen4_exp)
新 export package.h (122 行): kModelId/kTargetKey/kRuntimeStage +
                validate_stage_a_geometry() — 张量形状派生 7 量断言 + layers 名字扫描;
                任一缺失/不符 → 抛错报双侧值 (S28 writer 契约指名)
新 qwen4_exp/CMakeLists.txt: include(export, impl) 无源
```
复现 (全部通过):
```
patch -p1 --dry-run < _collab/B_s37_stage_a.diff  → 6×checking + S37_DRYRUN_OK
g++ -std=c++20 -fsyntax-only config.h:  static_assert intermediate==640/hidden==2560 → CONFIG_H_COMPILES
g++ -std=c++20 -fsyntax-only export 头 (-I src):  static_assert kModelId + 取 validator 地址 → EXPORT_PKG_AND_CROSSCHECK_COMPILE
  (host 检查还抓出一个真错: object_name 返回 string_view 非 string&, 已修)
```
registry.cpp TU 编译归应用窗口 (CUDA 头; 同 S20/S24/S33 先例); 分支体仅新增
validate 调用 + 字符串拼接 (同型用法 identity.model_id registry.cpp:285)。
生成器保留 EOL 修正 (首版曾把 CRLF 文件整文件改写, 已修为逐文件保持)。

## 4. FP8 PLE 分片阻塞声明
阻塞: 仅阶段 (e) 真数据验证 + 最终端到端"出正常文本"门 (layer-1 PLE 逐 token);
sidecar FP8→BF16 物化决策同样等 model-plefp8-* 十分片。不阻塞: (a) 已完、(b)
(model-bf16 在盘)、(c) (experts 分片全在盘)、(d)、(f) (mtp 随 bf16 尾片)。
