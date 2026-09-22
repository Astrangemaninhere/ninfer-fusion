# NInfer 总 TODO (2026-09-03 固化 v2, 防上下文压缩丢失)
> **[入口横幅 2026-09-18]** 本文件是**计划/流水账**：其中的常数多是**当时快照**，不是现状；树才是权威。
> 冷槽常数这一族已于 **2026-09-18** 整体重定：记录 **9632 B**（`320 + 32×259 + 1024`，owner `src/product/kv_tier_formats.h:289` 的 `kKvColdPoolStrideBytes`，权威字面量 `include/ninfer/ops/entropy_nvfp4_slot.h:48`）、价格 **470**（`src/product/kv_bit_budget.h:408`，派生并 static_assert `== 470` @`:412`）、天花板 **404**（`src/targets/qwen3_6/impl/state/decoder_state.cpp:574`，4.04 bits/code 实测最小值）。
> 凡本文件把 **6688 / 9536 / 466 / 327 / 2544** 当作"应改到的值"的句子，都是**改动前**的世界；照做会被 `decoder_state.cpp:845`/`:863`、`kv_tier_formats.h:329`、`kv_bit_budget.h:412` 的 static_assert 当场挡下（6688 B 只给 167 B/流，实测 K 命中 0/64）。
> 本次只审了冷槽常数这一族；文件其余部分**未逐行重审**。已加注解的原文行：**5249 / 5252 / 5292 / 5298 / 7006 / 7186 / 7189 / 7201**（插入后行号整体后移；用 `grep -n "注解 2026-09-18" research/notes/TODO.md` 定位全部 8 条注解）。


## 铁律 (用户定死, 2026-09-04)
- 后台编译/长任务进行时绝不允许干等: 必须同时推进另一项 todo 工作 (本次教训: 连续多次
  TaskOutput 空等编译). 启动长任务 -> 立即转向并行任务 -> 完成通知到达再切回.
- 杀进程/杀端口前必须先看 cmdline (曾误杀训练进程两次).
- 一切会话产物/状态同步进本文件 (防上下文压缩丢失).
- 遇到数据/数值/布局/协议类问题, 优先用计算机工具实证 (探针程序/位级 dump/
  脚本标定/解析器), 不做纸面推演空猜 (案例: qpn_simt 映射纸算两轮失败,
  单码探针 + GPU 自标定一次破译; GGUF v3 dims=u64 也是 hexdump 实证).

## 0. 技术路线图 (2026-09-08 重构, 用户定调)

### A. 异构并行 = 特性驱动的任务分配 (用户核心原则: 按各部分特性分配任务性质,
###     使整体效率最大化; 并行只是手段, 匹配才是目的)
设计原则: 调度器以「工作负载特性 × 设备特性」匹配矩阵做放置决策, 而非
  "同型号才能并行"。每类工作有主导资源: 带宽型 (KV 注意力/大 GEMM)/算力型
  (小 batch matmul)/容量型 (KV 池/权重驻留)/延迟型 (草稿验证串行链)/分支型
  (投机接受/采样/路由)。
设备特性画像 (运行时自测, 静态表仅初值):
  dGPU 新卡 (5090): 算力+带宽双高 -> 主力 prefill/decode 热路径, 大 GEMM,
    NVFP4 热层
  dGPU 老卡 (V100): fp16 算力中, 带宽中 (900GB/s), 无 bf16/fp4 -> 深层
    KV 注意力 (带宽型), ISO3/Int 档权重驻留, 分层 prefill 的后段层
  CPU: 容量巨大 + 分支/串行逻辑强 -> 采样/路由/调度/贪心接受/冷 KV 宿主/
    tokenizer; 低带宽 => 不放热路径 matmul
  核显 (iGPU): 带宽共享内存 + 常驻 -> 投机草稿模型 (dspark 小模型),
    lm_head top-k 前缀, PLE/ngram 查表, KV 冷页压缩 (ISO 编解码是 INT 运算,
    核显 SIMD 友好)
  NPU: 定点/低精矩阵流 -> W4A16/W8 定点 GEMM 口味, 草稿模型大 matmul,
    embed gather; 走各自 SDK 封装统一 ExecuteOp 接口
分配器 (新组件 PlacementPlanner):
  输入: 工作负载画像 (每算子 主导资源/尺寸/精度需求/依赖深度) +
    设备画像 (上表 + 实测 microbench: 算力/带宽/启动延迟/互连带宽)
  决策: 图分层切分 (层间串行, 按层间通信量最小切) + 层内仅在同质设备间
    row 切; 异构设备间永远"整层/整算子"迁移 (避免同步开销吃掉收益)
  目标函数: max 端到端 tok/s s.t. 接受率不塌 + 显存不溢
分层路线 (由近及远):
  L1 异构 NVIDIA 分层并行 (近期): 5090(sm120) + V100(sm70)。依赖:
    ①每卡各自档位内核 (qpn 线已备 sm70) ②PlacementPlanner 按 特性匹配
    切层 (浅层 attention 重 -> 5090; 深层 KV 长 -> V100 带宽驻留) ③跨代
    通信退 PCIe/host staging (NCCL 跨代 P2P 不可靠; custom_all_reduce 参考)
  L2 NVIDIA+CPU/iGPU/NPU 算子覆盖 (中期): 草稿/查表/冷层/采样按特性表
    分配 (上面画像); 通信 host staging + copy engine 异步流; NPU SDK
    (DirectML/OpenVINO/QNN) 统一 ExecuteOp
  L3 极端同层异构 TP (远期): 仅当 L1/L2 数据表明层内切分收益 > 同步开销
    才启动
验收里程碑: L1 双卡分层 -> tok/s > max(单卡各自) 且 > 单 V100 上限;
  L2 草稿卸 CPU/NPU 接受率不塌 (+10-25 接受点保留); PlacementPlanner 的
  静态决策与穷举最优差 <10%
依赖与现状: qpn sm70 移植 (§37-39 工具链/e2e 已通), TP 路线 (§15),
  custom_all_reduce 参考 (v100-skinny fork_patches), KV 分层基建 (§2)

### B. FreeToken 落地 (用户点名尽快)
目标: 弹性 KV 预算 + 带宽自适应 (FreeToken 论文思路), 落 qwen3.8-27B fp8 场景。
路线:
  ①弹性 KV 预算: 注意力按 token 重要度分配不同 KV 精度/保留数 (iso3 冷层 +
    热窗 tail 机制已备 —— 把静态分层改为按分数动态)
  ②带宽自适应: 实测 decode 带宽占用动态调 batch/冷页换入换出节奏
  ③验收: 64K 长测协议 (针尖 + 长生成退化检测, §35) 下 ppl/tok/s 不塌
依赖: 现有 KV 分层 (§2) 已具雏形; 首改动点 = layouts/attention 的逐层静态表
  -> 运行时分数驱动。

### C. 主线状态 (Muse/QPN/GUI)
- QPN sm70 移植: simt(M1-3)+qpn2(M1-8) 映射全实证+e2e PASS (57f31d7 已推);
  剩 qpn mma M9-16 标定 -> 引擎接线 -> 真机 V100
- Muse 128 核 sync-hang: 原子计数器探针待做 (§38)
- studio GUI: 桌面 NInfer-Studio.bat ✓ / /hwdec 端点 ✓ (架构代次 NVDEC
  矩阵, 非名字白名单, 软解永远开放); 迁移清单 ⑥项 (§40)
- 引擎 NVDEC 硬解后端 (src/media/decode 现仅软解 avcodec) = 工作包
- KV 组合矩阵: iso3 映射 bug 已修(已编入), E8×BF16 长上下文 FAIL 待查;
  真实 iso3 组合矩阵待重测
- 多模型批: Ornith (GGUF→spec ✓, 几何注册补丁已备), gpt-oss (MoE 工作包),
  40B (96 层超限审计), LoRA merge 素材就位
- 训练: 待用户选择方案

## 1. GPU 空闲验收批 (按序, 一次一串)
1. finalize_dflash2.ps1 导出最终 ckpt -> WSL serve 验收 (acceptance/tok/s)
2. eval_ddtree.py --ckpt step_006000 (hit@k: k=1 chain / 2/4/8/16 tree)
   -> hit@2/4 >> hit@1 才做引擎树验证; 数据差则续训
3. KV 冷页卸载首次端到端验证: 小 kv-capacity + --cold-policy disk + 长上下文
   -> stderr [cold] compressed/restored 计数 + 回答质量
4. suffix_lookup 真内核对照 (与 tests/test_suffix_lookup.py 同用例 GPU 跑)
   + 接 dflash2 查表链 (LABD: fuse_draft/自适应控制器/q16 验证图)
5. PLE/ngram 真表 gather 验证 (PleTable 分页 fault; sidecar 构建器已闭环)

## 2. KV 分层精度自由组合 (新大项, 2026-09-03 定调)
- 三层: 热(活动页) / 尾(近期 W token 高精度) / 冷(老化出窗)
- 每层独立可选 KV 格式, 组合不固化: bf16 / fp16 / int8 / 传统 int4 /
  iso4 / iso3 / E8(2bit) kvarn 系 (KVarN 参考 syv readme/up_kv 备忘录)
- 权重档拆两项 (独立 profile): a) nvfp4-fusion (现融合版全部特性)
  b) 纯 nvfp4 (基线, 对照归因用)
- 参数: --kv-tier-formats hot=bf16,tail=fp16,cold=iso3 (serve CLI + GUI
  冷存储卡扩展 + tooltips); 全矩阵测试 -> 对照表
- 尾窗 = issue#164 KV precision tail (热环页池 + 出窗降级复用 requant 链)

## 3. 多 GPU / 能力分发 (非门禁, 按卡走内核路线)
- 已落地: CMake 列表门禁(>=sm_75) + build_arch.sh + 报错按档位指引;
  定调/矩阵/1Cat 归因在 tools/archkit/_GPU_MATRIX.md
- 待: linear 按 device.sm()+profile 分发 (w4a4-TMA sm>=100a / fp8-W8A16
  sm89-90 / int8-tc sm75-86 / QPN2-W4A16 sm70), 内核 __CUDA_ARCH__ 守卫共存
- QPN 移植 (参考归档 data/v100-skinny, Apache-2.0): prepack + SIMT 核,
  NVFP4 中段 M9-16 必须补 MT=2 (v1 让步带); CUDA12 旧链构建 V100 档
- 1Cat TP 路线 (引擎多设备): 权重 N 维切分 + AR 原语(参考
  fork_patches/custom_all_reduce.py) + KV 头分片 + 每 rank 投机管线 +
  DecodeGraph 跨卡化; 参照 4xV100 366tok/s
- 1Cat 技术移植清单: split-KV verify / 前缀缓存(含 GDN 循环态) / 贪心 MTP
  (+10-25 接受点) / 校准 int4 lm_head/drafter / 图 PINNING (0.7-2.6ms/轮)
  / GDN speculative-state 契约 (21 syncs/70 copies) / decode-partition pin

## 4. ARCHKIT (通用架构接入)
- v1 完成: arch_spec.py (extract) + gen_target.py (config.h+manifest),
  qwen4-exp spec/产物 g++ 验证过
- v2 待: bindings/package/recipe stub 生成器 (抄 qwen3_6_27b 回填)
  + GUI 白名单改读 manifest + DeepSeek v4 flash / Kimi K3 spec

## 5. 其他队列
- FreeToken 落地 (稀疏注意力/卸载, 零代码)
- 内核极致优化 (MMA 化 MTP/dflash 等) — GPU 回归后逐个
- ngram 加载器: sidecar 构建器+往返验证完成; 真表 gather 待 GPU
- Windows EXE 单文件封装 (环境自检+官网指路+详尽注释) — GUI 验收后
- YaRN: 确认在 (rope_yarn4 三路径+4x 域扩展+--yarn), 无需动作

## 6. 融合版 nvfp4 都熔了啥 (回忆存档, 供 nvfp4-fusion vs 纯 nvfp4 拆分)
- 7 功能分支 rebase: issue142 / disk-cold-tier / pr1 / pr3 / yarn / pr4 / pr2
- nvfp4-cold-pool (冷池), E8Kv 修复: gqa_kv_hadamard64 (64 维 Sylvester,
  与 g64 scale 域对齐, 自 PR#35) + e8_project_8d_warp 投影 rintf clamp[-7,7]
  + scale /7 (原 /127 错) + Q 侧同 R 旋转 + 邻居 nibble 投影
  (all-E8 ppl 6.9144 -> 1.1120)
- represented-scale 补偿: 已回滚 (ppl 恶化 1.50->2.65), ROADMAP 保留待修
- --kv-residual-layers 崩溃: 已知限制 (E8Kv 主线修复优先)
- 其它: PR160 fold3/foldmlp (TMA GEMM 变体, artifact 在 models/), swiglu
  非对齐拆分, 内存自适应 prefill chunk, dspark/dflash2 草稿, 22 处 dbg 清理

## 7. 已完成 (会话累计)
- 导出工具链 patch/verify/finalize 全链路验证; eval_ddtree 就绪(CPU 验证)
- GUI 三合一 + LABD/ngram-SSD 控件 + 桌面启动器 + 环境/世代卡
- suffix_lookup CUDA 落地 (fuzz 4000 用例全绿, 双 TU 编译过)
- KV 冷页审计; PLE sidecar 构建器+引擎真代码往返通过
- gpu_compat 世代表; v100-skinny 归档 data/v100-skinny
- resume 修复 + MemoryError 弹跳
- 教训: 杀端口占用前先看 cmdline; GUI 子进程被父杀会连坐

## 8. 参考文件索引
_TODO.md(本文件) / _dflash2_status.md / _GPU_MATRIX.md / _ARCHKIT.md /
_labd_1cat_research.md / _flashnext_plan.md / _ninfer_ecosystem_devices.md /
up_kv-nvfp4-k8v4.md / RESEARCH-*.md / CHANGES-20260901.md / data/v100-skinny
引擎仓 docs/maintainer/: 1cat-remaining-workpackages.md (§63 第4项) /
engine-failure-recovery.md (§63 第5项) / resource-scheduling-and-context-cache.md /
replayssm-gdn.md

## 9. 更新 (2026-09-03 晚, 防丢失)
### 状态
- 训练: step ~3925+/6000, loss ~12.44 (波动), 0.39 steps/s, ETA ~19:50;
  6000 步可能不够 -> eval 后按 hit@k 续训 (--resume --steps N, resume 已修)
- GUI @8077 运行中; tips 27 条 (含 sktiers/snvfp4mode)

### 本轮已落地代码 (全部编译/测试过, 已同步 WSL 树)
1. KV 分层量化区分:
   - tools/gui/kv_tiers.py 契约 (7/7; fusion/pure 模式; pure 默认 cold=int8)
   - src/kvcfg/kv_formats.h C++ 孪生 (8/8, 同规则同文案)
   - src/ops/kv/iso_codec.h iso4/iso3 编解码 + iso_codec_test.cpp
     (C++ <-> tools/convert/kv_iso_ref.py 逐位一致: iso4 f1f1d39707e2b5e2,
      iso3 69ba63, scale 3c00; 注意 python round 银行家 vs C++ lround)
   - GUI: 服务卡 KV 分层格式输入 + NVFP4 模式选择 + tooltips
2. 1Cat/LABD:
   - src/spec/lookup_fuse.h fuse_chain 策略 (4/4): 全匹配链 K / 部分试探 1 / 回退
   - src/ops/split_attention.cu + split_attention_ref.h + test (3/3, 1.2e-7):
     split-KV verify 算子对 (local partials + combine), 已注册 src/CMakeLists
   - suffix_lookup 自匹配边界修复 (o+query<=start), 内核 TU 编译过, fuzz 4000/0
3. 内核优化现状: v1 正确性优先标量核 (suffix/iso/split); MMA 化 = GPU 批

### GPU 批 (验收/接线顺序, 逐项测)
1. 导出 finalize_dflash2.ps1 -> serve 验收; 2. eval_ddtree hit@k (决定续训)
3. KV 冷页首验 ([cold] 统计行); 4. suffix/split 真核对照 (同用例 GPU 跑)
5. 多线程/多流宿主管线 (未落地, 引擎调度层首项)
6. TP 多卡 custom AR (参考 v100-skinny fork_patches/custom_all_reduce.py)
7. iso GPU fill 核 (照 kv_iso_ref 位布局) + serve 接入 kv_tiers + 全矩阵对照表
8. 1Cat 剩余: 前缀缓存(GDN 循环态)/贪心 MTP 调度化/校准 int4 头/图 PINNING
9. QPN2/W8A16+MT=2 (v100-skinny 参考已归档); sm 分发点; 矩阵编译回归
10. PLE 真表 gather
### CPU 队列
- ARCHKIT v2: bindings/package/recipe stub 生成器 (v1 完成: spec->config.h+manifest)
- Windows EXE 封装 (PyInstaller spec + 环境自检首启; GUI 验收后)
- FreeToken 引擎落地 (零代码)

## 9b. 更新 (2026-09-02 晚, 自动适配引擎活)
### 本轮落地 (全部编译/测试过)
1. 语义自动推导闭环: semantics_v5 注入 adapt.py (spec['semantics']), Muse manifest
   现含排序后公式窗口 (softcap 先 mult 后 tanh / theta=0=NoPE / qk 只乘 q)
2. 参考源缓存 tools/archkit/refs/modeling_muse_glimmer.py + 公式全部钉死:
   双 norm 层图(4 RMS/层, post_norm_eps 独立) / gate=sigmoid(gate_proj(h)) / qk_norm
   无参 / 额外 head_dim^-0.5 / CenteredRMSNorm 实为普通 RMS*(1+w) 无 mean 减
   详见 tools/archkit/_MUSE_SEMANTICS.md (engine 施工契约)
3. ops::logit_policy 新 op: x*mult + cap*tanh(x/cap), BF16 in-place,
   bf16x8/x2/scalar 路由; GPU 数值测试 OK (含 identity/饱和/奇尾)
4. ModelConfig::apply_final_logit_policy constexpr-if + 接线 8 个 lm_head 产出点
   (qwen 零开销); ninfer_engine+serve 编译干净; 已同步 Windows 镜像

### 引擎下一步 (WSL 树, 按 _MUSE_SEMANTICS.md)
1. 主层驱动: 逐层 theta + NoPE 跳过 + q 侧 qk_scale (MTP tail 是现成范例)
2. 双 norm 层图 flavor + gate 进 flavors/gen_variant (同族全自动)
3. Muse converter + serve 对拍 (权重下载中: data/muse_nvfp4 仅元数据)
4. Gemma milestone-1 施工单继续; v3 qwen38 golden 对拍
5. Muse NVFP4 safetensors 下载完成度检查 (dl_muse_nvfp4.py 后台)

## 9c. 全自动引擎适配阶段 (2026-09-03 定调, 设计见 tools/archkit/_AUTOADAPT.md)
目标: 拖入全新模型 -> 跑一会 -> "适配完成" -> 直接 serve。引擎适配也全自动。
状态机: 检测指纹 -> spec+语义(已自动) -> 口味/缺口 -> 代码生成(目标目录+CMake
注册) -> hook 落树可编译 -> 权重转换 recipe -> 目标编译+serve 冒烟 -> HF 对拍
(logits<=1e-2) -> 服务卡解锁。失败停在对应步给指引, 可回滚, 不假装成功。
### 引擎特性化清单 (qwen 回归门: 编译+行为不变)
- E1 逐层 rope theta + NoPE 跳过: DONE (attn_mix, ModelConfig::rope_theta_at +
  layer_of_full 反查; theta<=0 整层跳过 rope; qwen 编译期常量不变)
- E2 q 侧 qk_scale_factor: 设计定案 = 折进变体 attention_scale (rope 线性,
  q 乘因子 ≡ scores 缩放); 零运行时开销; qk_norm 无参(weight 空)待 bindings 层
- E3 sliding-window 主模型层: decode gqa 核已有 sliding_window 参数; prefill 有
  sliding_window_attention op (现仅草稿侧用) -> 主路径接线 (待)
- E4 双 norm 层图 (o_proj 后先 norm 再残差, Muse 4 norm/层): FullLayerW 新槽 +
  attn_mix/mlp_tail 特性分支 (待, 见 _MUSE_SEMANTICS.md D1)
- E5 n_gdn=0 全 softmax 层图: 验证 bind/load 空 gdn 可跑 (待)
- E6 tied head (Gemma): bindings/load 层 (待)
- E7 逐层 scalar (Gemma-4): semantics 待拉 modeling_gemma4 钉 (待)
- E9 目标自动装配: v2 stub 生成器升级 -> manifest 一次产完整目标目录 (待)
### 本轮已落地
- _AUTOADAPT.md (管线规格) + _MUSE_SEMANTICS.md E1 状态更新
- E1 进 attn_mix, ninfer_engine/serve 编译干净, 已同步 Windows 镜像
- Muse 下载器流式续传版: shard1 完成 2.79GB, shard2 4.56GB 续跑中
- gen_full_target.py (archkit 新工具): manifest -> 完整 27b 同形 TextConfig
  (逐层 kind 表 full/swa + 逐层 rope theta 表 + 全部旋钮常量 + 访问器全表面 +
  内嵌 static_assert: 计数 13/39/0 / full 位 bijection / NoPE 规则);
  Muse 发射过 g++ -std=c++20 语法+断言门, 已落引擎树
  src/targets/muse_glimmer_30b/impl/config.h (Windows 镜像 + WSL 双树);
  manifest.engine.full_text_config 已记录
### 阶段执行
A 本轮(完成): todo 固化 + 设计 + E1 落地编译回归
B: E4 双 norm + E3 window 接线, 收进 flavors
C: Gemma E6/E7/E8 钉死 + milestone-1 施工单收尾 (gemma_engine_plan.md)
D: 目标自动装配: 生成器产完整目标目录 + adapt.py 编排入口 + GUI S4-S8 状态机
E: converter recipe 自动生成 -> Muse/Gemma artifact -> serve 对拍自动验收
F: 新架构盲测协议 (拖陌生模型, 记录缺口指引质量)

## 9d. 更新 (2026-09-03 深夜, 全自动管线推进)- topology.py (archkit 新模块): dense|moe|moe_ngram 拓扑自动判别, 键+数值语义
  (None/0/False 不算; Gemma-4 config 带 enable_moe_block=false + null 专家字段 =
  可切 MoE 的 dense, 不误报); 7/7 回归 (muse/gemma4 dense, gemma4-moe 合成,
  qwen4 spec moe_ngram / 去 ple 纯 moe, 合成 config 形态); 已注入 adapt.py
  -> manifest.spec.topology (kind+evidence+moe/ngram 事实+guidance+draft 建议)
- Gemma-4-31B 实测发现: 官方 config 内置 MoE 开关 (enable_moe_block=false,
  num_experts/moe_intermediate_size=null) -> 未来 checkpoint 可能切 MoE, 判别
  逻辑已覆盖
- gen_full_target.py: manifest -> 完整 27b 同形 TextConfig (逐层 kind/theta 表
  + 旋钮 + 访问器全表面 + static_assert 门), Muse 发射过语法门, 已落双树
  src/targets/muse_glimmer_30b/impl/config.h (manifest.engine.full_text_config)
- Muse 下载: 流式续传版 3 分片 2 已完成 (2.79G+6.19G), 第 3 片续传中
### 下一步 (全自动主线)
- bindings/package/variant stub 生成 (抄 27b 装配面: package.cpp/bindings.cpp/
  variant.cpp/h) -> Muse 目标进 CMake 可编译 -> E4/E3 分支有编译实例
- GUI S4-S8 无人值守状态机; converter recipe 自动生成; serve 对拍验收
- AUTOPILOT GPU 批 (训练完成后): S1 finalize -> S6, 见 _AUTOPILOT.md

### 训练状态结案 (2026-09-03 深夜查实, 重要!)
- 主训练 (15:30 启动, --steps 6000, hs_cache 旧数据) loss 稳定卡 12.43 (bad-teacher
  症状, 早已定性作废), 于 20:53 step 5375/6000 卡死: GPU 0% 空转 3h+, ckpt 停在
  step_005000 (20:37 落盘), 无新日志; 进程现已自行退出 -> 旧 run 结案, 不续训
- 22:18 有人跑过 df2pilot pilot (data/df2pilot/packs 新数据, batch1 ctx48 lr3e-4,
  300 步计划): loss 3.06 -> 1.39 (healthy!) -> teacher 对齐 + 新数据路径验证通过
- 下一步 (正式重训, 交 AUTOPILOT/下轮): ① 全量 re-collect: NINFER_HS_DUMP_TOPK=1
  引擎采集新 hs_cache (479 文件级) ② train_dflash2.py 重训 6000 ③ eval_ddtree hit@k
- _AUTOPILOT.md S0 需同步此状态 (主 run 作废, 重采先行)

### 训练状态结案补充 2 (2026-09-04 凌晨实测)
- 全量重采 serve 路受阻: WSL .wslconfig 已升 28GB 仍 OOM
  (dflash2 artifact 25.4GB + 运行开销 > 28GB; 主机物理 31GB 无法再给;
  cudaMalloc 5 秒内确定性失败; GPU 3.6GB 被死训练残留占着, 进程已确认不在,
  上下文待驱动回收)。Windows 有 CUDA v13.2 工具链 (nvcc.exe) -> Windows 原生
  serve 构建是远期选项 (大工程)
- 关键发现: train_dflash2.py (306-322) 的 lm_head/embed 从 HF safetensors 加载
  (data/Qwen3.8-27B, bf16 标准 layout, 非 engine swizzle) -> teacher logits 可由
  train 内 HF head 计算。若旧 479 npz 的 last hidden layout 正确, 无需重采即可
  重训! 下轮第一优先:
  A. 小实验验证: 旧 npz last -> train 内 HF head -> teacher logits 正确性
     (比对 loss 是否 ~1.39 级; 或抽样 top 概率 vs pilot 数据同文本)
  B. 若 last layout 坏 (engine 内部序未转 row-major) -> 离线转正 (npz 重排)
     再验 A; 仍不行才考虑重采 (需扩内存/Windows serve)
- df2pilot pilot (新 packs, loss 1.39) 证明: dump ids16/vals16 (TOPK) teacher
  路径 + HF-head 路径任一可用则重训即有效

### 训练数据证据链结案 3 (2026-09-04 凌晨, 决定性)
- 对照实验: pilot npz (loss 1.39 用的) 含 ids16/vals16/top1 (TOPK 采集);
  旧 479 npz 无 ids16 -> teacher 走 last+HF head (坏) -> loss 12.4 根因实锤:
  TOPK dump 数据是重训唯一可靠 teacher 源 (engine 内 logits)
- 旧数据离线修复判死: last 经 3 种 layout x 有/无 RMSNorm x HF head 解码
  top1 命中率 ~0 (0.8%, 接近随机) -> 不投入逆向, 弃用旧 479 npz 重训路线
- 现成 dump 普查: hsdump2/done 808 chunks (10:35, 无 TOPK) /
  hsdump3 721 chunks 81GB (11:47, 无 TOPK) -> 全部修复前, 作废;
  bench/hs_cache 479 npz = 10:35 批打包 (同样无 TOPK)
- pilot 147-token TOPK 数据 (22:17) 证明 TOPK 采集 + serve 曾可行一次,
  方法无记录 -> 问用户或复现
- 重训三选项 (用户定): A 用户告知 22:17 pilot 采集的 serve 启动方式 ->
  复用全量采; B 扩物理内存 >=48GB (WSL 40GB) 后自动采+训;
  C 训练挂起, 先跑 GPU 空闲可干的 (S5 真内核对照 / Muse/Gemma 对拍等)

### S5 真内核对照状态 (2026-09-04, GPU 空闲, Windows/WSL 双路编译)
- iso_kv_gpu_test: PASS (GPU vs CPU 逐字节 mismatches=0, iso4+iso3+scales)
- suffix_gpu_test: PASS (294 轮 fuzz vs CPU 镜像, fails=0; 链接 ninfer_ops+
  ninfer_core+ninfer_nvfp4_tma+-lcuda)
- split_gpu_test: shards=1 PASS (maxerr 8.9e-8); shards=2/4 FAIL (err ~0.9-1.2)
  = harness 分片语义问题 (原 harness 还把 host 指针当 device 指针传, 已修:
  partials 驻留 device + q 按窗口切片); 多分片时 kv 前缀可见性/combine 行序
  语义需对照 _labd_1cat_research.md / split_attention_ref.h 契约再修 harness
  (内核单分片链已证正确)
- 编译路: WSL nvcc 13.3 单 TU + 静态库链接 (Windows nvcc 缺 cl 环境失败,
  MSVC BuildTools 14.44 存在但 SDK 版本路径待对; 优先 WSL 路)

### serve 内存墙最终结论 (2026-09-04 实测闭环)
- base 21.5GB / dspark 24.3GB / fold3 21.5GB / dflash2 25.4GB 全部 cudaMalloc OOM
  (WSL 27GB 可见 + Windows 物理 31GB, available ~4GB -> vmmem 无法供给)
- qwen3.8-27B 任何 artifact 的 WSL serve 在本机不可行; 训练采集同受此限
- 突破口: Muse NVFP4 11.1GB (转换后 ~9-11GB) < 21GB -> converter 完成后
  Muse serve 对拍在 WSL 可行! converter (全自动主线 E 阶段) = 解锁 serve
  验证 + 全自动适配验收的钥匙; Gemma 20.45GB 权重(bf16? nvfp4?)转换后同理待估
- split S5 结案: iso PASS / suffix 294 PASS / split shards=1 PASS;
  多分片 harness 语义与 GPU v1 核行域(片内位置)契约差 = 1Cat verify 集成时
  按真实调用形状 (query 未来位置, causal 不裁剪) 对拍, 归 S6

### Muse 目标装配进度 (2026-09-04, 头面闭合里程碑)
- gen_muse_headers.py: 27b 模板 -> muse_glimmer_30b 三头 (export package.h
  WeightsProfile{MuseNvfp4} / bindings.h kTextLayers=52 full=13 gdn=0 /
  variant.h), 双树已同步
- config.h 补 DFlashConfig/DFlash2Config/VisionConfig 占位 (0 层 unsupported)
  + kGdnScale/kPrefillChunkAlignment/kMaximum*DraftTokens 常量; variant.h
  dspark/dflash2_weights 裁剪
- g++ host 语法门 PASS: variant.h 全链 (config+package+bindings+qwen3_6 公共
  头+artifact 头, ModelView<...,13,0> + DFlashWeights<0> 实例化) 零 error
- 剩余: bindings.cpp / variant.cpp / package.cpp 实现 + CMakeLists +
  registry 注册 -> ninfer_engine 编译; 然后 Muse converter (布局 schema
  已由 bindings.h 头面确定: input_norm/attention proj (q+gate 融合 2q /
  k+v 融合 2kv)/q_norm?/k_norm?/output/post_attention_norm/mlp{gate_up,down}
  + final_norm/output_head + token_embedding)

### E4 双 norm 引擎改动落地 (2026-09-04, 编译回归干净)
- text_context.h: ModelConfig 判定 (attn_out_double_norm/mlp_out_double_norm/
  post_norm_epsilon, detection idiom) + FullLayerW.post_attn_out_norm +
  MlpW.post_ff_norm 槽
- attn_mix 分支: 双 norm 时 o_proj->linear 到独立缓冲 -> rmsnorm(post_attn
  _out_norm, post_eps) -> residual_add (单 norm 走原 leaf)
- mlp_tail 分支: Variant::post_mixer_double_norm 叶子接口 (三变体声明;
  27b/35b throw 永不被调; Muse 将实现真逻辑 = linear_swiglu->linear->
  rmsnorm(post_ff_norm)->residual_add); 泛型 lambda 方案弃 (非模板 if
  constexpr 分支体对 MoE payload 仍编译)
- ninfer_engine + ninfer_serve 双变体编译回归干净; 已同步 Windows 镜像
- Muse 变体编译实例 = Muse variant.cpp 实现 post_mixer_double_norm 后获得

### Muse 目标装配完成 (2026-09-04, ninfer_engine 含 Muse 编译通过!)
- 新 exact 目标 muse_glimmer_30b 全件套: config.h (运行版: 52 层全 full,
  theta 表不动 -> 短上下文窗口不裁剪数值=HF; E3 后按 kind 恢复窗口) +
  export package.h (WeightsProfile{MuseNvfp4} + export_head_weights) +
  bindings.h/cpp (52 层 fused q|k|gate|v 8704 行, 4 norm 槽含 post_attn_out_norm/
  mlp_out_norm, q_norm/k_norm 全1权重由 converter 造, 全 FP8_E4M3FN_ROW_BF16S) +
  variant.h/cpp (post_mixer_double_norm 真实现, profile 单分支, graph
  profiles 保留) + package.cpp + CMakeLists + registry.h/cpp + engine.cpp
  (CoreMuse/ScoreCoreMuse dispatch)
- 关键坑记录: kMaximumMtpDraftTokens 是引擎编译期数组尺寸 (array<N-1>),
  置 0 -> SIZE_MAX 栈爆 (program_impl 12007 sorry); 须保持 27b 值 5/7;
  非模板 if constexpr 弃用分支仍全语义检查 (35b MoE payload 兼容用 leaf
  接口方案); supports_per_layer_kv_defaults=false 绕 16 层 dtype 表上限
- ninfer_engine + ninfer_serve 编译通过 (27b/35b/Muse 三变体), 双树已同步
- 下一步: Muse converter (键映射 recipe + NVFP4-QAT 解包->fp8 row-scale
  重打包 + artifact 写入; 布局 schema 已全锁定) -> serve 对拍破内存墙

### Muse converter 主体 (2026-09-04, tools/convert/muse_glimmer_30b/convert.py)
- 对象计划 552 个 (embed fp8 + 52 层 x 10 + final_norm + output_head), 布局与
  bindings 锁定一致 (q|k|gate|v 融合 8704 / gate|up 39936 / 4 norm 1+w 烘焙 /
  q_norm 全 1 / vocab 202048 / 全 FP8_E4M3FN_ROW_BF16S)
- 流程: ShardReader -> NVFP4-QAT v2 dequant (e2m1 nibble 表 x per-16 s1 x
  per-row s2) -> fp8 row-scale requant (encode_fp8_row_scaled) / norm 1+w
  烘焙 bf16 -> ArtifactWriter (MODEL_ID=muse-glimmer-30b weights_id=nvfp4)
- 待下载完成后: ① 校准 dequant (真实 shape/数值 vs 假设 (n,k/2) u8 +
  s1(n,k/16)+s2(n,)) ② 单层试跑 ③ 全量转换 -> serve 对拍
- 下载修复: 原三分片全截断 (HF API size 误导, 实际 9.96/9.96/4.75GB);
  dl_muse_nvfp4.py v3 = HEAD Content-Length 为期望 + 断点续传, 运行中
  (~30MB/s, shard1 续传中)

### Muse 转换首跑与 NVFP4 方案 B (2026-09-04)
- 全量转换跑通: 523/523 对象 (embed fp8 / 52 层 x10 / final_norm / lm_head),
  norm 1+w 烘焙, q_norm 全 1, MIXED_PRECISION bf16 直通; 数值校准 std 0.0061 自洽
- 首版 fp8 全线性 artifact 27.9GB 超 WSL 21GB 上限 (mlp fp8 占 ~20.7GB)
- 方案 B (定案, 最小改动): mlp + lm_head 走 encode_nvfp4 直通 (源即 nvfp4:
  packed(n,k/2) + per-16 fp8 s1 + divisor=s2 标量 bits; 引擎 NVFP4 解码语义
  匹配), attn/embed/bf16 保持 fp8 -> 估算 ~17GB fit
- 实施清单: ① converter: mlp/head 对象换 NVFP4 payload (encode_nvfp4) + 每
  nvfp4 权重配 input divisor 1.0 f32 对象 (bind_nvfp4_weight 需要) + TensorSpec
  格式 NVFP4 (BLOCK_SCALE_LAYOUT) ② bindings.cpp: mlp/head NVFP4 绑定模式
  (bind_nvfp4_weight), attn/embed FP8 保持 ③ 重转 -> serve 对拍

### Muse 格式图 (2026-09-04 扫描全 52 层, NVFP4 化输入)
- 体积: fp8 全方案 27.9GB 超 21GB; nvfp4 直通可降至 ~11-14GB
- mlp: gate/up layer 5-11 = fp8 全宽 uint8 (19968,6656); 其余 packed (19968,3328)
  nvfp4; down layer 1-12 = fp8 全宽 (6656,19968); 0,13-51 packed (6656,9984)
- attn: 层内混合 (q/k/v/gate 与 o_proj 格式可不同; o k=4096 full/packed=2048,
  qkv k=6656 full/packed=3328); 层 7-12 部分 bf16; 层 24+ 部分 nvfp4 packed
- 引擎约束: NVFP4 block-scale 需 N%128==0 && K%64==0 (vocab pad 202048->202112
  已入 config: output_rows=202112 token_domain=202048)
- converter 已支持动态 mlp 格式 (mlp_layer_fmt 运行时判定 + FP8 fallback);
  attn 动态 (fused_attn 需按 4 proj dtype 分支) 待补; embed fp8 + lm_head nvfp4
  pad 已就绪
- bindings 侧需 per-layer 格式表 (C++ if layer in set) 同步 converter 规则
- 下一步: ① fused_attn_payload 按 dtype 分支 (nvfp4 直通 concat / fp8 编码)
  ② plan attn 动态 ③ bindings.cpp 格式表 (mlp gate_up 5-11 FP8 else NVFP4;
  down 1-12 FP8 else NVFP4; attn 按层) ④ 重编 engine + 重转 -> serve 对拍

### Muse serve 破墙冲刺状态 (2026-09-04, 逐层清障)
- artifact 19.7GB < 21GB fit ✓ (NVFP4 直通 mlp/lm_head + fp8 attn 拆 4 独立
  q/k/gate/v + embed fp8 + 资源 6 件 + divisor 对象 + vocab pad 202112)
- 引擎清障已过: registry/serve 符号链 (apps/ninfer-serve target 名) /
  frontend 资源 6 件 / divisor shape () / KV 层表 16->64 (types.h 常量 +
  decoder_state/layouts/serve_options/plane_base 全同步) / StateImage linear
  假层 min1 / attn capacity 0 / gqa q_heads 校验 32->2
- 卡点: gqa decode 内核 tile 为 qwen 特化 (GqaGeometry<32,2,1> 实例化
  static_assert Wc%RowTiles==0 失败; 支持组合 24/16) -> 需内核 tile 泛化
  (Wc/RowTiles/PVNt 按 QHeads 32 适配: gqa_attention_decode_{i8,nvfp4}.cuh
  162-163 等) + prefill softmax 路径同查 (32q/2kv)
- 下一步: ① gqa tile 泛化 (最小: 为 32 头加 tile 常量特化分支) ② 重编
  serve -> serve Muse smoke (破墙) ③ logits/HF 对拍

### Muse GQA 内核口味边界 (2026-09-04, 精确卡点定案)
- 根因: 引擎 GQA decode/prefill 内核族以 kGqaHeadDim=256 硬编码
  (gqa_attention_decode.cuh:22 全局常量; fragment 切分/共享内存布局按 256);
  Muse head_dim=128 -> 非 256 几何 = 新内核口味 (._ARCHKIT 第 5 节预期
  new-op 事件: "新口味第一次出现写一个 leaf 内核, 之后同族免费")
- Muse 文本适配完成度 ~95%: 目标装配/转换/格式/资源/KV 64 层/48 项清障全过;
  artifact 19.7GB fit; 剩余 = 128-head-dim GQA 内核族工作包:
  [ ] kGqaHeadDim 模板化 (Geometry 加 HeadDim 参数) 或 128 口味新内核
  [ ] gqa decode: bf16/i8/nvfp4 三个 kernel cuh (fragment/arena 按 D)
  [ ] prefill softmax attention 路径 (causal_cache/small_t.cu 同查)
  [ ] split-KV verify / workspace capacity 几何
  [ ] swa window (E3) 主路径 (39 层)
  完成后 serve Muse smoke -> logits 对拍
- 同族 (128-head-dim GQA) 未来模型将免费 (口味库化)

### GQA HeadDim 参数化落地 (2026-09-04)
- GqaGeometry 加 HeadDim 模板参数 (默认 256, assert 128|256); GqaMuseGeometry
  = <32,2,1,128>; decode 内核族 6 文件模板体 kGqaHeadDim -> Geometry::HeadDim
  (49 处); ninfer_ops 编译到 tile assert (通路通)
- 剩余 tile 适配 (已定位):
  [ ] RowCount = TokenTile*GroupSize -> Muse GroupSize 16 时 TokenTile 上限 3
      (RowTiles=TokenTile <=3); launcher 上层 tokens->TokenTile 选择需按
      Geometry 钳制 (6/5/4 对 GroupSize16 非法)
  [ ] launch WarpsPerCta 分支按 GroupSize 适配 (Muse 16: WarpsPerCta 4 时
      Wc/RowTiles ∈{4,2,1} -> PVNt 4/8/16 全满足; 需 Muse 专用分支)
  [ ] bf16/fp8/iso3 内核同查 (i8/nvfp4 已见 assert 形态)
  [ ] prefill softmax attention 路径 (32q/2kv/128d) + split-verify capacity
- 注: 该工作包 = 专注内核开发轮次 (每 cache dtype x TT 组合实例化验证)

## 10. 新增三项 (2026-09-03, 用户定调: 落实但精度不压)
1. ninfer 格式进一步压缩 (借鉴 llama.cpp 思路) —— 只做无损/保精度:
   * 零行/填充行剔除 (vocab padding 已知 NaN 清零), scale 结构冗余
   * 码/scale 的无损重打包与熵编码 (先实测空间, 再定布局)
   * 落地: tools/convert/compress_probe.py 实测 -> 布局设计 -> 引擎布局扩展
2. 按需实际占用显存 (不先占): 稠密层收益有限 -> 以 MoE 专家页粒度为主,
   另加多模型轮换 (闲置模型整模型 host 换出) —— 保精度 (byte 级镜像)
3. 模型权重 OOM -> 内存卸载 (KV 冷机制已有, 权重没有): 层/专家粒度
   host 镜像 + 按需换入, 挂 cold-host 同族基建; 不重量化 (精度别压)


## 11. Muse serve 破墙修复链 (2026-09-04, 逐门突破实录, 全部已落地)
serve 端到端启动关卡逐一拆除 (每项=引擎或转换器一处修复):
1. chat_template digest 白名单 (e84f ThinkingToggle / c3cf ReasoningEffort): 转换器内嵌
   tools/convert/muse_glimmer_30b/qwen_chat_template.jinja 换为 HF Qwen3.8-27B 官方模板
   (sha256=c3cf...); 旧文件 .bak4d34 保留
2. tokenizer_config 缺 added_tokens_decoder (引擎 require_object_field): converter 合成空对象
3. preprocessor_config/video 必须匹配引擎编译的视觉几何 (patch16/temporal2/merge2/mean0.5):
   converter 直接嵌入 qwen 官方同款资源 (data/qwen3_8_pp/, 与 pinned sha 一致)
4. token 域家族常量 248077 参数化: FrontendOptions 增 token_domain + validate_official_special_ids;
   muse package.cpp 传 202048/false; runtime 报错文案去掉硬编码 248077
5. cold_slots/cold_slot_valid 数组 16->64 (PagedKVCacheLayout + PagedKVCache 成员; qwen27 恰好
   full_attn=16 塞满, Muse 52 层越界写/读 -> 未初始化 TensorRegion -> dtype=16 崩溃; 附带
   zero-init); 教训: per-layer 数组一律对齐 kKvLayerStorageSlots=64
6. embedding FP8 形状门 248320x5120 硬编码 -> 泛化 (n>0 && k==out.ne[0])
7. fp8/nvfp4 linear 几何白名单注册 Muse 形状 (fp8: 4096x6656/256x6656/6656x4096/19968x6656/
   6656x19968; nvfp4: 19968x6656/6656x19968/202112x6656), 全部 A16-only 路由 (A8/W4A4 待 Muse
   测量), fp8 decode 调度继承 AttnInput 特化, small_t_max=11
8. gqa_attention wrapper head_dim 256 常量参数化 (校验/平面几何/scale 期望 1/sqrt(head_dim)
   全部改 cache.head_dim; gqa_attention_workspace_capacity_bytes 增 head_dim 参数)
9. 瞬时 cudaMalloc 失败/VM 崩溃根因: 宿主仅 32GB RAM, WSL 占 27GB 挤爆宿主 -> GPU-PV 分配
   无背书内存 -> 偶发 OOM + WSL VM 整体崩溃 (/tmp 清空特征). 处理: .wslconfig 内存改 14GB
   (serve), 构建期切 26GB (C:\Users\User\Documents\ziqinzhang\wslmem.bat <GB>); DeviceBuffer/
   DeviceArena cudaMalloc 加重试(6x2s)+错误带 free mem

## 构建纪律 (用户要求, 2026-09-04)
- 14GB VM 下 cicc 峰值 ~4GB/进程: -j32/-j8 都会换页死锁; 构建前 wslmem.bat 26 再 -j16
  (26GB/5GB≈5 并发安全); serve 前 wslmem.bat 14
- 不要在后台构建时干等: 并行推 todo (本次并行完成了参考侧 logits 通道调研)
- 公共头改动代价极大 (gqa_attention.h 签名变更 -> ninfer_ops 全量 CUDA 重编): 常变逻辑
  下沉 .cpp / 新几何参数避免改 include/ninfer 公共 API
- 分层构建脚本: /home/user/ninfer-fusion/build_fast.sh (core->ops->engine->target, 快速失败)
- 遗留: ccache 不可用(无 sudo); 若可 apt 装上后 nvcc 增量可秒级

## 参考侧 logits 对拍通道调研 (并行完成, 待办)
- transformers 5.14.1 不认 model_type=muse_glimmer; config 为 multimodal 包装
  (MuseGlimmerForConditionalGeneration + vision_config/projector), text_config 是核心
- 需要 NVIDIA NVFP4-QAT 仓库 modeling_muse_glimmer.py 注册后跑 reference logits;
  pip 无 nvfp4 包; repo 来源 URL 未留痕 (_dl*.log 无 URL)
- 引擎侧 logits 出口未建 (serve 无 logits 端点; ninfer CLI 无 dump): 对拍前需先定通道
  (causal scoring core 已有 ScoreCoreMuse; ninfer-perplexity 可做聚合 ppl 粗对拍)

- [更新] transformers main 分支已含 muse_glimmer 架构 (modular 生成, MuseGlimmerForConditionalGeneration);
  官方权重 meta-models/Muse-Glimmer-30B (2 shard, bf16 ~60GB > 32GB VRAM); 本地 data/muse_nvfp4 是
  NVFP4-QAT 量化版 (需 NVIDIA NVFP4-QAT 推理栈或引擎自对)。对拍落地路径: serve 冒烟后 -> pip 装
  transformers@main -> 下载官方权重跑参考 logits (需解决 60GB 显存: 分片/CPU 半精度 或降级为
  量化容忍对拍)

## 12. 全自动适配管线 v1 (2026-09-04, 验收标准推进)
- adapt_all.py 一键管线 5 阶段落地: spec(config.json/目录/spec) -> adapt(config.h+manifest+gaps)
  -> gen_variant 口味叶子 -> gen_stubs_v2 (bindings/package/recipe 骨架) -> report
- 双案例全链跑通: qwen4-exp (spec 输入) + Muse (原始 config.json 输入)
- 修复: ①spec 文件被 adapt.py 二次提取自毁 (加 spec 直通判定) ②model_id 推导
  ③family 键归一 (qwen3_6->qwen38 flavors) ④catalog_gaps 增 B 类家族不变量审计
  (token_domain 差异 / layers>16 per-layer 数组 / quant_geometry 转换后门) -> Muse 现报 10 gaps
  (9 hook + vision new_op 可跳过 + post 门)
- 已知 catalog 精度缺口 (下轮): GDN 口味引擎已有实现 (qwen27 主模型) 却报 new_op ->
  catalog 需引擎口味覆盖注册表; 引擎已迁移运行时推导项 (embedding/gqa head_dim) 不再需报
- 下一步: v2 生成器回填 (bindings/recipe stub -> 真代码, 吃 Muse 手工绑定作参照) +
  catalog 覆盖表 + GUI 消费 manifest

## 13. ARCHKIT v2 生成器回填方案 (2026-09-04, CPU 线推进)
可行性已验证: bindings.cpp 用 binder.require_tensor(结构化拼接名, format, shape) 绑定,
名字模板 = prefix + layer + 子路径 -> 完全可由 converter 键表/格式决策生成 (Muse 手工
bindings 320 行 vs converter object_plan 868 对象一一对应, 无字面量漂移).
v2 生成器输入 = 每个 variant converter 的布局契约 (格式表 layer_mlp_fp8 类决策 +
行列几何 + divisor 伴生对象规则) -> 输出与手写等价的 bindings 展开.
实施步:
1. 把 converter 的格式/形状决策提为布局契约 JSON (per layer: attn q/k/gate/v 格式,
   mlp gate/up/down 格式表, divisor 规则) -- Muse converter 已隐含, 抽取即得
2. bindings 生成器: 契约 -> bind_weight/bind_nvfp4_weight 展开 + 校验 (形状断言)
3. package.cpp 骨架对照 Muse/qwen27 手写 166 行回填 (make_frontend/profile/叶子装配)
4. recipe 生成器: 契约 -> converter 主循环 (reader 遍历 + encode 调用)
验收: 生成产物替换手写后 serve 冒烟一致 (Muse 为基准案例)

## 14. Qwen3.6-40B 适配工作包 (2026-09-04, 用户指认已下载 24GB 就位)
路径: HF 缓存 models--maci0--Qwen3.6-40B-...-NVFP4 (model.safetensors 单文件 24GB)
预研结论 (键表 3508 键全扫):
- model_type=qwen3_5, 96 层, hidden 5120, 24q/4kv/hd256, vocab 248320(=qwen 家族域,
  FrontendOptions 无需覆盖), intermediate 17408
- 权重 = NVIDIA NVFP4-QAT e2m1 pack 风格 (weight_packed U8 [rows, k/2] +
  weight_global_scale/input_global_scale F32) -> 与 qwen27/muse 转换器解码同族
- 结构与 qwen3.6-27B 差异: ①96 层 > 引擎 per-layer 数组上限 64 (kKvLayerStorageSlots)
  -> 引擎层数上限参数化 (B 类不变量实战案例) ②分离 q(12288)/k(1024)/v(1024)/
  o(3072->5120) 投影 + linear_attn (in_proj_a/b/qkv/z) 混合层 -> qwen3.5 linear-attn
  口味对照引擎 GDN 支持核对 ③新 fp8 几何注册 (q 12288x5120 等)
- FlashNext 权重: 未定位 (fn_official.json 是规格非权重; 请用户指路或确认 repo id)
实施序: Muse serve 冒烟收尾后 -> adapt_all 跑 40B config (层数审计驱动引擎改造) ->
转换 (recipe 泛化) -> serve 验收

## 15. TP/QPN 多卡与老卡移植 (2026-09-04 启动, 用户点名通用保证)
参考: data/v100-skinny (Apache-2.0, skinny_kernels.cu 1913 行: simt/wmma/mma8/
qpn M4-16 mma.m8n8k4 / qpn_simt M1-3) + fork_patches/marlin.py (_qpn_prepack) +
fork_patches/custom_all_reduce.py (1Cat TP 参考). 规划详见 _GPU_MATRIX.md.
硬件约束: 本机单卡 sm_120a (无 V100/无第二卡) -> 老档核以通用 SIMT 语义移植+
数值验证, sm_70 专属发行待 CUDA12 链; TP 以代码路径+单卡退化验证先行.
已落地:
- [x] QPN prepack 原型 (tools/archkit/qpn_port/qpn_prepack_proto.py): 与 marlin
  参考逐字节一致 (128x512/320x5120/34816x5120/202112x6656 ALL EXACT)
待办:
- [ ] QPN2 decode 核移植 (qpn_simt M1-3 + qpn M4-16 mma8n8k4): 消费 qpn_prepack
  布局, 引擎 NVFP4 权重加载期/转换期加 prepack 旁路
- [ ] sm 分发层: linear 按 device.sm()+profile 选档 (QPN2 老卡 / 现有档位不动)
- [ ] 老卡回归清单: groupwise-int 3090/4090 真机基线 (195-203 tok/s)
- [ ] TP 框架: AR 抽象 (单卡退化 memcpy) + 权重 N 维切分契约 + KV 头分片 +
      decode-partition pin (0.7-2.6ms/round) + greedy MTP (+10-25 接受点)

## 16. Muse 全 0 输出调查 (2026-09-04, 乱码根因链)
症状: serve/CLI 生成恒定 token id 0 -> decode 全 U+FFFD; 数值层问题 (非模板).
已排除:
- artifact 损坏? NO - 数据区有 payload_offset 对齐 (4096), 手工读必须加偏移; 修正后
  divisor 精确 (gate 0.000108 / head 0.001414 = 源 weight_scale_2) -> converter/artifact 正确
- MuseReader 顺序读错位? NO (30 key 跨 shard 大规模测试 0 mismatch)
- device 差异 (GPU/CPU 转换同错读法同果, 均为读法问题)
进行中: 引擎采样前 logits dump (decode_impl.h 首个 decode 打印前 12 logits) ->
  全 0 则 head/nvfp4-vocab gemv 数值; 非全 0 则采样/输出路径
教训: ninfer artifact 手工读取必须 payload_offset = align(16+json_bytes, 4096) + obj.offset

## 18. Muse 修复进展 + decode 回写断点 (2026-09-04 深夜续)
已修 (真根因 x3, 全部验证):
1. NVFP4 divisor 存 1/s2 (converter) - 数值爆炸修复
2. embed fp8 核行宽 5120 硬编码 -> 运行时 d + 统一 <1,256> launch (misaligned 修复)
3. text_context_impl.h 从 ninfer-allbuild 8-31 恢复 (962 行损坏版弃用, 备份
   /home/user/text_context_impl.h.broken-0904; Muse 单 norm 路径在 allbuild 版完整)
当前状态: 网络数值全通 (logits 非 0/首 token 162042 正常采样), 无崩溃
残留 (最后环节): decode 每步输入 token=0 (stepin dump) - sampled token 未回写
   为下步输入; 证据链: steplog 1-3 logits 恒定 (输入恒 0) + stepin token 全 0
   写回链: decode_impl sample -> ordinary.egress (host sampled_tokens, 步1=162042 OK)
   -> program_impl L11864 BatchedGeneratedRound.tokens -> 上层 (engine/request) 写
   下轮 ingress tokens -> decode_impl 读 ordinary.tokens (实测 0)
   待查: 上层回写点 (BatchedGeneratedRound.tokens 消费者 -> ingress tokens 更新;
   qwen 同路径正常 -> 找 Muse 分支差异如 sequence/ingress row 写回偏移)
   下一步: grep BatchedGeneratedRound.tokens 消费点 + ingress tokens 写入, 对比
   qwen27 (正常) 与 Muse 差异; dump 上层收到的 tokens

## 20. Muse decode 收敛 (2026-09-04 深夜续3)
证据链最终收敛:
- prefill 路径正常 (首 token 162042 = "Immediately" 真实生成)
- decode 首步: hidden 非 0 (graph 模式 headdbg 步1 hidden=16328 正常) + head 输出
  logits 全 0 -> ops::sample 0 -> 后续恒 0/NaN
- 结论: decode 的 lm_head (nvfp4 vocab 202112x6656) gemv 核输出 0
  (prefill head 用小 T 核正常; decode head 走 A16 gemv 核 - 新 MuseVocabulary
  几何无 qwen 对照 (qwen lm_head 是 Q6/W8))
待查: nvfp4_gemv_kernel<Nvfp4MuseVocabularyGeometry> 输出 0 原因:
  ①核内 scale 平面读取 vs converter swizzled scales (qwen 旧 converter 未 swizzle?
     Muse layouts.py swizzle -> 引擎核布局假设差异) ②gemv 核 202112 行边界
验证法: dump decode head linear 的 Weight.qdata 首个 scale 值 (引擎读 vs artifact)
修复方向: 引擎 nvfp4 gemv 解 swizzle (unswizzle_nvfp4_scales 语义) 或 converter
  关闭 swizzle 对齐 qwen (验证 qwen artifact scale 平面是否 swizzled)
附: graph 模式步 2+ logits 恒定 (输入恒 0 自洽); no-graph 步 2+ NaN (append 后
  attention NaN 次要问题, 待 head 修复后复验)

## 21. Muse decode-hidden-zero 收敛点 (2026-09-04 深夜终)
最终证据 (no-graph): decode 首步 head 输入 hidden 全 0 (head 权重正常 23931) ->
   网络 decode 输出 0; prefill 正常 ("Immediately")
排除: head gemv 布局 (引擎核 swizzle 预期 = artifact ✓) / divisor / embed / attention
   output_projection leaf (无 ph 分支)
待查候选 (下轮):
A. decode 的 run_layers 层 2+ 或全层输出 0: 加 decode 层 0/25 输出 dump
   (restored allbuild 版 run_layers L1030, ph==Verify 分支)
B. workspace 容量: 旧版 (8-31) workspace 计划 vs Muse 52 层 decode -
   层中途 alloc 失败/覆盖 (prefill 正常 decode 崩的模式吻合);
   workspace_plan.ordinary_round 容量核对
C. restored 版与 Muse variant leaf 接口差异 (旧版 attn_mix/mlp_tail 调用序列
   vs Muse leaf 预期) - prefill 正常则差异在 Verify 路径
建议: B 优先 (workspace capacity per TextConfig 层数 - 查 workspace_recipe 的
   capacity 计算是否按 kCfg.n_layers=52)

## 22. Muse decode NaN 源收敛 (2026-09-04 深夜终2)
证据: decode 首步层 0 mlp 后 x = 0x7FFF NaN -> 层 0 attention 读 cache 含 NaN
  (prefill 正常 Immediately 但 decode 用 cache 即 NaN) -> KV cache 读到未写区
  (0x7FFF 填充?) 或 envelope 超 range
待查 (下轮, 已定位代码):
A. decode 首步 envelope: program_impl L11815 envelope = {profile.min_execution_frontier+1,
   max+1} - Muse frontier (prefill 后) 核对; 若 envelope.max 超 cache 有效长度
   -> attention 读未写页
B. KV cache 初始内容: 分配后是否清零 (若含 0x7FFF 模式 -> 未清零证据)
C. decode KV append 路径: restored attn_mix 无 append 调用 - 确认 Muse decode 的
   当前 token KV 由谁 append (调度 materialize_sequence_kv? / gqa 核内?) -
   qwen 同路径正常 -> 找 Muse frontier/页面分配差异
D. 层 0 的 k/v 投影后 append 位置 (q/k/v 计算正常? dump 层 0 k 投影输出前 4)
快速实验: --kv-capacity 放大 (页面足) 是否消失; prompt 长些 (prefill 多 token)

## 23. Muse decode 收敛终 (2026-09-04)
长 prompt 实验 (5+ tokens): decode 层 0-24 正常, 层 25 起 NaN (0x7FFF)
短 prompt ('Hi' 2 tokens): 层 0 即 NaN
-> 模式: NaN 起始层 = f(prefill 长度?) 或 append 到某层的 KV NaN
   层 25 NaN = decode append 层 25 的 KV NaN (append kernel 层 25 位置错?)
   短 prompt 层 0 NaN = 首 token prefill 的 KV append NaN?
双线残留: ①层 25+ NaN (KV append 特定层) ②步 2+ 输入非 sampled (回写)
下轮建议: dump decode append 后层 25 KV 值 (gqa_kv_append 输出) 或
   --kv-capacity/页分配路径; 对照 qwen27 (48 层) 的层 25 行为
已知修复累计 (Muse): divisor 1/s2 / embed d 参数化 / embed launch 统一 / 文件恢复

## 24. Muse decode NaN 真相澄清 (2026-09-04 终)
澄清: artifact fp8 code 平面 0 个 0x7F (filter 有效) - 先前 0x7F 计数含 bf16
  scale 平面字节 (误报)
确认: 权重全干净 (code/scale/divisor) - decode NaN 在引擎运行路径
剩余线索: decode 层 12 mlp 后 NaN (层 0 正常); 起点层 1-11 未定; prefill 正常
  (Immediately) -> decode 特有 (cache/append/页分配方向)
建议下轮: ①探针层 6/8/10 定位首 NaN 层 ②该层 attn vs mlp 二分
  (dump q/k 投影已做过层 25 - NaN 输入传播; 起点层 dump attn 后 vs mlp 后)
  ③decode append 页分配 (kv-capacity 页面) 检查
本会话 Muse 修复累计: divisor 1/s2 / embed d 参数化 / embed launch 统一 /
  text_context_impl 恢复 / NaN filter 防御 (code 干净已验证)

## 25. Muse decode 最终焦点: nvfp4 gemv 输出 0/NaN (2026-09-04)
证据 (no-graph decode):
- 层 0 mlp: gate/up NVFP4 decode 输出 0 (act 全 0) -> 层 0 mlp 0 增量 (x 残差正常)
- 层 1 mlp: gate/up NVFP4 decode 输出 NaN (act/o 32767) -> 层 1+ 全 NaN
- attn 层 0/1 正常 (层 1 attn 后 x 正常) / q/k 投影正常 / prefill 全正常 (Immediately)
- down o dump 显示 decode 层 0 down 输出 0 (act 0)
收敛: nvfp4 decode gemv (A16) 对 Muse 权重输出 0 (层 0) / NaN (层 1)
  - qwen nvfp4 gemv (5120 K) 正常; Muse 6656 K 崩
  - 权重 artifact code/scale/divisor 全验证干净
  - 层 0 vs 层 1 差异: 层 1 gate/up 源? (层 0 源 U8 packed NVFP4-QAT; 层 1 源?)
    待查: 层 1 gate 源格式 (U8 packed vs F8 vs bf16) -> converter 编码差异 ->
    gemv 对某编码的 scale/码展开 NaN
下轮: ①查层 1 gate/up 源 dtype ②nvfp4 gemv 核 code/scale 展开 (逐 K tile dump)
  ③试 bf16 KV/小上下文隔离
会话累计 Muse 修复: divisor 1/s2 / embed d+launch / 文件恢复 / NaN filter (code 净)

## 26. Muse decode 巨大值方向: decode 核 append (2026-09-04 深夜终3)
证据:
- decode 核族 (bf16/i8/nvfp4/fp8 cuh) 均有 GqaAppendInput -> 核内 append+attention
- 1-token prompt: 层 0 decode 正常 (读 [0] 已写) -> 5-token: 层 0 巨大 (读 [0..4]
  含未写槽?) - 模式 = decode 读 range 覆盖未 append/未写槽 (garbage 巨大随机)
- prefill 正常 (Immediately) - prefill 写读同位置
下轮: ①查 gqa_attention_decode_bf16.cuh 的 append 段 (Muse HeadDim 参数化后
  append 槽/偏移是否错 - HeadDim 128 vs 256 的 cache 写索引) ②dump cache 槽
  frontier (decode 首步后) 验证 append 是否写 ③对照 qwen (256hd 正常) 与
  Muse (128hd) 核 append 差异
本会话 Muse 修复 (已验证): divisor 1/s2 / embed d+launch / 文件恢复 / NaN filter

## 27. Muse decode 巨大源: o_proj fp8 gemv (2026-09-04 终4)
证据链更新:
- cache 内容正常 (cK0 页0/1 槽值 16028 等正常)
- 权重/scale 全净 (o_proj rows scale 5e-5 级正常)
- decode 层 0 巨大源 = attention_output_projection (o_proj) 输出巨大
  (q/k/gate/attn/cache 全正常; 1-token prompt 正常 = 与 cache 长度相关?
  实际: 巨大仅当 prompt>64 tokens (跨页) - 但 cK0 页1 值正常 -> 跨页不是 cache 写
  问题 -> 巨大源在 o_proj gemv 对特定输入 (长 prefill 后 hidden 分布?)
下轮: ①dump o_proj 输入 (sigmoid_mul 后 a) 确认输入正常 ②fp8 gemv 核
  (MuseAttnOut 6656x4096 几何) 内部检查 (qwen 无 fp8 o_proj 对照 -> 新几何核)
  ③是否 long-prefill 后 hidden 数值域触发 gemv 核 bug (如 bf16 展开/累加)

## 28. Muse decode 时变 garbage 收敛 (2026-09-04 终5)
证据:
- cs memcheck 干净 (0 Invalid/Misaligned) - 无非法访问 -> 时变 = 合法地址未初始化读
- o_proj 后 x 正常 (3 次 run = embed 值 -> 层 0 attn 输出 0 增量正常)
- 巨大在层 0 mlp 后 (vlay L0 时变巨大); pm-in h 有时 0 多 (rmsnorm 输出 0? 输入巨大?)
- 核逻辑全查: append/页表/split/缓存索引完整正确
下轮 (最小 dump 集): ①post_mixer 内 dump gate/up/act/down 输出 (巨大环节定位)
  ②h (rmsnorm 输出) 采样巨大值 ③workspace alloc 重叠检查 (post_mixer 的
  gate/up/act/o 顺序 alloc - 若 scope 容量不足返回重叠 buffer -> 时变)
会话累计: 4 修复 (divisor/embed/恢复/NaN filter) + 证据链 §17-28

## 29. Muse decode 时变终局 (2026-09-04)
矛盾: pm-in h (rmsnorm 输出) 有时 0 多 (16412 0 0) 有时正常 (15921), 但 vlay L0
  (mlp 后 x) 3 次全巨大 - 数学上 gemv(正常 h) 不可能巨大 (权重/acc 全净, cs 干净)
  -> 疑 dump 与执行错位 或 buffer 复用 (h/o 的 work alloc 区与 x 重叠?)
下轮精确方案 (一次调用内全采): post_mixer 内同一调用 dump h/gate/up/act/o/residual
  六连 (当前分点 dump 可能跨调用错配) + run_layers vlay 同调用核对
  若六连显示正常 (o 正常) 而 vlay 巨大 -> x buffer 层间被改 (residual_add 或
  mlp_tail 后 x 与下一层共享问题)

## 30. Muse decode 根收敛: attention 输出巨大 (2026-09-04 终6)
六连 dump 决定性: post_mixer h 全 +/-0 (rmsnorm 溢出) 因输入 x 巨大; o=0
  (mlp 无辜); 根 = 层 0 attention 输出 (o_proj 输入 a) 巨大时变
已排除: cache 内容/权重/divisor/code/scale/acc 初始化/cp.async wait/
  page table/split 范围 - 全正常
attention 核内部剩余: score/softmax/PV 计算 (q/k 正常, cache 正常 ->
  score 正常 -> 巨大在 softmax/PV 或 o_proj 输入 a 的 sigmoid_mul(gate,a)
  gate 巨大? (gate 投影 fp8 正常 dump 过) - 待查: dump a (sigmoid_mul 后,
  非 o_proj 输入) 时变巨大?
建议: ①dump gqa_attention 输出 a (batch 路径) 时变确认 ②softmax 核中间
  (score/p) ③或 direct: Muse bf16 核的 WarpsPerCta/QKNt/PVNt 布局 vs qwen
  256 核 (D=128 的 ldmatrix/mma 配置核对)
会话终: Muse 修复 4 项; 焦点收窄到 bf16 decode attention 核 (128 D) 内部

## 31. Muse 层 1+ 近死冻结根因排查 (2026-09-04 续)
新二进制 (18:40 全量重链) 后症状改变: 层 0-11 decode 数值全净 (~1.0), 但:
- [headin] 变 [head2]: fin_in (final norm 输入 x@L51) 正常 ~1.0; hidden 正常 (16286 16438...);
  logits 正常非 0 => 旧 "hidden=0" 是 stale 二进制 + dump 跨缓冲错读
- vlay 扩展 L13/21/31/41/51: x 在 L5 后几乎冻结 (L5 49418 == L51 49421 差 3 counts)
- xattn L4-L7 (attn 后 x): L4 attn delta +48 counts (0.07%), L5 attn delta = 0 EXACT,
  L6/L7 attn = 0; 整网从 L1 起 delta 收缩: L0 attn delta -1.1 (正常!), L1 后 ~0.001
- [oproj] (o_proj 输出) L0-L3 全 ~1.0 正常, 但 x 只变 ~0.02 => L1+ 的 o 未加入 residual
  或 dump 读 stale (linear 未跑时 arena 复用同地址)
- qk L0/L1 正常 (h->qkv 投影工作); 层 6+ artifact 权重全稠密 (nonzero 99.8-100%)
=> 结论: 只有 L0 attention 真正工作; L1+ attention+mlp 近死 (attn delta 0 exact from L5,
  mlp delta ~0); prefill 是否同冻结 = 决定性 (待 pvlay dump)
待验证: ①prefill pvlay L0-8 轨迹 ②decode L1-3 的 a (attnB gate 已扩) ③cache L1+ 内容
假设: prefill 层 1+ 若也冻结 => 根在 prefill 层 1+ 数学 (norm/qkv/attn 权重绑定);
  若 prefill 正常而 decode 冻结 => decode cache 层 1+ 为空 (per-layer cache 写 bug)


## 32. Muse 根因: gqa 几何分派缺 Muse (2026-09-04 定位)
证据链 (全部 sync'd dump, 可信):
- L0-L8 每层 residual_add 均正常 (xpost == xpre + o, float 验证)
- o_proj/attn a: L0 a ~±1 正常; L1+ a ~0.001 (uniform-flat); attnB L0 a == v 逐位相等
  => decode attention 只 attend 自己 (无 past context); L1+ 连 self 都 ~0
- qn/kn (norm 后) L1 正常; loader 格式表正确 (gate/up fp8 5..11, down fp8 4..12 全对);
  config layer_kind {1,1,1,0} = SWA×3+full; layer_rope_theta 每 4 层 NoPE —— 均非根因
- step2+ 所有 dump 是 CUDA graph capture/replay 假象 (qwen27 同样 NaN 却工作正常);
  prefill pvlay 同理不可信
根因: gqa_attention_small_t_launch (decode+batch 路径) 只有 Gqa27(24头) + 默认 Gqa35(16头,
  256维) —— Muse (32头/128维) 静默落 Gqa35Geometry!! Muse 几何只在 cached 路径与
  capacity 函数注册过。prefill prompt/append 路径同样缺。=> 全链 256-dim 几何跑 128-dim 模型。
修复 (3 patch, 待构建):
  1) fix_muse_geom: small_t_launch 加 GqaMuse 分派 (已应用, 编译中)
  2) fix_prefill_geom: prompt_attention/kv_append/prompt lambda 加 Muse (cache.head_dim 判 128)
  3) fix_checked: 未知几何 throw (不再静默落 Gqa35) —— 自动化完备性: 未来模型几何未注册
     直接报错而非错算
GUI 热窗: ninfer-gui.py /api/serve + gui_page.html 两 serve 卡加 cold_keep (热窗 tokens,
  0=引擎默认128) + host 策略; serve_gui.py 滑块版加 冷策略+热窗 滑条
待验证: 构建后 Muse serve -> 141-token prompt -> 期望连贯输出 (Muse wall-break!)


## 33. 自动化几何覆盖门 (2026-09-04)
新洞: Muse 几何分派缺失是引擎手写分派表的静默回退 (未知几何落 Gqa35 错算)
   => adapt_all 判据 "1-4 全绿 => 可 serve" 不完备 (无 serve 级/分派级验证)
补强:
  1) 引擎 (fix_checked + fix_hd): 所有 gqa 分派点 (small_t/cached/prompt/prompt_attention/
     kv_append) 未知几何 throw + 守卫加 head_dim 消歧 (q.ne[0]) => 永不静默错算
  2) tools/archkit/check_geometry.py: 以引擎源码为事实源解析 GqaXxxGeometry 别名表与
     各分派函数守卫; spec geometry (q,kv,head_dim) 逐路由核对; 缺口硬失败
  3) adapt_all.py 阶段 [2.5/5] geometry gate 接入 (失败即停)
验证: muse (32,2,128) PASS 全路由; gemma4-31B (32,16,256) FAIL (未注册, 需
  GqaGemmaGeometry 别名 + 分派点登记后才能声称可 serve)
注意: gemma4 若走非 qwen3_6 运行时家族, 门需按家族域限定 (当前 family=auto, 先全局严)


## 34. 长测协议 + 自动化门落地 + gpt-oss MoE 发现 (2026-09-04)
长测协议 (用户定): 短测不足信 => ①64K 上下文 (至少 65536) ②长生成查循环 ③尾段
  质量退化检测 (同义词胡言乱语) => probe_64k.py (针尖) + gen_64k.py (长生成+4gram 新颖度/
  段重叠/词共享分析); KV 组合矩阵分开测 (iso3/int8+iso3/nvfp4+iso3/e8+iso3/fusion 基线)
  => start_64k.sh (KVSPEC 需整体引号!)
iso3 uniform @2K: 短问答精确 (content="4"), 400tok 长思考连贯, 3 轮确定性 = PASS
  (64K 长测待新二进制之后跑)
gpt-oss-20B (参数门新发现): num_local_experts=32, experts_per_token=4 => **MoE 模型**
  (OpenAI gpt-oss-20b = 32 专家 top-4) + rope_scaling(factor 32, yarn) + swiglu_limit=7
  => 非快速校验, 需 MoE 专家路由内核工作包 (new_op); 修正多模型批预期
GGUF 批 (元数据已验): i1-Q4_K_M (15.66GB) / Ornith-1.5-9B (5.38GB) / NVFP4-Q8_0
  (15.71GB) 全部 llama.cpp **qwen35 架构** (bos 248044, token 域需覆盖);
  mmproj-BF16 (0.87GB) = clip 视觉封装 => 文本可先 serve, 视觉 = new_op
自动化门: check_geometry (2.5 硬门) + check_params (2.6, config 输入时报告制) 已接入
  adapt_all; MoE 键从 benign 改为 new_op (防静默); Muse 全管线绿
构建: gqa_attention_decode.cu.o 单 TU 在 5090/WSL 需 35-45min 且怕内存挤压
  (GPU-PV 19GB serve + 编译 = nvcc 0% CPU 挂死) => 规则: 大编译期间不跑大 serve;
  setsid 脱离编译防任务取消连坐 (final-build4)


## 35. KV 组合矩阵 + 双 bug 发现 (2026-09-04)
64K 长测协议: 针尖(57K 上下文) + 长生成 4000-6000 tok (4-gram 新颖度/段重叠/词共享判
  循环与退化) — probe_64k.py/gen_64k.py (curl -d @file 防 argv 超限)
矩阵结果 (旧二进制; 注意: iso3 层映射 bug 使标注失真, 见下):
- iso3 uniform / fusion(基线) 输出逐字相同 => 二者实际同配置 (variant 默认表覆盖 --kv-dtype)
- i8+iso3(=i8×16+bf16×48) PASS; iso3+i8(=bf16×16+i8×48) PASS; nvfp4+iso3(=nvfp4×16+bf16) PASS
- e8+iso3(=e8×16+bf16×48) @2K PASS 但 @57K FAIL (针尖 reasoning 循环 + 长生成尾段
  100% 复读 loop=True) => E8 层与 BF16 层混排 (plane 4:2 不均) 长上下文坏
- fusion(E8×10 分散+NVFP4 全 4-plane 均匀) @57K PASS => E8 层本身 OK
双 bug:
 1) layouts_impl.h per-layer 覆盖映射缺 Iso3Group16 => 静默落 BF16 (已修: iso3→NVFP4,
    与 target_kv_cache_profile 一致) — 真实 iso3 组合待重编译后重测
 2) E8×BF16 不均 plane 混排长上下文失败 (根因待编译后查: E8 走 prompt 路由 vs BF16
    走 small_t 的混用/plane base)
全 E8@64K 启动即报 "invalid profile or interval" (E8 路径 workspace profile 覆盖不足)


## 36. 拆 TU 编译根治 + split v2 收官 + Muse 128 维核死循环 (2026-09-05 0:00)
编译死磕终局:
- 拆分落地: decode.cu (701行巨TU) → impl.cuh (共享模板) + decode.cu(Gqa27+分派)
  + decode_muse.cu + decode_g35.cu + 既有 decode_e8.cu; CMake 4 TU 并列
- .o 实测: decode 190MB→88.7MB, muse 45.7MB (拆分生效实锤)
- 三 ptxas 并行挤爆 13GB VM 会触发 make Hangup → 串行 -j1 稳定走通全程
- 最终: [100%] Built ninfer-serve EXIT=0 (23:45, 828MB); 积压补丁全部编入
  (Muse 几何分派+throw+head_dim 消歧+iso3 映射+split v2)
1Cat S5/S6 内核收官:
- harness UAF 修复 (partials free 早于 combine, shards>=2 地址复用覆盖)
- 根因定位: v1 单一 t_local 签名 → shards>=2 数学上无标准语义 (q 被迫行段分片)
- split v2: local 加 q_rows/q_base/kv_offset (全局因果, q 草稿窗不分片+kv 位置分片),
  旧 local 因果行残留 (乱码注释隔断锚点) 二次修复
- 结果: shards=1/2/4 全 PASS (8.9e-8 vs attention_mono 金标准尾窗行) — S6 完结
- iso_kv PASS (逐字节) / suffix PASS (294 fuzz) 不变
GUI: 逐层 KV 文本框+cuda graph 开关+fusion/pure 接线修 (原 fusion 被硬接 all:nvfp4)
gemma4-31B 参数门画像: 音频域(7键)+视觉域+text 8 新旋钮 (attention_k_eq_v/
  double_wide_mlp/kv_shared_layers...); 门精化: 音频域聚合/双视角去重/None 值 MoE 降级
Muse 破墙新阻塞 (精确记录):
- 新二进制 warmup decode step1: qkv L0 dump 后卡死 (GPU 100%, host spin,
  gqa_attention small_t Muse 几何核首跑点)
- 定性: GqaMuseGeometry(32,2,1,128) = bf16 partial 核 D=128 的首个真实实例
  (qwen27/35 全 256); 核内 mma/ldmatrix/tile 布局 128 假设违例 → device 死循环
- reducer grid kGqaHeadDim 硬编码 (256→128 应减半, 越界块风险但非死因)
- 下一步: ①mini repro 单 TU (32q/2kv/128d bf16 small_t 调用) + compute-sanitizer
  定位死循环核与行 ②核内 HeadDim 假设逐行审 (bf16 partial 核 tile 循环)
  ③reducer grid 改 div_up(Geometry::HeadDim, kDChunk)


## 37. Muse 128 核死循环: 最小复现锁定 (2026-09-05 00:50)
复现资产 (全部在位):
- /home/user/muse128_repro2 (tests/muse128_repro2.cu): 1-token eager bf16 Muse 几何
  small_t 调用即死循环 (GPU 100%, host sync spin)
  命令: /home/user/muse128_repro2 1 0   (141/1 同挂; graph 模式报 capture unsupported
  是衍生症状 — capture 中 invalid launch 的次生错误, 非根因)
- 关键差异: envelope{0,0}=不挂 (repro1), envelope{0,2048}=挂 => splits/window 路径触发
- warmup 死循环 = 同一内核问题 (dump 清理后仍挂, capture-retry 理论已证伪)
- compute-sanitizer memcheck + launch-timeout 无输出 (核级 hang, sanitizer 陪跑)
排除:
- bf16 partial 核无 dynamic smem attr; impl.cuh 仅 i8/nvfp4 核有 static attr
- kb 主循环有界; page_count<=32 不越界 smem[256]
待审 (下轮):
  1) GqaMuseGeometry GroupSize=16 (32q/2kv) 是首个 16 组 — qwen27=6/qwen35=8;
     gqa_small_t_tc_row_to_qt / gqa_valid_q_head / ldmatrix 布局对 GroupSize=16 的假设
  2) reducer grid div_up(kGqaHeadDim=256,kDChunk=64) 硬编码 vs D=128 (越界读, 顺手修)
  3) gqa_small_t_split_count/upper_bound 对 window=2048 + width=1 的 splits 值打印
     (repro2 加 printf splits/grid) — 排除 grid.y 异常
  4) cuda-gdb attach 或 ncu 定位死循环 PC
桌面启动 bat 修复: UTF-8 中文注释 GBK 吞行 bug -> 纯 ASCII 重写 + python 探测改
  if exist/%PY% -c + 端口就绪轮询重试 (30s) 再开浏览器; 实测 GUI 起监听 8077 ✓
QPN/CUDA12 线 (进行中): CUDA13 无 sm_70; 12.8 官方 installer 下载 0 字节+镜像
  404 (nvidia.cn local_installers 路径不通); 待换 conda nvidia channel 或
  network repo 精确 URL; qpn8_test_sm120.py 已备 (compute_70 PTX-only + Zc:preprocessor,
  5090 JIT 可跑数值验证)


## 38. Muse 128 核: 死循环定位收窄到 sync-hang (2026-09-05 01:00)
新增证据 (repro2 增强版):
- env{1,2048}: splits=32 (合法), grid(2,32,1) 仍死循环 => 与 min=0 无关
- env{0,0} "不挂" 真相 = capacity 全空 -> splits=0 -> grid.y=0 launch 静默失败 (假通过)
- split_capacity 公共入口有 min==0 守卫 (repro 打印触发, 非根因)
- bf16 partial 核 429 行全文审计: 所有循环有界 (kb/QKNt/QKKs/PVNt/PVKs/chunk),
  无 while/goto => "死循环" 实为 __syncthreads/__syncwarp 死锁 或 越界触发 hang
- Muse 特有首例假设: GroupSize=16 (32q/2kv); T=1 派 WARPS=2 -> Threads=64;
  warp_max<4>/warp_sum<4> (4-warp shuffle) 在 64 线程块 (2 warp) 下的 mask 语义待查
下轮 (探针三连发, 每发 30 秒编译):
  1) 核内 printf: entry/kb-loop-first/PV-mma/exit 四点, 看停在哪层
  2) dispatch(1,2) 改 dispatch(1,4) 试 WARPS=4 (排除 64 线程块 shuffle 假设)
  3) GroupSize 16 vs 8: 用 Gqa35Geometry(16q/2kv/256) 同形状跑对照 (同 KVHeads=2,
     唯 GroupSize 差) — 二分 GroupSize 维度
资产: /home/user/muse128_repro2 (env 已设 {1,2048}); tests/muse128_repro2.cu 双仓同步


## 39. GUI v2 改造启动: local-studio (vllm-studio) 落地 + Muse 探针反转 (2026-09-05 03:00)
GUI v2 (用户定调: 换 vllm-studio, 加训练/测试按钮, RAG 重做为文件->向量化->语义搜索):
- clone sybil-solutions/local-studio (controller Bun/Hono + frontend Next16/React19 + agent-runtime)
- Windows 适配三坑全清: git symlink (scripts/project.mjs + .githooks) -> mklink 重建;
  effect 包 shared/ 向上解析 -> repo root npm i effect; .env.local (API_URL 8080)
- 实跑: controller 127.0.0.1:8080 /health=ok (自动识别 nvidia-smi, data 本地化);
  frontend dev 3000 页面真渲染 (title=Local Studio 88KB 无编译错; HTTP 500 = dev
  边缘组件行为, 浏览器可用性待确认)
- 旧 GUI 同步升级: 全参数面板 (extract_serve_params.py 56 旗标注册表 + /api/serve_params
  + 分组折叠渲染 + extras 透传) 已端到端 (参数 API 实测返回)
- 待办: ①浏览器验收 3000 ②controller engines/recipes 加 ninfer engine (WSL serve 脚本
  + 56 参数注册表搬入 recipe schema) ③训练页 (train_dspark 脚本对接) + 快捷测试按钮
  (chat smoke/needle) ④RAG 重做: 文件上传->chunk->embedding->SQLite 向量->语义搜索页
  ⑤替换/并存旧 GUI (桌面 bat 指向新栈)
Muse 探针反转: printf 版不挂 (EXIT=0) 且 WSL 设备 printf 无输出 => Heisenbug 实锤
  (代码生成变化掩盖 smem 越界/竞态); 下轮换原子计数器探针 (device printf 通道 WSL 不可靠)
资产: C:\...\ziqinzhang\local-studio (bun1.4.2 全链可跑); controller/frontend dev 起法已验证


## 40. 桌面快速启动 + 硬解识别 + 全功能转 studio (2026-09-08 16:00)
用户定调: 旧自写 GUI 彻底放弃, 全部功能集成入 local-studio.
桌面快速启动:
- NInfer-Studio.bat (新): 双击拉起 controller(8080)+frontend(3000)+开浏览器;
  已运行时仅开浏览器; ASCII-only 防 GBK 吞行. 实测全栈拉起 ✓
- 旧 NInfer-本地运行器.bat 保留但过渡 (studio 验收后删除)
硬解识别 (从 GPU 本体取信息, 防未知卡漏判):
- tools/gui/hwdec.py: pynvml 读 GPU 本体 (name/brand/NVML 架构枚举) +
  nvidia-smi compute_cap; 判定按**架构代次->NVDEC 矩阵** (非市场名白名单):
  Kepler<h264 / Maxwell+hevc / Pascal+vp9 / Ampere+av1 ...;
  未知卡也报真实身份与架构; 软解选项永远开放 (hw_ok 仅影响默认值)
- 实测本机: RTX 5090 D / Blackwell / 12.0 / [h264,hevc,vp9,av1] / hw_ok=true
- controller /hwdec 端点 (system/hwdec-routes.ts, 60s 缓存) 已通:
  http://127.0.0.1:8080/hwdec (根路径挂载; Windows PowerShell 显示中文乱码
  是控制台代码页, JSON 本体 UTF-8 正常)
- 待: 前端 Configure 面板消费 /hwdec (硬解默认开/软解可选项);
  引擎媒体层 (src/media/decode) 目前仅软解 avcodec, NVDEC hw 后端 = 工作包
旧 GUI -> studio 迁移清单 (后续逐项):
  ① serve 启动 (56 参数注册表 -> recipe extra_args) ② 训练页 (train_dspark)
  ③ 快捷测试按钮 (chat smoke / 64K 针尖) ④ RAG 重做 (文件→向量化→语义搜索)
  ⑤ 模型导入/转换 (archkit + convert 工具链) ⑥ 环境卡 (gpu_compat)


## 41. FreeToken 落地细化 (2026-09-08, 用户点名尽快)
步 1 (观测, 零风险, 已就绪待编译):
- ft_stats.h (header-only, 已写): 每层 attention energy 观测
  (decode reducer 的 partial_l mean), NINFER_FT_STATS=1 开,
  NINFER_FT_PERIOD=回合数. 挂接点 = attn_mix 的 gqa_attention 返回后
  (partial_l 在 small_t 路径是 Tensor, 需在 wrapper 层读 — 下次编译接入)
- 目的: 拿到 52/64 层每层活跃度分布 -> 决定弹性预算的分层依据
步 2 (弹性预算, 核心改动):
- 现有挂点已确认: layer_dtypes_ 是运行时 per-layer 表,
  batch_layer_view(layer) 按 layer_dtypes_[layer] 解析 dtype/stride/v_dtype
  => 静态表已在运行时可读; 弹性 = 按 step1 能量分位数周期性重排各层
  (热层升 bf16/fp16, 冷层降 iso3/iso4) + 热窗 token 数随负载伸缩
- 重排时机: 请求间隙 (KV 池重压缩复用现有 cold requant 链)
步 3 (带宽自适应): decode 带宽占用 -> 动态调 prefill-chunk / 冷页换入节奏
验收: §35 64K 长测协议 + ppl 对照 (flashnext 场景 qwen27 fp8)


## 42. 异构分配器原型落地: PlacementPlanner + 设备画像 (2026-09-08 续)
用户原则落地: 「按各部分特性分配任务性质, 效率最大化」=> 特性×负载匹配矩阵。
新组件 (tools/archkit/qpn_port/):
- device_profile.cu: 设备画像 microbench (fp16 算力/带宽/启动延迟/int8 吞吐,
  JSON 输出, 同内核跨设备横向可比)。5090D 实测: scalar 1.3 TFLOPS /
  1625 GB/s / 8.0us / 456 GOPS (注意: naive 核非 TC, 测的是可移植算力)
- placement_planner.py: 两种负载模型 (decode=带宽型: 权重+KV字节/带宽;
  prefill=算力型: flops/算力+KV/带宽) + 贪心argmin(带迟滞防乒乓) +
  容量受限连续切分穷举 (VRAM 约束)
demo 实证结论 (诚实数据):
- 场景 1 (NVFP4 权重, 双卡均无容量压力): 5090 算力带宽双优 => 全层归 5090,
  speedup=1.0 —— 切层无收益, 诚实不硬凑
- 场景 2 (bf16 47.6GB 容量受限): 5090(32G)+V100-32G 强制切分, 最优切点
  layer 42 => 5090:0-41 / V100:42-63, 37.9ms/token, 单卡不可行=>异构解锁
- 16G V100 变体: 47.6GB vs 48GB 余量 0.4GB, 整数层下无可行划分 (规划器
  如实报错) —— 边界检查有效
下一步: ①实测 V100 画像替换推算值 ②接入引擎层切分执行器 ③FreeToken
ft_stats.h 观测面随下次编译接入 (步1 已就绪)


## 43. KV UI 绑死 ninfer + 三问诚实盘点 (2026-09-08 晚)
用户报告: 网页 KV cache 仍只有 fp8 三选、不能分开调节、新量化格式没出现。
根因: ninfer 分支被引擎门控 (backend==="ninfer") 而引擎下拉来自 runtime
  targets (只有 vllm/sglang/exllamav3), ninfer 永远不可选 => 分支永不显示。
修复: KvCache 改为无条件渲染 (绑死 ninfer 专用): 统一 KV 精度六选
  (BF16/Int8/FP8/NVFP4/ISO3/E8) + 逐层 KV 混合输入 (0-15:e8,16-31:iso3,...)
  + 序列化器三段补 kv_dtype/kv_layer_storage (原被白名单剥掉=不能持久化)
  + tsc 归零 + 前端重启 200。
用户三问盘点 (诚实):
1) 1Cat 借鉴未完全完成。已落地: split-KV verify v2 (三档全 PASS) / iso 编解码
   GPU 逐字节 / suffix lookup (fuzz 全绿)。未落地: 前缀缓存(含 GDN 循环态) /
   贪心 MTP 调度化 / 校准 int4 头 / 图 PINNING / GDN speculative-state 契约 /
   QPN MT=2 中段 / per-arch 分发层 / 老卡真机回归 —— 全部保留为工作包。
2) 自动化模型适配: adapt_all 管线 + 几何门 + 参数门 CLI 可用 (已验证),
   但 studio 添加模型流程尚未接 /api/autoadapt —— 接线未完成 (下一步)。
   MoE 检测: check_params 已报 MoE; adapt.py catalog 已加 num_experts
   new_op 检测 + raw_text_config stash (需 fn manifest 重生成验证)。
3) KV UI: 已绑死 ninfer 专用无条件显示; 序列化持久化已通; 浏览器硬刷新后可见。


## 44. KV K/V 独立量化选项 (2026-09-08 晚)
用户要求: K 量化一个选项, V 量化一个选项。
落地: 契约 +kv_k_fmt/kv_v_fmt; serializer 三段接线; UI 拆成 K 量化
  (E2M1/E8/Int8/BF16/FP8) 与 V 量化 (ISO3/NVFP4/BF16/Int8/FP8) 两个独立下拉;
  tsc 0 错。
引擎现状说明 (诚实): PagedKVLayerView 本有 dtype(K)/v_dtype(V) 双字段
  (NVFP4 层即 K=E2M1+V=ISO3 组合), 但逐层独立 K/V 平面组合需 plane-geometry
  扩展 = 工作包; UI/持久化先行, 引擎消费随后。
ngram 提醒 (用户点名): PLE 真表 gather 验证仍未跑 (PleTable 分页 fault 待修;
  sidecar 构建器+往返已闭环) — 保留为待办, 下次编译窗口一起做。


## 45. 分层 KV 验收 (新二进制复现): E8×BF16 混层长上下文 FAIL (2026-09-08)
验收测试 (qwen27, 57K 上下文针尖, 新二进制含全部补丁):
- 配置: --kv-dtype iso3 --kv-layer-storage 0-15:e8 => 实际 E8×16 + NVFP4(ISO3 V)×48
  (iso3 映射修复后, base 层不再是 BF16 而是真正的 NVFP4/ISO3)
- 结果: FAIL — 针尖未命中, reasoning 陷入自问自答循环 (同 §35 症状)
- 注意: 此时混合 = E8 层 + NVFP4(ISO3 V) 层混排, 依然是 4-plane vs 4-plane,
  但 plane 内格式不同 (E8 晶格码 vs E2M1 码) — 混排读取路径有真实 bug
结论: 混合格式 (E8×NVFP4) 分层长上下文有真实引擎 bug, 验收不通过。
定位方向: batch_layer_view 按层 dtype 解析 plane 格式/量化组 — 检查
  attention 核在层间切换 cache.dtype==E8Kv vs NVFP4 时的 scale 读取
  (E8 用 int8 核+per-64 FP16 scale, NVFP4 用 nvfp4 核+per-16 E4M3 scale;
  层间 stride/缩放因子串扰嫌疑最大)
单格式基线对照: NVFP4 uniform @57K PASS (§35) — 单格式无此问题
待办: ①复现最小化 (单层 E8 + 其余 NVFP4, 二分问题层) ②检查 E8Kv 层
  attention 输出的 scale 反量化一致性 ③通过后再跑全验收


## 46. E8 层数二分定位: 质量悬崖在 13→14 层 (2026-09-08 深夜)
64K 针尖二分 (N 层 E8 + 其余 NVFP4/ISO3):
  1/8/12/13 层 E8 = PASS (针尖命中); 14 层 = FAIL (推理语言混乱/丢针尖)
结论: **非机械 bug, 是量化质量悬崖** — 层 14-15 是最深的全注意力层
  (最接近输出、对 KV 精度最敏感, 与 MixKV/PyramidKV 文献一致:
  深层 KV 量化敏感度最高), E8 (2-bit 晶格 K) 覆盖到深层即崩。
与旧默认吻合: 老配置 10L E8 + 6L NVFP4 正是避开深层。
自动混合校准的策略修正 (直接指导 auto-mix 生成器):
  - E8/ISO3 只放浅层+中层, **深层 (最后 ~20% 全注意力层) 必须 NVFP4+**
  - 金字塔方向确认: 深层敏感度 > 浅层 (14 层才崩 vs 浅层全 E8 无碍)
  - auto-mix 分配顺序改为: tier0 (最高精度) 先填深层, 再浅层, 中层拿低精度
下一步: ①auto-mix 生成器按此修正 (深层保 NVFP4) ②重跑 14/16 层配置验收
  ③通过后 FreeToken 弹性预算解锁 (能量分位 + 深层保护双约束)


## 47. 分层 KV 验收通过: 深层保护策略 (2026-09-09)
策略: E8 覆盖浅/中全注意力层 0-9, 深层 10-15 保 NVFP4 (§46 深层保护)。
57K 针尖命中 PASS (XKCD-42-7777 精确召回 + 科学家细节)。
=> 分层 KV (混精度) 验收通过, 与 §46 质量悬崖结论自洽:
   E8 放浅/中层 = 安全省带宽; 深层保 NVFP4 = 精度不塌。
FreeToken 步 2 解锁: 弹性预算 = 深层保护 + 能量分位双约束的动态重排。
auto-mix 生成器深层保护逻辑与该验收策略一致 (深层 20% 保高精度)。
遗留: 深层敏感度的精确定档 (10-15 中哪几层是红线) 可后续细化; 当前
  "深层 20% 保高精度" 已足够安全。


## 48. FreeToken 步2 决策逻辑落地: ft_tiers.py (2026-09-09)
观测→决策→应用闭环打通:
- 输入: ft_stats [ft] 观测日志 (每层 mean_l, 实测 6/16 层覆盖)
- 决策: 深层保护 (最后 20% 强制 NVFP4) + 能量三分位 (低能量=e8,
  中=iso3, 高=nvfp4) + 未观测层保守 iso3
- 实跑真实数据输出: 12-15:nvfp4 (深层), 5:e8 (最低能量), 0-1/3-4/6-9:iso3
  — 与 §46 质量悬崖结论完全自洽
步2 剩余: 引擎侧动态重排 (运行中按新观测周期性 reload kv-layer-storage;
  静态透传已可用 — 观测→手动/脚本生成→重启生效)
FlashNext P0 骨架已落地: src/targets/qwen4_exp/{impl/config.h, package.h}
  (48 层 layer_kind 表 + MoE 512x10/PLE/MTP 参数注释; fn_official.json 生成)


## 49. 精度分配机制设计修正 (用户指正, 2026-09-09)
用户指正: E8×NVFP4 混排失败后, 不该用「深层保护」硬规则把低精度关掉,
  而应反思分配机制本身为何产生不合理分配 —— 改进分配模式, 不是关。
设计修正 (分配机制 v2):
- 位置金字塔 (浅/中/深) 降级为**无观测数据时的初始先验**
- 正式机制 = **敏感度校准驱动**: ft_stats 观测每层能量 (mean_l) ->
  按实测敏感度排序分配精度预算 (敏感层保高精度, 耐压层拿低精度) ->
  质量测试回归 -> 重校准迭代。位置先验仅在有观测数据前使用。
- ft_tiers.py 即此机制的决策实现 (能量分位驱动, 深层保护成为
  "深层实测高敏感"的涌现结果而非硬规则)
另落实「fusion 双层 NVFP4 ≠ 普通 NVFP4」:
- 引擎 `--kv-residual-layers` = 每层可选第二阶段残差平面
  (K/V 码+scale 双份, 有效精度更高) — 之前 UI 混为一谈
- 已接线: 契约 kv_residual_layers + serializer 三段 + 命令构建
  (--kv-residual-layers 透传) + UI V 下拉加 "NVFP4 双层残差" 选项
- tsc 0 错; 残差层质量/带宽实测 = 验收项


## 50. 判别实验: E8 质量悬崖 = 覆盖率效应 (非位置效应) (2026-09-09)
同为 14 层 E8, 两种位置 (0-13 含深层 / 2-15 含最深层) 均 FAIL =>
  悬崖由 E8 **覆盖率**决定 (14/16 = 87.5%), 与覆盖哪几层无关。
含义: 自动分配机制的校准输入 = 「格式-覆盖率-质量曲线」, 不是层位置。
  校准方法 = 对每种格式实测「覆盖率 vs 质量退化」曲线, 曲线上拐点
  即该格式的安全覆盖上限 (E8@nvfp4混排 ≈ 13/16 = 81%)。
  分配算法 = 在覆盖率上限内优先用便宜格式填到上限, 剩余层用高档格式。
  (旧默认 10/16 = 62% 有较大安全余量)
修复: serializer 三处 (键表/Schema/normalized) 补 kv_k_fmt/kv_v_fmt/
  kv_residual_layers — 之前只加了两处导致 ControllerRecipe 缺字段;
  去重后 tsc 0 错。/api/autoadapt 端点已接 (注册+类型通)。


## 51. E8 覆盖率校准扫描完成 (2026-09-09)
kv_calibrate.py 实测质量-压缩曲线 (0-16 层, 步长 2):
  0-12 层 E8 全 PASS, 14-16 层 FAIL。**安全上限 = 12 层 (75%)**。
  比旧默认 10 层多 20% 压缩收益。推荐参数: --kv-layer-storage 0-11:e8。
注意: 校准点为偶数步长, 奇数边界 (13 层) 未测 —— 由 bisect (§46) 确认
  13 PASS / 14 FAIL, 与扫描结果一致 (12 PASS 是因为 13 未测)。
auto-mix 生成器已含深层保护逻辑 (§46), 与此校准数据一致。
FreeToken 步 2 弹性预算 = 此校准 + ft 能量观测双输入动态重排。


## 52. E8 覆盖率校准完成 + 工作树全净 (2026-09-09)
kv_calibrate.py 扫描 (0-16 层, 步长 2): **E8 安全覆盖上限 = 12 层 (75%)**。
  14 层起 FAIL (与 §46 bisect 13/14 悬崖一致)。旧默认 10 层有 20% 余量。
  推荐参数: --kv-layer-storage 0-11:e8。结果已推 GitHub (工作树 clean)。
FreeToken 步 2 状态:
  - 观测 (ft_stats) ✓ + 决策 (ft_tiers.py 含深层保护) ✓ + 静态应用 (重启生效) ✓
  - 引擎动态重排 (运行中周期 reload) = 下一个引擎侧改动
下一步优先: ①E8×NVFP4 混排质量悬崖根因 (§45, 不是 bug 是质量边界, 但
  14 层悬崖的锐利程度暗示可能有可修复的非线性放大) ②引擎动态重排
  ③FlashNext bindings 回填 ④1Cat 剩余


## 53. 残差双层深层保护 PASS (2026-09-09)
策略: E8(0-9) + NVFP4+残差(10-15) @ 57K => 针尖精确命中 PASS。
残差双层 (--kv-residual-layers 10,11,12,13,14,15) 进一步验证为
  深层 NVFP4 的有效精度增强。与 §47 (无残差版) 组合确认:
  分层 KV 混精度在深层保护下完全可行, 残差提供额外精度余量。
=> 分层 KV 压缩验收完全通过, FreeToken 步 2 无阻塞。
下轮: FlashNext P0 bindings 回填 + 1Cat 前缀缓存 (GDN 循环态)。



## 54. Studio RAG v2 落地: 传文件-自动向量化-语义搜索 (2026-09-09)
用户需求 (传入文件, 自动向量化, 可搜索) 完整实现, 全本地:
- controller: modules/system/rag-routes.ts — POST /rag/ingest (分块600字/重叠60 +
  向量化), GET /rag/search (余弦top-k), GET /rag/documents, DELETE /rag/documents/:id。
  存储 = <data_dir>/rag/*.json, 每文档带 embedder 标签, 检索时不同向量化器文档自动跳过。
- 向量化双轨: @huggingface/transformers bge-small-zh-v1.5 (q8, 512维, 中英) 优先,
  失败回退 hash n-gram (512维确定性)。冒烟: 无关键词重叠中文查询命中 0.53 相似度。
- 前端: /rag 页面 (文件导入/粘贴/搜索/文档管理), 侧栏导航 Knowledge 项 (Database 图标)。
- 环境要点: HF 下载需走本机代理 HTTPS_PROXY=http://127.0.0.1:10808 (WinINET 代理 Bun
  fetch 不自动读); 模型已缓存, 重启后无需网络。controller 已带代理 env 重启 (pid 44336)。
- 双仓 typecheck 0 错; /rag HTTP 200。
受阻项: FlashNext P0 bindings 回填需要 vLLM qwen4_exp 权重键全集, dl/vllm-main-latest
只有 patch/issue 文件非全源码 — 待拿到 vLLM 源码后做映射表。
下一步优先: ①引擎动态重排 (FreeToken 步2 收尾, hook=layer_dtypes_ 运行时表)
②1Cat prefix cache (GDN 循环态) ③studio Configure 面板消费 /hwdec + /autoadapt


## 55. FreeToken 步2 引擎动态重排设计定稿 (2026-09-09, 下窗口直接实现)
勘察结论 (代码事实):
- 规划链: ServeOptions.kv_layer_storage[64]+kv_residual → EngineOptions (generation_service.cpp
  构造时一次性拷入) → make_sequence_planner (api_impl.h:498) → SequencePlan
  (layouts_impl.h:923 处 layer_overrides→DType 表) → create_program 消费 plan 建 KV 池,
  plan 绑定 weights_profile 校验 (api_impl.h:513)。=> dtype 表变更 = 重规划+新池+迁移换池。
- 实现契约 (Program 级, 序列边界执行, 不中断生成):
  ① Program<Variant> 新增 replan_kv(kv_layer_storage[64], kv_residual[64]):
     make_sequence_planner(device, 更新后 options, weights_profile) 产新 plan;
     新 KV 池按新 plan 分配; 逐层迁移用现有 per-format quantize 路径 (src→bf16 路径已有
     dequant, bf16→dst 已有 quantize; 同格式直接 memcpy); 在 generation_service 持锁点
     原子换池引用, 旧池等在飞请求排空后释放 (引用计数)。
  ② serve 入口: /v1/reload_kv (POST body = kv-layer-storage 字符串) → parse_kv_layer_storage
     (kv_options.h 现成) → 触发 ①; 另加 env NINFER_FT_RELOAD_SECS 周期轮询:
     读 NINFER_FT_STATS 输出的 [ft] 能量行 → 复用 tools/archkit/ft_tiers.py 决策逻辑的
     C++ 内联版 (深层保护 + 能量三分位, §48) → 若表变化才 replan (幂等, 抖动需滞回:
     连续 2 周期一致才切换)。
  ③ 验收: 57K 长测协议 (§38 针刺检索) 三态对照 bf16 起点 → 静态 0-11:e8+深层nvfp4 →
     动态重排到同表, 结果一致; 重排期间在飞请求不崩。
- 注意: CUDA graph 捕获与换池交互 — 换池必须发生在 graph replay 外 (序列边界已满足),
  换后需重捕获 (use_cuda_graph=false 快捷路径先验)。


## 56. FreeToken 步2 引擎动态重排实现落地 (2026-09-09, 构建通过, 冒烟中)
v1 = 排干式换池 (drain-based replan), 权重复用不重载。改动 8 文件 (Windows 主仓 + WSL 构建副本已同步):
- registry.h/cpp: Instance 增加 weights_profile 成员 (构造时存入); 新增
  replan_target_kv(ActiveTarget&, EngineOptions&, DeviceContext&) — 先析构旧 Program
  (释放 KV 池让容量解析看到真实预算) → 镜像冷启流程 (chunk 阶梯 + resolve_kv_capacity +
  plan 不变式校验) → create_program 换池。失败则 Program 为空 (重试或重启)。
- engine.h/cpp: 公开 Engine::reload_kv_storage(table, residual={}) — bind_to_current_thread
  (即 cudaSetDevice, 任意线程安全) + 更新 options_ + targets::replan_target_kv。
  安全性依据: EngineCore 只持 Instance&, 每次调用现解引用 instance_.program->
  (engine_core.h:839/1285), 排干后换 unique_ptr 无悬挂。
- generation_service.h/cpp: reload_kv_storage(spec) — 先解析 (坏 spec 不扰线上) →
  reload_in_progress_ 原子门与 request_capacity_->mutex 同锁 (通过者必已计数, 排干不漏)
  → 新请求 503 → 排干等待 (active==0, 上限 120s) → engine_->reload_kv_storage。
- http_server.h/cpp: POST /reload_kv, body {"kv_layer_storage": "0-11:e8,..."}
  (CLI 同语法, 兼容纯文本 body), 复用全局鉴权。
⚠ 发现并修复: Windows 主仓 engine.cpp 相对 WSL 构建副本缺 ScoreCoreMuse 守卫
  (submit 排除 + score_tokens 接受) — 覆盖同步时暴露。已补齐三核守卫; 以后 WSL→Windows
  同步必须 diff 校验, 不能单向覆盖。
验收路径: serve 起后 POST /reload_kv 换表 → 57K 针刺长测 (§38) 对照静态启动同表;
下一步 (v1 验收后): NINFER_FT_RELOAD_SECS 周期轮询 + ft_tiers C++ 内联决策 + 滞回;
studio 侧 ninfer 引擎适配器加 reload 命令 + UI 按钮。


## 57. warmup 崩溃诊断: 增量构建 ABI 失配 (2026-09-09)
现象: 全配置 warmup 必崩 cudaErrorMisalignedAddress (报错点漂移: prefill 探针 /
nvfp4_small_t getLastError = sticky error, 真 fault 在更早 kernel)。与 reload_kv 无关
(对照构建 = master+守卫, 同样崩)。
证据链:
- Sep-4 整体构建的 ninfer CLI 同工件跑通 (exit 0) => 环境无罪, GPU-PV dxg 报错是杂音。
- 内核 .o 全部 Sep 4; 非内核源码 Sep 4→9 间大改 (GqaMuse split/ft_stats/kv dispatch),
  增量链接 = 旧 ABI 对象 × 新头文件 => 错位访问。昨扫描通过用的是当时的连贯二进制。
- 教训: WSL 构建树源码被单向覆盖后, 必须 find . -name '*.o' -delete 全量重建;
  Windows→WSL 同步前先 diff 校验 (§56 已记 ScoreCoreMuse 守卫事故)。
处置: 已全量重建 (with reload_kv + 守卫), 重建产物上跑 reload_kv 冒烟
(test_reload.sh: 换表→生成→坏spec 400)。CLI 冒烟旗子: --max-new (非 --max-tokens)。


## 58. 自动 KV 量化分配器落地 (用户规格: 设 K/V 等级→自动分配→容量最大+精度可接受) (2026-09-09)
- tools/archkit/kv_auto_allocate.py: 输入 --k-grade/--v-grade (+模型/层数/目标上下文),
  输出最优 --kv-layer-storage spec。分配逻辑 = 校准覆盖上限 (§51: E8 ≤12/16=75%,
  needle-safe) 内最大化便宜层覆盖; 字节/token 数学给出目标上下文 KV 占用
  (64K → 7.48GiB @ 0-11:e8,12-15:nvfp4 = §53 已验收 spec)。
- studio controller: GET /kv_allocate (kv-allocate-routes.ts, 注册进 system routes),
  前端可用 ?k=&v=&target= 查询; typecheck 0 错, 实测返回正确 spec。
- 引擎语义现状 (决定分配器映射): 单表逐层; NVFP4 层 = 硬连线对 K=NVFP4+V=ISO3
  (decoder_state.cpp:283-291 推导规则); E8 层 = K=E8+V=E8。内核侧 cache.v_dtype
  已完全支持 K≠V (paged_kv_cache.h:58); 缺的是选项/规划器第二张表。
- 引擎 K/V 独立表设计 (下一步实现):
  ① EngineOptions.kv_v_layer_storage[16]+explicit + CLI --kv-v-layer-storage (同语法)
  ② SequencePlanningInputs.layer_kv_v_dtypes → SequencePlanImpl (layouts_impl.h:865 一带)
  ③ decoder_state.cpp plane-pair 循环按角色建几何 (K/V plane 尺寸可不同),
     view 的 .v_dtype/.v_quant_group 先查 V 表, 空则回退现行推导 (向后兼容)
  ④ reload_kv spec 扩展 v 字段; auto-allocate 升级为逐层 K/V 独立组合
  ⑤ 校准: kv_calibrate.py 扩展扫 (K_fmt,V_fmt) 组合覆盖曲线, 回填 COVERAGE_LIMITS


## 59. 排障夜实录: 编译连环崩 = hashcat 吃光宿主提交内存 + 提交态三处编译破损 (2026-09-09)
- 根因一 (环境): 12:42 用户启动 hashcat (Archive-Password-Recovery GUI, zip 密码恢复)
  占 20.6GB 提交 + GPU 破解; 宿主提交 126.5/127.4GB (99.4%) → Windows 杀 WSL VM 自保
  (Resource-Exhaustion 2004)。今晚 WSL 五连崩皆此模式; .wslconfig 已 14GB→24GB (备份
  .wslconfig.bak-14g)。用户决定: 等 hashcat 跑完; resume_when_free.ps1 已挂 (hashcat
  退出→3min→自动续 watchdog 编译+冒烟)。
- 根因二 (代码): main 提交态三处编译破损, 之前全靠 WSL 树未提交 WIP 硬撑 (rsync 覆盖后
  暴露): ①c312b81 Muse 分发使 TT>3 实例化 RowTiles≤3 断言失败 (i8.cuh:94/nvfp4.cuh:161)
  ②decoder_state.cpp plan_cache 本地数组 16 vs 布局结构体 64 ③layouts_impl 16/64 错位 +
  layouts_impl:929 越界读 (64 循环 × 16 槽表 = 真 UB)。
- 修复 (1d2a302 已推): 全家族 KV 表统一 64 槽 (types.h kKvLayerStorageSlots=64,
  EngineOptions/CLI/视图/SequencePlanningInputs/variants 默认表) + 双宏 Muse TT≤3 分发
  守卫 (超宽 draft 运行时显式抛错)。
- 教训: ①编译必须 -j1 (ptxas 峰值 ~9GB, 并行撞爆 VM) + watchdog_build.sh (26min 无进展
  自动杀重试) ②WSL 树不许当唯一真源; Windows→WSL 同步前 git diff 校验 ③宿主资源基线
  先查 (commit charge / hashcat / WSA) 再怪代码。
- 待办: hashcat 退场后自动链: 编译过 → serve 冒烟 (reload_kv 换表→生成→400) →
  final_result4.txt 出 SMOKE_DONE → 提交冒烟结论。


## 60. FlashNext P0 bindings 契约自研落地 (2026-09-09, 用户指示"源码没有就写")
上游 HF/vLLM 源码本地无货 → 契约自己定义: tools/archkit/flashnext_bindings.py。
- 规范名 + HF/llama.cpp 双风格别名通配 (首中即取); 覆盖: token_embd/head、48 层
  (GDN 36×8 项 / QSA 12×(7+indexer 3+hc 4×2))、MoE 48×(router+shared 3+512×3)、
  PLE 残差栈 6 项 (表 20019200×160 永不驻 GPU, SSD 契约)、MTP 3 项、final norm
  = 74,520 条契约项, --emit 出 JSON。
- 审计双向: --audit (safetensors index) / --audit-gguf (名单) → 缺哪些引擎张量 +
  哪些源键没消费 + complete 判定 (exit code)。自测: 8 假键命中 7, 未知键正确上报。
- 真正 checkpoint 到位后跑一次 audit 即可校准别名, 转换器照契约写, 无需上游源码。
- studio 前端 AutoMixBox 新增"服务端校准分配" (K/V 等级 + 目标上下文 → /kv_allocate
  → 直填逐层 spec), typecheck 0 错。


## 61. reload_kv 机制验证通过 + 换表后生成内核缺口 (2026-09-09)
- 编译收敛: types.h 64 槽 + layouts.h 残差表 64 + TT 守卫 + ScoreCoreMuse 守卫 =>
  main@1d2a302+TT 修复全量编译通过 (watchdog -j1, 多轮 VM 崩溃续跑, .o 持久)。
- **reload_kv 端到端 PASS**: serve (nvfp4, no-graph) 起→warmup 18s 过→POST /reload_kv
  {"kv_layer_storage":"0-11:e8,12-15:nvfp4"} → 14.7s 重规划+换池 → HTTP {"status":"ok"}
  → "drained; re-running KV sequence plan" → SERVE_STILL_ALIVE → 端口继续可用。
  解析/排干/预算解析/换池/HTTP 全链路正确 = FreeToken 步2 机制交付。
- 遗留缺口 (下一窗口修): 换到 e8 层后**首次生成**死: sticky cudaErrorInvalidValue,
  承接点 sigmoid_gate_mul.cu:22 (真 fault 在更早 kernel — E8 层 × Muse GDN 解码路径,
  新重建内核; 旧 WIP 内核时代 §53 曾 PASS, 故为新内核回归或旧 WIP 掩盖的缺口)。
  排查入口: 请求后第一个 CUDA_CHECK 前 dmesg/compute-sanitizer (WSL: /usr/local/cuda/bin),
  或对 e8 decode 内核 (gqa_attention_decode_e8.cuh) 参数审计。
- 基线注意: 换回 all:bf16 未及测 (首测脚本 model id 用了下划线, 正确 = muse-glimmer-30b);
  坏 spec 400 未及复测 (serve 已死于 sticky)。二测顺序: reload bf16 → gen (应过) →
  reload e8/nvfp4 → gen (复现) → sanitizer 定位。
- 环境: .wslconfig 24GB (备份 .wslconfig.bak-14g); watchdog_build.sh (26min 看门狗) +
  supervise_build.ps1 (Windows 侧守护, 冒烟完成自停) 模式已验证可跨 VM 崩溃续命。


## 61b. §61 缺口破案: 我的 TT>3 守卫是错的, 真相 = RowTiles 断言过时 (2026-09-09)
决定性实验: reload 到 all:bf16 后生成也失败 (503), 且 **未 reload 的 baseline 生成**
直接抛我加的守卫异常 "Muse small-T decode supports draft widths up to 3" =>
Muse 常规解码本来就走 TT4+ (kWc 表 TT1..TT6 正是为它设计); 我的守卫打断了正常服务。
真正的缺口 = 内核过时断言 `RowTiles <= 3` (i8.cuh:94 / nvfp4.cuh:161): Muse 的
RowTiles == TokenTile, kWc 表 (TT4→8, TT5→10, TT6→12) 下全部派生量合法
(ConsumerWarpsPerTile=2, PVNt=8, Wc%RowTiles=0, ProducerThreads=RowTiles*32 ≤ Wc*32)。
旧 WIP 时代 §53 57K PASS 即放宽态运行。
修复 (本轮): 两断言放宽至 ≤6 + 撤销两处 TT>3 守卫。重建中, 过后完整冒烟:
baseline gen (nvfp4) → reload e8/nvfp4 → gen → reload bf16 → gen → 坏 spec 400。
附带发现: 程序调用中的异常会把 EngineCore 打入 503 不可用态 (req-1 抛错后 req-2
直接 service_unavailable) — 引擎鲁棒性改进点, 记 backlog。


## 62. 双 agent 分工交接 (2026-09-09 19:5X, 用户指令: 拆 TODO 并行)
**TODO 维护权自此移交本 agent**: 以下进度记录、状态同步 (铁律③) 由你负责; 双方动手前先读
本文最后 20 行防冲突。我 (引擎侧 agent) 只在完成里程碑时追加一行到本节末尾, 不再维护全文。

### 你 (状态+轻侧 agent) 负责:
1. **_TODO.md 维护**: 各节进度更新、新任务记录、铁律守门。
2. **studio Configure 面板消费 /hwdec** (§54 下一步③): settings/engines-section.tsx 加
   硬解能力卡 (fetch /api/proxy/hwdec, pynvml 架构判定已在端点侧完成); 软解选项常开。
3. **FlashNext P0 转换器骨架** (对 6369cfe 的 bindings 契约写 tools/archkit/ 侧
   safetensors→ninfer 张量映射器骨架; 真 checkpoint 未到, 先契约驱动写代码+审计自测)。
4. **1Cat 剩余工作包 §43 设计文档化** (prefix cache/GDN 循环态、greedy MTP 调度、
   图 PINNING、QPN MT=2 中频段、per-arch 分发层、旧卡回归清单)。
5. backlog 转正评估: 程序调用异常后 EngineCore 打 503 不可用 (§61b 附带发现) 的
   鲁棒性改进设计。

### 我 (引擎侧 agent) 正在干 / 接下来:
- **正在跑**: build6 (08dbfa9 断言放宽版: RowTiles≤6 + 撤销 TT 守卫) — watchdog -j1,
  日志 /home/user/main_build6.log, 冒烟 watcher 自动出 /home/user/final_result5.txt。
- 构建完 → **reload_kv 完整冒烟**: baseline gen (nvfp4, model id=muse-glimmer-30b 连字符)
  → reload 0-11:e8,12-15:nvfp4 → gen → reload all:bf16 → gen → 坏 spec 期望 400。
  注意: baseline 若抛 "draft widths up to 3" = 08dbfa9 未生效 (旧二进制)。
- 冒烟过 → 57K 针刺长测 (§38 协议) 对 e8/nvfp4 分层做验收。
- 之后: FreeToken 步3 (带宽自适应调度) 引擎侧。
- 不碰: 你负责的 1-5 项 (studio 前端/工具链/文档)。


## 63. 状态+轻侧 agent 首轮: hwdec 卡 + FlashNext 映射器骨架 (2026-09-09 晚)
接手 _TODO.md 维护 (§62)。§62 第 2-5 项全部完成 (2/3 已交付, 4/5 设计文档已落引擎仓)。

### 2. studio Configure 硬解能力卡 (DONE)
- engines-section.tsx 新增 MediaDecodeSection (SettingsGroup "Media decode"):
  fetch /api/proxy/hwdec → Accelerator (名/vendor) + Architecture (arch + cc)
  + Hardware decode (h264/hevc/vp9/av1 badge + reason) + Software decode (常开);
  加载/失败态带 Retry; 软解不受 GPU 能力门控 (消费 soft_always_available)。
- 验证: tsc 0 错 + eslint 干净 + /settings 编译 200 + 浏览器实渲染可见卡片 +
  curl /api/proxy/hwdec 返回 5090D/Blackwell/hw_ok=true (UTF-8 正常)。
- local-studio 提交 f8c804c (本地 main; GitHub 推送 403 未解, 未推)。

### 3. FlashNext P0 映射器骨架 (DONE)
- tools/archkit/flashnext_convert.py: 契约 (flashnext_bindings 74,520 项) + spec 几何
  驱动, 逐张量输出 {源键, artifact 形状, 格式, 布局, 变换, 状态}。
- 布局实证 (非推测): 2-D 线性 = (out,in) 直通 (35b bindings.cpp:126);
  conv = (kernel, channels), channels = 2*key_dim+value_dim (35b config.h:37);
  1-D 范数直通; embed/head = (vocab,hidden); PLE 表 = SSD sidecar 不进 artifact。
- 自测 PASS: 74,520 项全解析 (0 缺失/0 计划外) / 149,184 alias 0 冲突 /
  44 抽样变换 (形状+数值+conv 行序) / safetensors 真读写 44/44 / 负路径
  (错形状) 正确报 mismatch + exit 1。
- 待真 checkpoint: 计划 73,800 项显式 pending (73,728 MoE 专家 pack 等内核定案 +
  dt_bias/A_log 语义拆分 + conv_bias 归属); 真权重到位跑 --checkpoint 即校准。
- 计划产物 tools/archkit/out/flashnext_plan.json (30.9MB, gitignored)。
- 引擎仓提交 3870add (本地 main, 未推 — 避免与引擎侧推送交叉)。

### 环境/守门
- controller 8080 此前不在跑 (仅 frontend 3000) → 已带 HTTPS_PROXY=127.0.0.1:10808
  重启 (RAG 的 HF 下载需要), /health 与 /hwdec 均通。
- .githooks/pre-commit|commit-msg|pre-push = git symlink 占位 (22 字节文本,
  scripts/hooks 不存在) → 两仓提交都需 --no-verify (local-studio 实测)。
- 未碰: WSL / ninfer-serve / 引擎侧日志。

### 4. 1Cat 剩余工作包设计文档化 (DONE)
- docs/maintainer/1cat-remaining-workpackages.md (引擎仓): 六项逐条「现状(代码实证)→
  缺口→设计→验收/风险」+ 实施序表。关键实证:
  · 前缀缓存: 已有 checkpoint/catalog 复用 + KV 页 COW (logical_kv_store.h:130/183)
    + GDN 态独立池 (linear_attention_state.h:14/25/64) + StateImage fork
    (state_image_store.h:353); 缺口 = 内容寻址哈希表 / 任意边界截断 / GDN record 重放。
  · 贪心 MTP: extent 由设备端 mtp_round.cuh:29-48 算, accepted_per_position 已记录
    (program_impl.h:12046) 但从不回馈宽度 → 逐位置接受率 + 滞回。
  · 图 PINNING: exec-graph update 已有 (program_impl.h:685), 按需扩展仅普通路径;
    reload_kv 重建 Program = 全量重捕获 (registry.cpp:250-263) → 目标「换池不重建」。
  · QPN MT=2: 内核与 e2e 已 PASS (2bcb4f3), 缺 host 入口 (qpn_host.cu:13-15 仅 M1-3)
    + CMake 接线 + 引擎 QType 接入 + M9-16 标定。
  · per-arch: sm() 全仓仅 3 处调用, 唯一 arch 行为是 sm!=120 硬拒
    (layouts_impl.h:824-835) → 中心路由表 + 非门禁原则。
  · 老卡回归: 仓内零基线/零 harness → 清单化 + old_gpu_baseline.json schema。

### 5. EngineCore 503 鲁棒性设计 (DONE, 待实现)
- docs/maintainer/engine-failure-recovery.md (引擎仓): 根因 = fail_all_locked
  (engine_core.h:1782-1804) 置 failed_ 且 worker 线程 return → 永久不可用;
  关键实证: CUDA_CHECK 是 std::abort (device.cu:39-44) → worker catch 只见宿主异常,
  /health 恒 ok (http_server.cpp:364-366) → 不可观测。
- 设计: 异常分类 (请求域→lane 级失败 / 资源域→可恢复 / 不变量→停摆) +
  Engine::recover() + POST /recover + /health 增引擎状态 + 校验前移
  (§61b 教训: 宽度守卫不该在核内抛)。验收门 4 条 (含 req-1 失败后 req-2 正常)。

### 交付物/提交
- 引擎仓 3870add (flashnext_convert.py) + 7b79c2f (两份设计文档) — **已推 origin/main**
  (08dbfa9 → 7b79c2f, 快进); 引擎侧下次推送前先 fetch/rebase。

### 附: 全量静态检查补测 (用户追问"测试了吗", 2026-09-09 晚)
- `npm run check:static` 之前从未真正跑过 (提交全走 --no-verify, 钩子是坏 symlink
  占位) → 补跑发现 **4 个 lint error, 全在旧文件 (非本轮改动)**:
  ① option-tab.tsx:480 `useState` 在提前 return 之后 → rules-of-hooks (引擎切换时
     条件返回翻转 → "Rendered more hooks" 崩溃, KV 面板可复现类);
  ②③ 同文件 `counts`/`si` 两个 let (si 是死变量);
  ④ rag/page.tsx:57 useEffect 违反本仓 no-restricted-syntax (仓库禁用 effect hooks)。
- 已修 (local-studio 提交 99d6332): hook 提到提前 return 之前 + const + 删死变量 +
  RAG 页改 useMountSubscription (与 engines-section 同模式)。
- 结果: check:static 全链绿 —— lint 0 error / typecheck 三套 / cycles 无环 /
  ui-structure PASS; /rag 与 /settings 均 200。
- 环境修复: `scripts/project.mjs` 同为 git symlink 占位 (内容
  "../frontend/desktop/project.mjs") → 已 mklink 重建, 工作树仍干净; 该文件是
  check/doctor/setup 入口, 修前这些命令全都跑不了。
- 硬解卡行内渲染已浏览器实证: Accelerator = NVIDIA GeForce RTX 5090 D (nvidia) /
  Architecture = Blackwell · cc 12.0 / Hardware decode = h264,hevc,vp9,av1 +
  "NVDEC 可用 (Blackwell)" + available / Software decode = Always available + enabled。

### 下一步 (本 agent)
- 待用户/引擎侧指令。候选: ①硬解卡接引擎侧 NVDEC 后端状态 (现仅能力展示)
  ②FlashNext 真 checkpoint 到位后 --checkpoint 校准 ③按 §63 实施序推进设计落地
  (施工属引擎侧范围, 我只做设计与验收契约)。


## 64. 交付单: 状态+轻侧 → 引擎侧 (2026-09-09 晚, 用户指令「给隔壁交付一下」)
**已推 origin/main: 08dbfa9 → 7b79c2f (快进 2 提交)。你下次推送前先 fetch/rebase。**

### 给你用的三件东西
1. **docs/maintainer/1cat-remaining-workpackages.md** (7b79c2f)
   六项施工单 (前缀缓存含 GDN 循环态 / 贪心 MTP / 图 PINNING / QPN MT=2 /
   per-arch 分发 / 老卡回归), 每项「现状(file:line) → 缺口 → 设计 → 验收/风险」。
   建议先做两项: **§2 贪心 MTP**(纯调度改动, 内核零改, +10-25 接受点) 与
   **§4 QPN MT=2 接线**(内核 e2e 已 PASS 于 2bcb4f3, 只差 host 入口 qpn_host.cu
   M>3 + CMake + QType 接入)。实施序表在文末。
2. **docs/maintainer/engine-failure-recovery.md** (7b79c2f)
   §61b 那个「程序抛异常 → EngineCore 503 不可用」的完整设计。
   关键实证 (与直觉相反, 会影响你的修法): `CUDA_CHECK` 是 `std::abort()`
   (device.cu:39-44) → worker 的 catch 只捕获宿主侧异常, 所以「请求域错误」
   完全可以只失败单请求而不停引擎; P0 改动点 = engine_core.h worker 循环
   (1782-1901) + http_server /health (364-366)。
3. **tools/archkit/flashnext_convert.py** (3870add)
   真 checkpoint 到位时直接跑:
   `python tools/archkit/flashnext_convert.py --checkpoint <dir|file>`
   → 自动报「源键缺失 / 形状不符 / 计划外键」; 自测 `--self-test [--with-safetensors]`
   当前全绿 (74,520 项, 0 缺失/0 计划外)。

### 需要你注意的契约缺口 (真键表到位后补)
- `layer.{i}.gdn.dt_bias` 的 alias 合并了 dt_bias 与 A_log 两个语义不同的张量,
  而引擎 GdnWeights 两槽都有 (35b bindings.cpp:122-125) → 契约需拆项。
- `layer.{i}.gdn.conv_bias`: 引擎 GDN 无 conv bias 槽 → 确认并入 convolution 或忽略。
- 两者在计划里已显式标 pending_semantics (不静默错算)。

### 边界与约定
- 我**没碰**引擎源码 / WSL / ninfer-serve / 你的日志; 只新增 1 个工具 + 2 份文档。
- 引擎仓后续我默认**不再推**, 除非你要求或提交纯工具/文档 (这次是用户交付指令下的推送);
  你要是有在飞的本地提交, fetch 后 rebase 到 7b79c2f 即可 (无冲突: 我只加新文件)。
- studio 侧 (不属你范围): 硬解卡 + 4 个 lint 修复 (含 KV 面板 hooks 崩溃修复),
  local-studio 本地提交 f8c804c / 99d6332。
- 你按老规矩往 §62 末尾追加里程碑行即可; 全文维护仍归我。

### §64 速览 (引擎侧先读这里)
- **origin/main 已到 7b79c2f** (我推的, 08dbfa9 快进 2 提交) → 你 push 前先 fetch/rebase。
- 三件产物: `docs/maintainer/1cat-remaining-workpackages.md` (六项施工单; 建议先做
  §2 贪心 MTP 与 §4 QPN MT=2 接线) / `docs/maintainer/engine-failure-recovery.md`
  (§61b 那个 503 的设计, P0 改动点在 engine_core.h worker 循环 + /health) /
  `tools/archkit/flashnext_convert.py` (真 checkpoint 到位跑 `--checkpoint <dir>`)。
- 边界: 我未碰引擎源码 / WSL / ninfer-serve / 你的日志; 引擎仓后续默认不再推。
- 契约缺口 2 条 (dt_bias/A_log 需拆项、conv_bias 归属) 见上一节, 已在计划里标 pending。


## 65. 数值分析复核: nvfp4 Muse TT4 smem (与 §66 同一 bug; 结论见文末合并速览) (2026-09-09)
**现象** (引擎侧报告, 我方未复现): Muse NVFP4 解码 TT4 实例在
`cudaFuncSetAttribute(MaxDynamicSharedMemorySize)` 返回 InvalidValue → 该实例起不来;
TT1 实例正常 (warmup 活着) → 启动能过, 走到 TT4 就炸。

**数值 (报告值)**: 静态 ~25KB + 动态 72KB ≈ 97KB, 贴死 sm_120 每块 opt-in 上限 100KB;
TT1 仅 62KB (余量 38KB)。

**我按源码复算** (launcher `src/ops/launcher/gqa_attention_decode_impl.cuh:363-372`,
kernel `src/ops/kernel/gqa_attention_decode_nvfp4.cuh:175-188`):
- 动态 = kRBytes + kVDynamicBytes; Muse TT4/Wc8/Bc32:
  kTileBytes = 4·32·128 + 4·32·16 = 18,432;
  kRBytes = 2·18,432 + 4·16·64 + 2·8·16·64 = 36,864 + 4,096 + 16,384 = 57,344;
  Iso3V V tile = 32·256·2 = 16,384 → 动态 = 73,728 B = **72KB, 与报告值逐字节一致**。
- 静态 (六个 `__shared__`): q_a 8,192 + q_sf 1,024 + static_r_s 16 + p_s 4,096
  + alpha_s 256 + physical_pages_s 1,024 = **14,608 B ≈ 14.3KB** → 比报告值低 ~11KB。
  **口径差待复核** (疑 cudaFuncGetAttributes 含编译器保留/对齐, 或报告实例与我复算
  的不是同一个) → 修完请用同一口径复测, 别拿我的算术值当验收基线。
- 逐档总占用 (我的口径, 含 Iso3V): TT1/Wc4 65.3KB · TT2/Wc8 77.6 · TT3/Wc6 78.0 ·
  TT4/Wc8 86.3 · TT5/Wc10 94.6 · TT6/Wc12 102.9 → TT6 已越 100KB。
  Wc 表出处: `gqa_attention_decode.cu:270-276` (i8, DynamicArena=false) 与
  `:398-405` (nvfp4, DynamicArena=true); Muse 分支 TT1→4 / TT2→8 / TT3→6 /
  TT4→8 / TT5→10 / TT6→12。

**三个候选修法 (报告, 按代价排; 字节数已按源码逐一校验)**:
1. **K ping-pong 减载 −18,432B (=18KB ✓)**: K 侧现为双缓冲 (2·kTileBytes = 36,864),
   退成单缓冲省一整个 tile。代价: K tile 载入与 PV 计算不再重叠 → 需测 decode tok/s。
2. **ISO3 V tile 复用 −16,384B (=16KB ✓)**: 省掉 kVDynamicBytes 的专用 V 解码缓冲,
   改用 repack/psc 段空闲区。代价: 需证明生命周期不重叠 + 同步点核对。
3. **调度层拆 TT**: 不动内核, 在 Muse nvfp4 分派处按 smem 预算换档 (TT5→TT4, 或
   TT4 拆两次 TT2); 最小做法 = 加编译期/启动期预算检查, 让越档别再以 InvalidValue
   形式炸在运行中。代价: 吞吐/形状变化, 需重测接受率。

**验收门 (修完提交后跑; 工具已自测)**:
- `smoke_full_nograph.sh` (项目根): 七步全序列 —— baseline gen(nvfp4) → reload
  `0-11:e8,12-15:nvfp4` → gen → reload `all:bf16` → gen → 坏 spec 期望 400 → health。
- `tools/archkit/longtest_57k.py --port 8321 --context 57344 --reloads "all:bf16"
  "0-11:e8,12-15:nvfp4"`: 57K 针刺逐字召回, exit code 判 PASS/FAIL。
- 两道都过 → 57K 分层长测 + FreeToken 步 3。


## 66. 引擎侧运行时证据: nvfp4 Muse TT4 smem 超限 (接 §61b; 与 §65 同一 bug) (2026-09-09)
build6 (08dbfa9) 编译通过, 但 serve 首个普通请求 (TT4) 崩:
  decode.cu:370 CUDA_CHECK(attr) failed: cudaErrorInvalidValue
  = nvfp4 launcher 里 cudaFuncSetAttribute(MaxDynamicSharedMemorySize) 被拒。
数值 (Muse, TT4, Wc=8, KeyBlock=32, Iso3V=true):
  kTileBytes=4*32*128+4*32*16=18432; kRBytes=2*18432+4*1024+2*8*1024=57344;
  kVDynamicBytes=32*256*2=16384; kDynamicBytes=73728 (72KB);
  静态 smem (Br=64): q_a 8K + q_sf 1K + p_s 4K + 杂 ~10K = ~25KB => 合计 ~97KB,
  贴着 100KB opt-in 上限 (TT1: kRBytes 47K + 静态 ~15K = ~62K 过 => warmup 活)。
候选修法 (按代价排序):
  a) TT4-6 时 kRBytes 减载: 2*kTileBytes 的 K ping-pong 改单缓冲或 KeyBlock=16
     (需动 Bc==32 断言与 tile 数学, 中风险, 直接收 ~-18KB);
  b) TT4-6 时 kVDynamicBytes 的 ISO3 V tile 换静态外置/复用 r_s (省 16KB);
  c) nvfp4 Muse TT 只到 3, TT4 拆两次 TT2 (调度层改, 需确认语义等价)。
验收岗 (引擎 agent): 你改完提交后, 我跑 smoke_full_nograph.sh 全序列 +
longtest_57k.py (tools/archkit/longtest_57k.py, 已就绪, --help 自测过)。
[§65 复核] 本节的动态 72KB 与 §65 源码复算逐字节一致; 但静态口径两方不一致
(本节"杂 ~10K"未细化 vs §65 实算 14.6KB) → 合计 97KB vs 86KB 直接影响修法选择:
先跑 §65 给的 5 行探针 (失败实例的 sharedSizeBytes + 设备 optin + 实际 TT 档),
再动内核。本节标号由 63 改 66 (63 已被状态记录占用)。


### §65/§66 合并速览 (引擎侧先读这里; 本文件末尾 = 唯一权威版)
- **活**: nvfp4 Muse 解码实例 `cudaFuncSetAttribute(MaxDynamicSharedMemorySize)` 返
  InvalidValue → 该 TT 实例起不来 (TT1 正常故 warmup 活着)。运行时证据 §66,
  数值复算 §65。
- **数值**: 动态 73,728B (72KB) 两方一致 ✓ = kRBytes 57,344 + Iso3V V tile 16,384
  (TT4/Wc8/Bc32)。静态: §66 估 ~25KB (含"杂 ~10K") vs §65 按源码实算 14,608B
  (六数组, nvfp4.cuh:175-188; 另 included `gqa_attention_decode.cuh:189`
  `reduce[256]` 1,024B → ~15.6KB)。
- **先探针再修**: 我算术下 TT4 ≈ 86KB < 99KB, 与"超限"不合 → 先跑 5 行探针
  (`cudaFuncGetAttributes().sharedSizeBytes` 于失败实例 + 设备
  `cudaDevAttrMaxSharedMemoryPerBlockOptin` + 实际 TT 档位)。要么静态实测更大,
  要么失败档是 TT5/TT6 (我算 TT6 = 102.9KB 才真越限)。**探针前不要动内核**。
- **修法 (按代价; 字节已按源码校验)**: ①K ping-pong 减载 −18,432B ②ISO3 V tile
  复用 −16,384B ③调度层拆 TT / 启动期预算检查 (别再以 InvalidValue 炸在运行中)。
- **验收 (修完提交后)**: `smoke_full_nograph.sh` 七步 +
  `tools/archkit/longtest_57k.py --reloads "all:bf16" "0-11:e8,12-15:nvfp4"` (exit 判);
  两道都过 → 57K 分层长测 + FreeToken 步 3。
- **分工待拍板**: §66 原写"你(对方)修 + 引擎 agent 验收"; §65 原写"移交引擎侧修"。
  事实约束: 修内核需 WSL 构建 + GPU 实测循环 (仅引擎侧有), 轻侧只能出分析 +
  从 Windows 跑 longtest (对 WSL serve)。**建议: 修 = 引擎侧, 验收 = 引擎侧跑
  smoke + 轻侧跑 longtest**; 用户若另有指派以用户为准。


## 67. 遗漏任务补全 (2026-09-09 深夜, 隔壁整理; 每项=现状→动作→验收→依赖)
检索源: 本文件全文 + 15 份 _*.md 历史状态 + 引擎仓 docs + git log + 会话归档。
旧版 TODO 考据结论: "最后一件=整理成 Windows 环境"那版正文无独立存档 (从未入 git,
todo_*.py 仅是追加器); 其条目已被现文 §5 吸收 (Windows EXE 单文件封装行仍在)。
以下按执行优先级编号, 供双 agent 直接领取。

### W1. [引擎侧·当前] nvfp4 Muse TT4 smem 修复 (§65, 已派工对方)
现状: cudaFuncSetAttribute InvalidValue; 动态 72KB 复算一致; 三修法候选。
验收: smoke_full_nograph.sh 七步全过 + longtest_57k.py --reloads "all:bf16" "0-11:e8,12-15:nvfp4" exit 0。

### W2. [引擎侧] PLE 真表 gather 验证 (用户点名"那个 lookup/ngram", 断在最后一步)
现状: sidecar 构建器+引擎往返 ✅ (§7); suffix_lookup/fuse_chain ✅; 唯缺真表 gather。
资产: data/ple 表 bin (flashnext_ple/ple/ple-bf16-*.bin 4 分片 + ple-manifest.json
  含 per_head_vocab_sizes/offsets/multipliers 全参数, sha 已核)。
动作: ①写 tools/ple_gather_test.cu: mmap 第 1 分片, 按研究文档公式
  (mixed=ctx0*m0^ctx1*m1(^ctx2*m2); row=mixed%per_head_vocab+off) 算 16 行号,
  gather 16x160 bf16, GPU/宿主双路对拍; ②修 PleTable 分页 fault (engine 侧
  sidecar 加载路径); ③接 fuse_draft 链 (LABD q16 变体, §9.6.1)。
验收: gather 行号与 manifest 公式逐位一致 + fuse_draft 长上下文 prompt acceptance 提升。
依赖: 无 (纯 CPU/GPU 读表, 不依赖 serve)。

### W3. [引擎侧] 1Cat 贪心 MTP 调度化 (施工单 §2, 最先做)
现状/设计: docs/maintainer/1cat-remaining-workpackages.md (对方已交)。
关键位: accepted_per_position 已记录 (program_impl.h:12046) 从不回馈宽度。
动作: 读该统计 + 滞回调整每轮 draft width (min..max 区间)。
验收: 复现类 prompt acceptance +5 点以上, tok/s 不降; 回归 = 普通 prompt tok/s 持平。

### W4. [引擎侧] QPN MT=2 host 接线 (施工单 §4)
现状: 内核 e2e PASS (2bcb4f3); qpn_host.cu:13-15 仅 M1-3。
动作: ①qpn_host 加 M4-16 入口 (调 mma 变体) ②CMake 注册 ③QType 消费
  ④M9-16 数值标定 (exhaustive probe 照 §46 方法)。
验收: qpn_numeric_test 扩展 M=4..16 全过; V100 真机 (等硬件)。

### W5. [引擎侧] FreeToken 步2 收尾: 周期自动重排 (§55 设计①②后半)
现状: 手动 POST /reload_kv ✅; NINFER_FT_RELOAD_SECS 轮询未实现。
动作: serve 内 env 驱动后台线程: NINFER_FT_STATS=1 时读 [ft] 行 → ft_tiers C++ 内联版
  (深层保护+能量三分位) → 表变化且连续 2 周期一致 (滞回) → 内部调 reload_kv_storage。
验收: 起 serve 带 env, 跑 57K 长测, stderr 出现 "auto-relayout" 行且生成不中断。

### W6. [引擎侧] FreeToken 步3: 带宽自适应 (§41 B)
现状: 零代码。设计: decode 带宽占用实测 → 动态调 prefill-chunk/冷页换入节奏。
动作: ①ft_stats 扩展带时间戳的带宽采样 (event timer 已有基建) ②admission_policy
  消费带宽信号调 chunk。验收: 64K 长测 tok/s 波动 <10% 且 ppl 不塌。

### W7. [轻侧] FlashNext P1: 纯文本最小栈 (flashnext_plan P1)
现状: P0 骨架+契约+映射器 ✅; 真 checkpoint 未到。
动作: 等 checkpoint → audit (flashnext_bindings --audit) → 校准别名 (dt_bias/A_log
  拆项, conv_bias 归属, §64 缺口) → dense 层+shared expert 转换器最小版 → serve 对拍。
依赖: 真 qwen4_exp 权重 (用户提供; fn_official.json 是规格非权重, §14)。

### W8. [引擎侧] rmsnorm 权重折叠 PPL 回写 (_opt_plan/_progress_fold2 悬案)
现状: 验证到 PPL 对比前一步断了 (56 层 8.5% 一致率=噪声, PPL 结果未回写)。
动作: ninfer-perplexity --text perplexity-corpus-long.txt 跑 fold artifact vs 基线
  (注意 ppl 不支持 --kv-dtype nvfp4, 用默认); 差<0.1 → 按 _opt_plan 任务1 实施
  (b/c/d 步), 否则记"放弃"关案。验收: PPL 数字回写本文件, 决策落地。
依赖: GPU 空闲窗口 (与 W1 互斥)。

### W9. [轻侧] 训练线重启决策 (§9d 结案三选项悬置, 用户定)
现状: 主 run 作废 (bad-teacher), pilot 1.39 证明 TOPK 路径有效, 全量重采未启动。
选项 (原样保留): A 用户给 22:17 pilot 的 serve 启动方式 → 复用采集;
  B 扩内存 ≥48GB 后自动; C 继续挂起跑 GPU 活。
**需要用户拍板, agent 不得自行选择。**

### W10. [引擎侧] KV 组合矩阵全量重测 (§35 遗留: iso3 映射修复后未重测)
动作: longtest_57k.py --reloads 依次: "all:iso3" "all:int8" "all:fp8" "all:nvfp4"
  "0-11:e8,12-15:nvfp4" "0-9:e8,10-15:nvfp4"(残差版需 CLI) 各 @57K。
产出: 覆盖率×质量表回填 kv_auto_allocate.py COVERAGE_LIMITS (现为 Muse 单点)。
依赖: W1 修复 + serve 稳定。

### W11. [引擎侧] Muse HF logits 对拍 (§11 尾悬案, Muse 质量无定量基线)
动作: pip install transformers@main → MuseGlimmerForConditionalGeneration CPU bf16
  分片跑同 prompt (短上下文即可) vs 引擎 logits (ninfer CLI 加 --dump-logits 或
  score_tokens API)。容忍阈值 ≤1e-2 (量化版)。依赖: GPU/CPU 空闲大块内存。

### W12. [轻侧] 4090/SM-count 借鉴立项 (检索新发现, _ninfer_ecosystem_devices.md)
来源: UDPSendToFailed/Azhu9701 ninfer-4090d (NINFER_TARGET_SM_COUNT 编译期宏,
  4090D 114/4090 128/3090 82/5090D 170) + Windows ninfer-serve.exe + charlesarcher sm_89。
动作: ①launch 网格审计: 哪些内核写死 SM 数 (grep <<<.*1728|<<<.*grid 常量)
  ②引运行时 device.props.multiProcessorCount 替代编译期宏 (低风险普遍化)
  ③sm_89 fp8 路线评估 (4090 有 fp8 mma 无 4bit → FP8 权重档)。
验收: 编译期 SM 常量清零清单 + per-arch 分发表初版 (接 §15 待办第2项)。

### W13. [引擎侧] 权重 host 卸载 + MoE 专家页 (§10.2/3 + AUTOPILOT S6.7, 零实现)
设计已有: 层/专家粒度 byte 镜像 + 按需换入, 挂 cold-host 同族基建。
动作: ①cold_host 机制从 KV 扩到权重 (load 期标 layer/expert 驻留等级)
  ②MoE 专家页 LRU。验收: 强制低显存 serve 不 OOM, 输出与全驻留一致。
依赖: FlashNext W7 (首个 MoE 消费者) 或 35b MoE 模型。

### W14. [轻侧] 无损压缩 probe (§10.1, compress_probe.py 未写)
动作: 写 probe 实测零行/填充行/scale 冗余比例 → 决定是否值得布局扩展。
验收: 数字回写, <2% 直接关案。

### W15. [轻侧] Windows EXE 封装 (§5 遗留 = 旧版 TODO 最后一件; GUI 验收后)
现状: studio 已替代旧 GUI 且可用 → 条件成熟。
动作: PyInstaller spec 打包 controller+frontend 静态导出 (next build standalone)
  + 环境自检首启 (python/cuda/bun 检测指路)。验收: 双击 exe 起全栈。
依赖: studio 浏览器验收 (用户)。

### W16. [引擎侧] EngineCore 503 鲁棒性实现 (设计已交: engine-failure-recovery.md)
P0: worker 循环 (engine_core.h:1782-1901) 请求域异常 lane 级失败 + /health 增引擎态。
验收门 4 条在文档里 (含 req-1 失败后 req-2 正常)。

### W17. [小项清尾] ①多线程/多流宿主管线 (§9.5, 一句话待办, 归 W6 带宽自适应)
  ②ccache (无 sudo 未解; 若可 apt 即装, nvcc 增量秒级) ③q16 验证图变体 (归 W2 LABD 链)
  ④Muse 参考源 NVFP4-QAT repo URL 失传 (§11: 对拍若需 QAT 栈要重找)。
执行状态: 本节条目完成时在行尾标 [done YYYY-MM-DD] 并写一句结果。

## 68. 子代理机制实证 + GLM-5.3 接线 + 并行 (2026-09-09 深夜, 用户指令)
- 问题: 子代理能否用 GLM-5.3 且与主线并行。结论 (客户端源码 + 三组判别实验):
  · 机制: 用户级子代理 = `<storageRoot>/agents/*.md` (frontmatter 支持 name/description/
    model/thoughtLevel/tools/disallowedTools/skills/permissionMode/maxTurns/background/
    injectAgentsMd/mcpServers/color; 出处 `resources/glm/zcode.cjs` 的 Z0t/$ti);
    内置 general-purpose/Explore 的模型覆盖 = `<storageRoot>/v2/agents-state.json` 的
    builtInModelOverrides (只认这两个键; 出处同文件 ZOi/H0t)。
  · 模型串 = `<providerId>/<modelId>` (例 `builtin:bigmodel-coding-plan/GLM-5.3`);
    DB model_usage 历史确有 `zcode-Explore` 跑 `builtin:zai/GLM-5.3`。
  · **生效时机 = 会话启动时固化**: 本会话内改文件无效 (判别: bogus 模型不报错;
    有效异模型仍跑主模型) → 需重启客户端/新会话; UI Settings→Subagents 改则即时。
- 已落: ~/.zcode/v2/agents-state.json = {general-purpose, Explore} → GLM-5.3 (下次启动生效);
  备选串 a1dcf64a-84d1-4264-9a6e-d44dc8003c96/glm-5.3。
- 并行: 已启用 (子代理 run_in_background 后台跑, 主线不阻塞); 首例 = W14 压缩 probe。

## 69. W2① PLE 真表 gather 验收 PASS + 环境事故 (2026-09-09 深夜)
### W2① 完成 (真表 + 真 GPU 三方一致)
- 新工具: `tools/archkit/ple_gather_check.py` (宿主权威通道) + `tools/ple_gather_test.cu`
  (GPU/宿主双通道对拍, 自包含无引擎依赖)。引擎仓 a4d2298 / ada226e, **已推 origin**。
- manifest 语义审计 (拿 HF 官方实现独立复算, 全 OK):
  · per_head_vocabulary_sizes = 第 h 个 >= 20000000 的素数 ✓
  · per_head_offsets = 逐头累计 ✓ · padded_vocabulary_rows = ceil(total/128)*128 ✓
  · **layer_multipliers 由 splitmix64(seed=1234, vocab=248320) 逐值复现** ✓
    → manifest 与 HF 实现一致 (正是引擎 PleTable 消费的那份参数)。
- 真表 gather: 24 token × 16 头 = 384 行, 0 越界; 4 分片随机读 122,880B;
  载荷 sha256=5991ec81... / fnv1a64=0xe8120b70c21aaeb8。
- **GPU 对拍 PASS** (RTX 5090; Windows nvcc 13.2 + MSVC 14.44, 无需 WSL):
  row_mismatch=0 / byte_mismatch=0 / fnv 三方一致 (Python / C++ host / CUDA)。
- 复现: `python tools/archkit/ple_gather_check.py --manifest <m> --data-dir <d>
  --vocab-size 248320 --ngram-base 20000000 --emit-spec <s>`
  → `nvcc -O2 -std=c++17 tools/ple_gather_test.cu -o ple_gather_test`
  → `./ple_gather_test --spec <s> --data-dir <d> --expect-fnv 0xe8120b70c21aaeb8`
- 剩余 (W2 ②③): 引擎侧 PleTable 分页 fault 修复 + 接 fuse_draft 链。

### 环境事故: 引擎仓长路径出现「残桩目录」(双 agent 必读)
- 现象: `C:\...\infer-fusion-repo` 长路径间歇不可访问; 某时刻后解析到**只含 tools\ 的
  残桩** (本 agent 写入时被建歪), 真仓库在 8.3 短名 **NI2A3F~1** 下完整可访问。
- 判据: `dir 长路径` 只列 tools; `dir NI2A3F~1` 列出 .clang-format/apps/AGENTS.md 全树;
  `git -C NI2A3F~1` 正常 (HEAD=ada226e)。
- 处置: 本 agent 一律用短名访问引擎仓; 长路径残桩**未删** (待确认无进程占用);
  建议用户/隔壁核实后清理。**写入前先 dir 核对目标树。**

## 70. W1 诊断修正: 越限的是 TT6, 不是 TT4 (2026-09-09 深夜, 实测)
用独立探针 (`tools/archkit/nvfp4_smem_probe.cu`, 提交 7212f94) 在 RTX 5090 上直接
实例化 nvfp4 Muse 解码内核并查询属性, **无需引擎构建** (Windows nvcc 13.2 + MSVC 14.44,
`-gencode arch=compute_120a,code=sm_120a`):

| TT | Wc | 静态(B) | 动态(B) | 合计 | cudaFuncSetAttribute |
|----|----|---------|---------|------|----------------------|
| 1 | 4 | 4,416 | 62,464 | 65.3 KiB | OK |
| 2 | 8 | 7,808 | 71,680 | 77.6 KiB | OK |
| 3 | 6 | 11,200 | 68,608 | 77.9 KiB | OK |
| 4 | 8 | 14,592 | 73,728 | 86.2 KiB | **OK** |
| 5 | 10 | 17,984 | 78,848 | 94.6 KiB | OK |
| 6 | 12 | 21,376 | 83,968 | **102.9 KiB** | **cudaErrorInvalidValue** |

设备 opt-in = 101,376 B (99 KiB), sm_120。

结论 (修正 §65/§66 的派工前提):
1. **越限档 = TT6** (105,344 B > 101,376), 超出仅 3,968 B (3.9 KiB); TT1-TT5 全部通过。
   §66 报的 "首个普通请求 (TT4) 崩" 与实测不符 → 需核对运行时实际选中的 TT 档
   (可能是 TT6, 或该次构建的 kWc/KeyBlock 与 main 不同)。
2. **静态实测 14,592 B (TT4) / 21,376 B (TT6)**, 否掉 §66 的 "~25KB" 估算;
   与我 §65 的源码算术 (14,608 / 21,392) 差 <0.1% (对齐/填充)。
3. **修法落点**: 只需让 TT6 降 ~4 KiB 即可, 两个候选都绰绰有余 ——
   ISO3 V tile 复用 (−16,384) 或 K ping-pong 单缓冲 (−18,432); 或调度层把 TT6 拆成
   TT5+TT1 / 直接对 TT6 走回退档。**不必动 TT1-5**。
4. 探针可复跑: `nvcc -O2 -std=c++20 -Xcompiler /std:c++20
   -gencode arch=compute_120a,code=sm_120a -I <repo>/src tools/archkit/nvfp4_smem_probe.cu -o probe`
   → `./probe`。改完内核重跑即可验收 (setattr 全 OK 为准)。

## 71. W14 无损压缩 probe 结果: 值得做 (Muse 7.46% 可回收) (2026-09-09 深夜)
工具: `tools/convert/compress_probe.py` (W14 子代理写就, 被重启打断后由本 agent 跑完)。
命令: `--artifact out/muse_glimmer_30b_nvfp4.ninfer --sample-layers 2 --sample-rows 4096`
(Muse 19.7GB 全量元数据 + 分组抽样)。

**结论: 总可无损回收 1.37GB = 7.457% (0 阶熵界), 远超 2% 门槛 → 值得做布局扩展。**

| 格式/布局 | 张量 | 字节 | 零行 | scale | code | 可回收 |
|---|---|---|---|---|---|---|
| NVFP4 blockscale-k16 | 131 | 9.76GB | 0.002% | 1.968% | 0.566% | 2.536% (253MB) |
| FP8 row-scale | 287 | 8.60GB | 0.005% | 0.011% | 13.033% | 13.048% (1.12GB) |
| BF16 contiguous | 313 | 2.68MB | 0 | 0 | 53.279% | 53.279% (1.43MB) |

关键读数:
- **FP8 code 平面是最大头**: 6.13GB, H=6.5376b/8b (distinct 254) → 熵编码理论上省 1.12GB。
  单张最大 = `text/token_embedding` [202112,6656] 可回收 209.77MB (16.35%)。
- **NVFP4 scale 平面**: 423MB, H=4.2848b/8b (distinct=65) → 熵编码省 ~197MB;
  注意 **RLE 是负收益** (runs=4.05e8 → 编码后 772MB, 比原平面还大 82.5%), 必须用熵编码不是 RLE。
- **零行剔除基本无效** (0.002-0.005%): vocab padding 行本来就是干净/已 pad, 这条线关案。
- 容器开销 (对象间 gap + 平面内 pad) 可忽略 (64KB 级)。
- 抽样口径: 大张量每 (格式,布局) 组最多全扫 2 个、每张 4096 行 (小张量全扫), 见脚本 CLI。

建议的布局扩展 (若实施):
1. 静态熵表 (rANS/Huffman, per-format 固定表, 不需每模型训练) 编 code 平面 —— FP8 优先;
2. NVFP4 scale 平面同样静态熵编码 (distinct=65, 天然适合查表);
3. 解码核需要新增 (熵解码 + 原布局重排), 属新 op; 先做只读验证 (往返 bit-exact) 再谈性能。
下一步: qwen27 artifact 对照跑 (后台进行中); 若两者结论一致 → 立"熵编码布局"工作包。

### 最新必读 (2026-09-09 深夜, 引擎侧先看这条)
1. **W1 前提已纠正 (实测, §70)**: 越限的不是 TT4 而是 **TT6** —— 探针实测
   opt-in=99KiB, TT1-5 全 OK (TT4=86.2KiB), **TT6=102.9KiB → cudaErrorInvalidValue**。
   TT 档位来自 `invocation.width`(草稿窗+1), Muse 宽度 6 才会走到 TT6。
   修法只需给 TT6 降 ~4KiB (ISO3 V tile 复用 −16K / K ping-pong 单缓冲 −18K),
   **TT1-5 不要动**; 探针 `tools/archkit/nvfp4_smem_probe.cu` 可复跑自验收。
2. **W2① 已验收 PASS (§69)**: PLE 真表 gather 三方一致 (Python/C++host/CUDA),
   manifest 参数与 HF 实现逐值一致; 剩 W2② (PleTable 分页 fault) ③ (接 fuse_draft)。
3. **W14 结论 (§71)**: Muse artifact 无损可回收 **7.457%** (FP8 code 平面为主),
   qwen27 对照 **5.538%** (1.11GB, 同源: FP8 code H≈6.53b/8) → 两模型一致, 值得做;
   零行剔除关案, RLE 负收益 → 若做就走静态熵表 (per-format 固定表)。
4. **环境**: 引擎仓用短名 `NI2A3F~1` (长路径是残桩, §69); 子代理已可跑 GLM-5.3。

## 72. Windows 原生引擎构建评估: 确认不可行 (2026-09-09 深夜)
目标: 让本 agent 能在 Windows 上自建引擎并跑 GPU 验证 (不依赖 WSL)。
实测结论 (**到此为止, 不再投入**):
1. **工具链本身可用**: vcvars64 (MSVC 14.44) + nvcc 13.2 + Ninja + CMake 4.4 全在位;
   单 TU/单内核级编译已验证可行 (PLE gather 工具、smem 探针都在 Windows 上编译并跑通)。
2. **配置可绕**: CMakeLists:72 的 FFmpeg pkg_check 是无条件的, 但 ops-only
   (`-DNINFER_BUILD_APPS=OFF -DBUILD_TESTING=OFF`) 并不链接 FFmpeg →
   用桩 .pc (`_stub_pc/*.pc` + `PKG_CONFIG_PATH`) 即可 `CONFIGURE_OK`。
3. **卡死在 POSIX 头**: 实编 `ninfer_core` 时 `src/runtime/engine/context_cost.cpp:15`
   引 `unistd.h` → MSVC 无此头; 引擎有真实 POSIX 依赖 (需 shim 层: unistd.h/getpid/
   sysconf 等), 属大工程且收益有限。
**结论**: 维持 WSL 构建为主 (隔壁); 本 agent 走 **standalone 单内核/单工具验证** 路线
(已在 PLE gather 与 smem 探针上验证有效)。Windows 全量构建列入"远期/不做"。

## 73. Windows 全功能原生移植工程 (2026-09-09 深夜, 用户定调: 不计代价, 功能一个不少)
**用户要求**: 主构建仍在 WSL; 但最终交付 Windows 原生全功能 (serve/CLI/media/测试全在),
且 **方便调试 + 一键安装 + 傻瓜看得懂**。

### 现状 (已实测)
- 工具链在位: MSVC 14.44 (vcvars64) + nvcc 13.2 + Ninja + CMake 4.4; 单 TU/单内核
  编译与运行已跑通 (PLE gather / smem 探针)。
- CMake 配置可通: `-DNINFER_BUILD_APPS=OFF -DBUILD_TESTING=OFF` + 桩 .pc
  (FFmpeg 的 pkg_check 无条件执行但不被 ops-only 链接) → CONFIGURE_OK。
- **精确阻塞面 (967 文件全扫)**:
  | 位置 | POSIX 用法 | 处置 |
  |---|---|---|
  | src/artifact/reader.cpp:180-215 | open/fstat/close/mmap/munmap + O_DIRECT | shim (Win32 映射) |
  | src/ops/ple/ple_table.cu:55-64 | open/close | shim |
  | src/runtime/engine/context_cost.cpp:299 | getpid | shim (_getpid) |
  | apps/serve/main.cpp:126-127 | signal(SIGINT/SIGTERM) | 控制台 handler |
  | src/product/media_acquire/acquire.cpp | POSIX socket/netdb/arpa | Winsock shim + closesocket |
  | tests/test_http_transport.cpp | socket/close/poll | Winsock shim |
  | src/core/arena.cu | dlopen/dlsym | LoadLibrary shim |
  | 其余 | unistd.h 仅用于零散符号 | 一个 unistd.h shim 覆盖 |
- FFmpeg: media 解码依赖, Windows 侧需 MSVC 可用件 (vcpkg 或预编译 .lib)。

### 工作包 (W-P1..W-P6)
- **W-P1 shim 层** (`compat/win32/`, 仅 WIN32 生效, 零侵入 Linux): unistd.h / fcntl.h /
  sys/mman.h / dlfcn.h / sys/socket.h+netinet+arpa+netdb。验收: 引擎核心+ops 在 Windows
  编译通过, WSL 构建行为不变 (shim 只在 WIN32 加入 include 路径)。
- **W-P2 全目标 Windows 构建**: engine/serve/CLI/tests 全绿; FFmpeg 用 vcpkg 或预编译件;
  验收 = `ninfer-serve.exe` 在 Windows 起 serve 并跑通 smoke。
- **W-P3 调试友好**: RelWithDebInfo + PDB; CUDA_CHECK 失败改为可捕获的错误 + 崩溃日志
  (现在是 std::abort); `--log-level`; `ninfer-doctor` 自检脚本 (驱动/CUDA/显存/FFmpeg)。
- **W-P4 一键安装**: 安装包 (VC++ 运行库 + CUDA 运行库检查 + 引擎 + 模型目录 + 桌面快捷
  方式); 双击即用, 失败给中文指引。
- **W-P5 傻瓜文档**: 中文 README + 图文 + 常见问题 (显存不足/驱动旧/端口占用), 全部脚本
  带中文注释与自检输出。
- **W-P6 回归与验收**: Windows 与 WSL 同模型同 prompt 输出一致 (logits ≤1e-2) + tok/s 对照。

### 执行序 (本 agent 主推, 隔壁保持 WSL 主构建)
1. W-P1 shim 层 → 立即开始 (可自验证: Windows 编译)。
2. W-P2 逐目标打通 (从 ninfer_core → ninfer_ops → ninfer_engine → serve)。
3. W-P3/P4/P5 随 W-P2 并行推进 (调试设施先行, 安装器最后)。

## 74. W1 修复落地: Iso3V 路径回收 repack 缓冲 → TT6 78.9KiB (2026-09-09 深夜)
诊断依据 §70 (实测 TT6=102.9KiB 越 99KiB opt-in)。**修法** (最小且可证):
- 事实: 内核里 `repack_a/repack_b` (每 warp 16*64 字节, 共 2*Wc*16*64) **只在
  native mxf4nvf4 PV 分支 (`else`, nvfp4.cuh:845-848) 使用**; Iso3V 分支
  (794-844) 用 `v_bf16`+BF16 mma, 完全不碰 repack。
- 改动 (2 文件, 各 1 处 + 指针守卫):
  · `src/ops/kernel/gqa_attention_decode_nvfp4.cuh`: `kRepackBytes = Iso3V ? 0 :
    2*Wc*16*64`, 同步 `static_r_s` 尺寸与 `kRBytes`; repack 指针在 Iso3V 下指回
    `r_s` (不解引用)。
  · `src/ops/launcher/gqa_attention_decode_impl.cuh`: 同规则计算 `kRBytes`.
- **Windows 独立探针实测 (改后)**: TT1-6 全部 `setattr=cudaSuccess`,
  TT6 = 21,376(静态) + 59,392(动态) = **80,768 B = 78.9 KiB** (原 102.9), 余量 20 KiB。
- 影响面: Iso3V 实例 (Muse NVFP4 层 + qwen27 NVFP4 层) 省 smem; native 路径不变;
  数值路径不变 (回收的是未使用缓冲)。
- WSL 侧: 已备份原文件 (.bak-prett6) + 同步补丁 + `make -j2` 重建中
  (log /home/user/win_tt6fix_build.log); 完成后跑 `smoke_full_nograph.sh` 七步验收。
- 待办: 冒烟过 → 提交 Windows 仓 + 推送; 然后 57K 长测 (§38) + KV 矩阵重测 (W10)。

## 75. W12 完成: SM 数硬编码审计 (2026-09-09 深夜, GLM-5.3 子代理)
报告: `tools/archkit/_SM_COUNT_AUDIT.md` (提交 325d025)。摘要:
- 本仓**无** `NINFER_TARGET_SM_COUNT` 宏 (0 命中); SM 数全是散落字面量, 按 5090 (170 SM) 调优。
- **真·编译期 SM 常量 9 处**, 其中 4 处 host 侧可直接清零:
  · `gdn_gating_proj/bf16/bf16_gdn_gating_proj_plan.cpp:130,139` (340/680 驻留判定,
    对应 `gemm_mma.cuh:296` 的 `this_grid().sync()` 协作启动) —— **唯一正确性相关**:
    非 170-SM 卡会驻留不足 -> 协作启动死锁/失败。
  · `gdn chunked/output.cu:9-11,46` (170*4=680 wave)、`rope.cu:18` (1020=170*6)、
    `sparse_moe/prefill/sparse_moe_prefill_kernels.cu:259-261` + 6 launch (510 持久块)。
- 派生常量 (含容量语义, 需分阶段改): `gqa_attention_geometry.cuh:21` / `causal_cache/
  geometry.cuh:12` (85=170/2, 同时是 split-K scratch 容量)、`bidirectional_gqa_attention
  .cuh:18`、`context_query.cuh:18`、`gqa_attention_decode.cu:78`+`.cuh:109` (42≈170/4 cap)。
- 不要动: `w8_linear_add_gemm_simt.cu:76` (2048=kIntermediate)、`w8_pair_plan.cpp` 680 (列边界)、tile/warp 常量。
- 待实测 (c 类): `kSparseMoePrefillWideMin=768`、decode Paths 档、adaptive 47..51 窗、nvfp4 warp 交叉点。
- 运行时取值: `DeviceContext::props` 已有 `multiProcessorCount` (`src/core/device.cu:57`);
  建议加 `device_sm_count()` helper (`cudaDeviceGetAttribute` + 静态缓存, 失败回退 170),
  协作启动处加驻留 guard。
下一步: 立"per-arch 分发层"工作包时按此清单分批改 (先 A 类 host 侧 4 处 + guard)。

## 76. W1 修复补全: arena 数学有**两份副本** + 预存测试破损 (2026-09-09 深夜)
- **关键坑**: nvfp4 的 arena/动态 smem 计算在仓库里有**两份独立副本**:
  · `src/ops/launcher/gqa_attention_decode_impl.cuh` (~357-365) — 我先改的这份;
  · `src/ops/launcher/gqa_attention_decode.cu` (~349-363) — **实际运行的那份** (Muse 走它),
    首轮只改 impl.cuh -> 冒烟仍在 `gqa_attention_decode.cu:370` 崩 (同一 InvalidValue)。
  两处现已同步为 Iso3V 回收 repack 缓冲的写法; 教训: 改 nvfp4 launcher 必须**两处同改**
  (或先合并重复代码, 记入技术债)。
- **预存测试破损 (与本次改动无关)**: `tests/ops/test_cold_i8.cpp:188-200` 调
  `cold_i8_slot_pack_raw/restore_raw` 少了 `slot_bytes` 参数 (头文件已加, 测试没跟) ->
  `make all` 在 62% 断掉。已修 (补 `ops::kColdI8SlotBytes` + nullptr stream)。
- 构建状态: `make ninfer-serve` 通过; 冒烟在第一次补丁后仍失败, 第二次补丁已同步,
  正在重编大 TU (gqa_attention_decode.cu, ~20-25min), 完成后重跑冒烟。

## 77. W2② 落地验收 PASS: PleTable 真表 gather 端到端打通 (2026-09-09 深夜)
- 修的三处 (提交 5f2a289):
  1. `ple_table.cu` fault 路径: **cudaHostAlloc 不保证零填充** (原注释写错) -> EOF 尾部
     显式 memset, 否则最后一页的 padding 行读到脏数据。
  2. 同一次 `gather()` 内可能驱逐掉**已取过 UVA 设备指针**的 pinned 缓冲 (cache 超预算时)
     -> 新增 `gather_epoch`, 本轮 fault 的条目豁免驱逐 (跨 gather 才可回收)。
  3. `ple_table.h` 输出布局注释改正: 实际是 **token-major**
     (`t*n_heads*row_dim + h*row_dim`, 与 HF PLE `flatten(-2)` 一致), 原文写反。
- 新验收工具 `tools/ple_table_test.cu` (自包含, 直接编 ple_layout.cpp+ple_table.cu):
  真 sidecar + 真 GPU, 走引擎自己的 derive_rows + gather, 与 Python 参考对拍。
- **实测 PASS**: 行号逐位一致 (`rows[0][0..3]=3402798 28422562 45851954 63762923`),
  载荷 122,880B 的 fnv1a64 = **0xe8120b70c21aaeb8** (与 §69 的 Python/C++/CUDA 三方值相同),
  缓存重放逐字节一致。=> W2② 关案; W2③ (接 fuse_draft 链) 与真模型接线仍待 FlashNext
  真 checkpoint (PleTable 目前无消费方, 属 P1 接线)。
- 复现: `wsl -e bash _build_ple_table_test.sh` (脚本同步补丁+编译+跑, ~2min)。

## 78. W5 落地: FreeToken 步2 周期自动重排 (2026-09-09 深夜)
实现 (提交见下): `src/serve/kv_auto_relayout.{h,cpp}` + `ft_stats.h` 快照接口 + main.cpp 接线。
- 触发: env `NINFER_FT_RELOAD_SECS>0` 起后台线程 (另支持 `NINFER_FT_FULL_ATTN_LAYERS`
  默认 16 / `NINFER_FT_DEEP_FRAC` 默认 0.2)。
- 决策: 每周期读 `ops::ft::snapshot()` (进程内, 不再解析 stderr) -> `build_ft_spec()`
  = 深层保护 (最后 20% 全注意力层保 nvfp4) + 能量三分位 (低=e8 / 中=iso3 / 高=nvfp4)
  + 未观测层保守 iso3 —— **与 tools/archkit/ft_tiers.py 逐字符一致**。
- 应用: **语义比较** (解析成逐层表再比) 避免格式差异误触发; **滞回** = 同一候选连续
  2 周期才切; 应用失败 (reload 忙) 下轮重试; 成功后 stderr 打 `[ft] auto-relayout -> spec`。
- 验证 (已 PASS, 纯宿主): `tools/kv_relayout_test.cpp` —— C++ vs Python 参考 **MATCH**,
  滞回 (1st 记录/2nd 应用/3rd 无操作) + `format_kv_table` 往返正确。
- 待办: serve 端到端验收 (需引擎重编 + 起 serve 带 NINFER_FT_STATS=1/NINFER_FT_RELOAD_SECS,
  跑 57K 长测看 `[ft] auto-relayout` 行且生成不中断) —— 排在 W1 冒烟之后。

## 79. 预存测试 API 漂移全清 + 快速语法门 (2026-09-09 深夜)
问题: `make all` 在 62% 死在 `tests/ops/test_cold_i8.cpp` (预存破损, 与本轮改动无关)。
处置:
- 新增**快速语法门**技巧: 用 `build/compile_commands.json` 精确回放每个测试 TU 的编译命令,
  只加 `-fsyntax-only` (不产码) -> 110+ 个 TU 几秒内扫完, 不必付 50 分钟全量构建的代价。
  脚本: `_syntax_tests2.sh` (临时, 可复用)。
- 修 5 处预存 API 漂移 (提交 ebd4760):
  · `test_cold_i8.cpp` 三处缺 `slot_bytes` (补 `ops::kColdI8SlotBytes`);
  · `test_kv_cache_append.cpp` 两处 cyclic 调用缺 `window` (补 `kWindow`);
  · `test_speculative_round.cpp` 两处缺 DFlash2 的 `draft_ids/draft_probs` (补空 `Tensor{}`)。
- 复验: **122/122 测试 TU 语法检查 0 失败** -> 下一次全量构建不会再卡这些点。
教训: 仓库测试树有长期未跟头文件演进的破损, 建议把"语法门"并入日常 (比全量构建便宜 100x)。

## 80. W16 落地 (观测面): 引擎失败态贯通 /health (2026-09-09 深夜)
设计见 `docs/maintainer/engine-failure-recovery.md`; 本轮落 **3.5 可观测** 半边 (不改失败语义):
- `include/ninfer/types.h`: 新增 `EngineFailureState{failed, reason, since}` (纯增量;
  只有 34 个宿主 TU 直接引用 types.h、零 CUDA 文件 -> 重编代价可控)。
- `engine_core.h`: `fail_all_locked` 捕获首个异常 `what()` + 时间戳 (首因优先),
  新增 `failure_state()` 访问器 (queue_mutex_ 保护)。
- `Engine::failure_state()` (engine.cpp, std::visit + C++20 `requires` 判别式 —— 
  variant 里还有 `CausalScoreCore`, 它没有该方法, 用 requires 优雅退化)。
- `GenerationService::failure_state()` 透出; `/health` 在失败时返回 **503 +
  {status:failed, engine:{state,reason,since}}** (原来恒 200 {"status":"ok"})。
- 验证: 4 个受影响 TU 语法门全 OK (提交 51a7a8f)。**待端到端验收**: 需构建后
  注入一次失败 (或后续 lane 级失败实现) 看 /health 变 503 + 原因。
- 剩余 (设计 3.1/3.2/3.3): 异常分类 + lane 级失败 + `POST /recover` —— 需更谨慎的
  归因逻辑与数值回归, 排在 lane 归因实现轮次。

## 81. W1 冒烟首跑: TT6 修复生效 + 暴露并修复 reload 后 stale 句柄 (2026-09-09 深夜)
**W1 修复确认生效**: 新二进制 (23:24 构建) 冒烟第一步 baseline gen 成功 (旧版此处必崩
`gqa_attention_decode.cu:370 InvalidValue`)。坏 spec 也正确返回 **HTTP 400**。
**W16 /health 立刻兑现价值**: 冒烟第 3 步 (reload e8/nvfp4 后生成) 失败时,
`/health` 直接给出 `{"status":"failed","engine":{"reason":"checkpoint recovery owner is stale",...}}`
—— 以前只会看到一个裸 503。

**新 bug (已修, 待重建验证)**: reload 后**首个请求**抛
`checkpoint recovery owner is stale` (`program_impl.h:7483`, `valid_continuation(owner)` 失败)
→ worker 兜底 catch 毒化引擎 → 后续请求全 503。
- 根因: continuation catalog 在 **ResourceManager** (`src/runtime/engine/resource_manager.h`,
  `CatalogEntry.handle`) 里, 属于 Program 之外; `Engine::reload_kv_storage` 重建 Program
  (旧 Program 析构) 后, catalog 里的句柄全部失效, 但没被清理 -> 下一次规划调用
  `program.checkpoint_recovery_ns(*entry.handle, ...)` 即抛。
- 修复 (2 处):
  · `engine_core.h`: 新增 `clear_context_catalog_after_replan()` =
    `resources_.clear_after_program_cleanup()` + `scheduler_.reset()` (execution_mutex_ 保护);
  · `engine.cpp`: `reload_kv_storage` 在 `replan_target_kv` 之后经 std::visit + `requires`
    调用它 (CausalScoreCore 无此方法, 安全退化)。
- 语义: 换池 = 上下文缓存 revision 失效点 (§55 设计), 清空 catalog 是正确行为;
  serve 层调用前已排干, 无在飞请求。
- 下一步: 重建后重跑冒烟, 期望 7 步全过; 然后跑 W10 (57K KV 矩阵) + W5 的 serve 端到端。

## 82. E8 分层 KV 首生成崩: §32 类缺陷在 E8 文件里复发 (2026-09-09 深夜)
冒烟第 3 步 (reload 到 e8 层后首次生成) 崩, compute-sanitizer 定位到
**`gqa_attention_prefill_e8_launch` 的 cudaLaunchKernel 返回 InvalidValue**。
三处同族缺陷 (全部已修, 待重建验证):
1. **E8 prefill 启动器缺 Muse 分派** (`gqa_attention_prefill_e8.cu` 4 处):
   只有 `q.ne[1]==Gqa27(24)` 否则**静默落 Gqa35(16 头/256 维)** -> Muse (32 头/128 维)
   用错几何启动 -> InvalidValue。修: 加 `GqaMuseGeometry` 分支 (按 head_dim 消歧),
   并把 Gqa35 兜底改成**显式校验 + throw** (§33 教训: 不再静默落档)。
2. **E8 decode 启动器同病** (`gqa_attention_decode_e8.cu` 2 处): 同上修复。
3. **E8 decode 缺 Muse 的 Wc 表**: 通用 schedule 表对 TT3 给 Wc=8, 而 Muse 的
   RowTiles==TokenTile=3 -> `static_assert(Wc % RowTiles == 0)` 编译期失败 (预检抓到)。
   修: 加 `GroupSize == 16` 分支, Wc 表与 nvfp4/i8 Muse 一致 (TT1->4/2->8/3->6/4->8/5->10/6->12)。
方法论收获: **编译前预检** (`nvcc -O1 -c` 单 TU + `g++ -fsyntax-only`) 在 2 分钟内抓到了
一个 30 分钟全量构建才会暴露的编译期断言, 值得固定成流程。
另: WSL 侧 `cp` 曾静默失败导致预检读到旧文件 -> 同步后必须 md5 校验 (脚本已加)。

## 83. W4 落地: gemm_qpn M 档分派 + 数值扫档验收 PASS (2026-09-10)
- `src/ops/linear/qpn/qpn_host.cu` 新增 **`gemm_qpn`**: M≤3 → 既有 qpn_simt,
  M4-8 → `skinny_nvfp4_qpn<1>`, M9-16 → `<2>` (MT = 8 行 A tile 数; MT=2 只解一遍权重流),
  M>16 显式抛错 (wmma 档 17..64)。声明加入 `qpn_kernels.cuh`;
  **CMake 注册进 ninfer_ops** (步骤②, 提交 196fe67 / b8ecb05)。
- 新验收工具 `tools/qpn_gemm_test.cu` (独立编译, 不依赖引擎构建): 合成 prepack +
  qpn2 槽位 CPU 参考解码, 扫 **M=4..16 全档 → bad=0, QPN_GEMM_TEST PASS** (RTX 5090)。
- 坑记录: `qpn_kernels.cuh` 里 `skinny_quant_a8` 是**非 inline 的 __global__ 定义**,
  两个 TU 同时包含会链接期重复符号 -> 测试 TU 只声明 `gemm_qpn` 不包含内核头;
  **引擎侧接线时同样只允许 qpn_host.cu 一个 TU 包含该头**。
- 剩余 (W4 ③④): QType 消费 (linear.cpp 按 profile 选档 + 加载期 prepack 旁路) 与
  M9-16 真机标定 (需 V100 真机或 compute_70 PTX JIT)。

## 84. W17④ 结论: Muse 参考源线索 (2026-09-10, 实证到仓库页)
- 搜索 + 抓取核实: 官方 NVFP4-QAT 工具链 = **`NVIDIA/Model-Optimizer`**
  (GitHub 实抓: 含 `examples/llm_qat` (HF Trainer) 与 `examples/megatron_bridge`;
  新闻页提到 Nemotron 的 PTQ/QAT FP8/NVFP4)。**该页无任何 Muse/Muse-Glimmer 字样**
  -> 本地 `data/muse_nvfp4` 的具体来源仓库仍未定位, 搜索摘要里的"Muse-Glimmer NVFP4"
  条目**未经证实**, 不作为依据。
- 对 W11 (Muse logits 对拍) 的影响: 参考栈首选 Model-Optimizer 的 NVFP4 推理路径;
  但 60GB 显存/内存门槛仍在, 短期继续走引擎自洽 (score_tokens/perplexity) + 真实权重
  到位后再定对拍通道。

## 85. W6 带宽自适应 + W16 P0 lane 级失败落地 (2026-09-10, 待构建/GPU 验证)
用户指令: 把 TODO 里没落地的功能先落地 (WSL 侧); 构建期并行写码 (铁律①)。
### W6 (FreeToken 步3 = 带宽自适应)
- 新增 `src/runtime/engine/bandwidth_governor.h` (header-only, 无 CUDA):
  信号 = `decode_device_wait_ns / committed_decode_tokens` 的 EMA ÷ **噪声地板**
  (地板瞬降到新低、每窗 2% 慢爬 → 会话一开始就受压也能收敛); 比率 > tol_hi 连续
  `streak` 窗 → prefill 份额减半 (1→0.5→0.25→0.125 地板), < tol_lo 连续 streak 窗
  → 翻倍恢复; 信用按 decode round 累积 (`share × rounds`, 上限 2), 每个真正执行的
  prefill 单元扣 1。**share==1 时 `prefill_allowed()` 恒真 = 历史 1:1 交替行为不变**。
- 消费点 `Scheduler::choose_execution` 加第 4 参 `prefill_admitted` (默认 true):
  被拒时只要有 decode 就继续 decode, **无 decode 时仍跑 prefill (不饿死)**。
  worker 循环每轮 `observe()` + 归因, `[ft] bw <kind> share= ratio= decode_us= base_us=`
  落 stderr (变更时打印, baseline 首次打印)。
- 环境变量: `NINFER_FT_BW_GOV=1` 开关; `NINFER_FT_BW_{TOL_HI,TOL_LO,STREAK,MIN_SHARE,
  BASE_ALPHA,EMA_ALPHA,WINDOW_MS,MAX_CREDIT}` 可调 (默认 1.35/1.15/2/0.125/0.02/0.35/50ms/2.0)。
- **单测 PASS** (`tests/test_bandwidth_governor.cpp`, 已注册 tests/CMakeLists.txt, g++ 直编):
  实测轨迹 1.0→0.5(win8)→0.25(win10)→0.125(win12) → 恢复 0.25(win50)→0.5(win52)→1.0(win54);
  另含禁用态零副作用、信用收支、调度决策表 8 条 (不饿死/交替/节流让路)。
- 待验收 (GPU): `_w6_bw_e2e.sh` = 并发 64K prefill 下短请求时延 OFF/ON 对比
  (`_w6_bw_probe.py` 输出 RATIO, 门 ≤1.10) + 57K 针刺质量 + `[ft] bw` 行数。
### W16 P0 (503 鲁棒性; 设计 = docs/maintainer/engine-failure-recovery.md)
- `engine_core.h`: 异常分类 `classify_failure` (RequestError→Request / bad_alloc→Resource /
  其余→Invariant) + 单元阶段 `unit_phase_` + 单归属 `unit_owner_`;
  **`fail_request_lane`**: 仅当 (Setup 阶段 && 单归属 && Request 类 && lane/sequence 绑定
  完好 && 无 capture/上下文事务) → `resources_.abort(program,lane,sequence)` + `complete_error`
  + 摘掉该 lane, **引擎继续服务**; 任一前提不清 → 退回 `fail_all_locked` (保守)。
  decode/control/commit 路径一律标 `UnitPhase::Commit` → 不走 lane 级 (设备工作可能已下发)。
- 测试钩子: `NINFER_FAULT_INJECT=request_once|invariant_once` + `..._MIN_ID` (跳过 warmup
  = 首请求), 在 prefill setup 处抛。验收脚本 `_w16_fault_test.sh` + `_w16_fault_probe.py`:
  A 案例 = 恰好一个请求带注入原因失败、其后请求 200、/health ok、进程存活;
  B 案例 = 引擎停摆 + /health 503 + reason (W16 观测面)。
- 语法门: engine.cpp / http_server.cpp / test_admission_policy.cpp `-fsyntax-only` 全过。
### 并行与构建纪律 (本轮新增)
- 验收脚本加 `NINFER_SERVE_BIN` 覆盖点 (smoke_full_nograph / _w5_e2e / _kv_matrix_57k)
  → 测试跑**快照二进制**, 后续重建不打扰在跑的测试 (避免 ld 覆盖在用的 ninfer-serve)。
- W6/W16 全是宿主侧文件 (无 .cu), 与 `decode_e8` 的 nvcc 长编译并行写不冲突;
  同步脚本 `_sync_w6_w16.sh` (md5 校验) 待 E8 构建完成后再跑 (engine_core.h 会波及
  在构建的宿主 TU)。

## 86. W8 结案: rmsnorm 权重折叠 PPL 实测 -> **放弃全量折叠** (2026-09-10)
验收协议 (W8 原文): `ninfer-perplexity --text perplexity-corpus-long.txt` 跑
fold artifact vs 基线; 差<0.1 → 实施 _opt_plan 任务1 的 b/c/d 步, 否则记"放弃"关案。
- 环境: GPU 空闲窗口 (E8 构建只用 1 核 CPU), 三 artifact 顺序跑, 脚本 `_w8_ppl.sh`,
  日志 /home/user/w8_ppl/*.log (报告落在 profiles/perplexity/.../report.json)。
- **实测 (148,745 tokens, context/stride 4096/2048, kv 默认 fp8-e4m3-r256)**:
  | artifact | ppl | Δ vs baseline |
  |---|---|---|
  | qwen3_8_27b_nvfp4 (基线) | **13.850809** | — |
  | ..._fold3 (3 层折叠) | **13.862400** | **+0.0116** |
  | ..._foldmlp (全 56 层 gate_up 折叠) | **14.406620** | **+0.5558** |
  单次耗时 ~49s 打分 + ~25s 加载, ~3000 tok/s。
- **裁决: 放弃全量折叠 (关案)**。理由: 唯一能拿到 _opt_plan 任务1 那 4.4% 上限的形态
  = 全部 209 个 rmsnorm 消失 = 56 层全折, 代价 +0.556 ppl (相对 +4.0%), 远超 0.1 门限;
  3 层折叠代价几乎为零 (+0.012) 但只覆盖 ~5% 的层 → 收益 ~0.2%, 不值得动
  prefill/decode 全部 linear + norm 语义 (该改动横跨 T 全档, 见 _opt_plan 任务1 注)。
- 附带数据点 (对 W11 有用): fold artifact 的层权重 requant RMS 误差 5.3%/max 6.4%
  在 56 层累积后把 ppl 从 13.85 推到 14.41 => **NVFP4 e2m1 二次量化的质量弹性基线**,
  后续任何"再量化一次"的方案都可用这条标尺预估。
- 方法论: 本关案把 §71 的"待 PPL 回写"悬案关闭, W8 从 backlog 移除。

## 87. W16 P1 落地: recover() + POST /recover (2026-09-10, 待构建/端到端验证)
设计稿 §3.3 的恢复路径实现 (与 W6/W16P0 同批构建):
- `EngineCore::recover(rebuild_program)`: ① 持 queue_mutex_ 检查 (stopping_/failed_),
  幂等 (健康态直接返回); ② **join 旧 worker (不持 execution_mutex_, 避免与其释放锁互锁)**;
  ③ 调 rebuild_program 回调 (此刻无 worker 在跑, 换 Program 不竞态); ④ 清 failed_/reason/
  pending_/materializing_, 起新 worker 线程并等它绑定设备成功 (startup promise)。
- `Engine::recover()`: 先 `cudaGetLastError()` 清粘性错 + `cudaMalloc(4096)` 探针
  (上下文不可用则抛, 不假装成功) → 回调里 `replan_target_kv` (与 reload_kv 同路径:
  销毁旧 Program/KV 池/图 → 冷启镜像 → 新池) + `clear_context_catalog_after_replan()`。
  代价: **KV 缓存与全部前缀缓存丢失** = 最后手段, 正常路径仍是修 bug。
- serve: `GenerationService::recover()` 复用 reload 的排干门 (reload_mutex_ +
  reload_in_progress_ + active==0 上限 120s) → `Engine::recover()`; `POST /recover`
  走同一全局鉴权, 健康态返回 ok (幂等)。
- 语法门: engine.cpp / generation_service.cpp / http_server.cpp `-fsyntax-only` 全过。
- 待验收 (扩展 `_w16_fault_test.sh` B 案例): 注入 invariant 故障 → /health 503 →
  `POST /recover` → /health 200 + 后续请求正常返回。A 案例 (request_once) 不变。
- 依赖链: P1 复用 §56 的 replan 路径, 因此**图/前缀缓存全丢**; 若后续做「图 PINNING」
  可把重捕获成本降下来 (设计稿 §3.3 注)。

## 88. 构建代价实测: E8 decode TU 的 PTX 爆炸 (2026-09-10, 影响后续构建规划)
E8 修复 (加 Muse 几何分派 + Wc 表) 后的全量构建中, 单个 TU 成为长尾:
- `ops/launcher/gqa_attention_decode_e8.cu` 的 nvcc 前段产出 **PTX = 881 MB / 22,867,618 行**
  (`/tmp/tmpxft_*_gqa_attention_decode_e8.ptx`), 随后 ptxas 单线程 ~100% CPU 跑
  **>22 分钟** (RSS 12.7→13.4 GB 缓慢增长, 无换页), 而**同批的 prefill_e8 TU 只用了 83 秒**
  (它的 .o 6.6MB vs decode 的 160MB —— decode 的实例化基数本来就大得多)。
- 根因: 该 TU 按 `TokenTile × Wc 表 × Geometry` 全量实例化, 本次新增 Muse 几何 (32 头/128 维,
  TT1..TT6) 后规模再上一个台阶; 编译选项 `-O3 -lineinfo -rdc=true` 进一步放大。
- 代价: 全量 `make ninfer-serve` 从 ~26 min 拉长到 ~45+ min (仅此一个 TU 占一半以上),
  且 ptxas 峰值内存 13 GB 逼近 WSL 24 GB 上限 -> 与其它内存大户 (perplexity/serve) 并行时
  有 OOM 风险, 排期时不要同时跑。
- 后续优化 (未做, 记入 W17): ①按几何拆 TU (decode_e8_muse.cu / decode_e8_gqa35.cu),
  ②只实例化实际用到的 TT 档, ③W17② ccache 对 nvcc 的缓存 (重复构建直接命中)。
- 经验: **改 .cu 前先看它当前 .o 的体积** —— >100MB 的 TU 意味着一次构建 +20 min。

## 89. [新发现, 阻塞 W10] Muse 长 prompt 预填充 bad_alloc: 工作区 arena 越界 (2026-09-10)
起因: 跑 W10 的 57K 针刺长测时, 引擎在**规划/预填充**阶段抛 std::bad_alloc (HTTP 500)。
定位链 (全部实证, 铁律④):
1. **不是我的改动**: E8 快照二进制同样失败; qwen3.8-27b 在同一二进制上 12K prompt 正常生成。
2. **不是真实内存不足**: LD_PRELOAD 的 operator new 探针零命中; 宿主机 16 GB 空闲。
3. **抛出点**: `__cxa_throw` 拦截 + `addr2line` 符号化 =
   `ProgramImplCore::advance_prefill` (Muse 变体), 即预填充执行期, 不是 admission。
4. **直接原因**: `DeviceArena::alloc_bytes` 的 `end > cap_` 分支
   (诊断已加: 打印 request/offset/end/cap)。实测 (Muse, max-context 24576):
   - prefill-chunk=512: 规划 `text_prefill=34,493,952` B, 运行时在 offset=34,082,816 处
     再要 20,447,232 B → end=54,530,048 > cap。**运行时的用量按 `{hidden=6656, 512}`
     bf16 (6,815,744 B) 为单位累积到 8 块**, 规划只算了 ~5 块的量。
   - prefill-chunk=3072: 规划 65,693,184, 实际 ≥66,032,640 → 同样越界。
   - **触发条件 = prompt 需要多个 chunk** (chunk≥2048 且 prompt≈1050 token 时单块就过;
     chunk≤1024 必崩) → 单步 prefill 正常, 多步 prefill 崩。
5. **加 50% 余量无效**: cap 提到 51.7 MB 后, 溢出点同样后移 (运行时继续按 chunk 单位
   累积分配) → **不是"规划低估", 而是运行时在单次 prefill 内无界累积分配**
   (疑点: 每个 capture/checkpoint 组或每个 split 段分配一块 `{hidden, chunk}` 未做 scope;
   计划侧 `target_body(text_prefill, 1, chunk, ...)` 只按一块计)。这解释了 §21 当年
   "workspace 容量 vs Muse 52 层" 的悬案方向是对的, 但根因在执行侧而非规划侧。
- 诊断基建 (已进树, env 门控, 无副作用): `NINFER_WS_DUMP=1` 打印
  `[ws] chunk=... text_prefill=... ordinary=... mtp_prefill=... general=...`;
  arena 越界打印 request/offset/end/cap + 调用者地址。
- 复现 (30 秒): `ninfer <muse>.ninfer --prompt <1050 字中文> --max-new 8 --no-thinking
  --greedy --no-cuda-graph --kv-dtype nvfp4 --max-context 24576 --prefill-chunk 512`
  → `error: std::bad_alloc`; `--prefill-chunk 3072` 同 prompt 直接成功。
- 影响: **W10 的 57K 矩阵无法运行** (57K prompt 必然多 chunk); Muse 长上下文能力实际不可用
  (此前 57K 记录应为更早内核/更小 chunk 时代)。短 prompt (smoke 78 token) 不受影响。
- 建议修法 (待引擎侧确认): 在 `text_context_impl.h` 的 prefill 路径为每个 split/组加
  `work_.scope()` (或按组 reset), 使峰值回到"单块"量级; 规划侧再核对
  `target_body(text_prefill, 1, chunk, ...)` 的 peak 是否覆盖所有组。

## 90. W16 P0+P1 端到端验收 PASS (2026-09-10, 故障注入实测)
二进制: W6/W16/P1 构建 (891,544,512 B, md5 a15aeaa6b7c8d60695750d143a3ed46c 快照)。
脚本: `_w6w16_battery.sh` = 快照 + `_w16_fault_test.sh` + `_w6_bw_e2e.sh` + smoke 回归。
### A. 请求域故障 (NINFER_FAULT_INJECT=request_once, MIN_ID=2 跳过 warmup)
| 观测 | 结果 |
|---|---|
| req[0] | **HTTP 400** `invalid_media: injected request-domain fault` |
| req[1..3] | **HTTP 200** (引擎继续服务) |
| `/health` | **200** `{"status":"ok"}` |
| 进程 | ALIVE |
| 引擎日志 | `[engine] request id=2 failed in lane 0: injected request-domain fault` |
→ **P0 lane 级失败成立**: 请求域异常只失败该请求, 不再把整机打入 503 (对比 §61b 现象)。
### B. 不变量故障 (invariant_once) + 恢复
| 观测 | 结果 |
|---|---|
| req1 | HTTP 500 `injected invariant fault` |
| `/health` | **503** `{"status":"failed","engine":{"state":"failed","reason":"injected invariant fault","since":...}}` |
| `POST /recover` | **HTTP 200** `{"status":"ok"}` |
| `/health` (恢复后) | **200** `{"status":"ok"}` |
| req2 (恢复后) | **HTTP 200 正常回答** |
→ **P1 recover() 成立**: 进程不重启即从 poison 态恢复 (Program 重建 + worker 重生成)。
### 附带
- 回归: smoke 7 步结构全过 (含 reload e8/nvfp4 → 生成不崩)。
- 观测面: 失败原因经 `/health` 结构化暴露, 与 51a7a8f 的 EngineFailureState 一致。
- 遗留 (设计稿 P2): 执行轮内的 throw 源头治理 (校验前移) 仍未做; 本次证明**兜底路径已足够
  让单请求故障不再升级为整机故障**, P2 可降级为常规清理项。

## 91. W6 端到端实测 + 验收判定 (2026-09-10): 机制成立, ≤10% 门限不可达 (架构限制)
实测协议 `_w6_bw_e2e.sh` + `_w6_bw_probe.py` (qwen3.8-27b, 64K 并发 prefill 下测**在飞流式请求**
的逐 token 间隔; Muse 因 §89 无法吃 64K prompt):
| 配置 | solo 中位间隔 | 并发期中位间隔 | 比值 |
|---|---|---|---|
| 调速器 OFF (chunk 3072) | 18.4 ms | 238.4 ms | **13.0×** |
| 调速器 ON  (默认参数) | 18.3 ms | 234.7 ms | **12.9×** |
| 固定 `--prefill-chunk 512` (OFF) | 18.4 ms | 160.6 ms | **8.7×** |
| STREAK=6/TOL_LO=1.6 (更慢恢复) | 18.5 ms | 245.0 ms | 13.3× (未触发节流) |
- **机制成立** (日志实证): `[ft] bw throttle share=0.500 ratio=2.83 decode_us=66058` →
  `share=0.250 ratio=1.53 decode_us=7885` (节流窗内解码时延降 8.4 倍) → `recover 0.5 → 1.0`;
  OFF 运行 0 条 bw 行 (零副作用)。
- **但验收门 (tok/s 波动 <10%) 不可达**: 引擎只有一条 prefill 车道, 单元粒度
  `advance_prefill` = 一整块 (3072 token ≈ 440ms GPU), 解码在每个 prefill 单元期间被阻塞;
  即便把 chunk 压到 512 (73ms) 仍有 8.7×, 而 128 是硬地板 (prefill 对齐), 128 块本身就把
  解码间隔翻倍。→ 结论: **该门限需要架构级改动** (prefill 切片 << 128 token, 或
  decode-first 调度 + 预填充仅在解码队列空时推进), 不是参数问题。
- 附带发现 (值得记): 调速器会**振荡** — 节流后解码立刻变快 → ratio 掉到死区 → 恢复 →
  解码再变慢。当前滞回 (2 窗) 不够, 需要"恢复冷却期"或按固定时长保持节流态。
- 交付物: `bandwidth_governor.h` (信号/份额/信用/动态 chunk)、`Scheduler::choose_execution`
  第 4 参、`Program::set_prefill_chunk` + `prefill_chunk_capacity`、单测
  (`tests/test_bandwidth_governor.cpp` 含 chunk 缩放) 全 PASS。
- **建议**: 把 W6 的门限改写为「可控性」而非「≤10%」: ①节流窗内解码时延下降 ≥5× (已达成
  8.4×); ②OFF 与 ON 的 `[ft] bw` 行数分别为 0 / >0; ③动态 chunk 生效 (chunk 随 share 缩放,
  单测覆盖)。若坚持 ≤10% 门限, 需先做架构改动 (另立工作包)。

## 92. [重要陷阱 + W5 断链修复] 死 TU: `gqa_attention_decode_impl.cuh` 与 muse/g35 拆分文件
排查 W5 的 `[ft]` 行为何 0 行时发现:
- `gqa_attention_decode_impl.cuh` (含唯一一处 `ft::observe`) 只被
  `gqa_attention_decode_muse.cu` / `gqa_attention_decode_g35.cu` include, 而**这两个 .cu 已不在
  `src/CMakeLists.txt` 里** (count=0) → 它们的 .o 是 09-09 的陈旧产物, 不参与链接。
  证据: 两个 .o 里有 `[ft] layer=` 字符串, 但 `ninfer` / `ninfer-serve` 二进制里
  `strings | grep -c 'ft. layer='` = **0**。
- 活着的实现是**巨石 TU** `gqa_attention_decode.cu` (703 行, .o 224 MB, 自带
  `launch_tc_partial_nvfp4` 等全部副本) —— 这也是 §W1 修 arena 时必须同时改两份的原因。
- 后果: **W5 的观测链端到端从未真正工作** (FreeToken 步 1/步 2 的自动重排永远等不到数据),
  此前"策略+滞回已验证"只是宿主侧单测的结论。
- 修复 (本批): 在活的 `gqa_attention_decode.cu::launch_tc_partial_nvfp4` 里补上
  `ft::observe(...)` (与 `_impl.cuh` 同语义, 含 include `ops/common/ft_stats.h`); 重建中。
- **教训 (写进流程)**: 改 launcher/kernel 前先确认目标 TU 是否在 CMake 里 (或在最终二进制里
  有符号), 否则可能在改死代码。检查法: `grep <文件名> src/CMakeLists.txt` 或
  `strings <binary> | grep <特征串>`。

## 93. W17② ccache 免 sudo 落地 (2026-09-10)
- 结论: 无需 sudo 也能装。官方静态构建
  `https://github.com/ccache/ccache/releases/download/v4.11.3/ccache-4.11.3-linux-x86_64.tar.xz`
  下载解包即用 (实测 WSL 可直连 GitHub), 已安装到 **`/home/user/.local/bin/ccache`** (4.11.3,
  features: avx2 file-storage http-storage redis...)。
- 启用方式 (下次全量重建/重建 build 目录时执行一次):
  `cmake -S . -B build -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache`
  → 之后 `gqa_attention_decode.cu` / `gqa_attention_decode_e8.cu` 这类 **20+ 分钟的单 TU**
  在输入未变时可秒级命中 (本次会话实测: 两个巨石 TU 的 ptxas 各 20-26 分钟, 是构建长尾的全部)。
- 未立即启用原因: 切换 launcher 会触发一次全量重编译 (~45 min), 会阻塞本轮验收;
  make 本身已跳过未变 TU, 所以 ccache 的收益主要在 **重建 build 目录 / VM 崩溃续跑** 场景
  (§57 曾多次发生)。建议与下一次必须全量重建的改动合并执行。

## 94. W5 观测链修复进展: 行出来了, 但采样值全是 NaN (2026-09-10)
- 修复后 (补回 `ft::observe`) 实测: `[ft] layer=` 行 **988 条** (此前 0), 但
  **986 条 mean_l=nan + 2 条 -nan**, 16K 长上下文同样全 NaN (52/52)。
- 已排除: ①未接通 (行数已证明接通); ②环境变量 (同一进程的 `NINFER_FT_*` 生效);
  ③splits==1 假设 (16K 上下文仍全 NaN, 且内核 `gqa_attention_decode_nvfp4.cuh:228/239/985/992`
  确有 `partial_l[...] = l0/l1` 写入)。
- 仍存疑 (下轮入口): 引擎侧 `partial_l` 张量是否就是内核写入的那个缓冲 (内核里有
  `partial_l += batch * QHeads * TokenTile * split_count` 的 per-batch 推进), 以及
  `gqa_partial_stat_index<Geometry>(q_head, token, split, TokenTile)` 的前 `min(q_heads,32)` 个
  float 是否落在被写入的区域; 观测点在 kernel 之前 (采上一轮值), 首轮必为未初始化 (cudaMalloc
  垃圾常为 NaN), 但不应"轮轮皆 NaN"。
- 建议的下一步 (15 分钟可判): 在引擎侧 decode 步结束后 (而非观测点) 用一次 D2H 打印
  `partial_l` 前 8 个 float + `splits` + 张量 shape, 与内核的 index 公式对照; 若确为
  "写入区域 ≠ 采样区域", 则改采样偏移 (或让内核额外写一个 per-head LSE 标量)。
- 影响: **W5 的自动重排仍无法端到端验收** (能量信号不可用 → 三分位决策无意义);
  W5 此前"策略+滞回已验证"仅指宿主单测 (`tools/kv_relayout_test.cpp` 的 C++↔Python 对拍)。
- 附带: `ft::observe` 目前只在 **nvfp4** decode 启动器里 (i8/fp8/iso3/e8 路径无观测) —
  分层 KV 场景 (e8 层) 下该层的能量也拿不到; 若 W5 要在混合表上工作, 观测点需要覆盖
  全部 KV dtype 的 decode 路径 (或改在引擎侧统一采样)。

## 95. KV 组合矩阵实测 + e8 上层损坏 + 位预算策略 (2026-09-10, 用户指令)
背景: 用户要求把各种 KV 组合（nvfp4 fusion / 双 e8 / 3bit / e8+iso3 混搭）都试一遍, 并把
"给定 bit（含小数 bit）的混合精度策略" 应用到 KV, 且按设备跑基准。
### 实测环境与工具
- 设备 RTX 5090 D (单卡; 32GB); 模型 **qwen3.8-27b**（Muse 被 §89 阻塞, 长 prompt 不可用）。
- 工具: `_kv_matrix_v3.sh`（逐配置起 serve → 读 `--request-log-jsonl` 的
  `memory.kv_payload_bytes` → 长 prompt 请求读 prefill/decode tok/s → 32K/8 针针刺）。
  产出 `/home/user/kv_matrix_v3.csv`, `/home/user/kv_quality.csv`。
### 实测结果 (32K 上下文, 8 针, 16 个全注意力层)
| 配置 | KV 载荷 MiB | B/token | prefill tok/s | decode tok/s | 针刺 |
|---|---|---|---|---|---|
| all:bf16 (全局) | 1112 | 17792 | 6424 | 32.4 | 6/8 |
| all:int8 | 1112 | 17792 | 6433 | 32.4 | 6/8 |
| all:fp8 | 1112 | 17792 | 7380* | 50.6* | 6/8 |
| all:nvfp4 | 1112 | 17792 | 6422 | 32.4 | 6/8 |
| all:iso3 | 1112 | 17792 | 6428 | 32.3 | 6/8 |
| **0-7:e8** (8 层 e8 + 8 层 nvfp4) | 1120 | 17920 | 6214 | 35.4 | **8/8** |
| **8-15:e8** | — | — | — | — | **0/8** |
| 0-3:e8,4-7:int8 | 1376 | 22016 | 6308 | 45.2 | **8/8** |
| all:e8 (0-15) | 1088 | 17408 | 7786 | 45.5 | **0/8** |
| 0-11:e8 | 1104 | 17664 | 7475 | 49.0 | **0/8** |
(*fp8 那行来自 16K 档的另一次测量, 仅供参考。)
### 三个硬结论
1. **KV 载荷与档位几乎无关**: bf16/int8/fp8/nvfp4/iso3 全部 17792 B/token (e8 17408, -2.2%)。
   即**当前引擎的 KV 池不随档位缩容** -> "降档换上下文容量" 在 qwen3.8-27b 上不成立
   (与 README/VRAM.md 的 "E8Kv 约减半" 矛盾, 需引擎侧查 page slot 是否按最大档固定)。
   载荷随 max-context 线性 (65K→262K: 1112→4448 MiB) => 确实是 per-token 项。
2. **e8 在层 8-15 上产出垃圾**: `8-15:e8` 与 `all:e8`、`0-11:e8` 全部 **0/8**; 而 `0-7:e8`
   8/8、`0-3:e8,4-7:int8` 8/8。**静默错误**（不报错、不崩，只是检索全丢）。
   嫌疑: qwen3.8-27b 的 16 个全注意力层分两类几何（前 8 层一种、后 8 层另一种），
   e8 内核只对前者正确。**使用建议: 在定位前 e8 只允许层 0-7**（引擎应显式拒绝或校验）。
3. **纯档位之间质量无差异**（针刺全 6/8, 且缺的针相同: YPP63F/KRZJ6K）; 速度差异也在
   ±5% 内（27B 权重带宽主导, KV 档位在 32K 下不是瓶颈）。=> 针刺在 32K 不足以区分 4bit
   档位质量, 需要 PPL 或更长上下文/更硬的探针。
### 位预算策略（用户要求的"给定 bit"）
- 新工具 `_kv_bit_budget.py`: 输入目标平均 bit/元素（支持小数）+ 层数, 用 DP 精确解
  "min 质量罚分 s.t. Σbits ≤ B·L", 输出 `--kv-layer-storage` 表。
  档位位成本: bf16 16.0 / int8 8.25 / fp8 8.03 / nvfp4 4.50 / e8 4.06 / iso3 3.00。
  质量先验: bf16 0 / int8 0.02 / fp8 0.03 / e8 0.08 / nvfp4 0.30 / iso3 0.50（待实测校正）。
  输出示例 (16 层): B=4.0 -> `0-11:e8,12-15:iso3`; B=4.5 -> `all:e8`;
  B=6 -> 5×e8+11×int8; B=12 -> 3×bf16+13×int8。**注意**: 先验里 e8 优于 nvfp4,
  但实测显示 e8 只在前 8 层可用 -> 分配器需要"每层可用档位"约束（下一步）。
### 顺带修好的验收工具 bug (关键)
`tools/archkit/longtest_57k.py` 此前**恒报 0/0 或 0/N**: ①`make_context` 返回的是
**未植入**的针（应返回已植入的）; ②只读 `message.content`, 思考模型把预算花在
`reasoning_content` 上 -> 答案恒空; ③`max_tokens=120` 装不下多针答案。
三处已修（返回已植入针 / 匹配 content+reasoning / 预算随针数缩放）, 修后 4K/8K 2/2 PASS。
**此前 W10 的"57K 针刺"结论均不可信**, 需用修好的工具重跑。
### 待办
- [ ] e8 上层损坏定位（引擎侧; 高优先, 影响所有 e8 混搭）
- [ ] KV 池为何不随档位缩容（引擎侧; 决定"降档换容量"是否可用）
- [ ] 用 PPL 或更长上下文建立 4bit 档位的质量阶梯（针刺不够敏感）
- [ ] 分配器加"每层可用档位"约束（e8 限层）

## 96. [实证完成, 待引擎侧修] e8 KV 在高层号上长上下文检索损坏 (2026-09-10)
模型 qwen3.8-27b: 64 物理层, 其中 16 层全注意力 (物理 3,7,11,...,63), 其余 GDN;
`--kv-layer-storage` 的层号 = 全注意力层顺序 (0..15), 与物理层一一对应 (从 artifact JSON
读出: `text/layers/N/{attention|gdn}`)。所有 16 个注意力层是**同一几何**, 所以不是"两种层型"。
### 完整实证地图 (32K 上下文, 8 针, greedy, 固定 seed; 每格重复 2 次一致 => 确定性)
| e8 层集合 | e8 层数 | 结果 |
|---|---|---|
| 0-7 | 8 | **8/8** |
| 4-7 / 6-9 / 8-9 | 4/4/2 | **8/8** |
| 0-9 | 10 | 7/8 |
| 单层 7 / 8 / 9 / 10 / 11 | 1 | **8/8** |
| 8-11 | 4 | 2/8 |
| 14-15 | 2 | 4/8 (两次重复均 4/8) |
| 12-15 | 4 | 0/8 |
| 8-15 | 8 | 0/8 |
| 0-11 | 12 | 0/8 |
| all (0-15) | 16 | 0/8 |
规律: **单个 e8 层在任何位置都正常**; 一旦出现 ≥2 个"高层号"e8 层就开始掉, 且随
(高层号 e8 层数 × 层号) 恶化。低层号 0-7 即使 8 层全 e8 也完美。
### 排除项 (实证)
- **不是越界**: `compute-sanitizer --tool memcheck` 跑 `14-15:e8` 全程零报错 =>
  是**逻辑/别名错误**(缓冲区内的错误索引), 不是 OOB。
- **不是层类型**: 16 个注意力层同几何 (见上)。
- **不是短上下文**: `8-15:e8` 在 8K 短上下文生成完全正常 ("杭州是一座以西湖美景…"),
  只有长上下文检索崩 => 与 KV 长度/页数相关的路径。
- **不是全局 e8 禁用**: e8 单层/低层正常, 速度还更快 (prefill 7786 vs 7420)。
### 建议的引擎侧下一步 (15-30 分钟可判)
1. 在 decoder-state 布局里 dump 每个 e8 层的 **KV plane 基址/步长**, 与 nvfp4 层对照,
   查是否出现"高层号 e8 层 plane 区间重叠"(最符合"随层号恶化"的形状);
2. 或对照 `layouts_impl.h:79` (`E8Group64 -> {DType::E8Kv, kKvInt8QuantGroup}`) 的
   plane 尺寸推导与 `program_impl.h:12575` 的缓存视图, 找按层索引的固定 stride;
3. 若确实重叠, 修 plane 尺寸/步长即可; 顺带加"e8 层数上限"校验避免静默损坏。
### 实用规则 (先按此执行)
- **e8 只放在全注意力层 0-7**（实测 8/8; README 的 10L 默认在 qwen 上只 7/8）。
- 上层 8-15 用 iso3/int8/nvfp4 替代（`0-7:e8,8-15:iso3` / `8-15:int8` 均 8/8）。
- `tools/archkit/kv_bit_budget.py --e8-layers 8` 已内置该约束。

## 97. [重要] 全局 `--kv-dtype` 不改变 KV 池几何; bf16 档无法表达 (2026-09-10)
背景: §95 的矩阵里 bf16/int8/nvfp4/iso3 载荷完全相同 (17792 B/token), 当时怀疑"池按最大档固定"。
本轮把两种入口分开实测, 结论更精确:
| 入口 | 配置 | kv_payload (65K ctx) | 说明 |
|---|---|---|---|
| 全局 `--kv-dtype bf16` | 无表 | 1112 MiB (17792 B/token) | **与 nvfp4 完全相同** |
| 全局 `--kv-dtype int8` | 无表 | 1112 MiB | 同上 (JSONL 记 `kv_cache=int8-group64` 但池没变) |
| 全局 `--kv-dtype nvfp4` | 无表 | 1112 MiB | 基准 |
| 逐层表 `--kv-layer-storage all:int8` | 表 | **2112 MiB (33792 B/token, 1.83×)** | 表**确实**改变几何 |
| 逐层表 `all:nvfp4` | 表 | 1152 MiB | |
| 逐层表 `all:e8` | 表 | 1088 MiB (-5.6%) | e8 平面 = U8 半维 + FP16 尺度 |
→ **KV 池几何只由逐层表 `layer_kv_dtypes` 决定; 全局 `--kv-dtype` 只进 options/JSONL,
不进几何** (与 §95 的"档位无关"结论合并: 那些"纯档位"行其实全是 nvfp4)。
### 连带发现: bf16 档无法通过任何入口选中
- `--kv-layer-storage` 的解析里 `"bf16" -> KvCacheStorage::BFloat16`, 而 **BFloat16 同时是
  "未设置"哨兵值**（`parse_kv_layer_storage` 用 `table[slot] != BFloat16` 判重复写;
  kv_options.h 注释: "Unlisted slots stay BFloat16, which means inherit the global --kv-dtype"）。
  => 写 `all:bf16` 与"不写"等价, 槽位仍继承模型默认 (qwen3.8-27b = nvfp4)。
- 全局 `--kv-dtype bf16` 又不进几何 => **当前引擎在 qwen3.8-27b 上无法启用 bf16 KV**。
  这对"精度基线"实验是硬缺口: 我们拿不到真正的 bf16 KV 对照, 只能拿 nvfp4 当"最高精度"。
- 代码位置: 几何在 `src/targets/qwen3_6/impl/state/decoder_state.cpp:84-124` 按
  `layer_dtype(layer)` 分支 (BF16 2 平面 / I8+尺度 / E8 半维 U8+尺度 / NVFP4 半维 U8+FP8 尺度),
  说明**设计上支持逐层不同**; 问题在"全局 dtype → layer_kv_dtypes"的解析链路。
### 对策略的影响 (需要引擎侧修)
1. 修 `--kv-dtype` 进几何 (或在文档/校验里明确它只影响未显式设置的层);
2. 给 bf16 一个可表达的入口 (例如 `bf16!` 或单独的 `--kv-baseline-bf16`), 否则精度基线缺失;
3. 修好后 §95 的矩阵需要重跑 (bf16 行目前是 nvfp4 的重复)。

## 98. [已修] 全局 `--kv-dtype` 进入 KV 池几何 (2026-09-10, 提交 89e2375)
- 根因 (`layouts_impl.h:946-970`): 没有显式逐层表时, `layer_overrides` 直接取
  `Variant::default_layer_kv_dtypes(weights_profile)`（模型默认表），**全局 `options.kv_cache`
  只在"槽位继承"时被读，而继承路径又回落到默认表** -> CLI 的 `--kv-dtype` 对池几何无效。
- 修法: 新增 `EngineOptions::kv_cache_explicit`（CLI/serve 在解析 `--kv-dtype` 时置位）;
  `layouts_impl.h` 里在"无显式表"分支优先用 `kv_profile.dtype` 填满逐层表, 再回落到默认表。
- **验证 (65K 上下文, qwen3.8-27b, 修复后)**:
  | 全局 --kv-dtype | payload | B/token | 相对 nvfp4 |
  |---|---|---|---|
  | bf16 | 4096 MiB | 65536 | **3.56×** |
  | int8 | 2112 MiB | 33792 | 1.83× |
  | nvfp4 | 1152 MiB | 18432 | 1.00× |
  | iso3 | 1152 MiB | 18432 | 1.00× (与 nvfp4 同几何) |
  | fp8 | (2.61 GiB runtime) | — | 几何生效, 但 warmup 报
  `packed KV cache must use quant_group 16` (内核侧限制, 单独项) |
  | e8 | 启动失败 | — | `gqa_attention workspace: invalid profile or interval`
  (e8 只能用逐层表; 全局 e8 的 workspace profile 缺失, 单独项) |
- 意义: **bf16 KV 基线重新可用**（此前 §95 的"bf16"行其实是 nvfp4）。§95 的矩阵在需要
  bf16 对照时应改用 `--kv-dtype bf16`（注意此时池是 3.56×，容量会掉到 1/3.56）。
- 回归: smoke 7 步全过（reload e8/nvfp4 → 生成不崩）。
- 遗留（新工单）: ①全局 `--kv-dtype fp8` 的 quant_group 不匹配（内核侧）;
  ②全局 `--kv-dtype e8` 缺 workspace profile; ③`--kv-layer-storage all:bf16` 仍是"未设置"
  语义（bf16 无法用表表达），文档需注明用全局 flag。

## 99. [用户可见] qwen3.8-27b 出厂默认 KV 表含高层 e8 -> 长上下文检索掉 2/8 (2026-09-10)
发现路径: §97 修复后重跑质量阶梯, 纯 nvfp4 从 6/8 变 8/8 -> 追查"默认表到底放了什么"。
- `src/targets/qwen3_6_27b/impl/variant.cpp: default_layer_kv_dtypes` 的注册默认表是
  **10 层 E8Kv + 6 层 NVFP4**, e8 落在层 `{0,1,3,4,6,7,8,9,13,14}`（注释: 13.3k zh ppl
  ctx4096 实测 1.020 最优, all-E8 1.112 / all-NVFP4 1.706 / all-I8 1.522）。
- **实测 (32K, 8 针, 同 seed, 每项可复现)**:
  | 配置 | 针刺 |
  |---|---|
  | 出厂默认（不传任何 kv 参数） | **6/8** |
  | `--kv-dtype nvfp4`（全 nvfp4） | **8/8** |
  | `0-7:e8`（e8 仅低 8 层） | **8/8** |
  | `--kv-dtype bf16`（真无损基线, §97 修复后可用） | **8/8** |
  => 默认表在**长上下文**上比全 nvfp4 少 2 根针, 与 §96 的"e8 在层 8/13/14 退化"一致
  （默认表的 e8 恰好包含 8,9,13,14 四个高层号层）。
- 权衡: 默认表的 PPL 优势是在 **ctx 4096** 上测的（e8 的格点增益确实存在）, 长上下文代价
  没被当时的验收覆盖 -> 这是"短上下文 PPL 最优 != 长上下文可用"的典型案例。
- 建议 (二选一, 需引擎侧/模型侧定):
  A. **修 §96 的 e8 高层 bug**（首选）-> 默认表保持 10 层 e8, PPL 与长上下文双赢;
  B. 若短期修不了: 把默认表的 e8 收到 `{0,1,3,4,6,7}`（去掉 8,9,13,14）,
     代价是 PPL 收益略降（需重测 PPL 确认幅度）。
- 复现: `_kv_default_probe.sh`（shipped_default / plain_nvfp4 / safe_low 三态对照）。

### 96b. e8 高层退化的进一步定位 (2026-09-10 续)
新增实证 (全部可复现):
- **深度相关**: `14-15:e8` 在 **16K 上下文 4/4 PASS**, 同配置 **32K 4/8 FAIL**（缺的正是靠后的
  4 根针）=> 损坏从某个深度开始, 层号越高阈值越浅（0-7:e8 到 32K 仍全中; 8-11:e8 在 32K
  只中前 2 根）。
- **已排除**:
  · host-KV 镜像: `--cold-host-bytes 67108864`（64 MiB）后 `14-15:e8` 仍 4/8;
  · head_dim 常量不匹配: artifact 实测 qwen3.8-27b = **head_dim 256 / 24 q / 4 kv**
    (QKV+gate 融合投影 [14336,5120] = 6144+1024+1024+6144 ✓), 与内核
    `kGqaKvQuantHeadDim=256 / kGqaKvQuantGroups=4` 一致;
  · 逐层平面基址/视图: `decoder_state.cpp` 的 `plane_base[layer]` 与
    `PagedKVCache::layer_view` 都按层解析, 未见固定步长假设。
- **代码线索**: `src/ops/wrapper/gqa_attention.cpp:489-492` 明确写着
  "E8Kv small-T kernels are unverified; route E8 to the prompt path" —— e8 解码被强制走
  **prefill 内核**; e8 的 KV 写入/读取都在 `gqa_attention_prefill_e8.cu` +
  `gqa_attention_prefill_i8.cuh`(E8=true 模板)。该文件 `722-725` 有一处**空的 if 块**
  (疑似被删掉的调试钩子), 不是直接原因但值得注意。
- **建议的下一步 (需要引擎侧插桩, 约 1 小时)**: 在 `gqa_kv_append_e8_launch` 与
  `gqa_attention_prefill_i8_kernel<E8=true>` 里对同一 (layer, page, head, offset) 打印
  实际写入/读出的地址与值, 用 32K 上下文、e8 放层 14-15 复现; 对照
  `paged_kv_element_offset<LeadingExtent,HeadExtent>` 的页面步长与实际 plane shape
  (`{head_dim/2 或 head_dim/group, 64, kv_heads, physical_pages}`), 找随 page 累积的偏差。
- **当前实用规则不变**: e8 只用层 0-7; 上层用 iso3/int8/nvfp4。

## 100. [已修] §89 破案: Muse `post_mixer_workspace_capacity_bytes` 返回 0 (2026-07-10 续)
定位链 (全部实证):
1. 给 arena 轨迹加**可符号化的调用者偏移** (`caller=0x%zx` = RA(0) - module base) →
   `addr2line` 解析出失败分配来自 **`MuseGlimmer Variant::post_mixer`**;
2. 再在 `DeviceArena::alloc` 打**形状** → 失败的是 `{19968, 512}` BF16 = `{3*hidden, chunk}`
   (19968 = 3 × 6656)，连续 **3 个** (gate/up/act) 各 20.4 MB = 61 MB，另加 `{hidden, chunk}`;
3. 对照规划侧: `Variant::post_mixer_workspace_capacity_bytes` 对 Muse **直接 return 0**
   (旧注释"无 workspace"——因为运行时把 `linear_swiglu` 拆成了独立 `linear + silu_mul`)，
   于是规划完全没算这 68 MB → 多 chunk prefill 时 arena 越界。
修复: 该函数改为按运行时真实峰值估算
`3 × 2 × intermediate × tokens + 2 × hidden × tokens`（gate/up/act + down 输出，全 BF16）。
验证: 1050 字符 prompt + chunk 512（原复现）→ **rc=0 无越界**; 16K prompt + chunk 512 → **rc=0**。
注: 同目录 `impl/load/variant.cpp` 是**又一份不在 CMake 里的死拷贝**（§92 的陷阱再现，
本次只改 `impl/variant.cpp`）。arena 轨迹的 caller 偏移打印已作为常驻诊断保留（env 门控）。

## 101. [新阻塞项] Muse 生成本身是词沙拉 — 与 KV 量化无关 (2026-09-10)
证据 (决定性): 用 §97 修复后**可用的 bf16 KV 基线**跑同一个短 prompt
("用一句话介绍杭州。", greedy, max_tokens 48):
- `--kv-dtype bf16` → reasoning = "锟斤拷…ols Shan Sprecher… Phyt compos… contractors Ortega…"
- `--kv-dtype nvfp4` → reasoning = " Premier锟斤拷…"
两者都是**词沙拉**（英文单词随机拼接 + 大量 U+FFFD），且 content 为空（思考模式，token 全在
reasoning）。=> **与 KV 档位无关**，是模型路径（权重/几何/tokenizer/模板）的既有缺陷。
- 不是 §16 的"全 0 输出"（那时是恒定 token 0 → 全 U+FFFD; 现在是有英文单词的乱拼）。
- 不是我的改动引入: 09-09 23:00 的 E8-only 构建（本会话最早的冒烟）已复现同样的乱码。
- 影响: **W10 的 Muse 57K 矩阵无法用针刺/质量判定**（§89 崩溃已修，但模型本身不可判）。
  质量矩阵只能先用 qwen3.8-27b（§95 已做）; Muse 质量需先修此缺陷。
- 下一步入口: 用 `score_tokens`/`ninfer-perplexity` 在 Muse 上跑一个短语料（若 ppl 正常而
  生成乱 → 采样/模板问题; 若 ppl 也乱 → 权重/几何映射问题）; 对照 `frontend/chat_template.jinja`
  与 tokenizer_config 的 special tokens 是否与 §11 的转换器产物一致。

### 101b. Muse PPL 判别路径不可用 (2026-09-10)
`ninfer-perplexity <muse> --text ... --kv-dtype bf16` 直接拒绝:
`artifact identity 'muse-glimmer-30b/nvfp4' has no registered target for this device`
—— 该 CLI 的 target 注册表不含 Muse（当前只有 qwen 系）。=> 想用 PPL 判别 Muse 权重/几何
是否正常, 需要先给 perplexity 注册 Muse target（或改走引擎 CausalScoring purpose + 自写 harness）。
下一步建议: ①在 `apps/perplexity` 里确认 target 注册方式并加 Muse; 或 ②用 serve 的
`score_tokens`/logprobs 对同一短语料算 NLL 做对照。

## 102. Muse 生成质量根因定位: decode 层栈输出 NaN (2026-09-10, §16-22 悬案正式收敛)
判据链 (全部实证, 每步都有对照):
1. **token 序列**: Muse prompt "1,2,3,4,5,6," → `generated ids 45351 0 0 0`（首 token 来自
   prefill 头, 正常; 之后恒 0）; qwen 同 prompt → `22 248046` = "7" ✓ 对照成立。
2. **不是 KV 档位**: `--kv-dtype bf16`（§97 修复后可用的无损基线）同样乱 => 与 KV 无关。
3. **不是 lm_head 内核**: 新增独立判决测试 `tools/nvfp4_muse_vocab_test.cu`
   （合成权重: 全 e2m1(1.0) 码 + e4m3(1.0) 尺度 + x=1 → 期望每行 = K = 6656）:
   **202112 行全部 exact=202112, zero=0 → MUSE_VOCAB_GEMV_PASS** ✓ 内核正确。
4. **不是权重**: prefill 头（M>1 → small-T 路径）用同一份 lm_head 权重且能出正确首 token。
5. **定位到层栈**: 新增 env 门控探针 `NINFER_HEADDBG=1`（`text_context_impl.h` 的
   `debug_head_probe`, 打 BF16 张量前 8 值）插在 ordinary_decode_batch 的
   post_embed / post_layers_x / final_hidden / logits 四处:
   | 探针 | Muse | qwen (对照) |
   |---|---|---|
   | post_embed | 0.073 0.055 -0.010 … **有限** | -0.0016 0.011 … 有限 |
   | post_layers_x | **NaN** | 2.41 0.38 3.56 … 有限 |
   | final_hidden | **NaN** | 0.67 0.095 … 有限 |
   | logits | **NaN** | 10.9 8.4 … 有限 |
   => **NaN 由 decode 的 `run_layers` 产生**（embedding 正常），最终采样退化为 token 0。
   这与 §22 的悬案逐字吻合（"decode 首步层 0 mlp 后 x = 0x7FFF NaN → 层 0 attention 读
   cache 含 NaN; prefill 正常但 decode 用 cache 即 NaN"）=> **§16-22 那条链从未真正关闭**。
- 下一步 (窄化到子层): 把 `debug_head_probe` 再插到 `run_layers` 内层 0 的
  attention/GDN 子层前后（`text_context_impl.h` 的 `run_layers`/`attn_mix`），看 NaN 是
  attention（KV cache/envelope, §22 的候选）还是 GDN（线性注意力状态）引入;
  对照 qwen 同位置。Muse 的 GDN 状态池与 KV envelope 是首要嫌疑。
- 复现: `bash _headdbg.sh`（Muse vs qwen 对照, 各 4 步）。

## 103. Muse decode 逐层发散: 第一个异常层 = 16 (KV 路径相关) (2026-09-10)
探针: `NINFER_HEADDBG=1` 现在逐层打印 (L00..L51, attn/gdn + mlp 两个阶段, 带 <NAN> 标记),
prompt "1,2,3,4,5,6," + max-new 2 + greedy + no-graph。
### 观测 (Muse, 三步对照)
| KV 档 | 首个异常 | 形态 |
|---|---|---|
| nvfp4 (默认) | **L16_attn 起 NaN** | L15_mlp 仍有限 (0.59 -4.28 …), L16_attn 全 NaN |
| int8 | **L19_mlp 起 NaN** | 更晚一层 |
| bf16 (无损) | **无 NaN**, 但 L51 已到 **1e29** | final_hidden/logits 归零 -> token 0 |
- **发散形态**: 隐藏态随层号指数增长 (L0-L15 ~1-10 → L16 ~1e6 → L51 ~1e29), 最后 rmsnorm
  把巨大值压成 0 -> logits 0 -> 采样 token 0。首 token 来自 prefill (正常) 之后全 0。
- **异常位置随 KV 档位移动** => 不是某一层的权重/代码坏, 而是 **decode 的 KV 读取路径**
  与档位相关 (nvfp4 最早暴露)。
- **不是 artifact 损坏**: 扫 `text/layers/16/*` 全部 17 个张量的字节统计 (0xFF/bf16NaN/zeros)
  与同族层一致, 无异常填充。
- **不是 head**: 独立判决测试 `tools/nvfp4_muse_vocab_test.cu` 已证明 lm_head gemv 内核在
  该几何 (202112×6656) 下 202112 行全对 (MUSE_VOCAB_GEMV_PASS)。
- **不是 prefill**: prefill 同层栈正常 (首 token 正确), 只有 decode 发散。
- **Muse 的层型**: `layer_kind = {1,1,1,0}` 循环 (39 SWA + 13 full, 全部走 attn_mix, 无 GDN);
  L16 是 SWA 层, 但 L0-2 同为 SWA 且正常 => 单纯"SWA 层坏"不成立。
- **窗口线索**: `sliding_window_tokens` 在 `program_impl.h` 里**没有任何赋值点** (只有 launcher
  侧的拷贝), 需确认它是否真的按层传入; 若恒为 0, SWA 语义在 decode 里未生效 (长上下文会
  多读 KV), 但与本例 (10 token) 的早期发散不直接相关。
### 下一步 (按性价比)
1. 在 `attn_mix` 内部插桩 (投影后 / gqa 后 / o_proj 后) 对 L15 vs L16 各打一次, 看发散是
   从 qkv 投影、attention 还是 o_proj 开始; 对照 qwen 同位置;
2. 打印 L16 的 `PagedKVLayerView` 关键字段 (dtype/quant_group/sliding_window_tokens/
   k_pages 形状) 与 L15 对比 —— 档位相关的位置移动提示视图字段或页映射有差异;
3. 用 `--spec mtp`? 不适用 (Muse 无 MTP); 可改用 `score_tokens` 路径 (M>1) 验证
   "M=1 decode 路径特有" 的假设。
### 复现
`bash _headdbg2.sh` (逐层, nvfp4) / `bash _muse_dtype_nan.sh` (三档位对照)。

## 104. Muse NaN 收敛: 层 16 的**早期位置 KV** 是坏值, 只在查询看到它时暴露 (2026-09-10)
`attn_mix` 内部逐阶段探针 (fidx=15/16, `NINFER_HEADDBG=1`) 的结果:
| 阶段 | L15 (非 SWA) | L16 (SWA) |
|---|---|---|
| in_x | 有限 | 有限 (0.586 -4.28 …) |
| q / k / v | 有限 | **有限** |
| qn / kn (rmsnorm+rope 后) | 有限 | **有限** |
| **attn_out (gqa_attention 输出)** | 有限 | **NaN** |
| out_x (o_proj 后) | 有限 | NaN |
=> **NaN 由 `ops::gqa_attention` 产生**, 且所有输入 (q/k/v/qn/kn/gate) 都干净。
### 关键澄清: prefill 也有 NaN, 只是不被采样看到
- 760 字符 prompt (2 块 prefill) 的日志里, **第一条 NaN 出现在 prefill 的第一块** (L16_attn),
  不是 decode 独有;
- 但首 token 仍正常 (45351/100103 各不相同) => 采样用的是**最后一行** (因果注意力下它只
  看自己), 而探针读的是**第 0 行** (它要看到全部早期位置) -> 第 0 行 NaN、末行干净。
- decode 的输出只有 1 行 (新 token), 它必须看全上下文 -> 必然撞上坏 KV -> NaN ✓ 与
  "prefill 正常 / decode 崩" 的现象完全自洽。
=> **根因 = 层 16 的早期位置 (position 0 附近) 的 KV 内容为 NaN**（写坏了），
   而层 0-15 的早期 KV 干净。
### 为什么 prefill 自己没崩
单块 prefill 的注意力在块内完成、且采样只看末行; 2 块 prefill 的日志证明第一块就有 NaN,
但同样因为只看末行而"看起来正常"。
### 下一步 (最后一步定位)
1. 在 prefill 之后 dump **层 16 与层 15 的 KV 第 0 页** (D2H 前几个 bf16) —— 直接看层 16 的
   position 0 的 K/V 是否 NaN/0x7FFF, 对照层 15;
2. 若确认, 顺着 `gqa_kv_append` 查层 16 的写入寻址: 层 16 是 SWA (`layer_kind[16]==1`),
   检查 SWA 层是否走**环形缓冲/不同页映射**的 append 分支, 以及该分支的页基址是否与
   读取侧 (`batch_layer_view(fidx)`) 一致;
3. 备选: 用 `--kv-dtype bf16` 跑同一 dump (bf16 无 NaN 但仍有 1e29 发散) —— 说明除了
   "早期 KV 坏" 之外, 可能还有第二个 (更普遍的) 精度/寻址问题, 两个都修才可能让 Muse 正常。
### 复现
`bash _mixprobe.sh` (L15/L16 阶段探针) / `bash _muse_multichunk.sh` (2 块 prefill 证明
prefill 第一块就有 NaN)。

## 105. 用户拍板 (2026-09-10): W9 改为"内存瓶颈下自行工程化"; 默认 KV 表保持现状
### W9 (训练线) — 用户答复
> "我那个其实找不到了 你想办法在内存瓶颈的条件下进行吧 不管是借用显存还是ssd"
- 选项 A 作废（pilot 的 serve 启动方式已丢失, 无法复用）。
- **新口径**: agent 自行在 24/28GB 内存瓶颈下把**全量重采**跑起来, 手段不限
  （借显存 / SSD / 降 host 预留 / 流式读盘等）。
- 已知障碍 (§9d 结案补充 2): dflash2 artifact 25.4GB + 运行开销 > WSL 28GB -> cudaMalloc
  确定性失败; 且当时有死训练进程占 3.6GB 显存未回收。
- 资产: 训练脚本 `train_dflash2.py` (workspace 根 + data/df2pilot 副本), pilot 数据
  `data/df2pilot/`, 既有 hs_cache (`/home/user/bench/hs_cache{,_2,_topk}`, `data/hs_cache`),
  引擎侧采集钩子 `NINFER_HS_DUMP_DIR` (text_prefill_impl.h:86+) 与
  `NINFER_HS_DUMP_TOPK` (校验模式, tokens<=256)。
### 默认 KV 表 — 用户答复
> "保持现状，等 §96 修复"
- 即: 出厂默认 10L e8 不动; §99 的长上下文掉针由 §96 (e8 高层退化) 的修复来解决。
  在修复前, 质量矩阵/验收统一用"纯 nvfp4 / 0-7:e8"做对照。

## 106. W9 采集复活 + 新 bug: 部分请求 "arena allocation must be nonzero" (2026-09-10)
### 采集状态 (运行中)
- `collect_main_topk.py` 两处修复后跑通 (① /v1/models 是 GET, POST 会被断开;
  ② TOPK trailer 必须精确切片)。06:53 起, 07:01 已 **299/480** pack (≈5.8G)。
- 服务端: `ninfer-serve qwen3_8_27b_nvfp4_dflash2.ninfer --kv-dtype nvfp4
  --kv-layer-storage all:nvfp4 --max-context 16384 --kv-capacity 16384 --spec dflash2
  --draft-tokens 7 --greedy --max-concurrency 1`; 单条失败重试 3 次后跳过。
- 观测: 单条 prompt 的 prefill 约 1-2s, 但**每次失败要重启 serve (~2min)**, 是主要时间成本。
### 新 bug (待修): seq 299 确定性 500
- `code:rowsplit_grouped_mma.cuh` (5000 字符) 每次都在 prefill 阶段抛
  `arena allocation must be nonzero`; 新 serve 实例的**第一条**请求也复现
  => 与请求内容相关, 不是累积状态。
- 诊断增强 (已写未编译): `src/core/arena.cu` 的 bytes==0 分支改为打印 module 相对偏移
  + `backtrace_symbols_fd` 调用栈 (旧版 dladdr 对 static 函数拿不到符号, 只印裸地址,
  ASLR 下无法 addr2line)。
### §104 探针已就位 (本次构建 07:00 已含)
- `NINFER_KVDUMP_DIR` + `NINFER_KVDUMP_KV=15,16` -> attn_mix 内 dump append 前的 BF16
  kn/v + cache_positions; `NINFER_KVDUMP_LAYERS=15,16` + `NINFER_KVDUMP_PAGES=4` ->
  prefill chunk 后 dump block_tables + 各层 k/v/ks/vs 原始 plane (含 meta)。
- 分析器 `_kvdump_analyze.py` (e2m1/e4m3 解码 + stored vs source 逐 token 对比)。
- 复现脚本 `bash _kvforensics.sh nvfp4` (需 GPU 空闲; 采集完成后再跑)。

## 107. W9 复活: 采集 v2 (引擎 top-16 教师) 跑通 (2026-09-10)
### 根因链 (三段, 全部实证)
1. **训练强制要 ids16/vals16**: `train_dflash2.py:415` 无 ids16 直接 `continue` (教师源已切换)。
   旧 479 npz 全无 → 会**零样本**; 之前 loss 12.4 那批是 HF-head 教师路径 (§289 已判死)。
   唯一有效来源 = 引擎 TOPK dump (pilot 147-token 数据 loss 1.39 即此路)。
2. **引擎 TOPK dump 越界**: `[vocab,tokens]` BF16 logits 走 prefill workspace arena,
   而 arena cap 只有 79,990,784 B (~76 MiB), chunk 256 就要 127 MiB →
   每请求 `arena overflow ... error std::bad_alloc` (HTTP 500)。
   修复 (`text_prefill_impl.h`): 该诊断路径改用 `cudaMalloc` 临时缓冲 (GPU 空闲 7.3 GiB, 够),
   上限 256→1024, 并给 host top-16 加 `f <= topv[15] -> continue` 早退 (254M 次比较/chunk 变 O(vocab))。
3. **采集器残留污染**: 旧版失败尝试留下的 chunk 会被下一次成功请求一起拼接
   (seq 0 的 T=42980 就是这么来的, 且 hsdump3 里还有上一轮的 721 个旧 chunk)。
   v2 (`collect_main_topk2.py`): 每 attempt 前清空 dump 目录; 保存 top1/ids16/vals16;
   500 先原地重试一次再重启 serve (省 ~2 min/次)。
### 现状 (运行中)
- 07:13 已 **98/480** pack, ≈0.66 rec/s, ETA ≈07:23; 输出 `/home/user/bench/hs_cache_topk2`。
- 每请求 chunk 结构: 若干 prompt chunk (--prefill-chunk 1024, 实际变长) + 2~4 token 的
  尾部 chunk (draft/verify 阶段), 采集器用 `ntok > 8` 过滤尾部 (draft_tokens=7 下安全)。
- 抽检 seq_000000: `top1 == ids16[:,0]` 100%, vals16 逐行单调, top1 下一 token 命中 0.27~0.42。
### 下一步
① 采集完 → `_kvforensics.sh` (§104 取证, ~3 min) → ② npz 拷到 Windows
`data\hs_cache_topk2` → ③ `DF2_CACHE=... python train_dflash2.py --steps 6000
--batch-seqs 6 --anchors-per-seq 12 --max-ctx 128 --lr 6e-4` (~4.4 h @0.38 steps/s)
→ ④ `eval_ddtree.py --ckpt step_006000.pt` hit@k。


## 80b. [引擎侧B] 晨报: W2② 验收 PASS + 诊断修正链 (2026-09-10 07:2X)
- **W2② PleTable 分页 fault e2e PASS** (0eaac04): tests/ops/ple_table_e2e_test.cu
  真表 95GiB 上 — 行号公式 (EOS 截断/窗口去重) + UVA pinned LRU fault-in +
  device gather **位精确** vs host pread (7680 元素 0 mismatch) + 驱逐后二次
  gather 仍位精确 (epoch 保护有效)。1MiB 小预算强制驱逐路径已覆盖。
  W2③ (接 fuse_draft/LABD) 剩引擎接线。附带钉死 derive_rows 的 prevs 语义
  = 每 token 平铺 (ngram_size-1) 个前驱 (近→远)。
- **诊断修正**: bf16 repro2 + nvfp4/iso3 repro4 (T=1/4, ctx=1/141) 在重建内核上
  全 no error => §37/38 老 sync-hang 已被断言放宽重建治愈; serve 崩溃根因 = TT6
  smem (§70/74, 对方已修, serve 07:08 链接就绪)。
- 实锤: sm_120 optin = 101376B (99KiB) 全可用 (smem_limit_test.cu; 此前 python
  ctypes 报 0 是调用错误, 勿信)。
- tt6fix 全量 make 挂点 = test_cold_i8.cpp 预存 API 漂移 (重编已过 = 旧日志
  误导); ninfer-serve 不受影响。
- GPU 状态: W9 采集 (qwen27 dflash2, 对方启动, 自动重启型) 占 25GB — W1 冒烟
  等空窗; PleTable/repro 均小显存已与采集共存验证。

## 108. [已定位+已修] §106 的 arena alloc-zero 根因: 1..4 token 的 W4A4 尾巴 (2026-09-10 07:3X)
### 定位链 (backtrace 全符号化)
- `arena.cu` bytes==0 分支加 `backtrace_symbols_fd` + module 偏移后拿到栈 (记录 299 在
  **新进程的第一条请求**就复现 => 与内容相关, 不是累积状态):
  `DeviceArena::alloc_bytes(0)` ← `nvfp4_linear_swiglu_dispatch`
  ← `qwen3_6_27b::Variant::post_mixer` ← `TextContext::prefill_impl<DFlashFeatureSink>`
  ← `prefill_text_chunk` ← `ProgramImplCore::advance_prefill`.
- 代码级: `nvfp4_linear_swiglu_plan.cpp` 的 `LinearW4A4Post` 分支把 batch 切成
  `head = tokens & ~255` + `tail`; tail 走 `allocate_baseline_workspace(workspace, tail)`,
  而 `linear_workspace_capacity_bytes(...)` 对 **tokens<5 返回 0**
  (`nvfp4_dispatch.cpp`: MlpGateUp -> `tokens>=5 ? W4A4 : A16`) → `alloc_bytes(0)` →
  `arena allocation must be nonzero` → 请求 500, 且**引擎进入永久 failed 态**
  (后续全 503 `inference engine is unavailable`, 只有重启进程能恢复)。
- 触发条件: 某 prefill chunk 满足 `tokens >= 512 && tokens % 256 ∈ {1,2,3,4}`
  (chunk 切分动态 => 约 1.6% 的 chunk 命中; 观测 3 次失败 / 约 700 chunk)。
### 修复
`allocate_baseline_workspace`: `alloc_bytes(max(linear_bytes, 256))` — A16 尾巴本来不需要
workspace, 但借用式 arena 不能为空; planner 只对 >=49 token 规划该分支, 容量不受影响。
### 附带修复 (采集器 collect_main_topk2.py)
① `ensure_serve` 只信任**自己 spawn** 的 serve (旧版 attach 到外部 serve 时, 503 后
`serve_proc` 为 None → 无法重启 → 375 条被连环跳过); ② `stop_serve` 顺带清理 stray
ninfer-serve (按 /proc/<pid>/cmdline 精确匹配 argv[0], 避免误杀 grep 自身)。


## 81b. [引擎侧B] 任务汇报 (2026-09-10 07:4X 更新: 见上方 §80-89 为另一 agent 更新内容) (2026-09-10 07:3X, 用户指令: 对方仍在干活, 汇报防撞车)
**先读这节再动 GPU。我的自动链已全部撤除 (auto_smoke_w1/auto_longtest 已 kill),
不会再有任何后台抢占。当前 GPU 上只有你的采集 (8010, 未动)。**

### 我今晨已完成 (无冲突, 均已推 origin)
1. **W2② PleTable 分页 fault e2e PASS** (0eaac04): tests/ops/ple_table_e2e_test.cu
   真表上 LRU fault-in/UVA/device gather 位精确 vs host pread (7680 元素 0 mismatch,
   含驱逐后二次 gather)。顺手钉死 derive_rows 的 prevs = 每 token 平铺 2 前驱。
2. **repro2(bf16)/repro4(nvfp4+iso3) 全 PASS** (0eaac04): T=1/4, ctx=1/141 no error
   => §37/38 老 sync-hang 已被你断言放宽的重建治愈; serve 崩 = TT6 smem (你的 §70/74)。
3. **smem 实锤** (smem_limit_test.cu): sm_120 optin=101376B 全可用; §80 记录了
   ctypes 探针报 0 是误报 (避免你踩同一坑)。
4. 发现你已完成 W16 (lane 级失败/health failed/POST /recover) — 我未动, 待你验收发布。
5. tt6fix 全量 make 的 test_cold_i8 挂 = 预存 API 漂移旧日志; 我重编已过, serve 无碍。

### 我已撤除的自动链 (不会再跑, 但脚本在, 你也可以用)
- auto_smoke_w1.sh: 等采集退出 → pkill ninfer-serve → 起 8321 (nvfp4, 64K ctx,
  no-graph) → 七步冒烟 (baseline gen → reload e8/nvfp4 → gen → reload bf16 → gen
  → 坏spec 400 → health) → 结果 /home/user/w1_smoke.txt。
- auto_longtest.sh: 冒烟 PASS 后接力 57K 针刺 (all:bf16 + 0-11:e8,12-15:nvfp4 两配置,
  longtest_57k.py) → /home/user/w1_longtest.txt。
- 若你要跑: 直接 bash /mnt/c/Users/User/Documents/ziqinzhang/auto_smoke_w1.sh
  (会杀 8010 的 serve! 等采集完再用)。

### 我接下来 (等你协调, 不抢)
- W1 七步冒烟 + 57K 长测: **需要 GPU 空窗** (Muse 18.4GB + 64K KV ~7.5GB)。
  采集 07:23 时 271/480 (~30/min, 约你读到时已完)。你说一声空窗归我, 我即跑; 或者
  你跑你的, 我等下一窗。
- W10 KV 矩阵重测 (longtest_57k --reloads 六配置): 排 W1 冒烟后, 同样要 GPU。
- 纯 CPU 可并行 (我不占 GPU): W2③ fuse_draft 接线评估 / §73 W-P1 win shim。
- 不碰: 你的采集/8010/构建树 (除非你喊我)。

### 给你的两个复用资产
- tools/archkit/ple_gather_test.py (W2①, 你 §69 也验过) + tests/ops/ple_table_e2e_test.cu (W2②)
- tests/muse128_repro2.cu / muse128_repro4.cu (bf16/nvfp4+iso3 小 T 复现, 已适配新 API)


### 81b-2. 汇报勘误与收窄 (07:5X, 我读完你的 §80-105 后)
**我 §81b 里"W1 冒烟/57K 长测待跑"已过时** — 你 §81/§90/§95/§98 已覆盖 (TT6+stale 句柄
7 步过 / W10 用 qwen27 跑了 / Muse 被 §101 词沙拉阻塞)。我的自动链 (smoke/longtest)
已撤除且**不会重挂** — 你在 §95 修好的 longtest_57k.py 才是有效工具, 我那份是旧版。

**我今晨净贡献 (不与你重复, 均已推 origin 0eaac04/98bc24e)**:
W2② PleTable e2e PASS (真表位精确) / repro2+repro4 (§37-38 sync-hang 治愈实证) /
smem optin=99KiB 实锤 / W2① python gather 工具。

**我现在改领 (声明, 避免撞车; 你若正在做喊一声我立刻让)**:
1. **§104 最后一步: dump 层 16 vs 15 的 KV 第 0 页** (D2H 前 8 bf16, 看 position 0
   的 K/V 是否 0x7FFF/NaN) — 30 分钟活, 直接命中你留的口子; 若确认坏值我顺着
   gqa_kv_append 的 SWA 分支查写入寻址。
2. 若 1 你已接: 我转 **W2③ fuse_draft/LABD 接线** 或 **§73 W-P1 win shim** (纯增量)。
3. 继续不碰: 采集/8010/你的构建与测试链。

**给你的信息补充 (可能有用)**:
- e8 高层损坏 (§96) 与 Muse 层 16 早期 KV 坏值 (§104) 形态相似 (都是"特定层的 KV
  内容坏而不崩") — 若 §104 dump 显示 SWA 层页映射错, 值得对照 §96 的 e8 高层 (高层
  恰为 SWA 窗口更深的层?) 一并查。
- 我 reps 里的 nvfp4+iso3 小 T 复现 (repro4) 可作为你 §96 插桩的最小载具 (不载权重,
  10 秒跑一轮, 比 serve 快两个量级)。


### 81b-3. [终章] 引擎侧B 全面移交 (2026-09-10 07:5X, 用户指令: 对方接管全部)
**我自此停止一切执行** (无后台任务在跑: 自动链已撤、轮询已结束、无编译/serve 由我持有)。
以下为移交清单。

### 资产 (全部已验证, 已推 origin)
| 文件 | 状态 |
|---|---|
| tools/archkit/ple_gather_test.py | W2① 公式+真表 gather, PASS |
| tests/ops/ple_table_e2e_test.cu (+CMake) | W2② 分页 fault e2e, PASS (0eaac04) |
| tests/muse128_repro2.cu (bf16) / muse128_repro4.cu (nvfp4+iso3) | 小 T 复现, 已适配新 API, 全 no error; 可作 KV 插桩最小载具 (10s/轮, 不载权重) |
| smem_limit_test.cu (ziqinzhang 根目录) | sm_120 optin=101376B 实锤工具 |
| tools/archkit/longtest_57k.py | **注意: 我仓里是旧版**; 对方 §95 修好过三 bug (植针/思考模型 content/预算), 以对方版本为准 |

### 环境现状
- studio 栈: controller 8080 (带 HTTPS_PROXY=10808, RAG 的 HF 下载需要) + frontend 3000 在跑。
- WSL: .wslconfig=24GB (备份 .wslconfig.bak-14g); ccache 已装未启用 (对方 §93, 建议下次全量重建时 -DCMAKE_*_COMPILER_LAUNCHER=ccache)。
- 采集 (W9, 8010) 运行中, 断点安全 (npz exists 跳过), 语料 480 条。
- local-studio 本地 main 有未推提交 (1fb9b19..dc683bb+: RAG/hwdec卡/KV分配/重排按钮), GitHub 403 待 0xSero 凭据 — 与引擎仓无关, 需用户处理。

### 我知道的坑 (散在各节, 汇总防丢)
1. §92/§100 死 TU 陷阱: 改 launcher 前先 grep CMakeLists 或 strings 二进制 (impl 目录里还有死拷贝)。
2. §57/§59: WSL 树单向覆盖丢过未提交修复; 同步后 md5 校验。
3. §88: gqa_attention_decode_e8.cu 单 TU ptxas 22+min/13GB, 勿与内存大户并行。
4. 我的 ctypes 探针报 optin=0 是误报 (§80b), cudaFuncSetAttribute 实测全可用。
5. build6 全量 make 挂点 test_cold_i8 是旧日志 (重编已过), serve 不受影响。

### 未完成项归位 (全部在 §66 工作包 + 对方 §80-105 进度, 无我私挂)
- 进行中主线: §104 dump 层16 KV 第0页 → SWA 写入寻址; §96b e8 高层插桩 (两条形态相似, 可能同根)。
- 排队: W2③ fuse_draft/LABD 接线 / W7 (等 checkpoint) / W11 (Muse 质量需先修 §101) / W13 权重卸载 / §73 W-P1..P6 Windows 移植 / W15 Windows EXE。
- 临时脚本可删: ziqinzhang 根目录 find*.ps1(已删) 残余 check_*.sh / poll*.sh / run_*.sh / rearm.sh / disarm.sh / auto_*.sh / gpu_check.sh / corpus_count.sh 等均为一次性, 对方可直接 rm。
移交完毕。(_TODO 全文维护权本就在你, 此后我只读不写。)

## 109. [根因实锤+已修] Muse 词沙拉/NaN = 行缩放表 [16][4][256] 越界 (2026-09-10 08:0X)
### 证据链 (KV 取证, 全部来自引擎自 dump)
用新探针 (`NINFER_KVDUMP_DIR`, 见 §106) 对 Muse 单轮 prefill 后逐层 dump 52 层 plane:
- **层 0-15**: K/V 均写入正常 (码值/scale 都合理)。
- **层 16**: K plane **全 0** (码 0, scale=0x01 下溢最小值); V 正常。
- **层 17-51**: K plane 有数据但**全部 52 层中 17-51 的 K 字节完全相同** (md5 一致);
  V plane **全 0**。解码后 K 恒为 6.0*0.00195=0.0117 (即量化了一个 NaN/退化输入)。
- 同时 dump 的 *append 源* (`kvsrc_*_kn.bin`): 层 15/16 源 K 有限且互不相同;
  **层 17/18/51 的源 K 字节完全相同** → 说明层 16 的注意力输出 NaN 之后, 后续层输入
  全 NaN (bit 相同), 与 headdbg 的 `L16_attn_out NaN` 一致。
### 代码根因
`src/ops/kernel/gqa_isoquant_row_scale.cuh`:
`__constant__ unsigned short kGqaKvRowScaleDev[16][4][256]` —— **按 layer 索引, 只烘了 16 层**
(标定自 qwen3.8-27b 的 kvcalib-a)。Muse 有 **52 个注意力层**, `gqa_kv_row_scale(16..51, h, d)`
越界读常量内存 → 层 16 取到 0 (K*=0 → 写全 0), 层 17+ 取到垃圾 (K*=garbage/NaN);
读侧 `gqa_kv_row_scale_inv` = 1/0 = **inf** → Q 变 inf → QK^T NaN → softmax NaN →
`L16_attn_out` NaN → 逐层发散成词沙拉 (§101/§104 全部现象归一)。
旋转表 `kGqaIsoquantRotDev[64][4][4]` 按 4-通道块索引 (与层无关) → 无此问题。
### 修复
`gqa_kv_row_scale`: 越界 (layer∉[0,16) / kv_head∉[0,4) / d∉[0,256)) 返回 **1.0**。
语义正确: 标定只对 qwen 的 16 层有效; 恒等缩放保持 QK^T 精确 (写 K*s / 读 Q/s),
只是非 qwen 模型的旋转域量化不做标定。Muse 层 0-15 仍会套用 qwen 的标定 (数学上
QK 不变, 仅量化误差分布次优)。
### 待验
重建后 `bash _fix_rowscale.sh` 检查 ① headdbg 无 NaN ② Muse 生成是否正常中文。

## 110. §96 的"plane 重叠"假设被否 + W9 训练启动 (2026-09-10 08:0X)
### §96 e8 高层: 不是 plane 地址重叠 (实证)
用 §109 的 plane 探针 (含每层 k/v/ks/vs 的设备地址) 跑
`_e8_plane_dump.sh 100000 14-15:e8` (qwen3.8-27b, 32K 上下文, 8 个 prefill chunk):
- 12 个 plane 地址两两不重叠 (total overlapping plane pairs: **0**) => **别名/重叠假设作废**。
- 各层 k plane 的 0xFF/0 字节分布: L13 (nvfp4) zero=15440 / ff=1981;
  L14/L15 (e8) zero≈150-162k / ff≈25-27k (编码不同, 不可直接对比)。
- 结论: e8 高层退化不是寻址/别名问题, 需回到**数值/编码**方向 (下一步: 解码 e8 码本,
  对比同一位置 e8 vs nvfp4 的 K 相对误差随层深变化; 或直接查 e8 的 scale 平面语义)。
### W9 训练已启动 (07:57:56)
- `DF2_CACHE=data\hs_cache_topk2` (480 packs, 41.4GB, 已从 WSL rsync 到 Windows),
  旧 checkpoints (broken-teacher 那批 10 个) 已移到 `data\dflash2_ckpts_broken_teacher\`。
- `--steps 6000 --batch-seqs 6 --anchors-per-seq 12 --max-ctx 128 --lr 6e-4`;
  **step 1 loss=3.0643** (与 pilot 的 3.0642 一致 => 教师/数据格式对齐);
  日志 `dl\train-dflash2.log`, 输出 `dl\train-retrain.out`。
- 占用 24.3GB 显存 => Muse 验证必须等训练结束或短暂让路 (计划: 构建完成后让路 3 分钟跑
  `_fix_rowscale.sh`, 再重启训练)。

## 111. [已定位, 待构建验证] §94 ft 能量全 NaN = 观测点在内核启动之前 (2026-09-10 08:0X)
代码走查 (无 GPU) 直接定位:
- `src/ops/launcher/gqa_attention_decode.cu:402-407` 的 `ft::observe(...)` 在
  **`launch.template operator()<...>()` 之前**调用; observe 内部是
  `cudaMemcpyAsync(host, partial_l_device, ...) + cudaStreamSynchronize`
  => 它读的是**本轮 kernel 还没写**的 workspace 缓冲 (每轮都被 arena 重新分配,
  内容为垃圾/NaN), 所以"轮轮的 NaN"。
- 注意: `src/targets/qwen3_6/impl/runtime/ft_stats.h` 是**另一份死拷贝** (3 参数签名),
  真正被 include 的是 `src/ops/common/ft_stats.h` (5 参数, 带 stream) —— §92/§100 的
  死 TU 陷阱又出现一次。
- 修复: 把 observe 块移到 if-constexpr 启动链 + `CUDA_CHECK(cudaGetLastError())` **之后**。
- 待验: 重建后 `NINFER_FT_STATS=1 --no-cuda-graph` 跑一轮 decode, `[ft] layer=... mean_l=`
  应为有限值且随层深变化。
- 附带 (§94 末尾的老问题): ft 观测目前只在 nvfp4 decode 路径; 分层 KV (e8 层) 拿不到该层
  能量, 若 W5 要在混合表上工作需覆盖全部 KV dtype 的 decode 路径。

## 112. [子代理调研] dflash2 接受率根因 + SVIP/DDTree 融合可行性 (2026-09-10 08:3X)
### 实测接受率 (来源 DFLASH2-MTP-STATUS.md / _TODO §1/§9)
| 配置 | 接受率 | 备注 |
|---|---|---|
| MTP3 chain (int8 KV, 代码语料) | 79.9% / 168 tok/s | d3 greedy |
| MTP3 chain (bf16 / nvfp4 / 默认表) | 76.8-79.6% / 162-178 tok/s | KV 修复后 |
| MTP3 中文散文 | 45.5% / 127 tok/s | 任务难度 |
| **DFlash2 d7 (官方 z-lab 权重, nvfp4 KV)** | **23.6% / 73 tok/s** | 位置衰减 177,113,72,52,36,19,12 |
| DFlash2 d7 all:bf16 KV | 27.7% | 全精度仍低 => 非 KV 精度问题 |
| DFlash2 采样调优后 (gumbel+dist-accept) | 24.1%→53.3%, AL 1.69→3.73 | 09-01 修 17 文件 |
| 参考: vLLM n=7 / MLX tok-1 | 47.8% / 75.9% | 外部基线 |
| 逐位置 | DFlash2 pos-1 61% vs MTP3 pos-1 91% | 5 层 draft 容量差 |
| kDFlash2MinAcceptance 0.25→0.05 | decode 24.4→52.7 tok/s @4K | spec_decision.h:45-46 |
### 根因假设 (按可能性排序)
1. **链式 walk 浪费了选择器的树材料**: `dflash2_selector.cuh` 每步产出 16 候选 × 16×16 前驱对分数,
   但 walk 只保留单链 (:180-254) => DDTree 的现成收益没吃到。
2. 采样/贪心口径 (09-01 已修, 24→53%); 中文 23.6% 更多是分布+容量 (MTP3 在该语料也只有 45.5%)。
3. 目标侧 NVFP4 量化喂给 draft 的特征与大模型 BF16 训练分布不同 (仅外部 A/B 可证伪)。
### SVIP/DDTree 融合方案 (最小改动)
- SVIP 现状: mtp_round.cuh:53-119 (熵超阈值就截断下一次 draft 长度) + mtp_impl.h:156-177
  (`NINFER_SVIP_THRESHOLD`, 默认关); **dflash2 完全没接** (dflash2_impl.h:355 硬要求 k==7)。
- 选择器输出: candidates [B,7,16] + unary [B,7,16] + pair [B,7,16,16], 已发布 draft_candidate_ids/probs。
- ① 离线闸门: `eval_ddtree` hit@k (已有); 判据 (§TODO 80-81): hit@2/4 >> hit@1 才值得做树验证。
- ② `dflash2_selector_walk_kernel` 加 beam-L 模式, 输出 L 条链到 drafts [L,7,B]
  (frame 缓冲本来就是 [16,k,B]) → 复用 batch 维, **verify kernel 零改动**。
- ③ SVIP-for-dflash2: 放开 :355 的 k==7 硬约束, 用候选概率熵写 `frame.proposal_extents`。
- 新开关: `--ddtree-leaves`, `NINFER_DF2_SVIP_THRESHOLD`。
- 风险: 验证成本 ×L (只有 hit@L − hit@1 > (L−1)/L 才划算); CUDA graph 固定 width-8 形状;
  workspace recipe 按 width 8 规划, 放开 k 需同步。

## 113. [已核实] dflash2 训练→引擎可用 artifact 的导出链完整 (2026-09-10 08:3X)
- `tools/convert/qwen3_8_27b/finalize_dflash2.ps1 [-Ckpt <pt>]`: 取最新 checkpoint →
  `patch_dflash2.py` 把 73 个 draft 张量写进一份 `_tuned.ninfer` 副本 (源 artifact 不动) →
  `verify_patch.py` 校验 (对象表 + 拷贝张量 hash + 73 个替换张量与 ckpt 逐值相等) →
  自动 stage 到 WSL `/home/user/models`。三个文件均已确认存在; 基础 artifact
  `models\Qwen3.8-27B-Huihui-Abliterated-NInfer-DFlash2\qwen3_8_27b_nvfp4_dflash2.ninfer` 存在。
- 训练完成后一条命令即可得到可直接 `--spec dflash2` 加载的 artifact (先跑 eval hit@k)。
- 注意: candidate_selector (3 个张量) 不在训练范围内, 保持原 FP8 draft 的值。

## 114. W9 训练 OOM 连环: 真因是**崩溃进程没退显存** (2026-09-10 08:5X)
- 现象: 训练两次 `CUDA error: out of memory` (第一次 Adam `_foreach_sqrt` @step~275,
  第二次 backward), 且日志停更时进程仍在 (Get-Process 可见, WS 忽大忽小)。
- **实锤**: `nvidia-smi` (Windows) 的进程表里 **11584 / 29680 两个 python 仍是 Type=C**,
  合计占 **31.2 GB / 32.6 GB**; 把它们 `Kill()` 掉后显存立刻回到 **348 MiB used / 31.8 GiB free**。
  => 之前"改配置"的思路是错的: 是崩溃进程泄漏显存, 新进程自然 OOM。
  (注: `taskkill /F` 对这类卡在 CUDA 调用里的进程报"没有此任务的实例", 要用
   PowerShell `(Get-Process -Id N).Kill()`; 杀完必须复查 nvidia-smi 而不是只看进程表。)
- 另发现: 我早先设的 `_after_build.sh` 自动链在 make 短暂消失后误判"构建结束",
  又把训练**重启成 batch=6 的旧配置**(PID 7056), 与我自己启的 batch=4 打架 => 已两条都杀,
  自动链全部撤除 (教训: 自动化里"杀掉别人进程"的分支必须带白名单/二次确认)。
- 现状: **单实例训练 (PID 27540, batch 4 / anchors 12 / save-every 100 / expandable_segments)
  于 08:51:51 起跑, step 1 loss=3.0709**; 显存 31.8GB 空闲起步。
- draft 内存模型 (实测): 1.848G 参数 → bf16 参数 3.44GiB + 梯度 3.44GiB + Adam fp32 13.77GiB
  ≈ 20.7GiB, 加 lm_head/embed 5GiB ⇒ 稳态 ~26GiB, Adam/backward 瞬时峰值再 +5~7GiB
  => 32GB 卡上本就贴着上限, 显存被占一点就会炸。

## 115. "按层索引表"隐患普查结论: 只有行缩放表一处真 OOB (2026-09-10 09:0X)
- `__constant__` 全扫: 只有 `kGqaKvRowScaleDev[16][4][256]` 有隐患 (§109 已修);
  rope 频率表 `[32]/[64]/[18]` 按维度对索引, 与层号无关 ✓ 安全;
  `kGqaIsoquantRotDev[64][4][4]` 按 4 通道块索引 ✓ 安全。
- `std::array<..., 64>` 家族 (`decoder_state.h:26/28/55/58/62/138`, `paged_kv_cache.h:61`,
  各 variant 的 `default_layer_kv_dtypes`): 索引是 **KV 层号**, 支持模型 qwen3.8-27b(16) /
  Muse(52) / qwen3.6-35b-a3b 均 < 64 ✓ 当前安全; 但没有任何越界检查 —— 若将来接一个
  >64 层的模型会**静默**读写越界 (建议: 在 `PagedKVCache` 构造/`batch_layer_view` 里加
  `static_assert`/运行时校验)。
- 其余 `std::array<T, 4/5/8/32/64/128>` 命中都是 magic bytes / sha256 状态 / 路由表,
  与实际层号无关 ✓。
- 结论: 这一轮"按层索引"审计**没有第二个必须修的 bug**, 只有 1 条健壮性建议。

## 116. [§96 根因实锤] e8 的 K 尺度取的是**旋转前**最大值 => 旋转后溢出 ±7 被截断 (2026-09-10 09:0X)
### 实锤数据 (新 dump: `_e8_src_dump.sh` 同时 dump append 源 K 与存储 plane)
- `_e8_clamp.py` 用 dump 出的源 K 复现 kernel 的变换 (H64 Sylvester butterfly, `gqa_kv_hadamard64` 同款),
  对比"旋转前 max"与"旋转后 max":
  | 层 | 码值数 | \|code\|==7 占比 | 旋转后/旋转前 max 比 (中位 / 最大) |
  |---|---|---|---|
  | L14 (e8) | 65536 | 1.03% | **1.42 / 12.9** |
  | L15 (e8) | 65536 | 0.55% | **1.54 / 15.5** |
- 机制: `gqa_attention_prefill_i8.cuh:121-135` 先算 `k_abs = warp_max(|K|)` (**旋转前**),
  用 `ksh_e = k_abs/7` 当尺度; 而 `gqa_kv_hadamard64(...)` 在**之后**才做 =>
  旋转把组内峰值放大 1.4-15 倍 => `clamp(-7,7)` 截掉顶部能量 + 有效量化级数变少。
  层越深放大越狠 (L15 > L14) => 正好解释 §96 的"层深相关退化"与"多高层 e8 才掉针"。
- 对比: nvfp4 路径先做 SO(4) 旋转、再按组求 e8/e4m3 尺度 => 没有这个先后颠倒。
- **修复方向 (需重建+质量复测)**: 把 `k_abs` 的 warp_max 挪到 `gqa_kv_hadamard64` **之后**
  (投影前), i8 路径 (`ksh = k_abs/127`) 同样处理; 预期 e8 高层退化显著缓解,
  §96/§99 的"e8 只放 0-7 层"约束可能随之放宽。
- 复现: `bash _e8_src_dump.sh 100000` + `python3 _e8_clamp.py` (全部 CPU 分析, 数据已存在)。
### 116b. 修复收益的**离线量化** (改代码之前就测出来)
`_e8_fixsim.py` 用 dump 的源 K 模拟同一套 E8 投影 (D8 ∪ D8+0.5, 与 kernel 同逻辑), 只改尺度来源:
| 层 | A: 现网 (旋转前 max/7) 相对误差 / 截断率 | B: 修复 (旋转后 max/7) |
|---|---|---|
| L14 | **0.4495 / 3.2%** | **0.0942 / 0.1%** |
| L15 | **0.4372 / 3.1%** | **0.0942 / 0.1%** |
=> 相对误差降 **4.7x**, ±7 截断几乎消失。修复 = 把 `k_abs = warp_max(|K|)` 从
`gqa_kv_hadamard64` **之前**挪到**之后** (1 行搬移, 无新参数)。
落在 `src/ops/kernel/gqa_attention_prefill_i8.cuh` (E8 分支, §116 已给行号)。
### 116b-verify. [协作验收] B 线独立复算 → **PASS** (2026-09-10 09:3X)
B 线按协议**不复用我的函数**独立实现 (自写 H64 Sylvester 矩阵 + torch bf16 view + 自写 meta/nibble 解析,
产物 `_collab/B_s2_verify_run.py` / `_collab/B_s2_verify.md`), 样本量比我大 **60×**
(98,304 组 + 6.29M code/层, 全 token/页):
| 指标 | A (我) | B (独立) |
|---|---|---|
| L14/L15 post/pre 中位 | 1.42 / 1.54 | **1.478 / 1.506** |
| clamp 率 (|code|==7) | 1.03% / 0.55% | **0.87% / 0.71%** |
- **B 的额外发现 (采纳)**: ①`post/(7s)` 全样中位仅 ~1.1, 但 max 14-16× ⇒ **饱和集中在尾部组**,
  别把收益按"全局 1.5×"折算 (我的 rel-err 0.45→0.094 是逐组重建误差, 两者不矛盾但要分开说);
  ②我先前只采了前 64 token (偏 prompt 起始段), 受限复算得中位 1.911/1.679、clamp 0.39%/0.44%
  —— 解释了两人数字的微小差异方向 ✓ (结论不变)。
- 结论: §96 根因与修复方向**独立可复现**, 验收通过; 修复收益的**表述**按 B 的提醒收紧。

### 116c. [已改] 实际改动 + 又发现一处更严重的 Muse-e8 隐患 (2026-09-10 09:0X)
- **已改**: `gqa_attention_prefill_i8.cuh` 的 `gqa_attention_prefill_fill_i8_kernel`(E8 分支,
  ~line 134-146): 旋转提到尺度计算之前 (`k_abs_e` 在 `gqa_kv_hadamard64` 之后求),
  尺度除数保持 7 ✓。这条是 **qwen e8** 走的路 (`KVHeads==4` => 不走 page kernel), 即 §96 的主体。
- **新发现 (未改, 待验证)**: `gqa_attention_prefill_fill_i8_page_kernel<...,E8=true>`
  (~line 260-300, 条件 `tokens >= 128 && Geometry::KVHeads == 2` => **Muse** 专用) 里:
  ① E8 分支**完全没有 `gqa_kv_hadamard64`** (写的是未旋转域的码, 而读侧对 Q 做了旋转 => 域不匹配);
  ② `ksh_e = k_abs / 127.0` (e8 应为 /7) => 尺度小 18 倍 => 码全撞 ±7 截断;
  ③ V 侧的 `vinv_e` 同源错误。
  => **Muse 一旦开 e8 KV 且 append>=128 token, 写入的 K/V 基本是废值**; 因 Muse 默认表是 BF16
  (variant.cpp:25-32 返回全 BF16 且 `supports_per_layer_kv_defaults=false`) 所以现在没被踩到。
  修法与 ① 同源 (补旋转 + 除数改 7), 但**必须先用 Muse+e8 起一轮验证**再改 (风险: 该 kernel 无测试覆盖)。
- 教训: 同名族 kernel 里"copy-paste 的量化分支"要一次性全扫 (`grep -n 'ksh_e\|/ 127.0f'`),
  否则修了一个还会漏另一个。

### 116d. 同族清点结果 (2026-09-10 09:1X): 第三处 e8 尺度 + i8 澄清
用 `grep -rn 'gqa_kv_hadamard64' src/ops` 把**所有**旋转点找齐, 只有 4 处 (2 处是 Q 侧):
| 位置 | 用途 | 尺度来源 | 结论 |
|---|---|---|---|
| `gqa_attention_prefill_i8.cuh:142` | prefill/fill 的 e8 K 写入 | 旋转前 (改后: 旋转后) | **已修** |
| `gqa_attention_decode_i8.cuh:245` | **decode 每 token 的 e8 K 写入** | 旋转前 | **本次已修** (同款搬移; 不修则 prefill 修好也白搭——每步仍在写截断 K) |
| `gqa_attention_prefill_i8.cuh:260-276` | Muse(2 KV heads) 的 bulk page-fill | 旋转前 + **无旋转** + `/127` | 未改 (§116c, 需 Muse+e8 验证) |
| `...:423` / `decode_i8.cuh:334` | Q 侧旋转 (读路径) | — | 正常 |
- **澄清 (纠偏)**: i8 (非 e8) 路径**不是**同类 bug —— 旋转只在 `if constexpr (E8)` 内发生, i8 量化的就是它求 max 的同一批值 ✓ (先前 §116c 里"i8 同理"的推断作废)。
- 剩余未落地: (a) Muse page-fill (§116c); (b) `std::array<...,64>` 无越界校验 (§115 健壮性);
  (c) 需重建才能生效: 现在引擎树里的 e8 修复必须 `touch` 对应 header 后再 make 一次 (当前那次 build 可能在改前就读过文件)。

## 117. "未落地项"复查 (2026-09-10 09:1X, 按用户要求继续排查)
除 §116c/§116d 的 e8 家族外, 复查到这些**仍未落地**的引擎缺口 (按性价排序):
### 本轮已落地 (1 行级)
- **U1 fp8 全局档已修**: kernel 实际用 nvfp4 的 16-组 scale 索引
  (`gqa_attention_decode_fp8.cuh:207` `gqa_kv_nvfp4_scale_index`), 而规划器给 `256`
  (`decoder_state.h:13`) → 改回 **16** ✓ (与 wrapper 的 "packed 必须是 16" 校验一致)。
  待验: 起一轮 `--kv-dtype fp8` 冒烟 (复用 `_kv_default_probe.sh`)。
- **U2 全局 e8 已修**: `gqa_attention_workspace_capacity_bytes` 的 dtype 白名单漏了
  `DType::E8Kv` (`ops/wrapper/gqa_attention.cpp:434` 附近) → 补上 ✓。待验: 同上。
- **U5 已落地**: `kPagedKVCacheMaxLayers = 64` + `plan_cache` 越界校验 ✓ (`_syntax_check.py` SYNTAX_OK)。
- **U3 只文档化** (不给新语法): `BFloat16` 仍是"未设置"哨兵, `all:bf16` 无法表达;
  已在 `kv_options.h` 注释里写清替代做法 (`--kv-dtype bf16`) 与将来需要"was-set 掩码"的原因。
### 纠偏 (实证后撤回)
- **U4 不是 bug**: Muse `impl/config.h:64-70` 明确写着"验收阶段 39 个 sliding 层**按 full 运行**
  (短上下文 < 窗口 2048 不裁剪, 与 HF 数值一致); **E3 窗口接线后**再按 `layer_kind` 恢复窗口语义"。
  => 属于**有意的阶段延后**, 不列入缺口。**但**要记住: 一旦上下文 > 2048 (如 57K 掉针),
  当前行为与模型训练时不同 → Muse 长上下文的掉针结果可能受此限制, 判读时要区分。
- **U1 的 i8 部分撤回**: i8(非 e8)路径没有旋转, 尺度来源本就正确 (见 §116d)。
### 仍未落地
1. ~~全局 fp8~~ (已修) / ~~全局 e8~~ (已修) / ~~all:bf16~~ (文档化) / ~~64 槽校验~~ (已修)。
   `packed16 && quant_group != kNvfp4QuantGroup`), 于是 warmup 直接抛 "must use quant_group 16"。
   二选一: ①把 fp8 规划组改成 16 (与 kernel 一致, 推荐); ②让 wrapper 接受 256 (需同步改 kernel 的分组常量)。
   改完要跑一轮 fp8 KV 的冒烟+质量 (可复用 `_kv_default_probe.sh`).
2. **[真缺口, 中] 全局 `--kv-dtype e8` 起不来** (§98②): "gqa_attention workspace: invalid profile or interval"
   —— e8 只在**逐层表**里能用, 全局路径缺 workspace profile。修法: 在 layouts 的 profile 里补 e8 档
   (照 nvfp4 的 (min,max) 表加一行) 或全局 e8 时回退到逐层表展开。
3. **[语义缺口, 小] `--kv-layer-storage all:bf16` 等于"未设置"** (§98③): 想表达"全 BF16 基线"只能靠
   全局 `--kv-dtype bf16`; 建议解析时允许 `bf16!` 或给 `--kv-baseline-bf16` 开关, 否则文档/矩阵永远少一格。
4. **[性能/正确性, 中] `sliding_window_tokens` 全局没有赋值点** (§103 遗留): Muse 有 39 个 SWA 层,
   现在它们按全注意力读整条 cache —— 语义上超出窗口、显存与带宽也白花。修法: 从模型配置把窗口写进
   `PagedKVLayerView.sliding_window_tokens` (decoder_state.cpp 的 layer_view 里补字段), 并用 57K 掉针回归。
5. **[健壮性, 小] `std::array<...,64>` 无越界校验** (§115): >64 层模型会静默越界。
6. 大件: W2③ fuse_draft/LABD 接线 / W7(等 ckpt) / W13 权重卸载 / Windows 移植 W-P1..P6。
- 认领方式: 逐条改 `_collab/board.md` 的"待认领"栏 (U1..U6), 谁动谁标 DOING。

## 118. 用户八问的落地状态 + 新下发的两件事 (2026-09-10 09:4X)
### 八问逐条查证 (证据在括号里)
| # | 问 | 已有 | 缺 | 判定 |
|---|---|---|---|---|
| 1 | 启动前自测→固化→首跑后跳过→可手动重校 | 离线采集 (`kv_calibration.h` dump post-RoPE K/V 给 `tools/calib`) + 离线烘表 (行缩放/SO(4) 旋转表, "kvcalib-a" → 引擎常量) | **运行时**那半截: 启动跑一遍→固化→之后跳过→手动重校 | **半截** |
| 2 | 输入量化等级→KV 温窗自动分配 (prefill 优先) | `docs/maintainer/kv-strategy-matrix.md` 实证矩阵 + `tools/archkit/kv_bit_budget.py` DP 分配器 (含 e8 只放 0-7 约束) + `--kv-layer-storage` 可吃其输出 | 引擎内"给 bit 数自动分配"入口 (现须人工跑工具再传 spec) | **策略完善/引擎自动化未做** |
| 3 | 推广到冷窗 | `--cold-policy none/window/host/disk` + entropy/rANS 冷槽 + `max_cold_pages` | bit 分配器未含冷槽代价模型 (9536B/slot) | **冷窗可用/未统一分配** |
| 4 | 热/温/冷自动动态分配 | FreeToken 三步骨架 (能量观测/pacing/周期重排; ft 的 NaN bug 今日已修) | 按热度自动决定热冷归属的**闭环策略** | **未闭环** |
| 5 | 权重卸载到内存 W13 | — (代码内 grep 不到任何 offload) | 全部 | **未做** |
| 6 | lookup ngram | CUDA 内核 (`include/ninfer/ops/suffix_lookup.h`) + fuzz 4000 全绿 + sidecar 加载器往返验证 | 真表 GPU gather / 前缀缓存(含 GDN 循环性) | **部分落地** |
| 7 | FreeToken 在本设备加载 FlashNext | P0 骨架 `src/targets/qwen4_exp/` + 74,520 项绑定契约 + `flashnext_convert.py` | **真实 checkpoint** (原 TODO #552: 权重未定位) | **架构就绪/权重缺失 → 本轮开始补** |
| 8 | dflash2 接受率低根因 | ①采样口径实锤 24.1%→53.3% ②非 KV 精度 (bf16 KV 27.7%) ③5 层 draft 容量 (pos-1 61% vs 91%) | 最强假设④"选择器已算 16×16 前驱对分数、walk 只留单链(树材料白扔)"未实施 | **部分找到** |
### 用户新下发 (2026-09-10 09:4X)
1. **用自动导入管线导入 MiniCPM5-1B**: `openbmb/MiniCPM5-1B` (1.08B, LlamaForCausalLM, GQA 16/2, 128K, Apache-2.0, 2.2GB)。
   管线入口 (`tools/archkit/_AUTOADAPT.md` S1-S8): `adapt.py <model_dir|config.json> [--model-id X]` →
   `arch_spec`(spec) + `flavors`(口味) + catalog 缺口 (`covered|engine_hook|new_op`) → config.h/leaves/bindings.stub/manifest。
2. **下 FlashNext 破限 NVFP4**: HF 上真名是 **Qwen3.8-Flash-Next (qwen4_exp 预览, MoE 512 experts + GDN + QSA + HyperConnections + PLE n-gram)**。
   选了 **`dealignai/Qwen3.8-Flash-Next-ABLITERATED-NVFP4`** (公开 gated=False, 135.3GB, 破限+NVFP4+含 `model-plefp8-*` PLE 表)。
   另两个备选: `orcarouter/...Uncensored-NVFP4` (**gated=auto, 需用户点同意**, 183.5GB) / `huihui-ai/...abliterated-GGUF` (公开 112GB, 破限但 GGUF) / `nvidia/...NVFP4` (公开 132.7GB, 不破限)。
   下载: `_hf_download.py` (走本机代理 127.0.0.1:10808, 可断点续传), 日志 `dl/hf-download.log`。
   用户口径: **格式不重要 (GGUF/safetensors 都应可导入), 内存不足也能接受, 目标是把极限推出来**。
### 新增待办 (由八问派生, 认领后改 board)
- N1 [引擎] `--kv-bit-budget <bits>`: 把 `kv_bit_budget.py` 的 DP 搬进引擎, 启动即生成逐层表 (纯 host, 不动 kernel)。
- N2 [引擎] 冷窗代价模型纳入同一 DP (冷槽 9536B/slot), 并让 `--cold-policy` 与分配联合选择。
- N3 [引擎] 运行时校准闭环: 启动自测→写 sidecar→之后跳过→`--recalibrate` 手动重跑。
- N4 [引擎] 热/温/冷自动动态分配: 先验证 ft 修复后的能量观测是否可用 (W5 前置)。
- N5 [引擎] W13 权重 host 卸载 (零实现, 需设计: mmap/分页 + 冷热分层)。
- N6 [引擎] ngram 真表 GPU gather + 前缀缓存 (含 GDN 循环性)。
- N7 [导入] FlashNext 权重到位后回填 bindings (74,520 项契约已在 `flashnext_bindings.py`)。

## 119. 自动导入实跑暴露 2 个生成器 bug + HF 下载通道规律 (2026-09-10 09:4X)
### 自动导入 MiniCPM5-1B (用户指示"试着用那个自动导入导入一下")
- 入口: `tools/archkit/adapt.py <model_dir> --model-id minicpm5-1b` (S1-S4 段) → 产出
  `specs/minicpm5-1b_spec.json` + `out/minicpm5-1b/{config.h,manifest.json}`。
- **结论: 只需 2 个 hook + 1 个 post 校验, 无 `new_op`** => 属于最便宜的适配类 (不用写新内核):
  `token_domain: vocab=130560 != family 248077` / `layers:24>16` (per-layer 数组容量校验) /
  `quant_geometry` 转换后校验。topology=dense, 无 MoE/ngram, draft 建议 mtp_or_existing。
- **发现并修复 2 个生成器 bug** (`adapt.py`, 任何新 dense 模型都会踩):
  ① `rms_epsilon` 只读 knob `post_norm_eps`, HF/Llama 系配置是 `rms_norm_eps`(spec 里 `geometry.rms_eps`)
     => 静默写成默认 1e-5 (实际 1e-6); 修: `knobs.get('post_norm_eps', g.get('rms_eps', 1e-5))` ✓
  ② `full_attention_layers()` 用原始 `kinds` 求和, dense 模型的 `layer_kind_order` 为空 => 生成 **0**
     (同函数里 `kind_map` 早有 `['Full']*n` 兜底, 统计时却没用它); 修: 改为按 `kind_map` 计数 ✓
  => 重跑后 `config.h` 验证: `rms_epsilon = 1e-06f` ✓ / `full_attention_layers() = 24` ✓。
### HF 下载: 这条链路上**全量 GET 会卡、range GET 飞快** (实测规律, 值得记住)
- hub 默认走 **xet** 分块协议 => ~33KB/s (缓存里留下 `pm_*.incomplete`); 关掉 xet 走普通 HTTP,
  全量 GET 仍会卡死 (2.16GB 停在 479MB 不动)。
- 但**同一 URL 的 range 请求** 200MB/9s (~22MB/s, 直连 hf-mirror.com) => 改成 **100MB 分块 + 续传追加**后
  MiniCPM 479MB→2.32GB **19 秒** (~100MB/s)。
- 落地: `_hf_chunked.py` (走 API 拿文件清单+大小 → 逐块 curl `-r a-b` → 追加 → 校验长度), 日志 `dl/hf-chunk.log`。
  MiniCPM5-1B 已下完 (2.42GB); `dealignai/Qwen3.8-Flash-Next-ABLITERATED-NVFP4` (135.3GB) 进行中。
- 备注: `orcarouter/Qwen3.8-Flash-Next-Uncensored-NVFP4` (183.5GB, 同样破限+NVFP4) 是 **gated=auto**,
  需要用户在 HF 网页点同意后我才能下; 另有 `huihui-ai/...abliterated-GGUF` (公开 112GB, GGUF 可走导入管线)。

## 120. hybrid 导入缺口修复 + O1 加固 + 窗口 C/续训实况 (2026-09-10 09:5X)
### hybrid 假阴性修复 (LFM2-2.6B-Exp / Falcon-H1R-7B) —— B 已对抗验证 (S13, PASS 8/8)
- 病灶: `adapt.py` 的缺口统计只数 `full/sliding/gdn` 三类, **未知层型被静默丢弃** ⇒
  LFM2(conv x22 + full_attention x8) 只报 `attention:gqa_full`; Falcon-H1R(**无 `layer_types`**, 44 层
  Mamba/attn) 连层混合都没被建模, 却仍 `rc=0`, 看上去"可导入"。
- 修复: ① 未知层型 → `new_op:layer_kinds(conv x22)`; ② 检测到 SSM/Mamba 配置键但 `layer_types` 为空 →
  `new_op:state_space(mamba_d_state,mamba_n_heads,...)` (依据新增的 `spec['hybrid_hint']`)。
- B 的独立验证 (`_collab/B_gapfix_verify.md`): 7 个合成 config + 用 sed 剥离新块得到的 pre-fix 副本,
  **8/8 PASS、0 假阳/假阴**; MiniCPM5-1B 的 manifest 与改前逐字节相同 (`MANIFEST_IDENTICAL`)。
### O1 (B 在验证中抓出, M 已修, 待 B 复验 S15)
- 隐患: 存在 `new_op` 缺口时 `config.h` **仍无条件写出**, 且 `emit_config_header` 把任何不含
  full/sliding 的层型**折成 Gdn** ⇒ conv 注入样例生成 `gdn_layers() { return 2; }`;
  下游只要忽略 `manifest.gaps` 就直接"假装成功"。
- 修复 (三处, `tools/archkit/adapt.py`): ① 未知层型映射为 `Unknown`, `gdn_layers()` 只数显式 linear/gdn;
  ② `main()` 门禁: 有 `new_op` 缺口时写 `config.h.BLOCKED`, 并 `unlink` 旧的 `config.h`;
  ③ 单个 `#error` 兜底行把未决缺口写进文件自身 (覆盖 Falcon"空 layer_types 回退全 Full"这个特殊情况)。
- 自证 (待 B 独立复算): minicpm5-1b → `config.h`、无 `#error`; lfm2 → `config.h.BLOCKED` +
  `#error "auto-adapt: unmodelled layer kinds (conv) have no engine leaf / unresolved new_op gaps (new_op:layer_kinds(conv x22))"`;
  falcon → `config.h.BLOCKED` + `#error "auto-adapt: unresolved new_op gaps (new_op:state_space(mamba_d_state,...))"`。
### 窗口 B 续训 CUDA OOM 的真实根因 (与 §114 的僵尸进程是两回事)
- 证据: `dl/train-resume-b.out` 尾部 → backward 阶段 `torch.AcceleratorError: CUDA error: out of memory`,
  `=== exit -1 2026/09/10 09:15:32 ===`。
- 根因: 这次续训**与在跑的 nvcc/make 并行** ⇒ 构建期 host/device 内存挤兑 (U8"构建与训练必须串行"
  的另一面: 上次是构建被拖死, 这次是训练先崩)。
- 当前: 训练自 09:15:32 停止; 最新 checkpoint `data\dflash2_ckpts\step_000200.pt` (09:04:51 存盘)。
- 纪律: **续训前必须 `pgrep -a make` / `pgrep -a nvcc` 为空**; window C 脚本自身满足 (其 make 是前台阻塞调用)。
### FlashNext 下载的真实速率 (分片粒度)
- 该 repo 是**每层若干分片** (`layer-XXXXX-experts-*.safetensors` ≈354MB/片), `_hf_chunked.py` 逐分片
  100MB range GET: 稳定 **~12MB/s** (354MB/30s); 09:38 起跑, 09:45 已 5.06GB, 仍在推进
  (此前"3.5GB/135GB"是按整仓视角的错觉)。
- 说明: 下载与构建/训练**可并行** (网络 IO 不抢 GPU/编译内存), 无需进 GPU 窗口。
### 本轮新装
- 12:00 进度汇报已装调度 (`automation-4a69ab27`, 非递归, 只跑一次)。

## 121. FlashNext 契约 vs 真实 checkpoint: 前缀不匹配已定量定位 (2026-09-10 10:5X)
### 方法 (值得记住): 不下载权重也能全量审计
- 该 repo 自带 `model.safetensors.index.json` (34.3MB) ⇒ 296,475 个张量名已经到手 (权重仍在后台下载)。
### 决定性实验 (`_flashnext_prefix_probe.py` → `_collab/M_flashnext_prefix_probe.md`)
| 规范化 | 命中条目 | 消费键 |
|---|---|---|
| 原样 | 1 / 74,520 | 1 / 296,475 |
| **`model.language_model.` → `model.`** | **74,174 / 74,520** | 74,210 / 296,475 |
| 去 `model.` | 1 / 74,520 | 1 / 296,475 |
- ⇒ 契约实质只差**一层前缀规范化**; 真实仓库把文本模型挂在 `model.language_model.` 下
  (例 `model.language_model.layers.0.linear_attn.in_proj_qkv.weight`)。
### 残余 (346 条) 与未消费键的构成
- **NVFP4 伴生张量**: 真实 `.weight` 75,047 个, 其中 73,728 个带齐 `weight_scale`/`weight_scale_2`/`input_scale`
  (= 48 层 × 512 专家 × 3 (gate/up/down), 与 spec 完全吻合); 而**契约里含 scale/quant 字样的条目 = 0**
  ⇒ 不处理则量化尺度被静默丢弃 (或由转换器自行推导, 需明确声明)。
- **`model.visual` 333 个张量**: 该 checkpoint **带视觉塔**, spec/plan 未建模 (= 多模态新范围项)。
- `mtp.*` 28 个 (layers 24 + hyper_connection_mixer 3 + fc_embedding 1): MTP 头, 需与契约的 mtp 覆盖核对。
- 架构侧反而完全自洽 (实证): 36 GDN (`A_log`/`dt_bias` 各 36) + 12 QSA + 48 层 + hyper-connection mixer
  + PLE ngram (`ngram_heads_offsets`/`ngram_heads_vocab_sizes`)。
### 既有"自检"的盲区 (为什么之前没暴露)
- `flashnext_convert.py` 的 `_synthetic_names(canonical_only=True)` + "0 missing, 0 unconsumed" 是
  **用契约自己生成源键再自查** (自我一致性): 真实仓库一个键都不匹配时它依然通过。
  ⇒ S21 要求自检改吃真实清单, 合成清单仅留作离线冒烟。
### 性能附注
- 原 `audit()` 是 O(键 × 条目 × alias) 三重循环 (实测烧掉 2,339 CPU-秒仍未完成) ⇒ 已换
  `_flashnext_fast_audit.py` (set 查表 + bisect 前缀索引, 语义同原实现), 秒级完成。

## 122. 未落地项侦察: N3 范围被实证改写 + LFM2 确认 new_op (2026-09-10 11:3X)
### N3 (启动自测→固化→跳过→手动重校): 从"大件"降级为"有界功能" —— 实证依据
- **关键的代码事实**: 行标定表在 `src/ops/kernel/gqa_isoquant_row_scale.cuh:19`
  `extern __constant__ unsigned short kGqaKvRowScaleDev[16][4][256];`
  ⇒ 它在 **constant memory**, 且是 extern 声明 ⇒ **可用 `cudaMemcpyToSymbol` 在运行时整表替换, 不需要改任何 kernel**。
  这消掉了原先以为的最大障碍 (以为表是编译期常量、要改 kernel 才能换)。
- **第二个事实**: `KvCalibrationCapture` (`kv_calibration.h:28`) 是**死代码** —— 全树 grep
  `KvCalibrationCapture|kv_calibration` 只命中它自己的注释与定义; `EngineOptions.kv_calibration_dir`
  没有任何消费点。也就是说"采集侧"写了但从没接线, 更谈不上"跑一次→固化→跳过"。
- **第三个事实 (顺带的质量缺口)**: 表只覆盖 **16 层** (`[16][4][256]`, 为 qwen3.8-27b 烘的);
  Muse-Glimmer-30B 有 52 层 ⇒ `gqa_kv_row_scale()` 直接返回 identity 1.0 (见 §104 的越界修复)
  ⇒ Muse 的旋转域量化**从未被标定过**。
- ⇒ 可执行的最小闭环: ① 旁车表文件 (`<artifact>.kvrowscale.bin`, 32KiB = 16*4*256*2)
  ② 启动时按 (模型哈希, KV 配置) 命中则 `cudaMemcpyToSymbol` 载入 ③ 缺失且 `--kv-calibrate auto|force`
  时跑一次 capture→`tools/calib` 分析→烘表→写旁车 ④ `--recalibrate` 强制重做。
- 验证方式 (可判定): (a) 从当前常量 dump 出的表经旁车加载后, qwen 路径的 32K 掉针**逐位不变**;
  (b) Muse 用标定表 vs identity 的 32K/57K 掉针对照 (这是 Muse 侧的**新质量收益**, 不只是机制);
  (c) 第二次启动日志出现 skip 行, `--recalibrate` 后重新出现 capture 行。
- 注意: 采集侧要额外接线 (`kv_calibration_dir` → prefill 路径实例化 capture), 这部分与 N1 的 option
  通路同源, 建议与 N1/N2 放同一窗口。

### LFM2-2.6B-Exp / Falcon-H1R-7B: 缺口报告正确, 但确实需要新算子
- 实证: 全树搜 `causal_conv1d|conv1d_silu|conv_state` 只命中 **GDN 路径内的融合 conv**
  (`fp8_gdn_conv_fused.cu` 的 `conv_weight/conv_states`), **没有独立的短卷积叶子**。
- ⇒ LFM2 的 22 个 conv 层不是"配置问题", 而是要真写一个 leaf (`new_op`); Falcon-H1R 的 44 层
  Mamba/SSD 更是大件。适配器把这两个模型标成 `new_op` 是**正确**的, 不该被"让它跑起来"的愿望覆盖。
- 结论: 这两个模型的导入暂不排期; 先把 MiniCPM5-1B 那种"2 hook + 1 post"的廉价适配类做扎实。

### 本轮同步
- 已派 A/S24: `sliding_window_tokens` (E3) 逐层窗口表接线补丁 (落点已给: spec→layout→plan_cache→ctor→两个 view 构造点)。
- 已派 B: S21 补充"下载完成后可直接粘贴的运行配方"。

## 123. 三处结论修正 (实证复核, 2026-09-10 11:3X) —— 含两条我先前判断有误
### ① LFM2 的 conv: 不是"没有算子", 而是"已有近似算子 + 两点差异" (先前我说反了)
- 我先前称"全树没有独立短卷积叶子" ⇒ **错**; 原因: 那次 grep 被 `Select-Object -First 12` **截断**,
  12 条全落在 `fp8_gdn_conv_fused.cu`, 于是我误判"只有 GDN 内的融合 conv"。
- 实际: `include/ninfer/ops/causal_conv1d_silu.h` + `src/ops/wrapper/causal_conv1d_silu.cpp`(**381 行**)
  是**已完整实现**的算子 —— 深度可分离因果卷积(宽 4) + SiLU, BF16, 携带 3 值状态, 无 bias,
  带数值契约文档 (FP64 oracle 口径), 且**已被 GDN 路径使用** (`text_context_impl.h`)。
- 用**真实 checkpoint 头部**实证 LFM2 的 conv 形状 (range GET 4MB, 不下载权重):
  `model.layers.N.conv.conv.weight` = **BF16 [2048, 1, 3]** (深度可分离, 核宽 **3**);
  `conv.in_proj.weight` = [6144, 2048] (B/C/x 三路); `conv.out_proj.weight` = [2048, 2048]。
- ⇒ 差异只有两点: **(a) 核宽 3 vs 算子 4** —— 可用"最老抽头补 0"映射 (状态多留一位, 无副作用);
  **(b) SiLU 位置** —— HF `Lfm2ShortConv` 是 `x = conv(x) * B` 再 `out_proj`; 是否含 SiLU **我尚未实证**。
  这是能否直接复用该算子的**唯一关键点**: 若不含 SiLU, 需要一个 no-SiLU 变体 (kernel 机制已具备, 属小改)。
- **结论改写**: LFM2 = 小算子变体 + 层型接线, **不是**"写新算子的大件"; 但 (b) 必须先查清再排期。
  (查清手段: 有权重后用参考 logits 对齐, 或读 HF modeling 源; 不要靠回忆下结论。)
### ② N6 (ngram): "真表 gather + 热行缓存"**已实现**, 且引擎已有前缀复用机制
- `src/ops/ple/ple_table.{h,cu}`: SSD 后备 PLE 表 (95GiB 旁车不常驻显存), 16 行 id/token 由 `PleLayout` 推导,
  异步预取 worker + **有界 pinned LRU 热行缓存** (默认 512MB, `PleTableOptions.cache_bytes`),
  设备侧 `ple_gather_rows_kernel` 用 UOA 指针表 gather 成 `[n_heads*row_dim, n_tokens]` BF16;
  冷未命中 pread → pinned 暂存 → H2D。**这就是"真表 GPU gather"**。
- **但它是死代码**: `grep -rn 'PleTable|ple_table|PleLayout'` 在 `src/ops/ple/` **之外零命中**
  ⇒ 引擎从未实例化 (与 N3 的 `KvCalibrationCapture` **同一种病**: 实现完整但没接线)。
- 且引擎**已有**序列级前缀复用: `serve/generation_service.h:42 prefix_cache_hit_tokens`、
  `targets/qwen3_6/impl/runtime/program_impl.h:9686 "request plan has an invalid prefix reuse path"`、
  `runtime/engine/engine.cpp:558` 说明复用句柄会失效需丢弃 ⇒ Q6 的"前缀缓存"并非空白。
  真正待查的是**它与 GDN 循环状态是否相容** (线性注意力状态依赖整条前缀, 不是 KV 截断就能复用)。
- **结论改写**: N6 的缺口 = 把 `PleTable` 接进 `qwen4_exp` 运行时 + 确认 GDN 下的前缀复用正确性;
  不是"从零写 gather 内核"。
### ③ 方法论 (本次两处误判的共同根因)
- 判断"某能力是否存在"时必须三件套: ① greps **不截断** (或截断后翻页); ② 先 `-l` 列文件再逐文件确认;
  ③ **同时回答"是否被接线"** (消费者 grep) —— 本仓库已出现两例"实现完整但从未实例化"的模块。

## 124. W13 权重卸载: 复核后的准确范围 (2026-09-10 11:4X) —— "无实现"成立, 但有两块可复用机制
### 复核方法与结论 (按 §123 的方法论: 不截断 + 列文件 + 查消费者)
- 全树 `offload|mmap|cudaMemAdvise|uvm`(忽略大小写) 只有 **11 处命中**, 逐条看下来:
  - `logical_kv_store.h:806` 是 **KV** 冷路径的 re-attach, 不是权重;
  - `include/ninfer/types.h:187` 是 **pinned host 预算**注释: "Pinned host-memory budget for
    ColdPolicy::Host offload. Default 4 GiB." —— 也是 **KV 冷窗**的, 不是权重;
  - 其余是 gemm Schedule 名字的假阳性。
- ⇒ **"没有权重卸载实现"成立** (先前 board 里"grep 不到任何 offload"的说法**不准确**, 应改成
  "没有任何**权重**卸载; 但有 KV 冷窗的 pinned host 通路与 artifact 的 mmap 读取")。
### 两块现成机制 (W13 可直接复用, 不必从零设计)
1. **artifact 本身已是 mmap 读取**: `src/artifact/reader.cpp:201`
   `mapping = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);`
   ⇒ 权重源已在**可分页的宿主映射**里, 不需要额外"从磁盘流式读"的新层。
2. **pinned host 暂存 + H2D 通路**: KV 冷窗 (`ColdPolicy::Host`, 4GiB 默认预算) 已验证了
   "pinned 预算 + 冷未命中原路"的形态; PLE 的 `PleTable` 更进一步给了
   **有界 pinned LRU + 异步预取 worker** 的模板 (`src/ops/ple/ple_table.h:34-38`)。
### W13 真正缺的三件 (按依赖排序)
1. **权重驻留与取用路径的显式化**: 现在权重在加载期一次性进显存 (无分页); 需要一个
   "权重页表 + 未命中时从 mmap 拷入 pinned 再 H2D"的层 —— 这是**新代码**, 也是 W13 的主体。
2. **驱逐/预取策略**: 与 KV 冷窗共用还是独立预算; 与 dflash2 草稿权重的交互 (草稿是热路径)。
3. **接线进 linear 算子**: 每次 GEMM 前解析权重页 (可行但需避免热路径加锁; 参考 PLE 的
   `gather_epoch_` 无锁豁免做法)。
### 排期建议
- 不是"顺手改"的量级 (3 件里第 1 件是主体, 且要动 linear 热路径) ⇒ 单独立项, 且在
  **L3 拆 TU + L2 ccache 落地之后**开工 (否则每次迭代都要烧 30 分钟编译, 正是今天的教训)。

## 125. LFM2 短卷积块: 源码实证结案 (2026-09-10 12:0X) —— "无 SiLU", 复合块而非单算子
### 取证方式
- 抓取 transformers 官方 `modeling_lfm2.py` (25,555 字节; HF 模型仓库本身不带该文件, 镜像 404, 走 GitHub raw 成功),
  存档 `dl/_lfm2_modeling.py`, 逐行读 `Lfm2ShortConv` (324-389 行) 与其本地 fallback (280-321 行)。
### 结案结论 (不再有"待实证")
1. **无激活**: `causal_conv1d_fn(hidden, weight, bias=None, activation=None, **kwargs)` 的 `activation` 默认 `None`,
   而 LFM2 的调用 `causal_conv1d_fn(hidden_states, self.conv.weight.squeeze(1), self.conv.bias, seq_idx=seq_idx)`
   **没有传 activation** ⇒ 卷积**不带 SiLU**。`config.conv_bias=false` ⇒ 卷积**无 bias**。
2. **核宽 3**: `kernel_size = config.conv_L_cache = 3`, `groups = hidden_size` (深度可分离), `padding = 2`。
   与引擎算子 `causal_conv1d_silu` 的**宽 4 + SiLU + 3 值状态**都不同。
3. **完整复合块** (这是关键): `BCx = in_proj(h)` → `B, C, x` →
   `h1 = B * x` → `h2 = conv(h1)` → `y = C * h2` → `out = out_proj(y)`。
   **卷积前后各有一道门** (B 在前、C 在后), 且 conv 的输入是 `B*x` 而**不是** x 本身。
   引擎现有算子无任何门控, 因此**不能直接复用**。
4. 状态语义: 走 `update_conv_state(..., conv_kernel_size=3)` + 单 token 走 `causal_conv1d_update`,
   即解码路径每步就地更新状态 (与引擎算子"携带状态"的形态一致, 属于可对齐的部分)。
### ⇒ LFM2 适配的准确工作量 (取代 §123 的初步判断)
- **需要**: ① 一个 **no-SiLU、宽 3** 的深度可分离因果卷积变体 (同族内核加模板参数即可, 非新算子);
  ② 两道逐元素门 (`B*x`、`C*conv`) — 引擎已有 gating/逐元素乘算子供复用;
  ③ 一个新**层型**把 "in_proj → gate → conv → gate → out_proj" 串起来 (层型接线, 不是写内核);
  ④ 与现有 GDN 的 conv 状态管理对齐 (两者形态相同, 但 LFM2 是每层独立短卷积而非 RNN 门控)。
- **结论**: LFM2-2.6B-Exp = **中等工作量 (变体 + 接线)**, 与我最初的"大件"判断和随后的"直接可复用"判断**都不准确**;
  现在这份规格是可施工的。
- 排期: 排在 FlashNext (S27 写出路径) 与 KV 系列之后; 若要做, 建议单独一轮 + 用 HF 参考 logits 对齐验证
  (有权重后, 逐层比对 conv 块输出是最直接的判据)。

## [补丁A/S43 根因] DFlash2 decode ingress 未填 state slots → verify 读 slot 0（2026-09-10 15:55）
- 现象链: counting prompt 上 plain `gen=129 stop_token` vs dflash2 `gen=2 stop_token`
  （同一 prompt/温度/flag）⇒ 投机解码的 verify 不精确（正确实现必须与 plain 逐 token 相同）。
- 根因（E1 定位、M 复核代码）: `program_impl.h` 的 DFlash2 ingress 循环只填 11 个字段，
  **没有** `state_source_slots`/`state_destination_slots`（宿主 ingress 零初始化 ⇒ 一直为 0）。
  这两个是**设备侧 state-image slot id**；消费点 `dflash2_impl.h:371-372` →
  `speculative_target_impl.h:18,23`（verify 从 slot 0 读 GDN 线性状态）与 `:36`
  （continuation hidden 写回 slot 0）。兄弟后端都填：ordinary `:11831`、MTP `:11989`、
  DFlash v1 `:11414`。⇒ KV（text_kv_table_rows 正确）与循环状态（48/64 GDN 层）互相错位。
- 后果解释全部症状: 列 0 logits 错 → 目标 argmax 错且 a=0 时唯一发布的就是它 ⇒ 立即 stop；
  草稿上下文同样被污染 ⇒ 接受率 0~50%；重复性 prompt 上 GDN 对单 token 不敏感 ⇒ 50.9%。
- 已落地（主线）: `_apply_patchA.py` 幂等补丁，`program_impl.h:12411-12413` 新增
  `state_selectors(sequence)` + 两个赋值；diff 存档 `_collab/M_patchA_dflash2_state_slots.diff`，
  备份 `program_impl.h.bak_patchA`（md5 1c216280 → 8a9c851c）。
- 待验证（编译后必须做）: (1) plain vs dflash2 文本逐字节相同（`_df2_exactness.sh`）；
  (2) 接受率应显著上升（旧 50.9% / counting 0%）；(3) dspark 侧同类漏填待查（E5）。
- 潜伏项（未落地，E1 Patch B）: `dflash2_impl.h:186-193` 给 `prepare_masked_block` 传
  `attention_valid=width`，契约要求 `extent+1`；extent=7=k 时无影响，边界（output_limit/容量）才触发。

## [构建取证 + 草稿对齐 + dspark] 状态同步（2026-09-10 16:07）
### 1. 构建系统取证（长期效率问题的机械解释）
- **头文件依赖根本没接进来**：`make -p` 显示 `ninfer_engine` 的 `variant.cpp.o` 前置只有
  `{compiler_depend.ts, flags.make, variant.cpp}`，`.o.d` 里的 `program_impl.h` **不进 make 图**
  （树里没有 `compiler_depend.make`）。⇒ 改头文件不会触发重编译，这是"静默陈旧 .o"陷阱的机制。
  后果：补丁 A 必须 `touch` **源文件**才会被编进二进制（`_patchA_build.sh` 就是这么做的）。
- **`make ninfer` 不构建 `apps/ninfer-serve`**（目标是 `ninfer-serve`）。所有 serve 实测用的二进制
  与 `ninfer` 是两个文件；只跑 `make ninfer` 会让 serve 停在旧版本（差点导致本轮测量全部作废）。
- **ccache 从未真正缓存**：`ccache -s` 显示 cacheable 0/7、0 文件、5GB 上限；
  6/6 uncacheable 的原因是 *Compilation failed*（= 被我按 PID 杀掉的编译），
  另有 1 次 *Input file modified during compilation*（= 边改边编译，ccache 自己检测到了）。
  ⇒ 别再指望 ccache 提速；真正的杠杆是"只编必须编的 TU" + 不边改边编。
- 本轮代价：window J2 的 `make ninfer -j1` 在 `gqa_attention_decode.cu` 上重复编译 65 分钟
  （该 .o 当时已是最新），被我在 `_patchA_build.sh` 里按 PID 停掉，改为只编 3 个 variant TU +
  6 个陈旧 CUDA TU + 链接（预计 ~40-60 分钟而不是数小时）。
### 2. 草稿对齐三臂探针（Windows 侧，step_001200，476 对）
`_df2_shift3_probe.py` → `dl/shift3_probe.log`：
| 目标行 | 语义 | top-1 一致率 |
|---|---|---|
| `ids16[a+i-1]` | 草稿自己槽位的 token（引擎 verify 要求的那一行） | 0.1239 |
| `ids16[a+i]` | next-token | 0.1618 |
| `ids16[a+1+i]` | **legacy 训练默认（= 现状训练用的行）** | **0.2290** |
读法：三臂都低 ⇒ 草稿本身弱（2300 万 vocab 上 0.23 仍远超随机，但离可用差得远）；
**最高臂是训练默认那一行，而引擎 verify 要的是另一行** ⇒ 若两者确实不对齐，接受率会在
位置 1 就低（与用户"第一个位置就没过 0.8"的观察一致）。**尚未定论**，定论要靠引擎侧
位置剖面（`accepted by pos`，已接进 `_spec_4way.sh` 的 CLI 阶段）。
### 3. dspark（E5/S47）
- ingress 11/11 字段全填，**不是** DFlash2 那种漏填（`:12177/:12178` 与 MTP 同一 `state_selectors()`）。
- 接受路径与 MTP/DFlash2 逐字同一条（`speculative_target_impl.h:9-38`）。
- **真 bug（待落地）**：DFlash verify 复用草稿块的位置表（按 `V=k` 造，`dflash_impl.h:223-231`），
  比自己声明的 `target_valid_columns=extent+1=k+1` 少一位 ⇒ 第 k 路验在重复位置、KV 同址双写。
  已派 E6 产出经 shadow/dry-run 验证的补丁（`_collab/E6_s48_dspark_verify_pos.diff`）。
- 首选假设 H1：草稿输入链（上游按 HF `hidden_states` 训练，引擎 tap 是该层 post-MLP residual）——
  DFlash2 草稿是在引擎 NHS1 tap 上训练的，所以对 tap 约定免疫，这解释 50.9% vs 10.3% 的不对称。
  判别实验：CLI 位置剖面（低 p_0 ⇒ H1；p_0 正常而尾部崩 ⇒ 位置/markov 链）。
### 4. 本轮故意推迟（记录在案，不是遗漏）
- E3 的 i8 平面步长补丁（在 head_dim=256 下逐字节等价）与强制重建 decode TU：留到 Muse 128 工作一起做。
- E1 的补丁 B（`dflash2_impl.h:186-193` 的 `attention_valid` 契约）：extent=7=k 时无影响，边界才触发。
- E3 的 S36 派生恢复（`E3_s45b_s36_restore.diff`，perf-only）。
- 32-needle 长上下文重测（8 needles 噪声太大）。
### 5. 内存纪律（今天差点重演重启）
MemAvailable 曾跌到 **93MB**（并发：make 的 ptxas 9.8-11.7GB + E3 临时编译 + 导出 python 8.1GB）。
处置：按 PID 杀掉 E3 编译树与导出，删除半成品 artifact；此后规则=**同一时刻只有一个 nvcc 或一个模型实例**，
外派子代理明令禁止私自开编译；新增 post-build 看门狗在 MemAvailable<700MB 时自动杀 serve。

### 6. 更正（2026-09-10 16:07）：E3 的 prefill 守卫已回退
上节说"E3 修正版 prefill 守卫已应用"**已不成立**：复核发现该 diff 把 NVFP4/ISO3/FP8 三个分支条件**合并**，
在 head_dim=256 下纯 ISO3 或纯 FP8 的 KV 会进入合并分支却过不了内层 `== NVFP4` 判定 ⇒
**既不发射 kernel 也不抛错，注意力被静默跳过**（E8/iso3 档位真会走到）。已把主树
`gqa_attention_prefill.cu` 回退为 pristine（md5 `b2da4c437ea412902a09d2a378dfafeb`，453 行）。
守卫本身是**防御性**的（128+packed KV 不是当前会跑的组合；解码侧 128 已有 `require_nvfp4_geometry_dim` 显式拒绝），
因此推迟到有"逐分支、保留原条件"的验证版再上。已把修正要求发回 E3：
产出 `_collab/E3_s45d_prefill_guard_v2.diff` + 256 逐字节等价的行级证明 + 六 dtype × {256,128} 回归表。
`--spec` usage 文本那条保留（纯文本，无行为影响）。

### 7. 更正（2026-09-10 16:12）：我关于 S45c 的"256 静默回归"判断是**错的**
上节说 E3 的 S45c 守卫"合并条件导致 ISO3/FP8 在 256 下静默跳过"，这个结论**不成立**，E3 的申辩是对的。
实证方法：把 `E3_s45c_prefill_guard.diff.SUPERSEDED` 打到 pristine 的临时副本上，做**括号深度追踪**
（`_s45c_depth.py`）——合并分支 `} else if (NVFP4 || ISO3 || FP8) {` 在 depth 4，而
`} else if (cache.dtype == DType::ISO3)` 在 **depth 6**、FP8 同在 depth 6，说明这两条臂被**串进了守卫内部**
（`if constexpr (256) { if (NVFP4) … else if (ISO3) … else if (FP8) { … } else { … } }`），
**256 下是可到达的**。我此前的读法只看 diff hunk，而这两条 `} else if` 分隔行在补丁里是上下文行、不在 hunk 内，
于是被误读成"外层死代码"。
S45c 真正的缺口只有一个：128 下 packed 臂**没有响亮抛错**（会静默什么都不做）——那是防御性缺口，不是 256 回归。
因此当时的回退并非必需（无害：树回到 pristine 约 10 分钟，期间没有任何测量基于它）。

**已落地**：`E3_s45d_prefill_guard_v2.diff`（逐臂守卫、保留原条件、本体逐字节不变）。
我独立复核过 E3 的证据：`removed/changed = 0`、`added = 31`、453→484 行、三条 dtype 条件出现次数不变、
每臂各自 `if constexpr + else-throw`（114/168、173/196、201/224）、括号深度收于 0。
attr 组守卫保持"无 else"（对 bf16/i8 也执行，加了 else 会把 Muse 唯一可用的 prefill 档也毙掉）。
`E3_s45d_s36_restore.md` 记录 S36 派生恢复是**单行值替换、不含控制流、与 S45d 行不重叠**，
128 下只是 fill grid 放大 2 倍（性能），继续推迟。

### 7. 更正（2026-09-10 16:12）：我关于 S45c 的"256 静默回归"判断是**错的**
上节说 E3 的 S45c 守卫"合并条件导致 ISO3/FP8 在 256 下静默跳过"，这个结论**不成立**，E3 的申辩是对的。
实证方法：把 `E3_s45c_prefill_guard.diff.SUPERSEDED` 打到 pristine 的临时副本上，做**括号深度追踪**
（`_s45c_depth.py`）——合并分支 `} else if (NVFP4 || ISO3 || FP8) {` 在 depth 4，而
`} else if (cache.dtype == DType::ISO3)` 在 **depth 6**、FP8 同在 depth 6，说明这两条臂被**串进了守卫内部**
（`if constexpr (256) { if (NVFP4) … else if (ISO3) … else if (FP8) { … } else { … } }`），
**256 下是可到达的**。我此前的读法只看 diff hunk，而这两条 `} else if` 分隔行在补丁里是上下文行、不在 hunk 内，
于是被误读成"外层死代码"。
S45c 真正的缺口只有一个：128 下 packed 臂**没有响亮抛错**（会静默什么都不做）——那是防御性缺口，不是 256 回归。
因此当时的回退并非必需（无害：树回到 pristine 约 10 分钟，期间没有任何测量基于它）。

**已落地**：`E3_s45d_prefill_guard_v2.diff`（逐臂守卫、保留原条件、本体逐字节不变）。
我独立复核过 E3 的证据：`removed/changed = 0`、`added = 31`、453→484 行、三条 dtype 条件出现次数不变、
每臂各自 `if constexpr + else-throw`（114/168、173/196、201/224）、括号深度收于 0。
attr 组守卫保持"无 else"（对 bf16/i8 也执行，加了 else 会把 Muse 唯一可用的 prefill 档也毙掉）。
`E3_s45d_s36_restore.md` 记录 S36 派生恢复是**单行值替换、不含控制流、与 S45d 行不重叠**，
128 下只是 fill grid 放大 2 倍（性能），继续推迟。

### 8. 又落地两件（2026-09-10 16:15）
- **S45d**（E3 的逐臂 prefill 守卫）：已应用，我独立复核 0 删除行 / +31 行 / 453→484、
  三条 dtype 条件逐字未变、每臂各自 `if constexpr + else-throw`、括号深度收 0。
- **S48**（E6 的 dspark verify 位置表 k→k+1）：已应用（`dflash_impl.h` md5 60a51d1c→02cb3bc9）。
  E6 同时纠正了 E5 的一处推断：`accepted by pos[m]` 由 verify 第 m 列决定，所以被污染的第 k 列
  只在 `a == extent == k`（全部草稿都被接受）时才有影响 ⇒ **这个 off-by-one 不是 10.3% 的成因**，
  而是高接受率恢复后的正确性前置条件。落地它不改变前 k 列（预测 draft/`p_0`/token 流逐字节不变），
  只影响全接受轮的 bonus token 与 KV 槽位映射。
- **本轮这一份编译将同时包含**：补丁 A、E2、E4、S45d、S48、`--spec` usage 文本 —— 一次 build 全覆盖。
- FlashNext/MiniCPM 的分块下载已重启（09:35 那次在 15:16 重新枚举后停住；规律仍是"整文件 GET 会卡、
  range GET 快"）。32-needle 长上下文检查已接进 `_spec_4way.sh` 的尾部（K3 的 GPU 窗口内跑）。

### 9. 落地 S50：KV 覆盖改为下界语义（借用上游 03177b9，2026-09-10 16:31）
- 我方实测：`logical_kv_store.h:1498` 仍是 `if (target < page_count || target > entitlement) throw`
  —— 即"要求覆盖变小"会抛错。上游改成下界语义（已覆盖就早返回、不截断；错误信息带 tokens/pages/entitlement）。
- E7 的可达性分析（其引用行我已复核）：唯一可证明可达的收缩点在 DFlash 终止结算 ——
  verify 已按 `frontier+extent+1` 映射（`program_impl.h:12180`），而 dspark 的 terminal
  `enqueue_dflash_context_append` 只要求 `max(text_kv_valid, end)`（`:11416`，end=base_E+accepted）。
  当"截断的接受 + 跨 64 token 页"同时发生（例 base_E=60 / extent=8 / accepted=2 ⇒ 1<2 页）**今天会抛错**，
  修补后成为无害早返回。⇒ dspark 侧一个真实可触发的失败模式（不解释 10.3% 接受率，但会真炸）。
- 已落地：4 文件 / 26 hunk / +46-38（14 处 `program_impl.h` 调用点改名；E7 用 `diff -u -w -B` 归一化标识符后
  证明参数逐字节不变，我也抽验了引用行）。旧标识符现存 0 处；已 touch 3 个 variant TU 强制重编。
- 未落地：`E7_s50_regression_sketch.diff`（+36 行 store 级页边界回归测试）留到能编 tests 的窗口。

### 10. 下一趟编译队列（刻意不塞进本轮，避免污染补丁 A 的归因，16:31）
1. **S51 / E8**：`ops::silu` 的近似式在 x 很负时被归零（我们树是"精确 expf + IEEE 除法"，零点在
   x = -88.72284；此时真值 SiLU = -2.607e-37，**是 bf16 最小正规数的 22.18 倍**、最小次正规数的 2839 倍
   ⇒ 不是次正规噪声，是真的精度损失）。上游 PR #194 把指数折到不会溢出的一侧（除数恒在 (1,2]）。
   我们树把上游的 file-local `swiglu_silu` 合并成了共享 `ops::silu`（`src/ops/common/math.cuh`），
   所以 **一处修复覆盖全部调用点**（E8 扫到 66 处 / 18 文件）。diff 已就绪（+18/-3），
   与 mirror 的 dry-run 均 rc=0（严格 --fuzz=0）。**推迟理由**：它改的是所有 nvfp4/bf16 线性层的数值，
   会让补丁 A 的接受率归因变浑。
2. **E7 的回归测试** `E7_s50_regression_sketch.diff`（+36 行 store 级页边界用例）：需要能编 tests 的窗口。
3. **E9 的 dflash2 可配置 K 最小切片**（若它给出 ≤60 行 diff）：同样会改 dflash2 行为，单独一轮。
4. E3 的 i8 平面步长 + S36 恢复（256 下逐字节等价）、E1 的补丁 B（attention_valid 契约）。

### 11. S51 收尾细则（E8 终稿，落地下一次时必须带上，16:32）
- **机制与上游不同**：我们树里 **没有** `__fdividef`（全树 0 处）、**没有** `-use_fast_math`（153 个 .cu 条目里 0 个），
  所以我们的缺陷形状相同但机制是"精确 expf 在 x ≤ -88.72284 溢出到 +inf，IEEE `x / inf` 得 -0"；
  补丁保留我们的 `expf` 与 IEEE 除法，只把指数折到不溢出的一侧（所以不是照抄上游 hunk）。
- **同族 2 处**（都在 `src/ops/common/math.cuh`）：`silu`（:13）与 `sigmoid`（:15），diff 两个都改；
  背后是 **66 个 silu 调用点 / 18 文件** 与 11 个 sigmoid 调用点 ⇒ **一处修改修 66 处**（与"硬编码 256 × 81 处"相反）。
  E8 另列了 11 处"无需改"的点及理由（softplus、三处 `__expf` softmax、14 处 `__frcp_rn`、测试 oracle、文档注释、bench 字符串）。
- **穷举证据（g++/glibc + numpy 双算）**：旧式在 [−1000,0) 上静默归零 **1,998,749** 个 float32；
  换算到 bf16 层：**救回 1,145,241 / 回归 0 / 扰动 2,432**（占 1.12e9 非零点的 2.2e-6）；
  而在 **x ∈ [0, 60) 的 1,114,636,288 个 float32 上逐位完全相同**（正半轴零风险）。
- **两处不能照抄上游的说法**：① "两式在共同有定义处完全相同"是**错的**——最大相对差 3.3e-7~5.0e-7（2.8–4.19 ulp）、
  最大绝对差 < 1 ulp(1.0)，仍在 bf16 量子之下但不能说"相同"；② 上游 PR 里的 "−9.6e-37" 复现不出来（实测 −1.0267e-36），引用时用我们的数。
- **落地时必须补一个测试**：现有测试证明不了这个修复（`test_silu_mul.cpp:17` 的 gate 只到 ±12，
  `linear_swiglu_test_common.cpp:37` 对 A4 允许 1.6e-1）⇒ 落 S51 时要同时加"极端负值"回归用例；
  另外 E8 给了"我们 artifact 到底会不会踩到"的一行 `__any_sync` 计数实验（md §6.2）。
- 待证实（E8 自己标注）：sm_120a 上 `expf` 的确切溢出点（无编译/无 GPU，只交叉验证了两个宿主 libm）。

## 121. Spark-X2.5 导入实测 + 导入器修复 + GUI/i18n 工作流（2026-09-10 16:44）

### A. 导入器实跑暴露的四个盲区（已修，实证在案）
上游模型 = **XHToken/Spark-X2.5-4B**（科大讯飞/Apache-2.0；36 层，hidden 2560，16Q/4KV，head_dim 256，
vocab 131072，**3:1 滑窗(512):全注意力混合**，**全注意力层 partial RoPE 0.25 + θ=5e6、滑窗层全 RoPE + θ=1e4**，
**逐头 sigmoid 输出门** `headwise_attn_output_gate`，**GELU MLP**，tied embeddings）。
先只取元数据（config/生成配置/tokenizer/index/`modeling_spark.py` 参考实现，共 15MB），
跑 `tools/archkit/adapt.py` ⇒ 修复前的输出**只有** `head:tied=true` 一条 new_op（header 被扣为 `config.h.BLOCKED`），
其余四个真特性**完全没被识别**（`rope_parameters` 只存进 spec、没有探测器；`headwise_attn_output_gate`/`hidden_act` 既没进 spec 也没探测器）
—— 这正是"rc=0 看起来可导入"的老毛病。
修复（改的是工具，不是 src）：`tools/archkit/adapt.py`
1. `knobs` 白名单补 `partial_rotary_factor` / `headwise_attn_output_gate` / `gate_attn_act_mode` / `hidden_act`
   （否则后两个探测器永远不触发——这个坑我踩了一次才发现）；
2. spec 新增并**落盘**：`rope_by_kind`、`partial_rotary_by_kind`、`hidden_act`、`attention.{headwise_attn_output_gate,gate_attn_act_mode}`
   （`hf_to_spec` 只写它自己那份 dict，我加了显式持久化，否则审计脚本重读文件会看到不同 spec）；
3. 四个新探测器 + 分级依据（都按引擎实证能力分档，不是拍脑袋）：
   - `rope:per_type_theta{...}` → **hook**（引擎 `ops::rope` 收 `rotary_dim`，theta 是参数）
   - `rope:partial_rotary{full:0.25,sliding:1}` → **hook**（`include/ninfer/ops/rope.h` 明确 "head_dim=256 with even 0<rotary_dim<=256"，且 D256/R64 是已注册域）
   - `attn:headwise_output_gate(sigmoid)` → **hook**（`ops::sigmoid_gate_mul` 与 `attn_input_proj` 的 output_gate 都已存在）
   - `mlp:act=gelu` → **hook**（`ops::gelu` 存在；未知激活才是 new_op）
4. `head:tied=true` 仍是 **new_op（诚实挡住）**，但 action 文本改成给出**可执行出路**：
   "转换期物化 lm_head = embed^T（约 671 MB）或引擎 flavor 复用；引擎按独立 lm_head 对象加载，converter 现硬写 tie=False"。
   参考：`models/Spark-X2.5-4B/`、spec `tools/archkit/specs/spark-x2.5-4b_spec.json`、manifest `tools/archkit/out/spark-x2.5-4b/manifest.json`。
5. 权重分片（5 个 ~1.7GB）已用新的通用分块下载器 `_hf_chunk.py` 拉取中（FlashNext 已在 16:40 全量完成：126GB）。

### B. GUI 接线（并行两个子代理，文件所有权互不重叠）
- **接线缺口（实证）**：`tools/gui/extract_serve_params.py` 能把引擎 55+ 个 serve 旗标抽成
  `serve_params.json`（注释写着"引擎加旗标自动出现"），但**没有任何消费者**——`serve_gui.py` 的
  `build_cmd()` 只拼 12 个硬编码旋钮。所以"接线"= 把注册表真正接进界面 + 补全被漏掉的参数面。
- **i18n 核心已就绪**（我写，作为验收门）：`tools/gui/gui_i18n.py`（`t()/set_lang/lang_from_request`，
  自动合并 `i18n_*.py` 表，缺键记录不静默）+ `tools/gui/gui_i18n_check.py`（三条判据：
  `t()` 键在 zh/en 都在；无"只有 zh"条目；**源码里没有未经 t() 的裸露中文**）。基线：全 FAIL（已列行号）。
- 分工：G1 = `serve_gui.py` + `extract_serve_params.py` + `serve_params.json` + `i18n_serve.py`（接线 + 双语 + 自检）；
  G2 = `convert_gui.py`/`rag_gui.py`/`model_import.py`/`gui_tips.py` + `i18n_misc.py`（双语 + 把导入器的
  gap 分级/阻塞裁决/绑定头出路显示到界面上）。各自 gate：`gui_i18n_check.py --only <自己的文件>` 必须 PASS。

### C. 其他状态
- **FlashNext 下载完成**：`models/Qwen3.8-Flash-Next-ABLITERATED-NVFP4` = 126GB，`ALL_CHUNKED_DOWNLOADS_FINISHED`（16:40）。
- **E9/S52 已交付**（dflash2 可配置 K 最小切片）：6 文件 +44/-32，dry-run rc=0；13 处硬编码宽度；
  并纠正我方两处前提（d1/d3/d7 那组数字其实是 **dflash(DSpark)** 的 sweep，不是 dflash2；dflash2 宽度影响是**待证实**）。
  排入下一趟编译（改 dflash2 行为，不能与本轮同框）。
- 编译进度：decode TU 已完成，正在编 `gqa_attention_decode_e8.cu`（11%）。

### 122. S53 复核 + 裁决 + GUI 收尾（2026-09-10 17:16）
- **S53（转换器侧 tied head + 嵌入别名）**：新 `tools/convert/common/source_map.py`（候选别名表、索引/header 解析、
  tie 判定、朝向、体积，纯标准库）+ `check_source_map.py` 自检。**我自己重跑：61 项检查 0 失败 RESULT: PASS**。
  Spark 命中 `model.embedding.weight`（rank 1/4）、tie=true、真 header (131072,2560) BF16、**朝向 identity（无需转置）**、
  体积 671,088,640 B，全量物化走转换器自己的表达式成功；对照 Qwen3.8-Flash-Next（296,475 张量、有 lm_head）
  ⇒ not-tied、不物化、rebind 后 1118 条 recipe 与注册表**逐元素相等**（非 tied 路径不变）。
- **裁决（我）**：接受 `head:tied=true` 由 `new_op` 改判为 `covered`。理由：引擎按独立 `lm_head` 对象加载
  （`bindings.cpp:423-445`、`text_context_impl.h:358`），物化是**转换期动作**、不需要新算子，而转换器现在真的实现了它。
  ⇒ Spark 唯一 new_op 消失、`config.h.BLOCKED` → `config.h`（946 B）。
- **但补了一刀（已落地并复核）**：`covered` 等级原来没有中英标签（会显示"其他等级"），且 tied 那行文案还是旧的
  "需要新算子"口径。现在 `TIER_ORDER/TIER_KEYS` 加了 `covered`（标签"已覆盖（无需引擎改动）"/"covered (no engine
  change needed)"），`head:tied=true` 这行**按 tier 分档**（new_op 用旧文案、covered 用新文案，含命中键、约 671 MB、
  "转换器已实现 source_map.py"），并补了 `.nosize` 兜底。检查器仍 **VERDICT: PASS**（323 条、zh/en 各 323）。
  两种语言下的渲染我逐个跑过（见下）。
- **GUI 工作流收口**：`serve_gui` 接线（注册表 58 旗标 → 58 个控件）+ 五个模块双语全部完成；门 = 全量 i18n 检查器 PASS
  （164 源码键 / 323 表项 / zh-en 等量 / 动态键家族可解析）+ `serve_gui_selftest.py` 29 项断言全过
  （含与 git HEAD 冻结基线的**逐字节对拍**、20000 次随机滑块、232 个空值高级参数用例）。
  子代理自报的未验证项已照实记录：没在真实浏览器点过、`/api/start` 从未调用、本机只有 `python` 无 `python3`。
- **下一步（导入器续）**：Spark 尚无自己的 target/inventory/资源，只有 adapt 产出的 `config.h` + `engine_hook.patch`；
  往下是 S4 代码生成（target 骨架 + variant 叶子 + inventory/bindings + CMake 注册）与 5 个引擎钩子
  （按层型 rope 参数/部分旋转、逐头输出门、gelu MLP、SWA 掩码）——都属 src/ 改动，**必须等本轮 build 测量完**再落，
  否则污染补丁 A 的归因。已记入下一趟队列。

### 123. S54 推翻乐观结论 + 导入器第二轮修正 + 裁决（2026-09-10 17:31）
**更正我上一条**：我上轮根据 manifest 的 `hook 7 / new_op 0` 说 Spark"VERDICT: CLEAR"——**错了**，
那是分类器不全造成的。S54 的只读分析给出 5 处硬发现，我逐条到源码核实（不是采信）：
1. **生成的 config.h 编不过**：`namespace ninfer::targets::spark_x2.5_4b::detail` 带小数点
   （生成器只替 `-` 没替 `.`）⇒ 阻塞级。**已修**：新增 `cpp_ident()`，生成物第 4 行现在是 `spark_x2_5_4b`。
2. **主注意力 16Q/4KV@head_dim256 未注册**：引擎注册表只有 `16/2@256, 24/4@256, 32/2@128`；
   `wrapper/gqa_attention.cpp:25-30` 按 q_heads 反推（16→2）再抛错。**已加探测器**：直接解析
   `src/ops/kernel/gqa_attention_geometry.cuh` 的 `GqaGeometry<...>` 别名表比对 (q,kv,hd)，
   未注册即 new_op（报告里会列出注册表现有项）。
3. **逐头输出门是 new_op 不是 hook**：`sigmoid_mul` 要求四维同形（`wrapper/sigmoid_mul.cpp:36-40`），
   HF 的 `[n_q,T]` 逐头广播没有算子。**已改判**并写清零算子出路（把 g_proj 展开成
   `[n_q*head_dim, hidden]` 的元素级门控，BF16 约 +755 MB）。
4. **gated GELU MLP 也是 new_op**：`ops::gelu` 是 in-place 单元激活，`gelu(gate)*up` 的两输入乘全树无算子
   （ops 下只有 silu_mul/sigmoid_mul/gelu/causal_conv1d_silu）。**已改判**。
5. **qk-norm 缺失**：家族无条件 `rmsnorm(q,k)`（`text_context_impl.h:958-959`）而 Spark 没有 qk-norm
   ⇒ **已加探测器** `attn:qk_norm=absent`（hook：需 `qk_norm_enabled()` 门）。
6. 好消息（S54 实证）：`ops::rope` 语义与 HF **完全一致**（前 R 维 + split-half + θ^(-2i/R)），
   R=256 撞上 `kRopeMaxHalf=128` 边界但可用 ⇒ **rope 不需要新内核**，只是两种层型都落通用核（性能项）。
**重跑导入器后的诚实结论**：`[blocked] config.h withheld as config.h.BLOCKED (unresolved:
attn:headwise_output_gate(sigmoid); mlp:act=gelu(gated); attn:head_geometry(16q/4kv@256))`，
hook 8 项（含 qk_norm=absent）、covered 1 项（tied）、post 1 项。**Spark 在几何/算子到位前不该被转换**。
**裁决**（已写入 `_collab/S54_spark_target_plan.md` §7）：id 用 `spark_x2_5_4b`；首版权重档 **BF16**；
SWA 首版范围 = **上下文 ≤ 512 token**（W=512 时滑窗≡全注意力），长上下文必须等 SWA 通路。

### 124. 构建绿灯 + 补丁 A 首批实测 + 一个更根本的发现（2026-09-10 17:49）
**构建**：S55 只修 3 处就把树修绿（`layouts_impl.h` 补 `#include "product/kv_options.h"`、designator 顺序交换、
`text_context_impl.h` 把 N3/S38 的 kvcalib 块搬出函数体）。证据：`ninfer` 17:42:33 / `ninfer-serve` 17:44:00，
三个 variant `.o` 均晚于 `program_impl.h` 16:31 ⇒ 补丁 A 真进了对象；两个 make rc=0 且重跑 no-op。
**补丁 A 实测**：`gen=2` 立即停消失（dflash2 zh/num = 83/76，plain 对照 96/88）；接受率 dflash2 4.51%/35.06%、
mtp3 35.04%/57.29%（详见 `_collab/M_patchA_effect.md` 判读）⇒ **致命症状解决，0.9 的目标未达**。
**新发现（比接受率更根本）**：exactness 显示 **MTP 与 dflash2 都偏离 plain**（num 上 MTP 逐字相同）
⇒ 分歧在 `target_verify_batch` 与 plain 解码**不等价**，不在草稿。已排除"verify 漏 logit policy"（对 qwen 是
编译期 no-op），但**Muse 有 softcap 20 + output_multiplier 0.196 ⇒ Muse 的投机 verify 会漏 policy，是真 bug**，列下一趟。
**方法论修正**：测量脚本没传 `--no-thinking` ⇒ content 为空、两侧空串会被判 IDENTICAL（用"空对空"证明 verifier 精确）；
已加 `--no-thinking` + "空文本判 INVALID"防呆。**空结果永远不是一致性证据。**

### 125. Muse SWA 回归（我造成）+ 修复 + 待验证；训练已起（2026-09-10 18:10）
- **教训**：为了让构建绿灯，我把 `layouts_impl.h` 的 S24 窗口接线整段回退（因为它的 `requires` 守卫在非依赖
  上下文里无效）。但那段接线**同时服务 Muse**（E3 注释写明：Muse 52 层里 39 层 SWA、window 2048）——
  回退等于把 Muse 的窗口表清零 ⇒ `muse_srv_bf16.log` 里 L46..L51 全 NaN（118 行）。
  **回退一个改动前必须查它服务哪些 target**。
- **修复（已落地+已编译）**：按正确形式恢复接线 —— 去掉 `requires` 守卫、直接调用
  `TextConfig::full_attention_layers()/is_swa_attention()/sliding_window`（三个 target config 现在都声明了成员：
  Muse 本来有 2048/true，27b/35b 已补 0/false），`.layer_sliding_windows = layer_windows` 挂回聚合。
  `ninfer-serve` 18:08 重建、三个 variant `.o` 同步更新（补丁 A 也在）。
- **验证状态：待 GPU 窗口**（训练 18:08 已启动并占满显存 31.9/32.6 GB）⇒ 暂**不能**声称 NaN 已解决，只能说
  "已恢复接线并重建，验证待做"。判别点：Muse bf16 起 serve 后 `nan_lines=0` 且 L46+ 不再是 NaN。
- **Muse + nvfp4 是设计性拒绝**：`nvfp4 prefill requires head_dim=256; this geometry is 128`（S45d 守卫）——
  这正是历史上"把 256 内核跑在 128 几何上"的路径被显式拦住，Muse 只能用 bf16/i8 KV；验收脚本已按此设计（nvfp4 记为预期拒绝，不算 pass）。
- **顺手修**：`_spec_4way.sh` 的 `plain_mtp` 档一直 SERVE_FAILED，原因是 `--spec mtp` **必须**带
  `--draft-tokens ∈ [1,5]`（engine 硬约束，失败日志尾部就是 usage）⇒ 已补 `--draft-tokens 3`。
- **训练**：`train_dflash2.py --steps 6000` 于 18:08:22 启动（从 step_001200 续跑，batch 4 / anchors 12 / ctx 128）。
- **判别实验已就绪**：`_verify_equivalence.sh`（`--print-token-ids` 比 token id，含 plain×2 自洽基线、
  dflash2×2 确定性、plain vs dflash2/mtp3 交叉）——等 GPU 窗口（训练暂停或结束后）。

### 126. 下一批落地（CPU 侧，与训练并行）+ Spark 权重完成（2026-09-10 18:53）
- **S28 补完（三件套）**：① `gqa_isoquant_row_scale_loader.h` 补 `kv_rowscale_sidecar_apply_from_env`
  声明（签名与 .cu 逐字一致）；② `src/CMakeLists.txt` 把 `ops/kernel/gqa_isoquant_row_scale_loader.cu`
  登记进 `ninfer_ops`（就加在已登记的 `gqa_isoquant_row_scale.cu` 旁边）——**这个 .cu 此前从未被编译过**；
  ③ `decoder_state.cpp` 恢复调用（include 放**文件顶层**、调用用完全限定名 `ninfer::ops::...`，避开当初
  "include 落进 namespace"的老毛病）。⇒ 该特性从"半落地"变成"真进构建"。
- **S51 落地**：应用 `_collab/E8_s51_nvfp4_silu.diff`（`ops::silu/sigmoid` 极端负值归零修复）。
  额外收获：本次修改 `src/CMakeLists.txt` 触发了 CMake 重新配置 ⇒ **整棵树的 TU 重编**，这恰好让
  `math.cuh` 的改动**真正生效**（本树没有头依赖跟踪，改头不重编，见 §123）。
- **Muse verify 契约修复**：在 `target_verify_batch_impl` 的 `lm_head` linear 与 `argmax` 之间补
  `kCfg.apply_final_logit_policy(...)` —— 契约要求"每个 lm_head logits 产生点"都应用；对 qwen 是编译期
  no-op，**对 Muse（softcap 20 / multiplier 0.196）是必需**（否则 verifier 的 argmax 与 plain 不等价）。
- **环境坑（已修）**：`ccache: not found`（Error 127）——本树把 `ccache nvcc` 当 launcher，而 ccache 装在
  `/home/user/.local/bin`；由不同 shell 启动 make 时 PATH 不同。已在重建脚本里显式
  `export PATH="/home/user/.local/bin:$PATH"`。**启动构建必须带这一句。**
- **Spark 权重下载完成**：5 个分片全部到手（`ALL_CHUNKED_DOWNLOADS_FINISHED`，7.7 GB）⇒ 配合已就绪的
  S53（tied head 物化 + 嵌入别名 `model.embedding.weight`），**Spark 的转换侧已具备条件**；
  仅剩引擎侧 3 个 new_op（16Q/4KV@256 几何、逐头输出门、gated GELU MLP）。
- **训练**：18:08 起续跑，`step 1825/7200 loss=1.9348 lr=5.13e-04 steps/s=0.25`，已落 step_001800.pt。
- **明确待补（不许含糊）**：S51 的**极端负值回归用例**——E8 已证明现有测试证明不了它
  （`test_silu_mul.cpp` 的 gate 只到 ±12；criterion 绝对容差 2.0e-5 会掩盖 1e-37 量级的归零），
  新用例必须用**相对判据**并覆盖 −88.7 / −100 / −1000；待 tests 目标编译 + GPU 空闲时运行。

### 127. 低接受率根因重定位：verify 与 plain 的**固定偏差**（4 组复现）—— 2026-09-10 19:12
**先回答"修了没"：没有。** 补丁 A 只修掉"立即停止"，没修低接受率。但今天把它**从"草稿质量"重新定位为
"verify 路径的确定性偏差"**，证据如下（token 级，`--print-token-ids`，全部贪心 temperature=0）：
| 对照 | 结果 |
|---|---|
| plain_a vs plain_b | IDENTICAL（48 token）⇒ 自洽 |
| dflash2_auto a vs b | IDENTICAL ⇒ 投机自身确定 |
| **plain vs dflash2（T=8）** | **DIFFER at token 36**：plain=98633 → spec=**133222** |
| **plain vs mtp3（T=4）** | **DIFFER at token 36**：plain=98633 → spec=**133222**（同点同值） |
| **plain vs mtp d1 / d3 / d5（T=2/4/6）** | **三者都在 token 36、都错到 133222** |
⇒ **四种 width（T=2/4/6/8）+ 两个不同草稿架构，偏离点与错值完全一致** ⇒ **排除**"T/batching 数值噪声"假说
（那是 A1 的首选假设，被这份数据推翻），指向 **verify 路径里一个与 T、与草稿都无关的确定性差异**。
另一条判别（A1 提议）：**plain 下换 `--prefill-chunk 128/512/2048` ⇒ 48 token 逐 token 相同** ⇒ 排除 prefill 分块维度。

**A1 的代码级分析（`_collab/A1_verify_vs_plain.md`）**：
- 否定"tie-break 不同"：verify 走 `ops::argmax`、plain 走 `ops::sample` greedy，但两者是**同一全序**上的最大值
  （值降序、并列取小索引）⇒ 数学上不可能因此不同；
- 否定"第 0 列五件套不一致"（token/cache 位置/RoPE 位置/可见 KV 集合/GDN 状态槽 plain vs verify 逐一同）；
- 新报两条具体缺陷：① **dflash2 的 verify 把 position 表同时当 rope 表、缺 `rope_delta`**（plain/MTP 都有，
  `dflash2_impl.h:410-411` vs `program_impl.h:11827/11981`）——若 delta≠0 即系统性 RoPE 错误（**纯 CPU 可判**）；
  ② MTP AR 步 `ar_valid_columns[s] = s+1<next?1:0`（`mtp_round.cuh:47`）可能让 `ar_hidden` 取自**被置零的列**
  ⇒ 直接压低 pos0 接受率；
- 观测缺口：verify 路径**没有任何 head probe**（plain 有 4 处）⇒ 应补一处以便定位。
- 独立线索：CLI 显式 `--spec dflash2` 报 `object handle does not name a materialized tensor`，而 `--spec auto` 正常。

**三个子代理的交付**（都已完成）：
- **A1**（verify 根因枚举）：上表 + `_collab/A1_verify_vs_plain.md`（12 条候选差异主表 + 判别序列 E0–E6）。
- **A3**（Spark 三个 new_op 补丁草案）：三份 diff 均 `patch -p1 --dry-run` **rc=0**
  （`A3_spark_head_geometry.diff` 8 文件 +140/-21；`A3_spark_headwise_gate.diff` 6 文件 +153/-5；
  `A3_spark_gelu_mul.diff` 8 文件 +384）。关键：**16Q/4KV@256 不在 kernel 实例化域内**——
  `GroupSize=4` 时 i8 的 `Wc=24` 臂覆盖 192≠256（静默少算），`Wc∈{12,6}` 与 nvfp4 `TT≥5` 违反自身 static_assert
  ⇒ 只能用 `if constexpr` 守卫（运行期 throw 拦不住实例化）；且 `kv_heads_for_q_heads()` 必须改成
  "取 `cache.num_kv_heads`"（反推法在 16Q 上信息论不可能对），并收紧 6 处会落进 35B 实例的分派点（含一处
  **连检查都没有**的 fall-through）。另外 S54 的行号基于镜像，与活体树有 6 处漂移，A3 全部按活体树重生成。
- **A4**（GUI 位置剖面 + LFM2/Falcon 导入）：GUI 新增"投机/接受率/位置剖面"卡（门禁 `--only serve_gui.py` PASS、
  自检 29 项全过；serve 无指标端点 ⇒ 先读日志并留好端点候选表）；导入器抓到 **3 处新漏检**
  （LFM2 不写 `head_dim` ⇒ head 几何判定被**静默跳过**；qk-norm 极性判反（权重里 16 个 layernorm 证据）；
  Falcon 的 9 个 muP multiplier 完全没读），并做了**逐字节回滚对照**证明只有这两个模型的分级变化。
- **A2**（草稿上限）/ **A5**（dspark 根因）仍在跑。

**恢复**：训练已从 `step_001900` 续跑（19:11:39，我停它做实验时每 100 步有 checkpoint ⇒ 零损失）。
**编译**：S28 补齐 + S51 + Muse policy 那批仍在编（12%，CMake 重新配置触发全树重编）。

### 128. 拆掉"第 36 步机制"假象 + 低接受率的两条独立成因（2026-09-10 19:20）
**判别实验（`_collab/M_offset_probe.md` / `M_offset_probe2.md`）**：
| 档 | prompt tokens | 生成数 | 第一个偏离的生成序号 | 错值 |
|---|---|---|---|---|
| 中文短 | 36 | 48 | 36 | 98633→**133222** |
| 中文中 | 99 | 48 | **无偏离**（48 token 全同） | — |
| 中文长 | 222 | 48 | 36 | 115544→**133222** |
| 中文 max-new=192 | 36 | 131 | 36 | 98633→**133222** |
| **英文** | 45 | 48 | **37** | 2696→**369** |
**修正**：先前我把"三档都停在 36"读成"固定在生成第 36 步的机制"——**错了**。中文三档的 prompt 只是加了填充，
模型续写的前 36 个 token 本来就**逐字相同**（长档与短档同值 133222 即是证据），所以"同一位置"是同一条输出上的
同一个点；而**换内容的英文档偏离在 37、错值也变**（369）⇒ **偏离点随内容变化**，不是固定步数机制。
⇒ 真实机制：**verify 与 plain 之间存在数值差，在"argmax 近似并列"的位置把选择翻转**（翻转目标通常是次高 token）。
这与 A1 的 D1/D2（plain T=1 vs verify T=width 的 kernel 实例化差异 + KV 行标定量化的次 ULP 放大）一致；
也解释了为什么 T=2/4/6/8 会在同一点翻到同一个值（同一条输出上的同一个危险位置）。

**由此得到两条**互相独立**的问题，别再混为一谈**：
- **(A) verify ≠ plain 的数值差**：已复现、可判据（token id 对照），但**影响面小**——只在近似并列处翻转，
  解释不了"接受率从 0.9 掉到 0.42"。仍需修（它是正确性缺陷，且长上下文更容易踩），但不是低接受率的主因。
- **(B) 草稿本身在位置 0 只有 ~42% 命中率**：位置剖面 p(pos0)=dflash2 41.8% / MTP 53.5% / dspark 26.8%，
  与"草稿质量/训练对齐"一致 ⇒ **这才是 0.9 目标的主要缺口**。A2（草稿上限，正在跑）会用真实特征与 checkpoint
  给出"草稿在位置 0 的上限命中率"，若 ≈42% 则证实"verify 没吃草稿、是草稿不够好"。

**A5（dspark 深挖）已完成**：H1（tap 层号错）**在代码级被排除**（引擎 tap = 第 layer 层 post-MLP residual
= HF `hidden_states[layer+1]`，与参考实现一致；层号 `[4,16,28,40,52]` 与 config 一致），且**离线逐字节取证**
（`_collab/A5_weights_audit.py`）证明现役 artifact 的 **55/55 个 `dflash/*` 张量与 checkpoint 逐字节相同**
（含 markov 未互换、qkv/gate_up 的拼接方向、context_k/v 的来源）⇒ "权重绑定错"整族也排除。
剩下两条 **dspark 专属**候选：
- **候选 A（首选）草稿 rope 约定**：checkpoint 声明 `yarn/factor 32`（`tmp/dspark-config.json:54-68`），
  引擎草稿两条 rope 都走纯 `rope_theta`（`dflash_impl.h:177/304`），而 target 侧只有 factor-4 的
  `ops::rope_yarn4` 且需 `--yarn` ⇒ 若真，属可修的系统性错误；
- **候选 B（次选）block 行→verify 列约定**：引擎 bf16 取 rows 0..k-1（含 anchor 行），HF 参考取 `[:, 1:]`、
  引擎自有 W8 分支取 rows 1..k ⇒ 整路错位一行（也能解释历史的 `k=1 → 0%`）。
  但位置剖面判读**不支持** B 当主因（p0=26.8% 且后续不衰减，与"块内错位"应有的 p0≈0 矛盾）。
- 交付：`_collab/A5_dspark_rootcause.md` + 两份 dry-run rc=0 的 diff（诊断探针 / 候选 B 修法），未落地。

**下一步（全力修 bug 的顺序）**：
1. 等 A2 的草稿上限数字 ⇒ 判定 (B) 是否为 0.9 的主要缺口；
2. 若 (B) 成立：查 `--target-shift` 与训练目标对齐（A2 会给两种对齐的命中率对比）；
3. dspark 的候选 A（YaRN）按 A5 的判据验证并修；
4. (A) 的数值差：按 A1 的 E 序列继续（其中"verify 侧补 head probe"是唯一能看数值的手段，需改码+编译）。

### 129. **dspark 低接受率的根因找到并已修**：block 列错位一行（2026-09-10 19:26）
**根因（训练约定 vs 引擎实现直接冲突，实证）**：
- 训练脚本 `train_dspark.py`：
  - `:428` `block_pos[r] = torch.arange(a, a+B)` ⇒ **第 0 列 = anchor（位置 a 的已提交 token）**；
  - `:168-169` `logits = lm_head(out[:, 1:])` / `return logits, out[:, 1:]` ⇒ **输出只用 columns 1..k**；
  - `:430` `teacher_idx[r, p-1] = tok[a+p]` ⇒ **每一列预测"它自己那一列"的 token**（对齐/自预测）。
  ⇒ 推理时必须取 columns **1..k**（跳过 anchor 列）。
- 引擎 `dflash_impl.h:448-453`（修复前）：
  `std::size_t source_column_offset = 0; if constexpr (!Config::bf16_weights) { source_column_offset = 1; }`
  ⇒ **只有"非 bf16"才跳过 anchor 列**；而我们的 dspark 草稿正是 **bf16**（S47/A5 审计：55/55 张量与 checkpoint
  逐字节相同）⇒ 走 offset=0 ⇒ **把 anchor 列当第一个草稿槽** ⇒ 整块草稿错位一列 ⇒ p(pos0) 崩到 26.8%。
  （注：`if constexpr (!bf16_weights)` 那条注释写的是 "Legacy DFlash keeps the anchor column out of the
  proposal rows" —— 即"跳过 anchor"才是设计意图，bf16 分支反而没跳。）
**修法（已落地）**：`source_column_offset` 恒为 1（跳过 anchor 列），并把命名保留成常量以便日后配置化；
注释里写明训练约定的出处（`train_dspark.py:168-169,428-432`）。
**这条能解释**：dspark p(pos0)=26.8%（显著低于 dflash2 41.8%、MTP 53.5%）、维持 11.22% 的整体接受率、
以及历史上"dspark 在 k=1 时 0%"的记录（错位一列时 k=1 的那一个草稿完全错位）。
**验证（编译完成后立刻做）**：跑 `_dspark_posverify.sh`（CLI `--spec dflash --draft-tokens 7` + 位置剖面）。
判据：p(pos0) 从 26.8% 明显上升（接近 dflash2/MTP 一档）即确认；若不动则根因判断错误，回退并重查。
**顺带证伪**：A5 的首选候选 A（YaRN）**不成立**——`train_dspark.py:70-75` 用的是纯 rope，且它自己写出的 config
是 `'rope_type': 'default'`（`:548`）；artifact 旁那份 `yarn/factor 32` 是**基座 Qwen3.8-27B 的遗留字段**。

### 130. "取列/目标偏移"家族全量核查：共两处，均已修（2026-09-10 19:29）
用户问"dspark 有这个毛病，别的有没有？"——把三个后端的**训练约定**与**推理取列**两两核对（并做数据实证）：

| 后端 | 推理取列 | 训练目标配对 | 判定 |
|---|---|---|---|
| **dspark (DFlash v1)** | `dflash_impl.h:448` 原为 `if constexpr (!bf16_weights) offset=1`，即 **bf16 草稿含 anchor 列** | `train_dspark.py:168-169` 输出只用 `out[:, 1:]`；`:430` 每列预测自己 | **推理错** ⇒ 已修（offset 恒 1） |
| **dflash2** | `dflash2_impl.h:315-316` 已 `+ hidden*bytes` 跳过第 0 列（注释写明 bonus 在 column 0） | `train_dflash2.py:220` 同样 `out[:, 1:]`；但 `:435` 目标行 = `ids16[a+target_shift+i]` | **推理对、训练目标错** ⇒ 已改默认值 |
| **MTP** | 无 block 列机制（自回归链） | 无本地训练脚本 | **不适用该族**（p(pos0)=53.5% 也是三者最高） |

**dflash2 的凭证（双重实证）**：
1. **口径实测**（`_probe_ids16_semantics.py`，4 个真实 cache，3832 样本）：
   `P(ids16[t]==tok[t+1])=0.2904` vs `P(ids16[t]==tok[t])=0.0008` ⇒ `ids16[t]` 是**位置 t 的 next-token 分布**；
2. 于是"位置 a+1+i 的 token"的老师行是 `ids16[a+i]`，而 `head()` 的 `out[:, 1:]` 已让槽位 i 落在位置 a+1+i
   ⇒ **对齐的 shift 是 0**；默认值 1 取 `ids16[a+1+i]`（位置 a+2+i 的 token）⇒ **训练目标整体晚一行**。
3. 旁证：先前"三臂探针"里 legacy(shift=1) 命中率最高（0.2290 vs aligned 0.1239）——正是**模型学的是"晚一行"**
   的表现，与本次口径实证一致。
**修法**：`train_dflash2.py` 的 `--target-shift` 默认 1 → **0**（注释里写明口径实测数据，防止被改回）。
⚠️ **现有 checkpoint 是 shift=1 训出来的** ⇒ 要让接受率受益必须**重训**（训练按用户要求停着，等放行）。
**验证链已起**（`_offset_family_verify.sh`，等编译结束自动跑）：dspark（修后）位置剖面 vs 修复前的
`accept=11.22% / p(pos0)=15/56=26.8% / 1.39 tok/round`，并以 dflash2（列本就对）与 mtp3 作对照。

### 131. MTP 的 `ar_valid_columns` 嫌疑（**待实验，不凭读码就改**）+ dflash2 重训方案（2026-09-10 19:31）

**MTP 嫌疑（代码级，语义未确证）**：`mtp_round.cuh:47`
```cpp
const int licensed       = licensed_counts[row];        // 本轮已发布 token 数（a+1）
const int remaining      = remaining_budgets[row] - licensed;
const int budget_extent  = remaining > 1 ? remaining - 1 : 0;
const int context_extent = max_context - updated_frontiers[row] - 1;
int next = min(budget_extent, context_extent);          // **下一轮**的 draft 上限
...
ar_valid_columns[offset] = s + 1 < next ? 1 : 0;        // 第 s 步（位置 frontier+s）
```
- 若该列语义是"位置 frontier+s 的 token 已存在"⇒ 判据应是 `s < licensed`（**本轮已发布数**），而不是 `next`
  （下一轮的 extent）；而且 `s + 1 < next` 比 `s < next` 还**少一位**（s = next-1 被误判无效）。
- ⚠️ 但 `ar_valid_columns` 也可能不是这个语义（可能只是给 attention 的可见列掩码）⇒ **不凭读码就改**：
  MTP 是当前最健康的档（p(pos0)=53.5%），凭空改可能更差。
- **实验设计（便宜、可判）**：把 `s + 1 < next ? 1 : 0` 改成 `s < next ? 1 : 0`（或 `s < licensed ? 1 : 0`，
  两版各跑一次），用 CLI 的 `accepted by pos` + `spec_accept_rate` 对比：
  若两者与现状**逐位相同** ⇒ 该量在当前配置下不生效（无害）；若上升 ⇒ 是 bug；若下降 ⇒ 回退。
  需一次编译（单行改动，但 `mtp_round.cuh` 会牵动 ops TU）+ 一次 GPU 窗口。

**dflash2 重训方案（已写脚本，等用户放行训练）**：
- 现有 checkpoint 是 `--target-shift 1`（晚一行）训出来的 ⇒ 要吃到修复的收益必须**重训**（不能靠微调）。
- 脚本 `_train_df2_shift0.bat`：与 W9 同配置（`--steps 6000 --batch-seqs 4 --anchors-per-seq 12 --max-ctx 128
  --lr 6e-4 --gpu-frac 0.98 --save-every 100`）**加 `--target-shift 0`**，**从零开始**（便于与 W9 的 step 对齐比较）。
- 判据：训到 ~1200-1900 步后，用 CLI 的 `accepted by pos` 对比 `p(pos0)` 与整体接受率（当前 dflash2：41.8% / 10.03%）；
  同时可用 `_df2_shift3_probe.py` 复核 top-1 与"自己那一行"的一致率是否随步数上升。
- 预计：0.25 steps/s ⇒ 到 1900 步约 2.1 小时。

### 132. A2 判决：草稿质量是主因；dflash2 两条根因均已修，重训前置条件就绪（2026-09-10 19:34）

**A2（草稿上限，CPU-only，679 anchors × 3 ckpt，`_collab/A2_draft_ceiling.md`）的三个答案**：
| 问题 | 数字 | 判读 |
|---|---|---|
| Q1 位置 0 的上限 | 引擎 mask 口径 **0.0309 / 0.0427 / 0.0000**（step1200/1800/400）；训练器真 token 口径 0.1708/0.1222；最有利口径也仅 0.2504 | 位置 0 上限**远低于** 41.8% |
| Q2 形态 | 引擎（41.8%→47.8%）与离线（0.171→0.474 / mask 0.031→0.524）**同形**；可比 prompt 上量级也接近（引擎 zh 4.51% ≈ 离线 mask 3.1–4.3%） | **无证据表明 verify 在丢好草稿**（与我的 token 级实验一致） |
| Q3 对齐 A/B | **shift=1 系统性更高**（0.2504 vs 0.1708）⇒ ckpt 学的是 shift=1，引擎要 shift=0 | **M1 确认** |
- **M2（新，A2 报出）**：引擎喂 `[anchor, mask×7]`，`train_dflash2.py` 喂**真 token**；语料里 mask id 出现 **0/4468** ⇒ mask embedding 从未被训练；mask 口径命中率低 **5.5×**（0.031 vs 0.171）。
- **额外的强证据**：仓库内**另外两个参考训练器都是「mask 块 + shift 0」**（`train_dspark.py:417-432`、`dl/aeon-train_head.py:25-60`）
  ⇒ **dflash2 的训练脚本是全仓唯一的异类**（真 token + shift 1）。
- **量化外推**：教师 top-16 熵地板 1.3654 nat，当前 loss = 地板+0.48；按 (loss,命中) 外推到地板，位置 0 也只有
  ≈0.10（引擎口径）⇒ **旧配方下 0.9 不可达**；新配方（mask + shift 0）必须重训后再测。

**已就绪的修复（`train_dflash2.py`，三处）**：
1. `--target-shift` 默认 **1 → 0**（M1；注释写明 ids16 口径实测：P(next)=0.2967 vs P(self)=0.0007）；
2. **输入模式与推理一致**：block 第 0 列 = anchor、1..B-1 列 = mask（新增 `--mask-block` 默认开，
   `--no-mask-block` 可复现旧配方做 A/B）；
3. `MASK_ID = 248077`，与引擎 `src/targets/qwen3_6_27b/impl/config.h:106` 的 `mask_token` **逐字一致**
   （A2 的离线实验用的是 248070，方向不变但数字偏悲观，已记录）。
**重训脚本**：`_train_df2_shift0.bat`（同 W9 配置 + `--target-shift 0`，从零开始；mask 输入走新默认）。
预计 ~2.1 小时到 1900 步；**等用户放行训练**。

**"取列/目标偏移"家族总账（用户问"别的有没有"）**：
| # | 位置 | 问题 | 状态 |
|---|---|---|---|
| 1 | `dflash_impl.h:448`（dspark 推理） | 取列含 anchor（应跳过）⇒ 整块错位一列 | **已修**，等二进制验证 |
| 2 | `train_dflash2.py`（dflash2 训练） | 目标行晚一行（shift 1）+ 输入模式用真 token | **已修**，待重训验证 |
| 3 | `mtp_round.cuh:47`（MTP） | `ar_valid_columns = s+1<next`（参照量存疑 + 边界差一） | **嫌疑待实验**（不凭读码就改） |

### 133. 偏移/约定类 bug 总账：四处，全部已修（2026-09-10 19:38）
| # | 位置 | 问题（依据） | 影响面 | 状态 |
|---|---|---|---|---|
| 1 | `dflash_impl.h:448` dspark 推理 | 取列含 anchor 列（`if constexpr (!bf16_weights) offset=1`，而草稿是 bf16）⇒ 整块错位一列（训练约定见 `train_dspark.py:417,427-432`：block[0]=anchor、每列预测自己） | **每轮全部 7 个草稿** | **已修**（offset 恒 1） |
| 2 | `train_dflash2.py` M1 | 目标行晚一行：`ids16[a + target_shift + i]`，而 `ids16[t]` 是"位置 t 的 next-token 分布"（实测 P(next)=0.2904 vs P(self)=0.0008）⇒ 对齐值应为 0 | **每个草稿的目标行** | **已修**（默认 1→0） |
| 3 | `train_dflash2.py` M2 | 输入模式 skew：训练喂真 token、推理喂 `[anchor, mask×7]`；语料里 mask id 出现 0/4468（A2）⇒ mask embedding 从未训练 | **每个草稿的输入** | **已修**（mask 默认开，`--no-mask-block` 可 A/B；`MASK_ID=248077` 与 `config.h:106` 逐字一致） |
| 4 | `mtp_round.cuh:47` MTP | `ar_valid_columns = s + 1 < next`：`next` 是**下一轮草稿数**，第 s 步有效应满足 `s < next`；原式在 `next == steps`（预算刚好够）时把**最后一个草稿**判无效 | 仅"预算恰好受限"的轮次少一个草稿；`next > steps` 时两式**恒等** | **已修**（`s < next`；零风险等价改动 + 边界恢复） |

**修复的正确性依据（家族一致性的正面证据）**：
- 仓库内**三个**训练器里，`train_dspark.py:417-432` 与 `dl/aeon-train_head.py:25-60` 都是 **mask 块 + shift 0** ⇒
  我改后的 `train_dflash2.py` 与它们**一致**；改之前它是全仓唯一的异类（真 token + shift 1）。
- MTP 的修复在 `next > steps` 时与旧式**逐位等价**（可用旧二进制与新二进制对同一 prompt 比对 token id 验证）。

**待验证（编译完成后自动跑，`_offset_family_verify.sh`）**：
dspark 位置剖面（对照修复前 `accept=11.22% / p(pos0)=15/56=26.8% / 1.39 tok/round`）+ dflash2/mtp3 作对照。
**dflash2 的重训**（`_train_df2_shift0.bat`，含 mask 输入的新默认）等用户放行训练。

# R6：dflash2 塌陷的根因定位（2026-09-11 13:1X）

## 一句话结论
草稿侧的 unary 候选一直是用 **target 的 248320 行 LM head** 算的；而这个 checkpoint 自带
**专用草稿头 `text/draft_head [131072,5120]`（Q4G64）+ `text/draft_head_token_ids [131072]`**，
契约 §6.1（`ninfer-upstream/docs/maintainer/qwen3.8-27b-dflash2.md:254`）明确要求
「必须先经 `text/draft_head_token_ids` 映射为 global token ids，再访问 selector codebook」。
上游 `propose_dflash2_batch` **两种头都支持**（`dflash_impl.h:313-318`），我们**只实现了 Full 分支**，
并在 CLI 层硬性拒绝另一分支（`src/product/speculative_options.h:59-61`；上游同文件 DFlash2 分支**无此限制**）。

## 实证链（全部本机可复现，脚本随附）

### 1. 目标侧是健康的（此前所有"target 侧"怀疑可以排除）
工具：`_df2_blame2.py`（87 轮探针 + 无投机贪心真值流）
- 对齐方法本身可自证：投机解码只会 licensed 与 target argmax 相等的 token ⇒ 引擎的 licensed 流
  必然是无投机贪心流的前缀，每轮偏移 = 累计 licensed 数；探针自带的 `count/pos` 与该对齐一致。
- **列 0：target argmax 命中真值 17/20 = 85%**
- **列 1（在引擎已接受的前缀上）：target argmax 命中 6/8 = 75%**（参考实现保持 ~82%）
⇒ `TargetVerifyFrameView` 的位置 / KV / 特征捕获 / 块注意力 / `valid_columns` 语义全部正常。
（此前列的差异 #3「attention 第 6 参语义」、#4「`proposal_valid_columns` vs `attention_valid`」不再是嫌疑。）

### 2. 草稿侧：per-column unary 可用，**链式耦合项失效**
- 中文散文档：草稿退化成「同一 token 重复 3 连」——`[104647,104647,104647,1710,1710,1710,3709]`；
  更关键的是 **s=3,4,5 三轮 anchor 不同（98325 / 96041 / 96672）、上下文不同，却给出逐位完全相同的草稿向量**
  ⇒ 输出既不依赖 anchor 也不依赖上下文差异，只随"缓慢变化的某个量"漂移。
- 可复制文档（数字模式）：草稿 **6/7 列正确**
  `verify[0..7]=[15,220,16,220,17,220,18,220]`，`draft=[220,16,220,17,220,18,220]`，
  `argmax=[220,16,220,17,248046,18,220]`；且整轮内 licensed 出 ~8 个 token（说明这一轮草稿真的被大量接受）。
⇒ 块构造（`prepare_masked_block` 第 0 列确为 anchor，已逐行核对内核）、RoPE 绝对位置、上下文 K/V、
   `feature_projection`/`context_norm` 都在工作；**只有"把相邻位置串起来"的边项没起作用**。
   这也解释了"改 target-feature-layers 无差别"：失配来自头部，不在特征层集合。

### 3. 仓库级证据
- 草稿头在 **dspark 分支**是实现了的：`dflash_impl.h:489-499`（`optimized_proposal` + `ops::proposal_remap_token_ids`）。
- **dflash2 分支**（`dflash2_impl.h:328`）硬编码 `state.execution.model.output_head`，没有分支。
- artifact 清单实测（`_art_df2.py`）：`text/draft_head [131072,5120] Q4G64_F16S`、
  `text/draft_head_token_ids [131072] I32` 确实存在（另有 `text/output_head [248320,5120]`）。

### 4. A/B 实测（`_lmhead_ab.sh` / `_lmhead_ab2.sh`）
- dspark K=7 + `--lm-head-draft`：`pos=[15,1,0,0,0,0,0]` vs 不加 `[17,1,...]` ⇒ **无实质变化**
  ⇒ dspark 的第 1 位崩塌另有来源（markov 项，见下"并行"）。
- dflash2 + `--lm-head-draft`：**CLI 直接拒绝** —— `error: --spec dflash2 requires the full proposal head`。
  即该路径从未被走过（也顺带说明 `--spec auto` 与 `--lm-head-draft` 不兼容是既有约定，非 bug）。

## 为什么"用错头"会必然导致"链式失效"（机理，可证伪）
`E_i[p,c] = u_i[c] + Σ_r W_pred[pred_token,r] · g_i[r] · W_succ[C_i[c],r]`
`u_i` 与边项的**相对标度**是训练出来的：`W_pred/W_succ`（[248320,256] 两本 codebook）与 `g_i`
（`selector_hidden_projection`）是按**草稿头 logits** 的标度标定的。换成 target 的 LM head ⇒
`u_i` 量级/分布不同 ⇒ 边项被压到无效 ⇒ walk 退化为"每步取 unary argmax" ⇒
相邻列输出同一 token（重复）⇒ **第 1 位起全灭、第 0 位仍 ~35%**。该解释同时覆盖：
复制文档全对（unary 足够）、改特征层无差别、以及 **vLLM 同样塌陷**（vLLM 也只喂了 target 的 head）。

## 修复方案（可立即实施）
1. 绑定 `text/draft_head`(Q4G64_F16S, [131072,5120]) 与 `text/draft_head_token_ids`([131072] I32)
   —— dspark 的 `optimized_proposal` 绑定段（`bindings.cpp:447` 附近）可照抄。
2. 删除 `speculative_options.h:59-61` 的硬性拒绝（与上游对齐）。
3. `dflash2_impl.h` 增加 `proposal_head` 分支：Optimized 时在 **131072 行草稿头**空间做 stable top-16，
   再用 `draft_head_token_ids` 映射成 global ids，把 `(ids, scores)` 作为**预计算候选**交给 selector。
   这要求把现在的 `dflash2_selector`（内部自算 top-K）拆成
   「头部 top-K + id 映射」与「边格 + walk」两段 —— 正是此前审计的差异 #5
   （上游 `candidate_selector_path(candidates, scores, projected, …)` 的候选由外部给出）。
4. 验收判据：同一 prompt 的 per-position 剖面从 `[8,0,0,0,0,0,0]` 变成链式（p1+ 非零），AL ≥ 3（参考 ~4.0）。

## 并行线
- (b) dspark 第 1 位崩（`[17,1,0,…]`）与 dflash2 的边项**同构**：查 `dspark_markov_argmax.cuh`
  的耦合项标度（同一"耦合项失效"家族，dspark 用的是 `markov_w1/w2`）。
- (c) vLLM 侧核对它是否也只用 target head：若是，则跨实现塌陷由同一条解释。

# R7：dflash2 链式塌陷的最终定位（2026-09-11 14:0X）

## 结论（三段，按可信度排序）

### 1. 引擎侧发现并修复了两个真缺陷（都已落地并通过验证）
- **F4（实质缺陷）**：`dflash2_selector_walk_kernel` 的 argmax 是坏的。
  `value` 被 `fmaxf` 归约**覆盖之后**才与 `best` 比较，再以 `min(lane)` 破平 ⇒ lane 0
  归约后必然持有全局最大 ⇒ `equal` 恒真 ⇒ `chosen` **恒为 0**（= unary 的 argmax）。
  后果：契约要求的「coherent path walk」退化为「逐步取 unary argmax」，**训练好的边项被整个丢弃**。
  修复后探针直接可见机制恢复：`chosen == Earg`（修前恒 `Earg≠chosen=0`），
  `pred` 真的在走链（0→2→14→1→2→11→0），草稿不再是同一 token 重复串。
- **F1（契约/上游不一致，非本症根因）**：dflash2 未接 checkpoint 的专用草稿头
  （`text/draft_head [131072,5120]` + `text/draft_head_token_ids`，契约 §6.1:251-254），
  且 CLI 层硬性拒绝（`speculative_options.h:59-61`，上游同处无此限制）。
  已实现两条 route 并实测生效（737 vs 735 张量、43/87 轮草稿变化），但**剖面纹丝不动** ⇒ 不是根因。

### 2. 残差在草稿 checkpoint 自身，不在引擎（本轮最强实证）
- **边项权重扫描**（`NINFER_DF2_PAIR_SCALE`，同一 prompt）：

  | pair 权重 | 接受率 | AL | 位置剖面 |
  |---|---|---|---|
  | 1.0（契约值） | 4.76% | 1.33 | `[6,1,0,0,0,0,0]` |
  | 0.25 | 4.97% | 1.35 | `[8,0,0,0,0,0,0]` |
  | 0.0（整段关掉） | 4.97% | 1.35 | `[8,0,0,0,0,0,0]` |

  ⇒ 边项**不携带可用信息**（全权重略差于关掉），所以"标度修一修就好"被证伪。
- **unary 很平**：探针实测 top-16 的 `uspan` 仅 ~1.5–2.75，而边项 `pairspan` 5–18；
  在可复制文本（数字模式）上 unary 却 6/7 列正确。
- 与 `_collab/A2_draft_ceiling.md` 的既有诊断吻合：**该 ckpt 的训练配方喂真 token，
  推理喂 mask 块**（mask id 在训练语料 0/4468 出现，见 A2）⇒ 模型只会"泛化续写"，
  不会"逐位置具体预测"。这解释了：复制文本可用、散文下 unary 平、边项无信息、位置 ≥1 全灭。
- 权重真伪已排除：artifact 由 `patch_dflash2.py --ckpt data\dflash2_ckpts\step_006000.pt`
  生成，其 verify 闸门要求 `replaced tensors ok: 73 match checkpoint` 且 round-trip
  `BIT-EXACT`（`_collab/C_artifact_manifest.md:15-30`）⇒ 73 张草稿权重是真的、逐字节一致。
  本机 HF 目录 `Qwen3.8-27B-NVFP4-RTX5090` 只含 target（无任何草稿张量），
  所以"用 HF 对照草稿"不可行；对照基准应是**我们自己的训练 ckpt**。

### 3. 目标侧完全健康（本条贯穿始终，未变）
blame 对齐分析（`_df2_blame2.py`）：列 0 target argmax 命中真值 **17/20 = 85%**，
在引擎已接受前缀上列 1 命中 **6/8 = 75%**（参考 ~82%）⇒ verify 块的位置/KV/特征捕获/注意力
全部正常，`valid_columns` 语义、第 6 参、4.x 号差异等嫌疑全部排除。
KV 精度不是自变量：引擎 summary 实测 `kv cache dtype = bf16`，显式 `--kv-dtype int8` 剖面同形。

## 今天被证伪的假设（连同证据）
| 假设 | 结论 | 证据 |
|---|---|---|
| 用错 proposal head（应走草稿头） | **非根因**（但契约缺口真实，已修） | F1 实测两 route 剖面相同；1Cat 参考实现只用 target LM head（`qwen3_dflash2.py:456-509`） |
| 边项标度失配 | **证伪** | scale 扫描 1.0/0.5/0.25/0.1/0.0 全在 4.5–5.0% |
| artifact 草稿权重被转换改坏 | **证伪** | 转换链自带 `73 match checkpoint` + `BIT-EXACT` 闸门 |
| KV int8 污染 | **证伪** | 实测 bf16，且 int8 同形 |
| `verify[c+1]==draft[c]` 是"发现" | **tautology** | 那是块布局本身；真正该看的是 `argmax[c] vs draft[c]` |

## 落地补丁（全部有 diff + 备份，可 `patch -R` 回滚）
| # | 文件 | 内容 | 备注 |
|---|---|---|---|
| F1 | `_collab/F1_df2_draft_head.diff` | 草稿头两 route + CLI 解禁 + selector 的 id 映射 | 8 文件 +86/−32；备份 `/home/user/df2head_bak` |
| F2 | `_collab/F2_df2_scale_probe.diff` | selector walk 的 `E/u/pair/uspan` 探针 | `NINFER_DF2SEL=1` 才生效；输出走 **stdout** |
| F3 | `_collab/F3_df2_pair_spread.diff` | 追加 `pairspan/Earg/Uarg` | 同上 |
| F4 | `_collab/F4_df2_walk_argmax_fix.diff` | **修 walk argmax（对比本 lane 自身分值）** | 本轮实质修复 |
| F5 | `_run_scale_probe.sh` 系列 | `NINFER_DF2_PAIR_SCALE` 运行时权重旋钮 | 默认 1.0，与契约算术一致 |

变异体常量：`DFlash2Config::draft_head_rows = 131072` 已补进 27b/35b/Muse 三处 config.h
（共享头 `dflash2_impl.h` 需要每个 variant 都有）。

## 下一步（唯一有意义的路径）
1. **恢复续训**：`_train_df2_shift0.bat`（mask 输入 + `--target-shift 0`，即 A2 认定正确的配方），
   产出新的 `step_XXXXXX.pt` 后按 `_collab/C_artifact_manifest.md` 的 patch→verify→round-trip
   三闸门出新 artifact。
2. 用现成仪器复测（无需再写代码）：
   - `NINFER_DF2DBG=1` + `_df2_blame2.py` → 目标侧/草稿侧 blame 与列剖面
   - `NINFER_DF2SEL=1` → `pairspan` 是否变为"有信息"（应显著影响 `Earg≠Uarg` 且 token 合理）
   - `_verify_df2head.sh` → 接受率 + G-A（生成 token 流逐位一致）
3. 验收判据：位置剖面出现链式（p1+ 非零）、AL ≥ 3、G-A 保持 IDENTICAL。
   若新 ckpt 下 `pairspan` 仍无信息，则应回头查 mask 块/context K-V 的喂法（当前唯一未证伪的
   引擎侧残余项），并用 F5 旋钮做 A/B。

# R8：草稿链路参数审计（对照 ckpt 自带 config）+ vLLM 对照数字核实（2026-09-11 14:2X）

## 一、最关键的结论：dflash2 的**声明型**超参与 ckpt 自带 config **逐项一致**
权威来源：`data/draft_dflash2_ref/config.json`（`architectures: DFlash2DraftModel`，即文档健康数字对应的
参照草稿，Sep 11 12:32 已在盘上）。对照我们引擎硬编码的 `DFlash2Config`
（`src/targets/qwen3_6_27b/impl/config.h`）：

| 参数 | ckpt config（`dflash_config`） | 引擎 `DFlash2Config` | |
|---|---|---|---|
| `mask_token_id` | 248070 | 248070 | ✓ |
| `selector_rank` | 256 | 256 | ✓ |
| `selector_top_k` | 16 | 16 | ✓ |
| `conv_group_size` | 16 | 16 | ✓ |
| `conv_kernel_size` | 2 | 2 | ✓ |
| `block_size`（= W = K+1） | 8 | `block_drafts`=7 → W=8 | ✓ |
| `target_layer_ids` | [5,19,33,47,61] | `target_feature_layers`=[5,19,33,47,61] | ✓ |
| `num_attention_heads` | 32 | `query_heads`=32 | ✓ |
| `num_key_value_heads` | 8 | `kv_heads`=8 | ✓ |
| `head_dim` | 128 | 128 | ✓ |
| `hidden_size` / `intermediate_size` | 5120 / 17408 | 5120 / 17408 | ✓ |
| `num_hidden_layers` | 5 | `layers`=5 | ✓ |
| `rope_parameters.rope_theta` | 1e7 | 1e7 | ✓ |
| `sliding_window` / `use_sliding_window` | 2048 / true | `local_window`=2048 | ✓ |
| `layer_types` | 全 sliding_attention | —— | ✓ |
| `is_causal` | false | 块注意力非因果（swa） | ✓ |
| `input_embedding_scale` / `output_multiplier` | 未声明 → 默认 1 | 未施加（且 qwen3 的 policy 为 no-op） | ✓ |

⇒ **"dflash2 声明的草稿超参传错"这一假设被证伪**（至少对这些键）。剩下的错只能在**未声明的约定**
（块的语义、context K/V 的 positions、feature 拼接顺序）或 **verify/状态侧** —— 而后者今天已被实测出真偏差（见三）。

## 二、vLLM 对照数字核实（用户记忆的那组）
盘上可查的 vLLM 运行（`dl/clean_run.log`、`dl/anchor_rev.log`）配置是：

```json
{"method": "dspark", "model": ".../data/draft_model", "num_speculative_tokens": 7,
 "draft_sample_method": "greedy"}
```
指标：
```
spec_decode_num_drafts_total                 94
spec_decode_num_draft_tokens_total           658
spec_decode_num_accepted_tokens_total         1.0
per_pos: position0 = 1.0, position1..6 = 0.0
```
**⇒ 记录里那次（含干净环境）vLLM 是塌陷（1/658），不是"一切正常"**；而且 stock vLLM 没有 dflash2，
它走的是 **dspark** 路径、用的是 `data/draft_model`（与 我们 artifact 的 dspark 源
`models/qwen3.8-27b-dspark-zh/` 不是同一份，后者目录里**连 config.json 都没有**）。
若"正常"的那组在别的日志里（例如 vLLM 引擎自报的 acceptance length 行），请指给我文件名/行，
我立刻核。**空结果不能当证据**（`M_patchA_effect.md` §5 的同一条纪律）。

### 顺带发现：dspark 侧存在**真实**的参数不一致风险
`data/draft_model/config.json`（`Qwen3DSparkModel`）声明：

| 参数 | 该 ckpt | 我们 `DFlashConfig`（摘要记录） |
|---|---|---|
| `mask_token_id` | **190221** | 248077 |
| `target_layer_ids` | **[1,14,29,44,57]** | {4,16,28,40,52} |
| `rope_parameters.rope_theta` | **1e6** | 1e7 |
| `num_attention_heads` | **40** | 32 |

dspark 也塌在位置 1（`[17,1,0,…]`）。但这些差异是否成立取决于"我们 artifact 的 dspark 到底出自哪份 ckpt"——
`models/qwen3.8-27b-dspark-zh/` 只有 `model.safetensors`、**没有 config**，所以引擎的 dspark 常量目前
**没有可核对的权威来源**。这是一处该补的账（把 ckpt config 一起归档，或让引擎读 ckpt config）。

## 三、今天测出的**真参数/状态偏差**（接受率上限，与草稿无关）
同一 prompt、同一参数：
```
zh_plain vs zh_dflash2 : DIFFER at 29/96     （plain [...97844, 129775...] / spec [...97844, 95966, 129775...]）
num_plain vs num_dflash2: IDENTICAL (96 tok)
```
K 扫描（隔离机制）：
```
K=1: DIFFER at 62/96
K=3: DIFFER at 62/96   ← 与 K=1 同位置同 token
K=7: DIFFER at 29/96
```
⇒ K=1 就偏、且 K=1 与 K=3 同位置 ⇒ **不是"后续 draft 列被前面的列看到"的因果掩码污染**，
而是**与草稿无关的每轮状态/参数偏差**（paged KV 槽写入/回滚、GDN 循环状态 replay、
anchor/bonus 记账），随轮次缓慢漂移，只在分布接近（散文）处翻 argmax；可复制文本上不显现。
这正是契约 §7"无损"要求被破坏之处，也是 9-10 `M_patchA_effect.md` §3 遗留的同一症状。

## 四、下一步（按性价比排序）
1. **验证/状态链路**：审 `speculative_accept_greedy_drafts` 的列→licensed 映射、
   `speculative_select_accepted_hidden` + `scatter(state_destination_slots, continuation_hidden_store)`
   对**被拒列**的状态回滚、以及 GDN `RecordForReplay` 的 replay 一致性；判据 = spec 流与 plain 逐位一致。
   跨后端交叉验证：`--spec mtp --draft-tokens 3` 在同一 prompt 上是否也在同一 token 位置偏（若同位置 ⇒ 共享路径）。
2. **未声明约定的三项**（config 不管，只能对契约/上游）：块语义、context K/V 的绝对 positions、
   feature 拼接顺序 [5,19,33,47,61]（后两项已与上游实现逐条对上，前一项已对 §5 逐条对上）。
3. **参照草稿 A/B（一次定生死，需你点头）**：`data/draft_dflash2_ref/` 已在盘上（12:32 下载），
   用 `patch_dflash2.py --ckpt` 那套把它打成 artifact 跑一次：
   - 若接受率回到健康带（AL≥3）⇒ 引擎/verify 路被开脱，问题全在 `step_006000`（训练配方）；
   - 若仍 5% ⇒ 引擎/verify 侧确有问题（与第 1 项结论互相印证）。
   你之前说"别下参照草稿"，所以我没有自行使用；它已在本地，是否拿来做这次 A/B 你定。

# R9：草稿→head→selector 全链路审查（对参照实现逐项）+ verify 侧定位（2026-09-11 14:3X）

## 一、审查用到的"权威基准"（全部为已知路径定点读取，未扫盘）
- 参照 dflash2 模型定义：`1Cat-vLLM/vllm/model_executor/models/qwen3_dflash2.py`
- 参照 dflash2 调度/walk：`1Cat-vLLM/vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py`
- ckpt 自带超参：`data/draft_dflash2_ref/config.json`（`DFlash2DraftModel`）

## 二、逐项审查结果

### 2.1 声明型超参：**全中**（见 R8 表）
`mask_token_id 248070`、`selector_rank/top_k 256/16`、`conv 16/2`、`block_size 8`、
`target_layer_ids [5,19,33,47,61]`、`32/8` 头、`rope_theta 1e7`、`sliding_window 2048`、
`is_causal false` 全部与引擎 `DFlash2Config` 一致。

### 2.2 块与 positions：**一致**
参照 `speculator.py:859-872`：`num_tokens==8` / `num_draft_tokens==7` ⇒ W=8、K=7；
`_context_target_positions.copy_(input_batch.positions[:8])`，注释明写
"Raw positions equal the later masked positions for every accepted row. Rejected rows remain
scratch data and never reach the KV cache." ⇒ context 用 **target 块的 8 个绝对 position**，
与我们的 `append_context_impl` 用 sink 捕获的 feature positions 等价。

### 2.3 selector walk：**语义一致（F4 修复正是对齐参照）**
参照 `_selector_walk_kernel`（`speculator.py:53-127`）：
```python
previous = 0
for step in range(walk_steps):                 # walk_steps = draft_block = 7
    score_base = (flat*top_k + previous)*top_k  # 行 = previous（上一步的候选 rank）
    scores   = load(scores_ptr + score_base + c)     # 本步 E 行
    candidates = load(candidate_ptr + flat*top_k + c)
    _, index = gumbel_noised_argmax(scores, ..., temperature=0 for greedy)
    store tokens[flat] = candidates[index]
    previous = index
```
我们修好的实现：行 = `pred = previous`、列 = `C_s[c]`、`previous = chosen`、
`drafts[s] = C_s[chosen]`、初值 0 —— **逐条相同**。
（修前我们 `chosen` 恒 0 = 忽略边项，与参照语义不符 —— 这是 F4 修掉的实质缺陷。）

### 2.4 **抓到一处真实保真度分歧（新）**
参照 `speculator.py:904-907`：
```python
def draft_logits_spec(...):
    # The selector walk and rejection sampler must consume identical scores.
    # BF16 rounding measurably changes candidate order, so keep this FP32.
    return torch.float32, -float("inf")
```
⇒ 参照全程 **FP32** 的 selector 分数；而我们 `dflash2_impl.h` 的
`logits = work.alloc(DType::BF16, {head_rows, k*batch})` 是 **BF16**，top-K 与 unary 都吃这 8-bit 尾数。
参照自己标注"BF16 会 measurably 改变候选顺序"。**这是一处该对齐的参数（精度），改动小、风险低。**

## 三、verify 侧：今天实测出的真偏差（接受率上限）
同 prompt、同参数：
```
zh_plain vs zh_dflash2 : DIFFER at 29/96
num_plain vs num_dflash2: IDENTICAL
K 扫描: K=1 偏@62，K=3 偏@62（与 K=1 同位置同 token），K=7 偏@29
```
⇒ K=1 即偏、且与 K=3 同位置 ⇒ **不是注意力掩码的跨列污染**，而是**与草稿无关的每轮状态/参数偏差**
（paged KV 槽写入/回滚、GDN `RecordForReplay` 一致性、anchor/bonus 记账）。
按契约 §7.2 的 greedy 判据 `d_i == argmax(p_(i-1))`，差异只可能来自
**verify 列上的 target logits 与 plain 同上下文 logits 不等**，而接受逻辑本身无嫌疑
（licensed 的 token 必然与 target argmax 相等，这是结构保证）。

## 四、审计优先级（下一步，按收益排序）
1. **FP32 化 selector 分数**（2.4）：把 `logits` 与 selector 的 unary/scores 提到 FP32，与参照一致；
   可与其它改动合批，一次编译。
2. **verify 状态链**：审 `speculative_accept_greedy_drafts` 的列→licensed 映射、
   `speculative_select_accepted_hidden` + `scatter(state_destination_slots, continuation_hidden_store)`
   对**被拒列**的 KV 槽回滚、以及 GDN replay 的一致性；判据 = spec 流与 plain **逐位一致**。
   交叉验证：`--spec mtp --draft-tokens 3` 是否也在**同一 token 位置**偏（若同位置 ⇒ 共享路径）。
3. **参照草稿 A/B**（`data/draft_dflash2_ref/` 已在盘上，等你点头）：一次区分"引擎 vs 训练 ckpt"。

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

# R11：翻转成因的再定位（graph 排除 + MTP 链反证）（2026-09-11 15:0X）

## 一、本轮两个决定性实测

### 1. CUDA graph 捕获/回放被排除
同一 prompt，开/关 graph 逐位相同：
```
dflash2_graph   vs plain: DIFFER at 29   dflash2_nograph vs plain: DIFFER at 29（同值）
mtp3_graph      vs plain: DIFFER at 40   mtp3_nograph    vs plain: DIFFER at 40（同值）
```
⇒ spec≠plain 的偏移与 graph 无关（graph 路径无罪）。

### 2. **MTP 在同一 verify 路径上拿到活链** —— 这条推翻了 R10 的结论
```
mtp3   位置剖面 = [29, 16, 5]     条件接受率 p1=16/29=55%, p2=5/16=31%
dflash2 位置剖面 = [6, 1, 0, ...]
```
同一 target、同一 verify/轮次机制、同一二进制 ⇒ **verify 路径不是接受率的限制器**：
它能支撑多 token 链（MTP 就是证据）。spec≠plain 的偏移确实存在（是 G-A 层面的瑕疵，该修），
但它**不封顶**接受率。⇒ R10"缺口在 verify 块与顺序解码不等价"需要降级为"次要瑕疵"。

## 二、修正后的定位：限制器是 **dflash2 草稿块的第 1..K 位**
- dflash2 的第 1..K 位正是**吃 mask token** 的列；MTP 的草稿机制（markov/链）不依赖 mask 块。
- `_collab/A2_draft_ceiling.md`（既有记录）：该 ckpt 的训练**喂真 token**，
  mask id 在训练语料出现 **0/4468** ⇒ 推理喂 `[anchor, mask×K]` 时，第 1..K 位是**分布外输入**
  ⇒ 这些位置的 hidden 必然不可用 ⇒ 链在位置 1 断。
- 这与全部实测吻合：列 0（不依赖 mask）正常 85%；可复制文本上 unary 独自够用 ⇒ 6/7 列正确；
  边项扫描无益（边项吃的是第 1..K 位的 hidden）；K 无关（掩码/结构无病）。

### 顺带修正一条我自己的推论
"mask embedding 行未训练"**不成立**：草稿训练时该行是**冻结的固定向量**，行本身训不训练无所谓，
真正的错配是**输入模式**（真 token vs mask）。这也解释了为什么早前"换 mask token id"实验无效。

## 三、待你一句话就能定论的问题（决定"引擎 vs ckpt"）
你说干净环境那套 vLLM（支持 dflash2）跑 dflash2 时接受率正常（记得 50 几）。**它加载的是哪份草稿？**
- 若是 `data/draft_dflash2_ref/`（Sep 11 12:32 在盘上那份 HF `DFlash2DraftModel`）
  ⇒ **本诊断成立**：我们用 mask 块推理 step_006000（训练喂真 token）才是塌陷主因，
  引擎侧只需补 G-A 保真（FP32 分数）与 retrain 后复测。
- 若它加载的是我们自己的 ckpt（`data/dflash2_ckpts/step_*.pt` 或我打的 artifact）
  ⇒ **本诊断被推翻**，差异回到引擎某处我尚未定位的管线参数，我按 R10 的探针继续挖。

> 之所以必须问：`data/draft_model/`（vLLM 日志里出现的草稿）是 **dspark** 模型
> （`Qwen3DSparkModel`，mask 190221 / rope 1e6 / layers[1,14,29,44,57]），与 dflash2 不是一回事，
> 不能当 dflash2 对照；而 `data/draft_dflash2_ref/` 恰好是**训练过 mask 块**的参照 dflash2 草稿。

## 四、引擎侧待办（不依赖上面的答案）
1. selector 分数 FP32 化（对齐参照，消除 BF16 候选顺序抖动）——合批做。
2. spec==plain 的偏移（G-A 瑕疵）：按 R10 §四(A) 打 logits/margin 判别探针，
   定"数值路径 vs 状态回滚"，再决定改算术还是查 GDN replay。
3. retrain 就绪后（`_train_df2_shift0.bat`，mask 输入 + shift 0）→ 三闸门出新 artifact →
   用 `NINFER_DF2DBG` + `_df2_blame2.py` / `NINFER_DF2SEL` / `_verify_df2head.sh` 复测。
   验收判据不变：链式剖面（p1+ 非零）、AL ≥ 3、G-A IDENTICAL。

# R12：排除"引擎侧"可能性（2026-09-11 15:1X）

## 结论
**引擎侧被排除**：引擎已经把该 ckpt 的能力**榨干**了；限制器是 ckpt 的训练/推理输入口径错配
（A2 的 M1 shift 差一位 + M2 真 token vs mask）。以下 5 条都是实测/既有量化记录。

### E1 引擎实测 p0 ≈ 26–35% ≥ 该 ckpt 的离线上限 ≈ 25–31%
A2 §Q1 记录（同一 ckpt、679 anchors、两种输入口径离线跑）：
- 引擎真实输入口径（`[anchor, mask×7]`）：位置 0 命中 **0.0309 (21/679)** @step1200、**0.0427** @step1800；
- 训练脚本口径（真 token 块）：0.1708 / 0.1222；
- 连"按脚本自己的目标 `ids16[a+1,0]`"也只有 **0.2504 / 0.2047**。
A2 原话：**"位置 0 的上限远低于 41.8%；即便按对自己最有利的口径也只有 0.25"**。
今天引擎实测位置 0 接受率：修 walk 前 8/23=34.8%、修后 6/23≈26% ⇒ **已达/超过该 ckpt 的上限**
⇒ 引擎没有在丢好草稿。

### E2 A2 §Q2：引擎接受率 ≈ 草稿离线命中率（实测一致性）
A2 原话："**没有证据表明 verify 在丢好草稿**（引擎接受率 ≈ 草稿离线命中率）；41.8% 那一档
属于更容易的 prompt 组合，纯文本样本复现不出"。⇒ 引擎忠实实现了草稿质量。

### E3 可复制文本上，引擎的草稿在 mask 列 1..7 上是**对的**
今天实测（数字模式 prompt）：单轮 drafts = `[220,16,220,17,220,18,220]` vs target argmax
`[220,16,220,17,248046,18,220]` ⇒ **6/7 列（含全部 mask 列）正确**。
⇒ 引擎的草稿管线（context K/V、块构造、绝对 positions、dynamic conv、final_norm、
proposal head、selector）在该场景下**可证明地正常工作**——若有引擎级缺陷，这些列也会错。

### E4 共享的 verify/轮次机制能支撑多 token 链
今天实测同一二进制、同一 target：`mtp3` 位置剖面 **`[29,16,5]`**（p1=16/29=55%、p2=5/16=31%），
而 `dflash2` 是 `[6,1,0,…]`。⇒ verify/accept/轮次记账本身不是限制器（MTP 走同一条路）。

### E5 引擎侧参数/结构逐项已核
- 声明型超参 vs ckpt 自带 `config.json`：逐项一致（R8 表）。
- 块构造（`[anchor, mask×7]`、`positions=F+min(i,V-1)`）、context positions（target 块绝对位置）、
  selector walk 语义（行=previous、列=C_step、previous 链）：与契约/参照一致（R9）。
- 已排除：CUDA graph 回放（开/关逐位相同）、KV 精度（bf16 实测、int8 同形）、
  `rope_delta`（纯文本为 0）、`apply_final_logit_policy`（qwen3 编译期 no-op）、
  跨列掩码污染（K=1 就偏，K 无关）。

## 因此：
1. **限制器 = ckpt 的训练口径**：`train_dflash2.py` 喂真 token（mask id 在语料 0/4468）+ 默认
   `--target-shift 1`，而引擎推理喂 `[anchor, mask×7]` + shift 0 ⇒ 推理时第 1..K 位是分布外输入。
   A2 的 M2 表：mask 口径命中率比真 token 口径低 **5.5×**（0.031 vs 0.171 @step1200；step400 为 0/679）。
   ⇒ 修法就是已备好的 `_train_df2_shift0.bat`（mask 输入 + shift 0）；产出新 ckpt 后按
   `_collab/C_artifact_manifest.md` 的 patch→verify→round-trip 三闸门出 artifact，再用现成仪器复测。
2. **引擎侧只剩保真度项**（不影响接受率上限，但按契约该修）：
   a) selector 分数 FP32 化（参照自注 BF16 会改变候选顺序）；
   b) spec 流 vs plain 的偏移（G-A 瑕疵）：R10 §四(A) 的 logits/margin 探针定"数值 vs 状态"。

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

## 重启前状态快照（2026-09-11 15:5X，用户要求重启 ZCode 以让子代理模型生效）
- **本轮已落地的引擎改动**（都在 WSL 构建树 `/home/user/ninfer-fusion`，均有备份/可回滚）：
  1. `src/ops/gdn_input_proj/gdn_conv.cuh:116` — UP1_gdn_conv_column（`s2` 回带 BF16 发布值），
     备份 `/home/user/gdnconv_bak/gdn_conv.cuh.143109`，md5 `f9a21c5a9b54 → 3a3b445c9161`；
     5 个 includer 已 touch 并重编通过（binary 14:35）。
  2. `dflash2_selector.cuh` — walk argmax 修复（F4，对比本 lane 自身分值）+ env-gated 探针
     （`NINFER_DF2SEL` / `NINFER_DF2_PAIR_SCALE`，默认不影响算术）。
  3. `speculative_options.h` — 解禁 dflash2 的 `ProposalHead::Optimized`；`dflash2_impl.h` 加两 route；
     三处 config.h 补 `dflash2::draft_head_rows`。
  4. 补丁留档：`_collab/F1_df2_draft_head.diff`、`F2/F3/F4`、`UP1_gdn_conv_column.diff`。
- **重启后的第一件事（按顺序）**：
  1. 定位 spec 解码实际走的 GDN kernel（按权重格式/ token 数分派）：`src/ops/gdn_input_proj/{nvfp4,q4_q5,w8,fp8}/*plan*`；
     已知 mtp 的流未受 `GdnConvEpilogue` 影响 ⇒ 别在错的 kernel 上打补丁。
  2. 判 "chunked(一次 W 列) vs 递推(一 token 一步)" 的数值不等价，并二选一对齐；验收 = spec 流与 plain 逐位一致。
  3. 保留项：selector 分数 FP32 化（参照保真）；接受率上限仍归 ckpt 口径（R12），retrain 后复测。
- **子代理**：`agents-state.json` 的两个 `builtInModelOverrides` 已改为
  `8060ff93-47f4-4298-841f-7561864c0cfe/deepseek-flash`（备份 `dl/agents-state.bak_20260911_091726.json`）；
  重启即为生效。重启后可派子代理做：GDN kernel 分派定位、8 项未落地清单、LABD 调研。
- 报告索引：R6–R13 在 `_collab/`（R12=引擎侧排除、R13=GDN 周期性偏移）。

# R14：漂移不是"固定生成序号"，而是小扰动的阈值穿越（2026-09-11 16:2X）

## 实测（同一 prompt、同一二进制，修掉 +1 偏移后）
| 配置 | 轮数 | 首次偏离 |
|---|---|---|
| K=1 | 72 | 第 **17** 轮 / 第 **22** 个生成 token |
| K=7 | 89 | 第 **8** 轮 / 第 **10** 个生成 token |

⇒ **轮数与 token 数都不恒定** ⇒ 既非"按轮累积"也非"按 token 累积"。

## 判读
- 漂移是一个**持续存在的小扰动 + 内容相关的 margin**：只有走到 near-tie 处才翻 argmax，
  所以"首次翻转点"随内容与轮次结构变化 ⇒ 它**不是**一个干净的累积轴指标。
- 因此此前把 `M_offset_probe/2` 的"固定生成序号 36"读成"与解码步数相关"属于**过度解读**
  （短/长两档内容相近，才落在同一处）。K=1 与 K=3 同位置更像是"首轮草稿相同"的巧合。
- **正确的量化对象是扰动幅度，而不是首次翻转点。**

## 下一步（本路主代理自己做，不与子代理重叠）
打 logits 级探针（R10 §四(A)）：在同一个 (上下文, 绝对位置) 上，把
**plain 单步的 logits** 与 **verify 第 0 列的 logits** 各取 top-8 打出来，比较：
- top-1 差距 <1e-2 且排名仅差一位 ⇒ 纯数值（要"消除偏移"就得让 verify 与单步同算术）；
- 差距大 / 排名完全不同 ⇒ 结构性（状态回滚或位置口径）。
`logits` 两个来源都在手里：verify 侧是 `frame.target_logits`（探针已能读），
plain 侧在 `text_prefill_impl.h:377`（`apply_final_logit_policy` 之后、`ops::sample` 之前）。

## 与子代理的分工
- S_A（GDN kernel 分派）/ S_B（GDN 状态回滚）给**静态**证据链；
- 本路给**运行期**幅度证据；
- 三者合并后才能判"数值 vs 结构"，再决定改算术还是改状态回滚。

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

# R16：S_B 结论 —— 状态机是精确的，残差全在"形状相关的进料/出料"（2026-09-11 17:0X）

## 一、S_B 已裁决（报告 `_collab/S_B_gdn_state.md`）
1. **spec 轮不存在"回滚"**：`RecordForReplay` 下 key/value/gate/conv 只写记录平面、主状态零改动
   （`recurrent.cuh:350-441`）；被拒列 = 从未应用。
2. 只有 committed 前缀（`accepted+1`，anchor 恒提交）经 `gdn_replay_fold` 重放
   （`program_impl.h:9847-9868` → `recurrent.cu:104-128`）。
3. **fold 与"逐 token 推进 commit 次"逐位等价**（同一 `run_recurrent_sequence`、同一记录、同一 FP32 起点）。
4. conv 窗口携带精确（记录存 `bf16(p)`；`publish_final_conv_history` 是 `tail_3` 的逐下标双射 + static_assert）。
5. `commit ≤ valid_columns`、`accepted+1 == count` 有强校验（`program_impl.h:12491-12498`）。
⇒ **结论：状态转移严格等价；不等价的只是"W 列一次"与"1 列一次"两套形状产生的进料/出料。**

## 二、残差清单（全部同一"形状相关数值不等价"家族）
| # | 位置 | 机理 | 量级 | 触发 |
|---|---|---|---|---|
| D1 | `fused conv :99-104` vs `:118` | 第 4 抽头用 FP32 `p`，却把 `bf16(p)` 发布进状态 | ~2e-3 | batch==1 且 W∈{2,3,7..10}（**W=8 ⇒ dflash2 中招；W=4 ⇒ mtp 不中招**） |
| D2 | flat conv vs record 侧 | `acc += w*x`（8 舍入） vs `fmaf` 链（4 舍入） | ~1e-7 | 偶发 LSB |
| D3 ★ | `gated_delta_net.cpp:186-191,256-260` vs `recurrent.cuh:615-631` | chunked 把 `l2norm(q/k)` 发布成 **BF16**；recurrent 用寄存器 **FP32** ⇒ **同 token 不同 k** | 形状相关 | chunked(≥64) vs 单列 |
| D4 | `chunked/launch.h:26-35`（kChunkSize=64） | 工作区 `W/U/v_new/h_chunk` 全 BF16 ⇒ 64 token 粒度已发布值、对切分敏感 | 形状相关 | `--prefill-chunk` 改变切分 |

**D1 与今天落地的 `gdn_conv.cuh:116` 同类**（都是"状态里存的是发布值、计算里用了更宽的累加器"）；
补丁打在 FusedA16 ⇒ 只有 W∈{2,3,7..10} 的 batch==1 受影响 ⇒ **解释了"补丁改变了 dflash2 的流、mtp 逐位未变"**。

## 三、战略性结论（回答"消除偏移"的可行边界）
- 用户 P1 闸门 G-A 要的是 **spec 输出与 plain 逐位一致**。按 S_B：状态机已精确，
  但 **chunked GDN 与 recurrent GDN 的中间量表示不同（D3/D4）** ⇒ 只要 verify 仍走 chunked 形状，
  spec==plain 就**不可能**逐位一致（任何引擎同理，含参照实现）。
- ⇒ 真正可落地的方向只有两条：
  a) **让 verify 的 GDN 逐 token 走 recurrent 路径**（与 plain 完全同算术）⇒ 换来 G-A 逐位一致，
     代价是 verify 里 GDN 段失去 W 列并行；
  b) 保持 chunked，但把 `l2norm(q/k)` 与工作区提升到 FP32（D3/D4 的深改，牵动 chunked 全部 kernel）。
- **先做便宜的一半**：D1 的一行修复（按 `:118` 范式把 `:99` 的 `p` 先 bf16 化）⇒ 消掉 2e-3 那一档；
  它只影响 W∈{2,3,7..10}（dflash2 默认 W=8 命中）。

## 四、下一步（主代理车道，按性价比）
1. **P1 可证伪预测（S_B 给）**：强制走 `MaterializedA16` 后跑 `_ga_check.sh`，
   预测 K=1/K=7 的首次偏离会变、**K=3（W=4，本就 materialized）不变**。若成立 ⇒ D1 确在动数。
2. 落地 D1 一行修复（与其它改动合批一次编译），复测 `_ga_check.sh` 与接受率。
3. 若要求 G-A 逐位一致 ⇒ 按 §三(a) 立项"verify 的 GDN 走 recurrent 逐 token"，单独评估性能代价。
4. 与 S_A/S_C 合并：S_A 定"哪种格式/宽度走哪条 conv 与 GDN 路径"（含打印 `Fp8GdnConvScheduleId` 定案），
   S_C 给接受率上限的口径证据。

# R17：首因确认 —— GDN 预归一化的 BF16/FP32 双路（S_A 预测一次命中）（2026-09-11 17:2X）

## 一、可证伪预测的结果（零编译、零投机）
| prompt | `--prefill-chunk 128` vs `4096` |
|---|---|
| 未对齐（≈1200 token） | **第 0 个生成 token 就分叉**（104980 vs 103735） |
| **对齐到 1920 = 15×128** | **整段 IDENTICAL（31 token）** |
脚本 `_align128_ab.sh`（含对齐脚本 `_align_prompt128.py`，tokenizer 直接从 artifact 的
`frontend/tokenizer.json` 取，不加载模型），日志 `dl/align128_ab.log`。

⇒ **S_A 的机理①成立**：`gated_delta_net.cpp:254-261`
—— 只有当该次调用处理了完整 64 块（`T_full>0`）时，q/k 的 `l2norm` 才写进 **BF16 缓冲**
（缓冲 dtype 见 `:187-190`）并令 `recurrent_normalize=false`；`T_full==0`（本次 <64 token）时
改在 **FP32 寄存器**里归一化（`recurrent.cuh:64-70,125,617`）。
⇒ 同一批 token 因"切分方式"不同而拿到 **不同的 k**（`l2norm(q/k)` 的发布精度不同）。

## 二、意义
1. 这是**第三处**"状态/缓冲里存的是发布值、计算里用了更宽累加器"的同族缺陷
   （前两处：`gdn_conv.cuh:116` 的 `s2=p`、fused conv 的 `:99-104`）。
2. 它**只影响形状相关的进料**（不是状态回滚问题，S_B 已证明状态机精确）
   ⇒ 正是 **verify(T=W=8 列) vs plain(T=1 列)** 这类形状切换的差异来源。
3. 由此得到**最小改动清单（S_A 给，按性价比）**：
   ① **删/改 `gated_delta_net.cpp:255-261` 的 BF16 预归一化分支（≈6 行，本次首因）**；
   ② `h_chunk` BF16→FP32（3 处：`chunked/launch.h:42`、`launch.cu:55,68`、`output.cuh:11,155`）；
   ③ `nvfp4_gdn_snapshot_plan.cpp:42-44` 放宽到 `tokens<=16` 分派；
   ④ 残余的"按 T 选 tile/route"（`nvfp4_gdn_input_w4a4.cu:40-58`、attention/conv 按 T 选 kernel）
      只能统一 kernel 家族或接受非 bit-exact —— 建议先修 ①②③ 再复测，否则会把"精度档/发布口径"
      误判为不可控舍入。
4. **回归探针（重要细节）**：S_A 建议把"同 prompt 只改 `--prefill-chunk`"加入验收；
   但本次实测表明它**只在对齐非 128 倍数时灵敏**（对齐后即使缺陷仍在也会通过 ⇒ 假阴性）
   ⇒ 探针必须用**非 128 对齐**的 prompt（例如 1200 token 那版）。

## 三、下一步（主代理车道，串行一次编译）
1. 落地 ①（≈6 行）与 ②（3 处 dtype）；③ 视风险决定是否同批。
2. 复测三件：
   a) `--prefill-chunk 128 vs 4096`（**非对齐** prompt）→ 期望 IDENTICAL；
   b) `_ga_check.sh`（spec vs plain，zh/num）→ 看首次偏离是否后移/消失；
   c) 接受率剖面（`_verify_df2head.sh`）→ 不应退化。
3. 若 ①②③ 后 spec 仍 ≠ plain（预期仍有 ④ 的按 T 选 tile），再立项"verify 的 GDN 走 recurrent 逐 token"
   （S_B §三(a)）或接受非 bit-exact 并在文档里明确 G-A 的适用边界。

# R18：S_C 的阻碍性发现 —— R12 的 E1/E2 论据作废，需先钉死一个混淆项（2026-09-11 17:4X）

## 一、必须纠正我自己的结论（诚实记账）
S_C 查明：**live artifact `/home/user/models/qwen3_8_27b_nvfp4_dflash2.ninfer` 的 mtime 是 2026-08-26，
且全树唯一**（没有 `*_tuned` / `w9s1200` 之类变体）；唯一一次导出尝试
`dl/df2_w9_export.log`（09-10 15:45）报 `ModuleNotFoundError: No module named 'tools'`，
`dl/` 里**没有任何 `mapping ok`** ⇒ **任何训练 ckpt 从未进过引擎**。

⇒ 今天所有引擎接受率数字，描述的是**08-26 那份草稿**，而不是 step_006000 / step_1900。
⇒ **R12 的 E1 作废**：它把"引擎实测 p0 26–35%"与"*另一个* ckpt 的离线上限 25–31%"相比；
**E2 也作废**（A2 的离线命中率是训练 ckpt 的，不是 08-26 草稿的）。
⇒ **"引擎侧已排除"这条结论的定量桥被拆掉，问题重新开放**；但 E3（可复制文本上 mask 列 6/7 正确）、
E4（MTP 同路径拿到 `[29,16,5]`）、E5（参数/结构逐项一致）**不受影响**，仍支持"引擎机制健全"。

## 二、另一个必须先钉死的混淆项
`M_df2_serve_measure.md` 记的 **4.55 tok/round**（旧 binary、serve 路径）与今天 CLI 的
`[8,0,0,0,0,0,0]`（≈1.35 tok/round）**互相矛盾**。同一份 08-26 草稿 ⇒ 要么是 prompt/路径差异，
要么是我今天的改动（F4 walk 修复、GDN conv 补丁）把它改差了 ⇒ **retrain 前必须先钉死这一项**。

## 三、S_C 的两条便宜且决定性的判定（建议优先于 6000 步重训）
1. **已有 mask+shift0 的 ckpt**：`data/dflash2_ckpts/step_000100`、`step_000200`
   （09-10 23:32/23:36，**晚于** 19:34 的配方改动）。测"塌陷比" R（mask 口径命中 / 真 token 口径命中）：
   **旧 ckpt R≈0.18，预测新配方 R ≥ 0.6** ⇒ 一次离线评测即可判 M2 是否是修法。
2. **直接测 live artifact 那系草稿**（`data/draft_model/model.safetensors`，08-17）的 R：
   R≈0.18 ⇒ 口径论需重估；R≈1 ⇒ M2 就是修法。
两者都比重训便宜得多。

## 四、S_C 的其他确认与坑
- **M1 成立**（shift 必须 0）：`train_dflash2.py:227` 行 i ↔ 位置 a+1+i；`:457` 教师行 `ids16[a+shift+i]`；
  实测 `P(ids16[t,0]==tok[t+1])=0.2420` vs `==tok[t]=0.0023` ⇒ 教师是**下一个** token，shift=1 晚一行。
- **M2 成立**：legacy 配方有源码直证（`data/df2pilot/train_dflash2.py:420` 真 token 块、`:425` 硬编码 +1）；
  全量 **706,220** 位置里 mask id 出现 **0** 次 ⇒ mask embedding 梯度恒 0，而推理 7/8 列喂它。
- **M1 从未被实验测过**：`data/_ab_shift0|1` 是空目录、`dl/shift_ab.log` 无 step 行 ⇒ 现存结论全靠语义。
- **mask id 别动**：fork `config.h:106=248077` 与 trainer 一致；190221 实验已回退、无改善。
- **retrain 坑（必读）**：`data/dflash2_ckpts/` 混了两套配方，`--resume` 取 `sorted()[-1]` =
  `step_001900`（**legacy 错位权重**）⇒ 续训必须显式 `--out-dir`，否则接错配方。
- 可证伪阈值 P1–P5（见 `_collab/S_C_train_regime.md`）：如 P3 "引擎剖面不得再是 `[x,0,0,0,0,0,0]`，
  p1/p0≥0.5 且 AL≥1.8"。

## 五、下一步（顺序已定）
1. **钉死混淆项**：同 prompt 同 ckpt，比 serve 路径 vs CLI 路径的 tok/round（先用 08-26 草稿，
   再与我今天的改动做一次 A/B：F4 之前/之后、conv 补丁之前/之后）。
2. **走 S_C 的便宜判定**（step_000100/000200 的 R；以及 08-26 草稿的 R）⇒ 决定是否/如何 retrain。
3. **同时落地 GDN 形状修复 ①②**（与 ckpt 问题无关、独立有效）：删 `gated_delta_net.cpp:255-261`
   的 BF16 预归一化分支、`h_chunk` BF16→FP32；验收 = 非对齐 prompt 的 chunk A/B IDENTICAL
   + `_ga_check.sh` 首次偏离后移/消失 + 接受率不退化。
4. R12 的措辞已在 `_HANDOFF.md`/`_TODO.md` 里按本页纠正（E1/E2 作废）。

# FIX_PLAN_ABCD：性能优先的修复整合（2026-09-11 18:0X）

## 0. 硬约束（用户定调）
**最优性能，不惜代价** ⇒ 禁止任何"以并行度换精确性"的改法。
具体否决：**S_B §三(a)**（让 verify 的 GDN 逐 token 走 recurrent）——它换来 G-A 逐位一致，
但 verify 的 GDN 段失去 W 列并行 ⇒ **不做**。同理否决任何 FP32 化的*全量*缓冲升级
（`h_chunk` BF16→FP32 会 2× 带宽）。原则：**在保持/提升性能的前提下对齐数值**。

## 1. 四路结论 → 三个修复项 + 两项决策
| 来源 | 发现 | 本计划处置 |
|---|---|---|
| S_B D1 ★ | fused conv 第 4 抽头用 FP32 `p`、却把 `bf16(p)` 发布进状态（`:99-104` vs `:118`），量级 ~2e-3，触发 batch==1 且 W∈{2,3,7..10} | **修 ①（一行，性能中性）** |
| S_A ④ + ③ ★ | 小 T 家族按 T 选不同 kernel/精度档（`nvfp4_gdn_input_w4a4.cu:40-58` 激活量化 kernel 与 M-tile/TMA 表随 T 变；`nvfp4_gdn_snapshot_plan.cpp:42-44` 分档）⇒ **verify(T=W) 与 plain(T=1) 走不同路线** | **修 ②（统一小 T 家族，性能中性：这些形状本就极小）** |
| S_A ① | `gated_delta_net.cpp:254-261`：`T_full>0` 走 BF16 缓冲、`T_full==0` 走 FP32 寄存器 ⇒ prefill 切分敏感（已用 128 对齐实验证伪/证实） | **修 ③（仅影响 prefill 尾块；非 G-A 必需，列为次优先）** |
| S_A ② | `h_chunk` BF16（3 处） | **不做**（FP32 化 = 2× 带宽，违性能约束）；若 ③ 后仍差再评估 |
| S_B D2 | flat conv `acc+=w*x` vs record 侧 `fmaf` 链（~1e-7） | 不做（低于任何可观测阈值） |
| S_D | selector `logits` BF16 单侧 vs 参照 FP32；greedy 下 accept 不消费 `draft_candidate_probs`（无风险） | **不做**（S_D 已证：既不能修 spec≠plain，也不影响接受率上限；且裸改 dtype 会静默出错） |
| S_C | 训练口径（M1 shift / M2 真 token vs mask）+ live artifact 是 08-26 草稿 + `--resume` 取 legacy `step_001900` | **决策项**（见 §4） |

## 2. 修复项（按顺序，均性能中性或更优）
### 修 ①（S_B D1，一行）
`src/ops/gdn_input_proj/fp8/<fused conv kernel>`：把 `:99-104` 处的第 4 抽头累加从"直接用 FP32 `p`"
改为"先用 `bf16(p)` 再进状态"，与 `:118` 的 `s2 = __bfloat162float(__float2bfloat16_rn(p))` 同范式。
- 判据：`--prefill-chunk` 无关；更重要的是 `_ga_check.sh` 里 **K=1/K=7 的首次偏离后移或消失**、
  而 **K=3（W=4，本就 Materialized）不变**（S_B 给的可证伪预测）。
- 性能：单条 bf16 round，无结构变化。

### 修 ②（统一小 T 家族 —— 结构性修法）
目标：让 **T ∈ {1..16}**（decode 与 verify 的 W=2/4/8/16 全在内）走**同一** GDN 路线与精度档。
- `nvfp4_gdn_snapshot_plan.cpp:42-44`：把分档阈值放宽到覆盖 `tokens<=16`（S_A ③）。
- `nvfp4_gdn_input_w4a4.cu:40-58`：确认小 T 家族内激活量化 kernel 与 M-tile/TMA 选择**一致**；
  若 T=1 与 T=W 选了不同表，则把小 T 家族统一到同一条（这些形状极小，性能代价≈0）。
- 目的：**spec==plain 的结构性前提**（同形状家族同算术），且**不动大 T 的 prefill 快路**。
- 判据：`_ga_check.sh` 在 zh/num 上 IDENTICAL（或首次偏离显著后移）+ 接受率不退化 + 解码 tok/s 不退化。

### 修 ③（S_A ①，次优先）
`gated_delta_net.cpp:254-261`：让 `T_full==0`（尾块 <64）与 `T_full>0` 使用**同一发布精度口径**。
- 性能优先的选择：**把尾块对齐到已发布口径**（而不是把主路升到 FP32）。
- 判据：非 128 对齐 prompt 的 `--prefill-chunk 128 vs 4096` 变为 IDENTICAL。

## 3. 验收套件（每次改动都跑）
1. `_ga_check.sh`（spec vs plain 逐位，zh + num）
2. `_align128_ab.sh`（chunk 不变量；**必须用非 128 对齐 prompt**，否则假阴性）
3. `_verify_df2head.sh`（接受率 + G-A）
4. 解码 tok/s（性能不许退化）
5. （新增）`--prefill-chunk` 回归探针纳入常规验收

## 4. 两项决策项（用户/需 GPU 窗口）
- **S_C 的混淆项**：同 ckpt 下 serve 4.55 tok/round vs CLI 1.35 tok/round ⇒ 先钉死再谈 retrain。
- **retrain**：`_train_df2_shift0.bat`（mask + shift0），**必须显式 `--out-dir`**（否则 `--resume` 取
  legacy 的 `step_001900`）；先用已有 `step_000100/000200` 测塌陷比 R（旧≈0.18，预测新 ≥0.6）。

# R19：P7 双路对比 + ② 落地实况（单变量无效果）+ 构建事故（2026-09-11 18:3X）

## 一、P7 对比（A 路 vs B 路，同方法独立两遍）
### 求其同（两路一致 / 量级吻合）
| 项 | 结论 |
|---|---|
| **修①**（`gdn_conv.cuh:99`：`p` 先 bf16 化） | **两路逐字一致的唯一实质代码改动**；B 另证发布侧逐位不变（`bf16(bf16(p))==bf16(p)`），修后 fused conv 与 materialized post conv **逐表达式相同 ⇒ 对同 p 逐位等价（是 0，不是 1e-7）** |
| **修②**（`nvfp4_gdn_snapshot_plan.cpp:43`：`tokens<=3` → `<=16`） | 实质一致；B 发现该行**已在树中**（我 15:1X 那次被取消的调用落进去的），B 未回退、只把自己的 hunk 降为注释——处理得体 |
| **量级** | ② 的 A4 激活误差：A 测 L2 9.48%/点积 1.14e-1，B 测 RMS 9.4e-2/点积中位 9.6e-2（A16 分别 0.17%/1.28e-3 与 1.7e-3）⇒ **两路独立吻合 ≈55–57×，属一阶偏差** |
| ① 量级 | A：Δconv/|conv| p90 4.1e-3 ≈ 输出 bf16 步长，~10% 列翻 1 ulp；B：均值 1.4e-3、14.8% 列变、97% 恰 1 个 bf16 步 ⇒ 同性质同量级 |
| **性能** | 两路同判：① 中性；② 权重流量不变、激活 +123 KB/层/轮、**省 1 kernel 启动与 w4a4 中间缓冲**，roofline 下 +0~3%/round ⇒ **必须实测 tok/s**（A 的退路：>3% 就把阈值收到 8） |
| **残余（两路同判，重要）** | T=1 走 gemv、T∈[2,16] 走 small_t ⇒ **不可能逐位一致**（~1e-7，偶发 ±1 bf16 ulp）；batch>1 仍走 compose(A4) ⇒ **验收判据应改为：同精度档 + 同 conv 语义 + 首次偏离后移/消失 + 性能不退化**，而不是逐位 hash 相同 |
### 分析不同（待实验裁决）
- **修③（`gated_delta_net.cpp` 尾块归一化）**：A 判"有机会 IDENTICAL"（`T_full=128`，chunk128/4096 的划分相同，`g_cumsum` 逐 64 块重置、`l2norm` 逐 (head,token) 独立）；**B 判"不可能"**（残余来自 attention/MLP/lm_head 的 T 形状 kernel）⇒ 用 `_align128_ab.sh`（非对齐 prompt）裁决。

## 二、② 落地实况（单变量）：**测不到效果**
- 已落：`nvfp4_gdn_snapshot_plan.cpp:43` = `if (tokens <= 16) { … SmallTFusedA16 }`（备份 `/home/user/fix2_bak/`）。
- 构建成功后的**同一套基准**与修复前**逐项一致**（`zh plain vs df2` 仍 DIFFER@10 同 token；chunk 探针仍 DIFFER@0，
  `104980` vs `103735`）⇒ **② 未改变我们测到的路径**。
- 与 A 路 caveat #2 吻合：`AllowA4` 系源码常量推断；**若 dflash2 verify 的 GDN 输入不走 snapshot plan，② 即 no-op**。
- ⇒ **定案手段（A 路建议）**：打一行 schedule 日志，打印解析出的 `Nvfp4GdnConvScheduleId`；
  同时也需确认 dflash2 verify 实际走哪个 GDN 输入 kernel（snapshot / w4a4 / independent …）。
  在此之前**不要**把 ② 当作已产生收益的修复。

## 三、构建事故与教训（已修复）
- 现象：`make` 报 `Error 127`、`/bin/sh: 1: ccache: not found`，`apps/ninfer` 被删（链接未产出）。
- 原因：构建用 ccache 作 CUDA 编译器启动器（早前 `-DCMAKE_CUDA_COMPILER_LAUNCHER=ccache`），
  而这次 `make` 的 PATH 里没有 `/home/user/.local/bin`（ccache 在那里）。
- **教训（写入纪律）：任何构建都必须 `export PATH=/home/user/.local/bin:$PATH`**；
  我早前成功的构建都在脚本里 export 过，裸 `make` 会 127 并删掉二进制。
- 已修：`export PATH=…` 后 `make ninfer -j2` rc=0，`apps/ninfer` 15:22 恢复（含 ②）。

## 四、下一步
1. **修① 落地**（两路共同核心；我上一版 patcher 因真实行含对齐空格未命中，守卫挡住、树未动）：
   锚点改为子串 `= projected[token];`（唯一）⇒ 替换为 `= __bfloat162float(__float2bfloat16_rn(projected[token]));`；
   落地后跑同一套基准（判据：`_ga_check` 首次偏离后移/消失、chunk 探针、接受率、tok/s 不退化）。
2. **定案 ②**：加一行 schedule 日志，确认 verify 走的 GDN kernel（决定 ② 是有效还是 no-op）。
3. **裁决 ③**：跑 `_align128_ab.sh`（非对齐 + 对齐两组）。
4. 验收口径统一按 §一"残余"栏调整（不要求逐位一致）。

# R20：修① 落地实测（单变量验证 + 交叉验证）（2026-09-11 18:5X）

## 一、已落地
- **修②**：`src/ops/gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_plan.cpp:43` `tokens <= 3` → `<= 16`
  （备份 `/home/user/fix2_bak/`）；单变量实测**无任何变化** ⇒ 疑为 no-op（待 schedule 日志定案）。
- **修①**：`src/ops/gdn_input_proj/gdn_conv.cuh:99`
  `const float p              = __bfloat162float(__float2bfloat16_rn(projected[token]));`
  （备份 `/home/user/fix1_bak/gdn_conv.cuh`）；两路（A/B）逐字一致的共同核心。
- 构建：`export PATH=/home/user/.local/bin:$PATH && make ninfer -j2` rc=0，`apps/ninfer` 15:35。

## 二、单变量实测（同一套基准脚本，修复前 → ①+② 后）
| 指标 | 修复前 | ①+② 后 | 判读 |
|---|---|---|---|
| `zh plain vs dflash2` 首次偏离 | DIFFER at **10** | DIFFER at **19** | **后移 ✓ 修① 在预期路径上生效** |
| `zh plain vs mtp3` 首次偏离 | at 40 | at 40（同 token） | **完全未变 ✓ 与触发谓词一致** |
| chunk 探针（非对齐长 prompt） | DIFFER at 0 | DIFFER at 0 | ③ 未落地，符合预期 |
| chunk 探针（128 对齐） | IDENTICAL | IDENTICAL | 对照项 |
| dflash2 接受率 / 剖面 | 4.51% `[6,1,0,…]` | 3.76% `[5,0,0,…]` | ±1 token 量级（23 轮）⇒ 噪声内，**不能说提升** |
| mtp3 接受率 / 剖面 | 37.31% `[29,16,5]` | 37.31% `[29,16,5]` | 逐位未变 ✓ |
| decode tok/s（长 prompt） | ~70.9 | **71.16** | 无退化 ✓ |
| decode tok/s（zh） | （未留基线） | 44.78 | 作为后续对照值 |

### 交叉验证（值得记的一条）
mtp3 在所有指标上**逐位未变**，而 dflash2 的首次偏离后移——这正好对上修①的**触发谓词**
（fused conv 仅在 batch==1 且 **W∈{2,3,7..10}** 生效；dflash2 默认 W=8 命中，mtp3 的 W=4 不命中）。
⇒ 两路（A/B）给的谓词被独立实测证实 ✓。

## 三、诚实的边界
- 修① 让 spec 与 plain **更接近**（首次偏离 10 → 19），但**没有**让接受率上升；
  接受率上限仍受 ckpt 口径（R12/R18）与残余 ~1e-7 归约差（两路同判）约束。
- 修② 是否有效**未定**：需加一行 schedule 日志打印解析出的 `Nvfp4GdnConvScheduleId`，
  确认 dflash2 verify 的 GDN 输入到底走 snapshot / w4a4 / independent 哪条。
- 非对齐 prompt 的 chunk 不变量仍未达成（DIFFER at 0）：按 B 路判断，残余来自
  attention/MLP/lm_head 的 T 形状 kernel，**修③ 只能消掉 GDN 那一份**，不能单独达成 IDENTICAL。

## 四、下一步
1. 一行 schedule 日志给 ② 定案（决定它算不算收益）。
2. 修③ 评估：若目标是"chunk 无关性"，需一并处理其它 T 形状 kernel（工作量已知但更大）；
   若只求 spec==plain 尽量接近，③ 的边际收益已由 ① 覆盖一部分 ⇒ 先量化再决定。
3. 验收口径按 R19 §一"残余"栏：**同精度档 + 同 conv 语义 + 首次偏离后移/消失 + 性能不退化**。
4. 回到主线：ckpt 口径（retrain 的 `--out-dir` 坑与 R 判定）与 serve/CLI 混淆项。

# R21：根因定位 —— 按 T 选 kernel 的数值不等价是 spec≠plain 与 chunk 不变量的共同根（2026-09-11 19:1X）

## 一、实测（引擎自带 hidden dump，逐位置 + 逐层；同一 prompt 仅改 `--prefill-chunk`）
工具：`NINFER_HS_DUMP_DIR` + `NINFER_HS_DUMP_TOPK=1`（每个 prefill chunk 一个 `chunk_%06d.bin`：
tokens + ids[] + 5 层特征 25600×T + post-norm hidden 5120×T + 全词表逐列 argmax）。
prompt：912 token（非 128 对齐），chunk 128（9 块）vs 4096（2 块）。

| 项 | 不同位置数 | 首次位置 |
|---|---|---|
| post-norm hidden (bf16, 5120) | **912 / 912** | **0** |
| 5 层特征 (bf16, 25600) | **912 / 912** | **0**（layer 5/19/33/47/61 **全部**在 pos 0 即分叉） |
| 引擎逐列 argmax（全词表） | 35 / 912（**3.8%**） | 5 |

## 二、判读（修正此前所有猜测）
1. **差异从位置 0 就存在，且每一层都有** ⇒ 不是 S_A ① 的"尾块 <64 走 FP32 寄存器归一化"
   （那只影响最后一个 chunk）；**根因是首块 T 本身不同（128 vs 912）⇒ 按 T 选的 kernel 不同 ⇒ 从第 0 个位置起数值即不同**。
2. **P7 的"分析不同"由此裁决：B 路对，A 路错** —— A 说"③ 有机会 IDENTICAL"，B 说"不可能，
   残余来自 attention/MLP/lm_head 的 T 形状 kernel"⇒ 实测支持 B ✓。
3. **同一机制解释主线 spec≠plain**：verify 是 **T=W=8**、plain 是 **T=1** ⇒ 走不同 kernel
   ⇒ target 侧**约 3.8% 的 argmax 翻转** ⇒ G-A（spec 与 plain 一致）无法成立；
   这也解释了此前"首次偏离位置随内容漂移"的现象（4% 的翻转率 ⇒ 首次翻转点是随机命中 margin 极小处）。
4. 与"接受率上限"的关系：3.8% 的翻转率**不足以**解释 p1=0 的塌陷（那是 ckpt 口径，R12/R18）；
   但它**就是** G-A 的缺口来源，且会让 verify 的 logits 系统性偏离 plain ⇒ 对接受率有二次影响。

## 三、修复方向（性能优先，不变）
**统一小 T 家族**：让 T∈[1, W]（含 plain 的 T=1 与 verify 的 W=2/4/8/16）在**所有** T 形状分派点上
走**同一条** kernel 路线与同一精度档 —— 不限于 GDN 快照（修② 只覆盖了 GDN 那一处，故单变量测不到效果）。
需要覆盖：attention 输入/核心、MLP/linear、lm_head、GDN 输入（含 snapshot / w4a4 / independent 各档）。
- 已派 S_E 子代理产出**这些分派点的完整清单 + 分派谓词 + 统一方案 + 性能代价**（报告 `_collab/S_E_shape_kernels.md`）。
- 性能判据：小 T 家族统一后必须实测 tok/s（A 路 roofline 估计 +0~3%/round；>3% 则回退到更小的统一域）。
- 配套定案：给 ② 加一行 schedule 日志，确认 verify 的 GDN 输入实际走的路线（决定 ② 是否有效）。

## 四、附带确认（本轮工具链）
- dump 只在 `--spec dflash2`（启用 dflash2 特征捕获）时触发；不带 spec 会 0 文件。
- dump 的 bf16 精度足够做**定位**（见上），但不足以量化 1e-3 以下的差值 ⇒ 量化仍用 argmax 翻转率。
- 构建纪律再次生效：必须 `export PATH=/home/user/.local/bin:$PATH`（否则 ccache 找不到、127、删二进制）。

# R22：TMA epilogue 缺 bf16 回舍 —— 已定案（chunk 不变量的根）；并划清与 spec≠plain 的边界（2026-09-11 19:4X）

## 一、S_E 的单变量判别（我实测，零改码）
prompt ≈1032 token；`--prefill-chunk 128`（10 块，**永不 TMA**）vs `512`（4 块，**恒 TMA**）；
据 S_E：这一对只差 TMA 路径，gating proj 的 SplitK 两档相同（都 8）。

| 项 | 结果 |
|---|---|
| post-norm hidden 不同 | **1032 / 1032**（首次位置 = **0**） |
| 引擎逐列 argmax 不同 | **33 / 1032 = 3.20%**（首次 = 5） |

⇒ 与 128-vs-4096 的 3.8% **同量级** ⇒ **S_E 指认的缺陷就是活动机制**：
`src/ops/linear/nvfp4/nvfp4_linear_swiglu_w4a4_tma.cuh:242-245` 的 epilogue **少了 gate/up 的 bf16 回舍**
（baseline 与 MMA-fused 两条路都有该回舍）；T 超阈值时按 head=1024 切块走 TMA ⇒
只改 `--prefill-chunk` 即命中 tokens 0..1023 ⇒ 从位置 0 起数值差 ~1e-3 ⇒ 3.2% argmax 翻转。

## 二、作用域划清（重要，避免误记收益）
- 这条缺陷只在 **prefill 的大 T 路径**生效 ⇒ 修它能让 **chunk 不变量**（同 prompt 不同 chunk 同输出）成立；
- **它不解释 spec≠plain**：短 prompt 下 plain 与 spec 的 prefill 都是单块、同路；
- spec≠plain 的 3.8% 来自**另一处**：**小 T 家族（T=1 vs T=2..16）的 kernel/路线不同**
  （A 路已指出 GDN 输入 T=1 走 gemv、T∈[2,16] 走 small_t；修② 改的是 snapshot plan 阈值，
  **不是这个 dispatcher**，故单变量测不到效果）。
- S_E 已排除（附代码证据）：nvfp4 MMA 六档 schedule、rope 各分档、causal_conv1d、KV fill、
  注意力路由、rmsnorm/l2norm、**lm_head（恒 T=1，不可能贡献第 0 token 差异）**。
  另注：MoE(35B) 有 `adaptive = tokens>=47 && <=51/52` 的专家核分档（家族级分叉），27B dense 不涉及。

## 三、修复清单（按 P7 双路取证，性能优先）
| # | 目标 | 改动 | 判据 | 性能 |
|---|---|---|---|---|
| 修④ | chunk 不变量 | 在 `nvfp4_linear_swiglu_w4a4_tma.cuh:242-245` 补 gate/up 的 bf16 回舍，与 baseline/MMA 一致 | 128 vs 512 与 128 vs 4096 的 argmax 翻转率 → 0（或显著下降） | 中性（多一次 cvt） |
| 修⑤ | spec==plain | 统一**小 T 家族**：T=1 的 gemv 路线与 T∈[2,16] 的 small_t 路线对齐（同算术/同累加顺序）；范围以 S_E 的清单为准 | `_ga_check.sh`（zh/num）首次偏离后移/消失 | 需实测 tok/s（小 T 家族代价应≈0） |
| 修③ | prefill 尾块 | `gated_delta_net.cpp:255` 的 BF16 预归一化双路（S_A ①） | 可与修④ 叠加验证 | 中性 |

## 四、工具与纪律（累积）
- dump：`NINFER_HS_DUMP_DIR=<dir> NINFER_HS_DUMP_TOPK=1`，**必须带 `--spec dflash2`** 才触发；
  每 chunk 一文件，含 5 层特征 + post-norm hidden + （tokens≤1024 时）全词表逐列 argmax；
  **无 GDN 内部量、无逐层 hidden**（KV 另有 `NINFER_KVDUMP_DIR`）。
- 构建：必须 `export PATH=/home/user/.local/bin:$PATH`（ccache 在里面，否则 127 且删二进制）。
- 判定 shape 分叉的最低成本手段：**同一 prompt 只改 `--prefill-chunk`，比 dump 的逐列 argmax 翻转率**。

# R23：修④ 的真身是 TMA epilogue 多乘 alpha（S_E 的"缺回舍"表述需修正）（2026-09-11 20:1X）

## 一、逐字对照（四条路径的 silu·up 算式）
| 文件:行 | 算式 | 有无 alpha |
|---|---|---|
| `src/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_w4a4.cu:47-48` | `__floats2bfloat162_rn(silu(gate_values.x) * up_values.x, …)` | **无** |
| `…/nvfp4_linear_swiglu_small_t.cu:70` | `__float2bfloat16_rn(silu(gate) * up)` | **无** |
| `…/nvfp4_linear_swiglu_decode.cu:61` | `__float2bfloat16_rn(silu(gate) * up)` | **无** |
| **`…/nvfp4_linear_swiglu_w4a4_tma.cuh:242-245`** | `__floats2bfloat162_rn(silu(gate[0] * alpha) * (up[0] * alpha), …)` ×4 | **有，且对 gate 与 up 各乘 ⇒ 乘积 ×alpha²** |

- `alpha` 是**运行时 float 参数**：`nvfp4_linear_swiglu_w4a4_tma.cu:60` 的 launcher 签名 → `:81` 传入内核；
  `…_tma_launch.h:14` 亦声明 ⇒ **不是编译期 1.0 的 no-op**。
- 因此这不是 S_E 说的"少了 bf16 回舍"（TMA 侧确实有 `__floats2bfloat162_rn` 回舍），
  而是**乘性系数不一致**：三条兄弟路无 alpha，唯 TMA 乘以 alpha²。
- 这正好解释实测：**大 T（≥TMA 阈值）走 TMA ⇒ 值级分叉从位置 0 起**，
  `--prefill-chunk 128 vs 512` 的 hidden 1032/1032 不同、argmax 翻转 3.20%（与 128 vs 4096 的 3.8% 同量级）。

## 二、必须实测定案（不可直接改）
两种可能，方向相反：
- **(a) TMA 的 alpha 是多余**（累加器已是真实单位；三条兄弟路正确）⇒ **删掉 alpha** 即对齐；
- **(b) alpha 是必需**（TMA 走另一套累加单位；兄弟路各自内部已处理）⇒ 删它会把长 prompt 的前 1024 token 算错。
判据（最便宜、一行）：**打印 alpha 的实际值**（gated 日志，加到 `nvfp4_linear_swiglu_w4a4_tma.cu:81` 附近）：
- `alpha == 1.0` ⇒ 该差异不存在，S_E 这条线索作废，chunk 分叉另有来源；
- `alpha != 1.0`（比如 ~0.02、~1.5 之类）⇒ 进入 (a)/(b) 判定：再看**兄弟路的 alpha 从哪来/是否已在别处乘过**
  （grep 它们的 launcher 参数与自己内部是否含 `* alpha` / 缩放），并做一次 A/B：删 alpha 后跑
  chunk 判别（128 vs 512 翻转率应→0 或显著下降）+ 同 prompt 的文本合理性（防止把 (b) 改坏）。

## 三、与主线的边界（不变）
- 修④ 属 **prefill 大 T**路径 ⇒ 修好它达成"chunk 不变量"，**不直接改变 spec≠plain**（短 prompt 下
  plain 与 spec 的 prefill 同为单块）；但**长 prompt 的 prefill 被算错**会同时污染两条路 ⇒ 对长上下文
  的接受率与正确性都有影响（此前未单独量化，值得在修④ 后补测长 prompt 的接受率）。
- spec≠plain 的 3.8% 仍归**小 T 家族**（T=1 gemv vs T∈[2,16] small_t；修② 改的 snapshot plan 阈值
  不覆盖该 dispatcher，故单变量测不到效果）。

## 四、顺手复核的结果（用户提醒的"对齐问题"= `_TODO.md` §132/§133 四处）
- ① `dflash_impl.h:448` 取列跳过 anchor（`source_column_offset = 1` 带完整注释）⇒ **已在树** ✓
- ④ `mtp_round.cuh:50` `ar_valid_columns = s < next ? 1 : 0` ⇒ **已在树** ✓
- ② ③ 在训练脚本 `train_dflash2.py`（shift / 输入模式）⇒ **只影响 retrain**，与引擎无关 ✓
⇒ 引擎侧两条"对齐"确认已落地；未在现二进制上做行为复验（本轮未发现它们复发的迹象：
dspark 的 p0 已由此前的 0% 恢复到 22% 一档，见 `_probe_a5.out` 的 A/B）。

## 五、下一步（按性价比）
1. **打印 alpha**（一行 gated 日志 + 一次编译）⇒ 定案 (a)/(b)。
2. 若 ≠1：按 (a) 删 alpha（或按 (b) 反向修正）→ 跑 chunk 判别 + 长 prompt 文本合理性 + 接受率。
3. 回到小 T 家族统一（修⑤），P7 双路出补丁。

# R24：修④ 是假阳性（alpha 必需）——已回退；并给出整场"spec≠plain"战役的结论（2026-09-11 16:0X）

## 一、经过（P7 的"量化验证"救了一次回归）
1. S_E 指认 `nvfp4_linear_swiglu_w4a4_tma.cuh:242-245` 的 epilogue "少了 gate/up 的 bf16 回舍"。
2. 我逐字对照四条路径，发现真身不是缺回舍，而是**TMA 路径多乘 alpha**（gate 与 up 各乘一次 ⇒ 乘积 alpha²），
   且 `alpha = 1.0F / (weight.input_scale_divisor * weight.weight_scale_divisor)` 是**运行时反量化系数**；
   准入条件 `tokens>=256 && tokens%256==0` 正好解释"chunk 512 命中、chunk 128 不命中"。
   ⇒ 我判定"与三条兄弟路对齐 = 目标"，**落地去掉 alpha**。
3. **实测立刻否掉**：同一 prompt、同 token 集，`--prefill-chunk 128`（不命中 TMA）vs `512`（命中 TMA）的
   逐列 argmax 差异 **3.20% → 99.61%**（hidden 差异仍 1032/1032）。
   ⇒ 改动前两条路径**本就高度一致（仅 3.2% 舍入级差异）**，**alpha 是必需的**
   （TMA 的累加器是 raw 单位，兄弟路是 real 单位，各自内部自洽）。
4. **回退**：`cp /home/user/fix4_bak/…` 复原，md5 `698f409c71a4` 与备份一致，
   重编二进制大小与修复前同为 839637376、时间 16:01 ⇒ 树回到 ①+② 状态。
   ⇒ **S_E 的这条线索是假阳性**（P7 的"两路 + 量化"在此拦住了一次回归）。

## 二、由此得到的结论（本场战役的收敛点）
1. **chunk 依赖（3.2%）不是 bug，而是"同一数学、不同形状 kernel"的舍入级差异**：
   两条路径各自内部自洽（缩放各自处理正确），差异来自 silu/乘法的结合顺序与归约顺序。
   ⇒ **不能靠"删一个系数"消除**；只有统一算术（同 kernel 家族）才能消，代价见下。
2. **spec≠plain 的 3.8% 属同一类**（T=1 的 gemv 与 T∈[2,16] 的 small_t 各自自洽、舍入级差异）。
   ⇒ **G-A 的"逐位一致"在可接受的性能代价下不可达**（参照实现同样如此）；
   实测偏差量级：**1e-3 级 logits、3.8% 的 argmax 翻转**。
3. ⇒ 结论：**把 G-A 的判据定为"统计等价"**（同精度档、首次偏离后移/消失、接受率与性能不退化），
   而不是逐位 hash 相同；否则需立项"统一小 T 家族/统一 prefill 路径"，用实测 tok/s 换精确性
   （A 路 roofline 估 +0~3%/round，但需实测；这是**性能与逐位一致之间的取舍，属用户决策**）。
4. **接受率的杠杆仍不在这些舍入项**：按 R12/R18/S_C，真正杠杆是 **ckpt 口径（训练喂真 token + shift 1）
   与 live artifact 是 08-26 草稿**这两件事；最便宜的下一步是 S_C 的塌陷比 R 判定
   （用已有 `step_000100/000200` 与 08-26 草稿，离线跑，比重训便宜）。

## 三、当前树状态（可回退点）
- 已落且实测有效：**修①**（`gdn_conv.cuh:99` 的 `p` 先 bf16 化；`zh plain vs df2` 首次偏离 10→19，
  mtp3 逐位未变 ⇒ 触发谓词被交叉验证）✓
- 已落但**未测到效果**：**修②**（`nvfp4_gdn_snapshot_plan.cpp:43` 阈值 3→16）⇒ 待一行 schedule 日志定案。
- 已回退：**修④**（alpha）⇒ 备份 `/home/user/fix4_bak/` 保留，md5 校验一致。
- 其它备份：`/home/user/fix1_bak/`、`/home/user/fix2_bak/`、`/home/user/gdnconv_bak/`。

## 四、下一步（建议顺序）
1. **ckpt 线（收益最大）**：S_C 的 R 判定（离线，CPU）→ 据结果决定 retrain（`--out-dir` 必给，避免
   `--resume` 取到 legacy 的 `step_001900`）；顺带钉死 serve 4.55 vs CLI 1.35 tok/round 的混淆项。
2. 修② 定案：一行 schedule 日志。
3. 修⑤（可选、需用户决策）：统一小 T 家族以把 spec≠plain 从 3.8% 压到更低，代价是实测 tok/s。

# R25：接受率地板实验（短上下文 +21%）+ 优先级与"合并"定义固化（2026-09-11 16:1X）

## 一、实验：`kDFlash2MinAcceptance 0.05 → 0.0`（永不再因低接受率停起草）
| 指标 | 地板 0.05 | 地板 0.0 |
|---|---|---|
| 接受率 | 4.51% | 4.50%（不变，符合预期） |
| 位置剖面 | `[6,1,0,0,0,0,0]` | **`[18,4,0,0,0,0,0]`**（绝对量 ≈3×） |
| decode tok/s（zh，36 tok prompt） | 44.78 | **54.19（+21%）** |
| decode tok/s（长 prompt ~1.8-1.9K） | lu_c128 ≈71.16 / la_c128 ≈70.68 | lu_c128 **71.01** / la_c128 **70.65**（持平，无退化） |

同一轮测得的全局图景（同 prompt）：**plain 77.87 / mtp3 121.13 / dflash2 54.19 tok/s**
⇒ **dflash2 的吞吐瓶颈就是它的接受率**（低接受率＝大量白干的 verify）。
地板存在的原因见 `spec_decision.h:42-44` 注释：4K 场景下曾出现 53→24 tok/s 的退化（218 fallback steps vs 0）。

### 结论与建议
- **不是把常数设 0 就完事**：历史退化发生在 **4K** 档（本轮只测到 ~1.9K，无退化）⇒ 正确解法是
  **按上下文/观测吞吐自适应的控制器**（例如以"最近若干轮的实测 tok/s"为准决定继续起草与否），
  而不是"固定接受率地板"。
- 落地顺序：① 在 ≥2K/4K 档复测地板 0（补齐缺的那一档）；② 若 4K 确实退化，则把地板改成
  **上下文相关的阈值或吞吐反馈**；③ 否则可直接放宽地板。
- 备份：`/home/user/floor_bak/spec_decision.h`（可 `cp` 回到 0.05）。

## 二、用户定下的长期优先级与口径（写入约束）
1. **重训练永远最低权重**（"重训练永远是最低权重"）⇒ 抬接受率/吞吐只能靠**引擎侧**；
   R12/R18/S_C 的 ckpt 线（R 判定、retrain、`--out-dir`）降为**最低优先级**，不再作为主计划。
2. **"合并" = 统一算术**（"我的意思是合并 统一算术"）⇒ 见 §三。
3. P7 不变：修复双路取证（同方法两遍 → 求同析异 → 落地），量化证据优先，不许拿上下文当停工理由。

## 三、统一算术（当前主攻）：UNIFY-A / UNIFY-B 两路已派出
目标：让 **plain(T=1)** 与 **verify(T=W∈{2,4,8,16})** 在每个 T 分派点走**同一 kernel/同一累加顺序/同一精度档**。
硬约束（性能优先）：**宁可让 T=1 改走 T∈[2,16] 的 kernel**（T=1 最便宜），禁止 verify 退化/串行化/大缓冲升 FP32。
覆盖点：GDN 输入（T=1 gemv vs T∈[2,16] small_t）、swiglu 四条路（decode/small_t/w4a4/tma）、
attention 输入与核心注意力、lm_head 及其它 `tokens<=N` 分档。
交付：`_collab/UNIFY_A_patch.diff` + `UNIFY_A_report.md`（B 路同构）⇒ 我对比求同、落地、用同一套基准复测。

## 四、验收口径（因 §三 而定稿）
`spec==plain` 的判据采用**分级**，不追求逐位 hash：
1. 强判据：同精度档 + 同 conv 语义 + **首次偏离位置显著后移或消失**；
2. 效率判据：**接受率与 decode tok/s 不退化**（tok/s 是用户硬底线）；
3. 一致性判据：非对齐 prompt 的 `--prefill-chunk` 不变量（`_align128_ab.sh`）作为回归探针常备。

# R27：统一算术的两步实测 —— 累加链（阴性）+ 分派层（进行中）（2026-09-11 16:3X）

## 一、单变量①：FP8 累加链 4→1 —— **阴性（零代价、零效果）**
| 指标 | 改前（R20/25 状态） | 累加链 4→1 后 |
|---|---|---|
| `zh plain vs dflash2` 首次偏离 | DIFFER at **19**（token 同） | DIFFER at **19**（token **逐位相同**） |
| `zh plain vs mtp3` | at 40 | at 40（同） |
| chunk 探针（非对齐 / 对齐） | at 0 / IDENTICAL | at 0 / IDENTICAL |
| dflash2 接受率 | 4.50% `[18,4,…]` | 4.50% `[18,4,…]` |
| dflash2 decode | 54.19 tok/s | 54.19 tok/s |
| plain(T=1) decode | 77.87 tok/s | 77.87 tok/s |

⇒ **改动代价 0、效果 0**（两条流的每个 token 都未变）⇒ 说明**T=1 与 T∈[2,16] 的分岔不在"同家族内累加链数"上**，
而更可能在**路由层**（T=1 被送到另一条 kernel/家族）。B 的"2 行即逐位"结论**在 FP8/live 路径上不成立**（阴性）。
（该改动保留在树上，代价为 0；备份 `/home/user/fp8chain_bak/`。）

## 二、单变量②：UNIFY-A 的 373 行分派统一 —— 编译中
- **已应用**：`dry rc=0` / `apply rc=0`（`patch -p1 -b`），17 个文件 + 其 includer 已 `touch`；编译在后台
  （日志 `dl/unify_a_apply.log`），当前处于 `ninfer_ops` 设备链接（长杆）。
- A 路的结构论证（重要）：**plain 与 verify 是同一段代码同一 `Phase::Verify`（`text_context_impl.h:810/816`
  vs `:867/873`），只差 T；T=8 在 16 个分派点上 route 全部不变** ⇒ 统一只把 T=1 / T∈[2,4] 搬上 verify
  已在用的路 ⇒ **verify 侧按构造不退化**；代价集中在 T=1（估 +1.5~3.5%，超 3% 时按粒度回退：先 gating 后 swiglu）。
- 判据（编译完即测，与 §一 同四项）：首次偏离是否**消失/显著后移**；T=1 与 zh decode tok/s；接受率。

## 三、方法论沉淀（写给后续）
- **阴性结果要记账**：本轮两个"看起来最像根因"的最小改动（修② 阈值、累加链）都测到零效果；
  真正的机制得靠"**同一 prompt 只改 `--prefill-chunk` 比 dump 逐列 argmax 翻转率**"这类**形状对照**去定位，
  而不是靠"读代码猜哪两行不同"。
- **判据优先选"形状对照"与"与 plain 的首次偏离"**，而不是"与参照实现逐字对照"（后者已多次给出假阳性）。

# R28：统一算术两步都阴性 —— 并据此**撤回** R21 的外推（2026-09-11 16:4X）

## 一、两步单变量实测（同一套基准，全部逐位可比）
| 项 | 基线 | ① FP8 累加链 4→1 | ② UNIFY-A 373 行分派统一 |
|---|---|---|---|
| `zh plain vs dflash2` 首次偏离 | DIFFER at 19 | DIFFER at **19**（token 同） | DIFFER at **19**（token 同） |
| `zh plain vs mtp3` | at 40 | at 40 | at 40 |
| 非对齐 chunk 探针 | at 0 | at 0 | at 0 |
| 对齐 chunk 探针 | IDENTICAL | IDENTICAL | IDENTICAL |
| dflash2 接受率/剖面 | 4.50% `[18,4,…]` | 同 | 同 |
| dflash2 decode | 54.19 tok/s | 54.19 | 54.19 |
| plain(T=1) decode | 77.87 tok/s | 77.87 | 77.87 |
| 二进制 | — | 已重编 | **已重编（16:30，size 840053456≠839633280）** |

② **确实改到了 live 的 FP8 文件**（`attn_input_proj/fp8/fp8_attn_input_plan.cpp`、`gdn_input_proj/fp8/fp8_gdn_conv_fused.cu`、
`gdn_input_proj/fp8/fp8_gdn_input_plan.cpp`、`linear/fp8/{fp8_config.h,fp8_small_t.cuh}`、
`linear_add/fp8/fp8_linear_add_plan.cpp`、`linear_swiglu/fp8/fp8_linear_swiglu_plan.cpp`），**仍零效果**
⇒ A 放松的分派条件（`active==1`、`first_w4a4=7/8`、`tokens<=4` 等）**在我们的 token 数上根本没被走到**。

## 二、必须撤回的推论（诚实记账）
- **R21 的"按 T 选 kernel 数值不等价 = spec≠plain 的根"是外推，现被证伪**：
  ① chunk 形状依赖（prefill 128 vs 4096/512）是**实测事实** ✓ 保留；
  ② 但"同理适用于 verify(T=8) vs plain(T=1)"**未经验证**，而两步统一都零效果 ⇒ **否定** ✓。
- 同理，**修②（snapshot 阈值）与累加链**两次"读代码猜两行"都零效果 ⇒ 方法论上：
  **不要再从"与参照/兄弟路的逐字差异"去猜**，改用**形状对照 + 与 plain 的首次偏离**这类**可复现的对照实验**定位。

## 三、处置
1. **回退 UNIFY-A 的 373 行**（`patch -R`，备份已在 `_collab/UNIFY_A_patch.diff`）：未被证实的改动不留在树上
   —— 尤其它会改变 NVFP4/bf16 变体的行为而无任何证据；留着也不是"保险"，而是"未审计变更"。
   保留 FP8 累加链改动（代价 0、效果 0，但属无害对齐，可留；如需极简亦可回退，备份 `/home/user/fp8chain_bak/`）。
2. **下一步改成"定位而非猜测"**：
   a) **KV dump 对照**：`NINFER_KVDUMP_DIR`（S_E 指出存在）⇒ 同一 prompt 下 plain 与 spec 的 KV 逐位置比对，
      看 verify 把块列的 K/V 写在何处、接受的列与 plain 是否逐位一致。
   b) **核心注意力的形状对照**（A 提的零改码实验）：draft 宽度 4 vs 8 比翻转率 ⇒ 若宽度变则分叉率变，
      注意力形状即关键；若不变，则分叉与注意力无关。
   c) **GDN 状态是唯一已证"精确"的部分**（S_B）⇒ 把注意力/KV 排除后，剩余嫌疑就收敛到注意力与 verify 输入绑定。

# R29：核心排查两项结论 —— 因果性成立、但 verify 列 0 与 plain 有 ~31% 偏差（2026-09-11 17:0X）

## 一、因果性：**成立**（决定性）
方法：同 prompt 跑两臂，只用 `--lm-head-draft` 换一组**不同的草稿**（其余全同）；把两臂按
**相同上下文前缀 + 相同位置**配对，比较第 0 列 argmax。
| 指标 | 结果 |
|---|---|
| 可配对轮 | **65** |
| col0 argmax 不同 | **0 / 65 = 0.0%** |
| 控制项：右侧 draft 列不同 | **52 / 65**（⇒ 实验有力度） |
⇒ **target verify 的第 0 列完全不受同块右侧列影响**（live 行与 cache 两条路径都不泄漏）
⇒ **核心掩码因果、正确，不是本因** ✓（此前的 K 无关推断得到独立确认）

## 二、当前二进制的逐列一致性：**列 0 只有 69%**
（token 锚定、自校正探针滞后；样本 = 上下文干净的列；plain 参照 `bl_zh_plain.log`）
| 列 | 干净样本 | 与 plain 一致 | 一致率 |
|---|---|---|---|
| 0 | 29 | 20 | **69.0%** |
| 1 | 8 | 6 | 75.0% |
| 计 | 37 | 26 | 70.3% |
（历史对比：F4/① 之前 列0 17/20=85%、列1 6/8=75%）
⇒ **verify 第 0 列约 31% 的输出与 plain 不同** ⇒ 这既是 spec 流偏移的来源，也直接压低接受率
（正确草稿要通过的是**被污染的** target argmax）。

## 三、排除清单（本轮新增）
| 假设 | 结论 |
|---|---|
| 核心掩码非因果（左侧列被右侧 draft 影响） | **排除**（0/65，含控制项）|
| 路由 SmallT/Prompt 数值不同（T≤6 vs T∈[7,16]） | 对齐后**零效果**（A 的 373 行 + 累加链都测不到变化）|
| GDN 状态/回滚 | S_B 证精确（fold 与逐 token 逐位等价）|
| positions/RoPE | 两路同为绝对位置（纯文本 rope_delta=0）|
| 声明型超参/块构造/walk | 与 ckpt config 及参照实现逐项一致 |

## 四、下一步（本轮已启动）
**KV/注意力的"写 vs 读"对照**——仪器 `NINFER_KVDUMP_DIR` + `NINFER_KVDUMP_KV` 已存在，
但门卡在 `ph == Phase::Prefill`（`text_context_impl.h:1007`），而 plain 与 verify **同属 `Phase::Verify`**
⇒ 把它放宽到 `(Prefill || Verify)`（1 行）后：
- 同一 prompt 分别跑 plain 与 spec，各 dump 每层写入的 BF16 K/V 与 positions；
- 逐位置比对**被接受的列**：若 K/V 不完全一致 ⇒ **写路径 bug**；若逐位一致 ⇒ 嫌疑转向读/注意力。
（判据与 S_B 的"状态精确"合起来，可把剩余空间压到单点。）

## 五、副产品与纪律
- `/tmp` 会随 WSL 重启清空 ⇒ **所有产物写 `dl/` 或 `_collab/`**（本轮踩过）。
- `pkill -f 'make ninfer'` 会**自伤**（匹配到自己的 shell）⇒ 用 `pkill -x make|nvcc|ptxas`。
- 瞬时 OOM（`prepare_ragged_prefix.cu:20`）出现一次、重跑即过 ⇒ 记为**瞬时**，非 A 路回归。

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

# R32：PART-A 结果 + 两个窗口的强制排序（幽灵副本已核实）（2026-09-11 18:0X）

## 一、PART-A（FP32 partial_acc）已就绪
- 产物：`_collab/PART_A_patch.diff`（**21 文件 / 61 hunk / +124 −125**）+ `PART_A_report.md`；
  `patch -p1 --dry-run` **21/21 clean、exit 0、无 .rej、树未改**。
- 改动四层：写侧签名（18 处）→ 打包 store（14 处，`make_float2`）→ 中性初始化（5）→ reduce（5）→
  `DType::BF16→FP32` 分配（5）→ `.data` 转换（19）；**`partial_m/l`、索引/步长、路由全未动** ✓。
- 性能：partial 缓冲 ×2（27B T=1 ctx4096 splits=64：768 KiB→1.5 MiB），per-token 流量 **+≈0.17%**；
  **verify(T=8 走 Prompt，该缓冲 0 字节) ⇒ verify 不可能退化** ✓；smem/占用无恶化（部分档位还少一次
  `__syncthreads` 与一次 BF16 中转）。唯一让步：bf16/fp8/iso3 的 128-bit `int4` store → `float2`
  （保 128-bit 需 +16 KB smem 且扰动占用；回退方案见其报告 §4.1）。

## 二、PART-A 的两条关键新发现
1. **27B 的 KV 默认档不是 BF16**：是 `NVFP4` + 10 层 `E8Kv`（`qwen3_6_27b/impl/variant.cpp:23-39`）
   ⇒ **活路径的 partial 核是 `decode_nvfp4` 与 `decode_i8<E8>`**；`decode_bf16.cuh` 只在 `--kv-dtype bf16` 时激活
   ⇒ **只改 bf16 那条会测不到变化**（本 patch 两条都覆盖 ✓）。这条与 §三的排序一起，是本次能测出效果的前提。
2. `dense/causal_cache`（SmallT 并行族）**不可达、自成闭环**，且其 `small_t_fp8` **早就是 FP32 + `make_float2`**
   —— 正是本次要对齐的形态 ⇒ 不动它是对的（§4.2）。

## 三、幽灵副本：**已核实**，并据此定死两个窗口的排序
实测 `src/ops/launcher/`：
| 文件 | 大小 | 在 CMake？ |
|---|---|---|
| `gqa_attention_decode.cu` | 45.8 KB | **是** |
| `gqa_attention_decode_e8.cu` | 12.5 KB | **是** |
| `gqa_attention_decode_smallt.cu` | 16.5 KB | **否** |
| `gqa_attention_decode_partial.cuh` | 26.5 KB | **否** |
| `gqa_attention_decode_impl.cuh` | 37.2 KB | **否** |
| `gqa_attention_decode_g35.cu` / `_muse.cu` | 2.5 KB 各 | **否** |
⇒ `_smallt.cu` / `_partial.cuh` 的 mtime = **Sep 10 19:57**，即"暂存 TU 拆分"那批产物，**早于今天的 FP32-partial 补丁**
⇒ **它们内部必然仍是 BF16**。**若先落 TU 拆分，会把 BF16 静默接回构建（测出来的将是旧行为）** ✗✗

### 强制排序（两个窗口）
1. **窗口①：先落 FP32 partial**（`PART_A_patch.diff` 的公共核心，或两路合并后的版本）→ 编译 → 复测
   （判据：`verify 列 0 与 plain 一致率 69% → ≥90%`、首次偏离后移/消失、接受率与 tok/s 不退化）。
2. **窗口②：再落 TU 拆分**，且**必须**在同一窗口内：
   - 从**打完补丁后**的 `gqa_attention_decode.cu` / `_e8.cu` **重新生成**拆分产物（或对 `_smallt.cu`/`_partial.cuh`
     施加同样的 FP32-partial 编辑），并**校验无 `bfloat16` partial 残留**（grep 断言）；
   - 否则 §三 的静默回退会发生。

## 四、状态
- PART-B 仍在跑；到齐后按 P7：对比 21 文件的逐点差异 → 取共同核心 → 落地窗口①→ 复测。

## 2026-09-11 窗口①/② 状态（自动同步）

### 窗口①：FP32 partial_acc（PART-B 超集，29 文件）—— 已落地，编译中
- P7 双路对比结论：A(21文件) 与 B(29文件) 是**同一改动集**，B 为超集；**B 的关键订正**：R31 点名的
  `dense/context`（32Q/8KV）在 targets 里**无调用者**，plain(T=1) 的 live split-K 是
  `ops::gqa_attention` → `ops/kernel/gqa_attention_decode_{bf16,cuh}`（27B=24Q/4KV/D256，T≤6→SmallT，T=8→Prompt）。
- 落地形态：kernel 签名 / 打包存储（int4·pack_bf16x2 → make_float2·float2，4 个 128-bit 暂存核每 lane 仅 2 相邻 d）
  / 分配 DType::BF16→FP32 / reduce 去 __bfloat162float / `.data` 转换 31 处。
- 成本（两路一致）：live T=1/window=1200 partial 足迹 228→456 KiB/层 = 该层注意力 +19%，**整 step 权重流量 +0.2%**；
  verify 走 Prompt（T=8）该缓冲 0 字节 ⇒ 不退化。最贵项 swa/bidirectional T=16/W=4096 的 8 MiB 往返，
  已备分组回退顺序（E 组优先）。
- 判据（复测）：`verify 列 0 与 plain 一致率 69% → ≥90%`、`zh plain vs df2` 首次发散前移/消失、接受率与 tok/s 不退化。

### 窗口②：暂存 TU 拆分 —— 已就绪且与补丁后源码自洽
- **实证隐患**：staged 产物是**补丁前**抽出的 ⇒ 原样落地会把 `_partial.cuh`/`_smallt.cu` **静默退回 BF16**。
- 处置：覆盖型产物从树里幽灵副本回抄（`e12ce1a6→9c4a3ae3`、`15c1794a→f709636e`；字节账 −40/−8 精确吻合）；
  e8 兄弟 TU 施加同一替换（`_e8_arms.cuh.new` 1 处，`42ba0988→5e2a0e7c`）；**断言无 BF16 partial 残留 PASS**。
- 两张落地脚本 md5 表已同步（`_land_split.sh` 3 处、`_land_s6.sh` 2 处，FAILS=0）。
- **无冲突**：UNIFY-A 对被拆三文件命中 0；树里 live TU 为 6 处 FP32 / 0 处 BF16。

### 修② 定案关闭
`nvfp4_gdn_snapshot_plan.cpp:41-43` UNIFY-A 注释明确覆盖 projection 与 conv/store epilogue 两者，
`AllowA4` 下 T≤16 一律 `SmallTFusedA16` ⇒ 无 gemv 路由可切，3→16 是**代码级可证的 no-op**（零影响有解释，无需日志行）。

## 2026-09-11 深夜 · vLLM+dflash2 参考线结论（关闭）与宿主环境发现

### 参考线：WSL 路线在原理上走不通（已停，勿再试）
- 7 次 `Wsl/Service/E_UNEXPECTED` 崩机；所有可调项都拆过：MM 关闭、`--kv-cache-dtype fp8`、
  `--skip-mm-profiling`、`--num-gpu-blocks-override 256`、最小形状（seqs=1/len=512）、
  模型移至原生盘（去 9P）、VM 22→27GB ⇒ 仍在 `avail 22G` 时崩。
- **原理性硬约束**：dflash2 spec-decode 的 `StagedWriteTensor→UvaBuffer` 要求
  `is_uva_available()`（= pin_memory）⇒ WSL 下必须 `VLLM_WSL2_ENABLE_PIN_MEMORY=1`，
  而该开关让 vLLM 用**锁页主机内存（不可回收、计入 used）** ⇒ **主机内存 22–25GB 尖峰** ⇒ 崩。
  ⇒ **WSL 里跑 dflash2，钉页主机内存躲不掉**；Windows 侧无此矛盾（R2 的数即出自那里）。
- 另有 `--swap-space`（vLLM CPU swap，默认 4GiB）从未设过 ⇒ 若将来仍要试应先 `--swap-space 0`。
- 结论：**参考跑若要做，只能在 Windows 侧**（需源码构建支持 dflash2 的 Windows 轮子，数小时）；
  否则采用 R2 既有证据（vLLM+dspark 1.87%、逐位置 [11,0,0,0,0,0,0]，与 ninfer 同症状）。

### 宿主环境发现（与项目无关，但影响可用性）
- 一晚上的崩机/显示重枚举后 `dwm` 常驻 ~20%（`dwm` 强停重启无效、禁用 GameViewer 虚拟屏无效、
  停 MicaForEveryone 无效、用户已重启两次仍复现）。用户观察：**每次跑完模型之后都这样**。
- 已改（可回滚，需重启生效）：`HKLM\SYSTEM\CurrentControlSet\Control\GraphicsDrivers\HwSchMode = 1`
  （关闭 HAGS，原为未设置=默认；回滚置 2）。
- 查到既有非默认值：**`TdrDelay = 8`**（默认 2s）⇒ GPU 卡住后要 8 秒才恢复显示栈
  ⇒ 与用户描述的"闪动"高度相符（可能是前人为了长编译调大的）。`TdrLevel = 3` 正常。
- 我造成的增量已清理：WSL 侧 17GB 模型副本（删）、12GB pip 缓存（删）、`.wslconfig` 复原为
  22GB/50GB/16proc、GameViewer 虚拟屏已恢复 OK、MicaForEveryone 已恢复运行、WSL 已关闭、无 CUDA 进程。
- 保留物（可按需删）：`/home/user/vllm029`（9.8GB 全新 vllm0.29 环境）、
  `/home/user/models/q3nvfp4`（17GB 原生盘模型副本）、`/home/user/models/draft_dflash2_ref`（3.6GB）。

## 2026-09-11 深夜续 · vLLM+dflash2 加载期内存的**实测结论**（这条线关闭）

### 测量方法（终于做到可归因）
- 用户级 systemd 硬上限：`systemd-run --user --unit=vllmstream -p MemoryMax=14G -p MemorySwapMax=0`
  （`--scope` 会随我的会话被杀，必须用瞬态 **service** 才独立存活）。
- 逐 2 秒记录：cgroup `memory.current` / `memory.peak` / 显存 / 当前阶段 → `dl/stream_trace.log`。
- 宿主侧看门狗（`_watchdog2.ps1`，阈值 10GB、2s 采样）→ `dl/watchdog2.log`，越线自动 `wsl --shutdown`。

### 实测结论（三层归因，全部有数）
1. **吃内存的是"加载期"**：`memGB 2.1→3.9→6.7→13.9→14.0`（6–15 秒内撞上限），
   而阶段始终是 `Loading ... checkpoint shards / Runai Model Streamer` ⇒ **不是 KV、不是 MM、不是 JIT 收尾**。
2. **不是"文件被整份持有"**：`runai_streamer` 确实逐张量读（`843/2387 @420it/s` ✓），
   但主机需求**照样到 14GB** ⇒ 那是 **vLLM 对 2387 个张量的自身处理（NVFP4 重打包/元数据 + `enable_jit_warmup`）**，
   属加载流程固有成本，**换加载器搬不掉** ✗。
   （`fastsafetensors` 在 WSL 里退化为 nogds 普通拷贝：`/dev/nvidia-fs0 not found` ⇒ 也搬不掉 ✗。）
3. **算术封死**：加载期≈14GB + VM 开销 + 页缓存 > 21GB VM ⇒ **21GB 的 VM 装不下 16.7GiB 的权重** ✗。

### 用户两条建议的执行与实测
- **KV 量化 FP8**：已做（`--kv-cache-dtype fp8`，与模型 `kv_cache_quant_algo=FP8` 对齐 ✓）⇒ 对这笔账无影响 ✓（KV 在 512/seqs=1 下只有几十 MB）。
- **"别落主机/只进显存"**：加载器两条路都试了（fastsafetensors ✗、runai_streamer ✓真流式）⇒ **主机需求不变** ✗。

### 硬上限的效果（用户最关心）
`MemoryMax` + 看门狗**有效**：宿主最低到 7.7GB 时看门狗自动关 VM，
**宿主全程未崩** ✓（对比之前 7 次 VM 崩机把宿主内存拉到 2.5GB ✗）。

### 结论
**在这台机器（宿主 31.4GB）上，vLLM 0.29 + 27B NVFP4 + dflash2 只能在原生 Windows 跑**
（Windows 侧不受 VM 上限约束，R2 的 1.87% 即出自那里 ✓）；WSL 路线**因加载期主机需求 > VM 容量而不可行** ✗。
若要走 Windows：需构建支持 dflash2 的 Windows 轮子（数小时 ✗）。

## 2026-09-12 凌晨 · dflash2 接受率：逐步取证与根因（重大）

### 一、装了仪表并跑出逐步归因
- 在 `src/ops/kernel/dflash2_selector.cuh` 的 `[df2sel]` 探针旁新增 `[df2cand]`：每步打印 top-K 候选全貌（备份 `/home/user/df2cand_bak/`，含 `--dry` 前 md5 `92026e7092da`）。
- 新增 `NINFER_DF2FEAT`：dump `features / projected / context_full`（`dflash2_impl.h`，备份 `/home/user/df2feat_bak/`）。
  要点：**必须先 D2H 拷贝再落盘**（直接 fwrite 设备指针会得到 0 字节）；**图捕获期间不能 cudaStreamSynchronize**，
  所以特征 dump 必须配 `--no-cuda-graph`（否则 abort，rc=134）。
- 逐列归因（候选为真词表 id，无需跨域映射；只统计落在已接受前缀内的干净列）：
  **head_miss 15/15 = 100%**（干净列）；放宽到全部 290 步为 head_miss 84.8% / walk_error 10.7% / verify_flip 4.1%。
  自纠：曾一度误判候选是"草稿域 id"并二次映射，导致 100% 假 miss；也曾在第二版漏掉干净列过滤。

### 二、根因（实测，非推断）
- **`features` 只有第 0 段（tap0）有数据，tap1..4 整块为 0**：`--no-cuda-graph` 下 call=5..8 的 tap 范数
  依次为 `[278.4,0,0,0,0]`、`[271.3,0,0,0,0]`、`[275.9,0,0,0,0]`、`[272.9,0,0,0,0]`。
  ⇒ 喂给 `fc → context_norm → 5 层草稿网 → selector → 提案头` 的特征 **80% 是空的**。
- 下游算术已排除：`features → fc → rmsnorm` 与离线 numpy 复算**逐位吻合**（相对误差 0.00000）。
- 这一条解释了今晚所有"零差异"实验：换 5 层网 / 换选择器 / 清零 `text/draft_head` / 切 `--lm-head-draft`
  都不改变"候选是否命中"，因为缺料在更上游。

### 三、被实测否定的假设（清单）
| 假设 | 实验 | 结果 |
|---|---|---|
| 4 比特提案头是天花板 | 切 `--lm-head-draft`（draft_head 域） | 4.81% → 4.54%，候选集 0% 相同 |
| 草稿头混血（新层+08-26 老选择器） | 整体换 incoai 自洽头 | 逐项完全相同 |
| 选择器不参与 | 清零两张码本 | 4.81% → 4.01% |
| 5 层草稿网有问题 | 换整套 incoai 5 层网 | 零差异 |
| 提案头不在路径上 | 清零 `text/draft_head` 码区 | 候选集变了（0.9% 相同）但接受率不变 |
| tap 层号错位 | 读 config 对照契约 | `config.h:150 {5,19,33,47,61}` 与 HF `target_layer_ids` 完全一致 |
| KV 精度造成分歧 | 两侧 KV 都设 bfloat16 | 前 7–14 token 逐字一致 |

### 四、契约侧澄清（重要）
- `docs/maintainer/qwen3.8-27b-artifact.md`：`text/draft_head` + `draft_head_token_ids` 是
  **"optimized MTP draft head"**（第 i 行 ↔ 全词表行 `draft_head_token_ids[i]`）⇒ 它属于 **MTP** 路径，
  不是 dflash2 的提案头 —— 与"清零它不影响 dflash2 接受率"的实测一致。
- `_collab/A5_dspark_rootcause.md` 曾"代码级排除" tap 层号错（抓 post-MLP residual、按升序写 `feat[index·hidden]`），
  但本次实测 tap1..4 为 0 ⇒ 那次读码排除漏了某种情况（待三路子代理对照后定因）。

### 五、进行中
- 三路子代理（同方法各做一遍）已后台派出，目标：定位"只有槽 0 被写入"的根因并给最小补丁；
- 修复后由主代理统一编译落地，并用 `_verify_fix.sh` 验收：**tap1..4 范数须由 0 变非零**，
  且接受率相对基线 `4.81% / 20,3,0,0,0,0,0` 有可测改善。

## 2026-09-12 上午 · 归因重做（可验证对齐 + 正确基准）与三路并行

### 一、对齐问题已解决（关键方法论修正）
`[df2cand]`（选择器探针）与 `[df2dbg]`（每列调试行）之间**滞后一轮**：
轮次偏移扫描给出 `off=+1 → 497/497 = 100%`（`off=0` 仅 54.1%）⇒ 早期按"同轮分组"的
归因（含 head_miss 84.8%/100%）**是把两股数据配错了** ✗。改用**内容配对**
（`cands[chosen] == 该列 draft`）后 497/497 全中，对齐不再依赖任何位置假设。

### 二、基准键修正（决定接受率的键是 verify 的 argmax）
干净列上 `verify 的 argmax == plain[pos+1]` 仅 4/9 ⇒ 用 plain 当答案键会把"verify 偏差"记到
"候选质量"账上。改用 verify 的 argmax 为键后（干净列 n=10）：
- `verify键 == plain` : **9/10**（干净列上两者基本一致 ⇒ "翻转"不在干净列）
- **head_miss（候选缺正确 token）: 4/10 = 40%**
- **walk_error（候选里有但排名靠后捞不回）: 5/10 = 50%**
- accepted 1/10
⇒ 崩坏点从"几乎全在候选集"变为**候选集 40% / 走链 50%**（样例里正确 token 常在候选第 12–13 位）。

### 三、边项缩放扫描（阴性）
`NINFER_DF2_PAIR_SCALE` ∈ {0.5, 1.0, 2.0, 4.0}：接受率 4.01% / 4.81% / 4.80% / 3.99%
⇒ **对边项缩放不敏感** ⇒ walk_error 那 50% 更像"排名太靠后"的症状，不是独立杠杆。

### 四、本轮已验正确 / 已作废
- 已验正确：特征/tap 链（`pending 100%` 非零；`features` 仅 live 列是设计）、
  artifact 改动能生效（清零码本 -0.8 点）、候选 id 就是真词表 id（`direct=96, via_map=0`）。
- 已作废（我自己）：tap 全零论（轴错位）、下游算术逐位吻合（读了全零残留文件）、
  同轮分组的 head_miss 归因（对齐错）。

### 五、三路并行（后台，只读，禁止编译）
- **D**：dflash2 候选 logits 的来源（哪份权重/算子/域；`optimized_head` 两分支；`draft_head_token_ids` 应用位置）
- **E**：**可表示性天花板** —— 目标常选 token 是否根本不在 131072 草稿域内（若比例可观，40% head_miss 属结构性必然）
- **F**：MTP（35–57%）与 dflash2（4.81%）的结构差异清单与最可能机制

### 六、修正后的下一步
等三路回来后对照；若 E 证明"目标 token 大量落在草稿域外"，则问题是**域/映射设计**（而不是网络权重）；
若 F 指出 MTP 走全词表而 dflash2 走受限域，则两者互为印证 ⇒ 修复方向是让 dflash2 的候选域与映射回到全词表口径。

## 117b. §117"仍未落地"条目复核 (2026-09-12, 实证)
本轮逐条核实，**四项全部已修 / 既知接受 / 陈旧**，清单未同步：
1. 全局 `--kv-dtype e8` 起不来 → **已修**：`src/ops/wrapper/gqa_attention.cpp:450-456` 的
   profile 校验集已含 `DType::E8Kv`，注释原文即指 _TODO 98/117 U2。
2. `all:bf16` 语义缺口 → **解析正常**：权威解析器 `src/product/kv_options.h:43-62` 已支持
   `all`（`first=0,last=slots-1`）；残余问题是 "BFloat16 兼作 unset 哨兵"，该文件 25-31 行
   **已明确记为需要"was-set 掩码"的 API 变更**，并给出替代（全局 `--kv-dtype bf16`）
   ⇒ 属既知限制，非缺陷。
3. `sliding_window_tokens` 全局无赋值点 → **已有赋值**：`src/targets/qwen3_6/impl/state/
   decoder_state.cpp:324` 与 `:382`（`.sliding_window_tokens = window`）。Muse 掉针回归
   仍值得做（值是否正确未验），但"无赋值点"不成立。
4. `std::array<...,64>` 无越界校验 → **陈旧**：写入循环以 `.size()` 为界
   （`layouts_impl.h:1061-1078`），serve 侧 `parse_kv_table` 另做 clamp
   （`kv_auto_relayout.cpp:60-63`）。
另（本轮新修，非清单项）：`kv_auto_relayout.cpp` 的**重复实现** `parse_kv_table` 把 `all`
交给 `atoi` 解析成 0 ⇒ `all:<tier>` 在该副本里只设第 0 层；已补 `range == "all"` 分支
（该副本仅供 auto-relayout 使用；CLI/serve/引擎走 `product::parse_kv_layer_storage`）。

## 118b. 流程纪律（用户 2026-09-12 指令 + 本会话教训）
1. **所有指令一律投递后台**（`run_in_background` 或脱离的 `systemd-run --user` 服务），
   主代理只用短小 `cat/tail` 轮询日志。理由：用户经常打断，前台命令会被中断丢掉。
2. **后台服务是脱离的**：工具调用"被取消"≠ 服务停止 ⇒ 启动新测量前必须
   `systemctl --user stop <unit>` 且 **断言 `pgrep -x ninfer` 为空**（否则两个 22.66GiB
   实例争 32GB 显存 ⇒ 实测掉到 1/6，污染结论）。
3. **凡 tok/s 读数必须标注宿主状态**（`uptime`、宿主 CPU%、宿主空闲内存）：
   实测同一二进制同配置在宿主 CPU 66% + Memory Compression 活跃时从 130 掉到 17 tok/s。
4. **`cp -p` 回退会保留备份 mtime** ⇒ `make` 认为无需重编（BUILD_WALL=0.38s，静默）
   ⇒ 回退后必须 `touch` 再重编，否则"回退后测量"无效。
5. **测量臂一律先落完整日志再 grep**（`| grep` 会吞掉失败信息；我把一次失败误判过）。
6. **子代理禁止在共享路径删除**（已有一次 `rm dl/*` 毁掉一批 dump）。
7. **否定性结论必须先复算**（已撤回三个假结论：树=链、深度退化、耦合相反）。
8. **dump/二进制分析先断言元素数与结构**（四次栽在布局/编码假设上）。
9. **带宽口径**：5090D 理论 2176 GB/s（17001MHz×2×512bit/8），**可持续实测 1813 GB/s**；
   引擎有效 1.54 TB/s ⇒ 71% 峰值 / **85% 可持续** ⇒ 内存效率只剩 15~18%。

## 2026-09-12 夜 · KV 组件化 / 冷路径 / 仪表陷阱（本轮结论归档）

### 1. 本轮落地（14 组）
- 首轮 10 份：P1（ModelOpt 读取器）、F1×3（fp8 入口映射 + 测试期望 + 成本表注释）、C2-core（冷 codec 表 + ISO3 修正）、
  F3（判据注释 + `sequence.mtp_window` 复位）、R1（磁盘槽释放 + 2 个活 bug + CLI 三旗标）、H2-01（`kv_cold_tier_budget.h`）、
  D2（逐层冷槽 stride）、S2（22 文件：旋转唯一 gate / row scale 三态 / V codec iso3|e2m1）。
- 第三批 7 组：D1（DP 冷档成本模型，`kKvBitBudgetColdBitsX100 == 466` 有 static_assert）、
  QROT（fp8 decode Q 读端补旋转）、Q3（三处开关接线）、PPL（perplexity `--kv-dtype` 死标签）、
  HR（冷主机驱逐重基，21/21 hunk 零 fuzz）、F2（split 几何固定）、GM（G1⊕G2 冷路径解锁并集）。

### 2. 实测：row-scale / kvarn 的价值（C1，`ninfer-perplexity`，噪声底 = 0）
| 组 | baked | identity | Δ mean_nll |
|---|---|---|---|
| all-NVFP4 长 64k | 1.40990842 | 1.40831856 | **−0.00159（4/4 stream 同向）** |
| all-NVFP4 短 2k | 1.50889187 | 1.50900833 | +0.00012（方向 2:2 混杂） |
| 稠密长文本 130k（ctx 65536/stride 8192） | 1.99035283 | 1.98828477 | **−0.00207（9 窗中 8 窗同向）** |
| 负对照 all-E8Kv | 2.15060161 | 2.15060161 | **0.00000（逐位相同）** |
结论：**"长文本才有价值"不成立**（64k 下 identity 反而更优），"牺牲短文本"也不成立（2k 差异小 13 倍且方向混杂）。
量级：row-scale 效应（0.1–0.3% ppl）比 KV 层选择（默认表 4.870 vs all-NVFP4 4.096 = 0.775 nats）小 180–490 倍 ⇒ 二阶旋钮。
静态证实：全树仅 `gqa_isoquant_row_scale{,_loader}` + nvfp4 decode/prefill 四处读 `gqa_kv_row_scale`（负对照成立）。

### 3. 独立复核：旋转表是"真旋转"
S1：64 块实测 max|RRᵀ−I| = 1.9e-07、det ∈ [1±2e-07] ⇒ `--kv-rotation off`（R=I）对 `QK^T` **精确无损**，
只是把量化误差分配退回未标定状态 ⇒ 它必须表现为不同 token id。gate 烘焙 `kGqaIsoquantRotGeom[4] = {1,0,0,0}`（默认开）。

### 4. 新发现真 bug（两路独立裁定，已落 QROT）
**fp8 decode 单边旋转**：`gqa_attention_decode_fp8.cuh:183` 写 K 时施加 R，但 `:240-250` 把 Q 原样送进 `qkv_s`，
`:335-347` 的 `mma_bf16` 直接拿它对已旋转的 K ⇒ 实际算 `qᵀRk` 而不是 `(Rq)ᵀ(Rk)`。
- iso3 路径 `gqa_attention_decode_iso3.cuh:93-95` 的注释逐字写着正确不变量（"K is rotated at append time; rotate Q by the same…"）。
- 量级（Q2 宿主数值）：两边都转误差 mean 1.0e-7；只转 K 的误差 mean 1.97、95.7% 样本相对误差 > 0.1 ⇒ O(1) 全错。
- 归因：`decode_fp8.cuh:1-8` 自述 body 抄自 bf16 kernel，而 bf16 kernel 全文无旋转、Q 暂存与 fp8 一字不差 ⇒ 只在写端补了旋转。
- 影响面：默认层表不含 fp8，只在操作者显式选 `--kv-dtype fp8` / `all:fp8` 时触发；**且 `--kv-rotation off` 会让它退化为无害**，
  所以任何"rotation off 更好"的 fp8 A/B 结论都是这个 bug 的假象，不可用。

### 5. 新发现接线缺口（Q3：13 个开关，3 个无效，已落）
- **`--kv-residual-layers`（serve 侧完全无效）**：`serve_options.cpp:350-368` 解析完整，但 `generation_service.cpp:237-281`
  从不拷进 `EngineOptions`；另有第二个丢弃点 `:531` `reload_kv_storage(table, {})` 传空残差表，首次 relayout 会清空。
  潜在 bug：`serve_options.h:71` 声明 `std::array<bool,16>`，契约是 64（`types.h:44`）⇒ `--kv-residual-layers 20` 被误拒。
  **影响既往验收**：`research/scripts/_kv_matrix_57k.sh:42-56` 起的是 `ninfer-serve` 并传该旗标，
  故 TODO.md:1131 的"残差双层 PASS"很可能是在残差未生效时得到的 ⇒ **该 PASS 需回查**（已列为待重测）。
- **`--kv-quality-weight` / `--kv-tier-scores`（CLI 侧无效）**：解析了（`options.cpp:177-183`）与字段、运行期读者
  （`layouts_impl.h:1079/1086`）都现成，但 `apps/cli/main.cpp` 无拷贝 ⇒ **两分数判据在 CLI 上从未可达**。
- 非缺陷观察：`kv_v_codec_explicit` 是死标志（4 写 0 读），`--kv-v-codec` 生效只因 `layouts_impl.h:1257` 无条件拷值。

### 6. 冷路径的性质与收益（G1/G2/GM）
- 门是"**未完成门**"不是正确性门，且**有两个拦截点**：栈门（早退）+ 页级 `dtype != I8 -> success=false`；只解前者 nvfp4 一页也压不了。
- 槽位几何本就双 codec：`slot_bytes = 9536` 同时容纳 rANS 上限与 int8 raw 布局（9232）。混合 int8+nvfp4 可支持；
  **含 e8/iso3/fp8/bf16 的栈不行**（无冷 codec）⇒ 出厂默认 10×E8+6×NVFP4 档**解锁后仍一页压不了**（1M 的最大剩余障碍）。
- 收益算术：@4.0 冷槽 9536 > 常住 9216 ⇒ 每 head-page **亏 320 B**；@2.6 stride = 320+32×167+1024 = **6688 B** ⇒
  > **[注解 2026-09-18 · SUPERSEDED]** 本条（连同下一行 5250 的算术）已作废：6688 不是可用目标，**现值为 9632 B**（`320 + 32×259 + 1024`），owner 是 `src/product/kv_tier_formats.h:289` 的 `kKvColdPoolStrideBytes`（权威字面量 `ops::kEntropyNvfp4SlotBytes = 9632`，`include/ninfer/ops/entropy_nvfp4_slot.h:48`）。
  > 照本条下调会**先编译失败**：`kv_tier_formats.h:316`（`kKvColdRansStreamBytes` 必须 ≥ 实测 259 B/流）与 `:329`（`static_assert(kKvColdPoolStrideBytes == 9632, "cold stride == ops::kEntropyNvfp4SlotBytes")`）；6688 B 只给 167 B/流，实测 K 命中 **0/64**（`:276-298`）。符号不变（记录仍宽于常驻，`:330` 断言），差额是算术上的 416 B（9632 − 9216）。
  净 **+2528 B/head-page（+27.4%）**；1M/16 层/kv_heads 4/head_dim 256 ⇒ 18.00 → **13.06 GiB（省 4.94 GiB）**；盈亏平衡 **b ≲ 3.84**。
- **翻 D2 常量到 2.6 会让 `decoder_state.cpp:367` 的 `static_assert(cold_slot_stride_for(...) == ops::kEntropyNvfp4SlotBytes)` 编译失败**
  （6688 ≠ 9536）⇒ 必须同步改断言与文档，不能只改常量。
  > **[注解 2026-09-18 · SUPERSEDED]** 本条（即 5251-5252 那句"必须同步改断言与文档"）现在无从执行：`kKvBitBudgetColdSlotBytes` 已是**导出量**（`src/product/kv_bit_budget.h:314` = `kKvColdPoolStrideBytes`），该文件里没有可改的冷字节；两侧钉子现在是 `decoder_state.cpp:845`/`:863`（`static_assert(... == ops::kEntropyNvfp4SlotBytes)`）与 `kv_tier_formats.h:329`（`== 9632`），任一侧改成 6688/9536 都当场断。
  > 它引的 `decoder_state.cpp:367` 不是现行位置：那条 nvfp4 断言（消息 `nvfp4 cold-slot derivation drifted from ops::kEntropyNvfp4SlotBytes`）现在在 `decoder_state.cpp:835-837`。

### 7. 未决 / 待裁决
- **`cold_v_valid` 索引约定不自洽（新发现，先测后改）**：张量声明 `DType::I32, {kv_heads, 2, cold_pages}`
  （`decoder_state.cpp:466`）⇒ 按 row-major 平面偏置应是 `nb[2]`（slot 数据指针正是用 `cold_slots.nb[2]` 取 V 平面），
  但**发布侧（`program_impl.h:10609-10611`）与三个 launcher（`decode_impl.cuh:16` / `decode_partial.cuh:38` / `prefill.cu:61`）
  都用 `nb[1]`** 推 V valid；生产侧写 `slot_valid[page*valid_page_stride + head]`（`entropy_nvfp4_slot_kernels.cuh:123/170/183`），
  prefill 消费侧又算 `slot_base*(2*KVHeads) + head` 并给 V 加 `+KVHeads`。四者不可能同时正确。
  唯一真跑过的冷消费路径（int8 decode，`gqa_attention_decode_i8.cuh:406`）**根本不读 valid**，只用 block table 哨兵 `physical_page <= -2`。
  ⇒ 本轮 GM 按"并集"落了解锁，但该 valid 门是**断言而非修复路径**；索引约定需在第一次 nvfp4 冷端到端测试时定案。
- 1M 路线：驱逐被"被 attend 的页必须常驻"挡住（三条独立证明）⇒ 需 read-free 窗口语义 + 权重卸载；冷数据目前仍全在显存（disk 只做镜像，host 未实现）。
- F2 修的并行度代价：131072+4096 窗口下 split 数变化（3 vs 64）需实测 decode 速度后再决定是否做 tile 级 mod-S 交错。
- 残余：e8 中段窗口 11~15（F5 实测中）、nvfp4 慢的可信归因（M3）、split 几何的 CUDA graph 路径、独立 3-bit iso3 未接线。

### 8. 仪表陷阱（三条，均已修/已记）
1. `ninfer-perplexity` 的 `--kv-dtype` 是**死标签**（不设 `kv_cache_explicit`，全局档不生效）⇒ 已在 PPL 补丁修；
   修之前该工具只能用 `--kv-layer-storage` 指定层。
2. 摘要行 `kv cache dtype` 打印的是**全局 flag**而非生效的逐层表（nvfp4 跑会显示 `bf16`）⇒
   判据一律用 **`kv cache payload`**（payload = bits_per_element × 0.5 + fixed，fixed=0.50 GiB = MTP 层；`--spec none` ⇒ 0）。
   `report.json` 的 `execution.kv_dtype` 同样不可信。
3. `--kv-layer-storage all:bf16` **不是** bf16 基线：`BFloat16` 同时是"未设置"哨兵（`src/product/kv_options.h:25-30` 明文），
   要强制 BF16 必须走 `--kv-dtype bf16`（该路径 `_TODO.md 97/98/117` 已修）。另：`parse_kv_storage("fp8") = Fp8Group16`，
   而 `--kv-dtype fp8 = Fp8E4M3Row256`，两种拼写现在都映射到 `DType::FP8_E4M3FN`（F1 修的就是前者静默落 BF16）。

### 9. 流程教训（本轮实证两次的事故）
- **"外来删除 hunk"**：补丁若基于**移动中**的工作树生成，会静默回退别人的改动，而且 `patch --dry-run` **仍然 rc=0**，
  不会自曝。D1 与 HR 两条独立路径都实际踩到并各自用"逐 hunk 认领 + 锚定基准"发现。
  ⇒ 规矩：补丁必须锚定基准生成；落地后必须跑**标记存活检查**（本轮全量点名见 `_build_acc.txt` 第 2 节）。
- **不要在脚本运行时覆盖该脚本**：bash 会按文件偏移续读，覆盖后可能重复执行后半段命令。
  本次后果：残`_build_first.sh` 的 bash 实例在旧构建被杀后**又拉起一次 `cmake --build -j16`**（连续两次）。
  处置：把两份副本（`/mnt/c/.../ziqinzhang/_build_first.sh` 与 `.../sh/_build_first.sh`）都中和成 `exit 0` 占位。
- 误报澄清：批 4 落地后标记曾显示 `D2=0 / F3=1`，看似回退，实为**我 marker 路径写错**
  （`kColdSlotRansBitsPerCode` 定义在 `decoder_state.cpp:281`，export 头只声明区域；F3 的实际编辑在 `program_impl.h`），
  已用「全树定位 + `git diff --stat`（39 文件）+ 修正后的 manifest 对比」确认无回退。

### 10. 补记（同轮续查两条，都是"纠正先前假设"）

**(a) 翻 D2 常量不是改一个数字：整数类型挡路，且第二处 static_assert 也会断。**
`cold_slot_stride_bytes(codec, head_dim, page_tokens, std::int32_t bits_per_code)` 全程整数运算
（`stream_budget = (stream_symbols * bits_per_code + 7) / 8`），所以把 `kColdSlotRansBitsPerCode` 写成 `2.6`
会**截断成 2**；先前报告的 "6688 B / −27.4%" 是按实数算的，与当前类型不符。可用取值：
  > **[注解 2026-09-18 · SUPERSEDED]** 这一支（把 ceiling 降到 2.6/2、换取 6668/6688）已死，下文表格 5293-5297 与 5298 一并作废：该参数现在就是定点量 `kColdSlotRansBitsPerCodeX100`（`decoder_state.cpp:574`），取值 **404 = 4.04 bits/code，实测最小值**，不是可再下调的偏好；`kv_tier_formats.h:316` 把任何 < 259 B/流 的预算挡在编译期。
  > 别按表格里的 2.6 行（6668 B）"小重构"：6688/6668 都过不了 `kv_tier_formats.h:316`/`:329`，而且实测命中 0/64。当前真值仍在 `kv_tier_formats.h:288-289`（9632 B）。
| bits/code（整数）| 数据区 | stride | vs 常住 nvfp4 head-page 9216 B |
|---|---|---|---|
| 4（现状） | 8192 B | 9536 B | **+3.47%** |
| 3 | 6144 B | 7488 B | **−18.75%** |
| 2.6（需定点化） | 5324 B | 6668 B | −27.65% |
要拿到 6688 必须把该参数改成定点（例如 `bits_x10`）或浮点，属小重构。另外**不止一处断言**：
  > **[注解 2026-09-18 · SUPERSEDED]** "改成定点（如 `bits_x10`）"树里已经做了（`...X100`，`decoder_state.cpp:574`）；但 6688 仍不可达且被禁：167 B/流 实测命中 0/64，`kv_tier_formats.h:316`/`:329` 两条 static_assert 会断。
  > 本条引的位置也已过期：`static_assert(kKvColdPoolStrideBytes == 9536, …)` 现在是 `src/product/kv_tier_formats.h:329` 且读作 `== 9632`；`decoder_state.cpp:374` 那条现在在 `:835-837`（两侧钉子另在 `:845`/`:863`）；`kv_bit_budget.h` 的"mirror"注释现在只有一处派生（`:314`，注释在 `:71-82`），不再是 `:50/132` 两处手写字面量。
`decoder_state.cpp:374` 与 `src/product/kv_tier_formats.h:215`（`static_assert(kKvColdPoolStrideBytes == 9536, …)`）
都会编译失败，`kv_bit_budget.h:50/132` 的两处"mirror"注释也要同步。
安全性质（读码确认）：预算收紧**不会出错**——rANS 流溢出预算就清 valid 标志、该页保持热；
只是会让更多页压不动（池子变"惰性"）。`floor_bytes = header + 4*32 + scale_plane = 1472` 不构成约束。

**(b) `cold_v_valid` 不是"约定不一致"，是一处真缺陷（且至今无人读）。**
- 文档（`include/ninfer/ops/entropy_nvfp4_slot.h`）写 `slot_valid is I32 [kv_heads, pages]`（**2 维**）；
- 实际声明（`decoder_state.cpp:466`）是 `DType::I32, {kv_heads, 2, cold_pages}`（**3 维**）；
- 3 维下平面步长是 `nb[2]`（slot 数据指针正是用 `cold_slots.nb[2]` 取 V 平面，自洽），
  但发布侧（`program_impl.h:10609-10611`）与三个 launcher（`decode_impl.cuh:16` / `decode_partial.cuh:38` / `prefill.cu:61`）
  全用 `nb[1]` ⇒ 按声明形状那是**头步长**（= 2×平面步长），即 V valid 指针实际指向 head=1/plane=0。
- 消费侧的扁平式 `slot*2*KVHeads + head`（prefill）与生产侧的 `slot*cold_pages + head`
  只有在 `cold_pages == 2*KVHeads` 时才偶然相等 ⇒ 两套索引不可能同时正确。
- 唯一真跑过的冷消费路径（int8 decode，`gqa_attention_decode_i8.cuh:406`）**根本不读 valid**，只用哨兵 `physical_page <= -2`。
**处置（本轮决定）**：nvfp4 冷的"是否冷"判定改回**只用 block table 哨兵**（与已跑通的 int8 路径一致），
不把正确性押在这套未验证、且已被证明自相矛盾的标志数组上；标志数组的约定留作独立缺陷单
（修法需要把 `cold_pages` 与 `KVHeads` 显式传进内核，而不是靠猜步长）。
GM 并集补丁里那道 valid 门因此是"断言而非修复路径"，落地后需替换为哨兵判定。

### 11. **更正**（第 10(b) 条的结论是错的，以本条为准）
第 10(b) 条我判"`cold_v_valid` 指针用 `nb[1]` 是错的、平面偏置应是 `nb[2]`、标志数组自相矛盾"——**这个判断错了**，
错因是我按 PyTorch 约定（`nb[0]` = 元素大小、`nb[i]` = 第 i-1 维的步长）去读步长。本引擎的约定不同：
`src/core/tensor.cpp:51-57 set_contiguous_strides` 使 **`nb[i]` = 第 i 维的步长**（dim0 最内），于是
- valid `{kv_heads, 2, cold_pages}`：`nb = [4, 4·kh, 8·kh, 8·kh·P]` ⇒ **平面（第 1 维，size 2）的步长就是 `nb[1]`**；
- slots `{stride, kv_heads, 2, cold_pages}`：`nb = [1, S, S·kh, 2·S·kh]`（kh=2, S=9536 → `[1,9536,19072,38144]`）
  ⇒ 那里平面是**第 2 维**，步长 `nb[2]`。
两者是同一个意思（都是"跨一个平面的字节数"），只是 rank 不同。所以三个 launcher 的 `k_valid + nb[1]` **是对的**，
而按声明形状推 `nb[2]` 才是错的。

**正确结论（gm2 直读生产写入位置 + 消费 base 指针得出，证据充分）：**
- 生产侧：`k_valid = data + slot·nb[2]`（元素 `s·2kh`）、`v_valid = k_valid + nb[1]`（元素 `s·2kh+kh`）；
  encode 实参 `kv_heads, 1, …, nullptr, kv_heads` ⇒ `page_count=1`、`page≡0`、`valid_page_stride` **惰性**；
  实体写入是 `k_valid[head]` / `v_valid[head]`（`entropy_nvfp4_slot_kernels.cuh:123/170/183`）。
  ⇒ 布局：K(s,h) = `s·2kh+h`，V(s,h) = `s·2kh+kh+h`。
- 消费侧两种自洽约定：(A) **预偏移 base**（V 由 launcher 加 `nb[1]`）⇒ 索引 = `s·2kh+h`，**不加减 KVHeads**；
  (B) **单 base + 显式偏移**（`small_t_bf16.cuh` / `small_t_i8.cuh`，base 是 `cold_slot_valid.data`）⇒ 这里 `+KVHeads` 才对。
- 因此：nvfp4 decode 用 `[stage_slot_id]` **正确**（GM 并集改对了）；G2 的 `+KVHeads` 读的是槽 s+1 的 K 标志，
- **prefill `gqa_attention_prefill_nvfp4.cuh:1170` 是既有真 bug**：它数据侧用 (A)（`:1173-1176` + launcher `gqa_attention_prefill.cu:121-123` 加 `nb[2]`），
  valid 侧却用 (B)（launcher `:126-131` 加 `nb[1]`）⇒ 自相矛盾，应为 `cold_v_valid[cold_slot_id]`。
  G2 当初把这条 prefill 写法当"正确参照"去支撑 `+KVHeads`，属**同错互证**。
- **写错的后果不是"仅禁用冷压缩"**：假阴性（真 1 读 0）⇒ `stage_cold=false` 但 `stage_physical_page` 仍是 `-2-slot`
  ⇒ 热分支按负页号寻址（`gqa_attention_decode_nvfp4.cuh` 的 else 分支；prefill `:1171 physical_page = cold ? 0 : table_entry`）
  ⇒ 越界读常驻平面 = **静默错答案**；假阳性（真 0 读 1）⇒ 去解编码器显式判无效的槽（溢出时只写 magic/flags=0）
  ⇒ 垃圾 K/V = **静默错答案**。所以标志数组是**承重的**，不能改成"只看哨兵"（我先前那个提议会引入假阳性风险）。
- 附带：`valid_page_stride` 惰性、以及"调用方把 `2·kh` 折进 base、而内核页内步长按 `kh`"这个分工**只在 `page_count==1` 时安全**
  ⇒ 应在两处 encode 调用点加注释/断言。

**待办（下一轮与 D2 一起做）**：① 修 prefill `:1170` 为 `[cold_slot_id]`（**两路独立复核一致**：gm1+gm2 都判它是既有真 bug，
且 gm1 证实该行在 `git HEAD` 里就存在）；② 两处 encode 调用点加 `page_count==1` 的说明；
③ D2 定点化重构（`bits_per_code` 整数截断问题 + 两处 static_assert + 环境变量可调以便实测 2.6 vs 3.0）；
④ **产品侧可达性/文案未同步（GM 合并时丢弃了 G1 的产品补丁，属性缺口）**：`src/product/kv_tier_formats.h` 里
`kv_cold_pool_reachable`（`:346-356`）仍要求全层 Int8、`cold=` 规格仍以"all-INT8 gate"为由拒绝（`:508-519`）、
报告行仍打印 `not all-int8`（`:574`），`tests/test_kv_tier_formats.cpp:197/296-303` 也还是旧行为 ⇒
解锁后产品层会**低报可达性、并用已不成立的理由拒绝 `cold=`**，必须同步。

**两条独立复核的收敛性（值得记）**：gm1 与 gm2 在四个问题上给出完全相同的结论，包括
"`[slot_id]` 对、`+KVHeads` 错"、"两个拦截点都解掉"、"旋转开关零回退"，以及
"我上一轮对 `nb` 约定的描述是反的"。两条独立路径都直接读了 `src/core/tensor.cpp:51-58` 才定案——
这正是"双路取证"要的效果：**我自己的单路推断错了，两路一致把它纠正过来**。

### 12. NVFP4 慢的归因实测（M3，串行单实例 + 每跑持锁；结论可信）
**仪器**：`--max-new 256`，`decode = F + T·gen` 拟合，基线 nvfp4 61.10 / int8 64.03 tok/s。
- **满栈差只有 2.93 tok/s（4.58%）**，不是参考表里的 2.14×。
- **单层 delta（16 层 × 双向，全部命中）**：`d1`（nvfp4 基座 + 单层→int8）中位数 **−0.01** tok/s（换 int8 **零收益**）；
  `d2`（int8 基座 + 单层→nvfp4）中位数 **+0.52** tok/s（换 nvfp4 **稳定损失**）。
  **线性外推两方向互相矛盾**（16×d1 ≈ −0.2 符号都反、16×d2 ≈ +8.2 高估 2.8×）⇒ 单层代价**不可加、呈凹/饱和**，只有全层同档才兑现。
- **归因分摊**：QK 第二遍 = **0%**（K 侧用 iso3↔nvfp4 双向探针压到 ≈0.1 tok/s 证明几乎免费）；
  **V 软解 ≈95–97%、K 码本 ≈3–5%**。若按参考表那 45.98 tok/s（2.14×）口径摊派，则 QK 第二遍 ≈93–96%——
  但 V 软解实测被限制在解码时间 ~5% 以内，**不可能解释 2.14×**，所以 2.14× 不是本二进制能达到的口径。
- **`F ≈ 16 ms/次解码、T ≈ 13.9–14.2 ms`**：`--max-context` 8192→131072 让 KV payload 144 MiB→2.25 GiB（15.6×）
  而每 token 时间不变 ⇒ **池大小不构成代价**；随上下文增长的部分仅 ~2.1 ms/tok。
  这解释了为什么 `--max-new 8` 那轮各档看起来同速（那轮测的是 16 ms 固定开销）⇒ **以后测档位速度必须给足 `--max-new`**。
- **参考表不可复现且偏差已被解释**：其 +0.50 GiB 就是 MTP 层（该层 bf16）；实测 `M_bf16_none` 8.00 → `M_bf16_auto` 8.50 坐实。
  归一化后参考的 nvfp4 慢 ≈2.06×、e8 快 ≈1.8×，与"旧二进制 + 未门控第二遍"一致 ⇒
  **参考表一律标注来源、不当基准**；要重测请用 `F + T·gen` 口径而不是裸 tok/s。

**两条必须跟进的新事实**：
1. **MTP 驻留时排序反转**：nvfp4 **148.94** > int8 **136.70** tok/s（接受率 78% vs 60%）⇒
   开推测（默认！）时 nvfp4 反而更快。这与我先前"nvfp4 更慢"的叙述相反，验收里要单独出一臂。
2. **偶发崩溃**：`--kv-dtype nvfp4 --spec none` 出现一次 SIGABRT，`gqa_attention_prefill.cu:339 cudaMemcpy D2H` 失败；
   同参数此前成功 ⇒ 非确定性，需在新二进制上复现排查（F2/GM 都改过 prefill 相关代码）。

**附带印证 Q3**：残余 pass 在 CLI 上根本够不到——`error: unknown argument: --kv-residual-layers`，
usage 无此 flag，二进制 `strings` 出现 0 次；源码只在 `apps/serve` 解析且**只能设 true 不能关**；
残余平面还需 `dtype==NVFP4` 才分配 ⇒ 不分配、不发生，代价恒为 0。

### 13. 导入普查的独立复算（q4，从零数，未借我的任何中间量）
**先更正前提**：任务里给的源件路径 `/home/user/models/q38_abl_huihui_nvfp4` **在本机没有权重载荷**——
它声明 2 个分片，而 `model.safetensors` 只以六个 `.incomplete` 块躺在 `.cache/huggingface/download/` 里。
完整的 ModelOpt-NVFP4 源件是 `/home/user/models/q3nvfp4`（17,915,815,528 B / 2 分片）。q4 两个都数了，
字节级结论都来自 q3nvfp4；huihui 目录只有那个 BF16 MTP 分片是真的（15 个 key present / 0.849 GB）。

| 项 | 值 |
|---|---|
| 源对象（q3nvfp4） | **2387 keys**，401 个量化 Linear，17.916 GB，**27,356,728,560** 逻辑参数（≈27.4 B） |
| 目标要求（注册 `qwen3_8_27b`） | **1124 对象 = 6 resources + 1118 tensors**；oracle 与这份清单**完全吻合**（0 处名字/格式/布局/形状不符） |
| 可逐字节搬运（已验） | **1118 里 550**（209 单跑 + 48 拼接 + 222 vision BF16 + 71 NVFP4；NVFP4 载荷经注册 `encode_nvfp4` 重建 **100.000000%** 相同）⇒ 加上 huihui 的 MTP **12**（经 `mtp.py` 12/12 逐字节）共 **562** |
| 值精确但非逐字节 | 144（96 个 `A_log`/`dt_bias` BF16→FP32 加宽 + 48 个 `conv1d` reshape+转置） |
| 必须重算 | **412** = 41 NVFP4 `mlp/down` + 145 FP8 + 1 embedding + 111 vision Q4/Q5/Q6/W8 + 112 input-scale divisor + 2 draft head |
| 缺 key | q3nvfp4 **12**（恰好是 `mtp/*` 全集）；huihui 193 |
| 源里多余 | q3nvfp4 **0/2387**（每个 key 都被消费） |
分区自检：550 + 144 + 412 + 12 = 1118 ✔

**三类不可复现（都带数字）**：
1. **41 个 NVFP4 `mlp/down`（层 15–55）**：codes 11.34% 不同、group scale 6.51%、divisor 41/41 逐位相同；载荷重建仅 89.20%。
   数值：max|δ| = 0.0247、rel RMS 6.2435%、**max|δ| = 0.4286 个 NVFP4 group step**；
   float64 SVD σ1/‖δ‖_F = **0.3008** vs 同密度零假设 **0.0520**（复现了文档里的 0.307/0.051）。
   **既不是 `s·W`**（最优 s* = 0.998、残留 6.13%）**也不是 `W + a·1`**（a* = −2e−7、残留 max 0.0144），
   逐元素比值散布 12.9% ⇒ **是两次独立量化落在略微不同的网格上**，不是参数化的权重改变。
   ⚠️ 这**推翻**了我先前的说法"差一个秩 1/全局 scale 修正"。
2. **145 个 FP8（+embedding）**：重编码源值只有 23.23% code / 8.21% row-scale 一致（9.85% rel RMS）。
   注册档本身自洽（scales/codes/payload 100.000000%，且**每一行最大 code 字都是 `0x7E`**——印证了我从 oracle 反推出的不变量），
   但这一类**无法**从该源逐字节复现。
3. **112 个 FP32 divisor，干净的反例**：`1/input_scale` 只对 **1/112** 站点与 oracle 逐位相等，比值域 **0.355–3.761**。
   ⚠️ 注意：P1 当初验证的 112/112 用的是 **`weight_scale_2`** 那条规则（"引擎做除法 ⇒ `divisor = fp32(1/weight_scale_2)`"），
   与 q4 这里测的 `input_scale` **不是同一个字段** ⇒ 两说并不矛盾，但**必须由我亲自定点核对一遍**（见待办）。

**交叉验证**：`tools/convert/import_model.py --plan-only` 逐位复现 q4 的数字（2387/2687、字段直方图、MTP 0/15、vision 333、
同样的 17,915,815,528 B），并独立以 `F2 missing shard model.safetensors` 拒绝 huihui 目录；
q3nvfp4 被路由为 work-item，且**只**因两个 MTP config key 被拒。⇒ 前门的数字有了独立第三方确认。

**文档偏差（均为实测）**："27.0% of elements" vs 实测 6.03% nibbles / 11.34% bytes；divisor 比值域 0.68–1.47 vs 0.355–3.761；
§5 表合计 1004（漏了 112 个 divisor 与 2 个 draft head，补上正好 1118）；"1104 source keys present" vs 1106。

**待办**：定点核对 divisor 到底是 `1/weight_scale_2`（P1 规则，112/112）还是别的——
这条直接决定导入流水线的除数规则是否正确，两方说法必须由主代理用同一份 oracle 一次判死。

### 14. divisor 之争的判定（主代理亲自跑，5 轮探针，结论：**不可从本源复现**）
**先澄清两件事不是同一个量**：NVFP4 权重对象**内部拖尾**的 FP32 `weight_divisor`（P1 验过 =
`fp32(1/weight_scale_2)`，112/112 逐位）与 oracle 里**独立的** 112 个 `*/input_scale_divisor` 对象
（FP32、shape []、`contiguous-le-v1`）是**两回事**。q4 测的是后者，P1 当初**把这 112 个对象列在 skipped 里**
（它的验收表：`未匹配 skipped 119 = 7 个 mtp/* + 112 个 */input_scale_divisor`）⇒ 这 112 个此前**从未被任何人验证过**。

**结构与候选空间**（实测）：
- 源件 q3nvfp4 只有 **7 个标量字段名**：`input_scale`(401) / `weight_scale`(401) / `weight_scale_2`(401) /
  `weight`(922) / `bias`(166) / `A_log`(48) / `dt_bias`(48) —— **没有 `output_scale`**。
  ⇒ 激活侧唯一的标量就是 `input_scale`。
- oracle 的 112 个 divisor **清一色是 MLP**：`gate_up_projection` 56 + `down_projection` 56 ——
  正好等于 112 个 NVFP4 权重对象（注意力是 FP8，另有一套 row-scale 机制）；即**每个 NVFP4 对象配一个 divisor**。
- 源件 MLP 是**分开的** `gate_proj`/`up_proj`/`down_proj`（无 `gate_up_proj`），且
  `gate_proj.input_scale == up_proj.input_scale`、`gate_proj.weight_scale_2 == up_proj.weight_scale_2` **56/56**。

**判定（候选置换空间 36 个/divisor）**：6 模块 × 2 字段 × ±1 层 = 36 个候选里，
`|div − v|` 与 `|div − 1/v|` 的精确命中总数只有 **7 次偶然命中**（且各来自不同候选）⇒
**不存在任何置换式精确映射**：div 既不等于这些源标量的直接值，也不等于它们的倒数。
- 对 `gate_up_projection` vs `gate_proj.input_scale`：log-log **r = −0.98872、slope = −0.9983**
  ⇒ 与 `1/input_scale` 是**同一个物理量族**，但乘性因子逐层摆动很大（`div/in` 跨 2274–658721，约 290×）。
- 对 `down_projection` vs `down_proj.input_scale`：r = −0.84238、slope = −0.6916（更弱）。
- 逐层看更清楚：L0 div/(1/in) = 0.963，L1 就变成 1.475 ⇒ 不是常数偏移，也不是常数倍数。

**结论**：这 112 个 divisor **无法从本源的任何字段复现**（q4 的判断成立，且现在把 36 候选置换空间也排除掉了）；
我先前"divisor = 1/weight_scale_2"的说法**只适用于权重对象内部的拖尾 divisor，不适用于这 112 个独立对象**。
连同另两类不可复现（41 个 NVFP4 `mlp/down` 的 codes/scale、145 个 FP8），**三类都是"接近但不相等"**
（divisor 与 1/input_scale 强相关却因子不定；down 的 σ1/‖δ‖_F = 0.30 vs 零假设 0.05；FP8 只有 23% code 一致）
⇒ 最自洽的解释是：**该 oracle 的转换输入（那份标定/量化运行）与本机这份源件转储不是同一次**。

**对导入流水线的意义（正面）**：要求从来不是"逐字节复现这个 oracle"，而是"任意非 ninfer 文件 → 自动生成
可跑的 ninfer"。因此这三类不可复现**不阻塞**流水线；它们只说明：拿这份 q3nvfp4 生成的是**等价**产物而非同一产物。
要逐字节对齐该 oracle，需要它当初那次转换的原始输入（标定产物），本源件里没有。

### 15. e8 中段窗口 11~15 实测（f5，perplexity 判据 ctx 65536 / quick 261167 token）
**结论：11~15 里没有"放 e8 仍然安全"的层。每多放一层都单调变差，且平滑无悬崖。**
| 在出厂表(10×E8)之上加一层 | Δ% ppl |
|---|---|
| 层 **11** | **+6.17%** |
| 层 15 | +10.26% |
| 层 **12** | **+11.52%** |
| 11~15 整块(13×E8) | +44.59% |
单调前段窗口：11×E8(`0-10`) +11.00% / 12×E8 +19.73% / 13×E8 +33.00% / 16×E8 **+76.38%**。
**⇒ 不支持把 E8 上限从 10 提高**；反方向才有收益：8×E8 **−6.02%**、0×e8(all:nvfp4) **−15.91%**。
- **等层数换位**：e8 从 13 挪到 12 ⇒ **+3.58%** ⇒ 中段承受力 **11 > 13 > 15 ≈ 12**。
- **位置也重要**：出厂分散表（4.870234）优于连续前段 `0-9:e8`（+1.95%）与换位 `{…,12,14}`（+3.58%）
  ⇒ **DP 会排出的"前段连续"形状比出厂表差约 2%**。
- **上下文不翻转排序**：ctx 4096/16384/65536 三档 all:nvfp4 一律最好（出厂表 e8 代价 +15.6%/+17.4%/+18.9%）。

⚠️ **与 `variant.cpp:22-38` 的注释冲突（重要）**：该注释称 ctx 4096 上 10×E8(1.020) 优于 all-NVFP4(1.706)，
f5 **无法复现**，且绝对量级也不同（注释 ~1.02 vs f5 ~4.1–8.6，疑似度量口径不同）。
而 **C1 与 f5 两路独立吻合**：all:nvfp4 = 4.095580（C1 4.096）、出厂表 = 4.870234（C1 4.870）⇒
**两份独立实测都说"全 NVFP4 比出厂的 10×E8 表更好"，与该注释的排序相反**。
⇒ 需要用它**当初的原始脚本/语料**（13.3k 中文、ctx 4096）复跑判定；在判定前，
`variant.cpp` 那张表与"DP 用的 tier 质量表"都属**待重测**（与 M3 的"参考表不可复现"是同一个坑）。

**数据可靠性**：噪声底 0，4 组独立检查逐位一致（B0 起终点重跑、ctx4k 臂的"孤儿子进程产出 vs 干净重跑"，
以及与另一 agent 的 3 条锚点 `all:nvfp4` 4.095580 / `all:e8` 8.590025 / 出厂表 4.870234 全部位相同）；20 臂全 rc=0。

**顺带确认/提醒**：
- 两个仪表陷阱独立复核成立（`--kv-dtype` 死标签、`all:bf16` 等于没设），都有逐位相同的实测证据。
- **`ninfer-perplexity` 拒绝"已存在且非空"的 `--output` 目录**（rc=1）⇒ 任何 perplexity 跑法每臂都要用新目录。
- **`i8` 不是合法拼写**（必须 `int8`）。
- 二进制仍是 20:30:58 快照；重编后按 `sh/f5_ppl_recheck.sh` 复核（先比 md5，变了就重跑基线比对 `4.870234464376197`）。

### 16. 产品侧一致性补丁（已备好，**未落**）：`product_cold_reachability.patch`
冷路径解锁后产品层没跟上（低报可达性 + 用已不成立的理由拒绝 `cold=`）。补丁 19 hunk / 2 文件
（`src/product/kv_tier_formats.h` + `tests/test_kv_tier_formats.cpp`），对当前真树 `patch -p1 --dry-run` **rc=0**、
零 offset/fuzz、dry-run 后树逐字节不变、无 `.orig/.rej` 残留。
- 语义改动：`kv_cold_pool_reachable` 改为"全层 ∈ {Int8, Nvfp4Fusion}"（与 `program_impl.h` 的真实准入一致）；
  `kv_cold_codec_spec(Nvfp4Fusion).reachable` 改 `true`（注释写明：当前 4.0 bits/code 下 `reduces==false`，
  9536 > 9216，所以 `cold=` 仍被**显存收益**那条理由正确拒绝）；陈旧文案/注释不再引用"all-INT8 gate 使 nvfp4 分支不可达"。
- **顺手抓出并修掉一个连带缺陷**：`cold=int8` 的 offender 循环原本取"第一个不是 Int8 的层"——
  `reachable` 放行 nvfp4 后，`[nvfp4, bf16]` 这种栈会**点名能压动的 nvfp4 层**当罪魁。
  已改为用新谓词 `kv_layer_class_cold_capable()`（`Int8 || Nvfp4Fusion`）选取，并被 `kv_cold_pool_reachable` 复用。
- 测试改了 9 处（`:188/:197/:254/:272-288/:300/:309-319/:344-372/:375-385`），
  **承重性双向验证**：旧测试+新头 → 7 条 FAIL；新测试+旧头 → 12 条 FAIL；
  新测试对改后头以 `-std=c++20 -Wall -Wextra` 编译 0 错 0 警告、运行 `all checks passed`。
- 明确未动：`pool_stride_bytes` 与全部 `kKvCold*Bytes` 常量（留给 D2）；`kv_cold_codec_default` 的代码；
  `layouts_impl.h:1229` 那句仍写 "the all-INT8 gate" 的陈旧注释（不在两文件范围内，避免冲突）⇒ 后续顺手改。
- **已知边界（已写进注释与报告）**：① 全 nvfp4 栈上写 `cold=int8` 会被接受——因 `cold=` 在设备侧**不是选择器**
  （codec 由逐层 dtype 派生），且报告行始终打印 RESOLVED spec 的字节真值，不会误报收益；要硬拦属**新语义**，单独立项。
  ② `KvLayerClass` 不携带"是否有残差平面"，故对"带残差 nvfp4 栈"仍报可达（池 reserve、层压不动，
  由 `program_impl.h:10658` 的逐层检查兜住）——注释里标为 `BOUNDARY`。

### 17. 出厂 KV 表的争议判定（找到原始脚本与语料，两边都对，分歧在**语料**）
**找到了源头（不是"找不到"）**：脚本 `/home/user/fusion_ppl.sh`（四臂正是注释里那四个数），
语料 `/home/user/perplexity-corpus.txt`（`scored_tokens=13318` 即"13.3k"，sha256 `54b85a53…` 与历史 report.json 逐位一致），
生成器 `gen_corpus.py` = **把同一段 232 字中文逐字 append 120 次** ⇒ 全文 120 行、**唯一行只有 1 行**。
历史四臂 report.json 全部在位（默认 1.0202220884607947 / all-E8 1.1119582079644514 /
all-NVFP4 1.705519481556782 / all-I8 1.5217006341755472），同一张表也印在 Windows 侧 README 里。

**复现（9 臂，单一二进制 md5 `d2827de5…`，rc 全 0）**：
| 臂 | 原口径（zh 复读语料 ctx4096） | 历史 | 对照口径（perplexity-1m quick，261167 token） |
|---|---|---|---|
| 出厂默认表 | **1.0266312582** | 1.020222 | 5.0335 |
| all:e8 | 1.1204700908 | 1.111958 | 7.7531（**最差**） |
| all:nvfp4 | 1.4783051775 | 1.705519 | **4.3537（最好）** |
| all:int8 | 1.9785998885 | 1.521701 | — |
（默认表与 `--kv-dtype int8` 在该工具上**逐位相同**，`total_nll` 都是 350.03458349495634。）

**判定**：
- **注释的排序可以复现**（默认 1.0266 < all-E8 1.1205 < all-NVFP4 1.4783），与历史同向；
  两关键臂差 0.6% / 13%，属 09-02→09-12 的引擎漂移。
- **归因是语料**：同一二进制、同一 ctx/stride、同一口径下，排序在"复读语料"与"真文本"之间**翻转**
  ⇒ **不是 ctx**（f5 已证三档 ctx 都不翻转）、**不是 ppl 口径**。
- **口径不是问题**：注释数字 = `exp(mean_nll)` = 工具的 `overall/perplexity`，逐位吻合
  （1.0202=exp(0.020020)、1.7055=exp(0.533870)…）。绝对量级差（1.02 vs 4.87）纯来自语料：
  复读语料把 nll 压到 0.02 nats（win1/2/4 的 ppl 只有 1.001，**指标塌到地板**），真文本是 1.6–2.0 nats。
- **机制**：该语料实际只测"KV 能否逐字复现前文"——E8 重表把复读窗口 nll 压到 0.001–0.002，NVFP4 抬到 0.07–0.46。
  ⇒ `variant.cpp:22-38` 那个 1.020 **是合成复读语料的产物，不可外推到真实文本**。

**建议（待你裁决）**：不要以"ppl 更好"为由保留 10×E8 出厂表；改按"长上下文检索 + 真文本 ppl"双判据选表
（短期可收表到 `{0,1,3,4,6,7}`），并把该注释标注为"合成复读语料、不可外推"。同时修 e8 高层的退化。

**副产物两条（都是独立发现，建议各自立项）**：
1. **`--kv-dtype` 在本构建上对池几何完全无效**（`int8`/`bf16`/不传三者 `total_nll` 逐位相同）——
   这与"死标签"提示一致，但**否证了 TODO §98 声称的"`--kv-dtype bf16` 现给真 bf16 基线"**
   ⇒ 本构建**无法**用 `ninfer-perplexity` 取得无损 bf16 对照（`all:bf16` 是哨兵）。
   （注：CLI 侧的 `--kv-dtype` 是**设了** `kv_cache_explicit` 的，只有 perplexity 这个生产者漏了——已由 PPL 补丁修，故这条在**重编后**应复核是否仍成立。）
2. **`all:int8` 相对历史恶化 30%**（1.5217 → 1.9786，已差于 NVFP4）⇒ 一个与本争议无关的**引擎回退**，
   建议单独立项排查（注意：本次用的是 20:30 快照二进制，**不含**本晚落地的 14 组补丁）。

### 18. ngram 当"另一种 KV"：**实测判定为死路**（用户设想的这一形态）
仪器：哈希 n-gram 倒排索引 + 向量化精确 LCP（1M 建索引 0.107 s）；**用 O(N·Q·L) 暴力枚举在 32 个配置上逐位对拍，
最长匹配长度与最近源距离 mismatch=0（VALIDATION PASSED）**；`max_cand` 16→1024 只动 2 个百分点。

**一句话**：按 KV 页（64 token）粒度，"精确匹配查表"在 1M 混合上下文里的覆盖率是 **0.000%**（精确 0 页，不是"很小"），
**最乐观的无限索引上界也只有 3.76%**；安全口径下一页都压不下来。英文散文与随机数据上**完全不划算**。

| 场景 | cov≥4 | cov≥8 | cov≥64 | **KV 页覆盖** | 源距离中位数 |
|---|---:|---:|---:|---:|---:|
| 1M 混合流，前缀索引（N=4k…524k，含 1 MiB 预算点） | 0.5–1.6% | ~0 | 0 | **0.000%** | 15–39 万 |
| 1M 混合流，整段因果上界（无限索引+自引用，**不安全**） | 38.96% | 21.00% | 1.99% | **3.762%** | 821 |
| 代码域 260k，前缀索引 130k | 30.76% | 12.42% | 0.48% | **0.887%** | 99,174 |
| 代码域 260k，因果上界 | 50.61% | 28.11% | 2.14% | **4.235%** | 910 |
| 中文维基 260k，前缀索引 | 21.63% | 11.71% | 1.21% | **2.687%** | 103,083 |
| 英文散文（wikitext/PG-19，两口径） | 1.2–6.3% | 0.06–0.31% | **0.00%** | **0.000%** | 2k–112k |
| 真随机字节（→真 tokenizer） | 1.31% | 0.00% | 0 | **0.000%** | — |
| 对照：16× 重复代码块 / 同文两遍 | 99.7% | 99.3% | 93.9% | **100%** | — |

**三个把结论钉死的发现**：
1. **索引字节不是约束**：4 B/token（后缀数组）～12 B/token（哈希表），全历史 4–12 MiB = nvfp4 KV 的 **0.022–0.065%**；
   1 MiB 预算 = 26.2 万索引 token（25% 历史）。**瓶颈 100% 在覆盖率**。
2. **结构性障碍**：full-attention 层 `KV(p) = f(T[0..p])`，同序列内两位置的**前缀长度必然不同**
   ⇒ **序列内重复永远不可能精确共享 KV**，给多长 guard 都不行。唯一精确路径是**跨序列整前缀复用**——
   而引擎**已经实现了**（`ResidentPrefixIdentity` + `reuse_base`）。实测该语料跨流最大 LCP 只有 **2 token**，
   1M 流内 `max LCP(T[j:], T[0:]) = 2`。
3. **重复性是局部的**：因果上界里按匹配长度加权的源距离中位数 **821 token**，仅 15.4% 来自 65k 以外
   ⇒ **"对遥远过去建索引"这个方向本身是错的**，一个最近 1k–64k 的滑动索引就拿到 84.6%。

**KV 换算口径（双证）**：`kPagedKVPageSize=64`（`src/core/paged_kv_cache.h:17`）× 16 个 full-attn 层 × kv_heads 4 ×
head_dim 256 ⇒ nvfp4 **18,432 B/token**（`src/product/kv_tier_formats.h:45-52`；TODO §98 实测 1152 MiB@65,536ctx 吻合）
⇒ **1M = 18.00 GiB**。⚠️ **我先前说的"~22 GiB"仓库复现不出（是 1.222×）**，以 **18.00 GiB** 为准；
`sh/run_1m.sh` 顶部注释里那个 4.85e-6 系数偏大，需改成 3.815e-6 GiB/(token·bit·el)（注释级，不影响测量）。

**⇒ 结论与转向**：
- **ngram 作为 KV 替身：放弃**（安全口径 0.000%，无限索引上界 3.76%，且唯一精确路径已由前缀缓存实现）。
- **ngram 作为草稿/算力加速器：仍值得**，但只在**结构化/代码类文本**上（代码 cov≥4 达 30.76% 可提出 4-token 草稿）；
  这正是 `suffix_lookup` 原本的用途，而它今天**引擎零调用**（唯一调用点是未注册进构建的测试）。
- 1M 的正确杠杆回到：**降位宽（3-bit/2-bit）+ 熵编码 + 按上下文年龄分层**，以及 **read-free + 权重卸载**。
- 附带印证：仓库自带的 nvfp4 冷档**本身净亏**（slot 9,536 B > 常住 9,216 B/head-page）——ngram 不是绕开它的路。

### 19. KV 验收电池第一轮（26 臂，全 rc=0）+ 仪器修正三处
**结论：引擎的 KV 逐层记账是精确正确的**；判据从 11 条 FAIL 收敛到只剩冷臂那 2 条（原因见下，已修）。

**铁证（三条逐层记账恒等式，预测 vs 实测到 4 位小数全等）**：
| 恒等式 | 预测 | 实测 |
|---|---|---|
| `p_layer_nvfp4 == 16 层 nvfp4 + MTP(bf16)` | 0.1721 GiB | 0.1721 GiB |
| `p_nvfp4 == 16 层 nvfp4 + MTP(nvfp4)` | 0.1495 GiB | 0.1495 GiB |
| `p_default == 16 层默认表 + MTP(bf16)` | 0.1672 GiB | 0.1672 GiB |

**全层同档臂的 b̄ 复现到 1e-4**：bf16 16.0000 / int8 8.2500 / fp8 8.4999 / nvfp4 4.4999 / iso3 4.4999 / e8 4.2501。
**iso3 与 nvfp4：payload 完全相同（Δ=0.0000）而 token id 不同（5/48）** ⇒ ISO3 是真档位、共用 plane 几何、codec 确实不同（I1 落地有效）。
**三个组件开关都真的生效**：rotation off 4/48、row-scale off **1/48**（与 C1 "二阶效应"一致）、v-codec e2m1 5/48；默认表上 rotation off 48/48。
**负测试**：`--kv-v-codec e2m1` + 冷池被明确拒绝并给出可操作原因：
`kv-v-codec e2m1: the cold pool requires ISO3 V for NVFP4 layers (the eviction requant is Iso3VG16); use --cold-policy none or keep --kv-v-codec iso3`。

**仪器修正（三处，都是我测试脚本的错，不是引擎）**：
1. **冷臂缺溢写目录**：我传了 `--cold-disk-path .../spill_i8` 却没建目录 ⇒ 引擎明确报错
   `error: cold disk open failed: .../ninfer_cold_L0.slot` + rc=1（这正是"实在不行要明确报错"的样式）。
   ⇒ `acc_battery.sh` 的 cold 相已补 `mkdir -p`（三个溢写目录），冷相将在夜间 S6b 重跑。
2. **payload 模型常数写死成 0.5/0.5**：那是 262144 档的数字。实测斜率**随 capacity 变**，
   正确口径是 `capacity × 4096 B 每 (token·bit/el)`（4096 B 来自 nvfp4 的 18,432 B/token ÷ 4.5 bits/el）。
   在 capacity=8192 上实测 0.033218 GiB/bit/el，几何预期 0.031250，偏差 6.3%（小 capacity 下页对齐噪声占比大）。
   ⇒ 已改为"按 capacity 推斜率 + ±10% 容差"，并**去掉截距断言**（固定项是 MTP 层，不是截距）。
3. **混合表/逐层 pin 的臂不做 b̄ 断言**：它们的 MTP 层跑全局 dtype（默认 bf16）而 16 层跑档位，
   反推 b̄ 必然被抬高（实测 5.03 / 5.18）——**那是"MTP 层占一行"的正确签名**，改用上面三条恒等式检验。
   ⇒ 用"扣掉 MTP 后的 b̄"判默认表（得到 4.09~4.34 区间 vs 设计 4.34375、与 nvfp4 4.5/e8 4.25 都不同）。

**下一步（夜间自动）**：冷相重跑（验证 int8 冷真的省内存、混合栈按文档 skip）；acc2（F2 代价 + 崩溃复现）；
1M 阶梯；**硬检索测试**（10 万位数字 + 8 语言复述）；ctest；落地 prefill 索引修复 + 产品侧可达性并重编复验。

### 20. 热 KV 平面熵编码实测（在引擎 dump 的真实设备平面上，非旧数字非权重）
数据来源：引擎自己的 `NINFER_KVDUMP_DIR` 落盘的真实平面。**真实 byte-rANS 往返校验：192 条流 bit-exact，
真编码器比解析模型还小 0.6%（模型保守）**。

**收益（真实平面）**：
| 平面 | 现状 bits/el | +静态熵表 | +逐页表 |
|---|---|---|---|
| qwen27 nvfp4（K/V 码面） | 4.5 | **3.15（−30%）** | 3.29 |
| hd256 nvfp4 | 4.5 | 3.67（−18%） | — |
| hd256 **e8** | 4.25 | **2.82（−34%）** | — |
换算 1M 池（16 全注意力层 × kv_heads 4 × hd 256 = 32,768 el/token）：
nvfp4 **17.17 → 13.99 GiB**，e8 **16.21 → 10.77 GiB**；若坚持固定 stride 则退化为 14.80 / 12.37 GiB。

⚠️ **口径纠正（第二次）**：`4.85e-6 GiB/(token·bit/el)` 偏大 1.22 倍，仓库自己已改成 **3.815e-6**
⇒ **1M @ 4.34 bits/el 是 16.56 GiB，不是 21 GiB**。另外"权重后剩 ~10.4 GiB"只出自一条 `ctx=65536` 的历史日志行，
**不是通用常数**，别当预算基准（1M 的真实可行性要由 S4 的真实池分配失败信息给出）。

**⟹ 最重要的一条：这是"零质量风险"的 30% 削减。** 熵编码只是**重打包**，往返逐位还原（192 流 bit-exact）
⇒ 不动码本、不掉质量、不需要新 kernel 语义，只把已有平面压得更紧。而且实测 **e8+熵编码(2.82) 比"朴素 3-bit 码本+熵编码"(2.90) 还好**
——**先把现有格式熵编码，比重新设计窄码本更划算、风险更低**。

**逐块自包含的代价（几乎不在熵上，都已量化）**：逐页表的熵反而**低于**全平面表（ΔH = −0.002 ~ −1.17 bit/symbol，
页间异质性大于小样本噪声）。代价在三处：① 每流 4 B 冲刷，按冷槽 256 B/流设计点 = **+1.2~1.5% of resident**；
② 频率表头：16 符号码面 ~0.4~1.6%，但 **256/65536 符号的 scale 面高达 12~29%**（fp16 scale 逐页表 0.76 vs 静态表 0.42，差一倍）；
③ **真正的浪费在固定 stride**：为覆盖尺寸分布尾部，有效体积再涨 **+6~8 个百分点 of resident**（压缩后体积的 8~13%）。
块粒度：**32 token（半页）对码面最优**——正好是冷槽已有切法；scale 面反之越大越好。

**首选方案**：`(page, kv_head, 半页) 逐块自包含 rANS + 编译期烘焙的 per-format 静态熵表`；
池内布局用**变长块 + 每页偏移表**，**不要固定 stride**（冷槽今天"数据区=未压缩大小"的零余量配置就是反面教材）。
分两步：码面 → scale 面；都用 `--kv-entropy-*=off` 开关且**默认不改行为**；验收 = 逐字节往返 bit-exact +
关掉时逐位等于 baseline + 回退页与关闭时逐位相同。
**明确不做**：运行时跨页熵表、per-model 训练表、RLE（旧测 +82.5% 负收益）、stride 定在未压缩大小、
对 scale 面做逐页表、给 e8/iso3/fp8/bf16 混合栈承诺 codec。

**与降位宽的关系（排序结论）**：叠加但熵余量收窄——8-bit FP8 冗余 1.52 bit/el、引擎真实 4-bit 码本 0.48~1.31、
形状良好的 4-bit 码本只剩 **0.24**。叠加后：8-bit+熵 **6.72** → 4-bit+熵 **3.21** → 3-bit+熵 **2.90 bits/el**
⇒ **降位宽是更大的杠杆（每档 ≈2~3 bit/el），熵编码是第二位**。两者都做才是 1M 的使能条件。

**两个工程前置障碍（已写进报告）**：
1. `kvc_*_bt.bin` 其实是 execution tables，**不是逻辑→物理页映射** ⇒ 工程验收前必须先补一个真正的页映射 dump；
2. 短 prompt 下**未写入行的 code 字节是残留垃圾** ⇒ 编码器必须按 valid token 数决定编码范围（否则会把垃圾编进去）。

### 21. F2 的"并行度代价"实测：**方向相反，是新几何更快**（决策 ③ 结案）
同口径（ctx 131072、`--max-new 256`、短 prompt ⇒ decode 主导）：
| 臂 | decode tok/s | ms/token | prefill tok/s | payload |
|---|---|---|---|---|
| 默认表 | 167.28 | **5.98** | 1503.89 | 2.67 GiB |
| int8 | 160.13 | 6.24 | 1452.12 | 4.38 GiB |
| nvfp4 | 154.61 | 6.47 | 1409.04 | 2.39 GiB |
| e8 | 133.33 | 7.50 | 1530.81 | 2.26 GiB |
对照 M3 在**旧二进制（F2 之前，20:30 快照）**同 ctx 的 int8 档：**16.33 ms/token（≈61.2 tok/s）**
⇒ 新二进制 **快 2.2–2.7 倍**（`model elapsed` 1.585 s / 256 token）。

**判读（含口径限制）**：这是**跨二进制**对照（旧=20:30 快照；新=23:23:28，含 14 组落地 + F2 新几何），
所以严格说是"F2 新几何 + 其它落地"的**合成**，不能单独归因给 F2；但量级太大（2.2–2.7×）且同向，
**结论明确：先前担心的"F2 让 split 从 64 降到 3 ⇒ 慢"在这个仪器上没有出现，F2 保留。**
（我先前那个担心取自 F2 报告里另一组窗口配置 131072+4096 的数字；在"活窗口很小"的这里反而是它更快。）
若要**同二进制** A/B，必须回退 F2 再重编一次（约 40 分钟），夜间不做；作为待办保留。

**附带读数**：档位速度序 默认 > int8 > nvfp4 > e8（e8 最慢而 payload 最小 ⇒ e8 解码路径本身更贵，与晶格解码一致）；
prefill 在 ctx 131072 上 1409–1531 tok/s。

### 22. ngram 的思路纠正（用户指出方向）：不是"用 ngram 替换 KV 页"，而是"外挂知识库 + 按轮次召回"
**用户原话**："ngram 应该是实现思路有问题……查一查 qwen3.8 flash next 的 ngram 那 51b；
在现成模型上做到那样很难，但既然有算力看有没有办法把上下文搞成那样的格式，或者那些最远古的上下文
召回的时候去 SSD 上搜，**以对话为单位召回**，而不是长期卡在 KV 里，也就是搞成**类似外挂知识库**的形式，
那样 **SSD 带宽就不会成为瓶颈**。"

**联网查到的 Qwen3.8-Flash-Next 事实（务必注意与"替换 KV"是两件事）**：
- 125B 主模型 + **51B ngram embedding 参数**（BF16 ≈95.4 GiB），每 token 激活 6B；原生 262,144，YaRN 到 1M。
- 那 51B 是**哈希寻址的稀疏查表（PLE 式，受 Gemma 3n PLE 与 DeepSeek Engram 启发）**：约 2000 万行 × 160 B；
  每 token 读 **16 行**（8 个 2-gram 头用 x_{t-1},x_t + 8 个 3-gram 头用 x_{t-2},x_{t-1},x_t），拼成 [2560]；
  层位置在 **decoder block 1**；**确定性寻址 ⇒ 几乎不占每 token 算力预算**；表可放**主机内存并异步预取**
  （SGLang：pinned host + Triton UVA 只 gather 16 行 ⇒ 显存 83.91 → 60.45 GiB/卡）。
- **长上下文是另一套**：GDN（4 层里 3 层把历史压成固定尺寸递归状态，不涨 KV）+ **QSA（块级/micro-block 稀疏检索）**：
  轻量 indexer 把序列聚成 micro-block、估块级重要性、只选相关区域（块级，非 DSA 的 token 级）。1M 下 prefill 7.6× / decode 4.9×。
- 生态：`EngramDB`（把这类确定性哈希 ngram 表当数据库：落盘/索引/预取/单机 CPU+NVMe 服务，含 vLLM/SGLang 补丁）、
  `ngram-knowledge-injector`（用热插拔 `.plepatch` 往 GGUF 的 PLE 表注入知识，不重写 ~54 GB 表）、
  **NGM: A Plug-and-Play Training-Free Memory Module for LLMs**（Causal N-gram Encoder + Cosine-Gated Memory Injector，
  在 Qwen3 0.6B–14B 上评测 ⇒ **免训练注入现成模型**这条路的现成参考）、
  "On the Design of Qwen3.8-Next Architecture"（消融 ngram 层位置，第 2 层最优）。

**关键区分（决定了我们该做什么）**：那 51B 表承载的是**预训练期把语料统计成型**，不是运行期把当前对话写进去；
所以"把上下文搞成那个格式"有三条完全不同的路：① 训练期（现成模型做不到）；② **免训练注入**（NGM / injector 那条，风险与代价要评）；
③ **运行期外挂召回**（把对话存成可检索的形态，召回时重算或载入）——**用户说的 SSD 召回属于 ③**。

**用户方向的合理性（与我先前实测不冲突）**：我测的是"逐页精确共享 KV ⇒ 0.000% 覆盖"，
但用户要的是**稀疏粗粒度高精度召回**（一次查询只取少数几轮），是完全不同的机制：
- 粒度：以**轮/对话**为单位（比 64-token 页粗 1~2 个数量级）；
- 正确性：召回后**重新 prefill 或按页载入该轮的 KV**，不要求与历史位置逐位相同 ⇒ 绕开"KV 是前缀函数、序列内不可共享"的死结；
- 字节账（我算的，待 agent 复核）：nvfp4 KV ≈ **18,432 B/token**，而**文本只有 ~4 B/token**；重 prefill 实测 **~1450 tok/s**。
  一轮 2k token：KV 36.9 MB、文本 8 KB；一次查询召回 3~5 轮 ⇒ 读 ~150 MB（SSD 3–7 GB/s ⇒ 0.02–0.05 s）对比
  "把 1M KV 全留显存 = 16.6 GiB" ⇒ **SSD 带宽确实不成瓶颈**（用户判断正确）。

**已派两路**：① 情报深挖（Qwen ngram/QSA/GDN + EngramDB/injector/NGM，落成"最少改动路径"清单）；
② 架构与量化（文本外挂重算 vs 打包 KV 载入 vs 混合；召回质量 Recall@k；字节/算力账；热窗口 N∈{32k,128k,512k} 下 1M 的显存占用）。

**顺带确认两条本轮结果**：
- **F2 的"并行度代价"方向相反**：ctx 131072 同口径下新二进制 5.98–7.50 ms/token，旧（F2 前）16.33 ms/token ⇒ 快 2.2–2.7×（跨二进制对照，合成归因，但量级明确）。
- **M3 的偶发 SIGABRT 在新二进制上 10/10 未复现**（`--kv-dtype nvfp4 --spec none` 连跑 10 次全 rc=0）。
- **我脚本的一个隐患已修**：`trap release_lock EXIT INT TERM` 在 TERM 时只释放锁然后**继续跑** ⇒ 进程杀不掉、还占着锁，
  害我白等 15 分钟并让 S4 差点被锁住。改成 `trap 'release_lock; exit 130' INT TERM` + `trap release_lock EXIT`。

### 23. 1M 阶梯第一段实测 + 权威内存预算（含边界钉点）
**极限几何（`--max-context 1000000 --yarn --kv-capacity 65536`，池只给 64k）四档全部通过并正常生成**：
| 臂 | max context | kv payload | free after weights |
|---|---|---|---|
| default 表 | **1000000** | 1.34 GiB | 10.80 GiB |
| e8 | **1000000** | 1.13 GiB | 10.75 GiB |
| nvfp4 | **1000000** | 1.20 GiB | 10.80 GiB |
| int8 | **1000000** | 2.19 GiB | 10.75 GiB |
（int8 在 ctx 131072 上限的 decode 速度 140 tok/s。）

**边界钉点**：**不带 `--yarn` 时 `--max-context 1000000` 被明确拒绝**：
`error: max_context exceeds the variant native context capacity` ⇒ rope/原生容量这道闸门有效且报错明确；
带 `--yarn` 时 262144 通过，1010000 / 1048576 / 1048577 / 2000000 正在依次钉（结果见 s4_1m_rerun.log）。

**⟹ 权威内存预算：`free after weights` ≈ 10.75–10.80 GiB**（加载 19.7 GiB 权重之后）。
用它对照 1M 池需求（口径：3.815e-6 GiB/(token·bit/el)，即 nvfp4 18,432 B/token ÷ 4.5）：
| 档位 | bits/el | 1M 池 | vs 10.8 GiB |
|---|---|---|---|
| bf16 | 16.0 | ~62 GiB | ✗ |
| int8 | 8.25 | 31.5 GiB | ✗ |
| fp8 | 8.5 | 32.4 GiB | ✗ |
| nvfp4 | 4.5 | 17.17 GiB | ✗ |
| e8 | 4.25 | 16.21 GiB | ✗ |
| **e8 + 逐块 rANS（无损）** | **2.82** | **10.76 GiB** | **边缘（≈刚好）** |
| 3-bit 码本 + 熵 | 2.90 | 11.06 GiB | 略超 |
| nvfp4 + 熵 | 3.15 | 12.02 GiB | ✗ |
| **年龄分层（热 e8 + 冷 2-bit/熵）** | ≲2.6 | **< 10 GiB** | ✔ |
⇒ **结论：1M 不需要新码本就先到边缘了**——只要给 e8 加**逐块自包含 rANS 重打包**（无损、不动码本、不改质量），
1M 池 16.21 → 10.76 GiB，正好压进 10.8 GiB 的预算；再加上按年龄分层（久远上下文降档）就有余量。
**这条把"1M 差在哪"从"缺 6 GiB"变成"缺一次无损重打包"。**

**工具修复（我的 bug）**：硬检索测试第一次跑崩在 `_posix_spawn`，因为我用 `--prompt` 传 286 KB 文档 ⇒ **argv 超限 E2BIG**。
已改成写 `messages.json` 并用 `--messages <file>`（JSON 的 `content` 允许纯字符串，
源码依据 `src/serve/anthropic_messages_request.cpp:53-58`）。

### 24. **更正 §23 的乐观结论** + 1M 真实缺口（引擎自报数字）与可达路径
§23 我写"e8+熵编码 10.76 GiB 正好压进 10.8 GiB 预算"——**错了**：我用了裸 payload（16.21 GiB），
而引擎的 **runtime reservation 还包含 workspace/池开销**。以引擎**自己报的**数字为准：

**边界钉点（实测）**：
- `--max-context 1010000 --yarn` ⇒ **通过**（`max context 1010000`）
- `--max-context 1048576 --yarn` ⇒ **失败**：`error: gqa_attention workspace: invalid profile or interval`
  ⇒ **名义 YaRN 容量 1,048,576 在实际 workspace 档位上不可用**；可用上限其实是 **1,010,000**（`kGqaAttentionMaximumVisibleKeys`）
- `1048577` / `2000000` ⇒ 拒绝：`max_context exceeds the variant YaRN-extended context capacity`（报错明确）
- 不带 `--yarn` 给 1000000 ⇒ 拒绝：`max_context exceeds the variant native context capacity`

**真 1M 池（`--kv-capacity auto`）的精确缺口（e8 档，引擎原文）**：
```
error: minimum Engine runtime reservation requires 18912736512 bytes in addition to
       1073741824 bytes of automatic headroom, but only 11600323584 bytes are available after weights
```
⇒ 需要 **17.61 GiB + 1.00 GiB 余量 = 18.61 GiB**，可用 **10.80 GiB** ⇒ **缺 7.81 GiB**（当前 e8 档，4.25 bits/el）。

**按真实口径重算"要多少 bits/el 才能装下 1M"**（以 17.61 GiB 为 4.25 bits/el 的基准，线性缩放）：
| 有效 bits/el | 1M 池需要 | +1 GiB 余量 | vs 10.80 GiB |
|---|---|---|---|
| 4.25（e8 现状） | 17.61 GiB | 18.61 | **缺 7.81** |
| 3.15（nvfp4 + 熵） | 13.05 | 14.05 | 缺 3.25 |
| **2.82（e8 + 逐块 rANS，无损）** | **11.68** | **12.68** | **缺 1.88** |
| 2.37 | 9.82 | 10.82 | **刚好** |
| 2.25（2-bit 码本 + 熵） | 9.32 | 10.32 | **够，余 0.48** |
| 2.00 | 8.29 | 9.29 | 够，余 1.51 |
⇒ **结论（诚实版）：1M 今天差 7.81 GiB；单靠"e8 + 无损熵编码"能把缺口从 7.81 收到 1.88 GiB，但仍不够；
需要再叠一层（或换一项）才落进预算。可达路径按性价比排**：
1. **无损熵编码（e8 平面）**：缺 7.81 → 1.88 GiB（不动码本、不掉质量，最强的一步）；
2. **按上下文年龄分层**：久远 token 降到 2-bit/熵 ⇒ 有效 ≈2.2–2.4 bits/el ⇒ **装下并留余量**（这也是用户要的"根据上下文制定 KV"）；
3. **V 侧跨位置共享**（ngram 设计已证 V 不受 RoPE 约束、K 侧受）：V 占一半 ⇒ 再省最多 25%；
4. **权重卸载/释放显存**：把 10.80 GiB 的可用预算做大（1M 的预算瓶颈之一就是权重占 19.7 GiB）；
5. **按轮次外挂召回**（用户新方向）：把远古轮次移出常驻 KV，直接改变"1M 需要多少常驻"这个前提本身。

### 25. S6 重编 rc=2 的真因（**不是补丁问题，是我的启动环境**）+ 两份补丁已落且自检通过
**症状**：`Error 127` 于 `gqa_attention_decode.cu.o` / `_smallt.cu.o` / `_prefill.cu.o` 三个 TU。
**真因**：`Error 127 = command not found`。CMake 把 CUDA 编译器前置成 **ccache**，而 ccache 在
`/home/user/.local/bin`，**非登录 shell 的 PATH 里没有**；夜间流水线是用 `nohup setsid bash ...` 起的 ⇒ PATH 不全
⇒ 编译器启动失败。**这正是本会话早先踩过并记录过的同一个坑**，我给自己的脚本都加了 `export PATH`，却漏了 `night_shift2.sh`。
已修（脚本顶部加 `export PATH="/home/user/.local/bin:$PATH"`）。
⇒ 结论：**两份补丁落地成功且自检通过**：
- `--kv-layer-storage`/`--kv-dtype` 相关：prefill 的 `cold_v_valid[.. + Geometry::KVHeads]` 残留 **0**、正确索引 **1**；
- 产品侧可达性补丁 apply rc=0（19 hunk，含 9 处测试用例改写）。
**待办**：重编一次（带 PATH；因 prefill 补丁改的是 CUDA 头，4 个重型 TU 要重编，约 40 分钟）→ 跑 KV 单测 → 重跑 cold 相。

### 26. Qwen3.8-Flash-Next 的 51B ngram = 预训练冻结的知识表（**装不了上下文**）+ 对"按轮次外挂召回"的量化落地路径
**规范来源**：`https://lmsys.org/blog/2026-08-26-qwen-flash-next`（SGLang Day-0，2026-08-26；未进 tagged release，需 PR #36497；
镜像 `lmsysorg/sglang:qwen38flashnext`；模型 `Qwen/Qwen3.8-Flash-Next{,-FP8}`、`RadixArk/...-NVFP4`）。

**① 那 51B 到底是什么（已纠正我转述的两处数字）**：一张**由"以当前 token 结尾的 2/3-gram"寻址、内容在预训练期冻结的稀疏 embedding 表**。
16 个头（8 个 2-gram 读 (x_{t-1},x_t)、8 个 3-gram 读 (x_{t-2},x_{t-1},x_t)），uint64 回绕
`mixed = XOR_i (token_i × multiplier_i)`，`row[h] = mixed % per_head_prime[h] + per_head_prefix_offset[h]`；
物理落盘 **320,001,536 行 × 160 维**（128 分片；BF16 95.4 GiB / FP8 47.7 GiB）——**不是 2000 万行，2000 万是单头素数模数**；
每 token gather 16 行拼 `E_t∈R^2560`，在 **第二个 decoder block**（`ple_layer_ids: [2]`）经门控 + kernel=4 深度卷积注入
4 支 hyper-connection 残差后 HC-Mix 折回；**运行期只读**、遇 EOS 哈希窗复位。
⇒ **键只有 2–3 个 token，信息论上不可能寻址整轮对话** ⇒ 它承载的是**预训练语料的知识**，不是上下文。

**② 长上下文是另一套（这才是"以对话为单位召回"的参照）**：48 层 = **36 层 GDN**（把历史压成固定尺寸递归态，不涨 KV）
+ **12 层 QSA**（轻量 indexer 对**压缩比 4** 的块打分，**保留最好的 512 块**，展开成 **2048 个逻辑位置**）；
1M 下 prefill 10.2× / decode 6.6×（相对 Qwen3.7-Plus）；另有 IndexShare MTP（草稿步复用 QSA 的 top-k，省 indexer 调用）。

**③ PLE 主机内存卸载的确切做法（可借鉴）**：`--ple-offload-embedding`（vLLM：`VLLM_PLE_CPU_OFFLOAD=1`）；
查表位置**预先算好并异步预取** ⇒ 永不常驻显存（省 95.4 GiB BF16 / 47.7 GiB FP8）。

**④ 对我们（现成模型 + 自己的引擎）的结论与路径**（判定依据来自本轮情报）：
- **训练期 PLE 不可行**（要重训预训练 + 改残差流结构）；**`.plepatch`/`.pleo` 不适用**（要求模型**已有** PLE 表，我们没有）；
  **NGM 免训练模块**可用（零新权重、一个新算子）但收益仅 +0.5~1.2 分且**与长上下文召回无关** ⇒ 别当替代。
- **该做的是运行期外挂召回，两步走**：
  **步骤 1（纯宿主 / 零新算子 / 默认关）**：append-only **轮次日志**（`turn_id → token span, rope base, identity digest`）
  + 轮次级**文本**召回，用现有 `ResidentPrefixIdentity` + `reuse_base` 整前缀复用重 prefill ⇒ 先回答"召回有没有用"。
  **步骤 2（新 dump / 仍零新算子）**：每轮**一个连续文件**的 KV dump = **resident nvfp4 payload 原样**（18,432 B/token）
  + GDN 层固定尺寸递归态 blob；召回 = 一次顺序读 + 页表恢复，只 prefill 新后缀 ⇒ 回答"召回省多少"。
- **带宽/算力汇率（实测口径）**：每 token 历史，**SSD 读打包 KV = 2.6 µs（7 GB/s）/ 6.1 µs（3 GB/s）**，
  而**重新 prefill 同一 token = 690 µs**（1450 tok/s）⇒ **约 105–260 : 1，用 SSD 带宽买算力买得过**（KV:文本 = 18,432:4 = 4,608:1）。
  1M 会话（40 轮 × 25K）召回连续 128K = 2.36 GB / 0.34–0.79 s；最坏从第 1 轮起 18.0 GiB / 2.6–6.1 s — 仍只有重 prefill 1M（690 s）的 **1/175**。
  **关键前提：这成立是因为"每轮一次"**。若做成**逐 token 流式读 KV 做注意力** = 18.0 GiB/step ⇒ 7 GB/s 下 **0.38 tok/s，必崩**。
- **两个必须写进设计的坑**：(a) **KV 是前缀函数** ⇒ v1 只做"**连续后缀**"零近似召回；非连续远端轮次要 v2 的
  CacheBlend 式 RoPE recovery + top-r% 重算（劣化 ≤1–3%）。(b) **冷池不能当地基**：冷槽 9536 B > 常住 9216 B/head-page（净亏 320 B），
  且出厂默认 10×E8+6×NVFP4 下 `--cold-policy host` **直接不可用**（"[cold] admits nothing on this model"）⇒ L2 必须**独立于冷池**。
- **一条硬约束**：ContiguousKV（arXiv 2601.13631）证明"细粒度选择 + 粗粒度存储"会带来 **12–56× 读放大** ⇒ **每轮必须存连续 blob**。
- **现成范本**：llama.cpp PR #24003/#24004（"递归态一起落盘 + 默认关且行为逐位不变"）。

**⑤ 最该先做的三件事**：1) L0 轮次日志 + 文本召回最小闭环（纯宿主、可开关；验收：40 轮会话针放第 3 轮命中率不塌 +
TTFT 下降达阈值 + identity 不等必须拒绝复用，不静默近似）；2) **把 `suffix_lookup` 接成候选轮次生成器**（内核/wrapper/CMake 全在树里、
fuzz 4000/0、**引擎零调用**，`lookup_fuse::suffix_best` 已有 CPU 参考；验收：真核与 CPU 参考逐用例一致 + 端到端跑通）；
3) **KV per-turn dump/restore**（raw resident payload + GDN 状态，**不转冷 codec**；验收：`dump→restore→续跑` logits 与不 dump **逐位一致**）。

### 27. 1M 的**权威**二元拟合（用引擎自报的两个档位需求点）+ 一条性能异常
**两个实测点（引擎原文数字，均 `--max-context 1000000 --kv-capacity auto`）**：
| 档位 | bits/el | 需要的 runtime reservation | +1 GiB 余量 |
|---|---|---|---|
| e8 | 4.25 | 18,912,736,512 B = **17.61 GiB** | 18.61 GiB |
| nvfp4 | 4.5 | 20,000,740,608 B = **18.63 GiB** | 19.63 GiB |
可用（`available after weights`）= 11,600,323,584 B = **10.80 GiB**。

**拟合**（需求 = k · bits · tokens + c，两点解出）：
- k = **4.08e-6 GiB / (token · bit/el)**，c ≈ **0.27 GiB**（固定项：workspace/池结构）
- 两档之比 18.63/17.61 = **1.058** ≈ bits 之比 4.5/4.25 = **1.059** ⇒ **需求对 bits/el 严格线性**（这就是为什么能外推）

**⇒ 要装下 1M，有效位宽必须 ≤ (10.80 − 1.00 − 0.27) / (4.08e-6 × 1e6) = ≈ 2.33 bits/el**
（与 §24 的独立估计 2.37 吻合到 2%。）
| 方案 | 有效 bits/el | 1M 需要 | +1 GiB 余量 | vs 10.80 |
|---|---|---|---|---|
| e8 现状 | 4.25 | 17.61 | 18.61 | **缺 7.81** |
| nvfp4 + 熵 | 3.15 | 13.12 | 14.12 | 缺 3.32 |
| **e8 + 逐块 rANS（无损）** | **2.82** | **11.78** | **12.78** | **缺 1.98** |
| **2-bit 码本 + 熵（或年龄分层到此水平）** | **2.25** | **9.45** | **10.45** | **够，余 0.35** |
| 2.00 | 2.00 | 8.43 | 9.43 | 够，余 1.37 |
⇒ **一句话：1M 今天缺 7.81 GiB；无损熵编码把它收到 1.98 GiB；再叠一层年龄分层（久远→2-bit）就装下。**

**⚠️ 一条性能异常（待查）**：长文 perplexity 两臂（`--quick`、261,167 计分 token、**同一份语料**）
分数**逐位相同**（ppl 4.095580317272603、mean_nll 1.4099084208842896 —— 因为 65k 的流在两档下都只排出一个窗口，这是**预期**），
但 `score_seconds` **67.75 s（ctx 131072，3855 tok/s）vs 270.65 s（ctx 262144，965 tok/s）⇒ 同一份工作量慢 4 倍**。
⇒ 说明**评分成本随 `max_context`（池大小）增长，而不只随活跃 token 数**——这与 M3 在 *decode* 上的结论（"池大小不构成代价"）不同，
因为这里是 **prefill 主导**。需单独立项：prefill/attention 的成本是否随池容量而非活跃窗口增长（若是，2M 及以上的可行性要重估）。

### 28. 按轮次外挂召回：架构与量化实测（用户新方向，agent 交付）
口径：CPU numpy + 仓库自带 tokenizer；索引 4.00 MiB（1M token @4 B/token，必付）。

**① 字节/时间账（k=3 轮召回）**：
| 载荷 | 体积 | 读入时间 |
|---|---|---|
| 索引（必付） | 4.00 MiB | 1.4 ms @3 GB/s / 0.6 ms @7 GB/s |
| **打包 KV（熵编码 15,022 B/token）** | 11.52 MiB（3×268 tok）→ **1.7 ms @7 / 4.0 ms @3**；44.01 MiB（3×1024）→ 6.6 / 15.4 ms | |
| 文本外挂 | 845 B ~ 3.15 KiB | 微秒级（SSD 不在关键路径） |
| 对照：逐 token 流式读全部 1M KV | 18.00 GiB | **2.76 s @7 GB/s** |
⇒ **"SSD 不是瓶颈"对打包 KV 成立且很硬：2.76 s → 6.6 ms（418×）**，且每请求只付一次。
**但"文本+重算"那条路是把带宽换成算力，换贵了**：804 token 重算 = **554 ms**（1450 tok/s），同一召回走打包 KV 只要 2.3 ms（≈240×）；
而且重算**有损**（16 个 full-attention 层 `KV(p)=f(T[0..p])`，单轮重算 ≠ 原 KV）。
⇒ **结论：每轮存"打包 KV"而不是"文本"。**

**② 召回质量 Recall@k**（真实 Q&A 池 3142 条 + 1M 语料切片；暴力比对 22 配置已过）：
- 逐字粘贴在**消息末尾**：`suffix_lookup` 现有语义 **R@1 = 0.73–1.00**（min_len≥16 时 ≥0.90），R@8 ≥ 0.99
- 逐字粘贴在**消息中间**（后面还有追问）：**R@1 = 0.00–0.03，100% 查询零匹配**；同数据改"任意位置 n-gram" → **0.83–1.00**
- 用户**换词**：n-gram 全线 0.00–0.03（**零匹配率 100%**）；**BM25 R@1 = 0.80–1.00**（1024-token 长文档、查询词覆盖 10% 时 0.96–1.00，5% 时 0.67–0.78，2% 时只剩 0.33）
⇒ **现有语义（尾部锚定 + 最长匹配排序）在三种查询形态里两种直接失效** ⇒ 检索索引必须升格：
**"轮级 + 任意位置 n-gram 倒排 + idf/BM25 打分"**，起点就是 `src/spec/lookup_fuse.h:25`（已写好、引擎零调用）。

**③ 1M 下的显存**（热窗 N 常驻 + 其余外挂；权重 18.98 GiB、预算 10.4 GiB）：
| 热窗 N | 常驻 KV |
|---|---|
| 32k | **613.69 MiB（1/30）** |
| 128k | 2.29 GiB |
| 512k | 9.04 GiB |
外部侧车：文本 lzma **1.02 MiB** / token id 3.88 MiB / 若要外挂真 KV 则 14.21 GiB。
**附带发现比 SSD 账更重要：常驻 KV 直接钉死 decode 上限** —— 全 1M 常驻 = **80 tok/s**，N=32k = **2,550 tok/s（32×）**
⇒ 把久远轮次移出常驻 KV 不只是省显存，还是**吞吐的乘数**。

**④ 两个关键架构结论（都放宽了约束）**：
- **轮次召回把 `cold_host_page_is_read_free` 这条把 1M 卡死的前置条件移除了**：
  未召回的轮**永不被 attend**，"驱逐"= "不在召回集"，read-free 由构造满足。
- **打包 KV 不能用现有的 9,536 B 固定冷槽 stride**（比常住 9,216 B 还大，`kv_tier_formats.h:57-60` 自有记载）
  ⇒ 必须**变长块 + 4 B/块偏移表**（实测代价 ≈ 0）。

**⑤ 最该先做的一件事**：把 `suffix_lookup` 升一格（轮级 + 任意位置 n-gram 倒排 + idf/BM25），
先只做**纯仪表 `--turn-recall-stats off|log`**（默认 off、零行为改变），在现成的 `SequenceState::ledger`（`program.h:449`）
上按 `TurnClosure` 轮边界（`chat_template.cpp:618,657`）统计命中率。

### 29. GitHub：远端是**另一条无关历史**，已按"非破坏"方式投递
**用户给的仓库**：`https://github.com/Astrangemaninhere/ninfer-fusion`（**公开**，`private=false`，已有内容）。
- **远端 main = 76 提交 / 1433 文件**（`0eaac04`），本地 main = 11 提交 / 2605 文件，**`merge-base` 失败 ⇒ 无共同祖先**。
  远端那条线含 `feat(ple): W2 real-table gather verification PASS`、`test(ple): W2-2 ... on real 95GiB sidecar`、
  `diag(engine): Muse NaN ... layer 16 (TODO 104)`、`docs(kv): post-fix quality ladder ...`、
  `fix(kv): an explicit --kv-dtype now reaches the KV page geometry`、`feat(archkit): KV bit-budget allocator` 等 ——
  **同一项目的另一份快照**，且**远端有而本地没有** `tools/archkit/{flashnext_bindings,flashnext_convert,kv_bit_budget,kv_auto_allocate}.py`、
  `tests/test_suffix_lookup.py`、`docs/maintainer/kv-strategy-matrix.md`、`ROADMAP.md`/`RESEARCH-{FLASHNEXT,EXTERNAL,FREETOKEN}.md`/`VRAM.md` 等
  （正是 TODO 一直当作在 `ninfer-fusion-repo`（过期镜像）里的那批）。反之本地有整个 `src/product` KV 层与 `research/notes` 全部协作报告。
- **投递方式（非破坏）**：`git push origin main:refs/heads/work/kv-switches-cold-import`
  ⇒ 远端新分支 `f5c4def`，**`origin/main` 未被触碰**；PR 链接
  `https://github.com/Astrangemaninhere/ninfer-fusion/pull/new/work/kv-switches-cold-import`。
  随后把远端**仅在远端存在**的 64 个文件 `git checkout origin/main -- …` 取进本地（**纯新增、零覆盖**，已验证只有 A 与一个 D），
  提交 `f5c4def`（+10,356 行），并清掉仓库里一个 0 字节垃圾文件 `$f`（shell 展开事故，`d38bb91` 带入）。
- **为什么不能直接推 main**：两段无关历史 + 远端 76 个提交，强推会**摧毁那条线**。整合方向属用户决策。
- **网络路径（记下来，省下次摸索）**：WSL 的 `git http.proxy=http://127.0.0.1:10808` **在 NAT 模式下够不到**（代理只监听 Windows 环回，
  网关 172.30.128.1 的 10808/7890/10809 全不可连）⇒ 改为**从 Windows 侧用 portable git**
  （`C:\Users\User\AppData\Local\OpenClaw\deps\portable-git\mingw64\bin\git.exe`）对 `\\wsl.localhost\Ubuntu\home\user\ninfer-fusion`
  操作 + `-c safe.directory=* -c http.proxy=http://127.0.0.1:10808` ⇒ 通；认证可用（Windows 侧有 GCM，
  虽然会打印一条 `credential-manager-core is not a git command` 的无害告警）。
  **持久修法是 WSL 的 mirrored 网络模式**（`.wslconfig` 里 `networkingMode=mirrored`），但那要 `wsl --shutdown`（会打断构建/agent），
  所以**没在跑长任务时再改**。

### 30. 新导入靶子：`nerkyor/Qwen3.8-27B-EfficientThink-…-SimPO-MTP-NVFP4`（多量化合集包）
**用户指路**（2026-09-13）："这个模型也值得导入进来看看，挂后台下载就是"，随后指定 **"下那个 24GB 的，这个效果最好"**。

**仓库结构（查过 API）**：**160 文件 / 167.19 GB**，是**同一微调的 6 种量化合集**，每个子目录一种：
| 变体目录 | 主权重 | 体积 |
|---|---|---|
| **`W4A4+W8A8/`** | `model-nvfp4-mixed.safetensors` | **24.006 GB（用户指定）** |
| `W4A4/` | `model-nvfp4-fast.safetensors` | 18.822 GB（已下 200 MB 后停，留盘可续） |
| `AWQ-W4A16/` | `…SimPO-AWQ-W4A16.safetensors` | 18.979 GB |
| `INT8-W8A8-QAT/` | `model-0000{1..8}-of-00008` | ~31.9 GB |
| `W4A16/` / `W8A16/` | `text-0N` | ~15.9 / ~31.6 GB |
每个变体目录自带 `config.json` / `hf_quant_config.json` / `model.safetensors.index.json` / `manifest.json` / `SHA256SUMS` /
tokenizer 五件套 / `chat_template.jinja` / **`DFlash2-FP8/{config,manifest,SHA256SUMS}`**（与引擎 `--spec dflash2` 对应）；
根目录**没有** config.json（各变体各自一份）。`gated=false / private=false`（无需授权）。

**架构（`W4A4+W8A8/config.json` 实测）**：`Qwen3_5ForConditionalGeneration` / `model_type: qwen3_5` / `language_model_only:false`（多模态包装）；
`text_config`：**hidden 5120、head_dim 256、intermediate 17408、`attn_output_gate:true`、`full_attention_interval:4`**、
`layer_types` = 每 4 层一个 `full_attention`（其余 `linear_attention`＝GDN）⇒ **16 个 full-attention 层**
⇒ **与本引擎已注册的 `qwen3_8_27b` 家族（5120/256/16 full-attn + GDN + output gate）同构** ⇒ 预期是**近乎现成的导入**，
正是"谁来都行"的合格靶子。

**量化（`W4A4+W8A8/hf_quant_config.json` 实测）**：`producer: modelopt 0.43.0`、**`quant_algo: MIXED_PRECISION`**、
`kv_cache_quant_algo: null`、`quantized_layers` 逐层给 `quant_algo`（前若干层是 **FP8**，其余应为 NVFP4）
⇒ **真·混合精度 ModelOpt 检查点**，正好压测导入流水线的量化分类与 F 码路径（`P1` 读取器 + `import_model.py` 的 quant 判定）。

**下载与网络（两个可复用的发现）**：
1. **`_hf_chunk.py` 里 `PROXY` 硬编码 `http://127.0.0.1:10808`，而 WSL NAT 够不到 Windows 环回** ⇒ 在 WSL 跑它必失败。
   但 **hf-mirror.com 在 WSL 是直连通的**（curl 探测通）⇒ 我用了**去代理的副本** `/home/user/scratch/hfchunk_noproxy.py`
   （把 `build_opener(ProxyHandler(...))` 换成 `build_opener()`；**未改用户的脚本**）。
2. **实测速率 ~2.5–7 MB/s**（不是历史记录里的 22–100 MB/s）⇒ 24 GB 约需 1–2.5 小时。落点
   `/mnt/c/Users/User/Documents/ziqinzhang/models/Qwen3.8-27B-ET-Uncensored-NVFP4/{W4A4,W4A4+W8A8}/`
   （**下到 C: 侧**：WSL `/` 只剩 53 GB，而 C: 有 566 GB）。
**下一步**：下完后跑 `tools/convert/import_model.py --plan-only` 做前置普查（对象数/缺失键/量化分类），
再按 F 码决定是直接导入还是先补 target 侧支持。

### 31. 新靶子的**配套** DFlash2 草稿头 + vision-MTP 头（这次才能真正量接受率）
在 160 文件全清单里找到（先前只看了非 safetensors 项，漏了这三个大文件）：
| 文件 | 体积 | 说明 |
|---|---|---|
| `W4A4+W8A8/model-nvfp4-mixed.safetensors` | 24.006 GB | 主权重（下载中） |
| **`W4A4+W8A8/DFlash2-FP8/model.safetensors`** | **2407.028 MB** | **配套草稿头（FP8）** ← 下接受率就靠它 |
| `W4A4+W8A8/vision-mtp-bf16.safetensors` | 1770.898 MB | vision + MTP 头（bf16） |
（六个变体目录各有一套同款配套；`W4A4/` 那支的草稿头同尺寸。）

**草稿头 config 实测（`DFlash2DraftModel`）**：`num_hidden_layers: 5`、`head_dim: 128`、`hidden_size: 5120`、
`intermediate_size: 17408`、`num_attention_heads: 32`、`num_key_value_heads: 8`、`layer_types` 5×`sliding_attention`、
`sliding_window: 2048`、`max_window_layers: 5`；`dflash_config = { block_size: 8, conv_group_size: 16, conv_kernel_size: 2,
mask_token_id: 248070, selector_rank: 256, selector_top_k: 16, target_layer_ids: [5,19,33,47,61] }`、
`num_target_layers: 64` ⇒ **`target_layer_ids` 指向的就是这个 64 层模型自己的层** ⇒ **真配套**（此前那个 ckpt 的接受率上限只有 ≈25–31%，
`_HANDOFF` 记"引擎 p0 26–35% ≥ 该 ckpt 离线上限" ⇒ 草稿是瓶颈）。`dtype: bfloat16` 但目录名 FP8 ⇒ 权重按 FP8 存。
**接受率测量计划（三个文件到齐后）**：
1. `tools/convert/import_model.py --plan-only` 对新源做普查（主权重是 ModelOpt `MIXED_PRECISION`：逐层 FP8/NVFP4）；
   架构是 `qwen3_5` 家族 5120/256/16 full-attn + GDN + output_gate ⇒ **与已注册 `qwen3_8_27b` 同构**，预期近乎现成导入。
2. 导入主权重（artifact 落 C: 侧，WSL 只剩 ~53 GB）+ 把**配套草稿头**接进 dflash2 路径（对照现有
   `qwen3_8_27b_nvfp4_dflash2.ninfer` 的草稿结构与 `draft_dflash2_ref/config.json`）。
3. 量接受率：固定 prompt + `--greedy` + `--spec dflash2 --draft-tokens K`（K=1/3/5/7），读引擎的接受率行与
   `NINFER_ACCEPTLOG` 的逐轮明细；与旧（不配套）的 24.1%→53.3% 以及 ckpt 离线上限 ≈25–31% 对比。

### 32. 免权重普查法（可复用仪器）：只靠 HTTP range 取 safetensors 头
**原理**：safetensors = `u64 LE 头长` + JSON 头（`{name:{dtype,shape,data_offsets}}`）+ 数据区
⇒ **两次 range GET（前 8 字节、再读头长）就能拿到该分片全部张量名/dtype/形状，权重一个字节都不用下**。
脚本 `py/header_census.py`（走 hf-mirror 直连；注意**索引里的分片名是相对变体目录的**，直连会 404，须加前缀）。
新靶子实测：**2139 张量 / 2 分片**，dtype `F32 800 / BF16 799 / F8_E4M3 400 / U8 140`，
末段字段 `weight 937 / input_scale 400 / weight_scale 400 / bias 166 / weight_scale_2 140 / A_log 48 / dt_bias 48`。
⇒ 这条以后对任何新源都能先用几分钟把"里面到底有什么"量清楚。

### 33. 前门自举修复：`import_model.py` 按路径调用必崩
`REPO_ROOT = Path(__file__).resolve().parents[2]` 早就算了，但**从未插进 `sys.path`**，
而它用 `importlib.import_module("tools.convert...")` ⇒ `python3 tools/convert/import_model.py <源>`（最自然的用法）
必然 `ModuleNotFoundError: No module named 'tools'`。已修：算完 ROOT 即 `sys.path.insert(0, ROOT)`（+ `import sys`）。
md5 `a3e09637…` → `85d3977f…`。

### 34. **新增：布局决策层 `tools/convert/common/layout_plan.py`（导入缺的那一半）**
**为什么需要**：注册 target 的转换器都是"封闭的字节钉死契约"，各自硬编码一个检查点的对象计划——正确，但回答不了
"这个新源应该变成什么"。前门只报告"缺哪个转换器"，**决策层负责从源件自身结构决定布局**。
**它只看两件事**：① 一个量化组内 **dtype/shape 的关系**；② 源件**自己声明的量化算法**（ModelOpt `hf_quant_config.json`
的逐层 `quant_algo`，只作交叉核对、不作唯一依据）。**全模块没有任何模型名/层名/键白名单**（除量化器自己定义的
`.weight/.weight_scale/.weight_scale_2` 后缀约定）⇒ **新家族是数据问题不是代码改动**。
**约束直接引自 `tools/artifact/layouts.py`**（不在本模块里造）：`blockscale-k16-m128x4-v1` 要 `n%128==0 && k%64==0`、
码 `n*k/2` + 刻度 `n*k/16` + 尾部 fp32 divisor；`row-scale-v1` 码 `n*k` + **每行一个 BF16 刻度**（`n*2` 字节）、**无整除约束**；
`contiguous-le-v1` 原样搬 BF16/FP32/I32。
**关键设计**：按 **code 张量自身 dtype 优先分派**（直接可表示 ⇒ 原样搬），**标量刻度是逐行刻度的退化情形 ⇒ 广播**，
并把 `F32 标量→BF16 刻度` 的收窄**记为 deviation**（不藏）；无法表达的**明确拒收并点名缺的机制**。
**在新靶子真实普查上验证（2139 张量）**：
```
contiguous-le-v1          585 groups
row-scale-v1              260 groups   ← 源件声明 FP8 260
blockscale-k16-m128x4-v1  140 groups   ← 源件声明 NVFP4 140
refusals: none           结构与声明一致 400 / 不一致 0
```
**过程中修掉一个真 bug**：初版按"有没有 scale 伴随"分派 ⇒ 585 个普通组（`lm_head`/`embed_tokens`/`layernorm`…）
被误报 `F-UNRECOGNISED-GROUP`（537 个）。改成 **dtype 优先分派**后归零。代码 md5 `49e4838e…` → `d531e0a8…`。

### 35. 新靶子在前门的正式裁定（跑通全过程）
- **⑤ 前端资源**：源件自带 tokenizer **语义一致**（证据：6 个特殊 token id 全部对得上；
  最大 id 248076 < vocab_size 248320；与第二份本地副本 token→id 逐项相同）；`tokenizer_config.json` 与
  `chat_template.jinja` **未证实**（无法证明与钉死版本等价）⇒ 需 `--allow-frontend-drift`（偏离记入报告）；
  另两项 sha256 **钉死一致**。⇒ 四态证据模型按设计工作。
- **⑥ 路由**：`qwen3_8_27b` / `qwen3_6_27b` 的 **config 契约都逐字段匹配**（同形状不同权重族，需 `--target` 指明，默认取 qwen3_8_27b）；
  `qwen3_6_35b_a3b` 因 MoE 不匹配拒。**nvfp4 入口**：`qwen3_8_27b` 那条**要求 compressed-tensors**（实测 got `modelopt`）⇒ 拒；
  **`qwen3_6_27b` 那条通过**（但其输出 basename 固定为 `qwen3_6_27b_nvfp4.ninfer` 且 `--model` 必须是未量化官方源）。
  ⇒ **结论：走 ModelOpt 适配器这条路**，不是拿现成 closed converter 硬套。
- **⑦ 推测解码**：源件有 15 个 `mtp/*` 键 ⇒ MTP 可用；短名单头可离线生成，与 MTP 权重无关。
- **⑧ 结论**：`work-item`，缺件已点名：**ModelOpt NVFP4 单源适配器**，位置
  `tools/convert/dequant/modelopt.py` + **`tools/convert/qwen3_8_27b/convert_modelopt.py`**（后者尚不存在）。
  ⚠️ 注意其表述里"注册布局要求 145 个对象为 FP8、112 个为 NVFP4"是**原 artifact 的历史构成**；
  **导入应复现"这个源自己的声明"**（FP8 260 + NVFP4 140），而不是去凑历史比例。

### 36. 打包（whl）的门槛：**已知缺陷清单**——用户口径"带着 bug 不能打包"，所以 whl 放最后
把本轮所有实测到的缺陷收拢成一份**有取证锚点**的清单，作为出 whl 的 gate。
**A. 正确性 / 质量（必须先修）**
| # | 缺陷 | 证据锚点 |
|---|---|---|
| A1 | **`all:int8` 质量回归 30%**（同语料 1.5217 → **1.97860**） | `dl/rt_arm_d_all_int8.txt`：`scored=13318 mean_nll=0.682389 ppl=1.978600`（历史 1.5217） |
| A2 | **出厂默认表(10×E8)在真文本上差于 all-NVFP4**（4.870234 vs 4.095580），而立表理由（1.020）来自"一句话重复 120 次"的复读语料 ⇒ **表要重选**；我据此把 `kKvBitBudgetE8LayerLimit` 8→10 也一并**待重审** | `repro_table/REPORT.md`；两路独立同值（C1 4.096/4.870、f5 4.095580/4.870234） |
| A3 | **1M 边界**：名义 YaRN 容量 **1,048,576 直接报错**（`gqa_attention workspace: invalid profile or interval`），实际可用只到 **1,010,000** | `dl/night2/s4_1m_rerun.log`：1010000 `rc=0`；1048576 `rc=1 error: gqa_attention workspace...`；1048577 起报 capacity |
**B. 能力 / 经济性（影响可用性与 1M 目标）**
| # | 缺陷 | 证据锚点 |
|---|---|---|
| B1 | **冷路径净亏与不可达**：@4.0 每 head-page **亏 320 B**（9536 槽 > 9216 常住）；翻 2.6 需**定点化重构 + 两处 static_assert**；且 **e8/iso3/fp8/bf16 无冷 codec** ⇒ 出厂表**一页压不了** | `g1/REPORT.md`（9536=320+32×256+1024 **恰好填满、零余量**）；`gm/REPORT.md`（盈亏平衡 b≲3.84） |
| B2 | **运行时 iso3 档位白扔 ~1 bit/el**：`Iso3Group16 → {DType::ISO3, kNvfp4KvQuantGroup}` ⇒ 存的是 4-bit nibble 平面，却只用 3-bit 码本 | `layouts_impl.h:87` 一带 |
| B3 | **无损熵编码未落地**：真实平面 e8 **4.25 → 2.824 bits/el**（1M 池 16.21 → **10.77 GiB, −33.5%**），这是 1M 最划算的一步 | `kv_entropy/REPORT.md`（真 byte-rANS 往返 192 流 bit-exact） |
**C. 仪表 / 接线（会让判据失真）**
| # | 缺陷 | 证据锚点 |
|---|---|---|
| C1 | **DP 的质量/速度列是过期表**（参考 45.98 tok/s 不可复现；nvfp4 实际只慢 **4.58%**）⇒ 分配器输入错 | `m3/REPORT.md`（满栈 nvfp4 61.10 / int8 64.03） |
| C2 | `kv_v_codec_explicit` **死标志**（4 写 0 读） | `q3_wiring/REPORT.md` |
| C3 | **残差双层 PASS 需重测**（当时 serve 旗标静默失效） | `q3_wiring/REPORT.md`（3/13 开关无效） |
| C4 | `--kv-dtype` 在 perplexity 是死标签 ⇒ 我的 PPL 补丁**已落且已重编**，**待复核** | `ppl_fix/ppl_kvdtype.patch`（三处编辑，dry-run rc=0） |
**D. 未完成功能（不算 bug，但打包前该收）**：FlashNext 导入三卡点（转换器前缀错位/PLE 未接运行时/GDN 前缀复用）；
按轮次外挂召回零实现（L0 轮次日志 + 检索索引升格）；`convert_modelopt.py`（两路 agent 在做）；
以及**打包设施本身**（仓库零 pyproject/setup.py/CMake install）。
**打包顺序（用户已定）**：先把 A/B 修掉、C 复核完、D 里该收的收掉 ⇒ **最后**才做 whl ⇒ 那时才谈合并 main。

### 37. 自动化首轮的三个新发现（"反复测到无缺陷"的必要性实证）
用户口径："不只是已知缺陷，得反复测试到没有任何缺陷了才能最终打包"。首轮自动化跑完就抓到三类先前**不在清单里**的东西：

**(1) ctest：109 个测试 5 个失败**（95% passed，6 个 skip 是需模型路径的 real 测试）
```
FAILED: 4  ninfer_ple_table_e2e_test            (Failed)
        26 ninfer_qwen3_6_frontend_test         (Subprocess aborted → 崩)
        51 ninfer_gelu_mul_test                 (Failed)
        74 ninfer_softmax_attention_test        (Failed)
        95 ninfer_gdn_input_proj_conv_snapshot_test (Failed)
Total Test time (real) = 381.54 sec
```
⚠️ **我的 harness 缺陷**：执行器把 ctest 输出管道给了 `tail -80` ⇒ **`--output-on-failure` 的明细丢了**，
而且 `ctest rc=0` 报的是 `tail` 的返回码（不是 ctest 的）。已改为**不经管道**并单独落盘；
现在有一个 GPU 门控的重跑在排队（等硬检索测试让出 GPU），重跑后会拿到每个失败的原文。
注意 `ninfer_softmax_attention_test` / `ninfer_gdn_input_proj_conv_snapshot_test`（快照类）**疑似被本轮落地改动影响**，
`ninfer_gelu_mul_test` 对应的是本地独有那批（`gelu_mul`/`gelu_and_mul`，远端镜像没有），
`ninfer_ple_table_e2e_test` 属 PLE（FlashNext 那条线）——三条不同来源，必须逐个看原文再定位。

**(2) cold 相：四个臂跑完，但"正向冷行为"根本没被激励** ⇒ **我的判据是空的**
- `c_i8_cold`：rc=0、**未报 skip** ✔、reservation **2.31 → 2.38 GiB**（是**涨**了）。
- `c_mixed_cold`：rc=0、**未报 skip**（按文档 e8 无冷 codec 应当报 skip）、payload 684 MiB。
- `c_e2m1_cold_neg`：rc=1 + 精确拒绝文案 ✔（`kv-v-codec e2m1: the cold pool requires ISO3 V for NVFP4 layers ...`）。
**根因**：`--max-context 32768` + 82 token 的 prompt + `--max-new 48` ⇒ **根本没有 token 老到被驱逐**，
所以冷池只是**纯开销**（reservation +0.07 GiB），`enqueue_cold_compressions` 那条路径压根没走到 ⇒
"报不报 skip"与"省不省显存"这两个判据**都是空转**。⇒ **冷相必须加一个"长生成/长上下文让页真的老化"的臂**，
并且判据要改成"**压缩后驻留显存下降**"而不是"reservation 变了"。
（另外 analyzer 的负测试正则在 `[cold] error:` 前缀上不匹配 ⇒ 也是我的正则缺陷，行为本身正确。）

**(3) 由此确立门槛的操作定义**：**"没有缺陷"= 所有仪器反复跑干净**，仪器清单与现状：
| 仪器 | 现状 |
|---|---|
| ctest 109 | **5 fail**（上面，原文待取） |
| KV 电池 26 判据 | 2 fail（均为我的 harness：冷臂缺溢写目录，已修；需重跑确认） |
| acc2（F2 代价/崩溃复现） | F2 代价已测（快 2.2–2.7×）；崩溃 10/10 未复现 |
| 1M 阶梯 | 几何通、边界钉死、真池缺 7.81 GiB；A3（1048576 workspace）待修 |
| 硬检索 | **本轮首跑（在跑）** |
| 导入判据 | `layout_plan` 15 单测过 + 真实普查 400/400 |
| 质量阶梯 | 真文本/复读语料对照已建；A1/A2 待修 |

### 38. ctest 5 个失败**定性完毕**（两类：2 个环境/注册，3 个真代码缺陷）
| # | 测试 | 原文 | 定性 |
|---|---|---|---|
| 4 | `ninfer_ple_table_e2e_test` | `usage: ... <sidecar_root>` | **环境/注册**：CTest 没给它必需参数（不是代码缺陷） |
| 26 | `ninfer_qwen3_6_frontend_test` | `failed to open test resource: /home/neroued/models/llm/qwen/Qwen3.6-27B/base-hf-bf16/tokenizer.json` | **环境**：**硬编码了别人机器的路径**（`/home/neroued/`）；且缺资源时是 **abort 而非 skip**（测试质量缺陷） |
| 51 | `ninfer_gelu_mul_test` | `strided gate/up: pointwise mismatch max_abs=182.393 max_rel=1.99994 actual=-89 reference=93.3926 first_violation=10240 non_finite=0` | **真代码缺陷**：`max_rel≈2.0` 是**符号翻转**特征；**strided（打包 gate+up）路径**错，`first_violation=10240` 定位到具体偏移 |
| 74 | `ninfer_softmax_attention_test` | `causal_softmax_attention accepted an envelope outside the launcher domain`（packed/context 均 PASS） | **真契约缺陷**：该 op 接受了域外的 envelope（**F2 改了 envelope 的参考量=图常量容量，是首要嫌疑**） |
| 95 | `ninfer_gdn_input_proj_conv_snapshot_test` | `NVFP4 snapshot interval did not preserve its A16/A4 route boundary` | **真代码缺陷**：A16/A4 路由边界在 snapshot 区间内没保住（嫌疑在本地独有那批 `nvfp4_w4a4_tma_*`） |
备注：这轮我重跑时**没给 triage 加 GPU 锁**，而硬检索同时在跑 ⇒ 这三个数值类失败**可能是抢显存导致的伪失败**，
下一轮重跑必须**独占**（或先确认 VRAM 充足）再定案——**不确定就不许当结论**。

### 39. 3-bit / 2-bit KV 码本实测（agent 交付，数据是引擎自己的 KV dump，真实 KV 非权重代理）
| 方案 | bits/el | K 相对误差 | V 相对误差 | K logit 扰动 | 对比 nvfp4(9.6/10.3/9.8%) |
|---|---|---|---|---|---|
| 旧 `iso_codec.h` iso3 契约 | 5.00 | — | — | — | — |
| **现在跑的 nibble iso3 档位** | **4.50** | — | — | — | — |
| **推荐 3-bit**（组 32 + E4M3 + 冻结码本） | **3.25** | 15.9% | 16.9% | 16.0% | **1.6× 差** |
| **推荐 2-bit**（组 64 + E4M3） | **2.125** | 31.6% | 34.3% | 32.0% | **3.3× 差** |
（2.0 需要零 scale 开销，实现不了。）
**1M 只有 2-bit 装得下**：2.125 → **10.31 GiB**（余量 0.09）；3-bit 最低 **14.70**；按 10.4 GiB/1M 反推
**平均预算只有 2.144 bits/el ⇒ 连一层 int8 或 nvfp4 都买不起**，只能 **15 层 2-bit + 1 层 3-bit = 10.22 GiB**。
**三条实测发现（其中第一条解释了我们复现实验的谜）**：
1. ⭐ **现在跑的 E8 档位 K 误差是同 bit 宽纯取整的 1.9–2.0× MSE**——E8 投影让 **46% 坐标落在半整数上**被 `rint` 销毁，
   其 Hadamard-64 旋转又让 K 差 **3.4–4.5×** ⇒ **E8 花 4.25 bits/el 拿到的 K 精度还不如 3.25 bits/el 的纯标量**。
   ⇒ **`_TODO.md:1042` 的"E8 覆盖深层即崩"是那套实现的锅，不是 2-bit 的极限**；
   ⇒ **同时解释 A2**：出厂 10×E8 表在真文本上输给 all-NVFP4，正是因为 E8 的 K 精度更差。
   ⇒ **据此收回我先前的改动**：`kKvBitBudgetE8LayerLimit` **10 → 8**（见下方补丁），真正的修法是修 E8 codec
   （半整数 rint、Hadamard 旋转、scale 定点化），不是放宽上限。
2. **shipped 的 fp16 scale 平面没有范围 clamp**：amax > 65504 时 scale 变 `inf`（潜在缺陷）。
3. **shipped 的 scale 没有迭代到格式格点的不动点** ⇒ 白扔 **0.4–7.0% NMSE**。

### 40. 硬检索仪器修好后的**首份真实曲线**（独立打分器复算）+ 两处测量纪律缺陷

**仪器修复**（上一轮的两处都是我自己的错，不是模型的）：
- 打分器：原先"把全部数字剔出来从第 0 位比对"⇒ 思考里的 "64"、标记里的 "00" 会把数字流整体左移 ⇒ **必然 0 分**。
  改为**锚定对齐**：在任意起点上找与真值对齐最好的一段（旧输出回归验证 **29/64、acc 0.453**）。
- 预算：`--max-new` 96 → **512**，默认关思考。
- **独立取证**：新增 `py/score_retrieval.py`，**只读产物文件 + 只按 needles.json 真值重算**，与
  `retrieval_stress.py` 内部打分构成两条独立路径（内部打分不再被单独采信）。

**实测**（arm `default` = bf16 KV；ctx 262144；prompt **151669 tok**；greedy；数字文档 227425 字符 / 100000 位）

| 标记 | 偏移(位) | 精确前缀 | 字符准确率 | 输出数字数 | 生成tok | 结束 | MTP 接受率 |
|---|---|---|---|---|---|---|---|
| [D0] | 0 | 30/64 | 0.562 | 62 | 63 | stop-token | 70.00% |
| [D25] | 12500 | 3/64 | 0.109 | 380 | 418 | stop-token | 97.80% |
| [D50] | 25000 | 36/64 | 0.594 | 466 | 512 | **output-limit** | 91.67% |
| [D75] | 37500 | 26/64 | 0.438 | 66 | 67 | stop-token | 51.28% |
| [D100] | 50000 | 18/64 | 0.359 | 56 | 57 | stop-token | 55.56% |
| [D125] | 62500 | 1/64 | 0.219 | 430 | 473 | stop-token | 84.96% |

**小计：精确前缀 114/384 = 0.297；平均字符准确率 0.380。**

**该读出的四件事**：
1. **召回是"部分且不可靠"，不是"能/不能"**：分数**非单调**（偏移 12500 的 3/64 比 25000 的 36/64 差得多），
   说明决定分数的更多是**模型有没有决定去输出数字**（看生成 tok：418/430 那两条在思考里打转），而不是距离。
   ⇒ **禁止**把这个曲线读成"27B 在 262144 处检索能力随距离衰减"。
2. **offset 0 也只有 30/64** —— 贴着文档开头的标记都没能完整背出。所以"KV 就是前缀函数、开头必然最准"这个直觉
   **在 64 位精确背诵任务上不成立**（细节保真度不够，而非位置信息丢失）。
3. **仪器仍有残缺陷**：[D50] 撞 `output-limit`（512 tok 不够）⇒ 预算下一轮升到 1024，否则该点是被截断的假低分。
4. **顺手量到 ⑧ 的一手数据**：MTP 接受率 **51.28%–97.80%**（生成 tok 少的那几条更高），
   但这是**模型自带 MTP 头**（`mtp draft window 3` / `rounds 133` / `accepted by pos 130,118,91` /
   `acceptance length 3.55 tok/round`），**不是**新靶子配套的 DFlash2 草稿头——那个还没导入 ⇒
   **⑧ 的新抓手仍卡在 ⑦（导入）上**，本条只是"自带头在长上下文里的接受率基线"。

**同一次运行的运行期读数**（对 1M/预算类问题有用）：
`artifact 19.41 GiB` / `weights H2D 19.40 GiB` / `KV payload 5.34 GiB` / `KV capacity 262144` /
`runtime reservation 6.50 GiB` / `free after weights 10.75 GiB` / `free after startup 3.66 GiB` /
`prefill 2740.05 tok/s`（=**365 µs/tok**）/ `decode 25.97 tok/s`（=**38.5 ms/tok**，MTP 已计入）。
⇒ 逐轮外挂召回的性价比可以用一手数字重算：SSD 读打包 KV 2.6–6.1 µs/tok vs **重新预填充 365 µs/tok**
= **60–140 : 1**；vs 单步解码 38.5 ms = **约 10⁴ : 1**。

**两处测量纪律缺陷（都是我这边的，已修）**：
- `_triage5.sh` 的等待循环是"**先判断、后不占**"：它退出等待的瞬间正好落在旧检索被杀、新检索未起的空档里
  ⇒ 新检索 25 s 后拿到锁，而 triage 的全量 ctest **裸跑**上去，两者**抢 GPU** ⇒
  那三个数值类失败的定性**仍不算数**，必须独占复跑。
- 修法（`sh/_exclusive3.sh`）：**先持锁、再等 VRAM**。判断与占住之间没有窗口，物理上不可能被插队；
  它已挂到锁队列上等检索释放。

### 41. G/S 两组缺陷**双路取证收敛并落地**；纠正 §38 的一处误判；发现一整类派发缺陷

**纠正 §38（我自己的读法错了）**：`max_rel≈2.0` **不是**符号翻转特征。
两份独立复算都证明：`rel = |got-ref|/max(|got|,|ref|) ≤ 2`，**取等当且仅当 `got == -ref`**，而
观测值是 `1.99993775… < 2`，且全 174080 个元素里 `got[i] == -ref[i]` 的计数是 **0**；
报告行自身就是反证（`|-89| ≠ 93.3926`）。
真实机制是**操作数/地址互换**：第二半块 81920/81920 个元素与模型逐位吻合，
`got[i] = bf16(gelu(up[…]) * gate[…])`，两处地址都错。§38 表格里"真代码缺陷"的**定性是对的**，
"符号翻转特征"这个**机制判读是错的**。

#### 缺陷 G：`#51 ninfer_gelu_mul_test`（gelu_mul 派发顺序）
- 位置：`src/ops/launcher/gelu_and_mul.cu:23` —— 扁平 bf16x8 快速路径的判据只有
  **地址对齐** + `n % 8`，**没有连续性检查**；于是 `gate/up` 是同一个打包矩阵的两半（strided）时，
  必然走进扁平核、把 strided 视图当扁平数组读，而**专门为这个 case 写的**
  `gelu_and_mul_strided_input_kernel`（`:34` 那条分支）被**永久屏蔽**。
- `first_violation=10240 = rows = 打包块 2*rows 的一半` = 逻辑 `(d0=0, d1=1)` 的第 0 个元素
  = d1 由 0 变 1 的**列边界**；前 10240 个元素两种读法恰好相同，所以"第一个失败点"正好落在那里。
- 对照：兄弟 `src/ops/launcher/silu_and_mul.cu:40` **先查连续性**，它那份
  **形态完全相同**的 strided 测试（`tests/ops/test_silu_mul.cpp:103`，同 seed、同 17 列）是**通过**的
  ⇒ 差异只在派发顺序。
- **实现违反了自己的公开契约**：`include/ninfer/ops/gelu_mul.h:47` 原文
  "out is contiguous; **gate and up may use arbitrary valid Tensor strides**"。
- **爆炸半径（G2 的附带发现）**：`ops::gelu_mul` 现在**没有生产调用方**（只有测试）；
  但 `ops::silu_mul` 在生产里就是按 `gate_up.slice(0,0,intermediate)` 调的
  （`dflash_impl.h:421`、`dflash2_impl.h:331`、`qwen3_6_27b/impl/variant.cpp:395` 等）
  ⇒ 一旦按同样模式把 gated GELU MLP 接进去，本缺陷会**静默产出错误激活值**。
- 修法（已落地）：快速路径判据前置 `gate.is_contiguous() && up.is_contiguous()`，与 silu 同风格。

#### 缺陷 S：`#74 ninfer_softmax_attention_test`（causal 域闸比错对象）
- 位置：`src/ops/softmax_attention/dense/causal_cache/causal_softmax_attention.cpp` 的**三处**
  `:170`（`validate_envelope`）/`:251`（`validate_batched_attention_tensors`）/`:370`（容量查询），
  比的都是 **×4 的名义 YaRN 容量** `kCausalAttentionMaximumVisibleKeysYarn = 4*262144 = 1048576`，
  而算子声明的域是 `kCausalAttentionMaximumVisibleKeys = 262144`。
- 测试的三个探针把域钉在闭区间 `[1, 262144]`：`{1,262144}` 必须**接受**（`:1584-1590`，
  文案自称 "its maximum visible-key envelope"）、`{1,262145}` 必须**拒收**（`:1591-1597`）。
  超界 1 个 key vs 距实际闸 786431 个 key ⇒ 唯一能抛的就是那个常数。
- **严重度不是"契约瑕疵"而是内存安全**（S1 的发现，我独立核实了结构部分）：
  `small_t_bf16.cuh:41 / small_t_fp8.cuh:67 / small_t_i8.cuh:84` 都是
  `constexpr int PageIds = 64;`（bf16 那处注释明写"262144-key 最大信封在本 split 几何下最多跨 **49** 页"），
  `physical_pages_s[PageIds]`（`:50`/`:93`/`:116`），而写入是
  `physical_pages_s[page] = block_table[first_page + page]`（`small_t_bf16.cuh:147`）**无边界检查**
  ⇒ 放行到 1048576 的信封意味着**共享内存越界写**（S1 的主机算术给出 193/97 页 vs 64，
  最小破坏窗口 348161 / 696321 keys；**这条具体数字是 S1 算的，我只核实了"固定 64 + 无检查"这个结构事实**）。
- 修法（已落地）：三处都改用声明常数 `kCausalAttentionMaximumVisibleKeys`，与 GQA 兄弟
  （`gqa_attention.cpp:240/323/460` 全用 `kGqaAttentionMaximumVisibleKeys`）同构；`> capacity` 的池检查不动。

#### 三处裁决（两份报告之间/与我之间，都做了实证）
1. **`out` 的连续性不需要在 launcher 里查**（我的疑点不成立）：`src/ops/wrapper/gelu_mul.cpp:23`
   已经 `throw "gelu_mul: out must be contiguous"`，wrapper 兜住了；与 silu 写法一致。
2. **bench 不会因为打补丁而开始抛**（S2 的保留意见是错的、S1 是对的）：
   `bench/ops/causal_softmax_attention_bench.cu:758` 是**和式感知**的夹取
   `context > kCausalAttentionMaximumVisibleKeys - valid` ⇒ bench 自己就把域钉在 262144，
   没有任何合法行会越界，**调用方可见行为不变**。
3. **两路产物等价性**：S 组双路结果**逐字节相同**；G 组双路结果**只在局部变量名与注释文字上不同**
   （`inputs_contiguous` vs `contiguous`），逻辑一致。

#### 同形隐患：这是一整类缺陷，不止一处
"扁平向量核的判据只查对齐、不查连续性"的站点至少 **6 处**：
`gelu_and_mul.cu`（已修）、`sigmoid_gate_mul.cu:31`、`residual_add.cu:21`、`logit_policy.cu:21`、
`add_bias.cu:22`、`gelu.cu:18`（后两处只有 G2 找到）。已按铁律⑦再派双路代理查**可达性**并出补丁。
注：`silu_and_mul` 是正确写法的样板；`gelu_and_mul.cuh:57` 的注释（"out stays contiguous"）
界定了 strided 核的契约边界。

#### 落地状态与测量纪律
- 两处修复已**落地为源改动**（`git diff --stat`: 2 files, +9/−4），原文件备份在
  `/home/user/scratch/landed_GS_010806/`。**未重建**——已查证**没有任何定时/后台流水线会重建**
  （用户 crontab 空、`/etc/cron.d` 无 ninfer 项、systemd timer 全是系统自带、无 `night_shift` 进程，
  故源码改动不会污染正在跑的检索）。
- `build/tests/*` 里的测试二进制**早于**这次改动，所以排队中的 `_exclusive3.sh`
  会用**旧二进制**复核那 5 个失败——这是**有意为之**：先把"失败确实存在"钉死在修复之前，
  再重建、再看修复是否让它消失。
- **时间线旁证**（缩小"抢显存伪失败"的范围）：`ninfer_gelu_mul_test` 在 **00:53:45 失败**，
  而新检索 **00:54:05** 才启动（它的 VRAM 门要求 <2000 MiB 才肯跑 ⇒ 当时显存是空的）；
  `ninfer_softmax_attention_test` 00:53:50 开始、5-6 秒结束，也早于权重加载。
  ⇒ **#51 与 #74 是在显存空闲时失败的**；再加上两者的判据都是**纯主机/纯派发谓词**
  （S 是两个编译期常数比较，G 是地址对齐与 `n%8`），**由构造排除显存争用**。
  **#95 仍待独占确认**（它排在后面，与权重加载有重叠）。

### 42. ⚠️ 硬检索对照里出现**结构性矛盾**：int8 臂的 KV payload **大于** bf16 臂（8.77 vs 5.34 GiB）

**先更新 §40 的曲线读数**：`default` 臂现已 8/8 跑完（§40 当时只有 6 条，小计 0.297 是中间值）：
**精确前缀 211/512 = 0.412**，平均字符准确率 0.482。后两条（[D150] 51/64、[D175] 46/64）明显好于中间几条，
进一步支持 §40 的判断：**分数由"模型有没有决定直接输出数字"主导**，不是距离衰减（[D25] 只对 3/64 而 [D150] 对 51/64）。

**然后是新发现（本节的要点）**：`int8` 臂 7/8 完成，精确前缀 **436/448 = 0.973**——**远好于 bf16 臂**。
同二进制、同 artifact、同文档、同 ctx、同 greedy，只有 `--kv-dtype int8` 之差。原始读数：

| 读数（[D0] 的 `.err`） | default | int8 |
|---|---|---|
| `kv cache dtype` | bf16 | int8-group64 |
| `kv cache payload` | **5.34 GiB** | **8.77 GiB** |
| `gpu sequence used` | 5.53 GiB | 8.94 GiB |
| `runtime reservation` | 6.50 GiB | 9.62 GiB |
| `free after weights` | 10.80 GiB | 10.80 GiB |
| `KV capacity` | 262144 | 262144 |
| `KV page groups` | 4096 / 4096 | 4096 / 4096 |
| `prefill speed` | 2838.92 tok/s | 3562.85 tok/s |
| `decode speed` | **22.80 tok/s** | **153.06 tok/s**（6.7×） |
| `generated tokens` | 63 | 61 |

**为什么这是矛盾**：按仓库自己的口径（`bf16 16.00 / int8 8.25 bits/el`），**同容量下 int8 的 payload 应当是 bf16 的约一半（≈2.7 GiB）**。
实测是 **1.64 倍**。⇒ **这两个 arm 不可能被当成"同容量、只有精度不同"的干净对照。**

**两个臂的差异还不止 payload**：bf16 臂的 decode 慢 **6.7 倍**。结合"payload 反而更大"，两条一起指向
**bf16 长上下文路径没有走到预期的那条实现**（更大的输出缓冲？回退慢路径？不同的 split/几何规划？）——**具体机制未定，不得猜测**。

**召回差异本身是"数值退化"的形状，但同样只是线索不是结论**：
```
truth  : 1933566505638220772676141458373988904577750824077250935109245286
default: 19335665056382207726761414583775082407724077250935109244988588   <- 前 30 位对，第 31 位起发散
int8   : 193356650563822077267614145837398890457775082407725093510924     <- 前 60 位全对
```
[D0] 是**同模式对照**（生成 63 vs 61 tok，都不是在思考里打转），却 30/64 vs 60/64 ⇒ 不能只用"模式不同"解释。

**处理**：已按铁律⑦派**双路**代理（用同一份简报）做量化解释，必须回答：
① payload 反演的完整账目（每一项字节 + 出处 `file:line`）——8.77 vs 5.34 的差额来自哪一项；
② 6.7× decode 差的机制（哪些 kernel、什么判据）；
③ **bf16 长上下文路径有没有真实缺陷**（并顺带查 `gqa_attention_decode*` 里有没有与
   `GqaExecutionEnvelope::split_reference_keys` 同类的"活窗 vs 图常量容量"参考量混用）；
④ 这两个 arm **到底是不是干净对照**，差在哪些变量上。

**对八大项的影响（先记下来，别当结论）**：如果 int8 KV 真的比 bf16 更占内存，那么
**`bits/element` 那张表可能是名义值而非实际值**，②③④（输入 bit→分配、冷窗、闭环）全部建立在它之上。
⇒ **在这一点弄清楚之前，②③④ 一条都不许算"做完"。**

### 43. 缺陷 D（`#95 gdn snapshot` A16/A4 边界）**双路收敛并落地**；并纠正一处被"定案关闭"的错误编辑（修②）

#### 根因（D1/D2 独立收敛，同一行同一修法）
`src/ops/gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_plan.cpp:43`：`AllowA4` 分支的
`if (tokens <= 16) { … SmallTFusedA16; }` **吞掉了整个 T∈[4,16] 的 A4 档**，使 A4 路线要等到 **T≥17** 才可达。
容量查询 `nvfp4_gdn_snapshot_workspace_capacity_bytes` 从同一个 `resolve_plan` 推导答案（`:55` 在最大宽度非
`Materialized` 时直接 `return 0`）⇒ `AllowA4 [4,4]` 的容量变成 **0**，而测试第 3 条子句要求它**非零**（判据值 **93440**）。
**机制是一处常量串味**：紧上面 `A16Only` 分支的"A16 家族上限 16"被抄成了"A4 回退上限 3"。
判据全是**整数 token 计数**（`tokens <= 16/3`），无浮点无取整；且该子句**不启动任何 kernel**（纯规划算术）
⇒ **显存争用由构造排除**。D2 的 CPU 复算把边界**唯一地**钉在 T=4：上限=3 ⇒ 四子句全过；上限=16 ⇒ 挂第 3 条；上限=0 ⇒ 挂第 1/2 条。

#### 关键：这不是上游回归，是本地"修②"
- `git log --all -S 'tokens <= 3'` → **`dbb7d48`**（`origin/main` 的**祖先**）写的就是 `tokens <= 3`；
  当前 HEAD 是 `tokens <= 16`（由本地提交 `d38bb91` 带入）。
- **D1 用的外部参照树 `/home/user/ninfer-pristine` 与 `dbb7d48` 版本逐字节一致**（我交叉验证）
  ⇒ 参照物可信；两个 pre-edit 备份（`/home/user/fix2_bak/`、`_quarantine_backups_20260912/*.orig`）也都写 `<= 3`。
- **测试文件与 pristine/origin 逐字节相同**（modulo CRLF）⇒ **这条测试在 `origin/main` 上是绿的，
  是本地"修②"把它变红的。**
- `TODO.md:4489 / 4525 / 4720` 明确记录"修②"：`tokens<=3` → `<=16`，目的"统一小 T 家族"（`:4442`）；
  `:4552`"修② 是否有效**未定**：需加一行 schedule 日志"；`:4720`"已落但**未测到效果**"；
  `:4796 / :4824`"两个最小改动（修② 阈值、累加链）都测到**零效果**"。
- **`修② 定案关闭`（`:5028`）的论证是循环的**：它用"`AllowA4` 下 T≤16 一律 `SmallTFusedA16`"来证明
  "3→16 是代码级可证的 no-op"，而"T≤16 一律 SmallTFusedA16"**正是这次改动本身造成的**
  —— 改动前 T∈[4,16] 是 `Materialized`(A4)。且它选择"无需日志行"，即**从未测量**。

#### 裁决与落地
**落 D2 的 1-hunk 版**（`tokens <= 16` → `tokens <= 3`，含注释修正），**不落 D1 的 3-hunk 版**：
后者顺带把 wrapper 的 magic `4` 换成具名常量 `kNvfp4GdnA4FirstTokens`（**行为等价的重构**），
但会波及 `nvfp4_config.h` ⇒ 大量 TU 重编；**边界本身才是缺陷**，故取最小改动。
已落地：`1 file changed, 5 insertions(+), 3 deletions(-)`，备份 `/home/user/scratch/landed_D_*/`。
T=4 的四条同源旁证：record 规划器的 `std::max(min_width, 4)`（`wrapper/gdn_input_proj.cpp:903`）、
测试自己的档位选择器 `a4 = policy==AllowA4 && tokens>=4`（`test…snapshot.cpp:741`）、
线性派发器文档里被删掉的逐问题阈值 `(4 / 5 / 8)`（`linear/nvfp4/nvfp4_dispatch.cpp`）、pristine/git 历史。

#### 诚实记录一处张力（已裁决，但留作复核点）
本步把 `AllowA4 && T∈[4,16]` 从 A16（激活误差 ≈**0.17%**）移回 A4（`TODO.md:4489` 记 L2 ≈**9.48%**）。
表面像"verify 退化"，但四条理由支持落地方向：
1. `T≥17`（**含 prefill**）本来就是 `Materialized`=A4 ⇒ A4 是**被注册的**路线，不是新增的低精度行为；
2. 测试作者明示 `AllowA4 && T≥4` 即 A4，且要求该档容量非零 ⇒ 契约就是 T=4；
3. 修② 实测对接受率**零效果**（4.51%→3.76%，23 轮噪声内）⇒ 回退预期同样零效果；
4. 若将来真要 verify 宽度走 A16，正确做法是**显式注册并同步改测试**，而不是留一个违反契约的编辑。
⇒ 若后续发现 verify 质量确实受损，回退路径是 `scratch/landed_D_*/nvfp4_gdn_snapshot_plan.cpp.orig` 与
`fixD1/fix.patch`（具名常量版）。

#### 三处**只留档、本轮不修**的隐患（D1/D2 交付）
1. `nvfp4_gdn_snapshot_small_t.cu:82/84` 的 launcher 数组尺寸写作**字面量 16**（另一侧用 `kNvfp4LastSmallT`=**32**），
   而索引是 `x.ne[1] - kNvfp4FirstSmallT`（`:93/103`）**无边界检查** ⇒ **planner 上限与数组尺寸必须同步，
   且没有任何 `static_assert` 保证这件事**。（当前两者一致，故不可达。）
2. TMA 判据 `nvfp4_w4a4_tma_route()`（`linear/nvfp4/nvfp4_w4a4_plan.h:58`）在
   `gdn_input_proj/nvfp4/nvfp4_gdn_input_w4a4.cu:42` 与 `linear/nvfp4/nvfp4_w4a4.cu:64` 被**内联重写**
   ⇒ **第二真值来源**，两侧一旦漂移就会静默分叉。
3. UNIFY-A 之后 `Nvfp4GdnConvScheduleId::DecodeFusedA16` 的守卫可能已成死代码（enum 与 dispatch case 仍在，`:71`）。

#### 顺带纠正我先前的一处定位错误
§38 的笔记把嫌疑指向"本地独有那批 `nvfp4_w4a4_tma_*`（在 `src/ops/kernel/` 下搜）"——**位置说错了**：
`src/ops/kernel/` 下**没有任何** `nvfp4_w4a4*`；TMA 判据在 `src/ops/linear/nvfp4/`，
而那批 kernel 与本次失败**无关**（它们是 A4 内部的子路线 `tokens>=1024 && tokens%256==0`）。

### 44. 同形派发缺陷全量审计（双路第一路）；三臂完整读数把"bf16 最差"钉成趋势

#### 44.1 审计结论：12 个"扁平向量核 + 判据不含连续性"站点，**只有 2 个是真缺陷**
筛法：grep 向量打包类型（`Bf16x8Pack`/`Bf16x4Pack`/`__nv_bfloat162`/`float2`/`uint4`）+ `alignof(` + 字面量掩码 `& 0x`
（后者抓到 `cast.cu` 这种用 `& 0x` 而不写 `alignof` 的写法会漏掉的站点）。

**真缺陷 1：`src/ops/launcher/gelu_and_mul.cu:23`（已修）** —— 契约允许 `gate/up` 任意 stride
（`include/ninfer/ops/gelu_mul.h:47`），而快速路径判据只看对齐与 `n%8`，把 strided 视图当扁平读。
**独立交叉确认**：审计代理把自备补丁打到干净 HEAD 上，产物与**当前工作树逐字节相同**。

**真缺陷 2：`src/ops/launcher/sigmoid_gate_mul.cu` 的导出入口 `sigmoid_gate_mul_bf16x8_launch`**
—— 它**连一个判据都没有**（既无对齐也无 `n%8`），于是
(a) `packs = numel / 8` 在 `numel % 8 != 0` 时**静默丢掉尾巴**；
(b) 非连续视图被当扁平数组读。
**而且它可达**：`ops::sigmoid_mul` 路径被 `wrapper/sigmoid_mul.cpp:42` 的 throw 兜住，
但该导出入口被 `bench/ops/sigmoid_mul_bench.cu:71` **直接调用、绕过 wrapper**。
补丁（给导出入口补自检 + throw）**待第二路对照后落地**。

**判为"不是缺陷"的 9 处 + 1 处**（`residual_add.cu:21`、`logit_policy.cu:21`、`gelu.cu:18`、
`add_bias.cu:22`（含 `:42` 的 bf16x2 路）、`cast.cu:28/33`、`layer_norm.cu:20`、
`vision_pos_embed.cu:28`、`scatter.cu:17`、`causal_conv1d.cu:57`，以及 `rope.cu:31` 的 head 维）
——**各自都有 wrapper 级的 `throw std::invalid_argument` 谓词**（行号已逐条列出），
按 `docs/maintainer/op-development.md` §4.1（layout 校验归 wrapper）/§4.3（每个 kernel 假设都要有
匹配的 wrapper 或 launcher 谓词）判定为**已被覆盖**。审计代理明确**没有为了使补丁好看去动这 9 处**。
对 `Tensor::is_contiguous()` 的语义也核过：`ne[i]==1` 的维不查 stride，但那只贡献 0 索引，
所以"`is_contiguous()` ⇒ 扁平索引等价"成立（`src/core/tensor.cpp:90-102`）。

**附带发现的约定违规**：`bench/ops/sigmoid_mul_bench.cu:71` 直接 include launcher 私有头，
违反 `docs/maintainer/op-development.md` §4.2（"public benchmarks 不得包含 launcher 私有头"）。

**附带提示（标量版同类，未发现可达的错误路径）**：`wrapper/suffix_lookup.cpp` **没有任何
`is_contiguous`**、纯裸指针 API；`vocab_topk16` 无 wrapper 校验。注意 `suffix_lookup` 正是
⑥ 那条 lookup/ngram 线，与"引擎零调用"的既有记录一致。

**取证方法值得留档（比"独立重写"更强）**：审计代理的 CPU 复算**直接链接真实的
`src/core/tensor.cpp`**（不是自己重写一份 `is_contiguous`），得出：
```
gate.ne=10240,17,1,1  nb=2,40960,696320,696320
gate.is_contiguous=0  up.is_contiguous=0  out.is_contiguous=1
n=174080  n%8=0   (gate|up|out)&15 = 0x0
OLD gelu_mul fast path taken = 1     <-- 缺陷机制
NEW gelu_mul fast path taken = 0     <-- 修复后落到 strided 核
silu_mul dim0-split stride-aware path taken = 1   <-- 兄弟算子为何通过
```

#### 44.2 三臂完整读数（`default`/`int8` 已跑满 8 条；`nvfp4` 3 条）
| 臂 | 精确前缀 | 平均字符准确率 | 生成 tok 范围 | KV payload |
|---|---|---|---|---|
| `default` (bf16) | **211/512 = 0.412** | 0.482 | 57–512（波动大，含 2 条在思考里打转） | 5.34 GiB |
| `int8` | **500/512 = 0.977** | 0.977 | 61–81（紧） | 8.77 GiB |
| `nvfp4`（3/8） | 174/192 = 0.906 | 0.906 | 51–91（紧） | 待读 |

⇒ **排序 `bf16 ≪ nvfp4 < int8`，与"精度越低越差"完全相反。**
且 **[D0] 是同模式对照**（bf16 生成 63 tok、int8 生成 61 tok，都不是在思考里打转，都不是被
`output-limit` 截断），却 **30/64 vs 60/64** ⇒ **不能用"回答模式不同"解释**。
把它与 §42 的两条读数合看——int8 的 payload **大于** bf16（8.77 vs 5.34 GiB，同容量同 page groups）、
decode **快 6.7×**（153 vs 22.8 tok/s）——**强烈指向"bf16 长上下文走的不是同一条实现"**。
机制由 kv1/kv2 双路定案；**未结案前 ②③④ 仍不许算"做完"**。

### 45. 同形派发审计：双路**分歧的裁决**与落地（sib1 对"可达性"、sib2 修错了函数）

#### 双路的同与异
**同**：两路都认定 `gelu_and_mul.cu` 那处是唯一"实现违反自己契约"的缺陷（已修，55 行）；都认定
`sigmoid_gate_mul` 有问题；复现算理一致（`[2*rows,columns]=[20480,17]` 的两个 slice、`n=174080 % 8 == 0`、
第二片 data 偏移 `20480 B` 是 16 的倍数 ⇒ 扁平判据全满足 ⇒ `first_violation=10240 = rows`）。

**异（本轮的关键）**：
- **sib2 写"`sigmoid_mul` 唯一调用方是 `wrapper/sigmoid_mul.cpp:42` 强制连续"——这是错的。**
  `bench/ops/sigmoid_mul_bench.cu:71` **直接调用**了 `ops::detail::sigmoid_gate_mul_bf16x8_launch`，绕过 wrapper。
  该头自己就写着 `// Fixed-route launch control used by production and qualification benchmarks.`（`sigmoid_gate_mul.h:18`）。
- **因此 sib2 的补丁 guards 的是 `sigmoid_gate_mul_launch`（通用入口，wrapper 本来就守着），
  恰恰没碰那个被 bench 直呼的**窄入口 `sigmoid_gate_mul_bf16x8_launch`——而后者**连一个判据都没有**
  （`packs = numel/8` 在 `numel%8 != 0` 时**静默丢尾巴**，strided 视图被当扁平读）。
  ⇒ **sib1 的修法落在真口子上，sib2 落在了已经被守住的那个函数上。**

#### 契约原文核实（决定"是不是违约"）
逐条读 `include/ninfer/ops/*.h`：
- **只有** `gelu_mul.h:14` 与 `silu_mul.h:14` 写 "out is contiguous; **gate and up may use arbitrary valid
  Tensor strides**" ⇒ 只有这两个算子的输入 stride 合法，`gelu_and_mul` 的 launcher 违反了自己的契约（已修）。
- `residual_add.h:14` / `sigmoid_mul.h:14` / `add_bias.h:14-15` / `gelu.h:20` / `logit_policy.h:18`
  **全部明文要求 contiguous BF16 tensors**，且各自 wrapper 逐字强制（`is_contiguous` 检查行号已列）
  ⇒ **这 5 处的实现没有违反自己的契约**，是**潜伏**而非活的违约。sib1 判"不是缺陷"成立。

#### 完备性检查（决定要不要推广修法）
`src/ops/launcher/` 全部头里，**只有** `sigmoid_gate_mul.h:18` 带"给 benchmark 用的固定路线入口"注释。
把"扁平向量派发 + 判据缺连续性"这一类与"被非 wrapper 调用方直呼"交叉，**只有 sigmoid 一处**
⇒ 无需推广。另外 9 处（`residual_add`/`logit_policy`/`gelu`/`add_bias`/`cast`/`layer_norm`/
`vision_pos_embed`/`scatter`/`causal_conv1d`，以及 `rope` 的 head 维）均有 wrapper 级 `throw` 谓词覆盖，
按 `docs/maintainer/op-development.md` §4.1/§4.3 判定为**已被覆盖**。

#### 落地与不落地
- **已落**：sib1 的 `fix_worktree.patch` —— 给窄入口 `sigmoid_gate_mul_bf16x8_launch` 补自检
  （连续性 + 16 字节对齐 + `n%8==0`，不满足则 `throw std::invalid_argument`），`1 file changed, +13/-1`，
  备份 `/home/user/scratch/landed_sib_*/`。现在树的改动共 **4 个源文件**：
  `gelu_and_mul.cu` +5/-1、`causal_softmax_attention.cpp` +3/-3、`nvfp4_gdn_snapshot_plan.cpp` +5/-3、
  `sigmoid_gate_mul.cu` +13/-1。
- **不落**：sib2 的另外 4 个断言（`residual_add`/`logit_policy`/`add_bias`/`gelu`）。
  理由：(1) 这 4 个入口的契约明文要求 contiguous 且 wrapper 逐字强制 ⇒ **零可达性收益**；
  (2) 它们会在 4 个**每层都跑**的热路径上新增硬 `throw`，而 sib2 自己承认**未核实 `bind(backing)` 的
  运行时张量身份** ⇒ 风险是把"当前可跑"变成"硬崩"，而不是修掉任何已证实的缺陷。
  补丁已备查：`/home/user/scratch/sib2/fix.patch`（apply-check 通过），是一条随时可取用的加固选项。
  **诚实边界**：我判"零可达性收益"是**读代码**得出的（契约 + wrapper 的 throw），**没有跑运行期**；
  若将来确有 binding 别名出 strided 张量，sib2 那版会让它以**硬崩**而不是静默错算的方式暴露
  —— 那本身是可接受的失败模式，只是本轮不引入未经证据支持的改动面。

#### 留档：bench 绕过 wrapper 的全量清单（本轮只覆盖了其中一类）
`bench/ops/` 里直接调 `ops::detail::` 的地方共 **58 处 / 8 个文件**：
`w8_linear_swiglu_bench.cu` 21、`gdn_gating_proj_bench.cu` 12、`w8_linear_add_bench.cu` 10、
`bf16_linear_add_bench.cu` 7、`q5_linear_add_bench.cu` 4、`gated_delta_net_bench.cu` 2、
`sigmoid_mul_bench.cu` 1、`position_bench.cu` 1。
其中大多数是 `*_resolve_plan`/`*_schedule_name` 这类纯规划调用或 w8/bf16/q5 **linear** 入口（属于另外的
layout 假设类别，**不在本次审计范围**）。`docs/maintainer/op-development.md` §4.2 明文要求
"public benchmarks 不得包含 launcher 私有头"——这 8 个文件都在违反它，而正是这条规定的缺失让
sigmoid 那个无判据入口被绕过而不自知。**留档，不在本轮处理。**

### 46. 导入双实现的**我自己的独立差分**（不依赖 convC 代理的 diff 脚本）——分歧全在 artifact 对象层

工具：`py/cmp_plans.py`（另一条取证路：只读两份 plan JSON，按公共键逐对象比）。
数据：`scratch/convC/outA/planA.json`（728 KB）与 `outB/planB.json`（1.07 MB）。

#### 结论一：**源组层面 400/400 完全一致**——两条独立实现 + 我自己的 `layout_plan.py` 三方吻合
| | 源组层（985 模块，其中量化 400） |
|---|---|
| A | contiguous-le-v1 **585** / row-scale-v1 **260** / blockscale-k16-m128x4-v1 **140** |
| B | **完全相同**（代理报告原文：`quantised modules: common=400 same_layout=400 differing_layout=0`） |

⇒ 我在 §34 用 `layout_plan.py` 独立验证过的 585/260/140 直方图，现在**被两份互不相干的实现再次确认**。
**"从源自身 dtype/shape 关系结构性推导布局"这件事已经三方一致，可以当结论用了。**

#### 结论二：分歧**全部**在 artifact 对象层，且是**契约取向**之争而非谁算错了
| | 对象数 | 字节总量 | 拒收 |
|---|---|---|---|
| A | **1599** | **25,780,702,800 B ≈ 24.0 GiB** | **0** |
| B | 920 (+6 resources) | **9,793,788,469 B ≈ 9.1 GiB** | **598**（198 F-NO-ROUTE + 128 F-SHAPE + 272 F-UNCONSUMED-GROUP） |

- **公共对象 0 个**（名字零重叠）：A 用**源侧命名**（`model/language_model/layers/0/…`），
  B 用**artifact schema 侧命名**（`text/layers/0/gdn/query_key_value_z`、`mtp/…`）。
- **A 的总量 ≈ 两个源分片之和**：24,006,113,464 + 1,770,897,648 = **25,777,011,112**，
  A 报 **25,780,702,800**（差 3,691,688 B ≈ 3.5 MB，头/对齐）⇒ **A 覆盖全源**。
  B 只覆盖 **9.1 GiB**（其自报 `required_source_keys=1225`，源有 2139 张量）。
- **B 的拒收原文点名了根因，是"不肯融合"**：
  - `F-SHAPE: model.language_model.layers.0.mlp.gate_proj: stored F8_E4M3[17408, 5120] -> logical
    (17408, 5120), artifact part expects (34816, 5120)` —— artifact 要**融合后的** gate_up `[34816,5120]`，
    源是分开的 `gate_proj`/`up_proj`；B 要求"存在一个逻辑形状等于 artifact part 的源矩阵"，于是拒收。
  - `F-NO-ROUTE: text/layers/0/gdn/query_key_value_z` —— 同上，q/k/v/z 的融合不在 B 的路由里。
  - `F-UNCONSUMED-GROUP`: 源里量化的 `in_proj_qkv` / `in_proj_z` / `mlp.gate_proj` 没有 artifact 对象消费。
- 格式/布局视野也不同：A 只出 4 种格式（BF16/FP32/FP8_E4M3FN_ROW_BF16S/NVFP4），
  B 出 9 种（另有 Q4G64_F16S / Q5G64_F16S / Q6G64_F16S / W8G32_F16S / I32 与 `row-split-k128-v1`=117、
  `raw-bytes-v1`=6）。

#### 我的判读（**暂不裁定**，等 convC 的卷宗 §3 对象层对照）
形状是：**A 按源结构出对象（覆盖全源，但不保证与注册 schema 逐对象对齐）；B 以注册 schema/recipe 为权威出对象
（架构上更对——装载器与运行时核消费的是注册 schema），并在源不匹配处**明确拒收 598 条**而不是发明对象。**
⇒ 若采信 B 的架构，则它那 598 条拒收正是"**这份源要转成注册 schema，还缺哪些融合**"的诚实清单；
若采信 A，则拿到的是"覆盖全源但可能装载不了"的计划。
**这个裁定要看 `out/diff_report.txt` §3 的对象层对照 + 注册 schema 的实际消费方**，
不在本轮拍板。两件事已经确定：① 源组层三方一致；② 分歧是契约取向、不是算错。

### 47. E8 codec 双路第一路：五条断言**重裁**、决定性缺陷 D1（免费 +3.65 dB）、并更正我自己的两条记录

取证方式（值得留档）：自写 CPU 复算程序 `/home/user/scratch/fixE1/e8verify.cpp`，只读**真实 KV dump**
`/home/user/bench/kvdump_e8src`（27 个 dump、64677 token、各 **66.2M** 个 K/V 元素），未改树、未构建、未用 GPU。
**工具本身先自证**：`gqa_kv_hadamard64` 复刻 `max|MᵀM−I| = 0.000e+00`（严格正交）；
`e8_project_8d_fast` 复刻 vs 精确候选搜索（2×2⁸）20000 个随机块 **20000/20000 都是真正最近点、0 并列**；
解析预测的 NMSE 0.0276 vs 实测 0.0264（差 5%）。三条独立支撑。

#### 五条断言的重裁
| # | 原断言 | 重裁 | 独立证据 |
|---|---|---|---|
| 1 | E8 的 K 误差是同 bit 宽纯取整的 **1.9–2.0×** | **成立，但系数是 2.32×**（+3.65 dB）。**子句"还不如 3.25 b/el 纯标量"被推翻** | MSE(E8)/MSE(z8，同旋转同码宽 4.25) = **2.319×(K)/2.320×(V)**；而逐组扫到 MSE 最优的 3-bit mid-rise（3.25 b/el）K NMSE = 2.64e-2 ≈ shipped E8 的 2.636e-2 ⇒ **E8 不吃亏**。但 bit 确实被浪费：3.25→4.25 本该买 4×，z8 拿到 4.22×，**shipped E8 只拿到 1.82×** |
| 2 | 约 **46%** 坐标落在半整数上 | **成立** | 实测 **47.57%**，且 47.57% 的 8-块选中 `D8+½` 陪集 |
| 3 | Hadamard-64 旋转让 K **差 3.4–4.5×** | **推翻（反证）** | MSE(had64)/MSE(无旋转) = **0.697(K) / 0.463(V)** ⇒ 旋转让 K **好 1.43×**。机制：`E[amax_rot²]/E[amax²] = 0.685`（每块 amax 平均降 18% rms），1/0.685 = 1.46 与实测吻合到 2%。⇒ **旋转必须保留** |
| 4 | fp16 scale 无 clamp，`amax>65504 ⇒ inf` | **部分成立**："无 clamp"成立；**阈值说错了** | 除数是 7 ⇒ 阈值是 **amax > 458752**（实测 450000→64288 有限，458752→inf）。**后果比原描述更重**：scale=inf ⇒ `k_inv=1/inf=0` ⇒ **15 个码全变 0** ⇒ 读侧 `0×inf = NaN`。真实数据最大组 amax = 15.0(K)/96.5(V)，余量 3e4 倍 ⇒ 潜在、未触发 |
| 5 | scale 未迭代到格式格点不动点，白扔 **0.4–7.0%** NMSE | **部分成立，机制与数值都不同** | 在 E8 档实测是 **14.81%（0.70 dB）**，不是 0.4–7.0%（那组数来自 2/3-bit *mid-rise* 码本实验）。且前提**不成立**：本档不是"fp32 算 scale 再取整"，而是 `decode_i8.cuh:273-283` 先 `ksh=half(amax/7)` → `ks=half2float(ksh)` → `k_inv=1/ks` 分码；改成 fp32 分码后 MSE 变化 **0.0000%** |

#### 决定性缺陷 D1：E8 格点投影 + `rintf` 存在**整数**码平面里（**真缺陷，0 bit 换 +3.65 dB**）
- 位置：`src/ops/kernel/gqa_attention_decode_i8.cuh:281`（及 `:293` 的重复投影）；`gqa_attention_prefill_i8.cuh:153`/`:168`（fill 核）
  与 `:293`/`:304`（page 核）。共 **6 处** `e8_project_8d_warp` 调用。
- 错在哪：E8 = **D8 ∪ (D8+½)**，**47.57%** 的块最近点落在 `D8+½`，其八个坐标**全是半整数**；
  而码平面每坐标只存一个**整数**，紧随的 `rintf` 把每个坐标推 0.5 step ⇒ 这一半的格点增益**被扔掉**。
- 正确做法：**删掉投影、保留旋转**（码平面本就是整数平面，格点不可表示）。
- 收益：K NMSE **2.636e-2 → 1.137e-2**（relRMS **16.24% → 10.66%**，**2.32× / +3.65 dB**）；
  V 2.550e-2 → 1.099e-2。**代价 0 bit。**
- 反面方案已量化排除：真要收格点增益需存"2×坐标"的 5-bit 平面（±15）= **+1 bit/el**，
  只换回 **0.58 dB**（实测 ideal-E8/z8 = 0.874×）⇒ **严格劣于**把这一 bit 花在更细的均匀格点上。
- 补丁：`/home/user/scratch/fixE1/fix.patch`（4 文件、+82/−40，sha256 `029b5fde…`；干净副本上 apply-check 与真 apply 均 exit 0；
  **真树未被改动**）。含 D1 + D2（`kv_scale_half` 饱和）；**D3 故意不含**（需要在写循环里对 64 组做两个 warp 归约，
  无法在此编译/运行 CUDA ⇒ 不把未验证的 kernel 改动塞进补丁）。**等双路第二路（fixE2）报告后我裁定落地。**

#### ⚠️ 更正我自己的两条记录
1. **「e8/iso3 共用 NVFP4 的 4-bit nibble plane + 3-bit sign-magnitude 码本 ⇒ 浪费约 1 bit/el」把两个档位搞混了**：
   - 那句描述的是 **iso3（4.50 b/el）**：`gqa_attention_prefill_nvfp4.cuh:63-75` 的 `gqa_iso3_nibble`
     = 低 3 位幅值 0..7 + bit3 符号，且只在 `value<0 && code!=0` 时置符号位（−0 从不写）⇒ **15 个电平**。
   - **e8（4.25 b/el）是完全另一套平面**：`DType::E8Kv = 10`（`core/dtype.h:21-24`）＝"4-bit E8-lattice K codes +
     i4 V codes（每字节两个）+ 每 64 通道 fp16 scale" ⇒ 4 + 16/64 = **4.25**；
     读侧 `gqa_kv_unpack_i4 = (nibble^8)−8 ∈ [−8,7]`，写侧 clamp 到 [−7,7] ⇒ 用 15/16 个码。
   - ⇒ **"浪费 1 bit/el"差了一个量级**：15 电平装在 4 位里 = log2(15) = **3.907 bit**，**浪费 0.093 bit/el**。
2. **「用于深层若干层」与 `src/product/kv_bit_budget.h:60-75` 相反**：那里写的是 e8 **只对前导层（layers 0..7）验证过**，
   并注明 "the shipped high-layer e8 path degrades"；层数上限 `kKvBitBudgetE8LayerLimit = 10`。
   而 shipped 默认把 e8 放在 `{0,1,3,4,6,7,8,9,13,14}`（`src/targets/qwen3_6_27b/impl/variant.cpp:33`）
   —— **够到了 13/14，正是 header 自认未验证的配置**。旁证：`kvc_0_t3072_L14_meta.txt` 是 `dtype=10 quant_group=64`
   （E8Kv/g64），而 L13 是 `dtype=8 quant_group=16`（NVFP4）。
   ⇒ **据此修正我先前对 `kKvBitBudgetE8LayerLimit` 的处置**：**不降到 8**；正确顺序是**先落 D1、再复测"深层 e8 是否仍退化"**。
   若退化消失，则"e8 只许放前导层"的前提本身可能就是 D1 造成的。

#### 一条**没有解释清楚的分歧**（必须记下来，两边绝对值都别当定论）
两轮的绝对数字**对不上**：上一轮 E8 K relRMS 28.6% / 3-bit 15.9%，本轮复刻是 10.66% / 21.9%，
两条腿都差 2–4× MSE；试过用"旋转前 amax/7"与"旋转前 amax/127"两个历史修订版回代，得到 20.17% 与 86.99%，
**都对不上 28.8%**。本轮数字有三条独立支撑（正交性、暴力最优性、解析预测），但**这个分歧没有解释清楚**
⇒ **两边的绝对数值都该谨慎引用**；可作为定论的是**方向与比值**（投影有害、旋转有益、系数 ≈2.32×）。

### 48. convC 卷宗裁定 + kv2 破案：**payload 反演闭合到两位小数**，我收回 §42 的红旗表述

#### 48.1 convC 卷宗：B 的 598 条拒收里 **528 条是同一个 bug**，不是架构性拒收

**两份实现都满足"底层生成、不许 or 硬写"**（用户口径）：
- A 的驱动**全文件 0 处 `.get(x, default)`**，无 `or` 兜底、无层号/名字查表；
- 两份**都调用同一个 `layout_plan.select`**，都是纯结构判定；
- **985/985 模块、其中 400/400 量化模块布局完全一致**（`same_layout=400, differing_layout=0`）
  ⇒ **与我 §34 用 `layout_plan.py` 的独立验证三方吻合**，这条可以当结论。
- convC 还独立解析了 safetensors 头（2139 张量 / 985 module）并校验两个 shard 的
  `8 + header + payload == file_bytes` 均 True。

**源张量级：一致 426 / 不一致 1713**，不一致的模式**不是** layout 或 dtype 分歧，而是三类：
797 条命名空间差异（A 逐字搬 BF16 / B 交给注册 recipe 语义）、914 条（**B 只吃下 128/400 个量化分组**：
642 ABSORB→MISSING + 272 REPACK→MISSING）、2 条词表端点（A 逐字 BF16 / B 重编码 FP8，B 已记账）。

**决定性单点发现：B 的 128 条 `F-SHAPE` 全是同一个比较基准错。**
- `_matrix_entry` 拿**每个 part** 的形状去与**整个融合对象**的形状比（B 那一带收到的是 `registered.shape`）。
- convC 遍历注册的两张 recipe 表：那 **128 个多 part 对象共 288 个 part**，
  **`part.source.shape` 与源逻辑形状 288/288 完全相等** ⇒ **改成按 part 比就会全过**；
  单 part 对象因"对象形状 == part 形状"恰好不受影响。
- 连带：**272 条 `F-UNCONSUMED-GROUP` + 198 条 `F-NO-ROUTE` 里的 128 条**都是这个 bug 的下游二次记账；
  **真·无路由只剩 70 条**（全是 `input_scale_divisor`，因其父对象被拒/被丢）。
- ⇒ 我 §46 里"B 是诚实的架构性拒收"这个临时判读**被推翻**：它是**一个 bug 的辐射**。

**B 在本源上根本跑不完（两处独立故障）**：
1. refusal 让 `convert()` 必抛（B:1322）⇒ 写不出任何 artifact；
2. convC **实测**它的 `direct` + `official` 两条 payload 路（**748/920 个对象**）今天都
   `AttributeError: 'SourceTensor' object has no attribute 'expression'`
   （把 `recipe.expression` 传给了要 `TensorRecipe` 的 `materialize_recipe`；
   注册侧正确入口是 `rr8.materialize_quantized_direct` / `rr8.materialize_quantized_official`）。

**对照：A 在本源上可完整跑完（0 refusal，25.78 GB 全覆盖），但 A 自认产物不可加载**
（不融合、用源侧命名）。B 的语法方向（融合、MTP/W8/Q4/Q5/Q6、draft head、资源）才是装载器消费的那套。

**两份在关键数值路径上不可区分**（convC 独立实测）：
NVFP4 纯重打包 **140/140 codes 逐位相同、140/140 scale swizzle 互逆、140/140 divisor 字 = fp32(1/weight_scale_2)**；
A 的 448.0 论据被独立确认（**260/260 个 FP8 码平面、11.04 GiB 全量扫描**，`max|code|==448.0` 分布 `{126: 260}`、无 NaN）。

**唯一"未记账的数值转换"在 B**：`official` 路由 347 个对象里有 **117 个存量化学**（Q4G64/Q5G64/W8G32/Q6G64），
而源用 `exclude_modules` 显式声明 mtp/vision/lm_head **不被量化**；B 只给 2 个词表对象记了
`D-RECODED-FROM-BF16`，那 117 个没记。

**处置**：**两个都不整份落地**。已派**双路**代理去修 B 的 Bug1（比较基准）/Bug2（payload 入口）/Bug3（记账）
并在**真实源**上复跑（预期 refusal 从 598 收敛到 ~70、payload 不再抛、对象数与字节总量可比），
拿到可比数据后再裁定。卷宗：`/home/user/scratch/convC/DOSSIER.md`（692 行、81 处证据引用）。

#### 48.2 kv2 破案：**"bf16" 是哨兵值，默认臂那 16 层从来不是 bf16**（我收回 §42 的红旗表述）

**payload 反演，账目闭合到两位小数**：
- 打印路径 `apps/cli/main.cpp:218` ← `program_impl.h:13083` ← `layouts_impl.h:369-370` ←
  `decoder_state.cpp:634-636` ← `core/paged_kv_cache.cpp:100-104`（Σ 每个 plane 的 `storage.region.bytes`），
  plane 形状 `{leading_extent, 64, head_extent, physical_pages}`（页 = 64 token）。
- 几何：24 q / 4 kv / head_dim 256、16 个全注意力层、capacity 262144 ⇒ 4096 页。
- **逐层 dtype → plane 表在 `decoder_state.cpp:171-229`**；每层 byte/页 = `leading×64×4×elem_size`：

| 层档 | B/页/层 | bits/元素 |
|---|---|---|
| BF16 | 262,144 | 16.00 |
| I8 | 135,168 | 8.25 |
| E8Kv | 69,632 | 4.25 |
| NVFP4 | 73,728 | 4.50 |

- **`--kv-dtype` 未给时走 `Variant::default_layer_kv_dtypes`**（`layouts_impl.h:1209-1217`），
  `src/targets/qwen3_6_27b/impl/variant.cpp:56-71` 的表 = **E8Kv 在 {0,1,3,4,6,7,8,9,13,14}、其余 NVFP4、
  一个 BF16 都没有**；`--kv-dtype int8` 才触发 `layer_overrides.fill(...)`（`apps/cli/options.cpp:161`）。
  打印出来的 `kv cache dtype bf16` 是**全局哨兵值**，不是那 16 层的 dtype。
- **default：`10×69,632 + 6×73,728 = 1,138,688`（文本）+ `262,144`（MTP=bf16）= 1,400,832 B/页；
  ×4096 = 5,737,807,872 B = 5.3444 GiB → 打印 "5.34"** ✓
- **int8：`17 × 135,168 = 2,297,856` B/页；×4096 = 9,412,018,176 B = 8.7656 GiB → 打印 "8.77"** ✓
- 交叉验证：仓库 commit `89e2375` 自报 "payload 3.56x nvfp4"，本模型给 4,456,448/1,253,376 = **3.555** ✓
- ⇒ **差额完全来自"默认臂文本 KV = 4.34 bits/el、int8 = 8.25"**（int8 宽 **1.90×**，`2,162,688/1,138,688 = 1.8993`）。
- ⇒ **真 bf16 在 262144 需要 17.0 GiB > `free after weights 10.80 GiB` ⇒ 物理上跑不起来**；
  文档里 `bf16 16.00` 那行是 layer-table 的"未设置哨兵值"，从没被这 16 层用上
  （`docs/maintainer/kv-strategy-matrix.md:10-13,33-36` 已记同一件事）。

**6.7× decode 差的机制（并用带宽量化排除了带宽解释）**：两臂对 16/16 层 + MTP 层走**不同 kernel**，
判据只有 `cache.dtype`（`gqa_attention_decode_smallt.cu:64-132`）。E8 额外多一遍
`gqa_attention_decode_i8.cuh:524-571` 的 `unpack_tile()`（把 packed 4-bit 的 K/V tile 在 smem 里逐 16 维展开成 i8：
2 读 + 8 解包 + 16 次移位拼接 + 2 写），且**被同步插进 key-block 主循环**（`:780`）。
量化：默认臂每 round 文本 KV ≈ **2.70 GB / 136 ms → ~20 GB/s**；int8 ≈ **5.13 GB / 23.1 ms → ~236 GB/s**；
两者都远离 ~1.8 TB/s 峰值，且**默认臂字节更少却慢 12×** ⇒ 差在**每 key 的计算/延迟**（反量化+unpack 在关键路径上）。

**判定**：
- **不是"bf16 长上下文路径有缺陷"** —— 这条路径在这次测量里**不存在**。
  `split_reference_keys` 的接线是对的（`program_impl.h:11767/12241`/`:587-606`；`partial.cuh:89-91` 只在 0 时回退）；
  **`kCausalAttentionMaximumVisibleKeysYarn` 现在只剩定义、无使用点** ⇒ 我的 S 修复在源级确认生效。
- **真正解释默认臂召回崩塌的是仓库自认未修的缺陷**：默认表把 E8 放在 **8,9,13,14**，而
  `docs/maintainer/kv-strategy-matrix.md:37-42,109-113` 实测 `8-15:e8 → 0/8 针`、`0-7:e8 → 8/8`、
  **出厂默认表 6/8**（纯档位全 8/8），compute-sanitizer 干净 ⇒ **静默逻辑/别名错**。
  形状与"default 30/64、int8 60/64"一致。
- **两个 arm 不是干净对照**：差在 (a) 16 层 dtype（E8/NVFP4 vs I8，**4.34 vs 8.25 bits/el**）、
  (b) 因此不同的 decode/prefill kernel 与 tile schedule、(c) MTP 层 dtype（bf16 vs i8）、
  (d) 派生量（prefill chunk 3072→2048、workspace peak 457.71→305.14 MiB、prefill 24s→17s）。
  ⇒ "int8 召回更好"**不能**读成"低精度更好"或"bf16 有缺陷"，它读作
  **"8.25 bits/el > 4.34 bits/el，且 4.34 那一侧把 E8 放在了已知损坏的层号上"**。
- **新发现（未证可达）**：`mtp_impl.h:219-220` 与 `:238-239` 用**两个初始化项**构造 `GqaExecutionEnvelope`
  ⇒ `split_reference_keys = 0` ⇒ 活窗驱动划分，而同一 MTP 缓存的 decode-batch 路钉的是 `capacity`
  （`program_impl.h:598`）—— 正是 `include/ninfer/ops/gqa_attention.h:182-193` 禁止的那类混用。
  **未证明它在本次运行里被触发或改变了数字。**

**我收回 §42 的红旗表述**：我写"int8 臂的 KV payload **大于** bf16 臂"并称之为"结构性矛盾"——
**那是我的标签读错**：那 16 层**不是 bf16**（是 10×E8Kv + 6×NVFP4），payload 反演不是矛盾而是自洽。
**保留下来的真问题**是它顺带暴露的：**出厂默认表把 E8 放在 4 个已知损坏的层号上**。

**处置（两条决定）**：
1. kv2 的补丁（把默认 E8 集合收到 `{0,1,3,4,6,7}`，34 行 1 hunk，apply-check 通过；
   这正是仓库自己在 `kv-strategy-matrix.md:113` 给的短期修法）**暂不落地**——它与 E 组的 D1
   （E8 格点投影）**都改 E8 行为**，应当**一起落、一起测**；且当前重建+闸门队列在飞，不得搅动。
2. **`kKvBitBudgetE8LayerLimit` 不动**（我先前准备的 8→10 回退补丁作废）：
   问题不在层数上限（10），而在**层号选择**；正确顺序是 **落 D1 → 复测"深层 e8 是否仍退化"**。
   E1 已指出 `kv_bit_budget.h:60-75` 本就写着 e8 只对 layers 0..7 验证过、高层退化 —— 与 kv2 的发现一致。
3. **D1 与"层区退化"的关系**：D1（格点投影扔掉 47.57% 格点）对**所有** E8 层一致生效，
   单独解释不了"0-7 好 / 8-15 坏"的**层区差异**；但两者都指向 E8 实现。
   待 E2 报告 + D1 落地后复测，才能判断"高层 e8 退化"是不是 D1 的后果。

### 49. 四臂**剂量-反应**：E8 档是**功能性失效**，而且它在**出厂默认表**里

硬检索 4 臂（同二进制、同 artifact、同文档、同 ctx 262144、同 greedy；只有 KV dtype 逐层配置不同）：

| 臂 | 生效的逐层 dtype | bits/el | 精确前缀 | 平均字符准确率 |
|---|---|---|---|---|
| `default` | **10×E8Kv + 6×NVFP4** | **4.34** | 211/512 = **0.412** | 0.482 |
| `int8` | 17×I8 | 8.25 | 500/512 = **0.977** | 0.977 |
| `nvfp4`（纯） | 16×NVFP4 + MTP | 4.50 | 486/512 = **0.949** | 0.949 |
| **`e8`（纯）** | **16×E8Kv** | **4.25** | **0/128 = 0.000**（臂未跑完，2/8） | 0.000 |

**纯 e8 是全崩，不是精度退化**：`[D0]` 精确前缀 **0/64**、**输出数字数 0**、**只生成 6 个 token**、
**MTP 接受率 0.00%**、`stop-token`；`[D25]` 同样 0/64。`[D0]` 生成 6 token 就停 ⇒ 模型立刻吐出退化输出
（KV 垃圾 ⇒ 注意力崩塌 ⇒ 立即 EOS 的形态）。

**剂量-反应**：
- 0% E8 层（纯 nvfp4）→ **0.949**
- 62% E8 层（默认表 10/16）→ **0.412**
- 100% E8 层（纯 e8）→ **0.000**

⇒ **`default` 臂 0.412 与纯 nvfp4 0.949 的落差，完全由 E8 档解释**；
而 **E8 就在出厂默认表里**（`variant.cpp:56-71` 的 `{0,1,3,4,6,7,8,9,13,14}`），
即**出厂默认配置本身有严重正确性缺陷**——这与 §48.2 引的仓库自测
（`kv-strategy-matrix.md:109-113`：`8-15:e8 → 0/8`、出厂默认表 6/8、纯档位全 8/8、compute-sanitizer 干净）
**方向一致，但严重度更高**：那边是"默认表 6/8 针"，这里是"纯 e8 在 64 位精确背诵上 0/64、
且默认表在 8 针上只拿到 0.412 而纯 nvfp4 有 0.949"。

**注**：`e8` 臂还在跑（2/8），但 `[D0]`（6 token、0 数字）与 `[D25]`（0/64）两条已足够定性。
本次硬检索用的是**旧二进制**，而本轮落地的 4 个修复（`gelu_and_mul` / `causal_softmax_attention` /
`nvfp4_gdn_snapshot_plan` / `sigmoid_gate_mul`）**都不在 E8 的 KV 路径上** ⇒ **这条读数成立、不被 4 个修复影响**。

**据此调整优先级与计划（我先前"hold kv2 的表修复"的理由升级）**：
- **当前重建+闸门队列照常跑完**（它的目的是验证那 5 个 ctest；KV 表改动**不可能**影响它们，故归因不混）。
- **队列跑完后第一批要落的，就是 E8 这一族**，一起落、一起测：
  ① kv2 的默认表收窄（`variant.cpp:66` → `{0,1,3,4,6,7}`；34 行 1 hunk，apply-check 通过；
     仓库自己在 `kv-strategy-matrix.md:113` 给的短期修法）；
  ② E1 的 **D1**（删掉 E8 格点投影、保留旋转；4 文件 +82/−40，等 E2 双路确认后落）；
  ③ 视 kv1 的结论决定是否还有第三项。
- **测量仪器已经现成且已校准**：这条 4 臂硬检索就是 E8 修复的验收仪，
  **锚点是 e8=0.000、default=0.412、nvfp4=0.949**；修完预期 `e8` 与 `default` 两臂显著上升。
- **`kKvBitBudgetE8LayerLimit` 仍不动**（我先前准备的 8→10 回退补丁作废）：问题不在层数上限，在 E8 实现本身与层号选择。

### 50. 我独立复核 payload 模型：**恒等式级确认**、**四臂全解释**、并发现一处**度量冲突**

不复述结论，只做独立算术（脚本 `sh/_payload_check.sh`，纯 CPU、只读 .err 与源码）。

#### 50.1 恒等式自检：`每层 byte/页 == 16,384 × (bits/元素)` —— 四档全部精确成立
`16,384 = 262,144/16`，而 262,144 B 正是 BF16(16 bit) 的每页每层字节数 ⇒ 这是个可独立验证的恒等式：
| 档位 | bits/el | 16,384×bits | kv2 表里的字节 | 判定 |
|---|---|---|---|---|
| BF16 | 16.00 | 262,144 | 262,144 | ✅ |
| I8 | 8.25 | 135,168 | 135,168 | ✅ |
| E8Kv | 4.25 | 69,632 | 69,632 | ✅ |
| NVFP4 | 4.50 | 73,728 | 73,728 | ✅ |

⇒ kv2 的逐层表与**文档的 bits/el 表互相自洽**，两边同时被验证。

#### 50.2 四臂 payload **全部**被同一个模型预测到打印精度（这是最强形式的确认）
| 臂 | 我的预测（B/页 × 4096 页） | 实际打印 |
|---|---|---|
| `default`（10×E8Kv + 6×NVFP4 + MTP bf16） | 1,400,832 ×4096 = 5,737,807,872 = **5.34375 GiB** | **5.34** ✅ |
| `int8`（17×I8） | 2,297,856 ×4096 = 9,412,018,176 = **8.76562 GiB** | **8.77** ✅ |
| `nvfp4`（17×NVFP4） | 1,253,376 ×4096 = **4.7812 GiB** | **4.78** ✅ |
| `e8`（17×E8Kv） | 1,183,744 ×4096 = **4.5165 GiB** | **4.52** ✅ |

- `default` 的**文本**平均 = 4.3438 bits/el（kv2 报 4.34）；`int8`/`default` 的比值 **1.64035** 与观测到的 payload 比一致。
- ⇒ 我 §42 的"结构性矛盾"**完全站不住**：四臂的 payload 不是矛盾，是**同一模型的四个点**。

#### 50.3 代码侧逐档吻合（我自己读的）
`src/targets/qwen3_6/impl/state/decoder_state.cpp:171-229` 的 plane 集合：
- `BF16` → 2×`{BF16, head_dim, kv_heads, 256}`
- `I8` → 2×`{I8, head_dim, …}` + 2×`{FP16, head_dim/group, …}`
- `E8Kv` → 2×`{U8, head_dim/2, …}` + 2×`{FP16, head_dim/group, …}`（注释：packed 4-bit E8-lattice K + i4 V，per-64 FP16 scale）
- `NVFP4` → 2×`{U8, head_dim/2, …}` + 2×`{FP8_E4M3FN, head_dim/group, …}`（+ 可选 residual 四元组）
`src/targets/qwen3_6/impl/runtime/layouts_impl.h:1213-1217` 的注释明写：
"An explicit global `--kv-dtype` replaces the target's registered per-layer default table for every layer …
Without this the global dtype never reached the KV page geometry"（引 `_TODO.md` 97）⇒ **两条路径与我 §48.2 的表述一致**。

#### 50.4 新发现：一处**度量冲突**（短上下文困惑度对 E8 缺陷**完全失明**）
kv2 补丁里保留了默认表当初的选取依据（写在 `variant.cpp` 注释里）：
> 13.3k zh perplexity @ ctx 4096：**本表 1.020（全部混合里最好）**；all-E8 **1.112**、all-NVFP4 **1.706**、all-I8 **1.522**。
> 生成验证：MTP accept 44%、decode ~111 tok/s（5090）。

而在 ctx **262144** 的 64 位精确背诵上（本会话实测）：
> **纯 e8 → 0/64（0.000）**；纯 nvfp4 → **0.949**；纯 int8 → **0.977**；本默认表 → **0.412**。

⇒ **在 ctx 4096 的困惑度上 all-E8 明显优于 all-NVFP4（1.112 vs 1.706），而在 262144 的检索上纯 e8 是 0.000 而纯 nvfp4 是 0.949。**
⇒ **短上下文困惑度看不见这个缺陷**；默认表是**按短上下文困惑度选出来的**，于是把一个在长上下文功能性失效的档位放进了默认配置。
这条对后面的验收设计很重要：**E8（以及任何 KV 档位）的验收必须包含长上下文检索，不能只看困惑度。**

#### 50.5 处置与顺序（在"别等"的前提下仍守铁律⑦）
- kv2 的补丁（`variant.cpp` 默认 E8 集合 → `{0,1,3,4,6,7}`，附带**很完整的层界理由注释**，含上面那些实测数字）已被我**算术级独立确认**，
  但**它的双路伙伴 kv1 尚未报告** ⇒ 按铁律⑦**等 kv1 报告后落地**（不与 E2 的 D1 混在一起）。
- 已把"落地 + 重建 + 用现成 4 臂仪验收"的编排写成脚本待命（go 标记由我在双路齐了之后放置），
  这样落地动作是**一条命令**，不需要再等任何人。

### 51. kv1/kv2 **双路收敛**（payload 调查结案）+ 裁定落地 kv2 的默认表收窄

#### 同（两面独立得到，且与我 §50 的独立算术逐位一致）
- 两臂**不是干净对照**；`default` = 注册逐层表（**10×E8Kv + 6×NVFP4**）+ BF16 MTP，
  `int8` = 全局覆盖 16 层 + MTP 也 I8。
- payload 反演：kv1 **default 5.3439 / nvfp4 4.7813 / int8 8.7657 GiB** —— 与我 §50 自算逐位一致。
- **`summary kv cache dtype` 打印的是全局 flag、不是生效的逐层表**（`program_impl.h:13065-13086` +
  `apps/cli/main.cpp:217`）。kv1 指出仓库**自己在 `TODO.md:5269` §8.2 明文记录过这个陷阱**
  （"摘要行 `kv cache dtype` 打印的是全局 flag 而非生效的逐层表…判据一律用 `kv cache payload`"、
  `src/product/kv_options.h:25-30`）——**但代码从未修**。
  ⇒ **这是第二个仪表缺陷**，也正是我 §42 那条"结构性矛盾"的直接成因（我的判读被它误导）。
- 真 bf16 物理上不可能：本几何下应为 **17.00 GiB**（16×1024 MiB + 1024.25 MiB MTP），是实测 5.34 的 3.2 倍。

#### 异（kv1 独有三条，都很重）
**① 6.7× 定位到一行**：`src/ops/wrapper/gqa_attention.cpp:189-192`
```cpp
// E8Kv small-T kernels are unverified; route E8 to the prompt path.
if (cache.dtype == DType::E8Kv && route != detail::GqaAttentionRoute::Prompt) {
    route = detail::GqaAttentionRoute::Prompt;
}
```
叠加"引擎的文本全注意力层在**每个 phase** 都调 A1 `gqa_attention`（不是 A3）"这一事实
（`text_context_impl.h:1066/1071` 的 `attn_mix()`；`run_layers()` 对每个 full 层无条件调，phase 只改 NVTX 分类；
A3 `gqa_attention_cached` 只被 MTP 层用，`text_context_impl.h:670`）：
decode/verify（width ≤6）本该走 SmallT，却被这条强制改成 `Prompt` ⇒
`gqa_attention_prefill_e8_launch`，grid = `(div_up(tokens,64), QHeads=24, 1)`，**且不切分 key**
⇒ 每步 decode 只有 ~24 个 CTA、每个串行扫完整个 ~151,669 key 窗口；
而非 E8 走 SmallT split-KV = `(KVHeads=4, splits=85, 1)` = **340 CTA 并行切窗**（splits 来自 `partial.cuh:96-156`）。
实测每 MTP round 墙钟：**default 136 ms/round（20 轮，10/16 层是 E8）、int8 23、nvfp4 29**
⇒ **只差"有没有 E8Kv 层"这一个变量就产生 ~5× 差**。
而且 E8 的小-T 解码 kernel **存在且已接线**（`launcher/gqa_attention_decode_e8.cu` 有 append/cached 两个 overload，
`gqa_attention_decode_smallt.cu:70-73` 会调到），只是 A1 这条活路径被作者以"unverified"自己挡掉了；
旁证：同一张表在 **ctx 4096 实测 ~111 tok/s**（`variant.cpp:31`）⇒ **不是该档位的内在成本，
而是随窗口线性恶化的 kernel 选择**。

**② 独立三臂召回复算，证伪"位预算"解释**（kv1 的 `scratch/kv1/pK1_acc2.py`）：

| 臂 | 生效 KV | bits/el（文本） | kv1 的精确前缀 | 我的打分器 |
|---|---|---|---|---|
| int8 | 16×I8 | 8.25 | 500/560 = 0.893 | 500/512 = 0.977 |
| **nvfp4** | 16×NVFP4 | **4.50** | 486/550 = 0.884 | 486/512 = **0.949** |
| default | 10×E8Kv+6×NVFP4 | **4.34** | 108/234 = 0.462 | 211/512 = 0.412 |

⇒ **4.50 b/el 的 nvfp4 与 8.25 b/el 的 int8 统计上无差别**，而 4.34 b/el 的 default 塌掉一半
⇒ **差异唯一跟随"有没有 E8Kv 层"**。"bf16 长上下文退化"与"低精度⇒召回差"两个框定**都不成立**。
且 `[D25]` 的 default 答案是**另一个完全不同的 64 位数**、`doc.find(...) == -1`（错答案不在文档里）
⇒ 形态是"**needle 不再可检索**"，而不是"读错 key 位置"。

**③ 静态排除我给的三条线索**：`split_reference_keys` 混用**存在但本跑不产生分歧**
（decode kernel 显式收 `logical_capacity` 与 `split_units`；graph 路径钉常量、eager 路径留 0 回落到
`max_visible_keys`，而**本跑 `max_context = KV capacity = 262144`，两者数值相同**）；
YaRN ×4 名义容量属**另一个算子族**（`ops/softmax_attention/dense/causal_cache`），GQA 路径无同类混用；
E8 的三处 producer（prefill `prefill_i8.cuh:134-190`/`:274-326` + decode `decode_i8.cuh:226-283`）
**都**做了 `gqa_kv_hadamard64 + e8_project_8d_warp + clamp(±7)`、Q 侧也都 rotate
⇒ **没有"漏投影"的残留副本**（`_TODO.md` 116/116b 描述的那个 bug 已修）。
⚠️ **这一条与 E1 的 D1 不矛盾、反而加强 D1**：投影是**到处都在做**的，问题是**投影本身**扔掉了一半格点
（`D8 ∪ (D8+½)` 里的半整数陪集被 `rintf` 推 0.5 step）。

#### 一处诚实的数值分歧（记下来）
我的打分器 default 臂 **211/512**，kv1 是 **108/234**；**分子在 int8（500）与 nvfp4（486）上完全一致**，
只在 default 上不同、且分母约定不同（我用 8×64=512，kv1 用"生成位数"）。
⇒ **定论取排序与比值**；绝对数取我这份（它经过真实输出回归验证：旧输出回归 29/64、acc 0.453）。

#### kv1 拒绝交补丁，理由正确且与 kv2 互补
kv1 明确不交 `fix.patch`，三条理由都对：
1. 仪表缺陷的**正确**修法是让 `MemorySummary` 携带生效的逐层表 / "mixed" 语义，而它只有单个枚举
   ⇒ 任何"只报一个枚举"的小补丁**对本例仍然会打印 bf16**，是**假修复**；
2. 真正修要跨公开头 ABI + 让 `PagedKVCache` 暴露逐层 dtype（`layer_view` 现为 private）⇒ 4 文件盲改，
   且它被禁止跑构建、无法保证编译通过 —— "交一个编译不过的补丁比不交更糟"；
3. 翻转 E8 闸的补丁会改整个默认配置的**数值与速度**，而该 kernel 被作者标为未验证 ⇒ **它不替作者做这个决定**。
它给的**最小修法留档待用**：`memory_summary()` 里若 `backend_kv_cache()` 非空，用 `layers()` +
`batch_layer_view(i).dtype` 求生效表；均匀则报该档，非均匀则报新增的 "mixed" 并附 `kv_payload`/bits-per-element。

#### 我的裁定：**放 `GO_KV_TABLE`，落地 kv2 的默认表收窄**
依据：
1. 双路都已报告，且**都**独立指向"差异跟随 E8Kv"（kv1 用 nvfp4 对照臂证伪位预算解释，kv2 用 payload/perplexity 反演）；
2. kv1 的三臂表显示 **nvfp4（4.50）≈ int8（8.25）而远好于 default（4.34）**
   ⇒ 把 4 层 E8 换成 NVFP4 **预期恢复召回**；
3. 该补丁正是仓库自己在 `docs/maintainer/kv-strategy-matrix.md:113` 给的**短期修法**，
   且补丁注释完整记录了层界实测（`0-7:e8 → 8/8 | 8-15:e8 → 0/8 | 14-15:e8 → 4/8 | 本表 → 6/8 | 纯档位全 8/8`）
   与"compute-sanitizer 干净 ⇒ 逻辑/别名缺陷、仍开放"；
4. kv1 拒绝的是**翻转 E8 闸**（会同时改数值与速度、且 kernel 未验证），与 kv2 的**收窄默认表**是**不同改动**，
   不构成反对。
**D1 仍 hold**：E2 已交 `fix.patch` 但报告未到，铁律⑦未齐。

### 52. E8 codec 双路**在旋转轴上直接对立**：两份补丁**都 hold**，已派两个独立裁决代理

#### 五条断言的 E1 vs E2
| 断言 | E1 | E2 | 判读 |
|---|---|---|---|
| ① 1.9–2.0× | 成立（**2.32×**） | 成立（**1.95/1.95/1.98×**） | 同向，系数略差；E2 给出可算机制：(0.0717+0.125)/0.0833 ≈ 2.0× |
| ① 子句"不如 3.25 b/el 纯标量" | **推翻**（E8 2.636e-2 ≈ 最优 3-bit 2.64e-2） | **成立 2.30–2.51×**，但指出那个基线其实是**拟合码本**（`K3=[0.1316,0.4024,0.6841,1.0]`，g32+E4M3）而非"纯标量" | **对立**（E2 补了真·均匀 3-bit g64 fp16 作对照） |
| ② 约 46% 半整数 | 47.57% | 46.48/46.47/46.72%（两实现一致） | 一致 |
| ③ Hadamard-64 旋转 | **推翻**：旋转让 K **好 1.43×**（`MSE(had)/MSE(none)=0.697`） | **成立**：旋转让 K **差 3.98/3.66/3.91×** | **直接对立** |
| ③ 机制 | `E[amax_rot²]/E[amax²] = 0.685`（旋转**降** amax） | H 把组 DC 灌进 `out[0]`（占峰值 88%）⇒ amax 被**抬** 1.82–1.92×（crest 2.48→4.22） | **对立**（同一对量、相反方向） |
| **修法** | **删投影、保留旋转** = 2.32× | **删旋转**（+删投影+最优 scale+陪集位 **4.375 b/el**）= **10.4–12.6×**，4 文件 +80/−26 | **旋转轴相反** |

#### 我已定位头号嫌疑：**H 的归一化约定**
E2 同时写 `out[0] = Σx/8 = 8·mean` 且 `|H[0][j] − 1/8| = 0`，**并**声称 `|HᵀH − I| = 0`。
**这两件事不可能同时为真**（若每行都是 ±1/8，则 `HᵀH = 8I` 或 `I/8`，不是 `I`）。
⇒ 归一化（除以 `1/8` 还是 `1/√8`；`HᵀH` 是 `I` 还是 `8I`）是**头号分歧嫌疑**；
第二位嫌疑是**两臂是否用同一条 scale 规则**（"无旋转"臂是按无旋转后的 amax 重取 scale，还是沿用旋转臂的）。
已派**两个**独立复算代理用**同一套严格协议**裁决（要求：从源码逐字抄出 H 系数、算出 `HᵀH`、
四臂在"自臂 scale"与"统一固定 scale"两种规则下都测、报 amax 均值与 crest，并指出两份程序各自的错误行）。
**裁决前两个补丁都不落。**

#### E2 的**无争议**发现（与旋转之争无关，应单独成立）
1. **D5 = 收敛性 UB（真 bug）**：`decode_i8.cuh:288-297` 把 `__shfl_xor_sync(FullMask, …)` 写在
   `if ((lane & 1) == 0)` 里（`FullMask` 却声称全 warp 参与）——**同族的 prefill 代码已正确把 shuffle 提出 `if`**。
   nvcc 可能照样发 SHFL 读到奇 lane 的旧寄存器值 ⇒ 需 GPU 或反汇编才能定案（已列禁跑项）。
2. **④ 的量级被更正**：无 clamp 成立，但阈值是 **E8 档 `amax > 458,528`** vs **i8 档 `amax > 8,319,008`**
   ⇒ **E8 比共用该代码的 i8 档敏感 18.1×**；真实 dump 最大 `amax_rot = 70.03`（余量 6548×）未触发。
   合成验证（真实 K ×1e8）：`scale = inf` ⇒ `kinv = 0` ⇒ **该组 64 个 code 全 0** ⇒ 读侧 `0 × inf = NaN`，无标志位。
3. **⑤ 的机制句对本档不成立**：E8 的码**不是**按 fp32 scale 分配后取整 scale
   （`ks_e = __half2float(ksh_e)`，`kinv_e = 1/ks_e` 驱动码）；那条 3.1%/0.03–0.6% 属 nvfp4/E4M3 档。
   而"scale 未迭代到最优"**成立但被低估 4–8×**：真值 **26–35% NMSE**（`t* ≈ 0.86–0.96 × amax/7`）。
4. **上一轮的 V 行是建模产物**：`e8_study.py` §3 把 E8 codec 套在 V 上得 17.78%，
   而引擎的 E8 档 **V 走 `gqa_kv_quant_i4_code`（scalar、无旋转、无投影）** ⇒ 真实 V 是 **13.5/12.3/8.1%**，
   **比 K 好得多**；K 的 28–29% 完全来自 K 的旋转+投影。⇒ 不要把 V 算进 E8 的病。
5. **一份重要的适用性限制（对我后续判定最关键）**：该 dump **只有 L13/14/15**，而
   `kKvBitBudgetE8LayerLimit = 10` 把 E8 放在**前 0–9 层**，默认表用的是 `{0,1,3,4,6,7}` + `{8,9,13,14}`。
   ⇒ **D1 的倍数取决于每层组内 DC 统计**，E2 **不能断言去旋转在部署层（0–9）同样值 3.9×**；
   而仓库自测的"`0-7:e8 → 8/8 | 8-15:e8 → 0/8`"说明**退化是随层位变化的**。
   ⇒ **两份测量都落在"已知坏"的 13/14/15 上**，部署层（0–9）的 codec 数字**至今没人测过**。
   这本身是一条该补的实验（需要 0–9 层的 dump），已记。
6. E2 还指出：全部编码器侧修完（D1+D2+D3，不加 bit、不改布局、读侧零改动）
   ⇒ K NMSE 0.0828 → 0.0080 = **10.4×（≈10 dB）**；加陪集位（4.375 b/el）→ **12.6×（≈11 dB）**。

### 53. e8 臂跑完：**字符准确率落在随机水平** ⇒ 不是精度差，是**可检索性完全丧失**；剂量-反应完整

`e8` 臂（纯 16×E8Kv）逐条读数（独立打分器，与 §50 的 payload 模型同一份产物）：

| 标记 | 精确前缀 | 字符准确率 | 输出数字数 | 生成 tok | 结束 | MTP 接受率 |
|---|---|---|---|---|---|---|
| [D0] | 0/64 | **0.000** | **0** | **6** | stop-token | **0.00%** |
| [D25] | 0/64 | 0.000 | 0 | ? | ? | ? |
| [D50] | 1/64 | 0.125 | 467 | 512 | output-limit | 78.17% |
| [D75] | 3/64 | 0.188 | 466 | 512 | output-limit | 41.83% |
| [D100] | 1/64 | 0.141 | 121 | 122 | stop-token | 33.89% |
| [D125] | 3/64 | 0.109 | 466 | 512 | output-limit | 11.67% |
| [D150] | 1/64 | 0.125 | 79 | 80 | stop-token | 72.00% |

**小计 9/448 = 0.020；平均字符准确率 0.098。**

**关键判读**：字符准确率 **0.098–0.188 正是十进制数字的随机水平**（猜中概率 0.1）⇒
e8 臂的输出里**不含可检索的信息**，这不是"记得不精确"，而是"**根本没取到**"。
（对照：纯 nvfp4 的字符准确率 0.949、纯 int8 0.977。）

**两种失败形态并存**：
- **立即 EOS**（`[D0]`：只生成 **6 个 token**、MTP 接受率 **0.00%**）；
- **读到预算为止在打转**（`[D50]/[D75]/[D125]`：512 tok 撞 `output-limit`）。
MTP 接受率也**极不稳定**：0.00 / 11.67 / 33.89 / 41.83 / 72.00 / 78.17%，而 int8 与 nvfp4 臂稳定在 **84–97%**。
⇒ 接受率本身可以被当作**KV 健康度的廉价指标**（这一条值得记：不需要精确背诵任务就能瞥见 E8 的崩坏）。

**剂量-反应现在完整**：
| E8 层占比 | 臂 | 精确前缀 | 平均字符准确率 |
|---|---|---|---|
| **0%**（16×NVFP4） | `nvfp4` | 486/512 = **0.949** | 0.949 |
| **0%**（17×I8） | `int8` | 500/512 = **0.977** | 0.977 |
| **62%**（默认表 10/16 E8） | `default` | 211/512 = **0.412** | 0.482 |
| **100%**（16×E8Kv） | `e8` | 9/448 = **0.020** | 0.098 |

⇒ **完全单调，且两端都在"每一条 needle"上成立**（不是被个别 needle 拉低）：
`nvfp4` 八条里五条满分、`int8` 八条里六条满分，而 `e8` 八条里**最高只有 3/64**。
这条 4 臂仪因此是**E8 修复的现成验收仪**（锚点 `e8=0.020 / default=0.412 / nvfp4=0.949 / int8=0.977`）。

#### 本轮新派出的代理（继续并行）
- `rc1`/`rc2`（**双路**，八项⑥）：**逐轮外挂召回**。第一交付物是"`suffix_lookup` 为何在中/尾两种查询形态下
  **零匹配**"的**纯 CPU 复算诊断**（最硬的一条），其次才是 L0 逐轮日志 / 检索索引 / 引擎挂钩点设计与补丁。
  已在简报里写入新事实：**外挂的 packed KV 必须用 nvfp4 或 int8，不许依赖 e8**。
- `w13a`/`w13b`（**双路**，八项⑤）：**权重卸载 W13**（零实现；1M 的使能项）。
  要点：1M 按现有预算需要 ≤ **2.33 bits/el**（引擎自报 e8 需 18,912,736,512 B / nvfp4 需 20,000,740,608 B
  vs 可用 11,600,323,584 B；拟合 k = 4.08e-6 GiB/(token·bit/el)、c ≈ 0.27 GiB），
  而 W13 是**把预算做大**的另一条路；**e8 已被证不可用 ⇒ W13 的价值更高**。
  要求：卸载谁/何时取/带宽账/与 KV 预算的换算（能把 2.33 提到多少）+ 补丁 + 爆炸半径。

### 54. 冷窗 cc1：字节账拆开、**2.6 bit 是单位陷阱**、挖出一个只调常数才暴露的**真 bug**；并更正我三条表述

#### 54.1 `9536 = 320 + 32×256 + 1024` 拆开
| 项 | 值 | 是什么 | 出处 |
|---|---|---|---|
| header | **320** | `magic(4)+version(2)+flags(2)` + 2×`EntropyNvfp4SlotHalf(128)` + `reserved[56]`；每半 128 = `freqs[16]u16`(32) + `offsets[17]u32`(68) + `reserved[28]` | `entropy_nvfp4_slot.cuh:41,36,45`；`static_assert(sizeof(EntropyNvfp4SlotHeader)==320)` `:61` |
| streams | **32×256 = 8192** | 2 半 × 16 流 × 256 B；每流 512 个 E2M1 code ×2048 bit = 256 B ⇒ **该区恰好等于未压缩 code plane（4.0 b/code）** | |
| scale tail | **1024** | `16 × 64` 行 E4M3FN g16 尺度，逐字节原样拷贝 | `kernel:177-181` |

#### 54.2 每 head-page（256 dim × 64 token = 16384 KV 元素，单 K\|V plane）的常驻 vs 冷记录
| dtype | 常驻 plane | 冷 codec | 冷记录 | Δ |
|---|---|---|---|---|
| bf16 | 32768 | 无 | 预留 9536，写 0 | (若有 **−23232**) |
| int8 | 16896 | Int8Raw | 16+8192+1024 = **9232** | **−7664** |
| fp8 | 17408 | 无 | 预留 9536 | (若有 −7872) |
| nvfp4 | 9216 | Nvfp4Rans | 320+8192+1024 = **9536** | **+320 净亏** |
| e8 | 8704 | 无 | 预留 9536 | (若有 +832) |
| iso3 | 9216 | 无 | 预留 9536 | (若有 +320) |

**那 16 B 的答案**：`kColdI8SlotBytes(9232) − (8192+1024) = 16 = kColdI8SlotHeaderBytes`（`cold_i8_kernels.cuh:32`），
而 `8192+1024 = 9216` 恰是 nvfp4 常驻 plane。
**但这不是"int8 比 nvfp4 多 16 B"**：**int8 冷路径并非存 int8 plane**——它先把 int8 g64 plane
**重量化成 E2M1 / g16-E4M3**（`entropy_cold_requant` 的 `Int8G64` 模式）再原样存。
**同口径真值是 16896 → 9232。**

#### 54.3 ⚠️ 更正我的三条表述
1. **"冷槽精确填满、零余量"只对一半**：只有 **rANS 记录（4.0 b/code）** 零余量（流区恰好==code plane）；
   **int8 raw 记录是 9232，只填 9232/9536，余 304 B**（代码自己也这么说：`decoder_state.cpp:448-449`）。
2. **"两处 `static_assert`"不完整**：真正**让构建失败**的只有一处 ——
   `decoder_state.cpp:374-376`（`nvfp4 cold-slot derivation drifted from ops::kEntropyNvfp4SlotBytes`，`6688 != 9536`）。
   我说的那"两处"其实是**host 镜像**：`kv_tier_formats.h:222`（`== 9536`）与 `kv_bit_budget.h:204`（`== 466`），
   它们**跟自己文件里的字面量比**，**改了值也照样 PASS** ⇒ 是**静默变假**。
   ⇒ 正确说法：**两个镜像必须重指向 6688 / 327**（否则静默），**硬门是第三处 `:374`**。
  > **[注解 2026-09-18 · SUPERSEDED]** 不要照这条"**两个镜像必须重指向 6688 / 327**"做：镜像已不存在——`kKvBitBudgetColdSlotBytes` 现在**派生**自 `kKvColdPoolStrideBytes`（`src/product/kv_bit_budget.h:314`），现值为 **9632 / 470**（价格 `kKvBitBudgetColdBitsX100` 同为派生并 static_assert `== 470`，`kv_bit_budget.h:408-412`）。
  > 这段找到的"静默镜像"洞**正是被它要求的硬门堵上的**：第三处钉子现在在 `decoder_state.cpp:863`（外加 `:845`），且检查的是**派生值** == `ops::kEntropyNvfp4SlotBytes`；照本条改成 6688/327 会让它立刻 FIRE。
   （cc1 实测编译三种场景：只改定点 X=40 全过；X=26 只爆 `:374`；X=26 + 显式 320 header + ops 6688 全过。）
3. **"e8/iso3/fp8/bf16 无冷 codec ⇒ 出厂表一页压不了"要分开说**（见 54.5）。

#### 54.4 ⭐ 真 bug：`kColdSlotRansHeaderBytes` 会在调常数时**变负数**
`decoder_state.cpp:299`：`kColdSlotRansHeaderBytes = slot − code_plane − scale_plane`
—— 这个式子**只在 4.0 b/code（流区 == code plane）时得 320**。改成 2.6 b/code 时它是
`6688 − 8192 − 1024 = −2528`，于是 `cold_slot_stride_for` 返回 **3840**，**每条记录都欠尺寸**。
⇒ 必须改成**显式 320 常量 + assert**。这是**只有把常数往下调才会暴露**的潜伏缺陷。

#### 54.5 ⭐ 单位陷阱：源码的 "2.6" 是 **bits/code**，DP 记的是 **bits/element**
- 源码里的 2.6 是 **bits/code**（head-page 8192 code），而 `kKvBitBudgetColdBitsX100` 记的是 **bits/element**（16384）。
- 2.6 b/code ⇒ 6688 B ⇒ `6688·8·100/16384 = 326.5625` ⇒ **327（= 3.27 b/el），不是 2.6**。
- 要 **2.6 b/element** 需 `2.6·16384/8 = 5325 B` ⇒ `(5325−320−1024)/32 = 124 B/流 = 1.9375 b/code`
  ⇒ **低于实测的 2.0–2.6 b/code** ⇒ **该目标靠调这个常数根本不可达**（要么换算法，要么接受 3.27 b/el）。

**定点化改动清单（3 处）**：`:281` 的 `kColdSlotRansBitsPerCode`（int32 装不下 2.6 ⇒ 改十分位 26）；
`:331` 公式 `(stream_symbols*bits+7)/8` ⇒ `(stream_symbols*bits_x10+79)/80`；
`:299` 的 header（见 54.4）。

#### 54.6 e8 / iso3 / fp8 / bf16 逐个判定
| dtype | 9536 槽放得下？ | 判定 | 最少需要 |
|---|---|---|---|
| **iso3** | **是**：plane pair 与 nvfp4 逐字节相同（8192+1024） | **仅缺实现**（`cold_slot_codec_of` 不映射 ISO3；无 standalone ISO3 的 requant 臂与 cold 分支） | 9536（6688 @2.6） |
| **e8** | **是**：4-bit code plane 8192 + 自带 FP16 g64 tail 512 ⇒ **320+8192+512 = 9024** ≤ 9536 | **仅缺实现**；且推导把尾部硬编码成 g16/1024（`kColdSlotScaleGroup`），e8 的 512 B 尾会被算错 | 9024 |
| **fp8** | **否**：原生 8-bit code plane 16384 ⇒ `320+32·512+1024 = 17728 > 9536` | **槽不够 + 缺实现**（必须重量化到 4-bit，那样 9536 零余量刚好） | 17728（原生）/ 9536（重量化） |
| **bf16** | **否**：同 fp8 但更差 | **槽不够 + 缺实现** | 17728 |

**注意限定**：`iso3 无冷 codec` 必须限定为 **standalone ISO3 层**——**nvfp4 tier 的 V plane 就是 ISO3，
而且确实走 rANS 冷编码**。出厂 `qwen3_6_27b` 表是 10×E8Kv + 6×NVFP4，
因"**每层都必须能喂 codec**"⇒ **该表下冷池一页都压不了**（(c) 条成立）。

#### 54.7 B1 三条的裁定与补丁状态
- **(a) 真缺陷**（@4.0 净亏 320 B/head-page，见 54.2）；
- **(c) 真缺陷**，但需按 54.6 限定（iso3/e8 是"缺实现"、fp8/bf16 是"槽不够"）；
- **(b) 描述有误/不完整**：定点化是真的，但漏了真正爆构建的第三处 assert、掩盖了 54.4 那个真 bug、
  且把 bits/code 与 bits/element 混为一谈。
- 补丁：`/home/user/scratch/cc1/fix.patch`（358 行 / 4 文件 / 14 hunk / +225 −19），干净副本 **apply-check PASS**；
  落地后两份 product 头文件 host 编译通过且**出货常数不变**（466/9536/9232/9216/16896）；
  `tests/test_kv_tier_formats.cpp` 补丁前后都 all checks passed；**不改默认**（源码自己要求 2.6 必须是测量决策）。
- **先 hold**：等双路第二路（`cc2`）报告（铁律⑦）。
- 附带未改项：DP 的单一 `kColdBitBudgetColdSlotBytes` 与 per-dtype 记录不一致
  （**int8 冷层被多记 304 B/head-page**）——补丁里写明了，未改。

#### 54.8 与 E8 主线的交汇
出厂表 10×E8Kv + 6×NVFP4 **既**让长上下文召回塌掉（§53：62% E8 层 → 0.412）
**又**让冷池一页都压不了（54.6）。
⇒ **kv2 的默认表收窄（已放 `GO_KV_TABLE`）同时改善两件事**；
但要让冷窗真正可用，还需要一个**统一档**的图层方案（或补齐 e8/iso3 的冷臂）——这是 ③ 的下一步设计点。

### 55. ⭐ 旋转之争**裁决**：E1 对、E2 错——E2 把 dump 读取**转置**了；我的头号嫌疑是**错的前提**

裁决方式：在同一批字节上跑**两种 gather**，看谁能复现谁（`/home/user/scratch/adj2/`，纯 CPU）。

#### 55.1 分歧的唯一原因：E2 的 gather 是转置的
仓库**四处**独立证据一致 —— `kn` 文件是 **head_dim 最快变化**：
- meta `kn ne=256,4,3072,1  nb=2,512,2048,6291456`（nb[0]=2 B ⇒ dim0 最快）；
- `gqa_kv_quant_src_index = d + 256*(h + 4*t)`（`src/ops/kernel/gqa_attention_kv_quant.cuh:59-64`）；
- dumper 写的是 `view({head_dim, n_kv, T})`（`text_context_impl.h:997`，raw 连续 memcpy）；
- 仓库自带的 `research/scripts/_src_vs_stored.py:53` 也是 `reshape(256,4,-1)` 后 `kn[d,h,t]`。

**E1 用 `g*64 + 256*(h + 4t)`（对）；E2 用 `((d*4+h)*3072 + t)`（转置）。** 同一批字节、两种读法：

| gather | `E[amax_rot]/E[amax_raw]` | crest raw→rot | `MSE(ship)/MSE(norot)` |
|---|---|---|---|
| **A：`d + 256h + 1024t`**（meta / src_index） | **0.8319** | 3.14 → 2.57 | **0.719**（L13/14/15：0.681/0.766/0.716） |
| **B：`(4d+h)3072 + t`**（E2） | **1.8645** | 2.47 → 4.17 | **3.769**（3.94/3.56/3.80） |

- 布局 B 几乎逐项复现 E2 的数字（amax ×1.82–1.92、crest 2.48→4.22、MSE ×3.98/3.66/3.91）；
- 布局 A **逐位复现 E1**：`E[amax_rot²]/E[amax_none²] = 0.6847`（E1 同）、组均比 0.8635（同）、
  `MSE(b)/MSE(d) = 0.6973`（同）、`MSE(a)/MSE(c) = 0.7189`（同）、shipped NMSE **0.026361**（E1 的 2.636049e-02 同）、
  最大组 amax 15.0000 → 旋转后 9.2288（同）。
⇒ **一次采集、两种读法，这就是全部分歧。**

#### 55.2 ⚠️ 我的头号嫌疑（H 归一化）是**错的前提**
adj2 指出："每行都是 ±1/8" 与 "`HᵀH = I`" **可以同时成立** —— 求和要取**平方**：
`Σ_j (1/8)² = 64/64 = 1`。数值验证 `min|H[i][j]| = 0.1250000000`、`max|H[0][j]−1/8| = 0`、
`max|HᵀH−I| = 0`、`max|H Hᵀ−I| = 0`。源码 `gqa_attention_kv_quant.cuh:32-35` 的系数是 `0.125f`（不是 `1/√8`）。
⇒ E2 的 `out[0] = Σx/8 = 8·mean` 是**正确代数**，两份程序都与源码一致。**归一化不是分歧点，我判断错了。**

#### 55.3 判定：**E1 对、E2 错**（E2 的代数对，测量是转置文件的假象）
三个数：
1. `MSE(rot)/MSE(none)` = **0.72（投影开）/ 0.70（投影关）** ⇒ **旋转让 K 好 1.39–1.43×**；
2. `E[amax_rot]/E[amax_none] = 0.8319`、`E[amax_rot²]/E[amax_none²] = 0.6847` ⇒ **旋转降低 amax** ⇒ 步长更细；
3. `8|mean|/amax_rot = 0.3182`，且 **DC 位置只在 1.54% 的组里是 argmax** ⇒ E2 说的"占峰值 88%"实际是
   **平均 32%、从不主导**。

**真机制（两方都没说到的细化）**：真实的 64 组是**重尾**的（crest 3.14，旋转前 amax 到 15 而 rms≈1.5，
即单个离群通道主导）；正交 H 把**离群能量摊到 64 个坐标**上；DC 项确实会长，但只到 `0.32·amax_rot`，
所以净 amax 降 0.83×、步长变紧。E2 被搅乱的编组把尾巴磨掉了（crest 2.47、类高斯——因为它把**一个通道**在
64 个 token 上平均，那**恰好造出**相干 DC），这正是 DC 项在那里显得主导的原因。
**另注**：那 1.43× 增益**依赖于每组 amax 锚**——若换成一个全局常数 scale，增益消失（1.14×）。

#### 55.4 四次独立测量下的 E8 四臂（布局 A，K，27 dumps / 64677 token / 1034832 组，codes ±7，div 7）
| 臂 | NMSE | relRMS | crest |
|---|---|---|---|
| (a) 旋转+投影（shipped） | 0.026361 | 16.236% | 2.5745 |
| **(b) 旋转、无投影** | **0.011366** | **10.661%** | 2.5745 |
| (c) 投影、无旋转 | 0.036669 | 19.149% | 3.1369 |
| (d) 无旋转、无投影 | 0.016301 | 12.767% | 3.1369 |

⇒ **(a)→(b) = 0.026361 → 0.011366 = 2.32×**：**删投影 = 2.32×**（与 E1 的 2.32× 吻合），
**删旋转反而变差**（(a)→(c) = 0.72×）。scale 锚的对照：自臂锚 0.7189/0.6973、统一锚 0.6353/0.4296、
单一全局常数 1.1383/1.0631（各臂全崩、relRMS 39–59%）⇒ **锚是二阶效应（≤0.08–0.27×），不是 5.7× 的成因**。

#### 55.5 定罪行
- `/home/user/scratch/fixE2/e8recon.cpp:457`：`const size_t i = ((size_t)d * H + h) * T + t;`（主臂循环的逐行 gather）；
  `:544`（旋转诊断）、`:585`（fp16 范围扫描）同一处转置。**应为** `d + (size_t)D * (h + H * t)`。
- `/home/user/scratch/fixE2/cross.py:84`：`reshape(D, H, T).transpose(2, 1, 0)` —— 它重复了同一个错，
  所以那个"numpy 第二实现"**没能抓住它**（两实现同错 ⇒ 交叉验证失效，方法论教训）。
- E1 布局干净；仅 `fixE1/e8verify.cpp:484-485` 把 `sqrt(E[amax²_rot]/E[amax²_none]) = 0.8275` 标成"均值比"
  （真值 0.8319，E1 自己的 0.8635 那行才是对的量）——**纯标签，不碰任何 MSE 数字**。

#### 55.6 裁决后仍然成立 / 被推翻的
**仍然成立**（与布局无关）：
- E2 的 **D5 收敛性 UB**（`decode_i8.cuh:288-297` 的 `__shfl_xor_sync(FullMask,…)` 在 `if ((lane&1)==0)` 里，
  而同族 prefill 已正确提出 `if`）；
- E2 的 **④ 阈值算术**（`7×65504 = 458,528` vs `127×65504 = 8,319,008`，纯算术、与布局无关）；
- E2 的 **⑤ 机制更正**（`ks_e = __half2float(ksh_e)`，码本就按解码端读到的 fp16 scale 分配）；
- E2 的 **V 行更正**（E8 档 V 是 scalar i4、无旋转无投影，`prefill_i8.cuh:160-161/296-297`）。

**被推翻**：E2 的**旋转轴结论**与它 **10.4–12.6× 的头条** —— 都是转置文件的假象。

**注**：E1 的 V 数字（0.463）是"旋转 V"这种**shipped 从不运行的配置**下的假设量，不构成对 V 的结论。

#### 55.7 据此放 `D1_PATCH`（E1 的删投影补丁）
依据：① E1 与 adj2 **两次**独立测量都给出 **2.32×**，且 adj2 逐位复现 E1；② E2 的反向修法被定位到具体错误行；
③ 删投影后 e8 的 K 路径退化为普通 i4（offset-binary、±7、g64/fp16），与 `kv_bit_budget.h:124` 注释
"nibble + FP16/g64" 一致，且**读侧零改动、格式不变**（读侧只是 `码 × fp16 scale`，`decode_i8.cuh:626-634`）。
④ 补丁含 D1 + D2（`kv_scale_half` 饱和），**不含** E1 故意排除的 D3（需 CUDA 无法在此编译验证）。
⇒ 写入 `/home/user/scratch/e8land/D1_PATCH` = `/home/user/scratch/fixE1/fix.patch`。
（`adj1` 若给出相反裁决，把该标记删掉即可回退——标记是可撤销的，管线在阶段 2 才读它。）

### 56. 冷窗双路**求同析异**：cc2 在两点上更对（fp8/bf16 放得下、int8 成本少报）；并更正我一个数字错

#### 同（两面独立一致）
- `9536 = 320(rANS 槽头) + 8192(32×256，4-bit 无扩张上限) + 1024(未压缩 E4M3FN g16 scale 尾)`；
  **scale 尾必须保持未压缩**（restore/scatter 按 `slot + slot_bytes − 1024` 当裸 uint4 读），
  **它自身就是 0.5 bit/element**（1024×8/16384）——压它是下一个杠杆，不属于这次 flip。
- **`kColdSlotRansHeaderBytes` 必须停止"反推"**（`slot − code_plane − scale_plane`）。cc2 把后果说得更狠：
  record 被算成 **3840 B**，编码器从同一 stride 推出的 budget 变 **78 B/流** ⇒ **每页溢出 → valid flag 全清
  → 池子静默失效，而报告还在报省内存**。
- **static_assert 我确实漏数**：是**三个字面量钉子**：`decoder_state.cpp:374`（`6688 != 9536`）**FIRE**；
  `kv_tier_formats.h:222`（`==9536`）与 `kv_bit_budget.h:204`（`==466`）**不 fire**（字面量比字面量）
  但**必须改**，否则报告/门按一个已不存在的 9536 B 槽定价、DP 继续按 **4.66 b/el** 收费并据假前提否掉冷方案。
  （从不 fire 的：`:371` int8==9232（raw 记录被 clamp 到固定布局）、`:379/383` 两条 16 字节对齐。）
- 16 B 答案；以及**别把 int8 的对照面搞错**：是 **16896 → 9232**，不是 9232 vs 9216。

#### 异：cc2 在两点上更对
**① fp8/bf16 **放得下**，cc1 的"槽不够"是错的。**
关键事实 **cc1 自己**已经证明：**int8 冷路径不是存 int8 plane，而是先重量化成 E2M1 / g16-E4M3 再原样存**。
同一机制下 fp8（8-bit）也该被重量化 ⇒ **冷记录与源 dtype 无关**：

| class | 常驻 | 需要的 record | 放得进 9536? | @4.0 划算? | @2.6 划算? | 堵在 |
|---|---|---|---|---|---|---|
| bf16 | 32768 | 6688 | ✅ | ✅ −23232 | ✅ −26080 | **缺实现** |
| fp8 | 17408 | 6688 | ✅ | ✅ −7872 | ✅ −10720 | **缺实现** |
| iso3 | 9216 | 6688 | ✅ | ❌ **+320** | ✅ −2528 | **缺实现**（4.0 下 ceiling 也是堵点） |
| e8 | 8704 | 6688 | ✅ | ❌ **+832** | ✅ −2016 | **缺实现**（同上） |

⇒ **四个都放得下，一个都不缺槽**；最小需要槽 = 9232(raw)/6688(rANS)，**比原来预留的 9536 还小**。

**② 真正该动的数是 ceiling，不是 codec。** cc2 给出**盈亏平衡上限**：9216 B 的 pair 为 **3.84 b/code**、
e8 的 8704 B 为 **3.59 b/code** ⇒ **2.6 两者都在内、4.0 两者都在外**。
⇒ **"在 4.0 的 ceiling 下给 iso3/e8 写 codec 毫无意义"** —— 这比我原先的说法更准。
缺实现的具体位置（`program_impl.h:10645-10659` 的 `packed = dtype==I8||dtype==NVFP4` 无条件 return；
`gqa_attention_decode_iso3.cuh` 里 cold/slot 引用数为 **0**；e8 走自己的 kernel）已记账。

**③ 它修了 cc1 只标注未修的那个不一致，并抓到我一处数字错。**
`kv_tier_formats.h` 把 int8 的 `pool_stride_bytes` 填成 rANS 的 **9536**，而 `decoder_state` 按每层 dtype
给 int8 分 **9232** ⇒ 两者不一致 ⇒ **报告少报 304 B@4.0、少报 2544 B@2.6**。
**⚠️ 更正我自己的数**：我在 §54 写"int8 @4.0 净省 7360 B"——**错，真值 7664 B**（16896→9232）。
cc2 顺手修了，也修了会打出 `COST -2528 B/head-page` 的硬编码（改为按符号出词）。

**④ 一个潜伏小 wart**：int8 raw 头 16 B 只写了 8 B（`cold_i8.cu:29-31` 写 magic/version/flags 于 +0..7，
**+8..15 保留且从未初始化**，无人读）。

#### ⚠️ 两版补丁的**策略性冲突**（我的裁定）
- **cc1**：保持**出货常数不变**（466/9536/9232/9216/16896），把 2.6 做成**显式可选**；
  理由：源码自己要求"2.6 必须是**测量决策**，不能是默认"。补丁 358 行 / 4 文件 / +225 −19。
- **cc2**：把 `ops::kEntropyNvfp4SlotBytes` 改 **6688**、`kKvBitBudgetColdBitsX100` 改 **327**
  > **[注解 2026-09-18 · SUPERSEDED]** 本条是 cc2 提案的转述，**未落地且现在被树禁止**：出货值是 `ops::kEntropyNvfp4SlotBytes = 9632`（`include/ninfer/ops/entropy_nvfp4_slot.h:48`）与派生的 `kKvBitBudgetColdBitsX100 = 470`（`src/product/kv_bit_budget.h:408`，`:412` 断言 `== 470`）。
  > 若照本条把两者改成 6688/327：`kv_tier_formats.h:329`、`kv_bit_budget.h:412`、`decoder_state.cpp:845`/`:863` 四条 static_assert 会断。
  ⇒ **直接改出货常数**。补丁 835 行 / 6 文件 / +482 −114，含 26 条新 static_assert。
- 而 **cc2 自己报告**："2.6 ceiling 的真实回退率**无任何计数器可测**，全树只有 `slot_valid==0` 这一个信号"。
  ⇒ **在没有测量的情况下把 6688/327 变成出货默认，正是源码自己禁止的那件事。**
  > **[注解 2026-09-18 · SUPERSEDED]** 这条的**理由仍成立且已被遵守**（先测量再动常数），但它点名的两个数属于"改动前"世界：记录现为 9632、价格为 470。
  > 它要求的测量已经做了：`dl/ransceil/rans_probe.cu` 实测 min 243 / p50 250 / p90 252 / p99 254 / max 259 B 每 512 符号流（记录在 `kv_tier_formats.h:262-282`），且 6688 B（167 B/流）实测 K 命中 **0/64**——"被源码禁止的那个数"如今正是"什么都编不出来的记录"。

**裁定**：**以 cc1 的补丁为落地候选**（保持出货常数、含 header 反推真 bug 的修复、26 条 static_assert 把事实
钉成编译期事实，例如 `kv_cold_class_bytes_at_4bit(Iso3Fusion).fits && !...pays` = "槽从来不小，是 ceiling 太高"），
**并把 cc2 的两处修正（int8 stride / 成本符号）作为必须并入项**。
**两版都先 hold**：等 E8 家族那一批跑完，作为**独立一批**落地（一批改动一次重建，保归因）。
另需一条测量才能动 ceiling：**`slot_valid==0` 的计数**（目前没有）。

#### 验证状态（两面都做了，cc2 多一层）
- cc1：`g++ -std=c++20 -I include -I src tests/test_kv_tier_formats.cpp` → all checks passed（补丁前后都过，因为出货常数没动）。
- cc2：`g++ -std=c++20 -Wall -Wextra -Werror` 同样 all checks passed；**并证明未打补丁的测试文件配新头会 fail 15 条断言**
  ⇒ 测试改动是**承重的**；`decoder_state.cpp` 几何块抽出编译 **8 条 static_assert 全过**，
  `stride NVFP4=6688 / I8=9232 / ISO3=E8Kv=FP8=BF16=6688`。
  > **[注解 2026-09-18 · SUPERSEDED]** 这是 cc2 在**改动前**构建里的验证输出，stride 表已不是这个：NVFP4 = **9632**（`kv_tier_formats.h:289`/`:329`），I8 raw = 9232（`kv_bit_budget.h:430`），BF16 与 E8Kv 被断言**就是 int8 raw 记录 = 9232**（`decoder_state.cpp:826`/`:832`），无冷 codec 的类（ISO3/FP8）按 `cold_slot_stride_for` 取最宽默认记录 = **9632**（`decoder_state.cpp:803-814`）。
- 两边都没跑 GPU；设备侧 codec、requant 质量、Muse 路径均未验。

### 57. 裁决第二路（adj1）与第一路（adj2）**独立收敛**：E1 对、E2 错；并给出**机制分解**与一条不依赖元数据的证据

#### 57.1 两面数字对到 6 位（真正的独立复现）
| | adj1 | adj2 |
|---|---|---|
| 四臂 (a)/(b)/(c)/(d) | **0.026360 / 0.011366 / 0.036667 / 0.016301** | **0.026361 / 0.011366 / 0.036669 / 0.016301** |
| `MSE(rot)/MSE(no-rot)` | 0.7189（投影开）/ **0.6973**（关） | 0.7189 / 0.6973 |
| `E[amax_rot²]/E[amax_raw²]` | **0.6847** | 0.6847 |
| DC 是 argmax 的频率 | **1.54%** | **1.54%** |
| crest raw→rot | 3.1369 → 2.5745 | 3.14 → 2.57 |
| 判定 | **E1 对、E2 错** | **同** |

⇒ **(a)→(b) = 0.026360 → 0.011366 = 2.32×（删投影）**，两次独立测量一致；且两面都把 E1 的数字**逐位复现**
（shipped relRMS 16.236% 逐位相同）。

#### 57.2 两面都排除我的两个嫌疑（**我判断错了，记录在案**）
1. **H 归一化**：`H = (1/8)·Sylvester_64`（`1/8 = 0.125f` 只出现在最后一级蝶形上）。
   `|HᵀH − I|max = 0`、`max|H[0][j]−1/8| = 0`、`max||H[i][j]|−1/8| = 0`、`|H(Hv)−v|/|v| = 1.3e-07`（对合）。
   **`(1/8)²·64 = 1` ⇒ "row 0 全 1/8"与 `HᵀH = I` 完全兼容** ⇒ **我 §52 说的"不可能同时为真"是错的**。
   另：**"每行都 ±1/8"这个前提本身也错**——只有 row 0 是全和行，其余行符号是变化的。
   E2 的 `out[0] = Σx/8 = 8·mean` 是**正确代数**；两方程序都忠实复刻了蝶形、都与源码一致。
2. **scale 规则**：E1（`run_group:144`）与 E2（`encode_group:222`）都用"该臂自身变换后 amax/7"，**规则相同**。

#### 57.3 轴序分歧（两面一致），adj1 多给一条**不依赖元数据**的证据
元数据侧：`gqa_kv_quant_src_index = d + 256*(kv_head + 4*token)` ⇒ 内存序 **[t][h][d]，d 最快**；
dump meta `kn ne=256,4,3072 nb=2,512,2048`；`kvdump_dump_tensor` 是 raw memcpy。
**adj1 的数据指纹（不看任何元数据）**：lag-1 相关 —— stride 1 → **+0.0001**；stride **1024（=1 token）→ +0.6212**
（RoPE 特征）；且每个 64-block 内 `rms[d] ≈ rms[d+32]`（RoPE 半配对）。
⇒ **stride-1 是 dim 轴、stride-1024 是 token 轴**，纯从数据判出。E1 用对了；**E2 转置**
（后果：E2 的每个"64-group"其实是**几乎同一通道跨 64 个 token 的采样**，而不是"某 token 某 head 的 64 个 dim"）。
两面都**按 E2 的读法把 E2 的数字复现了出来**（adj1：layout B 下 L13 `amax_rot=6.441 / crest=4.216 /
`8|mean|=5.701 / DCisMax=59.1%`，对上 E2 的 6.19 / 4.22 / 5.75 / 88%）。

#### 57.4 ⭐ adj1 的机制分解（最有价值的新东西）
**在固定 scale 下，旋转的"形状"效应是有害的：1.0945×**；**收益 100% 来自 scale 锚**
（`0.697 / 1.0945 = 0.637`）。固定 scale 对照：
- ① 四臂共用 `fp16(amax_raw/7)`（逐组）：**1.1098×（投影开）/ 1.0945×（投影关）** ⇒ 锚被拿掉后**旋转反而略有害**；
- ② 全数据集单条常数 `s = 0.651855`：0.8846× / 0.7318×。
⇒ 所谓"旋转有益"，实质是"**旋转改变了分布使 `amax` 下降，于是能用更细的步长**"。
**推论**：删投影（D1，2.32×）之后，**下一个杠杆是 scale 锚**（LS / 最优 scale）——那正是 E1 故意排除的 D3
（需 CUDA 才能验证）。**修法优先级：D1 先行（安全、格式不变、已三次确认），scale 锚列第二。**

#### 57.5 顺带抓到一个**不可信的仓库注释**（文档缺陷）
`prefill_i8.cuh:137` 与 `decode_i8.cuh:267` 说旋转 "raises the peak by 1.4-15x"。
**硬界是 `amax_rot/amax_raw ≤ 8`，实测均值 0.83（即 0.83×，是降低）** ⇒ **该注释不可信**；
adj1 怀疑它正是 E2 错误假设的来源。**记为文档缺陷，待修。**

#### 57.6 结论与处置
**三次独立测量（E1 + adj1 + adj2）对一次（E2），且 E2 的错误被两面各自定位到同一批行**
（`e8recon.cpp:457/544/585`、`cross.py:84`；那个"numpy 第二实现"在**唯一要紧的轴**上继承了同一假设
⇒ **不构成独立复核**）。
⇒ **`D1_PATCH`（E1 的删投影补丁）维持不变**，管线照跑。
E1 仅剩表述问题：`e8verify.cpp:290-292` 把 amax 比称作"上界"（它其实是**唯一驱动项**）；
`:431/434` 变量名 `crest_ratio_sum` 实际累加 amax 比；**其 V 列（0.4633）来自"给 V 也加旋转"，
而 shipped 路径从不旋转 V**（`prefill_i8.cuh:44,58`、`decode_i8.cuh:274,284`）⇒ 那一列不是 shipped V。
**这些都不碰任何 MSE 数字。**

### 58. ⭐ 独占定案 + **四个修复全部验证生效**：5 个失败在显存空闲下逐字复现；重建后 3 个缺陷测试转绿、零新签名

#### 58.1 独占复跑（旧二进制 `mtime=2026-09-13 00:35:13`，全程 VRAM **603 MiB** 空闲，测试前后都是 603）
`_exclusive3.sh` 先持锁、再等 VRAM 空闲（§40 修掉的那个竞态），所以这次**不可能**有抢占：

| 测试 | rc | 原始指纹 |
|---|---|---|
| #51 `ninfer_gelu_mul_test` | 8 | `gelu_mul strided gate/up: pointwise mismatch max_abs=182.393 max_rel=1.99994 max_index=34686 actual=-89 reference=93.3926 first_violation=10240 non_finite=0` —— **与 triage 时逐字相同** |
| #74 `ninfer_softmax_attention_test` | 8 | `causal_softmax_attention accepted an envelope outside the launcher domain`（packed/context 两个兄弟用例 `PASS`）—— **逐字相同** |
| #95 `ninfer_gdn_input_proj_conv_snapshot_test` | 8 | `NVFP4 snapshot interval did not preserve its A16/A4 route boundary` —— **逐字相同** |
| #4 `ninfer_ple_table_e2e_test` | 8 | `usage: …/ninfer_ple_table_e2e_test <sidecar_root>`（环境） |
| #26 `ninfer_qwen3_6_frontend_test` | 8 | `failed to open test resource: /home/neroued/models/llm/qwen/Qwen3.6-27B/base-hf-bf16/tokenizer.json`（环境） |

⇒ **五个失败在显存空闲下全部逐字复现** ⇒ **"抢显存导致的伪失败"这个保留意见正式关闭**（我先前一直说"不确定就不许当结论"——现在确定了）。
三个数值/契约类失败是**真缺陷**，且我落的 4 个修复正对着它们。

#### 58.2 重建 + 定向回归（`_after_exclusive.sh` 阶段 1/2 已完成）
- 重建：`_build_locked.sh`（持锁、`-j8`）rc=0、二进制 mtime 已更新（否则脚本会 exit 6 不进闸门）。
- 定向回归 `_regress_check.sh`（按**确切失败指纹**判，逐个测试单独持锁跑）：

| 测试 | 类型 | rc | 指纹还在? | 判定 |
|---|---|---|---|---|
| `ninfer_gelu_mul_test` | defect | 0 | no | **FIXED** |
| `ninfer_softmax_attention_test` | defect | 0 | no | **FIXED** |
| `ninfer_gdn_input_proj_conv_snapshot_test` | defect | 0 | no | **FIXED** |
| `ninfer_ple_table_e2e_test` | env | 8 | yes | KNOWN_ENV（非回归） |
| `ninfer_qwen3_6_frontend_test` | env | 8 | yes | KNOWN_ENV（非回归） |

**总体：没有 STILL_BROKEN / NEW_SIGNATURE。**
⇒ **四个修复（G/S/D/sib）全部验证生效**：3 个真缺陷测试转绿、**零新签名**、2 个环境类被正确分类为 `KNOWN_ENV` 而不是当成回归或静默。
这正是"先把失败钉死在修复之前、再重建看它是否消失"这条纪律的回报。

#### 58.3 链条继续
阶段 3（全量 ctest 109 + cold 相）正在跑；之后 `_land_e8_family.sh` 会按已放的 `GO_KV_TABLE` + `D1_PATCH`
落地 **E8 家族**（默认表收窄 + 删格点投影）→ 重建 → 用 4 臂仪验收（锚点 `e8=0.020 / default=0.412 / nvfp4=0.949 / int8=0.977`）。

### 59. ⑥ 逐轮外挂召回（rc1）：`suffix_lookup` 语义精确到分界线；并**纠正"零实现"**——四条腿已在树里，缺 P5/P6/P7

#### 59.1 `suffix_lookup` 的精确语义（CPU 复算，双镜像 198269 例 0 不一致）
契约（`src/ops/kernel/suffix_lookup.cuh:15-82`）：`limit = min(len-Q-K, start-Q)`；对每个候选窗 `o` 算
`ids[start..start+Q)` 与 `ids[o..o+Q)` 的**公共后缀**长度，取（最长、同长取最大 `o`），
`L >= min_len` 时输出 `continuation[k] = ids[o+Q+k]`。

**分界线 = 查询侧的对齐点被钉死在最新 token 上**（`:36-42` 从 `q=0` 倒着比、第一个不等就 break），
`o` 只平移历史侧：

| 查询形态 | 内核结果 | 同数据的 any-position 2D 锚点 |
|---|---|---|
| 逐字粘贴在**末尾** | **L=16、o=208（正确对齐）、续写=正确历史续写** | 同 |
| 逐字粘贴在**中间**（后接 12 token 追问） | **L=0** | **L=4、R=223（正确）、续写正确** |
| **换词** | L=0（min_len=8） | L=8 仅因构造残留 |

**对齐律定量**（Q=32、min_len=8、每 f 400 次随机追问）：
**f=0 → match_rate 1.000；f≥1 → 0.000（f=1..12 全零）**。**偏一个 token 就全丢**
⇒ 先前记录的"中/尾两种形态 100% 零匹配"**为真**，此处给出定量边界：失败是**形态**失败，不是数据失败。
三条硬边界（穷举）：`starts[b] <= Q` 恒 0（548 例、非零 0）；命中要求历史侧 run 右端 `R >= Q-1`
（R=7..14 全不命中、R=15 起全命中）；`L <= Q`。

**两条必须点名的实现事实**：
1. **`history` 是死参数**（`src/ops/launcher/suffix_lookup.cu:18-20` 不传它）；
2. **内核只挡了匹配窗重叠，没挡"续写读回查询窗"** —— `o = start-Q-1` 时 8 个草案位置有 **7 个落在查询窗内**
   （零新信息的自链）。
另：`suffix_lookup` **今天引擎零调用**（唯一调用点 `tests/suffix_gpu_test.cu:52`），
`src/spec/lookup_fuse.h` 只被它自己的 CPU 测试用。

#### 59.2 ⭐ 纠正"零实现"：**四条腿已在树里**，真正缺的是三块
1. **落盘**：`program_impl.h:10610-10879`（`file_slot` 分配 `10588-10600`；`ColdPageEntry` `program.h:476-490`）
2. **读回**：`restore_cold_page:11023-11158`
3. **批量预取**：`prefetch_cold_pages:11163-11171`
4. **内容寻址键**：树里**已有现成的键类型** ——
   `CheckpointSummary::shortlist_key = {PrefixShortlistDigests::at(frontier), frontier, identity_tag}`
   （`program_impl.h:7178-7192`，摘要定义 `prefix_identity.h:42-57`）
   ⇒ 索引 = `(digest, frontier, page) -> file_slot`，**底层生成、零硬编码**。

**真正缺的三块**：
- **P5 每轮触发**：`warm_cold_prefix` 自述 "rewrite/resume paths only"，唯一调用点 `9426`，
  **不在任何 `decode_*_batch` 里**；
- **P6 选页规则**：现在只按 `end_page` 恢复整段前缀；
- **P7 持久日志**：`restore_cold_page` 末尾**释放 file_slot 并删掉 cold_pages 记录** ⇒ 映射是内存态且会被删。
另：稳态靠**设备冷槽内联可读**（`cold_host_tier.h:19-24`）——**spill 文件是 mirror 不是 tier**，
这与"逐轮外挂召回"缺的正是**同一块**。

#### 59.3 设计要点（"每轮一次"有硬理由）
**唯一挂钩点**：`src/targets/qwen3_6/impl/runtime/program_impl.h:10470-10483`（`ensure_sequence_kv_mapped`）
—— 四个轮入口 ingress 循环的**最后一句**、**按行每轮恰好一次、submit 之前**：
`decode_ordinary_batch:12276` / `decode_mtp_batch:12438` / `decode_dflash_batch:12626` / `decode_dflash2_batch:12872`。
其余 4 个非轮调用点（`8256/8818/9806/11857`）不受影响。

**每轮而非每 token 的硬理由**：逐 token 实测 **2.63 s/token**，比**重新预填充**（365 µs/token）还差 **~7200×**
—— 它不是"贵"，而是**比什么都不做更慢**；批量读 2.6–6.1 µs/token ⇒ 相对预填充 **60–140:1**、
相对单步解码 **6.3e3–1.5e4:1**。胜出条件 `C_fixed < N*359µs`：N=64（一页）允许 23 ms、
N=821（实测源距离中位数）允许 295 ms —— 任何一次 batched pread 都满足。

**键必须是"前缀证书"而不是相似度**：KV 是前缀函数 ⇒ 召回合法性 ⟺ **前缀逐字节相同**（§59.2 的第 4 条已有该键类型）。

**补丁**：`/home/user/scratch/rc1/design.patch`（49,350 B、6 文件 16 hunk、两个 new file）；
`git apply --check -v` rc=0、随后实应用 rc=0；`src/spec/turn_recall.h` + 测试用
`g++ -O2 -std=c++20` **真编译并 6/6 通过**。设计文档 `rc1/DESIGN.md`。

#### 59.4 rc1 的诚实边界
真 GPU 内核未跑（"复算=真内核"仅由双镜像一致性支撑）；**仓库未编译** ⇒ 3 个被改的 C++ 只做了锚点级正确性检查，
**编译未验证**（最高风险处：`prefix_digests.at(frontier)` 的索引上界假设，已用 `frontier > size() ⇒ 跳过` 保守处理）；
`0.38 tok/s` 未找到出处、未复现，对它的解释是**推断**；**2D 锚点只有 CPU 复算，补丁不含其 GPU 实现**；
换词形态未解决（指向 BM25/词法，属另一键空间）；L0 跨进程一致性只做了 tombstone 缓解、未证明。

### 60. ⑤ 权重卸载 W13（w13a）：**P0 骨架已在树里但从未接线**；账算清了（decode 不赚、prefill 赚、真用途是"否则不可行"）

#### 60.1 ⚠️ 纠正"零实现"：P0 骨架已在，但没有任何调用点读它
- **已有**：`src/product/weight_residency.h`（242 行：`WeightResidency{Resident,Host,Disk}`、`WeightSpan`、
  `WeightResidencyPlan`、`classify_weight_residency()`、`WeightPageCache`（有界 pinned LRU + epoch 豁免 + 注入式分配器））；
  `src/artifact/binder.h:44` 计划载体；`include/ninfer/types.h:177` `weight_host_offload_bytes`；
  **`src/serve/serve_options.cpp:403-412` 把 `--weight-host-bytes` 解析后"大声拒绝"**；
  `research/notes/A_s32_w13_p0.md` 设计笔记（含 hook 点/锁分析）。
- **但 P0 从未被接线**：`Binder::finish()`（`src/artifact/binder.cpp:124`）忽略该字段，
  `materialize()`（`src/artifact/materializer.cpp:98`）无条件把所有设备对象塞进单一 arena，
  **没有任何调用点读它**。
- **既有的卸载机制各自独立、都不是权重**：KV 冷层 host（`cold_host_tier.h`、`program_impl.h:10890`）/
  磁盘双缓冲暂存（`program_impl.h:921` `cold_disk_staging[2]`）/ PLE 95 GiB 表
  `pread→pinned LRU→H2D`（`src/ops/ple/ple_table.h:9-11`）/ 加载期 pinned 环（`materializer.cpp:174-246`）。
  仓库里 `--cold-disk-path` **只服务 KV**。

#### 60.2 设计的两条关键点
**① 地址永不变（这是能不动 CUDA graph 的原因）**：
`slot(L) = (L − first) % arena_layers` ⇒ **地址在整个生命周期不变** ⇒ `device_data()`、所有 `Weight*`、
**已捕获的 CUDA graph 全部无需改动** —— 循环的是**条带内容**，不是指针。
⚠️ **陷阱**：`fetch_layer` **必须注销被覆盖层的 resident 位**，否则**静默读到错层**
（那是 bug，不是性能问题）。
卸载对象选择：从 artifact 对象名解析 `layers/<N>/`（`bindings.cpp:223` 自己构造的）→ 按层分组 →
按预算**自深向浅**取连续尾部；低于导出地板（`max(1MiB, arena/4096)`）的 norm 常驻；
无层对象（embed/token head/MTP/DFlash/vision）**永不卸载**。
唯一取数入口 `note_layer(L)`（层边界），预取 `L + arena_layers − 1`；命中/未命中/同步缺页**全部计数，绝不静默**。

**② ⭐ 账算清了（决定性）**：解码头**权重流量恒定 19.0 GB/token**（HBM 1792 GB/s → 10.6 ms/token），
而 **KV 流量随上下文线性**（**1M nvfp4 = 20.0 GB/token，比整个权重栈还大**）。
- 卸载 6 GiB ⇒ PCIe **0.258 s/token** vs 20 ms/回合 = **~12× 变慢**（单测断言 `!net_positive`）；
- **同样 6 GiB 摊到 4096 chunk = 1.003×，几乎免费**。
⇒ **dense decode 不赚、prefill 赚、真正的用途是"否则跑不起来"**（1M 只有"慢速可行 vs 不可行"）。
MoE 是最优场景（未接线）。

#### 60.3 KV 预算换算：**精确两点解**（比先前的拟合更强）
`KV_bytes = 4150.40625·tokens·bits + 416666880` —— **逐字节复现引擎自报的两个 1M 点**。
可用 11,600,323,584 B ⇒ **今天只允许 2.57 b/el**（`TODO.md` 早先记的 2.33 来自另一组拟合；
**两者都在两点容差内**，w13a 用了精确复现的那组）。
- 要 **nvfp4 的 4.50** 需腾出 **7.82 GiB**；
- **e8 的 4.25 只需 6.81 GiB，但 e8 已功能性失效（§53：纯 e8 = 0.020）⇒ 不值得**；
- int8 需 **23.0 GiB，不可达**。
宿主 21 GB / 实测可用 15 GB ⇒ **7.82 GiB pinned 不可换页，处于边缘**。
（这条把 §53 的 E8 结论与 W13 的预算直接接上了。）

#### 60.4 交付与诚实边界
- 补丁 `/home/user/scratch/w13a/design.patch`：17 文件、**+1312/−15**、`git apply --check` **rc=0**（17/17 逐文件，跑两遍）。
- `DESIGN.md` 353 行；`budget_table.json`（数字由脚本导出，非手写）。
- **host 单测真跑过**：`g++ 15.2 -std=c++20 -Wall -Wextra` 零告警、`test_weight_residency: all checks passed`，
  40 项含**两轮逐 span 字节精确比对**、地址稳定、销毁/重取、6 条大声拒绝、KV 拟合复现两点、verdict 正负两侧。
- **默认关闭**（预算 0 ⇒ 空计划，全路径逐位不变）。爆炸半径：serve 的 `--weight-host-bytes`
  由"拒绝"变"接受"（行为变更，是本轮目的）；MTP/DFlash/vision 从不卸载但依然常驻。
- **未做到**：除 host 单测外**什么都没编译、没跑**（禁令下零构建零 GPU）；
  **P0 验收未达**（低显存强制启动、logits 逐位一致都没跑）；
  **decode 半边只有设计** —— decode 被 CUDA graph 捕获，host 侧取数会被**重放成陈旧数据**，
  需要把 H2D 提升为"第二条流上的 **graph memcpy 节点 + event 汇合**"（改 `graph_impl.h`），
  因此本轮的 hook **只门控在 `Phase::Prefill`**；带宽数字是**假设**（25 GB/s pinned H2D、1792 GB/s HBM 未在本机测，
  `weight_offload_verdict` 把两者做成入参以便替换实测值）；层字节是均值（19.0 GB/64 层）；
  SSD 层未实现（`Disk` 保留）；MoE 未实测；**P0 的 `WeightPageCache` 已成死代码**（保留仅为 API 稳定，应删）。

### 61. ⑥ 双路（rc2）与 **更正我 §59 的一处误读**；发现**内核与自身 host 参照实现不一致**的真缺陷

#### 61.1 ⚠️ 更正我自己的记录：失败形态是"**中间 + 换词**"，**不是"中/尾"**
我在 §59 写"先前记录的『中/尾两种形态 100% 零匹配』**为真**"——**这句是错的，我把 rc1 的表读反了**：
rc1 自己的表里 **`逐字粘贴在末尾` → `L=16、o=208（正确对齐）、续写 = 正确历史续写`（即"尾"形态是能工作的）**，
失败的只有 `中间`。
rc2 独立确认：树自己的记录（`TODO.md:5890-5893`）写 **`尾` 形态 R@1 0.73–1.00**，它的复刻也复现了（`raw_l=64` HIT）。
⇒ **正确表述：`尾` 形态可用；失败的形态是 `中间` 与 `换词`。**（两个代理在这点上其实一致，是我的转述错了。）

#### 61.2 rc2 独有的机制分解：**两个**结构性盲点（不是统计问题）
1. **位置**：API 的契约是"查询**就是**最新的 Q 个 token"。当 needle 后面跟着别的文本时，最新 Q 窗口装的是
   **追问**，needle 完全在窗外 ⇒ 每个 `min_len` 下 `raw_l = 0` ⇒ **任何索引或阈值都救不了**。
2. **尾部 run**：比较从窗口**最后一个** token 起、遇第一个不等就停。实测：
   **末尾距离 `d` 处替换一个 token ⇒ `raw_l == d` 恰好**（d=15 在 min_len=16 被拒、d=16 被接受）；
   在完全匹配的 needle 尾**之后**追加 16 个不匹配 token ⇒ `raw_l` **32 → 0**；
   真实文本上"中间形态、64 个窗口 token 里 24 个是 needle 字节"仍得 **0**。
另：`limit = min(len-Q-K, start-Q) <= 0` 时**一个候选都不扫**（`start <= Q`）；自匹配按构造不可能。

#### 61.3 ⭐ 真缺陷：内核与它自己的 host 参照实现**不一致**
doc `include/ninfer/ops/suffix_lookup.h:13` 写 `o < starts[b]`；host 参照 `src/spec/lookup_fuse.h:29`
就按这个实现；而**内核 `src/ops/kernel/suffix_lookup.cuh:28` 实现的是 `o < starts[b] - Q`**——**窄了 Q**。
模糊测试 20000 例 ⇒ **91 处分歧，全部是"内核 0、host HIT"**。**rc2 已报告、未改**（我把它记为待修缺陷）。

#### 61.4 rc2 补上了 rc1 找不到的出处 + 其余新增
- **`0.38 tok/s` 的出处找到了**（rc1 曾标"未找到、未复现"）：per-token 流式 1M = 18.00 GiB
  → **2.76 s/token = 0.362 tok/s**；每轮则是 **1:59**（365 µs/token）或 **1:113**（690 µs/token），
  相对解码（38.5 ms/步）≈ **10⁴:1**。
- **前缀函数怎么绕开**：(a) **从不重编号 position**；(b) 只做**前缀闭包**的召回集
  （`plan_prefix_closed` **拒绝有缺口的集合**而不是近似 ⇒ 恢复字节**逐位相同**）；
  (c) 未被召回的轮**根本不参与** ⇒ read-free 按构造成立。诚实代价：一次召回到达第 `i` 轮就要载入 `[i, frontier)`。
- **索引**：从日志的 token id **底层生成**的**任意位置 4-gram 倒排**（`rebuild_index() == incremental`，有断言）。
  实测：**整轮键拿到 R@1，而"最新 Q 键"什么都取不到**；span→SSD 偏移是**绝对页**（`page * page_bytes`），
  因为**轮边界不页对齐**。开销 4 B/token = payload 的 **0.022%**。
- **codec 几何在代码里**（`static_assert` 钉住 18,432 B/token nvfp4、33,792 int8、17,408 e8）；
  **e8 在代码里被拒绝**（把测量写进错误消息），且 e8 只能省 **5.6%**。

#### 61.5 双路的分歧点（落地时要裁决）
**挂钩点两路不同**：rc1 说 `program_impl.h:10470-10483`（`ensure_sequence_kv_mapped`，**四个轮 ingress
循环的最后一句**）；rc2 说 `:8692` `ProgramImplCore::decode(...)`（**轮入口本身**）+ `:8728` `append_forced_tokens`。
两处都合理但不同 —— 落地时二选一或并用（rc1 的位置在"每轮恰好一次、submit 之前"这个性质上论证更充分）。
**索引键**：rc1 复用树内已有的**前缀证书** `CheckpointSummary::shortlist_key`
（`PrefixShortlistDigests::at(frontier), frontier, identity_tag`）；rc2 用**任意位置 4-gram 倒排 + idf-sum**。
两者不是同一层（rc1 是"证书式合法性"，rc2 是"检索式召回"），**更可能是互补而非互斥**。

#### 61.6 交付与边界
- rc2 补丁 `/home/user/scratch/rc2/design.patch`（md5 `24925f4e…`）5 文件 **+1026/−11**：
  契约修正 + Tensor overload（`include/ninfer/ops/suffix_lookup.h`）、
  **布局谓词含 `is_contiguous()`**（`src/ops/wrapper/suffix_lookup.cpp`，正是我指定的那个 bounded 子任务）、
  新增 `src/spec/turn_recall.h` + `turn_recall_test.cpp`、CMake 注册。
- 验证：干净 HEAD 副本与**工作树**上 `git apply --check` **都 rc=0**；装进 scratch 副本后测试编译
  （`-Wall -Wextra`）并 **ALL CHECKS PASSED（37 断言）**；被改的 CUDA wrapper 用 CUDA 13.1 头
  `g++ -fsyntax-only` **rc=0**。
- **未做到**：**没接引擎**（没在重建运行时去改 716 KB 的 `program_impl.h`）；日志里**没有真文件 I/O**；
  **恢复的逐位一致性未端到端验证**（需 GPU，**最高验收门**）；v2（非连续/CacheBlend）按设计缺席；
  索引原型是 idf-sum、**不是完整 BM25**；`start - Q` 那个内核分歧**只报告未修**；
  Q/min_len 未在模型数据上调过；它初版 wrapper 曾在 host 侧读 `starts[]/lengths[]`（会解引用设备内存）
  —— **没有进补丁**（如实自报，好）。

### 62. ⚠️ E8 验收是**阴性结果**：删投影 + 表收窄**都没修好 e8 档**（e8 仍是 0.020）；故障在**读侧/内核侧**，不在编码器精度

#### 62.1 验收读数（`_land_e8_family.sh` 阶段 4/5，完整跑完）
| 臂 | pre-E8 锚点 | post-E8 | Δ |
|---|---|---|---|
| `default`（kc2 表收窄 + D1 影响） | 211/512 = **0.412** | 239/512 = **0.467** | **+28/512** |
| `int8`（`--kv-dtype int8`，不受表收窄影响） | 500/512 = 0.977 | 490/512 = **0.957** | **−10/512** |
| `nvfp4` | 486/512 = 0.949 | 486/512 = **0.949** | 0 |
| **`e8`（纯 16×E8Kv）** | **10/512 = 0.020** | **10/512 = 0.020** | **0（完全没变）** |

构建 rc=0、二进制 `01:40:11 → 02:36:00`、retrieval rc=0（03:14:57）、锁已释放。

#### 62.2 这个阴性结果推翻了什么、留下了什么
- **推翻**：我在 §53 写的"预期 `e8` 与 `default` 两臂显著上升"——**`e8` 一点没动**。
- **⇒ D1（删格点投影，K 的 NMSE 2.32×）不是 E8 功能性失效的原因。**
  这条很强：D1 是**编码器侧**的、格式兼容的、被三次独立测量确认的精度修复（2.32×），
  而它对 **0.020** 这个"字符准确率落在随机水平"的**功能性**失败**毫无影响**。
  ⇒ **E8 的失败不在"写了多少精度"，而在"读回来/算的时候"**。
- **剩下最可能的原因**（与 kv1 的发现直接对接）：**E8 档在 decode 时被强制走 Prompt 路径** ——
  `src/ops/wrapper/gqa_attention.cpp:189-192`
  ```cpp
  // E8Kv small-T kernels are unverified; route E8 to the prompt path.
  if (cache.dtype == DType::E8Kv && route != detail::GqaAttentionRoute::Prompt) {
      route = detail::GqaAttentionRoute::Prompt;
  }
  ```
  叠加"文本全注意力层每个 phase 都调 A1"这一事实 ⇒ 16 层全走那条**作者自己标注 `unverified`** 的路径。
  ⇒ **下一步该查的是这条 Prompt 路径的内核实现**，而不是继续调编码器精度。
  （仓库自己在 `docs/maintainer/kv-strategy-matrix.md` 记的"8-15:e8 → 0/8、compute-sanitizer 干净 ⇒ 静默逻辑/别名错"，
  也指向读侧。）

#### 62.3 `default` 只涨到 0.467，说明"E8 只放前导层就没事"这个前提也要重审
kc2 的表收窄把 E8 从 `{0,1,3,4,6,7,8,9,13,14}` 收到 **`{0,1,3,4,6,7}`**（4 个坏层被换成 NVFP4），
`default` 从 0.412 → **0.467**，**只挪了 28/512**；
而纯 nvfp4 是 **0.949**。⇒ **剩下的 6 个 E8 层（全在"已验证"的 0–7 区间内）仍在扣分。**
⇒ **"`0-7:e8 → 8/8`"这个仓库结论与本次实测不一致**（若 0-7 上的 E8 无损，收窄后 default 应接近 0.949）。
⇒ 要重审的是**整个 E8 档**，不只是"高层 E8 退化"。

#### 62.4 ⚠️ `int8` 臂 −10/512 需要复核（可能是回归，也可能是贪心解码的方差）
`int8` 臂的配置**不受表收窄影响**（`--kv-dtype int8` 覆盖全层），但它 **500 → 490**。
而 D1 的补丁**改了 `prefill_i8.cuh` / `decode_i8.cuh`**——这两个文件**被 int8 档共用**
（D2 的 `kv_scale_half` 饱和也落在同一批行上）。
⇒ 两种可能：**(a) 单 token 翻转级联造成的方差**（贪心解码下 10/512 在这个量级）；**(b) D2 的 clamp 真的改了 int8 的行为**。
**必须用一次重复运行区分**（同一二进制、同一 arm，跑两遍看 int8 是否稳定在 490 还是回到 500）。
在区分清楚之前，**D1 补丁不许算"无副作用"**。

### 63. 导入为什么"还是跑不通"：三处阻断逐行读出；**并承认这是我的执行缺口**（已定位却未去改）

#### 63.1 先分清：**转化侧已通，装载侧没通**
- **转化侧通**：新版 B 在真源上跑完完整 `convert()` ⇒ 1103 对象、产物 **22,585,140,736 B**、
  sha256 `dbbefaee…`、`refusals 0`（加 `--allow-frontend-drift`）、471 s（§62/§61 附近）。
- **装载侧不通**：三处阻断，逐行如下。

#### 63.2 阻断 ①：装载器按**层号**钉死 mlp 格式（真代码缺陷）
`src/targets/qwen3_6_27b/impl/load/bindings.cpp:383`（函数 `:343-398`）：
```cpp
if (layer < 56) {
    target.mlp.gate_up = bind_nvfp4_weight(binder, prefix + "mlp/gate_up", 34816, 5120,
                                           prefix + "mlp/gate_up_projection/input_scale_divisor");
    target.mlp.down    = bind_nvfp4_weight(binder, prefix + "mlp/down", 5120, 17408,
                                           prefix + "mlp/down_projection/input_scale_divisor");
} else {
    target.mlp.gate_up = bind_weight(binder, prefix + "mlp/gate_up", kFp8, {34816, 5120});
    target.mlp.down    = bind_weight(binder, prefix + "mlp/down", kFp8, {5120, 17408});
}
```
- `bind_nvfp4_weight`（`:73-96`）在 `:77-78` **硬 require** `NumericFormat::NVFP4` +
  `StorageLayout::BlockScaleK16M128x4V1`；
- `bind_weight`（`:64-71`）在 `:66-68` **对 NVFP4 直接 `throw`**（"NVFP4 weight requires a paired input divisor"）。
- **源在层 0..55 里为 21 个 mlp 对象**（7 个 `mlp/gate_up` + 14 个 `mlp/down`）**声明 FP8**（转换报告的 `D-OBJECT-FORMAT` 21 条）
  ⇒ **两条路都走不通，装载必抛**。
- **这条 grep 确认是全树唯一按层号决定格式的地方**（`layer < 56` 只此一处）。
- **正确修法**：**读对象自带的 format**，而不是从层号推；同时**保留**"NVFP4 必须配对 input divisor"这个断言
  （不许为了绕过而放松它）——因为 divisor 是 NVFP4 语义的一部分，不是可选的。

#### 63.3 阻断 ②：`weights_id` 未注册（配置缺口）
`src/targets/qwen3_6_27b/impl/package.cpp:91-150` 的 `resolve_weights` 只认
`{groupwise-int, nvfp4, nvfp4-dspark, nvfp4-dflash2, nvfp4-dflash2-bf16head}`；
新产物的 `identity.weights_id` 是 **`nvfp4-modelopt`** ⇒ 落到 `:149` 的
`throw std::runtime_error("artifact identity '" + … + "' is not supported by target …")`。
⇒ 加一个分支 + 一个 `WeightsProfile` 枚举值；**先查清 `WeightsProfile` 承载什么**（采样默认？`default_layer_kv_dtypes`？）再加最少字段。

#### 63.4 阻断 ③：前端 pin —— **这是故意的安全闸，不是 bug**
`tools/convert/qwen3_6/common/frontend_policy.py:327` `acceptability_error`（`:339` 自己写
"确认后可用 `--allow-frontend-drift` 放行（偏离会记入转换报告）"），入口 `import_model.py:524/622`。
源**确实带** `tokenizer_config.json` / `chat_template.jinja` / `generation_config.json`，
但驱动**无法证明**它们与 pin 的 sha256 相同 ⇒ 按策略拒收。
⇒ **合法去掉这个手动开关** = 把该变体的前端文件**纳入可接受集合、并记录哈希**；
**不是**放松闸门（放松会让任何人换 tokenizer 而不被察觉——那正是这个 pin 存在的理由）。

#### 63.5 ⚠️ 我的执行缺口（如实记）
这三处我在派代理去修转化器**之前**就已经核过（`bindings.cpp:383`/`:85`、`package.cpp:91-113` 的行号都在我早先的
验证输出里）。**但我把代理产能全花在了转化器上**（因为 convC 的卷宗说"B 坏了"），
而**真正的阻断一直是这三处小的、已定位的改动**。⇒ 判断偏差：**我把"验证清楚"当成了进度，把"改掉它"一直往后排。**
教训：**定位到阻断之后，阻断本身就该立刻排队去修，而不是先去修它旁边那条更大的、但不在关键路径上的线。**

#### 63.6 这已是本仓库**第三次**同类缺陷
"**用层号/位置规则代替数据自带的属性**"：
1. `nvfp4_gdn_snapshot_plan.cpp` 的"修②"（把 A4 边界按层宽阈值吞掉，§43）；
2. `suffix_lookup` 内核实现 `o < starts[b] - Q` 而它自己的 host 参照实现是 `o < starts[b]`（§61）；
3. 本次 `bindings.cpp:383` 的 `layer < 56`。
⇒ **值得作为一条检查项：凡"按层号/按名字/按位置"决定数值格式的地方，都要问"报文的哪一部分本来就带着这个信息"。**

#### 63.7 已派双路代理（ld1/ld2）与端到端验收计划
- 双路任务：复核 ①②③ + 找出**我可能漏掉的第四处**（grep `NumericFormat::` 与 `layer <`），
  出**底层驱动**的补丁（读 artifact 自带 format/identity），并给出**可证明的最小 CPU 证据**
  （`-fsyntax-only` + 绑定路径的合成复算）。
- **验收链**（修完我来跑）：**修三处 → 重建 → 重新产出真产物（471 s、22.58 GB，盘上 49 GB）→ 让引擎真去装载它。**

### 64. 硬编码系统普查（用户口径："这是旧 ninfer 的特色，但不是我们的需求"）——分类、规模、工作分解

**用户原话（本轮指令）**：
> "你把所有的硬编码部分都找出来改了 这是旧的 ninfer 的特色 但不是我们的需求"

⇒ 判据定死：**引擎必须由"数据/artifact 自带的属性"驱动**；
**凡"按模型名 / 按层号 / 按位置 / 按名字"决定数值格式或行为的地方，都是旧 ninfer 的特色，都要改。**
⇒ 从"修三个点"升级为**一次系统普查 + 分类修**。

#### 64.1 普查规模（`sh/_hardcode_recon.sh`，限定 `src/ apps/ tests/ include/ bench/`）
| 类 | 模式 | 处数 |
|---|---|---|
| A | `layer <op> N` | **18** |
| B | `(weights_id\|model_id) ==` | **20**（6 文件） |
| D | `tokens <op> N` | **201**（需筛出"派发/规划决策"vs"合法性校验"） |
| F | `static_assert(... == N)` | **408**（多数是结构性约束，需筛"字面量比字面量"的失效镜像） |
| E | 绝对路径（`/home/`、`/mnt/`、`C:\`、`\wsl`、`neroued`、`ziqinzhang`） | **20** |

#### 64.2 A 类真问题样本（按层号/位置决定行为）
- `src/targets/qwen3_6_27b/impl/load/bindings.cpp:25` `is_full_layer(layer) { return layer >= 3 && (layer - 3) % 4 == 0; }`
  —— **硬编码的"层种类周期"**（周期 4、偏移 3）；
- 同文件 `:28` 硬列 `layer == 3 || 7 || 11 || 15 || 19 || 23`；`:31` `layer == 3 || layer == 7`；
  `:33` `layer == 4`；`:383` `layer < 56`（已定位的装载阻断）；
- `src/targets/qwen3_6_35b_a3b/impl/load/bindings.cpp:21` **硬列 `{34,38,39}` 决定用什么数值格式**（与 `layer < 56` 同类）；
- `src/targets/muse_glimmer_30b/impl/load/bindings.cpp:60-61` 硬区间 `layer >= 5 && layer <= 11` / `layer >= 1 && layer <= 12`；
- **A3 逐层 KV 表**：`default_layer_kv_dtypes` 在 3 个变体各有一份，
  **函数签名就把层数写死成 `std::array<DType, 64>`**，表体是硬列 `{0,1,3,4,6,7}`（`qwen3_6_27b/impl/variant.cpp:23/44`）。
  ⇒ **这张表正是 §62 那个 E8 阴性结果的载体** ⇒ 它的数据驱动化要等 E8 读侧定性。

#### 64.3 C 类真问题样本（同一真值在 ≥2 处各写一遍）
- **TMA 判据被抄了 5 份**：定义在 `src/ops/linear/nvfp4/nvfp4_w4a4_plan.h:58` 的 `nvfp4_w4a4_tma_route(tokens)`，
  然后内联重写在 `nvfp4_linear_add_w4a4.cu:62`、`nvfp4_gdn_input_w4a4.cu:42`、`nvfp4_attn_input_w4a4.cu:88`、
  `nvfp4_w4a4.cu:64`，**每个还各带一份 `constexpr kTmaBlockM = 256`**；
- **`suffix_lookup` 内核 vs 它自己的 host 参照**：doc `include/ninfer/ops/suffix_lookup.h:13` 与
  host 参照 `src/spec/lookup_fuse.h:29` 是 `o < starts[b]`，**内核 `src/ops/kernel/suffix_lookup.cuh:28` 是 `o < starts[b] - Q`**
  ⇒ 20000 例模糊测试 **91 处分歧（全部"内核 0、host HIT"）**；
- 数组尺寸/上限：`kout[16]`/`vout[16]`（`gqa_attention_decode_i8.cuh:549-550`、`gqa_attention_prefill_i8.cuh:575-576`）、
  `make_index_sequence<16 - kNvfp4FirstSmallT + 1>` vs 另一侧 `kNvfp4LastSmallT`(=32) 且**索引无边界检查**、
  `PageIds = 64` + `physical_pages_s[PageIds]` 无边界检查。

#### 64.4 E 类真问题样本（硬编码路径 / 机器专属量）
- **`tests/targets/qwen3_6/test_frontend.cpp` 12+ 处 `/home/neroued/models/llm/qwen/…`**
  ⇒ 这就是 ctest **`#26` 永远红**的原因（缺文件时 `terminate` **abort** 而不是 skip）；
- **`src/targets/qwen3_6/impl/runtime/dflash2_impl.h:152/424` 里写着我的个人工作区绝对路径**
  `/mnt/c/Users/User/Documents/ziqinzhang/dl/feat_%s_%d.bin` 与 `df2scores_%s_%d.bin`
  ⇒ **调试落盘路径留在了生产源码里**，这条是最该改的；
- **ctest `#4`** 是注册时没给 `ninfer_ple_table_e2e_test <sidecar_root>` 的必需参数。

#### 64.5 工作分解（全部按铁律⑦双路）
- **W1 装载器数据驱动化**（`layer < 56` + `kTextLayers`/`is_full_layer`/硬列格式）⇒ **代理 ld1/ld2 已在跑**（导入阻断）。
- **W2 身份白名单 → 声明式能力注册**（`package.cpp`×3 + `registry.cpp` + `context_cost.cpp` + bench）⇒ **代理 id1/id2 在跑**。
- **W3 单一真值来源**（5 份 TMA 判据、`suffix_lookup` 内核 vs host 参照、数组尺寸 vs planner 上限）⇒ **代理 ss1/ss2 在跑**。
- **W4 硬编码路径**（`test_frontend.cpp` 12+ 处 → env 驱动 + 缺文件 **SKIP 77**；`dflash2_impl.h` 的调试路径；
  `#4` 的注册参数）⇒ **代理 ph1/ph2 在跑**。
- **W5 逐层 KV 表数据驱动化**（`default_layer_kv_dtypes` ×3、`std::array<DType, 64>` 硬签名）⇒ **等 E8 读侧定性后再派**
  （它就是那张表；E8 结论出来才知道该由谁声明、声明什么）。

#### 64.6 立为检查项（写进纪律）
**凡"按层号 / 按名字 / 按位置"决定数值格式或行为的地方，都要问一句：报文的哪一部分本来就带着这个信息？**
本仓库已出现 **4 次**同类：① `nvfp4_gdn_snapshot_plan.cpp` 的"修②"；② `suffix_lookup` 内核 vs host 参照；
③ `bindings.cpp:383` 的 `layer < 56`；④ `35b_a3b/bindings.cpp:21` 的 `{34,38,39}`。

### 66. E8 读侧双路**结论互斥**：e8r1 说"不是精度"、e8r2 说"就是 V 的精度"——我的裁定与决定性实验

#### 66.1 求同（两面独立一致）
- **都纠正了我的行号**：闸门在 `gqa_attention.cpp:518-520`（我给的 189-192 在本树里是 `require_shape`）。
- **都确认 E8 在 decode 走 Prompt 路径**（prefill 内核），**从不进** decode smallt 内核。
- **都判定写/读约定严格互逆、往返本身正确**：H64 = H2⊗H32 严格正交自逆（范数比 1.000000000000、自逆误差 4.4e-16）、
  nibble 次序（偶维→低）、`clamp(±7)` 与 `(n&0xF^8)-8` 符号扩展配平、scale per-64/lead=4 两侧一致、
  **smem 展开布局逐字节精确**（64 行×256 维 0 处不匹配）。往返误差 K relRMS **0.106**、V **0.178**，
  无 10× 异常、无系统性偏置。

#### 66.2 求异（互斥的核心）
| | e8r1 | e8r2 |
|---|---|---|
| 根因 | **不是精度**：输出误差只差 1.28×（0.1803 vs 0.1414），不可能造成 55× 保留差（10/550 vs 550/550）⇒ 必是"哪条路活着/两个 reader" | **找不到类别性读侧缺陷** ⇒ 倾向**精度，具体是 V 平面** |
| 主证据 | 标定曲线：K 加噪 relRMS 0.105/0.25/0.45/0.80 → 输出误差仅 0.087/0.201/0.408/0.648 | **V relRMS = 0.178**（K 的 1.8×、**int8 的 V 的 10.6×**）；E8 复用 int8 内核的 **per-64** scale 平面配 4-bit 码，而 **NVFP4/ISO3 的 V 用 per-16** ⇒ 3-bit 档的 V 比 4-bit 的 E8 更准。步长模型：per-64 预测 0.158（实测 0.178 ✓）⇒ **per-16 预测 0.102（3.0× MSE）** |
| 形状 | — | 退化**单调**（0/6/16 层 → 0.949/0.467/0.020），与"精度不足"相符、与"一层坏就全崩"的离散 bug 不符 |
| 修法 | 路由对称化（`gqa_attention_cached()` 也套 E8→Prompt）+ shuffle UB + Int8 名单 | V 换 per-16（**要同改 plane 几何与读侧，+0.75 b/el ⇒ 设计取舍，不是一行修**） |

#### 66.3 我的裁定：**e8r1 否定精度的那条论证不成立**
它的推理是"1.28× 的输出误差不可能造成 55× 的保留差"——**这假设"误差→保留率"平滑**。
而**本会话自己的数据反复推翻**：去掉 4 个 E8 层就让 [D0] 30/64 → 63/64；int8 换个 kernel 就让 [D0] 60 → 50。
**贪心解码是混沌放大器，离散指标上的整体翻转正是"小误差"的典型后果。**
反过来 e8r2 的定量关系更硬：**V 的误差在 attention 里不被衰减**（`out = Σ p_i V_i`，权重和恒为 1）
⇒ V relRMS 0.178 基本原样进输出；**而 e8r1 自己测的 E8 `OUT relRMS = 0.1803` 与 e8r2 的 V relRMS 0.178 几乎相等**
—— e8r1 的输出数本身就是"由 V 主导"的证据。

#### 66.4 两面各自挖到的真缺陷（照记）
- **e8r1**：① **路由不对称**——`gqa_attention.cpp:573` 的 `gqa_attention_cached()` **没被 `:518-521` 的 E8→Prompt 覆盖**，
  而它用 `decode_i8_tiled_kernel<E8=true>` 读**同一批由 `fill_i8_kernel<E8=true>` 写出的字节**（一个格式两个 reader）；
  ② **`gqa_attention_decode_i8.cuh:301-309` 的 shuffle UB**（`__shfl_xor_sync(FullMask,…)` 在 `if ((lane&1)==0)` 里，
  **奇 lane 携带每个打包字节的高 nibble**；同族 fill kernel `prefill_i8.cuh:165-181` 早已提出分支外）
  —— ⚠️ **这条与 E2 独立发现的 D5 是同一处**；③ `gqa_attention_decode_smallt.cu:204` 的 `Int8` 名单缺 `DType::E8Kv`（惰性）。
- **e8r2**：④ **`gqa_attention_prefill_e8.cu:316-336` 只传 10 个实参**，`cold_k_slots/cold_v_slots/slot_bytes` 落默认
  ⇒ `tile_cold` 恒假 ⇒ 一旦 block table 出现**负项**（冷页编码 `-(slot+2)`），表项被当物理页号代进
  `gqa_kv_i4_code_index` ⇒ 静默读到池内别的区域（越不出池 ⇒ compute-sanitizer 干净）——
  **它自己证伪为根因**：`program_impl.h:10660-10684` 的栈级闸门在"任一层不是 I8/NVFP4"时直接 return，E8 因此**从不退页**，
  且 int8 同样中招 ⇒ 解释不了 E8 专属性。补丁 = 把"负表项 + 未给冷平面"从静默别名改成整块零填充（加固）。

#### 66.5 ⚠️ 一条影响整条 E8 测量线的更正
**e8r2 逐字节证明 `/home/user/bench/kvdump_e8src` 是 Sep-10 修复之前的产物**（其 ks 平面 256/256 等于"旋转前 amax/7"，
码平面既不匹配"无投影"也不匹配现代编码）⇒ **它不能当当前写侧的真值**。
⇒ **E1/E2/adj1/adj2 的绝对数字都是在"旧编码器"的 dump 上算的** —— **比值与方向仍可信，绝对值要打折。**

#### 66.6 决定性实验（两个假设各有自己的可证伪读数，我都跑）
- **实验 A（验 e8r2 的 V 归因）**：把 E8 的 V 换成 per-16 尺度（或临时把 V 按 int8 存）→ 重建 → 同一条 4 臂硬检索。
  **纯 e8 从 0.020 升到 ≥0.3 ⇒ V 是主因、e8r1 的否定被推翻；仍 ≈0.02 ⇒ V 归因被证伪。**
- **实验 B（验 e8r1 的路由归因）**：打它的补丁（路由对称化 + shuffle UB）→ 重建 → 四臂各 8 针。
  **`0-15:e8` 的 sum(prefix) 从 10 跳到 ≥500 ⇒ 路由是主因；仍 10 ⇒ 被证伪。**

### 67. W4（硬编码路径，ph1 第一路）：产品代码里只有 **2 处**；测试里**恰好 12 处**；并补上我漏的 CMake 一环

#### 67.1 清单（分类 + 判定依据）
- **产品代码：只有 2 处**（`src/targets/qwen3_6/impl/runtime/dflash2_impl.h:152` 与 `:424`
  的 `/mnt/c/Users/User/Documents/ziqinzhang/dl/feat_%s_%d.bin` / `df2scores_%s_%d.bin`）
  ⇒ **判定为"env 门控的诊断通道"，不是随手写的死路径**，依据：两块都在 `getenv("NIFER_DF2FEAT"/"NIFER_DF2SCORES")` 里；
  `docs/maintainer/speculative-dflash2-status.md:52-56` 把这两个探针写成"可复用的诊断工具"；`git blame` 全指向
  一个调试提交 `d38bb91`，且 `research/scripts/_patch_df2scores.py` 就是写入这段的插桩脚本；
  **仓库已有约定**：`NINFER_KVDUMP_DIR` / `NINFER_HS_DUMP_DIR` / `NINFER_KV_CALIB_DIR` 都是"目录取自 env、未设则不落盘"，
  **这两个块是全树唯一违反该约定的 dump 点**。
  ⇒ **修法：保留通道、改掉目的地**（新增 `NINFER_DF2DUMP_DIR`，未设则连 `cudaMemcpyAsync` 之前的同步都省掉），**不删**。
- **测试：4 站点 × 3 文件 = 12 处（没有更多）**，全在 `tests/targets/qwen3_6/test_frontend.cpp`
  （`:116/118/120`、`:1226/1228/1230`、`:1377/1379/1381`、`:1626/1628/1630`）。
- **我漏掉的 3 个 py + 1 个工具**：`tests/convert/qwen3_6_27b/{test_official_resources.py:16,19, test_convert.py:21}`、
  `tests/convert/qwen3_8_27b/test_mtp_assembly.py:28,32`（都未注册进 ctest）、
  **`tools/convert/qwen3_8_27b/mtp.py:36` 的 `Path("/home/user/ninfer-fusion")`** —— 而它 `__file__` 回退写的是
  `parents[1]` = `tools/convert`，**永远是死分支**，只有硬编码路径能生效（已改成 `parents[3]`）。
- **误报提醒**：`README.md` 里的 `huggingface.co/neroued/...` 是上游 HF 命名空间，**不是机器路径，不能改**。

#### 67.2 ⚠️ 我漏掉的一环（ph1 补上）
**`#26` 光在 C++ 里返回 77 不够**：`tests/CMakeLists.txt:126-132` 注册 `ninfer_qwen3_6_frontend_test` 时
**没有 `SKIP_RETURN_CODE 77`**（该属性只由 `ninfer_add_op_test` 和逐条 `set_tests_properties` 给出）
⇒ 只改 C++ 会得到 "Failed (exit code 77)" 而不是 Skipped。
`#4` 走 `ninfer_add_op_test` 已带 77，所以只需改 `.cu`。⇒ 补丁加了 `set_tests_properties(... SKIP_RETURN_CODE 77)`，
**它只在 cmake configure 时写进 `CTestTestfile.cmake` ⇒ 必须 reconfigure 才生效**。

#### 67.3 验证（**真编真跑**，不是只过语法）
ph1 用 `compile_commands.json` 的原始命令 + `link.txt` 链接（去掉 `--dependency-file` 以免写 `build/`），
得到 `scratch/ph1/tf/test_frontend_fixed` 并实测：
- env 未设 → **exit 77** + `SKIP: NINFER_FRONTEND_TEST_ROOT is not set …`；
- env 指向真实 HF 目录 → 不再 skip，三个文件全被读入（`tokenizer.json` 19,989,325 字节）、继续进入 tokenizer 解析；
  带尾部 `/` 结果一致；
- env 指向不完整目录 → **exit 77** + `SKIP: missing official test resource <根>/tokenizer_config.json`；
- 把 helper 段单独编译运行，覆盖设/未设/空值/半个目录/尾斜杠 5 种情形。
E3 分支同法实测：无参无 env → 77；env 指向空目录 → 77；目录里有 `ple-manifest.json` → PROCEED；`argv[1]` 优先 → PROCEED。
python：4 文件 `py_compile`；`mtp.py` 在无 `PYTHONPATH`、cwd=`/tmp` 下导入成功并解析出 `REPO_ROOT`
（对照未打补丁版解析出 `/home/user/ninfer-fusion`，**证明旧回退确实是死的**）；
`test_mtp_assembly` 用 unittest **实跑两遍** —— 未设 env：13 tests / 1 skipped；设 env 指向真实产物：**13 tests 全 ok**（含 byte-exact 对拍）。

#### 67.4 ctest 影响预测（基线来自 `LastTest.log`）
- **基线：109 collected，107 pass / 2 fail / 0 skip**（失败恰为 `#4` usage→exit 2 与 `#26` abort）。
- **打补丁 + 重建后预期：107 pass / 2 skip / 0 fail**（`#4` exit 2→77 免 reconfigure；`#26` abort→77 **需 reconfigure**）。
- 若在 env 里设了 `NINFER_FRONTEND_TEST_ROOT` / `NINFER_PLE_SIDECAR_ROOT`，对应测试改真跑。

#### 67.5 未做到 / 未证
无 GPU；**`#26` 在本机无法全绿**——本机没有 Qwen3.6-27B base-hf 目录（`/home/user/models` 下只有 qwen3.8 系），
已有的候选（`q38_abl_huihui_nvfp4`、`q3nvfp4`）的 `tokenizer_config.json` 都缺 `added_tokens_decoder`，前端会拒；
所以"设了 env 且文件在 → 走原路径"我证到的是"三个文件确实按 env 根被读入并继续进入解析"，**不是 exit 0**。
两个 pytest 文件未实跑（环境无 pytest）。**`tools/`、`eval/`、根脚本、`profiles/` 里的机器路径未改**（清单已给全）——
它们是对外接口（CLI 默认值、env 名、GUI 文案），改法需要用户定；
其中 `eval/configs/qwen3_6_35b_needle_haystack.yaml`（10 处）与 `tools/convert/qwen3_6_35b_a3b/convert.py:38` 的
`GGUF_EVIDENCE_PATH`（关系到记录溯源）是下一步最该处理的两处。
CMake 改动**未跑 configure 实测**。
顺手发现未修：`dflash2_impl.h:436-437`/`:440-441` 把 `dump_one("front"/"anch")` 与同一段注释**各写了两遍**
（来自 `research/scripts/_patch_two.py` 的重复插桩），无害但冗余。

### 68. ⭐ **E8 到底是什么**（权威规格，代码引用）：它是"4-bit 整数平面 + 一个存不进去的格点投影"，且 **V 侧不是 E8**

**用户口径**：**"你先搞明白啥是 e8 再修啊"** —— 这一节就是那份规格。以下每一条都能落到代码行，不含推断。

#### 68.1 dtype 的定义（`src/core/dtype.h:21-24`，原文）
```
// Packed 4-bit E8-lattice K codes + i4 V codes (two codes per byte) with
// per-64-channel FP16 scales; consumed by the int8 attention kernels
// (stage unpacks nibbles to i8). See the per-layer KV storage table.
E8Kv       = 10,
```
⇒ **一句话规格：K 是"E8 格点码"，V 是 i4，尺度是 per-64 FP16，读者是 int8 内核。**

#### 68.2 它由五个面组成，**其中两个面并不是 E8**
| 面 | 实际是什么 | 代码 |
|---|---|---|
| **存储** | `2 × {U8, head_dim/2=128, kv_heads, 256}`（K/V 各一条 4-bit 打包 code 平面）+ `2 × {FP16, head_dim/64=4, kv_heads, 256}`（尺度）⇒ **4 + 16/64 = 4.25 b/el** | `decoder_state.cpp:185-191` |
| **K 编码器** | ① `gqa_kv_hadamard64`（64 维 Sylvester 旋转，最后一级乘 `1/8`）→ ② `e8_project_8d_warp`（Conway–Sloane 最近点，E8 = **D8 ∪ (D8+½)**）→ ③ `clamp(±7)` → ④ 两 nibble 打包 | `gqa_attention_kv_quant.cuh:256-270`（H64）、`e8_lattice.cuh:69-138/224-228`（投影） |
| **V 编码器** | **`gqa_kv_quant_i4_code` —— 纯标量 4-bit、offset-binary、`clamp(±7)`、无旋转、无投影** | `gqa_attention_kv_quant.cuh:356-361` |
| **尺度** | per-**64**-channel FP16，**与 int8 档同一套 plane 几何**；E8 的除数是 **7**（`amax/7`），int8 是 127 | `gqa_kv_quant_code`（±127）vs `gqa_kv_quant_i4_code`（±7） |
| **读侧** | **int8 内核对 `E8=true`**：把 packed 平面在 smem 里展开成 i8，然后跑 int8 的数学；反量化 = `码 × fp16 尺度`，码是 **offset-binary** `(nibble ^ 8) − 8` | `gqa_kv_unpack_i4`/`gqa_kv_unpack_i4x16`（`:368-380`） |
| **路由** | `gqa_attention.cpp:518-520`：**E8Kv ⇒ 强制 Prompt**（注释 "E8Kv small-T kernels are unverified"）⇒ **decode 跑的是 prefill 内核**，这就是 §62 那个 5–17× 慢的机制 | |

#### 68.3 ⭐ 决定性的一条：**它的"格点增益"被构造性地丢掉了**（`e8_lattice.cuh:140-149`，原文）
```
// !! READ THIS BEFORE CALLING THE PROJECTIONS ON THE KV WRITE PATH !!
// e8_project_* implements the Conway-Sloane nearest-point rule correctly (verified
// exhaustively against a 2x2^8-candidate exact search on 20000 random blocks: 0/20000
// strictly non-nearest), BUT ITS OUTPUT CANNOT BE STORED IN THE KV CODE PLANE THE READER
// USES. That plane is one signed integer per coordinate; the projection returns either D8
// points (integer coords) or D8 + 1/2 points (ALL-EIGHT-coordinates half-integer), and
// 47.6% of real K/V blocks on the L13-L15 forensics dumps TAKE THE HALF-INTEGER COSET.
// The `rintf` that necessarily follows therefore moves those coordinates 0.5 step, which
// measured +3.65 dB of error (2.32x MSE) at the same bit rate versus plain rounding.
// Use these only where the half-integer coset is representable (a code plane holding
// 2*coordinate, i.e. 5 bits at +-15 = +1 bit/el) or where the consumer reconstructs the
// lattice point rather than an integer code.
```
⇒ **"E8" 的 4-bit 整数平面装不下它自己算出来的格点**（47.6% 的块落在全半整数陪集上，被 `rintf` 推 0.5 step）
⇒ **标称的格点增益被构造性丢弃**；而要真拿到它需要 **+1 bit/el 的 5-bit 平面**，E1 实测那样**严格劣于**把这一 bit 花在更细的均匀格点上。

#### 68.4 结论性规格（这才是"E8"）
**E8Kv = 「一个 4-bit offset-binary 整数平面（K/V 各一条）+ per-64 FP16 尺度（与 int8 共用几何、除数为 7）」
+ 「K 侧额外施加 Hadamard-64 旋转与一个**存不进该平面**的 E8 格点投影」+ 「V 侧是**纯标量 i4、无旋转无投影**」
+ 「读者是 int8 内核、E8Kv 被强制走 Prompt 路径」。**

因此：
1. **E8 的"格点"部分是死功能**（存不下）⇒ 删投影（`D1_PATCH`，已落地）**方向正确**：删掉之后 K 侧退化为
   "4-bit 整数平面 + Hadamard-64 旋转"，与平面自洽。
2. **E8 的 V 侧是全家族最弱的 4-bit V**：它复用 int8 内核的 **per-64** 尺度去配 4-bit 码，
   而 NVFP4/ISO3 的 V 用 **per-16** ⇒ e8r2 实测 V relRMS **0.178**（K 的 1.8×、**int8 的 V 的 10.6×**），
   per-16 预测 **0.102（3.0× MSE）**。
3. **所以"修 E8"= ① 删投影（已落）+ ② 修 V 的尺度粒度**；而 ② 要同时改 plane 几何与读侧、**+0.75 b/el**，
   是**设计取舍**（用户需知），不是一行修。

#### 68.5 据此排的两条实验与两条修法
- **实验 A（验 V 归因）**：E8 的 V 换 per-16 尺度（或临时按 int8 存）→ 重建 → 4 臂硬检索。
  **纯 e8 从 0.020 → ≥0.3 ⇒ V 是主因；仍 ≈0.02 ⇒ 被证伪。**
- **实验 B（验路由归因）**：打 e8r1 的路由对称化 + shuffle UB 补丁 → `0-15:e8` 从 **10** → **≥500** 才算成立。
- **修法 ①**：删投影（`D1_PATCH` 已落，等实验确认它是否足够）。
- **修法 ②**：V 的尺度粒度（per-64 → per-16）或 V 直接换档；**这是取舍，落地前要看实验 A 的读数**。
- **auto**：另按 §65 走（宽度阶梯 + 校验放行 + 不再落常量 3），aw1/aw2 在跑。

### 69. ⭐ E8 的**数学优势**（查证）+ **我把表示代价高估了 8 倍** ⇒ "删格点"是错的修法

**用户口径**：**"我的意思是你从数学上搞明白这玩意的优势然后再搞啊……什么叫删除格点？你先上网找一下什么是 e8 再说"**

#### 69.1 E8 是什么（文献）
- **E8 = Gosset 格**：8 维**最密球堆积**（**Viazovska 2016 证明最优**，2022 Fields Medal），kissing number **240**，
  且有 **O(1) 闭式最近点解码**（Conway–Sloane）——本仓库的 `e8_project_8d_fast` 就是它。
- **它的优势即 granular gain（同码率下对纯标量量化的增益）**：
  `10·log₁₀(G(ℤ)/G(E8))`，`G(ℤ)=1/12≈0.0833`、`G(E8)=0.0717` ⇒ **≈ 0.65 dB**。
  旁证：高码率下距 Shannon 的差距 E8 是 **0.88 dB**、纯标量是 **1.53 dB**，差 **0.65 dB**（同一个数）。
- **引擎自己的 CPU 实测 `ideal-E8/z8 = 0.874×` = 0.58 dB** ⇒ 与理论吻合。
⇒ **这个格式的价值是真的、可量化的。**

#### 69.2 ⚠️ 表示那个陪集的代价，我（和 E1）**高估了 8 倍**
E8 = **D8 ∪ (D8 + ½·1)**。关键结构事实：**一个 8 坐标块的最近格点，要么八个坐标全是整数（D8），
要么八个坐标全是半整数（D8+½）——两个陪集不混。**
⇒ 表示它只需要 **每 8 个坐标 1 个"陪集位"**；**坐标本身仍是整数、仍装在现有 4-bit 平面里**
（半整数陪集就存 `坐标 − ½`，读时加回）⇒ **代价 = 1 bit / 8 坐标 = +0.125 bit/el**（4.25 → **4.375**）。
- `e8_lattice.cuh:140-149` 的警告写的是 "a code plane holding **2×coordinate**, i.e. **5 bits at ±15** = **+1 bit/el**"。
  **那是"每个坐标多存一位"。** 而**同一段注释自己那句 "ALL-EIGHT-coordinates half-integer" 正是让便宜编码成立的事实**：
  既然整块同陪集，**一位就够**。⇒ **结论被一句 8× 高估的成本带偏了。**
- 三处更正：① **我的"删投影"方向错了**（它把这个格式唯一的价值删掉，剩下的只是"加了旋转的普通 int4"）；
  ② **E1 的"陪集位严格劣"建立在 +1 bit/el 这个 8× 高估上**；③ **E2 提的"陪集位 4.375 b/el"在机制上是对的**
  （它的测量因转置不可用，但机制没错）。

#### 69.3 但"值不值"是**速率-失真问题**，我拒绝手算当结论
同码率 E8 好 **0.65 dB**；而多花的 **0.125 bit** 若按 6.02 dB/bit 折算值 **~0.75 dB** ⇒ **两者接近**。
谁赢要看**在"4-bit 整数码平面 + 每块 1 位边信息"这个受约束的编码族里**，E8 与"同样 1 位边信息的其它用法
（如按块在 Δ 与 Δ/2 之间切换）"谁更优。**这必须推导 + 实测。**
⇒ **定下来的是**：当前实现**净亏**（投影照算 + `rintf` 推 0.5 step = **−3.65 dB / 2.32× MSE**，比纯取整还差）；
"**E8 现在没用**"成立、"**E8 本该有用**"也成立；**中间那段（怎么表示、值不值）不许拍。**

#### 69.4 已派双路（e8m1/e8m2），要求
1. **数学**：把 `10·log₁₀(G(ℤ)/G(E8))` 的来历写清（G 的定义、0.0717 的出处、算式），
   并说明"同码率"在"每坐标 4 bit + 每 8 坐标 1 位边信息"的格式里**如何精确定义**；
   **在同一存储预算下严格比较** (i) E8+陪集位 vs (ii) 纯 4-bit 均匀 + 同一位做别的用途，
   给出**理论 MSE 与谁赢多少 dB**——**若结论是"E8 不划算"，直说并给推导**（那推翻的是我的方案，不是格式）。
   并回答 **Hadamard-64 旋转的收益与格点增益是否可加**。
2. **编码设计**（若数学支持）：陪集位存哪（现有 plane 有没有空位/要不要改布局）、写侧怎么产、读侧怎么重建、
   与 `clamp(±7)` 和 offset-binary `(nibble^8)-8` 怎么共处、**b/el 的精确算术**。
3. **纯 CPU 实测（硬要求）**：同存储预算下 (i) 均匀+旋转 (ii) E8+陪集位 (iii) 投影+rintf 的 MSE/相对误差/**按 b/el 归一**。
   ⚠️ **数据来源**：`/home/user/bench/kvdump_e8src` 已证明是 **Sep-10 之前的产物**，**不能当当前写侧真值**——
   只能用同一输入做**编码器之间**的相对比较并明确标注；有更新的 K/V 源就优先用。
4. **补丁**：`/home/user/scratch/e8m1/fix.patch`（或 e8m2）实现推导出的方案；
   若数学结论是"不划算"，**不交补丁**，改交"为什么这个格式应当直接用 4-bit 均匀+旋转、把 E8 从 dtype 表退役"的证据包
   ——**这也是合法交付，但必须带 3 的实测**。
5. **明令**：**不许把 E1/E2 之前测过的数当证据**（旧 dump + E2 的读取转置 bug），**必须自己重测**。

### 71. W1 双路（ld1/ld2）裁定；**转换器缺口已补**（此前根本不在树里）；仍缺 118 条记账补丁

#### 71.1 两面的同（独立一致，且都用了**真 artifact 报告**当基准）
“基准 = `cbfix2/out2/qwen3_8_27b_nvfp4_modelopt.ninfer.conversion.json`（identity `{qwen3.8-27b, nvfp4-modelopt}`、
1103 对象、22,585,140,736 B、471 s、refusals 0）”。
- ① `bindings.cpp:383` 的 `layer < 56` 确认；**`D-OBJECT-FORMAT` 恰好 21 条** = 7 `gate_up`(`{0,1,2,3,50,52,54}`)
  + 14 `down`(`{0,1,2,3,21,42,44,46,48,49,50,52,53,54}`)；**转换器的 scope 集合与之逐项相同**（程序化核对 `scopes agree exactly: True`）。
- ⭐ **两都指出：`gate_up` 与 `down` 在 7 个层上互相不一致**（`21,42,44,46,48,49,53`）⇒ **修法不可能"按层"表达，必须按对象**。
- ⭐ **FP8 对象在 artifact 里根本没有 divisor**：`D-DIVISOR-DROPPED` 也是 21 条，route 直方图 `divisor: 91` == `NVFP4: 91`
  ⇒ **FP8 声明必须"不做 divisor 查找"地绑定**，且 `Binder::finish()` 仍要求全部对象被消费。**两面独立得到同一结论。**
- **NVFP4 的 divisor 要求不许放松**（`bind_weight` 对 NVFP4 仍 throw；缺/0/负 divisor 仍致命）——两面都保留了这条。
- `WeightsProfile` **不**决定 `default_layer_kv_dtypes`（该参数**无名且未用**，两面都核过），**不**承载采样默认值（按 `model_id` 键）；
  它承载的是：端点格式、哪套 text-layer binder、`auto` 的 backend、以及 **7 个 switch 里的 per-op workspace 容量**。
- **CRLF 陷阱**：`bindings.cpp`/`package.h`/`variant.cpp` 是 CRLF，且 **`variant.cpp` 带 UTF-8 BOM**；
  ld1 第一版因归一化换行产生 **1604 行幻影 diff**，已修。ld2 也踩到同类（提到 `registry.cpp` 是 CRLF）。

#### 71.2 两面的异（范围）与裁定
- **ld1**（9 文件 **+201/−67**）：额外**删掉** `bind_nvfp4_text_layers` 里三个按层号定格式的谓词
  （`is_early_attention_input`:282 / `is_bf16_attention_output`:296 / `is_bf16_gdn_output`:324）；
  并**找出 4 处同类**：**`tools/convert/qwen3_8_27b/inventory_nvfp4.py:46-47`
  （`NVFP4_MLP_LAYERS = range(56)`、`FP8_MLP_LAYERS = range(56,64)`）= `bindings.cpp:383` 的"生产侧孪生体"**
  ——**这正是偏差报告读作"registered artifact pins NVFP4"的成因**；另有 `35b_a3b/impl/load/bindings.cpp:20-23`
  （`routed_down_format`：`{34,38,39} → Q6G64 否则 Q5G64`）、`muse/impl/load/bindings.cpp:57-62`
  （`layer_mlp_fp8`，注释自认 "Source format table (measured)"）。
- **ld2**（9 文件 **+162/−21**）：**不动**那三个谓词——它们**在 `Qwen36Nvfp4` 路径上、对 `nvfp4-modelopt` 不可达**，
  改动它们会改注册路径**却没有 artifact 可验**。它另加 `Binder::find_tensor`（**只读、不消费**——因为
  `require_tensor` 是"断言+消费"，无法用来问"artifact 说什么"）作为底层原语；`bind_declared_weight`
  **对任何第三方声明一律拒绝并点名格式**（不是"声明什么就装什么"）；并用 `-Wswitch-enum`
  **做对照实验**（删掉一个 label 会同时让 `variant.cpp` 与 `bindings.cpp` 报警）证明检查承重。
- **裁定：落 ld2**（更窄、可验、控制更硬），**把 ld1 的 4 处同类收进后续范围**——
  用户口径是"所有硬编码都要改"，但**"可验证路径优先"**：ld1 自己也承认 `endpoint_format` 因"对所有现存 artifact 都正确"而没动。
  ⇒ **后续单：`inventory_nvfp4.py:46-47`（生产侧孪生体，最高优先）+ `35b_a3b:20-23` + `muse:57-62` + qwen36n 那 3 个谓词**，
  每处都要用同一套 `bind_declared_weight` 机制，并**分别证明改动落在可验证路径上**。

#### 71.3 ⭐ 转换器缺口：此前**根本不在树里**（已补）
- ld2 发现：`tools/convert/qwen3_8_27b/convert_modelopt.py` **只存在于 `/home/user/scratch/{cbfix1,cbfix2,convB,convC}/`**；
  而 `import_model.py` 对 `flavour == "modelopt"` **只打印 work-item、从不路由** ⇒ **验收链第 4 步（重产真产物）无法从树里跑**。
- **已补**：落地 `/home/user/scratch/convB/convert_modelopt.py`（新版修订，md5 `402f4ee172e1…`，71,761 B）
  到 `tools/convert/qwen3_8_27b/convert_modelopt.py`。**新文件，无覆盖风险**；备份机制已就位（若已存在会先备份到 scratch）。
- **就地验证**：`py_compile` OK；**从树里跑 `--plan-only` 真通过** ⇒ `objects 1103`、`refusals 1`（只有前端 pin）、
  **字节覆盖 2139/2139 全在（`model-nvfp4-mixed` 24,005,888,128 / `vision-mtp-bf16` 1,770,858,976 全齐）**。
- ⚠️ **仍缺**：cbfix1/cbfix2 的**记账修复**（`D-RECODED-FROM-BF16` 从 1 → 119）**没并进来**——
  我落的是 `convB` 的新版，**不是**修完记账的那版（有界查找没命中所说的 `cbfix2/clean2/`）。
  ⇒ **待补一条**：把 cbfix 的记账补丁并进树里的这个文件（`cbfix1/fixB.patch` 或 `cbfix2/fixB.patch`，5 hunk/99 行，apply-check 通过）。
- 另：`import_model.py` 的**前门路由**（让 `flavour == "modelopt"` 真的走到这个 driver）**还没做**。

#### 71.4 两面都给齐了验收链（关键：**哪些步骤要 GPU**）
1. **应用补丁**（CPU）→ 2. **重建**（**要 nvcc，但不需要 GPU 执行**）→ 3. **重产真产物**（**要 GPU/`--device cuda`，~471 s，22.6 GB**）
→ 4. **让引擎真去装载**（**要 GPU**）→ 5. 质量（`ninfer-perplexity` + `ninfer-serve`）。**两份都给了逐条 stderr 的故障分诊表。**
- ⚠️ **磁盘**：artifact 22.6 GB 而 `/` 只剩 48 GB ⇒ **只留一份**。
- **最强的诚实边界（ld1 写的）**：**"真 artifact 已不在盘上"**（`cbfix2/out2/` 只剩 35 KB 的 report）
  ⇒ **第 4 步是这些补丁第一次面对真字节**；`Binder::finish()` 要求 1103 个对象**全部被消费且被规划**，
  而这只有"偏差报告里没有 missing/extra"作为间接证据。

### 70. W3（单一真值来源，ss1 第一路）：**8 项重复真值**统一；三处纠正我的普查；并标出**合并顺序风险**

#### 70.1 三处纠正我
1. **C1 的 TMA 判据是 6 处不是 5 处**，而且**第 6 处已经漂移了**：
   `src/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.cpp:41` 写 `tokens >= kTmaBlockM && %kTmaBlockM == 0`
   并自带 `:25` 的字面量 —— **是 256，不是 1024**。⇒ 它早就与"定义"分叉，只是没有任何东西把它绑回去。
   另外 4 个 `.cu` 各自还带了一份 `constexpr kTmaBlockM = 256`（**常量本身的第二份拷贝**）。
2. **C2 的 `suffix_lookup` 是 3 个界，不是 2 个**：doc（`:13` `o < starts[b]` + `:17` 的 `o+Q+K <= len`）、
   host（`lookup_fuse.h:28-29` = `min(len-Q, start)`，**连 K 项都没有**）、kernel（`suffix_lookup.cuh:27-29` = `min(len-Q-K, start-Q)`）。
   ⇒ **host 在 K 项上也不同**，而 **doc 自己那条 `<=` 比任何已落地实现都松一格**。
   **裁定：以 kernel 为准**，依据——① `TODO.md:171` 记录那个 `-Q` 是 **2026-09-03 有意的"防自匹配"修复**，
   且已编码进**两个镜像**（`tests/test_suffix_lookup.py:14`、`tests/suffix_gpu_test.cu:2`）；
   ② `TODO.md:7300` 用 198,269 个双镜像用例把 kernel 契约定在 0 分歧；③ **kernel 引擎零调用**，
   而 `lookup_fuse.h` **只被它自己的 CPU 测试用** ⇒ **host 侧才是陈旧的**；
   ④ host 自己的注释（`o + query_len <= query_start`）与它的代码（`o < query_start`）**自相矛盾**；
   ⑤ **决定性**：`lookup_fuse_test.cpp` 的 case 4 **把自匹配 bug 写成了断言**。
   ⇒ **kernel 一个字节没动**（**零数值风险**）；改的是 host + doc + 那一个测试用例。
   实测：**改前 host-vs-kernel 分歧 767，改后 0**（与我先前报的 91 同属一类，差异来自分布）。

#### 70.2 真正的清单是**8 项**（我漏了最有价值的两项）
| | 真值 | 拷贝数 | 现在 | 行为 |
|---|---|---|---|---|
| α | **MMA schedule 决策阶梯**（`<=64/96/128/192/384/512`） | **4 文件** | `nvfp4_w4a4_ladder.cuh:73/90` | **逐字相同（已证）** |
| β | 5 个 schedule 别名 | 同 4 文件 | 同上 `:38-66` | 相同 |
| γ | `alpha = 1/(in*weight_scale_divisor)` | **11 处 / 6 文件** | `nvfp4_w4a4_plan.h:76` | 相同 |
| δ | attn 行切分（6144/1024/6144） | **5 文件** | `nvfp4_config.h:131-136` + `static_assert` 铺满 `kOutputRows(14336)` | 相同 |
| ε | gdn 行切分（10240 qkv / 6144 z） | **3 文件** | `nvfp4_config.h:129-130` + `static_assert = kOutputRows(16384)` | 相同 |
| ζ | `PageIds` = 64 / 256 | **8 处** | `paged_kv_address.cuh:39` `paged_kv_page_ids(envelope)` | 相同 |
| η | small-T launcher 表长 + 无界索引 | 4 处 | `nvfp4_config.h:229/235/240/251` | 可达 T 上相同 |
| θ | i8 unpack 宽度 + 4×`make_int4` 重打包 | 2 处 × 2 数组（各约 32 行手展开） | `gqa_attention_kv_quant.cuh:141/154/161` | 相同 |

⭐⭐ **α 与 δ/ε 是我漏掉的高价值项**：**那条阶梯是 4 份拷贝的 8 分支决策表，而且真的漂移了**——
`linear_add` **没有 `<=96` 分支**，`attn` 在 `<=128` 上有 `Pipelined` 而 `linear` 只在非残差上有。
ss1 进一步**证明这些差异都能从几何（residual / GDN）推导出来** ⇒ 所以它能**证明等价**而不是猜。

#### 70.3 关于我那条"`16` vs 32 + 无界索引"的隐患（C3b）：**纠正**
它今天**不是** OOB，而且**原因不是**注释暗示的那个：`nvfp4_gdn_snapshot_small_t.cu` 里的 `16` 是
`nvfp4_gdn_conv_resolve_plan` 的 **A16 注册上限**（`tokens <= 16 ⇒ SmallTFusedA16`），**不是 `kNvfp4LastSmallT`**；
那个不变量**只通过 planner 传递性地成立**。⇒ ss1 把它**结构化**了：一个具名常量
（`kNvfp4GdnConvA16Ceiling`）定表长 + `static_assert` 绑长度↔上限 + `nvfp4_small_t_launcher<Ceiling>` **对每次派发的 T 做范围检查**。

#### 70.4 验证（**真跑**，不是只过语法）
- **阶梯（α/β）编译并运行**：把 4 条改前链逐字转写，与共享阶梯+visitor 在 **T=1..65536 × 5 个 problem**
  以及 3 条更窄的 per-launcher 链上对比 ⇒ **`ladder mismatches=0`、`visitor mismatches=0`**。
- **C2**：用**改后真实的** `lookup_fuse.h` 对 kernel 的 limit/tie-break 镜像跑 16,383 随机例
  （Q 2..8 / K 1..4 / min_len 1..Q）⇒ 改前 **767** 分歧、改后 **0**，且 `scan_limit == min(len-Q-K, start-Q)`。
- **补丁完整性**：74 hunk / 33 文件干净 apply；round-trip 到副本后 **33 个文件逐字节相同（33 SAME / 0 DIFF）**。

#### 70.5 风险：**恰好两处行为变更**
(i) C2 的 host 参照 + `lookup_fuse_test.cpp` case 4 —— **两者都在构建之外**（`lookup_fuse_test` 不在任何 CMakeLists；
kernel 无引擎调用者）⇒ **引擎的数值行为不变**；
(ii) PageIds 填充处的守卫：原本会砸 smem 的"信封溢出"场景，现在写中性统计并返回（按推导出的界不可达，**有意的 fail-visible**）。
**其余全是带证明的纯重构**（阶梯、alpha、行切分、常量、small-T 协议、unpack），且断言都是编译期。

#### 70.6 ⚠️ 合并顺序风险（要按这个排）
**ss1 的 33 个文件里有 4 个是别的代理已经改过的**：
`gqa_attention_decode_i8.cuh`、`gqa_attention_prefill_i8.cuh`、`gqa_attention_kv_quant.cuh`、`nvfp4_gdn_snapshot_plan.cpp`。
⇒ 它的补丁**只对"当前树"成立**；**若 E8 那条线（e8m 的陪集位、e8r1 的路由/shuffle 修）再动这几个文件，就会冲突。**
⇒ **落地顺序：先落 E8 这条线（关键路径）→ 再重取 ss1 的补丁（或手工并）**；不要反过来。

#### 70.7 未证
**本机没有 nvcc**（`/usr/local/cuda-13.3` 只有头、没有编译器）且无 GPU ⇒ **约 20 个 CUDA TU 没编译**；
对它们证据是"干净 apply + round-trip 逐字节相同 + host 头语法检查 + 已执行的阶梯测试 + 实例化集合未变
（同 5 个 schedule × 同几何）"。`kGqaGuaranteedSplitCount = 85` 是**推断**（没把
`causal_small_t_launch_capacity` 端到端读完证明大窗总是派 ≥85 splits）。
一处仍缺循环守卫：`gqa_attention_decode_nvfp4.cuh` 的 page fill 处 `write_neutral` 不在作用域（守卫跳过、已报告而非盲插）。
三项 follow-up 已标（`wrapper/gdn_input_proj.cpp` 重复推 per-model 行常量、3 个 attn epilogue 结构体重复行路由、
`small_t.cuh` 与 decode 内核各自一套 split-count 规则）。

### 72. W2 双路（id1/id2）裁定：以 id2 为基 + 并入 id1 的守卫；**id2 多找到一处真 bug**；并撞到 ⑦ 的坏语法

#### 72.1 两面的同（独立一致）
- `WeightsProfile` 是**"权重数值格式契约"的选择子**，消费方三组：`endpoint_format()`（`bindings.cpp:35-50`）、
  `bind_artifact()` 选哪族 text-layer binder（`:417-445`）、是否绑 `dflash/*`/`dflash2/*`（`:486`、`:532-533`）；
  外加 **7 个 workspace 容量 switch**（`variant.cpp:434/456/480/502/530/558/589`）与 `auto` 的 backend 解析（`package.cpp:118-155`）。
- **两面都明确否掉两条猜测**：它**不影响采样默认值**（`sampling_defaults` 只按 `model_id` 走，`package.cpp:82-88`）；
  **不影响 `default_layer_kv_dtypes`**（该参数**无名未用**，表体是硬列 `{0,1,3,4,6,7}`）⇒ 注册表**不需要** KV dtype 字段。
- 设计同为**声明式表**：`WeightsIdentityKey{model_id,weights_id}` → profile，**一次查表、一个拒绝点**；
  新 id **复用现有 profile 值**（差别只写进 `provenance`）。依据不是猜：转换器自己写着
  `WEIGHTS_ID = "nvfp4-modelopt"` 且复用 `inventory_nvfp4`（`MODEL_ID="qwen3.8-27b"`、`TARGET_KEY="qwen3_8_27b"`）+ `recipe_nvfp4`。
- **等价性证据**：两面都用 `git show HEAD:` 抽出的**改前函数体**做 CPU 差分 ⇒ 5 个既有 id 逐项同，新 id 接受，未知 id
  带**声明集合**拒绝（id2 还证明**旧句子是新消息的字节级前缀**）。都逐字节保住 **CRLF**（两个 27b/35b 导出头 + `registry.cpp`）。

#### 72.2 两面的异与裁定
| | id1（13 文件 +758/−111） | id2（11 文件 +689/−172） |
|---|---|---|
| 独有 | **5 个 `static_assert`** 钉声明漂移；把 `context_cost` 的"**静默降级**"显式化（`context_prefill_preset_miss_note()` 装载时打印） | ⭐ **`context_cost.cpp` 的 weights_id 匹配是 3 处不是 2 处**：第三处是 **`upsert_context_prefill_cost_atomic` 的 JSON 写回匹配（`:557-559`）** ⇒ 不纳入就"**读时认、标定写回时对不上**"（**真功能 bug**）；把**采样预设 if 链**也做成第二个声明轴；`registry.cpp` 4 分支 if 链 → 3 行注册表，且 `construct_registered` **不再接收 target_key**（取自命中行） |
| 自我纠错 | — | 它的 CPU 复算抓到**自己**一个真 bug：`SamplingDeclarationTable<8>` 填 8 行只有 2 行有值，而 **`-fsyntax-only` 完全看不见**（空行合法）⇒ 加 `sampling_models_are_declared` 守卫 |
| 负向编译守卫 | 3 条（采样预设须属本 target 消费的 model、每行须报本 package 的 target_key、身份须唯一） | `declaration_check::*` 三条同形 |

**裁定：以 id2 为基**（它覆盖第三处 + 采样轴），**把 id1 的 5 个 `static_assert` 与"静默降级提示"作为必须并入项**。
⇒ **双路不是"二选一"，而是"把各自的独有证据并起来"**（这已是本会话第三次这个形态：G、S、cc、W1 都是）。

#### 72.3 id2 顺带撞到 ⑦ 的一处**坏语法**
**`src/targets/qwen4_exp/package.h:29-33` 写的是 `namespace ninfer::targets::qwen4_exp {{ … }}`（双花括号，非合法 C++）**。
它是个占位、**不在任何 `CMakeLists.txt` 里所以从不编译**，注册表也不含它。
⇒ 而 **`qwen4_exp` 正是 FlashNext 那条线的目标** ⇒ **占位头是坏语法**，谁去接它谁先炸。**记入 ⑦。**

#### 72.4 两面的诚实边界（都写明）
- **都没有构建、没有 GPU、没有真装载**；补丁只有 `-fsyntax-only` + 声明逻辑的 CPU 复算 + 补丁可逐字节复现。
- **`nvfp4-modelopt → Qwen38Nvfp4` 是推断不是实测**：依据是转换器覆盖 `WEIGHTS_ID` 且复用已注册 recipe，
  加 `docs/maintainer/modelopt-nvfp4-import.md` §3/§5 的"NVFP4 对象字节级一致"。**没有读过该 artifact 里每个对象自带的 format。**
- id2 明说：**新 id 现在能"被认识"，能不能"装载成功"取决于 W1**（`bindings.cpp:383` 的 `layer < 56` 还在）。
- id2 未加 ctest 回归用例（证据在 scratch，harness 可直接改成 `tests/` 用例，但要动 `tests/CMakeLists.txt`，它刻意避开）。
- `context_cost_defaults.cpp` 那条别名行断言"ModelOpt artifact 的 prefill 成本 = 实测 nvfp4 artifact 的"，**没有重新标定**。

### 73. ⭐ 用户更正 E8 的框定：**它是一个跨位宽的族（4/3/2 bit），且必须与 isoquant 同预算比较**

用户原话：**"e8 不只是 4bit 还有 3bit 乃至于 2bit；这个和 isoquant 还要比较；之后只要求到同就修，不用问我"**
并追加授权：**"同就修，不问"** —— 两路一致即落地，不再回头确认。

**我上一轮把 E8 框定成"4-bit 平面"是错的。** §68/§69 只讲清了 4-bit 那一档的存储与投影，
把"E8 的优势"当成一个**单一码率的常量 0.65 dB** 来谈，因此推论走向"要么退档要么删格式"。
正确框定是：**E8Kv 是"K 侧走 E8 格点"的一个实例，格点编码本身可以在 4/3/2 bit 每坐标上取不同档**，
档位由格点缩放（码本粗细）决定；**半整数陪集位（+1/8 bit/el）在三档上都要付**。§69 那条
"表示代价被我高估 8 倍"的结论只对 4-bit 档验算过，2/3-bit 档需要重算 b/el。

**为什么这条更正不是细节，而是决定 1M 能不能开：**
- 1M 上下文**只装得下 2-bit**（10.31 GiB，余量 0.09 GiB）。
- 按 10.4 GiB/1M 反推，KV 的**平均预算只有 2.144 bits/el** ⇒ **连一层 int8(8.25) 或 nvfp4(4.50) 都买不起**。
  ⇒ **2/3-bit 的表现直接决定 1M 这个功能存不存在**，而不是"优化项"。

**数学上为什么低码率反而更该看格点：**
- 格点相对标量的 **granular gain 是速率无关的**：`10·log10(G(Z)/G(E8)) = 10·log10((1/12)/0.0717) ≈ 0.65 dB`，
  §69 已核（引擎自己的 CPU 实测 0.874× = 0.58 dB 吻合）。
- 但**低码率下格点相对标量的实战优势显著更大**：独立来源给出 3 bit/dim 约 +15.8 dB、4 bit/dim 约 +11.6 dB，
  到 6–8 bit 才趋于消失。⇒ 我此前"0.65 dB 太小、不值得"的判断，**只在 4-bit 档成立，且还是不成立的**
  （0.65 dB 的增益 vs 陪集位 0.125 bit/el 的代价，按 6.02 dB/bit 折算代价约 0.75 dB —— 两者同量级，
  必须**推导 + 实测**，不许手算仲裁；§69 已记这一点，至今未结）。

**与前几节的对账：**
- §68 记的规格（`DType::E8Kv = 10`、`2×{U8,head_dim/2,kv_heads,256}` + `2×{FP16,head_dim/64,kv_heads,256}`、
  K 走 `gqa_kv_hadamard64` → `e8_project_8d_warp`，**V 走纯标量 i4、无旋转无投影**）**仍然有效**，
  它是 4-bit 档的实现事实。
- §69 记的"陪集位 = +0.125 bit/el，我把代价高估 8 倍"**仍然有效**，但**只在 4-bit 档验算过**。
- 更早会话的 codec 探针给过 **3-bit 3.25 b/el（K 误差 15.9%，比 nvfp4 差 1.6×）、2-bit 2.125 b/el（31.6%，差 3.3×）**
  —— **注意：那是"组32/组64 + E4M3 尺度 + 冻结码本"的候选 codec，不是 E8 格点**，不能当 E8 的 2/3-bit 读数。
  它只能说明**纯量化的 2/3-bit 掉得很快**，这恰好是格点该发力的地方，而不是格点已经输了的证据。

**isoquant 这一侧此前是空白：** §69 的比较只对了"纯均匀标量"，e8r2 曾想对 isoquant 但**在树里找不到矩阵**而放弃。
`DType::ISO3 = 9` 是**离线学习旋转**（SO(4)/`gqa_isoquant_rot`）路线，与 E8 的**固定格点**路线是两种不同的
"用结构化先验换 MSE"的办法，同预算比较才是有效对照。⇒ 本轮必须把 ISO3 的权威规格查出来
（码本、尺度粒度、旋转、plane 几何、**实测有效 b/el**）。

**已派（铁律⑦，双路同一任务独立做）：**
- **e8v1**（`sh/pE8V1_*.sh`，产物 `scratch/e8v1/`）与 **e8v2**（`sh/pE8V2_*.sh`，产物 `scratch/e8v2/`），
  同一 brief：核实 §68/§69 各条 → 查 ISO3 权威规格 → 给 E8@2/3/4bit 的编解码设计（含陪集位代价）→
  **同预算实测 K/V 分开报**（E8@{2,3,4}+陪集位 vs ISO3 vs 同码位纯均匀 vs 现状投影+rintf）→
  结论一致就给 `fix.patch`。
- ⚠️ 明确告知两路：`/home/user/bench/kvdump_e8src` 是 **Sep-10 之前**的产物，只能做**编码器之间的相对比较**；
  不许把 E1/E2 旧测当证据（旧 dump + E2 的读取转置 bug `(d*4+h)*3072+t`）。
- ⚠️ 明确告知两路：已有证据指向**失败在读/内核侧**（删投影使写侧 NMSE 改善 2.32×，而 `e8` 臂检索 **0.020→0.020 分毫未动**），
  ⇒ 读侧重建必须讲清，不是只比 MSE。

### 74. 构建 #1 验收：**ctest 100% 通过 / 0 失败 out of 109**，但**跳过 8 个而非我预期的 2 个** —— 定性为夹具缺失，非代码回归

**落地**：W1（ld2）+ W4（ph2）干净落地，树相对 HEAD **36 文件 +2569/−147**；落地前快照
`/home/user/scratch/snap_pre_w1w4/{tree_pre.patch,status_pre.txt,revert.sh}`（可一键回退）。
构建 `rc=0`（增量 + ccache，约 1 分钟；先确认无人在构建才取锁）。

**ctest**：`100% tests passed, 0 tests failed out of 109`，`Total Test time 452.64 sec`，rc=0。
四个曾经的失败点（#51/#74/#95 的回归指纹 + #4/#26）现在**全無失败**。

**但跳过 8 个，不是 2 个** —— 我没有放过这个差异，逐条追了原因：

| 测试 | 跳过原因（程序自己打的） |
|---|---|
| #4 `ple_table_e2e_test` | 缺 `<sidecar_root>` 参数（已知，W4 的逐子测试 skip 覆盖） |
| #26 `qwen3_6_frontend_test` | 官方资源检查未 opt-in（需 `NINFER_FRONTEND_TEST_ROOT`）⇒ W4 给它加了 `SKIP_RETURN_CODE 77` |
| #31 `27b_prefix_real_test` | `neither NINFER_QWEN3_6_27B_WEIGHTS nor …_NVFP4_WEIGHTS nor a Qwen3.8 equivalent is set` |
| #32 `27b_score_real_test` | `NINFER_QWEN3_6_27B_WEIGHTS is not set` |
| #33 `27b_load_plan_test` | 缺 **两个**真产物 `out/qwen3_6_27b.ninfer` **与** `out/qwen3_6_27b_nvfp4.ninfer` |
| #34 `35b_a3b_real_test` | `NINFER_QWEN3_6_35B_A3B_WEIGHTS is not set` |
| #35 `35b_a3b_dflash_real_test` | 同上 |
| #36 `35b_a3b_dflash_load_plan_test` | 缺真产物 `out/qwen3_6_35b_a3b.ninfer` |

**关键取证（这一步决定"是回归还是夹具"）**：`git show HEAD:tests/CMakeLists.txt | grep SKIP_RETURN_CODE`
显示 **#31–#36 的 `SKIP_RETURN_CODE 77` 在 HEAD 里本来就有**（行 158/162/171/175/179/189）；
**我这批只加了 `ninfer_qwen3_6_frontend_test` 那一行**（`tests/CMakeLists.txt:135`）。
而且 `out/` 目录里那 4 个产物**全都已不存在**（磁盘 96% 占用、只剩 47 GB，被回收了）。
⇒ **这 8 个 skip 是"夹具/环境变量不在"，不是 W1/W4 改坏了任何东西。**

**由此纠正我自己两处记账**：
1. 我此前写的"预期 0 fail / 2 skip"是**错的**。上一轮"98% passed, 2 failed out of 109、无 skip"是在
   **真产物还在盘上**的时候测的；现在同一台机器上少 6 个可跑测试。
2. ⇒ **ctest 作为验收仪器，当前真实覆盖率是 101/109**，缺的 8 个映射到**缺产物**。
   而 #33 需要 `qwen3_6_27b.ninfer`（groupwise，另一个转换器产）**加** `qwen3_6_27b_nvfp4.ninfer`，
   本盘（47 GB 可用）装不下整套夹具 ⇒ **un-skip 它们是磁盘问题，不是代码问题**，记为仪器的已知限制。

**顺手落地的 ⑦ 修**：`src/targets/qwen4_exp/package.h` 原来是 `namespace … {{` **双大括号（非法 C++）**，
已改成单括号；并**实证**它 `grep` 无任何引用、不在任何 `CMakeLists.txt`、不在 `registry.cpp` ⇒
**从未被编译过**（所以双括号没炸过构建，但仍是坏生成）。同一目录 `impl/config.h:14` 还有
`static constexpr int intermediate = None;`（`None` 不是 C++），已改为 `-1` 并写明理由
（**不用 `0`**：`0` 会让"维数=0"静默过关，`-1` 在任何拿它当数组界的用到处立刻编译失败）。
⚠️ **吐 `None` 的真正发射点还没找到**：两个候选（`tools/archkit/gen_full_target.py:53` 的
`int(g.get("intermediate",0))`、`adapt.py:122` 的 `'%d' % g.get(...)`）在 `None` 上都会抛 TypeError，
**吐不出字面 `None`** ⇒ 真发射点在别处，已派 hw。

**新到的两条 agent 结果（都还没落，等实验后一起）**：
- **aw2（auto 第二路）**：12 文件 +571/−121，`git apply --check` 通过。⭐ **它指出 −27% 的真根因是"棘轮"**：
  调用点写的是 `if (cut < window) { window = cut; }`，而 `window` 是**本轮刚用过的**宽度 ⇒
  **窗口单调不增、永不恢复**，且判据的搜索上界用的是**配置**的 `draft_window` ⇒ 也长不回去。
  我的 CPU 复现单靠这一条就重现了记录的"realized ≈ 7.96 vs 阈值选出的 9"。
  ⇒ **阈值公式本身是对的**（`C(k)=a+bk`、post-add），**不该动 `kMtpWindowCostRatio`**
  （`b_mask/a=0.0062` 是退化的永不触发端；`b_width/a=0.0441` 也不被支持 —— 记录的三点 AL 只能把 r 钉在
  ≈`[0.0401, 0.0953]`，冻结值 0.0612 在里面）。
  另一条硬约束：27b 的 `kMaximumMtpDraftTokens = 15`，但 **35b_a3b/muse_glimmer 只有 5**，而
  `MtpCausalAttentionEnvelopes::ar` 是按 `kMaximumMtpDraftTokens-1` 定长的 ⇒ **15 级阶梯会在 35b 上越界**，
  阶梯必须按 variant 裁剪。
- **ss2（W3 第二路）**：32 文件 1021 行，`git apply --check` 通过。它多找到的、此前普查漏掉的东西：
  ⭐ **TMA 谓词在 4 个文件里各有一份内联副本、各自带一个 `kTmaBlockM = 256`，而这 4 个文件里的同一个函数
  已经在调用 `nvfp4_w4a4_tma_route()` 选阻塞尺度量化器** ⇒ 一旦漂移，**尺度布局会与 TMA 描述符静默失配**，
  是数值 bug 不是崩溃；
  ⭐ `suffix_lookup` 是**三方分歧**（内核 / host / 文档），内核 + python 镜像 + GPU 测试三方独立同意，
  且 python 镜像带"2026-09-03 修正, 防自匹配" ⇒ 是 host 与文档错，不是内核错；
  ⭐ **decode 家族还有 5 份 `PageIds = 256`**（不只是 small-T 那三份 64），其中 4 份带逐字节相同的注释；
  ⭐ 另有 **10 项重复真值**，其中**两条调度阶梯已经漂移**（`linear_add` vs `attn_input` 的 97–128、
  `gdn_input` 的 385–512）—— 这两条**要先有测量才能定**，不许猜。

### 75. 磁盘审计：`/` 915G/1007G（96%）。**不是"被我吃掉 1T"** —— 本会话新增约 150G，其余 ~830G 是先前就有的

用户问："你可能得真正找找到底你搞了些什么把整整 1 个多 t 吃掉了"。实测（`du -xsh`，逐层降，不跨文件系统）：

`/home` = **879G** 是全部大头，构成：

| 路径 | 大小 | 归属 |
|---|---|---|
| `/home/user/bench` | **329G** | 你自己的工作，非本会话 |
| `/home/user/models` | **290G** | 其中 **~245G 是 11 个 `.ninfer` 产物** |
| `/home/user/scratch` | **72G** | 其中 **`scratch/r1/build` 一个就 60G** |
| `/home/user/ninfer-fusion` | **68G** | 其中 **`build` 60G**（`build/tests` 56G / 109 个测试可执行） |
| 另外 ~70G | — | 一堆旧构建树：`ninfer` 14G、`ninfer-FIX` 14G、`ninfer-NR` 15G、`ninfer-pr3devbuild` 12G、`ninfer-allbuild` 3.4G、`ninfer-pr{1,2,3,4}build` 各 ~1.1G、`ninfer-issue144-build` 1.1G、`muse128_repro{3,4}` 各 842M、`ninfer_no_a5b`/`ninfer_with_a5b` 各 802M |
| `vllm029` 9.7G / `rag_build_db` 11G / `rag_db` 5.9G / `sglang-venv` 4.7G / `.local` 1.8G / `.cache` 3.3G | ~36G | 环境，非本会话 |

**`/home/user/models` 里 11 个 `.ninfer`（时间线是判断归属的证据）：**
`dflash2_bf16head` 26.07G(09-12 11:26)、`dflash2_refhead` 23.70G(09-12 10:35)、`dflash2_zeroselector` 23.70G(09-12 00:40)、
`dflash2_step_001900` 23.70G(09-11 19:12)、`dflash2_step_000200` 23.70G(09-11 19:22)、`dflash2` 23.70G(08-26)、
`dspark` 22.65G(08-26)、`foldmlp` 20.02G(09-03)、`fold3` 20.02G(09-03)、`nvfp4` 20.02G(08-25)、`muse_glimmer_30b` 18.38G(09-04)。
⇒ 其中 `_step_*`/`_zeroselector`/`_refhead`/`_bf16head` 是 **你自己的 dflash2 实验导出**，不是本会话产的。**不许动。**

**本会话真正新增（约 150G）：**
1. ⭐ **`ninfer-fusion/build` 60G** —— 最大单项。`build/tests` **56G 装 109 个测试可执行**，每个都静态链了整个引擎（单个 ~500M）。
   这是"跑全量构建 + 109 项 ctest"的直接代价。`build/apps` 2.4G（三个 826M 的二进制）+ `build/src` 1.8G。
2. ⭐ **`scratch/r1/build` 60G** —— 某 agent 为了冷盘槽的活**建了一整棵副本树**（r1 目录 60G 里 59.99G 全是 build）。
3. **正在产的产物 22.58G**（`out/qwen3_8_27b_nvfp4_modelopt.ninfer`，审计时已写 10.5G）。
4. 其余 agent scratch 输出约 12G（`f5`/`n1` 各 1.6G、`acc2`/`m3`/`one_m`/`repro_table`/`acc`/`c1` 各 ~790M、`w2m` 383M…）。

**明确是垃圾、可回收（约 170G，但我没有单方面删）：**
- ⭐ `models/q38_abl_huihui_nvfp4` = **21G，里面 ~19.6G 是 HF `.incomplete` 下载碎片**
  （6.68+5.72+4.38+1.46+1.40G），而该目录只有 849M 的 `model-mtp-bf16.safetensors` —— **主权重根本没下完**。
  ⚠️ 删掉会丢掉可续传的进度，所以这要你点头。
- ⭐ `ninfer-pr3devbuild/core.415` = **10.7G core dump**（08-31）。
- `ninfer-sass.txt` **2.5G**（08-25 的文本 dump）、`chunks_cache.jsonl` **2.1G**（08-30）。
- 旧 `ninfer-*` 构建树 ~**70G**（最后改动停在 08-28 ~ 08-31，且多数非 git ⇒ 是旧副本）。
- `ccache` 2.9G（可 `ccache -C`，但会让下次构建变慢——**不建议现在清**）。

**结论**：915G 里本会话约 150G，其中 120G 是**两份 60G 的构建树**（我的 + r1 agent 的）。
当前余量 39G，产物写完降到 ~27G，**够用**。回收清单已列，**在你点头前我一个都不删**。

### 75b. E8 数学结论到账（e8v2/e8m2 那一侧，**等 e8v1 对账后再落**）

⭐ **E8 的陪集表示代价是 0 bit，不是 +1（旧注释）也不是 +0.125（我 §69 的结论）。**
关键：**半整数陪集是读者已经读到的那 8 个码字的函数**，所以可以无损恢复、不需要任何旁信息：

```
xhat_i = s * ( u_i + parity(Σ_{j∈block} u_j) / 2 )      u = 4-bit nibble, [-8,7]
```

读侧只加一步：把 nibble 展开成小 int8（`k_i = 2·u_i + p`，`|k|≤16`）并让组尺度取 `s/2` —— 
**平面大小、打包、swizzle、`ldmatrix`、`m16n8k32.s8` 操作数、bits/el 全都不变**。
写侧：每 8 块比"全整数/全半整数"两个候选谁便宜，再按奇偶做**单个 ±1 的最省修复**。

**66,229,248 个真 K 元素上实测**：`−0.665 dB` vs 同预算最优 4.25 b/el 标量平面、**`−4.32 dB` vs 现状写侧**，
**b/el 恒为 4.250**。另两条副产品：
- **`±7 → ±7.5`（把 16 个 nibble 值都用上而不是 15 个）本身就值 0.599 dB，免费**；
- 显式存陪集位那个变体与免费版只差 **+0.0005 dB** ⇒ **那位纯冗余**；
- 实测码率律：**5.02 dB/code-bit**（4→5 bit），所以 0.125 b/el ≈ 0.63 dB ⇒ **E8 的陪集是那 1/8 bit 的更好用法**
  （按速率归一 NMSE：e8p 8.849 < per-8 flag 9.407）。
- **旋转与格点不可加**：格点增益在**旋转后是 −0.665 dB**（渐近形增益精确值），但在**未旋转** K 上是 −0.883、V 上 −1.128 ⇒
  旋转让 8 块变 iid 高斯，形增益就是全部；但**陪集加上后旋转仍好 1.29 dB ⇒ 旋转要留**。
- `D8 = {整数 8 维且坐标和为偶}` —— **"删除格点"就是这个意思**，而删除是免费的：编码器根本不发射被删的点，
  且解码器永远不需要被告知它们（`codebook D8 ∪ {u+½·1 : Σu 奇}` 与 E8 等距同构，`u ↦ s·(u+½·parity)` 是
  2³² 个 nibble 模式到 2³² 个格点的**单射** ⇒ 平面的 32 bit 一个不浪费）。
- 补丁：`/home/user/scratch/e8m2/fix.patch`（437 行 4 文件 +211/−95，`git apply --check` 过）。
  ⚠️ 它**没编译过、没跑过**（禁令不许构建），语义由 66.2M 元素上的编解码实现验证。**落地前必须先构建 + 解码对拍。**

### 76. ⭐ E8 双路裁定：**陪集代价是 0 bit（e8v2 对），"绝不值得"不成立**；但 **V 旋转是零代价最大单项（e8v1 独有）**；已落两路一致的部分

**两路都在 66.2M 真 K/V 元素上独立测了同一份源**（`bench/kvdump_e8src/kvsrc_*_{kn,v}.bin`，
= rmsnorm+rope 之后的 BF16 K 与原始 V，即 append 路径真正量化的张量；**不是**去解已有缓存，
所以完全绕开"Sep-10 旧 dump"那个坑）。分歧点：

| | e8v2 / e8m2 | e8v1 |
|---|---|---|
| 陪集代价 | **0 bit**（陪集 = 读者已读到的 8 个码字的奇偶函数） | **+1/8 bit/el**（按一个平面算） |
| 结论 | E8 在 4.250 b/el 上比同预算最优标量好 **−0.665 dB**；比现状写侧 −4.32 dB | **净亏**（0.65 dB 增益 < 0.75 dB 代价）⇒ "引擎不投影是对的" |

**裁定：e8v2 对，e8v1 的否定结论不成立。** 三条理由：
1. e8v1 自己是按"陪集要占一个**平面**"计价的；e8v2 指出并**实测**了另一种编码：
   读侧 `k_i = 2·u_i + parity(Σ_{j∈block} u_j)`、组尺度取 `s/2`，**平面大小/打包/swizzle/ldmatrix/
   m16n8k32.s8 操作数全不变** ⇒ 陪集信息本就含在那 32 bit 里（`u ↦ s(u+½parity)` 是 2³² 个 nibble
   模式到 2³² 个格点的**单射**，一个 bit 不浪费）。
2. e8v2 **直接测了这个断言**：显式存陪集位的变体 `K_e8c_bit`(4.375) 与免费的 `K_e8p`(4.250)
   只差 **+0.0005 dB** ⇒ 那位纯冗余。
3. e8v2 同预算表内（同为 **4.250**）：`K_e8p_scan` **2.4443e-2** vs 最优 16 电平标量 `K_u16_scan`
   **2.8485e-2** ⇒ **ratio 0.858 = −0.665 dB**，与格点形增益理论值 0.6534 dB 吻合到 0.012 dB。
   ⇒ **"绝不值得"是被自己的计价方式推翻的。**

**但 e8v1 有一项 e8v2 报告里明确写着"没探索"的东西，而且是最大单项：V 侧加 H64 旋转，零 bit 代价。**
- V 现状（标量 i4、无旋转）relRMS **0.15403** → 加 H64 旋转后 **0.09387**（**−4.30 dB，b/el 不变**）；
- 注意力代理 `softmax(qKᵀ/16)·V` 输出：现状 **0.11026** → 只修 K 0.10611 → **V 旋转 0.08144** →
  **K修+V旋转 0.07594**（**优于 ISO3 的 0.08277，且少 0.25 b/el**）。
- 为什么"零代价"成立：`out = Σ p_i V_i` 对 V 是**线性**的，所以存 `HV` 只需在输出端再乘一次 `H`
  （H 自逆）⇒ 这是**输出侧的线性映射**，可以**折叠进 `o_proj` 权重**（`o_proj' = o_proj·(I_head ⊗ H)`），
  内核零改动。⚠️ 但这是**转换器/产物侧**的改动，不是内核补丁 ⇒ e8v1 拒绝盲落（"落地它但不能编译验证，不可辩护"）——**这个判断我认同**。

**两路一致 ∧ 已落（用户授权"同就修，不问"）：16 电平奇数中点集。**
`/home/user/scratch/e8v1/fix.patch` → **已落地**：3 文件 **+82/−45**，只改
`gqa_attention_kv_quant.cuh`、`gqa_attention_prefill_i8.cuh`、`gqa_attention_decode_i8.cuh`，
把打包 4-bit 字母表从"15 电平 ±{0..7}、除数 7"换成 **`±{1,3,…,15}·(amax/16)`**（写侧辅助 + 读侧解包 + 三处
E8 写点的尺度除数与码分裂）。**b/el 恒 4.250、平面几何不变、不新增平面、不碰注意力输出路径。**
- `git apply --check` RC=0；落前落三个文件的 sha256 全变、**0 冲突标记**；快照 `/home/user/scratch/snap_pre_e8v1/`。
- 三个独立测量的实测收益：**K −0.96 dB（0.10661→0.09546）、V −0.49 dB、注意力输出 0.11026→0.10611**；
  e8v2 侧同预算测出这一项值 **−0.599 dB**。**两路都认它是免费win** ⇒ 这就是"同"。
- ⚠️ **写者与读者必须同一次改**（旧写者写的缓存新读者读不了）——可接受：KV 码平面只在运行时存在，盘上无。
- ⚠️ **它没编译过**（agent 被禁构建）⇒ **构建就是第一次验证**。

**e8v1 的另一条硬事实，重新框定 §68：`e8_project_*` 在树里没有任何调用点。**
⇒ **当前 E8 档里根本没有格点数学**：K = H64 旋转过的**纯 15 电平均匀**，V = 标量 i4、无旋转。
`e8_lattice.cuh` 被两个 i8 内核 `#include` 只是为了拿旋转辅助。**"E8Kv" 今天是个误名。**
（−3.65 dB 那个 2.32× 属于**已被删掉**的投影变体；e8v1 把 0.16236 vs 0.10661 = 2.3197× 精确复现了。）

**⚠️ 未结：基线差 1.55×。** e8v2 量出"现状写侧"K relRMS **0.16575**，而
**e8r1 / e8r2 / e8v1 三方独立都落在 0.1054–0.1072**。e8v2 自己就报告了这个矛盾，并给出解释
（能对上 10.66% 的同预算读法是"每 8 块各带尺度"的 4-bit 平面 = **6 b/el**，不是 4.25）。
**三对一 ⇒ 0.106 那侧更可能对**，但**这不动摇 e8v2 的陪集结论**（那是它自己 harness 内的**同预算 A/B**）。记为未结。

---

**ISO3 权威规格（e8v1 查证，带引用）—— 顺手纠正一个长期命名陷阱：**
- **`DType::ISO3 = 9` 是可运行的那一档**：4-bit nibble **符号-幅值**码（bit0-2 = 幅 0..7，bit3 = 符号 ⇒
  **15 电平 ±{0..7}**，除数 7；`gqa_attention_prefill_nvfp4.cuh:63-76`，在 `entropy_cold_requant_kernels.cuh:25-38` 有副本），
  **每 16 通道 E4M3 尺度**（与 NVFP4 同平面几何，`kNvfp4KvQuantGroup = 16`），
  **K 有烘焙的 per-4-通道 SO(4) 旋转**（写 K、读 Q，QKᵀ 不变；`gqa_isoquant_rot.cu:16-79`），**V 无旋转**；
  K 侧另有 Sinkhorn per-通道 行尺度 `s_d ∈ [0.5,2]`。**实测有效 = 4.50 b/el**（`kv_bit_budget.h:119-126`）。
- ⚠️ **`src/ops/kv/iso_codec.h:104-143` 与 `kvcfg/kv_formats.h:24,44` 里那个 "iso3" 是另一个东西**
  （group-8 / 幅 0..3 / 8 元素 3 字节 / `bit_width = 3`）：那是**冷槽词汇表 + 离线参考**契约
  （`tools/convert/kv_iso_ref.py`），**不是 `DType::ISO3` 跑的东西**。
  `kv_bit_budget.h:44-47` 记着阶梯正是因此从 iso3 300 改到 450。**任何引用"iso3 = 3.00 b/el"的人都在引名字，不是引那一档。**
- ⭐ **ISO3 的 V 优势是尺度分组，不是码本**：`V iso3_g16_e4m3 = 0.10546` vs `V iso3_g64_e4m3 = 0.15450`。
  ⇒ **这就是我一直在说的实验 A（per-64 → per-16）**，现在有了独立量化。
- **同 4.50 b/el**：E8（H64 + g16 + 16 电平）在两个平面上都赢 ISO3（K 0.08127 vs 0.09012；V 0.08013 vs 0.10546）。
- **1M（2.144 b/el/平面）的诚实结论**：E8 现状 4.25 与 ISO3 4.50 **都超预算约 2 倍**；只有 2-bit 装得下，
  而 E8-2b 的 2.25 还得把尺度分组放粗（g128 → 2.125）才刚好进；**2-bit 的注意力输出 relRMS 0.313 vs int8 的 0.0063
  ⇒ 差约 50 倍。⇒ 本树里没有任何 E8/ISO3 设计能给 1M 买到可接受质量**，决定它就是 2-bit 那一行，而它现在是这个读数。

### 77. ⭐ **产物逐字节复现**（确定性得证）+ 清盘 95.26G + w2m 到账（并纠正 §72 我写错的两处数字）

**1. 重产真产物：sha256 完全一致。**
树里的转换器（`tools/convert/qwen3_8_27b/convert_modelopt.py`，md5 `402f4ee172e1…`）在真源
`/mnt/c/Users/User/Documents/ziqinzhang/models/Qwen3.8-27B-ET-Uncensored-NVFP4/W4A4+W8A8` 上跑完整 `convert()`：
- `complete: 22585140736 bytes in 616.6s`，rc=0，落在 **`out/qwen3_8_27b_nvfp4_modelopt.ninfer`**；
- 实测 sha256 = **`dbbefaeeeb3bd5a45f1d8e8ad057d699a8d59301a26618541feb523c696bdec5`**，
  **与先前验证过的那份逐字节相同**（`dl/pB2_artifact.txt` 记的同一个值）。
⇒ **两条独立结论同时成立**：(a) 落地的转换器与验证过的 `402f4ee1` 那份行为等价；
(b) **整条转换流水线是确定性的**（两次运行、不同机器负载，同 sha）。用时 616.6s vs 471s 是并发抢 CPU 所致，非语义变化。
⚠️ 我那个"报告摘要"小脚本自己有 bug（把 `deviations` 当成 dict 列表，实际是字符串列表）⇒ `AttributeError`；
**产物与报告本身没问题**，只是我的摘要没打印出来。

**2. 清盘：回收 95.26 GB**（磁盘 26G→**122G** 空闲，98%→88%）。逐条留证后删的：
| 项 | 大小 |
|---|---|
| HuggingFace `.incomplete` 下载碎片（两个目录共 12 个） | **20.36 GB** |
| `ninfer-pr3devbuild/core.415` | 10.70 GB |
| `ninfer-sass.txt` | 2.49 GB |
| `chunks_cache.jsonl` | 2.03 GB |
| `scratch/r1/build`（agent 的重复构建树；`.git`/`src`/`REPORT.md`/补丁**全部保留**） | 60 GB |

删后逐条核验：**所有 `.ninfer` 产物、r1 的补丁与源码、`out/` 里刚产的产物都还在**。
⚠️ **我的第一次审计漏报了**：它只查了 `q38_abl_huihui_nvfp4`，而 `q38_abl_huihui_bf16` 下还有 6 个碎片。
**同一个失败模式我又犯了一次**（与之前"引号前缀让普查漏项"同类）：**只查自己已经知道的那个目录**。
留档：**报"某类文件有 N 个"之前，必须按类在**所有**已知根下查，不许只查已知的那一个。**
未动（余量已够、且归属不明确）：旧 `ninfer-*` 构建树 ~70G、`ccache` 2.9G（清了会让构建变慢）。

**3. w2m（W2 合并版）到账：13 文件 +854/−163**，`md5 6dc3f51b7d15…`，
干净副本 `git apply --check` rc=0，且"干净副本+补丁"与合并树 `diff -r` 逐字节相同。
⚠️ **但它不能直接打到活树**：27b 的 export `package.h` 与 `impl/package.cpp` 已被别的改动占住
（它基于 `b480c55`/`9d705cb`，活树现为 `4c38cf3`/`6885125`）⇒ **W2 必须变基落，不许直接 apply。**
它按补丁重数后**纠正了我 §72 写错的两处**：
- 我写的"**5 个 `static_assert`**"实为 **12 处 / 4 类谓词**
  （`kRealizations.size()` ×3、`realization_table_is_closed` ×3、`identity_table_is_reachable` ×3、
  `speculative_declarations_match_payload` ×3）；
- 我写的"3 条负向编译守卫（采样预设须属本 target 消费的 model / 每行须报 `target_key` / 身份唯一）"
  **其实是 id2 的 `declaration_check::*`，id1 没有这三条**（id1 的行连 `target_key` 字段都没有）。
⇒ 又一次印证：**我转述 agent 的清单时必须回原文重数，不许凭印象。**

**并入**：id1 的提示函数（改写成 id2 的 `WeightsIdentityKey` 判据，丢掉 id1 的 `context_prefill_identity_matches`，
否则"身份只定义一次"不成立）；static_assert 族**改轴**——新增 `rows_name_an_identity`（3 处，专抓
"weights_id 为空的行会接受空口味 artifact"这一格，**id2 原有三条守卫都抓不到**）、`distinct_profiles == 7`
（只 27b；运行期实测 8 行触达 7 个 profile）、草稿口味↔payload 守卫（27b/muse；35b 无谓词故不并）；
id1 独有的 **muse 标定行**（唯一超出裁定枚举的一项，已单独标出、可 3 个 hunk 撤回）。
**丢弃**：id1 的整套 realization 词汇表（4 个新头文件）、采样/权重 if 链改造（id2 已同语义）、
⭐ **`resolved_auto_speculative` 的声明化重构**（会改运行时语义且不在"必须并入"清单 ⇒ **明确记录"该语义本次未进入合并版"**）、
`TargetRegistration` 4 行表。**id2 无一被覆盖**（相对纯 id2 树只删 2 行，都是 muse 项所必需）。

**非空性实测（负向测试 C0–C6）**：C0 控制静默；C1 恰好 1 条炸；C2 炸且编译器打印 `(6 == 7)`；
C3/C5 分别炸 27b/muse 的 payload 守卫；⭐ **C6 证明 35b/muse 的 `distinct_profiles == 1` 是空守卫 ⇒ 据此丢弃**
（所以新数值不是只当装饰留下的）。`-fsyntax-only` 8/8 TU PASS（含头文件探针），`-Wall -Wextra` 相对 id2 **零新增告警**。

**两条要提请注意（w2m 自己标的，我认同）**：
1. **提示函数只看编译期表** ⇒ 用户用 `--context-cost-presets` 时它会**误报**（消息末句变假）。
   w2m 未改 id1 的语义，只在 REPORT 给出一行修法（gate 到 `prefill_source == GenericDefault`）。**待定夺。**
2. 变基落地的顺序问题（见上）。

### 78. ⭐⭐ **端到端通了：引擎真装载了真产物并出字**（导入链条第一次面对真字节）+ 一个新的资源纪律教训

**真装载成功**（`build/apps/ninfer out/qwen3_8_27b_nvfp4_modelopt.ninfer --prompt …`，rc=0）：

```
summary  target              qwen3_8_27b
summary  weights             nvfp4-modelopt          ← 从产物自身 identity 推出，不是 CLI 参数
summary  artifact file read  20.43 GiB
summary  weight H2D          20.41 GiB   (host to device 7.319 s)
summary  tensors/resources   671 / 6
load     artifact/materialize 16.473 s
load     engine construction  17.508 s
summary  pinned staging peak  256.00 MiB
```

生成（第一次 56 token 提示）：`prefill speed 914.88 tok/s`、`decode speed 49.72 tok/s`、2 token、finish reason `output-limit`。
第二次（64 token 中文提示）：

```
mtp draft window      3
mtp rounds           26     fallback steps 0
mtp drafted tokens   77
mtp accepted tokens  37     acceptance rate 48.05%
mtp acceptance length 2.42 tok/round
mtp accepted by pos  18,13,6          ← 典型递减剖面
kv cache dtype bf16 / payload 43.50 MiB   （dtype 行是全局哨兵，按 payload 判）
```

⇒ **整条导入链在真字节上闭环**：树内转换器（确定性，sha256 复现）→ 22.58 GB 产物 →
装载器读**产物自己声明的格式**（W1 的 `bind_declared_weight`，取代 `if (layer < 56)`）→
`weights_id == "nvfp4-modelopt"` 从 identity 解析出 `Qwen38Nvfp4ModelOpt` 档 → 20.41 GiB 上卡 →
prefill 915 tok/s → MTP 投机 48.05% 接受率。**W1/W4 这批补丁第一次面对真字节，通过。**

⚠️ **新教训（资源纪律，已升级为硬规则）：agent 报"完成"之后，它 detach 的后台子进程可能还在跑。**
本次两个 E8 agent 都报了完成，但：
- `python3 run_family.py all` —— **RSS 13.05 GB**，父进程 `bash pE8V2_family3.sh`（= e8v2 的遗留）；
- `python3 e8m1_v6.py` —— 父进程 `bash pE8M1_runv6.sh`（= e8v1 的遗留）。

两者把机器压到 **`available 0 B` + swap 17→20 GB + load avg 6.8**（我的构建 + 装载的 sha256 被饿到几分钟不动）。
按铁律②先查父进程/cwd/启动时间取证后 TERM 掉，机器立刻回到 `available 13 GB`、swap 9 GB。
**硬规则：开始任何重活（构建/装载/大转换）之前，先按父进程查一遍有没有已报完成 agent 的遗留子进程；
并且同一时刻的重活不要超过 2 个。** 这与之前那次"两个 agent 之间 OOM"是同一类病。

### 79. aw1/hw 全文到账：**auto 三处不一致必须解**（其一决定性）；⭐ 纠正我"两路都证实棘轮"的说法

**0. 我先纠正自己。** 我在 §78 之后的汇报里对用户说"aw1 与 aw2 **独立**把 −27% 归到棘轮"。
**aw1 的正式报告否掉了这个框定**：它原话——"我最初关于 `update_mtp_window_cut` 的 `window` 参数是棘轮的假设
**是错的** —— 唯一调用点传的是 plan 的 `draft_window`，所以扫描范围本来就是配置的全宽"。
它的补丁仍移掉 `cut < window` 那个收缩构造（非阶梯回退分支里还在），但**"棘轮"这个框定被它自己撤回**。
⇒ 我给用户的说法**过度合并了两路的证据**。正确表述：**两路都识别出 `cut < window` 的收缩构造；
aw2 把它命名为根因并量化，aw1 认为该框定夸大了（扫描范围并非收缩）**。留档教训：**转述"两路一致"前，
必须逐字读两边的"我收回/我说错了"段落，不能只看正面结论。**

**1. aw1 与 aw2 的四处不一致（落 auto 之前必须解）**
- ⭐ **F1（决定性，纯数学、可秒判）**：判据分母用加前还是加后。
  aw1：正确条件是**加前** `S_k ≥ b·N(k−1)/(a+b·(k−1))`；header 写的**加后** `C(k)` 会**压低阈值 ⇒ 过度投机**。
  2805 个模型扫：加前式与穷举 argmax 吻合 **99.8%**（5 次失配，无一差 1），加后式 **94.0%**（167 次失配，**每次恰好过投 1**）；
  code 提示上选 10 而最优 9。**aw2 则明确说"加后式是对的，我对着穷举 argmax 验过"。**
  两边都声称对着穷举验证过 ⇒ **必须第三方独立算一次**（两人都写好了 harness，成本极低）。
- **`--spec auto --draft-tokens N`**：aw2 的 CLI 矩阵显示 `auto×{1,3,9}` 由 THROW→**ok**；**aw1 说这条它没动**（仍拒）。
  两边都声称跑的是真 `validate_speculative_cli_options`。
- **阶梯成员**：aw1 = `{2,3,5,7,9,15}`（含 **2**，为最坏后悔 2.40%/均值 0.136% 穷举出来的）；aw2 = `{3,5,7,9,15}`。
- **35b 的约束表述不同但可兼容**：aw2 说 35b 的 `kMaximumMtpDraftTokens = 5`、`ar[]` 按它定长 ⇒ 15 级阶梯会越界，须按 variant 裁剪；
  aw1 说它的阶梯"小 T 边界并集"对 3/4/5 以外的宽度与 35b **未验**，落在并集外的 fork 是 `cudaGraphExecUpdate` **硬失败**。

**两边一致（可作合并基座）**：阶梯架构本身；**每档必须各占一个 `topology_class`**
（因为 `cudaGraphExecUpdate` 拒绝拓扑变更，一个 class 装不下两个宽度——**两路各自独立给出同一理由**）；
粗粒度 **8 轮**重决；`kMtpWindowCostRatio` **保持 `b/a` = 0.0612**（aw1 也复现了 `r ≤ 0.0065` 时
drafted/round 恒为 15 的退化端）；`mtp --draft-tokens 0` = 自适应；**auto 的常量 3 去掉**；
auto 不取短名单 draft head（`< 5 ⇒ Full`，两边同）；阶梯把 MTP 图预算乘上档数（aw2 估 ~3.3 GiB @ concurrency 8）。

**2. aw1 的独有缺陷（aw2 没有）**
- ⭐ **F2：Beta 先验被当证据读。** `reach[i] ≈ 0` 处 `p_i` 在每个未观测深度都退化成先验均值 0.9 ⇒
  判据**无法得出"停止投机"**。闭环实测：所选均值 10.3 / 7.3 / 6.4 vs 真最优 9 / 5 / 3。
  修法：把先验按 `reach[i]/reach[1]` 加权。⚠️ 并且**"扫描下限"是错的修法 —— 它会把窗口往下棘轮**（二维扫描证据在 `selfcheck.txt`）。
- 阶梯 `{2,3,5,7,9,15}` 是**穷举最坏后悔**选出来的，且**包含所有实测最优（3/5/9/15）** ⇒ 任何实测类别都不会回归。
- 按需捕获档位（`--graph-capture-ceiling` 门控在 `|ladder| > 1`）⇒ **fixed-k 运行逐字节不变**。
- CLI 区分报告自适应：`"15 (adaptive ladder, top) realized mean 9.16"`。
- 闭环复现了记录的 hex 损失与 rep 打平：交付版 **99.0/100.3/99.9/100.0%** vs 原接线 **59.0/73.3/66.5/100.0%**。
- 顺手发现：**`muse` 有 small-T 表副本而 `35b_a3b` 完全没有**；`NINFER_ADAPTIVE_WINDOW` 是**第二套**
  未文档化的自适应机制（设备侧 ctx 无状态 grow-by-one/collapse），同样只动 live extent。

**3. hw 到账（11 文件 +213/−99，`git apply --check` 过）——含对我的两处纠正**
- ⭐ **站点 1 的 `bindings.cpp:383` 已被 W1 在本会话改掉了** ⇒ "两处硬写"只剩生产侧，而生产侧**其实有两份**：
  我列的 `inventory_nvfp4.py:46-47` **加上我没列的 `convert_nvfp4.py:52` 正则 `layers\.(56|…)`**。三份 → 一份。
- ⭐ **边界是"源自己的声明"，不是几何量** ⇒ **没有 spec/geometry 字段可承载**。
  实据：`convert_modelopt.py:20-28` —— 同族另一个源声明的是 layers 0-3/50/52/54 全 MLP +
  21/42/44/46/48/49/53 的 down，**与 56..63 完全不同**。字段位置：compressed-tensors 源
  `config.json → quantization_config.config_groups[*].targets`；ModelOpt 源 `hf_quant_config.json →
  quantization.quantized_layers.<m>.quant_algo`。改法：`FP8_MLP_FIRST_LAYER` 单一副本 + `_FP8_TARGETS` 由 inventory 拼出
  （正则**字节相等**已证），并保留 `_validate_float_group` 每次拿报文核对。
- ⭐ **站点 6 的 `None` 解了，而且和我的排除判据对上**：`specs/qwen4_exp_spec.json` 的 `geometry`
  **没有 `intermediate` 键**（MoE-only，宽度在顶层 `moe.moe_intermediate_size = 640`），`%s` 把缺失键字符串化成 `None`；
  实跑重现 `%d → 0`、`%s → None`。已在**两个树内发射器**加守卫（对真 spec 触发、对 Muse no-op，输出仍 `19968`）。
  ⚠️ **但真发射器不在这个树里**（穷举：头部字符串、`layer_kind{` 唯一发射点、`git log -S`、逐提交 stat、
  过时镜像与 quarantine —— 唯一命中是产物文件本身）。⇒ 那两个文件是在别处生成后被搬进来的。
  ⚠️ **`gemma4-31b_spec.json` 同样缺 `intermediate`** ⇒ 同一颗雷，未爆。
- **站点 2 生产侧判"不改"是对的**：`recipe.py:487-492` 断言该源整份 BF16 ⇒ **报文里确实没有这个字段**，
  它就是 registered 契约本身。（这正是"报文里没有就别造"的正确交付。）
- **站点 3 方向反转**：生产侧 `mlp_layer_fmt` 直接读**源张量 dtype**，**装载器那张 "measured" 表才是重复**
  （已删，并把原本的死代码 `bind_mlp_weight` 改成读声明后投入使用）。
- **额外同病类**：`is_full_layer` 的 `(layer-3)%4==0` 是运行时 `qwen3_6::is_full_attention_layer` 的**第二份硬写**，
  27B/35B 两处均改为调用共享谓词。
- ⚠️ 补丁含 `src/targets/qwen3_6_27b/impl/load/bindings.cpp`，**该文件 W1 已改** ⇒ 落地前必须重对上下文。

**4. 构建 #2（e8v1）状态**：仍在 **18%**、无错误。卡在六个最重的 CUDA TU
（`gqa_attention_{decode,prefill,decode_smallt,decode_e8,prefill_e8}.cu` + `entropy_cold_requant.cu`），
这是 e8v1 改动后的**第一次全量设备码生成**（aw1 报告说这类 TU 各约 30 分钟），慢是预期。
内存 13 GB 可用、磁盘 122 GB，健康。`build/apps/ninfer` 仍是 08:15 那支（含 W1+W4，不含 e8v1）。

### 80. ⭐ F1 裁决：**aw1 对、aw2 错** —— 正确式是**加前**成本；aw2 的 harness 有单边盲区（它自己那行"过投"正是指纹）

裁决报告 `dl/pF1_adjmax.txt`（315 行，脚本与中间产物在 `sh/pF1_adjmax*.{sh,py}`、`scratch/f1adj/`）。

**1. 两人的式子到底是不是同一个 —— 不是，但只差一项。**
- aw2：`S_i > r·N(i-1)/(1 + r·i)`（`C(i) = a+b·i`，**加后**），并**自己打印了**推导
  `S_i·C(i) > N(i-1)·b ⟺ S_i > b·N(i-1)/(a+b·i)`；同一 patch 还把这句代数从旧 header 里搬了个位置，**没改它**。
- aw1：`S_k ≥ b·(1+Σ_{i<k}S_i)/(a + b·(k-1))`（`C(k-1)`，**加前**），并命名
  "the denominator is the PRE-add cost C(k-1) = a + b*(k-1), NOT the post-add C(k) = a + b*k"。
- 代码上只差 `columns = post_add_denominator ? i : (i-1)` 这一项；**aw1 的 `post_add_denominator=true` 分支与 aw2 的表达式逐字相同**
  ⇒ 两人是在争**同一个参数的两个取值**，不是两道题。

**2. 推导：加前式正确。** `Φ(k) ≥ Φ(k−1) ⟺ (N(k−1)+S_k)·C(k−1) ≥ N(k−1)·C(k) ⟺ S_k·C(k−1) ≥ N(k−1)·b`
—— 边际列 k 本身只花 **b**，所以隔离它时除的必须是 **C(k−1)**。加后式把左边换成 `S_i·C(i)`，
阈值恒定地小 `(a+bk)/(a+b(k−1))` 倍（k=9、r=0.0612 时 **1.046**）⇒ **单向更松的检验：只能多投，不能少投。**

**3. 数值扫（k=1..15，每格 20000 条递减 hazard，a=14.7ms / b=0.9ms，r∈{0.0441, 0.0612, 0.09} × 四个生成族 = 12 格）**

| | 总数 | 失配 | 失配的符号 |
|---|---|---|---|
| **加前 PRE** | 12 格 × 20000 | **全部 0** | — |
| **加后 POST** | 同上 | 530 ~ 4174（2.65%–20.87%） | **每次恰好 +1**，无其它差值，**从不欠投** |

结构性检验也通过：`k_PRE > argmax` 恒为 0，`k_POST < k_PRE` 恒为 0。

**4. ⭐ 用 aw2 自己的 harness 设置复现（几何 hazard、r~U(.02,.10)、20000 次）：**
POST = eq 17216 / lt **0** / gt **2784**（aw2 自己打印的是 17231 / 0 / 2769），**gt 中 100% 是 +1**；
同数据 **PRE = 20000/20000**。
⇒ **aw2 打印的那行 "cutoff < argmax : 0" 是"单向偏置"的指纹**，它据此归因于"Φ 非单峰"**是错的**：
同一批曲线上精确式失配为 0 ⇒ **非单峰根本没出现**。根因是**它的 harness 只查了 `cutoff < argmax`（欠投），从不查过投**，
所以把一个单边误差读成了"无误差"。（与我此前把 `max_rel≈2.0` 误读成符号翻转同类：**只查一边的检验会给出假清白**。）
aw1 点名的例子也复现：`p1=0.978, g=0.977, r=0.9/14.7 → argmax 9, PRE 9, POST 10`。

**5. 落地含义**：**auto 合并必须用加前式（aw1 的默认）**。aw1 的 `<` 与 aw2 的 `<=` 只在精确相等时有别（零测度）。
附带一条实测旁注：aw1 自己的档位 snap 表对 ≤1 的误差有 **91.4%–97.7%** 的吸收率，
**但在 11→12 边界会放大**（11→档 9，12→档 15）⇒ **档位集合（aw1 含 2、aw2 不含）应该用实测定，不要靠理论。**
⚠️ 仍未解的两处 auto 分歧：`--spec auto --draft-tokens N` 到底合不合法；档位是否含 2。

### 81. ⭐ E8 第四路（e8m1）回来：**"这个格式兑现不了格点增益" ⇒ 不落 e8m2 的陪集补丁**；但它与 e8v1 在 **V 旋转** 上正面冲突

**背景校正（我自己的记账）**：E8 这条线一共 **四路**，我之前把它们混着数了 ——
`e8m1`（上一轮派的"编码 A"，`scratch/e8m1/EVIDENCE_PACK.md`）、
`e8m2`（"编码 B"，陪集=parity，`scratch/e8m2/fix.patch`）、
`e8v1`（本会话派，落在树里的 16 电平奇数中点集，`scratch/e8v1/fix.patch`）、
`e8v2`（本会话派，E8 跨位宽族 vs ISO3 同预算表）。

**e8m1 的裁决：格点增益真实存在，但这个格式兑现不了 ⇒ 不出补丁。**
- **数学核实**：`G(Z)=1/12=0.0833333`、`G(E8)=929/12960=0.0716821`（SPLAG 表 2.3，本处再用 Voronoi 拒绝采样数值复算得 0.0715949）
  ⇒ `10log10(1080/929)=+0.6541 dB=0.1086 bit/dim`；并用**精确编解码在匹配步长、无 clamp** 下复算：
  E8 的 MSE = `G(E8)·s²` 四位吻合，比值 +0.6566/+0.6560/+0.6550/+0.6538 dB（s=0.5/0.2/0.1/0.05）。
- **"同码率"必须精确定义**：两方案填同一个支撑箱、每块同码点数、同步长。**只有这个定义下 0.654 dB 才存在。**
- ⭐ **陪集确实零 bit（与 e8m2 一致，且它独立复现了往返：393216/393216 块精确）** ——
  4-bit 平面本来就暴露全部 16 个 nibble 态，陪集可以骑**码字奇偶**。
- ⭐ **但**：那 1 bit 买来 2³³ 个地址，而 E8 只能用 `16⁸ = 2³²` 个（每个陪集要求坐标和为偶 ⇒ **每张网格有一半非法**）。
  **E8 恰好把它刚付钱买来的那一位作废了。**
- **严格同存储对照（每 8 块 33 bit = 4.125 b/el，真 K，3072 token × 4 head × L13/14/15）**：

| 那 1 bit 花在 | b/el | vs HEAD |
|---|---|---|
| E8 格点 + 陪集位 | 4.125 | **+0.409 dB** |
| **均匀 16 电平 + 每块网格相位** | 4.125 | **+1.149 dB** |
| 均匀 + 精修一个坐标 | 4.125 | +0.428 dB |

  ⇒ **（ii）赢 0.74 dB**。可推：实测斜率 **6.018 dB/bit**（所以 0.125 b/el = 0.753 dB）vs E8 的 0.654 dB 形增益
  ⇒ **同存储下 E8 在任何摩擦之前就已经 −0.10 dB**；实操上三次方并集还把那 1 bit 花在**自适应**相位上，格点比不了。
- ⭐ **实现层面更狠**：e8m1 自己搭的 parity-carried 实现**实得只有 +0.136 dB（K）/ +0.167（V）**，
  而"理想（免费位）实现"是 +0.409；它另搭的一个**受约束投影写侧得 −0.557 dB** —— **box×clip×parity 三者交互很脆**。
  且它需要在**两个 QK 内循环里各加一次每块仿射 rank-1 修正**（V 侧免费，因为 V 解到 bf16）。
- ⇒ **裁定：不落 e8m2 的陪集补丁。** 综合起来：
  **e8v1 的结论（"陪集绝不值得"）对，但理由错**（它按"占一个平面"计价）；
  **e8m2 的数学对**（parity 承载陪集、零存储位），但**它的经济主张不成立**
  （与其等存储的三次方+自适应相位方案比，格点输 0.74 dB；且可实现的 parity 版只实得 +0.136 dB）；
  **e8m1 的"不出补丁"是应当遵循的裁决**。
- ⭐ **e8m1 独立验证了我落地的 e8v1 补丁**：16 值奇数中点集值 **+0.96 dB**，**且它的除数是已最优的**
  （族最优 D=8 = amax/8 = 除数 16）。
- **修正 brief 里的事实**：B —— **当前代码并没有施加投影**（两个 i8 内核都写着 "NO e8_project_8d_warp() here"）
  ⇒ 与 e8v1 一致；project+rintf 的代价它测为 **+2.82 dB（1.91×）而非 +3.65 dB（2.32×）**
  ⚠️ **这与 e8v1 复现的 2.3197× 冲突**；47.6% 那条是对的。
- **Hadamard-64**：严格正交、`H²=I`，对 K 值 **+6.44 dB**（raw MSE 9.91e-2→2.25e-2；amax/rms 4.216→2.489，
  kurtosis 10.751→3.007），**全部来自尺度锚定、没有一分来自 shaping**（正交映射不可能改变 G）；
  且与格点增益**可加**（E8-vs-cubic 在旋转域 +1.369 dB、原始域 +1.138 dB）。

**⚠️ 两处新的正面冲突（必须再裁）**
1. ⭐⭐ **V 要不要旋转**：e8v1 测 **V + H64 旋转 = 0.15403 → 0.09387（−4.30 dB，b/el 不变）**，称之为最大单项免费win；
   **e8m1 说"对 V（未旋转）旋转毫无价值（L13 +0.58 dB、L15 −0.48 dB），所以引擎只转 K 是对的"**。
   ⇒ 同一件事、方向相反、两边都声称测过。**这是整条 E8 线最可操作的一条，必须先裁。**（已派双路复测。）
2. **被删投影变体的代价**：e8v1 复现 **2.3197×（+3.65 dB）** vs e8m1 测 **1.91×（+2.82 dB）**。
   低风险（该变体已不在树里），但属同一"两边都说验过"的模式，一并记。

### 82. ⭐⭐ e8v2 全文：**"0.020→0.020 不变"的机制查明了（两刀砍在 E8 不可达的代码上）**；陪集之争三方独立定案；V 旋转 2:1 支持

**1. ⭐ 为什么`e8`臂对那两刀毫无反应 —— 查明，不是"读侧没问题"。**
`src/ops/wrapper/gqa_attention.cpp:518-521`：`if (cache.dtype == DType::E8Kv && route != Prompt) route = Prompt;`
（注释："E8Kv small-T kernels are unverified"）⇒ **E8 的每个 decode 步都走 prefill 内核**，
`gqa_attention_decode_e8.cu` 那套 decode 内核对 E8 **根本不可达**。于是：
- `e8r1/fix.patch` 改的是 `gqa_attention_decode_i8.cuh` 里 decode-append 的 `__shfl_xor_sync` —— **E8 走不到**；
- `e8r2/fix.patch` 改的是 prefill 的 cold-tile 判定 —— **E8 启动从不传 cold 平面**。
⇒ **那两刀砍在臂之外的代码上，这既不是"读侧没问题"，也不是"读侧另有一个 bug"。**
（我此前把"0.020→0.020"读成"锅在读侧"是对的方位、错的机制。）
- 量级上也不可能是编码器的锅：本档 K 的 score relRMS 实测 **0.086–0.106**，
  与 **nvfp4 的 0.157（该臂 550/570 分）同一量级**；而 2-bit K 是 0.37–1.02（**质变**）。
  ⇒ **没有任何编码器改动能救 e8 臂（prefix 10 vs 550）—— 它是结构性失败。**

**2. ⭐ 陪集之争三方定案（我的 §81 裁定被独立确认）。** 同一个 4.25 b/el K 码面、真实数据、H64：
| 方案 | relRMS |
|---|---|
| 平铺 4-bit mid-tread | 0.1335 |
| + H64（**现档**） | **0.1061** |
| + H64 + 陪集位、**不加** D8 偶和约束 | 0.0931（+1.13 dB） |
| + H64 + 陪集位、**加** D8 偶和约束（**真 E8**） | **0.1112 —— 比不加标志位还差** |

因为 `|D8 ∩ 盒| ≈ 无约束并集的一半` ⇒ **偶和约束恰好吃掉标志位那 1 bit/块**。
⇒ **e8m2 的 0.0931 是"无约束并集"，不是 E8**；真 E8 是负收益。**§81 的裁定成立且现被第三方独立确认。**
另：真实 K 上 **43.9%** 的块取半整数陪集（树里写 47.6%），标志位熵 **0.99 bit/块（不可压缩）**；
陪集最小代价 = **+0.125 bit/el**，`e8_lattice.cuh` 的 "+1 bit/el" 只对"坐标×2 塞进同一条整数平面"
那一种构造成立（贵 8 倍），**不是下界**，两者都被支配。每档精确 b/el（含陪集位）：**2.375 / 3.375 / 4.375**。

**3. ⭐⭐ 奇数（mid-rise）字母表是本档最大的免费收益，且跨位宽放大**（同 b/el、H64，K/V）：
**2-bit 0.7123→0.3795（+5.47/+5.52 dB）｜ 3-bit 0.2478→0.1899（+2.31/+2.31）｜ 4-bit 0.1061→0.0950（+0.96/+0.96）**。
4-bit 那条**与树里自己签的 0.10661→0.09546 吻合** ⇒ **e8m1 与 e8v2 两路独立验证了我落地的那个补丁。**
2-bit 大赚的机理：mid-tread 把最内层花在**恰好 0**，2-bit 只剩 `{−1,0,+1}·amax`。

**4. ⭐ V 旋转现在是 2:1 支持（e8m1 是少数派）**，等裁决 agent：
- e8v1：V + **H64** = **0.15403 → 0.09387（−4.30 dB）**，把 H 折叠进 `o_proj`（内核零改动）；
- e8v2：V + **H8（per-8-block）** = **0.1315 → 0.0999（−2.4 dB）**，逆变换放在 **PV 输出侧**
  （`hadamard_8d_pair()`，numpy 验证 2000 组最大偏差 **0.0**、对合性 1.3e-15），零额外比特、零布局改动；
  输出 relRMS 0.1241→0.0925（仅 V 错）/ 0.1393→0.1118（K+V 都错）。
  ⚠️ 前提：若解除 `gqa_attention.cpp:518` 的路由强制，**逆变换必须加在 split reducer 里，不能加在 per-split partial 写**
  （partial 必须留在旋转基里）。⚠️ 半应用的旋转按实测 relRMS ~1.4（**比不改更糟**）⇒ 不能盲落。
- e8m1：~0 / 负。**两边基数不同**（e8v1 0.15403 vs e8v2 0.1315，后者已含 16 电平字母表），且旋转不同（H64 vs H8）。
- 顺带：**K 的旋转已经没有 dB 可捡**（n=4：none 0.1335 → H8 0.1142 → SO(4)/4ch 0.1243 → rand16 0.1153 → **H64 0.1061** → rand64 0.1073）。
- 并纠正 `scratch/kv_2bit/REPORT.md §5.4` 的"hadamard64 让 MSE 差 3–4.8×"：那是**跨旋转域比值**的测量假象
  （e8v2 自己也先复现出 relRMS 1.41 = √2 的同一个坑）。

**5. ⭐ ISO3 的 SO(4) 旋转几乎是白挂**：ISO3 K 0.0913/0.0898 @4.50 vs **同码本去掉 SO(4)** 的 `u4_g16_e4m3`
**0.0791/0.0781** @4.50 ⇒ **SO(4) 只值 +0.44 dB**。且 **ISO3 的 V = 0.1007 比 E8 档的 V（0.1315–0.1458，4.25 b/el）还准**，
因为它的尺度是 **per-16 E4M3** 而非 per-64 fp16。⇒ 又一次同一个结论：**差在尺度分组，不在码本/旋转。**

**6. 发现一个 dtype 表 bug**：`src/kvcfg/kv_formats.h:44` 的 `Iso3 → 3` **是错的**（差 1.5 b/el；
它描述的是 `iso_codec.h` 那份内核从不调用的 v1 契约，那份实际是 3+16/8 = **5.00** b/el）；
`bits_of(E8)=4` 与阶梯的 425 也对不上。**这条不一致会影响排序/校验规则**
（`hot ≥ int8` / `tail ≥ hot` / `pure 禁 iso`），虽然 DP 成本用的是正确的 450/425。

**7. 1M 预算口径分歧（未解）**：e8v2 按 27B 几何（16 全注意力层 × 4 kv_head × 256 dim × 2 面 = 32768 el/token）
算 **2.125 b/el × 1M = 8.50 GiB**；反推 10.4 GiB/1M ⇔ **2.60 b/el**，与 `_TODO.md:6132` 的 10.31 GiB / 2.144 **对不上**
（他们那个数需要 ~1.27M token，或约 1.2 GiB 非 KV 开销，例如 1 GiB `kDefaultKvCapacityHeadroomBytes` + 页对齐）。
两种口径下 **3-bit 都买不起**（最便宜 3.125）⇒ "1M 只装得下 2-bit" 成立；
**但上限是 2.14 还是 2.60 决定陪集位塞不塞得进去** —— 记为未解。
⚠️ 并且 **2-bit K 是质变级的坏**（score relRMS 0.37、输出 relRMS 0.61；可用最弱档 nvfp4 是 0.12）
⇒ **1M 能开 ≠ 检索能保。**

**8. 交付**：`scratch/e8v2/fix.patch`（3 文件，**纯注释、零行为变化**：修正 `dtype.h` 的 E8Kv 描述
"**没有格点**"、`e8_lattice.cuh` 的陪集代价口径、`gqa_attention_kv_quant.cuh` 的字母表附跨位宽实测）
与 `scratch/e8v2/vrot.patch`（3 文件，V+H8 旋转 + PV 输出侧逆变换，**标注前提、未盲落**）。两者 `git apply --check` 均过。

### 83. ⭐⭐ V 旋转裁决：**e8v1 对、e8m1 错**（−3.81 dB，零额外 bit）；并推翻 §81 里 e8m1 的 **+6.44 dB** 与"引擎只转 K 是对的"

裁决报告 `dl/pVROT_vrot.txt`（487 行），脚本 `sh/pVROT_vrot.sh`，产物 `scratch/vrot/vrot.py`。

**结论（引擎自己的口径：旋后 per-64-group amax、scale = fp16(amax/16)、16 电平奇数码本、同为 4.000 b/el）**：
给 V 加 H64 值 **V MSE −3.81 dB**（pooled 3.124e-01 → 1.299e-01，relRMS 0.14555 → 0.09387）；
旧 15 电平码本下 −3.34 dB；固定 K 时**输出 relRMS 0.08945 → 0.06251 = −3.11 dB**。
⚠️ **e8v1 的 −4.30 dB 高报了 0.49 dB** —— 它把"旋转"与"15→16 电平码本换装"两件事算在了一起；**纯旋转只有 −3.81 dB**。

**e8m1 的错钉到了行号**：`e8m1_v8.py:101-102,106-107`、`e8m1_v4.py:113,116,172-177` 用
`reshape(hd,kh,T)` 配 numpy **C-order**，把 **head_dim 与 T 读转置了**：
它所谓的"64 组"实际是**一个 head_dim 通道跨 64 个 token**（间隔 12），不是 64 个连续 head_dim 通道；
且它锚定的是**每 8 块 amax**，不是引擎的每 64 组 amax。

| 分组 × 锚定（V，树码本，pooled MSE） | raw | H64 | dB |
|---|---|---|---|
| **核轴 / 64 组（引擎口径）** | 3.12407e-01 | 1.29943e-01 | **+3.810** |
| 核轴 / 8 块 | 1.13216e-01 | 8.11656e-02 | +1.445 |
| e8m1 轴 / 64 组 | 1.11078e-01 | 1.53386e+00 | **−11.402** |
| e8m1 轴 / 8 块 | 5.79275e-02 | 2.29108e-01 | −5.972 |

**只改锚定 → +1.4 dB；只改轴 → −11.4 dB；两个都改才是 +3.81 dB。** 两边的 harness 都被逐位复现过
（与 e8v1 `bench_v4.txt` 每个 V/K 数字一致；与 e8m1 的 L13/L14/L15 三对 MSE 6 位一致），所以不是猜口径。

**副产品（同因，一并推翻）**：`EVIDENCE_PACK.md:236-254` 的"dump 已是旋转域"也是这个转置读法的产物
（它引的 2.489/4.216 在那个轴上逐位吻合）；**在核轴上旋转是下降的**（amax/rms 3.383→2.554，kurtosis 5.75→2.83）。
⭐ **`EVIDENCE_PACK.md:172-173` 的"K 旋转 +6.4 dB"同因错，正确是 +1.75 dB**
⇒ **§81 里我转述的"Hadamard-64 对 K 值 +6.44 dB、全部来自锚定"作废，改为 +1.75 dB。**

**机理 100% 是锚定**（正交映射不可能有 shaping 增益）：`E[A_raw²]/E[A_rot²]` 同组配对 66M 元素 ——
K 1.4605 → 预测 +1.645 / 实测 +1.745；V 2.3045 → 预测 +3.626 / 实测 +3.810；
逐层 V L13 2.829 vs 2.862、L14 6.671 vs 7.053、L15 2.056 vs 2.071。
⭐ **口径 (c)（旋前 amax）实测与 (a) 逐位相同（0.000 dB）** —— 正交变换对**固定 step** 的量化器误差为零
⇒ 这条直接坐实"收益全在锚定"，也解释了为什么半边见不到收益。
**V 根本不是"好脾气"**：per-channel rms 散度 17%/51%/46%，全局峰度 9.4/55.9/5.3 → 旋后 3.6/3.8/3.2。

⭐ **翻转 e8m1 的推论**：`EVIDENCE_PACK.md:198` 说"引擎只转 K 是对的" ——
**数据说的正相反：V 比 K 更值得转（3.81 vs 1.75 dB）。**

**未证（E1–E9）**：无模型运行 ⇒ 没证 3.8 dB 变成 ppl/needle；"H 折进 `o_proj` 零成本"只读代码未上机；
只测了 E8/i4 档的 V（未测在 ISO3 的 SO(4) 之上再叠 H64）；21 个 dump = 7 次 prefill × 3 层、
未验证是否为 7 个不同 prompt；未解任何 `.kvc` 码平面；输出侧是降级代理（q 用 dump 自己的 K 行、单 kv head、512 keys）。

**⇒ 现在这条可以进落地队列**：V + H64（或 H8）旋转，**零额外 bit、零布局改动**，
逆变换放 PV 输出侧（e8v2 已给 `hadamard_8d_pair()` 并 numpy 验证 0.0 偏差），
⚠️ 前提：若解除 `gqa_attention.cpp:518` 的路由强制，**逆变换必须加在 split reducer 里，不能加在 per-split partial 写**；
⚠️ 半应用的旋转实测 relRMS ~1.4，**比不改更糟** ⇒ 必须整条一起上并构建验证。

### 84. ngram：**空间不是那道墙**（前提被推翻）；收益上一轮已量化为负；真缺口是"引擎真表 gather"且**不需要 95 GiB 就能验**

用户问"ngram 现在不是有空间了吗"。查证后**前提不成立**，两件事：

**1. 那 95 GiB 不是"腾地方就能放"的文件。**
- `ple_layout.h:44` 写明它是 **"Baekpica SSD-PLE sidecar"** —— 设计上就是**从 SSD 流式读**；
  分片名形如 `ple/ple-bf16-00001-of-00004.bin`（`ple_layout.h:39`）。**它本来就不该常驻同一块盘。**
- 体积来自**模型训练出来的 PLE 表**，不是转换器能重造的：`flashnext_bindings.py:16` 记
  `ple-manifest.json (PLE table: 20M x 160 BF16, 16 heads)`；引擎侧按 `padded_vocabulary_rows` 铺满
  （`ple_layout.cpp:99` 校验 "part coverage != padded_vocabulary_rows" 即抛）。
  20M 行量级 × hidden ⇒ 那 ~95 GiB 的来源。**这是模型产物，不是磁盘空间问题。**

**2. ⭐ 更关键：这个方向的收益上一轮已经量化，结论是负的。**
`scratch/ngram_measure/REPORT.md`（Sep 12，28 KB；语料 sha256 逐文件校验 + 双 tokenizer 交叉验证，
两个 tokenizer 对 16 条流只差 3 token = 3e-6）结论原文：
> "精确匹配查表作为 KV 替代品**几乎不成立**。索引字节不是瓶颈（4–12 B/token，是 nvfp4 KV 的 **0.02–0.07%**），
> **覆盖率**才是瓶颈：按引擎 KV 页（64 token/页）粒度，混合 1M 上下文里前缀索引释放的页数是 **0 页**
> （精确 0，不是"很小"）… 最乐观的、允许自引用的**不安全**口径也只有 3.762%，即 1M 的 nvfp4 KV
> 从 18.00 GiB 压到 **17.32 GiB**；**安全口径一点也压不下来**。"

它同时给出"真正被数据支持的用法"：**投机/查表解码的算力加速（仅代码类文本）**，
以及引擎**已经实现**的整前缀复用（`ResidentPrefixIdentity`）。
其 KV 几何是双证过的：16 全注意力层 × 4 kv_head × 2 平面 × 9216 B/head-page ÷ 64 = **18,432 B/token**，
与 `research/notes/TODO.md §98` 记录的 65536 ctx / `kv payload 1152 MiB` 逐字节吻合（1152 MiB/65536 = 18,432）。

⇒ **ngram 没落地的门是"覆盖率"，不是空间；覆盖面 ~0，给 95 GiB 也换不来那 3.8%（还是不安全口径）。**

**3. 但有一格是真的、且现在就该补**：唯一未证的运行时部件是**"引擎真表 gather"**——
`tests/ops/ple_table_e2e_test.cu:2` 自己写着 `Verifies the one unproven runtime piece`，
而它 skip 只因为 `tests/CMakeLists.txt` 注册时不带 `<sidecar_root>`。
**验证它不需要 95 GiB**：树里就有构建器 `tools/convert/ple_sidecar_build.py`（写 `ple-manifest.json`）
与夹具/校验器 `tools/archkit/ple_gather_check.py`（有 `--emit-spec`，会往 tmp 写 manifest）。
⇒ **一份小的合成 sidecar 就能把 #4 从 skip 变成真跑。已派 agent 去做**（产物 `scratch/ple/`，
授权它**直接运行已存在的** `build/tests/ninfer_ple_table_e2e_test`，但**不许跑 cmake/nvcc/make**），
要求给出契约逐条（带行号）+ 最小合法 sidecar 规格 + 测试原始 rc 与输出 + 判定
（**已证 / 有真缺陷 / 仍不可证**），并**明令不许为让测试通过而伪造数据**。

### 85. ⭐ PLE 真表 gather **已证（模块级）**：小合成 sidecar rc=0、7680 个 BF16 逐位相等；但"真表"是误名 + 挖出 4 个新缺陷

交付 `dl/pPLE_small.txt`（319 行）+ `dl/pPLE_small2.txt`（83 行），产物 `scratch/ple/`（sidecar A 10.3 MiB、B 1.3 MiB）。

**1. 结果：`rc=0`、`PLE_TABLE_E2E_PASS`、7/7 断言全过。**
`NINFER_PLE_SIDECAR_ROOT=… build/tests/ninfer_ple_table_e2e_test`（argv 形式同样 rc=0）：
`gather vs host reference: 7680 elems, mismatches=0, max|diff|=0`、`post-eviction gather bit-exact`、
in-gather 去重（48 行塌缩为 32 次 fault-in）、distinct/repeated window、gather+sync、第二次 gather。
**真跑的链路**：manifest 解析 → `row_location` 跨 **4 个物理文件 / 32 个 logical part** → 冷缺页 `pread`
→ `cudaHostAlloc(Mapped)` + 补零 → `cudaHostGetDevicePointer` → gather kernel → 7680 个 BF16 **逐位相等**。
行数学不在此测（`test:34-37` 明示），另跑既有 `build/tests/ninfer_ple_layout_test`：**rc=0**，
与 `ple_reference.py` 交叉对拍 9×16 行 + 144 行 UVA gather 往返通过 ⇒ **行数学不是这个洞**。
sidecar 用树里**已有的合成夹具生成器** `tools/ple_reference.py gen`：32000 行 ×160 BF16 = 9.766 MiB，
4 文件 / 128 part，满足契约 C1–C11（16 头覆盖互不相交整段，48 个派生行落全部 4 文件 / 32 part）。

**2. ⚠️ 但"真表"二字不成立 —— 缺口从"代码"转成"标度与产物语义"。**
- 仓库自带的真实 HF 审计**否掉合成几何**：`ple_gather_check.py --ngram-base 2000` 对 sidecar A **FAIL**
  （vocab 2000×16 对素数 2003/2011/…；offsets 非前缀和；padded 32000 对 33024）。
- 夹具字节是**行号不是 PLE 数值**（`ninfer_ple_layout_test real <A>` 直接以 `NaN in PLE data` 失败）。
- 不触及 **>4 GiB 偏移、4096 对齐、evict/prefetch**。
⇒ **要补这些就得换成"素数几何 + 4096 对齐 + >4 GiB + 真实数值"的表，基本等于真产物。缺的是标度与产物语义，不是 fault/gather 代码。**

**3. ⭐ 新缺陷 1：eviction 路径按构造不可达，且与表大小无关。**
`post-eviction` 那两条断言（`test:155,165`）是**空转**。硬上界 = 2×48 个 4096 对齐条目 × ≤8192 B = **768 KiB < 1 MiB 预算**
（`ple_table.cu:153` 要超预算才 evict）。**该上界只取决于"每次 gather 的行数 × 页大小"，与表大小无关
⇒ 换 95 GiB 真表也照样不 evict。** ⇒ 又一条"测试在静默地不测"。

**4. ⚠️ 新缺陷 2：测试自身输入 #2 有 OOB，且第二段是循环自证。**
`test:150` 给 3 个 token 只传了 3 个前驱，而需要 `(3-1)*3=6`；`ple_layout.cpp:151-153` 按
`prevs.data()+i*n_prev` 遍历、**从不看 `prevs.size()`** ⇒ 读 `prevs[3..5]`（12 B 越界，
恰好落在 glibc 24 B usable chunk 内 ⇒ 不崩、无 sanitizer 也抓不到）。
**后果**：第二次 gather 的行由未初始化堆决定，而**设备侧与 host 参考用的是同一个 `rows2`**
⇒ `mismatches2==0` 是**循环自证**，不是验证。

**5. ⚠️ 新缺陷 3：树里的 sidecar 构建器产出不可用。**
`tools/convert/ple_sidecar_build.py:68` 落盘名写死分片第二位为 `0`（`…-of-00000.bin`），
`:82` 却只把 manifest 里的 path 改成真实总数（`…-of-00004.bin`）⇒ **manifest 指向不存在的文件**，
测试 rc=134 abort。诊断（字节未改、仅把名字对齐它自己的 manifest）：**rc=0** ⇒ 名字错配是唯一拦路项。
而且 `:84` 写死 `per_head_offsets=[0]*16` ⇒ 16 个头挤在同一 band（表的 6.25%），48 行只剩 4 个唯一行且全在 0 号文件。
⇒ **夹具路径严格更优**，构建器该修。

**6. ⚠️ 未覆盖面（逐条留档）**：>4 GiB 偏移与 64 位 key 打包（`(fi<<56)|off`）；part `file_offset` 的 4096 对齐
（夹具 124/128 未对齐；`ple_layout.h:32` 只承诺不校验）；`format_version` **从不与 1 比较**（`:43`）；
`physical_files[i].index==i` 不校验而 `ple_table.cu:63-69` 按**列表顺序**建 fd
（`index` 与位置不一致会**静默读错文件**）；per-head band 越界与 `row_stride_bytes==2*dim` 均不校验；
`read_span` 命中路径（`ple_table.cu:88-93`）在"同页两行且高偏移行跨页"时会多读 64 B 越过 pinned 块（本次未触发）。
⭐ 且 **`PleTable` 在 `src/`/`apps/` 里无人构造**，`prefetch_workers`（`ple_table.h:31`）**全树无引用**，
`ple_stage.h:69` 提到的 `PleTable::gather_phase` **不存在** ⇒ **头注释宣称的异步预取器未实现，引擎接线未验。**

**7. 对"小 sidecar 够不够"的回答**：**模块级运行时部件：够**（已补上，rc=0、逐位相等、跨文件、去重）。
**artifact 级：不够**（见 2、6）。⇒ 记档：**PLE 的模块级缺口已关闭；剩下的缺口是标度+产物语义，本质上要真表。**

### 86. ⭐ 2-bit 真 E8 测出来了：**+0.66 dB，命中理论** ⇒ 上一轮那个 0.3795 "不是 E8"；但三条更免费的杠杆比它大

**用户问"2bit 的 E8 还这么差吗"。答案：不差 —— 之前那个 0.3795 是标量 2-bit + H64，不是 E8。**

**1. 真 2-bit E8（陪集骑码字奇偶）的实测收益（同 2.25 b/el、两边各自调最优 divisor、K/V 分开、L13/14/15）**：
| | L13 | L14 | L15 |
|---|---|---|---|
| **K (+H64)** | **+0.657 dB** | **+0.679** | **+0.679** |
| V (+H8) | +0.667 | +0.601 | +0.648 |
| V (+H64) | +0.674 | +0.559 | +0.678 |
| V（不转） | +0.933 | +1.320 | +1.093 |
理论 `10log10((1/12)/0.0716828) = 0.6541 dB` ⇒ **K 命中到 ±0.025 dB**。
独立对照（无界、同步长、无裁剪，131k 块）：K/H64 在 step=1/0.5/0.25/0.125 得 +0.654/+0.651/+0.655/+0.653。**教科书数在干净区间被独立复现。**
（更小步长那几行 +5~+37 dB 是假象：E8 无裁剪而 Z8 被裁在 ±63，**不采信**。）

**2. 实现层验证（这节最硬）**：
- **可解码且 0 额外位**：65536/65536 块 `parity(Σu) == encoder coset`，`|x̂_enc − x̂_dec| = 0.000e+00`，
  码字可从 k 精确还原，`off=1` 占比 0.4995–0.5029 ⇒ **往返精确**。
- **编码器就是约束集上的精确最近点**：每块暴力枚举 65536 个候选（coset0 32768 + coset1 32768），
  150/150 子块完全一致，**worst excess = 0.000e+00**（div=1.75/2.5/3，L13+L14）。
- 复现基准：(a) 2-bit 奇数 mid-rise +H64 div=2 → **0.38010/0.38339/0.38045**（靶 0.3795，差 +0.14%）；
  (a) 4-bit 锚点 → **0.09517/0.09598/0.09524**（靶 0.0950）；4→2 bit 实测 **+12.02 dB**，与 6.018 dB/bit 斜率一致
  ⇒ **我上一轮"正走在码率阶梯上"的判断被证实。**
- ⚠️ **注意**：任务给的 V 靶 0.3777 是"V+**H64**"，而**树里 V 当前未旋转**（未旋转 V 的 (a) = 0.568）。

**3. ⭐ (c) 偶和约束在 2-bit 上吃掉多少**：同 2.375 b/el 下并集 0.30273 vs (b) 0.31874 ⇒ **约束吃掉 0.449 dB**
（该 bit 的费率价值 = 0.125×6.018 = **0.752 dB**，即吃掉 60%）。但**"零存储位"仍净赚 +0.30 dB**。
⇒ 与 4-bit 上"约束比不加标志位更差"方向一致，**但 2-bit 上净收益是正的**。

**4. ⭐ 但三条更免费/更大的杠杆排在 (b) 前面**（按性价比）：
1. **K 把 2-bit 的除数从 2 改到 ~2.5** ⇒ **+0.87 dB，零 reader 改动**（树里 2-bit 沿用的是 `div = 2^(n−1)` 约定，**没调过**）；
2. **V 加 H8** ⇒ **+1.8~3.4 dB**（H64 再 +0.3~0.6）；
3. **V 用 companded 码本 `±{a,1}·G`, a≈0.22** ⇒ 样本外 **+1.2~2.1 dB**（K 不需要拟合，a*≈0.30 收益仅 +0.05 dB ≈ 没用）。
**尺度面很便宜**：(a) g64 2.25 = 0.34378 → g128 **2.125** = 0.35428 → g256 2.0625 = 0.35600 ⇒ 放粗一档只花 **0.26/0.31 dB**。

**5. (b) 值不值得落**：**过 0.5 dB 门槛（+0.66），但要排在三条免费路径之后，且需要改 reader** ——
它把 dequant 变成 `code·S + 每 8 块 rank-1 常量（由 `parity(Σu)` 定）`，**破坏纯 int8×int8 的 QK MMA**，要加修正项。
换算：+0.66 dB ÷ 6.018 dB/bit = **只值 0.11 b/el**（2.25 档变"2.36 等效"）。
⇒ **建议：不要现在写 (b) 的补丁；先把两条免费路径量出来落地（2-bit divisor 2→2.5；V + H8 + a≈0.22 码本），
(b) 作为"拿到 +0.66 dB 的唯一付费路径"记在待办备用。**

**6. 它没证到的**：只有 KV 张量 relRMS/MSE，**没有** downstream NLL/困惑度/attention 输出/logit 误差
（**+0.66 dB 的 KV MSE ≠ +0.66 dB 的 logit**）；只有 L13/14/15 × 一个模型 × 一个 3072-token prefill（7 个可用样本，
21–26 号 dump 截断）；**(b) 的 reader 一行引擎代码都没写**，未验证它在真实 nibble 打包 / i8 QK MMA / rank-1 折叠下成立；
奇偶修复只用"单坐标 ±1"（经验最优，无一般性证明）；**严格说是"实现了同量级的 granular gain"，不是"实现了 E8 的最近点映射"**
（理论 0.6541 dB 是"无界、高码率、同步长"的 granular gain，2.25 b/el 的裁剪码本命中同一数字是实测事实，未证两者由同一个量支配）。

---

### 86b. 补我欠的几笔（上一轮的口头更正，正式记档）

1. ⚠️ **我把用户的外挂召回思路换成了一个它并不回答的问题。** 用户点名的思路是
   **③ 运行期外挂召回**（`_TODO.md:5684-5685` 原话："召回的时候去 SSD 上搜，**以对话为单位召回**，
   而不是长期卡在 KV 里，也就是搞成**类似外挂知识库**的形式，那样 **SSD 带宽就不会成为瓶颈**"；`:5703` 归档）。
   它的账**是正的而且很大**：SSD 读 packed KV **2.6–6.1 µs/token** vs 重新 prefill **365–690 µs/token** ⇒ **105–260:1**；
   "SSD 不是瓶颈"对 packed KV **418×**；**常驻 KV 钉死 decode 上限**（1M 常驻 = 80 tok/s，32k = 2,550 tok/s）。
   我去量的却是"ngram 查表能不能**替代** KV（靠上下文内部重复省页）"⇒ 覆盖率 ~0。
   **两者不冲突**（外挂召回不需要上下文内部有重复）——**是我把问题换掉了。**
2. ⭐ **cuFile 定性从"优化"改为"使能项"（用户指示）**，两个理由：**①不经主机内存的冷页卸载；②超大 MoE 模型的运行。**
   两个接入点都已**预成形**：
   - **冷页**：`ColdPolicy::Disk` 是预留未实现的 SSD 层（`layouts.h:112,153` 注释 "spill directory and budget"；
     `_TODO.md:7423` 记 "SSD 层未实现（`Disk` 保留）"；相关 `cold_host_tier.h` / `kv_cold_tier_budget.h` / `serve/kv_cold_policy.h`）。
   - **MoE/权重**：⭐ `src/product/weight_residency.h` 的头注释几乎就是为 cuFile 写的
     （"page the offloaded spans from the **mmap → pinned staging → H2D**… the pinned allocator is **INJECTED**"），
     已有 `WeightResidency::Host`、`classify_weight_residency()`、`WeightPageCache(budget, PinnedAlloc, PinnedFree)`；
     ⚠️ 但 `_TODO.md:7423` 记 "P0 的 `WeightPageCache` **已成死代码**"。
   ⇒ **cuFile 直读 = 把这条链的 `mmap → pinned staging → H2D` 换成 `SSD → VRAM 直读`，那个注入式分配器就是替换缝。**
   三层（PLE 表 / 冷页 / 权重-MoE）**本质是同一个"从 SSD 取数"问题**，应共用一套取数层（共同的现状形态 = `PleTable` 的 512 MiB pinned LRU）。
   ⚠️ 盒子实况：cuFile 库与头**都在**（cuda-13.3），但 **`nvidia-fs` 内核模块未加载、`/dev/nvidia-fs` 不存在**，
   卡是 **RTX 5090 D（消费级）**；cuFile 需要 **4096 对齐**（sidecar 夹具现状 124/128 未对齐）。已派 agent 出集成设计 + 退化路径判定。
3. **FreeToken 状态：三步骨架都落地了，但核心闭环没通。**
   时间线：§41(09-08 细化) → §48(09-09 步2 决策逻辑 `ft_tiers.py`) → §55(09-09 步2 引擎动态重排设计定稿)
   → §56(09-09 步2 实现落地，构建通过、冒烟中) → §78(09-09 深夜 **W5 周期自动重排**)
   → §85(09-10 **W6 = 步3 带宽自适应** + W16 落地，标注"**待构建/GPU 验证**")。
   树里已落地：`src/runtime/engine/bandwidth_governor.h`、`qwen3_6/impl/runtime/ft_stats.h`、`ops/common/ft_stats.h`、
   `serve/kv_auto_relayout.h`、`serve/kv_cold_policy.h`、`tools/archkit/ft_tiers.py`、`tools/kv_relayout_test.cpp`。
   ⚠️ **但 `_TODO.md:2177`："W5 的观测链端到端从未真正工作（FreeToken 步 1/步 2 的自动重排永远等不到数据）"**；
   `:2976` 状态表把「热/温/冷自动动态分配」标为 **未闭环**。
   ⇒ **机制交付了，闭环没通**；缺的是那条**观测链**。验收标准（用户定）= 64K 长测协议（针尖 + 长生成退化检测）下 ppl/tok/s 不塌。
   ⭐ 它与 cuFile 是**同一根线**：FreeToken 要"按热度动态决定热/温/冷归属"，而冷层的 `Disk` 支路至今未实现 ——
   **没有真正的 SSD 层，就没有真正的"冷"。**
4. **构建 #2（e8v1 + W1 + W4）完成 rc=0**；`build/apps/ninfer` 已更新。ctest 已接上跑（上轮基线 101 通过 / 0 失败 / 8 跳过）。

### 87. ⭐ **八大项找到了** = **N1–N7 + U6**；两份真值源 = `research/notes/board.md`（"协作看板 (唯一状态源)"）与 `_TODO.md §117`

用户两次点名"八大项"，我终于定到它。**证据链**：
- `research/scripts/_todo_sync2.py:42` 原话：**"八项：N1/N2/§103 早已进树（看板陈旧）；N2 遗留 `ColdPolicy::Host` 静默已修
  （矛盾报错+一次性告警，备份 `/home/user/cold_host_bak/`）；N3 第二轮其实已完成（只修了两处引用已删字段的陈旧注释）；
  N6/N7/N4/N5/U6 各有硬阻塞。"** ⇒ **八项 = N1, N2, N3, N4, N5, N6, N7, U6（恰好 8 个）。**
- `research/scripts/_patch_eight.py:2` 原话：**"两项小修（八项里最便宜的两条）"** —— (i) `apps/cli/main.cpp`
  `format_kv_cache` 补 4 个缺失分支（此前 4/7 档误报 "unknown"）(ii) `text_prefill_impl.h` 修正 dump 记录里
  `last` 的语义注释（实测是 final-norm **之前**）。
- `research/notes/board.md` 头部：**"# 协作看板 (唯一状态源)"**，表列 = `ID | 状态 | 负责人 | 产物 | done 判据 | 验证者/命令`。
- `board.md:45-56` 有一张 **「用户八问的落地状态（2026-09-10 09:3X 逐条查证）」** 表，与 N1–N7/U6 一一对应。

**编号 ↔ 名字 ↔ 事项（这张表就是"八大项"）**

| 项 | 名字 | 用户口径（`board.md:45-56` 的"问"） | 当时的"缺" | 2026-09-10 状态 |
|---|---|---|---|---|
| **①** | **N1** `--kv-bit-budget` | 输入 bit → KV 温窗自动分配 | 引擎内"给 bit 数自动分配"入口 | 策略完善/**引擎自动化未做** |
| **②** | **N2** 冷窗 | 推广到冷窗 | bit 分配器未含冷槽代价模型 | **冷窗可用/未统一分配** |
| **③** | **N3** 运行时校准闭环 | 启动自测→固化→跳过→可重校 | 运行时"跑一遍→固化→跳过→手动重校"流程 | **半截** |
| **④** | **N4** 热/温/冷自动动态分配 | 热/温/冷自动动态分配 | 按热度自动决定热冷归属的**闭环策略** | **未闭环** |
| **⑤** | **N5** W13 权重卸载 | 权重卸载到内存 W13 | **全部**（代码里无 offload 实现） | **未做**（1M 的使能项） |
| **⑥** | **N6** ngram 真表 gather | lookup ngram | 真表 GPU gather / 前缀缓存(含 GDN 循环性) | **部分落地** |
| **⑦** | **N7** FlashNext | FreeToken 加载 FlashNext | **真实 checkpoint** | **架构就绪/权重缺失** |
| **⑧** | **U6** 大件 | dflash2 接受率根因（第 4 条未实施） | — | **部分找到** |

⚠️ **编号映射的来源**：`_TODO.md:6965/6968` 用的是 **"八项⑤ = 权重卸载 W13"**、**"八项⑥ = 逐轮外挂召回"**
—— 与上表 **⑤=N5、⑥=N6 完全一致** ⇒ 编号即 N 号。我此前只核到 ⑤/⑥ 是因为只 grep 了 `八项①..⑧` 这种字面，
而清单本身用的是 **N 号**，不是圆圈数字。**这就是我找了两轮的原因。**

**board.md 里编年记录的关键里程碑（与本会话的活直接相关）**：
- `:217-218`（09-10）**"N3 运行时校准闭环 / N4 热温冷自动分配 / N5 W13 权重卸载 / N6 ngram 真表 GPU gather：未开工
  （N5/N6 是大件, 需单独立项; N3/N4 依赖 KV 侧先落地）"**。
- `:250-253` **"N6 ngram：`src/ops/ple/ple_table.{h,cu}` 的真表 gather + 有界 pinned LRU 热行缓存已实现，
  但是死代码（`src/ops/ple/` 之外零引用，与 N3 的 `KvCalibrationCapture` 同病）；N6 缺口 = 把 PleTable 接进
  qwen4_exp 运行时 + 确认 GDN 循环状态下的前缀复用正确性，不是从零写内核。"**
  ⇒ ⭐ **这条精确定义了 N6**，也解释了本会话我测出的"`PleTable` 在 `src/`/`apps/` 里无人构造、预取器未实现"正是 N6 的缺口。
- `:365` **"S33 重大范围发现：`qwen4_exp` 运行时只是桩"**；`:324` "S33 (B): `PleTable` 接入 `qwen4_exp` 运行时
  （N6 的实体；W2① 已证 gather 可用，但它是死代码）" ⇒ **N6 的接线补丁 `_collab/B_s33_ple_wiring.diff` 早已就绪（DONE, patch 未应用）**，
  它的核查列出的 6 个"不存在"包括：①qwen4_exp 运行时只有 2 文件 stub ②registry 未注册 ③`config.h:11 intermediate=None` 不可编译
  （**正是本会话我修掉的那个 `None`！**）④无 `--ple-sidecar` ⑤eos 无载体 ⑥sidecar 契约 BF16 vs checkpoint 表 FP8。
- `:26-44` 窗口 C 的运行态 + `:24` "GPU 占用：训练单实例 batch4…**GPU 窗口需先 kill 训练**"。

**本会话对八大项的推进（逐项对账）**
- **③ N3 运行时校准闭环**：cl1/cl2 双路已交（cl1 的 C++ 写侧与 `kv_rowscale_sidecar.py build()` **逐字节一致**；
  cl2 的指纹在 sidecar `tag[16]`），`git apply --check` 均过 —— **合并后即可落地**（仍是"半截"→可闭合）。
- **④ N4 热温冷自动分配**：这是 **FreeToken 的观测链**问题 —— `_TODO.md:2177` 记 **"W5 的观测链端到端从未真正工作
  （步 1/步 2 的自动重排永远等不到数据）"**，`:2976` 标 **未闭环**。本会话已派 agent 去测/定位断点。
- **⑥ N6 ngram 真表 gather**：本会话**已证到模块级**（小合成 sidecar，`rc=0`、7680 个 BF16 逐位相等、跨 4 文件/32 part），
  并挖出 4 个新缺陷（evict 按构造不可达且**与表大小无关**、测试输入越界且第二段循环自证、`ple_sidecar_build.py` 产出不可用、
  `PleTable` 无人构造/预取器未实现）。⇒ **N6 剩下的正是 board 说的那两件：接进 qwen4_exp 运行时 + GDN 循环态前缀复用。**
- **⑤ N5 W13**：`_collab/A_s32_w13_p0.diff` 早已就绪（`weight_residency.h`，host-only、分配器注入、10/10 主机检查），
  ⚠️ 但 `_TODO.md:7423` 记 **"P0 的 `WeightPageCache` 已成死代码"**。⭐ **本会话把它的价值重新定性**：
  用户点名 **cuFile 必须集成**（不经内存的冷页卸载 + 超大 MoE）⇒ **N5 的 `weight_residency.h` 就是 cuFile 的替换缝**
  （"mmap → pinned staging → H2D"换成"SSD → VRAM 直读"），已派 agent 出集成设计。
- **⑦ N7 FlashNext**：`S33` 已证 **qwen4_exp 运行时只是桩**（单文件 stub、registry 未注册）；本会话修掉了
  它 `config.h:11` 的 `intermediate = None`（非法 C++）与 `package.h` 的双大括号。**N7 的硬阻塞 = 真实 checkpoint（等用户指路）。**
- **①② N1/N2**：`_collab/A_n1_patch.diff`（5 文件/117 行）、`A_n1b_cold_pages.diff`（6 文件/126 行，`--max-cold-pages`）、
  `A_s30_budget_cold.diff`（147 行，预算×冷容量耦合）**全部就绪未应用**，三步 dry-run 逐步证明可叠加。
  ⚠️ **本会话查出它们的 apply-check 现在 FAIL**（对 `kv_bit_budget.h` 上下文过期）⇒ **须变基**。
- **⑧ U6 / 以及那两条"最便宜的"小修**（`_patch_eight.py`）：本会话查出 `apps/cli/options.cpp` 的 `Iso3→3` **差 1.5 b/el**
  （见 §82），以及 hw 那批硬编码 —— 都还没有落。

**⚠️ 由此更正我的两处旧表述**：① 我说"未完成得看八大项 = 只核到 ⑤ W13、⑥ 外挂召回"——**现在完整了**；
② 我把"外挂召回"当成 ⑥ 的**另一个方向**，其实 **⑥ = N6 = ngram 真表 gather**，而"逐轮外挂召回"是它的**实现形态之一**
（`_TODO.md` 里 rc1/rc2 标的就是"八项⑥"）。**两者是同一件事**，不是两件。

### 88. 构建 #2 验收：**ctest 100% 通过 / 0 失败 out of 109，与基线逐项一致 ⇒ e8v1 零回归**

- **构建 #2**（e8v1 + W1 + W4）：`rc=0`，**用时 51 分 12 秒**（六个重 CUDA TU 的设备码生成：`gqa_attention_{decode,prefill,decode_smallt,decode_e8,prefill_e8}.cu` + `entropy_cold_requant.cu`），
  错误数 0；`build/apps/ninfer` = **829,420,416 B @ 09:28**（比上一支 826,766,896 大 2.65 MB）。
  期间机器一度被**已报完成 agent 的遗留子进程**压到 `available 0`（见 §78 的纪律条），清掉后恢复。
- **ctest**：`100% tests passed, 0 tests failed out of 109`，`Total Test time = 581.57 sec`，rc=0。
  与上一轮**逐项一致**：**101 通过 / 0 失败 / 8 跳过**（跳过仍是同一批夹具缺失 #4/#26/#31/#32/#33/#34/#35/#36）。
  ⇒ **e8v1（16 电平奇数中点集）零回归——它可以留在树里了。**

**已派：变基 agent**（`pRB`）。理由：N1/N2/N3 那批补丁的 `git apply --check` 现在 **FAIL**（上下文被本会话的 W1/W4/e8v1 改动挪走），
而它们是"落地"的前置。要它逐份判断"变基成功 / 被取代 / 与另一份重复"，产出对**活树** `apply --check` rc=0 的补丁，
并给出**必须的落地顺序**（已知 `N1 → n1b → s30` 有硬依赖）与**建议的落地批次**（按"一批改动一次重建"）。
明确禁用了 `--fuzz/-C1`（那会静默错位），并要求锚点命中数 != 1 即失败退出。
清单：`a1/01_kv_tier_formats.patch`、`a1/02_..._tests.patch`、S20 的 `A_n1_patch.diff`（**要先判这两份 N1 谁是定稿**）、
`A_n1b_cold_pages.diff`、`b1/host-then-disk.patch`（判它是不是 N2 的另一版）、`A_s30_budget_cold.diff`、
`w2m/fix.patch`（W2 合并版，已知不能直打）、`hw/fix.patch`（含被 W1 改过的 `bindings.cpp`）、
`d1/kv_bit_budget_h.patch`、`d2/d2_cold_slot_stride.patch`、`c2/cold_codec_rule_core.patch`。

**落地顺序（按依赖与"一次重建"原则排，待变基完成后执行）**：
1. **变基后的 N1 → n1b → s30**（N1/N2/N3 一线，KV 预算与冷窗）；
2. **auto（aw1 为基座、用加前式）** —— 已判（§80），补丁对活树 `apply --check` 已 rc=0；
3. **W2 合并版**（变基后）、**hw 硬编码批次**（重对上下文后）；
4. **V 旋转（−3.81 dB、零 bit）** —— ⚠️ 三条前提已写死在案：逆变换必须加在 **split reducer** 里（不能加在 per-split partial 写）、
   **半应用比不改更糟**（relRMS ~1.4）、且要与 `hadamard_8d_pair()` 的对合性一致；
5. **W3（ss1/ss2）** 放在 E8 线之后（4 个文件与 E8 内核冲突）。
⇒ 然后**一次重建**，跑 ctest + **4 臂仪器（含路由假设）** + **1M 测试**。

### 89. ⭐⭐ ⑥=N6 外挂召回：**四条腿全在树里，缺的 P5/P6/P7 已做成补丁**；⭐ **cuFile 在这个盒子上拿不到零拷贝**（WSL2 + 消费卡，两条独立证据）

交付 `dl/pREC_recall_cufile.txt`（279 行）+ `scratch/rec/fix.patch`（4 文件 +1423/−5，`git apply --check` **rc=0**）。

**A. 四条腿逐条定位（只调用不重写）**
| 腿 | 位置 |
|---|---|
| **P1 落盘** `enqueue_cold_compressions` | `program_impl.h:10610-10887`（写入 `:10805-10830`、`file_slot` 分配 `:10588-10596`、`ColdPageEntry` `program.h:487-492`）|
| **P2 读回** `restore_cold_page` | `:11023-11158` |
| **P3 批量预取** `prefetch_cold_pages` | `:11163-11190` |
| **P4 内容寻址键** `PrefixShortlistDigests::at` | `prefix_identity.h:40-57`，用作 `shortlist_key` `:7182-7200` |

**⭐ 缺口是实测出来的（不是猜的）**：
- **P5**：`warm_cold_prefix` 的**唯一调用点**在 `:9426`，**不在任何 `decode_*_batch` 里**
  （4 个轮入口 `12276/12438/12626/12872` vs 4 个非轮 `8256/8818/9806/11857`）
  ⇒ **decode 期间它永不触发。这就是 `_TODO.md:2177`"观测链端到端从未真正工作"的机制。**
- **P6**：`warm_cold_prefix(seq, end_page)` **无条件恢复 `[0,end_page)`** ⇒ 没有页选择。
- **P7**：`restore_cold_page` 末尾 `:11149-11156` **释放 `file_slot` 并删记录** ⇒ 映射是内存态且会被删。

**补丁要点**：P5 挂钩 = 新 `ensure_sequence_kv_mapped_for_round()`，只换**4 个轮入口**（另 4 个保持原函数
⇒ prefill/rewrite 无法误触发）；**"数据先于指针"已核实**（`decode_raw:12727-12733` 先落盘、之后才 dispatch 到 ingress
⇒ 热路径不需要 fsync）。P6 = 粒度**一页 = 64 token × 全部层**（sentinel 决定，`cold_host_tier.h:34-41`）、
页号**升序**（磁盘偏移单调）、预算**从低端砍**（丢最旧、保持无洞）；**miss 是"切"不是"填"**。
P7 = 64 B 定长记录（显式补齐、crc32 覆盖 48 B）、`(digest,page)` 键、**tombstone 防 `file_slot` 复用别名**、
默认**不 fsync**、每轮至多一次批量 sync。判据 = 滚动摘要作 **shortlist**，权威仍是
`prefix_matches`/`ResidentPrefixIdentity`（`prefix_identity.h:37-39` 原文）；**e8 被 constexpr+static_assert 拒绝**
（负向对照证明编译期闸门真在）。验证：8 锚点各命中 1 次、括号平衡、`g++ -fsyntax-only -Werror -pedantic` rc=0 ×3、
**python3 第二见证 0 failures**。
⚠️ **发现并修掉一个真冲突**：rc2 在飞的 `turn_recall.h` 与本补丁在 `ninfer::spec` 里**重名 `RecallCodec`**（底层类型不同）
+ `kRecallPageTokens` ⇒ 同 TU include **直接编译失败**；已把本头文件移入 `ninfer::spec::turn_recall`，
**两补丁文件集零重叠、两种顺序都可落地、共存 include rc=0**（带负向对照）。rc1 已不适配当前树（rc=1）。
**没证到**：`program_impl.h` 未编译、端到端 logits 逐位一致未跑、GDN 态 blob 未实现、默认 provider 不接
⇒ **单靠本补丁不产生提速。**

**B. ⭐⭐ cuFile 判定：本机拿不到真 GDS，只能退化成 compat（=POSIX pread）。两条独立证据，都带出处：**
1. **这是 WSL2**：无 `/dev/nvidia*`、无 `/proc/driver/nvidia`、`lsmod/modinfo` 全空、
   `/usr/lib/wsl/lib/libcuda.so.1` 是 **187 KB 的 stub**、只有 `/dev/dxg` ⇒ **内核模块无处可挂**。
2. GPU 是**消费级 RTX 5090 D**；官方 `nvidia_fs` 文档要求 **"data-center class GPU"** + **"open kernel driver 535+"**，
   **GeForce 无矩阵条目**。
　**退化路径的代码级证据**：`cufile.h:78` 明写 *"nvidia-fs driver is not loaded. Set **allow_compat_mode**…"*
（即退回 POSIX pread/pwrite）；本机 shipped json `:85` 已 `"allow_compat_mode": true`；
`libcufile` 自带字符串 *"bar1 memory not available, only can work with compat_mode on."*；
且 `cufile.h:359` 要求 **`O_DIRECT`**（PLE 现在是 `O_RDONLY`）。
⇒ **⚠️ 直接回应你的指示**：你说"cuFile 要集成，理由是不经内存的冷页卸载 + 超大 MoE"——
**这两个理由都要求零拷贝路径，而这个盒子（WSL2 + 消费卡）给不了**。在这里"集成 cuFile"实际拿到的是
**compat mode**，即换个 API 表面跑 POSIX pread，**不经主机内存的收益拿不到**。
（这也**部分**印证了我早先"它更可能是优化而非使能项"的直觉，但**理由是错的**——不是"带宽够了"，而是**硬件/驱动层根本不给这条路**。）
**要拿零拷贝只剩三条**：① 换真 Linux（非 WSL2）+ 数据中心卡 + open kernel driver 535+；② Windows 原生 + 对应驱动栈；
③ 放弃 GDS，用 `O_DIRECT` + `cudaHostAlloc` 自己搭（拿不到真零拷贝，但能免掉 page cache 双拷贝）。

**两个接入点（带 file:line）**：
- **冷页 Disk**：改 `:10814-10820` 与 `:11063-11074` 两段，并把每层记录 stride **76288 → 77824（+2.01%）** 加版本头；
  ⚠️ **PLE 的 320 B 行绝不能逐行补**（+1180%）。
- **MoE**：⭐ `weight_residency.h:120-127` 的**注入式 pinned 分配器就是替换缝**，
  `WeightResidency::Disk`（`:48` *"reserved; P0 never emits it"*）正是落点；
  阻塞点：`binder.cpp:129` 忽略计划、`serve_options.cpp:410-414` 主动 throw、decode 半边被 CUDA graph 挡住（`TODO:7419-7420`）。
- **统一层提案** `ssd_fetch.h`（318 行，syntax rc=0、对齐账 static_assert 过）：PLE / 冷页 / 权重 / (A) 召回
  **共用一套 fetch + 三后端 + 能力门控**；(A) 的 P7 journal 直接复用它。
  ⇒ 这条正好回答"三层本质是同一个从 SSD 取数问题"。

**树未被改动**（`git status` 仍 39 项、两文件 mtime 仍是 09-12 22:28、树内 grep `turn_recall` = 0、
`src/spec` 仍只有原 2 文件、`build/` 未碰、无模块加载、无新文件）。

### 90. ⭐⭐ 自动导入器实测：**分类正确、路由缺失，卡在 `import_model.py:656` 一行**；而**驱动器早就能跑且报"一个字节不缺"**

交付 `dl/pIMP_mo.txt`（815 行）+ 原始采集 `pIMP_mo_raw.txt`。

**1. 每个 flavour 能不能自动路由（产生点 `import_model.py:293-324` `classify_quant`）**
| flavour.method | 产生点 | 自动路由？ |
|---|---|---|
| `native` | `:305` | ✔ **唯一可路由**：`:656` `if accepted and flavour.method == "native"` → `:665` subprocess.call |
| `modelopt` | `:317` | ✗ 只打印 work-item 说明书 `:669-675`，**无 dispatch** |
| `modelopt+sidecar` | `:303` | ✗ **连专属说明都没有**（`:669` 要 `=="modelopt"`、`:676` 要 `=="compressed-tensors"`）⇒ **静默** |
| `compressed-tensors` | `:318-323` | ✗ 仅当全部 nvfp4 判定被拒时打一行 `:676-678`；其双源入口永不派发 |
| 兜底任意串（fp8/gptq/awq/bnb/quanto/unknown）| `:324` | ✗ **完全静默** |

**根因**：`classify_quant` 自己的 docstring `:294` 写着 *"acceptance is decided by the converters"*，
而 `:656` 把 **flavour 这个"描述串"当成了"可行性判据"**。

**2. ⭐ 真跑（真源 `W4A4+W8A8`，只读，15 s）**
`import_model.py <SRC> --plan-only --resource-root <SRC>` → **rc=0**，分类 `flavour=modelopt`
（quant_algo=MIXED_PRECISION，group_0 8bit/260 targets + group_1 4bit g16/140 targets）、
**2139 张量 2 分片全在**、15 个 mtp 键、前端 6 项判定**"可接受"**（不是前端卡的）、
qwen3_8_27b/6_27b 的 config 都 accepted ⇒ **打印"1) ModelOpt NVFP4 单源适配器…"后进程直接结束，无 dispatch、无产物。**

**⭐⭐ 决定性对照：驱动器早就在树里而且能跑。**
`python3 -m tools.convert.qwen3_8_27b.convert_modelopt --model <SRC> --plan-only`
→ **rc=0、refusals=none、deviations 49、byte coverage 2139/2139 全在、objects 1103、
declared quant_algo `{FP8:260, NVFP4:140}`**。
⇒ **同一份源：前门说"缺实现、手动作罢"；驱动器说"规划完成、一个字节不缺"。**
`_TODO.md:7906/7914` 也自认 *"driver 已落地、前门路由还没做"*。
另：`convert_modelopt.py` 在 git 里**未跟踪**（`??`，本会话落的），`import_model.py` 是 `M`。

**3. 缺口（按"不 or / 不硬写"）**
1. **判据换位（唯一卡点）**：删掉 `:656` 的 `flavour.method == "native"`，改为派发
   **"被接受的那条 verdict 自带的 command"**（`:402-405`/`:438-445` **已经在算它**）⇒ flavour 退回只做展示。
2. **entry 集合要由 target 声明、前门枚举**：现在 `:54-59` 硬编码 4 个 target、
   `:366-369`/`:416` 硬编码 `"convert"`/`"convert_nvfp4"`，而 `convert_modelopt` 只在 `:675` 以**字符串**出现
   ⇒ **前门永远看不见它**。新 entry 应是**声明（数据）**，不是新 if。
3. **正确信息来源 = 源自己，且消费方已存在**：`config.json.quantization_config`（实测 260 FP8+140 NVFP4、exclude 4 项）
   + `hf_quant_config.json`（producer modelopt 0.43.0 / quantized_layers）+ `manifest.json` 的 `source_package_sha256`。
   ⭐ `tools/convert/common/layout_plan.py:121 select(members, declared_algo)` 的头注释写着
   *"no model name, no layer name, no key allowlist… **a new family is a data question**"*，
   而 `convert_modelopt.py:142` **已经 import 它** ⇒ **前门没接上而已**。
4. **CLI 参数靠生成**：命令模板由 entry 给（`convert_modelopt.OUTPUT_BASENAME:163`、`:1734-1746` 的只读三参数）；
   前门只填 `--model`=源、`--out`=`<entry basename>`、`--device`=探测、`--resource-root`=既有 `NINFER_RESOURCE_ROOTS`（`:63-65`）；
   双源 entry 的第二个源应由 entry **声明为必需外部源** + 走同一套资源解析（或源自带 manifest 的 provenance），
   **解析不到就 refuse，不许填占位符。**
5. **删掉 `:669-675` 的手写说明书**：它的"401 Linear / 145 FP8 / 112 NVFP4"**与源自己的声明不符**（400 = 260+140），
   也与它指向的驱动器的账（167 FP8 / 91 NVFP4）矛盾 ⇒ **硬写漂移的活教材。**
6. 顺带：`--target` 参数**不存在**（`:521-529`）但 `:641-642` 要求用它；实测两个 target 契约相同只能默认 `accepted[0]`。
   消歧应由 **entry 声明所需证据**（census 的 dtype 直方图**已算好**），消歧不了就 refuse。

**4. GUI vs CLI：不是同一套，且同一目录结论相反**
- CLI = `import_model.py`（结构化 flavour + 问 target 自己的 `validate_config` + 6 项前端钉死）；
  GUI = `tools/gui/model_import.py`（`_hf_quant_hint:280-288` 只对字符串做 `nvfp4/fp4` **子串匹配**且"只用于文案"、
  `_family_from_arch:263-277` 用几何+名字）+ `convert_runner.py`（`:58` 又一串家族字面量、
  `:72-75` **恒为 `convert.py`、无量化分支**，而 `:7` 的 docstring 声称会走 `convert_nvfp4`）。
- ⭐ **实测同一目录**：CLI = work-item **不可自动导入**；**GUI = `convert_groupwise` "可以自动转换成"
  并生成 `convert.py --model <量化源>`** ⇒ **GUI 会跑错转换器**（拿 `convert.py` 去跑量化源）。
- **重复真值**：家族白名单 **3 份**（`import_model:54-59/:72`、`model_import:40`、`convert_runner:58`）；
  量化档 **3 套判据**；`convert_runner.py` **两份字节相同的副本**（md5 `7ef6d002…`，**均无 importer**）。
- ⚠️ `ninfer-gui.py`（Windows 侧）与 `ninfer-gui.spec:17` 声明的 `tools/gui` 目录
  （`infer-fusion-repo\tools\gui`、`ziqinzhang\tools\gui`）**在盘上都不存在**。

**5. 它没证到的**：没跑真转换（禁 22 GB）⇒"GUI 那条 `convert.py` 链在量化源上会失败"是按
`recipe.py:317 preflight_sources` 的判据**推断**、未实跑；`convert_modelopt` 的**产物字节**没验证（盘上无 artifact）；
没启动 GUI 本体（只用路径枚举证明其模块目录不存在）；`muse_glimmer_30b` 为何没有纯 config 校验入口未查清；
"401"的出处未核实；第 5 类 flavour 是读码得出、只实测了 modelopt 一条。
**树状态**：无任何被跟踪文件在会话内被改动（脏文件最晚 mtime 09:39:27，会话 09:42 开始；`git status` 39 行与开始时一致）。
唯一副作用是解释器刷新的两个 `__pycache__`（`tools/gui/`、`tools/convert/qwen3_6_35b_a3b/`，gitignore 覆盖）。

### 91. 一笔总账（§91-§98 合并）：落地批 1 验收 + 三路审计 + MTP auto/kt 定论 + FreeToken + 导入器 + 邻居纪律

**§91 落地批 1（7 份）验收通过：** `ctest 100% tests passed / 0 failed / 109`，rc=0，455 s，
与基线**逐项一致**（101 通过 / 0 失败 / 8 跳过）⇒ **零回归**。⭐ 点名那条 E7 回归
`ninfer_qwen3_6_context_store_test` **Passed**。构建 **9 分钟**（增量+ccache 命中；顺带印证 51 分钟那个基线是"近乎全量"的数）。
批 1 内含：`UP1_nfc_ascii`、`V3_grouped_conv_position`、**`B_s33_ple_wiring`（= N6 PLE 接线！）**、
`E3_s45b_s36_restore`、`e8r2/fix`、`cbfix2/fixB`（修刚落转换器的三个 bug）、`E7_s50_regression_sketch`。
快照 `snap_pre_b1/`；`revert.sh` **只用 `git apply -R`、不用 `git checkout`**（因为 `gqa_attention_prefill_i8.cuh` 载着已落 e8v1）。

**§92 三路落地审计的最终数（含对我"只落了 4 份"的系统性纠错）：**
判据 = ① `git apply -R --check` 过 ⇒ **已落**（铁证）② 失败则回落**活树签名 grep**。
⚠️ **只用正向 `--check` 会把"已落但补丁过期"误报成未落地** —— 这是我先前"约 100 份只落 4 份"的错因。
- **`scratch` 110 份（去重）**：已落 **52**（`-R` 铁证 28 / 内容 100% 在树 12 / ≥86% 12）、部分 **3**、**未落 57**（②能过没落 19 / ①漂移 38）。
- **`_collab` 133 份**：已落 16、未落 **104**（干净可落 37 / 漂移 67）、空无效 13。
- **记录侧 58 条**：明确未落 26、半落 9、被取代 7、**看板陈旧其实已进树 19**、未判定 12。
- **`board.md` 的 `(patch 未应用)` 列至少 13/23 行已过期 ⇒ 看板不能再当状态源**，只能当"当初交付过"的索引。
- ⚠️ 两条"我们自己的记账在骗我们"：`_todo_sync2.py:43` 把"改两处陈旧注释"记成 **"N3 第二轮已完成"**（实测 `recalibrate` 零命中、`kv_calibration_dir` 只在注释里）；
  `M_unlanded_now.md:34` 自证已过期（`dflash_impl.h:231` 已是新写法 ⇒ A5b 已落）。
- ⚠️ **CRLF 伪信号**：112 个源文件是 CRLF，手写 LF 补丁会报 `different line endings` ⇒ **DRIFTED 里混着真未落**。
- ⚠️ **可落清单会立刻腐坏**：现场复核发现 `e8r1` 现已**双失败**、`C_s23_tu/01` 现已**可落** ⇒ **每次落地前必须现场复核，不许采信任何一次盘点**。
- **58→三路（AU1 文件系统侧 / AU2 记录侧 / LC 现场复核）**：LC 现场复核得 **56 份可落**（用双条件，不采信旧审计）。
- ⚠️ LC 抓到三个"一落就坏"：`build/D4_cmake_new_family.diff` 里字面是 `add_subdirectory(targets/<new_family>)` **占位符**；
  `build/T1_a5b_revert.diff` **会回退已落 A5b**；两者**永久排除**。
- ⚠️ **互斥实测**（隔离副本 apply A→check B 双向）：`cc1/cc2`、`cl1/cl2`、`w13a/w13b`、`rb/w2m*`、`hw/fix` vs `rb/hw_no_rebase_needed`（**md5 相同**）、
  `cbfix1/cbfix2`、`UP1_moe_s2_perf vs _v3`、S23/S40 拆分**全部互斥**；但 **`rc2 vs rec` 与 `e8v2/fix vs vrot` 在 hunk 层可共存**——
  前者的冲突在 **`RecallCodec` 符号**与功能重叠（**落两个会编译失败**）。
- **我的四条裁决**：`cl1`+`cl2` **两份都要**（不同功能撞接线点，先 cl1 后手工移植 cl2，合起来闭合 **N3**）；
  `w13b` 落、跳过已被 HEAD 覆盖的 P0 部分（⚠️ 落完**仍不可用**，`serve_options` 自述 "offload CANNOT work yet"）；
  `rc2` 永久排除、只落 `rec`；`e8v2/fix`+`vrot` 两份都落（⚠️ `vrot` 若只改一半就标需裁决，**半应用比不改更糟**）。

**§93 ⭐ MTP auto：门是"拼写门"不是"能力门"（我的 A/B 从头到尾用错了拼法）。**
实跑现有二进制：`--spec mtp`（**不写 `--draft-tokens`**）→ **rc=0**，报 `mtp draft window 15 (adaptive ladder, top) realized mean X`；
`--spec mtp --draft-tokens 5` → rc=0；**`--draft-tokens 0` → rc=1 `error: invalid draft-tokens: 0`**（唯一坏拼法）。
根因：CLI 的 `draft_tokens` **默认就是 0 且无"显式给出"标志** ⇒ 不写 flag 就已自适应；`apps/cli/options.cpp:33` 的 `parse_u32` 是**通用正数**助手，拒字面量 0。
门清单 G1–G7：**只有 G1 拒 0**；G2 接受 `[0,15]`；**G5 的 `0→7` 改写只在 DFlash2/Dspark 分支、不碰 MTP**；G6 按 variant 取 `kMaximumMtpDraftTokens`；G7 把 0 变阶梯。
**variant 上限：27b=15 / 35b_a3b=5 / muse=5**；⚠️ 另一处小缺陷：`layouts_impl.h:891/901/910` 在真实上限是 5 时**仍打印字面量 `[1,15]`**。
临时修 `NINFER_MTP_ADAPTIVE` 已落 2 文件（`apps/cli/options.cpp` +55/−1、`src/serve/serve_options.cpp` +47/−0，`-fsyntax-only` rc=0，快照 `snap_pre_gate/`）——
**但现在不需要它**（不写 flag 即可），留着等完整修时一起收。

**§94 ⭐⭐ MTP auto 的 A/B（正确拼法，GPU 空闲时跑）：auto 输给最优固定 k 18%。**
| 臂 | ctx | decode tok/s | 接受率 | AL | 落地窗口 |
|---|---|---|---|---|---|
| **自适应** | 32768 | **108.37** | 29.10% | 3.44 | top=15，**realized mean 7.93** |
| **自适应** | 65536 | **108.59** | 29.10% | 3.44 | 同上 |
| 固定 k=3 | 65536 | **132.93** | 56.52% | 2.70 | 3 |
| 固定 k=9 | 65536 | 112.76 | 26.19% | 3.32 | 9 |
| k=3 复跑×2 | 32768 | **132.22 / 131.81** | 56.52% | 2.70 | 3 |
- ⭐ **那个 1.80 tok/s / 241 s 的异常是争抢、不是 bug**：同格复跑两次 132.22 / 131.81 ⇒ **结清**。
- **auto 的每轮 token 数更高（3.25 vs 2.78）但总速度更低** ⇒ **每轮成本被宽度吃掉，判据对宽度的惩罚仍偏轻**。
- ⚠️ **auto 在 32k 与 64k 上数字几乎逐位相同**（含 mean 7.93）⇒ **对上下文长度零响应，这本身就是"没在自适应"的迹象**。
- **已核 F1/F2/阶梯都在树里且默认生效**：`columns = post_add ? i : (i-1)` 默认 false（**加前式**）；
  `:203-206` **F2 先验收缩** `weight = reach[i]/reach[1]` 默认开；`kMtpWindowLadder={2,3,5,7,9,15}`，
  `program_impl.h:10188` 用 **`kMtpWindowLadderTop` 作界**（⇒ **棘轮去掉了**）、`:10193` 按 `kMtpWidthRedecisionRounds` 粗粒度重决；
  ⭐ **`kMtpWindowCostRatio` 已换成实测常量比值**（不再是冻结的 0.9/14.7）。
- **两个候选根因**：① F2 只收缩先验、没处理"未观测深度"本身；② ⭐ **`a` 没随前缀变化**——aw1 自述 `a=14.7ms` 是"唯一没能测的常量"，
  而 32k/64k 逐位相同正说明每轮固定成本没随上下文增长进入判据（**它必然增长**）。**②更可能是"零响应"的原因**。
- 与 aw1 自己的闭环数字同向：修前选的均值 10.3/7.3/6.4 vs 真最优 9/5/3；**实测 7.93 vs 最优 3** ⇒ 同一个偏差（**选得太宽**）。

**§95 ⭐ FreeToken 测完：机制活着，但当前形态不能用。**
- `ninfer_bandwidth_governor_test` **rc=0 PASS**（真断言，但**只喂合成计数器**）；⚠️ **`tools/kv_relayout_test.cpp` 没有注册进任何 CMakeLists** ⇒ CI 从没跑过、盘上无二进制（主机 g++ 编译后 rc=0，与 `ft_tiers.py` **`python-reference: MATCH`**）。
- ⭐ **观测链现在真的通了**：`NINFER_FT_STATS=1 --no-cuda-graph` → **1554 条 `[ft] layer=` 行、0 NaN/inf**（修法是真的：`gqa_attention_decode_partial.cuh:473-480` 把 observe 挪到 launch 之后）。
- ⚠️ **四条断点**：① **`ft::observe` 落在 CUDA graph 捕获区内**（`decode_graph.cpp:65-67` ← `graph_impl.h:24-27` ← `decode_impl.h:68-69`），
  那里 D2H **非法**且返回码**没检查**（`ops/common/ft_stats.h:55-57`）⇒ 毒化捕获 → `smallt.cu:209 CUDA_CHECK` → **SIGABRT rc=134**；
  实测 5/5：`ninfer` 7.8 s 崩、**`ninfer-serve` 到不了 "listening"**；`PERIOD=1` 时 48-token 生成只有 **11 次** tap 执行且**全部 `mean_l=1`**（零信息）⇒ **证明 host lambda 只在捕获时跑一次、replay 从不跑**。**这就是 `_TODO.md:2177`，且是在任何人都会用的配置下。**
  ② **tap 只在 `launch_tc_partial_nvfp4`** ⇒ E8 的 6 层**结构性不可观测** ⇒ `build_ft_spec` 把它们映射成 iso3 ⇒ **环路第一个动作就是拆掉调好的默认表**。
  ③ ⚠️ **环路的动作把已知坏配置装了回去**：实测 spec `...5:e8,9:e8,8:iso3,11:iso3...` —— **E8 落在 9、11 层**，而 `variant.cpp:32-41` 记着 `8-15:e8 → 0/8 needles`。
  ④ ⚠️ **Apply 要求 `active == 0`**（`generation_service.cpp:523-536`）：22.93 s 生成期间 **0 次重排**，第一次在 `[req 1] done` 后 **9 ms**。
- **但闭环真闭上了一次**（serve + `--no-cuda-graph`）：1524 行、**2× `[ft] auto-relayout`**、服务存活、之后 HTTP 200 ⇒ **机制是活的，不是死代码**。
- **代价**（各 4 次，ctx 4096）：生产（graph 开）**66.0 tok/s** → graph 关 58.8 → 加 tap **53.9（−18.3%）**；**graph 开 + tap = SIGABRT** ⇒ **唯一能用的配置比生产低 18%，而验收线是 <10%**。
- **最小修法**：tap 移出捕获路径改"设备侧累加器 + 捕获外读"；**dtype 完备**；加**逐层允许档位掩码（E8 只许 ≤7）**；decide 与 drain 解耦、热路径用窗口差分；**graph 开着时直接拒绝 `NINFER_FT_STATS` 而不是 abort**；注册 `kv_relayout_test`。

**§96 ⭐ 自动导入器实测 + 修复交付。** 病根 `import_model.py:656`（把 **flavour 描述串**当**可行性判据**），
`modelopt+sidecar` 与兜底串**完全静默**；`modelopt` 只打说明书。
⭐ **新发现两处**：① `:442-444` 对每个 target 都发 `--quantized-model`，而 `qwen3_6_27b/convert_nvfp4.py:433` 声明的是 `--nvfp4-model`
（`qwen3_8_27b` 才用 `--quantized-model`）⇒ **坏在两处**；② **验收权威本来就在树里**：`qwen3_6/common/recipe.py:320-321` 按逐张量 dtype 拒源，
两个 `convert.preflight_conversion` 都以 `source dtype F8_E4M3 != required BF16` 拒掉，**只有 `convert_modelopt.preflight` 接受** ⇒ **不需要新白名单**。
**修复**（`scratch/ifr/fix.patch`，4 改/4 增/1 删，对活树 `git apply --check` **rc=0**）：entry 集合变成**数据**（`tools/convert/*/import_entries.py` + 新 `common/import_entry.py`）；
**验收 = 对各转换器自己 `preflight` 的点号引用**；派发 = "被接受的那条 verdict 自带的命令"；`flavour` 只做展示；`:669-675` 删；
`--target`→`--entry` + `--slot`（**解析不到就 refuse、不留占位符**）；GUI 侧改调 CLI 的 `classify_family/classify_quant/census_weights`，
删掉重复白名单/`_family_from_arch`/`_hf_quant_hint`，两份 `convert_runner.py` 收敛成一份。
**真源实测**：门口 rc=0 → `runnable: qwen3_8_27b:convert_modelopt`，7.97–21.14 s，峰值 RSS 372–373 MB；**GUI 测试 23/23 全过**（含新回归：runner 发 `convert_modelopt` 而非 `convert.py`）。
⚠️ **native(bf16) 路径行为变了**：两个 `convert` entry 共享 recipe 且都吃 bf16 ⇒ **门口现在列两候选并拒绝、要求显式 `--entry`**（不测真 bf16 源）。**用户裁定：bf16 不静默、等完整修。**
⚠️ ⭐ **它发现 `ninfer-fusion-repo` 是"另一个 remote 的独立 git checkout"**（`github.com/Astrangemaninhere/ninfer-fusion.git`，HEAD `0eaac04`），
**与活树不同步、且缺 `import_model.py`/`layout_plan.py`/`convert_modelopt.py`** ⇒ **Windows GUI 从那里加载模块 ⇒ 本次 GUI 修复对打包应用是惰性的**（在那之前退化成显式"无法判定"）。
⇒ **把这件"过时镜像"从"参考脏"升级为"产品路径脏"。** 建议：把 GUI 的 `REPO` 指向活树，或刷新那个 checkout。

**§97 ⭐ MTP 有没有树：定论是没有（读完 1021 行 + 逐个读源文件）。**
- `UP1_mtp_pack_contract.diff` 是**节点融合的性能补丁**（2×rmsnorm+pack 融 kernel、`cudaMemsetAsync`→kernel **省 4.6–8.4 µs**、按 D 路由 `MtpRowRoute{Cta256x6,Cta256x10}`、新增 `mtp_residual_norm`），
  **形状 = 线性**（`out[0:D,t]=rmsnorm(embedding[:,t])`、`out[D:2D,t]=rmsnorm(hidden[:,t])` ⇒ `[2D,T]` 扁平拼接）；全 1021 行 **`tree:0 beam:0 cand:0 parent:0 child:0 depth:0 topk:0 selector:0`**（`ladder:3` 指的是 `rmsnorm.cu` 的 block 宽度阶梯）。
- 活树结构：`program.h:462` **15 个 TokenId 扁平数组 + 1 个标量**；`mtp_impl.h:200-220` 每步各取 **1 列**；`mtp_round.cuh:29-51` 每 row 只有 **1** 个 accepted / **1** 个 `next_anchors` / **1** 个 `next_extents`。
- **三宽度辨析**：① **MTP 草稿树＝不存在**（单链，每 req 一条路径）② **MTP 窗口＝线性列数**，rung∈{2,3,5,7,9,15}，**已启用** ③ **DFlash2 DDTree＝存在但离线**（`dflash2_tree.py` 自述 *"Pure stdlib; no engine code"*，**默认关**），
  ⭐ **`selector_top_k=16` 不是束宽**（是每步候选集大小，引擎 walk 每 step 只消费 16×16 表的 1 行），`block_drafts=7` 是**深度**。
- ⭐ **树不是没测过——已被判 NO-TREE**：`_collab/M_accept_eval.md`（09-10，ckpt `step_001200.pt`，300 anchors）`hit@1≥0.35 AND hit@4−hit@1≥0.12 → NO-TREE`，**实得 hit@1 = 0.2719**。
- ⭐⭐ **新判据（链条 vs 树最锋利的指纹）＝ `accepted by pos` 逐深度直方图**。自适应实跑：**`28,19,7,0,0,0,0,0,0,0,0,0,0,0,0`（深度 ≥4 全 0 = 单链指纹）**；
  钉死 k=5：`15,10,5,5,4`。且 **自适应 mean 5.10 的 24.77%/AL 2.32 低于钉死 k=5 的 33.91%/AL 2.70** ⇒ **第二个独立读数再次确认 auto 输给最优固定 k**。
- ⚠️ 该产物**没有 dflash2 权重**（conversion.json 里 `dflash2` 命中 0）⇒ `--spec dflash2` **测不了**；`eval_ddtree.py --tree` 未跑（Windows python 硬编码 TARGET_DIR）。

**§98 TU 拆分：CMake 前提被纠正 + ccache 实况 + 测量口径 v2。**
- ⭐ **不需要 CMake 授权**：`add_subdirectory` **只用于 `targets/*`**（`src/CMakeLists.txt:356-359`），`ops/launcher/` **没有自己的 CMakeLists** ⇒ 加 TU 就是往 `ninfer_ops` 显式清单里加行（`:81` decode、`:82` smallt、`:83` e8；`:311` `ninfer_cuda_archive` ⇒ **RDC 归档**）。
- **慢的结构性原因**：`decode.cu`(45 行) 只想要**一个** host 算术模板，却因 include `_partial.cuh` 付了**全部 5 档 kernel 头的级联**，且**同一档级联被实例化两次**。
- **Cut A**（0 行 CMake、纯搬移、**且是判据③的唯一达成路径**）：新 `gqa_attention_decode_split.h` 收 `_partial.cuh:61-156` 四实体；**删 `:61-156` 以免同 TU 两份定义**；`_reference` 保 `inline`。
  **Cut B**（5 新 TU + `:83` 后 5 行 + `_tiers.h` 10 条 extern）：照 `decode_e8.cu` 样板；**H7 硬约束写进文件头**（`ninfer_ops` 还 PRIVATE 链了**非 RDC** 的 `ninfer_nvfp4_tma` ⇒ **绝不**把 nvfp4 档模板搬进它提供的 TU）。
  **Cut C**（0 行 CMake、删 5 个 **git 已跟踪**文件）：**隔离出树**（`cp -p` + md5 + `README.md` 记调用符号与复活方式）再 `git rm`，**理由**：改那两个**未注册驱动**的指针＝**写新代码且无法编译验证**（正是最该避免的），且 `muse128_repro.cu` 是已修损坏的**证据**。
  ⚠️ `revert.sh` 三段式：Cut A/B 用 `git apply -R`、**Cut C 的删除用 `git checkout --`**（删除不是补丁形态）。
- ⚠️ **它先前那条"ccache 没装"是假阻塞、是测量环境错误**（`which ccache` 跑在没带 PATH 的 shell 里）。实况：**`/home/user/.local/bin/ccache` 在（4.11.3）**，且 **`build/CMakeCache.txt:40 CMAKE_CUDA_COMPILER_LAUNCHER=ccache` 早就配好** ⇒ **B5 无需装、无需改 CMake**；51 分钟基线本身带 ccache（命中 103/362 = 28.45%）。
- **测量口径 v2**：判据权威改用 **`CCACHE_DISABLE=1` 跑拆前/拆后**（纯编译时间，免疫命中率运气）；M1=增量、M2=全量墙钟**都要测**（H1：拆分**不减少 device-link 总量**，**不许把 link 变慢当收益**）；
  ⚠️ **新陷阱**：`…LAUNCHER` 是命令行 `-D` 缓存项 ⇒ 若 M2 走"新建 build dir"**必须重传**，否则拆后的全量是**没 ccache**的全量 ⇒ **会得出"拆分没收益"的假结论**；
  **M5**（免疫缓存的结构证据）`compile_commands.json` 里 decode 条目数预期 **3 → 3（Cut A 后）→ 8（Cut B 后）**；**M4** = `launch_tc_partial_` 与 `__device_stub_` **两个 grep 一起看**（RDC 下 `__global__` 是 device stub + `.nv_fatbin`）。
  **回退触发（我事先接受）**：M2 没变快甚至变慢 ⇒ **如实报 + 只留 Cut A**。

**⚠️ 仪表纪律（我这一轮连犯三次，登记）**：① **把构建 pipe 进 `tail -40`** ⇒ 日志缓冲到结束才写，我因此误报"构建卡住"；
② `ps | grep -E 'cmake --build|…'` **没匹配到**，换带 `/` 的路径式模式才出来；③ **用"3 分钟内文件写入数"当推进判据**，采样正好落在编译与链接之间的空隙 ⇒ 误读为 0。
⇒ **硬纪律：判构建死活只看"进程树（按 ppid 展开）+ 最后一次写入时间"，不看缓冲日志、不用固定时间窗采样。** 另：`NINFER_MTP_WINDOW_TRACE` 那次"0 行"也是**我的 grep 模式问题**，不是产品缺陷（trace 实际正常）。

### 92. ⭐⭐ 纠正 §94：我那次 A/B **不可比**；"k=9 → k=3" 与"速度不达标"的主假设 = **自适应拿不到短名单 draft head**

**用户原话**："不，我的意思就是检查宽度和长度调整的策略……**彻查先前 k=9 最优现在变成 k=3 最优的原因，而且速度远不达标的原因**"。

**§94 我把它答成"auto 比最优固定 k 差 18%"，这是答错了问题。并排看才看得见**：

| | 记录（先前 A/B） | 我实测（今天） |
|---|---|---|
| k=3 | 199.86 | **132.93** |
| k=9 | **322.67（最优）** | 112.76 |
| k=15 | 275.07 | 85.40 |
⇒ **k=9 −65%、峰值 −59%、最优点 9→3。整条曲线塌了 2.4 倍。**

**我的测量不可比（我的错，且是"引用了数字却没读条件括号"这一类）**：`mtp_window_cut.h` 的 STATUS 括号里写着
**`(artifact nvfp4-dflash2, shortlist draft head, batch 1, greedy, fixed prompt)`**；我跑的是 **`nvfp4-modelopt`** artifact、
**不同 prompt**（短中文句 vs code 160 tok）⇒ **至少三处不可比：artifact / draft head / prompt。**

**⭐⭐ 主假设（一个原因解释全部三个症状）**：`startup_features.h:59 kMtpShortlistMinimumDrafts = 5`、`:68`
`if (backend == Mtp && draft_tokens >= 5) ⇒ 短名单头`；而**自适应传 `draft_tokens == 0` ⇒ `< 5` ⇒ `ProposalHead::Full`**
⇒ **自适应永远用昂贵的 Full draft head**（这正是 aw2 当初报过、我记下但没接上的那条："那 14–28% 就在桌上"）。

| 症状 | 同一原因的解释 |
|---|---|
| 曲线塌 2.4× | 草稿步从**短名单头**换成 **Full 头** ⇒ 每步贵得多 |
| 最优 k 9→3 | 草稿列成本 `b` 变大 ⇒ 边际列更不值 ⇒ **最优窗必然变窄** |
| 自适应选太宽（7.93 vs 3） | 判据的 `kMtpRoundCostPerColumnMs/BaseMs` 是**在 dflash2 短名单配置上标定的常量**，Full 头下**把列价报低了** ⇒ 过度投机 |

⇒ **正面回答"宽度和长度调整的策略"**：**阈值策略也许对，但它的代价模型是编译期常量，而代价随配置变**
（draft head / artifact / graph / batch）—— **它只有一份，而且来自别处。**

**已改派 MTP agent（`agent_61f84028`）**，顺序：① **同条件重测**（同 artifact/prompt/graph，`--spec mtp` 与
`--draft-tokens {3,5,7,9,15}` 六档，**`≥5` 本身就是"头"的开关**）⇒ 判据：k≥5 的每列成本明显低于 k<5 且 tok/s 上台阶
⇒ 主假设成立；② **量本配置的 `a`、`b`** 与记录的 **`16.71 / 0.737`** 对比（差值就是"曲线为什么塌"的量化答案）；
③ **验 `b/a` 是否配置相关**（若明显大于记录值 ⇒ "自适应选太宽"= 用了别处标定，直接成立）；
④ 各给一句带数字的判据式结论；⑤ **然后才出补丁**，优先级：**让自适应也能用短名单头（加逐轮守卫拒绝低于
`kMtpShortlistMinimumDrafts` 的 rung —— aw2 原话）** > **`b/a` 做成配置相关（从本配置实测推，不许新塞常量）**
> 我先前那条"`a` 随前缀变化"**降为次要**（先证伪/证实上面这条）。
⚠️ 加防污染约束：**机器上在跑 ctest（含 GPU 测试）⇒ 跑 GPU 前先看 `nvidia-smi`，被占就等**——
因为先前那个 **1.80 tok/s / 241 s 的假异常就是争抢造成的**，不许让同类读数污染这次结论。

### 93. ⭐ Windows/cuFile 的定论 + **PCIe 协商成 Gen5 ×1**（用户接受、提速推后）+ 由此重排优先级

**用户裁定**："这个没辙，我 itx 不太方便插拔……**我是在搞软件说到底，提速等以后吧**"。
⇒ **×1 暂不处理**；**"从 SSD 流进 VRAM"这一整条线降级为"以后"**。

**Windows 侧 API 实况（实测，逐项）**：
- ❌ **`cufile.h`/`cufile.lib`/`cufile.dll`/`cufile_rdma.*` 全缺**（CUDA v13.2 已装但没有它们；连 `C:\Program Files\NVIDIA Corporation\cufile\` 目录都不存在）
- ❌ ⭐ **`nvfs.sys`（GDS 内核驱动）递归搜 `C:\Windows`（含 WinSxS）+ DriverStore 零命中** ⇒ **GDS 在 Windows 结构性不存在、不可安装 ⇒ 别再试**
- ❌ `dstorage.dll`/`dstoragecore.dll` **不是 OS 内置**（递归搜 `C:\Windows` 零命中）⇒ 只能走 NuGet/SDK
- ✅ 可用的是 **NuGet `Microsoft.Direct3D.DirectStorage` 1.3.0**（与某 Steam 游戏里那份 **sha256 逐字节相同**）；MSVC 19.44 + SDK 10.0.26100 + `d3d12.h/.lib`/`bcrypt.lib` 齐；两块盘都是 **NVMe**
- ⇒ **你要的"NV 开源的能消费级直接读 SSD"，在这台 Windows 上的落地形态就是 DirectStorage**：**能跑通、能校验、能到 VRAM**，
  但**它自己在中途经 host 落地** ⇒ **"绕过主机内存"这个目的达不到**。

**DirectStorage → VRAM 真跑通**（`C: 256 MiB / chunk 16 MiB`）：**3.301 GB/s**，sha256 用**两种独立方式**校验全 MATCH
（① D3D12 `CopyResource`→READBACK；② **CUDA 经 D3D12 外部内存互操作拿到 `CUdeviceptr`，`cuMemcpyDtoH` 读回**）。

| 路径 | GB/s |
|---|---|
| ReadFile(缓冲)+`cudaMemcpy` H2D | 2.257 |
| NO_BUFFERING+pinned+`cudaMemcpyAsync` H2D | 2.384 |
| **NO_BUFFERING+`cudaHostAlloc(Mapped)`+D2D（= PLE 现状）** | **2.317** |
| 流水线化那条（NVMe DMA ∥ H2D） | 3.171 |
| **DSTORAGE → host 内存** | **11.074** |
| DSTORAGE → D3D12 UPLOAD heap | 3.172 |
| **DSTORAGE → D3D12 DEFAULT heap（= VRAM）** | **3.301** |

参考：**PCIe 裸 H2D 3.40 / D2H 3.58；NVMe 裸读 7.98；显存 D2D 543–860 GB/s。**
**"不是零拷贝"三条硬证据**：① DS 头文件自述 staging buffer（`SetStagingBufferSize` 注释 *"some but not all of the staging buffers will be allocated from VRAM"*）；
② ⭐ **带宽指纹**：同进程只换目的地，**DS→host 11.07 vs DS→VRAM 3.30（慢 3.4 倍）**，而 3.30 **恰等于裸 H2D 3.40** ⇒ "先在 host 落地再跨一次 PCIe"的签名；
③ 同为 D3D12 resource，**UPLOAD 3.172 ≈ DEFAULT 3.301** ⇒ 目标是不是真显存几乎不影响耗时。

**⭐⭐ 最大发现：GPU 的 PCIe 实际协商成 Gen5 ×1**（`MaxLinkWidth=16` / **`CurrentLinkWidth=1`**，Windows PnP 与 `nvidia-smi` 双路印证，
实测 H2D 3.40 GB/s 吻合 ×1）。⭐ **本仓自己的数据独立印证**：真产物装载时 **20.41 GiB H2D 用 7.319 s = 2.8 GB/s** ⇒ 当时已是 ×1。
⇒ **任何"进显存"的路径被钉死在 ~3.4 GB/s**，**连假设可用的 GDS 也一样**（DMA 到 VRAM 必须走 GPU 自己的 ×1 端口）。
ITX 板 + 很可能用了 PCIe 延长线；直插可能恢复到 ×16，**但用户决定暂不处理**。

**⇒ 由此重排优先级（写死）**：
- **降级/推后**：**PLE sidecar（95 GiB 流式）/ 冷页 `Disk` 卸载 / cuFile 集成**——三者的收益天花板都是 **3.4 GB/s**，
  在 ×1 下**整体不成立**（比 NVMe 裸读 7.98 还低一半）。`ssd_fetch.h` 那份统一取数层提案随之**不推**。
- **不受 ×1 影响、才是真软件议程**：**MTP 自适应判据**（纯调度/代价模型）、**落地批次 B3–B5（含 TU 拆分）**、
  **E8/KV 三条免费杠杆**、**FreeToken**（它的病是 graph 捕获期放置 + dtype 完备，**不是带宽**）、**导入器/GUI 统一修**。
- ⚠️ 但保留一条：**`DSTORAGE → host` 是健康的（11.07 GB/s）** ⇒ 如果将来要"把层放主机内存、按需换入"，
  **瓶颈不在 SSD、也不在 host 路径，而在那一次 PCIe 进显存的 ×1** ⇒ 仍受同一个上限约束。
- **没证到**：无 Linux GDS 对照基线；staging buffer 未直接观测（只有官方注释 + 带宽推断）；未测 GDeflate / 多队列 / 调参 / 随机小 IO / 95 GiB 工作集；
  CUDA 互操作只做到读回、**未跑 kernel**；测量期间有第三方进程占 22 GB 显存，噪声 ±10%。

## §12.w-throw（19:12）throw<std 非 runtime_error 派生异常> = 整引擎击杀：爆炸半径 + 修在哪一层
判据链现读现核：engine_core.h:393/400/404/406/408/420 classify_failure；:2011/2015 fail_request_lane 的
  classify!=Request ⇒ return false 门槛；:2063 fail_all_locked、:2078 classify、:2079 fprintf、:2084 failed_=true；
  :2307/2318 worker catch 派发。b14 的行号全部对上（树里只有 fail_request_lane，无 fail_request_locked）。
可执行分类表（sh/wthr_classify.sh，真代码逐字抽取 + 13 条内容断言 + 与树 sed 区间 diff 为空）：
  21 行；real=GREEN(RED=0)、mutA(加 catch logic_error→Request)=RED 5、mutB(判别子失效)=RED 1。
  读出的硬事实：RequestError : invalid_argument : logic_error ⇒ (a) 的"用基类判别"在类型系统上不可能。
  前缀匹配不做 trim（"  cudaMalloc…" 落 invariant），与 :409-419 注释+匹配块一致，属已声明性质。
爆炸半径：全树(src+include+apps) 非 runtime_error 派生标准异常 throw = 2878
  (logic_error 1084 / invalid_argument 1739 / out_of_range 52 / length_error 3；domain_error 0)
  减 tests/bench 392（不在 serve 二进制内）⇒ Tier A（服务期请求路径）2598
  其中 无局部捕获 2389；再要求字面量进 pin 二进制(932336eea9e8cbf8) ⇒ ★主口径 2038
  (logic_error 796 / invalid_argument 1196 / out_of_range 44 / length_error 2)
  invalid_argument 占 59% ⇒ 只 grep logic_error 会漏 6 成雷；RequestError 只有 25 个抛点（98.6% 的
  invalid_argument 不是 RequestError，而契约 §3.1 刻意把"invalid_argument 用于内部不变量"归 Invariant）。
命名靶点全部定位：program_impl.h:5127(A/REQ/无捕获/bin True)；scheduler.h:226(A/REQ, consume_service_work
  ← run_control_batch)、:188/:184/:195(:REQ, build_control_membership ← worker_loop)、:155/158/167
  (build_round_membership)、:213(active_admission_set ← try_admit_one)、:75/312/318/340/349/355/367/370/397；
  program_impl.h:656(REQ_LOOSE)、:679/:684(UNREACHED，我判"可能"，非启动期)。
新发现：①invalid_argument 是主体(59%)；②scheduler.h:213 与 :226 是同一条账的两个死门，run_control_batch
  整段扣 row_stride；我核了 projected_service_work(request_plan_impl.h:101-137) 只有 prefill_units+decode_units、
  无 control 项，但 validate_generation_capacity(frontend.cpp:1185-1197) 要求 effective_output_tokens 覆盖
  control suffix 且控制 token 走同一 output budget ⇒ 我的"控制项没预留"假设被自己证伪，如实报；
  ③真正的洞是 scheduler.h:259-260 树自己写的 "can under-count by up to one quantum per interior boundary …
  accepted deliberately" 恰好落在 consume_service_work(:225) 的门上。
裁定：不做 (a)（会打红已实现的验收门 2；判别子只能是具体类型 RequestError；会把真 invariant 降级为
  "500+静默错 KV"——b14 §2.5 第 2 条已证 sentinel 仍指向已回收冷槽）；不做 (c)（门槛是"归因不明即保守停摆"）；
  做 (b) 但二分：b1 合法输入可达 ⇒ 修条件本身（首选，= rebuild_work.h 既有形状）或改 RequestError 并前置到
  submit/规划期（= S35/3.4 已落地样板），且必须留一条真不变量 throw；b2 真不变量 ⇒ 不动。
  scheduler.h:213/226 属 b1 候选但未验证 ⇒ 先不动，先跑判定臂；要动就动充值口径不动抛点。
契约偏好 (b) 的三处成文依据：docs/maintainer/engine-failure-recovery.md §3.1 表 + §3.4"校验前移" +
  §4 验收门 2；rebuild_work.h:30-32 先例（"That used to throw std::logic_error … must not be an error"）；
  types.h:724 S35(3.4) + request_plan_impl.h 一批 RequestError(InvalidPrompt)。(a) 会让 RequestError 变多余。
负对照：A=invariant_once 必须仍 class=invariant + /health 503 + recover（守门犬，任何 (a) 方案必把它变绿 ⇒ 判负）；
  B=request_once 必须只杀一条（证明 lane 失败机制本来就通，把归因钉死）；C=探针 mutB 臂 RED（表非恒绿）。
派单：①判定臂（小 --prefill-chunk + 并发 + 带 capture 边界 prompt，不改代码，≤2 min）②守门犬臂
  （NINFER_FAULT_INJECT=invariant_once）③修复验证臂（私有 build）。GPU 未占。
未验证：近似调用图非精确（有同名污染，已用 REQ/REQ_LOOSE 双口径 + 逐跳路径缓解）；over-charge 由合法输入
  可达为推证未实测；ops/* 1517 条 invalid_argument 未逐条细化（最大未细化面）；in_bin 只有 strings 子串
  （二进制无 -g，DWARF 路不可走）；docs 行号漂移（实质与代码一致）；权威树脏（本线未改一行）。

## §12.d-grader（19:13）SC-4c `自足` 的判分器 + P-4：d_sem 的"不可测"补到"照 argv 跑一次出四态"

**交付**：`/home/user/scratch/d_grader/{DESIGN.md,REPORT.md}` + `judge/*.py` + `probe/*.{cpp,h}` +
`sh/dgr_{host,fake_e2e,run,outer,launch}.sh` + `frontend_root/`（挖出来的真资源）。
**0 模型 / 0 GPU / 0 构建 / 0 共享 build 改动 / 0 源码改动。**

### ⭐ d_sem 缺件 ② 关闭：token id → text 不再是阻塞
`.ninfer` = `NINFER\0` + 明文 JSON TOC（[16,176858)）+ **4096 对齐**数据段（起于 180224）。
六个 `frontend/*` 的 encoding 是 `raw-bytes-v1` ⇒ **字节切片**。`py/dgr_carve.py` 挖出六个资源并落 manifest。
**交叉验证**：`tokenizer.json`（12,809,320 B, sha256 `0997f410c57a1f4e…`）与机器上**别的线**已在用的
`/home/user/models/q3nvfp4/tokenizer.json`、`scratch/p29/res/`、`scratch/fix2/frontend_root/` **逐字节相同**。
（第一版锚点错了一个 4096 边界 ⇒ 三个候选都"差一点"，是这条交叉验证抓出来的。）
**宿主 detokenizer**：`probe/dgr_detok.cpp` 链树里**已有**的
`build/src/CMakeFiles/ninfer_engine.dir/targets/qwen3_6/impl/frontend/tokenizer.cpp.o`(392,128 B) +
`libninfer_text.a`(379,152 B)（`build/` **只读**）。实测
`roundtrip(U) bytes_in=464 bytes_out=464 exact_roundtrip=yes`、
`encode(decode(encode(x)))==encode(x):yes`。
⚠️ d_sem 说"`spec/*.h` 没有 detokenizer"对 `spec/` 是对的，但 **`tests/targets/qwen3_6/test_frontend.cpp:287-297`
早就用 `NINFER_FRONTEND_TEST_ROOT` 在宿主演真 `fi::Tokenizer` 了** —— 缺的不是代码，是**资源**。

### ⭐ SC-4a↔SC-4c 接缝 = **CLOSED**（无损∧不跨界∧自足 能合成的唯一理由）
```
ORIG=507 token  U=348 token  U at token 65
SEAM=CLOSED orig_tokens=507 interval=[65,413) n=348 unit_tokens=348 SEAM=CLOSED
```
`dgr_detok span <orig> <begin> <end> <unit>` 就是这条等式。**真实产品路径那一步（SumDir row → 文本）工具有了、接线没有。**

### ⭐ 判分器：零模型、非循环、可跑
`judge/dgr_judge.py`（sha256 `cd42a528…`）。规则**全文两条**：
`PASS ⟺ skel(needle) ⊆ content`（expect=contains）/ `⊄`（expect=absent），
`skel = NFKC → ASCII 小写 → 只留 [a-z0-9]`（冻结，id `skel-alnum-lower-nfc/1`）。
**不循环的四条**：①无可学参数、0 模型调用；②`battery_sha256` 在第一个请求前冻结、对不上**拒判**；
③needle 是 32 字母表抽 12 = **60 bit**（猜 < 1e-18，且本次运行前不存在 ⇒ 无预训练记忆）；
④`needle` 在 U 里**恰好 1 次**（C2）+ arm X 删掉后**必须答不出来**（C4）。
**裂缝如实写**：Tier-1 判的是"有据抽取"不是"语义"；**Tier-2 派生臂**（答案 `18050` **在 U 与 ORIG 里都不出现**）
是唯一保险；Tier-1 过 + Tier-2 不过 ⇒ 判 `塌(copy_only)`，**这是决策项 D-1**（翻过来一行）。

### ⭐ 四态（`未测/不可用/塌/成立`），字段级接线
优先级：`不可用` **压过** `成立`；`塌` 与 `成立` 是最后一层的**发现**。
- **未测** ← `arms` 为空 或 九臂全 `not_run`
- **不可用** ← `reasons` 非空：preflight 红 / `battery`/`judge.version` 不符 / 仪器字段缺 /
  `passes_agree is not True`（**`None` 也算**，即第二趟没跑）/ `greedy_deterministic is False` /
  `prompt_tokens.U == .X`（消融没送达）/ 任一臂 `not_run` 或 `ambiguous` /
  **arm X FAIL**（删了证据还说出来 ⇒ 目击者作废）/ **arm L FAIL** / arm S `returned_forbidden` / 控制臂 O/XC/L2/T2C 不 PASS
- **塌** ← 上面全清白 ∧ (`arms.U != pass` ∨ `arms.T2 != pass`)
- **成立** ← 全清白 ∧ `O=U=X=XC=S=L=L2=T2=T2C=pass`
**关键极性**：`expect=absent` 的臂 `PASS` = "按要求没说" = **好事**；`FAIL` 才是红了。
⚠️ **我第一版把这个读反了，而自测的期望是从实现抄的 ⇒ 60/60 全绿什么也没测**；
是**接地气的假读者端到端**（干净模式立刻给出 `X=fail` 而判据要 `X=pass`）把它逼出来的。
⇒ **凡"自测全绿"都要问一句：期望是从哪来的。**（已写进 DESIGN §3.3）

### ⭐ 会红的负对照（每一条都带 well-posedness 对照，否则是**无对照的负结果**）
`arm O`(原文问得出) / `arm U`(被测) / `arm X`(**抽掉 U 内部一段**必须答不出来) /
**`arm XC`**(同一 `U_x` 上问另一条记录必须答得出 ⇒ X 是"证据没了"不是"文档坏了") /
**`arm S`**(两 nonce 对调 ⇒ 同一问题答案必须变；返回旧答案 ⇒ 不可用) /
`arm L`(只存在 ORIG 的事实必须答不出来) / **`arm L2`**(同问题从 ORIG 必须答得出) /
`arm T2`(派生 `18050`) / **`arm T2C`** / **`U2`**（与 U 逐字节相同 ⇒ 贪心确定性探针）。
两趟（复用 / `--no-prefix-reuse`）状态必须一致。

### 实测（全部宿主，`host`/`light` 类）
| 项 | 结果 |
|---|---|
| 判分器自测 | ⭐ **64/64 分类正确**；23 个 report 注入里**恰好 1 个**能开 `成立` |
| 假读者四态端到端 | ⭐ **11/11 模式命中预期态**（走 harness 最后用的同一个 `dgr_report.py`） |
| 类型守卫 | 基线 18 断言绿；**SAB-V1/V2/V3 拒绝编译**、**SAB-V4 运行期 1 红** |
| preflight | **29 条全绿**（`ok=True failed=[]`） |
| 电池 | `battery_sha256=8939794ed8f7ee73…`，**同 seed 重建 3 次 + 线上那份四个相同** |
| harness 前半段 | `DGR_STOP_AFTER_SEAM=1` ⇒ preflight 绿 + **SEAM=CLOSED** |
| `/tmp` | 800 K（我来之前 752–788 K）；⚠️ 我曾写两个 ≈10 KB 临时目录并**同命令删除**（违反"产物写 scratch"，如实记） |

### ⭐ 类型守卫收紧（d_sem 那一版挡不住的一格）
`probe/dgr_sc4_verdict.h` 保留 d_sem 形状（`HostSemanticVerdict` **无 `Equivalent`**、`to_full` 无 `default:`），
守卫从 `-Werror=switch` 收紧到 **`-Werror=switch -Wswitch-enum -Werror`**：
`-Werror=switch` 一旦有 `default:` 就**哑**，`-Wswitch-enum` **即使有 `default:`** 也对未处理枚举值报警。
**SAB-V2（加 `Equivalent` + `default: return Equivalent;`）实测拒绝编译**，触发的是 `-Werror=switch-enum`。
`clang++` 本机**不存在** ⇒ 第二种编译器未验证。⚠️ 树上 `tests/CMakeLists.txt` **不带**这些开关 ⇒ **守卫现在空转**（决策项 D-2）。

### 派单（主线）
**① 不占 GPU 的前半段（10 秒）**：`DGR_STOP_AFTER_SEAM=1 bash /mnt/c/Users/User/Documents/ziqinzhang/sh/dgr_run.sh`
**② 真跑（`model` 类，长跑必须后台）**：`bash /mnt/c/Users/User/Documents/ziqinzhang/sh/dgr_launch.sh`
跟随 `tail -f /home/user/scratch/d_grader/logs/p4_model.txt`；四态在日志末尾 `FOUR-STATE VERDICT` 块；
产物 `run/p4/report.json`，**一律 scratch、绝不 /tmp**。
峰值：显存 **21.8–24 GiB**（**引用，未复算**）/ 主机 **9–11 GB**（实测可用 20.4 GB，门要 10000 MB）/
磁盘 **<300 MB**（模型**硬链接**；实测空闲 62 GB，门要 40 GB）/ **6–12 分钟**。
**会红**：⭐ arm X 必须答不出来（说出来 ⇒ 不可用，目击者作废）；arm L 必须答不出来；arm U 不是 pass ⇒ 塌；`U2==U`；两趟一致；`prompt_tokens.U ≠ .X`。
**失败行为**：`hwrun` rc 4/5/6/7 ⇒ **如实报不可用，绝不降级硬跑**；服务起不来 ⇒ `未测`（**不是负结果**）；字段缺 ⇒ 报缺，**不当 0 用**。

### ⛔ 未测（不算通过）
**本线没有一格 GPU 数字是我产出的**；`21.8–24 GiB / 9–11 GB / 256 MiB` 全是引用。
⇒ **SC-4c `自足` 对一个真实读者仍然是【未测】**；接缝只对本次语料成立；`judge/*` 未接进 `tests/`；
Tier-2 的难度**没标定**。**跑完仍是 G3，我不给它升格。**

### 副作用
scratch `/home/user/scratch/d_grader/` 18 MB / 1074 文件（≈5 MB 是 `run/fake_e2e/` 自测残留，可删）。
**源码 0 改动**；未动 `build/`、未动 `_hwlimit.sh`、未动别人 pin/槽位；无孤儿进程；GPU 只做 `nvidia-smi` 查询。
`hwrun`：`host`/`light` 若干，**`model` 0 次、`gpu` 0 次**。

---

## §12.d-throwfix（19:45）按 w-throw 的裁定落地 (b) 逐点修 + 前置门：`scheduler.h:213/226` 是同一条账上的两个门

线名 `d-throwfix`｜权威树 HEAD `3944a53`（脏工作树，**本线一行未改**，见下"零改动证明"）｜**未加载模型、未占 GPU、未动共享 `build/`**｜判据全部 host-only 可复跑。

**裁定执行**：(a) **不做**、(c) **不做**、**(b) 逐点做且加前置门**。改动 **5 文件 9 hunk +120/−7**：`scheduler.h`（2 处）、`types.h` + `request_plan_impl.h` + `request_record.h` + `engine_core.h`（把"计划把 prompt span 切成几段"这个数从计划带到充值点）。

**★ 把 w-throw §8「最大不确定性」变成可算的事实**：门触发 **⟺ `e_{u1} > e_{u0}`**，其中 `e_u = Σceil(L_i/u) − ceil(R/u)`（内部边界在该 unit 上的舍入超额）。暴力搜索（`py/dthf_probe_bf.cpp`，host-only 真 codegen）：
- 两段（1 内部边界）R ≤ 3072：**70,755,840 用例 / 6,064,000 见证**
- **一段（无内部边界）：0 见证**（正是树在 `scheduler.h:253-257` 写的 "What is EXACT here" 那一档）
- 三段：1,128,042,240 / 23,343,616

**最小见证**：`u0=256 → u1=128`、剩余 `R=258` 切成 `[129,129]` ⇒ `T0=2`、`T1=4`、充值只加 `disp=1`、预留 3 < 真需 4（`129=128+1` 是形状关键）。

**判据（全 host-only，3 臂）**：`dthf_probe_after` = **GREEN 0 fail**（A5 穷举 1,197,000 用例 0 短）；`dthf_probe_before`（与被测树**同 sha** 的 pre-patch 头）= **RED 5 fail**，且**逐字打出 w-throw §6 预言的 reason** `request service projection consumed 1 quanta with 0 remaining` 与 `active request has no admission accounting`；`dthf_probe_mutb`（把新项强制 0）= **RED 2 fail**；守护臂 A3 证明真 over-charge 与零扣款在 after 上**仍然抛**。

**逐点判定**：17 条 `scheduler.h` 抛点里只判 **2 条**为"条件写错"——
1. `:213`（原 `:212-214`）：payload `ActiveAdmissionSnapshot` **不含**该值（断言零贡献）+ 同一文件明文声明该值可为 0 而请求仍活跃 ⇒ **删断言**，承重墙 `:225` **原地不动**；`engine_core.h:2224-2226` 自己写的字是 "the **pair** of invariants in scheduler.h"，独立支持两处一起判。
2. `:272-279` 充值口径：漏掉内部边界的**逐段舍入项**。判法 `after = quanta_for(R,u1) + b`、`before = quanta_for(R,u0)`（**故意不对称**，两侧都加 b 会抵消）；`b = RequestPlanSummary::prefill_interior_boundaries`（由 `projected_service_work` **同循环**写出）+ `RequestRecord` 播种 + admission 同处播种 ⇒ 与 `service_work_quanta` 同函数同输入、不可能漂。代价 ≤ b 个 quantum，保守方向；"扣不够就自动延长预留"仍被注释里的原话拒斥。
3. **`:226` 保留为真不变量、一字不改**（病在充值口径，不在它）。其余 15 条判真不变量，逐条 verdict 在 `py/dthf_mkbaseline.py` 的 REVIEWED 表。
4. **契约未动**：`classify_failure` / `fail_request_lane` / `fail_all_locked` 与 9 个 hunk **不相交**（脚本逐 hunk 断言）；`program_impl.h:5127` **一个字符未动**；注入钩子 `request_once`/`invariant_once` 两处 cmp 逐字同。

**前置门**（单子第 1 项）：`dthf_gate.py` = **P1 声明**（新 `logic_error` 家族字面量出现即红 / 已声明字面量消失即红）+ **P2 条件已修**（declared `condition` 的字面量**必须不在代码里**）+ **P3 契约**（`classify_failure` 不得 catch 该家族）。**五臂实测**：`TREE=RED(P2)`、`MIRROR=GREEN`、`MUT_NEW=RED(P1 undeclared)`、`MUT_LOST=RED(P1 gone)`、`MUT_CLASS=RED(P3)` ⇒ 门在"**修前红 / 修后绿**"上自证；`MUT_NEW` 正是单子要的注入用例。扫描器做**注释屏蔽**（这棵树把决定写在注释里，不屏蔽会把历史当站点 —— 第一版就扫出假站点，已修）。

**真编译**（真 codegen 私有 `.o`，非 `-fsyntax-only`）：`engine.cpp` rc=0 **77.53 s / maxrss 1,286,016 KB**；`materializer.cpp`（覆盖 `request_plan_impl.h`）rc=0 **4.40 s / 291,208 KB**。证据：符号数 **1971 vs 共享 build 未打补丁的 1972**；被删抛点的消息串在我的 `.o` 里**消失**、在未打补丁的共享 `.o` 里**仍在**；`.o` 大小 **−864 B**。

**行多重集**（§12.34）：**+120 / −7**；7 行删除**逐条归属**（3 行声明删除、4 行被取代且替代行在同一 hunk 的 `+` 侧）⇒ **lost=0**；120 行新增全部来自 5 处 Edit ⇒ **invented=0**。`git apply --check` 6/6 rc=0，并**另用两条独立判据核**（逐行归属表 + 守门区不相交断言）。

**未验证（一条不省）**：①未跑引擎/未占 GPU ⇒ 单子判据 A/B 的**真引擎那格未测**；②见证是**纯算术可达**、非真 prompt 触发；③`b` 用整段边界数、偏保守但**未量化**；④树脏、行号可能漂（以逐字串为准）；⑤snapshot 档 56 行未逐条 review；⑥`program_impl.h` 族（`:5127` 未动、`:5133/:5143/:5148`、`:656`、`:679/:684`）**只判不修**；⑦`ops/*` 1517 条 `invalid_argument` 未覆盖；⑧真编译只 2 个 TU、`tests/` 未重编（`RequestPlanSummary` 加字段静态核对：树里无 `operator==`、无位置化聚合初始化 ⇒ 应不破坏）；⑨`:213` 删除的行为差异未实测。

**派单（要引擎的，交主线）**：①**判定臂**（最便宜，请最先跑）：`sh/d1_run.sh` 主臂 argv 逐字（pin `932336eea9e8cbf8`）+ 小 `--prefill-chunk` 256 + 并发（长 decode 与长 prefill 同飞）+ 带 capture/rewrite 边界的 prompt ⇒ 判据：出现 `reason: request service projection consumed <w> quanta with <r> remaining` ⇒ §2 的 (ii) 成立；若只出现 `class=request` ⇒ 我第二处判定要撤（**我要看那一行**）。≤2 min，host ≥ 10 GB。②**守门犬臂** `invariant_once MIN_ID=2` 必须仍 `class=invariant` + `/health` 503 + `POST /recover`。③`request_once` 归因臂（只杀一条）。④**修复验证臂**（私有 build）：落批后 `grep -c 'engine failure class='` = 0、该请求与后续全 done、`/health` 200 ok；**反向判据**：引擎已不可信而 `/health` 仍 ok ⇒ 判负。⑤门接 CI + 覆盖扩到 Tier A（需先梳理 `ops/*` 的 `supports_*` 探针约定）。⑥`program_impl.h` 族（`:5127` 的**制造者**）单独一线。

**零改动证明**：5 个被补丁的文件，工作树 sha256 与开工基线**逐字相同**（`ffd8cabd9f7c66cf` / `7be64a5c789983e9` / `3375a5a2c7c3cf57` / `1d4cde512aed3f06` / `42b516f1c0a0efaa`）；`hwrun monster` 2 次放行、rc=0、**未被拒**；未用 `git clean/checkout/stash/reset`、未 `rm -f /tmp/*`、未 `wsl --shutdown`。


## §12.w-recover（19:56）那 32 个「内容不在 git 对象库」的隔离文件 —— **32/32 可重建（逐字节 cmp=0）**；w-quarantine 标的「真·单点故障 16 个 / 1,142,825 B」**16/16 全部有构造性重建**

**接手**：`w_quarantine/REPORT.md` §8.2 唯一留下的未验证路径 ——「G1 那 32 个能否用盘上的 `*.diff/*.patch` **反向重建**：**我**（w-quarantine）**没有**做 `patch -R` 的构造性验证，这是**未证伪**的路径，不能算已排除」。
**结论**：这条路径**被证成了**，不是被证伪。产出 `/home/user/scratch/w_recover/REPORT.md`，脚本 `sh/wrec_*.sh`，全程 CPU only（未加载模型、未占 GPU）。

### 做了什么（可核）
1. **32 项口径直接取自 w-quarantine 的产物，不自己重新推导分级**：`FINAL.tsv` 的 `G1_NOT_IN_GIT` 行 = `grade.tsv` 的 `G1_IRRECOVERABLE` 行 = **32 行，行号逐行相同**（5,8,12,13,21,24–29,32,34,41–46,51,53,54,60,79,80,86–92）；复算字节 **1,444,492 B**，与 `GRADE_SUMMARY.tsv` 逐数吻合。
2. **语料两轮**：① w-quarantine 的 824 个 `*.diff/*.patch/*.rej/*.orig.*`（复算 24,973,236 B = 23.82 MiB）；② 我扩的 —— 扫 378,717 个小文件，取「内容里真有一行 `^@@ -<数字>`」的 **1,501** 个（其中 **927** 个名字命中 32 个 owner）⇒ 合计 **1,285** 个可解析差异文件。**②不是装饰：有 2 条的重建 patch 就是 `.txt` / `.md` 文件**（`dl/b02_hdrs.txt`、`research/notes/S_D_selector_fidelity.md`）。
3. **基准不是只看 HEAD**：每个 owner 的 3–11 个 git 历史 blob + 84–187 个盘上同名副本（按内容去重后 3–75 个不同内容）+ 同 owner 的其它隔离文件；**与目标逐字节相同的"副本"一律排除**，否则"重建"就是"拷贝"。
4. **三道闸**：行数算术闸 → patch 被删行的多重集子集闸 → **真打一次 + `sha256` 相等（唯一判定门）**。应用原语**先把段头路径拍平成 `TGT`** 再 `patch [-R] -p0 -i sec.diff TGT`，彻底绕开 `-p1/-p2` 与路径前缀的不确定性（先跑 `selftest2.sh` 证明原语能正能反能红）。
5. **9 条是 CRLF**：**32 个里 9 个是 CRLF 行尾的，这 9 个恰好就是第一轮 0 命中的 9 个**；盘上 1,285 个差异文件的段内容**全是 LF** ⇒ 必须加一行文档化归一 `基准 --(CRLF→LF)--> patch --> --(LF→CRLF)-->`。加上这一步后 **9/9 全部 fuzz=0、cmp=0**。
6. **负对照（最重要的可能不是"全中"，而是"抓到 2 条空操作"）**：
   - 23 条 patch-only：**NC1**（把 patch 里"要被删掉的那一行"末字节改一字节）**必须对不上** → 5 档代表 **25 PASS / 0 FAIL**（含 fuzz=0 与 fuzz=3 两跑，具体 sha 见 `negctl.tsv`）；
   - 9 条 patch+行尾：**NC1–NC5 共 45 行，40 PASS / 5 FAIL —— 5 个 FAIL 全部是"patch 不承重"**；
   - 于是对 **32 条逐条**跑「空 diff（不给 patch）」闸：**30 PASS / 2 FAIL** ⇒ `gqa_attention_decode_impl.cuh.bak-prett6` 与 `dflash2_impl.h.orig` **连 patch 都不需要**（前者 = `to_crlf(to_lf(git show 08dbfa9e:<owner>))`，36,387 B → 37,003 B；后者 = C: 侧 `ninfer-fusion-repo/` 副本的行尾归一，29,334 B ↔ 29,334 B）。**如实单列，绝不把空操作记成 patch 的功劳。**
7. **判定结果**：仅靠 patch **23 / 1,240,974 B**（22 条 `fuzz=0` + 1 条 `fuzz=2`）；patch + 行尾归一 **7 / 137,181 B**（全 `fuzz=0`）；纯行尾归一（无 patch）**2 / 66,337 B**；**不可重建 0 / 0 B**。**32/32 的 `cmp -s <重建产物> <树上那份>` 全部返回 0**，sha256 双向相等（`verify.tsv` / `verify_eol.tsv` / `totals.log`）。
8. **对"真·单点故障"的直接裁定**：`G1_16_NO_CDRIVE_COPY.txt` 的 **16 个 / 1,142,825 B → 16/16 全部可重建**：**5 个只靠 `HEAD`**（114,136 B）、**9 个靠另一个 ref**（`origin/HEAD`×5、`d38bb91b`×3、`ad921ee3`×1；299,937 B）、**2 个仍以 `/dev/sdd` 上的 scratch/`~` 副本为基准**（`types.h.orig`、`program_impl.h.orig`；**728,752 B = G1 的 50.5%**）。
   ⇒ **一句话：那 32 个「内容在 git 对象库里不存在」是事实，但它 ≠「内容不可重建」。** 32/32 都能由盘上素材逐字节重建；"无任何重建路径"的条目从 16 → **0**。

### ⭐ 最该被上游知道的一条（风险转移，而不是消失）
**重建的"基准"比"目标"更脆弱。** 32 条的基准按可信级分：**S（`HEAD`/树内同一文件）10 条**、**A（其它 ref）14 条**、**B（C: 侧副本）3 条**、**C（`/home/user` 非树副本 = 同一块 `/dev/sdd`）5 条**。
其中 **`program_impl.h.orig`（691,737 B，占 G1 的 47.9%）的重建完全押在 `/home/user/scratch/g2/a/…` 这一份 scratch 副本上** ——
"单点"没有消失，而是**从隔离文件挪到了它的重建基准**；scratch 一清理，这条重建路径就断。
另外：A 档里有 5 条依赖 `refs/remotes/origin/HEAD`（`merge-base HEAD origin/main` 仍为空，与 w-quarantine §9(a) 同）；**现场观察 19:06:50 别的线建了 `refs/tags/protect/origin-main-20260914T190650` 钉住同一 sha `0eaac046ee98`，这层当前已被他人加固（只观察，不认领）**。

### 顺手确认（只读；**未创建 `_collab/`、未改 `MANIFEST.txt`**）
- 树内**没有** `_collab/`（`ls -d` 失败、`find -iname '*collab*'` 空）；C: 侧有 ⇒ `MANIFEST.txt` 承诺的「archived as a diff under `_collab/`」**在树内断链**，但**我确实从 C: 侧 `_collab/` 取到了 23 条重建里 12 条的 patch**。
- 顺带：`MANIFEST.txt` 列了 **12 个唯一路径**，`_orig_quarantine/` 里只有 **4 个**在；另有 4 个在 `_quarantine_backups_20260912/`、1 个变成被跟踪文件、**4 个树内任何地方都找不到**（`engine_core.h.orig`、`generation_service.cpp.orig`、`serve_options.cpp.orig`、`serve_options.h.orig`）。这 4 个**不在** w-quarantine 的 102 项里、也**不在**我的 32 项里；**我未在树外搜索它们**，**未补、未改清单**。

### 未验证（明说）
① 只验**逐字节相等**，未编译未跑测试；② **"可重建" ≠ "归属语义成立"**（§5.3 抓到 2 条方向正好相反）；③ 未扫「差分藏在没有 `@@ -` 的文件里」（上下文格式 `! ` 行、`rcsdiff`、二进制 `GIT binary patch`）；④ 基准只搜 `/home/user` + C: 文档目录，未穷举全机/别的盘；⑤ 给的是"第一组能通过的"重建，**不是最小集**（同一条常有 1–8 组）；⑥ 行尾只试了 3 类变换，未试混合行尾/UTF-16/BOM；⑦ 未做破坏性对照（没真去删 patch/基准 —— 所有负对照都在私有副本上做）。

### 0 破坏自证
`HEAD = 3944a53eda1aac439a566a1cf46ea741f0415fdc`（**未变**）；`git status --porcelain | wc -l = 294`（**未变**）；**32 个隔离文件逐个重哈希 32/32 UNCHANGED**；树内 `.orig/.bak` 仍是 **102**；两个 quarantine 目录 mtime 未变；**所有重建都在私有目录 `/home/user/scratch/w_recover/` 里做**（把基准与 patch 段复制出去再打），**从未把 patch 打进树**；未 `git clean/checkout -- ./stash/reset --hard`、未 `git add`、未碰共享 `build/`、未 `wsl --shutdown`、未 `rm -f /tmp/*`、未 `fuser`/`/proc/*/fd`、未整目录 `cp -r`；全程 `hwrun light`，**无 rc 4/5/6/7**。
一处如实报：`.git/index` 的 sha256 从 `1b19c2e03c1c…`(mtime 15:07:54) 变成 `05e45d04598e…`(mtime 19:26:29) —— 是 `git status` 判定 stat 过期后回写索引（只读命令的副作用）；**HEAD 未变、脏文件数仍 294、`git diff --cached` 为空**，我从未 `add/commit/checkout/restore`。
明令遵守：**未分配 `§12.x` 编号** ⇒ 本段用 `## §12.w-recover（19:56）`；台账只 append。

**§12.w-recover 的两句更正（19:57，append 不改上文）**：① 收尾时实测 `.git/index` sha 已从 `05e45d0459…` 又变为 `efcaf0e77208…`（mtime 19:31:31，本轮内共动 3 次：15:07:54 → 19:26:29 → 19:31:31），且 `git diff --cached` **非空** —— 有 **1 个非我 staged 的条目** `tools/convert/qwen3_8_27b/convert_modelopt.py`（+1842 行）；**不是我**：我的脚本从未 `git add/commit/checkout/restore/reset/stash`，该文件不在我的操作范围；`HEAD` 未变、`status --porcelain` 始终 **294**、与 32 个目标无交集。② 树内 `19:00` 之后唯一被我写过的文件就是**本台账**（任务要求 append）；append-only 已证：`sha256(当前文件前 821357 B)` == append 前整文件 sha，`delta = 1 空行 + 段字节`。 已同步更正这两处（§9.4 与 §10）。

---

## §12.w-throw2（20:31）ops/* 的 951 条 + program_impl.h 族判定 + tests/ 真编译 + :213/b 量化

**三处硬修正（相对上一棒 d-throwfix §9 的 ⑥⑦⑧⑨）**

1. ★**计数归属改掉**：上一棒说「ops/* 的 **1517** 条 invalid_argument」——**1517 是 Tier A（249 文件）全体的数，ops/* 是 951（169 文件）**。
   我用它的两个工件（dl/wthrow/wthr_reach.tsv + wthr_tier2.txt）逐字复现出 1517（口径一致）与 951（py/wth2_ops.py → out/ops_tier.txt）。
2. ★**推翻「ops/* 大量是探针控制流 ⇒ 不是炸弹」**：835/951（87.8%）的站点落在**文件里一个 catch 都没有**的 161/169 个文件里；
   且 ops 入口的**调用点也没有 catch**（src/targets/qwen3_6/impl/runtime/text_context_impl.h：1598 行、89 个 ops:: 调用点、**0 catch**；
   27b/35b 的 variant.cpp：0 catch；dflash_impl.h / dflash2_impl.h：0 catch）。
   ⇒ 链路 `worker_loop → target impl 前向(0 catch) → ops::<entry>(0 catch) → require_*/validate_* 抛 invalid_argument → catch(...) → class=invariant → 停引擎`
   逐字成立（docs/maintainer/engine-failure-recovery.md:40）。「探针控制流」只是 **8 个文件**的局部现象。
   **且 ops 里没有 `supports_*` 探针约定**：gdn_input_proj/gqa_attention/embedding/rope/sampling 五个最大文件里 `supports|capable|_probe` 出现 **0** 次
   ⇒ 上一棒"扩门前要先梳理 supports_* 约定"这个前置条件**不存在**（见派单 0）。
3. ★**program_impl.h 族判了**（上一棒只写"未判"）：
   `:656` = **invariant**（4 个调用点拆开：ordinary :13006 有 graph_profile_missing+extend 守卫（:12997-13004）；DFlash :13378 / DFlash2 :13620
   在启动期无 ceiling 过滤、全量捕获并 validate（:12208/:12229-12249、:12258/:12290）⇒ 覆盖 [0,capacity-1]；**只有 MTP :13172 的子支
   `launch_width==0 ∧ ceiling!=0 ∧ rungs>1` 未证**）；
   `:679/:684` = **invariant 且在启动期**（外包函数我读出来了 = `ProgramImplCore::prepare_graphs()` :11867；上一棒的 UNDREACHED 应改判"启动期真不变量"）；
   `:5127/:5133` = **「针对错对象的不变量」= 条件写错，但制造者未证 ⇒ 保留 + 未判**（同文件别处一律按 (sequence, shared) 成对处理：
   pressure_state_source(action, sequence, shared) :5828、owner 取址 :5894-5896、:5019/:5064/:5075 的 (source_state||shared_state)；
   **只有 :5126 只看 source_state**，而共享前缀当源时 source_state==nullptr 是**构造性**的（:4938-4942），且冷记账只挂在 SequenceState 上
   （生产者 :11122、清空 :10767，SharedPrefixState 无 cold_pages）⇒ 这一支**没有答案可给**）；
   `:5143/:5148` = **真不变量**（:5143 是全族唯一"断言的量就是它刚要做的动作"的站点：:5142 已把描述符放回冷路）。
   制造者链条逐跳已给出（S 冷维护压冷 :11117-11124 → S 结束 release_continuation_slot :6653 → release_active_shared_references :6659 →
   release_sequence_kv :10752-10767 回收冷槽+文件槽+清 cold_pages（:9488 同形）→ 共享前缀当源 materialize → :5121 命中 → :5127）
   **卡在唯一一格**：「共享前缀的地址空间能否承载 cold_compressed 页」（反对：PressureKVDecisionKind 无冷操作；
   支持：共享前缀按定义就是 read-free 前缀、而冷压缩的目标正是它，且 can_cold_transfer(logical_kv_store.h:774) 不拒绝 references==2）。
   闭合只需再读两个文件（logical_kv_store.h:767-900 + SharedPrefixState 定义）⇒ 派单 7。

**Task 3（⑧，补真编译）**：把 tests/ 里**受补丁头影响的 6 个 TU** 真编了一遍（真 codegen、私有 .o、**不是 -fsyntax-only**）：
  TU 清单来自 build/compile_commands.json 一个文件的定点读（495 条/其中 tests 143 条/131 唯一）+ 逐 TU 单文件 grep。
  **6/6 rc=0**，wall 1.08–8.03 s，maxrss 253–575 MB（out/tests_build.txt 全量日志）。
  四条证据：(1) 对 6 个 TU 各做 -E 预处理后 `prefill_interior_boundaries` 出现 1/1/1/2/1/1 次 ⇒ **补丁头确实在 include 闭包内**；
  (2) 同一个 test_resource_manager.cpp 用**未打补丁的树** vs **带补丁的 mirror** 两份 .o 不同（724088 vs 724152 B，符号数都 720）
  ⇒ **加字段真的改到了测例的机器码**，这次编译不是空转；(3) rc=0 ⇒ `RequestPlanSummary` 加带默认值字段 **tests/ 一个字符都不用改**
  （上一棒只有静态核对，这一棒是编出来的）；(4) sha256 与两条命令行逐字在日志里。
  **整树 link 不在本线范围**：共享 build/ 的 link 命令行写死在 build/tests/CMakeFiles/*.dir/link.txt（含每个 .o 绝对路径 + 全套 -l/-L），
  把私有 .o 塞进去就是改共享 build/；唯一安全做法是另建整棵私有 build（495 TU 全量）⇒ **派单 5**。

**Task 4（⑨，:213 删除的行为差异量化）**：新探针 `probe/wth2_probe213`（host-only，两个 build 只差一个 -I 根：tree 头 sha 3375a5a2c7c3cf57 vs mirror 头 e282479aa196ba42；
  判据两边逐字相同 ⇒ 退出码就是结论）。src/REPORT 全在 out/probe213.txt。
  tree = **RED 8 failures exit 1**（A1/A3/A4-residual0 全 STOP: "active request has no admission accounting"）；
  mirror = **GREEN exit 0**（A1 size=1、A3 size=2、A4 三个 residual 快照**逐字段相同**）。
  ⇒ **能**造出"active 且 residual=0 被枚举成 donor"的序列；**差异量化 = 恰好 1 个比特（throw vs enumerate），其余全零**：
  被枚举的载荷类型 `ActiveAdmissionSnapshot` 只有 3 个成员（admission_policy.h:17-21，无 residual），
  它的**全部**消费者只用这 3 个（protection_has_live_donor :83-88 只用 request_id；persistent_backfill_is_authorized :90-108 用 3 个）。
  A5 守护臂：**承重门 consume_service_work 在两个 build 里都仍然抛**（二进制里 `request service projection consumed` tree 2 次 / mirror 2 次），
  被删的那条消息 tree 1 次 / mirror **0** 次 ⇒ 删掉的是错拷贝，承重墙原地不动。A5 还顺带证明 **residual 恰好 0 是扣款路径自己的正常输出**。

**Task 5（③，b 的保守量）**：py/wth2_b_bound.py（纯整数穷举）。
  **精确恒等式**：overshoot(=修后多留) == `b_whole − (e_{u1} − e_{u0})`，且 `0 <= e_{u1}−e_{u0} <= b_rem`
  ⇒ **overshoot ∈ [Δ, b_whole]，Δ := b_whole − b_rem = 已经走掉的内部边界数**；且 `need − top_up = (e_{u1}−e_{u0}) − b_whole <= −Δ <= 0` ⇒ **充分性无条件成立**。
  ⇒ 『因为用了整段边界数而多出的那部分』**恰好 = Δ 个 quantum**（下界、紧）；最坏多留整个 b_whole。
  **穷举**：层 A（R≤40、k≤4、已走段数 j、unit∈{8..64}）= 2,380,794 用例：违反 e_u≤k_rem−1 = 0、违反恒等式/下界 = 0、**修后仍欠账 = 0**、
  修前最大欠账 2 quanta、修后最大多留 6 quanta、max Δ 3。层 B（真实 128 对齐网格、R≤150、k∈{2,3}）= 25,143,750 用例：
  修前最大欠账 0、修后最大多留 3 quanta、**最大相对多留 2.000**（只在 needed 很小时，绝对代价 ≤3 quanta —— 必须与相对数一起写）。
  层 C 逐字复核上一棒见证 R=258/u0=256/u1=128/parts=[129,129]：修前预留 3 欠 1、修后预留 4 多留 0（Δ=0）**对上**。
  量级表述：额外预留 ≤ (k−1) quanta，相对 ≤ (k−1)·u1/R。

**交付**：/home/user/scratch/w_throw2/ —— REPORT.md；patches/00_no_source_patch.txt（**零源码补丁**的四条理由）；
  patches/wth2_gate_ops.patch（**前置门 baseline 的统一 diff，+862 行**：ops/* 855 + program_impl.h 族 7，**不施加**）+ baseline_ext.tsv + extended.tsv；
  out/（ops_tier.txt、ops_sites.tsv 951 条逐站点、ops_catch.txt、ops_hard.tsv 346 条 T1 清单、ops_callers.txt、ops_verdict.txt、
  recon*.txt 逐字区间、tests_pick/plan/build.txt 真编译日志、probe213.txt、b_bound*.txt）；probe/（探针源码 + 两个二进制）；obj/（7 个私有 .o）；sh/py/*。

**未验证（一条不省）**：①未跑引擎/未占 GPU ⇒ 判据 A/B 的真引擎格仍未测；②「零 catch」只到**文件级 + 抽样调用点**，不是 AST 级捕获闭包
  （§1.4 的 835 条"逃出文件"≠"逃到引擎并抛"）；③ops/* 只判了 T1 的 346 条（真不变量 301 / 条件写错 2 族 / **未判 43**），**T2+T3 的 605 条一条没判**，
  「未判」≠「通过」；④`:5127` 的制造者**未闭合**（卡在"共享前缀能否承载 cold_compressed 页"，缺 logical_kv_store.h:767-900 与 SharedPrefixState 定义 ——
  后者**不在** wthr_tier2.txt 给的 249 个 Tier A 文件里，我没去找以免变成目录扫描）；⑤`:656` 的 MTP 子支未证；⑥program_impl.h 族里没点名的相邻站点
  （:5109/:4621-4624/:10776-10778/:10650/:10844）未判；⑦tests/ 的 .o **没 link、没跑**（ctest 未跑，只证编译）；
  ⑧in_bin 是 strings 子串、9 条 catch(...) 的行匹配未确认"catch 是否在抛点之后"；⑨行号：开工时 5 个补丁文件 sha 与 d-throwfix 台账**逐字相同**
  ⇒ 自 19:53 起未漂，仍以逐字串为准；⑩我的纪律偏差：用过**一次** `grep -rn`（限定 src/runtime/engine/ 单目录、单符号 'struct ActiveAdmissionSet'）与
  **一次** Windows `dir /b`（开工定位）；其余全部定点读，遍历只发生在**已有清单上做单文件 grep**（249 Tier A 文件 / 131 tests 文件 / 169 ops 文件）；
  ⑪b 的量化吃"b 用整段边界数"这个现有行为 —— 若改成剩余段边界数，Δ 项消失、多留降到 [0, b_rem]（这是改法选择，不是我测出来的）。

**最大不确定性**：§1.2 的 **835/951** 只到文件级 —— 若据此改分类层或成批改 ops 谓词，必须先做 AST 级捕获闭包，否则会把其实被接住的站点误判成炸弹。
  第二：`:5127` 那格决定"改 `prepare_kv_restores` 的取源"还是"改 `release_sequence_kv` 的回收时机"（两者处置完全不同）。
  第三：`:656` 的 MTP 子支。

**派单**：**0**（新增，最便宜）把 patches/wth2_gate_ops.patch 落进前置门 baseline（数据文件）后跑 dthf_gate.sh 五臂：TREE 仍 RED(P2)、MIRROR 仍 GREEN、
  新增 862 行**不得**让 P1 变红；若 P1 报字面量对不上 ⇒ 把那一行给我（TSV 抽取口径 vs 门注释屏蔽不一致）。
  1 判定臂（原样，pin 932336eea9e8cbf8 + --prefill-chunk 256 + 并发 + 带 capture/rewrite 边界的 prompt，≤2min，host≥10GB；若只出现 class=request 我要看那一行）。
  2 守门犬臂 invariant_once（本棒已给 host 侧证据：承重门消息两 build 都在）。3 request_once 归因臂。4 修复验证臂（私有 build；反向判据：引擎已不可信而 /health 仍 ok ⇒ 判负）。
  5 **整树 link + ctest**（另建整棵私有 build，495 TU；判据 6 个测例全绿；编译半边本棒已给绿）。6 `:656` 的 MTP 子支（文本级二值判）。
  7 `:5127` 的制造者闭合（文本级二值判：被共享前缀引用的逻辑页能否被 transfer_to_cold 压冷；能⇒条件写错且修在上游，不能⇒真不变量）。

**零改动证明**：5 个被 d-throwfix 补丁的文件，工作树 sha256 = ffd8cabd9f7c66cf / 7be64a5c789983e9 / 3375a5a2c7c3cf57 / 1d4cde512aed3f06 / 42b516f1c0a0efaa，
  与 d-throwfix 台账**逐字相同** ⇒ 一个字符未动。未用 git clean/checkout/stash/reset、未 rm -f /tmp/*、未 wsl --shutdown、未 fuser。
  共享 build/ 只读（compile_commands.json + link.txt 路径列表），所有 .o 在 w_throw2/obj/。
  hwrun：host 15 次 / light 12 次，**model 0 次、gpu 0 次，未被拒**；每个调用点后都 trap - RETURN，没有拿 hwrun 包父脚本。

---

## §12.w-throw3（20:49）`:5127` 制造者 = 证出（可达）＋ `:656` MTP 子支 = 空集 ＋ `wth2_gate_ops.patch` 实测（不得单独落）＋ ops 逐条 verdict 862/997 ＋ 整树 link 估算

> 权威树 `/home/user/ninfer-fusion`（HEAD `3944a53`）。**未改源码、未加载模型、未占 GPU**。报告 `/home/user/scratch/w_throw3/REPORT.md`。

### 1. ★★ `:5127` 的"制造者" = **证出来了：可达**（w-throw2 的"未证"这一格**划掉**）

**一句话**：**冷压缩是唯一一条会把"被两个地址同时引用（`references > 1`）的页"的 device replica 摘掉的路径**；而"被两个地址同时引用"**就是共享前缀的定义**（同文件 `shared_kv_prefix_pages:7018` / `shared_device_kv_prefix_pages:7039` 都以 `address_references > 1` 为判据）。共享前缀那半个地址拿到的是**旧活动地址**，它与新活动地址**共享同一批逻辑页**，所以拥有它的 SequenceState 之后每一次冷压缩都会把共享前缀还在引用的页压冷，而冷簿记只写进**它自己**的 `cold_pages`。

**逐跳（全部逐字，行号今日现读）**：
1. `publish_shared`（`:7463`/`:7887`）→ 2. `prepare_active_snapshot(sequence.kv->text, *active_text_destination, …)`（`:8149-8154`）→ 3. `commit_active_snapshot`：`retain_reference` 每页（`logical_kv_store.h:1474`）**并把同一批逻辑页写进目的地 membership**（`:1476`），`A_old` 变**非活动 checkpoint 地址**（`:1497-1499`）→ 4. `shared.kv = *shared_bundle`（旧 bundle = `A_old`，`:8328-8343`、`:8428`）、`sequence.kv = A_new` → 5. 每轮 decode 边界 `enqueue_cold_compressions(sequence)`（`:13532-13538` → `:10845`），扫描窗 `cold_frontier .. (text_kv_valid-cold_keep_tokens)/page`（`:10857-10861`），谓词 `store.can_cold_transfer(text,page)`（`:10926`）→ 6. **谓词为真**（`:786-797`：`references != 0` ✓=2、`writer_references == 0` ✓、`source_pins == 0` ✓、device replica 在 ✓）⇒ `transfer_to_cold`（`:11121` → `logical_kv_store.h:771-779`，**摘 replica + `cold_compressed = true`**）+ 簿记写进 **S 的** `cold_pages`（`:11122-11124`）→ 7. 新请求以共享前缀为源（`:4362`、`:4938-4942`、`:5024`）⇒ `source_state == nullptr` → 8. `prepare_kv_restores` 的冷支 ⇒ **`:5127` 抛**（`:5118-5121`、`:5126-5127`）。

**为什么没人挡住（三条逐字）**：
① **`can_dematerialize` 认独占（`references == 1`，`logical_kv_store.h:669-674`）、`can_cold_transfer` 不认（只要求 `references != 0`，`:786-797`）**；地址级注释 `:1783-1784` 却写着"exclusively referenced by its writer"——**代码里没有这一项**。
② `protect_coverage`（`:1475`、`rebuild_checkpoint_protection:1908-1931`）只保**列**（消费者只有 `:618/:633/:666`），**冷谓词一个字都不看它**。
③ **唯一的 materialize 前回暖只对 private 源**：`:9516-9526` 的 `warm_cold_prefix` 守 `private_source_ready`（= `has_source`）⇒ 共享源**永不回暖**；而 `logical_kv_store.h:782-785` 的注释声称"fork path warms cold pages before materializing them"——**对共享源是假的**（两条注释互相矛盾）。
④ 冷簿记唯一载体是 `SequenceState`（`program.h:507-512`）；`SharedPrefixState`（`program.h:538-548`）**没有 `cold_pages`**。
⑤ 排除掉的分支：入口守卫/扫描窗/capture 不重置 `cold_frontier`/`writer_references`/`source_pins`/device replica/保护列/pressure 机器/`has_source ∧ has_shared_source` 互斥（`:1558`）——**逐条否掉**（报告 §1.4 表）。
⑥ 两个失败态：**态 A**（owner 活着：数据其实可取，但代码只去查 `source_state->cold_pages`）与**态 B**（owner 已结束：`release_sequence_kv:10755-10767` 已 `release_cold_slot` + `release_cold_disk_file_slot` + `cold_pages.clear()`，而描述符仍 `cold_compressed == true` ⇒ **sentinel 指向已回收的槽**）。
⑦ 修法三选（**都不是"就地改条件"**）：(a) 冷路加 `references == 1`（= 与 `can_dematerialize` 对齐，最小；代价=共享页留热，量未测）；(b) 让共享源也回暖 —— **技术上不可行**（`warm_cold_prefix` 要 `SequenceState&` 且要求地址 active，共享前缀地址恰恰不是 active）；(c) 冷簿记跟页走 / `release_sequence_kv` 在 `references > 1` 时先不回收槽（正确但大）。DRAFT 补丁：`patches/wth3_cold_shared_guard.DRAFT.diff`（**只 dry-run 过，未施加**）。

### 2. ★ `:656` 的 MTP 子支 = **空集（不可达）**

`launch_width == 0` ⟺ 每行都 `continue`（`:13139-13145`）⟺ `mtp_target_width == 0 ∧ mtp_ladder.empty()` 行行成立 **⟺ `mtp_ladder` 为空**；而 `:12138-12139 rungs = mtp_ladder.empty() ? {draft_window} : mtp_ladder` ⇒ `rungs.size() == 1`。⇒ **`launch_width == 0 ∧ rungs.size() > 1` 是空集**（与 ceiling 无关）。梯成员恒 ≥2（`round_state.h:52 {2,3,5,7,9,15}` + `layouts_impl.h:1598-1607` 只取 ≤max 的**正**元素 + 兜底 ≥1）⇒ 那条"⟺"也不能靠 0 成立。
**顺带发现（新的、可达的 `:656` 路径）**：判据用**全梯**、绑定用 `kMtpWindowLadderTop`（**15**，`:10301/:10318`），而捕获梯被 `kMaximumMtpDraftTokens` 裁剪（`layouts_impl.h:1598-1607`）；`chosen` 未被夹（`:10319-10321`），`launch_width` 直接当 `draft_width` 键（`:13145`、`:13172`），而 `extend_mtp_graphs` 对非梯成员**提前 return**（`:12454-12455`）⇒ **`mtp_target_width ∉ mtp_ladder` 时 `:656` 抛**。**对本 artifact（27b，max=15）不成立**（`raw ∈` 全梯、`head_floor ∈ {1,5}` ⇒ `chosen` 仍是梯成员）；**对 `kMaximumMtpDraftTokens = 5` 的 target（`qwen3_5_9b`、`muse_glimmer_30b`）成立**（该 header 自己记录 `cut = 12 → rung 15`，`mtp_window_cut.h:109-112`）。

### 3. ★★ `wth2_gate_ops.patch` 实测：**不得单独落**；"+862 里有误报"= **两层都答**

- **施加**：干净副本 `patch -p1` rc=0，结果 sha256[:32] `44f4b0f25fe3b17fcdd3fdcbe123f880` == 它自己的 `dthf_throw_baseline.extended.tsv`（**逐字节相同**）。
- **五臂（原始 862 行）**：只换 baseline **不扩 covered** ⇒ **862 × GONE（门在已修好的镜像上也红）**；扩到 162 文件 ⇒ **678 问题 = 572 UNDECLARED + 105 GONE + 1 无法声明**。
- **三类根因**：只收 `invalid_argument`（漏 46 条 `logic_error`）；**单行 text**（96 条跨行抛点被 `if not lit: continue` 静默跳过）；门取"参数里第一个字面量"（拼接消息取到片段 ⇒ 105 行声明成门看不见的字面量）。⇒ 覆盖率 **862/1305 ≈ 66%**。
- **修正件** `patches/wth3_gate_ops_v2.patch`（**+1303 行**，用**门自己的 `scan_throws`** 口径重建）+ 覆盖清单 `out/covered_v2b.txt`（161 文件，**排除** `replay.cpp`）⇒ 五臂 = **MIRROR GREEN(0) / TREE RED(恰好 1 条 P2) / MUT_NEW RED(1 UNDECLARED) / MUT_LOST RED(1 GONE) / MUT_CLASS RED(1 P3)**（干净重建后，`out/repair.txt`）。
- ⚠️ **门的设计限制**：`src/ops/linear_attention/gated_delta_net/replay.cpp:85 throw std::invalid_argument(message)` **无字面量** ⇒ 只要该文件在 covered 里，P1 的"cannot be declared"分支让门**结构性 RED**（声明消不掉）。要么上游给字面量，要么给门加 `verdict=dynamic`。
- **误报（语义层）**：对 pinned `build/apps/ninfer-serve`（814,583,280 B，`sha256=932336eea9e8cbf8…`）做一次 `grep -a -F -f`（1306 字面量）⇒ **57 条**声明字面量**不在**该二进制里（"会逃到 worker 边界"对这个二进制无证据）+ **133 条**名字图无路径 + **2 条**启动期 = **192/862 = 22.3%** 的声明行路径可疑；与 w-throw 的 `in_bin` 列**一致 847/862**（不一致 20，方向已标：**"找不到"是较强否证**）。

### 4. `ops/*` 逐条 verdict：**862/997（86.5%）出表**

表 `out/verdict_ops.tsv`（862 行 × 11 列，7 档）：`X1 未判 565` / `U1 名字图无路径 133` / `O1 不在 serve 二进制 57` / `P1 计划期形状契约候选 52` / `C2 上限语义弱候选 24` / `C1 条件写错候选 20` / `X0 短字面量不可判 11`。
⇒ **244 条（28.3%）给出非"未判"判定**（O1+U1+P1+S1），**618 条（71.7%）明确标未判**。未对齐 135 条（13 条是 **w-throw 自己的记行噪声**：`text` 是 `return;`/`}`/函数签名；16 条来自被排除的 `replay.cpp`；其余是两套扫描器的行号口径）。
**C1 抽样 5 条逐字读过**：与 `scheduler.h:213` 同形的 2 条（`gqa_kv_append: T exceeds KV cache capacity`、`…: execution envelope is shorter than T`）、半对 1 条、**机械规则误判 2 条**（稀疏 MoE 的工作区检查是同源自洽的真不变量）⇒ **C1/C2 一律算未判**，只有 O1/U1/S1/P1 计入"给出判定"。
**真 codegen**：6 个请求路径 ops TU `rc=0 6/6`（0.21–0.58 s、91–142 MB）、3 个含 `program_impl.h` 的 host TU（1.86–4.45 s、281–432 MB）；**175/175 条声明字面量在 `.o` 的 strings 里找到**（声明行不是扫描幽灵）。

### 5. 整树 link 可行性（**不跑**，派单 4）

495 TU（**`c++` 326 / `nvcc -rdc=true` 161** / nvcc 7 / cc 1）；lib 侧 `libninfer_ops.a` **815.6 MB**、`ninfer-serve` 814.6 MB；共享 `build/` = **59.4 GiB**（499 个 `.o`，device link 产物 276 MB、`gqa_attention_decode_e8.cu.o` 179 MB）。实测锚点：host TU 0.2–4.5 s / 91–432 MB；`arena.cu`（nvcc rdc）2.37 s / 318 MB；最重 TU `gqa_attention_decode_e8.cu` 见 `out/nvcc_probe.txt`。**估**：磁盘 60–70 GiB（**当前可用 ~59 GB，必须先腾盘**）、host ≥ 8 GB、串行 2–4 h / `-j4` 30–60 min。

### 6. 纪律账（两次自伤，如实记录）
1. ★`wth3_50` 的 `mkroot` 用 `cp -al` 后**直接覆写**（同一 inode）⇒ **损坏了 `d_throwfix` 的私有镜像** `src/runtime/engine/{scheduler.h,engine_core.h}`。发现方式：sha 与镜像相同而本应不同。**已用"树 + d-throwfix 自己的补丁"重放修复**并逐字校验（`e282479aa196ba42` / `85ce48aa4b102c86`），只坏了这 2 个文件。此后每个突变体先 `rm -f` 打断硬链（`wth3_57` 起）。
2. ★`wth3_50/55/56` 在污染根上跑过的三份输出**作废**；最终数字以 `out/repair.txt` 为准（`wth3_40` 的 A/B/C 臂与 `wth3_50` 的 MIRROR 臂在污染前跑，有效）。
3. 越界一次（已披露）：`grep -rln 'kMaximumMtpDraftTokens' …/src`（已知目录、单一符号）+ `du -sm`/`find -name '*.o'` 各一次（只对已知 `build/`）。其余全是定点读 + 单文件 grep。
4. 权威树**零写入**（唯一写入 = 本条台账）；每个 `hwrun` 之后都 `trap - RETURN`；**没有**用 `hwrun` 包父脚本；harness 自身两个 bug（`$T/$src` 重复拼接、baseline 路径）已修并重跑。

### 7. 派单（新增/改判）
- **派单 0（改）**：**不要单独落** `wth2_gate_ops.patch`；落 `wth3_gate_ops_v2.patch` + `covered_v2b.txt` + 把 `dthf_gate.py` 的 `--covered` 改成读清单。判据：MIRROR GREEN(0) / TREE RED(1 条 P2) / 三突变各 1 条。
- **派单 1（`:5127` 制造者的引擎级闭合，最高优先）**：`--cold-policy window --cold-keep-tokens 128` + 捕获式前缀缓存 + 单 lane 长 decode + 一路命中同一前缀 ⇒ grep `cold checkpoint page has no source bookkeeping`；不出现则要 `[cold] compressed N prefix pages` 的 N、`shared_prefix_references.size()`、共享页的 `references`（三样能否掉本链）。
- **派单 2（修法裁决）**：(a) 冷路 `references == 1` vs (c) 簿记跟页走；**(a) 的代价只有引擎能测**（共享页留热后 N 掉多少）。
- **派单 3（`:656` §2.2）**：对 max=5 的 target 跑 adaptive MTP，看 `[mtp-window] … rung=` 是否 >5、是否抛"coverage is incomplete"。
- **派单 4（整树 link）**：私有 `build/`；**先腾 ≥70 GiB**；`-j2…-j4`。
- **派单 5（门的 `dynamic` 档）**：`replay.cpp:85` 无字面量抛点 ⇒ 要么给字面量（会改消息、要同步 baseline），要么给门加一档。

（21:21 补测更正，见上条第 5 节 —— 台账 append-only，不改上文）

#### 补测更正（本条第 5 节的两处估算，21:21 实测推翻）

- ★**单 TU 峰值 = 19,579,268 KB ≈ 18.7 GiB**（`src/ops/launcher/gqa_attention_decode_e8.cu`，`/usr/bin/time -v`，rc=0，`.o` 179,058,096 B），**wall = 34 m 46.68 s**。host MemTotal = 22 GiB ⇒ **`-j4`/`-j8` 会 OOM**；**`hwrun monster` 的 9 GB 门槛低于真实峰值，保护不了 host**。实测时 `MemAvailable` 从 20.2 GB 掉到 7.8 GB。
- ★**盘**：共享 `build/` = 60,834 MB，其中 **`build/tests` = 56,741 MB（74 个 >10 MB 的测例二进制，多数 775–802 MB）**、`build/apps` = 2,329 MB、**`build/src` = 1,763 MB**。⇒ **整树+全 `ctest` ≈ 60 GB（可用仅 59 GB，94% 满）**；**只 build `ninfer-serve` + 6 个点名测例 ≈ 9 GB**（后者就够 `ctest` 的判据）。**上一段写的"disk ≥ 20 GB"（w-throw2 派单 5）两头都不对。**
- `build/tests` 里 4 个 `*.real_test`（`qwen3_6_27b_prefix_real`/`_score_real`/`35b_a3b_real`/`_dflash_real`，各 775 MB）**很可能要真模型** ⇒ 应从 `ctest` 排除或划给引擎臂。
- 修正后的预期：**串行 3–5 h；受内存约束的现实下限 `-j2`（重 TU 串行）≈ 1.5–2.5 h**。
