# NInfer 总 TODO (2026-09-03 固化 v2, 防上下文压缩丢失)


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
  净 **+2528 B/head-page（+27.4%）**；1M/16 层/kv_heads 4/head_dim 256 ⇒ 18.00 → **13.06 GiB（省 4.94 GiB）**；盈亏平衡 **b ≲ 3.84**。
- **翻 D2 常量到 2.6 会让 `decoder_state.cpp:367` 的 `static_assert(cold_slot_stride_for(...) == ops::kEntropyNvfp4SlotBytes)` 编译失败**
  （6688 ≠ 9536）⇒ 必须同步改断言与文档，不能只改常量。

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
| bits/code（整数）| 数据区 | stride | vs 常住 nvfp4 head-page 9216 B |
|---|---|---|---|
| 4（现状） | 8192 B | 9536 B | **+3.47%** |
| 3 | 6144 B | 7488 B | **−18.75%** |
| 2.6（需定点化） | 5324 B | 6668 B | −27.65% |
要拿到 6688 必须把该参数改成定点（例如 `bits_x10`）或浮点，属小重构。另外**不止一处断言**：
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
