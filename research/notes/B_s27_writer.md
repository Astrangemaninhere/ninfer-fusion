# B_s27_writer.md — S27: FlashNext NVFP4 writer = 重排器 (reformatter) 证明

日期 2026-09-09 · B · 改动: `flashnext_convert.py` (+`--measure-shapes`、measured-shapes
加载/回填、self-test 动态引脚) + 新数据文件 `specs/qwen4_exp_measured_shapes.json` +
round-trip 工件 `tools/archkit/out/flashnext_s27/` (12 payload + meta.json)。

## 核心问题的答案: 引擎 NVFP4 布局 vs modelopt 输出 — 需要什么重排?

| 成分 | modelopt 磁盘格式 (实测 header) | 引擎 `blockscale-k16-m128x4-v1` (源码实证) | 结论 |
|---|---|---|---|
| 权重码字 | `weight` U8 行主序 [N, K/2], 偶元素=低半字节 (fp4x2 惯例) | code plane 行主序 [N, K/2] @ offset 0 (`nvfp4_gemv.cuh:172` `row*kCodeBytesPerRow`; `decode_nvfp4_e2m1x2` .x=低半字节) | **逐字节原样, 无 nibble 重排** |
| 组尺度 | `weight_scale` F8_E4M3 自然行主序 [N, K/16] | **512 字节 128×4 tile 重排**: `offset=(m_tile*ktiles+k_tile)*512 + row_mod32*16 + quartile*4 + lane` (`nvfp4_gemv.cuh:87-95`) | **必须 swizzle** |
| 全局尺度 | `weight_scale_2` 标量 F32 | payload 尾部一个 F32 **除数** (引擎 `coefficient = s1 * inverse_weight_divisor`) | 存 **1/weight_scale_2** (Muse 先例 convert.py:317-326) |
| 激活尺度 | `input_scale` 标量 F32 | Weight 结构的 `input_scale_divisor` 字段 (metadata, 不在 payload) | 走 artifact 计划元数据 |
| 几何 | gate/up [640,2560], down [2560,640]; scale [640,160]/[2560,40] | N%128=0 ✓ K%64=0 ✓; payload = K/2*N + align256 + N*K/16 + 4 | 两例均 921,604 字节整 |

既有共享打包器 `tools/artifact/layouts.py::encode_nvfp4` (Muse 已投产) 正是这套语义:
codes 原样 + `swizzle_nvfp4_scales` + F32 尾字。**无需新写 swizzle 代码 — 复用同一打包器。**

## 任务 2: 单层 round-trip 证明 — 12/12 全部字节级相等

命令: `wsl.exe -e bash -c "cd /mnt/c/Users/User/Documents/ziqinzhang && python3 _collab/_b_tmp/b_s27_roundtrip.py"` (脚本随本报告归档要点; 产物在 `tools/archkit/out/flashnext_s27/`)
读 layer 0 experts 0..3 × {gate,up,down} (shard `layer-00000-experts-0000-0127.safetensors`) →
`encode_nvfp4`(codes, scale, pack("<f",1/s2), (N,K)) → 落盘 `.bin` + meta.json → 读回 →
`decode_nvfp4_words` 逆解 → 比对:
```
layer0.expert0.gate  shape=[640,2560] payload=921604 codes=True scale=True div=True geo=True swz=True
... (12 行全部相同, ROUNDTRIP: ALL 12 PASS, exit 0)
```
- codes/scale: 写盘往返后与源 shard 张量 **字节相等**; div: F32(1/weight_scale_2) 位级相等。
- `geo=True`: payload 长度与 `block_scale_geometry` 公式一致, scale_plane_offset==align256。
- `swz=True`: **独立**按 CUDA 核内公式 (`nvfp4_scale_offset` 逐字转写) 抽 12 个 (row,group)
  验证 payload[scale_off+offset] == 自然尺度字节 — 打包器与内核公式互证。

## 任务 1: 460 行 PENDING_SHAPE 回填 — 机制已落地, 数据被下载进度限制

实测时点磁盘真相: `layer-*-experts-*` 全可读; `model-bf16-00001` 下到一半 (文件在增长);
`model-bf16-00002..00012`、`model-plefp8-00000..00009` 未到位 (PENDING_SHAPE 族的张量全在
bf16 族分片; PLE 表在 plefp8 族)。因此:
- 落地 `flashnext_convert.py --measure-shapes <ckpt_dir>`: 容忍半截分片, 实测头维度写
  `specs/qwen4_exp_measured_shapes.json`; `build_plan` 对已测引擎 PENDING_SHAPE→READY,
  `artifact_shape` 用实测维度。**下载推进中重跑一次即已回填 18/460** (bf16-00001 部分层到位)。
- 剩余 442 行: 全部因分片未下完 (mtp.fc_* 在 model-bf16-00012)。recipe 步骤 3 完成后为 0。
- self-test 引脚改为动态: `n_pending == 460 − backfilled`, 基数 460 固定防回归; 当前 exit 0。

## 任务 3: exclude 清单的物理后果 (hf_quant_config.json + 分片族名双向印证)

- **NVFP4 路径 (encode_nvfp4 重排)**: 仅 48 层 × 512 专家 × {gate,up,down} = 73,728 张量
  (+221,184 伴生), 分片族 `layer-000NN-experts-*`。这是**唯一**走 swizzle/打包路径的权重。
- **BF16 原样 (contiguous-le-v1 直通)**: embed_tokens、lm_head、全部 self_attn (QSA q/k/v/o/
  q_norm/k_norm/indexer)、全部 linear_attn (GDN 含 in_proj_*)、mlp.gate (router)、
  shared_expert*、shared_expert_gate、全部 hyper_connection* (384+3)、mtp.* — 分片族
  `model-bf16-*` (exclude_modules 与分片内容一一对应, 已抽查 header dtype)。
- **PLE 表**: 存于 `model-plefp8-*` 10 分片 (**表本体是 FP8 存储**, 带表级 weight_scale)。
  测量时全部未到位 — FP8 子布局 (分组? 行主序?) **留待分片落地后钉死, 本轮不猜**; sidecar
  路径 (拼接 128 逻辑分片) 不变。
- visual 333: text-first 跳过。

## 任务 4: 下载完成后的最终配方 (取代 S26 版)
```bash
M=models/Qwen3.8-Flash-Next-ABLITERATED-NVFP4
# 0) 分片齐全: python3 -c 预检 (同 S21; 期望 206 shards missing 0)
# 1) 契约审计 (仅名字): python3 ninfer-fusion-repo/tools/archkit/flashnext_bindings.py \
#      --audit $M/model.safetensors.index.json
#    期望 matched=74931 engines=74804/74804 missing=0 companions=221184 residue={visual:333,mtp:27}
# 2) 回归门: python3 ninfer-fusion-repo/tools/archkit/flashnext_convert.py --self-test \
#      --real-names _collab/M_flashnext_names.txt → PASS exit 0
# 3) 维度回填 (S27 新): python3 .../flashnext_convert.py --measure-shapes $M
#    期望 "[measure] ... 460/460 PENDING_SHAPE engines backfilled (0 await)"
#    随后重跑 2) 期望 "shapes: 460 backfilled ... 0 still pending"
# 4) 计划+形状核对: python3 .../flashnext_convert.py --checkpoint $M
#    期望 missing=0 residue=360; mismatch_count=0 (形状全对齐) 或新可行动清单
# 5) artifact 写出 (S28+, 未开工): 逐引擎 encode_nvfp4 (仅专家族) / BF16 直通 / PLE sidecar
#    拼接; round-trip 已证 encode/decode 通路, 全量写出器仍缺 (本产物未含)。
```

## 附记
- round-trip 工件 `out/flashnext_s27/` (12×~900KB + meta.json) 保留为证据;
  `specs/qwen4_exp_measured_shapes.json` 保留 (55 实测, 18 已回填, 随下载增长)。
- 探针脚本 `_collab/_b_tmp/*` 已清理 (要点与命令在本文; `--measure-shapes` 已固化进转换器)。
- py_compile 通过; self-test exit 0; 未构建、未碰 GPU。
