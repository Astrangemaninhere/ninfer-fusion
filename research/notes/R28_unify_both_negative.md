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
