#!/usr/bin/env python3
"""State sync: the Spark-X2.5 import, the importer fix, the GUI/i18n workstream, FlashNext done."""
import datetime
import pathlib

T = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/_TODO.md")
stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
T.open("a", encoding="utf-8").write("""
## 121. Spark-X2.5 导入实测 + 导入器修复 + GUI/i18n 工作流（""" + stamp + """）

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
""" )
print("appended section 121")
