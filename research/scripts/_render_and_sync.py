#!/usr/bin/env python3
"""完整渲染 Spark 缺口报告（中/英）并写状态同步。"""
import datetime
import json
import pathlib
import sys

G = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/ninfer-fusion-repo/tools/gui")
sys.path.insert(0, str(G))
import gui_i18n as i18n  # noqa: E402
import model_import as mi  # noqa: E402

man = json.loads(pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/ninfer-fusion-repo/"
                              "tools/archkit/out/spark-x2.5-4b/manifest.json").read_text())
for lang in ("zh", "en"):
    i18n.set_lang(lang)
    print("=" * 74)
    print("LANG =", lang, " / entry = render_gap_report_text")
    print("=" * 74)
    print(mi.render_gap_report_text(man))
i18n.set_lang("zh")

T = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/_TODO.md")
stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
T.open("a", encoding="utf-8").write("""
### 122. S53 复核 + 裁决 + GUI 收尾（""" + stamp + """）
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
""" )
print("\n_TODO.md §122 已追加")
