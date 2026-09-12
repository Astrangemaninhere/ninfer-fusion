import pathlib

p = pathlib.Path(r"C:\Users\User\Documents\ziqinzhang\_TODO.md")
old = p.read_text(encoding="utf-8", errors="replace")
block = """
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
"""
p.write_text(old + block, encoding="utf-8")
print("_TODO.md ->", len(p.read_text(encoding="utf-8", errors="replace").splitlines()), "lines")
