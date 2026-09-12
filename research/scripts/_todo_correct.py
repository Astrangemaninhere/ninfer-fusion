import pathlib

p = pathlib.Path(r"C:\Users\User\Documents\ziqinzhang\_TODO.md")
old = p.read_text(encoding="utf-8", errors="replace")
block = """
## 2026-09-12 凌晨 · 更正：上一条"tap1..4 为零"的根因结论**作废**

### 事实更正（已独立复验，与子代理路线 A 一致）
上一条我写的"`features` 只有第 0 段有数据、tap1..4 整块为零 ⇒ 特征链输入缺料是根因"是**错的**，
成因是**我自己的离线读法轴错位**：引擎 `Tensor` 的 `nb[0]` 才是连续轴（`src/core/tensor.cpp:51-58`
`set_contiguous_strides`），故 `[25600, W]` 缓冲必须读成 `(W, 25600)`，我按 `(25600, W)` 读并切 tap 段，
把"整列范数"误当"tap0 段范数"。

正确读法（`dl/feat_features_{5..8}.bin`，各 204800 元素 = 8×25600）：
- 各列范数 = `[278.4, 0×7]`（只有第 0 列是 live 列）；
- **第 0 列的 5 个 tap 全部非零且随深度单调递增**：
  call5 `[102.36, 118.58, 119.42, 127.74, 149.55]`、call6 `[89.49,114.59,117.43,125.31,151.61]`、
  call7 `[101.01,116.62,117.62,127.21,149.38]`、call8 `[96.59,113.55,116.19,125.93,151.26]`
  ⇒ 反证"5 个槽写同一层"（同层范数应相等），5 槽确为 5 个不同深度；
- 只有第 0 列非零是**设计**：`prepare_ragged_prefix` 对 `column >= count` 写零
  （`prepare_ragged_prefix.cuh:24-27`），`count = frontier − context_frontier = 上一轮 licensed count`；
  实测 `accepted=0 count=1` ⇒ live=1 ⇒ **这是接受率塌的结果，不是原因**。
- tap 写入链无 bug：`text_context_impl.h:288-295` 偏移正确；`:311-314` 有硬门
  （任一 tap 漏采即抛 "did not publish every feature layer"），8 次调用均未抛 ⇒ 5/5 命中。

### 另一处作废：那条"下游算术逐位吻合 0.00000"
该验证是**空的**：脚本读的是旧版残留的**无编号** `feat_features.bin`（204800 元素全零），`0/0` 在 `1e-9`
保护下打印成 `0.00000`。用**有编号**文件与正确轴向复算：`fc @ features` 相对误差 **161.87**
（因为 `dflash2/feature_projection` 是**量化权重**，不能按 float16 直读）、rmsnorm 口径也不吻合（0.12）。
⇒ "下游算术已排除"这条**不成立**，只是尚未验证。

### 仍然站得住的部分
- `[df2cand]` 逐列归因（候选 id 是标量，无轴问题、无跨域映射）：干净列 **15/15 = 100% head_miss**、
  全部 290 步 84.8% head_miss / 10.7% walk_error / 4.1% verify_flip；
- 7 项假设排除（4bit 提案头 / 混血头 / 选择器 / 5 层网 / 提案头不在路径 / tap 层号 / KV 精度）；
- `text/draft_head` + `draft_head_token_ids` 属于 **MTP** 路径（artifact 文档），与"清零它不影响 dflash2"一致。

### 下一步
等子代理路线 B、C 回来后对照；若三路都否定"引擎 bug"而指向候选质量，则用独立口径复核
`head_miss` 归因本身（含"每个 walk 步对应哪个目标位置"的对齐），避免在分析层再犯同类错误。
"""
p.write_text(old + block, encoding="utf-8")
print("_TODO.md ->", len(p.read_text(encoding="utf-8", errors="replace").splitlines()), "lines")
