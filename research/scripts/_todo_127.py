#!/usr/bin/env python3
"""§127：低接受率根因重定位（4 组复现的固定偏差）+ 三个子代理的结论。"""
import datetime
import pathlib

stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
T = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/_TODO.md")
T.open("a", encoding="utf-8").write("""
### 127. 低接受率根因重定位：verify 与 plain 的**固定偏差**（4 组复现）—— """ + stamp + """
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
""" )
print("_TODO.md §127 已追加")
