# S44 · 投机接受率计数与上报 (MTP / DFlash / DFlash2)

产物：`_collab/E2_s44_acceptance_counter.diff`（未应用；`patch -p1`）+
`E2_s44_mkpatch.py`（生成器，可重放）/ `E2_s44_syntax_check.py` / `E2_s44_fmt_check.py`（证据脚本）。
本回合未写 `src/**`，未用 GPU，未启动 serve。

## 0. 四句话结论

1. **计数早就有了，而且三后端口径一致** —— 缺的是"带标签的上报"。旧 request-done 行里的
   `speculative=dflash2 4.55tok/round (50.9%)` 中的 `4.55` 与 `50.9%` 就是接受长度与接受率，
   但行里没有 `acceptance` 这个词，M 的关键字扫描自然 `acceptance fields found: (空)`
   （`M_df2_serve_measure.md`）。本项目里"引擎没有接受率计数"这个判断**是假**的，需要更正。
2. 补丁把**同一份已累加的计数**按字段名打印：`spec_drafted / spec_accepted / spec_accept_rate /
   spec_rounds / spec_fallback_steps`（另保留 `spec_accept_len`）；backend=None 时仍只打印
   `off`（零开销），不新增任何每轮工作。
3. `spec_accept_rate = accepted / drafted`，与运行时 DFlash2 降档判据用的是**同一个比值**
   （`spec_decision.h:106`，`kDFlash2MinAcceptance = 0.05`）；accepted 的定义是"验证器保留的
   草稿前缀长度"，部分接受轮只计它的前缀，每轮额外提交的纠正/bonus token（`licensed = accepted+1`）
   **永不计入**。
4. 验证：`patch -p1 --dry-run --fuzz=0` 三文件 clean（exit 0）；补丁后的三个 TU 用构建自身
   flags `-fsyntax-only` 0 失败；把补丁后的格式化函数抽出来在本机跑真数字 0 失败，且与今天日志
   的旧字段**逐位一致**（dflash2 → `spec_accept_rate=0.5085` ↔ 旧 `(50.9%)`，
   `spec_accept_len=4.55tok/round` ↔ 旧 `4.55tok/round`）。

## 1. 一轮投机在哪里结束、计数在哪里累加（既定事实，本轮未改）

| 后端 | 轮完成 + 累加点 (file:line) | drafted / accepted 的来源字段 |
|---|---|---|
| MTP | `src/targets/qwen3_6/impl/runtime/program_impl.h:12020` 起逐行循环，累加 `:12043-12049` | `pcur = mtp_host_ingress->current_extents[row]`（本轮被验证的草稿数）、`accepted_i = mtp_host_egress->accepted_drafts[row]` |
| DFlash | `program_impl.h:12210` 起逐行循环，累加 `:12232-12238` | `extent = dflash_host_egress->proposal_extents[row]`（**实现值**，不是请求值）、`accepted_i = …accepted_drafts[row]` |
| DFlash2 | `program_impl.h:12437` 起逐行循环，累加 `:12459-12465` | `extent = dflash2_host_egress->proposal_extents[row]`、`accepted_i = …accepted_drafts[row]` |

三个后端共用同一份 verify/accept 实现：`src/targets/qwen3_6/impl/runtime/speculative_target_impl.h:9`
的 `target_verify_accept()`（ops 调用在 `:27`）→ `ops::speculative_accept_greedy_drafts`
（`include/ninfer/ops/speculative_round.h`），该 Op 写 `accepted[b] = A`、`licensed_counts[b] = A + 1`。
**这是跨后端可比性的地基**：MTP / DFlash / DFlash2 的 accepted 是同一个核、同一个定义算出来的。

承载计数的**唯一长寿命结构**（禁止引入 per-round 局部计数器）：

```
RequestControl::speculative_stats            src/targets/qwen3_6/impl/runtime/program.h:509
  ← 初始化（每请求清零）install_sampling      program_impl.h:11343-11348
  ← 每轮累加（三个后端各一处）               program_impl.h:12043 / 12232 / 12459
  → FinishResult/AbortResult/CommitRowResult::speculative   program_impl.h:8982 / 9038 / 9056 / 8877
  → GenerationResult::speculative             include/ninfer/types.h:750（结构 :651-660）
                                              target export: src/targets/qwen3_6/export/ninfer/targets/qwen3_6/runtime.h:640/664/673
  → GenerationMetrics                         src/serve/generation_service.h:26-45（填充 generation_service.cpp:460-466）
  → 打印 speculative_str / speculative_json   src/serve/request_log.cpp:438（done 行 :591）/ :298（JSON 记录 :819）
```

任何一层写成 per-round 局部变量，都会在请求结束前被丢掉 —— 上面这条链就是"别写局部"的理由
（`M_accept_eval.md` 关心的正是这条链）。

## 2. `accepted` 的精确定义（可比性的关键）

- `spec_drafted` = 本轮**真正送进验证器**的草稿 token 数。anchor 不算草稿；`extent == 0` /
  `pcur == 0` 的轮贡献 0，落到 `spec_fallback_steps`。
- `spec_accepted` = 验证器**保留**的草稿前缀长度 `A`：greedy 下是与目标 argmax 逐位相符的前缀；
  sampling / DFlash2 selector 下是 `u < min(1, p_i/q_i)` 的前缀。即"没有重新采样、直接采用"的
  草稿 token —— 与任务书要求一致。
- **每轮额外提交的那 1 个 token 不是 accepted**：首个不匹配位置的纠正 token、或全接受后的 bonus
  token，都由 target 自己重采样（`licensed_counts = accepted + 1`）。因此 `accepted <= drafted` 恒成立。
- **部分接受轮**：只把前缀长度计入 accepted、把本轮 `extent` 全量计入 drafted，没有其它补记。
  例：d7 的一轮里接受 3 ⇒ `drafted += 7, accepted += 3`（不是 +3/+4，也不是把该轮记为 0）。
- `spec_accept_rate = accepted / drafted ∈ [0,1]`：**token 级**接受率（不是每轮）；
  `spec_accept_len = 1 + accepted / rounds` 是每轮平均提交长度（旧的 `tok/round`）。
- **恒等式（验收用强约束）**：单请求投机解码 `gen - 1 == spec_accepted + spec_rounds +
  spec_fallback_steps`（每轮提交 `accepted_r + 1` 个 token；dense 回退轮恰好 1 个）。
- **降档/退化轮**（DFlash2 接受率地板 `kDFlash2MinAcceptance=0.05` 触发，或
  `kSpecDemoteTokens=20480` 前沿降档）计入 `spec_fallback_steps`，**不计入** rounds/drafted/accepted。
  于是 `spec_accept_rate` 只描述"还在草稿的区间"的草稿质量（这正是要比的东西），而速度可比性必须
  连着 `spec_fallback_steps` 一起看（同一请求内已不再同质）。

## 3. 补丁（未应用；按符号锚定，行号为插入后位置）

| 文件 | 位置 | 改动 |
|---|---|---|
| `src/serve/request_log.cpp` | 新 `speculative_acceptance_rate()` 插在 `Json speculative_json(` 之前（旧 :298 → 新 :301-307） | text 与 JSON 共用同一比值函数，永不漂移 |
| 同上 | `speculative_json` 键表（旧 `accepted_tokens` 之后，新 :313-315） | `+ "acceptance_rate"`；`drafted == 0` 时为 `null`（对齐 bench JSON 语义） |
| 同上 | `speculative_str`（旧 :438-456 → 新 :464-478） | 打印 `spec_drafted= spec_accepted= spec_accept_rate=(4dp) spec_rounds= spec_fallback_steps=`，有草稿轮时追加 `spec_accept_len=`；backend None 仍返回 `off` |
| `apps/cli/main.cpp` | include 块（新 :5）、backend 名（新 :225） | 修 DFlash2 在 CLI 被显示成 `mtp` 的错标（旧 :221-222 的 DFlash-or-MTP 三目），三后端统一走 `product::speculative_backend_name` |
| `tests/test_request_log.cpp` | 新 :456-466 | 断言文本行字段串 + JSON `acceptance_rate` |

零成本论证：backend=None 提前 `return "off"`（一次比较）；累加路径**完全没动**（本来就每轮在写
`request.speculative_stats`）；新增只是请求结束时的 1 次除法与格式化。三文件均 LF，补丁保持 LF，
新增行均 ≤100 列（clang-format ColumnLimit）。

## 4. 证据（CPU，无 GPU，未写 src/）

```bash
# (a) 干跑：三文件 clean，exit 0
cd /home/user/ninfer-fusion && patch -p1 --dry-run --fuzz=0 < _collab/E2_s44_acceptance_counter.diff
# → checking file src/serve/request_log.cpp / apps/cli/main.cpp / tests/test_request_log.cpp ; exit=0

# (b) 补丁后的三个 TU 用 build/compile_commands.json 自身 flags 做 -fsyntax-only
python3 /mnt/c/Users/User/Documents/ziqinzhang/_collab/E2_s44_syntax_check.py /tmp/e2s44/b
# → exit=0 / exit=0 / exit=0 ; syntax-check failures: 0

# (c) 抽出补丁后的格式化函数，在本机跑真数字 + 不变量（drafted>=accepted, rate∈[0,1]）
python3 /mnt/c/Users/User/Documents/ziqinzhang/_collab/E2_s44_fmt_check.py /tmp/e2s44/b
# → format-check failures: 0

# (d) 现有测试目标本身可在 CPU 上跑（重建后复跑同一条）
/home/user/ninfer-fusion/build/tests/ninfer_request_log_test      # → ok (exit 0)
```

(c) 的实际输出（**这就是 rebuild 后应看到的字段形状**；数字用今天日志反解）：

```
[dflash2 d7, 192-gen run]   speculative=dflash2 spec_drafted=293 spec_accepted=149 spec_accept_rate=0.5085 spec_rounds=42 spec_fallback_steps=0 spec_accept_len=4.55tok/round
[dflash (dspark), 120-gen]  speculative=dflash  spec_drafted=320 spec_accepted=33  spec_accept_rate=0.1031 spec_rounds=86 spec_fallback_steps=0 spec_accept_len=1.38tok/round
[all-demoted dflash2]       speculative=dflash2 spec_drafted=0 spec_accepted=0 spec_accept_rate=0.0000 spec_rounds=0 spec_fallback_steps=57
[unused vector 900/720/300] speculative=mtp spec_drafted=900 spec_accepted=720 spec_accept_rate=0.8000 spec_rounds=300 spec_fallback_steps=2 spec_accept_len=3.40tok/round
```

反解怎么来的（可复核，**不是实测**）：`df2x_dflash2_explicit.log` 15:25:38 的 `gen=192` +
`4.55tok/round` + `(50.9%)` ⇒ `decode tokens = gen-1 = 191 = rounds + accepted`，
`1 + accepted/rounds = 4.55` ⇒ **唯一整数解 rounds=42, accepted=149**（`149/42 = 3.548 → 4.55` ✓），
再由 `accepted/drafted ∈ [0.5085, 0.5095]` ⇒ **drafted=293**（`149/293 = 0.50853 → 50.9%` ✓，
`drafted/rounds = 6.98 ≈ 7` 与 `--spec dflash2` 固定 d7 相符，有一轮被预算/上下文裁了 1）。
`s4w_dspark.log` 同法：`gen=120` + `1.38tok/round` + `(10.3%)` ⇒ rounds=86, accepted=33,
drafted ∈ {319,320,321}（取 320）。这说明**旧字段与新增字段是同一个比值**，故补丁不改变语义、只加标签。

## 5. 落地后检查 (post-build)

1. **重编范围**：只需 `ninfer_serve`（`request_log.cpp`）+ `ninfer`（`apps/cli/main.cpp`）+
   `ninfer_request_log_test`；三文件都不在 CUDA 侧，不必重编任何 `.cu`。
2. `build/tests/ninfer_request_log_test` → 期望 `ok`（新断言在 S44 hunk 里，一旦字段串不匹配即报错）。
3. **每后端一条 serve 运行**（同 prompt、同 192 token、greedy；各发 1 个请求即可）：
   ```bash
   # 注意：MTP 必须给 --draft-tokens（mtp ∈[1,5]，dflash ∈[1,15]，dflash2 固定 7 不可传别的值）
   <build>/apps/ninfer-serve <artifact> --spec mtp     --draft-tokens 3 ...
   <build>/apps/ninfer-serve <dspark artifact> --spec dflash --draft-tokens 4 ...
   <build>/apps/ninfer-serve <artifact> --spec dflash2 ...
   grep 'spec_drafted' <log>          # 每臂的 request-done 行
   ```
   今天 `--spec mtp` 报 `requires --draft-tokens in [1,5]`→`SERVE_FAILED`（`s4w_plain_mtp.log`）是
   **flag 校验**，不是引擎/MTP 缺陷；MTP 臂今天从未真正起来过，四档表里的 MTP 数据是缺的。
4. **判据（每条缺一不可）**：
   - 每臂行内都出现 `spec_drafted=… spec_accepted=… spec_accept_rate=… spec_rounds=…`；
     `--spec` 关闭那一臂只有 `speculative=off`（零成本回归检查）。
   - `spec_drafted >= spec_accepted` 且 `spec_accept_rate ∈ [0,1]`。
   - 恒等式 `gen - 1 == spec_accepted + spec_rounds + spec_fallback_steps`。
   - 交叉检查 `spec_rounds + spec_fallback_steps == engine_timing.decode_rounds`
     （同一行已打印 `decode-host=…us/round`，其分母即 decode_rounds）。若左边小于右边，说明有
     decode 轮绕过了三条 spec 循环（真缺陷，而不是计数缺失）。
   - 期望量级：dflash2 ≈ 0.50（今天 50.9%）、dflash ≈ 0.10（今天 10.3%）、MTP 未知待测。
5. 把四列（decode tok/s + `spec_accept_rate` + `spec_drafted` + `spec_fallback_steps`）填进
   `M_spec_4way.md` 那张表，MTP vs DFlash vs DFlash2 才可归因（草稿质量 vs 草稿成本 vs 回退占比）。
   注意 dflash2 是 d7、dflash 是 d4、MTP 是 d3 —— `spec_accept_rate` 可比"草稿质量"，
   但"每轮收益"要配 `spec_accept_len` 与 `spec_fallback_steps` 一起读。

## 6. 边界与未验证 (OPEN)

- **无 GPU**：没有实测四元组，293/149/42 是从旧日志两个已打印字段**反解**的唯一整数解；落地后
  第一条 serve 日志即为实测值。若实测与反解不符，先查 `spec_rounds + spec_fallback_steps ==
  decode_rounds`，再查 `gen-1 == accepted + rounds + fallback`（两条恒等式任一不成立就是引擎侧问题）。
- `drafted` 的一处语义差异（不影响可比性，但比较时必须知道）：MTP 的草稿在**上一轮**生成，
  `next_extents`（本轮新生成、尚未验证的草稿）**不计入**本轮 `drafted` —— 只有被验证的草稿才计数；
  DFlash/DFlash2 同轮生成同轮验证。所以"草稿成本"要按 `spec_drafted` 比，而不是按
  `draft_window` 名义值比。
- 本轮刻意不动的相邻问题：
  (i) `apps/cli/main.cpp` 的 DFlash2 错标在本补丁内修；bench 侧 CSV 的 `spec_acceptance_rate`
      一直是对的（`bench/targets/qwen3_6_27b/ninfer_bench_support.cpp:229-242`、表头 `:769`），
      `tools/bench/run_ninfer_bench_matrix.py:327` 读的就是它，不受影响。
  (ii) serve JSON 原本只有 rounds/drafted/accepted/fallback（`request_log.cpp:299-305`），补丁补上
      `acceptance_rate`；存量 JSON 消费方（`tools/bench/run_serve_corpus.py`、`run_serve_concurrency.py`）
      只取白名单键，新增键不破坏它们（它们按 `event == "request_done"` 抓记录、缺键才报错）。
  (ii-b) **四档对比的另一个真障碍（本轮未修）**：`tools/bench/run_serve_corpus.py:365-372` 的
      `block_fixture_names()` 只接受 `none|mtp|dflash`，遇到 `dflash2` 直接
      `CampaignError: unsupported speculative backend: dflash2` —— 就算接受率字段到位，语料/并发
      harness 仍跑不了 dflash2 臂；4 档 bake-off 前需要给它加一支（或用 `--spec` 手跑 serve + grep）。
  (iii) `spec_accept_len` 在降档请求上会把降档后的 dense 轮排除在分母之外，读数必须配
      `spec_fallback_steps`；需要"每 decode 轮平均提交 token"时用 `gen / (rounds + fallback)`。
- 未做（留给窗口）：`--spec dflash`（dspark 产物）与 MTP 臂的 serve 实测；`dflash2` 的
  `spec_accept_rate` 与 `kDFlash2MinAcceptance` 的事后核对（0.5 ≫ 0.05，今天不会触发地板）。
