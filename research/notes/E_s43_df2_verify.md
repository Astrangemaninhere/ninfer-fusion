# E_s43 — dflash2 verify/accept 根因 round（S43，CPU-only，未改 src/，未用 GPU）

日期 2026-09-10 · 调查者 E · 只读：`/home/user/ninfer-fusion`（canonical 构建树）
对象：`qwen3_8_27b_nvfp4_dflash2.ninfer --spec auto/dflash2` 下 `gen=2 finish=stop_token`

---

## 0. 结论：根因已定位，可只靠读码证明（patch 见 §6，未应用）

**DFlash2 的 decode ingress 从不填写 `state_source_slots` / `state_destination_slots`。**
其它三个后端都在自己的 ingress 填充里写这两项，只有 DFlash2 漏了：

| 后端 | 填充点 |
|---|---|
| ordinary | `program_impl.h:11831-11832` `ordinary_host_ingress->state_{source,destination}_slots[row] = selectors.{source,destination}` |
| MTP | `program_impl.h:11989-11990` |
| DFlash v1 | `program_impl.h:12177-12178` |
| **DFlash2** | **无**（全部赋值点只有 `program_impl.h:12396-12407` 的 11 个字段：anchors / execution_frontiers / context_frontiers / proposal_extents / target_valid_columns / text_kv_table_rows / dflash_kv_table_rows / active_lanes / sampling；`grep -n 'dflash2_host_ingress->'` 全文只余 9734-9735 的 append 路径，也是同一处遗漏） |

主机 ingress 在 setup 时被 `*dflash2_host_ingress = {}`（`program_impl.h:1068`）零初始化，之后 decode 路径不再重置，**于是这两项永远是 0**；而
`state_image_store.h:401-417` 的 `selectors()` 返回的是 **device slot 号**（`*source_object.device_slot`），
不是 0/1 语义的常量。消费侧 `dflash2_impl.h:371-372` 把 `frame.state_source_slots /
frame.state_destination_slots` 分别交给：

- `target_verify_batch(..., frame.state_source_slots, ...)`（`speculative_target_impl.h:18,23`）——verify 的
  GDN（`TextConfig::gdn_layers()=48/64` 层，`qwen3_8_27b/.../config.h:33-37,64-65`）线性注意力状态**从这个 slot 读**；
- `ops::scatter(frame.selected_hidden, frame.state_destination_slots, continuation_hidden_store)`
  （`speculative_target_impl.h:36`）——接续 hidden **往这个 slot 写**。

⇒ 每个 lane、每一轮 speculative 轮次，**target verify 的线性注意力状态都从 state image device slot 0 读，
而不是该 sequence 自己的 (source, destination) 镜像**（而 KV cache 走的是正确的 `text_kv_table_rows`，
所以 KV 与 recurrent state 处于**互相不一致**的状态）。这一个漏填写即可解释全部实测现象：

1. **verify 列 0 的 logits 被污染** ⇒ target argmax 变成停止符；而 a=0 时"发布列表"里**只有这一枚 target
   argmax**（见 §1），于是它就是那一轮唯一发布的 token ⇒ `finish=stop_token`、`content` 空、
   `gen=2 = 1 个非 spec 的 Begin token + 1 个 spec 轮 token`（§2 有日志算术）。
2. **draft 的 context 特征来自同一次被污染的 verify**（`DFlashFeatureSink` 在 verify 里抓
   `target_feature_layers`，`text_context_impl.h:872-878`、`dflash2_impl.h:405-406`）⇒ draft 提案系统性错误
   ⇒ 实测 `speculative=dflash2 1.00tok/round (0.0%)`（0 accepted / 7 drafted）。
3. **prompt 依赖**：被污染的 argmax 在"计数型/重复型"输入上往往仍是数字（文本看着健康、请求能继续），
   且 GDN 状态对重复序列的敏感度低 ⇒ 那条请求实测 `4.55tok/round (50.9%)`、gen=192 正常跑完；
   而在问答型 prompt 上状态差一个 token 就足以让模型直接收尾（停止符）。

---

## 1. 一轮 dflash2 的语义（file:line）——"发布列表"到底是什么

- ingress：`program_impl.h:12396-12407`；`extent = min(draft_window=7, remaining-1, capacity-frontier-1)`
  （`program_impl.h:12385-12386`）；`anchors[row] = sequence.ledger.back()`，`execution_frontiers[row] = frontier`。
  **锚点/位置约定与 plain 逐行一致**（plain 同款：`program_impl.h:11826-11831`）
  ⇒ coordinator 的"锚点错位"假设**排除**（唯一差异是 dflash 路径不加 `sequence.rope_delta`，
  而 delta 只在多模态非 0，且 `--spec dflash*` 与 `--vision` 被 CLI 互斥，`serve_options.cpp:457`）。
- 设备侧 `dflash2_impl.h:350-441`：`prepare_ragged_prefix`+`append_context_impl`（draft context）
  → `propose_batch_impl`（`dflash2_impl.h:169-348`：`prepare_masked_block` → 5 层 draft → `output_head`
  → `dflash2_selector`，walk 写 `drafts[b*steps+s]`，`dflash2_selector.cuh:180-254`）
  → `speculative_prepare_verify_ids`（`speculative_round.cuh:18-37`：`verify_ids[0]=anchor`、
  `verify_ids[j]=drafts[j-1]`，位置复用 `frame.proposal_positions`）
  → `target_verify_accept`/`target_verify_batch`（`text_context_impl.h:827-866`：
  `run_layers(Verify)` → final norm → lm_head → `ops::argmax` 写 `frame.target_argmax`）
  → `speculative_accept_greedy_drafts`（本轮走 `speculative_round.cuh:131-153` 的 temp=0 快路径，
  实测 `sampling[0]: temp=0 present=0 freq=0`）。
- **接受/提交语义**（= 单元测试 oracle，`tests/ops/test_speculative_round.cpp:104-119,148-163`）：
  `a = 最长等值前缀`（`row_targets[i] == row_drafts[i]`），`t_star = row_targets[a]`，
  `licensed_tokens = drafts[0..a-1] ++ [t_star]`，`licensed_counts = a+1`。
  ⇒ **发布序列里唯一"未被验证"的 token 是那一枚 target argmax**；a=0 时整条发布就是它。

## 2. `finish=stop_token` 在哪决定 + gen 的算术（回答 coordinator (b)）

- 唯一判定点：`src/targets/qwen3_6/impl/frontend/frontend.cpp:1078-1107`（`find(policy.token_ids, token)`
  → `complete(count, FinishReason::StopToken)`），按顺序吃**已发布**的 token
  （`engine_core.h:1152-1183` → `request->output.preview_model(row_tokens, ...)`）。
  ⇒ **(b) 不存在任何"对 draft 做 stop 检查"的路径。**
- 失败请求的算术（`_collab/M_df2_serve_measure.md:30` 原文）：
  `speculative=dflash2 1.00tok/round (0.0%)` ⇒ `1 + accepted/rounds = 1` ⇒ **accepted=0，accept%=0**
  （格式 `request_log.cpp:438-452`；`accepted`/`drafted` 累加见 `program_impl.h:12459-12467`）。
  `gen=2` 且 `decode=31.4tok/s`，而 `decode_tokens := completion_tokens-1 = 1`（`request_log.cpp:565`）
  ⇒ `decode_seconds ≈ 1/31.4 = 31.8ms ≈ decode-host=31515us/round × 1` ⇒ **decode 只有 1 轮**
  ⇒ 第 1 枚 token 来自**非 spec 的 Begin 行**（`resolve_non_speculative_pending` 强制单 token，
  `program_impl.h:1109-1119`），第 2 枚 = 该唯一 spec 轮的 `target_argmax[0]`（a=0）。
  ⇒ **停止符就是第一个 dflash2 轮次列 0 的 argmax**；"用户"来自 Begin 行（非 spec，正常）。
- 附带修正 board 末节"M 的待办"：**接受率不需要新计数器**，`speculative=<backend> X.XXtok/round (YY%)`
  每请求一行已在日志里；JSONL 另有 `accepted_per_position`（`request_log.cpp:298-305,819`）。

## 3. 已排除的假设（每条带读码证据）

| 假设 | 判定 | 证据 |
|---|---|---|
| stop 检查作用在 draft 上 | 排除 | §2；唯一判定点在 frontend，作用于已发布 token |
| max_new 把 draft 计数进去 | 排除 | 预算按发布数扣（`program_impl.h:12450-12451`）；记账错只会 `output_limit`，实测 `stop_token` |
| 接受率地板/降级 `dflash2_acceptance_too_low` | 排除 | 需 `drafted ≥ 128`（≈16 轮），`spec_decision.h:45,101-108`；本请求 decode 仅 1 轮 |
| 验尸器"总接受"（别名 / 拿 draft logits 当 target argmax） | 排除 | `target_argmax` ← `ops::argmax` over `frame.target_logits`（`text_context_impl.h:861-863`）与 `drafts` ← selector walk 是互不重叠的独立 region（`round_state.cpp:178-213`）；且实测 0% 接受 |
| 锚点/位置错位 | 排除 | §1（与 plain 同款 `ledger.back()`+`frontier`） |
| `attention_valid=width` 让无效尾列拿"未来位置" | **latent 契约违反**，本复现不激活 | `dflash2_impl.h:186-193` 传 `width`，dflash v1 传 `valid_columns=extent+1`（`dflash_impl.h:223-228`）；契约见 `include/ninfer/ops/prepare_masked_block.h:20-27`、`include/ninfer/ops/gqa_attention.h:80-84`（"invalid tail 重复最后有效位置"、"A1 不改 invalid 列 cache"）与 `speculative_round.h:26`；但 `extent=min(7,…)=7=k ⇒ width=8=extent+1 ⇒ 无 tail`（只在 output_limit 边界 / capacity 紧张时活）。顺手修法见 §6 Patch B |
| 接受计数缺失（M 的判断） | 排除 | §2 末：计数早已存在并已打印 |

## 4. 为什么"漏填 state slots"能同时解释 0% 接受与立即停止

- `continuation_hidden`/线性状态是一条**跨轮次、跨 lane 共享的 device 池**；写错 slot 会同时
  污染"下一轮读到的状态"和"别人的镜像"。48/64 层 GDN 全靠它 ⇒ logits 必错。
- KV 侧（`text_kv_table_rows`）是**正确**的，所以 prompt 的 129-token 基线完全正常，
  只有 spec 轮次坏 —— 与"同 prompt 基线 gen=129 vs dflash2 gen=2"逐字相符。
- draft 侧特征与 verify 同源 ⇒ 一并被污染 ⇒ 0.0% 接受（而 zh-wiki 上该机制的历史水平是 21-28%，
  counting prompt 是 50.9%，14/14 全落空在"只是 draft 弱"下概率 ≈ 0.1%）。

## 5. 最便宜的验证（按序；第 0 步是决定性的一步，都不需要新代码）

**实验 0（0 代码，决定性）**：同一请求先跑 dflash2、再跑 plain，比较第 1/2 枚 token；
再把 `--max-concurrency` 之外的 `--device-state-slots 1`/`--host-state-slots` 不动，
用**两个并发请求**跑 dflash2（slot 0 属于别的 sequence 时错误会放大）。
```bash
# 基线（注意：不传 --spec 才是 None；"none" 不是合法取值，speculative_options.h:12-16）
build/apps/ninfer-serve /home/user/models/qwen3_8_27b_nvfp4.ninfer \
  --no-thinking --greedy --request-log-jsonl /tmp/e43_base.jsonl --port 8080
# dflash2
build/apps/ninfer-serve /home/user/models/qwen3_8_27b_nvfp4_dflash2.ninfer --spec auto \
  --no-thinking --greedy --request-log-jsonl /tmp/e43_df2.jsonl --port 8081
for p in 8080 8081; do
  curl -s localhost:$p/v1/chat/completions -H 'content-type: application/json' \
    -d '{"model":"x","messages":[{"role":"user","content":"用三句话介绍杭州的地理与历史。"}],"max_tokens":192,"temperature":0}'
done
```
判读：两者**第 1 枚 token 相同、第 2 枚不同** ⇒ verify 列 0 已被污染（与 §0 一致）；
若把 dflash2 的并发数开到 2（两条不同 prompt 同时解码）后错误出现/加重 ⇒ state slot 串镜像的旁证。

**实验 1（0 代码）**：拿 `accepted_per_position`（位置 1 是否 0 接受）：
```bash
grep -o 'speculative=[a-z0-9]* [0-9.]*tok/round ([0-9.]*%)' /tmp/e43_df2.log | tail -5
python3 - <<'PY'
import json
for line in open("/tmp/e43_df2.jsonl", encoding="utf-8"):
    r = json.loads(line)
    if r.get("event") == "request_done":
        s = r.get("speculative") or {}
        print(s.get("backend"), s.get("accepted_tokens"), "/", s.get("drafted_tokens"),
              s.get("accepted_per_position"), "finish", (r.get("result") or {}).get("finish_reason"))
PY
```

**实验 2（1 行，若还要拿 token 级证据；回答 coordinator (c)）**：现有探针
（`NINFER_HEADDBG` `text_context_impl.h:55-58`、`NINFER_KVDUMP_*` `text_context_impl.h:86-130`、
`NINFER_HS_DUMP_DIR` `text_prefill_impl.h:92,140,300`、`NINFER_WS_DUMP` `layouts_impl.h:789`、
`NINFER_FT_STATS` `ft_stats.h:31`）**都不覆盖 speculative round**；要打印 round-1 的
drafted / target_argmax / accepted，加：

```cpp
// decode_dflash2_batch，device.synchronize() 之后（program_impl.h:12433 起）、for(row...) 循环内（:12437）
// 必须在 CUDA graph 之外，不可进 capture。
if (std::getenv("NINFER_DF2DBG") != nullptr) {
    std::int32_t tgt[kDFlashDecodeMaximumWidth]{}, drf[kDFlashDecodeMaximumDrafts]{};
    CUDA_CHECK(cudaMemcpyAsync(tgt, frame.target_argmax.slice(1, 0, batch).data, sizeof(tgt),
                               cudaMemcpyDeviceToHost, device.stream));
    CUDA_CHECK(cudaMemcpyAsync(drf, frame.draft_tokens.slice(1, 0, batch).data, sizeof(drf),
                               cudaMemcpyDeviceToHost, device.stream));
    CUDA_CHECK(cudaStreamSynchronize(device.stream));
    std::fprintf(stderr, "DF2DBG front=%u ext=%u acc=%d cnt=%d src=%d dst=%d "
                 "tgt=[%d %d %d %d %d %d %d %d] drf=[%d %d %d %d %d %d %d] lic=[%d %d]\n",
                 frontier, extent, accepted_i, count_i,
                 dflash2_host_ingress->state_source_slots[row],
                 dflash2_host_ingress->state_destination_slots[row],
                 tgt[0],tgt[1],tgt[2],tgt[3],tgt[4],tgt[5],tgt[6],tgt[7],
                 drf[0],drf[1],drf[2],drf[3],drf[4],drf[5],drf[6],
                 dflash2_host_egress->licensed_tokens[row*width],
                 dflash2_host_egress->licensed_tokens[row*width+1]);
}
```
一次运行即可同时看到 `src=0 dst=0`（漏填的直接证据）、draft 提案、target argmax、接受数、发布 token。

## 6. 补丁（未应用）

**Patch A（根因修，2 行）** —— 在 dflash2 batch 的 ingress 填充循环里补齐 state image slot，与
DFlash v1（`program_impl.h:12177-12178`）逐字同构：

```diff
--- a/src/targets/qwen3_6/impl/runtime/program_impl.h
+++ b/src/targets/qwen3_6/impl/runtime/program_impl.h
@@
             dflash2_host_ingress->active_lanes[row] = static_cast<std::int32_t>(sequence.lane);
             dflash2_host_ingress->sampling[row]     = request.sampling_host;
+            // DFlash2 verify reads the linear-attention (GDN) state through these slots and the
+            // accepted continuation hidden is scattered through the destination slot; the ingress
+            // zero-init leaves both at 0, so every lane/round used state image device slot 0
+            // instead of its own (source, destination) images. Same wiring as DFlash v1
+            // (program_impl.h:12177-12178) / MTP (:11989-11990) / ordinary (:11831-11832).
+            const StateImageSelectors selectors = state_selectors(sequence);
+            dflash2_host_ingress->state_source_slots[row]      = selectors.source;
+            dflash2_host_ingress->state_destination_slots[row] = selectors.destination;
```
算术/语义：`state_selectors(sequence)` = `state_store->selectors(sequence.state.read,
sequence.state.write)`（`program_impl.h:10028-10034`），返回值是 **device slot 号**
（`state_image_store.h:401-417`）；`*dflash2_host_ingress = {}`（`program_impl.h:1068`）⇒ 修前恒为
`src = dst = 0`，修后为该 sequence 自己的 (source, destination)。dflash2 的 append/context 路径
（`program_impl.h:9733-9735`）也漏了同样的两项，建议一并补（那条路径只做 draft context 投影，
可能无害，但同属同源遗漏）。

**Patch B（latent 契约修，1 行换参）** —— 让无效尾列位置按契约重复最后有效位置（draft 自身
`attention_valid=width` 保持不变，仍传给 `ops::swa`）：
```diff
-    ops::prepare_masked_block(anchors, frontiers, attention_valid, Config::mask_token, ids,
+    // positions must follow the target's valid-column contract: the invalid tail repeats
+    // positions[valid-1] (prepare_masked_block.h:20-27, gqa_attention.h:80-84);
+    // dflash v1 passes valid_columns (= extent+1) here too (dflash_impl.h:223-228).
+    ops::prepare_masked_block(anchors, frontiers, valid_columns, Config::mask_token, ids,
                               positions, state.execution.device.stream);
```
（`extent=7` 时 `min(j,extent)=j`，与现状等价 ⇒ 对本复现零影响；`extent<7` 时尾列位置由
`frontier+j` 变为 `frontier+extent`。）

## 7. 给 coordinator 的三个直接回答

- **(a) 首轮锚点/位置是否与 plain 一致**：一致（`ledger.back()` + `frontier`，`program_impl.h:12396`
  vs plain `:11826`）；差异只有 `rope_delta`（多模态才非 0，且 CLI 禁 vision+dflash）。
  **真正的错位不在锚点，而在 state image slot（§0）。**
- **(b) stop 是否可能作用在 draft 上**：不可能（§2）。唯一判定点在 frontend，作用于已发布 token；
  本次失败 a=0，发布序列里根本没有 draft。
- **(c) 有无现成探针打印 round-1 draft/接受前缀**：没有（见 §5 实验 2 列表）；但**接受率已经在日志行里**
  （`speculative=dflash2 1.00tok/round (0.0%)`），JSONL 还有 `accepted_per_position`；
  要 token 级证据需加 §5 实验 2 的那 1 行（含 `src/dst` 打印，直接给出 Patch A 的前后对比）。
