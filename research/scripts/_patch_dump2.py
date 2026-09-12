#!/usr/bin/env python3
"""修正 dump 位置：从 mtp_propose_batch 移到 proposal_argmax 的两个分支内。

根因（实测）：走短名单路径时 proposal_argmax 把 argmax 写进 proposal_tokens，
**从不写调用方传入的 logits 实参** ⇒ 在 mtp_propose_batch 里 dump 得到的是全零
（实测 48 步全为 top1=0/top2=1/margin=0/熵=ln(248320)=12.422，即均匀分布）✗

本补丁：
 - 在短头分支 dump 真实使用的 proposal_logits（131072×T BF16）+ 最终 tokens（映射后全局 id）
 - 在全词表分支 dump output_logits（248320×T）
 - 首次调用时一次性 dump proposal_head_ids_（131072×I32 = 524 KB）供离线反查
 - 同时撤掉 mtp_propose_batch 里那段（它测错了对象）
"""
import pathlib
import sys

R = pathlib.Path("/home/user/ninfer-fusion")
rel = "src/targets/qwen3_6/impl/runtime/text_context_impl.h"
p = R / rel
raw = p.read_bytes().decode("utf-8")
nl = "\r\n" if "\r\n" in raw else "\n"

BAK = pathlib.Path("/home/user/mtplog_bak") / (rel + ".v2")
BAK.parent.mkdir(parents=True, exist_ok=True)
BAK.write_bytes(raw.encode("utf-8"))
print("备份 ->", BAK)

# --- 1) 先撤掉 mtp_propose_batch 里那段（含 include）---
old_block_start = "    proposal_argmax(hidden, logits, draft_tokens);" + nl
start_idx = raw.find(old_block_start)
if start_idx < 0:
    print("FAIL 未找到 mtp_propose_batch 的调用行")
    sys.exit(3)
end_marker = "}" + nl + nl + "void TextContext::attn_mix"
end_idx = raw.find(end_marker, start_idx)
if end_idx < 0:
    print("FAIL 未找到该函数结尾")
    sys.exit(3)
raw = raw[:start_idx] + old_block_start + raw[end_idx:]
# 去掉我先前加的 include
raw = raw.replace("#include <cstdio>" + nl + "#include <cstdlib>" + nl + nl, "", 1)
print("已撤掉 mtp_propose_batch 里那段 + 两个 include")

# --- 2) 在两个分支内各插一段 dump ---
short_anchor = nl.join([
    "        ops::proposal_remap_token_ids(proposal_tokens, proposal_head_ids_, proposal_head_n_,",
    "                                      ctx_.stream);",
])
long_anchor = nl.join([
    "        ops::argmax(output_logits, proposal_tokens, kCfg.token_domain, ctx_.stream);",
])

dump_helper = nl.join([
    "        // 树的先决测量：dump 真实使用的 logits 与最终 token（门控 NINFER_MTPLOG）。",
    "        if (std::getenv(\"NINFER_MTPLOG\") != nullptr) {",
    "            static int mtplog_calls = 0;",
    "            if (mtplog_calls < 96) {",
    "                ++mtplog_calls;",
    "                auto mtplog_dump = [&](const char* tag, const Tensor& view) {",
    "                    if (view.data == nullptr || view.numel() == 0) { return; }",
    "                    std::vector<std::byte> host(view.bytes());",
    "                    CUDA_CHECK(cudaMemcpyAsync(host.data(), view.data, host.size(),",
    "                                               cudaMemcpyDeviceToHost, ctx_.stream));",
    "                    CUDA_CHECK(cudaStreamSynchronize(ctx_.stream));",
    "                    char path[256];",
    "                    std::snprintf(path, sizeof(path),",
    "                                  \"/mnt/c/Users/User/Documents/ziqinzhang/dl/mtplg2_%s_%d.bin\",",
    "                                  tag, mtplog_calls);",
    "                    if (std::FILE* fh = std::fopen(path, \"wb\")) {",
    "                        std::fwrite(host.data(), 1, host.size(), fh);",
    "                        std::fclose(fh);",
    "                    }",
    "                };",
    "                mtplog_dump(\"logits\", DUMP_LOGITS_VIEW);",
    "                mtplog_dump(\"tokens\", proposal_tokens);",
    "                if (mtplog_calls == 1 && DUMP_EXTRA_MAP) {",
    "                    Tensor map_view = make_tensor_like_ids();",
    "                    mtplog_dump(\"head_ids\", map_view);",
    "                }",
    "            }",
    "        }",
])

# 短头分支
ins_short = nl.join([
    "        ops::proposal_remap_token_ids(proposal_tokens, proposal_head_ids_, proposal_head_n_,",
    "                                      ctx_.stream);",
    "#define DUMP_LOGITS_VIEW proposal_logits",
    "#define DUMP_EXTRA_MAP 1",
]) + nl + dump_helper + nl
raw = raw.replace(short_anchor, ins_short, 1)

# 全词表分支
ins_long = long_anchor + nl + nl.join([
    "#define DUMP_LOGITS_VIEW output_logits",
    "#define DUMP_EXTRA_MAP 0",
]) + nl + dump_helper + nl
raw = raw.replace(long_anchor, ins_long, 1)

p.write_bytes(raw.encode("utf-8"))
chk = p.read_bytes().decode("utf-8")
for tok in ("mtplg2_%s_%d.bin", "DUMP_LOGITS_VIEW proposal_logits",
            "DUMP_LOGITS_VIEW output_logits"):
    assert chk.count(tok) == 1, f"缺失 {tok}"
assert chk.count("mtp_propose_batch") >= 1
print("OK dump 已移入 proposal_argmax 两分支（含短头/全词表 + 映射表）")
print("注意：还需要一个 proposal_head_ids_ 的 Tensor 视图辅助；见补丁后的编译错误提示。")
