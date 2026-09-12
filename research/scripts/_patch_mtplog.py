#!/usr/bin/env python3
"""MTP 逐步草稿 logits 落盘插桩（树的先决测量）。

现状：proposal_logits 每步被覆盖，只留最后一步 ⇒ 无法离线算 MTP 的树。
本插桩在每步 mtp_propose_batch 之后落盘 logits + draft_tokens（带递增序号），
门控 NINFER_MTPLOG，需 --no-cuda-graph。纯新增。
"""
import pathlib
import sys

R = pathlib.Path("/home/user/ninfer-fusion")
rel = "src/targets/qwen3_6/impl/runtime/text_context_impl.h"
p = R / rel
raw = p.read_bytes().decode("utf-8")
nl = "\r\n" if "\r\n" in raw else "\n"
BAK = pathlib.Path("/home/user/mtplog_bak") / rel
BAK.parent.mkdir(parents=True, exist_ok=True)
if not BAK.exists():
    BAK.write_bytes(raw.encode("utf-8"))
    print("备份 ->", BAK)

anchor = nl.join([
    "void TextContext::mtp_propose_batch(const Tensor& hidden, Tensor& logits, Tensor& draft_tokens) {",
    "    const std::int32_t batch = hidden.ne[1];",
    "    require_tensor_shape(hidden, DType::BF16, {kCfg.hidden, batch}, \"MTP proposal batch hidden\");",
    "    require_tensor_shape(logits, DType::BF16, {kCfg.vocab, batch}, \"MTP proposal batch logits\");",
    "    require_tensor_shape(draft_tokens, DType::I32, {batch}, \"MTP proposal batch tokens\");",
    "    proposal_argmax(hidden, logits, draft_tokens);",
    "}",
    "",
])
if raw.count(anchor) != 1:
    print(f"FAIL 锚点命中 {raw.count(anchor)} 次")
    sys.exit(3)

block = nl.join([
    "#include <cstdio>",
    "#include <cstdlib>",
    "",
    "void TextContext::mtp_propose_batch(const Tensor& hidden, Tensor& logits, Tensor& draft_tokens) {",
    "    const std::int32_t batch = hidden.ne[1];",
    "    require_tensor_shape(hidden, DType::BF16, {kCfg.hidden, batch}, \"MTP proposal batch hidden\");",
    "    require_tensor_shape(logits, DType::BF16, {kCfg.vocab, batch}, \"MTP proposal batch logits\");",
    "    require_tensor_shape(draft_tokens, DType::I32, {batch}, \"MTP proposal batch tokens\");",
    "    proposal_argmax(hidden, logits, draft_tokens);",
    "    // 树的先决测量：MTP 每步的草稿 logits 与选定 token。proposal_logits 每步被覆盖，",
    "    // 只留最后一步 ⇒ 离线无法重建逐深度分布与 hit@b。门控 NINFER_MTPLOG，需 --no-cuda-graph。",
    "    if (std::getenv(\"NINFER_MTPLOG\") != nullptr) {",
    "        static int mtplog_calls = 0;",
    "        if (mtplog_calls < 96) {",
    "            ++mtplog_calls;",
    "            auto dump_t = [&](const char* tag, const Tensor& view) {",
    "                if (view.data == nullptr || view.numel() == 0) { return; }",
    "                std::vector<std::byte> host(view.bytes());",
    "                CUDA_CHECK(cudaMemcpyAsync(host.data(), view.data, host.size(),",
    "                                           cudaMemcpyDeviceToHost, ctx_.stream));",
    "                CUDA_CHECK(cudaStreamSynchronize(ctx_.stream));",
    "                char path[256];",
    "                std::snprintf(path, sizeof(path),",
    "                              \"/mnt/c/Users/User/Documents/ziqinzhang/dl/mtplg_%s_%d.bin\",",
    "                              tag, mtplog_calls);",
    "                if (std::FILE* fh = std::fopen(path, \"wb\")) {",
    "                    std::fwrite(host.data(), 1, host.size(), fh);",
    "                    std::fclose(fh);",
    "                }",
    "            };",
    "            dump_t(\"logits\", logits);",
    "            dump_t(\"tokens\", draft_tokens);",
    "            std::fprintf(stderr, \"[mtplog] n=%d vocab=%d batch=%d logits_bytes=%zu\\n\",",
    "                         mtplog_calls, kCfg.vocab, batch, logits.bytes());",
    "        }",
    "    }",
    "}",
    "",
])
p.write_bytes(raw.replace(anchor, block, 1).encode("utf-8"))

chk = p.read_bytes().decode("utf-8")
assert chk.count("NINFER_MTPLOG") == 1, "getenv 未落"
assert chk.count("mtplog_calls") == 5, "计数器次数异常"
print("OK MTP 逐步 logits 插桩已落")
