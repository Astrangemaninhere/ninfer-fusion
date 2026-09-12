#!/usr/bin/env python3
"""MVP-1 插桩（自动探测行尾版）：NINFER_DF2SCORES 门控 dump 16x16 scores + cand + unary。"""
import pathlib
import sys

R = pathlib.Path("/home/user/ninfer-fusion")
rel = "src/targets/qwen3_6/impl/runtime/dflash2_impl.h"
p = R / rel

BAK = pathlib.Path("/home/user/df2scores_bak") / rel
BAK.parent.mkdir(parents=True, exist_ok=True)
BAK.write_bytes(p.read_bytes())
print("已备份 ->", BAK)

raw = p.read_bytes().decode("utf-8")
nl = "\r\n" if "\r\n" in raw else "\n"
print(f"行尾探测: {'CRLF' if nl == chr(13)+chr(10) else 'LF'}")

anchor = nl.join([
    "                          head_domain, head_token_ids, state.execution.device.stream);",
    "    state.execution.work.reset();",
    "",
])
if raw.count(anchor) != 1:
    print(f"FAIL 锚点命中 {raw.count(anchor)} 次"); sys.exit(3)

block = nl.join([
    "                          head_domain, head_token_ids, state.execution.device.stream);",
    "    // 离线树/束上界复算的原始材料：selector 已把每步 16x16 前驱-后继分数矩阵算满，",
    "    // 而 walk 每个 step 只消费其中 1 行。门控 NINFER_DF2SCORES 时把全表与候选落盘，",
    "    // 供离线重算 Viterbi 最优路径分与逐深度 hit@b（需 --no-cuda-graph：抓图期禁止同步）。",
    "    if (std::getenv(\"NINFER_DF2SCORES\") != nullptr) {",
    "        static int df2scores_calls = 0;",
    "        if (df2scores_calls < 12) {",
    "            ++df2scores_calls;",
    "            const cudaStream_t dump_stream = state.execution.device.stream;",
    "            auto dump_one = [&](const char* tag, const Tensor& view) {",
    "                if (view.data == nullptr || view.numel() == 0) { return; }",
    "                std::vector<std::byte> host(view.bytes());",
    "                CUDA_CHECK(cudaMemcpyAsync(host.data(), view.data, host.size(),",
    "                                           cudaMemcpyDeviceToHost, dump_stream));",
    "                CUDA_CHECK(cudaStreamSynchronize(dump_stream));",
    "                char path[256];",
    "                std::snprintf(path, sizeof(path),",
    "                              \"/mnt/c/Users/User/Documents/ziqinzhang/dl/df2scores_%s_%d.bin\",",
    "                              tag, df2scores_calls);",
    "                if (std::FILE* fh = std::fopen(path, \"wb\")) {",
    "                    std::fwrite(host.data(), 1, host.size(), fh);",
    "                    std::fclose(fh);",
    "                }",
    "            };",
    "            dump_one(\"scores\", scores);",
    "            dump_one(\"cand\", candidates);",
    "            dump_one(\"unary\", unary);",
    "            std::fprintf(stderr, \"[df2scores] n=%d steps=%d batch=%d scores_bytes=%zu\\n\",",
    "                         df2scores_calls, steps, batch_size, scores.bytes());",
    "        }",
    "    }",
    "    state.execution.work.reset();",
    "",
])
p.write_bytes(raw.replace(anchor, block, 1).encode("utf-8"))

chk = p.read_bytes().decode("utf-8")
assert chk.count("NINFER_DF2SCORES") == 1
assert chk.count("df2scores_calls") == 3
assert chk.count("state.execution.work.reset();") == 1
print("OK 插桩已落（NINFER_DF2SCORES x1 / df2scores_calls x3 / work.reset 仍 x1）")
