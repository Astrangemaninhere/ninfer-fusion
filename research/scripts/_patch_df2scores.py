#!/usr/bin/env python3
"""MVP-1 插桩：NINFER_DF2SCORES 门控下把 16x16 scores 全表 + candidates + unary 落盘。

纯新增：不改任何既有行为；仅在 selector 调用之后、work.reset() 之前插入。
二进制读写以保留 CRLF；锚点断言恰好命中 1 次。
"""
import pathlib
import re
import sys

R = pathlib.Path("/home/user/ninfer-fusion")
rel = "src/targets/qwen3_6/impl/runtime/dflash2_impl.h"
p = R / rel

BAK = pathlib.Path("/home/user/df2scores_bak") / rel
BAK.parent.mkdir(parents=True, exist_ok=True)
BAK.write_bytes(p.read_bytes())
print("已备份 ->", BAK)

src = p.read_bytes().decode("utf-8")

anchor = ("                          head_domain, head_token_ids, state.execution.device.stream);\r\n"
          "    state.execution.work.reset();\r\n")
if src.count(anchor) != 1:
    print(f"FAIL 锚点命中 {src.count(anchor)} 次"); sys.exit(3)

block = (
    "                          head_domain, head_token_ids, state.execution.device.stream);\r\n"
    "    // 离线树/束上界复算所需的原始材料：selector 已把每步 16x16 前驱-后继分数矩阵算满，\r\n"
    "    // 但 walk 每个 step 只消费其中 1 行。门控 NINFER_DF2SCORES 时把全表与候选落盘，\r\n"
    "    // 供离线重算 Viterbi 最优路径分与逐深度 hit@b（需 --no-cuda-graph：抓图期禁止同步）。\r\n"
    "    if (std::getenv(\"NINFER_DF2SCORES\") != nullptr) {\r\n"
    "        static int df2scores_calls = 0;\r\n"
    "        if (df2scores_calls < 12) {\r\n"
    "            ++df2scores_calls;\r\n"
    "            const cudaStream_t dump_stream = state.execution.device.stream;\r\n"
    "            auto dump_one = [&](const char* tag, const Tensor& view) {\r\n"
    "                if (view.data == nullptr || view.numel() == 0) { return; }\r\n"
    "                std::vector<std::byte> host(view.bytes());\r\n"
    "                CUDA_CHECK(cudaMemcpyAsync(host.data(), view.data, host.size(),\r\n"
    "                                           cudaMemcpyDeviceToHost, dump_stream));\r\n"
    "                CUDA_CHECK(cudaStreamSynchronize(dump_stream));\r\n"
    "                char path[256];\r\n"
    "                std::snprintf(path, sizeof(path),\r\n"
    "                              \"/mnt/c/Users/User/Documents/ziqinzhang/dl/df2scores_%s_%d.bin\",\r\n"
    "                              tag, df2scores_calls);\r\n"
    "                if (std::FILE* fh = std::fopen(path, \"wb\")) {\r\n"
    "                    std::fwrite(host.data(), 1, host.size(), fh);\r\n"
    "                    std::fclose(fh);\r\n"
    "                }\r\n"
    "            };\r\n"
    "            dump_one(\"scores\", scores);\r\n"
    "            dump_one(\"cand\", candidates);\r\n"
    "            dump_one(\"unary\", unary);\r\n"
    "            std::fprintf(stderr, \"[df2scores] n=%d steps=%d batch=%d scores_bytes=%zu\\n\",\r\n"
    "                         df2scores_calls, steps, batch_size, scores.bytes());\r\n"
    "        }\r\n"
    "    }\r\n"
    "    state.execution.work.reset();\r\n"
)
out = src.replace(anchor, block, 1)
if '\\"' in out.replace('\\"wb\\"', '').replace('\\"scores\\"', '').replace('\\"cand\\"', '').replace('\\"unary\\"', '') :
    pass  # 正常：C++ 源码里本就有 \" 转义引号
p.write_bytes(out.encode("utf-8"))

chk = p.read_bytes().decode()
assert chk.count("NINFER_DF2SCORES") == 1
assert chk.count("df2scores_calls") == 3
print("OK 插桩已落（NINFER_DF2SCORES x1, df2scores_calls x3）")
