#!/usr/bin/env python3
"""把特征 dump 改成"前 8 次调用都打摘要（各 tap 段的范数）"，并只在非零时才落盘。"""
import pathlib, hashlib

F = pathlib.Path("/home/user/ninfer-fusion/src/targets/qwen3_6/impl/runtime/dflash2_impl.h")
t = F.read_text()
old = """    if (std::getenv("NINFER_DF2FEAT") != nullptr) {
        static bool df2feat_dumped = false;
        if (!df2feat_dumped) {
            df2feat_dumped = true;"""
new = """    if (std::getenv("NINFER_DF2FEAT") != nullptr) {
        static int df2feat_calls = 0;
        if (df2feat_calls < 8) {
            ++df2feat_calls;
            const bool df2feat_dumped = false;
            (void)df2feat_dumped;"""
assert t.count(old) == 1, "anchor A count %d" % t.count(old)
t = t.replace(old, new)

# 落盘条件：只在前两次调用时写（避免重复覆盖），并在摘要里带调用序号
old2 = """            dump_tensor(state.execution.device.stream,
                        "/mnt/c/Users/User/Documents/ziqinzhang/dl/feat_features.bin", fv);"""
new2 = """            if (df2feat_calls <= 2) {
                dump_tensor(state.execution.device.stream,
                            "/mnt/c/Users/User/Documents/ziqinzhang/dl/feat_features.bin", fv);
            }"""
assert t.count(old2) == 1, "anchor B count %d" % t.count(old2)
t = t.replace(old2, new2)

# 摘要：加调用序号 + 各 tap 段范数（用 host 拷贝算）
old3 = """            std::fprintf(stderr, "[df2feat] features[%d,%d] projected[%d,%d] context[%d,%d]\\n",
                         fv.ne[0], fv.ne[1], context_roots.projected.ne[0],
                         context_roots.projected.ne[1], context_full.ne[0], context_full.ne[1]);"""
new3 = """            {
                std::vector<std::byte> hf(fv.bytes());
                CUDA_CHECK(cudaMemcpyAsync(hf.data(), fv.data, hf.size(),
                                           cudaMemcpyDeviceToHost, state.execution.device.stream));
                CUDA_CHECK(cudaStreamSynchronize(state.execution.device.stream));
                const auto* hp = reinterpret_cast<const std::uint16_t*>(hf.data());
                const std::size_t stride = static_cast<std::size_t>(fv.nb[1]) / sizeof(std::uint16_t);
                const std::size_t rows = static_cast<std::size_t>(fv.ne[0]);
                std::fprintf(stderr, "[df2feat] call=%d features[%d,%d] projected[%d,%d] context[%d,%d]",
                             df2feat_calls, fv.ne[0], fv.ne[1], context_roots.projected.ne[0],
                             context_roots.projected.ne[1], context_full.ne[0], context_full.ne[1]);
                for (int tap = 0; tap < 5; ++tap) {
                    double norm = 0.0;
                    const std::size_t base = static_cast<std::size_t>(tap) * 5120;
                    for (std::size_t r = 0; r < 5120; ++r) {
                        for (std::int32_t c = 0; c < fv.ne[1]; ++c) {
                            const float v = __bfloat162float(
                                *reinterpret_cast<const __nv_bfloat16*>(hp + (base + r) * stride + c));
                            norm += static_cast<double>(v) * v;
                        }
                    }
                    std::fprintf(stderr, " tap%d=%.4f", tap, std::sqrt(norm));
                }
                std::fprintf(stderr, "\\n");
            }"""
assert t.count(old3) == 1, "anchor C count %d" % t.count(old3)
t = t.replace(old3, new3)
F.write_text(t)
print("patched, md5", hashlib.md5(F.read_bytes()).hexdigest()[:12])
print("含 tap0 打印:", "tap%d=%.4f" in F.read_text())
