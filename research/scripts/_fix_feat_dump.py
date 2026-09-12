#!/usr/bin/env python3
"""修 dump：设备指针要先 D2H 拷贝再写文件。"""
import pathlib, hashlib

F = pathlib.Path("/home/user/ninfer-fusion/src/targets/qwen3_6/impl/runtime/dflash2_impl.h")
t = F.read_text()
old = """            auto dump_tensor = [](const char* path, const Tensor& view) {
                std::FILE* fh = std::fopen(path, "wb");
                if (fh != nullptr) {
                    std::fwrite(view.data, 1, view.bytes(), fh);
                    std::fclose(fh);
                }
            };"""
new = """            // The tensors live on the device: stage through host memory (same idiom as
            // the KV dump helper) before writing.
            auto dump_tensor = [](cudaStream_t stream, const char* path, const Tensor& view) {
                if (view.data == nullptr || view.numel() == 0) { return; }
                std::vector<std::byte> host(view.bytes());
                CUDA_CHECK(cudaMemcpyAsync(host.data(), view.data, host.size(),
                                           cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));
                std::FILE* fh = std::fopen(path, "wb");
                if (fh != nullptr) {
                    std::fwrite(host.data(), 1, host.size(), fh);
                    std::fclose(fh);
                }
            };"""
assert t.count(old) == 1, "anchor count = %d" % t.count(old)
t = t.replace(old, new)

# 三处调用同步加 stream 参数
for name in ("feat_features.bin", "feat_projected.bin", "feat_context.bin"):
    pass
t = t.replace("dump_tensor(\"/mnt/c/Users/User/Documents/ziqinzhang/dl/feat_features.bin\", fv);",
              "dump_tensor(state.execution.device.stream,\n                        \"/mnt/c/Users/User/Documents/ziqinzhang/dl/feat_features.bin\", fv);")
t = t.replace("dump_tensor(\"/mnt/c/Users/User/Documents/ziqinzhang/dl/feat_projected.bin\",",
              "dump_tensor(state.execution.device.stream,\n                        \"/mnt/c/Users/User/Documents/ziqinzhang/dl/feat_projected.bin\",")
t = t.replace("dump_tensor(\"/mnt/c/Users/User/Documents/ziqinzhang/dl/feat_context.bin\", context_full);",
              "dump_tensor(state.execution.device.stream,\n                        \"/mnt/c/Users/User/Documents/ziqinzhang/dl/feat_context.bin\", context_full);")
F.write_text(t)
print("patched, md5", hashlib.md5(F.read_bytes()).hexdigest()[:12])
print("含 cudaMemcpyAsync:", "cudaMemcpyAsync" in F.read_text())
