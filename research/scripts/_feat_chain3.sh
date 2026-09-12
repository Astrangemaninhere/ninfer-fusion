#!/bin/bash
# 还原 → 干净探针 → 编译 → 跑 → 离线分析（一条链）
set -u
R=/home/user/ninfer-fusion
J=/mnt/c/Users/User/Documents/ziqinzhang
export PATH=/home/user/.local/bin:$PATH
LOG=$J/dl/feat3.log
exec > >(tee -a "$LOG") 2>&1
echo "=== 还原 + 探针 $(date '+%H:%M:%S') ==="
cp -f /home/user/df2feat_bak/dflash2_impl.h $R/src/targets/qwen3_6/impl/runtime/dflash2_impl.h
echo "  restored md5=$(md5sum $R/src/targets/qwen3_6/impl/runtime/dflash2_impl.h | cut -c1-12) (期望 b83ac2a12756)"

python3 - <<'PYEOF'
import pathlib
F = pathlib.Path("/home/user/ninfer-fusion/src/targets/qwen3_6/impl/runtime/dflash2_impl.h")
t = F.read_text()
if "#include <cstdio>" not in t:
    lines = t.splitlines(); idx = 0
    for i, ln in enumerate(lines):
        if ln.startswith("#include"): idx = i + 1
    lines[idx:idx] = ["#include <cstdio>", "#include <cstdlib>", "#include <vector>"]
    t = "\n".join(lines) + "\n"
anchor = """    ops::rmsnorm(context_roots.projected, dflash2.context_norm, Config::rms_epsilon, false,
                 context_full, state.execution.device.stream);
"""
add = anchor + """    // Env-gated (NINFER_DF2FEAT): dump the feature chain for the first few calls, one
    // file per call, so the offline check can tell warmup snapshots from real ones.
    if (std::getenv("NINFER_DF2FEAT") != nullptr) {
        static int df2feat_calls = 0;
        if (df2feat_calls < 8) {
            ++df2feat_calls;
            Tensor fv = features.view({Config::feature_rows, columns});
            const cudaStream_t s = state.execution.device.stream;
            auto dump_one = [&](const char* tag, const Tensor& view) {
                if (view.data == nullptr || view.numel() == 0) { return; }
                std::vector<std::byte> host(view.bytes());
                CUDA_CHECK(cudaMemcpyAsync(host.data(), view.data, host.size(),
                                           cudaMemcpyDeviceToHost, s));
                CUDA_CHECK(cudaStreamSynchronize(s));
                char path[256];
                std::snprintf(path, sizeof(path),
                              "/mnt/c/Users/User/Documents/ziqinzhang/dl/feat_%s_%d.bin", tag,
                              df2feat_calls);
                if (std::FILE* fh = std::fopen(path, "wb")) {
                    std::fwrite(host.data(), 1, host.size(), fh);
                    std::fclose(fh);
                }
            };
            dump_one("features", fv);
            dump_one("projected", context_roots.projected);
            dump_one("context", context_full);
            std::fprintf(stderr, "[df2feat] call=%d features[%d,%d]\\n", df2feat_calls, fv.ne[0],
                         fv.ne[1]);
        }
    }
"""
assert t.count(anchor) == 1, "anchor %d" % t.count(anchor)
F.write_text(t.replace(anchor, add))
print("patched md5", __import__("hashlib").md5(F.read_bytes()).hexdigest()[:12])
PYEOF

echo "--- 编译 ---"
touch $R/src/targets/qwen3_6_27b/impl/variant.cpp $R/src/targets/muse_glimmer_30b/impl/variant.cpp
cd "$R/build" || exit 4
make ninfer -j3 2>&1 | tail -4
rc=${PIPESTATUS[0]}; echo "make rc=$rc"
[ "$rc" -ne 0 ] && { echo BUILD_FAIL; exit 5; }

echo "--- 跑 ---"
rm -f $J/dl/feat_features_*.bin $J/dl/feat_projected_*.bin $J/dl/feat_context_*.bin
P='请用中文写一段两百字左右的短文，介绍西湖一年四季的景色变化，要求语句连贯、不要列条目。'
NINFER_DF2FEAT=1 timeout 900 ./apps/ninfer /home/user/models/qwen3_8_27b_nvfp4_dflash2.ninfer \
  --prompt "$P" --max-new 16 --max-context 4096 --no-thinking --greedy --spec dflash2 \
  > $J/dl/feat_stdout3.log 2>&1
echo "  rc=$?"
grep -E 'df2feat' $J/dl/feat_stdout3.log | head -9
ls -1 $J/dl/feat_features_*.bin 2>/dev/null | wc -l

echo "--- 离线分析（各次调用、各 tap 段范数） ---"
/home/user/vllm029/bin/python - <<'PYEOF'
import numpy as np, pathlib
J = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/dl")
for call in range(1, 9):
    f = J / ("feat_features_%d.bin" % call)
    if not f.exists() or f.stat().st_size == 0:
        continue
    v = np.fromfile(f, dtype=np.float16).astype(np.float32)
    cols = v.size // 25600
    v = v.reshape(25600, cols)
    norms = [float(np.linalg.norm(v[i*5120:(i+1)*5120])) for i in range(5)]
    print("  call=%d cols=%d  tap norms = %s  total=%.4f" %
          (call, cols, ["%.3f" % x for x in norms], float(np.linalg.norm(v))))
PYEOF
echo FEAT3_DONE
