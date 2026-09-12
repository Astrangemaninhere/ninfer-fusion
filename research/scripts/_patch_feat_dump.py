#!/usr/bin/env python3
"""给 dflash2 特征链装仪表：env 门控，一次性 dump features / projected / context_full。"""
import pathlib, shutil, hashlib

F = pathlib.Path("/home/user/ninfer-fusion/src/targets/qwen3_6/impl/runtime/dflash2_impl.h")
BAK = pathlib.Path("/home/user/df2feat_bak"); BAK.mkdir(exist_ok=True)
shutil.copy2(F, BAK / F.name)
print("backup ->", BAK / F.name, "md5", hashlib.md5(F.read_bytes()).hexdigest()[:12])

t = F.read_text()
anchor = """    ops::rmsnorm(context_roots.projected, dflash2.context_norm, Config::rms_epsilon, false,
                 context_full, state.execution.device.stream);
"""
add = anchor + """    // Env-gated (NINFER_DF2FEAT): dump the draft feature chain once, for offline verification.
    if (std::getenv("NINFER_DF2FEAT") != nullptr) {
        static bool df2feat_dumped = false;
        if (!df2feat_dumped) {
            df2feat_dumped = true;
            auto dump_tensor = [](const char* path, const Tensor& view) {
                std::FILE* fh = std::fopen(path, "wb");
                if (fh != nullptr) {
                    std::fwrite(view.data, 1, view.bytes(), fh);
                    std::fclose(fh);
                }
            };
            Tensor fv = features.view({Config::feature_rows, columns});
            dump_tensor("/mnt/c/Users/User/Documents/ziqinzhang/dl/feat_features.bin", fv);
            dump_tensor("/mnt/c/Users/User/Documents/ziqinzhang/dl/feat_projected.bin",
                        context_roots.projected);
            dump_tensor("/mnt/c/Users/User/Documents/ziqinzhang/dl/feat_context.bin", context_full);
            std::fprintf(stderr, "[df2feat] features[%d,%d] projected[%d,%d] context[%d,%d]\\n",
                         fv.ne[0], fv.ne[1], context_roots.projected.ne[0],
                         context_roots.projected.ne[1], context_full.ne[0], context_full.ne[1]);
        }
    }
"""
c = t.count(anchor)
assert c == 1, "anchor count = %d" % c
F.write_text(t.replace(anchor, add))
print("patched, md5 now", hashlib.md5(F.read_bytes()).hexdigest()[:12])
print("df2feat 出现次数 =", F.read_text().count("[df2feat]"))
# 需要 <cstdio>/<cstdlib>
txt = F.read_text()
print("已有 cstdio?", "#include <cstdio>" in txt, " cstdlib?", "#include <cstdlib>" in txt)
