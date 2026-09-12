#!/usr/bin/env python3
"""给 dflash2 选择器探针补一个"本步 top-K 候选全貌"的 dump，供逐步归因。"""
import pathlib, shutil, hashlib

F = pathlib.Path("/home/user/ninfer-fusion/src/ops/kernel/dflash2_selector.cuh")
BAK = pathlib.Path("/home/user/df2cand_bak"); BAK.mkdir(exist_ok=True)
shutil.copy2(F, BAK / F.name)
print("backup ->", BAK / F.name, "md5", hashlib.md5(F.read_bytes()).hexdigest()[:12])

t = F.read_text()
anchor = """                   e_arg, u_arg);
        }
"""
add = """                   e_arg, u_arg);
            // 逐步取证（NINFER_DF2SEL 门控）：把本步的 top-K 候选与其 unary 全量打印，
            // 供离线按列归因：正确 token 是否在候选里（头的问题）/ 在却选错（walk 的问题）。
            printf("[df2cand] s=%d pred=%d chosen=%d k=%d", s, pred, chosen, (int)K);
            for (int c = 0; c < K; ++c) {
                printf(" %d:%.2f",
                       candidates[dflash2_selector_candidate_offset(b, batch, s, steps, c)],
                       unary[dflash2_selector_candidate_offset(b, batch, s, steps, c)]);
            }
            printf("\\n");
        }
"""
c = t.count(anchor)
assert c == 1, "anchor count = %d" % c
F.write_text(t.replace(anchor, add))
print("patched, md5 now", hashlib.md5(F.read_bytes()).hexdigest()[:12])
print("df2cand 出现次数 =", F.read_text().count("[df2cand]"))
