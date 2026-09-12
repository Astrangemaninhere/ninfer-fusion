#!/usr/bin/env python3
"""B：应用 S51（上游 PR #194 思路的 SiLU 修复）；C：verify 补 logit policy。"""
import pathlib
import re
import subprocess

R = pathlib.Path("/home/user/ninfer-fusion")

# ---- B: S51 ----
diff = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/_collab/E8_s51_nvfp4_silu.diff")
chk = subprocess.run(["patch", "-p1", "--dry-run", "--forward", "-f"], cwd=str(R),
                     stdin=diff.open("rb"), capture_output=True)
if chk.returncode == 0:
    out = subprocess.run(["patch", "-p1", "--forward", "-f"], cwd=str(R),
                         stdin=diff.open("rb"), capture_output=True)
    print("B: S51 " + ("已应用" if out.returncode == 0 else "失败: " + out.stderr.decode()[:200]))
else:
    print("B: S51 dry-run rc=%d（多为已应用）: %s" % (chk.returncode, chk.stderr.decode()[:120]))

# ---- C: verify 补 policy ----
T = R / "src/targets/qwen3_6/impl/runtime/text_context_impl.h"
tsrc = T.read_text(encoding="utf-8")
if "apply_final_logit_policy(flat_logits" in tsrc:
    print("C: 已在")
else:
    m = re.search(r"( *)ops::linear\(flat_hidden, \*lm_head_, flat_logits, stream\);\n", tsrc)
    assert m, "C 锚点未找到"
    ind = m.group(1)
    ins = (m.group(0) +
           ind + "// Contract (TextContext::apply_final_logit_policy): every lm_head logits\n" +
           ind + "// production site applies the architecture's final-logit policy. Verify was\n" +
           ind + "// the one site that did not: a compile-time no-op for the qwen family\n" +
           ind + "// (softcap 0, multiplier 1) but REQUIRED for Muse (softcap 20, multiplier\n" +
           ind + "// 0.196), where omitting it made the verifier's argmax differ from plain.\n" +
           ind + "kCfg.apply_final_logit_policy(flat_logits, stream);\n")
    tsrc = tsrc[:m.start()] + ins + tsrc[m.end():]
    T.write_text(tsrc, encoding="utf-8")
    print("C: verify 路径已补 policy")
print("done")
