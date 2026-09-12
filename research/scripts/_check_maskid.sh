#!/bin/bash
R=/home/user/ninfer-fusion
J=/mnt/c/Users/User/Documents/ziqinzhang
echo '=== ① 引擎侧 mask_token ==='
grep -n 'mask_token' $R/src/targets/qwen3_6_27b/impl/config.h | cut -c1-140
echo
echo '=== ② 训练脚本侧 MASK ==='
grep -n 'MASK_ID\|MASK_TOKEN_ID\|mask_token' $J/train_dflash2.py 2>/dev/null | head -6 | cut -c1-140
grep -n 'MASK_ID\|MASK_TOKEN_ID\|mask_token' $J/train_dspark.py 2>/dev/null | head -6 | cut -c1-140
echo
echo '=== ③ 草稿 config 侧（vLLM 用的那份）==='
grep -n 'mask_token_id' $J/data/draft_model/config.json 2>/dev/null | cut -c1-80
echo
echo '=== ④ 参照草稿（incoai/Qwen3.8-27B-DFlash2）侧 ==='
echo '  (前面已读: dflash_config.mask_token_id = 248070)'
echo
echo '=== ⑤ 产物里是否有 token 248077 / 190221 / 248070 的痕迹（draft_head 相关）==='
python3 - <<'PY'
import json, struct
for tag, P in (("dspark", "/home/user/models/qwen3_8_27b_nvfp4_dspark.ninfer"),
               ("dflash2", "/home/user/models/qwen3_8_27b_nvfp4_dflash2.ninfer")):
    with open(P, "rb") as f:
        f.read(8); n = struct.unpack("<Q", f.read(8))[0]
        raw = f.read(n).decode("utf-8", "replace")
    print("  [%s] 头部 JSON 里出现 248070: %s, 248077: %s, 190221: %s" % (
        tag, "248070" in raw, "248077" in raw, "190221" in raw))
PY
