#!/usr/bin/env bash
# S6 final acceptance (read-only): staged artifacts reproduce, evidence passes, live tree untouched
set -u
J=/mnt/c/Users/User/Documents/ziqinzhang
S=$J/_collab/build/staged2
T=$J/_collab/build/tmp2

echo "############ A. live tree untouched (must equal the pre-split md5s) ############"
cd /home/user/ninfer-fusion || exit 1
md5sum src/ops/launcher/gqa_attention_decode_e8.cu src/ops/launcher/gqa_attention_prefill.cu \
       src/ops/launcher/gqa_attention_prefill_e8.cu src/CMakeLists.txt
git -C /home/user/ninfer-fusion status --porcelain 2>/dev/null | head -5 || echo "(not a git repo)"
echo "no S6 files may exist in the tree:"
ls src/ops/launcher/ | grep -E '_(arms|batch|kv_append|append_g27|cached_g27|append_muse35|cached_muse35)' || echo "  (none) OK"

echo
echo "############ B. generator reproducibility (--check, no writes) ############"
tr -d '\r' < "$T/_s6_split.py" > /tmp/s6_split_check.py
python3 /tmp/s6_split_check.py --check 2>&1 | tail -8

echo
echo "############ C. evidence transcript (must end with ALL CHECKS PASSED) ############"
tr -d '\r' < "$T/_s6_evidence.py" > /tmp/s6_evidence2.py
python3 /tmp/s6_evidence2.py > "$T/s6_evidence_out.txt" 2>&1
echo "rc=$?"
tail -3 "$T/s6_evidence_out.txt"
grep -c IDENTICAL "$T/s6_evidence_out.txt"

echo
echo "############ D. staged2 inventory ############"
ls -l --time-style=+%H:%M "$S"
md5sum "$S"/* | sed 's|.*/staged2/||'
echo
echo "############ E. deliverables ############"
ls -l "$J/_collab/build/S6_e8_prefill_split.md" "$J/_land_s6.sh"
head -1 "$J/_collab/build/S6_e8_prefill_split.md"
file "$J/_collab/build/S6_e8_prefill_split.md"
