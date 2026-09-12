#!/bin/bash
W=/mnt/c/Users/User/Documents/ziqinzhang
echo "=== live-pipeline files still at the root? (must all be present) ==="
for f in _post_build_measure.sh _window_k3.sh _patchA_build.sh _spec_4way.sh \
         _muse_serve_accept.sh _train_df2_resume.bat _apply_patchA.py _hf_chunked.py \
         _df2_shift3_probe.py _needle_check.sh; do
  if [ -f "$W/$f" ]; then echo "  OK      $f"; else echo "  MISSING $f  <-- PROBLEM"; fi
done
echo
echo "=== root-level counts now ==="
echo "  _*.sh : $(ls $W/_*.sh 2>/dev/null | wc -l)   _*.py : $(ls $W/_*.py 2>/dev/null | wc -l)   _*.bat: $(ls $W/_*.bat 2>/dev/null | wc -l)"
echo "  _scratch dirs: $(ls -d $W/_scratch/*/ 2>/dev/null | wc -l)  moved files: $(find $W/_scratch -type f | wc -l)"
ls -d $W/_scratch/*/ 2>/dev/null | head -6 | sed 's/^/    /'
echo
echo "=== index head ==="
head -14 "$W/_scratch/README.md"
echo
echo "=== build / pipeline still healthy? ==="
grep -oE '^\[[ 0-9]+%\][^"]*' /tmp/pa_make_1.log 2>/dev/null | tail -2
for p in $(pgrep -f bin/nvcc); do echo "  nvcc $(ps -o etime= -p $p | tr -d ' ')"; break; done
pgrep -af '_window_k3|_post_build_measure' | cut -c1-70
tail -1 "$W/dl/postbuild_measure.log" 2>/dev/null | cut -c1-100
