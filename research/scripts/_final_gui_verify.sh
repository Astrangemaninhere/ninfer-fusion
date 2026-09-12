#!/bin/bash
G=/mnt/c/Users/User/Documents/ziqinzhang/ninfer-fusion-repo/tools/gui
J=/mnt/c/Users/User/Documents/ziqinzhang
cd "$G" || exit 3
echo "=== [1] 全量 i18n 检查器（我自己跑） ==="
python3 gui_i18n_check.py 2>&1 | tail -6
echo
echo "=== [2] serve_gui 自检（我自己跑） ==="
timeout 600 python3 serve_gui_selftest.py 2>&1 | tail -12
echo
echo "=== [3] 五个模块双语接线是否都在（关键调用点） ==="
for f in serve_gui.py convert_gui.py rag_gui.py model_import.py gui_tips.py; do
  n=$(grep -c "from gui_i18n import\|gui_i18n" "$f" 2>/dev/null)
  langapi=$(grep -c 'api/lang' "$f" 2>/dev/null)
  printf "  %-18s gui_i18n 引用=%s  /api/lang=%s\n" "$f" "$n" "$langapi"
done
echo
echo "=== [4] 编译 / 测量 / 下载 ==="
grep -oE '^\[[ 0-9]+%\] (Building|Linking)[^"]*' /tmp/pa_make_1.log | tail -2
for p in $(pgrep -f bin/nvcc); do echo "  TU 已编 $(ps -o etime= -p $p | tr -d ' ')"; break; done
tail -1 "$J/dl/postbuild_measure.log" | cut -c1-100
du -sh "$J/models/Spark-X2.5-4B" 2>/dev/null
