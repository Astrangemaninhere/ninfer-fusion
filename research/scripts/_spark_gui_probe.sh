#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
F=$J/ninfer-fusion-repo
echo "=== local hints for 'spark' as a MODEL name ==="
grep -rniE 'spark' "$J/dl/hf-download.log" "$J/hf-config.json" "$J/_hf_chunked.py" "$J/_hf_download.py" 2>/dev/null | head -8 | cut -c1-150
grep -rniE '"spark|spark-' "$J/models"/*/config.json 2>/dev/null | head -5
ls -d "$J/models"/*[Ss]park* 2>/dev/null | head
echo
echo "=== GUI: files + how it is launched + translation hooks ==="
ls -l "$F/tools/gui/" 2>/dev/null | awk '{print "  ", $5, $NF}' | head -12
echo "--- translate.cpp/h: what do they translate? ---"
head -30 "$F/src/serve/translate.h" 2>/dev/null | cut -c1-120
echo "--- gui strings / i18n tables in the GUI ---"
grep -rniE 'translat|i18n|locale|zh_|en_|LANG|语言' "$F/tools/gui"/*.py 2>/dev/null | head -12 | cut -c1-140
echo
echo "=== the importer entry + its recent state ==="
ls -l "$F/tools/archkit/adapt.py" 2>/dev/null | awk '{print $5, $NF}'
sed -n '1,25p' "$F/tools/archkit/_AUTOADAPT.md" 2>/dev/null | cut -c1-120
