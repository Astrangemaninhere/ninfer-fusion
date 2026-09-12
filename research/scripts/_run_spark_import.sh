#!/bin/bash
R=/mnt/c/Users/User/Documents/ziqinzhang/ninfer-fusion-repo
J=/mnt/c/Users/User/Documents/ziqinzhang
cd "$R" || exit 3
echo "=== adapt.py remaining CLI ==="
sed -n '289,320p' tools/archkit/adapt.py | cut -c1-140
echo
echo "=== run the auto-import pipeline on Spark-X2.5-4B ==="
PYTHONPATH="$R" timeout 900 python3 tools/archkit/adapt.py \
  "$J/models/Spark-X2.5-4B" --model-id spark-x2.5-4b > /tmp/spark_adapt.log 2>&1
echo "rc=$?"
tail -60 /tmp/spark_adapt.log | cut -c1-160
echo
echo "=== outputs produced ==="
ls -lt --time-style=+%H:%M "$R/tools/archkit/out/" 2>/dev/null | head -6 | awk '{print "  ", $6, $5, $NF}'
ls -lt --time-style=+%H:%M "$R/tools/archkit/specs/" 2>/dev/null | head -4 | awk '{print "  ", $6, $5, $NF}'
