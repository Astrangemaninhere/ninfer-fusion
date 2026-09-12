#!/bin/bash
R=/mnt/c/Users/User/Documents/ziqinzhang/ninfer-fusion-repo
J=/mnt/c/Users/User/Documents/ziqinzhang
cd "$R" || exit 3
echo "=== re-run the importer on Spark-X2.5-4B with the fixed detectors ==="
rm -rf "$R/tools/archkit/out/spark-x2.5-4b"
PYTHONPATH="$R" timeout 900 python3 tools/archkit/adapt.py \
  "$J/models/Spark-X2.5-4B" --model-id spark-x2.5-4b > /tmp/spark_adapt2.log 2>&1
echo "rc=$?"
cat /tmp/spark_adapt2.log | cut -c1-200
echo
echo "=== the gap list now (from the manifest) ==="
python3 - <<'PY'
import json, pathlib
m = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/ninfer-fusion-repo/tools/archkit/out/spark-x2.5-4b/manifest.json")
d = json.loads(m.read_text())
print("  model_id:", d.get("model_id"))
for g in d.get("gaps", []):
    print("  [%-7s] %-58s %s" % (g["tier"], g["need"][:58], g["action"][:60]))
PY
echo
echo "=== the spec now carries ==="
python3 - <<'PY'
import json, pathlib
s = json.loads(pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/ninfer-fusion-repo/tools/archkit/specs/spark-x2.5-4b_spec.json").read_text())
for k in ("rope_by_kind", "partial_rotary_by_kind", "hidden_act", "attention", "knobs"):
    print("  %-22s %s" % (k, json.dumps(s.get(k), ensure_ascii=False)[:150]))
PY
echo
echo "=== header state ==="
ls -l "$R/tools/archkit/out/spark-x2.5-4b/" 2>/dev/null | awk '{print "  ", $5, $NF}'
