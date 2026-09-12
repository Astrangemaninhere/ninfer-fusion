#!/bin/bash
echo "=== CLI 三臂的完整指标（含 rounds / acceptance rate） ==="
for f in /home/user/s4wpos_cli_mtp3.log /home/user/s4wpos_cli_dspark.log /home/user/s4wpos_cli_dflash2.log; do
  echo "--- $(basename $f) ---"
  grep -iE 'acceptance|accepted by pos|tok/round|round|draft|decode' "$f" 2>/dev/null | head -12 | cut -c1-140
done
echo
echo "=== 由直方图推"第 1 个位置接受率"：p(pos_i | pos_{i-1}) ==="
python3 - <<'PY'
hists = {"mtp3": [23, 12, 8], "dflash": [15, 5, 2], "dflash2": [23, 11, 3, 1]}
for name, h in hists.items():
    print("  %-8s counts=%s" % (name, h))
    for i in range(1, len(h)):
        prev, cur = h[i - 1], h[i]
        print("      p(pos%d | pos%d) = %d/%d = %.1f%%" % (i, i - 1, cur, prev, 100.0 * cur / prev))
PY
