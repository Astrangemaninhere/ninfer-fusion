#!/bin/bash
# U2: sweep all logs for dflash/dflash2/mtp acceptance + binary identity
set -u
echo "### ALL dspark(dflash) acceptance measurements in /home/user/*.log"
echo
for f in /home/user/*.log; do
  [ -f "$f" ] || continue
  r=$(grep -m1 'dflash acceptance rate' "$f" 2>/dev/null | sed -E 's/.*rate[[:space:]]+//')
  if [ -n "${r:-}" ]; then
    pos=$(grep -m1 'dflash accepted by pos' "$f" 2>/dev/null | sed -E 's/.*pos[[:space:]]+//')
    w=$(grep -m1 'dflash draft window' "$f" 2>/dev/null | sed -E 's/.*window[[:space:]]+//')
    mt=$(stat -c '%y' "$f" 2>/dev/null | cut -c1-19)
    wt=$(grep -m1 'weights ' "$f" 2>/dev/null | sed -E 's/.*weights[[:space:]]+//' | awk '{print $1}')
    printf '%-28s %-19s w=%-3s %-8s pos=%-16s %s\n' "$(basename $f)" "$mt" "${w:--}" "$r" "${pos:--}" "${wt:-}"
  fi
done | sort -k2
echo
echo "### ALL mtp acceptance measurements"
for f in /home/user/*.log; do
  [ -f "$f" ] || continue
  r=$(grep -m1 'mtp acceptance rate' "$f" 2>/dev/null | sed -E 's/.*rate[[:space:]]+//')
  if [ -n "${r:-}" ]; then
    pos=$(grep -m1 'mtp accepted by pos' "$f" 2>/dev/null | sed -E 's/.*pos[[:space:]]+//')
    mt=$(stat -c '%y' "$f" 2>/dev/null | cut -c1-19)
    printf '%-28s %-19s %-8s pos=%s\n' "$(basename $f)" "$mt" "$r" "${pos:--}"
  fi
done | sort -k2
