#!/bin/bash
# U2 final probe: profiles dspark timeline + protocol facts
set -u
R=/home/user/ninfer-fusion
cd "$R" || exit 1

echo "### dspark acceptance rate in profiles/ (by file mtime)"
mapfile -t FILES < <(grep -rl 'dflash acceptance rate' profiles/ 2>/dev/null | head -60)
if [ "${#FILES[@]}" -eq 0 ]; then echo "  (none)"; fi
for f in "${FILES[@]}"; do
  r=$(grep -m1 'dflash acceptance rate' "$f" | sed -E 's/.*rate[[:space:]]*//')
  p=$(grep -m1 'dflash accepted by pos' "$f" | sed -E 's/.*pos[[:space:]]*//')
  mt=$(stat -c '%y' "$f" | cut -c1-19)
  printf '  %s  %-9s pos=[%s]  %s\n' "$mt" "$r" "$p" "$f"
done | sort | tail -25

echo
echo "### how many profile files carry dspark acceptance at all"
grep -rl 'dflash acceptance rate' profiles/ 2>/dev/null | wc -l

echo
echo "### M_accept_eval.md"
cat /mnt/c/Users/User/Documents/ziqinzhang/_collab/M_accept_eval.md 2>/dev/null | head -32

echo
echo "### files in /home/user root that mention cmp_dsp_d7"
grep -ln 'cmp_dsp' /home/user/*.sh /home/user/*.md /home/user/*.txt 2>/dev/null | head
grep -ln 'cmp_dsp' /mnt/c/Users/User/Documents/ziqinzhang/*.sh /mnt/c/Users/User/Documents/ziqinzhang/*.md 2>/dev/null | head

echo
echo "### _collab/build/backup + staged listing (are there pre-A5b snapshots?)"
ls -la /mnt/c/Users/User/Documents/ziqinzhang/_collab/build/backup/ 2>/dev/null | head -20
