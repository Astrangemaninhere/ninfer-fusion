#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
echo "=== 起编译（含 PATH 修复与重试） ==="
setsid nohup bash "$J/_rebuild_after_s35.sh" > /dev/null 2>&1 < /dev/null &
sleep 20
tail -4 "$J/dl/rebuild_after_s35.log" | cut -c1-140
for p in $(pgrep -f 'bin/nvcc|cc1plus'); do echo "  在编 $(ps -o etime= -p $p | tr -d ' ')"; break; done
echo
echo "=== 顺便：写 dspark 验证脚本（等编译完就跑） ==="
cat > "$J/_dspark_posverify.sh" <<'EOS'
#!/bin/bash
# 验证 dspark 的 anchor 列错位修复：跑 CLI 的 dspark 位置剖面 + 接受率，与修复前对照。
# 修复前实测：p(pos0)=15/56=26.8%、接受率 11.22%、1.39 tok/round。
set -u
J=/mnt/c/Users/User/Documents/ziqinzhang
R=/home/user/ninfer-fusion
CLI=$R/build/apps/ninfer
MODEL=${MODEL:-/home/user/models/qwen3_8_27b_nvfp4_dspark.ninfer}
LOG=$J/dl/dspark_posverify.log
P='请用中文写一段两百字左右的短文，介绍西湖一年四季的景色变化，要求语句连贯、不要列条目。'
if pgrep -f 'train_dflash2' >/dev/null 2>&1; then echo "training running" | tee -a "$LOG"; exit 3; fi
: > "$LOG"
echo "=== dspark 位置剖面（修复后） $(date '+%F %H:%M:%S') ===" | tee -a "$LOG"
( cd $R/build && timeout 900 "$CLI" "$MODEL" --prompt "$P" --max-new 96 --max-context 4096 \
    --no-thinking --greedy --spec dflash --draft-tokens 7 > "$LOG.raw" 2>&1 )
echo "  rc=$?"
grep -iE 'acceptance|accepted by pos|tok/round|rounds|drafted' "$LOG.raw" | head -10 | tee -a "$LOG"
EOS
bash -n "$J/_dspark_posverify.sh" && echo "  验证脚本 SYNTAX_OK"
