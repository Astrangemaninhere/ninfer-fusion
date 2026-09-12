set -u
R=/home/user/ninfer-fusion
echo "=== LIVE md5 / sizes ==="
md5sum $R/src/ops/launcher/gqa_attention_decode.cu \
       $R/src/ops/launcher/gqa_attention_decode_partial.cuh \
       $R/src/ops/launcher/gqa_attention_decode_smallt.cu \
       $R/src/ops/launcher/gqa_attention_decode_g35.cu \
       $R/src/ops/launcher/gqa_attention_decode_muse.cu \
       $R/src/ops/launcher/gqa_attention_decode_impl.cuh 2>&1
wc -l $R/src/ops/launcher/gqa_attention_decode.cu \
      $R/src/ops/launcher/gqa_attention_decode_g35.cu \
      $R/src/ops/launcher/gqa_attention_decode_muse.cu \
      $R/src/ops/launcher/gqa_attention_decode_impl.cuh
echo "=== CR counts ==="
for f in gqa_attention_decode.cu gqa_attention_decode_partial.cuh gqa_attention_decode_smallt.cu gqa_attention_decode_g35.cu gqa_attention_decode_muse.cu gqa_attention_decode_impl.cuh; do
  printf "%s cr=%s\n" "$f" "$(grep -c $'\r' $R/src/ops/launcher/$f || true)"
done
echo "=== CMakeLists launcher section ==="
grep -n 'ops/launcher' $R/src/CMakeLists.txt | head -40
echo "=== compile_commands entries ==="
grep -o '[^"]*ops/launcher/[^"]*\.cu' $R/build/compile_commands.json 2>/dev/null | sort -u
echo "=== build objects for target files ==="
find $R/build -name '*gqa_attention_decode*' 2>/dev/null | head -40
echo "=== running compilers ==="
ps -eo pid,rss,etime,comm,args --sort=-rss 2>/dev/null | head -12
