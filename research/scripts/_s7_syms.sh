set -u
B=/home/user/ninfer-fusion/build/src/CMakeFiles/ninfer_ops.dir/ops/launcher
inv() {
  o=$1; tag=$2
  echo "########## $tag : $o ##########"
  nm -C "$o" 2>/dev/null | awk '$2=="T"||$2=="t"||$2=="W"||$2=="w"{print}' > /tmp/raw_$tag.txt
  wc -l < /tmp/raw_$tag.txt | sed 's/^/raw T/t symbols: /'
  # device stubs = one per kernel instantiation (host side)
  grep -c '__device_stub__' /tmp/raw_$tag.txt | sed 's/^/device stubs: /'
  # unique kernels: strip prefix/suffix, take name up to '<'
  sed 's/^[0-9a-f]* [A-Za-z] //' /tmp/raw_$tag.txt | sed 's/__device_stub__//;s/ \[clone.*//' | grep -o '[A-Za-z_0-9]*kernel' | sort | uniq -c | sort -rn
  echo "--- unique instantiation signatures (name + template args, deduped, no clone/cold) ---"
  sed 's/^[0-9a-f]* [A-Za-z] //' /tmp/raw_$tag.txt | sed 's/__device_stub__//;s/ \[clone.*//' | grep 'kernel' | sed 's/(.*//' | sort -u > /tmp/uniq_$tag.txt
  wc -l < /tmp/uniq_$tag.txt | sed 's/^/unique kernel signatures: /'
  echo "--- per-family unique signatures ---"
  sed 's/<.*//' /tmp/uniq_$tag.txt | sort | uniq -c | sort -rn
}
inv $B/gqa_attention_decode_g35.cu.o g35
inv $B/gqa_attention_decode_muse.cu.o muse
inv $B/gqa_attention_decode.cu.o main
echo "=== g35 unique signatures full list ==="
cat /tmp/uniq_g35.txt
echo "=== muse unique signatures full list ==="
cat /tmp/uniq_muse.txt
