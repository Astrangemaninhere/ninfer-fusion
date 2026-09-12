#!/bin/bash
# 1) 把 §117 条目复核写回 _TODO.md（避免重复排查）
# 2) 用独立 py 文件重写下载循环（不用 heredoc-in-substitution）
set -u
T=/mnt/c/Users/User/Documents/ziqinzhang/_TODO.md
cat >> "$T" <<'EOF'

## 117b. §117"仍未落地"条目复核 (2026-09-12, 实证)
本轮逐条核实，**四项全部已修 / 既知接受 / 陈旧**，清单未同步：
1. 全局 `--kv-dtype e8` 起不来 → **已修**：`src/ops/wrapper/gqa_attention.cpp:450-456` 的
   profile 校验集已含 `DType::E8Kv`，注释原文即指 _TODO 98/117 U2。
2. `all:bf16` 语义缺口 → **解析正常**：权威解析器 `src/product/kv_options.h:43-62` 已支持
   `all`（`first=0,last=slots-1`）；残余问题是 "BFloat16 兼作 unset 哨兵"，该文件 25-31 行
   **已明确记为需要"was-set 掩码"的 API 变更**，并给出替代（全局 `--kv-dtype bf16`）
   ⇒ 属既知限制，非缺陷。
3. `sliding_window_tokens` 全局无赋值点 → **已有赋值**：`src/targets/qwen3_6/impl/state/
   decoder_state.cpp:324` 与 `:382`（`.sliding_window_tokens = window`）。Muse 掉针回归
   仍值得做（值是否正确未验），但"无赋值点"不成立。
4. `std::array<...,64>` 无越界校验 → **陈旧**：写入循环以 `.size()` 为界
   （`layouts_impl.h:1061-1078`），serve 侧 `parse_kv_table` 另做 clamp
   （`kv_auto_relayout.cpp:60-63`）。
另（本轮新修，非清单项）：`kv_auto_relayout.cpp` 的**重复实现** `parse_kv_table` 把 `all`
交给 `atoi` 解析成 0 ⇒ `all:<tier>` 在该副本里只设第 0 层；已补 `range == "all"` 分支
（该副本仅供 auto-relayout 使用；CLI/serve/引擎走 `product::parse_kv_layer_storage`）。
EOF
echo "TODO 已追加复核段"
python3 - <<'PY'
import pathlib
p = pathlib.Path("/home/user/dl_one.py")
p.write_text('''import sys
from huggingface_hub import snapshot_download
try:
    snapshot_download(repo_id=sys.argv[1], local_dir=sys.argv[2], max_workers=2)
    print("SNAPSHOT_OK")
except Exception as e:
    print("FAIL:", type(e).__name__, str(e)[:160])
''')
print("dl_one.py 已写")
PY
cat > /home/user/dll2.sh <<'EOS'
#!/bin/bash
set -u
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
REPO=sakamakismile/Huihui-Qwen3.8-27B-abliterated-NVFP4
DEST=/home/user/models/q38_abl_huihui_nvfp4
LOG=/mnt/c/Users/User/Documents/ziqinzhang/dl/hf_dl_loop2.log
echo "start $(date '+%H:%M:%S')" >> "$LOG"
i=0
while [ "$i" -lt 500 ]; do
  i=$((i+1))
  echo "try $i $(date '+%H:%M:%S') $(du -sh "$DEST" 2>/dev/null | cut -f1)" >> "$LOG"
  out=$(python3 /home/user/dl_one.py "$REPO" "$DEST" 2>&1)
  echo "$out" >> "$LOG"
  case "$out" in *SNAPSHOT_OK*) echo "DONE $(date '+%H:%M:%S')" >> "$LOG"; break;; esac
  sleep 5
done
echo "LOOP2_DONE" >> "$LOG"
EOS
chmod +x /home/user/dll2.sh
echo "dll2.sh 已写"
