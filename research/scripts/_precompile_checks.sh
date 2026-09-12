#!/bin/bash
# 编译窗口打开前，先把两份待落补丁的"纯主机"部分验掉（不碰 GPU、不碰编译）。
# ① E 组的几何响亮失败验收（26/26，含 before/after 差分）
# ② S3 行标定表的载荷不变性（crc32=45795edc 判据）
# ③ S3 的几何闸门 2×2（旧 header 必拒 35b/Muse，新 header 必收，超池必拒）
set -u
J=/mnt/c/Users/User/Documents/ziqinzhang
C=$J/_collab
export PATH="/home/user/.local/bin:$PATH"

run() {  # $1=标签 $2...=命令
  local tag=$1; shift
  echo "=== [$tag] $* ==="
  timeout 600 "$@" 2>&1 | tail -25
  echo "  rc=$?"
  echo
}

echo "############ ① E 组：几何越界必须响亮失败 ############"
if [ -f "$C/E_geo_acceptance.sh" ]; then
  bash "$C/E_geo_acceptance.sh" 2>&1 | tail -30
else
  echo "  缺 $C/E_geo_acceptance.sh"
fi
echo

echo "############ ② S3：载荷逐字节不变（crc32 判据）############"
if [ -f "$C/build/S3_rowscale_payload_check.py" ]; then
  python3 "$C/build/S3_rowscale_payload_check.py" \
      /home/user/ninfer-fusion/src/ops/kernel/gqa_isoquant_row_scale.cu 2>&1 | tail -12
else
  echo "  缺 S3_rowscale_payload_check.py"
fi
echo

echo "############ ③ S3：几何闸门 2×2 ############"
if [ -f "$C/build/S3_rowscale_geom_test.sh" ]; then
  bash "$C/build/S3_rowscale_geom_test.sh" 2>&1 | tail -20
else
  echo "  缺 S3_rowscale_geom_test.sh"
fi
