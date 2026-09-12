#!/bin/bash
echo "=== 转换器支持哪些量化格式（modelopt / compressed-tensors / nvfp4）==="
grep -rn 'modelopt\|compressed\|nvfp4\|quant_method\|weight_scale\|weight_scale_2\|scale_global' \
  /home/user/ninfer-fusion/tools/convert/qwen3_8_27b/ 2>/dev/null | head -30
echo
echo "=== qwen3_8_27b 转换器文件清单 ==="
ls -la /home/user/ninfer-fusion/tools/convert/qwen3_8_27b/ 2>/dev/null | head -20
echo
echo "=== 公共转换层支持格式 ==="
ls /home/user/ninfer-fusion/tools/convert/common 2>/dev/null | head -20
grep -rn 'modelopt\|compressed-tensors' /home/user/ninfer-fusion/tools/convert/common/*.py 2>/dev/null | head -15
echo
echo "=== convert_runner 用法 ==="
sed -n '1,40p' /home/user/ninfer-fusion/tools/convert/convert_runner.py 2>/dev/null
