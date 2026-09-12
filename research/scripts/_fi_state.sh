#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
L=$J/dl/vref_df2c.err
echo "=== 最新一次的根因（run2，含 TRITON_ATTN 尝试） ==="
grep -nE 'RuntimeError|ValueError|Error:|FlashInfer|flashinfer|attention backend|No available|unsupported' "$L" 2>/dev/null | head -12
echo
echo "=== 首个 Traceback 前 20 行 ==="
N=$(grep -n 'Traceback (most recent call last)' "$L" 2>/dev/null | head -1 | cut -d: -f1)
[ -n "${N:-}" ] && sed -n "$(( N>22 ? N-22 : 1 )),${N}p" "$L" | head -26
echo
echo "=== flashinfer 是否可用（实证） ==="
/home/user/vllm029/bin/python - <<'PY' 2>&1 | tail -6
import os
os.environ.setdefault("VLLM_WSL2_ENABLE_PIN_MEMORY","1")
try:
    import flashinfer
    print("import flashinfer ok:", flashinfer.__version__)
    import flashinfer.cubin
    print("cubin ok")
except Exception as e:
    print("flashinfer err:", type(e).__name__, str(e)[:200])
try:
    from vllm.utils.flashinfer import has_flashinfer
    print("vllm 认 flashinfer:", has_flashinfer())
except Exception as e:
    print("vllm flashinfer 检查失败:", type(e).__name__, str(e)[:160])
PY
