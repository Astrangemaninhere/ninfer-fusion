#!/bin/bash
# WSL direct downloads are ~14KB/s to pythonhosted. Test whether a China PyPI mirror is fast.
test_url() {
  local name="$1"; local url="$2"
  local out
  out=$(timeout 25 curl -sS -o /dev/null -w '%{http_code} %{speed_download} %{time_total}' --max-time 20 "$url" 2>&1 | tail -1)
  echo "$name -> $out"
}
echo "=== mirror / index reachability + speed (bytes/sec) ==="
test_url "tuna      " "https://pypi.tuna.tsinghua.edu.cn/simple/vllm/"
test_url "aliyun    " "https://mirrors.aliyun.com/pypi/simple/vllm/"
test_url "ustc      " "https://mirrors.ustc.edu.cn/pypi/simple/vllm/"
test_url "sjtu      " "https://mirror.sjtu.edu.cn/pypi/web/simple/vllm/"
test_url "pypi.org  " "https://pypi.org/simple/vllm/"
echo ""
echo "=== download a real wheel chunk from tuna (vllm 0.29.0) ==="
timeout 30 curl -sS -o /dev/null -w 'tuna_whl http=%{http_code} speed=%{speed_download}\n' --max-time 25 \
  "https://pypi.tuna.tsinghua.edu.cn/packages/ca/09/7f79450e21bd1c2a0897ab946a544816a4f7e04c54f0ca49849b14b12d6a/vllm-0.29.0-cp38-abi3-manylinux_2_28_x86_64.whl" 2>&1 | tail -2
echo ""
echo "=== cuda-tile on pypi ==="
timeout 25 curl -sS "https://pypi.org/pypi/cuda-tile/json" 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print('cuda-tile latest:', d['info']['version']); [print('  ', f['filename']) for f in d['urls']]" 2>&1 | head -15
