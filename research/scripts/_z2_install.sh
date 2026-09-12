#!/bin/bash
# Z2: install latest vLLM into an ISOLATED env for DFlash2 reference data.
# Hard rules honored: never touch C:\vllm\venv , ziqinzhang\vllm-venv , /home/user/ninfer-fusion
# No make/nvcc/cmake invoked -> --only-binary=:all: forbids source builds.
set -x
VENV=/home/user/vllm-dflash2-venv
LOG=/home/user/_z2_install.log
exec > >(tee -a "$LOG") 2>&1

echo "===== Z2 STEP 1: create isolated venv (python3.12) ====="
date -Is
rm -rf "$VENV"
python3.12 -m venv "$VENV"
"$VENV/bin/python" --version
"$VENV/bin/python" -c "import sys; print('prefix=', sys.prefix)"

echo "===== Z2 STEP 2: upgrade pip tooling ====="
"$VENV/bin/pip" install --upgrade pip setuptools wheel 2>&1 | tail -5

echo "===== Z2 STEP 3: pip install vllm==0.29.0 (wheels only) ====="
date -Is
"$VENV/bin/pip" install --only-binary=:all: "vllm==0.29.0"
RC=$?
echo "PIP_EXIT=$RC"
date -Is

echo "===== Z2 STEP 4: freeze snapshot ====="
"$VENV/bin/pip" freeze > /home/user/_z2_freeze.txt 2>&1
echo "freeze lines: $(wc -l < /home/user/_z2_freeze.txt)"
echo "----- torch / vllm lines -----"
grep -iE '^(torch|vllm|flashinfer|transformers|numpy|xformgram|xgrammar)' /home/user/_z2_freeze.txt

echo "===== Z2 STEP 5: verify import ====="
"$VENV/bin/python" -c "import vllm; print('VLLM_VERSION=', vllm.__version__); print('VLLM_FILE=', vllm.__file__)" 2>&1 | tail -10

echo "===== Z2 STEP 6: grep dflash2 in site-packages ====="
SP="$VENV/lib/python3.12/site-packages"
echo "--- dflash2 hits ---"
grep -ril 'dflash2' "$SP" 2>/dev/null | head -40
echo "--- dflash2 hit count ---"
grep -ril 'dflash2' "$SP" 2>/dev/null | wc -l
echo "--- dflash (any) hits ---"
grep -ril 'dflash' "$SP" 2>/dev/null | head -40
echo "--- dflash hit count ---"
grep -ril 'dflash' "$SP" 2>/dev/null | wc -l
echo "--- dspark hits (existing baseline had dspark) ---"
grep -ril 'dspark' "$SP" 2>/dev/null | head -20
echo "=== Z2 DONE ==="
