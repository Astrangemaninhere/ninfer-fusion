# setup_env.ps1
# Sets up isolated Python virtual environment for Qwen3.8-Flash-Next Oracle

$ErrorActionPreference = "Stop"

$VENV_DIR = "E:\NInfer\venv-qwen4exp"
$PYTHON_EXE = "C:\Python314\python.exe"

if (-not (Test-Path $PYTHON_EXE)) {
    Write-Error "Python 3.14 executable not found at $PYTHON_EXE"
}

Write-Host "Creating virtual environment at $VENV_DIR using $PYTHON_EXE ..."
& $PYTHON_EXE -m venv $VENV_DIR

$VENV_PY = Join-Path $VENV_DIR "Scripts\python.exe"
$VENV_PIP = Join-Path $VENV_DIR "Scripts\pip.exe"

Write-Host "Upgrading pip ..."
& $VENV_PY -m pip install --upgrade pip

Write-Host "Installing dependencies (torch CPU, safetensors, numpy, huggingface deps) ..."
& $VENV_PIP install --index-url https://download.pytorch.org/whl/cpu torch
& $VENV_PIP install safetensors numpy tokenizers huggingface_hub regex pyyaml requests packaging filelock tqdm

Write-Host "Installing transformers from reference commit ..."
& $VENV_PIP install git+https://github.com/huggingface/transformers@fc5c5bde8e

Write-Host "Setup complete. Virtual environment ready at $VENV_DIR"
