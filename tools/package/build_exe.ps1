# build_exe.ps1 — NInfer Windows EXE 封装构建 (单文件, 纯标准库 GUI).
# 前置: Windows Python 3.10+; 自动装 pyinstaller (官网: https://pyinstaller.org)
# 产物: dist\NInfer.exe —— 双击即起 GUI 并自动开浏览器; 环境自检在 GUI 首页。
$ErrorActionPreference = 'Stop'
$py = 'C:\Program Files\Python312\python.exe'
if (-not (Test-Path $py)) { throw 'python not found: 请到 https://www.python.org/downloads/ 安装' }

# 1. pyinstaller
& $py -m pip show pyinstaller *> $null
if ($LASTEXITCODE -ne 0) {
  Write-Host '== installing pyinstaller'
  & $py -m pip install pyinstaller
}

# 2. 构建
$repo = 'C:\Users\User\Documents\ziqinzhang\ninfer-fusion-repo'
Write-Host '== building NInfer.exe (onefile)'
& $py -m PyInstaller (Join-Path $repo 'tools\package\ninfer-gui.spec') --noconfirm --clean
if ($LASTEXITCODE -ne 0) { throw 'build failed' }

# 3. 冒烟: 起 exe, 探 8077
$exe = 'C:\Users\User\Documents\ziqinzhang\dist\NInfer.exe'
Write-Host '== smoke test'
$p = Start-Process -FilePath $exe -PassThru -WindowStyle Hidden
Start-Sleep 5
try {
  $r = Invoke-WebRequest -Uri 'http://127.0.0.1:8077/api/env' -TimeoutSec 5
  Write-Host ('smoke ok: HTTP ' + $r.StatusCode)
} catch {
  Write-Host ('smoke warn: ' + $_.Exception.Message)
}
Write-Host "== done: $exe"
