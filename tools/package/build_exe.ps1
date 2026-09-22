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
# This script lives at <repo>\tools\package\build_exe.ps1, so the repository is two
# directories up and dist/ is written inside it.  It used to name a sibling
# `ninfer-fusion-repo` checkout: a stale mirror, i.e. the exe was built from
# whatever that copy held.  The entry (ninfer-gui.py) and its page now live at
# the repository root, so a clean checkout packages without a workspace around
# it.
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$root = $repo   # dist/ lives inside the checkout; the entry no longer sits one level above it
Write-Host '== building NInfer.exe (onefile)'
# --workpath: PyInstaller's default work directory is .\build, and in this repository
# that is CMake's out-of-source build tree, which is very much alive (this very GUI
# launches build\apps\ninfer-serve).  Package intermediates do not belong there, so
# name the work directory explicitly and keep it outside the checkout.
$wpath = if ($env:LOCALAPPDATA) { Join-Path $env:LOCALAPPDATA 'NInfer\pyi-build' } `
         else { Join-Path $env:TEMP 'NInfer\pyi-build' }
New-Item -ItemType Directory -Force -Path $wpath | Out-Null
& $py -m PyInstaller (Join-Path $repo 'tools\package\ninfer-gui.spec') `
    --noconfirm --clean --workpath $wpath --distpath (Join-Path $root 'dist')
if ($LASTEXITCODE -ne 0) { throw 'build failed' }

# 3. 冒烟: 起 exe, 探 8077
$exe = Join-Path $root 'dist\NInfer.exe'
Write-Host '== smoke test'
# The probe below is the only thing between a broken build and `== done`.  A onefile
# exe is a PARENT that unpacks plus a CHILD that serves: stopping the parent leaves
# the child holding 8077, and such an orphan left over from an EARLIER run answers
# the probe, so the smoke goes green for whatever exe it is handed -- a green that
# is not evidence about this build at all.  So clear 8077 BEFORE starting the exe
# under test, and if something still answers, refuse: this run cannot prove
# anything, and a loud abort is the only honest outcome left.
Get-Process -Name 'NInfer' -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Milliseconds 700
# Port level, not HTTP level: ANY listener owns the port, whatever it answers (404 counts).
$busy = Get-NetTCPConnection -LocalPort 8077 -State Listen -ErrorAction SilentlyContinue
if ($busy) {
  $owner = ($busy | Select-Object -First 1).OwningProcess
  $oname = if ($owner) { (Get-Process -Id $owner -ErrorAction SilentlyContinue).ProcessName } else { '?' }
  Write-Host "== smoke ABORTED: 8077 already has a listener (pid $owner = $oname) before the exe under test started."
  Write-Host "   Any 200 from here on would be that process's, not this build's, so this run proves nothing and is not reported as a pass."
  exit 1
}
$p = Start-Process -FilePath $exe -PassThru -WindowStyle Hidden
# Three defects, each measured on 2026-09-14 (Windows PowerShell 5.1.26100):
#  1. Invoke-WebRequest without -UseBasicParsing goes through the IE DOM parser and
#     throws System.NullReferenceException 3/3 with the server confirmed up, so this
#     check could NEVER pass here -- a false RED on a healthy exe.
#  2. A single shot at t=5 s had no headroom: measured time-to-200 was
#     1.4 / 1.4 / 1.5 / 2.0 / 2.1 / 2.5 / 4.6 / 22.1 s.
#  3. Stopping only $p.Id killed the onefile PARENT; the CHILD kept 8077, so every
#     later run's probe answered 200 from the orphan regardless of the exe under
#     test.  Stop by image name on BOTH paths.
$smokeOk = $false
for ($i = 0; $i -lt 40; $i++) {          # up to 40 s, still bounded
  Start-Sleep -Seconds 1
  try {
    $r = Invoke-WebRequest -Uri 'http://127.0.0.1:8077/api/env' -TimeoutSec 5 -UseBasicParsing
    if ([int]$r.StatusCode -eq 200) {
      $smokeOk = $true
      Write-Host ('smoke ok: HTTP ' + $r.StatusCode + ' after ' + ($i + 1) + 's')
      break
    }
  } catch {
    if ($i -eq 39) { Write-Host ('smoke FAILED: ' + $_.Exception.Message) }
  }
  if ($null -ne $p -and $p.HasExited) { break }
}
if (-not $smokeOk) {
  Get-Process -Name 'NInfer' -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
  Write-Host '== smoke test failed: the packaged exe did not answer GET http://127.0.0.1:8077/api/env'
  exit 1
}
# a build script must not leave a live server holding 8077 behind it
Get-Process -Name 'NInfer' -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Write-Host "== done: $exe"
