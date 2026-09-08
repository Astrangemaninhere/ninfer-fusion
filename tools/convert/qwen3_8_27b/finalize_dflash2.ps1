# finalize_dflash2.ps1 -- end-to-end DFlash2 training export.
# Usage:  finalize_dflash2.ps1 [-Ckpt <path>] [-SkipWsl]
#   * patches the newest train checkpoint into a _tuned copy of the
#     qwen3.8-27B nvfp4-dflash2 artifact (source artifact untouched)
#   * verify_patch.py: object table + hash equality on copied tensors +
#     ckpt equality on the 73 replaced tensors
#   * stages the tuned artifact into WSL /home/user/models for engine runs
$ErrorActionPreference = "Stop"
param(
    [string]$Ckpt = "",
    [switch]$SkipWsl
)
$py = "C:\Program Files\Python312\python.exe"
$base = "C:\Users\User\Documents\ziqinzhang"
$tool = Join-Path $base "ninfer-fusion-repo\tools\convert\qwen3_8_27b"
$srcArt = Join-Path $base "models\Qwen3.8-27B-Huihui-Abliterated-NInfer-DFlash2\qwen3_8_27b_nvfp4_dflash2.ninfer"

if (-not $Ckpt) {
    $Ckpt = Get-ChildItem (Join-Path $base "data\dflash2_ckpts\step_*.pt") |
        Sort-Object { [int]([regex]::Match($_.BaseName, '\d+').Value) } |
        Select-Object -Last 1 -ExpandProperty FullName
}
if (-not (Test-Path $Ckpt)) { throw "checkpoint not found: $Ckpt" }
$out = Join-Path $base ("data\dflash2_ckpts\" + [IO.Path]::GetFileNameWithoutExtension($Ckpt) + "_tuned.ninfer")
Write-Host "== patch: $Ckpt -> $out"
& $py (Join-Path $tool "patch_dflash2.py") --src $srcArt --ckpt $Ckpt --out $out
if ($LASTEXITCODE -ne 0) { throw "patch failed" }
Write-Host "== verify"
& $py (Join-Path $tool "verify_patch.py") --src $srcArt --out $out --ckpt $Ckpt
if ($LASTEXITCODE -ne 0) { throw "verify failed" }

if ($SkipWsl) { Write-Host "== staged locally: $out"; exit 0 }
Write-Host "== staging to WSL /home/user/models"
$wslName = [IO.Path]::GetFileName($out)
$target = "\\wsl.localhost\Ubuntu\home\user\models\$wslName"
Copy-Item $out $target -Force
if ($LASTEXITCODE -ne 0) { throw "wsl copy failed" }
Write-Host "== done. Engine artifact: $target"
