# C 线产物清单：从 step_XXXXXX.pt 到引擎可加载（最终 30 秒版）

> 唯一状态源：`_collab/board.md`。本页是操作卡片，不是状态。全部 CPU-only，不动源 artifact。

## 0) 变量（cmd，工作区根 `C:\Users\User\Documents\ziqinzhang`）
```bat
set PY=C:\Program Files\Python312\python.exe
set SRC=models\Qwen3.8-27B-Huihui-Abliterated-NInfer-DFlash2\qwen3_8_27b_nvfp4_dflash2.ninfer
set CK=data\dflash2_ckpts\step_006000.pt
set OUT=data\dflash2_ckpts\step_006000_tuned.ninfer
```

## 1) 三闸门（patch → verify → round-trip，实测 ~2+1+1 min）
```bat
%PY% ninfer-fusion-repo\tools\convert\qwen3_8_27b\patch_dflash2.py --src %SRC% --ckpt %CK% --out %OUT%
%PY% ninfer-fusion-repo\tools\convert\qwen3_8_27b\verify_patch.py --src %SRC% --out %OUT% --ckpt %CK%
%PY% _dflash2_roundtrip_tmp.py %CK% %OUT%
```
gate 期望字符串（逐字）：
- patch: `mapping ok: 73 tensors queued` / `written: %OUT%`；且 `note:` 行**恰好 10 条**、
  全部形如 `layers.{0..4}.{attention_conv,mlp_conv}.base reshaped (2, 2, 320, 16) -> (2, 2, 5120)`。
- verify: `identity+table ok: 1200 objects` / `copied tensors ok: 1118/1118 hash-equal` /
  `replaced tensors ok: 73 match checkpoint` / `verify ok`，exit 0。
- round-trip: `checked=73/73 bit-exact=73 mismatched=[]` / `readback NaN=0 Inf=0` / `VERDICT: BIT-EXACT`。
失败时看哪行：
- `MAPPING ERROR: artifact missing:` → 源 artifact 不是 dflash2 版（用错 SRC）。
- `CKPT ERROR: missing keys: [...]` → ckpt 缺层/缺键（exit 2，不会静默通过；多余键会被无视，无告警）。
- `SHAPE MISMATCH <name>` → 训练侧结构改了 → 停，对 `_dflash2_export_order_contract.md` 逐条 diff。
- 非 10 条 `note:`（多出其它 reshape）→ 视为顺序风险，停并查 contract §1-§5。

## 2) 装载到引擎（WSL）
```bat
copy %OUT% \\wsl.localhost\Ubuntu\home\user\models\step_006000_tuned.ninfer
```
确认（size/weights_id 区分不了 tuned 与 baseline，只有 名字+mtime+sha256）：
`wsl.exe -e bash -c "ls -l /home/user/models/step_006000_tuned.ninfer && sha256sum /home/user/models/step_006000_tuned.ninfer"`

## 3) A/B 接受率（一个 GPU 窗口内跑完）
```bash
# WSL 内；binary 必须指向重建后的实战路径（~/ninfer/build 是旧路径，勿用）：
NINFER_BIN=/home/user/ninfer-fusion/build/apps/ninfer \
B_ARTIFACT=/home/user/models/step_006000_tuned.ninfer \
bash /mnt/c/Users/User/Documents/ziqinzhang/_df2_ab.sh
```
输出 side-by-side：draft window / rounds / drafted / accepted / **acceptance rate** /
acceptance length / **accepted by pos** 直方图 / decode tok/s + `VERDICT: A_BETTER|B_BETTER|TIE`。
**坑**：引擎把 dflash2 指标标成 `mtp` 前缀（`apps/cli/main.cpp:222` 三目漏了 DFlash2）——
读日志/对账时按子串 grep `acceptance rate`、`accepted by pos`，不要带前缀。
训练存活时脚本自动 ABORT（exit 3）。

## 4) 快速失败注入（已验证 2026-09-09）
构造缺 layer4 的 60 键 ckpt 跑 `--dry-run` → `CKPT ERROR: missing keys: ['layers.4....']`，**exit 2**
（缺键=硬错误；多余键被静默忽略）。

## 自检（本文件自身；复制执行应全绿）
```bat
bash -n 校验：wsl.exe -e bash -c "bash -n /mnt/c/Users/User/Documents/ziqinzhang/_df2_ab.sh && echo SYNTAX_OK"
关键字存在：powershell -NoProfile -Command "Select-String -Path C:\Users\User\Documents\ziqinzhang\_collab\C_artifact_manifest.md -Pattern 'mapping ok: 73','1118/1118','BIT-EXACT','accepted by pos','ninfer-fusion/build/apps/ninfer' | Measure-Object | Select-Object -ExpandProperty Count"
（期望 Count=6+）
```
