#!/usr/bin/env python3
import pathlib, datetime
J = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang")
T = J / "_TODO.md"
old = T.read_text(errors="replace") if T.exists() else ""
block = """
## 2026-09-11 深夜 · vLLM+dflash2 参考线结论（关闭）与宿主环境发现

### 参考线：WSL 路线在原理上走不通（已停，勿再试）
- 7 次 `Wsl/Service/E_UNEXPECTED` 崩机；所有可调项都拆过：MM 关闭、`--kv-cache-dtype fp8`、
  `--skip-mm-profiling`、`--num-gpu-blocks-override 256`、最小形状（seqs=1/len=512）、
  模型移至原生盘（去 9P）、VM 22→27GB ⇒ 仍在 `avail 22G` 时崩。
- **原理性硬约束**：dflash2 spec-decode 的 `StagedWriteTensor→UvaBuffer` 要求
  `is_uva_available()`（= pin_memory）⇒ WSL 下必须 `VLLM_WSL2_ENABLE_PIN_MEMORY=1`，
  而该开关让 vLLM 用**锁页主机内存（不可回收、计入 used）** ⇒ **主机内存 22–25GB 尖峰** ⇒ 崩。
  ⇒ **WSL 里跑 dflash2，钉页主机内存躲不掉**；Windows 侧无此矛盾（R2 的数即出自那里）。
- 另有 `--swap-space`（vLLM CPU swap，默认 4GiB）从未设过 ⇒ 若将来仍要试应先 `--swap-space 0`。
- 结论：**参考跑若要做，只能在 Windows 侧**（需源码构建支持 dflash2 的 Windows 轮子，数小时）；
  否则采用 R2 既有证据（vLLM+dspark 1.87%、逐位置 [11,0,0,0,0,0,0]，与 ninfer 同症状）。

### 宿主环境发现（与项目无关，但影响可用性）
- 一晚上的崩机/显示重枚举后 `dwm` 常驻 ~20%（`dwm` 强停重启无效、禁用 GameViewer 虚拟屏无效、
  停 MicaForEveryone 无效、用户已重启两次仍复现）。用户观察：**每次跑完模型之后都这样**。
- 已改（可回滚，需重启生效）：`HKLM\\SYSTEM\\CurrentControlSet\\Control\\GraphicsDrivers\\HwSchMode = 1`
  （关闭 HAGS，原为未设置=默认；回滚置 2）。
- 查到既有非默认值：**`TdrDelay = 8`**（默认 2s）⇒ GPU 卡住后要 8 秒才恢复显示栈
  ⇒ 与用户描述的"闪动"高度相符（可能是前人为了长编译调大的）。`TdrLevel = 3` 正常。
- 我造成的增量已清理：WSL 侧 17GB 模型副本（删）、12GB pip 缓存（删）、`.wslconfig` 复原为
  22GB/50GB/16proc、GameViewer 虚拟屏已恢复 OK、MicaForEveryone 已恢复运行、WSL 已关闭、无 CUDA 进程。
- 保留物（可按需删）：`/home/user/vllm029`（9.8GB 全新 vllm0.29 环境）、
  `/home/user/models/q3nvfp4`（17GB 原生盘模型副本）、`/home/user/models/draft_dflash2_ref`（3.6GB）。
"""
T.write_text(old + block)
print("_TODO.md: %d -> %d 行" % (len(old.splitlines()), len(T.read_text().splitlines())))
