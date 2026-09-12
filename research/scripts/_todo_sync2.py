#!/usr/bin/env python3
import pathlib, datetime
J = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang")
T = J / "_TODO.md"
old = T.read_text(errors="replace") if T.exists() else ""
block = """
## 2026-09-11 晚 · vLLM+dflash2 参考跑（环境线）与引擎线状态

### 参考跑：软件墙全破，卡在 WSL GPU 直通
- **dflash2 草稿终于完整**：`data/draft_dflash2_ref`(=`incoai/Qwen3.8-27B-DFlash2`) 之前是**截断的**
  （头声明 3,848,817,896 B，实际 902 MB=23.4%，随后还被删 ✗）⇒ 这就是"跑不起来"的真因；
  已用 hf-mirror 整份重下并校验 `COMPLETE` ✓（81 张量/5 层；`architectures: DFlash2DraftModel`；
  `dflash_config{block_size 8, selector_top_k 16, conv_kernel_size 2, conv_group_size 16,
  target_layer_ids [5,19,33,47,61]}`）。
- **全新 vllm 0.29.0**（WSL, python3.12；torch 2.13.0+cu130）✓ 原生带
  `v1/worker/gpu/spec_decode/dflash2/speculator.py` + `models/qwen3_dflash2.py` ✓（方法名仍是 `dflash`，
  dflash2 由草稿架构+`dflash_config` 选中）。
- 破掉的墙：HF 不可达→`hf-mirror.com` ✓；WSL `pin_memory=False` 默认→`VLLM_WSL2_ENABLE_PIN_MEMORY=1` ✓；
  `flashinfer-cubin 0.6.13 ≠ flashinfer-python 0.6.18`→`FLASHINFER_DISABLE_VERSION_CHECK=1` ✓；
  drvfs 慢→17 GB 模型拷到原生盘 ✓；主存 22→27 GB（`.wslconfig`，原值 22GB+swap50GB+16proc 可回滚）。
- **实测引擎已能初始化**（target + dflash2 草稿都加载 ✓）。
- **但 WSL VM 连崩 4 次**（`Wsl/Service/E_UNEXPECTED`）：最后一次 **25 GB available 仍崩** ⇒
  **不是 OOM**，是 **WSL GPU 直通/驱动在重载下不稳定** ✓（每次都是显存刚上/推理期；
  dmesg 只有 memory-pressure 无 OOM kill）。⇒ **参考跑必须回到 Windows 侧**，
  而 Windows 侧支持 dflash2 的轮子本机不存在（0.29 需源码构建 ✗ 数小时）。**暂不再试 WSL 路线。**

### 参考侧已有的决定性证据（R2，Windows 侧）
| 实现 | 接受率 | 逐位置 |
|---|---|---|
| vLLM（dspark 原生草稿） | **1.87%** | `[11,0,0,0,0,0,0]` |
| ninfer dflash2 | 4.97% | `[8,0,0,0,0,0,0]` |
| ninfer dspark | 7.63% | `[17,1,0,0,0,0,0]` |
⇒ 两个独立实现同一症状 ⇒ **不是 ninfer 缺陷**；`K=1=23.68%` ⇒ 第一列健康、退化在**块内后续列**。
你记忆中的"五十几"来自 **LABD（lookup 上下文复制）**，是换机制、不是草稿变准；R3 判定我们的权重
跑不了 1Cat-vLLM（Hq6/Hkv1/D256 + SM70 + 另一份 checkpoint）。

### 引擎线（同一批）
- ① FP32 partial 落地并复测：列0 一致率 **69%→78.3%**（列1 87.5%）✓，接受率几乎不动（4.51%→4.81%）
  ⇒ 贡献者、非机制（修正此前"真凶锁定"）；W=2 判别实验排除"路由不同"；DF2SEL 证明边项是活的（54% 步改变决定）。
- ② TU 拆分落地且编译链接全绿 ✓（208 MB→17.9 KB 调度器）；但**长杆转到 `_smallt.cu`（单 ptxas 15 min）**
  ⇒ 干净构建 36 min，还需再拆一刀。
- 八项：N1/N2/§103 早已进树（看板陈旧）；**N2 遗留 `ColdPolicy::Host` 静默已修**（矛盾报错+一次性告警，
  备份 `/home/user/cold_host_bak/`）；**N3 第二轮其实已完成**（只修了两处引用已删字段的陈旧注释）；
  N6/N7/N4/N5/U6 各有硬阻塞。
- 换头 A/B：两个新 artifact 三闸门全过（`73 match checkpoint`、BIT-EXACT）但引擎 **0/945=0.00%** ✗ ⇒
  源码定因 `train_dflash2.py:355` 随机初始化、仅 `--resume` 才装载 ⇒ 那炉是"随机头训 200 步"；
  续训脚本已修成 `--resume` + 独立 `--out-dir`（种子 `step_001900`）。
"""
T.write_text(old + block)
print("_TODO.md 已同步: %d -> %d 行" % (len(old.splitlines()), len(T.read_text().splitlines())))
