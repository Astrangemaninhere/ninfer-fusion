#!/usr/bin/env python3
"""§130：offset 家族全量核查 + 第二处同族 bug（dflash2 训练目标偏移）已修。"""
import datetime
import pathlib

stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
T = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/_TODO.md")
T.open("a", encoding="utf-8").write("""
### 130. "取列/目标偏移"家族全量核查：共两处，均已修（""" + stamp + """）
用户问"dspark 有这个毛病，别的有没有？"——把三个后端的**训练约定**与**推理取列**两两核对（并做数据实证）：

| 后端 | 推理取列 | 训练目标配对 | 判定 |
|---|---|---|---|
| **dspark (DFlash v1)** | `dflash_impl.h:448` 原为 `if constexpr (!bf16_weights) offset=1`，即 **bf16 草稿含 anchor 列** | `train_dspark.py:168-169` 输出只用 `out[:, 1:]`；`:430` 每列预测自己 | **推理错** ⇒ 已修（offset 恒 1） |
| **dflash2** | `dflash2_impl.h:315-316` 已 `+ hidden*bytes` 跳过第 0 列（注释写明 bonus 在 column 0） | `train_dflash2.py:220` 同样 `out[:, 1:]`；但 `:435` 目标行 = `ids16[a+target_shift+i]` | **推理对、训练目标错** ⇒ 已改默认值 |
| **MTP** | 无 block 列机制（自回归链） | 无本地训练脚本 | **不适用该族**（p(pos0)=53.5% 也是三者最高） |

**dflash2 的凭证（双重实证）**：
1. **口径实测**（`_probe_ids16_semantics.py`，4 个真实 cache，3832 样本）：
   `P(ids16[t]==tok[t+1])=0.2904` vs `P(ids16[t]==tok[t])=0.0008` ⇒ `ids16[t]` 是**位置 t 的 next-token 分布**；
2. 于是"位置 a+1+i 的 token"的老师行是 `ids16[a+i]`，而 `head()` 的 `out[:, 1:]` 已让槽位 i 落在位置 a+1+i
   ⇒ **对齐的 shift 是 0**；默认值 1 取 `ids16[a+1+i]`（位置 a+2+i 的 token）⇒ **训练目标整体晚一行**。
3. 旁证：先前"三臂探针"里 legacy(shift=1) 命中率最高（0.2290 vs aligned 0.1239）——正是**模型学的是"晚一行"**
   的表现，与本次口径实证一致。
**修法**：`train_dflash2.py` 的 `--target-shift` 默认 1 → **0**（注释里写明口径实测数据，防止被改回）。
⚠️ **现有 checkpoint 是 shift=1 训出来的** ⇒ 要让接受率受益必须**重训**（训练按用户要求停着，等放行）。
**验证链已起**（`_offset_family_verify.sh`，等编译结束自动跑）：dspark（修后）位置剖面 vs 修复前的
`accept=11.22% / p(pos0)=15/56=26.8% / 1.39 tok/round`，并以 dflash2（列本就对）与 mtp3 作对照。
""" )
print("_TODO.md §130 已追加")
