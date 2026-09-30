# KV 分层策略矩阵（2026-09-10 实测）

来源：`_TODO.md` §95（用户指令：各种 KV 组合都试一遍，含"给定 bit（可小数）的混合精度策略"）。
设备：**RTX 5090 D（单卡）**；模型 **qwen3.8-27b**（16 个全注意力层；Muse 因 §89 长 prompt
不可用，本表不含 Muse）。上下文 32768，针刺 8 针（`tools/archkit/longtest_57k.py`，已修）。

## 1. 实测矩阵

> **重要更正（§97）**：全局 `--kv-dtype` **不进入 KV 池几何**——下表里 `all:bf16` 那一行实际
> 就是 nvfp4（载荷/速度/质量与 nvfp4 行逐项相同），真正的 bf16 KV 目前**无法启用**
> （bf16 是逐层表的"未设置"哨兵值）。逐层表 `--kv-layer-storage` 才改变几何
> （`all:int8` → 2112 MiB = 1.83×）。

| 配置 | KV 载荷 MiB | B/token | prefill tok/s | decode tok/s | 针刺 |
|---|---|---|---|---|---|
| all:bf16（**实为 nvfp4**） | 1112 | 17792 | 6424 | 32.4 | 6/8 |
| all:int8（全局，实为 nvfp4） | 1112 | 17792 | 6433 | 32.4 | 6/8 |
| all:nvfp4 | 1112 | 17792 | 6422 | 32.4 | 6/8 |
| all:iso4e（全局，实为 nvfp4） | 1112 | 17792 | 6428 | 32.3 | 6/8 |
| all:int8（逐层表） | 2112 | 33792 | — | — | — |
| 0-7:rk4v4（8 rk4v4 + 8 nvfp4） | 1120 | 17920 | 6214 | 35.4 | **8/8** |
| 0-3:rk4v4,4-7:iso4e（+nvfp4 上 8 层） | — | — | 5827 | 44.9 | **8/8** |
| 0-7:rk4v4,8-15:iso4e | — | — | 6205 | 35.5 | **8/8** |
| 0-7:rk4v4,8-15:int8 | — | — | 7401 | 35.9 | **8/8** |
| 0-3:rk4v4,4-7:int8（+nvfp4 上 8 层） | 1376 | 22016 | 6308 | 45.2 | **8/8** |
| **8-15:rk4v4** | — | — | — | — | **0/8** |
| **all:rk4v4** | 1088 | 17408 | 7786 | 45.5 | **0/8** |
| 0-11:rk4v4 | 1104 | 17664 | 7475 | 49.0 | **0/8** |

## 2. 三条硬结论

1. **KV 池几何只由逐层表决定**（`--kv-layer-storage`），全局 `--kv-dtype` 不进入几何
   （§97）：`--kv-dtype bf16/int8/nvfp4` 三者的载荷逐项相同；逐层表 `all:int8` 才是 1.83×。
   而 bf16 是逐层表的"未设置"哨兵值 → **当前无法启用 bf16 KV**，所以本表缺少真正的
   无损基线（"bf16"行实为 nvfp4）。速度差异也在 ±5%（27B 权重带宽主导）。
   ⚠️ **已推翻 [RK4V4-CONTROL 2026-09-18 vs PATCHSET/RK4V4-CONTROL/REPORT.md]**：bf16 是逐层表的"未设置"哨兵值这一条**不成立**——`--kv-layer-storage 0-15:bf16` 在**那个 fixture 与那个 pin 上**
   得到 `kv cache dtype bf16`、8.50 GiB、27/27。
   ⚠️ **[dl/backlog item1-R2]** 本行原文把 `0-15:bf16` 与 `0-15:fp8` 并列表述为"两者今天都可用"，这一并列必须拆开：
   在 pin `8c566fba8843b15b3a4632e009d5cd936a6981d8db93f64a498639c3d0bd15` 上，配 `examples/cli/messages/long_niah_64k.json`，
   **`--kv-layer-storage 0-15:fp8` 这个拼写被引擎的 cold-host window 守卫按名拒绝**（layer 0 声明 `sliding_window_tokens = 646720`，
   而 fp8 的 decode 路径不读该字段；`ninfer-perplexity --kv-dtype fp8` 被同一守卫以同样方式拒绝）。
   ⇒ **bf16 那一半是 fixture 相关的读数；fp8 那一半在本 pin 上不可复现**。要看复现，**fixture 与 pin 必须和读数一起引用**，
   否则读者无从判断哪一半在生效。原文那句并列在 pin 上是错的；而**bf16 今天无法启用 的推理同样不成立**，
   因为哨兵值不是 bf16 不能用的原因（真正的原因是**当初没测**）。
2. **rk4v4 只在前 8 个全注意力层可用（长上下文）**：完整实证地图见 `_TODO.md` §96。
   ⚠️ **本条规则已被控制重测推翻 [RK4V4-CONTROL 2026-09-18 vs PATCHSET/RK4V4-CONTROL/REPORT.md]**：`{0..12}:rk4v4 → 9/27` 而 `{0..13}:rk4v4`（**多一层**）
   `→ 27/27`，且 rk4v4 在第 **15** 层于两个不同集合里都是 27/27；下面那行 `0-7:rk4v4 = 8/8`
   在本树自己的 8 针探针上重测为 **1/8**。另外 **"27" 不是针数**，是一根 27 字符针的前缀
   （`ORCHID=493817; COLOR=COBALT`）。仍然成立的是：**全 16 层 rk4v4 = 0/27，两次同值（sigma=0），
   且 `--spec none` 下同样失败** ⇒ **结论维持，推理（层号/计数规则）不成立，层成员关系 OPEN**。
   原始逐针数据在 `PATCHSET/RK4V4-CONTROL/`。
   要点：单个 rk4v4 层在任何层号都正常；≥2 个"高层号"rk4v4 层开始掉分（`14-15:rk4v4` 稳定 4/8、
   `8-11:rk4v4` 2/8、`12-15:rk4v4` 0/8），而 `0-7:rk4v4` 8 层全 rk4v4 稳定 **8/8**（两次重复一致）。
   `compute-sanitizer` 零报错 => 逻辑/别名错误，非越界。`8-15:rk4v4` 短上下文生成正常。
   **实用规则：rk4v4 只用层 0-7；上层用 iso4e/int8/nvfp4 补**（`0-7:rk4v4,8-15:iso4e` 与
   `0-7:rk4v4,8-15:int8` 均 8/8）。`kv_bit_budget.py --rk4v4-layers 8` 已内置该约束。
3. **纯档位之间质量无差异**：bf16/int8/nvfp4/iso4e 全部 6/8 且缺的针相同（YPP63F/KRZJ6K）。
   针刺在 32K 不足以区分 4bit 档位质量——需要 PPL（`ninfer-perplexity` 只支持
   bf16/int8/fp8）或更长上下文/更硬探针来建立质量阶梯。

## 3. 位预算分配器（给定 bit，含小数）

`tools/archkit/kv_bit_budget.py`：DP 精确解 `min 质量罚分 s.t. Σbits ≤ B·L`，输出
`--kv-layer-storage` 表。位成本：bf16 16.0 / int8 8.25 / fp8 8.03 / nvfp4 4.50 / rk4v4 4.06 /
iso4e 3.00；质量罚分先验：bf16 0 / int8 0.02 / fp8 0.03 / rk4v4 0.08 / nvfp4 0.30 / iso4e 0.50
（**先验，待 PPL 校正**）。

⚠️ **上面这行位成本已过时，以代码为准**（`src/product/kv_bit_budget.h` 的 `kKvBitBudgetTiers`，
每行都有 `static_assert` 钉着）：**int8 8.25 / fp8 8.50 / nvfp4 4.50 / rk4v4 4.25 /
iso4e 4.50（被人为钉出候选集）**，罚分为 **int8 2 / fp8 3 / rk4v4 8 / nvfp4 30 / iso4e 200**（×100）。
原文的 `fp8 8.03 / rk4v4 4.06 / iso4e 3.00` 与 0–1 尺度罚分都已被后续补丁更正。

16 层示例（`--layers 16`，rk4v4 限前 8 层）：

⚠️ **[TABLEREGEN 2026-09-18] 下表是 `tools/archkit/kv_bit_budget.py` 的逐行输出**（已逐行核对：
七行的 achieved/spec 与该工具完全一致），而该工具的阶梯已分叉（`fp8 803 / rk4v4 406 / iso4e 300`，
权威为 `850 / 425 / 450`）。所以下表的数字**描述的不是引擎的阶梯**。

更要紧的是：**判据本身也在 2026-09-18T14:21:55Z 变了**。本表写的是「`min 质量罚分
s.t. Σbits ≤ B·L`」，而 `src/product/kv_bit_budget.h` 现在声明的是**字典序**——
「MORE BITS first, penalty only on an equal-bit tie」（BUDGETSAT 的饱和规则）。
两条判据给出**不同**的表，所以照着本节的判据去读引擎输出的 `penalty=` 列会得出
「引擎算错了」的结论。列单位与重生成命令见 `tools/archkit/kv_budget_regen.sh`；
本次实测的当前表与逐列单位见 `PATCHSET/TABLEREGEN/REPORT.md`。

| 目标 B | 实际 | 表 |
|---|---|---|
| 3.5 | 3.46 | `0-6:rk4v4,7-15:iso4e` |
| 4.0 | 4.00 | `0-7:rk4v4,8-12:nvfp4,13-15:iso4e` |
| 4.5 | 4.42 | `0-7:rk4v4,8:int8,9-14:nvfp4,15:iso4e` |
| 5.0 | 4.98 | `0-7:rk4v4,8-10:int8,11-15:nvfp4` |
| 6.0 | 5.92 | `0-7:rk4v4,8-14:int8,15:nvfp4` |
| 8.0 | 7.99 | `0:rk4v4,1-15:int8` |
| 16.0 | 16.00 | `0-15:bf16` |

## 4. 按设备定策略（框架就绪）

矩阵脚本把设备名写进 CSV 头（`# device=...`）。换设备只需在目标机上重跑
`_kv_matrix_v3.sh` / `_kv_quality.sh` 并把 CSV 追加成一张跨设备表；分配器的质量罚分表
按设备替换即可。**本机只有一张 5090 D**，其它设备（V100/4090/3090 等）待有机器时补跑。

## 5. 复现

⚠️ **[dl/backlog item2-scope] 本节原文是一次 SCOPE 错误，不是脚本缺失。** [TABLEREGEN 2026-09-18] 只在
`sh/` 与 `sh2/` 两侧查过，就断言四个脚本全部缺失；它们实际在 **`research/scripts/`**。
本次在 pin `8c566fba8843b15b3a4632e009d5cd936a6981d8db93f64a49863958d0bd15` 上逐项复核：

| 脚本 | 状态 | 字节 |
|---|---|---|
| `_kv_matrix_v3.sh` | **PRESENT**（`research/scripts/`） | 3,733 |
| `_kv_matrix_57k.sh` | **PRESENT**（`research/scripts/`）——**本文档从未点名它** | 2,525 |
| `_kv_quality.sh` | ABSENT | — |
| `_rk4v4_probe.sh` | ABSENT | — |
| `_rk4v4_iso4e.sh` | ABSENT | — |

CSV 侧同样被原文说错：`/home/user/kv_matrix_v3.csv` **PRESENT**（737 B），其第 1 行逐字就是
本文档 §4 引用的设备头 `# device=NVIDIA GeForce RTX 5090 D model=qwen3.8-27b ctx=65536 needle_ctx=32768 needles=8`；
`/home/user/kv_quality.csv` **PRESENT**（371 B）——**脚本不在而它的 CSV 在**。

⚠ **第二处更正，§4 的"追加成一张跨设备表"与脚本不符。** 在 USE SITE 读重定向（脚本把路径拼成
变量，grep 一个字面文件名是查不到的）：`_kv_matrix_v3.sh:13` 是 `OUT=${KV_OUT:-/home/user/kv_matrix_v3.csv}`，
`:28` 是 `echo "# device=..." > "$OUT"`——**单箭头 `>`，整表每次运行被截断**。照 §4 的命令跑第二台设备
会覆盖第一台的 CSV。跨设备表只能人工指定 `KV_OUT=` 分开保存后再合并。`_kv_matrix_57k.sh` 根本不写 CSV
（只写 `LOG`），§4 的追加程序对它不适用。

⇒ **本表的三列可以复现**，命令见下（**不要复用默认 `KV_OUT`**）：

```
KV_OUT=/home/user/kv_matrix_v3.<device>.csv bash research/scripts/_kv_matrix_v3.sh
```

第 3 节那张位预算表**可以**一条命令重生成，而且**不再用** `tools/archkit/kv_bit_budget.py`
——该文件自己的头声明它的阶梯已经从权威（`src/product/kv_bit_budget.h` 的
`kKvBitBudgetTiers`）分叉、其 stdout 不得用于决策：

```
# 位预算表：request -> achieved -> penalty -> spec，并对离线解做交叉核对
bash tools/archkit/kv_budget_regen.sh                                # 默认 3.5 4 4.5 5 6 8 12 16
bash tools/archkit/kv_budget_regen.sh --bits 4 4.5 6 8 --cold-cap 8  # 冷池打开
```

它只 include 引擎自己的头、host-only 编译，**不需要 GPU、不需要锁、不需要模型**，因此
任何时候都能跑；输出带 header sha256、日期、判据，并逐预算给出裁决。

```
# 位预算表的历史生成器（已分叉，仅作对照，不要当权威）
python tools/archkit/kv_bit_budget.py --layers 16 --bits 3.5 4 4.5 5 6 8 12 16
```

## 6. 补充实测（2026-09-10 追加）

- `0-3:rk4v4,4-7:iso4e`（+nvfp4 上 8 层）8/8；`0-7:rk4v4,8-15:iso4e` 8/8；`0-7:rk4v4,8-15:int8` 8/8。
  => **iso4e/int8 在层 8-15 正常**，上层损坏是 rk4v4 特有。
- `8-15:rk4v4` 短上下文（8K）生成正常（"杭州是一座以西湖美景…闻名的城市"），长上下文检索 0/8
  => rk4v4 上层问题是**长上下文路径**的静默错误，不是全场景不可用。
- 纯档位（bf16/int8/nvfp4/iso4e）在 32K 检索上都是 6/8 且缺同一批针（YPP63F/KRZJ6K）
  => 针刺不足以区分 4bit 档位质量。

### 6.1 修复 `--kv-dtype` 之后（§97/98）的复测

`--kv-dtype` 修复后（提交 89e2375）逐项复测，结论修正：

| 配置 | 32K 针刺 |
|---|---|
| `--kv-dtype bf16`（真无损基线） | **8/8** |
| `--kv-dtype nvfp4`（全 nvfp4） | **8/8** |
| `--kv-dtype int8` / `iso4e` | 8/8 |
| `0-7:rk4v4`（rk4v4 限低 8 层） | **8/8** |
| **出厂默认表**（10 层 rk4v4，含 8/9/13/14） | **6/8** |
| `8-15:rk4v4` | 0/8 |

- 之前"纯档位 6/8"其实是**出厂默认表**（`--kv-dtype` 当时不进几何，全部回落到模型默认的
  10L-rk4v4 表）——修复后纯 nvfp4/bf16 都是 8/8。
- **出厂默认 KV 表在长上下文上比全 nvfp4 少 2/8 针**（rk4v4 落在 8,9,13,14 等高层号层，与 §96
  的退化区间一致）。默认表的 PPL 优势是在 ctx 4096 上测的，长上下文代价未被覆盖。
- 建议：优先修 §96 的 rk4v4 高层 bug；短期可把默认表 rk4v4 收到 `{0,1,3,4,6,7}`。


