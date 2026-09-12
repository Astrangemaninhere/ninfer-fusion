# S51 · 上游 PR #194 适配（approximate SiLU 的错零）+ 同族全树扫描

交付物：
- `_collab/E8_s51_nvfp4_silu.diff` —— `patch -p1` 补丁，**1 hunk / +17 / −2**，只动 `src/ops/common/math.cuh`；
- `_collab/E8_s51_silu_probe.py` + `_collab/E8_s51_silu_probe.txt` —— numpy float32 模拟探针与原始输出；
- 本报告。附加证据（非要求产物，留在 scratch 供复核）：`_collab/s51_scratch/cross_check.cpp` +
  `cross_check_out.txt`（g++ 主机侧**穷举全部 float32** 的独立交叉验证）、`sweep.sh`/`sweep2.sh`/
  `sweep5.sh` 的原始 grep 输出。
- 本回合 **纯 CPU**：只读源码 + python + `diff`/`patch --dry-run`/`g++`（主机侧，不碰 CUDA）。
  **未写任何 `src/`**（dry-run 前后 `md5sum src/ops/common/math.cuh` 均为 `0ef644e3bd6b721b85b888ee67cac9af`）。
  **未跑 nvcc/ptxas/make**。

---

## 0. 一句话结论（English abstract）

Upstream PR #194's target — `__fdividef` returning 0 once `1 + __expf(-x)` reaches 2^126 — **does not
exist in our tree**: the whole tree contains **zero** `__fdividef` calls, and our device TUs are compiled
without any fast-math switch (`build/compile_commands.json`, all 153 `.cu` entries), so `/` is IEEE and
`expf` is the accurate one. The **same defect** is nevertheless present, with a **different threshold and
a different mechanism**: `expf(-x)` overflows to `+inf` for x ≤ −88.72284 and `x / inf` then returns
−0, while the true SiLU there is a perfectly normal bf16 (−2.6073e-37 = **22.18 × bf16 min normal**).
There is exactly **one** definition of the helper (`src/ops/common/math.cuh:13`) behind **66** device call
sites in 18 files — our tree merged upstream's file-local `swiglu_silu` into a shared `ops::silu`, so the
fix lands on the shared definition and repairs all 66 at once. Exhaustively over every representable
float32 in [−1000, 0): the old form silently zeroes **1,998,749** values, the fixed form zeroes **0** of
the old form's non-zero outputs, and at bf16 level it **rescues 1,145,241 / regresses 0 / perturbs 2,432**
(2.2e-6 of 1.12e9 non-zero points). The sibling `sigmoid` at `math.cuh:15` has the identical divisor shape
and is included as the second (droppable) half of the same hunk.

---

## 1. 缺陷与证据

### 1.1 上游 PR #194 改了什么

`git show pr194`（`ninfer-upstream`，commit `dc108d43`）只改一个文件
`src/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_w4a4_tma.cuh`：

```diff
-__device__ __forceinline__ float swiglu_silu(float x) { return __fdividef(x, 1.0f + __expf(-x)); }
+__device__ __forceinline__ float swiglu_silu(float x) {
+    const float e = __expf(-fabsf(x));
+    const float r = __fdividef(1.0f, 1.0f + e);
+    return (x >= 0.0f ? x : x * e) * r;
+}
```

机制：`__fdividef` 在除数 ≥ 2^126 时返回 0。`1 + __expf(-x)` 在 x < −87.3365（= −ln 2^126）时越过
2^126。我的模拟把该边界定位在 **x = −87.33655**（`E8_s51_silu_probe.txt` §1，`divisor = 8.507086e37`），
与 PR 文本的 “−87.34” 到 5 位有效数字一致。
**但 PR 文本里的 “SiLU 仍是 −9.6e-37” 复现不出来**：该处真值 = −87.33655 / 2^126 = **−1.0267e-36**
（同处 g++ 交叉验证给出 −1.0266e-36）。差 6.6%，不影响结论，但这条数字**不要引用**。

### 1.2 我们树的真实形态（这是本次适配的关键）

我们的文件 `src/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_w4a4_tma.cuh` **没有**自己的 helper。
它第 3 行 `#include "ops/common/math.cuh"`，第 242–245 行调用的是共享的 `ninfer::ops::silu`：

```
src/ops/common/math.cuh:13
__device__ __forceinline__ float silu(float x) { return x / (1.0f + expf(-x)); }
```

（`namespace ninfer::ops`，而调用点在 `ninfer::ops::detail` 内，靠外层命名空间查找命中。）

因此上游那个 hunk **不能照抄**，原因有三，逐条有证据：

1. **没有 `__fdividef`**。全树（含 `third_party`）grep `__fdividef\s*\(` 命中 **0 处**
   （`s51_scratch/sweep_raw.txt` §C）。引入它反而会打破我们 “IEEE 除法” 的既有语义。
2. **没有 fast-math**。`build/compile_commands.json` 里 `.cu` 条目 **153 个**，带 `-fno-fast-math` 的
   **0 个**、带 `-use_fast_math` 的 **0 个**（`sweep3_raw.txt` §M）；nvfp4 swiglu 的 TU 实际命令是
   `nvcc … -O3 -DNDEBUG -std=c++20 "--generate-code=arch=compute_120a,code=[compute_120a,sm_120a]"
   -lineinfo -x cu -c …`。⇒ nvcc 默认 `--prec-div=true --ftz=false`，`expf` 是精确版本。
   （`-fno-fast-math` 只出现在 `tests/**` 的 **主机** C++ 编译里，与 device 侧无关。）
3. **溢出机制不同、阈值不同**。精确 `expf` 在参数 ≈ ln(FLT_MAX) = 88.722839 处溢出为 `+inf`，
   于是 `1.0f + expf(-x)` 在 x ≤ **−88.72284** 变成 `+inf`，`x / inf` 得 **−0.0**（IEEE 保号）。
   比上游的 −87.3365 更负 1.386 —— **我们比上游晚了 1.39 才开始错，但错的形态一样**。

⇒ 补丁形态 = 保留我们的 `expf` 与 IEEE `/`，只把指数折到不会溢出的那一侧：

```cpp
const float e = expf(-fabsf(x));
return (x >= 0.0f ? x : x * e) / (1.0f + e);
```

除数恒在 (1, 2]，对任意有限 x 都不会溢出；唯一可能成为次正规的是 `e`，而它进入**乘法**。

### 1.3 为什么这次不是 “一个 bug 有 N 份拷贝”

- `silu` 的**定义全树只有 1 处**：`grep -rn -E '(float|double)\s+(silu|swiglu_silu|silu_approx)\s*\('`
  在 `src include tests bench apps tools` 内只回 `src/ops/common/math.cuh:13`（`sweep3_raw.txt`）。
- 磁盘上 `math.cuh` 也只有 **2 份且 md5 相同**：build tree 与 mirror 都是 `0ef644e3bd6b721b85b888ee67cac9af`。
- `_orig_quarantine/` 里的 15 个 patch 残留物没有 `math.cuh`（`MANIFEST.txt` + `grep` 均空）。
- 调用点 **66 处、分布在 18 个文件、全部在 `.cu`/`.cuh`**（`sweep5_raw.txt`）：
  `src/ops/kernel` 19 · `src/ops/linear_swiglu` 18 · `src/ops/linear` 16 · `src/ops/sparse_moe` 10 ·
  `src/ops/gdn_input_proj` 3。**改 1 行 = 修 66 处**，与 “硬编码 256 有 81 处” 那族正好相反。

---

## 2. 补丁与 dry-run

`_collab/E8_s51_nvfp4_silu.diff`（`diff --git a/src/ops/common/math.cuh b/src/ops/common/math.cuh`，
`@@ -10,9 +10,24 @@`，1 hunk / +17 / −2 / 30 行，纯 LF）。因为两处改动只隔一行，diff 合成**一个** hunk。

生成器：`_collab/s51_scratch/mkdiff.py`（把目标文件复制到 `/tmp` 后改副本，再 `difflib.unified_diff`；
断言 “旧行恰好出现 1 次”，否则拒绝出补丁）。

```
=== patch --dry-run -p1 inside /home/user/ninfer-fusion ===
rc = 0
stdout: checking file src/ops/common/math.cuh
stderr:

=== patch --dry-run -p1 --fuzz=0 (strict) ===      <-- 严格模式也过（无模糊匹配）
rc = 0
stdout: checking file src/ops/common/math.cuh
```

同一补丁对 **mirror** 也 dry-run 通过：`cd /mnt/c/.../ninfer-fusion-repo && patch -p1 --dry-run --fuzz=0`
→ `rc=0`。dry-run 前后 `md5sum src/ops/common/math.cuh` 均为 `0ef644e3bd6b721b85b888ee67cac9af`（未落盘）。

影子验证（`s51_scratch/verify_shadow.sh`）：在 `/tmp/s51_shadow` 复制该文件、真打补丁 →
`cmp` 与生成器意图文本 **IDENTICAL**，补丁后 sha256 `14660f726fe4db71d5ceb3de639717077210b93d6dfab665b41532b059fa62bb`。

**若只要最小爆炸半径**（只修 SiLU、不动 `sigmoid`）：该 diff 是单 hunk，手工删掉 `sigmoid` 那 7 行
`+` 即可；或直接对旧文件用 `sed` 一行的等价替换。第 4 节给了 sigmoid 的独立证据，是否一并落由 M 定。

---

## 3. 数值证据（CPU-only，numpy float32 / `E8_s51_silu_probe.txt`）

### 3.0 模拟边界（先说清哪些能信、哪些不能）

| 项 | 我怎么代替 | 可信度 |
|---|---|---|
| `expf` | numpy float32 `exp`（精确版，溢出 → `+inf`） | **定位“在哪儿错零”可信**（与 g++/glibc `expf` 独立复算一致，见 3.4）；**逐位复现 CUDA `expf` 不可信** |
| `__expf` | 同上（只用于复算上游形态） | 仅在该形态**有定义**的区间量级可信；`__expf` 有 ~2^-21 相对误差，最后几位不可信 |
| `__fdividef` | CUDA 编程指南语义：除数的绝对值在 `[2^-126, 2^126]` 内走精确除法，越界 → 0 | 用于定位上游阈值，与 PR 文本互证到 5 位 |
| IEEE `/` | numpy float32 除法（**这才是我们树实际用的算子**，见 §1.2 第 2 条） | 可信 |
| bf16 舍入 | “加 `0x7FFF+lsb` 后截低 16 位” 的标准 RNE 花招 | 与另一套 struct 位手术实现对 217 个采样值 **0 mismatch**（`probe.txt` §0b） |

### 3.1 旧式在哪里开始返回精确的 0

| 形态 | 最大被错零的 x | 该处真 SiLU | 相对 bf16 最小正规数 (1.1755e-38) | 相对 bf16 最小次正规 (9.1835e-41) |
|---|---|---|---|---|
| **我们树**（精确 `expf` + IEEE `/`） | **−88.72284** | **−2.607329e-37** | **22.18 ×** | 2839 × |
| 上游基线（`__fdividef` + `__expf`） | −87.33655 | −1.026633e-36 | 87.34 × | 11179 × |

我们树最后一个非零结果出现在 x = −88.72283（值 −2.607349e-37），与真值只差 7.6e-12 相对。
即：**错零不是“反正要变成 0”，边缘被吞掉的值是 bf16 最小正规数的 22 倍**。

### 3.2 要求的表（x = −80, −87, −87.34, −88, −100, −120, −200；另加括号行标出我们的边缘）

「exact」列在**同一个 fp32 输入**上求值（用 double 字面量 −87.34 而非 `f32(−87.34)` 会凭空造出
3.6e-6 的假误差）。

| x | old（我们树） | old（上游） | new（我们的修法） | new（上游修法） | exact SiLU (f64) |
|---|---|---|---|---|---|
| −80 | −1.443881e-33 | −1.443881e-33 | −1.443881e-33 | −1.443881e-33 | −1.443881e-33 |
| −87 | −1.431856e-36 | −1.431856e-36 | −1.431856e-36 | −1.431856e-36 | −1.431856e-36 |
| −87.34 | −1.023139e-36 | **0** | −1.023139e-36 | −1.023139e-36 | −1.023139e-36 |
| −88 | −5.328050e-37 | **0** | −5.328049e-37 | −5.328049e-37 | −5.328050e-37 |
| **−100** | **−0** | **0** | **−3.783506e-42** | −3.783506e-42 | −3.720076e-42 |
| −120 | −0 | 0 | −0 | −0 | −9.201178e-51 |
| −200 | −0 | 0 | −0 | −0 | −2.767793e-85 |
| −88.5 | −3.249987e-37 | 0 | −3.249987e-37 | −3.249987e-37 | −3.249987e-37 |
| −88.72283 | −2.607349e-37 | 0 | −2.607350e-37 | −2.607350e-37 | −2.607349e-37 |
| **−88.72284** ← 我们开始错零 | **−0** | 0 | **−2.607329e-37** | −2.607329e-37 | −2.607329e-37 |
| −90 | −0 | 0 | −7.374608e-38 | −7.374608e-38 | −7.374611e-38 |
| −95 | −0 | 0 | −5.245060e-40 | −5.245060e-40 | −5.245028e-40 |

**bf16（真正离开 kernel 的东西）**：上述 7 个要求点里，我们的旧式**没有一处产生错零** ——
−80 / −87 / −87.34 / −88 高于我们的阈值 −88.72，根本没触发；−100 / −120 / −200 虽已越过阈值，
但真值在 bf16 里本来就是 `-0.0`，所以那里取 0 是对的。**我们与上游的差别恰好落在这张表的空档里**：
上游形态在 −87.34 与 −88 两行把 −1.0227e-36 / −5.3191e-37 打成 bf16 **`-0.0`**，我们没打
（`probe.txt` §2 第二张表）。我们的错零只在 x ∈ (−88.7228, −97.46] 这一段可见，见 3.3。

### 3.3 偏差与 bf16 判决

**a) 两者都非零时的偏差**（题目要求的那一条）

| 网格 | 两者都非零的点数 | max abs(旧−新) | 出现在 | max rel(旧−新) | 出现在 |
|---|---|---|---|---|---|
| x ∈ [−1000, 0]，linspace 4e6（间距 2.5e-4） | 354,891 两者非零 | **5.960464e-08**（0.500 × ulp(1.0)） | @ x = −2.4167 | **3.769334e-07**（3.162 ulp） | @ x = −88.5897 |
| x ∈ [−96, 0]，linspace 9.6e6（间距 1e-5） | 8,872,283 两者非零 | **8.940697e-08**（0.750 × ulp(1.0)） | @ x = −1.7229 | **4.996920e-07**（4.192 ulp） | @ x = −88.6905 |
| g++ 穷举**全部** float32 ∈ [−1000, 0) | 1,118,925,334 两者非零 | **5.960464e-08**（0.5 × ulp(1.0)） | @ x = −2.07909679 | **3.336033e-07**（2.80 ulp） | @ x = −88.6919022 |

⇒ **“两者都有定义时完全相同” 是假的**：相对差最大 3–4 ulp（≈ 3.3e-7 ~ 5.0e-7）。
原因：x < 0 侧新式是 `(x*e)/(1+e)`，比 `x/(1+expf(-x))` 多一次舍入，且边缘处 `e` 落到次正规区
（x = −88.69 时 e = e^−88.69 = 2.99e-39，已低于 fp32 最小正规数 1.1755e-38，次正规量化步长
`2^-149` 在那里就是 ~4 ulp 的相对误差）。3–4 ulp 比 bf16 的 2^-8 = 3.9e-3 小 4 个数量级，
但**不是 0**，必须写明。

精度代价（x ∈ [−20, 0]，与 f64 真值比）：old max rel 2.06 ulp / new max rel 2.55 ulp
（`probe.txt` §4）。**修法最多贵半位**。

**b) bf16 判决 —— 修法到底改动了什么**

| 类别 | 网格 2.8e6（x∈[−140,0]） | g++ 穷举全部 float32 |
|---|---|---|
| **RESCUE**：`bf16(old) == 0` 且 `bf16(new) != 0` | 174,750 | **1,145,241** |
| **REGRESS**：`bf16(old) != 0` 且 `bf16(new) == 0` | **0** | **0** |
| **PERTURB**：两边都非零但差 1 个 bf16 ulp | 19（0.0007% 网格） | 2,432（占 1.12e9 非零点的 **2.2e-6**） |
| x ∈ [0, 60) 逐位不同 | 0 / 600,001 | **0 / 1,114,636,288** |

- 被救回的那一段：**x ∈ (−88.7228394, −97.4611588]**，宽 8.7383 个 x。上端值 = 22.18 × bf16 最小正规，
  下端 = 0.500 × bf16 最小次正规（即真值刚好停在 bf16 的舍入边界上，再负就该是 0）。
- x ≥ 0 侧 `(x>=0 ? x : x*e)` 与旧式**是同一个表达式**，穷举 11 亿个数**逐位不变** ⇒ 正半轴零风险。
- 反向错零 **在所有 11 亿个 float32 上都是 0 处** ⇒ 新式严格支配旧式，不存在“修一处坏一处”。

**c) 修法残留的零点（是否还有同类缺陷）**：新式在 fp32 层面仍会在 x ≤ **−103.9721** 得 0，
该处真值 7.2847e-44 = 52 × fp32 最小次正规 —— 但成因是 **`expf` 自己下溢**（除数没溢出、
也没越 2^126），**不是同一个缺陷**；bf16 层面这一整段本来就该是 0。⇒ 修法是完整的。

### 3.4 独立交叉验证（第二个语言 / 第二套 libm）

`s51_scratch/cross_check.cpp`（主机 g++ −O2，真 `expf`/`fabsf`/`/`，无 CUDA）：

```
expf overflow edge: last finite expf(88.7228317)=3.40279852e+38, first inf expf(88.7228394)
our tree: largest x with silu_old(x)==0 : -88.7228394
  exact SiLU at that edge : -2.607329276e-37        (numpy 探针: -2.607329276e-37)
  bf16 of the swallowed value : -2.600781251e-37
  / bf16 min normal (1.175494e-38) : -22.18 x
FULL float32 sweep over every representable x in [-1000, 0):
  largest x where old == 0 but the true SiLU is a NON-ZERO bf16 (spurious) : -88.7228394
  only-new-nonzero (old zeroed) : 1998749      only-old-nonzero (new zeroed) : 0
```

两套实现独立给出**同一个阈值**（−88.7228394）与**同一个边缘真值**（−2.607329e-37）。
两套的 max-rel 略有差别（numpy 4.997e-7 vs glibc 3.336e-7），说明差别来自各自的 `exp` 末位，
量级都在 3–4 ulp —— 报告按区间 [3.3e-7, 5.0e-7] 引用。

---

## 4. 同族扫描结果（这是重点）

### 4.1 汇总表

| # | 位置 | 形态 | 可达？ | 要不要同样的修 |
|---|---|---|---|---|
| 1 | `src/ops/common/math.cuh:13` | `silu(x) = x / (1.0f + expf(-x))`，除数 x ≤ −88.72284 时 → `inf` | **是**：66 个 device 调用点（见 §5） | **要 —— 已含在补丁里（hunk 唯一主体）** |
| 2 | `src/ops/common/math.cuh:15` | `sigmoid(x) = 1.0f / (1.0f + expf(-x))`，同一个除数 | **是**：11 个 device 调用点（`src/ops/kernel` 5 · `src/ops/gdn_gating_proj` 5 · `src/ops/sparse_moe` 1） | **同族同形态，已作为同一 hunk 的第二段给出；可一行丢弃（见下）** |
| 3 | `src/ops/common/math.cuh:17` | `softplus(x) = (x>20) ? x : log1pf(expf(x))` | 是 | **不要**。x < 0 侧指数参数为负、除数不存在；`expf(x)` 舍入到 0 的落点在 x ≈ −104.0，那里真值 `e^x < 7.01e-46`（半个 fp32 最小次正规）⇒ 得 0 是**正确舍入**。这不是“被除法冲掉”，是子表达式本身到了极限 |
| 4 | `src/ops/kernel/dflash2_selector.cuh:240,248` | `__expf(score - max)` | 是 | **不要**。指数恒 ≤ 0，不可能溢出 |
| 5 | `src/ops/kernel/mtp_round.cuh:88` | `__expf(x - m)` | 是 | **不要**。同上（softmax 减最大值） |
| 6 | `src/ops/kernel/sampling_device.cuh:199` | `__expf(cand*inv_temp - m)` | 是 | **不要**。同上 |
| 7 | `gqa_attention_prefill_{bf16,i8,nvfp4}.cuh`、`causal_cache/prompt_{bf16,fp8,i8}.cuh`、`dense/packed/kernel.cuh`（共 7 文件 14 处） | `l > 0 ? __frcp_rn(l) : 0.0f` | 是 | **不要**。已带 `> 0` 守卫；且 `__frcp_rn` 不是 `__fdividef`，2^126 那条规则不适用 |
| 8 | 全树 | `__fdividef` | —— | **0 处命中**（含 `third_party`）。上游那条机制在我们树里不存在 |
| 9 | `tests/**` 里 7 个文件 | `std::exp` 的**双精度 CPU 参照实现** | 否（host oracle） | **不要**。它们在 `±12` 一类的输入域上求值，且是 double |

判别标准我写成一句可判定的：**“某个子表达式的极限恰好是 0” 不算缺陷；只有 “两个都非零，
但商被硬件冲掉了” 才算**。按这条：`softplus` 不属于这一族（`log1pf(0) = 0` 是它的极限值），
`silu` / `sigmoid` 都属于 —— 它们的分母被冲到 `+inf` 而分子非零，商才有定义（真函数值
非零且可表示）。`sigmoid` 的极限**也是** 0，所以它更接近边界：它之所以仍被算作这一族，是因为
在 x ∈ (−88.72, −92.88] 这一段真值**仍是可表示的 bf16**：

| sigmoid 证据（`probe.txt` §8） | 值 |
|---|---|
| `sigmoid_old(−88.72284)` | **0.0** |
| 该处真值 | **2.9387352e-39** = **32.00 × bf16 最小次正规**（bf16(真值) = 2.9387359e-39，非零） |
| 错零窗口 | x ∈ (−88.7228, −92.8817]，宽 4.159 |
| x ∈ [−100,0] 网格 | 仅新式非零 112,772 点；仅旧式非零 **0** 点；max rel 差 4.767719e-07（4.0 ulp） |
| bf16 | rescue 41,589 / perturb 9 |

### 4.2 “differently-shaped, needs a decision”（不进补丁，列出待定）

| 位置 | 为什么形状不同 | 我的判断 |
|---|---|---|
| `include/ninfer/ops/silu_mul.h:12`、`sigmoid_mul.h:12`、`softmax_attention.h:38` | 只是**契约文档里的公式**，无代码 | 不用动 |
| `bench/ops/w8_linear_swiglu_bench.cu:181` | `"composed.linear+silu"` 是**字符串标签**，不是调用 | 不用动 |
| `tests/ops/*.cpp`（7 个文件） | 双精度 host 参照，域在 ±12 | 不用动。**但见 §6 第 4 条：现有测试覆盖不到这个缺陷** |
| `src/ops/kernel/causal_conv1d.cuh:4` | 注释里写着 `SiLU is computed as x / (1 + exp(-x))` | 注释仍然成立（代数等价），不必改 |

### 4.3 计数命令（可复现）

```
grep -ro --include=*.cu --include=*.cuh -E "(^|[^a-zA-Z_])(silu)\s*\(" src include apps tools | grep -v common/math.cuh | wc -l
# -> 66   （全部落在 .cu/.cuh；.cpp/.h 里 0 处）
grep -ro --include=*.cu --include=*.cuh -E "(^|[^a-zA-Z_])(sigmoid)\s*\(" src include apps tools | grep -v common/math.cuh | wc -l
# -> 11   （include/*.h 里另 3 处是文档注释）
grep -rn --include=*.cu --include=*.cuh -E '__fdividef\s*\(' . | wc -l          # -> 0
```

---

## 5. 可达性：我们 artifact 的哪些 kernel 真的走到这个 helper

调用链（每一跳都有 file:line）：

```
ops::linear_swiglu(..., text_policy(...), ...)          src/targets/qwen3_6_27b/impl/variant.cpp:373
  │  text_policy 在权重格式为 nvfp4 时返回 kNvfp4TextPolicy = LinearPolicy::AllowA4
  │                                                      src/targets/qwen3_6_27b/impl/variant.cpp:64,74
  │  （muse_glimmer_30b 同构：impl/variant.cpp:58,68；另有
  │    src/targets/qwen3_6/impl/runtime/dflash_impl.h:431 直接调 ops::linear_swiglu）
  ▼
ops::linear_swiglu(x, gate_up_weight, out, policy, ws, stream)
                                                         src/ops/wrapper/linear_swiglu.cpp:73
  ▼   (QType::NVFP4 分支)
detail::nvfp4_linear_swiglu_dispatch(...)                src/ops/wrapper/linear_swiglu.cpp:124
                                                         src/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.cpp:123
  ▼   resolve_route():
  │     tokens >= 256 && tokens % 256 == 0  -> TmaFusedW4A4    (plan.cpp:48-50)
  │     5 <= tokens <= 48                   -> FusedW4A4       (plan.cpp:47)
  ▼
launch_nvfp4_linear_swiglu_w4a4_tma(...)                 plan.cpp:141, plan.cpp:178
                                                         src/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_w4a4_tma.cu:56,80
  ▼
nvfp4_linear_swiglu_w4a4_tma_kernel<Geometry, M256N128S3>
                                                         .../nvfp4_linear_swiglu_w4a4_tma.cuh:49
  ▼
silu(gate[i] * alpha)                                    .../nvfp4_linear_swiglu_w4a4_tma.cuh:242-245
  ▼
ninfer::ops::silu  ==  src/ops/common/math.cuh:13        （该 .cuh 第 3 行 #include "ops/common/math.cuh"）
```

- **nvfp4 W4A4 prefill（T=256 的整数倍，T ≥ 256）**：走 TMA kernel，`silu` 4 处/线程（242–245）。
- **nvfp4 W4A4 小 T（5–48）**：走 `nvfp4_linear_swiglu_w4a4.cu:47,48`，同一 helper。
- **T = 1**：`nvfp4_linear_swiglu_decode.cu:61`；**T = 2..4 或 5..16**：`small_t.cu:70`。同 helper。
- 同一 helper 还被 **`silu_mul`**（`src/ops/kernel/silu_and_mul.cuh:19,20,29,59,117`）用到，
  而 `plan.cpp:157`（`LinearW4A4Post` 的 tail 分支）与 `plan.cpp:193`（全 baseline 分支）都调
  `silu_mul` —— 也就是说**同一次 dispatch 的每条分支最终都落到同一个 helper**，
  不存在“修了融合路径、漏了回退路径”的风险（这是 `silu_mul` 与融合 kernel 共享定义带来的额外收益）。
- 另有 `causal_conv1d`（10 处）、`gated rmsnorm`（`src/ops/kernel/rmsnorm.cuh:24`）、
  `gdn_input_proj`（3 处）、`sparse_moe`（10 处）、w8/q4/fp8 的 linear_swiglu —— 全在同一行上被一起修掉。

**参数方向**：`alpha = 1.0F / (weight.input_scale_divisor * weight.weight_scale_divisor)`
（`plan.cpp:139,176`）是正标量，所以 `gate[i]*alpha` 的符号 = `gate[i]` 的符号；`silu` 的参数确实可以
走到很负。**是否真的走到 ≤ −88.72 是数据问题**，见 §6 第 2 条。

**影响上界（闭式）**：错零时输出差 `|SiLU(x)| × |up·alpha|`。在 x = −88.72284 处
`|SiLU| = 2.6073e-37`，只要 `|up·alpha| ≳ 1.8e-4`，这个差就是一个**非零的 bf16**
（bf16 最小次正规 9.18e-41）。⇒ 这不是“反正要变 0”的注水，而是**该出数的位置出了 0**。

---

## 6. 残余不确定（未证实项，不要当成已验证）

1. **sm_120a 上 `expf` 的确切溢出落点 —— 待证实。** 我用两套主机实现（numpy float32 `exp`、glibc
   `expf`）独立定位到同一个 `88.7228394`（= ln(FLT_MAX) 向上取到 fp32 网格），但**本回合没有跑
   nvcc/ptxas，也没有 GPU**，没有在 sm_120a 上实测。CUDA `expf` 的文档保证是 max ulp error 2、
   溢出返回 `+inf`，落点应当一致；但若 sm_120a 的 `expf` 落点好在 1–2 ulp 之外，**边界 x 会平移
   1e-5 量级，结论不变**（真正会变的是“最大被错零的 x”这个小数的第 6 位有效数字）。
   同理，`__fdividef` 的 2^126 规则来自编程指南文本，我也没在硬件上复现 —— 上游的 −87.33655 是
   我的**模拟**结果。
2. **我们的 kernel 到底有没有真的踩到 —— 待证实（无 GPU）。** 需要数 `gate*alpha ≤ −88.72284` 的
   实例，或对某一层 dump `gate` 的 min。没有这个数，就**不能**声称我们的 artifact 已经出过错。
   可做的最小实验：在 `nvfp4_linear_swiglu_w4a4_tma.cuh:242` 前加一个 `__any_sync(gate[i]*alpha < -88.0f)`
   的计数器（一次 prefill 就能给答案），或离线对 `weight.input_scale_divisor/weight_scale_divisor`
   反推 `alpha` 量程。**本报告只证明“缺陷存在且在路径上”，不证明“已经发生”。**
3. **上游的 “SASS 完全相同（40 条 / 2 MUFU / 无 CALL）” 对我们的补丁不成立 —— 待证实。**
   那是**他们的**形态（`__expf` + `__fdividef`）的结论。我们保留了精确 `expf` 与 IEEE `/`，新式
   相对旧式多一次条件乘法与一次 select（预计 ~1 FMUL + 1 SEL，`expf` 与除法次数不变）。没有
   nvcc/ptxas，**我没有 SASS 证据**。
4. **现有测试证明不了这条修复 —— 已验证的负面结论。** `tests/ops/test_silu_mul.cpp:17` 的 gate
   取值范围是 `[−12, 12]`，`tests/ops/linear_swiglu/linear_swiglu_test_common.cpp:37` 的 A4 判据是
   `{1.6e-1, 1.0e-2, 1.6e-1}` —— **都够不到 −88，也都远大于 1 个 bf16 ulp**。⇒ 落补丁后跑现有测试
   **既不会红也不会绿**：需要一个新增的极端负值用例（思路：把 gate 造成 −90/−95/−100 的定值，
   与 f64 双精度参照逐元素比），否则这条修复没有任何回归门。**建议与 M 的 “移植
   `speculative_page_boundary.h` 边界用例思路” 一起排。**
5. **PR 文本的 −9.6e-37 复现不出来**：我算 −1.0267e-36（0.5 ulp 内两套实现一致）。不影响结论；
   引用时请用我的数。
6. **`sigmoid` 要不要一起改 —— 需要 M 定。** 它同族同形态（§4.1 第 2 行有 32 × bf16 最小次正规的
   证据），但影响面含 MoE 路由（`src/ops/sparse_moe/sparse_moe_route.cuh:80` 的共享专家门）与 GDN
   门控（`src/ops/kernel/gdn_gating.cuh:26` 等），一旦落批会扰动这些 kernel 的末位。我把它放进同一个
   hunk 是因为“同缺陷 + 机械改”，**但它比 silu 更偏 judgment call**。要最小爆炸半径就删掉那段 7 行 `+`。
7. **`sfu_rcp`/`__frcp_rn` 家族的除数上界未逐个推**：树里 14 处 `__frcp_rn(l)` 都带 `l > 0` 守卫，
   我没能证明 `l` 一定有上界（比如 softmax 的行和理论上可以到 2^126 以上吗？）。按 2^126 规则
   `__frcp_rn` 与 `__fdividef` 不同（`__frcp_rn` 是精确倒数，不 flush），所以**判定为“不需要”**，
   但这个“不需要”是基于指令语义而非实测 —— 记为待证实。

---

## 7. 建议

1. **落这个补丁**（1 hunk / +17 / −2 / 单文件 / dry-run 双树 rc=0）。它是零成本正确性修复：
   `expf` 与除法次数不变，代价 ~1 FMUL + 1 SEL，正半轴逐位不变（穷举 11 亿个 float32 验证）。
2. **同批加一个回归测试**（§6 第 4 条）——否则这条修复没有门，下一次重构会把它悄悄改回去。
3. **`sigmoid` 的取舍由 M 定**；若保守，只取 hunk 的第一段（silu），把 sigmoid 留到有 GPU 窗口时
   单独落、单独看 MoE 路由与 GDN 的末位变化。
4. §6 第 2 条那个 `__any_sync` 计数器是**最便宜的一次性实验**（一次 prefill 出数字），能把
   “缺陷在路径上” 升级为 “我们的 artifact 踩没踩到”。建议排在下一个 GPU 窗口里顺手做掉。
