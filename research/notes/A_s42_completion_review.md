# A/S42 — 两条 completion 的独立复核 + 下一个 Muse 缺陷排序 (2026-09-10 15:0X)

复核对象: 落树态（`/home/user/ninfer-fusion`，非 `A_s36_headdim.diff`——那份**已过期**，见 §5）
① `gqa_attention_decode_i8.cuh:78` `Groups = D / kGqaKvQuantGroup`
② `gqa_attention_decode.cu:532/:548` 的 `if constexpr (HeadDim == kGqaKvQuantHeadDim)` 门
纪律: CPU only、未写 `src/**`、未碰 GPU；本文件 + `A_s42_i8_stride_fix.diff`(未应用) + `A_s42_mkfix.py`
（另有 `A_s42_ctx/` = 我复核时用 Read 工具读的 5 个源文件**只读快照**，**不是补丁源**，勿从它取文件）。

---

## 0. 结论（先给判词）

- **② 的机制对、范围不够**：`if constexpr` 确实阻止 128 侧实例化，256 侧语义逐字不变（§2.1）。
  但它只堵了 **decode**；同一个 nvfp4 模板在 **prefill** 侧被 `cudaFuncSetAttribute` **强制实例化**，
  而那里 `Mxf4QKKs = D/64` 仍是 4 的静态断言 ⇒ **window J2 会在 `gqa_attention_prefill.cu` 编译期失败**
  （且该 `.o` 是 09:31 的陈旧对象，必重编）。这是**当前唯一会挡住整个验收的事**，见 §2.2。
- **① 的取值对、实现不完整**：`Groups=2` 与 kernel 自己的 tile 循环/寻址**全部自洽**（§1），
  但**三处仍按 `Groups==4` 写死**：`cp_async<8>`/`store_vec(make_int2)` 的 8 字节宽度（**smem 越界 4 B +
  跨键覆盖竞态**）、`cold_i8_decode_row` 的 256 写死（**栈越界 128 B + 4 B**）、
  以及**不在你改的两个文件里**的平面 stride 家族 `gqa_attention_kv_quant.cuh:47/:54/:103`（256 家族，
  对 Muse 是 2×；i8/E8 档整体仍坏）。修正补丁见 §3（只含前两类，第三类必须与 prefill_i8 几何化耦合）。
- **验收 harness 会假 PASS**：`_muse_serve_accept.sh` 只数 `NAN` 行且不看 text/HTTP ⇒ 现在 Muse+nvfp4
  会变成"显式拒绝"（好事），但脚本会把它记成 `nan_lines=0` = **PASS**；崩溃（`cudaErrorIllegalAddress`，
  13:44–13:48 的四份 Muse 日志全是这个）同样会被记成 PASS。见 §2.4。

---

## 1. `Groups = D / kGqaKvQuantGroup` 逐点对账

`D = Geometry::HeadDim`（:75），`kGqaKvQuantGroup = 64`（`gqa_attention_kv_quant.cuh:41`）⇒ 128→**2** / 256→4。

| 站点 | 判词 | 依据（算术） |
|---|---|---|
| 定义 `i8.cuh:78` | ✅ | `D/kGqaKvQuantGroup`；Muse 32q/2kv `GroupSize=16`（`gqa_attention_geometry.cuh:27`） |
| `static_assert(QKKs == Groups*GroupKc)` `:99` | ✅ | `QKKs=D/32`、`GroupKc=kGqaKvQuantGroup/32=2`；128: 4==2·2 / 256: 8==4·2 |
| QK tile 循环 `:562`(`g<Groups`)、`:566`(`k=g*2+kk`)、`:579-581` | ✅ | `k∈[0,QKKs)` 且 `bcol=k*16+b_koff ≤ 3*16+8=56 < DB16=D/2=64` |
| Q 量化 `:216-218`、`:329-332`、`:346` | ✅ | `pair/unit` 分解 `token=pair/Groups, grp=pair-token*Groups`；`d0=grp*64+lane`、`d1=d0+32` ⇒ d∈[0,128) |
| q_scale 寄存器 `:360-374` | ✅ | `float q_scale_r0[Groups]`，`q_scale_tmp[row*Groups+g]` 与写入侧同式 |
| smem 尺寸 `:118-119` | ✅ | `k_scale_s[Bc*Groups]`（Muse Bc=32 ⇒ 64 half）与寻址 `:418/:427/:594/:694` 的 `key_l*Groups+g` 一致 |
| V 反量化组选择 `:692-695` | ✅ | `grp=d>>6`，`d=dc*8<D` ⇒ `grp<Groups`；`__shfl(vs, grp*8)` 的 lane `grp*8 ≤ 8`(<32) |
| cold `row_scales[...]` **声明** `:409` | ✅ | `HeadDim/kGqaKvQuantGroup`=2，与 `:417/:426` 的 `g<Groups` 一致 |
| **`cp_async<8>` `:450-451`** | ❌ **8 B=4 half 写死** | Muse: `key_l=31` → half `[62,66)`，数组 `k_scale_s[64]` ⇒ **越界 2 half=4 B**；且 `key_l` 的拷贝覆盖 `(key_l+1)` 的前 2 槽（映源是**本键**平面的第 2/3 组→陈旧值），与 `(key_l+1)` 自己的拷贝**同址不同值 ⇒ cp.async 完成序未定义 ⇒ 间歇错尺度**（与旧缺陷同一间歇签名） |
| **`store_vec(..., make_int2(0,0))` `:438-439`、`:453-454`** | ❌ 同上 | 8 B 清零同样越界 4 B / 覆盖邻键；正确宽度=`Groups*sizeof(__half)` |
| **`cold_i8_decode_row`** `:410/:420` | ❌ **256 写死**（在 `cold_i8_kernels.cuh:72-95`） | 被调用者写 `codes_out[0..255]`、`scales_out[0..3]`；调用者数组是 `row_codes[HeadDim=128]`、`row_scales[2]` ⇒ **栈越界 128 B + 4 B**（前 128/2 个值本身是对的） |
| **平面 LeadingExtent** `kv_quant.cuh:47/:54/:103` | ❌ **不在本次改动文件内** | 见 §1.2 |

### 1.1 三个 ❌ 的算术

1. **smem 尺度槽**（`k_scale_s`/`v_scale_s[Bc*Groups]`）:
   旧 `cp_async<8>`/`make_int2` = 4 个 `__half`；`Groups=2` 时每键只需 2 个。
   `Bc=32`：`key_l=31` 写到 half `[62,66)` ⊄ `[0,64)` ⇒ 越界 2 half；`[62,64)` 是 `key_l=31` 的合法槽、
   `[64,66)` 落到数组外（`__shared__` 相邻对象/填充，取决于 ptxas 布局）。
   256 侧（`Bc=64`, `Groups=4`）旧宽度恰好等于一条键行 ⇒ **无越界**（§3 的补丁在该侧宽度不变）。
2. **cold 行解码**（`cold_i8_kernels.cuh:75-95`）：外层 `for (g=0; g<4; ++g)`、内层 `for (i=0;i<64;i+=2)`
   ⇒ 写 `d∈[0,256)`；scale 写 `scales_out[0..3]`。调用者数组按 `HeadDim=128`/`HeadDim/64=2` 声明 ⇒
   写穿 128 B（codes）+ 2 half（scales）。*注意：内核侧读侧（`:411-428`）只消费 `d<HeadDim`、`g<Groups`，
   值是对的；错的是"写穿"本身。*
3. **平面 stride**（§1.2）：写死 256 家族的那 3 个 helper 与**布局侧分配**（`decoder_state.cpp:106-116`
   用 `head_dim`、`head_dim/2`、`head_dim/group`）在 256 相等、在 128 **差 2×**。

### 1.2 为什么第三类不能用"自洽"免责（重要，且不在你的两条 completion 里）

`paged_kv_element_offset<L,H>(page,head,off,d) = L*64*(head + H*page) + L*off + d`
（`paged_kv_address.cuh:23-32`），`L` 就是**每 token 的 leading 宽度**；tensor 形状 `{L,64,H,pages}`
（`paged_kv_cache.cpp:74`）与之逐字对应。于是：

| 平面 | 分配（`decoder_state.cpp`） | kernel helper | 128 时的偏差 |
|---|---|---|---|
| I8 codes | `head_dim`=128（:106） | `kGqaKvQuantHeadDim`=256（`kv_quant.cuh:47`） | 2× |
| I8/E8 scales | `head_dim/64`=2（:108/:115） | `kGqaKvQuantGroups`=4（`kv_quant.cuh:54`） | 2× |
| E8 K codes(i4) | `head_dim/2`=64（:113） | `kGqaKvQuantHeadDim/2`=128（`kv_quant.cuh:103`） | 2× |

最大索引 ≈ `2×` 平面元素数 ⇒ 后半段 (head,page) **越出平面**写进池内相邻平面（同层 v 平面 / 下一层），
末尾越出池。**写侧(prefill fill, `prefill_i8.cuh:100-104` + 其 page-fill z 维 `prefill.cu:236`/`prefill_e8.cu:56` = `kGqaKvQuantGroups`)
与读侧共用同一错误 stride ⇒ 往返自洽**（这也解释了为什么单测/往返看不出），但越界写是真的。
这正是"**prefill 不得自动免责**"的实体：`prefill_i8.cuh:367` 的 `D = kGqaPrefillHeadDim` 仍 256、
`Groups = kGqaPrefillI8Groups = 4`（:371，`static_assert(kGqaPrefillI8Groups == 4)` :49）⇒ i8/E8 的
**prefill 整条 256-only**（无编译期保护，静默 OOB）。所以修 helper 必须与 `prefill_i8` 几何化**同批**，
否则 128 侧 prefill(2×写入) 与 decode(1×读取) 失配——我**不**把这类放进小补丁（§3 末尾）。

---

## 2. `if constexpr` 门复核

### 2.1 真的阻止实例化吗 → 是；256 侧变了吗 → 没有

- 规则：丢弃分支在**包含它的模板实例化时**不实例化（[stmt.if]/2）；`launch_tc_partial_nvfp4<Geometry,…>`
  依赖 `Geometry`，条件 `Geometry::HeadDim == kGqaKvQuantHeadDim` 实例化后成为常量 ⇒ 128 侧**不实例化**，
  于是 `static_assert(QKKs == 4)`（`gqa_attention_decode_nvfp4.cuh:167`，`QKKs=D/64`）在 128 下**不会触发**。
- 必要性（不是洁癖）：128 时 `QKKs=2` ⇒ 该断言**必然失败**，所以没有门就是编译错误（不是"跑错数"）。
  其余结构证据：`nvfp4.cuh:149` 自己的 `Groups = HeadDim/kGqaKvNvfp4Group` 说明"几何派生"就是该内核惯例；
  `kTileBytes = 4*Bc*128 + 4*Bc*16`（`:174`）、`q_a[Br*128]`（`:180`）都是 64 宽 tile 的 256 前提。
- 256 侧：条件恒真 ⇒ 语句块原样；被调用的模板实参、参数列表逐字未动；`else` 分支（`:537-540/:553-556`）
  为丢弃语句，只要求可解析（`require_nvfp4_geometry_dim` 定义在 `decode.cu:27`，早于宏 `:514`）⇒ 生成码不变。
  `(void)nvfp4_k; (void)nvfp4_v;` 只为丢弃分支消警告，不影响取用分支。

### 2.2 ❌ 缺口 1：prefill 侧同一个模板被强制实例化 ⇒ **J2 构建必失败**

证据链（全部落树）：
1. `gqa_attention_prefill.cu:36` `gqa_attention_prompt_attention_launch_for<Geometry, Metadata>`
   在函数体**开头**取址：`:54-57` `cudaFuncSetAttribute(gqa_attention_prefill_nvfp4_kernel<Geometry, Metadata, DType::NVFP4>, …)`，
   `:59-62` 同 `<…, DType::NVFP4, DType::ISO3>`，`:63-73` 还有 FP8/ISO3 变体 ⇒ **odr-use ⇒ 强制实例化**（runtime `if` 挡不住）。
2. 该函数被 Muse 直接调用：`:377-378`、`:431-433`（`<GqaMuseGeometry>`）。
3. `gqa_attention_prefill_nvfp4.cuh:1002` `constexpr int D = Geometry::HeadDim;`（A 的 S36 diff 从
   `kGqaPrefillHeadDim` 改来，见 `A_s36_headdim.diff` 的 prefill-nvfp4 hunk），`:1024` `Mxf4QK = (KVDType==NVFP4)`，
   `:1025` `Mxf4QKKs = D/64`，`:1026` `static_assert(!Mxf4QK || Mxf4QKKs == 4);` ⇒ 128: `2 != 4` **触发**。
4. 必然重编：`build/src/CMakeFiles/ninfer_ops.dir/ops/launcher/gqa_attention_prefill.cu.o` mtime **09:31:48**，
   `src/ops/launcher/gqa_attention_prefill.cu` mtime **14:45:23**（J2 脚本只 `rm` 了两个 decode `.o`，`:27-28`）
   ⇒ J2 的 `make` 必编它；`_window_j2.sh:41` 三次重试后 `MAKE_FAILED`、**不验证**。

预期日志形状（给 M 的判据）：`gqa_attention_prefill_nvfp4.cuh(1026): error: static assertion failed`（含 `Mxf4QKKs == 4`）。
最便宜的**事前**一条命令（不必等编译）：
`wsl.exe -e bash -c "cd /home/user/ninfer-fusion && sed -n '1024,1026p' src/ops/kernel/gqa_attention_prefill_nvfp4.cuh"` + `echo head_dim=128`（Muse `config.h:33`）⇒ 手工算术 128/64=2≠4。
**修法同 ②**：在 prefill 的两处 attr/launch 外包 `if constexpr (Geometry::HeadDim == kGqaKvQuantHeadDim) {…} else { require_nvfp4_geometry_dim(Geometry::HeadDim); }`（判词与 decode 相同：bf16/int8 可用、nvfp4 未移植到 128）。

### 2.3 ❌ 缺口 2：第二份 dispatch 宏副本没有门（今天不在构建里，但 C/S40 正落在它上面）

`src/ops/launcher/gqa_attention_decode_impl.cuh:489-503` 是宏的**另一份活体副本**（`:491`/`:502` 两处
`launch_tc_partial_nvfp4`，**无门**），被 `gqa_attention_decode_muse.cu:15`（`:25` 实例化 `<GqaMuseGeometry>`）
与 `gqa_attention_decode_g35.cu:4` include。今天不炸是因为 `src/CMakeLists.txt:77-78` 只列了
`gqa_attention_decode.cu` / `gqa_attention_decode_e8.cu`（**没有** muse/g35 TU），而
`_collab/C_s40_tu_split_v2/`（14:40，plan/patch/review 目录还空着）的业务正是把实例挪进这些 TU
⇒ **S40 落树前必须把门镜像进 `impl.cuh`，否则 Muse 一编译就回到 `QKKs==4` 断言。**

### 2.4 ❌ 缺口 3：验收 harness 会把"拒绝"和"崩溃"都判成 PASS

- `_muse_serve_accept.sh:52` 只 `grep -c 'NAN'`，`:62-69` 的 verdict 把 `0` 判 PASS、**完全不看** `text`/HTTP，
  只在 `SERVE_FAILED|MISSING` 才 FAIL。⇒ ① Muse+nvfp4 现在会在**第一次 decode 抛** `invalid_argument`
  （消息里没有 "NAN"）⇒ `nan_lines=0` ⇒ **PASS**（假 PASS，且掩盖了"这一档现在根本不出数"）；
  ② serve 在 health 之后崩（`/home/user/musee8_*.log` 13:44–13:48 四份全是
  `device.cu:126 CUDA_CHECK(cudaStreamSynchronize) failed: cudaErrorIllegalAddress`，`NAN`=0 行）同样会被判 PASS。
- `_e8_muse_check.sh` 反而**是**健壮的（`:44-47` 要 `needle hits N/M`，`:64` 无计数即 FAIL）——但 `:55-57`
  三档以 `--kv-dtype nvfp4` 为底 ⇒ 门落地后必 `SERVE_FAILED` ⇒ `MUSE_E8_VERDICT=FAIL`。
  **这是预期结论（"档位不存在"），不是 256 修复失败**；要么把这三档换成 `--kv-dtype bf16|int8` 底，
  要么在 verdict 里把"显式拒绝"单列一档。

---

## 3. 修正补丁（**未应用**，已 dry-run/反向/drop-in 证明）

`_collab/A_s42_i8_stride_fix.diff`（2 文件，由 `A_s42_mkfix.py` **锚定生成**，锚点唯一性失败即 abort；保留 EOL）。
内容：`i8.cuh` 新增 `gqa_i8_scale_row_clear<Groups>()` + 4 处清零宽度、2 处 `cp_async<Groups*sizeof(__half)>`、
2 处 `cold_i8_decode_row<Groups>`；`cold_i8_kernels.cuh` 加 `template <int Groups = 4>` 并把外层 `g<4` 改 `g<Groups`。

- 生成器实测输出：`dry-run rc=0 | apply rc=0 | shadow==new: True | reverse-dry rc=0`，`S42_FIX_OK`。
- 算术（同一次运行打印）：
  `Muse  head_dim=128 Bc=32 Groups=2  array=64 halves | 旧 8B 写 key_l=31 → [62,66) = 2 halves OOB | 新 4B → [62,64) OK`
  `qwen  head_dim=256 Bc=64 Groups=4  array=256 halves | 旧 8B 写 key_l=63 → [252,256) = 0 halves OOB | 新 8B → [252,256) OK`
- 256 侧为什么"语义逐字同"：`Groups==4` 时 `cp_async<8>` 仍是 8 B；清零改由 4 次 half 存 0 完成，
  写到的**字段集合完全相同**（可能少一条 8 B 合并存，语义/可见值不变）。
- **不含**（必须另开一批、且要与 `prefill_i8` 几何化耦合，理由见 §1.2）：
  `gqa_attention_kv_quant.cuh:47/:54/:103` 的 LeadingExtent → `Geometry::HeadDim` / `HeadDim/kGqaKvQuantGroup` /
  `HeadDim/2`，以及 `prefill_i8.cuh:26-41/:367/:371` 的 256 常量族。

---

## 4. 下一个 Muse 缺陷排序（含一条命令）

| # | 项 | 何时会咬 | 一条命令 / 判据 | 概率 |
|---|---|---|---|---|
| 1 | **J2 在 `prefill.cu` 编译失败**（§2.2：prefill nvfp4 的 `Mxf4QKKs==4`） | **现在**（本次构建内） | `wsl.exe -e bash -c "grep -n 'error:' /tmp/wj2_make.log\|head"` 期望 `gqa_attention_prefill_nvfp4.cuh(1026) static assertion failed`；事前版见 §2.2 | **确定** |
| 2 | **验收语义不可读**：`_muse_serve_accept.sh` 假 PASS + `_e8_muse_check.sh` 三档 nvfp4 底必 FAIL（§2.4） | 构建修好之后立刻 | `sed -n '52p;62,69p' _muse_serve_accept.sh`；`sed -n '55,58p' _e8_muse_check.sh` | **确定** |
| 3 | **Muse 的 I8/E8 档仍整体 2×**（§1.2 三处 stride + `prefill_i8` 256-only + §1 的两个宽度 bug） | 任何 `--kv-dtype int8`/e8 层跑起来时（换掉 nvfp4 底之后） | `sed -n '40,56p;100,105p' src/ops/kernel/gqa_attention_kv_quant.cuh` 与 `sed -n '106,116p' src/targets/qwen3_6/impl/state/decoder_state.cpp` 对照（256/4 vs 128/2）；症状=与今天同族（garbage/NaN/illegal address） | 高 |
| 4 | **SWA 对 Muse 默认 bf16 档完全无效**（窗口只在 iso3/nvfp4 内核里） | >2048 token 的长上下文语义（32K/57K 针尖） | `grep -c sliding_window src/ops/kernel/gqa_attention_decode_bf16.cuh src/ops/kernel/gqa_attention_decode_i8.cuh src/ops/kernel/gqa_attention_prefill_bf16.cuh` ⇒ 全 0；对照 `grep -n sliding_window src/ops/kernel/gqa_attention_decode_nvfp4.cuh`（有）。运行判别：`--no-cuda-graph` 下同 prompt 改**第 0..K 个 token**（K<末尾−2048）看末尾输出是否变；变=没窗 | 中 |
| 5 | U7 Muse page-fill E8 | **代码已修**（`prefill_i8.cuh:282-285` 旋转+/7+投影，参照位 `:142-145`），只剩"改前基线/改后复跑"的判定 | `sed -n '276,300p' src/ops/kernel/gqa_attention_prefill_i8.cuh`；判定仍依赖 #2/#3 解开后才能跑 e8 档 | 中低（且是测量阻塞、非新缺陷） |
| 6 | 行标定表对 Muse 0-15 层用 qwen 表 | 只有 nvfp4 档消费；守卫落地后**当前不可达** | `sed -n '20,31p' src/ops/kernel/gqa_isoquant_row_scale.cuh`（`layer<16 && kv_head<4 && d<256` 才查表 ⇒ Muse 0-15 全命中）；消费点 `nvfp4.cuh:323/:477`、`prefill_nvfp4.cuh:531/:766/:1105` | 低（nvfp4 回来后是"精度小差"而非坏） |
| 7 | `v_dtype=ISO3` 映射不对称 | 同一 nvfp4 档在"全局 `--kv-dtype nvfp4`"=原生 E2M1-V、在"任一逐层 dtype 表"（如 `--kv-layer-storage`/预算表）=ISO3-V | `sed -n '316,322p;374,380p' src/targets/qwen3_6/impl/state/decoder_state.cpp`；写/读两侧都按 `cache.v_dtype`（`prefill.cu:118/:272`、`decode.cu:525/:541`）⇒ **不是坏**，是 A/B 可比性与精度的混淆项 | 低 |

说明：#1/#2 是"现在就会发生"，#3 是**下一个真引擎缺陷**（若 J2 后看到 Muse+e8 崩，先归这里，别当成回归），
#4 是静默语义项（无 NaN，靠"和参考不一致"暴露），#5 是测量被 #2/#3 阻塞，#6/#7 当前被 nvfp4 守卫掩蔽。

---

## 5. 给 M 的最便宜核对（三条，全部 CPU）

1. 我的三处 ❌：`wsl.exe -e bash -c "cd /home/user/ninfer-fusion && sed -n '436,456p' src/ops/kernel/gqa_attention_decode_i8.cuh"`（看 8 B 宽度）+
   `sed -n '72,95p' src/ops/kernel/cold_i8_kernels.cuh`（看 `g<4`）+ `sed -n '40,56p;100,105p' src/ops/kernel/gqa_attention_kv_quant.cuh`。
2. 补丁复算：`wsl.exe -e bash -c "python3 /mnt/c/Users/User/Documents/ziqinzhang/_collab/A_s42_mkfix.py"` ⇒ 期望末行 `S42_FIX_OK`（内含 dry-run/反向/shadow 三项与算术打印）。
3. J2 失败预测：`wsl.exe -e bash -c "tail -40 /tmp/wj2_make.log"`（或 `dl/window_j2.log`）⇒ 期望 `prefill_nvfp4.cuh(1026)` 断言；`_window_j2.sh:41` 会打印 `MAKE_FAILED rc=… — 不验证`。

**警告（已实测）**：`_collab/A_s36_headdim.diff` 与落树态**已完全漂移**——`cp -a src → /tmp/a42b` 后
`patch -p1 --dry-run -f < A_s36_headdim.diff`：**13/13 文件、53 hunks 全 FAILED**（其中 i8 hunk 的
`Groups = kGqaKvQuantGroups` 现在是**上下文行**，`i8.cuh:78` 已变成 `D / kGqaKvQuantGroup`；两条 completion
的门也不在该 diff 里）。⇒ **该 diff 不可重放**（重放会顶掉两条 completion），要重放先 `A_s36_mkpatch.py` 重生成。
