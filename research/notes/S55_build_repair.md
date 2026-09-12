# S55 — build repair loop (make ninfer / make ninfer-serve)

Tree: `/home/user/ninfer-fusion` (WSL, canonical build tree).
Mirror `/mnt/c/Users/User/Documents/ziqinzhang/ninfer-fusion-repo` is read-only reference.
Constraint: `-j1` only (22GB box, prior OOM/reboots from parallel compile).

Entry format: 错误原文（首行）→ 文件:行 → 归类 → 修法与理由 → 是否回退/注释

---

## R1

### 1-1. `parse_kv_layer_storage` not visible in `ninfer::product`

- 错误原文: `src/targets/qwen3_6/impl/runtime/layouts_impl.h:1039:37: error: 'parse_kv_layer_storage' is not a member of 'ninfer::product'`
  (log line numbers 1039/1113 are pre-rollback; after the S24 rollback the same sites sit at 1022/1096 —
  `layouts_impl.h` is now byte-identical to `_orig_quarantine/.../layouts_impl.h.orig`, md5
  `90a9bbc975cb4579102f4fcbbab6f080`, 1153 lines, `diff -q` rc=0.)
- 文件:行 → `src/targets/qwen3_6/impl/runtime/layouts_impl.h:22` (include block)
- 归类: **消费者已有、生产者缺失**（missing include）
- 修法: 加一行 `#include "product/kv_options.h"`，紧跟已有的 `#include "product/kv_bit_budget.h"`。
- 理由: 调用点用的 `product::kv_bit_budget_spec` 来自 `product/kv_bit_budget.h`（已 include，故无报错），
  而 `product::parse_kv_layer_storage` 定义在 `product/kv_options.h:43`；该头文件从未被 include，
  且 `kv_bit_budget.h` 自己并不 include `kv_options.h`（已核实：`grep -n kv_options` 无命中）。
  只补 include、不改调用限定名、不删功能。
- 是否回退/注释: **否**。仅新增 include。

### 1-2. designated-initializer 顺序与声明顺序不符

- 错误原文: `src/targets/qwen3_6/impl/runtime/layouts_impl.h:1113:5: error: designator order for field '...SequencePlanningInputs::max_cold_pages' does not match declaration order`
- 文件:行 → `src/targets/qwen3_6/impl/runtime/layouts_impl.h:1088` (`SequencePlanningInputs inputs{...}` 初始化列表)
- 归类: **designator 顺序**（C++20 要求按成员声明顺序）
- 修法: 交换两行 designator —— `.max_cold_pages` 提到 `.cold_keep_tokens` 之前（并加一行注释指出
  声明顺序依据）。
- 理由: 声明顺序在 `src/targets/qwen3_6/impl/runtime/layouts.h:85-115`：
  `cold_policy`(105) → `max_cold_pages`(107) → `cold_keep_tokens`(108) → `cold_host_bytes`(109)。
  初始化列表把 `cold_keep_tokens` 写在了 `max_cold_pages` 前面。
  designated initializer 本身与顺序无关，交换后语义完全不变，属纯合规改写。
  选择改初始化列表而不是改 `layouts.h` 的字段声明顺序，因为后者会牵动 `SequencePlanImpl` 等
  其它结构体/使用点，风险不成比例。
- 是否回退/注释: **否**。纯顺序交换。

### 1-3. `kvcalib` 块被插进了另一个函数体内

- 错误原文: `src/targets/qwen3_6/impl/runtime/text_context_impl.h:116:31: error: a function-definition is not allowed here before '{' token`
  （伴随 `116:28: warning: empty parentheses were disambiguated as a function declaration [-Wvexing-parse]`
  与 `123:51` 同款错误，以及 `1030/1037: 'kvcalib_enabled'/'kvcalib_capture' was not declared in this scope`）
- 文件:行 → `src/targets/qwen3_6/impl/runtime/text_context_impl.h:109-134`
- 归类: **插入位置错**（函数定义落在另一个函数体内）
- 修法: 把整块（注释 109-115 + `kvcalib_enabled()` 116-119 + `kvcalib_capture()` 121-134）从
  `kvdump_layer_enabled()` 体内移出，放到该函数闭合大括号之后、`kvdump_write_file()` 之前，
  即仍在 `namespace ... ::schedule { namespace {` 的匿名命名空间作用域内。
- 理由: 该块被插在 `kvdump_layer_enabled()` 的 `while` 循环闭合 `}`（108）与 `return false;`（135）
  之间，于是 `kvcalib_enabled` 变成了局部函数定义 ⇒ 116/123 报错；`return false;` 也变成了
  `kvcalib_capture` 的尾部语句 ⇒ 函数体被打乱。移出后 `kvdump_layer_enabled` 恢复为
  `while{...} return false; }`，`kvcalib_*` 回到命名空间作用域，1030/1037 的调用点即可解析。
  **不删功能**：调用点（1024-1039 的 Prefill 捕获块）原样保留。
- 依赖核实: `KvCalibrationCapture` 声明在 `src/targets/qwen3_6/impl/runtime/kv_calibration.h:32`，
  同名 `...::schedule` 命名空间；该头文件已在 `text_context_impl.h:50` include。
  `capture(...)` 签名（`kv_calibration.h:48`）与调用一致。故移动即可，无需补 include。
- 是否回退/注释: **否**。整块搬移，一行未删。

---

## R1 收尾 — 修复后的验证构建（全部 -j1，串行）

修完上述 3 处后**没有出现新的第二层错误**，两个目标一次性转绿。

```
$ cd /home/user/ninfer-fusion/build && make ninfer -j1
real 1m3.953s / user 1m0.511s
MAKE_RC=0
[100%] Linking CXX executable ninfer
[100%] Built target ninfer
```

```
$ cd /home/user/ninfer-fusion/build && make ninfer-serve -j1
real 1m17.841s / user 1m13.925s
MAKE_SERVE_RC=0
[100%] Linking CXX executable ninfer-serve
[100%] Built target ninfer-serve
```

grep 构建日志 `error|Error` 计数 = **0**（`/tmp/fix_make.log`）。
重复执行两个 make 均为纯 no-op（`Built target ...`，无编译动作）。

### 证据 1 — 两个二进制（`ls -l --time-style=full-iso`）

```
-rwxr-xr-x 1 user user 825317128 2026-09-10 17:42:33.671595484 +0800 ./apps/ninfer
-rwxr-xr-x 1 user user 827527176 2026-09-10 17:44:00.427571268 +0800 ./apps/ninfer-serve
```

### 证据 2 — 三个 variant `.o`（均晚于 `program_impl.h` mtime，证明补丁 A 进了对象）

```
-rw-r--r-- 1 user user 2250128 2026-09-10 17:42:25.799599754 +0800 src/CMakeFiles/ninfer_engine.dir/targets/muse_glimmer_30b/impl/variant.cpp.o
-rw-r--r-- 1 user user 2404424 2026-09-10 17:41:47.675610713 +0800 src/CMakeFiles/ninfer_engine.dir/targets/qwen3_6_27b/impl/variant.cpp.o
-rw-r--r-- 1 user user 2323832 2026-09-10 17:42:07.151606631 +0800 src/CMakeFiles/ninfer_engine.dir/targets/qwen3_6_35b_a3b/impl/variant.cpp.o
```

（`muse_glimmer_30b` 是被共享头 `layouts_impl.h` 变更连带重编的产物，本轮**未改动**该目录任何文件。）

### 证据 3 — `ninfer-serve` 时间戳晚于 `program_impl.h`

```
ninfer-serve         2026-09-10 17:44:00.427571268 +0800
libninfer_engine.a   2026-09-10 17:42:25.875599722 +0800   (静态库晚于三个 .o)
program_impl.h       2026-09-10 16:31:00.400722983 +0800
```

17:44:00 > 16:31:00 ✅，且 `ninfer-serve` 链接发生在 `libninfer_engine.a`(17:42:25) 之后。

### 受保护补丁未动 — 复核

- 补丁 A：`program_impl.h` 12411-12413 `dflash2_host_ingress->state_source_slots[row] = selectors.source;`
  一带原样在位。
- S50：`program_impl.h` 仍有 9 处 `ensure_sequence_kv_mapped` 调用；其声明在
  `program.h:1282`（`logical_kv_store.h:1497` 提供 `ensure_mapped_to_tokens`）。未改名。
- S48：`dflash_impl.h` 的 `frame.proposal_positions` 用法在位。
- S45d：`gqa_attention_prefill.cu` 的逐臂 `if constexpr` 守卫在位（7 处）。
- S28：`gqa_isoquant_row_scale_loader` 在 `CMakeLists.txt` / `src/CMakeLists.txt` 中命中数
  均为 **0**（保持未登记）。
- 本轮仅 3 处 edit，全部落在 `layouts_impl.h`（2 处）与 `text_context_impl.h`（1 处）。

### 未触碰

`_collab/board.md`、`tools/**`、`src/targets/muse_glimmer_30b/**`。未启动引擎、未跑 GPU。
全程 `-j1`，每次 make 前 `pgrep -af 'make|nvcc'` 确认无并发编译。

**结论：`make ninfer` rc=0，`make ninfer-serve` rc=0，树已能编过。**
