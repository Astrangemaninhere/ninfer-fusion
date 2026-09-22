# 验收与复现

本册给出三样：**测试面规模**（实测）、**本目录每条数字的复现命令**、**本目录没有做什么**。

## 1. 测试面规模（实测）

| 项 | 数字 | 命令 |
|---|---|---|
| `tests/CMakeLists.txt` 里 `add_test` **文本**行数 | **124**（2026-09-22 复核；见第 6 节） | `grep -c 'add_test' tests/CMakeLists.txt` |
| 其中用包装宏 `ninfer_add_test` 的行数 | **101** | `grep -c 'ninfer_add_test' tests/CMakeLists.txt` |
| `add_executable` 行数 | **14** | `grep -c 'add_executable' tests/CMakeLists.txt` |
| `NEEDS_SOURCE_DIR` 出现次数 | **23**（2026-09-22 复核未变） | `grep -c 'NEEDS_SOURCE_DIR' tests/CMakeLists.txt` |
| `tests/` 下文件总数（排除 `__pycache__`） | **264** | `find tests -type f -not -path '*__pycache__*' \| wc -l` |
| 按后缀 | `cpp` 181、`py` 44、`cu` 18、`h` 14、`pre_land` 2、`jinja` 2、`txt` 1、`md` 1、`json` 1 | `find tests -type f -not -path '*__pycache__*' \| sed 's/.*\.//' \| sort \| uniq -c \| sort -rn` |

⚠ **一个陷阱，写文档时别踩**：`grep -c 'add_test\|ninfer_add_test\|add_executable'` 得 **134**，
而 `ninfer_add_test` **包含子串** `add_test`，所以 121 里已经含了 99，**134 是重复计数**。（2026-09-22 复核：同一条命令今天得 **138**；
**这两个数都不是 ctest 的测试数** —— 当天 `ctest -N` 是 **179**。见第 6 节。）
（同类陷阱还有 `npu` ∈ `i-n-p-u-t`、`CANN` ∈ `SCANNER`、`ascend` 作英文词；本目录的
[`unfinished.md`](unfinished.md) 第 D 段记了我自己踩的一次。）

## 2. 构建与测试开关（默认值实测）

| 选项 | 默认 | 出处 |
|---|---|---|
| `NINFER_BUILD_APPS` | **ON** | `CMakeLists.txt:238` |
| `BUILD_TESTING` | **OFF** | `CMakeLists.txt:239` |
| `NINFER_BUILD_BENCHMARKS` | **OFF** | `CMakeLists.txt:240` |
| `NINFER_ENABLE_ARCH_SIM` | **OFF** | `CMakeLists.txt:253-254` |
| `NINFER_ENABLE_VENDOR_SIM` | **OFF** | `CMakeLists.txt:274-275` |
| `CMAKE_CUDA_ARCHITECTURES` 默认 | **`120a`** | `CMakeLists.txt:20-21`；版本闸在 `:423-436` |
| 语言标准 | C++20 / CUDA C++20 | `CMakeLists.txt:277-280` |
| 未设 `CMAKE_BUILD_TYPE` 时 | Release | `CMakeLists.txt:282-284` |

⇒ 默认构建**不含**测试、基准与维护者工具（`README.md:40-41` 也这么写）。

## 3. 部署与验收路径

```bash
# 构建产品二进制
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# 安装（唯一部署步骤）
cmake --install build --prefix /path/to/prefix
```

- 只有三个可执行程序被装：`apps/CMakeLists.txt:60-62`（`ninfer`、`ninfer-serve`、`ninfer-perplexity`）。
- 文档组件装到 `${CMAKE_INSTALL_DOCDIR}`：`CMakeLists.txt:515-520`（`COMPONENT documentation`）。
- 配置期生成 `ninfer-install-manifest.txt` 并一起装：`CMakeLists.txt:525`（生成）、`:567-569`（安装）。
- 有意**不**装的东西（引擎静态库、Dockerfile、`tests/`、`bench/`、`tools/`、`docs/`）写在
  `CMakeLists.txt:509-513`。

**验收闸（都不需要 GPU）**：

| 闸 | 作用 | 出处 |
|---|---|---|
| `--capability-report` | 打印这个二进制被编译到的 arch 清单与逐格式 tensor-core 底线，各带 kernel 引证 | `apps/cli/options.cpp:479-484`、用法 `:207-215` |
| `--kv-score-table show` | 打印 planner 会用的罚分表 + 每列真实出处 | `apps/cli/options.cpp:485-490` |
| `tools/convert/artifact_diff.py` | 两个 `.ninfer` 的逐对象比较（导入产物的验收闸），按退出码判 | `tools/convert/artifact_diff.py`（224 行） |
| `tools/archkit/check_geometry.py` / `check_params.py` | 几何覆盖门 / 参数完备门，**会红** | 147 / 179 行 |
| `tools/archkit/test_probe_provenance.sh` | 探测结果的**出处校验** | 294 行 |
| `--draft-tree 1,d` 对 `--draft-tokens d` | 树路径退化链必须逐 token 复现链路径（树路径自己的仪器检查） | `apps/cli/options.cpp:223` |

## 4. 本目录每条数字的复现命令

本目录的数字都来自对**同一棵树**（`HEAD = 3944a53`）的只读测量。逐条对应：

| 数字 | 命令 |
|---|---|
| `HEAD` 与工作区规模（794 / 236 / 168） | `git rev-parse HEAD`；`git status --porcelain \| wc -l`；`git status --porcelain \| grep -c '^A'`；`git status --porcelain \| grep -c '^??'` |
| 旗标数与逐条行号 | `grep -n 'arg == "--' apps/cli/options.cpp`；`grep -n -o -- '--[a-zA-Z][a-zA-Z0-9-]*' apps/cli/options.cpp` |
| 旗标默认值 | `cat -n apps/cli/options.h` |
| 用法文本与互斥规则 | `awk 'NR>=146 && NR<=1016 {printf "%d\t%s\n", NR, $0}' apps/cli/options.cpp` |
| 源文件行数 / 文件数 | `wc -l <file>`；`find <dir> -type f \| wc -l` |
| 目录规模（各目标族） | `for d in src/targets/*/; do find "$d" -type f \| wc -l; done` |
| 运行期环境变量表 | `grep -rn 'getenv("NINFER' src/ apps/ include/ tools/ \| grep -v __pycache__` |
| 编译期宏与目标注册 | `grep -n 'add_subdirectory(targets' src/CMakeLists.txt`；`grep -rn '^option(' CMakeLists.txt` |
| `install(` 规则 | `grep -rn 'install(' CMakeLists.txt apps/CMakeLists.txt` |
| Windows 原生：0 命中 | `grep -c -E 'WIN32\|MSVC\|windows' CMakeLists.txt src/CMakeLists.txt apps/CMakeLists.txt` |
| `_WIN32` 分支文件 | `grep -rln '_WIN32' src/ apps/ tools/` |
| 严格词界的厂商判别（NPU / Intel / AMD） | `grep -rnwiE '<pattern>' src apps tools include docs tests` |
| `docs/` 跟踪比 30/55 | `git ls-files docs \| wc -l`；`find docs -type f \| wc -l` |
| 测试面规模 | 见本册第 1 节的命令 |

**一个纪律**：`--` 开头的模式要写成 `grep -o -- '--[a-z]...'` 或 `grep -- '-x'`，
否则会被 grep 当成选项吞掉。上一轮吃过一次空产物
（`grep -rln "kv-codec-preference"` 返回空），本目录所有带 `--` 名字的 grep 都加了 `--`。

## 5. 本目录没有做什么（明说）

1. **没有跑引擎、没有编译、没有加载模型**。全部结论来自读代码、读既有日志与只读文件系统测量。
   因此「能跑」「性能如何」这类结论本目录一律不写；`README.md` 的 Quick start 与
   `docs/performance.md` 是那类结论的落点。
2. **没有改任何源码**。本目录只新增文件：`README_FEATURES.md` 与 `docs/features/` 下的 10 个文件
   （`docs/features/` 之前不存在），不覆盖任何既有文件。
3. **没有验证「跨厂商能跑」**。见 [`unfinished.md`](unfinished.md) 的 P10 / P12 / P13：
   AMD 只有构建配置行，Intel 是被具名的拒绝，NPU/国产卡只有注释。回退内核族（SIMT FFMA、
   通用 rowsplit）已落地并有测试，但**没有任何厂商的端到端验证**。
4. **没有做「上一轮清单里我没能证明的」那部分的证明**。它们按名留在 `unfinished.md`，
   不是被删掉、也不是被写成已完成。
5. **没有引用任何仓库外的私有日志作为出处**。凡「实测数字」，都在本册第 4 节给出**可复跑的命令**；
   凡「源码事实」，都给出 `路径:行号`。

## 6. 2026-09-22 复核（`accept34` 线，只读测量）

本节由**另一条线**在**同一天的树**上重测第 1 至第 4 节里可复跑的那部分。结论一句话：
**第 1 节的数字已漂且有一处读法要纠正；第 2 节的默认值逐条仍成立；第 3、4 节的行号类出处已漂**
（原因见 6.3）。

**被测二进制**（本节凡引用二进制处都是这两个 sha256，逐条实测）：

| 二进制 | mtime | sha256 前 16 |
|---|---|---|
| `build/apps/ninfer` | 2026-09-21 22:05 | `713cfcd6a1ad1150` |
| `build/apps/ninfer-serve` | 2026-09-19 11:21 | `91327a8ae55f70e0` |

### 6.1 第 1 节的测试面（重测）

| 项 | 数字（2026-09-22） | 命令 |
|---|---|---|
| `tests/CMakeLists.txt` 里 `add_test` **文本**行数 | **124** | `grep -c 'add_test' tests/CMakeLists.txt` |
| `ninfer_add_test` 文本行数 | **101** | `grep -c 'ninfer_add_test' tests/CMakeLists.txt` |
| `add_executable` 文本行数 | **14** | `grep -c 'add_executable' tests/CMakeLists.txt` |
| `NEEDS_SOURCE_DIR` 出现次数 | **23**（与第 1 节相同，未漂） | `grep -c 'NEEDS_SOURCE_DIR' tests/CMakeLists.txt` |
| `tests/` 下文件总数 | **264**（`cpp` 181 / `py` 44 / `cu` 18 / `h` 14 / `pre_land` 2 / `jinja` 2 / `txt` 1 / `md` 1 / `json` 1） | `find tests -type f -not -path '*__pycache__*'` |
| 三个模式合起来（第 1 节记的「陷阱数」） | **138**（第 1 节记 134） | `grep -c 'add_test\|ninfer_add_test\|add_executable' tests/CMakeLists.txt` |

⚠ **本节要按名纠正一处读法**：上表的 **124 是文本行数，不是测试数**。同日 `ctest -N`（在 `build/` 里跑）
报 **`Total Tests: 179`**，展开件 `build/tests/CTestTestfile.cmake` 里也正好 **179** 行 `add_test`。
差额来自四个包装函数与一个 `foreach` 列表：`ninfer_add_test(` **97** 处、`ninfer_add_op_test(` **29** 处、
`ninfer_add_linear_test(` **8** 处、`ninfer_add_fused_linear_test(` **8** 处（各计数含自己的定义行），
其中 `ninfer_op_tests` 列表（`tests/CMakeLists.txt:601-626`，24 项）由 `:627-631` 的 `foreach` 批量注册。
⇒ **任何「121 / 124 个测试」的说法都是把文本行数当成了测试数。**

### 6.2 第 2 节的默认值（重测，逐条仍成立）

`CMakeLists.txt:238/239/240` 依次是 `NINFER_BUILD_APPS ON`、`BUILD_TESTING OFF`、
`NINFER_BUILD_BENCHMARKS OFF`；`:253-254` 与 `:274-275` 两个 SIM 开关默认 `OFF`；
`:20-21` 默认 `CMAKE_CUDA_ARCHITECTURES 120a`；`:423-436` 是 arch 版本闸；
`:277-284` 是 C++20 / CUDA C++20 与未设 `CMAKE_BUILD_TYPE` 时 Release。**逐条与第 2 节相同，未漂。**

### 6.3 第 3 与第 4 节的行号（已漂，原因是树在动）

- `apps/cli/options.cpp` 今天 **1045** 行（第 4 节记 1016；`unfinished.md` 的 D2 记 1016）。
- `--capability-report` 的 case 今天在 `:490`（第 3 节与 `cli-flags.md` 记 `:479-484`）；
  `--kv-score-table` 的 case 今天在 `:496`（记 `:485-490`）。
- `--stop` / `--reasoning-stop` 的共用分支今天在 `:707`（`cli-flags.md` 记 `:690`），
  三元表达式在 `:714`（记 `:697`）。
- `apps/CMakeLists.txt` 今天 **62** 行、`install(TARGETS ...)` 在 `:60`（第 3 节记 `:60-62` ✓）。
- `CMakeLists.txt` 今天 **569** 行；`COMPONENT documentation` 在 `:520` 与 `:569`、
  `ninfer-install-manifest` 在 `:525`（第 3 节记 `:515-520` / `:567-569` / `:525` ✓ 落在范围内）。
- **原因**：`apps/cli/options.cpp` 的 mtime 是 **2026-09-21 21:20**，而基准修订
  `3944a53eda1aac439a566a1cf46ea741f0415fdc` 不变（`git rev-parse HEAD` 实测仍是它）
  ⇒ **HEAD 不动、工作区在动**，于是「带行号的出处」会随别人的编辑漂。
  **行号类出处必须以重测当天的实测为准，并带上当天读到的文件 sha256。**

### 6.4 本节没有做的（明说）

没有跑引擎、没有加载模型、没有占用 GPU（本节涉及二进制的调用全部带 `CUDA_VISIBLE_DEVICES=` 空值，
调用前后各查一次 `nvidia-smi`，`utilization.gpu` 与 `memory.used` 未变）；没有改任何源码；
没有回滚任何别人的改动；没有动 `dl/_orch/DEFECTS.md`。本节只新增文字，外加 6.5 列出的三处数字更新。

### 6.5 本节的写入方式（前像可回退）

每个被改文件的前像落在记录目录 `dl/accept34/pre/`（前缀即文件名）；写入是「临时文件 + `os.replace`」，
写后重算 sha256 与行数，逐文件记录在 `dl/accept34/logs/s70_docpatch.txt`。
