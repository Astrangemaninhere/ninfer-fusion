# B_s33_ple_wiring.md — S33: PLE 表接入 qwen4_exp 运行时 (接线规格 + 补丁)

日期 2026-09-09 · B · 产物: `_collab/B_s33_ple_wiring.diff` (仅新增一个头文件,
`patch -p1 --dry-run` **DRYRUN_OK**; 未落树 — window G 在编 src/)。

## 接线规格 (任务 1)

- **构造与所有权**: `PleRuntime` (新头 `src/targets/qwen4_exp/impl/ple_runtime.h`) 持有
  `std::unique_ptr` 语义的唯一 `ops::ple::PleTable`; 一个引擎实例一个, 随 instance 析构
  (PleTable 不可拷贝, 自持 4 个 fd + pinned LRU — ple_table.h:35-41)。构造镜像
  registry 的 construct_registered 流程位 (src/targets/registry.cpp:112 起)。
- **喂什么 id**: 当前步已提交 token 批。ctx0=token 本身; prevs=2 个前驱**最旧在前**、
  token-major 共 (ngram_size-1)*T 个; 缺前驱/EOS 截断处传 eos (PleLayout::derive_rows
  语义, ple_layout.h:81-95)。
- **gather 结果进入哪里**: `[n_heads*row_dim, T]` = **[16×160, T] = [2560, T] BF16**
  (恰为 hidden 维) token-major 设备张量, 同 stream 产出, 由 layer 1 的 PLE 栈消费
  (checkpoint 键 `model.layers.1.ple.*`, S26 契约) — 在 key/value 投影之前。
  **该消费端代码今天不存在 (见发现 2), 本补丁只立 seam 不虚构消费端。**
- **sidecar 根发现**: 优先级 显式覆盖 > artifact 相邻目录 `<artifact_dir>/ple-root/`
  (须含 ple-manifest.json; 构建器写 `<out>/ple-manifest.json` + `<out>/ple/ple-bf16-*.bin`,
  tools/convert/ple_sidecar_build.py:54-75) > 关。

## 默认关 + 失败模式 (任务 2)
- 未配置且相邻目录无 manifest → `attach()` 返回 **nullptr**, 文本路径零改动、零引用。
- 配置了但 manifest 缺失/损坏 → `attach()` **抛 std::runtime_error**, 报文含完整路径与
  关闭方法 ("disable PLE by removing --ple-sidecar / the adjacent ple-root directory")。
  gather 全链无零填充回退 (ple_table.cu:173-251 无 zero-fill 分支) — 响亮失败, 不静默零。

## 调用路径符号核查 (任务 3; 全部 file:line 实证)

**存在 (接线可直接引用):**
| 符号 | 位置 |
|---|---|
| `PleTableOptions{sidecar_root, cache_bytes=512MiB, prefetch_workers=16}` | src/ops/ple/ple_table.h:28-32 |
| `PleTable` (ctor/dtor/不可拷贝/自持资源) | ple_table.h:35-41 |
| `PleTable::derive_rows(tokens, prevs, eos, rows_out)` | ple_table.h:46-48 |
| `PleTable::gather(rows, n_tokens, dst, stream)` → [n_heads*row_dim, T] BF16 token-major | ple_table.h:50-56; impl ple_table.cu:173 |
| `ple_gather_rows_kernel` (dst __half, `t*(n_heads*row_dim)+h*row_dim` — 与 HF flatten(-2) 对齐) | ple_table.cu:33-44 |
| `PleLayout::from_manifest(path)` (结构不符即抛) | ple_layout.h:73 (impl ple_layout.cpp) |
| 编译注册 (op 已在构建里, 只是无人实例化) | src/CMakeLists.txt:29 (ple_layout.cpp), :110 (ple_table.cu) |
| 几何自洽: 16×160=2560=hidden | config.h:8 + ple_layout.h:36-38 |

**不存在 (本轮发现, 不发明):**
1. **qwen4_exp 运行时不存在**: `src/targets/qwen4_exp/` 仅 package.h + impl/config.h
   (占位)。gather 张量的消费端 (layer-1 PLE 栈) 无代码可接 — 这是 N6 后续轮的主体。
2. **registry 未注册**: `construct_target` (src/targets/registry.cpp:280+) 只分派
   qwen3_6_27b/35b_a3b/muse; qwen4_exp artifact 今天 load 即抛 "has no registered target"。
3. **stub 不可编译**: `impl/config.h:11` `static constexpr int intermediate = None;` —
   `None` 未定义; 目前无人 include package.h 所以树能编, 首个消费者即破。PLE 旋钮
   (ngram=3/heads=8/vocab_base=20M) 只存在于注释 (config.h:45-47), 非代码常量。
4. **EngineOptions 无 PLE 字段** (include/ninfer/types.h:143) 且 serve CLI 无 `--ple-sidecar`
   (parse 先例 serve_options.cpp:266) — 本补丁的 attach() 以参数收显式根, 旗标归 serve 层轮。
5. **eos id 无载体**: qwen4_exp 无 frontend; eos 由调用方传入 (头文件已注明)。
6. **FP8 缺口 (S27 已录)**: sidecar 契约是 BF16 (row_stride 320B, 内核 __half), 而 checkpoint
   表本体 FP8 (`model-plefp8-*`) — 转换器建 sidecar 时须 FP8→BF16 物化 (95 GiB BF16) 或
   op 增 FP8 行路径; 子布局未钉死前两者都不可定。

## 证据命令 (可复现)
```
wsl.exe -e bash -c "cd /mnt/c/Users/User/Documents/ziqinzhang/ninfer-fusion-repo && patch -p1 --dry-run < /mnt/c/Users/User/Documents/ziqinzhang/_collab/B_s33_ple_wiring.diff"
→ checking file src/targets/qwen4_exp/impl/ple_runtime.h
→ DRYRUN_OK
```
补丁本身仅新增头文件 (无既有文件改动 → 与 window G 零冲突); 头文件的 TU 编译验证归
应用窗口 (同 S20/S24 先例: 不在本机起 nvcc)。

## 必须等待的事项 (任务 4)
1. **FP8 PLE 子布局钉死** (model-plefp8-* 10 分片落地后; 决定 sidecar 物化路径)。
2. **真实 prompt 的引擎内数值核对** (gather 结果 vs HF 参考逐元素; §69 的 3-way 对拍是
   独立 gather 验证, 不是引擎内)。
3. **"PLE 接上后模型出正常文本" GPU 窗口测试** (需全量 artifact + qwen4_exp 运行时)。
4. 前置工程轮 (本轮发现 1-5): qwen4_exp 运行时主体、registry 注册、config.h 修复、
   serve 旗标、eos 载体。
