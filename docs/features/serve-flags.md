# `ninfer-serve` 前端旗标

**为什么单独一册**：两个前端的旗标集**不相等**，一个前端有而另一个没有的旗标是真实存在的差异，
照着一个前端写另一个的文章就会错。权威来源是 `src/serve/serve_options.cpp` 的 `arg == "--"` 链
（实测 `grep -n 'arg == "--' src/serve/serve_options.cpp`）。

## 1. 只有服务端有的旗标（抽样，按功能分组）

| 旗标 | 语义 | 出处（`src/serve/serve_options.cpp`） |
|---|---|---|
| `--host` / `--port` / `--api-key` / `--model-id` | 监听地址、端口、鉴权、对外模型名 | `:225,227,229,231` |
| `--max-concurrency` / `--max-pending-requests` / `--pending-timeout-ms` | 并发上限与排队 | `:242,245,248` |
| `--log-stats-interval-ms` | 周期统计打印 | `:265` |
| `--max-request-mib` | 单请求上限 | `:268` |
| `--media-cache-mib` / `--media-live-mib` / `--media-preprocess-threads` | 媒体缓存与前处理线程 | `:275,282,289` |
| `--device-state-slots` / `--host-state-slots` / `--host-kv-mib` | 状态槽与 host KV 预算 | `:296,300,304` |
| `--max-private-continuations` / `--max-shared-prefixes` / `--max-long-anchors-per-continuation` | 前缀复用与续写 | `:311,316,321` |
| `--request-log-jsonl` | 请求日志落 JSONL | `:326` |
| `--response-store-max-records` / `--response-store-max-mib` | 响应存储上限 | `:331,338` |
| `--context-cost-presets` | 上下文代价预设 | `:260` |
| `--kv-auto-relayout SECS` | **周期从 FreeToken 观测重新推导逐层 KV 表**（这就是 CLI 里那个悬空名字的真正落点） | `:535`（解析 `:541-542`）；用法 `:121,177`；消费 `:550` 起 |
| `--ft-vram-axis on\|off` | 重排是否把 VRAM 轴计入 | `:543` |
| `--default-max-tokens` / `--default-thinking-budget` | 请求未给时的默认 | `:609,613` |
| `--no-prefix-reuse` / `--no-auto-system-shared-prefix` | 关前缀复用 | `:624,626` |
| `--preserve-thinking` | 保留思考段 | `:638` |
| `--cors` | CORS 开关 | `:640` |

## 2. 两个前端都有的旗标（服务端侧的行号）

服务端也解析 `--kv-dtype`、`--kv-layer-storage`、`--kv-tier-formats`、`--nvfp4-mode`、`--kv-rotation`、
`--kv-row-scale`、`--kv-v-codec`、`--kv-bit-budget`、`--kv-bits`、`--kv-k-bits`、`--kv-v-bits`、
`--kv-bits-mode`、`--kv-tier-scores`、`--kv-k-tier-scores`、`--kv-v-tier-scores`、`--kv-quality-weight`、
`--kv-score-table`、`--kv-residual-layers`、`--cold-policy`、`--cold-disk-path`、`--cold-disk-bytes`、
`--cold-keep-tokens`、`--max-cold-pages`、`--kv-unload-watermark-pages`、`--ft-stats`、
`--cold-host-bytes`、`--weight-host-bytes`、`--weight-device-arena-bytes`、`--weight-prefetch-layers`、
`--weight-span-floor-bytes`、`--yarn`、`--spec`、`--draft-tokens`、`--draft-tree`、`--vision`、
`--no-cuda-graph`、`--lm-head-draft`、`--no-lm-head-draft`、`--no-thinking`、`--temperature`、`--top-p`、
`--top-k`、`--min-p`、`--presence-penalty`、`--frequency-penalty`、`--seed`、`--greedy`、`--device`、
`--max-context`、`--kv-capacity`、`--prefill-chunk`、`--prefill-chunk-mode`。

对应行号依次在 `src/serve/serve_options.cpp`：`:347,350,359,365,375,388,394,418,438,443,448,453,457,459,461,463,466,471,482,495,497,503,509,521,547,552,555,560,566,575,578,580,583,587,620,622,628,630,636,642,645,648,652,655,658,661,663,345,236,239,251,254`。

⚠ **别假设语义逐字一致**：CLI 侧的 `--kv-dtype` 多接受 `iso4e`/`iso3`/`rk4v4`/`e8` 两个废弃别名
（`apps/cli/options.cpp:104-105`），且 CLI 侧的 `fp8` 映射到 `Fp8E4M3Row256` 而不是服务端的
`Fp8Group16`——`apps/cli/options.cpp:100-103` 自述这两个表**故意**不同，不是抄写错误。

## 3. 环境变量在两个前端的读取点

同一个开关在两个前端各有一处读取，这是「两个前端不能漂移」的实现方式：

| 环境变量 | CLI 读取点 | serve 读取点 |
|---|---|---|
| `NINFER_KV_UNLOAD_WATERMARK_PAGES` | `apps/cli/options.cpp:748` | `src/serve/serve_options.cpp:905` |
| `NINFER_MTP_ADAPTIVE` | `apps/cli/options.cpp:796` | `src/serve/serve_options.cpp:697` |

`apps/cli/options.cpp:741-746` 自述两者共用一个 `*_explicit` 门，使「flag 是否胜过 env」这件事
在两个前端**不可能不一致**。

## 4. ⚠ 服务端 `--help` 的用法行**漏列**了自己 parser 里注册的 18 个旗标（2026-09-22 `accept34` 线实测）

`ninfer-serve --help` 的第一行用法文本是**面向用户的第一份文档**，而它与 `serve_options.cpp` 的 parser 不齐。
口径：取 `src/serve/serve_options.cpp` 里所有 `"--flag"` 字面量的并集，减去用法文本里出现的旗标名，
差集实测 **18 个**：

`--cold-policy`、`--cold-keep-tokens`、`--cold-disk-path`、`--cold-disk-bytes`、`--cold-host-bytes`、
`--max-cold-pages`、`--kv-bit-budget`、`--kv-layer-storage`、`--kv-residual-layers`、
`--kv-k-tier-scores`、`--kv-v-tier-scores`、`--kv-unload-watermark-pages`、
`--weight-host-bytes`、`--weight-device-arena-bytes`、`--weight-prefetch-layers`、
`--weight-span-floor-bytes`、`--yarn`、`--help`。

复现（两条命令取差集）：

```bash
grep -o -- '"--[a-z0-9-]*"' src/serve/serve_options.cpp | tr -d '"' | sort -u > /tmp/srv_parser.txt
./build/apps/ninfer-serve --help | grep -o -- '--[a-z0-9-]*' | sort -u                > /tmp/srv_help.txt
comm -23 /tmp/srv_parser.txt /tmp/srv_help.txt
```

**判定：这不是「旗标不存在」，而是用法文本不完整。** 后果是照 `--help` 写文档的人会以为
冷窗族（`--cold-*`、`--max-cold-pages`）与权重卸载族（`--weight-*`）在服务端**不可达**；
实测相反，这两族都能过 parser 并进入装载：

```
ninfer-serve <model> --cold-policy disk --max-cold-pages 4        -> 过 parser（进 "loading model..."）
ninfer-serve <model> --weight-host-bytes 8589934592 --weight-prefetch-layers 2 -> 过 parser
ninfer-serve <model> --kv-bit-budget 0-7:8,8-63:4.5               -> 过 parser
```

本册第 2 节那串「两个前端都有的旗标」**是对的**；**不完整的是二进制自己 `--help` 的第一行**。

⚠ **第二个具名差异（同一批测量）**：用法行写 `--kv-dtype bf16|int8|fp8`，而
`ninfer-serve <model> --kv-dtype nvfp4` **被 parser 收下**（不报错、直接进入装载）。
按本项目的「dead label」教训，这种「文档说不可用、parser 收下」的格子**不能**写成「服务端支持 nvfp4」：
要么补用法行，要么在 parser 里拒。本条只报实测，不下裁定。

⚠ **测量口径**（本节的证据强度）：以上四条 serve 侧结论都是在 `build/apps/ninfer-serve`
sha256 前 16 **`91327a8ae55f70e0`**（mtime 2026-09-19 11:21）上取的，调用带空 `CUDA_VISIBLE_DEVICES`，
所以「过了 parser」= 打印 `loading model...` 后死于设备探测，**没有真跑**。
该二进制的 parser 源（`src/serve/serve_options.cpp` 2026-09-20 14:25、`src/product/kv_options.h`
2026-09-20 09:11）**比它新** ⇒ 服务端结论必须带上这个 sha16，见 `unfinished.md` 的 E2。
证据文件：`dl/accept34/logs/s52_parse_matrix.txt`、`dl/accept34/logs/s54_grammar2.txt`。
