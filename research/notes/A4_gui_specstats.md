# A4 · 投机档位/接受率/位置剖面接入 GUI + LFM2-2.6B-Exp / Falcon-H1R-7B 导入复核

日期 2026-09-10（窗口 A4，Windows 侧）。

**纪律**：只改 `tools/**`（`tools/gui/serve_gui.py`、`tools/gui/i18n_serve.py`、`tools/archkit/adapt.py`）。
未碰 `src/**`，未编译，未开 nvcc，未用 GPU，未启动 serve，未下载任何权重（只用本机 `models/` 下已有的
config / index / tokenizer）。本机 `python3` 不可用，全程 `py -3`（Python 3.12.10）。
本轮**没有**跨机（WSL）操作：所有命令都在 Windows 侧跑。

---

## 0. 一句话结论

1. 引擎**当前没有**投机指标端点（`src/serve/http_server.cpp` 只注册 `/health` 与 `/v1/*`），所以 GUI 走日志解析；
   但"端点优先"的顺序是**写进代码**的（候选表 + 10 s TTL + 0.4 s 超时），引擎哪天补上就自动切过去。
2. 位置剖面（`accepted by pos` 直方图）**只存在于两处**：`--request-log-jsonl` 的 JSONL 记录
   （`accepted_per_position`）与 CLI 的汇总行。serve 的文本行不打印它 —— 所以卡片在拿不到直方图时会
   明说"本轮日志里没有位置直方图"，而不是画一根假柱子。
3. 两个模型**都被扣了 header**（`config.h.BLOCKED`）：LFM2 的 22 个 conv 层、Falcon 的 44 层
   Mamba+attention 并行块，**都是真缺口**（不是导入器误报）。§122 的判断（"标成 new_op 是正确的"）**成立**。
4. 但重跑同时抓到导入器的 **3 处新的漏检/误判**（两类模型各命中）：LFM2 的 head 几何被**静默跳过**、
   LFM2 的 qk-norm 极性被**判反**、Falcon 的 9 个 muP multiplier 旋钮**完全没读**。
   三处都给了最小修法并重跑证明分类变化（§2.5），并做了 pre-A4 / live 的**对照实验**证明只动了这两个模型（§2.6）。

---

## 活 1：投机档位 + 接受率 + 位置剖面进 GUI

### 1.1 先查"引擎有没有指标端点"（结论：没有）

```
src/serve/http_server.cpp:364  server_.Get("/health", ...)          <- 只有 status/engine 两键
src/serve/http_server.cpp:385  server_.Get("/v1/models", ...)
src/serve/http_server.cpp:391  server_.Post("/v1/chat/completions", ...)
src/serve/http_server.cpp:395  server_.Post("/v1/responses", ...)
```

全文件没有 `/metrics`、没有 `/api/*`。⇒ 按任务书的次序走到第二档："没有就解析日志尾部"。
但候选表与探测逻辑照写：

```python
ENGINE_METRIC_PATHS = ('/api/spec_stats', '/metrics', '/v1/metrics')
PROBE_TTL_SECONDS = 10.0     # /api/state 每 1.5 s 轮询一次，不能每次都打同一个 404
```

`probe_engine_metrics()` 只在 serve 子进程活着时探测、超时 0.4 s、连接被拒立即返回，拿到 JSON 后
`_spec_from_engine_json()` 按同一套字段名归一 ⇒ 引擎补端点后界面零改动。

### 1.2 三个日志来源（字段同名同义）

| 来源 | 出处 | 给的字段 | 有位置直方图？ |
|---|---|---|---|
| serve stdout request-done 行 | `src/serve/request_log.cpp:464-479` | `speculative=<backend> spec_drafted= spec_accepted= spec_accept_rate= spec_rounds= spec_fallback_steps= spec_accept_len=` | **没有** |
| `--request-log-jsonl FILE` 的 `request_done` 记录 | `request_log.cpp:307-318`（`speculative_json`，含 `accepted_per_position`）；`src/serve/generation_service.cpp:455` 填充 | 同上 + `draft_window` + `accepted_per_position[]` | **有** |
| CLI 汇总行 | `apps/cli/main.cpp:226-248` | `<backend> acceptance rate %` / `acceptance length tok/round` / `accepted by pos a,b,c` | **有** |

合并优先级（低 → 高，按字段覆盖）：`log-tail < cli-summary < jsonl < engine-http`；
卡片右上角显示 `来源: log-tail + jsonl` 这样的字样，读数永远可归因。

### 1.3 接口与界面

- `/api/state` 新增 `spec` 键（任务要求 1 的"并把它写进 /api/state 的 JSON"）。
- 新增只读端点 `GET /api/spec_stats`（任务要求 1 允许的第二条路），返回同一份 dict。
- **`serve_params.json` 未改**（别的线在用，要求 2）。
- 界面：滑块卡下面新开一张 "投机档位与接受率（当前会话）" 卡，8 个数字格
  （档位/草稿后端 · 草稿窗 · 草稿轮数 · 接受率 · 接受长度 · 空转步 · 草稿 token · 被接受 token）、
  一行原文 `23,11,3,1,0,0,0  /  rounds=55`、以及一排位置柱（柱高按同图最大值归一，
  悬停 title 显示 `p3: 1 / 0.018/round`）。
- 徽章三态：`未启用` / `已配置，等待实测`（档位来自请求里的 spec 滑块，还没跑过） / `实测中`（来自日志）。

### 1.4 门禁（原文，逐字贴回）

```
$ cd ninfer-fusion-repo/tools/gui && py -3 gui_i18n_check.py --only serve_gui.py
keys used by sources : 13
keys in tables      : 343 (zh 343 / en 343)
template keys (runtime-resolved): 2; families without concrete entries: none
VERDICT: PASS
```

（顺手把全量检查也跑了一遍，未受影响：）

```
$ py -3 gui_i18n_check.py
keys used by sources : 164
keys in tables      : 343 (zh 343 / en 343)
template keys (runtime-resolved): 8; families without concrete entries: none
VERDICT: PASS
```

既有自检（`serve_gui_selftest.py`，包含与 git HEAD 基线的逐字节对拍 + 20000 次随机滑块 + 无引擎 HTTP 冒烟）：

```
$ py -3 serve_gui_selftest.py | tail -3
gui_lang.txt restored to: 'en'
==========================================================================
29 passed, 0 failed
VERDICT: PASS
```

### 1.5 实测证据（解析器 / 接口 / 前端）

解析器（`_collab/a4_scratch/a4_gui_verify.py`，喂的是 §M_patchA_effect / M_spec_4way 里的真数字）：

```
--- 1. parsers ---
serve line   : {"backend": "dflash2", "enabled": true, "drafted": 154, "accepted": 54, "accept_rate": 0.3506, "rounds": 22, "fallback_steps": 0, "accept_len": 3.45}
serve (zh)   : {"backend": "dflash2", "enabled": true, "drafted": 133, "accepted": 6, "accept_rate": 0.0451, "rounds": 19, "fallback_steps": 57, "accept_len": 1.32}
spec off     : {"backend": "", "enabled": false}
jsonl record : {"backend": "dflash2", "enabled": true, "draft_window": 7, "drafted": 154, "accepted": 54, "accept_rate": 0.3506, "rounds": 55, "fallback_steps": 0, "by_position": [23, 11, 3, 1, 0, 0, 0]}
cli summary  : {"backend": "dflash2", "accept_rate": 0.1003, "accept_len": 1.69, "by_position": [23, 11, 3, 1, 0, 0, 0]}
garbage      : null null null
```

合并后的会话视图（`configured` 来自请求里的 spec=3/draft=7，位置直方图来自 JSONL）：

```json
{ "running": true, "configured": "dflash2", "draft_tokens": 7, "backend": "dflash2",
  "draft_window": 7, "drafted": 154, "accepted": 54, "accept_rate": 0.3506, "rounds": 55,
  "fallback_steps": 0, "accept_len": 1.69, "by_position": [23,11,3,1,0,0,0],
  "sources": ["log-tail", "cli-summary", "jsonl"] }
```

真起一个 GUI 进程打两个端点（**不启引擎**，`/api/start` 从未调用）：

```
$ py -3 _collab/a4_scratch/a4_gui_http.py
GET /api/state      -> 200, keys=['gpu', 'log', 'running', 'spec']
   state["spec"]    -> {"running": false, "configured": "", "draft_tokens": 0, "backend": "", "enabled": null,
                        "draft_window": null, "drafted": null, "accepted": null, "accept_rate": null,
                        "rounds": null, "fallback_steps": null, "accept_len": null, "by_position": [], "sources": []}
GET /api/spec_stats -> 200
   body             -> （同上）
GET /?lang=en       -> 200, 28243 bytes, leftover @@: 0, spec card present: True
GET /api/i18n       -> 200, 28 strings
GET /?lang=zh       -> 200, spec card CJK chars: 119
```

前端（node 24 语法检查 + 真跑 `paintSpec()`，DOM 用最小桩）：

```
$ node _collab/a4_scratch/a4_js_check.js _collab/a4_scratch/a4_page_en.html
JS PARSE+EXEC: OK
paintSpec[no engine]: OK  badge="Not enabled" tier="—" rate="—" len="—" posline="—" src=""
paintSpec[measured dflash2 + positions]: OK  badge="Measured" tier="dflash2 d7" rate="35.06%" len="4.55 tok/round" posline="23,11,3,1,0,0,0  /  rounds=55" src="source: log-tail + jsonl"
paintSpec[measured, no positions]: OK  badge="Measured" tier="mtp" rate="35.04%" len="2.04 tok/round" posline="—" src="source: log-tail"
paintSpec[undefined payload]: OK  badge="Not enabled" tier="—" rate="—" len="—" posline="—" src=""
```

`posline` 正是任务书给的形状：`dflash2 23,11,3,1,0,0,0 / rounds=55`。

双语渲染（英文页那张卡 0 个 CJK 字符，中文页 119 个；两页都是 0 个遗留 `@@`）：

```
[zh] bytes=26876  leftover @@: []   spec card CJK chars: 119
[en] bytes=28243  leftover @@: []   spec card CJK chars: 0
```

新词条 19 条全部在 `tools/gui/i18n_serve.py` 的 `serve.*` 下（`spec_title / spec_none / spec_configured /
spec_measured / spec_na / spec_tier / spec_rate / spec_len / spec_len_unit / spec_drafted / spec_accepted /
spec_rounds / spec_rounds_word / spec_fallback / spec_window / spec_pos / spec_pos_none / spec_src /
spec_per_round / spec_hint`）；JS 侧只读 `JS_KEYS` 生成的字符串表，没有硬编码文案。

### 1.6 已知边界（写清楚，不含糊）

- **没有 `--request-log-jsonl` 就没直方图**：serve 文本行不打印位置。卡片会显示
  "（本轮日志里没有位置直方图）"，接受率/轮数/草稿数照常显示。要看剖面，把该旗标填进"全参数"面板即可
  （它本来就是注册表里的 58 项之一，GUI 已会把它拼进命令行并据此读文件）。
- **多源合并是按字段取"高优先级来源的最新一条"**。一次会话里跑了多个请求、且各源的最新条不是同一条时，
  卡片可能混用两个请求的数字（`来源:` 那行会显示来源串，可据此判断）。单请求会话无此问题。
- 任务书说"CLI 侧更容易拿位置剖面时**也可以**新开只读小页"——优先改 `serve_gui.py` 已满足要求，故**未**新开页，
  `convert_gui.py` / `rag_gui.py` / `model_import.py` / `gui_i18n.py` / `gui_i18n_check.py` / `i18n_misc.py`
  一个字未动（git status 见 §4）。

---

## 活 2：用导入器跑 LFM2-2.6B-Exp 与 Falcon-H1R-7B

### 2.1 跑法与产物

```bash
cd ninfer-fusion-repo && py -3 tools/archkit/adapt.py \
    C:/Users/User/Documents/ziqinzhang/models/LFM2-2.6B-Exp   --model-id lfm2-2.6b-exp
cd ninfer-fusion-repo && py -3 tools/archkit/adapt.py \
    C:/Users/User/Documents/ziqinzhang/models/Falcon-H1R-7B  --model-id falcon-h1r-7b
```

（`PYTHONPATH=$PWD` 在实际执行里不需要：`adapt.py` 自己 `sys.path.insert(0, REPO)`；
本机是 cmd 风格 shell，`PYTHONPATH=. py -3 ...` 会被当成外部命令，所以直接用 `py -3`。
两者都只有 config/index/tokenizer，**没有下载权重**。）

产物路径：

| 模型 | spec | manifest | header |
|---|---|---|---|
| LFM2-2.6B-Exp | `ninfer-fusion-repo/tools/archkit/specs/lfm2-2.6b-exp_spec.json` | `ninfer-fusion-repo/tools/archkit/out/lfm2-2.6b-exp/manifest.json` | `.../out/lfm2-2.6b-exp/config.h.BLOCKED` |
| Falcon-H1R-7B | `ninfer-fusion-repo/tools/archkit/specs/falcon-h1r-7b_spec.json` | `ninfer-fusion-repo/tools/archkit/out/falcon-h1r-7b/manifest.json` | `.../out/falcon-h1r-7b/config.h.BLOCKED` |

（各自还有 `engine_hook.patch`。两个 `config.h` **都不存在**，只有 `.BLOCKED` —— `main()` 会
`unlink` 掉另一分支的文件，不会留下会误导审计的旧 header。）

### 2.2 缺口表（原始输出，逐字贴回）

```
$ py -3 tools/archkit/adapt.py .../models/LFM2-2.6B-Exp --model-id lfm2-2.6b-exp
adapted lfm2-2.6b-exp -> ...\tools\archkit\out\lfm2-2.6b-exp
  [blocked] config.h withheld as config.h.BLOCKED (unresolved: new_op:layer_kinds(conv x22); attn:head_geometry(32q/8kv@64))
  [covered] attention:gqa_full                             v3 gen
  [new_op] new_op:layer_kinds(conv x22)                   短卷积层 x22 需 no-SiLU/宽度变体, 不能直接复用 causal_conv1d_silu 叶子 (叶子=宽4+SiLU+无门控; 本模型=宽3/无bias+无激活+卷积前后各一道逐元素门 B*x / C*conv, 卷积输入是 B*x 而非 hidden) ⇒ 变体内核+两个逐元素门+一个新层型 (依据 _TODO.md §125; 参考实现 dl/_lfm2_modeling.py Lfm2ShortConv)
  [covered] head:tied=true                                 tied head: 转换期把嵌入键 model.embed_tokens.weight 物化为 text/output_head ([vocab,hidden], 约 268 MB), 引擎按独立 lm_head 对象加载, 无需新算子; 转换器已实现 (tools/convert/common/source_map.py + qwen3_6_27b/recipe.py resolve_sources); 判定依据: index has no lm_head.weight and the config omits tie_word_embeddings
  [new_op] attn:head_geometry(32q/8kv@64)                 引擎几何注册表未收 (现有 16/2@256, 24/4@256, 32/2@128); wrapper 还会按 q_heads 反推 KV 头 (wrapper/gqa_attention.cpp:25-30: 16->2) 再抛错
  [hook] attn:qk_norm=present(weights,16)                 config 无 qk-norm 旋钮, 但 checkpoint 带 16 个 q/k norm 张量 (e.g. model.layers.13.self_attn.k_layernorm.weight) ⇒ qk_norm_enabled() 必须为真 (家族无条件 rmsnorm(q,k), text_context_impl.h:958-959), 按"config 没写=关闭"处理会静默丢掉 q/k 归一化
  [hook] token_domain:vocab=65536!=family 248077         FrontendOptions.token_domain + official_specials 覆盖
  [hook] layers:30>16                                    cold_slots 类 per-layer 数组容量核对 (家族已扩 64, 新家族须审计)
  [post] quant_geometry                                  转换后校验: fp8/nvfp4 形状对照几何注册表 (A16 起步)

$ py -3 tools/archkit/adapt.py .../models/Falcon-H1R-7B --model-id falcon-h1r-7b
adapted falcon-h1r-7b -> ...\tools\archkit\out\falcon-h1r-7b
  [blocked] config.h withheld as config.h.BLOCKED (unresolved: new_op:state_space(mamba_d_state,mamba_n_heads,mamba_d_conv,mamba_expand); new_op:layer_structure(parallel attn+ssm x44/44); attn:head_geometry(12q/2kv@128))
  [new_op] new_op:state_space(mamba_d_state,mamba_n_heads,mamba_d_conv,mamba_expand) 检测到 SSM/Mamba 配置键但无 layer_types: 层混合未建模, 需按 hybrid pattern 展开
  [new_op] new_op:layer_structure(parallel attn+ssm x44/44) checkpoint 每层同时带 self_attn.* 与 mamba.* 张量 ⇒ 同层并行两支求和, 不是"部分层注意力/部分层 SSM"的经典混合 (Falcon-H1: "Every FalconH1 decoder layer is hybrid (attention + mamba in the same block)")
  [hook] scale:muP{attention_out_multiplier=0.104167,embedding_multiplier=5.65685,key_multiplier=0.0306904,lm_head_multiplier=0.0130208,mlp_multipliers=[0.294628,0.0325521],ssm_in_multiplier=0.416667,ssm_multipliers=[0.353553,0.25,0.176777,0.5,0.353553],ssm_out_multiplier=0.117851} 嵌入/head/两支路/MLP 的逐支路缩放 (元素级乘, 含 ssm z/x/B/C/dt 与 mlp gate/down 的逐项表) ⇒ 漏掉即权重被按错误尺度使用
  [covered] head:tied=false                                独立 lm_head 对象 (嵌入键=model.embed_tokens.weight)
  [new_op] attn:head_geometry(12q/2kv@128)                引擎几何注册表未收 (现有 16/2@256, 24/4@256, 32/2@128); wrapper 还会按 q_heads 反推 KV 头 (wrapper/gqa_attention.cpp:25-30: 16->2) 再抛错
  [hook] attn:qk_norm=absent                            家族无条件 rmsnorm(q,k) (text_context_impl.h:958-959), 本模型无 qk-norm ⇒ 需 qk_norm_enabled() 门
  [hook] token_domain:vocab=130048!=family 248077       FrontendOptions.token_domain + official_specials 覆盖
  [hook] layers:44>16                                   cold_slots 类 per-layer 数组容量核对 (家族已扩 64, 新家族须审计)
  [post] quant_geometry                                 转换后校验: fp8/nvfp4 形状对照几何注册表 (A16 起步)
```

**分档计数**（改后 / 改前）：

| 模型 | new_op | hook | covered | post | 合计 |
|---|---|---|---|---|---|
| LFM2-2.6B-Exp | 2 (1) | 3 (3) | 2 (2) | 1 (1) | 8 (7) |
| Falcon-H1R-7B | 3 (2) | 4 (3) | 1 (1) | 1 (1) | 9 (7) |

（括号 = 改前的数，来自下面的 pre-A4 对照实验。）

**`config.h` 是否被扣**：**两个都被扣**，都只有 `config.h.BLOCKED`，都没有可编译的 `config.h`。
文件自身的 `#error` 行（第二道兜底）：

```
LFM2:  #error "auto-adapt: unmodelled layer kinds (conv) have no engine leaf / unresolved new_op gaps (new_op:layer_kinds(conv x22); attn:head_geometry(32q/8kv@64))"
Falcon:#error "auto-adapt: unresolved new_op gaps (new_op:state_space(mamba_d_state,mamba_n_heads,mamba_d_conv,mamba_expand); new_op:layer_structure(parallel attn+ssm x44/44); attn:head_geometry(12q/2kv@128))"
```

残余（**新发现，记录不改**）：Falcon 的 `.BLOCKED` 里 `full_attention_layers() { return 44; }` ——
`layer_kind_order` 为空时 `kind_map` 回退 `['Full'] * n`，数字是"回退值"，不是"44 层都是全注意力"。
这正是 §120 第 ③ 条说的"空 layer_types 回退全 Full"的特殊情况；因为文件带 `#error`，编译不出去，
所以是**文档性问题**而不是安全缺口。要更干净的话可让回退分支把 `full_attention_layers()` 也置 0 并加注释。

### 2.3 与 README / 模型卡对照：哪些是真缺口、哪些是漏检

取证方式（不下载权重）：

* 两个模型的 `model.safetensors.index.json`（LFM2 266 张量 / Falcon 751 张量）逐键统计 —— 本机文件；
* LFM2 参考实现 `dl/_lfm2_modeling.py`（§125 已存档的 HF `modeling_lfm2.py`，25,555 B）`Lfm2ShortConv` L324-389；
* Falcon-H1 参考实现 `transformers/models/falcon_h1/modeling_falcon_h1.py`（GitHub raw 实取）。

```
$ py -3 _collab/a4_scratch/a4_probe_index.py models/LFM2-2.6B-Exp
tensors: 266   layers seen: 30
non-layer tensors (2): model.embed_tokens.weight / model.embedding_norm.weight      <- 没有 lm_head
layer 0 keys: ['conv.conv','conv.in_proj','conv.out_proj','feed_forward.w1','feed_forward.w2',
               'feed_forward.w3','ffn_norm','operator_norm']
layer composition: {'other': 22, 'attn': 8}
$ py -3 _collab/a4_scratch/a4_probe_index.py models/Falcon-H1R-7B
tensors: 751   layers seen: 44
non-layer tensors (3): lm_head.weight / model.embed_tokens.weight / model.final_layernorm.weight
layer 0 keys: [... 'mamba.A_log','mamba.D','mamba.conv1d','mamba.dt_bias','mamba.in_proj','mamba.norm',
               'mamba.out_proj','self_attn.k_proj','self_attn.o_proj','self_attn.q_proj','self_attn.v_proj']
layer composition: {'attn+ssm': 44}      <- 44/44 层同时带两支
```

**LFM2-2.6B-Exp**

| 缺口 | 判定 | 依据 |
|---|---|---|
| `new_op:layer_kinds(conv x22)` | **真缺口** | config `layer_types` 22×conv + 8×full_attention；index 的 22 层带 `conv.{conv,in_proj,out_proj}`。§125 结案：`in_proj(3x) → B,C,x → B*x → 深度可分离因果卷积(宽3, groups=2048) → C* → out_proj`，**无激活、无 bias**，而引擎叶子 `causal_conv1d_silu` 是宽4+SiLU+无门控 ⇒ 要一个 no-SiLU/宽3 变体 + 两道逐元素门 + 一个新层型。**中等工作量，不是"配置问题"** |
| `attn:head_geometry(32q/8kv@64)` | **真缺口（改前被漏检）** | config 没有 `head_dim`，HF 约定 `2048/32=64`；引擎几何注册表只有 `16/2@256, 24/4@256, 32/2@128`（`engine_geometry_registry()` 实测）⇒ 32q/8kv@64 未注册 |
| `attn:qk_norm=present(weights,16)` | **真特性（改前判反了）** | config 里根本没有 qk-norm 旋钮，但 index 里 8 个注意力层各带 `self_attn.{q,k}_layernorm.weight`（共 16 张）。改前报 `=absent`("本模型无 qk-norm ⇒ 需 qk_norm_enabled() 门")，**极性相反** |
| `head:tied=true` → covered | 真特性、正确 | config `tie_embedding: true`（**别名**，见 §2.6 备注），index 无 `lm_head.weight` ⇒ 物化 268 MB |
| `attn:qk_norm` / `layers:30>16` / `token_domain` | 真 hook | 与引擎证据一致（家族上限 64 层、frontend token_domain） |

**Falcon-H1R-7B**

| 缺口 | 判定 | 依据 |
|---|---|---|
| `new_op:state_space(mamba_d_state,mamba_n_heads,mamba_d_conv,mamba_expand)` | **真缺口** | config 有全套 `mamba_*` 键、**没有 `layer_types`**；引擎无 SSD/Mamba 主路径 |
| `new_op:layer_structure(parallel attn+ssm x44/44)` | **真结构（改前只会说"层混合未建模"）** | index 44/44 层同时带 `self_attn.*` 与 `mamba.*`；参考实现原话 *"Every FalconH1 decoder layer is hybrid (attention + mamba in the same block)"*，两支**并行**后相加 ⇒ 不是"部分层注意力/部分层 SSD"的经典混合 |
| `hook scale:muP{...}` | **真缺口（改前完全没读）** | config 有 9 个 multiplier：`embedding_multiplier=5.657`、`lm_head_multiplier=0.013`、`attention_out_multiplier=0.104`、`key_multiplier=0.0307`、`ssm_in_multiplier=0.4167`、`ssm_out_multiplier=0.1179`、`ssm_multipliers[5]`、`mlp_multipliers[2]`。全是**语义性标量**（逐支路元素级乘），漏掉 = 权重按错误尺度使用 |
| `attn:head_geometry(12q/2kv@128)` | 真缺口 | 注册表未收 12q/2kv@128（`attn_layer_indices: null` 说明注意力在**每一层**，不是按索引稀疏出现） |
| `attn:qk_norm=absent` | 真特性、正确 | index 里 0 个 q/k norm 张量（与 LFM2 正好相反） |
| `head:tied=false` → covered | 真特性、正确 | index 有 `lm_head.weight`；`lm_head_multiplier` 另行拦住 |

⇒ 两个模型**都是真缺口**，导入器把 `config.h` 扣成 `.BLOCKED` 是**诚实的**：
§122 的结论（"要真写新算子，不该被'让它跑起来'的愿望覆盖"）在修补后**依然成立**，只是现在
**理由更完整**（conv 变体的具体规格、并行块结构、muP 缩放、几何未注册）。

### 2.4 导入器的 3 处新漏检/误判 + 最小修法（全部落在 `tools/archkit/adapt.py`）

| # | 病灶 | 最小修法 | 改后分类变化 |
|---|---|---|---|
| ① | **head 几何静默跳过**：`catalog_gaps` 的判定是 `if q and kv and hd`，而 LFM2 类 config 不写 `head_dim` ⇒ `hd=None` ⇒ **一个缺口都不报**，导入看起来干净 | `extract_spec` 里补 HF 约定的推导（`hidden // query_heads`），并记 `spec['head_dim_derived']=True` | LFM2 新增 `[new_op] attn:head_geometry(32q/8kv@64)` |
| ② | **qk-norm 极性判反**：只看 config 旋钮，而 LFM2 把 qk-norm 写成**模块名**（`q_layernorm`/`k_layernorm`），config 里没有任何标志 ⇒ 报 `=absent`（要求关掉），与权重证据相反 | 复用转换器已经读过的 index：`embedding_facts()` 改成同时回传张量名，新增 `layer_facts()`（只读 index，不读权重载荷），探测器先用权重证据 | LFM2 `[hook] attn:qk_norm=absent` → `[hook] attn:qk_norm=present(weights,16)` |
| ③ | **muP multiplier 完全没读**：`knobs` 白名单里没有这 10 个键，`catalog_gaps` 里也没有探测器 ⇒ 9 个缩放静默当成 1 | 白名单 +10 键；新增一个探测器（`hook`，与既有 `layer_scale:output_multiplier` 同档：标量逐支路乘，不是内核） | Falcon 新增 `[hook] scale:muP{...9 个缩放...}` |
| ④ | **未知层型的处置文案是错的**：原文 "短卷积可复用 causal_conv1d_silu 叶子" —— 而 §125 + 参考实现实证：叶子是宽4+SiLU+无门控，LFM2 是宽3+无激活+前后各一道门 | 文案改成按 `raw_text_config` 的实际 conv 旋钮（`conv_L_cache`/`conv_bias`）描述真实工作量 | LFM2 的 `layer_kinds(conv x22)` 行文案变化（档位不变） |
| ⑤ | **hybrid 结构说得太糊**："层混合未建模" 读起来像经典分段混合 | `layer_facts()` 增加"每层两支是否同时存在"的计数，命中即补一条 `new_op:layer_structure(...)` | Falcon 新增 `[new_op] new_op:layer_structure(parallel attn+ssm x44/44)` |

改动量：`tools/archkit/adapt.py` 一个文件，5 处（新增 `layer_facts()` / `_mult_str()` 两个小函数
+ 白名单 + 3 个探测点），**没有重构**，`catalog_gaps` 的既有分支一个都没删。

#### 回归控制（pre-A4 vs live）—— 证明只动了这两个模型

做法：把我的 `adapt.py` 存成 `_collab/a4_scratch/adapt_a4.py`，用 Edit **逐条反向回滚**成 pre-A4
（回滚后 `adapt.py` = **28851 B / 529 行**，与改动前 `ls` 记录**逐字节一致**），在**同一台机器、同一组
checkpoint** 上跑四个模型，再恢复 A4 版重跑一遍：

```
$ py -3 _collab/a4_scratch/a4_diff.py _collab/a4_scratch/pre
baseline: ...\_collab\a4_scratch\pre

=== lfm2-2.6b-exp ===
   baseline: 7 gaps {'covered': 2, 'hook': 3, 'new_op': 1, 'post': 1}
   live    : 8 gaps {'covered': 2, 'hook': 3, 'new_op': 2, 'post': 1}
   +ADDED   [new_op] attn:head_geometry(32q/8kv@64)
   +ADDED   [hook] attn:qk_norm=present(weights,16)
   -REMOVED [hook] attn:qk_norm=absent
   config.h          baseline=no         live=no         IDENTICAL
   config.h.BLOCKED  baseline=yes(1045)  live=yes(1078)  CHANGED

=== falcon-h1r-7b ===
   baseline: 7 gaps {'covered': 1, 'hook': 3, 'new_op': 2, 'post': 1}
   live    : 9 gaps {'covered': 1, 'hook': 4, 'new_op': 3, 'post': 1}
   +ADDED   [new_op] new_op:layer_structure(parallel attn+ssm x44/44)
   +ADDED   [hook] scale:muP{attention_out_multiplier=0.104167,embedding_multiplier=5.65685,key_multiplier=0.0306904,lm_head_multiplier=0.0130208,mlp_multipliers=[0.294628,0.0325521],ssm_in_multiplier=0.416667,ssm_multipliers=[0.353553,0.25,0.176777,0.5,0.353553],ssm_out_multiplier=0.117851}
   config.h          baseline=no         live=no         IDENTICAL
   config.h.BLOCKED  baseline=yes(1075)  live=yes(1125)  CHANGED

=== minicpm5-1b ===
   baseline: 6 gaps {'covered': 1, 'hook': 3, 'new_op': 1, 'post': 1}
   live    : 6 gaps {'covered': 1, 'hook': 3, 'new_op': 1, 'post': 1}
   config.h          baseline=no         live=no         IDENTICAL
   config.h.BLOCKED  baseline=yes(997)   live=yes(997)   IDENTICAL

=== spark-x2.5-4b ===
   baseline: 11 gaps {'covered': 1, 'hook': 6, 'new_op': 3, 'post': 1}
   live    : 11 gaps {'covered': 1, 'hook': 6, 'new_op': 3, 'post': 1}
   config.h          baseline=no         live=no         IDENTICAL
   config.h.BLOCKED  baseline=yes(1060)  live=yes(1060)  IDENTICAL
```

⇒ **只有 LFM2 与 Falcon 的缺口集变化**；MiniCPM5-1B 与 Spark-X2.5-4B 的 manifest 缺口集**逐项相同**、
`config.h.BLOCKED` **逐字节相同**（`IDENTICAL`）。原始日志：`_collab/a4_scratch/pre_run_output.txt`
（pre-A4）与 `_collab/a4_scratch/after_run_output.txt`（A4）。

**顺带更正一处此前记录**：`_collab/a4_scratch/before/`（我最初抓的快照，MiniCPM5 manifest 时间戳 09:32）
显示 MiniCPM5 当时**没有** `attn:head_geometry` / `attn:qk_norm` 两条 —— 那是**快照过期**（09:32 的 manifest
早于别人 09:4X-10:0X 加入这两个探测器：同一时刻 10:02 的 Falcon manifest 已经有它们）。
`MiniCPM5-1B` 现在 `config.h.BLOCKED`（新增 `attn:head_geometry(16q/2kv@128)`）**不是本次改动造成的**，
上面的 pre-A4 对照实验已经把它钉死（A4 前后逐字节相同）。

### 2.5 仍然成立/需要别人接手的（OPEN）

1. `tie_embedding` **别名**没被读：LFM2 的 config 写的是 `tie_embedding: true`，而
   `tools/convert/common/source_map.py:387 read_tie_flag()` 只认 `tie_word_embeddings`。
   本次**结论没受影响**（index 里没有 `lm_head.weight`，`decide_head` 走的是"没有 head 张量 = tied"这条路，
   `materialize=True` 正确），但 `manifest` 里的理由串 "the config omits tie_word_embeddings" 是**字面为真、
   实则误导**。修法在 `source_map.py`（S53/E10 的所有权），**我没动**：把 `tie_embedding` 加进
   `read_tie_flag` 的别名集即可。
2. `tools/archkit/check_params.py` 的 `COVERED_KEYS` 没有跟着加这 10 个 multiplier 键 —— 该工具是另一条
   独立审计线，本次**未动**；同一个人接活时补一行即可（否则它会把 Falcon 的 multiplier 报成"未覆盖"）。
3. Falcon 的 `full_attention_layers() { return 44; }` 回退数字（见 §2.2 末），文档性瑕疵。
4. 两个模型都**没有权重**（只有 config/index/tokenizer）⇒ 本轮所有几何/结构判定都是
   **config + index 级**证明；权重形状未复核（§123 曾用 range GET 4 MB 实证 LFM2 conv 形状 `[2048,1,3]`，
   那个结论被本次的 config/index 证据再次支持，但没有重复下载）。
5. 位置剖面的"真数字"没有在本轮产生：GUI 侧是**用 §M_patchA_effect / M_spec_4way 的历史数字**喂的解析器测试
   （`154/54/0.3506/22`、`133/6/0.0451/19/57`、`23,11,3,1,0,0,0`），**没有**启动引擎重跑（GPU 在训练）。
   真跑时需要 `--request-log-jsonl` 才有直方图。

---

## 3. 看板

`_collab/board.md` **主状态表**（文件顶部那张 6 列表）末尾追加了一行 `| A4 | ... |`（现为表格最后一行，
紧接 A1 之后、`## GPU 占用` 之前）。重排脚本：`_collab/a4_scratch/a4_board_fix.py`。

## 4. 本轮触达的文件（`git status --porcelain` 片段）

```
$ git -C ninfer-fusion-repo status --porcelain -- tools/gui tools/archkit
 M tools/archkit/adapt.py              <- 本次在其上做了 5 处最小修（该文件本轮之前已是 M，含 §119/§120/§121/S53 的改动）
 M tools/gui/serve_gui.py              <- 本次新功能（解析器 / 端点 / 卡片）
?? tools/gui/i18n_serve.py             <- 本次 +19 条词条（该文件尚未纳入 git）
```

本次的**净改动量**（不是 `git diff --stat` 的绝对值 —— 那两个文件在本轮之前就已经是 M）：

| 文件 | 本次改动 | 说明 |
|---|---|---|
| `tools/gui/serve_gui.py` | Python 新块 = `serve_gui.py:569-776`（**208 行，全新增**），另 CSS/HTML/JS/JS_KEYS 约 +95 行 | 既有函数只动了三处：`/api/state` 加一个键（+1 行）、新增 `/api/spec_stats` 分支（+2 行）、`do_POST` 的 `/api/start` 里多记 3 行会话状态。**`build_cmd()` 一行未动** —— 由 `serve_gui_selftest.py` 的"与 git HEAD 基线逐字节对拍 + 20000 次随机滑块"保证 |
| `tools/gui/i18n_serve.py` | +19 条 `serve.spec_*` | 只有新增，无改动 |
| `tools/archkit/adapt.py` | **28851 B / 529 行 → 37046 B / 662 行（+133 行）** | 新增 `layer_facts()` / `_mult_str()` 两个函数 + 白名单 10 键 + 3 个探测点 + `extract_spec` 的 head_dim 推导；revert 校验逐字节回到 28851 B |

（`_collab/` 在仓库之外，属工作目录：`_collab/A4_gui_specstats.md` 本文件、
`_collab/a4_scratch/` 证据脚本 + `before/` `pre/` `after/` 三份快照 + 原始日志。）

未改（逐个核对过）：`tools/gui/gui_i18n.py`、`tools/gui/gui_i18n_check.py`、`tools/gui/i18n_misc.py`、
`tools/gui/convert_gui.py`、`tools/gui/rag_gui.py`、`tools/gui/model_import.py`、
`tools/gui/serve_params.json`（结构未动）、`tools/gui/serve_gui_selftest.py`、
`tools/archkit/check_params.py`、`tools/convert/**`、`src/**`（工作树里 `src/**` 的 M 是别的窗口在编的，
本轮一次都没写）、`tests/**`。

## 5. 复跑命令

```bash
# 活 1 门禁
cd ninfer-fusion-repo/tools/gui && py -3 gui_i18n_check.py --only serve_gui.py     # 期望 VERDICT: PASS
py -3 serve_gui_selftest.py                                                        # 期望 29 passed, 0 failed
# 活 1 功能
cd ../../.. && py -3 _collab/a4_scratch/a4_gui_verify.py                           # 解析器/载荷/双语渲染
py -3 _collab/a4_scratch/a4_render.py en _collab/a4_scratch/a4_page_en.html
node _collab/a4_scratch/a4_js_check.js _collab/a4_scratch/a4_page_en.html          # 前端语法 + paintSpec
py -3 _collab/a4_scratch/a4_gui_http.py                                            # 真起 GUI 打两个端点
# 活 2
cd ninfer-fusion-repo && py -3 tools/archkit/adapt.py ../models/LFM2-2.6B-Exp --model-id lfm2-2.6b-exp
py -3 tools/archkit/adapt.py ../models/Falcon-H1R-7B --model-id falcon-h1r-7b
py -3 ../_collab/a4_scratch/a4_diff.py ../_collab/a4_scratch/pre                   # pre-A4 对照
```

（`a4_probe_*.py` / `a4_copy.py` / `a4_runall.py` / `a4_show.py` / `a4_size.py` / `a4_gui_verify.py` /
`a4_gui_http.py` 里的路径都是绝对路径常量，从哪个 cwd 跑都一样；`a4_diff.py` / `a4_render.py` 收命令行参数。
`a4_js_check.js` 需要 node，本机 v24.18.0。）
