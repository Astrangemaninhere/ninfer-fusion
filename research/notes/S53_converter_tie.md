# S53 — 转换器侧：嵌入别名解析 + tied head 物化

**一句话结论**：`tools/convert/` 里加了两件可复用能力 ——（a）按优先级识别真实嵌入键并**把命中的名字写进报告**（不静默猜）；
（b）`tie_word_embeddings=true` 或索引里没有 `lm_head.weight` 时，把嵌入物化成独立的 `text/output_head`（`[vocab, hidden]`），
并在报告里给出**来源 + 体积**。用真实 index（Spark-X2.5-4B + 4 个对照）自检 **61 项检查 0 失败（RESULT: PASS）**；
真权重只跑到"head 物化"那一步（671,088,640 B 物化成功）。**没有**跑完整转换、**没有**写 src/、**没有**开 nvcc/GPU。
未验证项见 §6。

---

## 1. 改动前现状（已核实）

| 事实 | 位置 |
|---|---|
| 嵌入键硬写在 recipe | `tools/convert/qwen3_6_27b/recipe.py:56` → `model.language_model.embed_tokens.weight`（shape `(248320,5120)`） |
| head 键硬写在 recipe（两处） | `recipe.py:203`（`text/output_head`）、`recipe.py:215`（`text/draft_head` 的 `GatherRows` 源）→ 都是 `lm_head.weight` |
| tie 硬写 False | `convert.py:40`（`_ROOT_CONFIG`）、`convert.py:63`（`_TEXT_CONFIG`）—— 本次两行都删除 |
| 输出对象名 | `text/token_embedding`（`inventory.py:48`，Q6，`[248320,5120]`）、`text/output_head`（`inventory.py:91`，Q6，`[248320,5120]`） |
| 引擎要什么 | `src/targets/qwen3_6_27b/impl/load/bindings.cpp:423-445` 按 `{248320,5120}` 绑定两个对象，`text_context_impl.h:358` 取 `weights_.output_head` ⇒ 引擎加载的是**独立 head 对象、朝向 `[vocab, hidden]`**，tied 与否与引擎无关 |
| 报告里原本没有任何名字信息 | `conversion.py:221-226` 的 `source_preflight` 只有 recipes/tensors/shards/dtypes 计数 |

⇒ 缺口定位成立：tied checkpoint 的出路就是**转换期物化**，引擎侧不需要新算子（按现成的独立 head 路径加载）。

## 2. 改了哪些文件

| 文件 | 变化 | 说明 |
|---|---|---|
| `tools/convert/common/source_map.py` | **新增** 507 行（纯标准库，不 import torch） | 候选表 `EMBEDDING_NAME_PRIORITY`(`model.embed_tokens.weight`,`model.embedding.weight`,`embed_tokens.weight`,`model.language_model.embed_tokens.weight`)、`OUTPUT_HEAD_NAME_PRIORITY`(`lm_head.weight`,`model.lm_head.weight`)；`load_weight_map`（索引，或单文件 `model.safetensors` 的 header）；`read_safetensors_tensors/read_safetensors_header`（**只读 header，不读权重数据**）；`tensor_fact`（shape/dtype/分片是否下完）；`resolve_name/resolve_embedding/resolve_head`（返回命中名 + `matched_rank` + 所有存在的候选；**都不存在则报错**，错误里列出候选与样本键）；`read_tie_flag`；`decide_head`（物化判定 + `reason` + `flag_conflict`）；`orient_embedding_for_head`（identity / transposed，其它形状报错）；`source_tensor_bytes` |
| `tools/convert/qwen3_6_27b/recipe.py` | +314/−8 | 两个源键提成常量 `EMBEDDING_SOURCE`/`OUTPUT_HEAD_SOURCE`；新增 `SourceBinding`（含 `rebind()`、`head_expression()`、`to_report()`）、`resolve_sources()`、`source_recipes()/recipes_for()`；`preflight_sources(model_dir, binding=None)` —— **binding=None 时行为与今天完全一致**；rebind 只改写这两个键（含 draft head 的 `GatherRows` 源），tied 时 head 表达式 = 嵌入本身（若嵌入存成 `[hidden,vocab]` 则 `Transpose((1,0))`），并且 `[hidden,vocab]` + draft head 组合会**显式报错**而不是静默错取行 |
| `tools/convert/qwen3_6_27b/convert.py` | +64/−5 | 删掉两处硬写 `tie_word_embeddings: False`；`config_summary` 改为**上报**解析到的 tie 值；`preflight_conversion` 先 `resolve_sources()` 再 preflight（tied 时来源换了，必须先把 head recipe 换掉），并打印命中名/rank/provenance/体积；`materialize_tensor(..., recipes=None)` 默认仍取注册表（`qwen3_8_27b/convert.py:145` 的委托调用不受影响）；报告新增 `source_binding` 段 |
| `tools/convert/qwen3_6/common/conversion.py` | +14/−2 | `build_conversion_report(..., extra=None)`：只有传了才加键，**其它调用方报告逐字节不变**（离线用例已断言 `source_binding` 不出现、`source_preflight` 字典原样） |
| `tools/convert/common/__init__.py` | +26/−0 | 导出 source_map 的公共名字 |
| `tools/convert/check_source_map.py` | **新增** 702 行 | 自检脚本（真实 index + 合成索引 + 有界真权重干跑 + 全量物化开关 `--full-materialize`） |

**报告新增段落**（真实 Spark 数据跑出来的 `source_binding.output_head`，节选）：

```json
{
  "object": "text/output_head",
  "object_shape": [131072, 2560],
  "registered_object_spec_applies": false,
  "tie_word_embeddings": true,
  "head_in_index": false,
  "head_name": null,
  "materialized": true,
  "materialized_from": "model.embedding.weight",
  "transpose": false,
  "orientation_verified": true,
  "provenance": "identity-from-embedding",
  "reason": "tie_word_embeddings is true and the index has no lm_head.weight",
  "tie_flag_conflict": false,
  "source_tensor": "model.embedding.weight",
  "source_shape": [131072, 2560],
  "source_dtype": "BF16",
  "source_bytes": 671088640,
  "embedding_header": {"shard": "model-00001-of-00005.safetensors", "shard_complete": true}
}
```

（`registered_object_spec_applies=false` 是因为这是"探测另一套几何"的路径：Spark 的对象格式要在它自己的 target 里注册；
在本机它没有 inventory，所以不报 Q6 体积而不是报一个错的数。Spark 的真对象体积 = 同一个张量 = 671,088,640 B @BF16。）

**「逐字节一致」的边界（重要）**：tie=False 且索引里有 `lm_head.weight` 时，**artifact 字节一致** —— 绑定后的 recipe 与注册表逐元素相等
⇒ 同样的源、同样的表达式、同样的编码。变化只出现在**描述性报告**里：① 多出 `source_binding` 段（本次要的审计信息：命中键/rank/provenance/体积）；
② `config_summary` 里原先硬写的 `text.tie_word_embeddings` 移到顶层 `tie_word_embeddings` 并改为**实测值**（注册 checkpoint 实测同样是 `False`，
值不变、只是不再写死）。报告其余字段（`source_preflight` 计数 / `objects` / `artifact`）不变；离线用例还断言了"不传 binding 时连 `source_binding`
键都不出现"，保证其它调用方零影响。

## 3. 自检原始输出（完整，`python tools/convert/check_source_map.py --full-materialize`）

```
==============================================================================
real safetensors indexes
==============================================================================

--- spark-x2.5-4b: TIED: config tie_word_embeddings=true, 290 tensors, no lm_head.weight at all
model dir      : C:\Users\User\Documents\ziqinzhang\models\Spark-X2.5-4B
weight map     : C:\Users\User\Documents\ziqinzhang\models\Spark-X2.5-4B\model.safetensors.index.json (290 tensors, synthetic=False)
config         : tie_word_embeddings=True geometry(vocab,hidden)=(131072, 2560)
candidates     : ['model.embed_tokens.weight', 'model.embedding.weight', 'embed_tokens.weight', 'model.language_model.embed_tokens.weight']
present        : ['model.embedding.weight']
embedding      : model.embedding.weight (rank 1 of 4, primary=False)
head resolution: in_index=False name=None
decision       : materialize=True -- tie_word_embeddings is true and the index has no lm_head.weight
embedding      : (131072, 2560) BF16 in model-00001-of-00005.safetensors (complete=True)
head           : header unavailable (index-only or shard missing)
  [ok  ] spark-x2.5-4b: embedding == model.embedding.weight
  [ok  ] spark-x2.5-4b: matched_rank points at model.embedding.weight
  [ok  ] spark-x2.5-4b: tie_word_embeddings == True
  [ok  ] spark-x2.5-4b: materialize == True
orientation    : head object (131072, 2560) vs embedding (131072, 2560) -> identity (transpose=False)
source volume  : (131072, 2560) BF16 = 671088640 bytes (640.0 MiB)
  [ok  ] orientation is identity (no transpose) as the checkpoint layout implies
  [ok  ] tied embedding (131072, 2560) already has the head object shape (131072, 2560) (no transpose needed)
real preflight : text/token_embedding <- model.embedding.weight (131072, 2560), text/output_head <- model.embedding.weight (131072, 2560)
                 recipes=2 tensors=1 shards=1 dtypes={'BF16': 1}
  [ok  ] resolved names resolve to real tensors in the real shards
binding report : object=text/output_head object_shape=[131072, 2560] registered_spec=False provenance=identity-from-embedding transpose=False orientation_verified=True
                 source=model.embedding.weight [131072, 2560] BF16 = 671088640 bytes; head object bytes=None
  [ok  ] spark-x2.5-4b: head materialized from model.embedding.weight
  [ok  ] spark-x2.5-4b: orientation_verified matches embedding header availability
  bounded dry run of the materialized head:
  head expression: SourceTensor(name='model.embedding.weight', shape=(131072, 2560), dtype='BF16')
  provenance     : identity-from-embedding
  read           : (16, 2560) from model.embedding.weight (BF16)
  materialized   : (16, 2560) (identity)
  [ok  ] materialized slice keeps the embedding row width 2560
  [ok  ] materialized slice survives a BF16 encode/decode round trip
  bf16 round trip: 81920 bytes for 16 rows
  encoder probe  : text/output_head Q6G64_F16S of the registered 27B object format encodes 16 rows to 32000 bytes
  full volumes   : embedding (131072, 2560) BF16 = 671088640 bytes (640.0 MiB); the tied head object carries the same tensor
  full head expression: SourceTensor(name='model.embedding.weight', shape=(131072, 2560), dtype='BF16')
  materialized shape  : (131072, 2560) dtype=torch.bfloat16
  [ok  ] whole head materializes to the object contract (131072, 2560)
  materialized bytes  : 671088640
  [ok  ] materialized size equals the BF16 source size
  row 0 first values  : [-0.007598876953125, -0.0029144287109375, 0.0015106201171875]
  [ok  ] row 0 is not all zeros

--- lfm2-2.6b-exp: real index with no lm_head.weight and no tie key in config (no shards on disk)
model dir      : C:\Users\User\Documents\ziqinzhang\models\LFM2-2.6B-Exp
weight map     : C:\Users\User\Documents\ziqinzhang\models\LFM2-2.6B-Exp\model.safetensors.index.json (266 tensors, synthetic=False)
config         : tie_word_embeddings=None geometry(vocab,hidden)=(65536, 2048)
candidates     : ['model.embed_tokens.weight', 'model.embedding.weight', 'embed_tokens.weight', 'model.language_model.embed_tokens.weight']
present        : ['model.embed_tokens.weight']
embedding      : model.embed_tokens.weight (rank 0 of 4, primary=True)
head resolution: in_index=False name=None
decision       : materialize=True -- index has no lm_head.weight and the config omits tie_word_embeddings
embedding      : header unavailable (index-only or shard missing)
head           : header unavailable (index-only or shard missing)
  [ok  ] lfm2-2.6b-exp: embedding == model.embed_tokens.weight
  [ok  ] lfm2-2.6b-exp: matched_rank points at model.embed_tokens.weight
  [ok  ] lfm2-2.6b-exp: tie_word_embeddings == None
  [ok  ] lfm2-2.6b-exp: materialize == True
orientation    : UNVERIFIED (no embedding header on disk)
real preflight : SKIPPED (needs a config geometry and a local header)
binding report : object=text/output_head object_shape=[65536, 2048] registered_spec=False provenance=identity-from-embedding (orientation unverified) transpose=False orientation_verified=False
                 source=None None None = None bytes; head object bytes=None
  [ok  ] lfm2-2.6b-exp: head materialized from model.embed_tokens.weight
  [ok  ] lfm2-2.6b-exp: orientation_verified matches embedding header availability
  bounded dry run of the materialized head:
  dry run: SKIPPED (embedding shard not on disk; index-only)
  full materialization: SKIPPED (embedding shard not on disk)

--- qwen3.8-flash-next: UNTIED control: index ships lm_head.weight (same alias as the registered 27B)
model dir      : C:\Users\User\Documents\ziqinzhang\models\Qwen3.8-Flash-Next-ABLITERATED-NVFP4
weight map     : C:\Users\User\Documents\ziqinzhang\models\Qwen3.8-Flash-Next-ABLITERATED-NVFP4\model.safetensors.index.json (296475 tensors, synthetic=False)
config         : tie_word_embeddings=False geometry(vocab,hidden)=(248320, 2560)
candidates     : ['model.embed_tokens.weight', 'model.embedding.weight', 'embed_tokens.weight', 'model.language_model.embed_tokens.weight']
present        : ['model.language_model.embed_tokens.weight']
embedding      : model.language_model.embed_tokens.weight (rank 3 of 4, primary=False)
head resolution: in_index=True name=lm_head.weight
decision       : materialize=False -- index ships lm_head.weight
embedding      : (248320, 2560) BF16 in model-bf16-00012.safetensors (complete=True)
head           : (248320, 2560) BF16 in model-bf16-00012.safetensors (complete=True)
  [ok  ] qwen3.8-flash-next: embedding == model.language_model.embed_tokens.weight
  [ok  ] qwen3.8-flash-next: matched_rank points at model.language_model.embed_tokens.weight
  [ok  ] qwen3.8-flash-next: tie_word_embeddings == False
  [ok  ] qwen3.8-flash-next: materialize == False
orientation    : head object (248320, 2560) vs embedding (248320, 2560) -> identity (transpose=False)
source volume  : (248320, 2560) BF16 = 1271398400 bytes (1212.5 MiB)
  [ok  ] orientation is identity (no transpose) as the checkpoint layout implies
real preflight : text/token_embedding <- model.language_model.embed_tokens.weight (248320, 2560), text/output_head <- lm_head.weight (248320, 2560)
                 recipes=2 tensors=2 shards=1 dtypes={'BF16': 2}
  [ok  ] resolved names resolve to real tensors in the real shards
recipe head    : SourceTensor(name='lm_head.weight', shape=(248320, 5120), dtype='BF16')
recipe embed   : SourceTensor(name='model.language_model.embed_tokens.weight', shape=(248320, 5120), dtype='BF16')
recipe draft   : SourceTensor(name='lm_head.weight', shape=(248320, 5120), dtype='BF16')
registered recipes identical: True (matched alias equals the registered literal: True)
  [ok  ] object order preserved
  [ok  ] registered literal aliases rebind to the identical recipes
  [ok  ] untied control reads the head straight from the index (no transpose)

--- falcon-h1r-7b: untied control, index-only (no shard headers on disk)
model dir      : C:\Users\User\Documents\ziqinzhang\models\Falcon-H1R-7B
weight map     : C:\Users\User\Documents\ziqinzhang\models\Falcon-H1R-7B\model.safetensors.index.json (751 tensors, synthetic=False)
config         : tie_word_embeddings=False geometry(vocab,hidden)=(130048, 3072)
candidates     : ['model.embed_tokens.weight', 'model.embedding.weight', 'embed_tokens.weight', 'model.language_model.embed_tokens.weight']
present        : ['model.embed_tokens.weight']
embedding      : model.embed_tokens.weight (rank 0 of 4, primary=True)
head resolution: in_index=True name=lm_head.weight
decision       : materialize=False -- index ships lm_head.weight
embedding      : header unavailable (index-only or shard missing)
head           : header unavailable (index-only or shard missing)
  [ok  ] falcon-h1r-7b: embedding == model.embed_tokens.weight
  [ok  ] falcon-h1r-7b: matched_rank points at model.embed_tokens.weight
  [ok  ] falcon-h1r-7b: tie_word_embeddings == False
  [ok  ] falcon-h1r-7b: materialize == False
orientation    : UNVERIFIED (no embedding header on disk)
real preflight : SKIPPED (needs a config geometry and a local header)
recipe head    : SourceTensor(name='lm_head.weight', shape=(248320, 5120), dtype='BF16')
recipe embed   : SourceTensor(name='model.embed_tokens.weight', shape=(248320, 5120), dtype='BF16')
recipe draft   : SourceTensor(name='lm_head.weight', shape=(248320, 5120), dtype='BF16')
registered recipes identical: False (matched alias equals the registered literal: False)
  [ok  ] object order preserved
  [ok  ] rewritten alias appears in the bound embedding recipe
  [ok  ] untied control reads the head straight from the index (no transpose)

--- minicpm5-1b: untied control with a local single shard
model dir      : C:\Users\User\Documents\ziqinzhang\models\MiniCPM5-1B
weight map     : C:\Users\User\Documents\ziqinzhang\models\MiniCPM5-1B\model.safetensors.index.json (219 tensors, synthetic=False)
config         : tie_word_embeddings=False geometry(vocab,hidden)=(130560, 1536)
candidates     : ['model.embed_tokens.weight', 'model.embedding.weight', 'embed_tokens.weight', 'model.language_model.embed_tokens.weight']
present        : ['model.embed_tokens.weight']
embedding      : model.embed_tokens.weight (rank 0 of 4, primary=True)
head resolution: in_index=True name=lm_head.weight
decision       : materialize=False -- index ships lm_head.weight
embedding      : (130560, 1536) BF16 in model-00000-of-00001.safetensors (complete=True)
head           : (130560, 1536) BF16 in model-00000-of-00001.safetensors (complete=True)
  [ok  ] minicpm5-1b: embedding == model.embed_tokens.weight
  [ok  ] minicpm5-1b: matched_rank points at model.embed_tokens.weight
  [ok  ] minicpm5-1b: tie_word_embeddings == False
  [ok  ] minicpm5-1b: materialize == False
orientation    : head object (130560, 1536) vs embedding (130560, 1536) -> identity (transpose=False)
source volume  : (130560, 1536) BF16 = 401080320 bytes (382.5 MiB)
  [ok  ] orientation is identity (no transpose) as the checkpoint layout implies
real preflight : text/token_embedding <- model.embed_tokens.weight (130560, 1536), text/output_head <- lm_head.weight (130560, 1536)
                 recipes=2 tensors=2 shards=1 dtypes={'BF16': 2}
  [ok  ] resolved names resolve to real tensors in the real shards
recipe head    : SourceTensor(name='lm_head.weight', shape=(248320, 5120), dtype='BF16')
recipe embed   : SourceTensor(name='model.embed_tokens.weight', shape=(248320, 5120), dtype='BF16')
recipe draft   : SourceTensor(name='lm_head.weight', shape=(248320, 5120), dtype='BF16')
registered recipes identical: False (matched alias equals the registered literal: False)
  [ok  ] object order preserved
  [ok  ] rewritten alias appears in the bound embedding recipe
  [ok  ] untied control reads the head straight from the index (no transpose)

==============================================================================
synthetic name maps (no local checkpoint carries these spellings)
==============================================================================
- hf embed_tokens + lm_head
    embedding=model.embed_tokens.weight (rank 0) materialize=False reason=index ships lm_head.weight
  [ok  ] hf embed_tokens + lm_head: embedding name
  [ok  ] hf embed_tokens + lm_head: materialize == False
- bare embed_tokens + lm_head
    embedding=embed_tokens.weight (rank 2) materialize=False reason=index ships lm_head.weight
  [ok  ] bare embed_tokens + lm_head: embedding name
  [ok  ] bare embed_tokens + lm_head: materialize == False
- nested alias + model.lm_head
    embedding=model.language_model.embed_tokens.weight (rank 3) materialize=False reason=index ships model.lm_head.weight
  [ok  ] nested alias + model.lm_head: embedding name
  [ok  ] nested alias + model.lm_head: materialize == False
- tied: bare embedding, no head
    embedding=embed_tokens.weight (rank 2) materialize=True reason=index has no lm_head.weight and the config omits tie_word_embeddings
  [ok  ] tied: bare embedding, no head: embedding name
  [ok  ] tied: bare embedding, no head: materialize == True
- registered 27B names (alias + head present)
    embedding=model.language_model.embed_tokens.weight (rank 3) materialize=False reason=index ships lm_head.weight
  [ok  ] registered 27B names (alias + head present): embedding name
  [ok  ] registered 27B names (alias + head present): materialize == False
    registered-name binding head=SourceTensor(name='lm_head.weight', shape=(248320, 5120), dtype='BF16') embed=SourceTensor(name='model.language_model.embed_tokens.weight', shape=(248320, 5120), dtype='BF16') identical=True
  [ok  ] registered names rebind to the identical recipe set
- orientation helper
  [ok  ] equal shapes -> identity, no transpose
  [ok  ] reversed shapes -> transposed
    raised: embedding shape (131072, 2048) cannot serve as output head (131072, 2560) (neither identical nor transposed)
  [ok  ] mismatched shapes raise SourceMapError
- missing embedding is a hard error, never a guess
    raised: no embedding tensor found in <synthetic: no embedding>: tried ['model.embed_tokens.weight', 'model.embedding.weight', 'embed_tokens.weight', 'model.language_model.embed_t
  [ok  ] unresolvable embedding raises
- tie=false with no head materializes but is flagged as a conflict
    materialize=True conflict=True reason=index has no lm_head.weight (or model.lm_head.weight) although tie_word_embeddings is false
  [ok  ] materialized and flagged as tie_flag_conflict

==============================================================================
summary
==============================================================================
checks: 61, failures: 0
RESULT: PASS
```

## 4. 判定结果（真实 index）

| 案例 | 数据来源 | 命中嵌入键 | rank | tie | 物化 | 依据/备注 |
|---|---|---|---|---|---|---|
| **Spark-X2.5-4B** | 真索引 (290 张量) + 真分片1 header + 真权重首段 | **`model.embedding.weight`** | 1/4 | `true` | **是** | 索引里**没有** `lm_head.weight`；header 实测 `(131072,2560) BF16`，分片1 `complete=True`；朝向 identity（嵌入已经是 `[vocab,hidden]`）；物化体积 671,088,640 B |
| LFM2-2.6B-Exp | 真索引 (266) | `model.embed_tokens.weight` | 0/4 | 缺失(None) | 是 | 真索引里也没有 head ⇒ 走"索引无 head"分支；无分片 ⇒ 朝向标 `orientation_verified=false` |
| Qwen3.8-Flash-Next | 真索引 (296,475) + 真分片 header | `model.language_model.embed_tokens.weight` | 3/4 | `false` | 否 | 索引有 `lm_head.weight` ⇒ 直接按索引读；绑定的 1118 条 recipe 与注册表**逐元素相等**（`identical=True`） |
| Falcon-H1R-7B | 真索引 (751)，无分片 | `model.embed_tokens.weight` | 0/4 | `false` | 否 | 索引有 head；别名与注册字面量不同 ⇒ 绑定 recipe 正确改写嵌入名、head 不动 |
| MiniCPM5-1B | 真索引 (219) + 真单分片 | `model.embed_tokens.weight` | 0/4 | `false` | 否 | 同上 |
| （合成）`model.lm_head.weight` | 合成索引 | `model.language_model.embed_tokens.weight` | 3/4 | — | 否 | **只有合成数据**覆盖第二 head 候选 |
| （合成）嵌入存成 `[hidden,vocab]` | 合成/helper | — | — | — | 是(转置) | `orient_embedding_for_head` 返回 `transposed`；真实 checkpoint 全是 identity |
| （合成）4 个候选都不在 | 合成索引 | **报错** | — | — | — | `SourceMapError`：列出候选 + 前 8 个真实键 |

真权重干跑（有界 + 全量）：

- 有界：读 `model.embedding.weight[0:16]`（真分片）→ identity → BF16 往返 81,920 B；注册 27B 对象格式 Q6 编码 16 行 = 32,000 B（`encoded_size` 一致）。
- **全量**（`--full-materialize`）：走转换器自己的 `materialize_expression`，`materialized shape (131072, 2560) dtype=bfloat16`，
  `materialized bytes = 671088640`（= BF16 源大小），row0 前 3 值 `[-0.007598876953125, -0.0029144287109375, 0.0015106201171875]`（非零真数据）。

## 5. `tools/archkit/adapt.py`：只动 tie/嵌入名（**请 M 裁决一处改判**）

- 新增 `embedding_facts()`（`adapt.py:121`）：**复用转换器的 `source_map` 解析器**（不另抄一份候选表），把 `name/matched_rank/candidates/present/head_name/head_in_index/materialize/reason/flag_conflict/source_bytes` 记进 `spec['embedding']`（`adapt.py:101`），因此 manifest/spec 里现在有**命中键**而不是假设。
- `head:tied=false` 行：覆盖时附带 `(嵌入键=…)`。
- `head:tied=true` 行：由 `new_op` 改判为 **`covered`**（当索引确认物化时），文本写明"转换期把嵌入键 X 物化为 `text/output_head`，引擎按独立 lm_head 加载，无需新算子；已实现于 …"。**副作用**：Spark 的 `head:tied=true` 原本是它唯一的 `new_op` ⇒ `out/spark-x2.5-4b/config.h.BLOCKED` 变成 `config.h`（`blocked` 集合空）。
- `manifest.json` diff（对照 `_collab/s53_scratch/before_manifest.json`）：gap 列表 **need/顺序完全一致**，只有该行 tier 变化 + 新增 `spec.embedding`；`config.h` 其余字段不变（hidden 2560/layers 36/vocab 131072/…）。
- **若要回滚这条改判**：把 `adapt.py` 里 `'covered' if emb.get('materialize') else 'new_op'` 改成 `'new_op'` 一行即可；`out/` 可从 `_collab/s53_scratch/before_*` 还原。
- `adapt.py` 工作区总 diff 是 +200/−11，但其中绝大部分是**上一个会话**已存在的 Spark 旋钮/rope/gate 改动；本次新增只有 3 处：`embedding_facts()`（:121-）、`spec['embedding']`（:101）、tie 两行（:245-262）。

## 6. 未验证项（严格区分）

1. **真实 qwen3.6-27B checkpoint 不在本机** ⇒「tie=False 且索引里有 lm_head 时逐字节一致」是在**表达式级**证明的（绑定后的 1118 条 recipe 与注册表逐元素 `==`，且 `GatherRows`/`Transpose` 结构不变），**不是**在真实 27B 权重上跑出来的。无本机反例可跑。
2. Spark 的**完整转换未跑**：分片 2..5 仍在下载（分片1 `complete=True`），且 Spark 尚无自己的 target/inventory/资源文件。
3. 全量物化只到 **head 张量**这一步：**没有**编码成 artifact、没写 `.ninfer`、没做引擎加载/数值比对。
4. 引擎/GPU/nvcc 全程未用；`tests/convert` 需要 pytest（本机无），改用 `_collab/s53_scratch/run_offline_tests.py` 等价跑两条离线用例（唯一差异：原测试的 `report["source"]["model_path"].endswith("/model")` 是 POSIX-only 断言）。
5. `model.lm_head.weight`（第二 head 候选）**只有合成索引**覆盖，本机没有真实 checkpoint 命中。
6. 转置分支（嵌入存成 `[hidden,vocab]`）只有 helper/合成级验证；真实 checkpoint 全部是 identity。
7. `flag_conflict`（config 说 tie=false 但索引没有 head）只做了合成验证：行为=物化 + 报告 `tie_flag_conflict=true` + preflight 打印 warning。

## 7. 复现

```bash
cd ninfer-fusion-repo
python tools/convert/check_source_map.py                  # 61 checks, RESULT: PASS（有界干跑）
python tools/convert/check_source_map.py --full-materialize   # 额外做一次 671MB 全量物化（约 0.7GB 峰值内存）
python tools/convert/check_source_map.py --model <dir>     # 只测一个目录
python tools/archkit/adapt.py <model_dir>                  # 重新生成 manifest/config.h（tie 行+索引解析）
python "C:/Users/User/Documents/ziqinzhang/_collab/s53_scratch/run_offline_tests.py"   # 离线等价用例
```

产物/中间件：`_collab/s53_scratch/`（`before_*` 是 adapt.py 改动前的 Spark 输出备份、`s53_check_output.txt` 是原始输出、`s53_report_section.json` 是报告样例）。
