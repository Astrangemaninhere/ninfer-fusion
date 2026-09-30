# CONTRACT — the shared writer + the shared gate, in one page

One sentence:

> **A new model writes a SPEC plus three tables (roles, expressions, ties) and
> nothing else; the shared driver owns every byte, every refusal and every gate,
> and the shared gate battery refuses the declaration unless it accounts for the
> checkpoint index exactly.**

The model-side code is `tools/convert/<id>/{spec,inventory,recipe,declaration,convert}.py`.
The shared code is `tools/archkit/kit/`. The boundary between them is the whole
design: a model-side file that opens the output for writing, or that decides what a
missing `config.json` means, has crossed it and `kit/check.py` will say so.

---

## 1. What a new model writes

| # | file | what |
|---|---|---|
| 1 | `tools/archkit/specs/<id>_spec.json` | geometry + layer kinds. Validated by `tools/archkit/arch_spec.py:load_spec`, which refuses a spec whose `geometry` is empty — an extraction that found no geometry keys is not a spec. |
| 2 | `tools/convert/<id>/inventory.py` | `MODEL_ID`, `WEIGHTS_ID`, `IGNORED_SOURCE_PREFIXES`, and the ordered `(object name, source keys, role)` vocabulary. |
| 3 | `tools/convert/<id>/recipe.py` | `RECIPES: role -> (obj, source, ctx) -> payload`, plus the tie rule. |
| 4 | `tools/convert/<id>/declaration.py` | `build_declaration(source) -> Declaration`. Pure: no file opened, no byte read. |
| 5 | `tools/convert/<id>/convert.py` | `driver.main(build_declaration)`. Five lines plus argparse is the target size. |

Measured sizes for MiniCPM5-1B: `inventory.py` 122 lines, `recipe.py` 58,
`declaration.py` 108, `convert.py` 51. The muse_glimmer_30b converter is 555 lines
with a 40-arm if/elif chain over object names inside `convert()`; spark's is 1,194.
The difference is not terseness, it is that those two each own a copy of the driving.

## 2. What the shared driver owns

`tools/archkit/kit/driver.py`:

1. `config.json` — absent ⇒ `REFUSED: missing config.json under <dir>`.
2. `model.safetensors.index.json` — absent ⇒ `REFUSED: missing model.safetensors.index.json under <dir>`.
3. the spec, through the tree's own loader.
4. the object plan, through `tools/artifact/container.plan_objects` (offsets, alignment, encoded sizes).
5. the gate battery (`kit/gate.py`).
6. the write, through `tools/artifact/container.ArtifactWriter` — the one place in this path that opens the output.
7. the re-open, through `tools/artifact/container.Artifact`, plus bytes and sha256.

Every failure is one exception type, `KitRefusal`, whose message begins by naming
the missing file or the gate that refused. The container's own `ArtifactError` is
re-raised as `container-writer: …` so a caller never needs to know which module
raised.

## 3. The gate battery (`kit/gate.py`)

| gate | reading | refuses when |
|---|---|---|
| `plan` | N objects, M payload bytes | the container rejects a shape/format/layout |
| `spec-vs-config` | K geometry fields agree | the spec disagrees with `config.json` |
| `layer-kinds` | `full=n, sliding=m, gdn=k` | the schedule does not cover every layer, or uses a kind outside the shared runtime's vocabulary |
| `source-coverage` | `objects=N index_keys=M consumed=C ignored=I C+I=M disjoint=yes` | any index key is neither consumed nor ignored; an ignore rule matches nothing; a key is consumed twice by non-aliases; a consumed key is not in the index |
| `geometry` | K objects checked against the shard headers | a declared shape is not what the shard holds; a fused object's sources do not stack to the declared shape; a BF16 source declared as another format |

## 4. Ties are aliases, not copies

`ObjectDecl.tie_to` names another object. The driver resolves the payload through
the tie target, so the byte stream is produced once per read and the declaration
stays a declaration. A tied object still declares its full SHAPE (the artifact
carries a full-length object under that name) and narrows its `sources` to the keys
the checkpoint actually stores — a checkpoint that never wrote `lm_head.weight`
still ties, and `source-coverage` stays exact.

## 5. Acceptance: three tiers, never mixed

| tier | means | how it is established |
|---|---|---|
| **a** | a spec / registration exists, no runtime | `registry.cpp` refuses by name |
| **b** | a converter exists AND produced an artifact | `kit/verify.py`: object count, order, per-object geometry and byte count against the shard headers, sha256, magic |
| **c** | an artifact was really loaded by the engine and emitted tokens | an engine run, its own log |

`AdaptReport`/`check.py` can only ever report **a**; tiers b and c need a produced
file and a real load, and a scaffold that claimed them would be claiming somebody
else's evidence.

## 6. Verifying a produced artifact

```
python3 -m tools.archkit.kit.verify --artifact <file>.ninfer --model <ckpt-dir> \
  --declaration tools.convert.<id>.declaration:build_declaration
```

The `--declaration` argument is required for the source-equality reading and is not
a convenience: the artifact's object names are the ENGINE's vocabulary
(`text/layers/3/attention/query`), the index's keys are HF's
(`model.layers.3.self_attn.q_proj.weight`), and a verifier that compared the two
sets directly would report 219 missing objects while measuring its own ignorance.
