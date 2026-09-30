# AI_GUIDE — how another AI (or another contributor) adds support for model X

You are the next contributor. This file is written for you and it is meant to be
read in one sitting. It has four parts: what to read, what to fill in, what you are
forbidden to generate, and the gates that decide whether you are done.

Read time: about ten minutes. Work: about one file of real thinking plus four small
tables. The parts that used to be 1,194 lines per model are now shared and you do
not write them.

---

## 0. The minimum reading set

| # | file | why |
|---|---|---|
| 1 | `tools/archkit/kit/CONTRACT.md` | the interface in one page: what you write, what the driver owns |
| 2 | `tools/archkit/kit/AI_GUIDE.md` | this file: the procedure and the gates |
| 3 | `tools/convert/minicpm5_1b/` (5 files, 339 lines total) | a complete worked example, end to end, that produced a real artifact |
| 4 | `tools/archkit/scaffold/minicpm5_1b/adapt_report.md` | what the generator measured about the family side for that model, generated |

That is the whole reading set. You do **not** need to read
`tools/artifact/container.py` (the writer), `tools/archkit/kit/driver.py` (the
driver), or any existing converter: all three are behind the interface in item 1,
and the one converter you should compare yourself against is item 3, which exists
precisely so you do not have to read a large one.

## 1. The input contract

You need a checkpoint directory containing, at minimum:

| file | used for |
|---|---|
| `config.json` | geometry, and the `architectures` / `model_type` names |
| `model.safetensors.index.json` | the authoritative key set, and the R29 denominator |
| the shard `.safetensors` files it names | headers (dtype/shape) and payload byte ranges |

Both named files are required. Missing either one is refused **by name** before
anything else happens, so the first thing you will see if you point the driver at
the wrong directory is `REFUSED: missing config.json under <dir>`.

Weights must be readable as byte ranges. A BF16 checkpoint is a pass-through: the
artifact stores what the shard stores, so the conversion is CPU-only, needs no GPU,
and needs no torch for the matrix path.

## 2. The fill-in list, in order

Each row gives the file, what "correct" looks like, and the gate that checks it.

**Step 1 — extract the spec.**

```
python3 -m tools.archkit.arch_spec extract <ckpt>/config.json <model-id> \
        tools/archkit/specs/<id>_spec.json
```

Correct = `geometry` is non-empty and every number is a copy of the checkpoint's
own `config.json`, not a default. Also correct = the file names the `family` you
intend to instantiate, and `layer_types` covers every layer with a kind from
`{full|full_attention, sliding|sliding_attention, gdn|linear_attention}`.
Gate: `spec-vs-config`, `layer-kinds`.

**Step 2 — write `inventory.py`.** The ordered vocabulary of artifact object names
and, for each, the source key(s) it is built from and its role.

Correct = every source key of the checkpoint appears exactly once across the whole
table (or is matched by an explicit `IGNORED_SOURCE_PREFIXES` entry), and the object
names follow the family vocabulary already in the tree (`text/token_embedding`,
`text/layers/N/attention/query`, `text/layers/N/mlp/gate`, `text/final_norm`,
`text/output_head`; see `src/targets/spark_x2_5_4b/impl/load/bindings.cpp:98-198`).
A second spelling of the same concept is a defect in waiting.
Gate: `source-coverage`. Reading: `objects=N index_keys=M consumed=C ignored=I
C+I=M disjoint=yes`.

**Step 3 — write `recipe.py`.** One function per role, taking `(obj, source, ctx)`
and returning bytes or a chunk iterable; plus the tie rule.

Correct = a pass-through role yields `source.raw_range(key)`; a role that computes
yields bytes from `tools/artifact/layouts.py`'s encoders; the tie rule returns the
name of the object this one aliases, or `""`. Nothing here writes a file.
Gate: `geometry` (each object's declared shape against the shard header).

**Step 4 — write `declaration.py`.** `build_declaration(source) -> Declaration`.
Inventory + recipe + spec, and the `WORK_ITEMS` list.

Correct = no file is opened and no byte is read; the `WORK_ITEMS` entries name the
engine-side gaps **by name** rather than describing them. A `WORK_ITEM` that says
"needs family work" is not an entry; "`qk_norm_enabled` has 0 references in the
shared runtime, so this stack's missing q/k norms are not gated by anything" is.

**Step 5 — write `convert.py`.** `driver.main(build_declaration)` plus argparse.
Correct = under 60 lines and zero `open(`.

**Step 6 — generate the scaffold and read the gap report.**

```
python3 -m tools.archkit.kit.scaffold \
  --spec tools/archkit/specs/<id>_spec.json --id <id> \
  --out tools/archkit/scaffold/<id>
```

**Step 7 — run the gates.** See section 4.

## 3. What you are FORBIDDEN to generate or auto-fill

The generator refuses these and so should you. Each line is a measured failure, not
a style preference.

1. **`impl/config.h`'s `TextConfig` body.** A generator was written for this and
   produced a file whose own review listed four reasons it was unusable; the
   sharpest is that `full_attention_layers()` returned 9 and `gdn_layers()` returned
   0 on a 36-layer stack. Nine plus zero is nine: **27 layers would never execute
   and nothing would fail.** The real file is 548 hand lines plus about twenty
   `static_assert`s. The generator states the invariant
   (`impl/spec_contract.h` asserts `schedule_total == layers`) and never the answer.
2. **The per-layer tables** (`layer_kind`, `layer_rope_theta`, `layer_rotary_dim`,
   `kFullLayers`). Their contents are the model's semantics; a wrong table is silent.
3. **The family hook accessors.** Which hooks the shared runtime actually reads is a
   measurement, not a guess: run `python3 -m tools.archkit.kit.refcount --target
   src/targets/<family>`. Everything it reports as ZERO-REFERENCE is a declaration
   the runtime never consumes, so declaring it changes nothing and relying on it
   changes nothing either. Making a dead hook live is a family-side land with its
   own review.
4. **`rope_theta` when the spec carries per-kind values.** Declining is correct;
   emitting a default is the defect. The generator emitted `0.0F` for spark and muse
   for exactly one revision, and the reverse diff caught it by name
   (`MECHANICAL MISMATCH rope_theta: real '10000.0F', generated '0.0F'`).
5. **`src/targets/registry.h`, `src/targets/registry.cpp`, `src/CMakeLists.txt`.**
   These are shared files that other lines edit. The generator prints exact
   `file:line` insertion points into `registry_insert.tsv` instead of editing them,
   and the insertion is a person's land with a pre-image taken at that moment.
6. **`tests/` and the front door's `REGISTERED_TARGETS`.** Same reason.

## 4. The acceptance gates

Run these in order. Each one prints a reading with a denominator; none of them is
optional, and a green `rc` is not the evidence -- the READING is.

```
# 1. plan only: config, index, spec, the whole gate battery, nothing written
python3 -m tools.convert.<id>.convert --model <ckpt> --out /tmp/x.ninfer --plan

# 2. the real write
python3 -m tools.convert.<id>.convert --model <ckpt> --out <file>.ninfer

# 3. read it back and compare it to the checkpoint's own index, per object
python3 -m tools.archkit.kit.verify --artifact <file>.ninfer --model <ckpt> \
  --declaration tools.convert.<id>.declaration:build_declaration

# 4. the negative battery: prove the gates can REFUSE
python3 -m tools.archkit.kit.test_kit

# 5. the compile gate on the generated contract (R41 + R52 + RED arms)
bash tools/archkit/kit/compilegate.sh tools/archkit/scaffold/<id>

# 6. the ruler: does the scaffold agree with its spec, and how many places does it touch
python3 -m tools.archkit.kit.check --check tools/archkit/scaffold/<id> \
  --spec tools/archkit/specs/<id>_spec.json
```

What each gate must show:

| gate | the reading that means done |
|---|---|
| plan | all five gates green, `consumed+ignored = index_keys`, `disjoint=yes` |
| write | `magic b'NINFER\x00\x02'`, a sha256, `objects=N` |
| verify | `objects_declared = objects_in_file = index_keys`, `order_equal yes`, `geometry_equal N/N`, `problems 0` |
| negatives | 6/6 cases refuse by name and produce no artifact |
| compile | `RC_FAILED_ARMS=0`, and the POS arm prints the absolute path of the file it compiled |
| ruler | 0 failures; tier **a** at most |

## 5. What you may claim when you are done

Tier **b** — "there is a converter and it produced an artifact" — is claimable only
with a `kit/verify.py` reading of the produced file: path, bytes, sha256, magic,
`ArtifactIdentity`, and a per-object geometry equality against the shard headers.

Tier **c** — "the engine loaded it and it emitted tokens" — is claimable only with a
run log. Neither the scaffold nor the generator can claim b or c for you; a
generated file that claims a load is claiming somebody else's evidence.

## 6. A worked example, end to end

`tools/convert/minicpm5_1b/` is that example, and its produced artifact is
`/home/user/models/minicpm5_1b_bf16.ninfer`:

```
bytes      2161302528
sha256     3dd1cac0093192228b3a2042b9b41295d2af8a7f4886874648ea8d6652ab4350
magic      b'NINFER\x00\x02'
identity   model_id=minicpm5-1b weights_id=bf16
R29        objects_declared 219  objects_in_file 219  order_equal yes
           index_keys 219  matched 219  set_equal yes
           geometry_equal 219/219  problems 0
```

Reproduce it with:

```
python3 -m tools.archkit.kit.demo                # a synthetic checkpoint, no download
python3 -m tools.convert.minicpm5_1b.convert --model <ckpt> --out <out>.ninfer --plan
```
