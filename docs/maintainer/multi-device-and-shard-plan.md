# Multi-device sharding, the arch route table, and the capability gate

Three host surfaces landed together that answer questions **about a card and about a world of cards**.
They are grouped in one document because they share the same property and the same status:

- `src/core/shard_plan.h` — the multi-device shard arithmetic, the per-rank persistence naming, and the
  one predicate the KV policy needs before any of it is safe.
- `src/core/kernel_route.h` — the central per-architecture GEMM route selector.
- `src/core/arch_caps.h` — the capability ladder and the artifact-format floor table, plus the gate that
  consumes them (`require_artifact_formats_supported()`), and the two build/run switches around them.

Anchors are given as file and symbol rather than line numbers, because line numbers in this tree drift
within a session. Where a line is given it is marked with the revision it was read at
(`@3944a53` = `git rev-parse --short HEAD`).

## 0. Status — landed, host-only, and **not yet on any call path**

State this first, because everything below reads as a capability claim and none of it is one.

| Surface | In the tree | Reached by the engine? |
|---|---|---|
| `src/core/shard_plan.h` (`@3944a53`, 564 lines) | yes | **no ranked-world call path**. `kernel_route` / `shard_plan` have **0** occurrences in `src/CMakeLists.txt`, in the top-level `CMakeLists.txt`, and in `src/targets/registry.cpp`, `src/runtime/engine/engine.cpp`, `src/ops/linear/linear.cpp`, `apps/cli/main.cpp` — a bounded probe of those seven named files, not a tree-wide search. The header is included by its own host test only. |
| `src/core/kernel_route.h` (`@3944a53`, 416 lines) | yes | **not wired to dispatch.** `select_route()` is a pure function; nothing in `src/ops/linear/` or in `registry.cpp` calls it. The tombstone it replaces **is** still in HEAD (§3.3). |
| `src/core/arch_caps.h` | yes, and **this one is wired**: `src/targets/registry.cpp` reaches the gate (`arch_caps` = 2 occurrences there) | yes — `require_artifact_formats_supported()` is an artifact-load-time gate |
| the three host tests | yes | **the `shard_plan` / `kernel_route` / multi-device tests are NOT registered**: `kernel_route` = 0, `shard_plan` = 0, `multidev` = 0 in `tests/CMakeLists.txt@a22d29074e08ca7e` (554 lines). The three source files exist and total 1221 lines (§7). Registration is a build-side change and is tracked separately; this document records the state, not the fix. |

So: read this as **a contract that exists and is exercisable on one GPU**, not as "the engine can run on
two cards". §2.6 names exactly what is missing for that.

## 1. `src/core/shard_plan.h` — the shard arithmetic

Host-only by construction: no CUDA header, no device query, no collective. Every rule below is a pure
function, which is why the whole thing is testable on a single-GPU host.

The header states its own premise in the top comment, and the premise is worth repeating because it is
the reason the file exists at all: the engine today has exactly one device —
`cudaGetDeviceCount` / `cudaSetDevice` occur only in `src/core/device.cu`, whose only job is to pick one
card and bind the thread to it; there is no rank, no world size, no collective, no peer access, no NCCL,
and no shard of anything. (`shard` hits only `src/ops/split_attention.cu`, which is a **position** split
of a single attention's token range inside one process — unrelated.)

### 1.1 The world shape

```
enum class ParallelAxis : std::uint8_t { None, Tensor, Pipeline };
struct WorldShape { std::uint32_t world_size = 1; std::uint32_t rank = 0; ParallelAxis axis = None; };
```

Deliberately **one** axis, not a mesh: a 2-D (tp x pp) world can be added by composing two `WorldShape`s,
and declaring it now would put an unexercised degree of freedom in the type.

`world_shape_refusal()` is a free function rather than an `assert` so the engine can **refuse** a
malformed world at construction and name the field. Two of its four refusals are the load-bearing ones:

- `axis=none` with `world_size != 1` is refused, because **the granularity contract depends on which axis
  it is** and a multi-device world that does not name its axis cannot be checked;
- `axis=tp|pp` with `world_size == 1` is refused, because a single-device world is `axis=none`.

### 1.2 Geometry in, shard ranges out

`plan_shards(WorldShape, ModelGeometry)` returns a `ShardPlan` that is either `ok` or carries a
non-empty `reason`. The two axes split different things and are refused for different reasons:

**Tensor (`tp`) splits KV heads and weight `N`; text layers stay whole on every rank.**

- `kv_heads % world_size != 0` is refused: the split would cut a GQA group across ranks, so a rank's query
  heads would attend to kv heads it does not own. No cross-rank reduction repairs that — the missing
  head's K/V bytes are simply not on that card.
- `weight_columns % world_size != 0` is refused: a partial column block would put one output element's
  partial sum on two ranks with no way to distinguish it from a complete one. (For a GQA model the
  attention QKV projection's `N` is `(q_heads + 2 * kv_heads) * head_dim`, which is **not**
  `kv_heads * head_dim`; the caller supplies the real number so the header does not guess a layout.)
- `q_heads % kv_heads != 0` is refused before either, because then the GQA group size is not an integer
  and no head split is defined at all.

**Pipeline (`pp`) splits text layers; KV heads and weight `N` stay whole on every stage.**

- the split is **allowed to be uneven** — unlike the head split, an uneven pipeline costs only balance,
  because each stage's layers are complete within that stage and no single layer is ever cut. The first
  `remainder` stages take one extra layer.
- `text_layers / world_size == 0` is refused: a stage would own no layer at all.

The plan carries two facts the KV policy has to consult:

| Predicate | `None` | `tp` | `pp` |
|---|---|---|---|
| `page_is_layer_complete()` (= `axis != Pipeline`) | true | true | false |
| `residency_decision_is_replicated()` (= `axis != Tensor`) | true | false | true |

### 1.3 Per-rank persistence naming

With one device the cold artifacts are named for the **artifact** alone (`<artifact>.kvrowscale.bin` and
its four siblings — see [`kv-rowscale-persistence.md`](kv-rowscale-persistence.md)). Under a split that
changes the KV's own shape the table is no longer the artifact's table, it is **the artifact's table for
one rank**: two ranks writing the same name would silently overwrite each other with tables of a
different geometry, and a reader started with a different world would then apply a table whose row count
does not describe its own KV.

So the name carries the shard, inserted **immediately before the suffix**:

```
base="m.kvrowscale.bin", {tp, rank 1, world 4}  ->  "m.tp1of4.kvrowscale.bin"
```

`sharded_artifact_name()` returns `base` **unchanged** for a single-device world, so every existing
artifact name and every file on disk today keeps working. `parse_sharded_artifact_name()` recognises the
tag, and `sharded_name_acceptance()` is the reader's rule: an **unsharded** name is a refusal for a
multi-device run, because a table written by a single-device run does not describe one rank's KV; a tag
for another axis, rank or world is a refusal too.

### 1.4 The page residency bit across a partition — the load-bearing part

The KV policy's eviction unit is **one page**, and "one page is one residency bit"
(`src/targets/qwen3_6/impl/runtime/cold_host_tier.h`). Splitting the KV across cards splits the **state**
that carries that bit while leaving its **decision inputs** shared, and that asymmetry is the whole
question:

- decision inputs, all shared by construction: `frontier`, `layer_windows`, the cold byte budgets;
- decision state, all **per rank** once the world is split: `ColdHostTier::usage_` / `ColdTierUsage` (the
  counter `admit_cold_page` reads), the `HostKVArena` (its capacity is a per-rank byte cap), and
  `allocate_cold_disk_file_slot()` (whose free list is process-global, i.e. per rank).

Two ranks can therefore run the **same** predicate to the **same** answer and still reach **different**
admits, purely because one rank's arena is full. That is not a defect in the predicate and no amount of
making the predicate smarter fixes it. Hence three named states:

| `ResidencyState` | Name | Meaning |
|---|---|---|
| `FullyAbsent` | `fully-absent` | every rank refuses the page |
| `FullyResident` | `fully-resident` | every rank admits it |
| `PartiallyResident` | `partially-resident` | **ranks disagree — a state the model does not define** |

`global_admit()` is the replicated decision: the page leaves the device only if **every** rank can take
it (an AND-reduce over ranks). Under the tensor axis that is a **new collective on the eviction driver's
critical path** — the price of keeping the page atomic.

**The factoring fact, stated as the property it is.** `cold_host_page_is_read_free(...)` is a pure
**conjunction over layers**. A conjunction factors across a partition of its conjuncts, so a **layer**
partition (the pipeline axis) can evaluate the predicate per stage, AND the answers, and get exactly the
global answer — the replication is free and exact. It does **not** factor across a partition of the KV's
**content** (the tensor axis): heads are not conjuncts of this predicate at all, and "this rank can drop
its heads for page `p`" says nothing about whether page `p` is still a legal attention input. The
counterexample is in `tests/test_shard_plan.cpp` (a closed-form attention over two heads with exact
powers of two, so its expected values are literals rather than the code under test's output).

Three consequences are exposed as separate predicates rather than folded into one:

| Predicate | `tp` | `pp` |
|---|---|---|
| `read_free_predicate_factors_over(axis)` | false | **true** |
| `residency_decision_needs_replication(axis)` | **true** | false |
| `page_content_is_layer_complete(axis)` | true | false |
| `recall_record_shape_survives(axis)` | true | false |

`recall_record_shape_survives()` is the recall-journal consequence: the journal stores **one**
`layer_bytes` per page and refuses `RecallCodec::Mixed` at compile time. A head split multiplies every
layer's stride by the same `1/world_size` — still one stride, still one codec per layer — so the record
shape survives untouched. A **layer** split does not: stages may legitimately choose **different**
per-layer codecs (that is exactly what `--kv-layer-storage 0-7:rk4v4,4-7:int8` does), which makes
`RecallCodec::Mixed` reachable and the existing `static_assert` a wall the feature cannot get past.

### 1.5 The cold spill cell path

The disk tier's cell name is built today as (in `program_impl.h`, the Disk / HostThenDisk arm)

```
dir + "/ninfer_cold_L" + std::to_string(layer) + ".slot"
```

— a **layer** axis with **no rank axis**, and `dir` defaults to `/tmp`. Two ranks of the same world
therefore open the same file, size the same `file_slot` space from it, and both start at
`(L0.slot, offset 0)`: the per-process slot allocator hands out the same first slot in every process, so
the writes do not merely interleave, they **collide deterministically**. Spilling is not best-effort: a
collision silently swaps two different pages' bytes. `cold_spill_cell_path(dir, layer, world)` is the
rank-aware spelling, and for a single-device world it returns the string above **byte for byte**.

### 1.6 What is missing — named, not stubbed

The header's own closing section lists what is **required** to make any of the above run, and states that
none of it exists:

- a rank/world plumbing layer: process launch, rank discovery, and the **deterministic iteration order**
  the replicated admit decision needs (all ranks must walk the same page order or "replicated" is
  meaningless);
- the collectives themselves — for tensor parallelism an all-reduce over the attention combine triple
  (row max, row sumexp, partial output) per layer, and an all-reduce AND of the eviction admit verdict;
- `cudaDeviceEnablePeerAccess` / `cudaMemcpyPeer` (or an NCCL equivalent) plus a topology decision;
- a per-rank weight loader: the artifact reader produces **one** byte stream and there is no N-dim column
  slicer anywhere in `src/artifact/`;
- a rank/world field in the recall journal record. `RecallRecord` is exactly 64 bytes with two free words
  (`pad0`, `pad1`) and a `static_assert` pinning the size, so the field can be added without moving the
  stride — but it has not been (§6);
- a device count > 1 exercised anywhere. `tools/archkit/_GPU_MATRIX.md`'s "我们的落地路线" item 1
  ("单进程 TP 骨架") is a plan, and the bounded probe in the header comment says none of it is code.

## 2. `src/spec/turn_recall_journal.h` — the shard axis

The recall record is 64 bytes. Its budget is now stated explicitly, because a shard axis has to occupy
words that were free:

| Constant | Value | What it is |
|---|---|---|
| `kRecallRegionDescriptorBytes` | 8 | `layer_bytes` + `file_slot`, the size of one region descriptor |
| `kRecallRecordFreeBytes` | 16 | the four unassigned words (`reserved`, `reserved2`, `pad0`, `pad1`) |
| `kRecallShardAxisBytes` | 8 | `shard_rank` + `shard_world` |
| `recall_extra_regions_after_shard_axis()` | `(16 - 8) / 8` = **1** | extra regions the record can still describe once the shard axis is present |

The two `static_assert`s around it are a two-sided pin, which is the point: `>= 1` says the shard axis
must leave room for at least one more region descriptor, and `< 2` says there is **no whole byte left**
after the shard axis — if the second ever passes, the ledger above it has drifted and the record has
grown.

The record's own comment states the reader's rule in the same shape `shard_plan.h` uses: a reader must
**refuse** a record whose `(shard_rank, shard_world)` is not its own, rather than mis-read a record
written by a different world.

## 3. `src/core/kernel_route.h` — the per-arch GEMM route selector

### 3.1 What it is

One place where "which GEMM route does this `(card, weight format, problem shape)` get?" is answered,
as `select_route(sm, format, shape)`. It is **table-driven**, and its two tables are not new: it consumes
`arch_caps.h`'s `kArchLadder` (the per-card capability set) and `kFormatRequirements` (the per-format
hardware floor, each row carrying the file:line of the mma intrinsic the engine actually emits). A route
decision is therefore derived from the same kernel evidence the artifact-format gate uses, instead of
restating it in a second table that can go stale.

The routes are named for the **mechanism**, not for a card, so every row can be justified by a kernel
file:

| `KernelRoute` | `<name>` | Note |
|---|---|---|
| `None` | `none` | no route: nothing in this tree can serve the request |
| `ConservativeSimt` | `conservative-simt` | lowest floor, no tensor-core requirement at all (the fallback) |
| `QpnW4a16` | `qpn-w4a16` | 4-bit codes expanded inline into fp16 mma; the Volta route |
| `MmaBf16` | `mma-bf16` | serves BF16 **and** all groupwise-int formats |
| `Fp8A16Bf16` | `fp8-a16-bf16` | fp8 weights dequantized to bf16, then `mma_bf16` |
| `Fp8A8KindF8f6f4` | `fp8-a8-kind-f8f6f4` | `kind::f8f6f4` A8 fp8 (a Blackwell form) |
| `Nvfp4W4a4Tma` | `nvfp4-w4a4-tma` | `kind::mxf4nvf4.block_scale` W4A4 (sm_100a / sm_120a) |

### 3.2 The three outcomes

| `RouteOutcome` | `<name>` | What the caller does |
|---|---|---|
| `Selected` | `selected` | log the line at info; nothing to do |
| `UnknownArchFallback` | `unknown-arch-fallback` | log the line at **WARN**; the run may proceed on the fallback |
| `NoKernelInTree` | `no-kernel-in-tree` | **refuse that format** and quote `why`, which names the kernel that has to be written |

Three values and not a bool, because the three call for three different reactions. `RouteChoice::ok()` is
`outcome != NoKernelInTree` and `warns()` is `outcome == UnknownArchFallback`, so a caller does not
re-derive either from the text.

### 3.3 Non-gating by construction, and the tombstone

An unknown compute capability **does not throw and does not refuse**: it selects the **lowest-floor**
route and reports `UnknownArchFallback`. What still comes back as `NoKernelInTree` is a route whose
kernel is genuinely absent from this tree — and then `why` names the **missing kernel**, which is
actionable, instead of naming the card.

It is **not** a gate and it does **not** decide whether an artifact may load. That is
`caps::require_artifact_formats_supported()` at artifact-load time (`src/targets/registry.cpp`,
`construct_target`). The two answer different questions over the same facts — "may this run at all" vs
"which route does this run take" — and a route can be `Selected` on a card whose artifact gate refuses a
different format.

**The tombstone, named because the two mechanisms must not be confused.** The statement this table
replaced was, verbatim,

```cpp
if (device.sm() != 120) { throw std::invalid_argument(
    "this ninfer build targets compute capability 12.0 only; your GPU is sm_" ...); }
```

It is **still present in HEAD `3944a53`** at `src/targets/qwen3_6/impl/runtime/layouts_impl.h` (the throw
whose message begins "this ninfer build targets compute capability 12.0 only") and is **deleted in the
working tree**, where the capability-probe gate (`src/core/device_capabilities.h` +
`src/core/device_probe.cu`) replaced it. Both of those are gates; `kernel_route.h` is the route **table**
that was missing from both. The reason the `sm() != 120` form was wrong is not that the number was wrong:
it is that **a number is not evidence for an instruction**. `device.sm()` is `major * 10 + minor`
(`src/core/device.cu`), so it cannot even see the `-a` suffix that decides whether the fp4/TMA kernels
exist in the binary at all (`tools/archkit/_GPU_MATRIX.md`: "这张表的键是 (sm 号 + 特性后缀)").

The selector is a pure function that takes `sm` in, so every rung of the ladder — including the cards
this host does not have — is exercisable on a single-GPU machine. That is the point of the parameterized
form: "what route does a V100 take for an nvfp4 artifact" is a **table** question, not a hardware one.

## 4. `src/core/arch_caps.h` — the gate half

### 4.1 The ladder and the floor table

`kArchLadder` is the per-card capability set and `kFormatRequirements` is the per-format hardware floor;
each floor row carries the `file:line` of the mma intrinsic the engine emits, so the floor is
kernel-evidence-backed rather than prose. `evaluate_artifact_formats(sm, formats)` returns a
`CapabilityReport` whose verdict is one of four, and only two of them are refusals:

| `Verdict` | Is it a refusal? |
|---|---|
| `Supported` | no — every format's floor is met |
| `Unsupported` | **yes** — at least one format's floor is not met on this compute capability |
| `UnknownArch` | **no** — the compute capability is not in the ladder; a warning, not a refusal |
| `UnknownFormat` | **yes** — an artifact format has no row here (a table defect, not a GPU fact) |

`require_artifact_formats_supported()` is the entry point the artifact loader calls. It is **silent when
it can**, and it stays silent on `UnknownArch` on purpose: the one refusal path that remains is the
evidence-backed one. `render_capability_report()` is the operator-facing text (the GPU, the binary's arch
list, the missing capability with the kernel that needs it, and the routes that would work on this card).

`unknown_arch_warning()` is the text for the one verdict that is a warning. It says what the verdict is a
fact about, and it names the trap directly: sm_120 and sm_120a share the number 120 and do **not** share
the fp4/TMA kernels, so a reader must not assume a neighbouring row's capability set.

### 4.2 `NINFER_ARCH_WARN` — the one runtime switch here

By default an unknown compute capability produces **no output at all**: the verdict is a fact about the
table rather than about the card, the run proceeds on the conservative route, and the route selector
refuses only the shapes it has no kernel for. Set `NINFER_ARCH_WARN` (any value; the check is
`std::getenv("NINFER_ARCH_WARN") != nullptr`) and the same warning is written to **stderr**:

```
if (std::getenv("NINFER_ARCH_WARN") != nullptr) {
    std::fprintf(stderr, "%s\n", warning.c_str());
}
```

This is the only `getenv` in `src/core/arch_caps.h`. It is a real environment read, not a reserved name
and not a compile-time macro: it is read at the moment an artifact's formats are checked. `arch_caps.h`
does not honour the "unset flag leaves the operator's environment alone" convention of the KV options,
because it has no flag counterpart at all.

## 5. `NINFER_BUILD_CUDA_ARCHES` — a compile-time macro, **not** an environment variable

A fourth entry in the family of "names that look like switches and are not" (the others are
`NINFER_QWEN36_VARIANT`, a per-target instantiation macro, and the CMake cache variables).

`NINFER_BUILD_CUDA_ARCHES` is a **preprocessor macro** the build exports from
`CMAKE_CUDA_ARCHITECTURES`:

```cmake
# src/CMakeLists.txt -- note the spelling of the macro name
string(REPLACE ";" "," NINFER_CUDA_ARCHS_TEXT "${CMAKE_CUDA_ARCHITECTURES}")
set_source_files_properties(core/device_probe.cu PROPERTIES
  COMPILE_DEFINITIONS "NINFER_BUILD_CUDA_ARCHS=\"${NINFER_CUDA_ARCHS_TEXT}\"")
```

```cpp
// src/core/arch_caps.h
inline std::string_view build_arch_list() noexcept {
#ifdef NINFER_BUILD_CUDA_ARCHES
    return std::string_view(NINFER_BUILD_CUDA_ARCHES);
#else
    return std::string_view{};
#endif
}
```

It is **not** set by `setenv`, so it cannot be changed at run time, and `#ifdef` is the only way to see
it. `CMAKE_CUDA_ARCHITECTURES` defaults to `120a` (`CMakeLists.txt`, `set(... CACHE STRING "CUDA
architectures to build")`). Setting the environment variable of the same name does nothing.

### 5.1 ⚠️ The two spellings do not meet, so `build_arch_list()` is dead in this build

The CMake statement above defines **`NINFER_BUILD_CUDA_ARCHS`** — no `E` — and scopes it to a **single
translation unit** (`core/device_probe.cu`). `arch_caps.h` reads **`NINFER_BUILD_CUDA_ARCHES`** — with
`E`. Under a bounded probe of the eight files that own the name (`CMakeLists.txt`, `src/CMakeLists.txt`,
`src/core/arch_caps.h`, `src/core/device_capabilities.h`, `src/core/device_probe.cu`,
`src/targets/registry.cpp`, `tests/test_arch_caps.cpp`, `tests/test_device_capabilities.cpp`):

| Spelling | Where it occurs |
|---|---|
| `NINFER_BUILD_CUDA_ARCHS` (no `E`) | `src/CMakeLists.txt:44` (the one definition), `src/core/device_capabilities.h` (3 occurrences, including `#ifndef` / `#define` and `facts.build_architectures = NINFER_BUILD_CUDA_ARCHS;`) |
| `NINFER_BUILD_CUDA_ARCHES` (with `E`) | `src/core/arch_caps.h`, 4 occurrences (the comment at 230, `#ifdef` at 302, the `return` at 303, the user-visible sentence at 400); `tests/test_arch_caps.cpp:227`, as a **string literal** inside a `mentions(...)` check |

Consequence, stated flatly: `#ifdef NINFER_BUILD_CUDA_ARCHES` could **never be true** in a build
this tree produces, so `build_arch_list()` returned the `#else` branch — the empty view — in every configuration, and
the operator-visible sentence that tells a reader to "rebuild with `CMAKE_CUDA_ARCHITECTURES` exported as
`NINFER_BUILD_CUDA_ARCHES`" names a macro **this build cannot define**. The string it promises can never
appear.

What this does **not** say, and the distinction matters: the arch list is not unreported to the engine as
a whole. `src/core/device_capabilities.h` uses the no-`E` spelling and *is* part of the translation unit
the CMake definition reaches, so the capability probe carries the real list. The defect is confined to
the second, doc-facing spelling in `arch_caps.h` — one of two spellings for one concept.

**The test used not to catch it. It now does, and this paragraph is what changed.**
The earlier wording here was "**The test does not catch it, on purpose.**" -- a deliberate,
documented blindness, justified by the rule *document, do not rename, because that is a build
change*. That decision has been surfaced to the owner as a conflict rather than reversed quietly:
this fleet's contract is that a gate must be able to fail and must be able to show that it ran, and
the two positions cannot both hold. **What the code does now:** the assertion is two-branch, and in
the configuration where the macro is unread it goes red. **What it cost:** the CMake change the
earlier wording declined to make -- an `INTERFACE` target, `ninfer_build_cuda_arches`. `tests/test_arch_caps.cpp` states the check as

```cpp
check(!ninfer::caps::build_arch_list().empty() ||
          mentions(nvfp4_on_ada, "NINFER_BUILD_CUDA_ARCHES"),
      "an unreported build arch list is named as unreported");
```

— a disjunction, so the empty branch passed and the check was satisfied by the message naming the
unreported list. That shape was defensible for a check whose subject is "the report says the truth
about what it knows" -- and it was **not** a check on the CMake macro's name. The replacement is the
check that was missing: it branches on whether this TU was given the macro, and demands the matching
behaviour in each case. **"Adding one would need the build to assert that the macro it defines is the
macro the header reads" is now done, not impossible** -- see the `INTERFACE` target above and
`dl/routeprobe/proof_arches.txt` for the mutation that proves the assertion can fail.

Why the string exists at all: a compile-time `-a` suffix (for example `sm_120a`) is **not observable
through `cudaDeviceProp`**, which reports only `12.0`. So this string is the **only** way the runtime can
name the arch-accelerated target of the binary it is running in — which is why it appears inside the
user-visible capability text:

```
"<unreported> (rebuild with CMAKE_CUDA_ARCHITECTURES exported as "
"NINFER_BUILD_CUDA_ARCHES to have the arch list named here)\n"
```

## 6. The other consumers of the same facts

Three further host surfaces landed alongside and are worth naming, because a reader of the capability
text will meet them:

| File | What it owns | Status |
|---|---|---|
| `src/core/device_capabilities.h` (`@3944a53`, 440 lines) | the **host half** of the capability probe: pure C++, no CUDA header, replacing "decide by core number" | landed; `src/CMakeLists.txt` names `device_probe` |
| `src/core/device_probe.cu` (`@3944a53`, 424 lines) | the **device half**: it really launches a tiny probe per capability, classifies launch / sync / copy errors (no image, launch failure, numerics disagree), and caches once per device | landed |
| `src/core/device_sm_count.h` (`@3944a53`, 47 lines) | the **SM count of the device owning the current context** — explicitly a **tuning input, not a capability test**: a grid size is a performance decision and nothing here may refuse a device or a request | landed |

`device_capabilities.h` names the statement it replaced, and it is the same one `kernel_route.h` names:
`src/targets/qwen3_6/impl/runtime/layouts_impl.h` `if (device.sm() != 120) throw`. The two defects it
records are (1) `device.sm()` has no `-a` half, so `sm_120` and `sm_120a` are indistinguishable at run
time, and (2) a number is not evidence for an instruction.

`tests/test_arch_caps.cpp` (`@3944a53`, 302 lines) and `tests/test_device_capabilities.cpp`
(`@3944a53`, 219 lines) are registered; `ninfer_device_capabilities_test` carries `NEEDS_SOURCE_DIR`
because it reads the tree. See [`op-development.md`](op-development.md) §6.4.

## 7. Tests — two registered, three not

| ctest name | File | Lines | Registered? |
|---|---|---|---|
| `ninfer_arch_caps_test` | `tests/test_arch_caps.cpp` | 302 | **yes** (`tests/CMakeLists.txt`, 79) |
| `ninfer_device_capabilities_test` | `tests/test_device_capabilities.cpp` | 219 | **yes** (83) |
| — | `tests/test_kernel_route.cpp` | 340 | **no** — `kernel_route` = 0 occurrences in `tests/CMakeLists.txt@a22d29074e08ca7e` |
| — | `tests/test_shard_plan.cpp` | 534 | **no** — `shard_plan` = 0 |
| — | `tests/test_multidev_wiring.cpp` | 347 | **no** — `multidev` = 0 |

The three unregistered files are real host tests, not drafts: `test_kernel_route.cpp` groups its checks so
each group fails for one reason (soundness invariant / non-gate principle / …), `test_shard_plan.cpp`
carries the head-split counterexample `shard_plan.h` §1.4 refers to, and `test_multidev_wiring.cpp`
opens by recording that the first version of this analysis was **wrong in its framing** — it argued that
"splitting KV by attention head breaks the page's atomicity", when the page's eviction *predicate* does
not break and the on-disk arithmetic breaks for a much more concrete reason.

Each of the three is a host-only `int main` program with no CUDA include and no device dependency, so
registering them is a `tests/CMakeLists.txt` change and nothing else. **Until that change lands, the
checks above are not run by ctest**, and this document does not claim they pass.

## 8. Adding a device axis — checklist

1. Decide the **axis** first, then the world size. `axis=none` with `world_size != 1` and the reverse are
   both refused; a world that does not name its axis cannot be checked against the KV granularity
   contract.
2. Check the divisor rules before anything else: `q_heads % kv_heads`, then `kv_heads % world_size` and
   `weight_columns % world_size` for `tp`, or `text_layers / world_size > 0` for `pp`.
3. Decide whether the page's *content* survives the axis (`page_content_is_layer_complete`) and whether
   the *decision* needs a new collective (`residency_decision_needs_replication`). Those two answers are
   independent, and conflating them is the framing error §7 records.
4. If the axis changes the KV's own shape, give every persisted artifact a shard tag
   (`sharded_artifact_name`) and make the reader **refuse** a name that is not its own.
5. If the axis changes the disk cell naming, use `cold_spill_cell_path` rather than spelling the path at
   the call site; a collision there silently swaps pages' bytes.
6. Add the record field to the recall journal only through the ledger in §2, and update both
   `static_assert`s. A field that does not fit is a stride change, not a local edit.
7. Update the route table only if the axis changes which kernel serves a shape. The route table answers
   **one card at a time**; placement is `shard_plan.h`'s question.
