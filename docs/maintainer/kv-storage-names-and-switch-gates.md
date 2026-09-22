# `KvCacheStorage` name tables, `dtype_of` refusals, and the switch-coverage gate

Three mechanisms that share one subject: **an enumeration that grew, and the hand-written tables over
it that did not.** This file is the maintainer reference for the name table, for the `DType`
classification refusals, and for the CMake gate that turns the next omission into a build failure.

Revision note: file-and-symbol anchors are given rather than bare line numbers because line numbers
drift within a session. Numbers, where given, are marked `@3944a53`.

## 1. The structural problem

This tree carried **nine hand-written `KvCacheStorage -> text` tables**, and every one of them was at
least one enumerator short. The reason is structural, not careless: **all nine are `switch`
statements**, and in this build a short `switch` is not even a warning — `CMakeLists.txt` opened no
warning flag at all. Even with `-Wall -Wswitch` the message is a warning, never an error.

So a comment that claims "a new enumerator breaks this build" was **false**, and two user-visible
consequences followed from it:

- `apps/perplexity/main.cpp`'s `kv_name()` named 4 of 8. `--kv-dtype nvfp4` parsed, loaded the
  artifact, scored the corpus and then died at the first `kv_name` call
  (`prepare_output_directory`). A name table shorter than its enum is a **runtime failure with a
  compile-time warning next to it**, and the warning is the only thing that was ever going to catch
  it — so it must be silent.
- `src/serve/request_log.cpp`'s `kv_cache_name()` named 3 of 8. That string goes into the
  **request-log line** (`{"kv_cache", kv_cache_name(engine_options.kv_cache)}`), and
  `engine_options.kv_cache` is filled by `serve_options.cpp parse_kv_dtype` /
  `apps/cli/options.cpp parse_kv_cache`, both of which map `nvfp4`/`iso4e`/`rk4v4` onto their real
  enumerators. So `--kv-dtype nvfp4` produced a request log saying `"kv_cache":"unknown"` — the
  service's persistent record claiming a tier the engine does not have, with no warning anywhere.

## 2. The canonical table — `src/product/kv_storage_dtype.h`

`kKvStorageNames` is `inline constexpr auto ... = std::to_array<std::string_view>({...})` — **an
array, not a switch** — and `kKvCacheStorageCount` is pinned to the last enumerator. Nine
`static_assert`s close both directions:

| Assertion | Catches |
|---|---|
| `static_cast<std::size_t>(KvCacheStorage::Dropped) + 1 == kKvCacheStorageCount` | an enumerator was appended and the count was not bumped |
| `std::size(kKvStorageNames) == kKvCacheStorageCount` | a name was added or dropped without growing the pin |
| eight `kKvStorageNames[static_cast<std::size_t>(KvCacheStorage::X)] == "x"` | a **complete but reordered** table |

`std::to_array` is deliberate: its size is **deduced** from the initialiser list, so
`static_assert(std::size(...) == N)` *can* fail. With `std::array<T, N> t{...}` it cannot — the
unwritten tail is value-initialised and `size()` is `N` no matter how many entries were written.

The positional assertions are not hypothetical busywork. `serve/kv_auto_relayout.cpp`'s switch lists
`Fp8Group16` **before** `Nvfp4Group16` while the enumeration has it **after**, so a mechanical
switch-to-array conversion there would swap those two tiers. A reordered table names every value and
mislabels all of them — the other silent failure mode.

Vocabulary (`@3944a53`, enumerator order):

| Index | Enumerator | Token |
|---|---|---|
| 0 | `BFloat16` | `bf16` |
| 1 | `Int8Group64` | `int8-g64` |
| 2 | `Fp8E4M3Row256` | `fp8-e4m3-r256` |
| 3 | `Nvfp4Group16` | `nvfp4-g16` |
| 4 | `Fp8Group16` | `fp8-g16` |
| 5 | `Iso4eGroup16` | `iso4e-g16` |
| 6 | `Rk4v4Group64` | `rk4v4-g64` |
| 7 | `Dropped` | `dropped` |

The three tokens that already shipped are byte-identical to what the old switches returned, so
existing log consumers keep parsing.

**`kv_storage_token()` keeps a LOUD contract.** The table is total over the
enum, but `KvCacheStorage` is a `std::uint8_t` fed from option text, so a cast or an untrusted input
can produce a value no enumerator names. The lookup refuses instead of reading past the end:

```cpp
if (static_cast<std::size_t>(storage) >= kKvStorageNames.size()) {
    throw std::invalid_argument(
        std::string("KV storage code ") + std::to_string(static_cast<unsigned>(storage)) +
        " names no enumerator, so it has no short name");
}
```

**This table does not absorb the register an operator reads.** `serve/kv_auto_relayout.cpp` uses a
different vocabulary (`"int8"`, `"fp8"`, `"nvfp4"`) and stays where it is. There is a second,
bench-only short vocabulary (`bf16` / `int8-group64` / `fp8-e4m3-row256`) in
`bench/targets/qwen3_6_27b/ninfer_bench_support.cpp`; it stays a short vocabulary by design, and
what changed there is only that a **future** enumerator is still reported by `-Wswitch` instead of
being absorbed. The canonical engine-side table is `core/device_capabilities.h`'s
`kv_storage_name()`.

## 3. `dtype_of` — one answer per meaning of `DType::BF16`

`DType::BF16` has **two meanings** and `dtype_of()` must only ever produce one of them:

- As an **answer** it means "this layer is bf16". That is what `case BFloat16: return DType::BF16`
  says, and it is correct.
- As an **absent answer** it is the "inherit the global `--kv-dtype`" sentinel
  (`decoder_state.cpp plan_cache()`'s `layer_dtype()` via
  `product/kv_component_switch.h kv_resolve_slot_dtype`), i.e. "no answer, use the operator's
  `--kv-dtype`".

So the function's codomain is the set of **real** tiers, and a storage that carries none is
**refused**. There is no arm, and no trailing statement, that returns `DType::BF16` because it did
not know. Concretely, `case KvCacheStorage::Dropped:` throws, because a discarded layer
(`NINFER_KV_DROP_LAYERS`) owns no KV planes and has no codec to resolve — and classifying it as bf16
would simultaneously report "this layer is bf16" **and** "ignore the operator's `--kv-dtype` here".
A value above the enumerator count throws too.

This is the tree's own idiom for this enum, not an invention: `product/kv_storage_dtype.h` and
`layouts_impl.h target_kv_cache_profile()` both switch over the **same** enum with no `default:` arm
and throw after the switch.

**The refusal is pinned executably**, in two places over the same enum:

- `tests/test_kv_tier_formats.cpp`'s EXECUTABLE GUARD block: a positive control
  (`dtype_of(KvCacheStorage::BFloat16) == DType::BF16`, so the two checks cannot pass vacuously),
  `rejects("owns no KV planes", ...)` for `Dropped`, and `rejects("names no enumerator", ...)` for
  the out-of-range byte `9`. Before this block existed, **reverting `case KvCacheStorage::Dropped:`
  to the old `default: return DType::BF16;` left the whole suite GREEN.** A guard nobody executes is
  a comment.
- `tests/test_kv_component_switch.cpp`, whose 256-code sweep requires exactly **249 refusals** —
  code 7 (the `Dropped` arm) among them — and **zero** codes resolving to `BF16`. Returning anything
  else from the production mapping turns that test red.

### 3.1 The message that was wrong, not the semantics

`src/product/kv_storage_dtype.h`'s `Dropped` arm exists because falling **past** the switch already
threw, so the semantics were right — but the message was not. The trailing throw said
"KV storage code 7 has no DType", which **claims no enumerator names it**. `include/ninfer/types.h`
does name it, and the truth is that the layer was **discarded** (`NINFER_KV_DROP_LAYERS`) and owns
no KV planes to build a codec for. Naming the arm also makes this switch answer the question the
header comment above it promises.

### 3.2 Sibling closures in the same idiom

Five more hand-written tables over the same two enums were closed in the same batch, each with the
answer its trailing `return`/`throw` already gave, so only the **reporting** changed:

| File | What was named | Answer kept |
|---|---|---|
| `src/core/device_capabilities.h` (capability switch) | `KvCacheStorage::Dropped` | `kKernelImageBaseline` — a discarded layer adds no codec capability to the preflight. A **decision**, not a fallthrough that happens to work |
| `src/core/device_capabilities.h` (`kv_storage_name()`) | `KvCacheStorage::Dropped` | `"dropped"` |
| `src/core/dtype.cpp` (`dtype_size()`) | `DType::NVFP4`, `ISO4E`, `Rk4v4Kv` | refusal, with a corrected message (below) |
| `src/ops/linear_pair/w8/w8_pair_plan.cpp` | the seven `ExactConcatMma<R,C>` ids | `logic_error` with a message naming which situation this is |
| `src/targets/qwen3_6/impl/runtime/layouts_impl.h` (`target_kv_cache_profile()`) | `KvCacheStorage::Fp8Group16`, `SpeculativeBackend::Auto` | see below |
| `bench/targets/qwen3_6_27b/ninfer_bench_support.cpp` | all of `KvCacheStorage` and `ProposalHead` | `"unknown"` |

**`dtype_size()` must refuse packed plane formats rather than call them invalid.** `NVFP4` / `ISO4E` /
`Rk4v4Kv` are **packed** plane formats (two codes per byte, scales in a separate plane), so there is no
scalar element size to return. The message is now:

```
"dtype_size: this DType has no scalar element size. A packed plane format "
"(nvfp4/iso4e/e8kv) is sized by the per-layer KV storage table, not by "
"rows*columns*element_size; any other code names no enumerator at all."
```

A caller that multiplied `rows*columns` by this would size a plane **wrong**
(`core/host_kv_arena.cpp`, `core/tensor.cpp`). The old refusal called a perfectly valid `DType`
"invalid".

**`Fp8Group16` was a real mapping hole, not just an unnamed arm.** Both fp8 spellings name the
**same** target tier, and the tree says so in three places: `src/product/kv_storage_dtype.h`,
`layouts_impl.h`'s own note ("target_kv_cache_profile() maps both to ..."), and
`core/device_capabilities.h`, which groups them under one capability set. `product::parse_kv_storage`
emits `Fp8Group16` for a per-layer `fp8` spec (`src/product/kv_options.h`). Only `Fp8E4M3Row256` was
named, so the `Fp8Group16` spelling fell past the switch into the trailing "unknown KV-cache storage
profile" throw — **the comments claimed a mapping the code lacked.**

**`case SpeculativeBackend::Auto:` is tolerated on purpose, and named rather than defaulted.** `Auto`
is resolved before planning by construction: `construct_registered()` calls
`Target::resolved_auto_speculative()` up front "so the planner, the load plan and the program all
see the same concrete backend" (`src/targets/registry.cpp`), and each package resolves it the same
way (`src/targets/qwen3_6_27b/impl/package.cpp`). So this arm is **not** a hole in the validation: it
states that `Auto` has nothing left to validate here. It does **not** throw, because nothing in that
batch established that `Auto` is unreachable on **every** path into the planner, and converting a
skip into a startup failure on an unproven reachability claim is exactly the sort of change that
must not be made silently. A stricter throw is a reported candidate, not a landed change.

**`w8_pair_plan.cpp`'s seven ids can never be entered.** The seven `ExactConcatMma<R,C>` ids are
mapped by `homogeneous_schedule()` onto their `ConcatMma<R,C>` representative, and this switch is ON
the homogeneous form. Named rather than `default:`ed so a future enumerator still reports here:

```
"w8 pair: exact concat schedule has no runtime column tile "
"(homogeneous_schedule maps it onto its ConcatMma representative)"
```

## 4. The switch-coverage gate — `CMakeLists.txt`

The assertions above close the tables that have an array behind them. The gate in `CMakeLists.txt`
closes the sites that are still switches, by making a missing arm **fail the build**.

Two CMake cache variables:

| Variable | Default | Values | Meaning |
|---|---|---|---|
| `NINFER_WARN_SHADOW_SWITCH` | `off` | `off` \| `report` \| `error` | switch-coverage gate strength |
| `NINFER_WARN_ERROR_CHECKS` | `switch` | semicolon list | which check names `-Werror` applies to when the gate is `error` |

Three states, and only the third is a hard gate:

- **`off`** — no flag (historical). The removed-`default:` protection in `program_impl.h` is
  **inert** here.
- **`report`** — adds `-Wswitch -Wswitch-enum`. Zero build risk on this tree (measured: 326/326 host
  TUs still `rc=0`), and the protection now **prints**.
- **`error`** — `report` plus `-Werror=<check>` for each entry of `NINFER_WARN_ERROR_CHECKS`.

### 4.1 Two decisions a maintainer must not undo

**CXX only, no nvcc.** The diagnostic targets a host header reached only through the four host
`variant.cpp` carriers (`program_impl.h` is reached solely via
`src/targets/qwen3_6/impl/runtime/instantiate.h`), so a CUDA half would add an **unmeasured** surface
for no coverage. Add it in a separate change, with `.cu` evidence.

**Do not name `switch-enum` in `NINFER_WARN_ERROR_CHECKS` on this tree.** `-Wswitch-enum` is
measured at **767 warnings over 19 files**, and most of those switches name only part of their
enumeration **on purpose**. Escalating that check turns 767 warnings into 767 build errors — that is
not a landing, it is a rewrite of 19 files. The check the removed `default:` needs is plain
`-Wswitch` ("an arm is missing **and** there is no `default:`"), which is a strict subset of
`-Wswitch-enum`.

`-Wall` is deliberately **not** used to open `-Wswitch`: on this tree it would also open ~600
unrelated warnings, which is a different decision from this gate.

### 4.2 The site-level escalation — `program_impl.h`

The gate above is a build-tree decision. The guarantee "adding a `DType` fails the build" needs the
escalation **at the site**, because on this tree — measured with a 12th `DType` added to
`core/dtype.h` and the real carrier TU `src/targets/qwen3_6_27b/impl/variant.cpp`:

| Flags | Result |
|---|---|
| none | `rc=0`, and the switch is not mentioned at all. The only default-on diagnostic is `-Wreturn-type` |
| `-Wswitch`, `-Wall -Wswitch` | `rc=0`: a **warning**, never an error |
| `-Wswitch -Werror=switch` | `rc=1`, but that same flag **also** errors on this tree's other `-Wswitch` sites, so it is not a CMake gate that can be switched on tree-wide today |
| the `#pragma` alone, no flags | **`rc=1`** |

So `program_impl.h` pushes `#pragma GCC diagnostic error` for `-Wswitch` **and** `-Wswitch-enum`
around the `storage_of` lambda. `-Wswitch-enum` is escalated as well as `-Wswitch` because the lambda
is exhaustive at both strengths, and `-Wswitch-enum` is the stricter one — it still fires if a
`default:` is ever re-added next to a missing arm. It costs nothing here and cannot leak: every one
of `DType`'s 11 enumerators is named inside the region, so the region is silent on the complete enum.

`storage_of` also lost its `default:` arm. It used to end in
`default: return KvCacheStorage::BFloat16;` — the per-layer "inherit the global `--kv-dtype`"
sentinel, i.e. a silent no-op **reported as success**. The six KV tiers are the real mapping; the
five control dtypes (`FP32`, `I32`, `U8`, `I64`, `FP16`) are named in their own arm as values that
cannot reach either caller. `plan.kv_dtype` comes from `target_kv_cache_profile()` (image = the six
above) and `layer_view.dtype` comes from `PagedKVCacheLayout::layer_dtypes`, which `plan_cache()`
validates to exactly those six ("Paged KV per-layer dtype is invalid"). The reachable answer for the
control dtypes is kept (`BFloat16`) because this function is `noexcept` by contract —
`memory_summary()` is `noexcept` — so the fix cannot be a throw; making the arm explicit is the
available one.

## 5. Adding a `KvCacheStorage` enumerator — checklist

1. Bump `kKvCacheStorageCount` in `src/product/kv_storage_dtype.h` and add the token to
   `kKvStorageNames` **at its own enumerator index**. The two `static_assert`s catch a miss either
   way; the eight positional ones catch a reorder.
2. Decide, for the new value, which of the two cases `dtype_of()` is in: a real tier, or a
   "carries none" value that must **throw**. There is no third option and no `default:`.
3. Name it in `core/device_capabilities.h` (both the capability switch and `kv_storage_name()`), in
   `layouts_impl.h target_kv_cache_profile()`, and in `src/core/dtype.cpp dtype_size()` if it is a
   packed format with no scalar element size.
4. Decide explicitly whether the bench-only vocabulary in
   `bench/targets/qwen3_6_27b/ninfer_bench_support.cpp` and the operator-facing register in
   `serve/kv_auto_relayout.cpp` should carry it. Both are **deliberately narrower** than the
   canonical table; name the enumerator and keep the narrower answer if that is the decision.
5. If the value is a **discarded** layer, nothing else is needed — `Dropped` is the worked example.
   Otherwise add the tier to the budget ladder and to the format table and update
   `tests/test_kv_tier_formats.cpp`'s guard with a **positive control** next to the new check.
6. Run the build with `-DNINFER_WARN_SHADOW_SWITCH=report` at minimum. With `error` and the default
   `NINFER_WARN_ERROR_CHECKS=switch` the gate is a hard failure, which is what a table closure is
   for.
