# Changelog

This file records what changed in NInfer, newest first. Entries name the flag, environment
variable, build option, or contract that changed, and the file that owns it. Where a change is
internal only, it says so rather than being written as a user-visible feature.

The repository has no release tags yet; the sections below are keyed by the landing batch rather
than by a version number. When the first tag is cut, replace the batch heading with the tag and
keep the entries.

## Unreleased — documentation completeness increment

This increment documents **four landed change families that the batch below does not describe**, plus
three names that were documented wrongly or not at all. Nothing here is a code change: every claim is a
read of the tree at the base revision, and the base is the same tree as the batch below.

An increment is **bounded by the patches it covers**, and the bound is stated here rather than left for
a reader to infer.

Base revision: `3944a53eda1aac439a566a1cf46ea741f0415fdc`.

### Added

- **Multi-device sharding: the shard arithmetic, the per-rank persistence naming, and the one predicate
  the KV policy needs.** `src/core/shard_plan.h` adds the rank/world shape (`ParallelAxis::None |
  Tensor | Pipeline`, with `axis=none` and `world_size > 1` refused in **both** directions), the GQA-safe
  shard rules (`kv_heads % world_size` and `weight_columns % world_size` for a head split — the first is
  the one no cross-rank reduction can repair; an uneven but layer-complete split for a stage split), the
  per-rank artifact naming (`.tp1of4` inserted before the suffix, single-device names returned byte for
  byte), the `ResidencyState` triple including `partial-resident` — *a state the model does not define* —
  the conjunct-factoring result that makes the pipeline axis free and the tensor axis a new collective on
  the eviction driver's critical path, and the rank-aware cold spill cell path (today two ranks open the
  same `/tmp` file and collide **deterministically**). It is **host-only and has no call path**; the
  header's closing section names every missing piece (rank plumbing, collectives, peer access, a per-rank
  weight loader, a rank field in the recall record). Contract:
  [`docs/maintainer/multi-device-and-shard-plan.md`](docs/maintainer/multi-device-and-shard-plan.md).
- **The central per-architecture GEMM route table.** `src/core/kernel_route.h` adds
  `select_route(sm, format, shape)` over seven mechanism-named routes and three outcomes, table-driven
  from `arch_caps.h`'s ladder and per-format floors so a route decision is derived from the same kernel
  evidence the artifact gate uses. It is **non-gating by construction** (an unknown compute capability
  falls back and warns instead of throwing; only a genuinely absent kernel is a refusal, and then `why`
  names the missing kernel). It is **not** the artifact gate and **not** a placement planner. Owning file:
  `src/core/kernel_route.h`. The statement it replaces — `if (device.sm() != 120) throw ...` — is **still
  in HEAD** at `src/targets/qwen3_6/impl/runtime/layouts_impl.h` and is deleted in the working tree,
  where `src/core/device_capabilities.h` + `src/core/device_probe.cu` replaced it. Both of those are
  gates; this is the route **table** that was missing from both.
- **`--prefill-chunk-mode dynamic|manual`** — which of the two modes owns the prefill unit. `manual` pins
  it to `--prefill-chunk` for the whole run with the governor constructed disabled; `dynamic` (default)
  lets the governor re-derive it. Case-sensitive, an unknown value is refused, and the resolution order is
  `CLI > environment > default` with `NINFER_FT_BW_GOV` as the environment spelling — so
  `--prefill-chunk-mode manual` reaches exactly the same state as `NINFER_FT_BW_GOV=0`. Owning files:
  `src/runtime/engine/bandwidth_governor.h` (the spellings, the mode → enabled mapping and the
  resolution), `include/ninfer/types.h` (`PrefillChunkMode`), `apps/cli/options.{h,cpp}`,
  `src/serve/serve_options.cpp`, `apps/perplexity/main.cpp` (which accepts only `manual`). User text:
  `docs/cli.md` § Common options.
- **The MTP proposal-side fill.** `src/targets/qwen3_6/impl/runtime/mtp_tree_proposal_fill.h` is the push
  that makes the draft-tree path more than a layout type: it reads the round's **own device frame**
  (`MtpDecodeEgress::next_proposal_ids`, laid out `s * depth_stride + i * lanes + t`), checks the lattice
  (`mtp_proposal_row_defect()` returns the defect as text; three classes it **cannot** decide are named in
  place rather than covered by a rule), and publishes through the receiving seam in one call
  (`fill_mtp_tree_round()`). The spine-vs-argmax agreement is a **same-source consistency** check, not two
  independent sources, and the header says so. Owning files:
  `src/targets/qwen3_6/impl/runtime/mtp_tree_proposal_fill.h`,
  `src/targets/qwen3_6/export/ninfer/targets/qwen3_6/round_state.h` (`kMtpTreeMaximumPaths` =
  `kMtpDecodeMaximumDrafts`, `kMtpTreeProposalDepthStride` = `kMtpTreeMaximumPaths *
  kMaximumConcurrency`, `kMtpTreeProposalEntries`), `.../mtp_impl.h` (the extraction call sites),
  `.../program_impl.h` (the two callers: the decode hand-off and the prefill bridge). Contract:
  [`docs/maintainer/mtp-draft-tree.md`](docs/maintainer/mtp-draft-tree.md) §2.9.
- **Two names around the capability gate that are not what they look like.** `NINFER_BUILD_CUDA_ARCHES`
  is a **compile-time macro** the build exports from `CMAKE_CUDA_ARCHITECTURES` (default `120a`), not an
  environment variable, and `#ifdef` is the only way to see it — it exists because a `-a` suffix is not
  observable through `cudaDeviceProp`. `NINFER_ARCH_WARN` **is** a real environment variable: by default
  an unlisted compute capability produces no output at all, and setting this variable is what sends the
  same warning to stderr. Owning file: `src/core/arch_caps.h`. User text: `docs/cli.md` § The build's
  arch list, and the unknown-arch warning.

### Fixed

- ⚠️ **The build's arch-list macro and the header that reads it use two different names, so
  `arch_caps.h`'s `build_arch_list()` is dead in this build.** `src/CMakeLists.txt` defines
  **`NINFER_BUILD_CUDA_ARCHS`** (no `E`) and scopes it to the single translation unit
  `core/device_probe.cu`; `src/core/arch_caps.h` reads **`NINFER_BUILD_CUDA_ARCHES`** (with `E`) in its
  `#ifdef`, its `return` and its operator-visible sentence. `#ifdef NINFER_BUILD_CUDA_ARCHES` therefore
  cannot be true, so `build_arch_list()` always returned the empty view. **FIXED**: the reader now
  looks for the name the build writes (`..._ARCHS`, both spellings accepted), so every TU that is
  given the macro reads it. The sentence that told a reader to "rebuild with
  `CMAKE_CUDA_ARCHITECTURES` exported as `NINFER_BUILD_CUDA_ARCHES`" named a macro this build cannot
  define -- **an operator who followed it rebuilt and read the SAME line, and nothing anywhere
  reported that the prescribed remedy had failed**; the remedy now names `src/CMakeLists.txt` and
  the `ninfer_build_cuda_arches` INTERFACE target instead. `src/core/device_capabilities.h` uses the no-`E` spelling and does receive the
  list, so **the engine still learns its arch list** — the defect is confined to the second, doc-facing
  spelling. `tests/test_arch_caps.cpp` accepted either branch on purpose and therefore could not
  catch this; the assertion is now two-branch, and in the configuration where the macro is unread it
  goes RED. Behavioural proof (control clean, mutants compile and are caught by the run):
  `dl/routeprobe/proof_arches.txt`.
  Owning files: `src/CMakeLists.txt` (the definition) and `src/core/arch_caps.h` (the reader). This
  increment first **documented** the mismatch in `docs/cli.md` and in
  [`docs/maintainer/multi-device-and-shard-plan.md`](docs/maintainer/multi-device-and-shard-plan.md) §5.1;
  it does not rename either side, because that is a build change.
- **The CLI reference did not list `--prefill-chunk-mode`.** `ninfer --help` prints it
  (`apps/cli/options.cpp`), `apps/cli/options.h` carries it, and `src/serve/serve_options.cpp` parses it,
  but `docs/cli.md` listed only `--prefill-chunk` — the **ceiling**, not the mode. The mode *was*
  described in `docs/serving.md` § KV strategy on the server; the CLI reference, which is the authority
  for option spelling, was the document that lacked it. It now has a row and the paragraph under it.
  Owning file: `apps/cli/options.cpp`.
- **Neither `NINFER_ARCH_WARN` nor `NINFER_BUILD_CUDA_ARCHES` was documented anywhere.** The first is the
  only `getenv` in `src/core/arch_caps.h` and the second appears in the operator-visible capability text;
  both were absent from every document in the tree. Owning file: `src/core/arch_caps.h`.
- **Three landed host tests were unreachable.** `tests/test_kernel_route.cpp` (340 lines),
  `tests/test_shard_plan.cpp` (534) and `tests/test_multidev_wiring.cpp` (347) exist in the tree and are
  referenced by nothing: `kernel_route` / `shard_plan` / `multidev` each occur **0** times in
  `tests/CMakeLists.txt@a22d29074e08ca7e`. The three are host-only `int main` programs, so registering
  them is a `tests/CMakeLists.txt` change and nothing else; this increment **records** the state and
  names the subject matter, and does **not** register them. Owning file: `tests/CMakeLists.txt`.
- **`docs/maintainer/op-development.md` had no list of the batch's new host test targets.** Twenty
  registrations are new in this session; §6.4 now names each one with the source it pins, separates the
  two that had never been compiled by anything from the eighteen that were merely unnamed, states that
  `NEEDS_SOURCE_DIR` occurs **eight** times as an argument (of eleven occurrences in the file, three of
  which are the helper's own machinery and a comment), and records the three unregistered tests above.
  Owning file: `tests/CMakeLists.txt`.

### Owner files of this increment

**Eighteen** rows, one per owner file. The count is given here because a table whose heading describes a
different set from its rows cannot be checked:

| Owner file | What it owns |
|---|---|
| `src/core/shard_plan.h` | the rank/world shape, the shard arithmetic, the per-rank naming, the residency-bit question |
| `src/core/kernel_route.h` | the per-architecture route table and its three outcomes |
| `src/core/arch_caps.h` | the capability ladder, the per-format floor table, the gate, `NINFER_ARCH_WARN`, `build_arch_list()` |
| `src/core/device_capabilities.h` | the host half of the real-probe capability gate |
| `src/core/device_probe.cu` | the device half of the same probe |
| `src/core/device_sm_count.h` | the SM count of the current context — a tuning input, not a capability test |
| `src/spec/turn_recall_journal.h` | the recall record's 64-byte budget and the shard axis |
| `src/targets/qwen3_6/impl/runtime/mtp_tree_proposal_fill.h` | the proposal-side fill and its row gate |
| `src/targets/qwen3_6/export/ninfer/targets/qwen3_6/round_state.h` | the egress layout and its three stride constants |
| `src/targets/qwen3_6/impl/runtime/mtp_impl.h` | the extraction call sites that write the lattice |
| `src/targets/qwen3_6/impl/runtime/program_impl.h` | the two callers of `fill_mtp_tree_round()` |
| `src/runtime/engine/bandwidth_governor.h` | `PrefillChunkMode`, its two spellings and the `CLI > environment > default` resolution |
| `include/ninfer/types.h` | the `PrefillChunkMode` enumerator |
| `apps/cli/options.{h,cpp}` + `src/serve/serve_options.cpp` | the two front ends that parse the mode |
| `apps/perplexity/main.cpp` | the front end that accepts only `manual` |
| `src/product/kv_perlayer_policy.h` | the per-layer KV policy the `ninfer_kv_perlayer_policy_test` registration pins |
| `src/CMakeLists.txt` | the arch-list macro's definition (the `NINFER_BUILD_CUDA_ARCHS` spelling, and the one-TU scope) |
| `tests/CMakeLists.txt` | the twenty registrations, the `NEEDS_SOURCE_DIR` keyword, and the three test sources nothing references |

Two of those rows are build files rather than modules, and they are here because the entries above name
them as the owner of a defect that spans a build file and a header. They are not new subject matter.

### Owner files of the batch below that its own entries do not name

The batch below says its entries name "the file that owns it". **Four** of the modules it describes were
never named anywhere in it, and one row here is a **modified carrier rather than a new file**, which is
called out so the table is not read as an inventory of new files:

| Owner file | What it owns | New file? |
|---|---|---|
| `src/product/kv_component_switch.h` | the KV component switches, and the `NINFER_KV_ROTATION` / `NINFER_KV_ROWSCALE` environment spellings | yes |
| `src/targets/qwen3_6/impl/runtime/mtp_tree_produce.h` | the producing half of the draft-tree layout | yes |
| `src/targets/qwen3_6/impl/runtime/mtp_tree_publish.h` | the receiving half — the single place a tree layout enters the runtime | yes |
| `include/ninfer/ops/mtp_proposal_topk.h`, `src/ops/kernel/mtp_proposal_topk.cuh`, `src/ops/launcher/mtp_proposal_topk.cu` | the per-token top-L extraction the lattice is read from | yes |
| `src/serve/kv_auto_relayout.cpp` | the auto-relayout pass and its environment knobs (`NINFER_FT_COLD_PAGES` is its only read; `NINFER_FT_COLD_MODE` is reserved and **not wired**) | **no — modified carrier** |

### Scope of this increment, stated as a bound

The four families above are the whole of this increment's subject matter:

| Family | Carriers |
|---|---|
| multi-device sharding | `src/core/shard_plan.h`, `src/core/kernel_route.h`, `src/core/arch_caps.h`, `src/spec/turn_recall_journal.h` |
| the MTP proposal-side fill | `src/targets/qwen3_6/impl/runtime/mtp_tree_proposal_fill.h`, `.../round_state.h`, `.../mtp_impl.h`, `.../program_impl.h` |
| the prefill-chunk mode | `src/runtime/engine/bandwidth_governor.h`, `include/ninfer/types.h`, `apps/cli/options.{h,cpp}`, `src/serve/serve_options.cpp`, `apps/perplexity/main.cpp` |
| the per-layer KV policy | `src/product/kv_perlayer_policy.h` |

**In the tree at the same base revision, and NOT described by this increment** — named rather than left
silent, because "not in this increment" and "not documented anywhere" are different claims and only the
first one is being made: the DFlash2/ddtree Op family (`include/ninfer/ops/dflash2_ddtree.h`,
`dflash2_ddtree_beam.h`, `dflash2_tree_walk.h`, their kernels, launchers, wrappers and
`tests/ops/test_dflash2_ddtree*.cpp` with its golden generator), the `ISO3` KV codec
(`src/ops/kernel/gqa_iso3_codec.cuh`), the NVFP4 W4A4 ladder
(`src/ops/linear/nvfp4/nvfp4_w4a4_ladder.cuh`), the per-codec decode launchers
(`src/ops/launcher/gqa_attention_decode_{bf16,fp8,i8,iso3,nvfp4}.cu` and the split/tiers headers), the
`qwen3_5_9b` and `qwen4_exp` (PLE) targets, `tools/convert/gguf_kquant.py` + `gguf_names.py`,
`tools/archkit/archkit_target.py`, and the `tools/gui/` i18n family. None of them is described here.

### Verification status of this increment

- **Documentation only.** No code file was changed, nothing was compiled, linked or run, and no GPU was
  touched. Every statement about a mechanism is a read of the file named beside it.
- The two counts the increment states — twenty new registrations at
  `tests/CMakeLists.txt@a22d29074e08ca7e` (554 lines, 64 registrations) and eight `NEEDS_SOURCE_DIR`
  argument uses in that file — were counted at that revision rather than inherited from a review.
- The three unregistered multi-device tests are **recorded as unregistered**. Registering them is a
  build-side change, and this increment makes no claim about whether they would pass.

## Unreleased — MTP draft-tree execution, KV storage name table, switch-coverage gate

Base revision for this batch: `3944a53eda1aac439a566a1cf46ea741f0415fdc`.
Scope: 14 tree patches over 33 files (16 modified, 17 created).

### Added

- **MTP draft-tree verify shape (`--draft-tree L,d`, `L > 1`) now has an execution path.** The
  option itself is older; what this batch adds is the machinery behind it. Four new files carry it
  and all four are host-only, engine-free, standard-library-only so a CPU test can pin the whole
  tree arithmetic:
  - `src/targets/qwen3_6/impl/runtime/mtp_tree_produce.h` — the **producing** half. Turns the
    proposal side's per-depth candidate lattice into a verify-column layout. Two entry points:
    `mtp_tree_columns_from_lattice()` (full lattice: per-depth top-k ids plus the top-k x top-k
    predecessor x successor score grid, runs the real `ops::ddtree::build_tree` reference builder)
    and `mtp_tree_columns_from_siblings()` (sibling lattice: per-depth top-paths ids only, no extra
    forward). Each has a `<name>_defect()` predicate that returns the defect as text and a
    validating wrapper that throws it.
  - `src/targets/qwen3_6/impl/runtime/mtp_tree_publish.h` — the **receiving** half, the single
    place a tree layout can enter the runtime. `MtpTreeColumns`, `mtp_tree_columns_from_ddtree()`,
    `mtp_tree_layout_defect()` (12 named defects), `validate_mtp_tree_columns()`,
    `MtpTreePublishSlot`, `publish_mtp_tree()`, `clear_mtp_tree()`, `mtp_tree_required_extent()`.
  - `include/ninfer/ops/mtp_proposal_topk.h` + `src/ops/kernel/mtp_proposal_topk.cuh` +
    `src/ops/launcher/mtp_proposal_topk.cu` — per-token top-L rows of a row-major 2-D bf16 table.
    **Additive and, as landed, not yet called**: the proposal loop that must call it is the
    remaining half. The closest precedent in this tree is `include/ninfer/ops/dflash2_tree_walk.h`
    ("ADDITIVE and DEFAULT OFF ... Nothing calls it yet"), landed ahead of its consumer for the same
    reason.
- **`ops::mtp_tree_commit_history`** (`include/ninfer/ops/mtp_round.h`) — commits a tree round's
  accepted chain (which occupies arbitrary verify columns) into the position prefix the next round
  reads as history. Publishes five named consistency flags in `kMtpTreeFlag*`:
  `ColumnOutOfRange`, `MaskMissingSelf`, `DepthMismatch`, `ChainOrder`, `PositionOutOfRange`.
- **`ops::mtp_draft_align_hidden`** (`include/ninfer/ops/mtp_round.h`) — re-indexes the verify
  hidden from verify-column order to accepted-chain depth order. Required because
  `alignment_ids[j] = verify_ids[j+1]` pairs index `j` with `hidden[:,j]`, which is depth-correct
  only while the accepted nodes are the column prefix — true for a chain round, false for a tree
  round. Refuses `out` aliasing `hidden`.
- **`kMtpTreeFlag*` diagnostics are consumed as a hard error, not a silent wrong history**
  (`src/targets/qwen3_6/impl/runtime/program_impl.h`).
- **A tree round is refused on the host, before the round, on the non-greedy routes**
  (`program_impl.h`): the mask-driven accept reads the verifier's own argmax, which only the greedy
  route produces, so the sampling and penalty routes would verify a tree with the masks silently
  ignored.
- **`MtpDecodeIngress::target_column_depths` and `MtpDecodeEgress::accepted_columns`,
  `chain_sources`, `tree_commit_flags`** (`src/targets/qwen3_6/impl/runtime/schedule.h`,
  `src/targets/qwen3_6/export/ninfer/targets/qwen3_6/round_state.h`).

- **The canonical short-name table of every `KvCacheStorage`** (`src/product/kv_storage_dtype.h`):
  `kKvCacheStorageCount`, `kKvStorageNames`, `kv_storage_token()`. The table is a
  `std::to_array<std::string_view>` so its size is **deduced** from the initialiser list, and nine
  `static_assert`s pin it: one bidirectional count/length pair and eight positional checks. This
  closes both silent failure modes at compile time — a missing name and a complete-but-reordered
  table. The assertions are real compile errors, unlike a `switch` missing an arm, which this build
  does not even warn about by default.

- **Switch-coverage build gate** (`CMakeLists.txt`), two new CMake cache variables:
  - `NINFER_WARN_SHADOW_SWITCH` = `off` (default) | `report` | `error`. `off` keeps today's
    behaviour exactly. `report` adds `-Wswitch -Wswitch-enum` for CXX only. `error` additionally
    applies `-Werror=<check>` for each entry of `NINFER_WARN_ERROR_CHECKS`.
  - `NINFER_WARN_ERROR_CHECKS` = `switch` (default). A semicolon list of check names.
  - Both are CXX-only on purpose: the diagnostic target is a host header reached through the four
    host `variant.cpp` carriers, so a CUDA half would add an unmeasured surface for no coverage.
  - **Do not put `switch-enum` in `NINFER_WARN_ERROR_CHECKS` on this tree.** `-Wswitch-enum` is
    measured at 767 warnings over 19 files, most of them switches that name part of their
    enumeration on purpose; escalating it turns 767 warnings into 767 build errors.

- **Two host tests**, registered in `tests/CMakeLists.txt`:
  - `ninfer_qwen3_6_mtp_tree_publish_test` (receiving half: layout validation and the sequence
    seam, against the real ddtree reference builder).
  - `ninfer_qwen3_6_mtp_tree_produce_test` (producing half: builds the verify-column layout from the
    proposal side's candidate lattice and, for the sibling shape, against `ddtree::column_depth` /
    `column_ancestors`, then publishes it through the receiving seam).

### Changed

- **MTP draft widths are snapped to the ladder the target actually captures**
  (`program_impl.h`, `mtp_ladder_round()`). The judgement is still made on the full decision ladder
  (the cost model is calibrated on it), but the capture domain is the target's. A target with
  `kMaximumMtpDraftTokens == 5` captures `{2,3,5}`, so a cut taken from the full ladder could name a
  rung of 7/9/15 that no captured graph covers, and `select_graph_profile` then threw "MTP batch
  CUDA Graph coverage is incomplete" — an engine stop for a rung this same code chose on purpose.
  Ties go up, matching `mtp_window_ladder_index`. For a target whose effective ladder is the full
  ladder this is a no-op. The statistics/trace rung now names `mtp_ladder.back()` instead of the
  decision ladder's top.
- **A chosen width that is not a rung of the captured ladder is now a named error**
  (`program_impl.h`): `throw std::logic_error("MTP target width is not a rung of the captured
  ladder")`. `head_floor` can still leave the ladder (it is `kMtpShortlistMinimumDrafts` under the
  Optimized proposal head), and the previous failure surfaced as a CUDA Graph coverage message that
  named the wrong thing. No target has such a ladder today — this is an unguarded shape, not a
  reachable bug.
- **`-Wswitch` / `-Wswitch-enum` are escalated to errors at the site that depends on them**
  (`program_impl.h`, `#pragma GCC diagnostic error` around the `storage_of` lambda). The guarantee
  "adding a `DType` fails the build" needs the escalation at the site because on this tree no flag
  is opened at all, `-Wswitch` is a warning even under `-Wall`, and `-Werror=switch` tree-wide also
  errors on this tree's other `-Wswitch` sites. The lambda names every one of `DType`'s enumerators,
  so the region is silent on the complete enum.
- **The cold-Host transfer rule is stated once** (`logical_kv_store.h`,
  `cold_host_device_replica_releasable()`). Three sites (`can_cold_host_prepare`,
  `can_cold_host_transfer`, and the inline guard inside `transfer_to_cold_host`) used to spell the
  same conjunction out by hand, so a one-sided edit could only wedge them: the eviction consumer
  keys on `can_cold_host_prepare` and then calls `transfer_to_cold_host`, whose guard would throw
  `logical KV page is not cold-Host-transferable` — a legal eviction raising a logic-error-family
  exception. `transfer_to_cold_host` now asks `can_cold_host_transfer(handle)` first, so if the
  eviction consumer's predicate says yes it cannot throw.
- **`storage_of` no longer has a `default:` arm** (`program_impl.h`). It used to end in
  `default: return KvCacheStorage::BFloat16;`, which is the per-layer "inherit the global
  `--kv-dtype`" sentinel — an unmapped dtype came out as a silent no-op reported as success. Every
  `DType` is now named, and the five control dtypes (`FP32`, `I32`, `U8`, `I64`, `FP16`) are named
  in their own arm as values that cannot reach either caller.
- **`--kv-bits`/`--kv-k-bits`/`--kv-v-bits` and their environment spellings are refused in
  combination with `--kv-dtype`, `--kv-layer-storage`, and `--kv-bit-budget`** in both front ends.
  These contradictions were previously refused by `ninfer-serve` only; the CLI accepted
  `--kv-bit-budget 4.5 --kv-dtype int8` and then read the ceiling with nothing, so the `int8` was
  accepted and dropped while producing the same dtype and payload as the ceiling alone.

### Fixed

- **A chunked tree round's mask admitted the wrong keys**
  (`src/ops/kernel/gqa_attention_decode_bf16.cuh`). The visibility test used the launch's **chunk**
  base, so for a width above `kSmallTChunkTokens` every earlier chunk's column was admitted
  unconditionally (`rel < 0`), silently over-visible-ing a tree round's branches. The test now uses
  the **round** base. For a chain the mask is the prefix `(1 << (j+1)) - 1`, so every column the old
  spelling admitted is admitted by the mask too — bit-for-bit unchanged, and identical for
  `column_begin == 0`.
- **A cold-path detach could drop the bytes of a page a second address still mapped**
  (`logical_kv_store.h`). The cold path was the only detach path that did not already refuse these
  pages, and the cold bookkeeping lives on the owning `SequenceState` alone, so a shared prefix that
  later materialised such a page found it `cold_compressed` with no source bookkeeping, and once the
  owner was gone the bytes were not recoverable. The refusal lasts exactly as long as the second
  address exists.
- **`kv_name()` in `apps/perplexity/main.cpp` named 4 of 8 `KvCacheStorage` values and threw for the
  rest.** `--kv-dtype nvfp4` parsed, loaded the artifact, scored the corpus, and then died at the
  first `kv_name` call. All eight are now named.
- **`kv_cache_name()` in `src/serve/request_log.cpp` named 3 of 8**, so `--kv-dtype nvfp4` produced
  a request log line saying `"kv_cache":"unknown"` — the service's persistent record claiming a tier
  the engine does not have, with no warning anywhere. All eight are now named, and the three that
  already shipped are byte-identical so existing log consumers keep parsing.
- **`dtype_of()` refused a `Dropped` layer and an out-of-range storage byte instead of returning
  `DType::BF16`** (`tests/test_kv_tier_formats.cpp` owns the test copy,
  `src/product/kv_storage_dtype.h` the production mapping). `DType::BF16` has two meanings and this
  function must only ever produce one of them: as an answer it means "this layer is bf16", as an
  absent answer it is the "inherit the global `--kv-dtype`" sentinel. A `Dropped` layer owns no KV
  planes, so classifying it as bf16 would simultaneously report "this layer is bf16" and "ignore the
  operator's `--kv-dtype` here".
- **`case KvCacheStorage::Dropped:` is named in `src/product/kv_storage_dtype.h`.** Falling past the
  switch already threw, so the semantics were right — but the message said "KV storage code 7 has no
  DType", which claims no enumerator names it. `include/ninfer/types.h` does name it, and the truth
  is that the layer was discarded (`NINFER_KV_DROP_LAYERS`) and owns no KV planes.
- **`dtype_size()` names `NVFP4` / `ISO3` / `E8Kv` and refuses them with the right message**
  (`src/core/dtype.cpp`). They are packed plane formats, so there is no scalar element size to
  return; the previous refusal called a perfectly valid `DType` "invalid".
- **`w8_pair_plan.cpp` names the seven `ExactConcatMma<R,C>` ids** and throws a message that says
  which situation this is, instead of a trailing throw that named nothing.
- **`device_capabilities.h` names `Dropped`** in both the capability switch and `kv_storage_name()`,
  returning what the trailing `return` already gave.
- **The bench-only short vocabulary in `bench/targets/qwen3_6_27b/ninfer_bench_support.cpp`** names
  every `KvCacheStorage` and `ProposalHead` enumerator, so a future enumerator is still reported by
  `-Wswitch` at that switch instead of being absorbed. The bench vocabulary stays bf16 /
  int8-group64 / fp8-e4m3-row256 by design; the canonical engine-side table is
  `core/device_capabilities.h`'s `kv_storage_name()`.
- **`target_kv_cache_profile()` names both fp8 spellings** (`layouts_impl.h`). `Fp8Group16` fell past
  the switch into the trailing "unknown KV-cache storage profile" throw, so a per-layer `fp8` spec —
  which `product::parse_kv_storage` emits as `Fp8Group16` — was rejected by a mapping whose own
  comments claimed it. `case SpeculativeBackend::Auto:` is named too, tolerated on purpose and
  documented as such: `Auto` is resolved before planning by construction, and converting that skip
  into a startup failure on an unproven reachability claim is not a change this line may make
  silently.
- **`ExactConcatMma` refusal in `w8_pair_plan.cpp`** (see above) and **`storage_of`'s removed
  `default:`** (see Changed).

### Internal

Not user-visible; recorded because they are contract changes another maintainer needs.

- The five `kMtpTreeFlag*` bits are a **contract**: a non-zero word means the tree metadata the draft
  side published is inconsistent and the round's accepted chain cannot be trusted. The runtime turns
  a non-zero word into a hard error rather than committing a guessed history.
- `mtp_draft_align_hidden` clamps a `chain_sources` value outside `[0,W)` to column 0 instead of
  using it as an offset, so a malformed table can degrade to a wrong gather but never to an
  out-of-range read. The table's own validation is `mtp_tree_commit_history`'s flag word.
- `mtp_tree_layout_defect()` and `mtp_lattice_defect()` return the defect as text and are separable
  from the throwing wrappers, so a test can assert the specific defect rather than just "it threw".
- `src/product/kv_storage_dtype.h` gained `#include "core/dtype.h"` and `#include <array>`; it
  remains untracked in git at the time of writing, so `git restore` cannot roll it back.

### Verification status of this batch

Stated precisely so the changelog is not read as stronger than it is.

- Every change in this batch is **landed in the working tree and matches its post-image hash**
  (`w_land4x/EXPECTED.tsv`, all 35 carriers `MATCH`).
- **Nothing in this batch was compiled, linked, or run** by the landing audit. The strongest
  statement established for the patches is "applies cleanly and matches a pinned post-image". Four
  pieces put code on the device path that **no one has compiled**: the mask-chunk-base change in
  `gqa_attention_decode_bf16.cuh`, the `.cu`/`.cuh` files of the tree-commit piece, and
  `mtp_proposal_topk.{cuh,cu}`. `nvcc -fsyntax-only` is not available in this environment.
- The two new host tests are registered with ctest; whether they are built and run is a build-side
  question.
