find_package(Python3 REQUIRED COMPONENTS Interpreter)

# This target deliberately receives no src/, CUDA, artifact, kernel, or target
# include root. It proves that the installed product headers stand alone.
add_executable(ninfer_public_api_test test_public_api.cpp)
target_include_directories(ninfer_public_api_test PRIVATE ${PROJECT_SOURCE_DIR}/include)
add_test(NAME ninfer_public_api_test COMMAND ninfer_public_api_test)
# The GQA split-geometry contract (see ops/test_gqa_split_contract.cu). This target deliberately
# links NO ninfer library and no device code of its own: the split policy is header-only and the
# route decision is inline, so it stays buildable -- and runnable -- while the engine's device link
# is not. Adding a library here would make it depend on cmake_device_link.o and destroy that
# property.
# The `allowed_tokens` mask contract of the sampling Op (ops/kernel/sampling.cuh,
# ops/kernel/sampling_device.cuh). The mask must defeat the greedy no-penalty RAW-logit fast path,
# not only the routes that already consult sampling_adjusted_logit(), so that a masked-out token
# can never be committed.  Same build shape as the two contract targets below and for the same
# reason -- header-only device code plus CUDA::cudart, NO ninfer library. A GPU is needed to RUN
# it: without one it prints SKIP and returns 77, which ctest reports as a failure of the test.
add_executable(ninfer_sampling_allowed_tokens_mask_test ops/test_sampling_allowed_tokens_mask.cu)
target_include_directories(ninfer_sampling_allowed_tokens_mask_test PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/third_party
  ${PROJECT_SOURCE_DIR}/third_party/utf8proc)
target_link_libraries(ninfer_sampling_allowed_tokens_mask_test PRIVATE CUDA::cudart)
add_test(NAME ninfer_sampling_allowed_tokens_mask_test COMMAND ninfer_sampling_allowed_tokens_mask_test)

add_executable(ninfer_gqa_split_contract_test ops/test_gqa_split_contract.cu)
target_include_directories(ninfer_gqa_split_contract_test PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/third_party
  ${PROJECT_SOURCE_DIR}/third_party/utf8proc)
target_link_libraries(ninfer_gqa_split_contract_test PRIVATE CUDA::cudart)
add_test(NAME ninfer_gqa_split_contract_test COMMAND ninfer_gqa_split_contract_test)

# The split-geometry DETERMINISM contract of the same family (ops/kernel/gqa_attention_decode.cuh):
# for one window, the split partition and the active-split count a partial kernel selects must be
# identical for a 1-column decode and a 6-column verify chunk, the reducer must read each column's
# OWN visible-key count, and the partial combination must not take its shape from blockDim.x.
# Same build shape as the contract test above and for the same reason: header-only host code plus
# CUDA::cudart, no ninfer library, so it keeps building and running while the engine's device link
# does not. It reads the kernel header AS TEXT for the combine-shape check, hence NINFER_SOURCE_DIR.
add_executable(ninfer_gqa_decode_split_exact_test ops/test_gqa_decode_split_exact.cu)
target_include_directories(ninfer_gqa_decode_split_exact_test PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/third_party
  ${PROJECT_SOURCE_DIR}/third_party/utf8proc)
target_compile_definitions(ninfer_gqa_decode_split_exact_test PRIVATE
  NINFER_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
target_link_libraries(ninfer_gqa_decode_split_exact_test PRIVATE CUDA::cudart)
add_test(NAME ninfer_gqa_decode_split_exact_test COMMAND ninfer_gqa_decode_split_exact_test)

# The RUNTIME form of the same contract: one binary, run twice -- NINFER_VERIFY_EXACT unset and
# NINFER_VERIFY_EXACT=1 -- printing how many (window, token-tile) pairs give a batch-1 decode a
# different split geometry than the verify chunk that replaces it. It needs no GPU and no build
# lock (the flag is read from the environment on the host and passed explicitly to the
# __host__ __device__ policy functions), which is the point: the neighbour's engine build and the
# model runs cannot block it. Build shape is the contract test's -- header-only host code plus
# CUDA::cudart, no ninfer library.
add_executable(ninfer_gqa_split_geometry_probe ops/gqa_split_geometry_probe.cu)
target_include_directories(ninfer_gqa_split_geometry_probe PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/third_party
  ${PROJECT_SOURCE_DIR}/third_party/utf8proc)
target_link_libraries(ninfer_gqa_split_geometry_probe PRIVATE CUDA::cudart)
add_test(NAME ninfer_gqa_split_geometry_probe_legacy
         COMMAND ninfer_gqa_split_geometry_probe)
# Pin BOTH arms: a globally exported NINFER_VERIFY_EXACT would otherwise relabel the legacy
# arm as the exact one and the pair would stop being a 2x2. Anything that is not "1" is legacy.
set_tests_properties(ninfer_gqa_split_geometry_probe_legacy PROPERTIES ENVIRONMENT "NINFER_VERIFY_EXACT=0")
add_test(NAME ninfer_gqa_split_geometry_probe_exact
         COMMAND ninfer_gqa_split_geometry_probe)
set_tests_properties(ninfer_gqa_split_geometry_probe_exact PROPERTIES ENVIRONMENT "NINFER_VERIFY_EXACT=1")


function(ninfer_add_test name)
  cmake_parse_arguments(arg "NEEDS_SOURCE_DIR" "" "SOURCES;LIBRARIES;ARGS" ${ARGN})
  add_executable(${name} ${arg_SOURCES})
  target_include_directories(${name} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/third_party
    ${PROJECT_SOURCE_DIR}/third_party/utf8proc)
  if(arg_LIBRARIES)
    target_link_libraries(${name} PRIVATE ${arg_LIBRARIES})
  else()
    target_link_libraries(${name} PRIVATE ninfer_core)
  endif()
  if(arg_NEEDS_SOURCE_DIR)
    target_compile_definitions(${name} PRIVATE
      NINFER_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
      NINFER_PYTHON_EXECUTABLE="${Python3_EXECUTABLE}")
  endif()
  add_test(NAME ${name} COMMAND ${name} ${arg_ARGS})
endfunction()

function(ninfer_add_op_test name)
  ninfer_add_test(${name} ${ARGN})
  set_tests_properties(${name} PROPERTIES SKIP_RETURN_CODE 77)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${name} PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:-fno-fast-math>
      $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>)
  endif()
endfunction()

function(ninfer_add_linear_test name source)
  ninfer_add_op_test(${name}
    SOURCES ${source}
            ops/linear/linear_test_common.cpp
    LIBRARIES ninfer_ops)
endfunction()
ninfer_add_op_test(ninfer_entropy_cold_requant_test
  SOURCES ops/test_entropy_cold_requant.cpp
  LIBRARIES ninfer_ops)
ninfer_add_op_test(ninfer_cold_i8_test
  SOURCES ops/test_cold_i8.cpp
  LIBRARIES ninfer_ops)
ninfer_add_op_test(ninfer_ple_table_e2e_test
  SOURCES ops/ple_table_e2e_test.cu
  LIBRARIES ninfer_ops ninfer_core)

function(ninfer_add_fused_linear_test name source common)
  ninfer_add_op_test(${name}
    SOURCES ${source} ${common}
    LIBRARIES ninfer_ops)
endfunction()

# Artifact framing and model-independent runtime contracts.
ninfer_add_test(ninfer_artifact_reader_test
  SOURCES test_ninfer_artifact_reader.cpp
  LIBRARIES ninfer_artifact)
ninfer_add_test(ninfer_artifact_materialization_test
  SOURCES test_artifact_materialization.cpp
  LIBRARIES ninfer_artifact)
ninfer_add_test(ninfer_media_decode_test
  SOURCES test_media_decode.cpp
  LIBRARIES ninfer_media_decode)

# Per-architecture capability ladder and the artifact-format capability gate
# (src/core/arch_caps.h). Host-only by construction: the gate takes the compute capability as
# a parameter, so every rung of the ladder -- including the cards this machine does not have --
# is asserted here on CPU rather than by owning one of each GPU. That is the point of the
# parameterized form: "does a V100 reject the nvfp4 artifact" is a table question.
ninfer_add_test(ninfer_arch_caps_test
  SOURCES test_arch_caps.cpp
  LIBRARIES ninfer_artifact
  # NEEDS_SOURCE_DIR since the AMD rungs landed: the per-row file:line citations in
  # src/core/arch_caps.h (kPtxSites / kAmdFormatBlockers / kAmdLdsBlockerSites) are READ BACK
  # OFF THE TREE, so a mark cannot be copied out of an assessment document and the number of
  # marks is not evidence by itself. Without this define the test still passes and prints SKIP
  # -- which is the "a guard that is never consulted" shape this tree keeps closing, hence the
  # define rather than a silent skip.
  NEEDS_SOURCE_DIR)

# MOTION A (b) -- recall1m's test, registered from dl/recall1m/REPORT.md section 7.2.
# Host-only: sum_dir.h and sum_dir_reach.h contain 0 occurrences of cuda (that line's measurement).
ninfer_add_test(ninfer_recall_catalogue_fidelity_test
  SOURCES test_recall_catalogue_fidelity.cpp)

# MOTION A (b) -- E8BAR's test, registered from dl/applier/E8BAR-HANDOFF.md.
# No target_include_directories and no LIBRARIES: ninfer_add_test supplies src/include/third_party
# and links ninfer_core by default, and this file does not include <cuda_runtime.h> (measured).
ninfer_add_test(ninfer_e8_width_contract_test
  SOURCES ${PROJECT_SOURCE_DIR}/src/ops/kv/e8_width_contract_test.cpp)
ninfer_add_test(ninfer_device_test       SOURCES test_device.cpp)
ninfer_add_test(ninfer_device_capabilities_test
  SOURCES test_device_capabilities.cpp
  NEEDS_SOURCE_DIR)
# Multi-device routing and the 64-byte page record's region budget. All three are host-only table
# tests: kernel_route.h takes the compute capability as a PARAMETER (like arch_caps above), so
# "which route does a V100 take for an nvfp4 artifact" is answered on CPU here rather than by
# owning one of each card.
ninfer_add_test(ninfer_kernel_route_test
  SOURCES test_kernel_route.cpp
  LIBRARIES ninfer_core ninfer_artifact)
# The arch-generic fp16 fallback route, the TEST-ONLY architecture simulator, and the
# non-Blackwell fp4 codec -- one binary, because the parts have to be checked against each
# other: the gate's fallback exit and the route arm must agree, and the simulated path must be
# the one the fallback selects.
#
# IT LINKS ninfer_ops FOR A REASON BEYOND THE CALL: the test takes the address of
# ops::qpn::gemm_qpn, so NINFER_HAVE_QPN (src/CMakeLists.txt, published on ninfer_core) and
# the object file that makes it true must arrive together. A definition without the source is
# a LINK error here rather than a route that names a kernel the binary cannot launch.
#
# NINFER_ARCH_SIM_ENABLED=1 IS SET ON THIS TARGET ONLY. The simulator needs a BUILD-time key
# as well as its two environment keys, so a shipping binary cannot be talked into simulating
# anything, and the one binary that can is the one whose job is to test it. The test also
# exercises the not-built-in world directly, through the pure decision function's
# `sim_built_in` parameter, so the refusal path is covered without a second binary.
ninfer_add_test(ninfer_arch_generic_fallback_test
  SOURCES test_arch_generic_fallback.cpp
  LIBRARIES ninfer_artifact ninfer_ops)
target_compile_definitions(ninfer_arch_generic_fallback_test PRIVATE NINFER_ARCH_SIM_ENABLED=1)

# THE VENDOR-AXIS CAPABILITY OVERRIDE (src/core/vendor_sim.h) and the keyed add-only announce set
# it is built on (src/core/announce_once.h). A SEPARATE BINARY from the arch-sim test above, on
# purpose: the two axes have SEPARATE BUILD KEYS, so a binary that can be talked into simulating a
# vendor is not the same binary that can be talked into simulating a V100, and neither key can
# enable the other. Folding them into one target would make the separation untestable.
#
# ⚠ NINFER_VENDOR_SIM_ENABLED=1 IS SET ON THIS TARGET ONLY, and NOTHING IN src/ SETS IT. That is
# not an oversight and it is the deliverable's finding: the vendor axis has NO engine consumer --
# measured, every AMD symbol in src/core/arch_caps.h has ZERO callers outside that header, so
# there is no TU whose behaviour the key could change. src/CMakeLists.txt's arch-sim block states
# the rule this obeys in its own words: the key's only possible effect is on a TU that includes the
# header, and a define on a target that includes nothing is build surface with no effect. The day a
# caller lands, the engine key block lands WITH it -- and the absence of one today is what makes
# "this override changes no shipping behaviour" a build fact rather than a claim.
#
# NEEDS_SOURCE_DIR: GAP 4 of the test is a source read (the `sim_vendor` field must be ABSENT from
# format_probe.h's row schema), with a two-way control so a failed read cannot report a false clean.
ninfer_add_test(ninfer_vendor_sim_test
  SOURCES test_vendor_sim.cpp
  LIBRARIES ninfer_core ninfer_artifact
  NEEDS_SOURCE_DIR)
# ---------------------------------------------------------------------------
# THE WIRE. Root CMakeLists.txt:274 declares option(NINFER_ENABLE_VENDOR_SIM), and until this
# if() landed NO if() ANYWHERE IN THE TREE read it -- a DEAD SWITCH. The cost was not cosmetic:
# the NotBuiltIn refusal (src/core/vendor_sim.h:385-395) names its remedy as
# "-DNINFER_ENABLE_VENDOR_SIM=ON", so a reader who did exactly that got THE SAME REFUSAL. A
# refusal that lies about how to fix it is worse than a plain one: the reader spends time on it.
# This if() is what makes the name in that message true.
#
# WHY HERE AND NOT src/CMakeLists.txt: the arch axis's option gates an engine block there, and
# root CMakeLists.txt:248-252 records the SAME defect on that axis with its landed fix. The
# vendor axis has no engine consumer at all -- measured, every AMD symbol in src/core/arch_caps.h
# has ZERO callers outside that header -- so this test target is the ONLY target the key can
# affect. That is exactly what the option's own help text says it gates.
#
# WHAT THIS CHANGES IN A DEFAULT BUILD: with the option OFF (the default) this target no longer
# carries NINFER_VENDOR_SIM_ENABLED, so the binary reports "vendor sim built in: no". The test
# takes built_in as a PARAMETER, so it runs in both worlds; the default build's result is
# unchanged. What changes is that the option now means something instead of nothing.
# ---------------------------------------------------------------------------
if(NINFER_ENABLE_VENDOR_SIM)
  target_compile_definitions(ninfer_vendor_sim_test PRIVATE NINFER_VENDOR_SIM_ENABLED=1)
endif()

ninfer_add_test(ninfer_shard_plan_test
  SOURCES test_shard_plan.cpp
  LIBRARIES ninfer_core ninfer_artifact)
ninfer_add_test(ninfer_multidev_wiring_test
  SOURCES test_multidev_wiring.cpp
  LIBRARIES ninfer_core ninfer_artifact)
ninfer_add_test(ninfer_decode_graph_test SOURCES test_decode_graph.cpp)
ninfer_add_test(ninfer_tensor_test       SOURCES test_tensor.cpp)
ninfer_add_test(ninfer_arena_test        SOURCES test_arena.cpp)
ninfer_add_test(ninfer_admission_policy_test SOURCES test_admission_policy.cpp)
ninfer_add_test(ninfer_bandwidth_governor_test SOURCES test_bandwidth_governor.cpp)
# The prefill unit has TWO operator-selectable modes (PrefillChunkMode, include/ninfer/types.h):
# dynamic (the governor above owns the unit) and manual (the unit is --prefill-chunk verbatim, the
# governor constructed disabled). This test pins the vocabulary both front ends spell, the "CLI >
# environment > default" precedence in BOTH directions, and the identity property that separates
# manual from "a governor that has not throttled yet" -- each with the negative control that shows
# the check can go red. Header-only on purpose: the mode's resolution lives in the governor header,
# so no device and no artifact are needed to test it.
ninfer_add_test(ninfer_prefill_chunk_mode_test SOURCES test_prefill_chunk_mode.cpp)
# The same mode through the two FRONT ENDS' own parsers: `--prefill-chunk-mode manual` has to land
# in apps/cli's Options and in src/serve's ServeOptions, and an unknown mode has to be refused by
# both. Compiles the front ends' real translation units, like ninfer_cli_options_test above, because
# the step a flag normally dies at is the copy out of the parser.
ninfer_add_test(ninfer_prefill_chunk_mode_cli_test
  SOURCES test_prefill_chunk_mode_cli.cpp
          ${PROJECT_SOURCE_DIR}/apps/cli/options.cpp
          ${PROJECT_SOURCE_DIR}/src/serve/serve_options.cpp)
target_include_directories(ninfer_prefill_chunk_mode_cli_test PRIVATE
  ${PROJECT_SOURCE_DIR}/apps/cli
  ${PROJECT_SOURCE_DIR}/src)
# W6 wiring: the governor class above is dead code in the engine unless a target's Program exposes
# set_prefill_chunk()/prefill_chunk_capacity(), and engine_core.h installs the shrunken unit behind
# a compile-time guard. This test asserts that guard's own named predicate for each target package
# and odr-uses both members, so deleting either forwarder is a build/link failure instead of the
# silent compile-out the mechanism started as. It links ninfer_engine for the same reason it
# includes the target packages: the definitions live behind the PIMPL facade there.
ninfer_add_test(ninfer_prefill_chunk_wiring_test
  SOURCES test_prefill_chunk_wiring.cpp
  LIBRARIES ninfer_engine ninfer_core)
target_include_directories(ninfer_prefill_chunk_wiring_test PRIVATE
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/export
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6_27b/export
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6_35b_a3b/export
  ${PROJECT_SOURCE_DIR}/src/targets/muse_glimmer_30b/export)
ninfer_add_test(ninfer_context_cost_test SOURCES test_context_cost.cpp)
ninfer_add_test(ninfer_kv_perlayer_policy_test SOURCES test_kv_perlayer_policy.cpp)
ninfer_add_test(ninfer_resource_manager_test SOURCES test_resource_manager.cpp)
ninfer_add_test(ninfer_weight_residency_test SOURCES test_weight_residency.cpp)
ninfer_add_test(ninfer_kv_capacity_test SOURCES test_kv_capacity.cpp)
ninfer_add_test(ninfer_sampling_defaults_test
  SOURCES test_sampling_defaults.cpp
  LIBRARIES ninfer_engine ninfer_core)
target_include_directories(ninfer_sampling_defaults_test PRIVATE
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/export
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6_27b/export
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6_35b_a3b/export)
ninfer_add_test(ninfer_perplexity_evaluation_test
  SOURCES test_perplexity_evaluation.cpp
          ${PROJECT_SOURCE_DIR}/apps/perplexity/evaluation.cpp)
target_include_directories(ninfer_perplexity_evaluation_test PRIVATE
  ${PROJECT_SOURCE_DIR}/apps/perplexity)
ninfer_add_test(ninfer_cli_options_test
  SOURCES test_cli_options.cpp ${PROJECT_SOURCE_DIR}/apps/cli/options.cpp)
target_include_directories(ninfer_cli_options_test PRIVATE ${PROJECT_SOURCE_DIR}/apps/cli)
# --kv-tier-formats landing (product/kv_tier_formats.h) and the vocabulary's own
# CPU twin that had no build target at all: both are header-only host tests.
ninfer_add_test(ninfer_kv_tier_formats_test
  SOURCES test_kv_tier_formats.cpp)
# The model's own directory over recall blocks (src/spec/sum_dir.h): host-only and
# std-only with no CUDA in the include graph, so it is a header-only host test like the
# two above. It pins its 64-token block granularity by value against
# core/paged_kv_cache.h's kPagedKVPageSize (that header needs <cuda_runtime_api.h>, so
# it is cited, not included).
ninfer_add_test(ninfer_sum_dir_test
  SOURCES test_sum_dir.cpp)
# The recall UNIT's identity and the overlap arithmetic over it
# (src/spec/recall_identity.h), and the syntax boundary a recall has to land on
# (src/spec/stop_boundary.h). Both are std-only host headers, so both are header-only host
# tests like the one above. The identity test also includes src/spec/sum_dir.h and
# product/kv_recall_block.h (both std-only) because the point of that header is to collapse
# the tree's three digest spellings to one scheme plus one named legacy spelling, and that
# claim is only worth anything if it is checked against the landed files themselves.
ninfer_add_test(ninfer_recall_identity_test
  SOURCES test_recall_identity.cpp)
ninfer_add_test(ninfer_stop_boundary_test
  SOURCES test_stop_boundary.cpp)

# src/spec is a header-only directory with no CMake target of its own (the same shape as
# src/kvcfg above), and its two host test programs were referenced by nothing in this tree:
# no target compiled them, no ctest could run them, and the test they were believed to be
# therefore never executed once. They are plain int-main host programs -- std-only headers
# plus ops/suffix_lookup.h, a declaration-only header -- so they register through
# ninfer_add_test exactly like the header-only host tests above. Deliberately WITHOUT
# SKIP_RETURN_CODE: neither file has a 77 path, and a skip property with no skip behind it is
# the "state with no check behind it" this project has already paid for.
ninfer_add_test(ninfer_lookup_fuse_test
  SOURCES ${PROJECT_SOURCE_DIR}/src/spec/lookup_fuse_test.cpp)
ninfer_add_test(ninfer_turn_recall_journal_test
  SOURCES ${PROJECT_SOURCE_DIR}/src/spec/turn_recall_journal_test.cpp)
# The two FNV-1a 64 conventions this tree uses (spec/fnv_convention.h owns both):
# the header's static_asserts are the compile-time half of the pin, this test is the
# runtime half -- golden values per convention, kDigestOffset/kDigestPrime in the
# file that owns the engine convention (read by text, deliberately not included), and
# a walk of src/+tests/+tools/ proving no third convention can appear unnoticed.
# NEEDS_SOURCE_DIR: it reads the tree, so it needs NINFER_SOURCE_DIR.
ninfer_add_test(ninfer_fnv_convention_test
  SOURCES test_fnv_convention.cpp
  NEEDS_SOURCE_DIR)
ninfer_add_test(ninfer_kvcfg_formats_test
  SOURCES ${PROJECT_SOURCE_DIR}/src/kvcfg/kv_formats_test.cpp)
# Cold-tier byte budgets (product/kv_cold_tier_budget.h): the admission ladder
# over --cold-host-bytes / --cold-disk-bytes is pure integer arithmetic, so it
# is a header-only host test like the two above.
ninfer_add_test(ninfer_kv_cold_tier_budget_test
  SOURCES test_kv_cold_tier_budget.cpp)
# RECALLSPAN: the admission test for the paging pre-allocation wiring. product/
# kv_paging_preallocation.h landed a correct contract with ZERO non-test consumers, so its
# refusal could not fire on anything; the wiring now lives at layouts_impl.h:1808+ and is
# opt-in (NINFER_KV_PAGING_PREALLOC). This test pins the two properties that wiring's
# soundness rests on: the refusal on D1's own argv, and its invariance over the
# byte-per-cold-page -- the one input the pool-sizing site cannot supply. Pure std, no CUDA,
# so it is a header-only host test like the cold-tier-budget one above.
ninfer_add_test(ninfer_kv_paging_admission_test
  SOURCES test_kv_paging_admission.cpp)
# R1 recall: the block-level KV ownership encoding (product/kv_recall_block.h) is
# pure std code, so it is a header-only host test like the three above. It needs the
# source tree because its central claim -- that its unit IS the engine's Paged-KV
# page -- is pinned against the authoritative text rather than by including
# core/paged_kv_cache.h (which would pull cuda_runtime_api.h into a plain-g++ test).
ninfer_add_test(ninfer_kv_recall_block_test
  SOURCES test_kv_recall_block.cpp
  NEEDS_SOURCE_DIR)
# N3 runtime loop for the KV row-scale table (product/kv_rowscale_persist.h):
# the one-shot/skip/--recalibrate decision, the applicability gates and the
# frame -> words solve are all host code, so this is a header-only host test with
# the two device readbacks stubbed (the loader's own parser gates the output).
ninfer_add_test(ninfer_kv_rowscale_persist_test
  SOURCES test_kv_rowscale_persist.cpp)
# The three KV component switches (--kv-rotation / --kv-row-scale / --kv-v-codec)
# are each honoured by a SUBSET of the KV tiers, so a switch set outside its
# domain used to be accepted and change nothing. product/kv_component_switch.h
# is the single host-testable predicate the planner now refuses with, so this is
# a header-only host test like the ones above.
ninfer_add_test(ninfer_kv_component_switch_test
  SOURCES test_kv_component_switch.cpp)
# The K/V bit-width entries and the score-table entry point (product/kv_kv_bits.h):
# two entries that must not disagree (joint = ONE overall ceiling; split = each
# plane's per-layer layering solved in its own budget), one realizability oracle,
# and the penalty table's own runnable producer. Host-only, no device, so this is a
# header-only host test like the two above -- and it carries its negative controls,
# because an oracle pinned only on its positive cases cannot tell a working one from
# one that always says yes.
ninfer_add_test(ninfer_kv_kv_bits_test
  SOURCES test_kv_kv_bits.cpp)
# BUDGETSAT: the CEILING IS A SATURATING CONSTRAINT. Measured at the engine
# (--kv-bits 4.5 -> achieved=4.38 with all-nvfp4 at exactly 4.50 available) and at the
# solver: the selection rule minimised penalty among the states at or below the ceiling,
# so a ceiling was a one-sided constraint the objective was free to undershoot. The oracle
# here is an EXHAUSTIVE ENUMERATION of the reachable totals, not a second DP, and the
# sweep asserts saturation, monotonicity, no-overshoot and never-below-the-default; it
# carries its own negative control (the ladder must really leave gaps, or the assertions
# would be vacuous) and a refusal control (below the floor the solver must REFUSE, never
# silently return a worse tier). Header-only host test, no device.
ninfer_add_test(ninfer_kv_budget_saturation_test
  SOURCES test_kv_budget_saturation.cpp)
# BUDGETSAT: the slider's INSPECTION entry, and the score-table/ladder size invariant it
# shares with product/kv_bit_budget.h. `ninfer --kv-score-table show` segfaulted (rc=139)
# on its SUCCESS path -- two loops were bounded by the LADDER's size (8) while indexing
# the SCORE TABLE (6 rows) -- and every earlier test only ever reached the ERROR path, which
# is why it survived. This one reaches the success path, asserts the rendered table names
# exactly the six scoreable rows (the old code emitted eight), asserts a MUTATED input table
# is rendered as mutated (so a renderer that ignored its input cannot pass), and round-trips
# the text back through the real loader.
ninfer_add_test(ninfer_kv_score_table_entry_test
  SOURCES test_kv_score_table_entry.cpp)

# BITSLIDER: the SLIDER selects among EQUAL-BIT mixes. The rule the axis exists for is
# "the bit count is an INPUT and the knob picks among codecs that cost the SAME bits", so
# the property under test is not reachability (iso4e shares nvfp4's 450-bit plane geometry
# exactly) but the TIE-BREAK: the solver returned ONE spec at 4.50 for every weight and
# every score table, while 17 codec-different mixes all cost exactly 4.5000. This test
# enumerates that equal-bit set, asserts the default answer is unchanged (`0-15:nvfp4`,
# the same pin tests/test_kv_budget_saturation.cpp holds), asserts that a caller CAN select
# the codec-different alternative and that BOTH report the SAME achieved bits, and carries
# the negative controls that keep the knob from being a penalty override (the shipped
# ladder's iso4e pin still wins) and from accepting a gear outside the candidate grammar.
ninfer_add_test(ninfer_kv_bit_mix_menu_test
  SOURCES test_kv_bit_mix_menu.cpp)
ninfer_add_test(ninfer_kv_cache_test
  SOURCES test_kv_cache.cpp
  LIBRARIES ninfer_core)
ninfer_add_test(ninfer_state_store_test
  SOURCES test_state_store.cpp
  LIBRARIES ninfer_core)
ninfer_add_test(ninfer_gdn_replay_records_test
  SOURCES test_gdn_replay_records.cpp
  LIBRARIES ninfer_core)
set_tests_properties(
  ninfer_artifact_materialization_test
  ninfer_device_test
  ninfer_decode_graph_test
  ninfer_arena_test
  ninfer_kv_cache_test
  ninfer_state_store_test
  PROPERTIES SKIP_RETURN_CODE 77)
# Family frontend and one opt-in real-artifact prefix/MTP integration test.
ninfer_add_test(ninfer_qwen3_6_frontend_test
  SOURCES targets/qwen3_6/test_frontend.cpp
  NEEDS_SOURCE_DIR
  LIBRARIES ninfer_engine ninfer_core)
target_include_directories(ninfer_qwen3_6_frontend_test PRIVATE
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/export
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/impl)
# The official Qwen3.6-27B resource checks inside this test are opt-in through
# NINFER_FRONTEND_TEST_ROOT; without it the test reports 77 instead of failing.
set_tests_properties(ninfer_qwen3_6_frontend_test PROPERTIES SKIP_RETURN_CODE 77)
ninfer_add_test(ninfer_qwen3_6_runtime_mechanisms_test
  SOURCES targets/qwen3_6/test_runtime_mechanisms.cpp
  LIBRARIES ninfer_engine ninfer_core)
target_include_directories(ninfer_qwen3_6_runtime_mechanisms_test PRIVATE
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/export)

# Host-only: the PRODUCING half of the --draft-tree L,d (L > 1) publication. It builds
# the verify-column layout from the proposal side's candidate lattice with the REAL
# ddtree reference builder (and, for the sibling shape, against ddtree::column_depth /
# column_ancestors), then publishes it through the receiving seam. No engine, no CUDA,
# no variant macro.
ninfer_add_test(ninfer_qwen3_6_mtp_tree_produce_test
  SOURCES targets/qwen3_6/test_mtp_tree_produce.cpp)
target_include_directories(ninfer_qwen3_6_mtp_tree_produce_test PRIVATE
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/export)

# Host-only: the RECEIVING half of the --draft-tree L,d (L > 1) publication -- layout
# validation and the sequence seam. It uses the real ddtree reference builder, no
# engine/CUDA/variant header.
ninfer_add_test(ninfer_qwen3_6_mtp_tree_publish_test
  SOURCES targets/qwen3_6/test_mtp_tree_publish.cpp)
target_include_directories(ninfer_qwen3_6_mtp_tree_publish_test PRIVATE
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/export)
# Host-only: the PROPOSAL-SIDE FILL of the --draft-tree L,d (L > 1) publication -- the
# egress lattice read, the spine cross-check that makes "the lattice was published"
# checkable, the planning-time shortlist-head refusal, and the fill into the receiving
# seam. It compiles with no engine, no CUDA and no variant macro.
ninfer_add_test(ninfer_qwen3_6_mtp_tree_proposal_test
  SOURCES targets/qwen3_6/test_mtp_tree_proposal.cpp)
target_include_directories(ninfer_qwen3_6_mtp_tree_proposal_test PRIVATE
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/export)
ninfer_add_test(ninfer_qwen3_6_state_image_test
  SOURCES targets/qwen3_6/test_state_image.cpp
  LIBRARIES ninfer_engine ninfer_core)
target_include_directories(ninfer_qwen3_6_state_image_test PRIVATE
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/export)
set_tests_properties(ninfer_qwen3_6_state_image_test PROPERTIES SKIP_RETURN_CODE 77)
ninfer_add_test(ninfer_qwen3_6_state_image_layout_test
  SOURCES targets/qwen3_6/test_state_image_layout.cpp
  LIBRARIES ninfer_engine ninfer_core)
target_include_directories(ninfer_qwen3_6_state_image_layout_test PRIVATE
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/export)
ninfer_add_test(ninfer_qwen3_6_context_store_test
  SOURCES targets/qwen3_6/test_context_store.cpp
  LIBRARIES ninfer_engine ninfer_core)
target_include_directories(ninfer_qwen3_6_context_store_test PRIVATE
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/export)
set_tests_properties(ninfer_qwen3_6_context_store_test PROPERTIES SKIP_RETURN_CODE 77)
ninfer_add_test(ninfer_qwen3_6_27b_prefix_real_test
  SOURCES targets/qwen3_6_27b/test_engine_prefix_real.cpp
  LIBRARIES ninfer_engine)
set_tests_properties(ninfer_qwen3_6_27b_prefix_real_test PROPERTIES SKIP_RETURN_CODE 77)
ninfer_add_test(ninfer_qwen3_6_27b_score_real_test
  SOURCES targets/qwen3_6_27b/test_engine_score_real.cpp
  LIBRARIES ninfer_engine)
set_tests_properties(ninfer_qwen3_6_27b_score_real_test PROPERTIES SKIP_RETURN_CODE 77)
ninfer_add_test(ninfer_qwen3_6_27b_load_plan_test
  SOURCES targets/qwen3_6_27b/test_load_plan.cpp
  NEEDS_SOURCE_DIR
  LIBRARIES ninfer_engine)
target_include_directories(ninfer_qwen3_6_27b_load_plan_test PRIVATE
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/export
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6_27b/export)
target_link_libraries(ninfer_qwen3_6_27b_load_plan_test PRIVATE CUDA::cudart)
set_tests_properties(ninfer_qwen3_6_27b_load_plan_test PROPERTIES SKIP_RETURN_CODE 77)
ninfer_add_test(ninfer_qwen3_6_35b_a3b_real_test
  SOURCES targets/qwen3_6_35b_a3b/test_engine_real.cpp
  LIBRARIES ninfer_engine)
set_tests_properties(ninfer_qwen3_6_35b_a3b_real_test PROPERTIES SKIP_RETURN_CODE 77)
ninfer_add_test(ninfer_qwen3_6_35b_a3b_dflash_real_test
  SOURCES targets/qwen3_6_35b_a3b/test_engine_dflash_real.cpp
  LIBRARIES ninfer_engine)
set_tests_properties(ninfer_qwen3_6_35b_a3b_dflash_real_test PROPERTIES SKIP_RETURN_CODE 77)
ninfer_add_test(ninfer_qwen3_6_35b_a3b_dflash_load_plan_test
  SOURCES targets/qwen3_6_35b_a3b/test_dflash_load_plan.cpp
  NEEDS_SOURCE_DIR
  LIBRARIES ninfer_engine)
target_include_directories(ninfer_qwen3_6_35b_a3b_dflash_load_plan_test PRIVATE
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/export
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6_35b_a3b/export)
target_link_libraries(ninfer_qwen3_6_35b_a3b_dflash_load_plan_test PRIVATE CUDA::cudart)
set_tests_properties(ninfer_qwen3_6_35b_a3b_dflash_load_plan_test
  PROPERTIES SKIP_RETURN_CODE 77)

# Product protocol contracts.
ninfer_add_test(ninfer_openai_schema_test
  SOURCES test_openai_schema.cpp
  LIBRARIES ninfer_serve)
ninfer_add_test(ninfer_openai_responses_test
  SOURCES test_openai_responses.cpp
  LIBRARIES ninfer_serve)
ninfer_add_test(ninfer_openai_responses_store_test
  SOURCES test_openai_responses_store.cpp
  LIBRARIES ninfer_serve)
ninfer_add_test(ninfer_anthropic_schema_test
  SOURCES test_anthropic_schema.cpp
  LIBRARIES ninfer_serve)
ninfer_add_test(ninfer_tool_call_parser_test
  SOURCES test_tool_call_parser.cpp
  LIBRARIES ninfer_engine)
ninfer_add_test(ninfer_serve_options_test
  SOURCES test_serve_options.cpp
  LIBRARIES ninfer_serve)
ninfer_add_test(ninfer_request_log_test
  SOURCES test_request_log.cpp
  LIBRARIES ninfer_serve)
ninfer_add_test(ninfer_host_timing_test
  SOURCES test_host_timing.cpp
  LIBRARIES ninfer_core)
ninfer_add_test(ninfer_http_error_handler_test
  SOURCES test_http_error_handler.cpp
  LIBRARIES ninfer_serve)
target_include_directories(ninfer_http_error_handler_test PRIVATE
  ${PROJECT_SOURCE_DIR}/third_party/cpp-httplib)
ninfer_add_test(ninfer_http_transport_test
  SOURCES test_http_transport.cpp
  LIBRARIES ninfer_serve)
target_include_directories(ninfer_http_transport_test PRIVATE
  ${PROJECT_SOURCE_DIR}/third_party/cpp-httplib)
ninfer_add_test(ninfer_bench_support_test
  SOURCES test_ninfer_bench_support.cpp
          ${PROJECT_SOURCE_DIR}/bench/targets/qwen3_6_27b/ninfer_bench_support.cpp
  NEEDS_SOURCE_DIR
  LIBRARIES ninfer_engine)
target_include_directories(ninfer_bench_support_test PRIVATE ${PROJECT_SOURCE_DIR}/bench/targets/qwen3_6_27b)

# Pure host-side qualification of the fixed-dimension non-negative fit and the exact coefficient
# quantization used by the context-cost calibrator. It deliberately links no Engine/CUDA target.
add_executable(ninfer_context_cost_measure_test
  test_context_cost_measure.cpp
  ${PROJECT_SOURCE_DIR}/bench/context_cost/context_cost_measure.cpp)
target_include_directories(ninfer_context_cost_measure_test PRIVATE
  ${PROJECT_SOURCE_DIR}/bench/context_cost)
add_test(NAME ninfer_context_cost_measure_test COMMAND ninfer_context_cost_measure_test)

# Shared Op numerical tests. Target-composed schedule tests are registered separately below.
set(ninfer_op_tests
  add_bias
  gelu
  gelu_mul
  silu_mul
  residual_add
  sigmoid_mul
  logit_policy
  rmsnorm
  gated_rmsnorm
  l2norm
  gated_delta_net
  causal_conv1d_silu
  layer_norm
  embedding
  argmax
  gdn_gating
  gdn_gating_proj
  rope
  vision_pos_embed
  sampling
  scalar
  cast
  scatter
  target_logprobs
  position)
foreach(op IN LISTS ninfer_op_tests)
  ninfer_add_op_test(ninfer_${op}_test
    SOURCES ops/test_${op}.cpp
    LIBRARIES ninfer_ops)
endforeach()

# Stateful, fused, and exact-shape Ops remain central even when the current implementation domain
# is one checkpoint/device.
ninfer_add_op_test(ninfer_softmax_attention_test
  SOURCES ops/softmax_attention/main.cpp
          ops/softmax_attention/causal_cache.cpp
          ops/softmax_attention/plain_and_packed.cpp
          ops/softmax_attention/context.cpp
  LIBRARIES ninfer_ops)
ninfer_add_op_test(ninfer_sliding_window_attention_test
  SOURCES ops/test_sliding_window_attention.cpp
  LIBRARIES ninfer_ops)
ninfer_add_op_test(ninfer_kv_cache_append_test
  SOURCES ops/test_kv_cache_append.cpp
  LIBRARIES ninfer_ops)
ninfer_add_op_test(ninfer_prepare_masked_block_test
  SOURCES ops/test_prepare_masked_block.cpp
  LIBRARIES ninfer_ops)
ninfer_add_op_test(ninfer_sparse_moe_test
  SOURCES ops/test_sparse_moe.cpp
  LIBRARIES ninfer_ops)
ninfer_add_op_test(ninfer_ple_layout_test
  SOURCES test_ple_layout.cpp
  LIBRARIES ninfer_ops
  NEEDS_SOURCE_DIR)
ninfer_add_op_test(ninfer_mtp_pack_test
  SOURCES ops/test_mtp_pack.cpp
  LIBRARIES ninfer_ops)
ninfer_add_op_test(ninfer_dspark_markov_argmax_test
  SOURCES ops/test_dspark_markov_argmax.cpp
  LIBRARIES ninfer_ops)
ninfer_add_op_test(ninfer_mtp_round_test
  SOURCES ops/test_mtp_round.cpp
  LIBRARIES ninfer_ops)
ninfer_add_op_test(ninfer_speculative_round_test
  SOURCES ops/test_speculative_round.cpp
  LIBRARIES ninfer_ops)
ninfer_add_linear_test(ninfer_linear_q4_a16_test ops/linear/test_q4_a16.cpp)
ninfer_add_linear_test(ninfer_linear_q5_a16_test ops/linear/test_q5_a16.cpp)
ninfer_add_linear_test(ninfer_linear_q6_a16_test ops/linear/test_q6_a16.cpp)
ninfer_add_linear_test(ninfer_linear_w8_a16_test ops/linear/test_w8_a16.cpp)
ninfer_add_linear_test(ninfer_linear_nvfp4_a16_test ops/linear/test_nvfp4_a16.cpp)
ninfer_add_linear_test(ninfer_linear_nvfp4_a4_test ops/linear/test_nvfp4_a4.cpp)
ninfer_add_linear_test(ninfer_linear_fp8_a16_test ops/linear/test_fp8_a16.cpp)
ninfer_add_linear_test(ninfer_linear_fp8_a8_test ops/linear/test_fp8_a8.cpp)
ninfer_add_op_test(ninfer_linear_bf16_a16_test
  SOURCES ops/linear/test_bf16_a16.cpp
  LIBRARIES ninfer_ops)
ninfer_add_op_test(ninfer_attn_input_proj_test
  SOURCES ops/test_attn_input_proj.cpp
  LIBRARIES ninfer_ops)
ninfer_add_op_test(ninfer_gdn_input_proj_test
  SOURCES ops/test_gdn_input_proj.cpp
  LIBRARIES ninfer_ops)
ninfer_add_op_test(ninfer_gdn_input_proj_conv_snapshot_test
  SOURCES ops/test_gdn_input_proj_conv_snapshot.cpp
  LIBRARIES ninfer_ops)
ninfer_add_op_test(ninfer_gdn_input_proj_conv_record_test
  SOURCES ops/test_gdn_input_proj_conv_record.cpp
  LIBRARIES ninfer_ops)
ninfer_add_op_test(ninfer_gated_delta_net_replay_record_test
  SOURCES ops/test_gated_delta_net_replay_record.cpp
  LIBRARIES ninfer_ops)
ninfer_add_op_test(ninfer_gdn_replay_fold_test
  SOURCES ops/test_gdn_replay_fold.cpp
  LIBRARIES ninfer_ops)
ninfer_add_fused_linear_test(
  ninfer_linear_add_q5_a16_test
  ops/linear_add/test_q5_a16.cpp
  ops/linear_add/linear_add_test_common.cpp)
ninfer_add_fused_linear_test(
  ninfer_linear_add_bf16_a16_test
  ops/linear_add/test_bf16_a16.cpp
  ops/linear_add/linear_add_test_common.cpp)
ninfer_add_fused_linear_test(
  ninfer_linear_add_w8_a16_test
  ops/linear_add/test_w8_a16.cpp
  ops/linear_add/linear_add_test_common.cpp)
ninfer_add_op_test(ninfer_linear_add_nvfp4_test
  SOURCES ops/linear_add/test_nvfp4.cpp
  LIBRARIES ninfer_ops)
ninfer_add_op_test(ninfer_linear_add_fp8_test
  SOURCES ops/linear_add/test_fp8.cpp
  LIBRARIES ninfer_ops)
ninfer_add_fused_linear_test(
  ninfer_linear_pair_w8_a16_test
  ops/linear_pair/test_w8_a16.cpp
  ops/linear_pair/linear_pair_test_common.cpp)
ninfer_add_fused_linear_test(
  ninfer_linear_swiglu_q4_a16_test
  ops/linear_swiglu/test_q4_a16.cpp
  ops/linear_swiglu/linear_swiglu_test_common.cpp)
ninfer_add_fused_linear_test(
  ninfer_linear_swiglu_w8_a16_test
  ops/linear_swiglu/test_w8_a16.cpp
  ops/linear_swiglu/linear_swiglu_test_common.cpp)
ninfer_add_fused_linear_test(
  ninfer_linear_swiglu_nvfp4_test
  ops/linear_swiglu/test_nvfp4.cpp
  ops/linear_swiglu/linear_swiglu_test_common.cpp)
ninfer_add_fused_linear_test(
  ninfer_linear_swiglu_fp8_test
  ops/linear_swiglu/test_fp8.cpp
  ops/linear_swiglu/linear_swiglu_test_common.cpp)

# The generic (unregistered-shape) decoder's LAYOUT contract. Two measured defects live in
# src/ops/generic/rowsplit_generic.cu:
#   (1) both operands were indexed for a token-inner layout this engine does not have --
#       x[j*t+col0] and out[row*t+col0+c] -- whereas the REGISTERED q4 kernels read
#       x[(col0+col)*k+j] and out[(col0+col)*out_ld+row], out_ld = out.nb[1]/sizeof(bf16).
#       The same defect sat in generic_gating_kernel, whose own store used the engine
#       convention while its read of h did not.
#   (2) generic_swiglu_split_dispatch passed an element COUNT into the kernel's `total_rows`,
#       whose guard multiplies by `columns`, so the guard admitted rows*columns*columns
#       indices: dead at every engine MLP (rows is a multiple of 256 there, so
#       blocks*threads == total exactly) and out of bounds on a 30-element destination at 6x5.
# The test carries the registered dispatcher as a POSITIVE CONTROL and asserts the registered
# and generic routes agree element-wise on a shape both serve, so a revert to the token-inner
# index cannot pass. It needs a GPU: without one it returns 77 and ctest reports SKIP.
ninfer_add_op_test(ninfer_generic_rowsplit_layout_test
  SOURCES ops/generic/test_rowsplit_generic_layout.cpp
  LIBRARIES ninfer_ops)

ninfer_add_test(ninfer_qwen3_6_27b_visual_scatter_test
  SOURCES targets/qwen3_6_27b/test_visual_scatter.cpp
  LIBRARIES ninfer_engine ninfer_core)
set_tests_properties(ninfer_qwen3_6_27b_visual_scatter_test PROPERTIES SKIP_RETURN_CODE 77)

# These tests deliberately compile against the exact target's internal contracts; the target does
# not export impl/ to ordinary consumers.
set(ninfer_qwen3_6_27b_internal_tests
  ninfer_qwen3_6_27b_visual_scatter_test)
foreach(test_target IN LISTS ninfer_qwen3_6_27b_internal_tests)
  target_include_directories(${test_target} PRIVATE
    ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6_27b/impl
    ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6_27b/export
    ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/export)
endforeach()

# Host-only DDTree layout test: pure standard C++ against the S17/S19 reference
# builder's golden vectors, no engine/CUDA/artifact header. See
# include/ninfer/ops/dflash2_ddtree.h for why it exists and what it is NOT.
ninfer_add_op_test(ninfer_dflash2_ddtree_test
  SOURCES ops/test_dflash2_ddtree.cpp
  LIBRARIES ninfer_ops)

# Host-only cross-check for the DEVICE half of the same layout contract: the
# fixed-capacity beam core used by src/ops/launcher/dflash2_tree_walk.cu must agree
# node-for-node with the slice-1 reference builder above. No CUDA, no engine, no
# golden file -- both sides are compiled into this one test on purpose.
ninfer_add_op_test(ninfer_dflash2_ddtree_beam_test
  SOURCES ops/test_dflash2_ddtree_beam.cpp
  LIBRARIES ninfer_ops)

# NGRAM-BUILD: the n-gram (PLE) seam composition and the table's carrier layer. Both are
# pure HOST code, so neither needs a GPU, a sidecar, or a fixture -- which is the point:
#   * test_ple_seam_window.cpp -- the wiring that joins PleNgramWindow (the predecessor
#     producer, which nothing called) with the PLE gather seam, plus the controls that go
#     red if the window is removed again. The CUDA half enters the composition as a
#     template parameter and is instantiated against the real op by the .cu siblings.
#   * test_ple_carrier.cpp -- the memory/SSD carrier decision ladder and its budget
#     arithmetic (header-only, no upstream library needed).
# Neither registers SKIP_RETURN_CODE: nothing here depends on an opt-in artifact, so a
# skip could only ever mean "green without running".
ninfer_add_test(ninfer_ple_seam_window_test
  SOURCES test_ple_seam_window.cpp
  LIBRARIES ninfer_engine ninfer_core)
ninfer_add_test(ninfer_ple_carrier_test
  SOURCES test_ple_carrier.cpp)
# The rank axis itself (core/shard_rank_axis.h) and the virtual-device guard
# (core/virtual_device.h). Both are host-only and take every input as a PARAMETER, so the three
# arithmetic structures that had no rank dimension, and the guard's whole refusal space, are
# exercised on a host with no GPU -- which is the only form of evidence this box can produce for
# tensor parallelism (one physical GPU, no MIG, no MPS).
ninfer_add_test(ninfer_shard_rank_axis_test
  SOURCES test_shard_rank_axis.cpp
  LIBRARIES ninfer_core ninfer_artifact)
ninfer_add_test(ninfer_virtual_device_test
  SOURCES test_virtual_device.cpp
  LIBRARIES ninfer_core ninfer_artifact)
# The collective seam (core/tp_transport.h, core/decode_graph_peer.h) and the N-dim column
# slicer (core/n_dim_slice.h). All three are host-only and take every input as a PARAMETER, so
# they compile and run with plain g++ -- no CUDA header, no device. What that proves is the
# interface, the ordering and the refusals; the transport's COST is not this component's number.
ninfer_add_test(ninfer_tp_transport_test
  SOURCES test_tp_transport.cpp)
ninfer_add_test(ninfer_n_dim_slice_test
  SOURCES test_n_dim_slice.cpp)
# The AND-reduce (core/and_reduce.h) -- the ONE reduction the plan asks for and NO reference on
# this box supplies: the donor has no boolean op and 1Cat's allreduce dispatches float/half/
# bf16 only. Host-only, every input a parameter, so it runs with no GPU.
ninfer_add_test(ninfer_and_reduce_test
  SOURCES test_and_reduce.cpp)

# THE FIX FOR THE MEASURED RED (core/tp_transport.h sections 3b/3c/4b): the slotted protocol with a
# DEVICE-OBSERVED acknowledgement, plus the exhaustive interleaving scan that shows the old event pair
# leaves the in-place combine unordered against the peer's read in 30140 of 48620 merges while both
# device-observed mechanisms leave it ordered in every one. Host-only: pure functions of their inputs,
# no CUDA header and no device, so it keeps building and running while the engine's device link does
# not.
ninfer_add_test(ninfer_tp_slotted_ack_test
  SOURCES test_tp_slotted_ack.cpp)

# tools/mg4/tp2_sum_stress.cu -- the DEVICE half of the same work: the stress reproducer (N iterations,
# the a+2b signature counted, the old path against both mechanisms of the fix) and the capture
# questions. It is deliberately an EXECUTABLE AND NOT A CTEST TEST: it takes the GPU, and every GPU run
# in this project goes through the fair lock in sh/_lock.sh, because a ctest test that grabbed the
# device without the lock would contend with whichever model chain holds it and contaminate that
# chain's own measurement. Its invocation path is therefore sh4/_mg4_run*.sh, which sources the lock,
# and that is where its numbers come from. It is registered here so the tree can BUILD the evidence.
add_executable(ninfer_tp2_sum_stress ${PROJECT_SOURCE_DIR}/tools/mg4/tp2_sum_stress.cu)
target_include_directories(ninfer_tp2_sum_stress PRIVATE
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/third_party
  ${PROJECT_SOURCE_DIR}/third_party/utf8proc)
target_link_libraries(ninfer_tp2_sum_stress PRIVATE CUDA::cudart CUDA::cuda_driver)

# ---------------------------------------------------------------------------
# SIMFIX2 -- the three tests that guard the simulator's verdicts, REGISTERED so that a normal
# `ctest` actually runs them. Before this, all three existed and nothing invoked them, which is
# the same defect they were written to close: a guard that is never consulted is decoration.
#
#   tools/archkit/test_probe_provenance.sh  a simulated arm may not enter the list through the RAW
#                                           (the stale `# probe_sim=` append path)
#   tools/gui/test_gpu_compat.py            a capability row must be MEASURED, not stamped with a
#                                           constant `ok: True` and rendered as a green tick
#   tests/test_sim_no_support.cpp           (SIMPROBE) a simulated verdict may not enter the list
#
# NONE OF THE THREE TAKES THE GPU. That is what makes them safe to register here, and it is the
# doctrine stated just above for ninfer_tp2_sum_stress: a ctest test that grabbed the device
# without sh/_lock.sh would contend with whichever model chain holds it. These are host-only --
# the driver test takes no lock (--derive-only acquires none, and its run-branch checks assert
# that the refusals precede the lock), and the other two are pure host code.
#
# A SCRIPT TEST HAS NO BUILD TARGET, so it is a direct COMMAND and not the
# `add_test(NAME <name> COMMAND <name>)` form the binaries above use. That idiom was verified in a
# scratch project before this patch was written (`cmake` + `ctest -N` lists it, `ctest -R` runs it).
#
# SKIP_RETURN_CODE 77 is the tree's own convention (set for the op tests above) and is attached to
# the GPU-compat test so that a hazard probe can report SKIP rather than reddening the suite.
# ---------------------------------------------------------------------------
find_program(NINFER_BASH_FOR_TESTS NAMES bash HINTS /bin /usr/bin REQUIRED)

add_test(NAME ninfer_simfix_probe_provenance_test
         COMMAND ${NINFER_BASH_FOR_TESTS} ${PROJECT_SOURCE_DIR}/tools/archkit/test_probe_provenance.sh)
set_tests_properties(ninfer_simfix_probe_provenance_test PROPERTIES SKIP_RETURN_CODE 77)

add_test(NAME ninfer_simfix_gpu_compat_test
         COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/tools/gui/test_gpu_compat.py)
set_tests_properties(ninfer_simfix_gpu_compat_test PROPERTIES SKIP_RETURN_CODE 77)

ninfer_add_test(ninfer_sim_no_support_test
  NEEDS_SOURCE_DIR
  SOURCES test_sim_no_support.cpp
  LIBRARIES ninfer_artifact ninfer_core)

# The OPERATOR-visible KV tier name against the CANONICAL one, per tier: the printed
# "kv cache dtype" line is the line an operator reads to confirm which tier actually
# ran, and apps/cli/main.cpp used to print a drifted vocabulary while the engine ran
# another. Four independent checks -- literal pin, derivation, totality (a NEW
# enumerator fails here instead of passing unpinned), and a text census over the CLI
# sources that catches a re-introduced second copy. NEEDS_SOURCE_DIR because the
# census reads those sources as text. Host-only, no CUDA, no device, no model.
ninfer_add_test(ninfer_kv_operator_name_test
  NEEDS_SOURCE_DIR
  SOURCES test_kv_operator_name.cpp)


# ---------------------------------------------------------------------------
# ROUTE D: the test-only architecture simulator must not reach the engine the GUI spawns.
#
# tools/gui/serve_gui.py built its child's environment with `dict(os.environ, ...)`, so a
# NINFER_SIM_ARCH exported in the shell that launched the GUI was inherited by the engine it
# started -- under a pane still showing the REAL card's nvidia-smi badge. A simulated run was
# therefore indistinguishable from a real one to the operator, which is the one thing this project
# forbids: a build configuration is not a support claim, and only a probe on real hardware is
# evidence of support. The same file's log pane also kept only a trailing window, so the SIMULATED
# banner (printed ONCE per process, at the first arch-view call) was evicted after a few kB.
#
# Both are closed, and this is the half that FAILS when either reopens. It drives the GUI's REAL
# POST /api/start with the simulator pair exported (subprocess.Popen is replaced, so no engine is
# ever spawned), then drives the real log pinning under ~60 kB of output, then asserts as an AST
# PROPERTY that every `env=` keyword passed to a child comes from engine_env(). The AST form is
# deliberate: the first draft of that check grepped, and matched its own explanatory comment --
# the same defect SIMFIX2 recorded one layer down.
#
# Deliberately WITHOUT SKIP_RETURN_CODE: it has no 77 path, and a skip property with no skip behind
# it is the exact defect this file exists to avoid.
# Host-only: no CUDA, no device, no model, no nvidia-smi, no lock, no engine process.
add_test(NAME ninfer_serve_sim_route_test
         COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/tools/gui/test_serve_sim_route.py)

# ---------------------------------------------------------------------------
# E8 WIDTH CODEC: the 55-check suite for the 3-bit / 2-bit K-plane variants
# (product/kv_e8_width.h + product/kv_e8_width_codec.h).
#
# This program was written and running but had NO CMake target and was UNTRACKED, so nothing in
# the tree could reach it: not built, not run by ctest and -- because it was untracked rather
# than merely unregistered -- one `git clean` from being deleted rather than merely skipped. An
# unregistered test does not run; an untracked one can vanish. It is load-bearing rather than
# historical because AMAXFIX moved the 4-bit scale divisor amax/7 -> amax/8 and inverted this
# suite's section 7.8 from a divergence pin to a convergence pin against the shipped row: it is
# the runtime half of the pin whose compile-time half is the static_asserts in
# ops/kernel/gqa_attention_kv_quant.cuh.
#
# SOURCE PATH: the file lives in src/ops/kv/ and is referenced IN PLACE, exactly as
# ninfer_lookup_fuse_test / ninfer_turn_recall_journal_test / ninfer_kvcfg_formats_test above
# reference src/. It is deliberately NOT copied into tests/: a copy would leave the original
# untracked and would itself become the second thing to drift.
#
# SIDE EFFECTS, checked before registering, because registering a test makes it run in EVERYONE's
# ctest: pure host std C++ -- no CUDA, no device, no model, no lock, no engine process, no
# subprocess, and no argv (int main() takes none). Its ONLY filesystem access is two std::fopen
# calls in READ-ONLY modes ("rb" and "r"), guarded by the optional E8V_REAL_K env var; it opens
# and closes a file and never creates, truncates, removes, renames or chmods one, and with
# E8V_REAL_K unset it touches no file at all.
#
# CHECK COUNT, REBUILT 2026-09-18. This comment used to say "Exit code is 0 iff all 55 checks
# pass". The 55 are the SCALAR codec's, and every one of them still runs and still compares
# against the same literal -- the pre-image of this test file reports 55/55 and 0 FAIL. What
# changed is that section 8 was APPENDED for the real E8 lattice as the W3/W2 codec of record
# (src/ops/kv/e8_lattice_plane_codec.cuh), so the target now reports 75/75. Stating both numbers
# here because a count in a comment that no longer matches the binary is the second thing to
# drift, and the reason the 55 are not renumbered is that section 7.6 pins the scalar writer's
# STORED BYTES at W3 and W2 and those bytes are the lattice's control arm.
#
# THE LATTICE SECTION NEEDS ONE EXTRA INCLUDE DIR, and it is the only build change this carried.
# src/ops/kv/e8_lattice_plane_codec.cuh reaches the real Conway-Sloane projection in
# src/ops/kernel/e8_lattice.cuh, which includes <cuda_runtime.h>; the point of the lattice
# section is that it compiles the REAL headers rather than a paraphrase, so the shim
# tools/e8_verify/host_shim -- the same one tools/e8_verify/run_lattice_codec.sh has used since
# that file was written -- is put on this target's include path. Still no CUDA, no device, no
# lock: the shim is 3 lines of <cmath>/<cstdint>.
#
# Deliberately WITHOUT SKIP_RETURN_CODE: it has no 77 path, and a skip property with no skip
# behind it is the "state with no check behind it" this file has already paid for.
ninfer_add_test(ninfer_e8_width_codec_test
  SOURCES ${PROJECT_SOURCE_DIR}/src/ops/kv/e8_width_codec_test.cpp)
target_include_directories(ninfer_e8_width_codec_test PRIVATE
  ${PROJECT_SOURCE_DIR}/tools/e8_verify/host_shim)


# THE COLD-HOST WINDOW (targets/qwen3_6/impl/runtime/cold_host_tier.h). The band
# max(0, F-K-H) <= W <= D-K, the feasibility condition F <= D+H, the DERIVED --cold-host-bytes
# value, and the per-round cost -- all host-only: every input is a PARAMETER, so the three
# quantities that were prose become numbers with red controls on a box with no free device.
# It reads cold_host_tier.h, which reaches core/paged_kv_cache.h, so it links the same pair the
# shard-rank-axis test above does.
ninfer_add_test(ninfer_cold_host_window_test
  SOURCES test_cold_host_window.cpp
  LIBRARIES ninfer_core ninfer_artifact)

# THE TWO COLD PAGE UNITS, PINNED -- AND THE PIN THE WINDOW TEST ABOVE CANNOT CARRY.
# tests/test_cold_host_window.cpp:79 names the COLD RECORD `kPageBytes`, and its own comment
# at :67 says in words that 1,181,696 B is "the engine's MEASURED cold-slot page record".
# Nothing in the tree read the HOST page unit beside it, and it could not have:
# core/host_kv_arena.h and product/kv_tier_formats.h have DISJOINT includer sets, and
# layouts_impl.h:2066-2067 states that both are invisible at the paging-plan site. This test
# is that convergence point, in test space: plan_host_kv_page_layout() on a DECLARED geometry
# beside the cuda-free product/kv_tier_formats.h record.
#
# CARD-FREE, so it is registerable in the same shape as the window test above and for the
# same reason: every input is a PARAMETER (a declared geometry and two product constants), so
# there is no decoder, no device and no ninfer_engine. What it CANNOT reach is the live pair
# -- turn_recall_page_bytes() is a member of ProgramImplCore and needs a planned decoder --
# and that half is covered at runtime by the constructor check in program_impl.h instead
# (dl/coldagree/shadow/.../program_impl.h, held unlanded). See dl/coldagree/REPORT.md TASK A.
ninfer_add_test(ninfer_cold_page_units_test
  SOURCES test_cold_page_units.cpp
  LIBRARIES ninfer_core ninfer_artifact)


# [FALLBACKBYTES] WHAT A COLD-TIER FALLBACK DOES TO THE DEVICE BYTE BUDGET. The claim under
# test is "a page that falls back costs no bytes, so the fallback rate is a hit-rate (capacity)
# term, never a bytes/component term". The true contract has four clauses, and they are NOT the
# ones that sentence states: (1) the cold slot pool is ONE fixed `{stride, kv_heads, 2,
# max_cold_pages}` device tensor per layer, sized at PLAN time from --max-cold-pages and its
# derivation -- a fallback cannot shrink it, and nothing in its sizing path can see how many
# pages were delivered; (2) `release_cold_slot` returns the SLOT IDENTITY, one byte in a host
# bitmap; (3) it therefore reaches no arena, no deallocator and no size field -- which is what
# makes it byte-free, NOT the page's 9216 B plane (product::kKvColdResidentNvfp4Bytes), which is
# the RESIDENT plane living in a different structure; (4) and that is the ONLY place it is
# byte-free, because the single byte-moving call in the pass is the success path's
# `transfer_to_cold`, which every abandonment path precedes -- so the fallback rate IS a byte
# term on the RESIDENT arena, in the direction of LOSS.
#
# Clauses (1), (2) and (4) are properties of decoder_state.cpp / program_impl.h, which cannot be
# LINKED here (the engine's device link is broken at this revision: the half-landed
# iso3->iso4e / e8->rk4v4 rename leaves ops::EntropyColdRequantMode without
# Iso4eVG16/Rk4v4KvG64). They are therefore asserted against those SOURCES AS TEXT -- the shape
# already in this file at ninfer_fnv_convention_test and ops/test_gqa_decode_split_exact.cu --
# with a NEGATIVE CONTROL on a synthetic broken copy behind every text check, so the test cannot
# pass vacuously. The byte ARITHMETIC is EXECUTED, from the cuda-free product/kv_tier_formats.h
# and product/kv_paging_preallocation.h plus the real cold_fallback_census.h.
#
# NO ninfer library and NO CUDA, deliberately and for the same reason ninfer_public_api_test
# above is built that way: this target must keep building and running while the engine's device
# link does not. It is NOT a ninfer_add_test() call because that helper links ninfer_core by
# default, which would make a contract test depend on the very link it exists to survive.
add_executable(ninfer_cold_slot_release_bytes_test test_cold_slot_release_bytes.cpp)
target_include_directories(ninfer_cold_slot_release_bytes_test PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/src)
target_compile_definitions(ninfer_cold_slot_release_bytes_test PRIVATE
  NINFER_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
add_test(NAME ninfer_cold_slot_release_bytes_test COMMAND ninfer_cold_slot_release_bytes_test)

# ===========================================================================
# [INEXACTGATE] THE RECALL PATH'S REFUSALS, MADE AUDITABLE
# ===========================================================================
# Five sections, one per measured finding (dl/holegate/REPORT.md): the INEXACT word is an
# OBSERVER computed after the restore; a plan cut EMPTY prints nothing while a plan cut SHORT
# prints INEXACT (the louder signal on the milder case); the second gate (`net_positive()`)
# compares two constants so `refused_cost=0` proves no check ran; `pages=R/P` is ANTI-correlated
# with exactness because its denominator is the already-cut plan; and the 227-page budget edge
# TRUNCATES the anchor away instead of refusing as sum_dir_reach.h:257/:773 requires.
#
# NO ninfer library and NO CUDA, deliberately -- the same shape and the same reason as
# ninfer_cold_slot_release_bytes_test above: the plan/cost/line half is EXECUTED against
# src/spec/turn_recall_journal.h (which is std-only by its own declaration, :90-91), and the
# ENGINE half is asserted against program_impl.h AS TEXT, with a negative control behind every
# text check, because it cannot be linked here.
#
# It deliberately does NOT replace src/spec/turn_recall_journal_test.cpp (registered as
# ninfer_turn_recall_journal_test above): that test pins the PLAN's arithmetic, this one pins the
# REFUSALS and the counters the fleet reads out of a run.
add_executable(ninfer_turn_recall_inexact_gate_test test_turn_recall_inexact_gate.cpp)
target_include_directories(ninfer_turn_recall_inexact_gate_test PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/src)
target_compile_definitions(ninfer_turn_recall_inexact_gate_test PRIVATE
  NINFER_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
add_test(NAME ninfer_turn_recall_inexact_gate_test COMMAND ninfer_turn_recall_inexact_gate_test)

# ===========================================================================
# [PREFILLBUDGET] THE BOUNDED PREFILL BUDGET'S EDGE, MADE AUDITABLE
# ===========================================================================
# dl/vectorkey/REPORT.md ROW 2 proposes a token-denominated prefill budget and states its edge in
# one word: a NAMED REFUSAL, "refusing not truncating -- never a partial injection" (:44), firing
# "at PLAN time, on the plan's own `needed` count, before a single token is re-prefilled" (:287-288).
#
# NO ninfer library and NO CUDA, deliberately -- the same shape and the same reason as
# ninfer_cold_slot_release_bytes_test and ninfer_turn_recall_inexact_gate_test above: the PLAN half
# is EXECUTED against src/spec/turn_recall_journal.h (std-only by its own declaration), and the
# ENGINE half is asserted against program_impl.h AS TEXT with a negative control behind every text
# check, because it cannot be linked here.
#
# It deliberately does NOT replace ninfer_turn_recall_journal_test (the plan's arithmetic) or
# ninfer_turn_recall_inexact_gate_test (the refusals): this one pins the TOKEN BUDGET's own axis --
# that it refuses rather than truncates, that a zero budget refuses nothing, and that the refusal is
# what the specification's string says it is.
add_executable(ninfer_recall_prefill_budget_test test_recall_prefill_budget.cpp)
target_include_directories(ninfer_recall_prefill_budget_test PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/src)
target_compile_definitions(ninfer_recall_prefill_budget_test PRIVATE
  NINFER_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
add_test(NAME ninfer_recall_prefill_budget_test COMMAND ninfer_recall_prefill_budget_test)


# ===========================================================================
# [RECALLCHEAP] THE RESTORE-FIRST PREDICATE. A bare `add_executable`: NO ninfer library and
# NO CUDA, deliberately -- the same shape and the same reason as
# `ninfer_cold_slot_release_bytes_test` (:1032) and `ninfer_recall_prefill_budget_test`
# (:1086) above. It includes ONLY `src/spec/turn_recall_journal.h`, which is host-only and
# std-only, and it EXECUTES the predicate over its whole truth table.
# The MUTANT / SPECIFIC arms are NOT built here: they are a host-only control device under
# `dl/recallcheap/control/` that compiles THIS SAME source against a mutated header image with
# `-I` precedence, exactly as `dl/fusionrefresh/control/` and `dl/prefillland/s/probe.sh` do.
# A control that lives in the build tree would be the "test that can only pass" shape.
# ===========================================================================
add_executable(ninfer_recall_restore_first_test test_recall_restore_first.cpp)
target_include_directories(ninfer_recall_restore_first_test PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/src)
target_compile_definitions(ninfer_recall_restore_first_test PRIVATE
  NINFER_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
add_test(NAME ninfer_recall_restore_first_test COMMAND ninfer_recall_restore_first_test)


# ===========================================================================
# THE 1M-CONTEXT ROPE FREQUENCY TABLES -- the FIRST fork-survey island that is
# INTEGRATED, not merely landed.
#
# What was landed on 2026-09-18 by the additive merge: src/models/qwen3_5/program/
# rope_scaling.h (cometkim/ninfer @ feat/1m-context, the YaRN table builder) and
# tests/models/qwen3_5/test_rope_scaling.cpp. What was NOT landed: the vocabulary they
# consume -- ops::RopeFrequencies, ops::rope_linear_frequencies, ops::rope_vision_frequencies
# -- which lived in the fork's include/ninfer/ops/rope.h, a file in the "modifies an existing
# tree file" bucket that the merge deliberately refused. So the borrow could not compile:
# MEASURED RED, 15 g++ errors, every one of them rooted in
#   "'RopeFrequencies' in namespace 'ninfer::ops' does not name a type"
# The missing half is landed now (include/ninfer/ops/rope.h, +67 insertions / 0 deletions).
#
# Why this island first: the owner's first branch requirement is "the index must reach 1M
# context -- use the index to break past the model's context ceiling". This builder is that
# knob: it parameterises the table by factor / original_positions / beta_fast / beta_slow,
# where the engine's shipped ops::rope_yarn4 is frozen at factor 4, theta 1e7, 262144. The
# fork's own header names the mechanism -- "position scaling past the checkpoint's trained
# range ... the legacy FP32 product loses ~0.03 rad on the lowest pairs at 1M positions".
#
# Both targets are HOST-ONLY C++ and deliberately link NO ninfer library, only CUDA::cudart
# for <cuda_runtime.h>'s cudaStream_t typedef. That is the same property
# ninfer_public_api_test and the contract targets above require, and it is load-bearing here:
# the engine's device link does not currently resolve, so a target that linked ninfer_core
# would not build and this island could not be shown to be live at all.
add_executable(ninfer_rope_scaling_test models/qwen3_5/test_rope_scaling.cpp)
target_include_directories(ninfer_rope_scaling_test PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/third_party
  ${PROJECT_SOURCE_DIR}/third_party/utf8proc)
target_link_libraries(ninfer_rope_scaling_test PRIVATE CUDA::cudart)
add_test(NAME ninfer_rope_scaling_test COMMAND ninfer_rope_scaling_test)

# The pin that the borrow AGREES WITH THIS TREE instead of being a parallel universe: it
# reproduces the SHIPPED kernel tables (src/ops/kernel/rope.cuh's kTextRopeInvFrequency,
# kDflashRopeInvFrequency, kVisionRopeInvFrequency, kTextRopeYarn4InvFrequency and
# kYarn4AttentionScaling), read from the header AS TEXT because they are __device__
# __constant__ arrays no host TU can link against -- the same technique the split-geometry
# contract target above already uses.
add_executable(ninfer_rope_frequencies_kernel_agreement_test
  models/qwen3_5/test_rope_frequencies_kernel_agreement.cpp)
target_include_directories(ninfer_rope_frequencies_kernel_agreement_test PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/third_party
  ${PROJECT_SOURCE_DIR}/third_party/utf8proc)
target_compile_definitions(ninfer_rope_frequencies_kernel_agreement_test PRIVATE
  NINFER_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
target_link_libraries(ninfer_rope_frequencies_kernel_agreement_test PRIVATE CUDA::cudart)
add_test(NAME ninfer_rope_frequencies_kernel_agreement_test
         COMMAND ninfer_rope_frequencies_kernel_agreement_test)
# ===========================================================================
# ISLAND #2 -- src/ops/kv_cache/append/geometry.cuh, the 16-line junction.
#
# Six borrows across two islands include "ops/kv_cache/append/geometry.cuh"
# (kv_cache/append/{k8v4,nvfp4,rk2v4e8,hq}_kernel.cuh and
# causal_cache/{prompt_hq,small_t_hq}.cuh) and the file did not exist.
# MEASURED RED under the project's own device oracle
# (nvcc -O3 -DNDEBUG -std=c++20 -c -rdc=true --generate-code=arch=compute_120a,...):
# the FIRST error of 6/6 consumers was
#     fatal error: ops/kv_cache/append/geometry.cuh: No such file or directory
#
# The tree already carries the same four declarations INLINED at
# src/ops/kv_cache/append/kernel.cuh:17-26 (tracked, present at HEAD); the fork's
# kernel.cuh:9 #includes them from geometry.cuh instead. So the borrow is the fork's own
# extraction of a block this tree kept inlined, and the risk of landing it is exactly that
# the tree ends up with two definitions of one vocabulary. The pin below makes that
# impossible to be true silently: it asserts a THREE-WAY byte agreement between the
# expected literal block, the landed header's block, and kernel.cuh's block -- measured
# 348 bytes, sha256 prefix ed61f2b8843a779c on all three sides -- and it fails if the live
# append dispatcher (launch.cu) stops naming KVCacheAppendD256Kv4 / KVCacheAppendD256Kv2,
# so the header cannot drift into supplying a vocabulary nothing selects.
#
# Host-only C++20, links no ninfer library: the engine's device link does not currently
# resolve (ops::EntropyColdRequantMode), and a target that needed it could not be built,
# so this island could not be shown to be live at all. Same property, and the same
# reasoning, as the rope-scaling targets above.
add_executable(ninfer_kv_cache_append_geometry_pin_test
  ops/test_kv_cache_append_geometry_pin.cpp)
target_include_directories(ninfer_kv_cache_append_geometry_pin_test PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/src)
target_compile_definitions(ninfer_kv_cache_append_geometry_pin_test PRIVATE
  NINFER_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
add_test(NAME ninfer_kv_cache_append_geometry_pin_test
         COMMAND ninfer_kv_cache_append_geometry_pin_test)


# ---------------------------------------------------------------------------
# The cold_i8 public surface must keep ONE spelling from the declaration in
# include/ninfer/ops/cold_i8.h to the definition in src/ops/wrapper/cold_i8.cpp. The e8 -> rk4v4
# rename half-landed there for the THIRD time on 2026-09-18: the header declared
# cold_i8_slot_restore_rk4v4_raw, four ProgramImplCore::restore_cold_page instantiations inside
# variant.cpp called it, and the wrapper still DEFINED cold_i8_slot_restore_e8_raw -- so the engine
# link died at 100% with an undefined reference to a name no file spelled. Nothing could see it
# earlier because in C++ a definition whose spelling misses the declaration is not an error: it
# silently introduces a new function, and only the last link objects.
# The test takes the ADDRESS of every public entry point -- a real odr-use, so a one-sided move is
# a link error inside the test binary, one build before the engine link -- and it reads this tree
# AS TEXT, which is the only one of its checks that needs no build at all: the configuration the
# link died in (build-sim) ships BUILD_TESTING=OFF, so a guard that only linked would be dormant
# exactly where it was needed. NEEDS_SOURCE_DIR because the census reads the sources as text.
ninfer_add_op_test(ninfer_cold_i8_symbol_pin_test
  NEEDS_SOURCE_DIR
  SOURCES ops/test_cold_i8_symbol_pin.cpp
  LIBRARIES ninfer_ops)

# ---------------------------------------------------------------------------
# A REGISTRATION IS NOT COVERAGE.  `ctest -N` counts every add_test NAME in its total, so a
# registered test whose binary was never built reports `***Not Run` while still inflating the
# count that a reader takes for coverage. MEASURED 2026-09-18: 165 registered, 11 registered
# with no binary, and none of the 11 visible in the total -- two of them are real defects
# (one fails 5 checks on the sm_70 rung, one cannot link: undefined
# ops::cold_i8_slot_restore_rk4v4_raw), so the tests that would have said so are the ones that
# are not running.
# tools/check_link_health.sh CHECK 3 now resolves each registration to the executable it
# actually RUNS (a registered name is not a target name: ninfer_gqa_split_geometry_probe_legacy
# and _exact are two configurations of one binary, and a script test's artefact is a source
# file), and FAILS on a registered-but-unbuilt test, naming each one with the object-presence
# discriminator: objects present == a link ran and failed, no objects == never built. Before
# this it reported five findings, ALL of them false positives, and filed the eleven real rows
# as `info`.
# It is registered HERE so that it is consulted -- the SIMFIX2 note above states the rule this
# file applies to itself: "a guard that is never consulted is decoration".
# Read-only: no CUDA, no device, no model, no nvidia-smi, no lock, no engine process, no build.
# `--allow-unbuilt` accepts a mid-flight build directory; the count and every name are still
# printed, so accepting it is a visible act rather than a silent omission.
# No SKIP_RETURN_CODE on purpose: the script has no 77 path, and a skip property with no skip
# behind it is the exact defect this file exists to avoid (see ninfer_serve_sim_route_test).
add_test(NAME ninfer_registration_coverage_test
         COMMAND ${NINFER_BASH_FOR_TESTS} ${PROJECT_SOURCE_DIR}/tools/check_link_health.sh)

# ---------------------------------------------------------------------------
# [RUNTHEM] THE SEVEN FILES THAT NO CMakeLists IN THIS TREE MENTIONED.
#
# MEASURED 2026-09-19 (dl/falsegreen/REPORT.md section 1 item 7; re-measured by dl/runthem in
# logs/s00_preflight.out): a depth-bounded walk finds 14 CMakeLists.txt in the whole tree (the
# deepest ones at depth 4, printed as proof), and NOT ONE of them contains any of these seven
# names -- so none of the seven was ever compiled by this tree's build, and none was ever run by
# its ctest. One of them, tests/test_sum_dir_vector.cpp, is UNTRACKED as well as unregistered:
# one `git clean` removes it rather than merely skipping it. An unregistered test does not run;
# an untracked one can vanish.
#
# FIVE OF THE SEVEN REGISTER HERE; TWO COULD NOT, AND THE REASON IS THE COMPILER, NOT POLICY:
#   tests/test_tensor_slice.cpp      'SliceRange' and 'TensorSlice' have not been declared in
#       'ninfer::artifact'. src/artifact/reader.h declares neither type and
#       tensor_row_slice/tensor_column_slice appear nowhere in src/ (READ 2026-09-19, s06 log) --
#       the fork-survey borrow landed the caller without the API half.
#   tests/test_kv_capacity_tp2.cpp   'resolve_kv_capacity_symmetric' is not a member of
#       'ninfer::runtime': src/runtime/engine/kv_capacity.h (403 B, READ 2026-09-19) declares ONLY
#       resolve_kv_capacity.
#   Both verdicts come from a REAL COMPILE of the file with the exact flag shape these
#   registrations give it (g++-13 -std=c++20 -O2 -DNDEBUG, includes tests/ include/ src/
#   third_party/ + the CUDA include root), not from -fsyntax-only: logs/s07_trial_compile.out and
#   logs/s08_details.out. Registering either would be a registered test whose binary can NEVER be
#   built -- the "A REGISTRATION IS NOT COVERAGE" defect named at the bottom of this file, and the
#   exact shape tools/check_link_health.sh CHECK 3 exists to catch.
#
# WHAT THE FIVE DO WHEN RUN. All five were compiled AND linked AND run before this block was
# written; the reds are REPORTED here rather than fixed, because a red test is a finding about
# the tree and turning it green is not this registration's decision:
#   ninfer_sum_dir_vector_test            RED    4 failure(s), rc=1, stdout 0 B (stderr 234 B)
#   ninfer_sum_dir_reach_test             RED    checks=38 failures=6782, rc=1, VERDICT: FAIL
#   ninfer_format_probe_test              green  121 checks, 0 failure(s)
#   ninfer_kv_paging_preallocation_test   green
#   ninfer_kv_storage_coverage_test       green  45/45
# ---------------------------------------------------------------------------

# Host-only, std-only: src/spec/sum_dir_vector.h has 0 occurrences of "cuda" (MEASURED) and its
# only include outside <...> is spec/sum_dir.h, so this registers exactly like its sibling
# ninfer_sum_dir_test above. The test's own header says "no CUDA, no artifact, no fixture, plain
# g++", and its subject is the vector retriever's wire format, ranking and refusals.
# MEASURED RED (54 checks, 4 failures) in both the hand-built trial and the CMake build; the
# failures are reported, not repaired. See logs/s08_details.out for the four FAIL lines.
ninfer_add_test(ninfer_sum_dir_vector_test
  SOURCES test_sum_dir_vector.cpp)

# Host-only as well: spec/sum_dir.h and spec/sum_dir_reach.h are std-only (0 occurrences of
# "cuda" in either, MEASURED) and src/spec has no CMake target of its own -- the same shape
# ninfer_lookup_fuse_test / ninfer_turn_recall_journal_test are registered in above, except that
# this file lives in tests/ rather than in src/spec/.
#
# IT TAKES AN ARGUMENT AND ctest GIVES IT NONE. int main(int argc, char** argv) reads
# [cargo] [term-file] [mode] and, with no argv, falls back to the corpus path hardcoded in its own
# main() -- /mnt/c/Users/User/Documents/ziqinzhang/dl/recallspan/d1/ninfer_text.cargo (READ
# 2026-09-19: 68128 B, present). That default is another line's record directory, so the honest
# statement of what this registration runs is "the corpus that path names today". If that file is
# absent the test prints "VERDICT: NOT MEASURED (no corpus)" and returns 2, which ctest reports as
# a failure: that is the test's own design (rc=2 means NOT MEASURED) and it is deliberately NOT
# given SKIP_RETURN_CODE 77, because it has no 77 path and a skip property with no skip behind it
# is the "state with no check behind it" this file has already paid for.
# MEASURED RED on that corpus: checks=38 failures=6782, rc=1. Reported, not repaired.
ninfer_add_test(ninfer_sum_dir_reach_test
  SOURCES test_sum_dir_reach.cpp)

# The contract of src/core/format_probe.h. LINKS ninfer_artifact FOR ONE SYMBOL, and the reason is
# measured rather than assumed: format_probe.h calls artifact::format_name() at :683, :1094,
# :1280, :1344, :1439 and :1474, and that symbol is DEFINED only in artifact/storage_layouts.cpp
# (:67) -- src/CMakeLists.txt:469-516 states the dependency in its own words. The registration
# note at the bottom of test_format_probe.cpp says this target "links no ninfer library"; that
# sentence is contradicted by the file's own header comment, which puts libninfer_artifact.a and
# libninfer_core.a on the hand-build line for exactly that symbol. The link verdict is what
# settles it, and this target is registered with the library that carries the definition.
# NEEDS_SOURCE_DIR: T1 checks the criterion constants against the file they are QUOTED from, so
# the test reads the tree; without the define it cannot make that check.
ninfer_add_test(ninfer_format_probe_test
  SOURCES test_format_probe.cpp
  LIBRARIES ninfer_artifact
  NEEDS_SOURCE_DIR)

# product/kv_paging_preallocation.h is std-only (its include list carries no <cuda...> and no
# ninfer/types.h), so this is a header-only host test like ninfer_kv_paging_admission_test above --
# which pins the same header's ADMISSION half. This file is the WIRING half's unit test: the
# pool-sizing expression it replaces, the working-set size, the refusal when a page has neither a
# replica nor a tier, and the spill-capacity contract. NO ninfer library beyond ninfer_add_test's
# default ninfer_core, and no NEEDS_SOURCE_DIR: the file reads no source (0 occurrences of
# NINFER_SOURCE_DIR, MEASURED).
ninfer_add_test(ninfer_kv_paging_preallocation_test
  SOURCES test_kv_paging_preallocation.cpp)

# A PORTABLE replacement for -Wswitch over KvCacheStorage: include/ninfer/types.h declares 15
# enumerators and four switching sites (device_capabilities.h x2, kv_storage_dtype.h,
# layouts_impl.h) each name only ten of them. Host-only: 0 occurrences of "cuda" in the test file
# itself, and it takes every input as a parameter, so it registers like
# ninfer_device_capabilities_test above -- including the default ninfer_core link, which is what
# puts the CUDA include root on the line for core/device_capabilities.h.
ninfer_add_test(ninfer_kv_storage_coverage_test
  SOURCES test_kv_storage_coverage.cpp)

# ---------------------------------------------------------------------------
# THE ENGINE'S SHARD CONTRACT (src/core/shard_plan.h section 6).
#
# Section 6 is the missing half of the multi-device work: sections 1-4 are the shard
# ARITHMETIC and are correct for any world_size, while section 6 is the ENGINE's answer to a
# different question -- is there anything in this build that can BIND a second device, MOVE a
# byte between two devices, or READ a per-rank weight stream. All four pieces are absent in
# this revision, by measurement, and a declared world is now REFUSED BY NAME where a machine
# with two cards used to get a silent single-card run.
#
# HOST-ONLY, and the default ninfer_core link is enough: neither core/shard_plan.h nor
# core/virtual_device.h contains a CUDA header or a device query (their own header comments
# say so), so this is a header-only host test in the shape of ninfer_and_reduce_test above.
# It takes no lock, spawns nothing, and reads no file -- it is registered for EVERYONE's ctest.
ninfer_add_test(ninfer_shard_contract_test
  SOURCES test_shard_contract.cpp)
# `ARGS` IS LOAD-BEARING HERE AND IS NOT DECORATION (added 2026-09-22 by line `ctest12`).
# The arm is selected by argv inside the test itself (tests/test_shard_contract.cpp:486-489):
# `--multidev-required` runs the CAPABILITY assertion (exits 1 in this revision), while no
# argv runs the seven default G1..G7 arms (exit 0). Registered with no argv, `WILL_FAIL`
# TRUE inverted the DEFAULT arm's 0 into a FAIL, so the entry was red the moment it was
# built and said nothing about the capability. The argv cannot be dropped: without it the
# `WILL_FAIL` latch inverts the wrong arm. See dl/_orch/landq/ctest12/README.md.
ninfer_add_test(ninfer_shard_contract_multidev_pending_test
  SOURCES test_shard_contract.cpp
  ARGS --multidev-required)

# THE SECOND REGISTRATION IS THE ONE THAT CARRIES THE PENDING CAPABILITY, and it is a
# DELIBERATE INVERSION. The binary is the same; the arm is selected by argv.
#
#   `--multidev-required` asserts the CAPABILITY: "this build can carry a world of two ranks
#   across two physical devices". That is FALSE today, so the PROGRAM EXITS 1 and prints the
#   refusal verbatim. WILL_FAIL TRUE is what turns that red program into a green ctest entry,
#   because a genuinely red test in a shared suite would make every other line's `ctest` run
#   fail -- which is not what "this capability is not landed yet" means.
#
#   `WILL_FAIL` IS THIS TREE'S FIRST USE OF THE PROPERTY, and it is a ONE-WAY LATCH on the
#   absence rather than a way of hiding a failure. The day the four pieces land, the program
#   exits 0, ctest inverts THAT into a FAIL, and the pin retires itself loudly: the contract
#   text, `kCrossDeviceFactsThisRevision`, G4's block and this registration must then move in
#   the same commit. A pin that could not go red in either direction would be decoration.
# THE REGEX IS KEYED ON THE CAPABILITY CLAIM, NOT ON THE RED LINE, AND THAT IS THE WHOLE
# LATCH. ctest applies PASS_REGULAR_EXPRESSION to the output and THEN applies WILL_FAIL as an
# inversion of THAT verdict, so the verdict is `NOT(regex_found)`. MEASURED 2026-09-22 in
# dl/ctest12/logs/s14_countercontrol.txt + s15_latchfix.txt (ctest -V, six arms, two real
# binaries): with the RED line as the regex the entry is ***Failed "Required regular
# expression found" TODAY and PASSES the day the capability lands -- the exact inverse of a
# one-way latch. Keyed on the capability claim the polarity is right: today the claim is
# absent, the regex "fails", WILL_FAIL inverts that, and the entry is GREEN while the program
# is RED and prints why; the day the four pieces land the claim appears, the regex "passes",
# WILL_FAIL inverts THAT, and the entry goes ***Failed loudly, which is the signal to retire
# this second registration and the arm. Neither line may be removed to "simplify" it.
set_tests_properties(ninfer_shard_contract_multidev_pending_test PROPERTIES
  WILL_FAIL TRUE
  PASS_REGULAR_EXPRESSION "capability: PRESENT.")

# ---------------------------------------------------------------------------
# [TESTHYG] THE ORPHAN SOURCES, TRIAGED BY COMPILING EACH ONE. Added 2026-09-22 by line `testhyg`.
#
# A walk of tests/ found 200 sources and 25 of them absent from build/compile_commands.json, so no
# target compiled them and no ctest could run them
# (dl/testhyg/logs/r08_orphans_fast.txt). Each of the 25 was then trial-compiled OUTSIDE the tree
# with the EXACT flag shape ninfer_add_test gives its targets (g++-13 -O3 -DNDEBUG -std=gnu++20 plus
# tests/ include/ src/ third_party/ third_party/utf8proc + the CUDA include root, and the
# -fno-fast-math -ffp-contract=off pair ninfer_add_op_test adds), so that each verdict below is a
# COMPILE RESULT rather than an inference about the file's name:
#   dl/testhyg/logs/r13_trial_attach.txt, r14_trial2.txt, r15_trial_rest.txt
#
#   COMPILE AND LINK, AND ARE REGISTERED HERE (5):
#     tests/test_recall_restore_units.cpp            host-only; reads no tree, returns 0 or 1.
#     tests/ops/test_deferred_lookup_instantiation.cu  A COMPILE INSTRUMENT, and its own header says
#         so: "The fix for that class is to FORCE the instantiation, which is what this TU does".
#         A TU that is never compiled cannot force anything, so its inert state was the whole defect.
#         NEEDS LIBRARIES ninfer_ops for the same reason its .cu siblings carry it.
#     tests/ops/test_gdn_input_proj_conv_restart.cpp  the RESTART arm of the fused
#         projection/conv/snapshot Op. Its two siblings (_snapshot, _record) were registered above;
#         this one was not, so the restart equivalence was never run by anything. A device gate was
#         added to its main() in the same change, so the skip is the repository's 77.
#     tests/ops/ple_seam_gather_e2e_test.cu           the CUDA half of the ngram seam. It carries
#         this tree's three-state sidecar convention INCLUDING the 77 path, and it holds the
#         load-bearing control that window-derived predecessors must gather DIFFERENT rows than an
#         eos fill would -- so SKIP_RETURN_CODE 77 is correct here and not a dead property.
#     tests/ops/test_e8_root_codec.cu                 drives the shipped src/ops/kernel/
#         e8_root_codec.cuh directly (tools/test_kv/ reimplements the E8 maths and never includes
#         that header, so it cannot catch a regression in the code the fill path compiles). Its
#         main() already returns 77 with no device.
#
#   COMPILES **AND LINKS** AND IS DELIBERATELY NOT REGISTERED (1), because it is RED here:
#     tests/test_recall_codec_blind.cpp -- see the WITHDRAWN note at the bottom of this block. The
#       verbatim failure is `FAIL: PD the sequence-end census carries the count, appended` with
#       `checks_failed=1` and `VERDICT=FAIL_WRONG`, rc=3, no device required.
#
#   EVERY REGISTRATION ABOVE WAS TRIAL-LINKED AND TRIAL-RUN BEFORE IT WAS WRITTEN, and the run rc is
#   recorded per target in dl/testhyg/logs/f03_newtargets_link.txt: run_rc must be 0, 1 or 77 --
#   a negative rc (a signal) or 124 (a timeout) means the source must not be registered. This is the
#   check whose absence put ninfer_ple_seam_gather_e2e_test into `make all` with the wrong
#   LIBRARIES for part of an hour: compiling a TU is not linking it, and linking it is not running it.
#
#   DO NOT COMPILE, SO THEY CANNOT BE ATTACHED (10). A registration for any of these would be a
#   registered test whose binary can never be built, which is the exact shape
#   tools/check_link_health.sh CHECK 3 exists to catch. The blocking symbol is named in each case:
#     tests/test_tensor_slice.cpp   'SliceRange' and 'TensorSlice' are not declared in
#         ninfer::artifact (src/artifact/reader.h declares neither type).
#     tests/test_kv_capacity_tp2.cpp  ninfer::runtime::resolve_kv_capacity_symmetric is not a member
#         of ninfer::runtime (src/runtime/engine/kv_capacity.h declares only resolve_kv_capacity).
#     tests/targets/qwen3_6_27b/test_shard_map.cpp, ..._sharded_materialization.cpp,
#     .../test_engine_tp2_real.cpp, .../test_engine_mtp_tp2_real.cpp, .../test_graph_tp2.cpp
#         -- all five fail inside src/targets/qwen3_6_27b/impl/config.h, whose first include is
#         <ninfer/targets/qwen3_6/frontend.h>, a header that does not exist in this tree; the three
#         engine ones additionally reference EngineOptions members `tp`/`devices` that this
#         revision's include/ninfer/engine.h does not declare.
#     tests/muse128_repro.cu, tests/repro_g35.cu
#         ninfer::ops::detail::gqa_attention_small_t_muse / _g35 are not declared.
#     tests/muse128_repro2.cu, tests/muse128_repro4.cu
#         ops::launcher::gqa_attention's parameter list drifted from these call sites.
#     tests/ops/softmax_attention/test_hq_attention.cu  KvCacheStorage::HqE8Rice2B does not exist.
#     tests/ops/test_hq_codec.cu   std::memcpy is reached with <cstring> never included.
#
#   COMPILE, AND ARE DELIBERATELY NOT REGISTERED (9), split by reason:
#     tests/iso_kv_gpu_test.cu, tests/split_gpu_test.cu, tests/suffix_gpu_test.cu -- ad-hoc
#       reproducers written to chase one fault each (their own first lines say what they reproduce).
#       They have no verdict structure and no device gate, so registering them would put ungated
#       device binaries into a suite other lines share -- the same defect this line removed from
#       ninfer_cold_i8_test / ninfer_entropy_cold_requant_test / ninfer_ple_layout_test.
#     tests/ops/test_allreduce.cpp, tests/ops/test_kvarn.cpp, tests/ops/test_hq_retrieval.cu --
#       these three DO compile and DO carry the repository's 77 convention, and they are the honest
#       near-misses of this list. They are left out BY NAME and for a stated cost, not by oversight:
#       all three put a multi-minute DEVICE job into every `ctest` run of a suite several lines
#       share, and test_hq_retrieval builds a 1,000,000 x 256 corpus to do it. Attaching each is one
#       line of this file; this line did not spend another line's turn on a decision that is about
#       suite cost rather than about coverage.
#     The remaining six of this group are the qwen3_6_27b GPU tests and the two muse repros already
#       listed above as not compiling.
# ---------------------------------------------------------------------------
# WITHDRAWN, BY MEASUREMENT, ON THE SAME DAY IT WAS PROPOSED. tests/test_recall_codec_blind.cpp
# compiles and links (its only include is spec/turn_recall_journal.h, a std-only host header), but on
# THIS tree it is RED, and a red entry in a suite several lines share is worse than an unregistered
# file. Verbatim, no device required:
#
#     $ ./ninfer_recall_codec_blind_test
#     FAIL: PD the sequence-end census carries the count, appended
#           the mutation this check rejects: drop it from the census: the single-run line would be the
#           only evidence
#     checks_failed=1
#     VERDICT=FAIL_WRONG          rc=3
#
# So it is reported here rather than registered: the file pins text and counters in
# src/targets/qwen3_6/impl/runtime/program_impl.h, and that producer has drifted from the pin. The
# repair is a decision about which side is right -- the pin or the producer -- and it is not this
# registration's to make. One `ninfer_add_test(... NEEDS_SOURCE_DIR)` line attaches it the day the red
# is resolved. See dl/testhyg/REPORT.md and dl/testhyg/logs/f03_newtargets_link.txt.
ninfer_add_test(ninfer_recall_restore_units_test
  SOURCES test_recall_restore_units.cpp)
ninfer_add_test(ninfer_causal_deferred_lookup_instantiation_test
  SOURCES ops/test_deferred_lookup_instantiation.cu
  LIBRARIES ninfer_ops)
ninfer_add_op_test(ninfer_gdn_input_proj_conv_restart_test
  SOURCES ops/test_gdn_input_proj_conv_restart.cpp
  LIBRARIES ninfer_ops)
ninfer_add_op_test(ninfer_ple_seam_gather_e2e_test
  SOURCES ops/ple_seam_gather_e2e_test.cu
  LIBRARIES ninfer_engine ninfer_core)
ninfer_add_op_test(ninfer_e8_root_codec_test
  SOURCES ops/test_e8_root_codec.cu
  LIBRARIES ninfer_ops)

# ---------------------------------------------------------------------------
# K40 OPERATOR GATE (line `k40land`, 2026-09-22).  A TEXT check over the source tree: no CUDA,
# no card, no lock, no build, no model.  It writes nothing.
#
# WHY IT EXISTS.  The K40 route (`dl/_orch/landq/k40finish/`) lands 27 rewritten comparison
# operators: the C++20 `friend bool operator==(X, X) noexcept = default;` / `operator<=>`
# spellings that nvcc 11.8's front end cannot read become EXPLICIT member-by-member
# comparisons.  That rewrite has one silent failure mode: from then on a structure that GAINS
# A MEMBER compares one field short, the compiler says nothing, and the semantics change with
# no line moving.  That is not hypothetical -- the gate below found it IN THE PREPARED LANDING
# IMAGE (`src/runtime/contract/types.h`, struct `IdentityMaterializationAssessment`, missing
# `physical_status`), and found a second defect of the same family in `context_cost.h` (a
# duplicated `bool` plus a one-parameter `friend operator==`, which does not compile at all).
# The `= default` form is PASSED THROUGH on purpose: the compiler still follows it, so a tree
# that has not landed that rewrite is green and says so rather than pretending to check.
#
# WHAT IT RUNS.  tools/k40gate/opcheck.py, two phases.
#   Phase 1: for each of the 65 structures named in tools/k40gate/opcheck.manifest, the set of
#     members compared by the explicit operator== must EQUAL the set of non-static data
#     members, else RED naming file:line plus every missing and every extra field name.
#   Phase 2: over the 45 files of both landing sets, a declaration-form scan (S1 a duplicated
#     return-type specifier in one declaration head; S2 a `friend operator==` whose arity is
#     not 2) and a brace-balance check.
# The script runs its own two fixtures FIRST on every invocation and reddens if either stops
# being detected, so a green here cannot come from the checker having become a no-op; the two
# fixtures are a structure whose comparison misses `beta`, and a declaration carrying the
# duplicated `bool`.  Measured controls live in dl/k40land/REPORT.md.
add_test(NAME ninfer_k40_operator_gate_test
         COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/tools/k40gate/opcheck.py
                 ${PROJECT_SOURCE_DIR})

