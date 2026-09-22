# src/targets/gemma4_31b/CMakeLists.txt
#
# Identity-only registration for Gemma-4-31B, in the shape
# src/targets/qwen4_exp/CMakeLists.txt:1-7 established for "the identity is
# recognized but the runtime does not exist yet".
#
# What this directory IS: the identity header (export/) plus the measured
# geometry constants (impl/config.h) plus the geometry cross-check that
# src/targets/registry.cpp:539-560 runs before the generic refusal. Those three
# are header + one host TU; nothing here needs CUDA.
#
# What this directory is NOT, by name: the engine-facing Package surface
# (LoadPlan/LoadedModel/Frontend/SequencePlanner/Program) and the Variant leaf
# set that src/targets/muse_glimmer_30b/impl/{variant,load/variant}.cpp supply.
# Two measured blockers keep those from being writable today, both recorded as
# static_asserts in impl/config.h: the ten full_attention layers carry
# head_dim 512 / 4 KV heads, which src/ops/kernel/gqa_attention_geometry.cuh:15
# cannot instantiate at all (it admits 128 or 256), and rotary_dim differs by
# layer kind while the shared TextConfig has a single slot. Writing a Package
# over one flattened geometry would load silently wrong numbers on ten of sixty
# layers, so this target refuses by name instead.

target_include_directories(ninfer_engine PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/export
  ${CMAKE_CURRENT_SOURCE_DIR}/impl)

# One host-only TU, and it is the compilation owner of the geometry cross-check
# declared in the export header. Without an owner that function would accumulate
# compile errors no build could see -- the same reasoning src/CMakeLists.txt:595-600
# records for the qwen4_exp PLE seam.
target_sources(ninfer_engine PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/package.cpp)
