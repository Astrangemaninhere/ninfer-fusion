# S37 stage (a): identity-only registration. No sources yet; the export header
# (identity + geometry cross-check) and impl/config.h join the engine include path
# so the registry recognition branch in src/targets/registry.cpp compiles. Stage (b)
# adds impl sources here (package.cpp / variant.cpp / load/bindings.cpp).
target_include_directories(ninfer_engine PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/export
  ${CMAKE_CURRENT_SOURCE_DIR}/impl)

# N6: the first implementation source this directory owns. It is host-only (no CUDA
# include), so it is also the first piece of this target a plain host compiler can check
# without a GPU. impl/ple_runtime.cpp (the CUDA-side seam) is still registered directly on
# ninfer_engine in src/CMakeLists.txt -- moving that entry here is that patch's decision,
# not this one's, and the file must never be listed twice.
target_sources(ninfer_engine PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/ple_ngram.cpp
  # The per-sequence PLE session set and the ONE definition of the engine-side reader of
  # EngineOptions::ple_sidecar_root (impl/ple_attach.h / impl/ple_session.h). Registered here
  # rather than in src/CMakeLists.txt because it is this directory's own source; the existing
  # impl/ple_runtime.cpp entry is deliberately left where it is (src/CMakeLists.txt:645-650
  # says moving it is that patch's decision), and no file is listed twice.
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/ple_session.cpp)

# ---------------------------------------------------------------------------------------
# STAGE (b) COMPLETENESS, wired to the registry's load gate.
# ---------------------------------------------------------------------------------------
# The header of this file names impl/package.cpp / impl/variant.cpp / impl/load/bindings.cpp
# as stage (b). Whether they EXIST is now a build-time fact with two consequences, taken in
# one place:
#   * the sources that do exist join ninfer_engine (none do today);
#   * NINFER_QWEN4_EXP_STAGE_B_COMPLETE is published, and src/targets/registry.cpp's refusal
#     for this family is conditional on it -- so the day stage (b) lands, the gate opens with
#     no second edit.
# stage_b.cmake is deliberately self-contained so its verdict can be re-run on its own, against
# either file-set shape, with `cmake -P src/targets/qwen4_exp/stage_b.cmake`. That is how both
# arms are rehearsed here without a build.
include(${CMAKE_CURRENT_SOURCE_DIR}/stage_b.cmake)
if(Q4_STAGE_B_COMPLETE)
  target_sources(ninfer_engine PRIVATE ${Q4_STAGE_B_REQUIRED_SOURCES})
  target_compile_definitions(ninfer_engine PRIVATE NINFER_QWEN4_EXP_STAGE_B_COMPLETE=1)
else()
  # =0 rather than undefined, so the refusal cannot fall back onto a different default, and
  # PER SOURCE rather than one string list: `-DNAME=a.cpp b.cpp` gives the preprocessor a token
  # sequence where a string literal is needed, and the refusal that NAMES the missing files then
  # does not compile (measured). Integers need no quoting.
  target_compile_definitions(ninfer_engine PRIVATE
    NINFER_QWEN4_EXP_STAGE_B_COMPLETE=0
    NINFER_QWEN4_EXP_STAGE_B_HAS_IMPL_PACKAGE=${Q4_STAGE_B_HAS_IMPL_PACKAGE}
    NINFER_QWEN4_EXP_STAGE_B_HAS_IMPL_VARIANT=${Q4_STAGE_B_HAS_IMPL_VARIANT}
    NINFER_QWEN4_EXP_STAGE_B_HAS_IMPL_BINDINGS=${Q4_STAGE_B_HAS_IMPL_BINDINGS})
endif()
