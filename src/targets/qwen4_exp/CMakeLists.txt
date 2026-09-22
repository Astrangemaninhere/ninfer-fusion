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
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/ple_ngram.cpp)
