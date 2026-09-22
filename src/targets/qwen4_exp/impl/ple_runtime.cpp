// Compile coverage for the N6 PLE seam (impl/ple_runtime.h).
//
// Until this translation unit existed the seam had no compilation owner at all:
// nothing under src/targets/qwen4_exp was enumerated by the build (no
// add_subdirectory, no source list entry), so no build and no test ever parsed
// ple_runtime.h. It had accumulated six compile errors that only a real compiler
// could see. This TU registers the header on ninfer_engine (src/CMakeLists.txt)
// so an ordinary build parses it from now on.
//
// It deliberately defines nothing: every PleRuntime entry point is an inline
// member defined in the header, so the object file is empty of symbols. Move this
// source entry into src/targets/qwen4_exp/CMakeLists.txt once that directory owns
// more sources.
#include "targets/qwen4_exp/impl/ple_runtime.h"
