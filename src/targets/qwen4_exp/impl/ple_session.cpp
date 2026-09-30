// qwen4_exp / compile coverage for the PLE session set, and the ONE definition of the
// engine-side reader of EngineOptions::ple_sidecar_root.
//
// WHY A TRANSLATION UNIT. impl/ple_runtime.h (the CUDA-bearing seam) and impl/ple_session.h
// (the per-sequence set) are header-only, so without a source of their own nothing under
// src/targets/qwen4_exp is enumerated by the build and neither header is parsed by any
// compiler -- impl/ple_runtime.cpp exists for exactly that reason (src/CMakeLists.txt:645-650).
// This file is registered by this directory's own CMakeLists.txt so the target owns its own
// compile coverage, and it is the TU the R41 gate compiles with -H to prove WHICH copy of the
// headers it opened.
//
// It also DEFINES attach_ple_sidecar. Splitting the declaration (a CUDA-free header) from the
// definition (this TU, which includes ple_runtime.h) is what lets src/targets/registry.cpp
// call the reader without <cuda_runtime.h> reaching every TU that includes registry.h.
#include "targets/qwen4_exp/impl/ple_attach.h"
#include "targets/qwen4_exp/impl/ple_session.h"

#include <cstdio>
#include <filesystem>
#include <memory>
#include <string>
#include <utility>

namespace ninfer::targets::qwen4_exp {

PleSidecarHandle attach_ple_sidecar(const EngineOptions& options) {
    // ---------------------------------------------------------------------------------------
    // THE READ. options.ple_sidecar_root is the field both front ends fill and the startup
    // validator stats; before this line, nothing else in the engine looked at it.
    // ---------------------------------------------------------------------------------------
    const std::filesystem::path explicit_root = options.ple_sidecar_root;
    std::unique_ptr<PleRuntime> runtime =
        PleRuntime::attach(options.artifact_path, explicit_root);
    if (runtime == nullptr) { return PleSidecarHandle{}; } // PLE off, and that is not an error

    PleSidecarHandle handle;
    // The root, asked of the SAME single point attach() used (ple_runtime.h:64); the answer is
    // deterministic in its two inputs, so this is a lookup and not a second discovery rule.
    handle.root = resolve_ple_sidecar_root(options.artifact_path, explicit_root).string();
    handle.gathers = runtime->forensics().gathers.load(std::memory_order_relaxed);
    const ops::ple::PleLayout& layout  = runtime->layout();
    const std::uint32_t ngram     = layout.ngram_size;
    const std::uint32_t heads     = layout.n_heads;
    const std::uint32_t row_dim   = layout.embedding_row_dimension;
    // The owner is type-erased at the boundary, so registry.h (included by front ends) never
    // sees a CUDA-bearing type. The deleter is the only place the type comes back.
    handle.owner = std::shared_ptr<const void>(runtime.release(), [](const void* raw) {
        delete static_cast<const PleRuntime*>(raw);
    });

    // The observable. An attached stage must be visible in a log, not only in a struct field:
    // this is the line a future acceptance run greps for, and it is the same convention as
    // NINFER_HEADDBG / NINFER_KV_ROWSCALE. Gated so a run that does not ask pays nothing.
    if (ops::ple::ple_stats_enabled()) {
        std::fprintf(stderr,
                     "ninfer: ple sidecar attached at '%s' (ngram=%u heads=%u row_dim=%u "
                     "gathers=%zu)\n",
                     handle.root.c_str(), ngram, heads, row_dim, handle.gathers);
        std::fflush(stderr);
    }
    return handle;
}

} // namespace ninfer::targets::qwen4_exp
