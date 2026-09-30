#pragma once

// qwen4_exp / THE ENGINE-SIDE READER of EngineOptions::ple_sidecar_root.
//
// WHY THIS HEADER EXISTS
// ----------------------
// `--ple-sidecar` was parsed by both front ends (apps/cli/options.cpp:686,
// src/serve/serve_options.cpp:538), stored (include/ninfer/types.h:571) and STAT'd at startup
// (product::validate_ple_sidecar_root, called from apps/cli/main.cpp:503 and
// src/serve/generation_service.cpp:313) -- and then read by NOTHING on the engine side. The
// only thing in the engine surface that touched the value at all was
// resolve_ple_sidecar_root() (impl/ple_runtime.h:64), and nothing called that either. So
// `--ple-sidecar <valid root>` and no flag produced the same load: the flag changed nothing an
// operator could observe. That is the silently inert switch F391 names, and its cure is not a
// comment -- it is a caller.
//
// THIS FILE IS THE DECLARATION HALF; impl/ple_session.cpp is the definition half. The split is
// deliberate: ple_runtime.h includes <cuda_runtime.h>, and src/targets/registry.cpp (the
// caller) must not drag that into every TU. The declaration is CUDA-free, and the handle is
// type-erased to shared_ptr<const void>, which is the whole relationship the engine has with
// the sidecar: keep it alive, read its root, read its counters.
//
// THE THREE STATES, and which of them refuses:
//   * empty root            -> PLE off. No attachment, no filesystem access.
//   * root + this family    -> attach() (impl/ple_runtime.h:93), non-empty handle.
//   * root + another family -> REFUSED BY NAME by the caller (declares_ple_stage is false).
//                              The PLE residual is ADDITIVE, so a load that succeeded with the
//                              sidecar attached-and-ignored is indistinguishable from PLE off
//                              in every downstream number.

#include "ninfer/types.h"
#include "targets/qwen4_exp/export/ninfer/targets/qwen4_exp/package.h"

#include <cstddef>
#include <memory>
#include <string>
#include <string_view>

namespace ninfer::targets::qwen4_exp {

// The identity whose runtime can consume a PLE stage. ONE place, so "which family" cannot
// drift between the attach and the refusal that guards it.
[[nodiscard]] inline bool declares_ple_stage(std::string_view model_id) noexcept {
    return model_id == kModelId;
}

// The attached sidecar, type-erased. `owner` is null exactly when PLE is off; otherwise it
// holds a PleRuntime (4 fds + a bounded pinned cache, ple_table.h:35-38) alive.
struct PleSidecarHandle {
    std::shared_ptr<const void> owner;
    std::string root;        // the root actually used (explicit, or the adjacent ple-root/)
    std::size_t gathers = 0; // forensics snapshot at attach time, so a caller can prove it ran
};

// Reads EngineOptions::ple_sidecar_root (this is the line the acceptance criterion names) and
// attaches the sidecar through PleRuntime::attach. Throws the target's own named error when a
// root is configured but unusable; the caller decides what a valid-but-unconsumable root
// means. Returns a handle with a null owner when PLE is off -- it does NOT return an error,
// because "no sidecar" is a legal state and the one every existing run is in.
//
// `artifact_path` comes from options; the adjacent ple-root/ discovery is the target's own
// documented rule (ple_runtime.h:59-73) and stays here rather than being applied to every
// family by the caller.
[[nodiscard]] PleSidecarHandle attach_ple_sidecar(const EngineOptions& options);

} // namespace ninfer::targets::qwen4_exp
