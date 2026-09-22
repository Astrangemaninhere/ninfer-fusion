#pragma once

// ---------------------------------------------------------------------------
// "Can this artifact draft?" -- the one question `--spec auto` turns on.
//
// WHY IT IS A FUNCTION IN ITS OWN HEADER.  This decision is the reason a source with no MTP
// layer could not be loaded at all: the front end defaults `--spec` to auto, auto resolved to
// Mtp unconditionally, and the binder then demanded the 12 `mtp/*` objects that a source with
// no `nextn_predict_layers` never had.  A decision that no test can reach is a decision that
// comes back, so it is stated here as a pure function of (what the ARTIFACT is, what the RUN
// asked for) and is exercised on every combination, with no device and no artifact, by
// `dl/mtpcopt/cpp/test_draft_resolution.cpp` (compiled by
// `dl/mtpcopt/evidence/run_cpp_evidence.sh`).  `package.cpp` calls it, so the tested text and
// the shipped text are the same function; nothing in this header includes CUDA or the artifact
// reader, which is what makes that test runnable on this box.
//
// The flavour is read from the ARTIFACT (its identity's `weights_id`), never from the run's
// wishes: `tools/convert/qwen3_5_9b/convert.py` writes `gguf-kquant-nomtp` exactly when the
// source GGUF declared no `nextn_predict_layers` block.
// ---------------------------------------------------------------------------

#include <cstdint>

namespace ninfer::targets::qwen3_5_9b::detail {

enum class DraftResolution : std::uint8_t {
    //: The artifact declares a draft block: `--spec auto` stays the MTP path, unchanged.
    UseMtp,
    //: It declares none: `--spec auto` lands on None, because there is nothing to draft with.
    DisableSpeculation,
    //: It declares none and the run asked for a draft window anyway: refuse by name.
    RefuseDraftWindow,
};

[[nodiscard]] constexpr DraftResolution resolve_auto_draft(bool artifact_declares_draft,
                                                          bool draft_window_requested) noexcept {
    if (artifact_declares_draft) { return DraftResolution::UseMtp; }
    return draft_window_requested ? DraftResolution::RefuseDraftWindow
                                  : DraftResolution::DisableSpeculation;
}

} // namespace ninfer::targets::qwen3_5_9b::detail
