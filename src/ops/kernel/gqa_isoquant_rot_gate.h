#pragma once

// Host-side switch for the baked SO(4) rotation gate (gqa_isoquant_rot.cuh).
//
// This header is host-only (no CUDA headers), same idiom as
// gqa_isoquant_row_scale_loader.h, so the option/planner layer can decide
// "rotation on/off" without seeing the device symbol. The upload itself lives
// in gqa_isoquant_rot.cu, the one translation unit that sees the __constant__.
//
// Three-way spec, matching the row-scale switch:
//   "auto" / "" / unset -> restore the baked enable word (1). The descriptor is
//                          process-global device state, so "auto" is written
//                          explicitly rather than left alone: an engine built
//                          with "off" followed by an engine built with "on" in
//                          the same process must not inherit the off gate. The
//                          value written is the baked initializer, so the
//                          on-path device behaviour is unchanged.
//   "off"              -> upload the disabled gate: K is written unrotated and
//                          Q is read unrotated, so QK^T stays exact but the
//                          K/Q quantization happens in the unrotated domain
//                          (i.e. exactly what a model without the IsoQuant
//                          calibration does).
// Anything else is refused -- a typo must not silently pick a mode.

#include <string>

namespace ninfer::ops {

enum class KvRotationMode {
    Auto,  // baked table (default)
    Off,   // identity: gqa_isoquant_rot_block4() early-returns
};

[[nodiscard]] inline bool kv_rotation_mode_from_spec(const std::string& spec,
                                                     KvRotationMode& out,
                                                     std::string& err) {
    if (spec.empty() || spec == "auto" || spec == "on" || spec == "default") {
        out = KvRotationMode::Auto;
        return true;
    }
    if (spec == "off" || spec == "none" || spec == "identity") {
        out = KvRotationMode::Off;
        return true;
    }
    err = "kv-rotation: expected auto|on|off, got '" + spec + "'";
    return false;
}

// Reads NINFER_KV_ROTATION (unset => Auto, i.e. the baked enable word) and
// applies the mode. Returns true when the gate is OFF, false when it is ON.
// Throws on an unknown spec: silently ignoring a spec is indistinguishable
// from success.
bool kv_rotation_apply_from_env();

// Same, with the spec passed explicitly by the option layer (--kv-rotation).
bool kv_rotation_apply_spec(const std::string& spec);

} // namespace ninfer::ops
