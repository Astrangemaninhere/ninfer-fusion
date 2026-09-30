#pragma once

// The three KV component switches -- --kv-rotation, --kv-row-scale and
// --kv-v-codec -- and the TIER DOMAIN each of them can actually reach.
//
// WHY THIS HEADER EXISTS. A component switch is only a switch if some kernel
// reads its gate, and each of the three is honoured by a SUBSET of the KV
// tiers. Setting one outside its domain parses, validates, is reported as
// accepted and then changes NOTHING -- indistinguishable from success, which
// this repository refuses by convention (a flag that parses and is ignored is
// worse than no flag). This header makes the domain a single, host-testable
// predicate so plan_decoder_state() can refuse instead of staying silent.
//
// Host-only and std-only (same convention as product/kv_tier_formats.h and
// product/kv_bit_budget.h) so a plain host test can exercise it.
//
// ---------------------------------------------------------------------------
// THE DOMAINS, AND THE EVIDENCE FOR EACH
// ---------------------------------------------------------------------------
// The per-layer KV tier is ONE DType per layer. decoder_state.cpp's plan_cache()
// resolves it as
//     selected = (layer_override == BF16) ? global_kv_dtype : layer_override
// so a BF16 entry INHERITS the global dtype, and an empty override table
// inherits wholesale. Every predicate below mirrors that rule exactly. Each one
// admits a tier only when a call-site census finds the gate actually read in
// that tier's kernels; no predicate is looser than its census, because a tier
// admitted on a guess lets the switch through to a kernel that never reads it --
// the same fake-completeness failure, in the other direction.
//
//   Rotation -- SO(4) IsoQuant rotation of K on cache write, Q before
//   quantization. Gate: gqa_isoquant_rot_block4() in
//   ops/kernel/gqa_isoquant_rot.cuh (the only reader). Call-site census over
//   src/:
//       ops/kernel/gqa_attention_decode_nvfp4.cuh:86    NVFP4 decode
//       ops/kernel/gqa_attention_decode_fp8.cuh:207     FP8 decode
//       ops/kernel/gqa_attention_decode_iso3.cuh:212    ISO4E decode
//       ops/kernel/gqa_attention_prefill_nvfp4.cuh:44,45,104,486,616,707,875
//                                                       NVFP4/FP8/ISO4E prefill
//   (gqa_attention_prefill_nvfp4.cuh is the shared prompt TU for all three
//   packed-16 tiers -- gqa_attention_decode_fp8.cuh includes it for exactly this
//   reason -- so one census entry covers the FP8 and ISO4E prefill paths too.)
//   The domain is therefore {NVFP4, FP8_E4M3FN, ISO4E} and NOTHING ELSE.
//
//   THE I8/Rk4v4 CONTRADICTION IS RESOLVED, AND THE OLD COMMENT WAS RIGHT ABOUT I8.
//   The old comment in decoder_state.cpp (rowscale_domain_active) claimed 'the
//   i8, rk4v4, fp8 and iso4e paths rotate but never scale'. The census above finds no
//   I8/Rk4v4 kernel that includes THIS gate header. Both statements hold at once,
//   because the engine has TWO live KV kernel stacks and neither I8/Rk4v4 path is
//   gated:
//     * I8 (DType::I8) rotates on the LEGACY D256 STACK and not on the GQA stack.
//       Legacy: ops/kv_cache/append/kernel.cuh's full-i8 writers call
//       normalized_hadamard_d256_inplace(k_values, lane) on the K row BEFORE
//       quantizing -- :234 in kv_cache_append_full_i8_kernel and :305 in its page
//       variant -- unconditionally. That is the stack behind
//       ops::kv_cache_append()/kv_cache_append_prefix(), which the DFlash and
//       DFlash2 runtimes call (targets/qwen3_6/impl/runtime/dflash_impl.h:184/189,
//       dflash2_impl.h:211) and which tests/ops/test_kv_cache_append.cpp pins. V
//       is NOT rotated on that path.
//       GQA: the I8 branch of gqa_attention_decode_i8.cuh (:333-359) and the I8
//       branches of gqa_attention_prefill_i8.cuh take the g64 scale straight off
//       the raw value (kamax / 127) and encode with gqa_kv_quant_code(); there is
//       no rotation call on either the K write or the Q read. That is the
//       documented contract of the public Op (include/ninfer/ops/gqa_attention.h,
//       INT8-G64: code = clamp(RNE(FP32(x) * inv)) with no pre-transform), and it
//       is the stack the TEXT path uses (ops::gqa_kv_append at
//       text_context_impl.h:624, ops::gqa_attention at :523/:528/:1066/:1071).
//       So an int8 run rotates K or not depending on the ROUTE, and the tree's own
//       conformance test could not have caught the legacy rotation: its I8 branch
//       asserts an exact match against a rotation-free oracle for V and for both
//       scale planes, but it never checks the I8 K codes.
//     * Rk4v4 (DType::E8Kv) rotates on BOTH stacks, unconditionally. GQA:
//       gqa_kv_hadamard64() (a 64-dim Sylvester Hadamard / 8,
//       ops/kernel/gqa_attention_kv_quant.cuh:22), called inside
//       `if constexpr (Rk4v4)` at gqa_attention_decode_i8.cuh:292 (K write) and :389
//       (Q read), and gqa_attention_prefill_i8.cuh:164/305 (K write) and :454
//       (Q read). Legacy: Rk4v4Kv is routed to the same int8 kernels
//       (ops/kv_cache/d256_profile.h:26-28, 'Packed 4-bit E8-lattice K / i4 V
//       with g64 scales (int8-kernel path)'), so it rotates K the same way.
//       Neither stack's I8/Rk4v4 kernels include gqa_isoquant_rot.cuh or read
//       kGqaIsoquantRotGeom.
//   What follows for the switch, and it is true on every route: '--kv-rotation
//   off' leaves every I8 and Rk4v4 rotation RUNNING. For Rk4v4 that is worse than a
//   no-op -- the option help promises 'the identity SO(4) map on BOTH the K write
//   and the Q read', and no spelling of this flag can deliver that, because the Rk4v4
//   map is a Hadamard, not SO(4), and is not gated. Hence the refusal, which
//   names the mechanism that does rotate.
//   A BF16 layer applies no rotation either: the plane stores the cache
//   verbatim. Because BF16 is the engine's DEFAULT KV dtype, a bare
//   "--kv-rotation off" is a no-op on a default run, and that is the case this
//   guard exists for.
//
//   RowScale -- Sinkhorn row scale on K write, inverted on Q read. Gate:
//   gqa_kv_row_scale()/_inv() in gqa_isoquant_row_scale.cuh. Call sites:
//   gqa_attention_decode_nvfp4.cuh:330 (K write) and :489 (Q read),
//   gqa_attention_prefill_nvfp4.cuh:489,710,1042. All of them are in the NVFP4
//   tier's kernels, which is exactly what decoder_state.cpp's own
//   rowscale_domain_active() already encoded. The domain is {NVFP4}.
//
//   VCodec -- E2M1 vs ISO4E V on the NVFP4 tier. kv_layer_v_dtype()
//   (decoder_state.cpp) returns its `dtype` argument UNCHANGED for anything
//   other than NVFP4, so the switch can only ever affect NVFP4 layers. The
//   domain is {NVFP4}.
//
// ---------------------------------------------------------------------------
// WHAT THIS HEADER IS NOT
// ---------------------------------------------------------------------------
// It is not, and cannot be, the validation for a K-side codec switch. K's codec
// is not a knob inside a tier -- it SELECTS the tier (DType::NVFP4 feeds
// gqa_attention_decode_nvfp4.cuh, DType::ISO3 feeds
// gqa_attention_decode_iso3.cuh), so there is no --kv-k-codec to validate. See
// REPORT.md, section "K".

#include "core/dtype.h"

#include <cstdint>
#include <span>
#include <string>

namespace ninfer::product {

// The switches whose reached-domain is checked. This is the validation domain's
// own vocabulary, not the option strings.
enum class KvComponentSwitch : std::uint8_t {
    Rotation,  // --kv-rotation / NINFER_KV_ROTATION
    RowScale,  // --kv-row-scale / NINFER_KV_ROWSCALE
    VCodec,    // --kv-v-codec
};

// The tier a per-layer entry resolves to, mirroring plan_cache()'s rule: a BF16
// entry inherits the global KV dtype, anything else is itself. That is the rule
// for a slot the spec did NOT write; a slot it DID write as BF16 stays BF16
// (kv_resolve_slot_dtype below), which is what makes a per-layer BF16 baseline
// expressible at all.
[[nodiscard]] inline DType kv_resolved_layer_dtype(DType kv_dtype, DType layer_dtype) noexcept {
    return layer_dtype == DType::BF16 ? kv_dtype : layer_dtype;
}

// The SAME rule, with the one fact the 2-argument form cannot carry: whether the
// per-layer spec actually WROTE this slot.
//
// `slot_explicit` true + layer_dtype == BF16 is the case that had no spelling
// before: the operator asked for BF16 on this layer, and the pool must store this
// layer as BF16 instead of inheriting a quantized global --kv-dtype. The two
// arguments inside the table type cannot express it, because KvCacheStorage::
// BFloat16 doubles as the "unset" sentinel -- which is why the mask exists
// (product/kv_options.h, KvLayerStorageSpec). `slot_explicit` false keeps the
// historical inheritance rule bit-for-bit, so an all-default table and every
// caller that passes no mask behave exactly as before this change.
[[nodiscard]] inline DType kv_resolve_slot_dtype(DType kv_dtype, DType layer_dtype,
                                                 bool slot_explicit) noexcept {
    if (layer_dtype == DType::BF16 && !slot_explicit) { return kv_dtype; }
    return layer_dtype;
}

// True when AT LEAST ONE layer of this configuration resolves to a tier whose
// kernels honour `which`. `layer_dtypes` is the per-layer override table
// (DecoderStateSpec::layer_kv_dtypes), already truncated to the layer count; an
// empty table inherits the global dtype wholesale, as plan_cache() does.
[[nodiscard]] inline bool kv_component_switch_domain_active(
    KvComponentSwitch which, DType kv_dtype, std::span<const DType> layer_dtypes,
    std::span<const bool> layer_dtypes_set = {}) noexcept {
    const auto honoured = [which](DType resolved) noexcept {
        switch (which) {
        case KvComponentSwitch::Rotation:
            // {NVFP4, FP8_E4M3FN, ISO4E} exactly: the census of
            // gqa_isoquant_rot_block4() readers. BF16 stores verbatim; I8 rotates
            // via normalized_hadamard_d256_inplace on the legacy DFlash stack and
            // not at all on the GQA stack; Rk4v4 rotates through gqa_kv_hadamard64().
            // None of those three is a reader of this gate.
            return resolved == DType::NVFP4 || resolved == DType::FP8_E4M3FN ||
                   resolved == DType::ISO3;
        case KvComponentSwitch::RowScale:
        case KvComponentSwitch::VCodec:
            return resolved == DType::NVFP4;
        }
        return false;
    };
    if (layer_dtypes.empty()) { return honoured(kv_dtype); }
    // Indexed, not range-for: the mask is positional. An empty or short mask
    // means "no slot was written", i.e. the pre-existing inheritance rule.
    for (std::size_t i = 0; i < layer_dtypes.size(); ++i) {
        const bool slot_explicit = i < layer_dtypes_set.size() && layer_dtypes_set[i];
        if (honoured(kv_resolve_slot_dtype(kv_dtype, layer_dtypes[i], slot_explicit))) {
            return true;
        }
    }
    return false;
}

// Empty when the switch can take effect on this configuration, else the refusal
// text. The caller decides WHETHER the switch was requested; this only answers
// whether the request would reach any kernel.
[[nodiscard]] inline std::string kv_component_switch_domain_error(
    KvComponentSwitch which, DType kv_dtype, std::span<const DType> layer_dtypes,
    std::span<const bool> layer_dtypes_set = {}) {
    if (kv_component_switch_domain_active(which, kv_dtype, layer_dtypes, layer_dtypes_set)) {
        return {};
    }
    switch (which) {
    case KvComponentSwitch::Rotation:
        return "kv-rotation off: no layer of this KV configuration is on a tier whose "
               "kernels read the SO(4) rotation gate, so the switch cannot take effect. "
               "The gate (gqa_isoquant_rot_block4, ops/kernel/gqa_isoquant_rot.cuh) is "
               "read by the NVFP4, FP8_E4M3FN and ISO4E kernels and by nothing else "
               "(gqa_attention_decode_nvfp4.cuh:86, gqa_attention_decode_fp8.cuh:199, "
               "gqa_attention_decode_iso3.cuh:212, "
               "gqa_attention_prefill_nvfp4.cuh:43/44/114/496/626/717/885). On this "
               "stack: BF16 stores the cache verbatim; I8 rotates through "
               "normalized_hadamard_d256_inplace on the legacy D256 stack "
               "(ops/kv_cache/append/kernel.cuh:234/305, the DFlash/DFlash2 route) and "
               "not at all on the GQA stack; Rk4v4 rotates through "
               "gqa_kv_hadamard64() unconditionally "
               "(gqa_attention_decode_i8.cuh:284/379, gqa_attention_prefill_i8.cuh:164/303/450). "
               "None of those is wired to this gate, so 'off' would leave them rotating. "
               "Use --kv-dtype nvfp4|fp8|iso4e (or a per-layer --kv-layer-storage SPEC "
               "naming one of them) if the switch must take effect, or drop "
               "--kv-rotation off";
    case KvComponentSwitch::RowScale:
        return "kv-row-scale: no layer of this KV configuration uses the NVFP4 tier, and the "
               "Sinkhorn row scale is applied by the NVFP4 K/V kernels and nowhere else "
               "(gqa_attention_decode_nvfp4.cuh:330/489, "
               "gqa_attention_prefill_nvfp4.cuh:499/720/1052), so the switch cannot take "
               "effect. Use --kv-dtype nvfp4 or a per-layer --kv-layer-storage SPEC, or drop "
               "--kv-row-scale";
    case KvComponentSwitch::VCodec:
        return "kv-v-codec e2m1: no layer of this KV configuration uses the NVFP4 tier, and "
               "the V codec switch only re-encodes NVFP4-tier V (kv_layer_v_dtype() is the "
               "identity for every other dtype), so the flag cannot take effect. Use "
               "--kv-dtype nvfp4 or a per-layer --kv-layer-storage SPEC, or drop "
               "--kv-v-codec";
    }
    return {};
}

// ---------------------------------------------------------------------------
// THE SLIDING WINDOW IS A COMPONENT SWITCH TOO
// ---------------------------------------------------------------------------
// `PagedKVLayerView::sliding_window_tokens` is the ONLY bound the Cold Host tier has:
// `cold_host_page_is_read_free()` releases a device page to the host on the strength of
// it. The field has TWO consumers per tier, not one -- a DECODE kernel and a PREFILL
// kernel -- and they do not agree with each other. Re-censused by content (F1227) rather
// than by citation; the numbers below are the current lines, grepped:
//
//   DECODE READS IT  gqa_attention_decode_nvfp4.cuh:413
//                          token_begin = (sliding_window > 0) ? window_full - sliding_window : 0
//                    gqa_attention_decode_iso3.cuh:250   the same line
//                    gqa_attention_decode_bf16.cuh:166    the same line, same clamp
//                    gqa_attention_decode_i8.cuh:215      the same line
//                    gqa_attention_decode.cuh:341-361, :494  the small-T family's count/origin
//   DECODE IGNORES IT  gqa_attention_decode_fp8.cuh (no reader at all)
//                    gqa_attention_simt_ffma.cuh (its own gate, :1114)
//   PREFILL READS IT gqa_attention_prefill_nvfp4.cuh:1097 -- and ONLY on KVDType == DType::NVFP4:
//                      `const int window = (sliding_window > 0 && KVDType == DType::NVFP4) ? sliding_window : 0;`
//                    That single line is the whole of the prefill census. `sliding_window`
//                    appears nowhere else under ops/kernel/gqa_attention_prefill*.
//   PREFILL IGNORES IT
//                    * ISO3 / FP8_E4M3FN: the launcher instantiates the SAME template with
//                      KVDType == ISO3 (ops/launcher/gqa_attention_prefill.cu:195 -> :201),
//                      the condition above is false, `window` is 0 and :1098 takes
//                      `visible_start = 0` => the prompt pass is FULL ATTENTION.
//                    * BF16: gqa_attention_prefill_bf16.cuh:132-137 -- the kernel has NO
//                      sliding_window parameter at all, so there is nothing to pass it.
//                    * I8: gqa_attention_prefill_i8.cuh -- same shape, no reader.
//                    * SIMT FFMA: refused by name at gqa_attention_simt_ffma.cuh:1385-1409.
//
// So the correct predicate is a CONJUNCTION over the two phases, and before F1227 the
// function below was the decode census wearing the name of the whole answer: a tier whose
// decode reads the field installed a window that the same tier's prompt pass ignored. That
// is the spark_x2_5_4b shape (27 of 36 layers at 512 tokens: installed, honoured in decode,
// absent from the bf16 prefill) and it is invisible at or below the window length, because
// there the clip is a no-op in both phases.
//
// On an ignoring tier the consequence is NOT a quality loss: the tier releases a page on
// the strength of a window the kernel never applies, and the kernel then reads a page that
// is no longer resident. That is SILENT CORRUPTION, and it is silent precisely because no
// needle probe can attribute it to the window. So it is refused BY NAME here, in the same
// vocabulary and at the same commit point as Rotation / RowScale / VCodec.
//
// The refusal is deliberately NARROW: it fires only where the tier could actually release
// a page, i.e. only when the WHOLE window table is non-zero (`cold_host_layers_are_windowed`,
// cold_host_tier.h:96-101 requires all_of; a single zero entry makes the tier structurally
// inert -- `cold_host_page_is_read_free`, :81-91, returns false on the first zero it sees).
// muse_glimmer_30b declares 2048 on 39 of its 52 layers and is therefore inert, not
// refused; a table of all zeros (every other variant today) is likewise inert.
// THE DECODE CENSUS, AND NOTHING ELSE. "Does this tier's DECODE kernel read
// sliding_window_tokens?" -- the question the name of the OLD function answered with this
// body. It is not a census of every reader; see kv_window_tier_prefill_honoured below.
[[nodiscard]] inline bool kv_window_tier_decode_honoured(DType resolved) noexcept {
    // i8win ADDS I8, and the evidence is the kernel: gqa_attention_decode_i8.cuh now derives
    // token_begin from the field (the max(0, window_full - sliding_window) clamp, :215).
    //
    // bf16win ADDS BF16, and the evidence for it is the kernel, not this comment: the bf16
    // decode kernel derives token_begin from the field (gqa_attention_decode_bf16.cuh:166,
    // `const int window_full = last_pos + 1;` followed by the max(0, ...) clamp).
    //
    // i8win: I8 IS added here. A comment block claimed it while this return was left
    // untouched, and the acceptance run caught the discrepancy by name
    // (`--kv-layer-storage 0:16:int8` still answered `cold-host window: layer 0 ...`).
    return resolved == DType::NVFP4 || resolved == DType::ISO3 || resolved == DType::BF16 ||
           resolved == DType::I8;
}

// THE PREFILL CENSUS. "Does this tier's PROMPT kernel read the same field?" One tier does,
// and the proof is a single line of the shared prompt TU (see the table above):
//   NVFP4 -- gqa_attention_prefill_nvfp4.cuh:1097, `window` non-zero only for NVFP4.
//   every other tier -- 0 there, or no parameter to pass at all.
// The list below must be edited together with that file, and the two cannot drift silently:
// a tier added here that the kernel does not honour installs a window the prompt pass
// ignores, which is the defect this split exists to make loud.
[[nodiscard]] inline bool kv_window_tier_prefill_honoured(DType resolved) noexcept {
    return resolved == DType::NVFP4;
}

// `kv_window_tier_honoured` now means WHAT ITS NAME SAYS: honoured in BOTH phases. This is
// the predicate an INSTALL site needs, because a window installed on the strength of one
// phase is a bound the other phase does not apply, and the disagreement is silent exactly
// where it matters -- above the window length, which no short-context probe can reach.
//
// The install site is targets/qwen3_6/impl/runtime/layouts_impl.h (the `honoured_here`
// expression in the per-layer window loop); it feeds the layer window table AND the named
// report of the layers whose declaration did not survive. Before F1227 that expression read
// the DECODE census, so a bf16 / i8 / ISO3 plan installed a window its own prompt pass
// ignored, and the report never fired for the tiers that most needed it.
[[nodiscard]] inline bool kv_window_tier_honoured(DType resolved) noexcept {
    return kv_window_tier_decode_honoured(resolved) &&
           kv_window_tier_prefill_honoured(resolved);
}

// ---------------------------------------------------------------------------
// THE RELEASE CENSUS: THE PHASES THAT CAN STILL READ A PAGE AFTER IT LEAVES
// ---------------------------------------------------------------------------
// `kv_window_tier_honoured` above answers "does EVERY phase that will read this layer
// apply the declared window". That is ALSO the question a page-release decision has to
// answer, because a released page is read by whichever kernel runs NEXT -- and the prompt
// kernel that reaches an ISO3 layer does not apply the window:
//   prefill gqa_attention_prefill_nvfp4.cuh:1097 -- `window` is 0 unless
//           KVDType == DType::NVFP4, and :1098 then takes `visible_start = 0`.
//   launch  ops/launcher/gqa_attention_prefill.cu:195
//           `} else if (cache.dtype == DType::ISO3) {` -> :201
//           `gqa_attention_prefill_nvfp4_kernel<Geometry, Metadata, DType::ISO3>`.
// So a page retired on ISO3's window is read by the next prefill that covers it: the class
// named above as SILENT CORRUPTION, refused by name at the release boundary instead of
// being trusted to a phase the caller remembers to check.
//
// PER TIER, AND WHY EACH ANSWER IS WHAT IT IS:
//   NVFP4 -- release-safe. Decode reads the field (gqa_attention_decode_nvfp4.cuh:413) and
//            the prompt kernel's NVFP4 instance reads it too (:1097).
//   ISO3 / BF16 / I8 -- NOT release-safe. Their decode reads it
//            (gqa_attention_decode_iso3.cuh:250, gqa_attention_decode_bf16.cuh:166,
//            gqa_attention_decode_i8.cuh:215); their prompt kernel does not. They stay out
//            until the prefill side applies the window as well -- making the kernel do that
//            is a numerics change on the compute path and is deliberately NOT this
//            predicate's job.
//   the other tiers -- not in the decode census either, so they cannot reach a release site
//            on a window's strength at all.
//
// WRITTEN THROUGH `kv_window_tier_honoured` AND NOT BESIDE IT (F1227). The release question
// and the install question are the same conjunction over the same two censuses, and this
// header already carried the cost of keeping two answers by hand: BF16 sat under "IGNORE IT"
// in prose while the code admitted it. One census, two readers.
[[nodiscard]] inline bool kv_window_tier_release_safe(DType resolved) noexcept {
    return kv_window_tier_honoured(resolved);
}

// Empty when every layer the window is declared on resolves to a tier that honours it
// (or when no window can ever release a page), else the refusal text.
[[nodiscard]] inline std::string kv_sliding_window_domain_error(
    std::span<const std::uint32_t> layer_windows, DType kv_dtype,
    std::span<const DType> layer_dtypes, std::span<const bool> layer_dtypes_set = {}) {
    if (layer_windows.empty()) { return {}; }
    // Inert: the tier cannot admit unless EVERY layer is windowed. Nothing to refuse.
    for (const std::uint32_t window : layer_windows) {
        if (window == 0) { return {}; }
    }
    for (std::size_t i = 0; i < layer_windows.size(); ++i) {
        const bool slot_explicit = i < layer_dtypes_set.size() && layer_dtypes_set[i];
        const DType resolved     = i < layer_dtypes.size()
                                       ? kv_resolve_slot_dtype(kv_dtype, layer_dtypes[i], slot_explicit)
                                       : kv_dtype;
        if (kv_window_tier_honoured(resolved)) { continue; }
        return "cold-host window: layer " + std::to_string(i) +
               " of this KV configuration declares sliding_window_tokens = " +
               std::to_string(layer_windows[i]) +
               ", but that layer's KV tier does not read the field, so the window would "
               "not bound what its attention kernel reads. The Cold Host tier releases a "
               "device page to the host on the strength of this field, so on this tier a "
               "released page would be read anyway: SILENT CORRUPTION, not a quality loss. "
               "`sliding_window_tokens` is read by the NVFP4 and ISO3 decode kernels and by "
               "nothing else (gqa_attention_decode_nvfp4.cuh:259, "
               "gqa_attention_decode_iso3.cuh:132); the BF16, FP8_E4M3FN, I8, simt_ffma and "
               "small-t decode paths hard-code full attention "
               "(gqa_attention_decode_bf16.cuh:153, gqa_attention_decode_fp8.cuh:131, "
               "gqa_attention_decode_i8.cuh:202, gqa_attention_simt_ffma.cuh:394, "
               "gqa_attention_decode.cuh:359). Use --kv-dtype nvfp4|iso4e (or a per-layer "
               "--kv-layer-storage SPEC naming one of them), or clear the variant's "
               "sliding_window";
    }
    return {};
}

} // namespace ninfer::product
