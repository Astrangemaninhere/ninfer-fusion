#pragma once

// N3 / S28 round 1: host-side parser/validator for the KV row-scale SIDECAR
// (format spec: tools/kv_rowscale_sidecar.py). Round 1 deliberately touches
// NO option plumbing: the engine hook is the environment variable
// NINFER_KV_ROWSCALE=<path> (same idiom as NINFER_HEADDBG / NINFER_KVDUMP_DIR),
// consumed once at decoder-state planning. All failures HARD-FAIL with the
// offending field named -- a foreign or corrupt table must never be applied
// silently (the baked table covers 16 layers of qwen3.8-27b; Muse's layers
// 0..15 currently read it through gqa_kv_row_scale's in-range path).
//
// This header is host-only (no CUDA headers) so the validation logic is
// unit-testable with plain g++ (_collab/C_s28_rowscale/C_s28_host_test.cpp).

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace ninfer::ops {

inline constexpr std::uint32_t kKvRowScaleSidecarVersion = 1;
inline constexpr std::uint32_t kKvRowScaleFlagIdentity = 0x1;
// Capacity of the device constant pool the payload must fit in
// (gqa_isoquant_row_scale.cuh: kGqaKvRowScalePool[kKvRowScalePoolWords]).
// This is an UPPER BOUND, not an equality: any geometry with
// layers * kv_heads * head_dim <= capacity loads, and the runtime geometry
// descriptor tells the kernels how those words are strided. loader.cu
// static_asserts this equals kKvRowScalePoolWords.
inline constexpr std::uint32_t kKvRowScalePoolCapacity = 16u * 4u * 256u;

struct KvRowScaleSidecar {
    std::uint32_t version = 0;
    std::uint32_t layers = 0;
    std::uint32_t kv_heads = 0;
    std::uint32_t head_dim = 0;
    std::uint32_t flags = 0;
    std::uint32_t crc = 0;
    std::uint64_t model_hash = 0;
    std::string tag;
    std::vector<unsigned short> words;  // BF16 bit patterns, [layer][kv][d]
};

namespace detail {

inline std::uint32_t le_u32(const unsigned char* p) {
    return static_cast<std::uint32_t>(p[0]) | (static_cast<std::uint32_t>(p[1]) << 8) |
           (static_cast<std::uint32_t>(p[2]) << 16) |
           (static_cast<std::uint32_t>(p[3]) << 24);
}

inline std::uint64_t le_u64(const unsigned char* p) {
    std::uint64_t v = 0;
    for (int i = 7; i >= 0; --i) {
        v = (v << 8) | p[i];
    }
    return v;
}

inline std::uint32_t crc32_reflected(const unsigned char* data, std::size_t n) {
    // CRC-32/ISO-HDLC (zlib.crc32): poly 0xEDB88320, init/final 0xFFFFFFFF.
    static std::uint32_t table[256];
    static bool ready = false;
    if (!ready) {
        for (std::uint32_t i = 0; i < 256; ++i) {
            std::uint32_t c = i;
            for (int k = 0; k < 8; ++k) {
                c = (c & 1u) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
            }
            table[i] = c;
        }
        ready = true;
    }
    std::uint32_t c = 0xFFFFFFFFu;
    for (std::size_t i = 0; i < n; ++i) {
        c = table[(c ^ data[i]) & 0xFFu] ^ (c >> 8);
    }
    return c ^ 0xFFFFFFFFu;
}

}  // namespace detail

// Parse the 64-byte header + payload. Every failure names the field.
[[nodiscard]] inline bool kv_rowscale_sidecar_parse(const std::string& blob,
                                                    KvRowScaleSidecar& out,
                                                    std::string& err) {
    const unsigned char* p =
        reinterpret_cast<const unsigned char*>(blob.data());
    if (blob.size() < 64) {
        err = "size: " + std::to_string(blob.size()) + " bytes < 64 header";
        return false;
    }
    if (std::memcmp(p, "NINFERKVRS1", 11) != 0 ||
        p[11] | p[12] | p[13] | p[14] | p[15]) {
        err = "magic: sidecar is not a NINFERKVRS1 file";
        return false;
    }
    out.version = detail::le_u32(p + 16);
    if (out.version != kKvRowScaleSidecarVersion) {
        err = "version: " + std::to_string(out.version) + " != 1";
        return false;
    }
    out.layers   = detail::le_u32(p + 20);
    out.kv_heads = detail::le_u32(p + 24);
    out.head_dim = detail::le_u32(p + 28);
    out.flags    = detail::le_u32(p + 32);
    out.crc      = detail::le_u32(p + 36);
    out.model_hash = detail::le_u64(p + 40);
    out.tag.assign(reinterpret_cast<const char*>(p + 48), 16);
    const std::size_t nul = out.tag.find('\0');
    if (nul != std::string::npos) { out.tag.resize(nul); }
    const std::uint64_t expect_words =
        static_cast<std::uint64_t>(out.layers) * out.kv_heads * out.head_dim;
    const std::uint64_t have_words = (blob.size() - 64) / 2;
    if (expect_words != have_words || (blob.size() - 64) % 2 != 0) {
        err = "length: payload " + std::to_string(blob.size() - 64) +
              " bytes != " + std::to_string(expect_words * 2) + " for " +
              std::to_string(out.layers) + "x" + std::to_string(out.kv_heads) +
              "x" + std::to_string(out.head_dim);
        return false;
    }
    const unsigned char* payload = p + 64;
    if (detail::crc32_reflected(payload, blob.size() - 64) != out.crc) {
        err = "crc32: payload crc mismatch";
        return false;
    }
    out.words.resize(static_cast<std::size_t>(expect_words));
    for (std::uint64_t i = 0; i < expect_words; ++i) {
        out.words[static_cast<std::size_t>(i)] =
            static_cast<unsigned short>(payload[2 * i] |
                                        (payload[2 * i + 1] << 8));
    }
    if (out.flags & kKvRowScaleFlagIdentity) {
        for (const unsigned short w : out.words) {
            if (w != 0x3F80) {  // BF16 1.0
                err = "identity.payload: non-1.0 word in identity-flagged table";
                return false;
            }
        }
    }
    return true;
}

// N3/S28 round 1 engine hook (definition: gqa_isoquant_row_scale_loader.cu).
// Reads NINFER_KV_ROWSCALE; unset/empty => false (feature off). Parses and
// validates against the loaded model identity, then uploads the table to the
// device constant. Any validation failure THROWS (no silent fallback to the
// baked table: a silently-ignored sidecar is indistinguishable from success).
bool kv_rowscale_sidecar_apply_from_env(std::uint32_t model_layers,
                                        std::uint32_t model_kv_heads,
                                        std::uint32_t model_head_dim,
                                        std::uint64_t model_hash);

// ---- SEPARATION: the row scale becomes a three-state component ----
//
// The switch used to be "sidecar file or baked table", i.e. there was no way
// to turn the row scale OFF without generating an all-ones sidecar file. Off
// is now a first-class state and it is expressed KERNEL-SIDE, in the accessor
// that already exists:
//
//   Auto  "auto" / "" / unset -> restore the baked calibration geometry
//                                (kKvRowScaleBakedGeom) into the descriptor. A
//                                restore, not a no-op: the descriptor is
//                                process-global, so an engine that ran with
//                                "off" must not leak the zeroed geometry into a
//                                later engine in the same process. The value
//                                written is the baked initializer, so the
//                                on-path device behaviour is unchanged.
//   Off   "off" | "none" | "identity" -> upload an all-zero geometry
//        descriptor. kGqaKvRowScaleGeom[0..2] are the accessor's range checks,
//        so layers=0 makes EVERY (layer, kv_head, d) out of geometry and
//        gqa_kv_row_scale() answers 1.0 -- the identity path that already
//        exists and is already exercised by a model wider than the pool
//        (gqa_isoquant_row_scale.cuh comment + _TODO.md 104). K*s = K and
//        Q/s = Q exactly, and NO new kernel instruction is added: "off" reuses
//        the guard the on-path already pays for.
//   Path  anything else -> the existing NINFERKVRS1 sidecar.
//
// A literal path named "off"/"auto" is shadowed by the keyword; a path with a
// separator ("./off") is unambiguous. That is the only spelling hazard, and it
// is the same trade the tier vocabulary already makes.
enum class KvRowScaleMode {
    Auto,
    Off,
    Path,
};

[[nodiscard]] inline bool kv_rowscale_mode_from_spec(const std::string& spec,
                                                     KvRowScaleMode& mode,
                                                     std::string& path,
                                                     std::string& err) {
    path.clear();
    if (spec.empty() || spec == "auto" || spec == "on" || spec == "default") {
        mode = KvRowScaleMode::Auto;
        return true;
    }
    if (spec == "off" || spec == "none" || spec == "identity") {
        mode = KvRowScaleMode::Off;
        return true;
    }
    mode = KvRowScaleMode::Path;
    path = spec;
    (void)err;
    return true;
}

// Explicit-spec entry point (--kv-row-scale). Returns true when the row scale
// is OFF, false when it is ON (baked Auto or a loaded sidecar). Every state
// writes the descriptor: the switch is state-based, not event-based, so the
// last engine planned in a process wins and no mode leaks into the next one.
bool kv_rowscale_sidecar_apply_spec(const std::string& spec,
                                    std::uint32_t model_layers,
                                    std::uint32_t model_kv_heads,
                                    std::uint32_t model_head_dim,
                                    std::uint64_t model_hash);

// Identity gate against the LOADED model. Non-identity tables must match the
// model geometry EXACTLY (a 16-layer table under a 52-layer model is the
// foreign-table hazard and is refused). Identity tables may under-cover.
// model_hash is enforced when both sides are non-zero.
[[nodiscard]] inline bool kv_rowscale_sidecar_check(
    const KvRowScaleSidecar& sc, std::uint32_t model_layers,
    std::uint32_t model_kv_heads, std::uint32_t model_head_dim,
    std::uint64_t model_hash, std::string& err) {
    const bool identity = (sc.flags & kKvRowScaleFlagIdentity) != 0;

    if (!identity && sc.layers != model_layers) {
        err = "identity.layers: table " + std::to_string(sc.layers) +
              " != model " + std::to_string(model_layers) +
              " (foreign-model table refused)";
        return false;
    }
    if (identity && sc.layers > model_layers) {
        err = "identity.layers: identity table " + std::to_string(sc.layers) +
              " > model " + std::to_string(model_layers);
        return false;
    }
    if (sc.kv_heads != model_kv_heads) {
        err = "identity.kv_heads: table " + std::to_string(sc.kv_heads) +
              " != model " + std::to_string(model_kv_heads);
        return false;
    }
    if (sc.head_dim != model_head_dim) {
        err = "identity.head_dim: table " + std::to_string(sc.head_dim) +
              " != model " + std::to_string(model_head_dim);
        return false;
    }
    if (sc.words.size() > kKvRowScalePoolCapacity) {
        err = "symbol_extent: table has " + std::to_string(sc.words.size()) +
              " words > kGqaKvRowScalePool capacity " +
              std::to_string(kKvRowScalePoolCapacity);
        return false;
    }
    if (model_hash != 0 && sc.model_hash != 0 && sc.model_hash != model_hash) {
        err = "model_hash: sidecar " + std::to_string(sc.model_hash) +
              " != model " + std::to_string(model_hash);
        return false;
    }
    return true;
}

}  // namespace ninfer::ops
