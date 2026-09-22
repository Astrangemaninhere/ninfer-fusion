#pragma once

// N3 runtime loop, solve half: turn the `.kvc` calibration frames that a capture
// run left behind into the BF16 words of a NINFERKVRS1 row-scale table.
//
// DOMAIN (the one thing that is easy to get wrong here).  The kernels apply the
// row scale AFTER the baked IsoQuant SO(4) rotation of each 4-channel block:
// gqa_attention_decode_nvfp4.cuh rotates (`gqa_nvfp4_load_rotate_4`), then
// `kx[j] *= gqa_kv_row_scale(layer, kv_head, grp * 16 + lane * 4 + j)`, then
// packs.  The captured K is the tensor the attention op CONSUMES, i.e. still
// PRE-rotation, so balancing the captured channels directly would balance the
// wrong quantity.  The solve therefore reconstructs the rotated values from the
// very matrix the kernels rotate with -- copied out of constant memory through
// kv_rowscale_device_rotation_matrix() (declared in the loader header, defined
// in gqa_isoquant_row_scale_loader.cu), never duplicated here, so the two cannot
// drift -- and uses the identity map when the device gate says the rotation is
// off, again as read back from the device rather than assumed from an option.
//
// FRAMES.  Producer: src/targets/qwen3_6/impl/runtime/kv_calibration.h.  Layout
// mirrored from the tree's existing reader (tools/calib/analyze_kv.py:149-168),
// which is the only other consumer:
//
//   0  64  header: "NINFERKVCAL1" | u32 header_bytes | u32 full_layer |
//                  u32 head_dim | u32 kv_heads | u32 tokens | u32 record_index |
//                  i32 first_position | i32 last_position | u32 reserved[4]
//  64 ..  i32  positions[tokens]
//         bf16 K[head_dim * kv_heads * tokens]
//         bf16 V[head_dim * kv_heads * tokens]
//
// and the K/V order is HEAD-DIM-FASTEST: channel d of kv_head h at token t is at
// flat index d + head_dim * (h + kv_heads * t).  The row-scale words come out in
// the pool's own order instead -- [layer][kv_head][d], i.e.
// (layer * kv_heads + h) * head_dim + d, which is what
// gqa_isoquant_row_scale.cuh indexes and what
// tools/kv_rowscale_sidecar.py writes.
//
// Host-only (no CUDA header): the whole solve is unit-testable with plain g++
// (tests/test_kv_rowscale_persist.cpp), with the two device readbacks stubbed.
//
// FAIL CLOSED.  Nothing here invents a table: a frame whose geometry disagrees is
// skipped (and named), a rotation domain that cannot be reconstructed is an
// ERROR, and a layer with no usable frame keeps the identity scale 1.0 -- logged,
// never guessed.

#include "ops/kernel/gqa_isoquant_row_scale_loader.h"
#include "product/kv_rowscale_frame_identity.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <string>
#include <system_error>
#include <vector>

namespace ninfer::product {

// The bake reproduces the kernels' pipeline, so it uses the kernels' constants:
// 4-channel blocks (gqa_isoquant_rot.cuh) and s_d in [0.5, 2.0]
// (gqa_isoquant_row_scale.cuh: "per-channel scale s_d in [0.5, 2.0] balances
// rotated K row RMS").
inline constexpr std::uint32_t kKvRowScaleBlockChannels = 4;
inline constexpr std::uint32_t kKvRowScaleRotationBlocks = 64;  // kGqaIsoquantRotDev[64]
inline constexpr float kKvRowScaleMin = 0.5F;
inline constexpr float kKvRowScaleMax = 2.0F;
// The geometric-mean normalisation is exact only up to the box clamp; four
// clamp/renormalise passes leave the product within ~1e-3 of unity, which is far
// below the BF16 grid the words land on.
inline constexpr int kKvRowScaleNormalizePasses = 4;

struct KvRowScaleBakeStats {
    std::uint64_t frames_read            = 0;
    std::uint64_t frames_used            = 0;
    std::uint64_t frames_skipped         = 0;
    std::uint64_t tokens_used            = 0;
    std::uint32_t layers_calibrated      = 0;
    std::uint32_t layers_identity        = 0;
    std::uint32_t heads_identity         = 0;
    std::uint32_t channels_identity      = 0;
    // max over heads of max(rms_d)/min(rms_d), before and after the solve: the
    // number that says whether the table does anything at all.
    double worst_channel_range_before    = 1.0;
    double worst_channel_range_after     = 1.0;
    double scale_min                     = 1.0;
    double scale_max                     = 1.0;
    // exp(mean(ln s_d)) over every calibrated channel: 1.0 is "no net gain".
    double product_drift                 = 1.0;
    bool rotation_applied                = false;
    std::string first_skip_reason;
    // THE IDENTITY LEDGER. `frames_unnamed` is the count of frames whose 16 reserved
    // bytes are ALL ZERO -- every artifact written before the identity existed -- and
    // `frames_foreign` the count whose identity is present but is not this run's.
    // Neither is a skip: a directory that holds one of them is a directory this solve
    // cannot describe, and the gate at the end of kv_rowscale_solve_frames() refuses
    // the whole bake by name instead of producing a table for the layers that happened
    // to be stamped.
    std::uint64_t frames_unnamed         = 0;
    std::uint64_t frames_foreign         = 0;
    std::string   identity_refusal;      // the first one, named
    // What the CONSUMED frames say: the stamp the table must carry. The stamp used to
    // be taken from the run that WROTE the table (persist.h's finish), which is how a
    // mislabelled capture was laundered into the table's own key.
    std::uint64_t producer_fingerprint   = 0;
    std::uint32_t producer_version       = 0;
    bool          producer_seen          = false;
};

namespace detail {

// BF16 with round-to-nearest-even, bit-for-bit the same rule as
// tools/kv_rowscale_sidecar.py:float_to_bf16().
[[nodiscard]] inline std::uint16_t kv_rowscale_bf16_from_float(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const std::uint32_t rounded = (bits + 0x7FFFu + ((bits >> 16) & 1u)) >> 16;
    return static_cast<std::uint16_t>(rounded & 0xFFFFu);
}

[[nodiscard]] inline float kv_rowscale_float_from_bf16(std::uint16_t word) {
    const std::uint32_t bits = static_cast<std::uint32_t>(word) << 16;
    float out = 0.0F;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

// EPOCH.  `std::filesystem::file_time_type` is NOT a unix time.  Its clock is
// `file_clock`, and since GCC 13 that clock has a DIFFERENT EPOCH from
// `system_clock`, so the raw `time_since_epoch()` of a mod time is a large
// NEGATIVE number (measured here with g++ 15.2.0 / -std=c++20: -4648162546 for
// "now", a fixed offset of 6437663999 s).  The staleness floor this header is
// compared against -- `not_before_unix`, filled in by the CLI from
// `std::time(nullptr) - 1` -- IS a true unix second.  Comparing the raw value
// against it is therefore unconditionally TRUE, which skipped EVERY frame and
// made the default (unnamed `--kv-dtype`) domain unable to bake a table at all.
// `clock_cast` is the C++20 conversion across the two clocks.
//
// THIS IS THE ONLY ONE.  `detail::kv_rowscale_mtime_unix()` in
// `kv_rowscale_persist.h` -- the loop's other time reader -- no longer repeats
// this expression: it stats the path and then calls THIS function.  persist.h:78
// includes this header, so the dependency runs persist.h -> bake.h, the call is
// legal, and this header stays host-only and self-contained.  One definition, so
// the two readers cannot drift by construction instead of a test noticing later.
// Pre-C++20-`chrono` (and pre-GCC-13) the two clocks share an epoch and the raw
// value already IS a unix second, which is exactly why the fallback is the raw
// value rather than an error.
[[nodiscard]] inline std::int64_t kv_rowscale_file_time_unix(
    const std::filesystem::file_time_type& stamp) {
#if defined(__cpp_lib_chrono) && __cpp_lib_chrono >= 201907L
    return static_cast<std::int64_t>(
        std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::clock_cast<std::chrono::system_clock>(stamp).time_since_epoch())
            .count());
#else
    return static_cast<std::int64_t>(
        std::chrono::duration_cast<std::chrono::seconds>(stamp.time_since_epoch()).count());
#endif
}

// FNV-1a/64 folded to 48 bits, and its hex spelling.  THE ONLY DEFINITION: this
// header is the lower one (persist.h includes it), so the fingerprint's own text form
// lives here and persist.h's detail::kv_rowscale_hex48 delegates -- the same "one
// expression, so the two readers cannot drift" rule the file-clock conversion above
// already follows.
[[nodiscard]] inline std::uint64_t kv_rowscale_fnv1a48(const std::string& text) {
    std::uint64_t hash = 1469598103934665603ULL;
    for (const unsigned char byte : text) {
        hash ^= static_cast<std::uint64_t>(byte);
        hash *= 1099511628211ULL;
    }
    hash ^= hash >> 48;
    return hash & 0xFFFFFFFFFFFFULL;
}

[[nodiscard]] inline std::string kv_rowscale_hex48(std::uint64_t value) {
    char buffer[16];
    std::snprintf(buffer, sizeof(buffer), "%012llx",
                  static_cast<unsigned long long>(value & 0xFFFFFFFFFFFFULL));
    return std::string(buffer);
}

}  // namespace detail

// Solve one directory of frames into `words` (empty on failure), for a model of
// `layers` full-attention layers x `kv_heads` x `head_dim`.  `not_before_unix`
// drops frames older than the capture run that is being persisted (the record
// names restart at 0 on every run, so an mtime floor is what keeps a shared or
// reused directory from mixing two runs).  `not_before_unix` is a TRUE unix
// second (the CLI passes `std::time(nullptr) - 1`, apps/cli/main.cpp:499), and
// the frame side of the comparison is converted into the same unit by
// detail::kv_rowscale_file_time_unix(); a raw `file_time_type` reading is NOT a
// unix second and comparing one against this floor rejects everything.
//
// Every refusal names its cause; `stats` describes what was actually consumed so
// the caller can log it instead of claiming success.
// `expected_fingerprint` is THIS run's kv_rowscale_config_fingerprint, and it is a
// REQUIRED argument with no default on purpose: a default would be the wildcard this
// gate exists to remove.  A frame whose identity is absent (all-zero reserved), whose
// producer version is not this build's, whose flags carry a bit this build does not
// know, or whose fingerprint is not `expected_fingerprint` is REFUSED BY NAME; if any
// such frame was seen, the whole solve fails and NO table is produced.  A partial table
// over the frames that did match would silently fall back to the identity scale 1.0 for
// the layers it did not.
[[nodiscard]] inline bool kv_rowscale_solve_frames(const std::filesystem::path& directory,
                                                   std::uint32_t layers, std::uint32_t kv_heads,
                                                   std::uint32_t head_dim,
                                                   std::int64_t not_before_unix,
                                                   std::uint64_t expected_fingerprint,
                                                   std::vector<std::uint16_t>& words,
                                                   KvRowScaleBakeStats& stats, std::string& err) {
    words.clear();
    stats = KvRowScaleBakeStats{};
    if (layers == 0 || kv_heads == 0 || head_dim == 0) {
        err = "geometry: layers/kv_heads/head_dim must all be positive";
        return false;
    }
    if (head_dim % kKvRowScaleBlockChannels != 0) {
        err = "geometry: head_dim " + std::to_string(head_dim) + " is not a multiple of " +
              std::to_string(kKvRowScaleBlockChannels);
        return false;
    }

    // The rotation domain has to be reconstructed, not assumed.  A model wider
    // than the 64-block table would be rotated out of bounds by the kernels, so
    // the bake refuses it here instead of producing a table for a domain that
    // does not exist.
    bool rotation_enabled = false;
    try {
        rotation_enabled = ninfer::ops::kv_rowscale_device_rotation_enabled();
    } catch (const std::exception& error) {
        err = std::string("rotation gate: ") + error.what();
        return false;
    }
    std::vector<float> rotation(static_cast<std::size_t>(kKvRowScaleRotationBlocks) *
                                kKvRowScaleBlockChannels * kKvRowScaleBlockChannels);
    if (rotation_enabled) {
        if (head_dim / kKvRowScaleBlockChannels > kKvRowScaleRotationBlocks) {
            err = "rotation: head_dim " + std::to_string(head_dim) + " needs " +
                  std::to_string(head_dim / kKvRowScaleBlockChannels) +
                  " SO(4) blocks, the device table has " +
                  std::to_string(kKvRowScaleRotationBlocks);
            return false;
        }
        try {
            if (!ninfer::ops::kv_rowscale_device_rotation_matrix(rotation.data())) {
                err = "rotation: the device SO(4) matrix could not be read back";
                return false;
            }
        } catch (const std::exception& error) {
            err = std::string("rotation: ") + error.what();
            return false;
        }
    }
    stats.rotation_applied = rotation_enabled;

    std::error_code ec;
    if (!std::filesystem::is_directory(directory, ec)) {
        err = "frames: " + directory.string() + " is not a directory";
        return false;
    }
    std::vector<std::filesystem::path> frames;
    for (std::filesystem::directory_iterator it(directory, ec), end; !ec && it != end; it.increment(ec)) {
        if (it->is_regular_file(ec) && it->path().extension() == ".kvc") { frames.push_back(it->path()); }
    }
    if (frames.empty()) {
        err = "frames: no .kvc record in " + directory.string();
        return false;
    }
    // Deterministic order: the sum is order-dependent in principle only (it is
    // not, doubling sums are added per frame), but a stable traversal makes a
    // rerun on the same directory byte-identical, which is what the acceptance
    // test compares.
    std::sort(frames.begin(), frames.end());

    const std::size_t channels = static_cast<std::size_t>(kv_heads) * head_dim;
    std::vector<double> energy(static_cast<std::size_t>(layers) * channels, 0.0);
    std::vector<std::uint64_t> tokens_per_layer(layers, 0);

    for (const std::filesystem::path& frame : frames) {
        ++stats.frames_read;
        if (not_before_unix > 0) {
            std::int64_t mtime = 0;
            const auto stamp = std::filesystem::last_write_time(frame, ec);
            if (ec) { ++stats.frames_skipped; continue; }
            mtime = detail::kv_rowscale_file_time_unix(stamp);
            if (mtime < not_before_unix) {
                ++stats.frames_skipped;
                if (stats.first_skip_reason.empty()) {
                    stats.first_skip_reason = frame.filename().string() + ": older than this run";
                }
                continue;
            }
        }
        std::string blob;
        {
            std::FILE* file = std::fopen(frame.string().c_str(), "rb");
            if (file == nullptr) { ++stats.frames_skipped; continue; }
            char buffer[65536];
            std::size_t got = 0;
            while ((got = std::fread(buffer, 1, sizeof(buffer), file)) > 0) { blob.append(buffer, got); }
            std::fclose(file);
        }
        if (blob.size() < 64) {
            ++stats.frames_skipped;
            if (stats.first_skip_reason.empty()) {
                stats.first_skip_reason = frame.filename().string() + ": shorter than the 64-byte header";
            }
            continue;
        }
        const auto* p = reinterpret_cast<const unsigned char*>(blob.data());
        // "NINFERKVCAL1" is 12 bytes (kv_calibration.h kMagic: the eleven
        // letters, the '1', then four NULs), UNLIKE the row-scale sidecar's
        // "NINFERKVRS1" which is 11 + 5. Getting this wrong makes the bake refuse
        // every real frame, which is why the cross-check against the tree's own
        // reader (tools/calib/analyze_kv.py) is part of the acceptance.
        if (std::memcmp(p, "NINFERKVCAL1", 12) != 0 || p[12] | p[13] | p[14] | p[15]) {
            ++stats.frames_skipped;
            if (stats.first_skip_reason.empty()) {
                stats.first_skip_reason = frame.filename().string() + ": not a NINFERKVCAL1 record";
            }
            continue;
        }
        const std::uint32_t header_bytes = ninfer::ops::detail::le_u32(p + 16);
        const std::uint32_t layer        = ninfer::ops::detail::le_u32(p + 20);
        const std::uint32_t frame_dim    = ninfer::ops::detail::le_u32(p + 24);
        const std::uint32_t frame_heads  = ninfer::ops::detail::le_u32(p + 28);
        const std::uint32_t tokens       = ninfer::ops::detail::le_u32(p + 32);
        // The identity: the 16 bytes the header has always reserved, at the offsets
        // product/kv_rowscale_frame_identity.h pins.
        std::uint64_t frame_identity = 0;
        for (int i = 0; i < 8; ++i) {
            frame_identity |= static_cast<std::uint64_t>(p[ninfer::product::kKvRowScaleFrameIdentityOffset + i])
                              << (8 * i);
        }
        const std::uint32_t frame_version =
            ninfer::ops::detail::le_u32(p + ninfer::product::kKvRowScaleFrameVersionOffset);
        const std::uint32_t frame_flags =
            ninfer::ops::detail::le_u32(p + ninfer::product::kKvRowScaleFrameFlagsOffset);
        const auto note_skip             = [&](const std::string& reason) {
            ++stats.frames_skipped;
            if (stats.first_skip_reason.empty()) {
                stats.first_skip_reason = frame.filename().string() + ": " + reason;
            }
        };
        // THE IDENTITY GATE, before every geometric one: a frame that cannot say
        // WHO produced it is not a frame this solve may use, whatever its shape.
        const auto refuse_frame = [&](const std::string& reason) {
            ++stats.frames_skipped;
            if (stats.identity_refusal.empty()) {
                stats.identity_refusal = frame.filename().string() + ": " + reason;
            }
            if (stats.first_skip_reason.empty()) {
                stats.first_skip_reason = frame.filename().string() + ": " + reason;
            }
        };
        if (frame_identity == 0 && frame_version == 0 && frame_flags == 0) {
            ++stats.frames_unnamed;
            refuse_frame("carries NO configuration identity: the 16 reserved bytes of the .kvc "
                         "header are all zero, so this frame was written before the identity "
                         "field existed or by a producer that does not stamp one; re-run the "
                         "capture with this build, or clear " + directory.string() +
                         " if the frames are stale");
            continue;
        }
        if (frame_version != ninfer::product::kKvRowScaleFrameIdentityVersion) {
            ++stats.frames_foreign;
            refuse_frame("was captured by producer version " + std::to_string(frame_version) +
                         ", this build reads version " +
                         std::to_string(ninfer::product::kKvRowScaleFrameIdentityVersion) +
                         "; re-run the capture with this build");
            continue;
        }
        if ((frame_flags & ~ninfer::product::kKvRowScaleFrameKnownFlags) != 0) {
            ++stats.frames_foreign;
            refuse_frame("carries unknown flags 0x" + std::to_string(frame_flags) +
                         " (known mask 0x" +
                         std::to_string(ninfer::product::kKvRowScaleFrameKnownFlags) +
                         "): an old reader must never accept a newer frame's semantics");
            continue;
        }
        if (frame_identity != expected_fingerprint) {
            ++stats.frames_foreign;
            refuse_frame("was captured for KV configuration " +
                         detail::kv_rowscale_hex48(frame_identity) + ", this run is " +
                         detail::kv_rowscale_hex48(expected_fingerprint) +
                         "; a table solved from it would be mislabelled");
            continue;
        }
        if (!stats.producer_seen) {
            stats.producer_seen        = true;
            stats.producer_fingerprint = frame_identity;
            stats.producer_version     = frame_version;
        }
        if (header_bytes != 64) { note_skip("header_bytes " + std::to_string(header_bytes)); continue; }
        if (layer >= layers) { note_skip("layer " + std::to_string(layer) + " outside 0.." +
                                         std::to_string(layers - 1)); continue; }
        if (frame_dim != head_dim || frame_heads != kv_heads) {
            note_skip("geometry " + std::to_string(frame_dim) + "x" + std::to_string(frame_heads) +
                      " != model " + std::to_string(head_dim) + "x" + std::to_string(kv_heads));
            continue;
        }
        if (tokens == 0) { note_skip("zero tokens"); continue; }
        const std::uint64_t plane =
            static_cast<std::uint64_t>(head_dim) * kv_heads * tokens;
        const std::uint64_t wanted = 64ULL + 4ULL * tokens + 4ULL * plane;
        if (blob.size() != wanted) {
            note_skip("size " + std::to_string(blob.size()) + " != " + std::to_string(wanted));
            continue;
        }
        const auto* k_words = reinterpret_cast<const unsigned char*>(blob.data() + 64 + 4ULL * tokens);
        const auto read = [](const unsigned char* plane_words, std::uint64_t index) {
            return static_cast<std::uint16_t>(plane_words[2 * index] |
                                              (plane_words[2 * index + 1] << 8));
        };
        const std::uint64_t blocks = head_dim / kKvRowScaleBlockChannels;
        for (std::uint32_t head = 0; head < kv_heads; ++head) {
            double* head_energy = energy.data() + static_cast<std::size_t>(layer) * channels +
                                  static_cast<std::size_t>(head) * head_dim;
            for (std::uint64_t token = 0; token < tokens; ++token) {
                const std::uint64_t base =
                    (static_cast<std::uint64_t>(head) + static_cast<std::uint64_t>(kv_heads) * token) *
                    head_dim;
                for (std::uint64_t block = 0; block < blocks; ++block) {
                    float x[kKvRowScaleBlockChannels];
                    for (std::uint32_t j = 0; j < kKvRowScaleBlockChannels; ++j) {
                        x[j] = detail::kv_rowscale_float_from_bf16(read(k_words, base + block * 4 + j));
                    }
                    float y[kKvRowScaleBlockChannels];
                    if (rotation_enabled) {
                        const std::size_t matrix = static_cast<std::size_t>(block) * 16;
                        for (std::uint32_t row = 0; row < kKvRowScaleBlockChannels; ++row) {
                            y[row] = rotation[matrix + row * 4 + 0] * x[0] +
                                     rotation[matrix + row * 4 + 1] * x[1] +
                                     rotation[matrix + row * 4 + 2] * x[2] +
                                     rotation[matrix + row * 4 + 3] * x[3];
                        }
                    } else {
                        for (std::uint32_t j = 0; j < kKvRowScaleBlockChannels; ++j) { y[j] = x[j]; }
                    }
                    for (std::uint32_t j = 0; j < kKvRowScaleBlockChannels; ++j) {
                        const double value = static_cast<double>(y[j]);
                        head_energy[block * 4 + j] += value * value;
                    }
                }
            }
        }
        tokens_per_layer[layer] += tokens;
        stats.tokens_used += tokens;
        ++stats.frames_used;
    }

    // FAIL CLOSED ON A FRAME THAT CANNOT SAY WHO PRODUCED IT.  A directory that holds
    // one stamped and one unnamed frame is not a directory this solve can describe: the
    // unnamed one is either a leftover from an older capture or a frame from another
    // configuration, and the layers it covered would silently come out as the identity
    // scale.  No table, a named reason, and the fix in the message -- the alternative
    // (skip it and solve the rest) is exactly the "wrong result that looks right" the
    // identity field exists to make impossible.
    if (stats.frames_unnamed > 0 || stats.frames_foreign > 0) {
        err = "frames: " + std::to_string(stats.frames_unnamed) + " of the " +
              std::to_string(stats.frames_read) + " record(s) in " + directory.string() +
              " carry no configuration identity and " + std::to_string(stats.frames_foreign) +
              " carry an identity that is not this run's; a table solved from a mixed "
              "directory would be mislabelled, so none was produced" +
              (stats.identity_refusal.empty() ? std::string()
                                              : " (" + stats.identity_refusal + ")");
        return false;
    }

    if (stats.frames_used == 0) {
        err = "frames: none of the " + std::to_string(stats.frames_read) +
              " record(s) in " + directory.string() + " could be used" +
              (stats.first_skip_reason.empty() ? std::string()
                                               : " (" + stats.first_skip_reason + ")");
        return false;
    }

    // The solve.  Per (layer, kv_head) the channels are balanced to a common
    // geometric mean: s_d = gm(rms) / rms_d, clamped into the kernel's box and
    // renormalised so that prod(s) == 1, i.e. no net gain in K (a global factor
    // would cancel in QK^T anyway, but it would move the quantization grid).
    words.assign(static_cast<std::size_t>(layers) * channels, 0);
    double log_sum = 0.0;
    std::uint64_t scaled_channels = 0;
    double scale_min = 0.0;
    double scale_max = 0.0;
    for (std::uint32_t layer = 0; layer < layers; ++layer) {
        if (tokens_per_layer[layer] == 0) {
            ++stats.layers_identity;
            for (std::size_t index = 0; index < channels; ++index) {
                words[static_cast<std::size_t>(layer) * channels + index] = 0x3F80;  // BF16 1.0
            }
            continue;
        }
        ++stats.layers_calibrated;
        const double tokens = static_cast<double>(tokens_per_layer[layer]);
        for (std::uint32_t head = 0; head < kv_heads; ++head) {
            const std::size_t base = static_cast<std::size_t>(layer) * channels +
                                     static_cast<std::size_t>(head) * head_dim;
            std::vector<double> rms(head_dim, 0.0);
            double log_sum_head = 0.0;
            std::uint32_t finite = 0;
            for (std::uint32_t d = 0; d < head_dim; ++d) {
                rms[d] = std::sqrt(energy[base + d] / tokens);
                if (std::isfinite(rms[d]) && rms[d] > 0.0) {
                    log_sum_head += std::log(rms[d]);
                    ++finite;
                }
            }
            if (finite == 0) {
                ++stats.heads_identity;
                for (std::uint32_t d = 0; d < head_dim; ++d) {
                    words[(base + d)] = 0x3F80;
                    ++stats.channels_identity;
                }
                continue;
            }
            const double target = std::exp(log_sum_head / static_cast<double>(finite));
            double low = 0.0;
            double high = 0.0;
            bool first = true;
            std::vector<double> scale(head_dim, 1.0);
            for (std::uint32_t d = 0; d < head_dim; ++d) {
                if (!std::isfinite(rms[d]) || rms[d] <= 0.0) {
                    scale[d] = 1.0;
                    ++stats.channels_identity;
                    continue;
                }
                scale[d] = target / rms[d];
                if (first || scale[d] < low) { low = scale[d]; }
                if (first || scale[d] > high) { high = scale[d]; }
                first = false;
            }
            double range_before = (low > 0.0) ? high / low : 1.0;
            // The renormalisation runs over the channels still INSIDE the box.  A
            // channel whose ideal scale is outside it is a channel the box itself
            // forbids balancing; letting it drag the geometric mean would inflate
            // every free channel until it hit the opposite bound and the table
            // would stop being a balance at all (measured: a 4x spread converged
            // to s = [2.0, 0.57, 2.0, 0.57] instead of the intended shape).
            for (int pass = 0; pass < kKvRowScaleNormalizePasses; ++pass) {
                double log_sum_scale = 0.0;
                std::uint32_t free_channels = 0;
                for (std::uint32_t d = 0; d < head_dim; ++d) {
                    if (scale[d] > kKvRowScaleMin && scale[d] < kKvRowScaleMax) {
                        log_sum_scale += std::log(scale[d]);
                        ++free_channels;
                    }
                }
                if (free_channels == 0) { break; }
                const double correction =
                    std::exp(-log_sum_scale / static_cast<double>(free_channels));
                if (std::abs(correction - 1.0) < 1e-12) { break; }
                for (std::uint32_t d = 0; d < head_dim; ++d) {
                    scale[d] = std::min<double>(
                        kKvRowScaleMax,
                        std::max<double>(kKvRowScaleMin, scale[d] * correction));
                }
            }
            double range_after = 0.0;
            double low_after = 0.0;
            double high_after = 0.0;
            first = true;
            for (std::uint32_t d = 0; d < head_dim; ++d) {
                const double scaled = (rms[d] > 0.0 && std::isfinite(rms[d])) ? rms[d] * scale[d] : 0.0;
                if (scaled > 0.0) {
                    if (first || scaled < low_after) { low_after = scaled; }
                    if (first || scaled > high_after) { high_after = scaled; }
                    first = false;
                }
                words[base + d] = detail::kv_rowscale_bf16_from_float(static_cast<float>(scale[d]));
                log_sum += std::log(scale[d]);
                ++scaled_channels;
                if (scale[d] < scale_min || scale_min == 0.0) { scale_min = scale[d]; }
                if (scale[d] > scale_max) { scale_max = scale[d]; }
            }
            range_after = (low_after > 0.0) ? high_after / low_after : 1.0;
            if (range_before > stats.worst_channel_range_before) {
                stats.worst_channel_range_before = range_before;
            }
            if (range_after > stats.worst_channel_range_after) {
                stats.worst_channel_range_after = range_after;
            }
        }
    }
    if (scaled_channels == 0) {
        err = "solve: every channel was degenerate; no table to write";
        return false;
    }
    stats.scale_min = scale_min;
    stats.scale_max = scale_max;
    stats.product_drift = std::exp(log_sum / static_cast<double>(scaled_channels));
    return true;
}

}  // namespace ninfer::product
