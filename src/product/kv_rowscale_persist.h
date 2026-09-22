#pragma once

// N3 runtime loop, policy half: the PERSISTED KV row-scale table.
//
// WHAT WAS MISSING.  The three pieces that already existed were the offline half
// of a loop: an offline capture (kv_calibration.h writes .kvc frames), an offline
// bake into the engine's constant pool (gqa_isoquant_row_scale.cu, round-tripped
// by tools/kv_rowscale_sidecar.py), and a three-state switch (auto|off|<path>,
// gqa_isoquant_row_scale_loader.h) that could only be pointed at a table an
// OPERATOR had baked by hand.  Nothing decided WHEN to calibrate, nothing
// remembered that it had been done, and nothing noticed that the remembered
// table no longer belongs to the model or the KV configuration in front of it.
// This header is that half:
//
//   1. one-shot      no usable table -> arm the existing capture for THIS run and
//                    write the table at the end of it (kv_rowscale_persist_finish);
//   2. skip          a usable table -> load it (through the unchanged loader) and
//                    say so, including the file's age; the capture never runs;
//   3. --recalibrate ignore the table, capture again, OVERWRITE it;
//   4. invalidation  the table applies only while its producer schema, its KV
//                    configuration fingerprint, the live KV geometry and the
//                    artifact's mtime all still agree.  Any disagreement is a
//                    CAPTURE, never a silent reuse of a table baked for something
//                    else.
//
// WHERE THE STATE LIVES.  On disk, next to the artifact, under the path the
// design note already fixed (research/notes/TODO.md §122, "旁车表文件
// <artifact>.kvrowscale.bin"):
//
//   <artifact>.kvrowscale.bin          the table: a byte-for-byte NINFERKVRS1
//                                      file, so tools/kv_rowscale_sidecar.py
//                                      validates it and the engine's loader loads
//                                      it with no format change at all
//   <artifact>.kvrowscale.bin.d        the capture records of a recalibration run
//   <artifact>.kvrowscale.bin.skip     "nothing to calibrate for this
//                                      configuration" (fingerprint + reason):
//                                      stops the loop from arming a capture for a
//                                      model whose KV tier has no rotated /
//                                      quantized domain at all
//   <artifact>.kvrowscale.bin.rejected the previous table, renamed aside when it
//                                      could not be applied (see resolve)
//
// PRECEDENCE IS UNCHANGED.  An explicit `--kv-row-scale off|<path>` still wins
// over everything, an explicit `--kv-row-scale auto` still wins over
// NINFER_KV_ROWSCALE, and NINFER_KV_ROWSCALE still wins over the loop.  The loop
// owns only the case that had no owner: mode auto with neither an explicit file
// nor the environment variable set.
//
// WHY THE APPLICABILITY GATE LOOKS LIKE THIS.  The NINFERKVRS1 header is exactly
// 64 bytes with no spare field (magic[16] + 6 u32 + u64 + tag[16]), so the two
// fingerprints that are not already in it -- the KV configuration the table was
// baked for, and the producer's schema version -- ride in `tag`, whose 16 bytes
// the loader itself documents as "carried for self-description only":
//
//   tag[16] = "rs1." + 12 hex digits of the KV configuration fingerprint
//
// "rs1." is the producer schema (a table without it was produced by something
// else, e.g. tools/kv_rowscale_sidecar.py by hand, and the loop will not adopt
// it).  Model identity is the geometry the loader already enforces exactly
// (layers/kv_heads/head_dim via kv_rowscale_sidecar_check) plus the artifact's
// mtime, which is what catches "the table is older than the model it sits next
// to".  The format's own model_hash field is passed through untouched: the loop
// writes 0 (unspecified) and honours whatever the caller supplies, so an
// externally stamped table still gets the exact gate.  Computing sha256 over a
// 21-54 GB artifact was measured at ~540 MB/s on this machine (40+ s), which
// would make every startup with a valid table slower for no added coverage here.
//
// ORDERING CONTRACT (same shape as the loader's own process-global contract).
// kv_rowscale_persist_begin() must run before the first prefill of the process:
// it installs NINFER_KV_CALIB_DIR if it is not already set, and the capture reads
// that variable once, at its first call.  One row-scale mode per process, as
// before.
//
// Host-only (no CUDA header): the whole decision path is unit-testable with plain
// g++ (tests/test_kv_rowscale_persist.cpp).

#include "ops/kernel/gqa_isoquant_row_scale_loader.h"
#include "product/kv_rowscale_bake.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace ninfer::product {

inline constexpr char kKvRowScaleTableSuffix[] = ".kvrowscale.bin";
inline constexpr char kKvRowScaleTagPrefix[] = "rs1.";
inline constexpr std::size_t kKvRowScaleTagPrefixLen = 4;
inline constexpr std::size_t kKvRowScaleTagDigits = 12;

// What the loop decided, for one run.
enum class KvRowScalePlan {
    Disabled,             // loop off: explicit spec, or the environment owns it
    UsePersisted,         // a valid table was found; capture skipped
    Capture,              // no valid table; capture this run and persist at the end
    NothingToCalibrate,   // no rotated/quantized KV domain in this configuration
};

[[nodiscard]] inline const char* kv_rowscale_plan_name(KvRowScalePlan plan) {
    switch (plan) {
    case KvRowScalePlan::Disabled: return "disabled";
    case KvRowScalePlan::UsePersisted: return "use-persisted-table";
    case KvRowScalePlan::Capture: return "capture";
    case KvRowScalePlan::NothingToCalibrate: return "nothing-to-calibrate";
    }
    return "unknown";
}

// The operator-visible KV configuration the table is baked FOR.  Every knob that
// changes what the rotated K domain is quantized to has to be in here: a table
// baked under one configuration and reused under another is the silent
// wrong-budget failure the loop exists to prevent.
struct KvRowScaleConfigKnobs {
    int kv_cache_code = 0;
    bool kv_cache_explicit = false;
    std::string layer_storage_spec;
    std::string tier_formats_spec;
    bool nvfp4_pure = false;
    bool rotation_off = false;
    bool rotation_explicit = false;
    int v_codec = 0;
    double bit_budget_bits = 0.0;
    std::string bit_budget_ranges;
    bool bit_budget_explicit = false;
    double quality_weight = -1.0;
    std::string tier_scores;
    // The K/V bit-width entries (product/kv_kv_bits.h). A K ceiling changes what the
    // rotated K domain is quantized to, so it belongs in the fingerprint for the same
    // reason bit_budget_bits does.
    double joint_bits = 0.0;
    double k_bits = 0.0;
    double v_bits = 0.0;
    int kv_bits_mode = 1;
    bool kv_bits_explicit = false;
    std::string k_tier_scores;
    std::string v_tier_scores;
    // SLIDERWIRE: --kv-codec-preference, flattened to the named ladder slots. It changes
    // which per-layer dtype table the fit emits, so it belongs in the fingerprint for the
    // same reason joint_bits does. Empty == the shipped order, and the fingerprint mixes it
    // in ONLY when set, so a table baked by a run that named no preference keeps its hash.
    std::string codec_preference;
    // ROPE REGIME: --yarn.  The row scale is solved from K that has ALREADY been through
    // the rope (bake.h's DOMAIN paragraph), so two runs whose rope regimes differ are two
    // different rotated domains and neither's table may validate for the other.  This was
    // the last operator-visible knob that reaches the captured K and was not in here.
    //
    // MIXED IN ONLY WHEN NOT DEFAULT (see the fingerprint below), exactly like
    // codec_preference: the default regime appends NOTHING, so every table baked before
    // this field existed keeps its historical hash and therefore its validity.
    int rope_regime = 0;  // 0 = the model's own rope; 1 = --yarn (static factor-4)
    int cold_policy = 0;
    std::uint32_t max_cold_pages = 0;
};

namespace detail {

}  // namespace detail

// FNV-1a/64, folded to 48 bits so the whole value fits the 12 tag digits, and its hex
// spelling.  DELEGATES to product/kv_rowscale_bake.h (included at the top), which now
// holds the only definition: this header is the upper one, and the frame side of the
// solve needs the same two functions, so keeping a second copy here is precisely the
// drift the clock conversion above was collapsed for.  The values are unchanged -- the
// test pins the default fingerprint against the historical constant (0xf3b77882c97d).
using detail::kv_rowscale_fnv1a48;
using detail::kv_rowscale_hex48;

// Canonical text, then one hash.  snprintf (not iostreams) keeps the spelling of
// the doubles locale- and library-independent, so the fingerprint of a given
// command line is the same on every machine that reads a table.
[[nodiscard]] inline std::uint64_t kv_rowscale_config_fingerprint(const KvRowScaleConfigKnobs& knobs) {
    char buffer[512];
    std::string text;
    std::snprintf(buffer, sizeof(buffer), "kv=%d/%d;", knobs.kv_cache_code,
                  knobs.kv_cache_explicit ? 1 : 0);
    text += buffer;
    text += "ls=" + knobs.layer_storage_spec + ";";
    text += "tf=" + knobs.tier_formats_spec + ";";
    std::snprintf(buffer, sizeof(buffer), "pure=%d;rot=%d/%d;vc=%d;", knobs.nvfp4_pure ? 1 : 0,
                  knobs.rotation_off ? 1 : 0, knobs.rotation_explicit ? 1 : 0, knobs.v_codec);
    text += buffer;
    std::snprintf(buffer, sizeof(buffer), "bb=%.6f;", knobs.bit_budget_bits);
    text += buffer;
    text += "bbr=" + knobs.bit_budget_ranges + ";";
    std::snprintf(buffer, sizeof(buffer), "bbe=%d;qw=%.6f;", knobs.bit_budget_explicit ? 1 : 0,
                  knobs.quality_weight);
    text += buffer;
    text += "ts=" + knobs.tier_scores + ";";
    // The K/V bit widths are mixed in ONLY when set, so an existing table keeps its
    // old hash and therefore its validity: a run that names no K/V ceiling must not
    // invalidate a table baked by a run that named none either.
    if (knobs.kv_bits_explicit) {
        std::snprintf(buffer, sizeof(buffer), "jb=%.6f;kb=%.6f;vb=%.6f;km=%d;",
                      knobs.joint_bits, knobs.k_bits, knobs.v_bits, knobs.kv_bits_mode);
        text += buffer;
        text += "kts=" + knobs.k_tier_scores + ";vts=" + knobs.v_tier_scores + ";";
        // SLIDERWIRE: ONLY when set -- an unset preference must not move any existing hash.
        if (!knobs.codec_preference.empty()) {
            text += "cp=" + knobs.codec_preference + ";";
        }
    }
    std::snprintf(buffer, sizeof(buffer), "cold=%d/%u", knobs.cold_policy, knobs.max_cold_pages);
    text += buffer;
    // ROPE REGIME, APPENDED and ONLY when it is not the default.  Appending is what makes
    // "a default run's text is byte-identical to what it was before this arm existed"
    // true without a reordering hazard, and the "only when set" rule is what makes every
    // existing table's tag -- and therefore its validity -- survive this change.  The
    // price is that the default and the --yarn text are not prefix-free in the middle of
    // the string; they do not need to be, the hash is taken over the whole text.
    if (knobs.rope_regime != 0) {
        std::snprintf(buffer, sizeof(buffer), ";rope=%d", knobs.rope_regime);
        text += buffer;
    }
    return detail::kv_rowscale_fnv1a48(text);
}

// CAN THIS CONFIGURATION EVER BE CALIBRATED?  The row scale is read by the NVFP4
// K/V kernels only (product/kv_component_switch.h, KvComponentSwitch::RowScale:
// the gate is honoured for DType::NVFP4 and for nothing else).  A configuration
// in which NO full-attention layer resolves to the NVFP4 tier has no rotated
// domain to measure, so kv_rowscale_persist_resolve() writes the .skip note and
// returns NothingToCalibrate INSTEAD of arming a capture (this file, the
// !domain_active early return), and kv_rowscale_persist_finish() then returns
// without writing anything because the plan is not Capture.
//
// WHY THE PRE-FLIGHT NEEDS THIS.  kv_rowscale_persist_begin() decides Capture
// from the fingerprint alone (no geometry exists yet), and the CLI used to turn
// CUDA graphs OFF for the whole run on that basis.  For a configuration with no
// NVFP4 layer that is a pure loss: the capture is never armed, no table can ever
// be produced for it (not by --recalibrate either: finish() requires Capture),
// and the operator pays a graphs-off run for a calibration that is impossible.
//
// WHAT IS DECIDABLE HERE.  Only the shape that is decided by one knob: an
// operator who pinned EVERY layer to one global tier with --kv-dtype and gave no
// per-layer table (--kv-layer-storage), no --kv-tier-formats and no ceiling.
// apps/cli/options.cpp states the rule this relies on -- "--kv-dtype ... fills
// every slot with one global tier" -- and it also makes the three excluded knobs
// mutually exclusive with --kv-dtype in two of the three cases; the guard is
// written out anyway so that a future option cannot silently widen this.
//
// `global_tier_is_nvfp4` is the CALLER's reading of its own storage enum, so this
// header keeps no second DType mapping.  Every shape this cannot decide returns
// TRUE ("maybe"), and the caller keeps disabling the graphs there: an
// undecidable configuration behaves exactly as it did before this predicate
// existed.
[[nodiscard]] inline bool kv_rowscale_config_can_calibrate(const KvRowScaleConfigKnobs& knobs,
                                                          bool global_tier_is_nvfp4) {
    // --nvfp4-mode pure forbids the NVFP4 tier outright, whatever else is named.
    if (knobs.nvfp4_pure) { return false; }
    if (!knobs.kv_cache_explicit) { return true; }
    if (!knobs.layer_storage_spec.empty()) { return true; }
    if (!knobs.tier_formats_spec.empty()) { return true; }
    if (knobs.bit_budget_explicit || knobs.kv_bits_explicit) { return true; }
    return global_tier_is_nvfp4;
}

[[nodiscard]] inline std::string kv_rowscale_make_tag(std::uint64_t fingerprint) {
    return std::string(kKvRowScaleTagPrefix) + detail::kv_rowscale_hex48(fingerprint);
}

// Reads the schema + fingerprint back out of a tag.  False when the table was not
// written by this producer, which is a refusal, not a fallback.
[[nodiscard]] inline bool kv_rowscale_tag_fingerprint(const std::string& tag, std::uint64_t& out) {
    if (tag.size() != kKvRowScaleTagPrefixLen + kKvRowScaleTagDigits ||
        tag.compare(0, kKvRowScaleTagPrefixLen, kKvRowScaleTagPrefix) != 0) {
        return false;
    }
    std::uint64_t value = 0;
    for (std::size_t i = kKvRowScaleTagPrefixLen; i < tag.size(); ++i) {
        const char c = tag[i];
        std::uint64_t digit = 0;
        if (c >= '0' && c <= '9') { digit = static_cast<std::uint64_t>(c - '0'); }
        else if (c >= 'a' && c <= 'f') { digit = static_cast<std::uint64_t>(c - 'a' + 10); }
        else { return false; }
        value = value * 16ULL + digit;
    }
    out = value;
    return true;
}

[[nodiscard]] inline std::filesystem::path kv_rowscale_table_path(const std::filesystem::path& artifact) {
    return std::filesystem::path(artifact.string() + kKvRowScaleTableSuffix);
}

[[nodiscard]] inline std::filesystem::path kv_rowscale_records_path(const std::filesystem::path& table) {
    return std::filesystem::path(table.string() + ".d");
}

[[nodiscard]] inline std::filesystem::path kv_rowscale_skip_note_path(const std::filesystem::path& table) {
    return std::filesystem::path(table.string() + ".skip");
}

[[nodiscard]] inline std::filesystem::path kv_rowscale_rejected_path(const std::filesystem::path& table) {
    return std::filesystem::path(table.string() + ".rejected");
}

// The note the authoritative half found CONTRADICTED (see kv_rowscale_persist_resolve):
// moved aside, never deleted, exactly as `.rejected` is for a table.  The note is the
// only durable record of WHY the loop once concluded that this KV configuration has
// nothing to calibrate, so losing it would lose the reason, not just the file.
[[nodiscard]] inline std::filesystem::path kv_rowscale_skip_note_stale_path(const std::filesystem::path& table) {
    return std::filesystem::path(table.string() + ".skip.stale");
}

// True when the spec leaves the decision to the loop (the loader's own parser is
// the single definition of "auto", so the CLI cannot drift from it).
[[nodiscard]] inline bool kv_rowscale_spec_is_auto(const std::string& spec) {
    ninfer::ops::KvRowScaleMode mode = ninfer::ops::KvRowScaleMode::Auto;
    std::string path;
    std::string err;
    if (!ninfer::ops::kv_rowscale_mode_from_spec(spec, mode, path, err)) { return false; }
    return mode == ninfer::ops::KvRowScaleMode::Auto;
}

namespace detail {

// std::filesystem's file clock is NOT the system clock: since GCC 13 the two
// have different epochs, so the raw time_since_epoch() of a mod time is not a unix
// time at all (measured here: -4648429486 for "now"). clock_cast is the C++20 way
// across; without it the fallback still ORDERS correctly, which is all the
// staleness gate needs, but a date cannot be printed.
//
// ONE CONVERSION IN THE PRODUCT.  This is only the path-taking wrapper; the
// conversion is `detail::kv_rowscale_file_time_unix()` in
// product/kv_rowscale_bake.h (included at line 78), which is now the ONLY
// clock_cast for file_clock in src/. It delegates instead of repeating the
// expression so that "the two readers agree" is structural and not something a
// test has to police: there is no second expression left to drift. The contract
// here is unchanged (a path in, 0 on stat failure). The one thing that does
// change is the `#else` arm, which is not reachable in this build (measured
// __cpp_lib_chrono = 201907, so the gate above is true): it used to return raw
// file_clock ticks -- the clock's period is 1/1000000000, i.e. NANOSECONDS -- and
// now inherits bake.h's duration_cast<seconds>, which is the unit this function is
// documented to return. Both consumers below are unaffected either way: 495/496
// compare two values from THIS function (order-preserving under a constant factor)
// and 594 only formats a date.
[[nodiscard]] inline std::int64_t kv_rowscale_mtime_unix(const std::filesystem::path& path) {
    std::error_code ec;
    const auto stamp = std::filesystem::last_write_time(path, ec);
    if (ec) { return 0; }
    return kv_rowscale_file_time_unix(stamp);
}

[[nodiscard]] inline std::int64_t kv_rowscale_now_unix() {
    return static_cast<std::int64_t>(std::time(nullptr));
}

[[nodiscard]] inline std::string kv_rowscale_time_text(std::int64_t unix_seconds) {
    if (unix_seconds <= 0) { return "unknown"; }
    const std::time_t stamp = static_cast<std::time_t>(unix_seconds);
    std::tm parts{};
#if defined(_WIN32)
    if (gmtime_s(&parts, &stamp) != 0) { return "unknown"; }
#else
    if (gmtime_r(&stamp, &parts) == nullptr) { return "unknown"; }
#endif
    char buffer[96];
    std::snprintf(buffer, sizeof(buffer), "%04d-%02d-%02d %02d:%02d:%02dZ", parts.tm_year + 1900,
                  parts.tm_mon + 1, parts.tm_mday, parts.tm_hour, parts.tm_min, parts.tm_sec);
    return std::string(buffer);
}

[[nodiscard]] inline bool kv_rowscale_read_file(const std::filesystem::path& path, std::string& blob) {
    blob.clear();
    std::FILE* file = std::fopen(path.string().c_str(), "rb");
    if (file == nullptr) { return false; }
    char buffer[65536];
    std::size_t got = 0;
    while ((got = std::fread(buffer, 1, sizeof(buffer), file)) > 0) { blob.append(buffer, got); }
    std::fclose(file);
    return !blob.empty();
}

// temp + rename: a table that is being read by another process is never observed
// half written, and a failed write leaves the previous table in place.
[[nodiscard]] inline bool kv_rowscale_write_file_atomic(const std::filesystem::path& path,
                                                       const std::string& blob, std::string& err) {
    const std::filesystem::path temporary = std::filesystem::path(path.string() + ".tmp");
    {
        std::FILE* file = std::fopen(temporary.string().c_str(), "wb");
        if (file == nullptr) {
            err = "write: cannot create " + temporary.string();
            return false;
        }
        const std::size_t wrote = std::fwrite(blob.data(), 1, blob.size(), file);
        const int closed = std::fclose(file);
        if (wrote != blob.size() || closed != 0) {
            std::error_code ignored;
            std::filesystem::remove(temporary, ignored);
            err = "write: short write to " + temporary.string();
            return false;
        }
    }
    std::error_code ec;
    std::filesystem::rename(temporary, path, ec);
    if (ec) {
        err = "write: rename " + temporary.string() + " -> " + path.string() + ": " + ec.message();
        return false;
    }
    return true;
}

}  // namespace detail

// Assemble the table.  Byte-for-byte the same file tools/kv_rowscale_sidecar.py
// build() produces for the same words/geometry/tag/hash (the acceptance test
// compares the two byte strings, not just the parsed fields).
[[nodiscard]] inline bool kv_rowscale_build_table(const std::vector<std::uint16_t>& words,
                                                  std::uint32_t layers, std::uint32_t kv_heads,
                                                  std::uint32_t head_dim, std::uint64_t model_hash,
                                                  const std::string& tag, bool identity,
                                                  std::string& blob, std::string& err) {
    const std::uint64_t expected =
        static_cast<std::uint64_t>(layers) * kv_heads * head_dim;
    if (words.size() != expected) {
        err = "length: " + std::to_string(words.size()) + " words != " + std::to_string(layers) +
              "x" + std::to_string(kv_heads) + "x" + std::to_string(head_dim);
        return false;
    }
    if (words.size() > ninfer::ops::kKvRowScalePoolCapacity) {
        err = "symbol_extent: " + std::to_string(words.size()) + " words > pool capacity " +
              std::to_string(ninfer::ops::kKvRowScalePoolCapacity);
        return false;
    }
    std::string payload;
    payload.resize(words.size() * 2);
    for (std::size_t i = 0; i < words.size(); ++i) {
        payload[2 * i]     = static_cast<char>(words[i] & 0xFFu);
        payload[2 * i + 1] = static_cast<char>((words[i] >> 8) & 0xFFu);
    }
    const std::uint32_t flags =
        identity ? ninfer::ops::kKvRowScaleFlagIdentity : 0u;
    if (identity) {
        for (const std::uint16_t word : words) {
            if (word != 0x3F80) {
                err = "identity.payload: non-1.0 word in identity-flagged table";
                return false;
            }
        }
    }
    blob.assign(64, '\0');
    auto* out = reinterpret_cast<unsigned char*>(&blob[0]);
    std::memcpy(out, "NINFERKVRS1", 11);
    const auto put_u32 = [&](std::size_t offset, std::uint32_t value) {
        out[offset + 0] = static_cast<unsigned char>(value & 0xFFu);
        out[offset + 1] = static_cast<unsigned char>((value >> 8) & 0xFFu);
        out[offset + 2] = static_cast<unsigned char>((value >> 16) & 0xFFu);
        out[offset + 3] = static_cast<unsigned char>((value >> 24) & 0xFFu);
    };
    put_u32(16, ninfer::ops::kKvRowScaleSidecarVersion);
    put_u32(20, layers);
    put_u32(24, kv_heads);
    put_u32(28, head_dim);
    put_u32(32, flags);
    put_u32(36, ninfer::ops::detail::crc32_reflected(
                    reinterpret_cast<const unsigned char*>(payload.data()), payload.size()));
    for (int i = 0; i < 8; ++i) {
        out[40 + i] = static_cast<unsigned char>((model_hash >> (8 * i)) & 0xFFu);
    }
    std::memcpy(out + 48, tag.data(), std::min<std::size_t>(tag.size(), 16));
    blob += payload;
    return true;
}

struct KvRowScalePersistConfig {
    bool enabled = false;
    std::filesystem::path artifact;
    std::filesystem::path table;
    std::filesystem::path records;
    std::uint64_t fingerprint = 0;
    bool recalibrate = false;
    bool graphs_enabled = true;
    std::int64_t run_start_unix = 0;
    std::uint64_t model_hash = 0;  // 0 = unspecified, exactly as the format defines it
    // Filled in by kv_rowscale_persist_resolve(), consumed by
    // kv_rowscale_persist_finish().
    std::uint32_t layers = 0;
    std::uint32_t kv_heads = 0;
    std::uint32_t head_dim = 0;
    KvRowScalePlan plan = KvRowScalePlan::Disabled;
};

// Process-global by construction (the row-scale state IS process-global: see the
// loader's own contract).  A function-local static in an inline function is one
// object for the whole program, so the CLI's configure and the planner's resolve
// see the same config.
[[nodiscard]] inline KvRowScalePersistConfig& kv_rowscale_persist_config() {
    static KvRowScalePersistConfig config;
    return config;
}

struct KvRowScaleResolution {
    KvRowScalePlan plan = KvRowScalePlan::Disabled;
    std::string apply_spec;  // "auto" | "<path>", handed to the loader
    std::string report;      // one line for stderr; empty = say nothing
};

namespace detail {

// The gates that need no geometry.  `table_mtime` is returned for the report.
[[nodiscard]] inline bool kv_rowscale_table_usable(const std::string& blob,
                                                  const KvRowScalePersistConfig& config,
                                                  std::string& err) {
    ninfer::ops::KvRowScaleSidecar sidecar;
    if (!ninfer::ops::kv_rowscale_sidecar_parse(blob, sidecar, err)) { return false; }
    // An identity-flagged table is deliberately NOT refused once it carries this
    // producer's schema and this run's fingerprint: it says the solve found no
    // channel imbalance, and refusing it would make the loop recapture forever.
    // A table that is not ours never reaches here (the schema check below).
    std::uint64_t stored = 0;
    if (!kv_rowscale_tag_fingerprint(sidecar.tag, stored)) {
        err = "tag: producer schema " + std::string(kKvRowScaleTagPrefix) +
              " expected, found tag '" + sidecar.tag + "'";
        return false;
    }
    if (stored != config.fingerprint) {
        err = "config: table fingerprint " + detail::kv_rowscale_hex48(stored) +
              " != current KV configuration " + detail::kv_rowscale_hex48(config.fingerprint);
        return false;
    }
    return true;
}

// "<fingerprint> <reason>", one line.  The note is only honoured while the
// fingerprint still matches, so any change to the KV configuration re-arms the
// loop without anyone deleting a file.
[[nodiscard]] inline bool kv_rowscale_skip_note_applies(const KvRowScalePersistConfig& config,
                                                       std::string& reason) {
    std::string text;
    if (!kv_rowscale_read_file(kv_rowscale_skip_note_path(config.table), text)) { return false; }
    if (text.size() < 12) { return false; }
    std::uint64_t stored = 0;
    try {
        stored = std::stoull(text.substr(0, 12), nullptr, 16);
    } catch (const std::exception&) {
        return false;
    }
    if (stored != config.fingerprint) { return false; }
    reason = text.size() > 13 ? text.substr(13) : std::string("no reason recorded");
    while (!reason.empty() && (reason.back() == '\n' || reason.back() == '\r')) { reason.pop_back(); }
    return true;
}

}  // namespace detail

// Pre-flight (CLI, before the first Engine: no geometry is known yet, and the
// graph-capture decision has to be made here).  Installs the config and returns
// what the run is expected to do, logging the reason.  When the answer is
// Capture the caller MUST run without CUDA graphs: the capture synchronizes the
// producing stream, which is illegal inside a graph capture.
[[nodiscard]] inline KvRowScalePlan kv_rowscale_persist_begin(KvRowScalePersistConfig config) {
    KvRowScalePlan plan = KvRowScalePlan::Disabled;
    std::string report;
    if (!config.enabled) {
        report = "[kvrowscale] loop off: an explicit --kv-row-scale owns this run";
    } else {
        std::string reason;
        const std::int64_t artifact_mtime = detail::kv_rowscale_mtime_unix(config.artifact);
        const std::int64_t table_mtime    = detail::kv_rowscale_mtime_unix(config.table);
        std::string blob;
        std::string err;
        if (config.recalibrate) {
            plan   = KvRowScalePlan::Capture;
            reason = "--recalibrate: capturing again and overwriting the table";
        } else if (detail::kv_rowscale_skip_note_applies(config, reason)) {
            plan = KvRowScalePlan::NothingToCalibrate;
            report = "[kvrowscale] nothing to calibrate: " + reason +
                     " (note " + kv_rowscale_skip_note_path(config.table).string() +
                     "); using the baked table";
        } else if (!detail::kv_rowscale_read_file(config.table, blob)) {
            plan   = KvRowScalePlan::Capture;
            reason = "no persisted table at " + config.table.string();
        } else if (!detail::kv_rowscale_table_usable(blob, config, err)) {
            plan   = KvRowScalePlan::Capture;
            reason = "persisted table not applicable (" + err + ")";
        } else if (artifact_mtime > table_mtime) {
            plan   = KvRowScalePlan::Capture;
            reason = "table baked " + detail::kv_rowscale_time_text(table_mtime) +
                     " is older than the artifact (" + detail::kv_rowscale_time_text(artifact_mtime) +
                     ")";
        } else {
            plan = KvRowScalePlan::UsePersisted;
            report = "[kvrowscale] persisted table " + config.table.string() + " accepted: baked " +
                     detail::kv_rowscale_time_text(table_mtime) + ", fingerprint " +
                     detail::kv_rowscale_hex48(config.fingerprint) + "; capture skipped";
        }
        if (plan == KvRowScalePlan::Capture) {
            report = "[kvrowscale] one-shot calibration: " + reason + "; this run captures into " +
                     config.records.string() + " and writes " + config.table.string() +
                     (config.graphs_enabled
                          ? " (CUDA graphs must be off for this run: the capture synchronizes the "
                            "producing stream)"
                          : "");
        }
    }
    config.plan = plan;
    kv_rowscale_persist_config() = std::move(config);
    if (!report.empty()) { std::fprintf(stderr, "%s\n", report.c_str()); }
    return plan;
}

// Plan time (decoder_state.cpp, the row-scale commit point): the authoritative
// gate, because only here are the live KV geometry and the resolved per-layer
// store known.  Re-decides everything the pre-flight decided, plus the geometry.
[[nodiscard]] inline KvRowScaleResolution kv_rowscale_persist_resolve(std::uint32_t layers,
                                                                     std::uint32_t kv_heads,
                                                                     std::uint32_t head_dim,
                                                                     bool domain_active,
                                                                     std::string_view store_spec) {
    KvRowScalePersistConfig& config = kv_rowscale_persist_config();
    KvRowScaleResolution out;
    out.plan       = KvRowScalePlan::Disabled;
    out.apply_spec = "auto";
    if (!config.enabled) { return out; }

    // STALENESS, RE-DECIDED HERE.  `artifact_mtime > table_mtime` used to live ONLY in
    // kv_rowscale_persist_begin(), the pre-flight; this function re-decides from the tag
    // and the geometry alone and never re-read the two stamps, so a table the pre-flight
    // had already rejected as stale was APPLIED here anyway -- after the run had paid the
    // graphs-off cost that rejection bought.  It is silent, it is the authoritative half,
    // and the artifact it accepts is "the same path with a different file on it", i.e.
    // exactly what a model swap, a restore or a container layer produces.  Both stamps
    // are read here for the same reason the geometry is: the commit point must re-decide
    // EVERYTHING the pre-flight decided, or the pre-flight's decision is not a gate.
    const std::int64_t artifact_mtime = detail::kv_rowscale_mtime_unix(config.artifact);
    const std::int64_t table_mtime    = detail::kv_rowscale_mtime_unix(config.table);

    // THE .skip NOTE IS PART OF THE DECISION, HERE TOO.  begin() honoured it and this
    // half did not, so the note's stated purpose -- "keep the loop from arming a capture
    // for a configuration whose KV tier has no rotated/quantized domain at all"
    // (kv_rowscale_persist.h:226-234) -- was defeated in the half that actually arms the
    // capture.  Read it first: when it AGREES with the live stack its own reason is what
    // the operator should see; when the live stack contradicts it, the contradiction is
    // NAMED rather than silently resolved in either direction.
    std::string note_reason;
    const bool note_applies = detail::kv_rowscale_skip_note_applies(config, note_reason);

    // The KV tier in front of us has no rotated / quantized domain: nothing to
    // measure, and the note keeps the loop from arming a capture on every run.
    if (!domain_active) {
        if (!note_applies) {
            const std::string note =
                detail::kv_rowscale_hex48(config.fingerprint) +
                " no full-attention layer resolves to the NVFP4 tier that reads the row scale"
                " (resolved per-layer KV store: " + std::string(store_spec) + ")\n";
            std::string err;
            (void)detail::kv_rowscale_write_file_atomic(kv_rowscale_skip_note_path(config.table), note, err);
        }
        out.plan   = KvRowScalePlan::NothingToCalibrate;
        out.report = "[kvrowscale] nothing to calibrate: no full-attention layer of this stack "
                     "resolves to the NVFP4 tier, and the row scale is applied by the NVFP4 K/V "
                     "kernels only (gqa_attention_decode_nvfp4.cuh:330 on the K write and :489 on "
                     "the Q read); resolved per-layer KV store: " +
                     std::string(store_spec) + "; using the baked table";
        if (note_applies) {
            out.report += "\n[kvrowscale] the .skip note recorded by an earlier run of this "
                          "configuration agrees: " +
                          note_reason + " (note " +
                          kv_rowscale_skip_note_path(config.table).string() + ")";
        }
        config.plan = out.plan;
        return out;
    }

    if (note_applies) {
        // The note says this configuration cannot be calibrated; THIS stack resolved a
        // live NVFP4 domain, so the note is contradicted.  It is moved aside (evidence,
        // not deletion -- the `.rejected` precedent) so that the two halves agree from
        // the next run on, and the contradiction is NAMED: silently arming a capture
        // over a note that says a capture is impossible is how a two-half decision
        // drifts, and silently honouring the note would leave a calibratable
        // configuration permanently uncalibrated.
        std::error_code ec;
        std::filesystem::rename(kv_rowscale_skip_note_path(config.table),
                                kv_rowscale_skip_note_stale_path(config.table), ec);
        out.report = "[kvrowscale] the .skip note for this KV configuration says '" + note_reason +
                     "' but this stack DOES resolve a live NVFP4 domain (resolved per-layer KV "
                     "store: " + std::string(store_spec) + "): the note is contradicted" +
                     (ec ? " (and could not be moved aside: " + ec.message() + ")" :
                           " and was moved aside to " +
                               kv_rowscale_skip_note_stale_path(config.table).string()) +
                     "; continuing with the capture decision";
    }

    std::string blob;
    std::string err;
    bool usable = false;
    if (!config.recalibrate && detail::kv_rowscale_read_file(config.table, blob) &&
        detail::kv_rowscale_table_usable(blob, config, err)) {
        // Geometry, exactly as the loader will check it: a table baked for
        // another model must never be applied (the loader would throw, and the
        // loop's job is to recapture instead of failing the run).
        ninfer::ops::KvRowScaleSidecar sidecar;
        std::string parse_err;
        if (!ninfer::ops::kv_rowscale_sidecar_parse(blob, sidecar, parse_err) ||
            !ninfer::ops::kv_rowscale_sidecar_check(sidecar, layers, kv_heads, head_dim,
                                                   config.model_hash, parse_err)) {
            err = parse_err;
        } else if (artifact_mtime > table_mtime) {
            // THE SAME FLOOR THE PRE-FLIGHT APPLIES, APPLIED HERE TOO.  Not usable does
            // NOT mean "stale" (see the classification below: a configuration-fingerprint
            // mismatch is the valid state of a DIFFERENT configuration and is left in
            // place), so the reason is spelled out where the classification reads it.
            err = "stale: the table was baked " + detail::kv_rowscale_time_text(table_mtime) +
                  " and the artifact was written " +
                  detail::kv_rowscale_time_text(artifact_mtime) +
                  ", so the table describes an artifact that is no longer the one on disk";
        } else {
            usable = true;
        }
    }
    if (usable) {
        out.plan       = KvRowScalePlan::UsePersisted;
        out.apply_spec = config.table.string();
        out.report     = "[kvrowscale] using persisted table " + config.table.string() + " (baked " +
                         detail::kv_rowscale_time_text(detail::kv_rowscale_mtime_unix(config.table)) +
                         ", fingerprint " + detail::kv_rowscale_hex48(config.fingerprint) +
                         ", geometry " + std::to_string(layers) + "x" + std::to_string(kv_heads) +
                         "x" + std::to_string(head_dim) + "): capture skipped";
        config.plan = out.plan;
        return out;
    }

    // Not usable.  The ways a table can be unusable are NOT equivalent, and only
    // some of them are this configuration's to clean up:
    //   * a foreign PRODUCER (tag without this schema) or a foreign GEOMETRY
    //     (another model) is a genuinely stale table: it is moved aside, which
    //     (a) keeps it as evidence and (b) makes the NEXT run's pre-flight see
    //     "no table" and capture -- the only way a geometry mismatch can heal
    //     without an operator;
    //   * a CONFIGURATION-fingerprint mismatch is NOT stale.  It is the valid,
    //     calibrated state of a DIFFERENT KV configuration, and it is here
    //     because kv_rowscale_table_path() keys the sidecar on the artifact
    //     alone -- every KV configuration of one model shares one table path.
    //     Moving it aside would destroy that configuration's calibration AND
    //     guarantee the ping-pong: config A bakes, config B renames it away, A
    //     bakes again, forever -- and every one of those runs pays the
    //     graphs-off capture.  So it is left exactly where it is; this run
    //     falls through to the baked table instead.
    bool foreign_config = false;
    if (!blob.empty()) {
        ninfer::ops::KvRowScaleSidecar sidecar;
        std::string parse_err;
        std::uint64_t stored = 0;
        foreign_config = ninfer::ops::kv_rowscale_sidecar_parse(blob, sidecar, parse_err) &&
                         kv_rowscale_tag_fingerprint(sidecar.tag, stored) &&
                         stored != config.fingerprint;
    }
    if (!config.recalibrate && foreign_config) {
        out.report = "[kvrowscale] persisted table " + config.table.string() +
                     " was baked for another KV configuration and was left in place, not moved "
                     "aside (" + err + ")";
    } else if (!config.recalibrate && !blob.empty()) {
        std::error_code ec;
        std::filesystem::rename(config.table, kv_rowscale_rejected_path(config.table), ec);
        out.report = "[kvrowscale] persisted table " + config.table.string() +
                     " is not applicable (" + err + "); moved aside to " +
                     kv_rowscale_rejected_path(config.table).string();
    }

    if (config.graphs_enabled && !config.recalibrate) {
        // The capture cannot run inside a graph capture (it synchronizes the
        // producing stream), and the graph decision was taken before this point,
        // so arming it here would break the run.  That is the tree's own
        // precedent for offline modes (NINFER_KVDUMP_DIR / NINFER_FT_STATS use
        // --no-cuda-graph), so say it instead of doing something unsafe.
        out.plan   = KvRowScalePlan::NothingToCalibrate;
        out.report += (out.report.empty() ? std::string() : std::string("\n")) +
                      "[kvrowscale] capture deferred: CUDA graphs are on for this run and the "
                      "calibration capture synchronizes the producing stream; re-run with "
                      "--no-cuda-graph to calibrate (nothing was changed)";
        config.plan = out.plan;
        return out;
    }

    out.plan = KvRowScalePlan::Capture;
    config.plan = out.plan;
    config.layers   = layers;
    config.kv_heads = kv_heads;
    config.head_dim = head_dim;
    {
        // THE IDENTITY OF THE PRODUCING RUN.  The capture reads this once, at its first
        // call, and writes it into the .kvc header's 16 reserved bytes; the solve then
        // refuses, BY NAME, any frame that does not carry THIS run's identity, and
        // kv_rowscale_persist_finish() stamps the table with the identity the FRAMES
        // carry -- so the stamp can no longer be taken from the run that writes the
        // table rather than the one that produced the frames.  Installed here, before
        // the first prefill, for exactly the reason NINFER_KV_CALIB_DIR is below: the
        // capture reads its environment once.
        const std::string identity = kv_rowscale_frame_identity_text(
            config.fingerprint, kKvRowScaleFrameIdentityVersion);
#if defined(_WIN32)
        const int set_identity = ::_putenv_s(kKvRowScaleFrameIdentityEnv, identity.c_str());
#else
        const int set_identity = ::setenv(kKvRowScaleFrameIdentityEnv, identity.c_str(), 1);
#endif
        if (set_identity != 0) {
            out.plan    = KvRowScalePlan::NothingToCalibrate;
            config.plan = out.plan;
            out.report += (out.report.empty() ? std::string() : std::string("\n")) +
                          "[kvrowscale] capture not armed: cannot set " +
                          std::string(kKvRowScaleFrameIdentityEnv) + " to " + identity +
                          ", and this build's solve REFUSES an unstamped frame by name";
            return out;
        }
    }
    if (!config.records.empty() && std::getenv("NINFER_KV_CALIB_DIR") == nullptr) {
        // The engine's capture reads this variable once, at its first call, and
        // plan time is before the first prefill, so installing it here is what
        // makes "no table -> capture this run" work without new engine plumbing.
        const std::string directory = config.records.string();
#if defined(_WIN32)
        const int set_result = ::_putenv_s("NINFER_KV_CALIB_DIR", directory.c_str());
#else
        const int set_result = ::setenv("NINFER_KV_CALIB_DIR", directory.c_str(), 1);
#endif
        if (set_result != 0) {
            out.plan    = KvRowScalePlan::NothingToCalibrate;
            config.plan = out.plan;
            out.report  = "[kvrowscale] capture not armed: cannot set NINFER_KV_CALIB_DIR to " +
                          directory;
            return out;
        }
        out.report += (out.report.empty() ? std::string() : std::string("\n")) +
                      "[kvrowscale] capture armed into " + directory +
                      " (one prefill, <= NINFER_KV_CALIB_MAX_TOKENS=4096 tokens per layer); the "
                      "table is written at the end of this run and loaded from the next one";
    }
    return out;
}

// End of the run (the CLI, while the engine is still alive: the solve reads the
// SO(4) matrix back from the device).  NEVER throws: a bake failure must not turn
// a good generation into an error, and a table that cannot be produced
// confidently must stay ABSENT -- the next run then captures again -- rather than
// be produced wrong.
inline void kv_rowscale_persist_finish() {
    KvRowScalePersistConfig& config = kv_rowscale_persist_config();
    if (!config.enabled || config.plan != KvRowScalePlan::Capture) { return; }
    std::string report;
    try {
        std::vector<std::uint16_t> words;
        KvRowScaleBakeStats stats;
        std::string err;
        if (!kv_rowscale_solve_frames(config.records, config.layers, config.kv_heads,
                                      config.head_dim, config.run_start_unix, config.fingerprint,
                                      words, stats, err)) {
            report = "[kvrowscale] no table written: " + err;
        } else if (stats.producer_fingerprint != config.fingerprint) {
            // Cannot happen: the solve refuses any frame whose identity is not this
            // run's, so the stamp and this run agree BY CONSTRUCTION.  Written out
            // rather than assumed, because an unreachable branch that would write a
            // mislabelled table is the defect this hunk exists to close.
            report = "[kvrowscale] no table written: the solve consumed frames stamped " +
                     detail::kv_rowscale_hex48(stats.producer_fingerprint) +
                     " while this run is " + detail::kv_rowscale_hex48(config.fingerprint);
        } else {
            bool identity = true;
            for (const std::uint16_t word : words) {
                if (word != 0x3F80) { identity = false; break; }
            }
            // THE STAMP OF THE RUN THAT PRODUCED THE FRAMES, not of this one.  This
            // line used to read `config.fingerprint` -- the CURRENT run's -- whatever run
            // produced the frames, which laundered a mislabelled capture into the table's
            // own key and made it permanently trusted by the only gate downstream.
            const std::string tag = kv_rowscale_make_tag(stats.producer_fingerprint);
            std::string blob;
            if (!kv_rowscale_build_table(words, config.layers, config.kv_heads, config.head_dim,
                                         config.model_hash, tag, identity, blob, err)) {
                report = "[kvrowscale] no table written: " + err;
            } else {
                const std::filesystem::path words_path =
                    std::filesystem::path(config.table.string() + ".words.bin");
                std::string words_blob;
                words_blob.resize(words.size() * 2);
                for (std::size_t i = 0; i < words.size(); ++i) {
                    words_blob[2 * i]     = static_cast<char>(words[i] & 0xFFu);
                    words_blob[2 * i + 1] = static_cast<char>((words[i] >> 8) & 0xFFu);
                }
                std::string write_err;
                if (!detail::kv_rowscale_write_file_atomic(config.table, blob, write_err)) {
                    report = "[kvrowscale] no table written: " + write_err;
                } else {
                    (void)detail::kv_rowscale_write_file_atomic(words_path, words_blob, write_err);
                    char buffer[512];
                    std::snprintf(
                        buffer, sizeof(buffer),
                        "[kvrowscale] wrote %s: geometry %ux%ux%u words=%zu identity=%d "
                        "crc=%08x tag=%s (stamped by the run that produced the frames)",
                        config.table.string().c_str(), config.layers, config.kv_heads,
                        config.head_dim, words.size(), identity ? 1 : 0,
                        ninfer::ops::detail::crc32_reflected(
                            reinterpret_cast<const unsigned char*>(blob.data() + 64),
                            blob.size() - 64),
                        tag.c_str());
                    report = buffer;
                    char detail_buffer[512];
                    std::snprintf(detail_buffer, sizeof(detail_buffer),
                                  "[kvrowscale] bake: frames %llu used / %llu skipped, tokens %llu, "
                                  "rotated=%d, layers calibrated %u / identity %u, heads identity "
                                  "%u, channels identity %u, channel range %.3g -> %.3g, s in "
                                  "[%.4f, %.4f], mean(s) %.6f",
                                  static_cast<unsigned long long>(stats.frames_used),
                                  static_cast<unsigned long long>(stats.frames_skipped),
                                  static_cast<unsigned long long>(stats.tokens_used),
                                  stats.rotation_applied ? 1 : 0, stats.layers_calibrated,
                                  stats.layers_identity, stats.heads_identity,
                                  stats.channels_identity, stats.worst_channel_range_before,
                                  stats.worst_channel_range_after, stats.scale_min,
                                  stats.scale_max, stats.product_drift);
                    report += "\n";
                    report += detail_buffer;
                    if (!stats.first_skip_reason.empty()) {
                        report += "\n[kvrowscale] first skipped frame: " + stats.first_skip_reason;
                    }
                    if (stats.frames_skipped > 0) {
                        report += " (skipped frames are named once; the rest are the same class)";
                    }
                }
            }
        }
    } catch (const std::exception& error) {
        report = std::string("[kvrowscale] no table written: ") + error.what();
    } catch (...) {
        report = "[kvrowscale] no table written: unknown failure";
    }
    if (!report.empty()) { std::fprintf(stderr, "%s\n", report.c_str()); }
}

}  // namespace ninfer::product
