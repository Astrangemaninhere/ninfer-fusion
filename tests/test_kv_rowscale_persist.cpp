// N3 runtime loop: host-only acceptance for the persisted row-scale table
// (src/product/kv_rowscale_persist.h + src/product/kv_rowscale_bake.h).
//
// Everything the loop decides is pure host code, and the two facts it needs from
// the device are two function calls, so the whole loop -- the format, the gates,
// the tag, the capture arming, the solve, the persistence -- is testable without
// a GPU.  The two device readbacks are stubbed here.
//
// The tests are the acceptance the design note asks for (§122): a table the loop
// writes is byte-compatible with tools/kv_rowscale_sidecar.py and is accepted by
// the ENGINE's own parser (`kv_rowscale_sidecar_parse`/`_check`), the second run
// SKIPS the capture, --recalibrate OVERWRITES, and every invalidation case
// (schema, configuration fingerprint, geometry, artifact drift) recaptures
// instead of silently reusing a stale table.

#include "product/kv_rowscale_persist.h"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

void check(bool condition, const std::string& what) {
    if (condition) { return; }
    ++g_failures;
    std::cerr << "FAIL: " << what << '\n';
}

void check_equal(const std::string& got, const std::string& want, const std::string& what) {
    check(got == want, what + " (got '" + got + "', want '" + want + "')");
}

// ---- the two device readbacks, stubbed (declared in the loader header) ----

bool g_rotation_enabled = true;
float g_rotation[64][4][4] = {};

}  // namespace

namespace ninfer::ops {

bool kv_rowscale_device_rotation_enabled() { return g_rotation_enabled; }

bool kv_rowscale_device_rotation_matrix(float* out_row_major_64x4x4) {
    if (out_row_major_64x4x4 == nullptr) { return false; }
    for (std::size_t block = 0; block < 64; ++block) {
        for (std::size_t row = 0; row < 4; ++row) {
            for (std::size_t column = 0; column < 4; ++column) {
                out_row_major_64x4x4[(block * 4 + row) * 4 + column] = g_rotation[block][row][column];
            }
        }
    }
    return true;
}

}  // namespace ninfer::ops

namespace {

namespace fs = std::filesystem;
using ninfer::ops::detail::crc32_reflected;
using ninfer::product::KvRowScaleConfigKnobs;
using ninfer::product::KvRowScalePersistConfig;
using ninfer::product::KvRowScalePlan;
using ninfer::product::KvRowScaleResolution;

// One 4-channel block: 45 degrees in (0,1), 30 degrees in (2,3).  Every column
// sum is non-zero, which is what the "rotated domain" test needs.
void install_test_rotation() {
    const double c45 = std::sqrt(0.5);
    const double c30 = std::sqrt(3.0) / 2.0;
    for (std::size_t block = 0; block < 64; ++block) {
        g_rotation[block][0][0] = static_cast<float>(c45);
        g_rotation[block][0][1] = static_cast<float>(-c45);
        g_rotation[block][1][0] = static_cast<float>(c45);
        g_rotation[block][1][1] = static_cast<float>(c45);
        g_rotation[block][2][2] = static_cast<float>(c30);
        g_rotation[block][2][3] = static_cast<float>(-0.5);
        g_rotation[block][3][2] = static_cast<float>(0.5);
        g_rotation[block][3][3] = static_cast<float>(c30);
    }
    g_rotation_enabled = true;
}

std::uint16_t bf16(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return static_cast<std::uint16_t>(((bits + 0x7FFFu + ((bits >> 16) & 1u)) >> 16) & 0xFFFFu);
}

float from_bf16(std::uint16_t word) {
    const std::uint32_t bits = static_cast<std::uint32_t>(word) << 16;
    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

// The identity every synthetic frame carries and the identity the solve is asked for.
// They are the same number on purpose: a fixture that leaves the header's 16 reserved
// bytes zero is REFUSED BY NAME by the solve (product/kv_rowscale_frame_identity.h), so
// a fixture has to say who produced it -- which is the whole point of the field.
constexpr std::uint64_t kFixtureFingerprint = 0x123456789abcULL;

// The default configuration's fingerprint, PINNED BY ARITHMETIC from the snprintf arms
// of kv_rowscale_config_fingerprint.  The rope-regime arm is the last one added, and the
// whole reason it is mixed in ONLY when it is not the default is that this number does
// not move: every table on every disk keeps its tag, and therefore its validity, across
// this change.  The pin lives here rather than in a static_assert because the fingerprint
// function is not constexpr (snprintf + std::string); the smallest change that would make
// it compile-time is a constexpr integer formatter, and then this check moves into a
// static_assert next to kKvRowScaleFrameIdentityVersion.
constexpr std::uint64_t kHistoricalDefaultFingerprint = 0xf3b77882c97dULL;

// The frame writer: mirrors src/targets/qwen3_6/impl/runtime/kv_calibration.h
// (magic, field order, positions then K then V, head-dim-fastest channels, and now the
// identity in the header's reserved[4] @48).
//
// ⚠ `version` DEFAULTS TO THE LIVE SCHEMA VERSION, and that default is why an "unstamped" frame
// written without also passing version=0 is NOT unstamped. The real writer initialises
// `identity_fingerprint_` AND `identity_version_` to 0 and leaves both alone when
// NINFER_KV_ROWSCALE_IDENTITY is absent (kv_calibration.h:203-207), so the frame the engine writes
// with no identity variable has ALL SIXTEEN reserved bytes zero -- which is what
// product/kv_rowscale_frame_identity.h:28-33 calls "no stamp" and what
// kv_rowscale_bake.h:92-94 defines `frames_unnamed` to count. A fixture that passes identity=0 and
// lets version keep its default writes fingerprint 0 with producer version 1: a STAMPED frame
// claiming configuration 000000000000, which the solve counts FOREIGN. Both states are now written
// below on purpose, and the pair is the case that tells the two counters apart.
void write_frame(const fs::path& path, std::uint32_t layer, std::uint32_t head_dim,
                 std::uint32_t kv_heads, const std::vector<float>& k,
                 const std::vector<float>& v,
                 std::uint64_t identity = kFixtureFingerprint,
                 std::uint32_t version = ninfer::product::kKvRowScaleFrameIdentityVersion,
                 std::uint32_t flags = 0) {
    const std::uint32_t tokens = static_cast<std::uint32_t>(k.size() / (head_dim * kv_heads));
    std::string blob(64, '\0');
    auto* out = reinterpret_cast<unsigned char*>(&blob[0]);
    std::memcpy(out, "NINFERKVCAL1", 12);
    const auto put_u32 = [&](std::size_t offset, std::uint32_t value) {
        out[offset + 0] = static_cast<unsigned char>(value & 0xFFu);
        out[offset + 1] = static_cast<unsigned char>((value >> 8) & 0xFFu);
        out[offset + 2] = static_cast<unsigned char>((value >> 16) & 0xFFu);
        out[offset + 3] = static_cast<unsigned char>((value >> 24) & 0xFFu);
    };
    for (int i = 0; i < 8; ++i) {
        out[ninfer::product::kKvRowScaleFrameIdentityOffset + i] =
            static_cast<unsigned char>((identity >> (8 * i)) & 0xFFu);
    }
    put_u32(ninfer::product::kKvRowScaleFrameVersionOffset, version);
    put_u32(ninfer::product::kKvRowScaleFrameFlagsOffset, flags);
    put_u32(16, 64);
    put_u32(20, layer);
    put_u32(24, head_dim);
    put_u32(28, kv_heads);
    put_u32(32, tokens);
    for (std::uint32_t token = 0; token < tokens; ++token) {
        std::string positions(4, '\0');
        const std::uint32_t position = token;
        positions[0] = static_cast<char>(position & 0xFFu);
        blob += positions;
    }
    for (int plane = 0; plane < 2; ++plane) {
        const std::vector<float>& values = plane == 0 ? k : v;
        for (const float value : values) {
            const std::uint16_t word = bf16(value);
            blob.push_back(static_cast<char>(word & 0xFFu));
            blob.push_back(static_cast<char>((word >> 8) & 0xFFu));
        }
    }
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    file.write(blob.data(), static_cast<std::streamsize>(blob.size()));
}

// K with a constant amplitude per channel: rms[d] == amplitude[d], exactly.
std::vector<float> constant_channel_k(std::uint32_t head_dim, std::uint32_t kv_heads,
                                      std::uint32_t tokens, const std::vector<float>& amplitude) {
    std::vector<float> out(static_cast<std::size_t>(head_dim) * kv_heads * tokens, 0.0f);
    for (std::uint32_t head = 0; head < kv_heads; ++head) {
        for (std::uint32_t token = 0; token < tokens; ++token) {
            for (std::uint32_t d = 0; d < head_dim; ++d) {
                out[d + head_dim * (head + kv_heads * token)] = amplitude[d];
            }
        }
    }
    return out;
}

// ---- cases ----

void test_tag_and_fingerprint() {
    check_equal(ninfer::product::kv_rowscale_make_tag(0xabcULL), "rs1.000000000abc",
                "tag spelling");
    std::uint64_t parsed = 0;
    check(ninfer::product::kv_rowscale_tag_fingerprint("rs1.000000000abc", parsed) && parsed == 0xabc,
          "tag round trip");
    check(!ninfer::product::kv_rowscale_tag_fingerprint("qwen3_8_27b", parsed),
          "a hand-baked tag (tools/kv_rowscale_sidecar.py --tag) is not adopted");
    check(!ninfer::product::kv_rowscale_tag_fingerprint("rs1.000000000ab", parsed),
          "a short tag is refused");
    check(!ninfer::product::kv_rowscale_tag_fingerprint("rs1.000000000abz", parsed),
          "a non-hex tag digit is refused");

    KvRowScaleConfigKnobs base;
    const std::uint64_t reference = ninfer::product::kv_rowscale_config_fingerprint(base);
    check(reference == ninfer::product::kv_rowscale_config_fingerprint(KvRowScaleConfigKnobs{}),
          "the fingerprint is a pure function of the knobs");
    const std::vector<std::pair<std::string, KvRowScaleConfigKnobs>> mutations = {
        {"kv_cache_code", [&] { auto k = base; k.kv_cache_code = 3; return k; }()},
        {"kv_cache_explicit", [&] { auto k = base; k.kv_cache_explicit = true; return k; }()},
        {"layer_storage", [&] { auto k = base; k.layer_storage_spec = "all:bf16"; return k; }()},
        {"tier_formats", [&] { auto k = base; k.tier_formats_spec = "hot=bf16"; return k; }()},
        {"nvfp4_pure", [&] { auto k = base; k.nvfp4_pure = true; return k; }()},
        {"rotation_off", [&] { auto k = base; k.rotation_off = true; return k; }()},
        {"rotation_explicit", [&] { auto k = base; k.rotation_explicit = true; return k; }()},
        {"v_codec", [&] { auto k = base; k.v_codec = 1; return k; }()},
        {"bit_budget_bits", [&] { auto k = base; k.bit_budget_bits = 4.5; return k; }()},
        {"bit_budget_ranges", [&] { auto k = base; k.bit_budget_ranges = "0-7:8"; return k; }()},
        {"bit_budget_explicit", [&] { auto k = base; k.bit_budget_explicit = true; return k; }()},
        {"quality_weight", [&] { auto k = base; k.quality_weight = 0.5; return k; }()},
        {"tier_scores", [&] { auto k = base; k.tier_scores = "a=1"; return k; }()},
        {"cold_policy", [&] { auto k = base; k.cold_policy = 2; return k; }()},
        {"max_cold_pages", [&] { auto k = base; k.max_cold_pages = 8; return k; }()},
    };
    for (const auto& [name, knobs] : mutations) {
        check(ninfer::product::kv_rowscale_config_fingerprint(knobs) != reference,
              "changing " + name + " must change the fingerprint");
    }
}

void test_paths_and_writer() {
    check_equal(ninfer::product::kv_rowscale_table_path("/models/m.ninfer").string(),
                "/models/m.ninfer.kvrowscale.bin", "table path convention");
    check_equal(ninfer::product::kv_rowscale_records_path("/models/m.ninfer.kvrowscale.bin").string(),
                "/models/m.ninfer.kvrowscale.bin.d", "record directory convention");

    std::vector<std::uint16_t> words(16 * 4 * 256, bf16(1.25f));
    std::string blob;
    std::string err;
    const std::string tag = ninfer::product::kv_rowscale_make_tag(0x123456789abcULL);
    check(ninfer::product::kv_rowscale_build_table(words, 16, 4, 256, 0xDEADBEEF, tag, false, blob,
                                                  err),
          "build a table: " + err);
    check(blob.size() == 64 + 2 * words.size(), "table size is 64 + 2 * words");
    check_equal(tag, std::string(blob.data() + 48, 16), "the tag lands in tag[16] verbatim");

    // The ENGINE's own parser and gate, not a second implementation of them.
    ninfer::ops::KvRowScaleSidecar sidecar;
    std::string parse_err;
    check(ninfer::ops::kv_rowscale_sidecar_parse(blob, sidecar, parse_err),
          "the engine's parser accepts the loop's table: " + parse_err);
    check(sidecar.layers == 16 && sidecar.kv_heads == 4 && sidecar.head_dim == 256,
          "geometry survives the round trip");
    check(sidecar.tag == tag, "the producer tag survives the round trip");
    check(sidecar.crc == crc32_reflected(reinterpret_cast<const unsigned char*>(blob.data() + 64),
                                         blob.size() - 64),
          "the crc is the payload crc");
    check(ninfer::ops::kv_rowscale_sidecar_check(sidecar, 16, 4, 256, 0, parse_err),
          "the gate accepts the same model: " + parse_err);
    check(!ninfer::ops::kv_rowscale_sidecar_check(sidecar, 52, 4, 256, 0, parse_err),
          "the gate refuses a 52-layer model (foreign table)");
    check(sidecar.model_hash == 0xDEADBEEF, "the model hash field round trips");

    // Identity is a flag, and an identity-flagged table must be all 1.0.
    std::string identity_blob;
    std::vector<std::uint16_t> ones(16 * 4 * 256, 0x3F80);
    check(ninfer::product::kv_rowscale_build_table(ones, 16, 4, 256, 0, tag, true, identity_blob,
                                                  err),
          "build an identity table: " + err);
    check(ninfer::ops::kv_rowscale_sidecar_parse(identity_blob, sidecar, parse_err),
          "the parser accepts an identity table: " + parse_err);
    check((sidecar.flags & ninfer::ops::kKvRowScaleFlagIdentity) != 0, "the identity flag is set");
    std::vector<std::uint16_t> wrong = ones;
    wrong[0] = bf16(2.0f);
    check(!ninfer::product::kv_rowscale_build_table(wrong, 16, 4, 256, 0, tag, true, identity_blob,
                                                   err),
          "a non-1.0 word under the identity flag is refused");
    const std::vector<std::uint16_t> too_few(4, 0x3F80);
    check(!ninfer::product::kv_rowscale_build_table(too_few, 16, 4, 256, 0, tag, false,
                                                   identity_blob, err),
          "a wrong word count is refused");
}

void test_solve_and_rotated_domain() {
    const fs::path root = fs::temp_directory_path() / "ninfer_kvrowscale_test";
    std::error_code ec;
    fs::remove_all(root, ec);
    fs::create_directories(root / "frames", ec);

    const std::uint32_t layers = 3;
    const std::uint32_t heads = 2;
    const std::uint32_t dim = 8;
    const std::uint32_t tokens = 4;

    // A 2x spread: no channel needs a scale outside [0.5, 2.0], so the solve is
    // exact -- s_d == gm(rms)/rms_d, and the result is a perfect balance.
    std::vector<float> amplitude(dim);
    for (std::uint32_t d = 0; d < dim; ++d) {
        amplitude[d] = (d % 2 == 0) ? 0.5f : 1.0f;
    }
    const auto k = constant_channel_k(dim, heads, tokens, amplitude);
    const std::vector<float> v(static_cast<std::size_t>(dim) * heads * tokens, 0.5f);
    // Rotation off: the plain domain, so the solve sees `amplitude` directly.
    g_rotation_enabled = false;
    for (std::uint32_t layer = 0; layer < layers; ++layer) {
        write_frame(root / "frames" / (std::to_string(layer) + ".kvc"), layer, dim, heads, k, v);
    }

    std::vector<std::uint16_t> words;
    ninfer::product::KvRowScaleBakeStats stats;
    std::string err;
    check(ninfer::product::kv_rowscale_solve_frames(root / "frames", layers, heads, dim, 0, kFixtureFingerprint,
                                                    words,
                                                    stats, err),
          "solve synthetic frames: " + err);
    check(words.size() == static_cast<std::size_t>(layers) * heads * dim, "word count matches");
    check(stats.frames_used == layers && stats.frames_skipped == 0, "every frame was used");
    check(stats.layers_calibrated == layers, "every layer was calibrated");
    check(stats.rotation_applied == false, "the stub reports the rotation domain as off");

    // The expected scale is target/rms, renormalised into [0.5, 2.0]; with a
    // 4x spread and a geometric target the clamp is not reached.
    const double geometric = [&] {
        double log_sum = 0.0;
        for (const float value : amplitude) { log_sum += std::log(value); }
        return std::exp(log_sum / static_cast<double>(dim));
    }();
    for (std::uint32_t d = 0; d < dim; ++d) {
        const float want = static_cast<float>(geometric / amplitude[d]);
        const float got = from_bf16(words[d]);
        check(std::abs(got - want) <= 0.01f * std::abs(want),
              "scale[" + std::to_string(d) + "] == gm/rms");
        check(got >= 0.5f && got <= 2.0f, "scale stays inside the kernel's box");
    }
    check(stats.worst_channel_range_before > 1.9 && stats.worst_channel_range_after < 1.01,
          "the solve collapses the channel range (before " +
              std::to_string(stats.worst_channel_range_before) + ", after " +
              std::to_string(stats.worst_channel_range_after) + ")");
    check(std::abs(stats.product_drift - 1.0) < 1e-3,
          "no net gain: mean(ln s) ~ 0 (" + std::to_string(stats.product_drift) + ")");
    check(stats.layers_identity == 0, "no layer fell back to identity");

    // A 6x spread: the two extreme channels need scales outside the kernel's box,
    // so they pin to the bound and the FREE channels carry the geometric mean.
    // This is the case the free-set renormalisation exists for.
    fs::remove_all(root / "clamped", ec);
    fs::create_directories(root / "clamped", ec);
    std::vector<float> wide(dim);
    for (std::uint32_t d = 0; d < dim; ++d) {
        wide[d] = std::vector<float>{0.25f, 0.5f, 1.0f, 1.5f}[d % 4];
    }
    write_frame(root / "clamped" / "0.kvc", 0, dim, heads, constant_channel_k(dim, heads, tokens, wide), v);
    std::vector<std::uint16_t> wide_words;
    ninfer::product::KvRowScaleBakeStats wide_stats;
    check(ninfer::product::kv_rowscale_solve_frames(root / "clamped", 1, heads, dim, 0, kFixtureFingerprint,
                                                    wide_words,
                                                    wide_stats, err),
          "solve the wide-spread frame: " + err);
    double free_log_sum = 0.0;
    int free_count = 0;
    int pinned = 0;
    for (std::uint32_t d = 0; d < dim; ++d) {
        const float got = from_bf16(wide_words[d]);
        check(got >= 0.5f && got <= 2.0f, "a clamped solve still stays inside the box");
        if (got <= 0.5f + 1e-3f || got >= 2.0f - 1e-3f) {
            ++pinned;
        } else {
            free_log_sum += std::log(static_cast<double>(got));
            ++free_count;
        }
    }
    check(pinned > 0, "the wide spread does pin channels at the box bounds");
    check(free_count > 0 && std::abs(std::exp(free_log_sum / free_count) - 1.0) < 1e-3,
          "the free channels carry the geometric mean (no net gain on them)");
    check(wide_stats.worst_channel_range_after < wide_stats.worst_channel_range_before,
          "and the solve still improves the channel range (before " +
              std::to_string(wide_stats.worst_channel_range_before) + ", after " +
              std::to_string(wide_stats.worst_channel_range_after) + ")");

    // ROTATED DOMAIN.  A frame whose captured channels are IMBALANCED but whose
    // rotated channels are perfectly balanced must solve to identity with the
    // rotation on, and to something else with it off.  This is the check that
    // the bake works in the domain the kernel applies the scale in.
    install_test_rotation();
    fs::remove_all(root / "rotated", ec);
    fs::create_directories(root / "rotated", ec);
    std::vector<float> rotated_amplitude(dim, 1.0f);
    // x = R^T z with z balanced per token: the captured plane is what the kernel
    // consumes, so it must NOT be balanced for this test to mean anything.
    std::vector<float> pre_rotation(static_cast<std::size_t>(dim) * heads * tokens, 0.0f);
    const std::vector<float> u = {1.0f, -0.5f, 0.25f, 2.0f};
    for (std::uint32_t head = 0; head < heads; ++head) {
        for (std::uint32_t token = 0; token < tokens; ++token) {
            for (std::uint32_t block = 0; block < dim / 4; ++block) {
                for (std::uint32_t j = 0; j < 4; ++j) {
                    double column_sum = 0.0;
                    for (std::uint32_t row = 0; row < 4; ++row) {
                        column_sum += g_rotation[block][row][j];
                    }
                    pre_rotation[block * 4 + j + dim * (head + heads * token)] =
                        static_cast<float>(column_sum) * u[token];
                }
            }
        }
    }
    write_frame(root / "rotated" / "0.kvc", 0, dim, heads, pre_rotation, v);
    std::vector<std::uint16_t> rotated_words;
    ninfer::product::KvRowScaleBakeStats rotated_stats;
    check(ninfer::product::kv_rowscale_solve_frames(root / "rotated", 1, heads, dim, 0, kFixtureFingerprint,
                                                    rotated_words,
                                                    rotated_stats, err),
          "solve the rotated frame: " + err);
    check(rotated_stats.rotation_applied, "the rotation domain was read back as enabled");
    bool all_one = true;
    for (const std::uint16_t word : rotated_words) {
        all_one = all_one && (word == 0x3F80);
    }
    check(all_one, "balanced in the rotated domain -> identity scales");
    g_rotation_enabled = false;
    std::vector<std::uint16_t> plain_words;
    ninfer::product::KvRowScaleBakeStats plain_stats;
    check(ninfer::product::kv_rowscale_solve_frames(root / "rotated", 1, heads, dim, 0, kFixtureFingerprint,
                                                    plain_words,
                                                    plain_stats, err),
          "solve the same frame without the rotation: " + err);
    bool any_scaled = false;
    for (const std::uint16_t word : plain_words) { any_scaled = any_scaled || (word != 0x3F80); }
    check(any_scaled, "the same capture, read in the WRONG domain, is not identity (so the "
                      "domain handling is load bearing)");
    install_test_rotation();

    // Frame filtering: geometry, magic, size and age must all be refusals.
    fs::remove_all(root / "junk", ec);
    fs::create_directories(root / "junk", ec);
    write_frame(root / "junk" / "0.kvc", 0, dim * 2, heads, k, v);  // wrong geometry
    {
        std::ofstream tiny(root / "junk" / "1.kvc", std::ios::binary | std::ios::trunc);
        tiny << "not a frame";
    }
    write_frame(root / "junk" / "2.kvc", 0, dim, heads, k, v);
    std::vector<std::uint16_t> junk_words;
    ninfer::product::KvRowScaleBakeStats junk_stats;
    check(ninfer::product::kv_rowscale_solve_frames(root / "junk", 1, heads, dim, 0, kFixtureFingerprint,
                                                    junk_words,
                                                    junk_stats, err),
          "one good frame among junk is still usable: " + err);
    check(junk_stats.frames_skipped == 2, "the two bad frames are skipped");
    check(!junk_stats.first_skip_reason.empty(), "the first skip is named");
    const std::int64_t now_unix = static_cast<std::int64_t>(
        std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count());
    const std::int64_t future = now_unix + 3600;
    check(!ninfer::product::kv_rowscale_solve_frames(root / "junk", 1, heads, dim, future, kFixtureFingerprint,
                                                     junk_words, junk_stats, err),
          "an mtime floor above every frame refuses the whole directory");
    check(err.find("could be used") != std::string::npos, "and the refusal names the cause: " + err);
    fs::remove_all(root, ec);
}

KvRowScaleConfigKnobs base_knobs() {
    KvRowScaleConfigKnobs knobs;
    knobs.kv_cache_code = 3;
    knobs.rotation_off = false;
    return knobs;
}

KvRowScalePersistConfig make_config(const fs::path& artifact, const KvRowScaleConfigKnobs& knobs,
                                    bool graphs, bool recalibrate) {
    KvRowScalePersistConfig config;
    config.enabled = true;
    config.artifact = artifact;
    config.table = ninfer::product::kv_rowscale_table_path(artifact);
    config.records = ninfer::product::kv_rowscale_records_path(config.table);
    config.fingerprint = ninfer::product::kv_rowscale_config_fingerprint(knobs);
    config.recalibrate = recalibrate;
    config.graphs_enabled = graphs;
    return config;
}

void write_artifact(const fs::path& artifact) {
    std::ofstream file(artifact, std::ios::binary | std::ios::trunc);
    file << "a model artifact stand-in";
}

void test_loop_end_to_end() {
    const fs::path root = fs::temp_directory_path() / "ninfer_kvrowscale_loop";
    std::error_code ec;
    fs::remove_all(root, ec);
    fs::create_directories(root, ec);
    const fs::path artifact = root / "m.ninfer";
    write_artifact(artifact);
    install_test_rotation();
    g_rotation_enabled = false;
    ::unsetenv("NINFER_KV_CALIB_DIR");

    constexpr std::uint32_t kLayers = 2;
    constexpr std::uint32_t kHeads = 2;
    constexpr std::uint32_t kDim = 8;

    // 1. no table: capture, then persist.
    KvRowScalePlan first = ninfer::product::kv_rowscale_persist_begin(
        make_config(artifact, base_knobs(), false, false));
    check(first == KvRowScalePlan::Capture, "run 1 with no table captures");
    KvRowScaleResolution resolution = ninfer::product::kv_rowscale_persist_resolve(
        kLayers, kHeads, kDim, /*domain_active=*/true, "0-1:nvfp4");
    check(resolution.plan == KvRowScalePlan::Capture, "plan time agrees: capture");
    const char* armed_dir = std::getenv("NINFER_KV_CALIB_DIR");
    check(armed_dir != nullptr, "the capture was armed through the engine's own env gate");
    check_equal(armed_dir == nullptr ? std::string("<unset>") : std::string(armed_dir),
                ninfer::product::kv_rowscale_records_path(
                    ninfer::product::kv_rowscale_table_path(artifact))
                    .string(),
                "the capture directory follows the table path");
    fs::create_directories(ninfer::product::kv_rowscale_records_path(
                               ninfer::product::kv_rowscale_table_path(artifact)),
                           ec);
    std::vector<float> amplitude(kDim);
    for (std::uint32_t d = 0; d < kDim; ++d) { amplitude[d] = 0.25f * static_cast<float>(1 + d % 4); }
    const auto k = constant_channel_k(kDim, kHeads, 4, amplitude);
    const std::vector<float> v(static_cast<std::size_t>(kDim) * kHeads * 4, 0.5f);
    for (std::uint32_t layer = 0; layer < kLayers; ++layer) {
        write_frame(ninfer::product::kv_rowscale_records_path(
                        ninfer::product::kv_rowscale_table_path(artifact)) /
                        (std::to_string(layer) + ".kvc"),
                    layer, kDim, kHeads, k, v,
                    // The capture writes the identity the LOOP installed, which is the
                    // fingerprint of the run that armed it. A fixture that stamped
                    // anything else would (correctly) be refused by the solve.
                    ninfer::product::kv_rowscale_config_fingerprint(base_knobs()));
    }
    ninfer::product::kv_rowscale_persist_finish();
    const fs::path table = ninfer::product::kv_rowscale_table_path(artifact);
    check(fs::exists(table), "the run left a table behind");
    check(fs::exists(fs::path(table.string() + ".words.bin")), "and the words it was built from");
    check(!fs::exists(fs::path(table.string() + ".tmp")), "the atomic write left no temporary");
    std::string blob;
    {
        std::ifstream file(table, std::ios::binary);
        blob.assign((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    }
    ninfer::ops::KvRowScaleSidecar sidecar;
    std::string parse_err;
    check(ninfer::ops::kv_rowscale_sidecar_parse(blob, sidecar, parse_err),
          "the engine parses the table the loop wrote: " + parse_err);
    check(sidecar.layers == kLayers && sidecar.kv_heads == kHeads && sidecar.head_dim == kDim,
          "the table carries the live geometry");
    std::uint64_t stored = 0;
    check(ninfer::product::kv_rowscale_tag_fingerprint(sidecar.tag, stored) &&
              stored == ninfer::product::kv_rowscale_config_fingerprint(base_knobs()),
          "the table carries this build's schema and this run's configuration");

    // The artifact is older than the table it just produced; make that explicit.
    fs::last_write_time(artifact, fs::file_time_type::clock::now() - std::chrono::hours(1), ec);

    // 2. table present and applicable: skip the capture.
    ::unsetenv("NINFER_KV_CALIB_DIR");
    KvRowScalePlan second = ninfer::product::kv_rowscale_persist_begin(
        make_config(artifact, base_knobs(), true, false));
    check(second == KvRowScalePlan::UsePersisted, "run 2 skips the capture");
    resolution = ninfer::product::kv_rowscale_persist_resolve(kLayers, kHeads, kDim, true, "0-1:nvfp4");
    check(resolution.plan == KvRowScalePlan::UsePersisted, "plan time loads the table");
    check_equal(resolution.apply_spec, table.string(), "and hands the loader the persisted path");
    check(resolution.report.find("using persisted table") != std::string::npos &&
              resolution.report.find("baked") != std::string::npos,
          "the skip is logged with the table's age: " + resolution.report);
    check(std::getenv("NINFER_KV_CALIB_DIR") == nullptr, "and the capture is NOT armed");
    ninfer::product::kv_rowscale_persist_finish();
    std::string after;
    {
        std::ifstream file(table, std::ios::binary);
        after.assign((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    }
    check(after == blob, "a skipped run does not rewrite the table");

    // 3. --recalibrate: capture again and OVERWRITE.
    resolution = ninfer::product::kv_rowscale_persist_resolve(kLayers, kHeads, kDim, true, "0-1:nvfp4");
    (void)resolution;
    KvRowScalePlan forced = ninfer::product::kv_rowscale_persist_begin(
        make_config(artifact, base_knobs(), false, true));
    check(forced == KvRowScalePlan::Capture, "--recalibrate captures again");
    resolution = ninfer::product::kv_rowscale_persist_resolve(kLayers, kHeads, kDim, true, "0-1:nvfp4");
    check(resolution.plan == KvRowScalePlan::Capture, "--recalibrate captures at plan time too");
    check(std::getenv("NINFER_KV_CALIB_DIR") != nullptr, "--recalibrate arms the capture");
    for (std::uint32_t layer = 0; layer < kLayers; ++layer) {
        write_frame(ninfer::product::kv_rowscale_records_path(table) /
                        (std::to_string(layer) + ".kvc"),
                    layer, kDim, kHeads, k, v,
                    ninfer::product::kv_rowscale_config_fingerprint(base_knobs()));
    }
    ninfer::product::kv_rowscale_persist_finish();
    std::string recalibrated;
    {
        std::ifstream file(table, std::ios::binary);
        recalibrated.assign((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    }
    check(recalibrated.size() == blob.size(), "--recalibrate overwrote the table in place");
    check(!fs::exists(ninfer::product::kv_rowscale_rejected_path(table)),
          "a good table is never renamed aside");
    ::unsetenv("NINFER_KV_CALIB_DIR");
    fs::last_write_time(artifact, fs::file_time_type::clock::now() - std::chrono::hours(1), ec);

    // 4. INVALUATION: the KV configuration changes -> the table is not reused.
    KvRowScaleConfigKnobs changed = base_knobs();
    changed.v_codec = 1;
    KvRowScalePlan reconfigure = ninfer::product::kv_rowscale_persist_begin(
        make_config(artifact, changed, false, false));
    check(reconfigure == KvRowScalePlan::Capture, "a changed KV configuration recaptures");
    check(ninfer::product::kv_rowscale_persist_config().fingerprint !=
              ninfer::product::kv_rowscale_config_fingerprint(base_knobs()),
          "and the installed fingerprint followed the change");

    // 5. INVALIDATION: the artifact is newer than the table -> recapture.
    KvRowScalePlan drifted = ninfer::product::kv_rowscale_persist_begin(
        make_config(artifact, base_knobs(), false, false));
    check(drifted == KvRowScalePlan::UsePersisted, "same configuration, table fresh enough");
    fs::last_write_time(artifact, fs::file_time_type::clock::now() + std::chrono::hours(1), ec);
    drifted = ninfer::product::kv_rowscale_persist_begin(
        make_config(artifact, base_knobs(), false, false));
    check(drifted == KvRowScalePlan::Capture, "a newer artifact invalidates the table");
    fs::last_write_time(artifact, fs::file_time_type::clock::now() - std::chrono::hours(1), ec);

    // 6. INVALIDATION: geometry.  Plan time cannot apply it, so it is moved aside
    //    (which is what makes the NEXT run's pre-flight capture) and the run falls
    //    back to the baked table instead of failing.
    KvRowScalePlan geometry = ninfer::product::kv_rowscale_persist_begin(
        make_config(artifact, base_knobs(), false, false));
    check(geometry == KvRowScalePlan::UsePersisted, "the pre-flight cannot see the geometry");
    resolution = ninfer::product::kv_rowscale_persist_resolve(kLayers + 1, kHeads, kDim, true, "0-1:nvfp4");
    check(resolution.plan == KvRowScalePlan::Capture, "a geometry mismatch recaptures");
    check(fs::exists(ninfer::product::kv_rowscale_rejected_path(table)),
          "and the foreign table is kept as evidence");
    check(!fs::exists(table), "and is out of the way");
    fs::remove(ninfer::product::kv_rowscale_rejected_path(table), ec);

    // 7. the domain gate: bf16 KV has nothing to calibrate, and says so once.
    resolution = ninfer::product::kv_rowscale_persist_resolve(kLayers, kHeads, kDim,
                                                             /*domain_active=*/false, "0-1:nvfp4");
    check(resolution.plan == KvRowScalePlan::NothingToCalibrate, "bf16 KV does not calibrate");
    check(fs::exists(ninfer::product::kv_rowscale_skip_note_path(table)),
          "and leaves a note so the loop stops planning captures");
    KvRowScalePlan noted = ninfer::product::kv_rowscale_persist_begin(
        make_config(artifact, base_knobs(), false, false));
    check(noted == KvRowScalePlan::NothingToCalibrate,
          "the note keeps the next run from planning a capture");
    KvRowScalePlan noted_reconfigured = ninfer::product::kv_rowscale_persist_begin(
        make_config(artifact, changed, false, false));
    check(noted_reconfigured == KvRowScalePlan::Capture,
          "but a configuration change re-arms the loop without anyone deleting the note");

    // 8. graphs on and a capture needed: defer loudly instead of breaking the run.
    fs::remove(ninfer::product::kv_rowscale_skip_note_path(table), ec);
    ::unsetenv("NINFER_KV_CALIB_DIR");
    KvRowScalePlan graphs = ninfer::product::kv_rowscale_persist_begin(
        make_config(artifact, base_knobs(), true, false));
    check(graphs == KvRowScalePlan::Capture, "the capture is planned");
    resolution = ninfer::product::kv_rowscale_persist_resolve(kLayers, kHeads, kDim, true, "0-1:nvfp4");
    check(resolution.plan == KvRowScalePlan::NothingToCalibrate,
          "with graphs still on the capture is deferred, not armed");
    check(resolution.report.find("--no-cuda-graph") != std::string::npos,
          "and the deferral tells the operator what to do: " + resolution.report);
    check(std::getenv("NINFER_KV_CALIB_DIR") == nullptr, "nothing was armed");

    // 9. capture armed but no frames: no table, no invention.
    fs::remove_all(ninfer::product::kv_rowscale_records_path(table), ec);
    KvRowScalePlan empty = ninfer::product::kv_rowscale_persist_begin(
        make_config(artifact, base_knobs(), false, false));
    check(empty == KvRowScalePlan::Capture, "a missing table captures (again)");
    resolution = ninfer::product::kv_rowscale_persist_resolve(kLayers, kHeads, kDim, true, "0-1:nvfp4");
    check(resolution.plan == KvRowScalePlan::Capture, "capture");
    fs::remove(table, ec);
    ninfer::product::kv_rowscale_persist_finish();
    check(!fs::exists(table), "a capture with no records writes no table at all");

    // 10. explicit off / path: the loop stands down entirely.
    KvRowScalePersistConfig disabled = make_config(artifact, base_knobs(), false, false);
    disabled.enabled = false;
    check(ninfer::product::kv_rowscale_persist_begin(std::move(disabled)) == KvRowScalePlan::Disabled,
          "an explicit --kv-row-scale owns the run and disables the loop");
    check(ninfer::product::kv_rowscale_persist_resolve(kLayers, kHeads, kDim, true, "0-1:nvfp4").plan ==
              KvRowScalePlan::Disabled,
          "and plan time agrees");
    // --recalibrate with the loop disabled must not capture either.
    resolution = ninfer::product::kv_rowscale_persist_resolve(kLayers, kHeads, kDim, true, "0-1:nvfp4");
    check(resolution.apply_spec == "auto", "the loop's fallback is the baked table");

    check(ninfer::product::kv_rowscale_spec_is_auto(""), "empty spec is auto");
    check(ninfer::product::kv_rowscale_spec_is_auto("auto"), "'auto' is auto");
    check(!ninfer::product::kv_rowscale_spec_is_auto("off"), "'off' is not");
    check(!ninfer::product::kv_rowscale_spec_is_auto("/tmp/t.kvrs"), "a path is not");

    fs::remove_all(root, ec);
}


// ---------------------------------------------------------------------------
// THE IDENTITY, THE ROPE KNOB, AND THE TWO GATES THAT USED TO LIVE ONLY IN THE
// PRE-FLIGHT.  Every check below is a case that used to be SILENTLY ACCEPTED.
// ---------------------------------------------------------------------------
void test_identity_and_dead_gates() {
    const fs::path root = fs::temp_directory_path() / "ninfer_kvrowscale_identity";
    std::error_code ec;
    fs::remove_all(root, ec);
    fs::create_directories(root / "frames", ec);
    install_test_rotation();
    g_rotation_enabled = false;

    constexpr std::uint32_t layers = 1;
    constexpr std::uint32_t heads = 1;
    constexpr std::uint32_t dim = 4;
    constexpr std::uint32_t tokens = 2;
    const std::vector<float> amplitude(dim, 1.0f);
    const auto k = constant_channel_k(dim, heads, tokens, amplitude);
    const std::vector<float> v(static_cast<std::size_t>(dim) * heads * tokens, 0.5f);
    std::vector<std::uint16_t> words;
    ninfer::product::KvRowScaleBakeStats stats;
    std::string err;

    // ---- THE ROPE REGIME ------------------------------------------------------
    KvRowScaleConfigKnobs plain;
    KvRowScaleConfigKnobs yarn = plain;
    yarn.rope_regime = 1;
    const std::uint64_t plain_fp = ninfer::product::kv_rowscale_config_fingerprint(plain);
    const std::uint64_t yarn_fp = ninfer::product::kv_rowscale_config_fingerprint(yarn);
    check(plain_fp == kHistoricalDefaultFingerprint,
          "the default rope regime does not move any existing table's hash: " +
              ninfer::product::kv_rowscale_hex48(plain_fp));
    check(KvRowScaleConfigKnobs{}.rope_regime == 0, "the default is the model's own rope");
    check(yarn_fp != plain_fp,
          "and --yarn is a DIFFERENT configuration, so neither table validates for the other");
    check(ninfer::product::kv_rowscale_hex48(yarn_fp) !=
              ninfer::product::kv_rowscale_hex48(plain_fp),
          "and the two tags differ, not just the numbers");

    // ---- THE IDENTITY TEXT FORM ----------------------------------------------
    std::uint64_t parsed = 0;
    std::uint32_t version = 0;
    std::string parse_err;
    const std::string text = ninfer::product::kv_rowscale_frame_identity_text(
        kFixtureFingerprint, ninfer::product::kKvRowScaleFrameIdentityVersion);
    check(text.size() == 14 && text[12] == ':',
          "the identity text is '<12 hex>:<version>': " + text);
    check(ninfer::product::kv_rowscale_frame_identity_parse(text.c_str(), parsed, version, parse_err) &&
              parsed == kFixtureFingerprint &&
              version == ninfer::product::kKvRowScaleFrameIdentityVersion,
          "and it round trips: " + parse_err);
    check(!ninfer::product::kv_rowscale_frame_identity_parse("", parsed, version, parse_err),
          "an empty identity is refused: " + parse_err);
    check(!ninfer::product::kv_rowscale_frame_identity_parse("123456789ABC:1", parsed, version,
                                                             parse_err),
          "an uppercase identity is refused: " + parse_err);
    check(!ninfer::product::kv_rowscale_frame_identity_parse("123456789abc:2", parsed, version,
                                                             parse_err),
          "a producer version this build does not write is refused: " + parse_err);
    check(!ninfer::product::kv_rowscale_frame_identity_parse("123456789abc", parsed, version,
                                                             parse_err),
          "a missing version is refused: " + parse_err);

    // ---- ALL-ZERO RESERVED IS A REFUSAL, NOT A WILDCARD ----------------------
    // ALL SIXTEEN reserved bytes zero: identity_fingerprint == 0 AND producer_version == 0 AND
    // flags == 0, which is exactly the frame kv_calibration.h writes with no
    // NINFER_KV_ROWSCALE_IDENTITY in the environment. `version=0` is therefore part of the
    // fixture, not an accident of the default argument.
    write_frame(root / "frames" / "0.kvc", 0, dim, heads, k, v, /*identity=*/0, /*version=*/0);
    check(!ninfer::product::kv_rowscale_solve_frames(root / "frames", layers, heads, dim, 0,
                                                     kFixtureFingerprint, words, stats, err),
          "an UNSTAMPED frame (all-zero reserved) produces NO table");
    check(stats.frames_unnamed == 1, "and is counted as unnamed, not as a plain skip");
    check(err.find("NO configuration identity") != std::string::npos,
          "and the refusal names the cause: " + err);
    check(err.find("clear") != std::string::npos,
          "and tells the operator what to do about it: " + err);

    // ---- A FRAME FOR ANOTHER CONFIGURATION IS REFUSED BY NAME ----------------
    // ---- A ZERO FINGERPRINT WITH A NON-ZERO VERSION IS **FOREIGN**, NOT UNNAMED ----------
    // THE DISCRIMINATING CASE, and the one that keeps the two counters from being one counter.
    // "No stamp is not a stamp" is defined on the SIXTEEN reserved bytes being all zero
    // (product/kv_rowscale_frame_identity.h:28-33). A frame that stamps producer version 1 and
    // leaves the fingerprint at 0 is a stamped frame whose claimed configuration is
    // 000000000000 -- a foreign configuration, refused with `was captured for KV configuration`,
    // NOT with the unnamed wording. A solve that folded the two together would pass the block
    // above while losing the distinction the identity field exists to draw.
    write_frame(root / "frames" / "0.kvc", 0, dim, heads, k, v, /*identity=*/0,
                ninfer::product::kKvRowScaleFrameIdentityVersion);
    check(!ninfer::product::kv_rowscale_solve_frames(root / "frames", layers, heads, dim, 0,
                                                     kFixtureFingerprint, words, stats, err),
          "a zero fingerprint with a live version produces NO table");
    check(stats.frames_foreign == 1 && stats.frames_unnamed == 0,
          "and it is counted FOREIGN, not unnamed: the reserved bytes are not all zero");
    check(err.find("was captured for KV configuration") != std::string::npos,
          "and the refusal is the foreign one, naming the configuration it claims: " + err);

    // ---- A FRAME FOR ANOTHER CONFIGURATION IS REFUSED BY NAME ----------------
    write_frame(root / "frames" / "0.kvc", 0, dim, heads, k, v, kFixtureFingerprint + 1);
    check(!ninfer::product::kv_rowscale_solve_frames(root / "frames", layers, heads, dim, 0,
                                                     kFixtureFingerprint, words, stats, err),
          "a frame captured for another KV configuration produces NO table");
    check(stats.frames_foreign == 1, "and is counted as foreign");
    check(err.find("was captured for KV configuration") != std::string::npos,
          "and the refusal names both configurations: " + err);

    // ---- AN UNKNOWN PRODUCER VERSION IS REFUSED ------------------------------
    write_frame(root / "frames" / "0.kvc", 0, dim, heads, k, v, kFixtureFingerprint, /*version=*/7);
    check(!ninfer::product::kv_rowscale_solve_frames(root / "frames", layers, heads, dim, 0,
                                                     kFixtureFingerprint, words, stats, err),
          "a frame from a producer version this build does not read produces NO table");
    check(err.find("producer version") != std::string::npos, "and names it: " + err);

    // ---- AN UNKNOWN FLAG IS REFUSED -----------------------------------------
    write_frame(root / "frames" / "0.kvc", 0, dim, heads, k, v, kFixtureFingerprint,
                ninfer::product::kKvRowScaleFrameIdentityVersion, /*flags=*/2);
    check(!ninfer::product::kv_rowscale_solve_frames(root / "frames", layers, heads, dim, 0,
                                                     kFixtureFingerprint, words, stats, err),
          "a frame carrying a flag this build does not know produces NO table");

    // ---- A MIXED DIRECTORY IS REFUSED WHOLE ---------------------------------
    write_frame(root / "frames" / "0.kvc", 0, dim, heads, k, v, kFixtureFingerprint);
    write_frame(root / "frames" / "1.kvc", 0, dim, heads, k, v, /*identity=*/0, /*version=*/0);
    check(!ninfer::product::kv_rowscale_solve_frames(root / "frames", layers, heads, dim, 0,
                                                     kFixtureFingerprint, words, stats, err),
          "one stamped and one unnamed frame is refused WHOLE, never solved in part");
    check(stats.frames_used == 1 && stats.frames_unnamed == 1,
          "the stamped frame was read and then NOT used (no partial table)");
    fs::remove(root / "frames" / "1.kvc", ec);

    // ---- A CORRECTLY STAMPED FRAME SOLVES, AND REPORTS ITS PRODUCER ----------
    check(ninfer::product::kv_rowscale_solve_frames(root / "frames", layers, heads, dim, 0,
                                                     kFixtureFingerprint, words, stats, err),
          "a correctly stamped frame solves: " + err);
    check(stats.producer_seen && stats.producer_fingerprint == kFixtureFingerprint &&
              stats.producer_version == ninfer::product::kKvRowScaleFrameIdentityVersion,
          "and the solve reports the PRODUCER's identity -- the value the table is stamped from");

    // ---- THE LAUNDERING: a run whose frames are not its own writes NO table ---
    {
        const fs::path artifact = root / "launder.ninfer";
        write_artifact(artifact);
        ::unsetenv("NINFER_KV_CALIB_DIR");
        std::string stale_note_err;
        (void)stale_note_err;
        const KvRowScalePersistConfig config = make_config(artifact, base_knobs(), false, false);
        (void)ninfer::product::kv_rowscale_persist_begin(config);
        const KvRowScaleResolution armed =
            ninfer::product::kv_rowscale_persist_resolve(layers, heads, dim, true, "0:nvfp4");
        check(armed.plan == KvRowScalePlan::Capture, "the capture is armed");
        check(std::getenv("NINFER_KV_CALIB_DIR") != nullptr, "and the capture directory with it");
        check(std::getenv(ninfer::product::kKvRowScaleFrameIdentityEnv) != nullptr,
              "and the identity the capture must stamp: " +
                  std::string(std::getenv(ninfer::product::kKvRowScaleFrameIdentityEnv) == nullptr
                                  ? "<unset>"
                                  : std::getenv(ninfer::product::kKvRowScaleFrameIdentityEnv)));
        // Frames that belong to a DIFFERENT configuration land in the same directory.
        fs::create_directories(ninfer::product::kv_rowscale_records_path(config.table), ec);
        write_frame(ninfer::product::kv_rowscale_records_path(config.table) / "0.kvc", 0, dim,
                    heads, k, v, /*identity=*/kFixtureFingerprint + 5);
        ninfer::product::kv_rowscale_persist_finish();
        check(!fs::exists(config.table),
              "frames from another run are NOT laundered into a table stamped with this run's "
              "key: no table was written at all");
    }

    // ---- THE DEAD STALENESS GATE, AT THE AUTHORITATIVE POINT -----------------
    const fs::path artifact = root / "m.ninfer";
    write_artifact(artifact);
    KvRowScalePersistConfig config = make_config(artifact, base_knobs(), false, false);
    (void)ninfer::product::kv_rowscale_persist_begin(config);
    std::string table_blob;
    const std::vector<std::uint16_t> ones(static_cast<std::size_t>(layers) * heads * dim, 0x3F80);
    check(ninfer::product::kv_rowscale_build_table(
              ones, layers, heads, dim, 0,
              ninfer::product::kv_rowscale_make_tag(config.fingerprint), true, table_blob, err),
          "build an applicable table for the staleness case: " + err);
    {
        std::ofstream file(config.table, std::ios::binary | std::ios::trunc);
        file.write(table_blob.data(), static_cast<std::streamsize>(table_blob.size()));
    }
    fs::last_write_time(config.table, fs::file_time_type::clock::now() - std::chrono::hours(1), ec);
    fs::last_write_time(artifact, fs::file_time_type::clock::now(), ec);
    check(ninfer::product::kv_rowscale_persist_begin(
              make_config(artifact, base_knobs(), false, false)) == KvRowScalePlan::Capture,
          "the PRE-FLIGHT rejects the stale table and demands a capture");
    {
        const KvRowScaleResolution stale =
            ninfer::product::kv_rowscale_persist_resolve(layers, heads, dim, true, "0:nvfp4");
        check(stale.plan == KvRowScalePlan::Capture,
              "and the COMMIT POINT re-decides the same staleness instead of applying it: " +
                  stale.report);
        check(stale.report.find("stale") != std::string::npos,
              "naming the staleness: " + stale.report);
        check(!fs::exists(config.table), "the table the pre-flight refused is out of the way");
        check(fs::exists(ninfer::product::kv_rowscale_rejected_path(config.table)),
              "and kept as evidence, not deleted");
    }

    // ---- THE .skip NOTE, IN THE AUTHORITATIVE HALF ---------------------------
    {
        std::string note_err;
        const std::string note_text = ninfer::product::kv_rowscale_hex48(config.fingerprint) +
                                      " an earlier run recorded this: no NVFP4 layer\n";
        check(ninfer::product::detail::kv_rowscale_write_file_atomic(
                  ninfer::product::kv_rowscale_skip_note_path(config.table), note_text, note_err),
              "write a .skip note: " + note_err);
        // It AGREES with a stack that resolves no domain: the note's OWN reason is what
        // the operator is shown.  It used to be replaced by a fixed sentence.
        const KvRowScaleResolution agreeing = ninfer::product::kv_rowscale_persist_resolve(
            layers, heads, dim, /*domain_active=*/false, "0:bf16");
        check(agreeing.plan == KvRowScalePlan::NothingToCalibrate, "an agreeing note keeps its plan");
        check(agreeing.report.find("the .skip note recorded by an earlier run") !=
                  std::string::npos,
              "and the note's own reason is reported verbatim: " + agreeing.report);
        check(agreeing.report.find("no NVFP4 layer") != std::string::npos,
              "including the reason text itself: " + agreeing.report);
        // It is CONTRADICTED by a stack that does: named, and moved aside, never silent.
        const KvRowScaleResolution contradicted = ninfer::product::kv_rowscale_persist_resolve(
            layers, heads, dim, /*domain_active=*/true, "0:nvfp4");
        check(contradicted.report.find("contradicted") != std::string::npos,
              "the commit point NAMES the contradiction between the note and the live stack: " +
                  contradicted.report);
        check(contradicted.report.find("no NVFP4 layer") != std::string::npos,
              "and quotes the note's own reason: " + contradicted.report);
        check(fs::exists(ninfer::product::kv_rowscale_skip_note_stale_path(config.table)),
              "the contradicted note is kept as evidence");
        check(!fs::exists(ninfer::product::kv_rowscale_skip_note_path(config.table)),
              "and is out of the way, so the two halves agree from the next run on");
        fs::remove(ninfer::product::kv_rowscale_skip_note_stale_path(config.table), ec);
    }

    fs::remove_all(root, ec);
}

}  // namespace

int main() {
    try {
        test_tag_and_fingerprint();
        test_paths_and_writer();
        test_solve_and_rotated_domain();
        test_loop_end_to_end();
        test_identity_and_dead_gates();
    } catch (const std::exception& error) {
        std::cerr << "FAIL: unexpected exception: " << error.what() << '\n';
        return 1;
    }
    if (g_failures != 0) {
        std::cerr << g_failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "kv_rowscale_persist: all checks passed\n";
    return 0;
}
