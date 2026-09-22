#pragma once

// The `.kvc` calibration-frame header's 16 RESERVED bytes (offsets 48..63), given
// a meaning.  ONE DEFINITION, for both sides of the format.
//
// WHY THIS FILE EXISTS AT ALL.  The frame header is written by
// targets/qwen3_6/impl/runtime/kv_calibration.h (a TARGET header, which pulls
// cuda_runtime.h and the target's own namespace macro) and read by
// product/kv_rowscale_bake.h (deliberately host-only and target-free, so the whole
// solve is unit-testable with plain g++).  Neither may include the other.  The
// tree's own rule -- turn_recall_journal.h: "a second literal that happens to agree
// today is how the previous ledger went stale" -- is that the numbers the two sides
// must agree on get exactly one home.  This is that home: <cstdint> and <string>,
// nothing else, so it can be included from either side without moving the layering.
//
// WHAT IT FIXES.  The header has reserved these 16 bytes since the format was
// written and NOTHING ever read or wrote them, so a calibration directory carried no
// record of WHICH configuration, WHICH producer and WHICH run produced its frames.
// The solve therefore accepted frames on geometry alone, and the table it built was
// stamped with the CURRENT run's fingerprint whether or not that run produced them.
// Both halves of that are closed by putting the producing run's identity here.
//
//   offset 48..55  u64 LE  config_fingerprint  (the 48-bit
//                          kv_rowscale_config_fingerprint of the PRODUCING run)
//   offset 56..59  u32 LE  producer_version    (this producer schema)
//   offset 60..63  u32 LE  flags               (bit0.. reserved; MUST be 0 today)
//
// ALL-ZERO IS A REFUSAL, NOT A WILDCARD.  Every artifact written before this field
// existed has all-zero reserved bytes, and no stamp is not a stamp: a frame whose
// identity is absent is REFUSED BY NAME by the solve, with the fix in the message.
// Reading zero as "anything goes" would restore exactly the silent acceptance this
// field exists to remove, and would do it in the deployment path (a copied, restored
// or image-layered calibration directory) that the owner cares about most.

#include <cstdint>
#include <string>

namespace ninfer::product {

// The ledger.  These are the ONLY places these counts are written down, and they must
// add up to the reserved extent exactly.  If a field is added, removed or moved, fix
// the table above and this fires -- the completeness check turn_recall_journal.h:387
// applies to its own record for the same reason.
inline constexpr std::uint32_t kKvRowScaleFrameHeaderBytes      = 64U;
inline constexpr std::uint32_t kKvRowScaleFrameIdentityOffset   = 48U;
inline constexpr std::uint32_t kKvRowScaleFrameVersionOffset    = 56U;
inline constexpr std::uint32_t kKvRowScaleFrameFlagsOffset      = 60U;
inline constexpr std::uint32_t kKvRowScaleFrameReservedBytes    = 16U;  // reserved[4] @48
inline constexpr std::uint32_t kKvRowScaleFrameIdentityBytes    = 8U;   // config_fingerprint
inline constexpr std::uint32_t kKvRowScaleFrameVersionBytes     = 4U;   // producer_version
inline constexpr std::uint32_t kKvRowScaleFrameFlagsBytes       = 4U;   // flags
static_assert(kKvRowScaleFrameIdentityBytes + kKvRowScaleFrameVersionBytes +
                      kKvRowScaleFrameFlagsBytes ==
                  kKvRowScaleFrameReservedBytes,
              "the frame identity must be COMPLETE AND DISJOINT: 8 + 4 + 4 of the 16 bytes the "
              ".kvc header has always reserved. If a field is added, removed or moved, fix the "
              "table above and this fires");
static_assert(kKvRowScaleFrameIdentityOffset == 48U &&
                  kKvRowScaleFrameHeaderBytes == kKvRowScaleFrameIdentityOffset +
                                                    kKvRowScaleFrameReservedBytes,
              "the .kvc header is one 64-byte record whose last 16 bytes are reserved[4]");

// The producer schema. Bump ONLY when a reader of this version would misread a frame
// written by the previous one; a bump makes every older frame a NAMED refusal.
inline constexpr std::uint32_t kKvRowScaleFrameIdentityVersion = 1U;

// Flags. Bit 0..31 are reserved and the READER REFUSES ANY BIT IT DOES NOT KNOW, so a
// future writer cannot smuggle a semantic past an old reader by setting one.
inline constexpr std::uint32_t kKvRowScaleFrameKnownFlags = 0U;

// The variable the loop installs (kv_rowscale_persist.h, resolve) and the capture
// reads once, at its first call -- the same mechanism and the same moment as
// NINFER_KV_CALIB_DIR.  ABSENT is legal: the frame is written UNSTAMPED and the solve
// refuses it by name.  That keeps a hand-run `NINFER_KV_CALIB_DIR=... ninfer` capture
// usable by tools/calib (which reads reserved as opaque) while making it unusable as
// the source of a table.
inline constexpr char kKvRowScaleFrameIdentityEnv[] = "NINFER_KV_ROWSCALE_IDENTITY";

// "<12 hex digits>:<version>".  One line, no whitespace, so a variable cannot smuggle
// a second value in behind the first.
[[nodiscard]] inline std::string kv_rowscale_frame_identity_text(std::uint64_t fingerprint,
                                                                 std::uint32_t version) {
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%012llx:%u",
                  static_cast<unsigned long long>(fingerprint & 0xFFFFFFFFFFFFULL), version);
    return std::string(buffer);
}

// The exact inverse, and the ONLY reader of the text form.  Returns false with a reason
// that names what was seen and what was wanted; the caller refuses on false, never
// substitutes a zero.
[[nodiscard]] inline bool kv_rowscale_frame_identity_parse(const char* text,
                                                          std::uint64_t& fingerprint,
                                                          std::uint32_t& version,
                                                          std::string& err) {
    const std::string value = text != nullptr ? std::string(text) : std::string();
    if (value.size() != 12U + 1U + std::to_string(kKvRowScaleFrameIdentityVersion).size() ||
        value[12] != ':') {
        err = "'" + value + "' is not '<12 hex digits>:<version>'";
        return false;
    }
    std::uint64_t digits = 0;
    for (std::size_t i = 0; i < 12U; ++i) {
        const char c = value[i];
        std::uint64_t digit = 0;
        if (c >= '0' && c <= '9') { digit = static_cast<std::uint64_t>(c - '0'); }
        else if (c >= 'a' && c <= 'f') { digit = static_cast<std::uint64_t>(c - 'a' + 10); }
        else {
            err = "'" + value + "': byte " + std::to_string(i) + " is not a lowercase hex digit";
            return false;
        }
        digits = digits * 16ULL + digit;
    }
    std::uint32_t parsed = 0;
    for (std::size_t i = 13U; i < value.size(); ++i) {
        if (value[i] < '0' || value[i] > '9') {
            err = "'" + value + "': the version is not decimal";
            return false;
        }
        parsed = parsed * 10U + static_cast<std::uint32_t>(value[i] - '0');
    }
    if (parsed != kKvRowScaleFrameIdentityVersion) {
        err = "'" + value + "': producer version " + std::to_string(parsed) +
              " is not the " + std::to_string(kKvRowScaleFrameIdentityVersion) +
              " this build writes";
        return false;
    }
    fingerprint = digits;
    version     = parsed;
    return true;
}

}  // namespace ninfer::product
