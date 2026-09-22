// src/spec/sum_dir_determined_test.cpp
//
// [INDEXGAPS] THE `DETERMINED` CONJUNCTION, EXECUTED. Positive arms, controls, and the runs that
// MUST GO RED.
//
// WHAT THIS FILE IS FOR. `DETERMINED` was written down as a conjunction (dl/vectorkey/REPORT.md
// sec.1.3, dl/THIRDKIND.md sec.3.2) and every clause had a carrier in the tree, but the four clauses
// had never been put into ONE EXECUTION that can FAIL. This is that execution, and this file is
// where it is RUN rather than asserted.
//
// HOW IT IS BUILT, AND WHY THAT WAY. It is host-only and std-only -- it includes
// `spec/sum_dir_determined.h`, which includes `spec/sum_dir_query_key.h`,
// `spec/sum_dir_model_binding.h`, `spec/sum_dir.h` and `spec/recall_identity.h`, and none of the
// five brings in CUDA, an engine header or the ninfer library. So this target is registered as a RAW
// `add_executable` with no library link, the same shape `tests/test_cold_slot_release_bytes.cpp`
// uses and for the same reason: it must keep building and running while the engine's device link
// does not.
//
// THE ARMS, AND WHAT EACH ONE CLAIMS
//
//   POSITIVE / CONTROL
//     `green`                        key from the block's own ids, plan names that row, cargo ids equal
//                                    => the conjunction HOLDS
//     `ctl-ambiguous`                two rows with the SAME content                 => refused-ambiguous
//     `ctl-nocand`                   a key for a block that is not in the store     => refused-no-candidate
//     `ctl-notsorted`                no sort_rows()                                 => refused-not-sorted, NOT a throw
//     `ctl-unset`                    a key with no ids / an over-long span          => refused-key-unset
//     `binding-seal`                 the fourth artefact's wire round-trip, and one flipped bit
//                                    => sealed / unsealed
//     `unification`                  recall_identity_tokens == sum_dir_block_digest on this corpus
//                                    (the tree's own tests/test_recall_identity.cpp sec.4 fact,
//                                    re-checked here because the query key is produced by the
//                                    directory's function while (D3) is written with the engine's)
//
//   NEGATIVE CONTROLS -- the mutant MUST BE REFUSED, and the refusal must be NAMED
//     `neg-len`               one id too few in the read-back          => refused-length-mismatch
//     `neg-ident`             one id XOR 0x5A5A5A5A in the read-back   => refused-identity-mismatch
//     `neg-substitute`        the plan names a DIFFERENT row           => refused-substituted
//     `neg-key-missing`       the key is another block's               => refused-no-candidate
//     `neg-weights-moved`     the CURRENT fingerprint's weights differ => refused-model-changed
//     `neg-tokenizer-moved`   the CURRENT fingerprint's tokenizer differs
//     `neg-other-directory`   the stored binding belongs to ANOTHER directory
//     `neg-other-directory-named`  the same, asserting the SPECIFIC name refused-directory-mismatch
//     `neg-unset-foreign`     the stored binding carries no model      => refused-fingerprint-unset
//
//   THE RUNS THAT MUST GO RED -- `--drop <clause>`
//
//     This is the load-bearing half. A conjunction whose clauses cannot be individually broken
//     cannot be shown to be anything: a check that passes because it never reaches a clause looks
//     exactly like one that passes because the clause holds. `--drop D1|D2|D3|D4|MODEL|POSITIONAL`
//     breaks exactly ONE clause and re-runs EVERY arm; the mutant the full clause set refuses is
//     then ACCEPTED, the corresponding arm's MUST-REFUSE check fails, and the AGGREGATE exits 1.
//
//     So `--arm all` is GREEN (rc=0) and each of the six `--drop` runs is RED (rc=1), with the
//     failing arm's expectation printed verbatim. The red is produced by the CHECK admitting a
//     mutant that it refuses when whole -- not written into the test as an expected failure.
//
// USAGE
//   sum_dir_determined_test --list
//   sum_dir_determined_test --arm all
//   sum_dir_determined_test --arm all --drop D3
//   sum_dir_determined_test --arm neg-ident [--drop D3]
//
// rc: 0 = every arm behaved, 1 = at least one arm did not, 2 = NOT MEASURED (a setup failure or an
// unknown arm/clause name -- never a verdict).

#include "spec/recall_identity.h"
#include "spec/sum_dir_determined.h"
#include "spec/sum_dir_model_binding.h"
#include "spec/sum_dir_query_key.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace sd = ninfer::spec::sum_dir;
namespace det = ninfer::spec::sum_dir::determined;
namespace qk = ninfer::spec::sum_dir::query_key;
namespace mb = ninfer::spec::sum_dir::model_binding;
namespace ri = ninfer::spec::recall_identity;

namespace {

int g_checks = 0;
int g_failures = 0;

void check(bool ok, const std::string& what) {
    ++g_checks;
    if (!ok) {
        ++g_failures;
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    }
}

// ---------------------------------------------------------------------------
// the corpus -- ids explicit, a store of four distinct blocks, and a fingerprint pair
// ---------------------------------------------------------------------------

constexpr std::uint32_t kBlockTokens = sd::kSumDirBlockTokens; // 64

[[nodiscard]] std::vector<std::uint32_t> block_ids(std::uint32_t seed) {
    std::vector<std::uint32_t> ids;
    ids.reserve(kBlockTokens);
    for (std::uint32_t i = 0; i < kBlockTokens; ++i) {
        ids.push_back((seed * 1000003U + i * 37U + 11U) % 250000U);
    }
    return ids;
}

[[nodiscard]] sd::SumDir build_directory() {
    sd::SumDir directory(0x1234ULL, kBlockTokens);
    for (std::uint32_t b = 0; b < 4U; ++b) {
        const std::vector<std::uint32_t> ids = block_ids(b + 1U);
        (void)directory.append(ids.data(), ids.size(), b * kBlockTokens, 1U, sd::SumDirCodec::Int8,
                               static_cast<std::int32_t>(b), b);
    }
    directory.sort_rows();
    return directory;
}

// The same, plus a SECOND row carrying block 0's content again -- the duplicate (D1) must count,
// and it is also the store on which the forbidden positional fallback would pick a winner.
[[nodiscard]] sd::SumDir build_directory_with_duplicate() {
    sd::SumDir directory(0x1234ULL, kBlockTokens);
    for (std::uint32_t b = 0; b < 4U; ++b) {
        const std::vector<std::uint32_t> ids = block_ids(b + 1U);
        (void)directory.append(ids.data(), ids.size(), b * kBlockTokens, 1U, sd::SumDirCodec::Int8,
                               static_cast<std::int32_t>(b), b);
    }
    const std::vector<std::uint32_t> again = block_ids(1U);
    (void)directory.append(again.data(), again.size(), 4U * kBlockTokens, 2U,
                           sd::SumDirCodec::Int8, 4, 4U);
    directory.sort_rows();
    return directory;
}

constexpr std::uint64_t kTokenizer = 0x746f6b656e697a31ULL; // "tokeniz1"
constexpr std::uint64_t kWeights   = 0x7765696768747331ULL; // "weights1"
constexpr std::uint64_t kGenIds    = 0x67656e6964733031ULL; // "genids01"

[[nodiscard]] mb::SumDirModelFingerprint fingerprint(std::uint64_t weights,
                                                    std::uint64_t tokenizer = kTokenizer,
                                                    std::uint64_t generation = kGenIds) {
    mb::SumDirModelFingerprint f;
    f.tokenizer_identity = tokenizer;
    f.weights_identity = weights;
    f.generation_ids_digest = generation;
    return f;
}

// ---------------------------------------------------------------------------
// arm scaffolding
// ---------------------------------------------------------------------------

struct ArmResult {
    bool ok = true;
    std::string detail;
};

using ArmBody = ArmResult (*)(const det::SumDirDeterminedClauses&);

[[nodiscard]] std::string verdict_line(det::SumDirDeterminedVerdict v,
                                      const det::SumDirDeterminedReport& r) {
    return std::string("verdict=") + det::sum_dir_determined_verdict_name(v) +
           " candidates=" + std::to_string(r.candidates) +
           " model=" + mb::sum_dir_model_binding_verdict_name(r.model_verdict) +
           " d1=" + (r.clause_d1 ? "1" : "0") + " d2=" + (r.clause_d2 ? "1" : "0") +
           " d3=" + (r.clause_d3 ? "1" : "0") + " d4=" + (r.clause_d4 ? "1" : "0");
}

// ---------------------------------------------------------------------------
// the positive arm
// ---------------------------------------------------------------------------

[[nodiscard]] ArmResult arm_green(const det::SumDirDeterminedClauses& clauses) {
    ArmResult result;
    const sd::SumDir directory = build_directory();
    const std::vector<std::uint32_t> ids = block_ids(1U);

    qk::SumDirQueryKeyReport key_report;
    const qk::SumDirQueryKey key = qk::sum_dir_query_key_from_ids(ids, &key_report);
    check(key_report.on_axis, "green: the producer put the key on the axis");
    check(key.token_count == kBlockTokens, "green: the key carries the block's token count");
    check(key_report.refused_empty_span == false && key_report.refused_over_block == false,
          "green: neither refusal fired");
    check(qk::sum_dir_query_key_is_identity_of(key, ids.data(),
                                               static_cast<std::uint32_t>(ids.size())),
          "green: the key is the identity of exactly those ids");

    const qk::SumDirQueryKeyLookup lookup = qk::sum_dir_query_key_lookup(directory, key);
    check(lookup.verdict == qk::SumDirQueryKeyVerdict::Determined,
          "green: (D1) the key names exactly one row");
    check(lookup.candidates == 1U, "green: (D1) candidates == 1");

    det::SumDirDeterminedReadBack read_back;
    read_back.ids_present = true;
    read_back.row = lookup.first_row;
    read_back.ids = ids;

    const mb::SumDirModelBindingHeader binding =
        mb::sum_dir_model_binding_make(fingerprint(kWeights), directory.payload_digest());

    det::SumDirDeterminedReport report;
    const det::SumDirDeterminedVerdict verdict = det::sum_dir_determined(
        directory, key, lookup.first_row, read_back, binding, fingerprint(kWeights), clauses,
        &report);

    result.detail = verdict_line(verdict, report);
    check(verdict == det::SumDirDeterminedVerdict::Determined,
          "green: the conjunction holds on the untouched store");
    check(report.clause_d1 && report.clause_d2 && report.clause_d3 && report.clause_d4,
          "green: every clause is true, so the green verdict is not vacuous");
    check(report.model_verdict == mb::SumDirModelBindingVerdict::Bound,
          "green: the fourth artefact binds");
    check(report.read_back_tokens == report.row_tokens && report.row_tokens == kBlockTokens,
          "green: (D2) the length came from the ROW and is 64");
    check(report.rederived_identity == report.row_identity,
          "green: (D3) the re-derived identity equals the stored one");
    result.ok = g_failures == 0;
    return result;
}

// ---------------------------------------------------------------------------
// controls
// ---------------------------------------------------------------------------

[[nodiscard]] ArmResult arm_ctl_ambiguous(const det::SumDirDeterminedClauses& clauses) {
    ArmResult result;
    const sd::SumDir directory = build_directory_with_duplicate();
    const std::vector<std::uint32_t> ids = block_ids(1U);
    const qk::SumDirQueryKey key = qk::sum_dir_query_key_from_ids(ids);
    const qk::SumDirQueryKeyLookup lookup = qk::sum_dir_query_key_lookup(directory, key);

    det::SumDirDeterminedReadBack read_back;
    read_back.ids_present = true;
    read_back.row = lookup.first_row;
    read_back.ids = ids;
    const mb::SumDirModelBindingHeader binding =
        mb::sum_dir_model_binding_make(fingerprint(kWeights), directory.payload_digest());
    det::SumDirDeterminedReport report;
    const det::SumDirDeterminedVerdict verdict = det::sum_dir_determined(
        directory, key, lookup.first_row, read_back, binding, fingerprint(kWeights), clauses,
        &report);

    result.detail = verdict_line(verdict, report);
    check(report.candidates == 2U, "ctl-ambiguous: the cardinality is COUNTED, and it is 2");
    check(verdict == det::SumDirDeterminedVerdict::RefusedAmbiguous,
          "ctl-ambiguous: (D1) refuses an ambiguous key -- no score and no POSITION ever crowned one");
    result.ok = g_failures == 0;
    return result;
}

[[nodiscard]] ArmResult arm_ctl_nocand(const det::SumDirDeterminedClauses& clauses) {
    ArmResult result;
    const sd::SumDir directory = build_directory();
    const std::vector<std::uint32_t> ids = block_ids(99U); // not in the store
    const qk::SumDirQueryKey key = qk::sum_dir_query_key_from_ids(ids);

    det::SumDirDeterminedReadBack read_back;
    read_back.ids_present = true;
    read_back.row = 0;
    read_back.ids = ids;
    const mb::SumDirModelBindingHeader binding =
        mb::sum_dir_model_binding_make(fingerprint(kWeights), directory.payload_digest());
    det::SumDirDeterminedReport report;
    const det::SumDirDeterminedVerdict verdict = det::sum_dir_determined(
        directory, key, 0, read_back, binding, fingerprint(kWeights), clauses, &report);

    result.detail = verdict_line(verdict, report);
    check(report.candidates == 0U, "ctl-nocand: the cardinality is 0, and it is COUNTED");
    check(verdict == det::SumDirDeterminedVerdict::RefusedNoCandidate,
          "ctl-nocand: a key with no row is a NAMED no-candidate, not a silent miss");
    result.ok = g_failures == 0;
    return result;
}

[[nodiscard]] ArmResult arm_ctl_notsorted(const det::SumDirDeterminedClauses& clauses) {
    ArmResult result;
    sd::SumDir directory(0x1234ULL, kBlockTokens);
    for (std::uint32_t b = 0; b < 4U; ++b) {
        const std::vector<std::uint32_t> ids = block_ids(b + 1U);
        (void)directory.append(ids.data(), ids.size(), b * kBlockTokens, 1U, sd::SumDirCodec::Int8,
                               static_cast<std::int32_t>(b), b);
    }
    // DELIBERATELY NOT SORTED.

    const std::vector<std::uint32_t> ids = block_ids(1U);
    const qk::SumDirQueryKey key = qk::sum_dir_query_key_from_ids(ids);

    // The tree's own `find()` THROWS in this state (sum_dir.h:1748). Prove that, so "we refuse
    // instead of throwing" is a reply to a real hazard and not a preference.
    bool threw = false;
    try {
        std::size_t begin = 0;
        std::size_t end = 0;
        (void)directory.find(key.identity, begin, end);
    } catch (const std::logic_error&) {
        threw = true;
    }
    check(threw, "ctl-notsorted: SumDir::find really throws before sort_rows(), as the header says");

    const qk::SumDirQueryKeyLookup lookup = qk::sum_dir_query_key_lookup(directory, key);
    check(lookup.verdict == qk::SumDirQueryKeyVerdict::RefusedNotSorted,
          "ctl-notsorted: the query-key lookup gives a NAMED refusal where find() would have thrown");

    det::SumDirDeterminedReadBack read_back;
    read_back.ids_present = true;
    read_back.row = 0;
    read_back.ids = ids;
    const mb::SumDirModelBindingHeader binding =
        mb::sum_dir_model_binding_make(fingerprint(kWeights), directory.payload_digest());
    det::SumDirDeterminedReport report;
    const det::SumDirDeterminedVerdict verdict = det::sum_dir_determined(
        directory, key, 0, read_back, binding, fingerprint(kWeights), clauses, &report);

    result.detail = verdict_line(verdict, report) + " threw_in_find=1";
    check(verdict == det::SumDirDeterminedVerdict::RefusedNotSorted,
          "ctl-notsorted: the conjunction refuses with the same name");
    result.ok = g_failures == 0;
    return result;
}

[[nodiscard]] ArmResult arm_ctl_unset(const det::SumDirDeterminedClauses& clauses) {
    ArmResult result;
    const sd::SumDir directory = build_directory();
    qk::SumDirQueryKeyReport key_report;
    const qk::SumDirQueryKey key = qk::sum_dir_query_key_from_ids(nullptr, 0U, &key_report);
    check(key_report.refused_empty_span, "ctl-unset: the producer NAMES the empty span");
    check(key.is_unset(), "ctl-unset: an empty span is not a key");

    const std::vector<std::uint32_t> too_long(kBlockTokens + 1U, 7U);
    qk::SumDirQueryKeyReport over;
    const qk::SumDirQueryKey key2 = qk::sum_dir_query_key_from_ids(too_long, &over);
    check(over.refused_over_block, "ctl-unset: the producer NAMES the over-long span");
    check(key2.is_unset(), "ctl-unset: an over-long span is not a key either");
    // A span at exactly the bound IS on the axis -- the bound is inclusive, not a truncation.
    qk::SumDirQueryKeyReport at_bound;
    const qk::SumDirQueryKey key3 =
        qk::sum_dir_query_key_from_ids(block_ids(1U), &at_bound);
    check(at_bound.on_axis && !key3.is_unset(),
          "ctl-unset: 64 ids are ON the axis (the bound is inclusive)");

    det::SumDirDeterminedReadBack read_back;
    read_back.ids_present = true;
    read_back.row = 0;
    read_back.ids = block_ids(1U);
    const mb::SumDirModelBindingHeader binding =
        mb::sum_dir_model_binding_make(fingerprint(kWeights), directory.payload_digest());
    det::SumDirDeterminedReport report;
    const det::SumDirDeterminedVerdict verdict = det::sum_dir_determined(
        directory, key, 0, read_back, binding, fingerprint(kWeights), clauses, &report);

    result.detail = verdict_line(verdict, report) + " empty_span=1 over_block=1";
    check(verdict == det::SumDirDeterminedVerdict::RefusedKeyUnset,
          "ctl-unset: a key that is not a key is refused by name");
    result.ok = g_failures == 0;
    return result;
}

[[nodiscard]] ArmResult arm_binding_seal(const det::SumDirDeterminedClauses&) {
    ArmResult result;
    const sd::SumDir directory = build_directory();
    const mb::SumDirModelBindingHeader binding =
        mb::sum_dir_model_binding_make(fingerprint(kWeights), directory.payload_digest());
    const std::vector<std::uint8_t> wire = mb::sum_dir_model_binding_serialize(binding);
    check(wire.size() == mb::kSumDirModelBindingHeaderBytes,
          "binding-seal: the artefact is one fixed 48-byte unit");

    mb::SumDirModelBindingLoadReport load;
    const mb::SumDirModelBindingHeader back = mb::sum_dir_model_binding_load(
        wire.data(), wire.size(), &load);
    check(load.ok && load.sealed, "binding-seal: the seal agrees on the round trip");
    check(back.tokenizer_identity == kTokenizer && back.weights_identity == kWeights &&
              back.generation_ids_digest == kGenIds,
          "binding-seal: all three components survive the round trip");
    check(back.directory_payload_digest == directory.payload_digest(),
          "binding-seal: the directory binding survives the round trip");

    // ONE BIT. The seal must notice, or it is decoration.
    std::vector<std::uint8_t> broken = wire;
    broken[16] ^= 0x01U; // inside weights_identity
    mb::SumDirModelBindingLoadReport broken_load;
    const mb::SumDirModelBindingHeader broken_header =
        mb::sum_dir_model_binding_load(broken.data(), broken.size(), &broken_load);
    check(!broken_load.ok && !broken_load.sealed,
          "binding-seal: a one-bit change is reported UNSEALED, not accepted");
    check(broken_header.magic != mb::kSumDirModelBindingMagic,
          "binding-seal: an unsealed header is returned with its magic zeroed, not half-believed");
    result.detail = "wire=48 sealed=1 one_bit=>unsealed";
    result.ok = g_failures == 0;
    return result;
}

[[nodiscard]] ArmResult arm_unification(const det::SumDirDeterminedClauses&) {
    ArmResult result;
    // The tree's own tests/test_recall_identity.cpp sec.4 asserts that the ENGINE's identity function
    // and the DIRECTORY's digest are the same 128 bits on the same tokens. It matters HERE because
    // the query key is produced by the directory's function while (D3) is written with the engine's:
    // if the two ever drifted, a key would stop matching the row it names.
    bool all_equal = true;
    for (std::uint32_t seed = 1U; seed <= 4U; ++seed) {
        const std::vector<std::uint32_t> ids = block_ids(seed);
        const sd::SumDirDigest via_directory = sd::sum_dir_block_digest(ids);
        std::vector<std::int32_t> signed_ids(ids.size());
        for (std::size_t i = 0; i < ids.size(); ++i) {
            signed_ids[i] = static_cast<std::int32_t>(ids[i]);
        }
        const ri::RecallIdentity via_identity =
            ri::recall_identity_tokens(signed_ids.data(), signed_ids.size());
        if (via_directory.lo != via_identity.lo || via_directory.hi != via_identity.hi) {
            all_equal = false;
        }
    }
    check(all_equal,
          "unification: recall_identity_tokens and sum_dir_block_digest agree on every block here");
    result.detail = all_equal ? "4/4 blocks: identical 128 bits" : "DIVERGENCE";
    result.ok = g_failures == 0;
    return result;
}

// ---------------------------------------------------------------------------
// the negative controls -- each mutant MUST BE REFUSED, under the NAMED verdict
// ---------------------------------------------------------------------------

enum class MutantKind : std::uint8_t {
    LengthShort,
    IdentityFlipped,
    PlanSubstituted,
    KeyMissing,
    WeightsMoved,
    TokenizerMoved,
    OtherDirectoryBinding,
    UnsetBinding,
};

[[nodiscard]] ArmResult run_mutant(const det::SumDirDeterminedClauses& clauses, MutantKind kind,
                                   det::SumDirDeterminedVerdict expected,
                                   const std::string& what) {
    ArmResult result;
    const sd::SumDir directory = build_directory();
    const std::vector<std::uint32_t> ids = block_ids(1U);
    const qk::SumDirQueryKey key = qk::sum_dir_query_key_from_ids(ids);
    const qk::SumDirQueryKeyLookup lookup = qk::sum_dir_query_key_lookup(directory, key);
    const std::size_t named = lookup.first_row;

    det::SumDirDeterminedReadBack read_back;
    read_back.ids_present = true;
    read_back.row = named;
    read_back.ids = ids;

    mb::SumDirModelBindingHeader binding =
        mb::sum_dir_model_binding_make(fingerprint(kWeights), directory.payload_digest());
    mb::SumDirModelFingerprint current = fingerprint(kWeights);
    qk::SumDirQueryKey key_used = key;
    std::size_t plan_named = named;

    switch (kind) {
    case MutantKind::LengthShort:
        read_back.ids.pop_back(); // (D2): the length stops being the ROW's
        break;
    case MutantKind::IdentityFlipped:
        read_back.ids[137U % read_back.ids.size()] ^= 0x5A5A5A5AU; // (D3): one id moves
        break;
    case MutantKind::PlanSubstituted:
        plan_named = (named + 1U) % directory.rows().size(); // (D4): the plan names another row
        break;
    case MutantKind::KeyMissing:
        key_used = qk::sum_dir_query_key_from_ids(block_ids(99U)); // (D1): a key with no row
        break;
    case MutantKind::WeightsMoved:
        current = fingerprint(0x7765696768747332ULL); // "weights2"
        break;
    case MutantKind::TokenizerMoved:
        current = fingerprint(kWeights, 0x746f6b656e697a32ULL); // "tokeniz2"
        break;
    case MutantKind::OtherDirectoryBinding: {
        sd::SumDir other(0x9999ULL, kBlockTokens);
        const std::vector<std::uint32_t> other_ids = block_ids(7U);
        (void)other.append(other_ids.data(), other_ids.size(), 0U, 1U, sd::SumDirCodec::Int8, 0, 0U);
        other.sort_rows();
        binding = mb::sum_dir_model_binding_make(fingerprint(kWeights), other.payload_digest());
        break;
    }
    case MutantKind::UnsetBinding:
        // The components are unset but the DIRECTORY link is intact, so the refusal that fires is
        // the fingerprint one and not the directory one. Otherwise this arm would silently be a
        // second copy of `neg-other-directory`.
        binding =
            mb::sum_dir_model_binding_make(mb::SumDirModelFingerprint{}, directory.payload_digest());
        break;
    }

    det::SumDirDeterminedReport report;
    const det::SumDirDeterminedVerdict verdict = det::sum_dir_determined(
        directory, key_used, plan_named, read_back, binding, current, clauses, &report);

    result.detail = verdict_line(verdict, report) +
                    " expected=" + det::sum_dir_determined_verdict_name(expected);
    check(verdict == expected,
          std::string("MUST-REFUSE: ") + what + " -- expected " +
              det::sum_dir_determined_verdict_name(expected) + ", got " +
              det::sum_dir_determined_verdict_name(verdict));
    check(verdict != det::SumDirDeterminedVerdict::Determined,
          std::string("MUST-REFUSE: ") + what + " -- the conjunction must NOT accept it");
    result.ok = g_failures == 0;
    return result;
}

[[nodiscard]] ArmResult arm_neg_len(const det::SumDirDeterminedClauses& c) {
    return run_mutant(c, MutantKind::LengthShort,
                      det::SumDirDeterminedVerdict::RefusedLengthMismatch,
                      "one id too few in the read-back");
}
[[nodiscard]] ArmResult arm_neg_ident(const det::SumDirDeterminedClauses& c) {
    return run_mutant(c, MutantKind::IdentityFlipped,
                      det::SumDirDeterminedVerdict::RefusedIdentityMismatch,
                      "one id XOR 0x5A5A5A5A in the read-back");
}
[[nodiscard]] ArmResult arm_neg_substitute(const det::SumDirDeterminedClauses& c) {
    return run_mutant(c, MutantKind::PlanSubstituted,
                      det::SumDirDeterminedVerdict::RefusedSubstituted,
                      "the plan names a DIFFERENT row");
}
[[nodiscard]] ArmResult arm_neg_key_missing(const det::SumDirDeterminedClauses& c) {
    return run_mutant(c, MutantKind::KeyMissing,
                      det::SumDirDeterminedVerdict::RefusedNoCandidate,
                      "the key is another block's");
}
[[nodiscard]] ArmResult arm_neg_weights(const det::SumDirDeterminedClauses& c) {
    return run_mutant(c, MutantKind::WeightsMoved,
                      det::SumDirDeterminedVerdict::RefusedModelChanged,
                      "the weight set moved");
}
[[nodiscard]] ArmResult arm_neg_tokenizer(const det::SumDirDeterminedClauses& c) {
    return run_mutant(c, MutantKind::TokenizerMoved,
                      det::SumDirDeterminedVerdict::RefusedModelChanged,
                      "the tokenizer moved");
}
[[nodiscard]] ArmResult arm_neg_otherdir(const det::SumDirDeterminedClauses& c) {
    return run_mutant(c, MutantKind::OtherDirectoryBinding,
                      det::SumDirDeterminedVerdict::RefusedModelChanged,
                      "the stored binding belongs to ANOTHER directory");
}
[[nodiscard]] ArmResult arm_neg_unset(const det::SumDirDeterminedClauses& c) {
    return run_mutant(c, MutantKind::UnsetBinding,
                      det::SumDirDeterminedVerdict::RefusedModelChanged,
                      "the stored binding carries no model");
}

// The specific-name arm: the conjunction's verdict is `refused-model-changed`, but the ARTEFACT's own
// verdict must be `refused-directory-mismatch` -- a caller reading the report has to be able to say
// WHICH of the three components disagreed.
[[nodiscard]] ArmResult arm_neg_otherdir_named(const det::SumDirDeterminedClauses& clauses) {
    ArmResult result;
    const sd::SumDir directory = build_directory();
    const std::vector<std::uint32_t> ids = block_ids(1U);
    const qk::SumDirQueryKey key = qk::sum_dir_query_key_from_ids(ids);
    const qk::SumDirQueryKeyLookup lookup = qk::sum_dir_query_key_lookup(directory, key);
    det::SumDirDeterminedReadBack read_back;
    read_back.ids_present = true;
    read_back.row = lookup.first_row;
    read_back.ids = ids;
    sd::SumDir other(0x9999ULL, kBlockTokens);
    const std::vector<std::uint32_t> other_ids = block_ids(7U);
    (void)other.append(other_ids.data(), other_ids.size(), 0U, 1U, sd::SumDirCodec::Int8, 0, 0U);
    other.sort_rows();
    const mb::SumDirModelBindingHeader binding =
        mb::sum_dir_model_binding_make(fingerprint(kWeights), other.payload_digest());
    det::SumDirDeterminedReport report;
    const det::SumDirDeterminedVerdict verdict = det::sum_dir_determined(
        directory, key, lookup.first_row, read_back, binding, fingerprint(kWeights), clauses,
        &report);
    result.detail = verdict_line(verdict, report);
    check(report.model_verdict == mb::SumDirModelBindingVerdict::RefusedDirectoryMismatch,
          "neg-other-directory-named: the ARTEFACT's verdict is refused-directory-mismatch, BY NAME");
    check(verdict == det::SumDirDeterminedVerdict::RefusedModelChanged,
          "neg-other-directory-named: and the conjunction turns it into refused-model-changed");
    result.ok = g_failures == 0;
    return result;
}

// The forbidden shape ITSELF, run as an arm: a positional first-fit over an ambiguous key. Under the
// full clause set this is refused by (D1); with POSITIONAL installed it is ADMITTED, which is the
// mutant that makes the `--drop POSITIONAL` run go red.
[[nodiscard]] ArmResult arm_ctl_positional(const det::SumDirDeterminedClauses& clauses) {
    ArmResult result;
    const sd::SumDir directory = build_directory_with_duplicate();
    const std::vector<std::uint32_t> ids = block_ids(1U);
    const qk::SumDirQueryKey key = qk::sum_dir_query_key_from_ids(ids);
    const qk::SumDirQueryKeyLookup lookup = qk::sum_dir_query_key_lookup(directory, key);
    // The row chosen BY POSITION -- the first of the duplicate run -- and its own content read back,
    // so the read-back is internally consistent and only the POSITION decided it.
    const std::size_t by_position = lookup.first_row;
    det::SumDirDeterminedReadBack read_back;
    read_back.ids_present = true;
    read_back.row = by_position;
    read_back.ids = ids;
    const mb::SumDirModelBindingHeader binding =
        mb::sum_dir_model_binding_make(fingerprint(kWeights), directory.payload_digest());
    det::SumDirDeterminedReport report;
    const det::SumDirDeterminedVerdict verdict = det::sum_dir_determined(
        directory, key, by_position, read_back, binding, fingerprint(kWeights), clauses, &report);
    result.detail = verdict_line(verdict, report) + " chosen_by=position";
    check(verdict == det::SumDirDeterminedVerdict::RefusedAmbiguous,
          "ctl-positional: the forbidden positional first-fit is refused by (D1) -- the ambiguity is "
          "COUNTED and never resolved by position");
    result.ok = g_failures == 0;
    return result;
}

// ---------------------------------------------------------------------------
// the arm table
// ---------------------------------------------------------------------------

struct Arm {
    const char* name;
    ArmBody body;
};

const Arm kArms[] = {
    {"green", arm_green},
    {"ctl-ambiguous", arm_ctl_ambiguous},
    {"ctl-nocand", arm_ctl_nocand},
    {"ctl-notsorted", arm_ctl_notsorted},
    {"ctl-unset", arm_ctl_unset},
    {"ctl-positional", arm_ctl_positional},
    {"binding-seal", arm_binding_seal},
    {"unification", arm_unification},
    {"neg-len", arm_neg_len},
    {"neg-ident", arm_neg_ident},
    {"neg-substitute", arm_neg_substitute},
    {"neg-key-missing", arm_neg_key_missing},
    {"neg-weights-moved", arm_neg_weights},
    {"neg-tokenizer-moved", arm_neg_tokenizer},
    {"neg-other-directory", arm_neg_otherdir},
    {"neg-other-directory-named", arm_neg_otherdir_named},
    {"neg-unset-foreign", arm_neg_unset},
};

constexpr std::size_t kArmCount = sizeof(kArms) / sizeof(kArms[0]);

// The six clause-breaks the run accepts. An unknown name is NOT MEASURED, so a typo can never turn
// a red run green.
const char* const kDrops[] = {"none", "D1", "D2", "D3", "D4", "MODEL", "POSITIONAL"};
constexpr std::size_t kDropCount = sizeof(kDrops) / sizeof(kDrops[0]);

[[nodiscard]] bool drop_is_known(const std::string& name) {
    for (std::size_t i = 0; i < kDropCount; ++i) {
        if (name == kDrops[i]) { return true; }
    }
    return false;
}

void print_arm_table() {
    std::printf("arm                            what it claims\n");
    std::printf("%-30s %s\n", "green", "the conjunction HOLDS on an untouched store");
    std::printf("%-30s %s\n", "ctl-ambiguous", "(D1) counts 2 candidates and refuses");
    std::printf("%-30s %s\n", "ctl-nocand", "(D1) counts 0 and names the no-candidate");
    std::printf("%-30s %s\n", "ctl-notsorted", "(D1) refuses where find() would throw");
    std::printf("%-30s %s\n", "ctl-unset", "an empty / over-long span is not a key");
    std::printf("%-30s %s\n", "ctl-positional", "the forbidden positional first-fit is refused");
    std::printf("%-30s %s\n", "binding-seal", "the fourth artefact's 48-byte wire + its seal");
    std::printf("%-30s %s\n", "unification", "the engine identity == the directory digest");
    std::printf("%-30s %s\n", "neg-len", "(D2) refused-length-mismatch");
    std::printf("%-30s %s\n", "neg-ident", "(D3) refused-identity-mismatch");
    std::printf("%-30s %s\n", "neg-substitute", "(D4) refused-substituted");
    std::printf("%-30s %s\n", "neg-key-missing", "(D1) refused-no-candidate");
    std::printf("%-30s %s\n", "neg-weights-moved", "the fourth artefact: weight set moved");
    std::printf("%-30s %s\n", "neg-tokenizer-moved", "the fourth artefact: tokenizer moved");
    std::printf("%-30s %s\n", "neg-other-directory", "the fourth artefact: another directory");
    std::printf("%-30s %s\n", "neg-other-directory-named", "and the artefact names THAT reason");
    std::printf("%-30s %s\n", "neg-unset-foreign", "an unset fingerprint is not a wildcard");
    std::printf("\ndrops accepted by --drop: ");
    for (std::size_t i = 0; i < kDropCount; ++i) { std::printf("%s ", kDrops[i]); }
    std::printf("\n");
}

// A per-arm failure snapshot. `g_failures` is CUMULATIVE across the run, so an arm's own verdict
// must be read off the DIFFERENCE, not off the running total -- otherwise the first arm that fails
// paints every arm after it red and the aggregate stops saying WHICH arm failed.
struct ArmScope {
    int before = g_failures;
    [[nodiscard]] bool ok() const { return g_failures == before; }
};

[[nodiscard]] int run_one(const Arm& arm, const det::SumDirDeterminedClauses& clauses) {
    ArmScope scope;
    const ArmResult result = arm.body(clauses);
    const bool arm_ok = scope.ok() && result.ok;
    std::printf("ARM %-30s %s  %s\n", arm.name, arm_ok ? "GREEN" : "RED",
                result.detail.c_str());
    return arm_ok ? 0 : 1;
}

} // namespace

int main(int argc, char** argv) {
    std::string arm = "all";
    std::string drop = "none";
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--arm") == 0 && i + 1 < argc) {
            arm = argv[i + 1];
            ++i;
        } else if (std::strcmp(argv[i], "--drop") == 0 && i + 1 < argc) {
            drop = argv[i + 1];
            ++i;
        } else if (std::strcmp(argv[i], "--list") == 0) {
            arm = "list";
        }
    }

    if (arm == "list") {
        print_arm_table();
        return 0;
    }
    if (!drop_is_known(drop)) {
        std::fprintf(stderr, "NOT MEASURED: no clause named '%s' (--list prints the list)\n",
                     drop.c_str());
        return 2;
    }

    // The clause set is derived from `--drop`, so a broken-clause run is the SAME code with ONE
    // clause removed -- not a second, separately-written expectation.
    det::SumDirDeterminedClauses clauses = det::sum_dir_determined_clauses_all();
    if (drop != "none") {
        clauses = det::sum_dir_determined_clauses_with_dropped(drop);
    }
    // Self-check the selector before it is used as evidence: the right clause must be the only one
    // that moved, and an unknown name must move nothing.
    if (drop == "D1") {
        check(!clauses.d1_uniqueness && clauses.d2_length_from_row &&
                  clauses.d3_rederived_identity && clauses.d4_no_substitution &&
                  clauses.model_binding && !clauses.allow_positional_substitution,
              "selector: --drop D1 drops exactly D1");
    }
    if (drop == "D4") {
        check(clauses.d1_uniqueness && clauses.d2_length_from_row &&
                  clauses.d3_rederived_identity && !clauses.d4_no_substitution &&
                  clauses.model_binding && !clauses.allow_positional_substitution,
              "selector: --drop D4 drops exactly D4");
    }
    if (drop == "POSITIONAL") {
        check(clauses.d1_uniqueness && clauses.d2_length_from_row &&
                  clauses.d3_rederived_identity && clauses.d4_no_substitution &&
                  clauses.model_binding && clauses.allow_positional_substitution,
              "selector: --drop POSITIONAL installs the substitution and drops no clause");
    }
    const det::SumDirDeterminedClauses typo = det::sum_dir_determined_clauses_with_dropped("D9");
    check(typo.d1_uniqueness && typo.d2_length_from_row && typo.d3_rederived_identity &&
              typo.d4_no_substitution && typo.model_binding,
          "selector: an unknown clause name drops nothing");

    std::printf("sum_dir_determined_test: arm=%s drop=%s arms=%zu\n", arm.c_str(), drop.c_str(),
                kArmCount);

    int bad_arms = 0;
    if (arm == "all") {
        for (std::size_t i = 0; i < kArmCount; ++i) {
            bad_arms += run_one(kArms[i], clauses);
        }
    } else {
        const Arm* found = nullptr;
        for (std::size_t i = 0; i < kArmCount; ++i) {
            if (arm == kArms[i].name) { found = &kArms[i]; }
        }
        if (found == nullptr) {
            std::fprintf(stderr, "NOT MEASURED: no arm named '%s' (--list prints the table)\n",
                         arm.c_str());
            return 2;
        }
        bad_arms += run_one(*found, clauses);
    }

    std::printf("CHECKS=%d FAILURES=%d DROP=%s\n", g_checks, g_failures, drop.c_str());
    if (g_failures != 0 || bad_arms != 0) {
        std::printf("VERDICT=RED arm=%s drop=%s\n", arm.c_str(), drop.c_str());
        return 1;
    }
    std::printf("VERDICT=GREEN arm=%s drop=%s\n", arm.c_str(), drop.c_str());
    return 0;
}
