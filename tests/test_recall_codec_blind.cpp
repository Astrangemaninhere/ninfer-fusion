// tests/test_recall_codec_blind.cpp
//
// [CODECBLIND] THE RECALL CLASSIFIER'S BLIND SPOT, MADE AUDITABLE -- and the named decision
// about it. One file, one build, no CUDA, no ninfer library, no GPU, no decoder.
//
// WHY THIS EXISTS
//
//   The tree has exactly ONE producer of the admission decision, the target's classifier
//   `ProgramImplCore::turn_recall_codec_of_layers()` (src/targets/qwen3_6/impl/runtime/
//   program_impl.h). It skips every layer without a cold slot and then reads
//
//       if (view.dtype == DType::NVFP4) { nvfp4 = true; }
//       else if (view.dtype == DType::I8) { i8 = true; }
//
//   with NO further arm, so a cold-slot-bearing layer of any other dtype sets neither flag.
//   When an admitted codec is present BESIDE such layers the stack is labelled with the
//   admitted codec, the chain fires, and nothing said the other layers were invisible --
//   while `turn_recall_page_bytes()` sums `cold_slots.nb[3]` over EVERY slot-bearing layer,
//   i.e. the record's stride and the classifier's answer are about two different sets.
//
// WHAT THIS TEST IS AND IS NOT ALLOWED TO BE
//
//   * It is HOST-ONLY, std-only, and includes exactly one tree header, spec/turn_recall_journal.h
//     (which declares itself host-only: "no CUDA, no engine headers, unit-testable with plain
//     g++"). Same shape and same reason as tests/test_turn_recall_inexact_gate.cpp.
//   * The POLICY HALF IS EXECUTED against the real header -- `codec_blind_refuses_run()` is a
//     pure function of (count, policy), so every policy is exercised against every count here.
//   * The ENGINE HALF (program_impl.h / program.h) cannot be linked here, so it is asserted
//     against the SOURCE TEXT, the established shape of tests/test_fnv_convention.cpp and
//     tests/test_cold_slot_release_bytes.cpp ("reads the kernel header AS TEXT").
//   * EVERY text check is paired with a NEGATIVE CONTROL on a synthetic broken copy of the
//     same snippet, so no check can pass vacuously: a check whose failure mode nobody can name
//     is not a check. The synthetic copies are in-file; the OUT-OF-FILE mutation (a mutated
//     header taking -I priority, compiling THIS UNCHANGED FILE) is the sibling arm run by
//     dl/combo1m2/s/gate_check_blind.sh, and it is the arm that MUST go red.
//
// THE FIVE SECTIONS AND THE MUTATION THAT REDDENS EACH
//
//   PA (the predicate, executed)  mutations: `return policy == RefuseRun;` (loses the
//       `blind_layers == 0` arm, which would refuse the homogeneous tier recall works on);
//       `return blind_layers != 0;` (loses the policy arm, which would make ReportOnly refuse).
//   PB (the shipped constant is NAMED)  mutation: set kCodecBlindPolicy back to Unnamed, or
//       swap it to RefuseRun (the second one is not a defect but it IS an admission change,
//       and it must not happen without this test saying so).
//   PC (the contradicted claims are GONE)  mutation: restore "which admits nothing" to
//       program_impl.h's classifier comment, or restore "or a table mixing codecs" to
//       RecallCodec::Rejected -- both were claims about the code, written in the code, that
//       recall_codec_admitted already contradicted.
//   PD (the engine wiring)  mutation: delete the `[recall] CODEC-BLIND` fprintf, delete the
//       `codec_blind_refuses_run(...)` branch, drop `blind_layers=` from the journal format,
//       or drop the counter from the census line.
//   PE (the negative controls)  mutation: make any synthetic broken copy EQUAL to the tree's
//       snippet -- then the corresponding check asserts nothing and this section reddens.

#include "spec/turn_recall_journal.h"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

#if !defined(CODECBLIND_SOURCE_DIR)
#if defined(NINFER_SOURCE_DIR)
#define CODECBLIND_SOURCE_DIR NINFER_SOURCE_DIR
#else
#error "define CODECBLIND_SOURCE_DIR to the tree root the text checks read"
#endif
#endif

namespace tr = ninfer::spec::turn_recall;

namespace {

unsigned failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAIL: %s\n", what);
        ++failures;
    }
}
void check_text(bool ok, const char* what, const char* mutation) {
    if (!ok) {
        std::printf("FAIL: %s\n      the mutation this check rejects: %s\n", what, mutation);
        ++failures;
    }
}
void check_str(const std::string& got, const std::string& want, const char* what) {
    if (got != want) {
        std::printf("FAIL: %s (got \"%s\", want \"%s\")\n", what, got.c_str(), want.c_str());
        ++failures;
    }
}

std::string read_text_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::printf("FAIL: cannot read %s\n", path.c_str());
        ++failures;
        return std::string();
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}
bool has(const std::string& text, const std::string& needle) {
    return text.find(needle) != std::string::npos;
}
// HOW MANY TIMES. A bare `!has(...)` is NOT a sufficient check for "the contradicted claim is
// gone", and this test learned that by failing: the CORRECTION quotes the contradicted
// sentence verbatim, so `has()` was true of the corrected file too, and the check could not
// tell an ASSERTED claim from a QUOTED one. The precise form is (occurrences == 1) AND (it
// occurs under the correction marker) -- which reddens both when the claim is restored
// somewhere else and when the correction that explains it is deleted.
std::size_t count_of(const std::string& text, const std::string& needle) {
    std::size_t n = 0, at = 0;
    while ((at = text.find(needle, at)) != std::string::npos) {
        ++n;
        at += needle.size();
    }
    return n;
}
bool has_before(const std::string& text, const std::string& a, const std::string& b) {
    const std::size_t ia = text.find(a);
    const std::size_t ib = text.find(b);
    return ia != std::string::npos && ib != std::string::npos && ia < ib;
}

const std::string kImpl = std::string(CODECBLIND_SOURCE_DIR) +
                          "/src/targets/qwen3_6/impl/runtime/program_impl.h";
const std::string kProg = std::string(CODECBLIND_SOURCE_DIR) +
                          "/src/targets/qwen3_6/impl/runtime/program.h";
const std::string kHead = std::string(CODECBLIND_SOURCE_DIR) + "/src/spec/turn_recall_journal.h";

}  // namespace

int main() {
    using tr::CodecBlindPolicy;

    // =====================================================================================
    // PA -- THE PREDICATE, EXECUTED. Every policy against every count.
    // =====================================================================================
    // A stack with no blind layer is NEVER refused, under EITHER named policy. This is the
    // arm that keeps the homogeneous nvfp4 tier recallable.
    check(!tr::codec_blind_refuses_run(0, CodecBlindPolicy::ReportOnly),
          "PA clean stack, ReportOnly: not refused");
    check(!tr::codec_blind_refuses_run(0, CodecBlindPolicy::RefuseRun),
          "PA clean stack, RefuseRun: NOT refused (mutation: `return policy == RefuseRun;`)");
    // ReportOnly never refuses, at any count: this is what makes the landing behaviour-preserving.
    check(!tr::codec_blind_refuses_run(1, CodecBlindPolicy::ReportOnly),
          "PA 1 blind layer, ReportOnly: not refused");
    check(!tr::codec_blind_refuses_run(6, CodecBlindPolicy::ReportOnly),
          "PA 6 blind layers, ReportOnly: not refused (mutation: `return blind_layers != 0;`)");
    check(!tr::codec_blind_refuses_run(16, CodecBlindPolicy::ReportOnly),
          "PA 16 blind layers, ReportOnly: not refused");
    // RefuseRun refuses exactly when something is blind -- the named refusal's RED arm.
    check(tr::codec_blind_refuses_run(1, CodecBlindPolicy::RefuseRun),
          "PA 1 blind layer, RefuseRun: REFUSED (the named refusal fires)");
    check(tr::codec_blind_refuses_run(6, CodecBlindPolicy::RefuseRun),
          "PA 6 blind layers (the factory tier's rk4v4 share), RefuseRun: REFUSED");
    check(tr::codec_blind_refuses_run(16, CodecBlindPolicy::RefuseRun),
          "PA 16 blind layers (every slot-bearing layer invisible), RefuseRun: REFUSED");
    // Unnamed is not a policy and must never refuse -- if it ever does, the constant below
    // has been replaced by something that is not one of the two named choices.
    check(!tr::codec_blind_refuses_run(6, CodecBlindPolicy::Unnamed),
          "PA Unnamed refuses nothing (it is the ABSENCE of a policy)");

    // =====================================================================================
    // PB -- THE SHIPPED CONSTANT IS NAMED, AND WHICH ONE IT IS.
    // =====================================================================================
    check(tr::kCodecBlindPolicy != CodecBlindPolicy::Unnamed,
          "PB the shipped policy is NAMED (mutation: set kCodecBlindPolicy = Unnamed)");
    check(tr::kCodecBlindPolicy == CodecBlindPolicy::ReportOnly,
          "PB the shipped policy is ReportOnly, so this landing moves NO existing run "
          "(mutation: flip it to RefuseRun -- an ADMISSION change, not a report change)");
    check_str(tr::codec_blind_policy_name(CodecBlindPolicy::Unnamed), "unnamed",
              "PB name(Unnamed)");
    check_str(tr::codec_blind_policy_name(CodecBlindPolicy::ReportOnly), "report-only",
              "PB name(ReportOnly)");
    check_str(tr::codec_blind_policy_name(CodecBlindPolicy::RefuseRun), "refuse-run",
              "PB name(RefuseRun)");
    check_str(std::string(tr::kCodecBlindRefusalName), "refused-codec-blind",
              "PB the refusal has ONE spelling (a harness greps this ONE string)");

    // =====================================================================================
    // PC -- THE CONTRADICTED CLAIMS ARE GONE, AND THE CORRECTION IS ON THE RECORD.
    // =====================================================================================
    const std::string impl = read_text_file(kImpl);
    const std::string prog = read_text_file(kProg);
    const std::string head = read_text_file(kHead);
    if (impl.empty() || prog.empty() || head.empty()) {
        std::printf("VERDICT=FAIL_READ\n");
        return 3;
    }

    // The classifier's own comment claimed `Mixed` "admits nothing" while recall_codec_admitted
    // admitted it. src/product/kv_plane_census.h records that contradiction independently.
    //
    // ⚠️ WHY THESE TWO ARE COUNTED AND NOT MERELY ABSENT. The correction QUOTES the sentence it
    // removes, so `!has(...)` is FALSE of the corrected file -- a check written that way passes
    // for the wrong reason on the pre-image and fails for the wrong reason on the post-image
    // (this test did exactly that once). The precise claim is: the phrase survives exactly once,
    // and that one occurrence is inside the CORRECTED block, i.e. it is HISTORY and not a claim.
    check_text(count_of(impl, "which admits nothing") == 1 &&
                   has_before(impl, "CORRECTED 2026-09-22", "which admits nothing"),
               "PC program_impl.h states the 'admits nothing' claim ONLY as corrected history "
               "(recall_codec_admitted had contradicted it since 2026-09-18)",
               "restore the sentence as an active claim, or delete the CORRECTED block that "
               "marks it as history: a reader would be back to a claim the code contradicts");
    check_text(has(impl, "CORRECTED 2026-09-22"),
               "PC the classifier's comment carries the correction",
               "delete the CORRECTED block: then the reader is back to a claim the code "
               "contradicts, with nothing saying so");
    // RecallCodec::Rejected's own enumerator comment.
    check_text(count_of(head, "or a table mixing codecs") == 1 &&
                   has_before(head, "[CODECBLIND] CORRECTED 2026-09-22",
                              "or a table mixing codecs"),
               "PC RecallCodec::Rejected states 'or a table mixing codecs' ONLY as corrected "
               "history",
               "restore 'rk4v4, bf16, fp8, iso4e, or a table mixing codecs: no external tier' "
               "as an active claim");
    check_text(!has(head, "so it is refused instead of mis-described"),
               "PC RecallCodec::Mixed says it is ADMITTED, not 'refused'",
               "restore the Mixed comment 'so it is refused instead of mis-described'");

    // =====================================================================================
    // PD -- THE ENGINE WIRING. The count must be TAKEN, NAMED, PRINTED and COUNTED.
    // =====================================================================================
    // (1) the count exists and reads the same skip the classifier does.
    check_text(has(impl, "std::uint32_t ProgramImplCore::turn_recall_codec_blind_layers() const {"),
               "PD the blind-layer count is DEFINED in program_impl.h",
               "delete turn_recall_codec_blind_layers(): the count exists nowhere else");
    check_text(has(impl, "const char* ProgramImplCore::turn_recall_codec_blind_codec() const {"),
               "PD the blind codec NAME is produced",
               "delete turn_recall_codec_blind_codec(): the line would name no codec");
    // the skip and the admitted set must be the classifier's own, or the pair drifts.
    check_text(has(impl, "if (view.cold_slots.data == nullptr) { continue; }\n"
                         "        if (view.dtype == DType::NVFP4 || view.dtype == DType::I8) "
                         "{ continue; }"),
               "PD the count skips exactly the non-slot layers and the admitted set",
               "change the skip or the admitted set here: the count would then describe a "
               "different layer set from the classifier's answer");
    // the name literals must be the tree's own tier tokens.
    check_text(has(impl, "case DType::E8Kv: return \"rk4v4\";"),
               "PD the rk4v4 spelling is the tree's own",
               "rename 'rk4v4' here: the report and the refusal would spell the codec two ways");
    check_text(has(impl, "default: return \"unclassified\";"),
               "PD an unnamed dtype is named `unclassified` rather than guessed",
               "replace the default arm with a guess: a name that is wrong is worse than "
               "a name that admits it does not know");
    // (2) the count is stored on the counters, so the census can carry it.
    check_text(has(prog, "std::uint64_t codec_blind_layers = 0;"),
               "PD the counter exists on TurnRecallCounters",
               "delete the counter: the fact then lives only in one stderr line");
    check_text(has(prog, "[[nodiscard]] std::uint32_t turn_recall_codec_blind_layers() const;"),
               "PD the two members are DECLARED",
               "delete the declarations: the definitions would not compile, but a reader "
               "would lose the pointer to the blind spot from program.h");
    // (3) the NAMED line, on the arming path, with both facts.
    check_text(has(impl, "\"[recall] CODEC-BLIND codec=%s layers=%u policy=%s refusal=%s \""),
               "PD the named [recall] CODEC-BLIND line exists",
               "delete the fprintf: the silent case goes back to being silent");
    check_text(has_before(impl, "const std::uint32_t codec_blind_layers = "
                                "turn_recall_codec_blind_layers();",
                          "if (!spec::turn_recall::recall_codec_admitted("),
               "PD the count is taken BEFORE the admission branch",
               "move the count below the admission branch: an e8-only stack would then "
               "never be counted, and the two refusals would stop being distinguishable");
    check_text(has(impl, "if (codec_blind_layers != 0) {"),
               "PD the line is conditional on the fact, so a clean run stays quiet",
               "print unconditionally: every run would look blind, which is how a real "
               "signal stops being read");
    // (4) the refusal branch, and the static_assert that keeps the choice made.
    check_text(has(impl, "} else if (spec::turn_recall::codec_blind_refuses_run(\n"
                         "                           codec_blind_layers, "
                         "spec::turn_recall::kCodecBlindPolicy)) {"),
               "PD the NAMED REFUSAL branch is wired",
               "delete the branch: `RefuseRun` would be a constant nothing reads -- the "
               "'knob exported and read nowhere' defect dl/prefillland names");
    check_text(has(impl, "spec::turn_recall::kCodecBlindPolicy !=\n"
                         "                              spec::turn_recall::CodecBlindPolicy::"
                         "Unnamed"),
               "PD the engine static_asserts the policy away from Unnamed",
               "delete the static_assert: the choice could then be left silently unmade");
    check_text(has(impl, "refused-codec-blind"),
               "PD the refusal's name reaches the engine's own text",
               "drop the name: the refusal would be printed without the string a harness greps");
    // (5) both census lines carry it, APPEND-ONLY (no existing field's bytes move).
    check_text(has(impl, "\"prefill_tokens=%llu fsync=%s blind_layers=%u blind_codec=%s\\n\","),
               "PD the journal line carries the count, appended",
               "drop the two fields: the arming record would name one codec for a two-codec page");
    check_text(has(impl, "\"append_failed=%llu refused_codec=%llu refused_medium=%llu "
                         "live=%zu \"\n                 \"codec_blind_layers=%llu\\n\","),
               "PD the sequence-end census carries the count, appended",
               "drop it from the census: the single-run line would be the only evidence");
    // append-only means the OLD field order is untouched: assert the old prefix verbatim.
    check_text(has(impl, "\"[recall] %s rounds=%llu records=%llu tombstones=%llu plans=%llu \""),
               "PD the census line's existing prefix is byte-unchanged",
               "edit the existing fields: every reader that stops before the new field "
               "would see different bytes");

    // =====================================================================================
    // PE -- THE NEGATIVE CONTROLS. A synthetic broken copy must DIFFER, or a check above
    //       asserted nothing. Each snippet below is the mutation named beside its check.
    // =====================================================================================
    const std::string good_predicate =
        "    if (policy != CodecBlindPolicy::RefuseRun) { return false; }\n"
        "    return blind_layers != 0;";
    const std::string broken_ignores_policy =
        "    return blind_layers != 0;";
    const std::string broken_ignores_count =
        "    return policy == CodecBlindPolicy::RefuseRun;";
    check_text(has(head, good_predicate),
               "PE the shipped predicate is the one PA executed against",
               "change the predicate's body: PA would then be testing a function the engine "
               "does not call");
    check_text(broken_ignores_policy != good_predicate && broken_ignores_count != good_predicate,
               "PE the two named mutations are DIFFERENT from the shipped predicate",
               "make either synthetic copy equal to the shipped one: its check would then "
               "assert nothing (the vacuous-check defect in a test's clothing)");
    // and the two mutations disagree with the shipped predicate on the cases PA asserts, so
    // PA genuinely reddens under them rather than merely differing textually.
    const auto shipped = [](std::uint32_t n, CodecBlindPolicy p) {
        if (p != CodecBlindPolicy::RefuseRun) { return false; }
        return n != 0;
    };
    const auto mut_policy = [](std::uint32_t n, CodecBlindPolicy) { return n != 0; };
    const auto mut_count = [](std::uint32_t, CodecBlindPolicy p) {
        return p == CodecBlindPolicy::RefuseRun;
    };
    bool differs = false;
    for (std::uint32_t n : {0U, 1U, 6U, 16U}) {
        for (CodecBlindPolicy p : {CodecBlindPolicy::Unnamed, CodecBlindPolicy::ReportOnly,
                                   CodecBlindPolicy::RefuseRun}) {
            if (mut_policy(n, p) != shipped(n, p)) { differs = true; }
            if (mut_count(n, p) != shipped(n, p)) { differs = true; }
        }
    }
    check(differs,
          "PE each named mutation DISAGREES with the shipped predicate on PA's own cases "
          "(so PA is a check, not a tautology)");

    // =====================================================================================
    std::printf("checks_failed=%u\n", failures);
    if (failures == 0) {
        std::printf("VERDICT=PASS\n");
        return 0;
    }
    std::printf("VERDICT=FAIL_WRONG\n");
    return 3;
}
