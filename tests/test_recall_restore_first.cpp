// ===========================================================================
// tests/test_recall_restore_first.cpp
// ===========================================================================
// THE RESTORE-FIRST PREDICATE, ITS EDGE, AND ITS DEFAULT.
//
// WHY THIS FILE EXISTS. `recall_cold_pages_for_round` (program_impl.h) has had TWO restore
// legs since the text-cargo landing -- the BYTE leg (`restore_cold_page`, decode the packed
// record back into the K/V plane IN PLACE) and the TEXT leg (re-prefill the cargo's 256 B
// block) -- and it has had ONE bit of policy between them: the PRESENCE of the
// `NINFER_RECALL_TEXT` env var.  That is a PROCESS bit, not a PLAN bit, so arming the text
// route replaces the byte leg for every plan, including the plans whose bytes are still
// materialisable and for which the byte leg is measured at 60x the cheaper side.
//
// `recall_text_route_wins` (src/spec/turn_recall_journal.h) makes the choice PER PLAN and
// additive: the byte leg is taken when every planned page still has its bytes, and the text
// route becomes the FALLBACK for the plans whose bytes are gone.  With
// `NINFER_RECALL_RESTORE_FIRST` absent the predicate reproduces today's single bit exactly.
//
// WHAT IS EXECUTED HERE (card-free, host-only, std-only, no GPU, no engine build):
//   * the POSITIVE arm    -- mode `bytes`, route armed, every planned page has bytes =>
//                            the TEXT route must NOT run (the byte leg wins);
//   * the FALLBACK arm    -- mode `bytes`, route armed, at least one page's bytes are gone =>
//                            the TEXT route MUST run, and the whole plan goes to it;
//   * the CONTROL arm     -- mode `text` (the default) reproduces today's `armed` bit for
//                            EVERY (planned_pages, pages_with_bytes) pair;
//   * the UNARMED arm     -- no text route at all => this predicate is FALSE for every mode;
//   * the PARSE arm       -- absent/nullopt/`""`/unrecognised all give the DEFAULT, and the
//                            default is NOT entered by a word that was never spelled;
//   * a NEGATIVE CONTROL  -- the predicate is asserted to be NON-CONSTANT in
//                            (planned_pages, pages_with_bytes) under mode `bytes`, so no check
//                            above can pass vacuously on a predicate that ignores its arguments.
//
// IT MUST NOT, and does not: assert anything about the ENGINE, the GPU, the answer quality, or
// which leg a real run took.  Those are measured in `dl/recallcheap/` by the engine arms.  This
// file pins the DECISION TABLE and nothing else.
//
// The MUTANT / SPECIFIC arms are NOT in this file and NOT in the build tree: they compile THIS
// SAME source against a mutated header image with `-I` precedence, the shape
// `dl/fusionrefresh/control/` and `dl/prefillland/s/probe.sh` already use.
#include <cstdint>
#include <cstdio>
#include <string>

#include "spec/turn_recall_journal.h"

namespace tr = ninfer::spec::turn_recall;

static int g_checks   = 0;
static int g_failures = 0;

static void ck(bool ok, const char* what) {
    ++g_checks;
    if (ok) {
        std::printf("  ok   %s\n", what);
    } else {
        ++g_failures;
        std::printf("  FAIL %s\n", what);
    }
}

int main() {
    using tr::RecallRestoreFirst;
    const RecallRestoreFirst TEXT  = RecallRestoreFirst::Text;
    const RecallRestoreFirst BYTES = RecallRestoreFirst::Bytes;

    std::printf("=== test_recall_restore_first ===\n");

    // ---- the name/parse vocabulary -------------------------------------------
    std::printf("[PARSE]\n");
    ck(tr::recall_restore_first_parse(nullptr) == TEXT,
       "PARSE: absent env is the DEFAULT, not a third mode");
    ck(tr::recall_restore_first_parse("") == TEXT,
       "PARSE: an empty value is the DEFAULT");
    ck(tr::recall_restore_first_parse("bytes") == BYTES,
       "PARSE: `bytes` selects the byte leg first");
    ck(tr::recall_restore_first_parse("text") == TEXT,
       "PARSE: `text` selects today's behaviour, by name");
    ck(tr::recall_restore_first_parse("0") == TEXT,
       "PARSE: `0` is NOT a mode word -- it falls back to the DEFAULT");
    ck(tr::recall_restore_first_parse("1") == TEXT,
       "PARSE: `1` is NOT a mode word -- it falls back to the DEFAULT");
    ck(tr::recall_restore_first_parse("Bytes") == TEXT,
       "PARSE: no case folding -- an unspelled value is the DEFAULT");
    ck(std::string(tr::recall_restore_first_name(TEXT)) == "text",
       "NAME: Text is spelled `text`");
    ck(std::string(tr::recall_restore_first_name(BYTES)) == "bytes",
       "NAME: Bytes is spelled `bytes`");
    ck(tr::kRecallRestoreFirstDefault == TEXT,
       "DEFAULT: the shipped default IS today's behaviour");
    ck(std::string(tr::kRecallRestoreFirstEnv) == "NINFER_RECALL_RESTORE_FIRST",
       "ENV: the name is the one the site reads");

    // ---- POSITIVE: bytes present, byte leg wins ------------------------------
    std::printf("[POSITIVE] mode=bytes, armed, every planned page has bytes\n");
    ck(tr::recall_text_route_wins(BYTES, true, 1, 1) == false,
       "POSITIVE: 1 planned page, 1 with bytes -> the TEXT route must NOT run");
    ck(tr::recall_text_route_wins(BYTES, true, 8, 8) == false,
       "POSITIVE: 8 planned pages, 8 with bytes -> the TEXT route must NOT run");
    ck(tr::recall_text_route_wins(BYTES, true, 227, 227) == false,
       "POSITIVE: the full 227-page pass, all with bytes -> the TEXT route must NOT run");

    // ---- FALLBACK: a byte-absent page sends the WHOLE plan to the text route --
    std::printf("[FALLBACK] mode=bytes, armed, at least one page's bytes are gone\n");
    ck(tr::recall_text_route_wins(BYTES, true, 1, 0) == true,
       "FALLBACK: 1 planned page, 0 with bytes -> the TEXT route MUST run");
    ck(tr::recall_text_route_wins(BYTES, true, 8, 7) == true,
       "FALLBACK: 7 of 8 with bytes -> the WHOLE plan goes to the TEXT route");
    ck(tr::recall_text_route_wins(BYTES, true, 8, 1) == true,
       "FALLBACK: 1 of 8 with bytes -> the WHOLE plan goes to the TEXT route");
    ck(tr::recall_text_route_wins(BYTES, true, 8, 0) == true,
       "FALLBACK: none with bytes -> the TEXT route MUST run");
    ck(tr::recall_text_route_wins(BYTES, true, 0, 0) == true,
       "FALLBACK: an empty plan is NOT a licence to skip a real route");

    // ---- CONTROL: mode=text reproduces today's single bit, everywhere --------
    std::printf("[CONTROL] mode=text (the default) == today's `armed` bit\n");
    {
        const std::size_t plans[4] = {0, 1, 8, 227};
        const std::size_t have[4]  = {0, 1, 7, 227};
        bool all_text = true;
        bool all_zero = true;
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                if (tr::recall_text_route_wins(TEXT, true, plans[i], have[j]) != true) { all_text = false; }
                if (tr::recall_text_route_wins(TEXT, false, plans[i], have[j]) != false) { all_zero = false; }
            }
        }
        ck(all_text, "CONTROL: armed + mode=text is the TEXT route for all 16 (planned, have) pairs");
        ck(all_zero, "UNARMED: no text route + mode=text is the BYTE route for all 16 pairs");
    }

    // ---- UNARMED: mode=bytes but nothing to fall back TO --------------------
    std::printf("[UNARMED] mode=bytes, route NOT armed\n");
    ck(tr::recall_text_route_wins(BYTES, false, 1, 1) == false,
       "UNARMED: bytes present -> byte leg, as it already was");
    ck(tr::recall_text_route_wins(BYTES, false, 1, 0) == false,
       "UNARMED: bytes absent and no text route -> still FALSE (the byte leg is the only leg)");

    // ---- NEGATIVE CONTROL: the predicate cannot be an argument-ignorer -------
    std::printf("[NEGATIVE CONTROL] the predicate must be NON-CONSTANT in its counts\n");
    {
        const bool a = tr::recall_text_route_wins(BYTES, true, 8, 8);
        const bool b = tr::recall_text_route_wins(BYTES, true, 8, 7);
        ck(a != b, "NEG-CONTROL: (8,8) and (8,7) MUST differ, else the positives assert nothing");
        const bool c = tr::recall_text_route_wins(BYTES, true, 8, 8);
        const bool d = tr::recall_text_route_wins(TEXT,  true, 8, 8);
        ck(c != d, "NEG-CONTROL: (mode=bytes) and (mode=text) MUST differ at (8,8)");
    }

    std::printf("CHECKS=%d FAILURES=%d\n", g_checks, g_failures);
    std::printf("VERDICT=%s\n", g_failures == 0 ? "PASS" : "FAIL_WRONG");
    return g_failures == 0 ? 0 : 3;
}
