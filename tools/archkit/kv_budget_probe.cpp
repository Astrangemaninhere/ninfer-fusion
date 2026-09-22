// kv_budget_probe.cpp -- HOST-ONLY probe (no CUDA, no model, no GPU, no lock) that drives the
// ENGINE'S OWN KV bit-budget selectors, so a table can be regenerated against the code that
// actually runs instead of against a re-implementation.
//
// Why this file exists: docs/maintainer/kv-strategy-matrix.md section 3 (the request -> achieved
// -> spec ladder) was generated from tools/archkit/kv_bit_budget.py, and that file carries a
// header banner declaring that its ladder has FORKED from the authority
// (src/product/kv_bit_budget.h kKvBitBudgetTiers) and that its stdout must not be used for
// decisions. Regenerating the table from the forked tool would reproduce the defect.
//
// The two ladders differ (authority first):
//   fp8   850 vs 803      rk4v4   425 vs 406      iso4e 450 (pinned) vs 300
//   penalties int8 2 / fp8 3 / rk4v4 8 / nvfp4 30 / iso4e 200   vs   2 / 3 / 8 / 30 / 50
//   rk4v4 layer limit 8 (both, after the banner's claim of 10 was itself corrected)
// and the engine's DEFAULT path (cold_cap == 0) is not the multiset DP the Python mirrors at
// all: kv_bit_budget_solve() hands cold-free solves to the layer-exact "gearbox" solver
// (src/product/kv_bit_budget.h, search D6). Only the cold path (cold_cap > 0) stays on
// kv_bit_budget_solve_impl, the DP the Python tool is a twin of.
//
// Modes:
//   ladder   -- the authority's own tier rows, rk4v4 limit, cold grid point, pack orders
//   rows     -- machine-readable rows for both flag spellings, for a --bits list
//   report   -- the engine's verbatim [kv-bits] report text for one budget
//   score    -- the engine's [kv-score] provenance + penalty-table entry
//   oob      -- the score-table / tier-ladder size mismatch (compile this mode with ASan)
//
// Build (host only):
//   g++ -std=c++20 -O1 -I include -I src tools/archkit/kv_budget_probe.cpp -o /tmp/kbp
//   g++ -std=c++20 -O0 -g -fsanitize=address,undefined -I include -I src \
//       tools/archkit/kv_budget_probe.cpp -o /tmp/kbp_asan

#include "product/kv_kv_bits.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

using ninfer::product::KvBitsPlan;
using ninfer::product::KvTierScoreTable;
using ninfer::product::kKvBitBudgetTiers;

std::vector<double> parse_bits(int argc, char** argv, int first) {
    std::vector<double> out;
    for (int i = first; i < argc; ++i) {
        // Stop at the first option: otherwise `--layers 16` contributes 0.0 and 16.0 as
        // budgets and the run gains two phantom rows (which is how this parser was wrong).
        if (argv[i][0] == '-') { break; }
        out.push_back(std::strtod(argv[i], nullptr));
    }
    if (out.empty()) { out = {3.5, 4.0, 4.5, 5.0, 6.0, 8.0, 12.0, 16.0}; }
    return out;
}

int find_int(int argc, char** argv, const char* name, int fallback) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], name) == 0) { return std::atoi(argv[i + 1]); }
    }
    return fallback;
}

std::string find_str(int argc, char** argv, const char* name, const char* fallback) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], name) == 0) { return argv[i + 1]; }
    }
    return fallback;
}

const char* const kBitsMode[] = {"joint", "split", "ceiling"};
const char* const kColdPolicy[] = {"none", "window", "host", "disk", "host-then-disk"};

// Every fact a reader of the regenerated table needs in order to know WHAT produced a row.
void print_ladder(std::int32_t layers, std::int32_t rk4v4_limit, std::int32_t cold_cap) {
    std::printf("LADDER|tier|bits_x100|penalty_x100|selectable\n");
    for (const auto& t : kKvBitBudgetTiers) {
        std::printf("LADDER|%s|%d|%d|%d\n", t.spec_name, t.bits_x100, t.penalty_x100,
                    t.selectable ? 1 : 0);
    }
    std::printf("CONST|kKvBitBudgetE8LayerLimit|%d\n", ninfer::product::kKvBitBudgetE8LayerLimit);
    std::printf("CONST|cold_bits_x100|%d\n", ninfer::product::kKvBitBudgetColdBitsX100);
    std::printf("CONST|cold_penalty_x100|%d\n", ninfer::product::kKvBitBudgetColdPenaltyX100);
    std::printf("CONST|cold_slot_bytes|%d\n", ninfer::product::kKvBitBudgetColdSlotBytes);
    std::printf("CONST|tiers_count|%zu\n", kKvBitBudgetTiers.size());
    std::printf("CONST|score_table_rows|%zu\n", sizeof(KvTierScoreTable) / sizeof(ninfer::product::KvTierScoreRow));
    std::printf("CONST|layers_arg|%d\n", layers);
    std::printf("CONST|rk4v4_limit_arg|%d\n", rk4v4_limit);
    std::printf("CONST|cold_cap_arg|%d\n", cold_cap);
    std::printf("PACK|hot=");
    for (const char* s : ninfer::product::kKvBitBudgetPackOrder) { std::printf("%s,", s); }
    std::printf("\nPACK|cold=");
    for (const char* s : ninfer::product::kKvBitBudgetPackOrderCold) { std::printf("%s,", s); }
    std::printf("\n");
}

std::string csv_escape(std::string_view s) {
    std::string out;
    for (char c : s) { out += (c == '|' ? '/' : c); }
    return out;
}

// One row per (spelling, request). `spelling` names the CLI flag whose resolution path is
// exercised, because the two spellings are separate entry points in product/kv_kv_bits.h.
int rows_mode(const std::vector<double>& bits, std::int32_t layers, std::int32_t rk4v4_limit,
              std::int32_t cold_cap, std::string_view codec_name) {
    const KvTierScoreTable scores = ninfer::product::kv_bit_budget_default_scores();
    const ninfer::KvVCodec codec = (codec_name == "e2m1") ? ninfer::KvVCodec::E2M1
                                                         : ninfer::KvVCodec::Iso3;
    std::printf("# row= one (request, achieved, spec) triple as the ENGINE computes it\n");
    std::printf("# spelling=kv-bits -> kv_kv_bits_entry_joint (the --kv-bits / --kv-k-bits joint entry)\n");
    std::printf("# spelling=kv-bit-budget -> kv_bit_budget_spec (the --kv-bit-budget entry)\n");
    std::printf("# achieved unit = bits per KV element, K+V AVERAGED, i.e. ONE plane's width\n");
    std::printf("#   (the engine's own line says 'joint realized per plane: K=<a> V=<a>'); a\n");
    std::printf("#   reader must not double it to get 'bits per element pair'\n");
    for (const double request : bits) {
        // Spelling 1: --kv-bits -> KvBitsMode::Joint -> kv_kv_bits_entry_joint -> kv_bits_solve_plane
        {
            try {
                const KvBitsPlan plan = ninfer::product::kv_kv_bits_entry_joint(
                    layers, request, scores, /*quality_weight=*/-1.0, rk4v4_limit, cold_cap, codec);
                std::printf("ROW|spelling=kv-bits|request=%.2f|layers=%d|rk4v4_limit=%d|cold_cap=%d|"
                            "achieved=%.4f|penalty=%.4f|deployed=%d|spec=%s\n",
                            request, layers, rk4v4_limit, cold_cap, plan.achieved_bits, plan.penalty,
                            plan.deployed ? 1 : 0, csv_escape(plan.spec).c_str());
            } catch (const std::exception& e) {
                std::printf("ROW|spelling=kv-bits|request=%.2f|layers=%d|rk4v4_limit=%d|cold_cap=%d|"
                            "REFUSED|reason=%s\n",
                            request, layers, rk4v4_limit, cold_cap, csv_escape(e.what()).c_str());
            }
        }
        // Spelling 2: --kv-bit-budget -> kv_bit_budget_spec (the same solver by name, but a
        // DIFFERENT call site and a different report line; if the two ever disagree the table
        // would be measuring the flag, not the allocator).
        {
            try {
                const auto solved = ninfer::product::kv_bit_budget_solve(layers, request, rk4v4_limit,
                                                                        cold_cap);
                std::printf("ROW|spelling=kv-bit-budget|request=%.2f|layers=%d|rk4v4_limit=%d|"
                            "cold_cap=%d|achieved=%.4f|penalty=%.4f|deployed=1|spec=%s\n",
                            request, layers, rk4v4_limit, cold_cap, solved.achieved_bits,
                            solved.penalty, csv_escape(solved.spec).c_str());
            } catch (const std::exception& e) {
                std::printf("ROW|spelling=kv-bit-budget|request=%.2f|layers=%d|rk4v4_limit=%d|"
                            "cold_cap=%d|REFUSED|reason=%s\n",
                            request, layers, rk4v4_limit, cold_cap, csv_escape(e.what()).c_str());
            }
        }
    }
    return 0;
}

int report_mode(double request, std::int32_t layers, std::int32_t rk4v4_limit,
                std::int32_t cold_cap, std::string_view codec_name) {
    const KvTierScoreTable scores = ninfer::product::kv_bit_budget_default_scores();
    const ninfer::KvVCodec codec = (codec_name == "e2m1") ? ninfer::KvVCodec::E2M1
                                                         : ninfer::KvVCodec::Iso3;
    std::printf("--- the engine's own [kv-bits] report text, verbatim ---\n");
    try {
        const KvBitsPlan plan = ninfer::product::kv_kv_bits_entry_joint(
            layers, request, scores, -1.0, rk4v4_limit, cold_cap, codec);
        std::cout << plan.report;
    } catch (const std::exception& e) {
        std::printf("THROWN: %s\n", e.what());
        return 1;
    }
    std::printf("--- the [kv-bit-budget] line's fields, verbatim from kv_bit_budget_solve ---\n");
    try {
        const auto solved =
            ninfer::product::kv_bit_budget_solve(layers, request, rk4v4_limit, cold_cap);
        std::printf("[kv-bit-budget] full_attention_layers=%d bits=%.2f ranges=- cold_pages=%d "
                    "spec=%s\n",
                    layers, request, cold_cap, solved.spec.c_str());
    } catch (const std::exception& e) {
        std::printf("THROWN: %s\n", e.what());
        return 1;
    }
    return 0;
}

int score_mode() {
    const KvTierScoreTable scores = ninfer::product::kv_bit_budget_default_scores();
    const ninfer::product::KvScoresProvenance prov =
        ninfer::product::kv_scores_provenance(std::string_view{});
    std::printf("PROV|provisional=%d|quality_measured=%d|speed_measured=%d\n", prov.provisional,
                prov.quality_measured, prov.speed_measured);
    std::printf("PROVLINE|%s\n", csv_escape(prov.line).c_str());
    std::printf("SCORES|tier|quality_x100|speed_x100\n");
    for (std::size_t i = 0; i < sizeof(KvTierScoreTable) / sizeof(ninfer::product::KvTierScoreRow);
         ++i) {
        std::printf("SCORES|%s|%d|%d\n", kKvBitBudgetTiers[i].spec_name, scores[i].quality_x100,
                    scores[i].speed_x100);
    }
    std::printf("# The two loops below walk the tier ladder (8 rows) while indexing the score\n");
    std::printf("# table (6 rows). Run this mode under ASan to see the overrun.\n");
    // (1) the documented no-model entry point --kv-score-table show:
    std::string error;
    const bool ok = ninfer::product::kv_score_table_run("show", std::string_view{}, &error,
                                                       std::cout, std::cerr);
    std::printf("SCORETABLE|ok=%d|error=%s\n", ok ? 1 : 0, error.c_str());
    // (2) the ladder the scored DP fits against, with a weight that makes both columns live:
    const auto ladder = ninfer::product::kv_bit_budget_scored_ladder(scores, 0.5);
    std::printf("SCOREDLADDER|tier|penalty_x100\n");
    for (std::size_t i = 0; i < ladder.size(); ++i) {
        std::printf("SCOREDLADDER|%s|%d\n", ladder[i].spec_name, ladder[i].penalty_x100);
    }
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "ladder";
    const std::int32_t layers  = find_int(argc, argv, "--layers", 16);
    const std::int32_t rk4v4_limit = find_int(argc, argv, "--rk4v4-limit",
                                           ninfer::product::kKvBitBudgetE8LayerLimit);
    const std::int32_t cold_cap = find_int(argc, argv, "--cold-cap", 0);
    const std::string codec = find_str(argc, argv, "--v-codec", "iso4e");

    std::printf("# probe=kv_budget_probe.cpp mode=%s layers=%d rk4v4_limit=%d cold_cap=%d "
                "v_codec=%s\n",
                mode.c_str(), layers, rk4v4_limit, cold_cap, codec.c_str());

    if (mode == "ladder") {
        print_ladder(layers, rk4v4_limit, cold_cap);
        return 0;
    }
    if (mode == "rows") {
        return rows_mode(parse_bits(argc, argv, 2), layers, rk4v4_limit, cold_cap, codec);
    }
    if (mode == "report") {
        const double request = argc > 2 ? std::strtod(argv[2], nullptr) : 4.5;
        return report_mode(request, layers, rk4v4_limit, cold_cap, codec);
    }
    if (mode == "score") {
        return score_mode();
    }
    std::fprintf(stderr, "unknown mode '%s'\n", mode.c_str());
    return 2;
}
