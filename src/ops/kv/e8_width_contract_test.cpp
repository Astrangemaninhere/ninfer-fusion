// e8_width_contract_test -- the runnable bar for the e8 3-bit / 2-bit TIER wiring:
// vocabulary, ladder cost, ladder admissibility, and the refusal an operator actually
// meets. Host-only, no CUDA, no build/ :
//
//   g++ -O2 -std=c++20 -I/home/user/ninfer-fusion/src -I/home/user/ninfer-fusion/include \
//       /home/user/ninfer-fusion/src/ops/kv/e8_width_contract_test.cpp -o /tmp/e8wc
//   /tmp/e8wc
//
// The two things worth stating up front, because they are the ones a reader will doubt:
//   * the SHIPPED row is unchanged and the new rows do not move it (check 2);
//   * the new rows ARE choosable -- the ladder's own `selectable` field is true for them since
//     dl/e8mixwire opened the gate on 2026-09-19 -- so "cannot be chosen" is no longer the
//     thing that keeps a defined-but-unrunnable codec off the engine. What does is check 4b:
//     the COUPLING between "the engine declares the codec readable" and "the resolver resolves
//     it". Pinned from BOTH sides, so neither half can be satisfied by emptying the other;
//   * and the one that is a PIN rather than a totality check: check 7 pins the ladder's own
//     `selectable` field per rk4v4-family row as a literal. Check 4 can only say whether the
//     resolver refuses TODAY; check 7 is the row-level policy, so opening the gate cannot
//     happen without this file moving in the same landing.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/dtype.h"
#include "core/device_capabilities.h"
#include "kvcfg/kv_formats.h"
#include "product/kv_bit_budget.h"
#include "product/kv_e8_width.h"
#include "product/kv_kv_bits.h"
#include "product/kv_options.h"
#include "product/kv_storage_dtype.h"
#include "product/kv_tier_formats.h"

using namespace ninfer;
using namespace ninfer::product;

namespace {
int g_ok = 0, g_total = 0;
void check(bool pass, const char* what) {
    ++g_total;
    if (pass) { ++g_ok; }
    std::printf("%s  %s\n", pass ? "PASS" : "FAIL", what);
}
std::string catch_msg(void (*fn)()) {
    try { fn(); } catch (const std::exception& e) { return e.what(); }
    return std::string();
}
} // namespace

int main() {
    std::printf("=== e8 width contract test (host-only) ===\n\n");

    // --------------------------------------------------- 1. vocabulary + ordering
    std::printf("--- 1. vocabulary ---\n");
    check(kvcfg::bits_of(kvcfg::KvFormat::Rk4v4) == 4, "bits_of(rk4v4) == 4 (unchanged)");
    check(kvcfg::bits_of(kvcfg::KvFormat::Rk3v4) == 3, "bits_of(rk3v4) == 3");
    check(kvcfg::bits_of(kvcfg::KvFormat::Rk2v4) == 2, "bits_of(rk2v4) == 2");
    check(std::string(kvcfg::name_of(kvcfg::KvFormat::Rk3v4)) == "rk3v4", "name_of(rk3v4)");
    check(std::string(kvcfg::name_of(kvcfg::KvFormat::Rk2v4)) == "rk2v4", "name_of(rk2v4)");
    check(kvcfg::format_from_name("rk3v4").value_or(kvcfg::KvFormat::Auto) == kvcfg::KvFormat::Rk3v4,
          "format_from_name(\"rk3v4\") round-trips");
    check(!kvcfg::format_from_name("e8k4").has_value(), "e8k4 is not a format");

    // -------------------------------------------- 2. ladder: shipped row NOT moved
    std::printf("\n--- 2. ladder ---\n");
    check(kKvBitBudgetTiers.size() == 8, "the ladder is 8 rows");
    check(kKvBitBudgetTiers[kKvBitsE8Index].bits_x100 == 425, "row 4 is still rk4v4/4.25 (shipped)");
    check(kKvBitBudgetTiers[kKvBitsNvfp4Index].bits_x100 == 450, "row 3 is still nvfp4/4.50");
    check(std::string(kKvBitBudgetTiers[kKvBitsE8Index].spec_name) == "rk4v4", "row 4 is still named rk4v4");
    check(kKvBitBudgetTiers[kKvBitsE8K3Index].bits_x100 == 375, "row 6 is rk3v4/3.75");
    check(kKvBitBudgetTiers[kKvBitsE8K2Index].bits_x100 == 325, "row 7 is rk2v4/3.25");
    // The ladder cost must be the geometry's own number, not a parallel literal.
    check(kKvBitBudgetTiers[kKvBitsE8K3Index].bits_x100 == e8_kv_bits_x100(E8KvWidth::W3),
          "row 6 cost == product/kv_e8_width.h W3 cost");
    check(kKvBitBudgetTiers[kKvBitsE8K2Index].bits_x100 == e8_kv_bits_x100(E8KvWidth::W2),
          "row 7 cost == product/kv_e8_width.h W2 cost");
    check(kKvBitBudgetColdTierIndex == 8, "the cold pseudo-tier moved to 8 (it was at 6)");

    // ------------------------------- 3. all three e8 rows map to their own DType/class
    std::printf("\n--- 3. row -> DType -> class ---\n");
    check(kv_bits_tier_dtype(kKvBitsE8Index) == DType::E8Kv, "row 4 -> DType::E8Kv");
    check(kv_bits_tier_dtype(kKvBitsE8K3Index) == DType::E8K3Kv, "row 6 -> DType::E8K3Kv");
    check(kv_bits_tier_dtype(kKvBitsE8K2Index) == DType::E8K2Kv, "row 7 -> DType::E8K2Kv");
    check(kv_layer_class_of(DType::E8Kv) == KvLayerClass::Rk4v4Fusion, "E8Kv -> Rk4v4Fusion (unchanged)");
    check(kv_layer_class_of(DType::E8K3Kv) == KvLayerClass::Rk3v4Fusion, "E8K3Kv -> Rk3v4Fusion");
    check(kv_layer_class_of(DType::E8K2Kv) == KvLayerClass::Rk2v4Fusion, "E8K2Kv -> Rk2v4Fusion");
    check(kv_layer_class_is_fusion(KvLayerClass::Rk2v4Fusion), "rk2v4 is a fusion tier");
    check(kv_cold_resident_bytes_of(KvLayerClass::Rk3v4Fusion) == 6656, "rk3v4 prices 6656 B");
    check(kv_cold_resident_bytes_of(KvLayerClass::Rk2v4Fusion) == 4608, "rk2v4 prices 4608 B");
    check(!kv_layer_class_cold_capable(KvLayerClass::Rk2v4Fusion),
          "rk2v4 is NOT cold-capable (no encoder could pack it)");

    // -------------------------------------------------- 4. the admissibility gate
    std::printf("\n--- 4. the refusal an operator meets ---\n");
    {
        auto thrower = +[]() { (void)kv_dtype_for_storage(KvCacheStorage::E8K3Group64, "--kv-layer-storage[0]"); };
        const std::string m = catch_msg(thrower);
        check(!m.empty(), "kv_dtype_for_storage(rk3v4) THROWS instead of resolving");
        check(m.find("rk3v4") != std::string::npos, "the refusal names 'rk3v4' back to the operator");
        // RE-ANCHORED (dl/e8bar), and the reason is on the record: the old text pinned
        // "no codec", and the tree has since CORRECTED that claim in its own words --
        // kv_storage_dtype.h's comment block now says the CODEC EXISTS (the host codec of
        // record plus the device arm, 75/75) and the missing piece is an ARM for the tier in
        // this engine. A literal `find("no codec")` is therefore FALSE on today's tree.
        // The property the check was standing in for -- "this is the tier's own refusal,
        // not the generic one that tells the operator no enumerator names their storage" --
        // is spelled as what it actually is. The REASON sentence is deliberately NOT
        // pinned: by the flip batch's own ordering it is the last text to move.
        check(m.find("names no enumerator") == std::string::npos,
              "the refusal is the TIER's own, not the generic 'names no enumerator' one");
    }
    {
        auto thrower = +[]() { (void)kv_dtype_for_storage(KvCacheStorage::E8K2Group64, "--kv-dtype"); };
        const std::string m = catch_msg(thrower);
        check(!m.empty() && m.find("rk2v4") != std::string::npos, "kv_dtype_for_storage(rk2v4) throws by name");
    }
    // The regression that matters: the SHIPPED e8 must still resolve, or this whole
    // change would have broken the live tier to add two dead ones.
    check(kv_dtype_for_storage(KvCacheStorage::E8Group64, "regression") == DType::E8Kv,
          "REGRESSION: E8Group64 still resolves to DType::E8Kv");
    check(kv_dtype_for_storage(KvCacheStorage::Iso3Group16, "regression") == DType::ISO3,
          "REGRESSION: Iso3Group16 still resolves to DType::ISO3");
    // The name table is total over the enum and positional (pinned by static_assert in
    // the header; this is the runtime half).
    check(kv_storage_token(KvCacheStorage::E8K3Group64) == "rk3v4-g64", "storage token rk3v4-g64");
    check(kv_storage_token(KvCacheStorage::E8K2Group64) == "rk2v4-g64", "storage token rk2v4-g64");
    check(kv_storage_token(KvCacheStorage::E8Group64) == "rk4v4-g64", "storage token rk4v4-g64 (shipped)");
    check(kv_storage_token(KvCacheStorage::Dropped) == "dropped", "storage token dropped (== 7, unmoved)");
    check(static_cast<int>(KvCacheStorage::Dropped) == 7,
          "Dropped keeps code 7: the new rows were appended AFTER it on purpose");

    // ------------------------------------------- 4b. THE COUPLING, PINNED FROM BOTH SIDES
    // WHY THIS SECTION EXISTS. Section 7 pins the ladder's `selectable` field per row. That field
    // is TRUE for rk3v4/rk2v4 today, so the ladder no longer keeps the engine off a codec it
    // cannot read. What stands between the operator and a silent misread is that
    // kv_dtype_for_storage() REFUSES BY NAME -- and a refusal is only load-bearing while it is
    // the LAST thing rather than the only thing, so the property worth pinning is the COUPLING
    // between "the engine declares this codec readable" (kvcfg::is_readable_codec) and "the
    // resolver agrees". Pinned on BOTH sides, it cannot be satisfied by emptying either one:
    //   * declared READABLE      => the resolver MUST resolve, and to THAT row's own DType;
    //   * declared NOT readable  => the resolver MUST throw, the message MUST name the tier, and
    //                               it MUST NOT be the generic "names no enumerator" refusal.
    // MEASURED RED CAPABILITY, three directions, all by mutation in a shadow include tree (never
    // in the working tree; dl/rk3reader/out_mut_*.txt):
    //   (i)   flip is_readable_codec(Rk3v4) to true and change nothing else
    //         -> the READABLE arm fires and names rk3v4 (and kv_formats.h's own static_assert
    //            fires first, also naming rk3v4 -- both directions are loud);
    //   (ii)  drop the rk3v4 arm from kv_dtype_for_storage (the .cpp-free way: shadow the header)
    //         -> the NOT-READABLE arm fires and names rk3v4;
    //   (iii) flip BOTH consistently (predicate + resolver) while no reader exists
    //         -> the non-vacuity pin below fires, because exactly two rows must be on the
    //            REFUSING side until the batch that wires the reader moves this line with it.
    std::printf("\n--- 4b. the coupling: declared-readable <=> resolvable ---\n");
    {
        struct ERow {
            int             index;
            const char*     tier;
            KvCacheStorage  storage;
        };
        const ERow kERows[] = {
            {kKvBitsE8Index,   "rk4v4", KvCacheStorage::E8Group64},
            {kKvBitsE8K3Index, "rk3v4", KvCacheStorage::E8K3Group64},
            {kKvBitsE8K2Index, "rk2v4", KvCacheStorage::E8K2Group64},
        };
        int readable_side = 0, refusing_side = 0;
        for (const ERow& row : kERows) {
            const auto fmt      = kvcfg::format_from_name(row.tier);
            const bool readable = fmt.has_value() && kvcfg::is_readable_codec(*fmt);
            const std::string where = std::string("--kv-layer-storage[") + row.tier + "]";
            bool threw  = false;
            DType got   = DType::BF16;
            std::string msg;
            try {
                got = kv_dtype_for_storage(row.storage, where);
            } catch (const std::exception& e) {
                threw = true;
                msg   = e.what();
            }
            if (readable) {
                ++readable_side;
                check(!threw && got == kv_bits_tier_dtype(row.index),
                      (std::string("COUPLING: '") + row.tier + "' is declared READABLE, so the "
                       "resolver must resolve it to that row's own DType and must not throw")
                           .c_str());
            } else {
                ++refusing_side;
                check(threw, (std::string("COUPLING: '") + row.tier + "' is NOT declared readable, "
                              "so the resolver must REFUSE it rather than resolve it").c_str());
                check(msg.find(row.tier) != std::string::npos,
                      (std::string("COUPLING: the refusal for '") + row.tier +
                       "' names the tier back to the operator").c_str());
                check(msg.find("names no enumerator") == std::string::npos,
                      (std::string("COUPLING: '") + row.tier + "' gets the TIER's own refusal, not "
                       "the generic 'names no enumerator' one").c_str());
            }
        }
        // NON-VACUITY, BOTH SIDES, WITH TODAY'S EXPECTATION STATED AS A PIN. The second line is
        // the one that moves: it says exactly two e8 rows are unreadable TODAY, and it moves only
        // with the batch that wires the 3-bit/2-bit K plate reader (product/kv_e8_width.h and
        // kv_storage_dtype.h's own refusal name the missing CALLER). Flipping it alone is the
        // silent-misread this whole bar exists to prevent.
        check(readable_side >= 1,
              "COUPLING PIN: at least one e8 row is on the READABLE side (rk4v4)");
        check(refusing_side == 2,
              "COUPLING PIN: exactly TWO e8 rows are on the REFUSING side -- rk3v4 and rk2v4, "
              "whose K plate is 3/2-bit and has no reader on any runtime path. THIS LINE MOVES "
              "ONLY WITH THE BATCH THAT WIRES THE READER");
    }

    // --------------------------------- 5. device capability: still a complete switch
    std::printf("\n--- 5. capability + names ---\n");
    const auto need_e8 = requirements_for_kv_storage(KvCacheStorage::E8Group64);
    const auto need_k3 = requirements_for_kv_storage(KvCacheStorage::E8K3Group64);
    const auto need_k2 = requirements_for_kv_storage(KvCacheStorage::E8K2Group64);
    check(need_k3 == need_e8 && need_k2 == need_e8,
          "rk3v4/rk2v4 demand exactly the rk4v4 codec's capabilities");
    check(kv_storage_name(KvCacheStorage::E8K2Group64) == "rk2v4-g64", "capabilities name map");
    check(kv_storage_name(KvCacheStorage::Dropped) == "dropped", "capabilities name map keeps dropped");

    // ------------- 6. --kv-layer-storage SPELLS them, so selection is not the blocker
    std::printf("\n--- 6. per-layer spec spelling ---\n");
    {
        const auto spec = parse_kv_layer_storage_spec("0-7:rk3v4,8-15:rk2v4");
        check(spec.set[0] && spec.table[0] == KvCacheStorage::E8K3Group64, "spec names rk3v4 on layer 0");
        check(spec.table[15] == KvCacheStorage::E8K2Group64, "spec names rk2v4 on layer 15");
    }
    {
        bool threw = false;
        try { (void)parse_kv_layer_storage_spec("0:e8k9"); } catch (const std::exception&) { threw = true; }
        check(threw, "spec still rejects an unknown token (e8k9)");
    }

    // ------------------------------------------------- 7. POLICY PIN (row-level, literal)
    // WHY THIS SECTION EXISTS, and why it is NOT section 4 repeated.
    //
    // Section 4 asks "does the engine REFUSE the tier?" -- a property of the resolver, read
    // through catch_msg(). That is a TOTALITY check: it is satisfied by any refusal, and the
    // moment the ban is lifted the throw is gone and its three checks become tests of the
    // empty string. It cannot say WHICH ladder row moved, and it cannot see a row that gained
    // reachability without the resolver changing at all -- which is exactly the shape of the
    // flip the owner's order is about (product/kv_bit_budget.h:289-290, the 4th field).
    //
    // This section pins the gate itself, PER ROW, AS A LITERAL: the ladder's own `selectable`
    // field. `selectable == false` is what keeps a row in the ladder (cost, geometry and name
    // all readable and pinnable) while the DP refuses to CHOOSE it -- kv_bit_budget_solve_impl
    // skips unselectable rows. Opening the gate is therefore a DELIBERATE edit of this table
    // in the same landing as the batch, and can never be a silent drift.
    //
    // THE COUPLING, stated rather than duplicated: a row may not be marked selectable unless
    // the score table (a PREFIX of the ladder) covers it -- kv_bit_budget.h:1021 makes the
    // table 6 rows, :1037-1042 `scores_cover_every_selectable_gear()`, and :1048 asserts it.
    // That invariant is a COMPILE-time red in the header and this bar does not restate it: a
    // second spelling of one question is the defect the header's own comment at :1022-1036
    // was written about. What the header cannot see is a table grown WITHOUT real priors; that
    // is a value defect for the file's owner, not a wiring one for this bar.
    struct RowPolicy {
        int         index;       // ladder slot
        const char* tier;        // the ladder's own spec_name for that slot
        bool        selectable;  // the policy THIS tree declares for it
    };
    // RE-ALIGNED (dl/rk3reader, 2026-09-20). These pins said `false` while the tree's policy said
    // `true`: the gate was opened by dl/e8mixwire on a ruling recorded verbatim in
    // dl/e8mixwire/land/patch_bar_pins.py:118-124 ("... Firing buys a selectable tier that is
    // loudly refused where it cannot yet be served"), and the pin was left behind -- so this bar
    // was red on its own tree and nobody saw it (registry A13c). The pin now says what the tree
    // says; what replaces its old job is check 4b. A pin that only restates a field can be
    // satisfied by flipping that field; the COUPLING cannot, because it is pinned from both sides.
    const RowPolicy kPolicy[] = {
        {kKvBitsE8Index,   "rk4v4", true },   // shipped: the 4-bit row is choosable
        {kKvBitsE8K3Index, "rk3v4", true },   // GATE OPEN (dl/e8mixwire): choosable;
        {kKvBitsE8K2Index, "rk2v4", true },   //   what is missing is the READER -> check 4b
    };
    for (const RowPolicy& row : kPolicy) {
        check(std::string(kKvBitBudgetTiers[row.index].spec_name) == row.tier,
              (std::string("POLICY: ladder row ") + std::to_string(row.index) + " is named '" +
               row.tier + "'").c_str());
        check(kKvBitBudgetTiers[row.index].selectable == row.selectable,
              (std::string("POLICY PIN: ladder row ") + std::to_string(row.index) + " (" +
               row.tier + ") selectable == " + (row.selectable ? "true" : "false") +
               " -- this row moves only with the batch that makes the tier runnable").c_str());
    }

    std::printf("\n== %d/%d ==\n", g_ok, g_total);
    return g_ok == g_total ? 0 : 1;
}
