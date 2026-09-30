#pragma once

// Fractional KV bit-budget allocator (the C++ side).
//
// CORRECTION (D12/M11). The text here used to say that the Python partner
// tools/archkit/kv_bit_budget.py "is no longer in the tree". That is FALSE and
// was already false when it was written: the file is present at
// tools/archkit/kv_bit_budget.py, 154 lines, sha256
// 3d5558b53a9624bae0279c407228e385114edd31a0f752232e5b95ea53d54fff, and it shows
// up in `git diff --stat` as +29 lines of uncommitted work, i.e. someone is
// actively editing it. The correct statement is the weaker and more useful one:
// it is present but FORKED, and its own header banner says so. Do not treat a
// disagreement between the two as a bug in either -- the banner is the contract.
//   * its ladder has drifted from this file's by construction (it kept nominal
//     widths after the plane-geometry fix moved this file's numbers;
//     see the tier table below, which is the authority);
//   * its banner claims "C++ rk4v4 layer limit = 10" while
//     kKvBitBudgetE8LayerLimit below is 8, so the banner is stale on that one
//     datum too -- which is the same lesson as the cold-stride comment in
//     kv_tier_formats.h: a comment is not a contract.
// The parity claim this header used to make ("agree byte-for-byte") therefore
// no longer holds and must not be re-asserted without re-deriving it.
//
// Given a target average bit budget per KV element, pick one KV tier per full-attention
// layer so that the total bit cost stays inside the budget while the total quality
// penalty is minimised. Solved exactly with a small DP whose state is (bits used, cold
// layers used); the rk4v4 count is carried in the surviving path rather than in the state key.
// The resulting per-layer assignment is packed into the existing --kv-layer-storage grammar,
// so the rest of the engine keeps its single resolution path.
//
// Tier costs (bits/element, K+V averaged) and penalties (needle-sweep prior, lower is
// better) follow the ENGINE's plane geometry and the measured needle/speed columns, not the
// deleted Python tool's nominal widths. The ladder below is the authority; the summary here
// is only a reading aid, and it has been stale before (it kept the Python tool's fp8 8.03 /
// rk4v4 4.06 / iso4e 3.00 after the geometry fix moved them): bf16 16.00/0.00, int8 8.25/0.02,
// fp8 8.50/0.03, nvfp4 4.50/0.30, rk4v4 4.25/0.08, iso4e 4.50/2.00 (pinned out).
//
// rk4v4 is only allowed on the leading layers: the rk4v4 tier is verified for layers 0..7 while
// the shipped high-layer rk4v4 path degrades (_TODO.md 96/116); iso4e/nvfp4 take over above.
// kKvBitBudgetE8LayerLimit below is that window's size (8), so the DP's rk4v4 block -- which the
// packing always emits first -- cannot leave it.
//
// Opt-in cold tier (engine knobs: --cold-policy none/window/host/disk/host-then-disk +
// --max-cold-pages):
// with a cold capacity in pages, a layer may instead live in the cold pool. At most
// cold_cap layers may be cold; cold_used is an exact DP dimension. Packing keeps rk4v4 FIRST
// (the rk4v4 limit is a leading-layer constraint, rk4v4 must keep slots 0..count-1 inside its
// verified window), then places cold on the next-shallow block; deep layers keep the hot
// high-precision tail (_TODO.md 46/47). "cold" is NOT a --kv-layer-storage tier (grammar:
// bf16/int8/fp8/nvfp4/iso4e/rk4v4): cold-planned layers are deployed via
// --cold-policy host|disk|host-then-disk --max-cold-pages N.
//
// A cold layer is charged its REAL slot bytes, not the "0.00 hot bits/element" this header
// used to declare: the slot pool is device memory (kKvBitBudgetColdSlotBytes and the geometry
// block below carry the evidence), so the old "the layer's GPU footprint is freed" story only
// ever described the RESIDENT page pool and it hid a net device-memory increase on every
// sub-int8 stack. See kv_bit_budget_plane_bytes() / kv_bit_budget_cold_delta_bytes() for the
// per-tier head-page comparison, and kv_bit_budget_solve_audited() for the real-byte audit
// against the same budget solved with the cold pool off.
//
// With cold_cap == 0 the allocation is bit-identical to the pre-cold header (the cold
// candidate is only enumerated under `cold_cap > 0`). The recorded cold-mode grid lives in
// research/notes/A_kvbit_cold_cpp.md and research/notes/A_kvbit_cold.md; both describe the
// pre-fix "cold is free" model AND the pre-correction ladder (fp8 803 / rk4v4 406 / iso4e 300,
// rk4v4_limit 8), so neither can be replayed against this header - the ladder was later corrected
// to fp8 850 / rk4v4 425 / iso4e 450 (rk4v4_limit 10 then, 8 -- the value below -- now).
// Cross-check with a fresh Python mirror
// instead; the cold-off column is the part that must stay identical, and it is, by the
// cold_cap == 0 argument above.
//
// kKvBitBudgetColdSlotBytes is the cold rANS record. It is no longer SPELLED here: it is
// DERIVED from product/kv_tier_formats.h kKvColdPoolStrideBytes (which is host-only by
// design -- that is why the arithmetic is repeated there instead of including the ops
// header, which needs cuda_runtime.h). It used to be a literal that only ever compared
// against its own hand-entered neighbours in this file, so any self-consistent world
// satisfied it; on 2026-09-18 that literal was 6688 while ops::kEntropyNvfp4SlotBytes and
// kKvColdPoolStrideBytes were 9632, and the three could not be moved together because
// nothing in the build tied THIS one to the other two. Now this file has no cold byte of
// its own to move: the record is defined once, in product/kv_tier_formats.h, and the
// remaining two-sided pin (this value == ops::kEntropyNvfp4SlotBytes == the record
// cold_slot_stride_for() hands the arena) lives in decoder_state.cpp, the one TU that can
// see both an ops header and this one.

#include "ninfer/types.h"
#include "product/kv_options.h"        // parse_kv_storage / parse_kv_layer_storage_spec
#include "product/kv_storage_dtype.h"  // kv_dtype_for_storage: the ONE deploy decision point
#include "product/kv_tier_formats.h"   // kKvColdPoolStrideBytes: the cold record, once
// [F1231 items 1+6] THE TWO HEADERS THAT OWN THE GEOMETRY THIS FILE'S `bits_x100` COLUMN IS
// PRICED FROM, included so the column can be CHECKED against them at compile time rather than
// asserted in prose: `cell_rung(CellMode).plane_bytes` is the descent's own plane price and
// `e8_kv_pair_bits_x100(K, V)` is the CLI storage grammar's pair price. The edge is a DAG --
// neither header includes this one (checked by content, not by assumption), and both are
// header-only product headers with no engine dependency.
#include "product/kv_cell_modes.h"     // cell_rung / CellMode / CellRung::plane_bytes
#include "product/kv_e8_width.h"       // e8_kv_pair_bits_x100 / E8KvPlaneFormat
// kvreach-k1-includes

#include <array>
#include <cmath>
#include <optional>
#include <sstream>
// kvreach-k2-optional
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::product {

// How many layers the DP may place on rk4v4, anywhere in the stack. This is a count, not a
// leading window: the solver carries the running count in its state and refuses to exceed the
// cap, so the cap is the only thing that bounds rk4v4 exposure.
//
// The value is 8, and it is bounded above by measurement, not chosen for headroom:
//   * docs/maintainer/kv-strategy-matrix.md section 2 rule 2: "rk4v4 只在前 8 个全注意力层可用
//     （长上下文）", with the per-arm record
//       0-7:rk4v4 = 8/8 (repeated twice)   8-15:rk4v4 = 0/8   8-11:rk4v4 = 2/8
//       12-15:rk4v4 = 0/8                 14-15:rk4v4 = 4/8  0-11:rk4v4 = 0/8   all:rk4v4 = 0/8
//     and the rule it states: "实用规则：rk4v4 只用层 0-7；上层用 iso4e/int8/nvfp4 补".
//   * section 6.1 measures the 10-layer table this comment used to cite (rk4v4 on 8,9,13,14) at
//     6/8 against all-nvfp4's 8/8 at the same context: two needles worse. The only measured
//     ten-layer configuration is the degraded one.
//   * the shipped table agrees: qwen3_6_27b/impl/variant.cpp builds rk4v4 on {0,1,3,4,6,7} under
//     a "CAP THE Rk4v4 LAYERS AT FULL-ATTENTION INDEX 7" comment that cites these same arms.
// So 11..15 are NOT unmeasured (that claim, and the variant.cpp table it quoted, were both stale).
// BUT the count rule that replaced it is UNREPRESENTABLE, per the same control re-measurement: the
// failure is neither monotone in the count nor explained by the highest index --
//     {0..12}:rk4v4 -> 9/27      {0..13}:rk4v4 -> 27/27   (one layer MORE, and it passes)
//     rk4v4 on layer 15 -> 27/27 in two different sets
// so "the safe count is at least 10 and below 16" cannot be stated as a count at all. That is an
// independent confirmation of this file's own "a SET, not a COUNT" argument, arrived at from data
// rather than from the factory table's shape. The silent logic/aliasing defect (_TODO.md 96,
// compute-sanitizer clean) remains the only candidate mechanism and is still open.
//
// The count is also the highest layer index rk4v4 can reach, because the packing emits rk4v4 first
// (kKvBitBudgetPackOrder[0]); at cap 10 the solver's own output is `0-9:rk4v4,...` for every
// budget between 4.50 and 6.00, i.e. it spends the cap on layers 8 and 9 -- which the control
// re-measurement above leaves as OPEN membership, not as a measured harm. At cap 8 the same
// budgets return `0-7:rk4v4,...`. The binding fact is the count ceiling itself, not the layer list.
//
// RESOLVED (re-measured, PATCHSET/RK4V4-CONTROL/REPORT.md): the "27" was never a needle count. The
// harness grades the exact prefix of ONE 27-CHARACTER needle string (ORCHID=493817; COLOR=COBALT),
// so `0/27` reads "0 of 27 characters of one needle". No 27-needle suite exists -- which is why the
// 8 and the 27 could never be aligned: they were never two counts of the same thing.
// What IS re-verified and decisive: all 16 layers on rk4v4 answers 0/27, TWICE, identically (sigma=0
// on every retrieval and acceptance metric; only throughput is noisy at ~2-3%), with the engine's
// own `kv cache dtype rk4v4-group64` line proving the tier actually ran -- and it still answers 0/27
// under `--spec none`, i.e. with no draft module in the loop at all. So the retrieval failure is a
// KV-fidelity failure, NOT a speculative-decoding artifact.
//
// Raise this only with a measurement that covers the layer range being raised over.
//
// =====================================================================================
// RAISED 8 -> 16 ON 2026-09-29 BY OWNER INSTRUCTION. THIS IS A POLICY CHANGE, NOT A
// MEASUREMENT, AND THE MEASUREMENT ABOVE IS THE ONE IT KNOWINGLY ACCEPTS.
// =====================================================================================
// The owner's sentence, verbatim: 「我允许你在**任何层用e8 2bit** 这个没有办法，**我可以放宽限制**
// …总之 e8 2bit 因为预算不得不开那就开吧。」 -- i.e. e8-2bit may be opened on ANY layer, because
// the budget has to open it. What that unblocks, in the comparison's own numbers: with the cap at 8
// the layer-KV floor is (8 x 4.25 + 8 x 4.50) / 16 = 4.375 b/el on a 16-layer stack, which is why
// 2.8 and 3.5 measured UNREACHABLE on the engine side while the instrument side could express them.
// At 16 every budget point is expressible on both sides.
//
// WHAT IS ACCEPTED, ON THE RECORD: the paragraph above measures all-16-rk4v4 at 0/27 retrieval,
// twice, identically, with the tier proven to have run. That reading is not withdrawn and this
// relaxation does not claim it wrong; it says the budget requirement outranks it for the comparison.
// Any accuracy reading taken at a budget that only the raised cap can express is therefore taken on
// a configuration this file's own evidence rates as degraded, and must be reported as such.
//
// THE MEASUREMENT THE SENTENCE ABOVE STILL ASKS FOR IS OWED, BY NAME: a retrieval/acceptance arm
// over the 8..15 band at the deep budgets. Nothing in this tree has taken it (dl/musesparkfix F1227
// did not; dl/kvcompare's arms could not express the budgets).
//
// SCOPE, SO THE NUMBER IS NOT MISREAD: 16 is this comparison stack's full-attention layer count, so
// on qwen3_6_27b it means "no cap". On a stack with MORE paged layers (muse_glimmer_30b 52,
// spark_x2_5_4b 36) this constant is still a cap, deliberately: the owner's relaxation was given for
// the 16-layer comparison, and extending it to another target needs that target's own reading.
inline constexpr std::int32_t kKvBitBudgetE8LayerLimit = 16;

// ---------------------------------------------------------------------------
// D4/B5: the rk4v4 constraint is a SET, not a COUNT.
// ---------------------------------------------------------------------------
// The count above is the DEFAULT CONSTRUCTION of that set -- the leading window
// 0..limit-1 -- and kv_bit_budget_solve_impl still consumes it exactly that way,
// so every allocation this header already produced is unchanged. What was wrong
// is the REPRESENTATION, and the two things that show it up are both in this
// tree, not constructed:
//   * the factory table's own rk4v4 set is {0,1,3,4,6,7} -- HOLES at 2 and 5.
//     A prefix family cannot name it at any limit, which is why a sweep of
//     (budget, rk4v4_limit) only ever returns prefixes and never the shipped table.
//   * the factory table's own rk4v4 set is {0,1,3,4,6,7} and the harmful layers are not contiguous
//     either, so "one clean block above 8" is refuted by measurement -- which is what this
//     argument needs. NOTE: the specific list formerly written here as {8,9,13,14} is NOT
//     confirmed by the control re-measurement (layer 15 passes 27/27 in two different sets, and
//     {0..13}:rk4v4 passes), so treat the membership as open while the non-contiguity stands.
// A COUNT is a lossy encoding of a SET that happens to round-trip only when the
// set is a prefix. This struct is the faithful encoding.
//
// It is deliberately ADDITIVE: nothing in this header reads it unless a caller
// passes one in, so the default path (and the cold_cap == 0 byte identity the
// notes below rely on) is untouched.
struct KvBitBudgetRk4v4Set {
    std::array<bool, kKvLayerStorageSlots> allowed{};

    [[nodiscard]] std::int32_t count() const noexcept {
        std::int32_t n = 0;
        for (const bool a : allowed) { if (a) { ++n; } }
        return n;
    }

    // True iff the set is exactly 0..count-1, i.e. iff the COUNT encoding could
    // have carried it. This is the predicate that decides whether an answer
    // produced from a set is even expressible in the count grammar, so a caller
    // can say "this plan needs the set form" instead of silently losing layers.
    [[nodiscard]] bool is_prefix() const noexcept {
        bool gap_seen = false;
        for (const bool a : allowed) {
            if (!a) {
                gap_seen = true;
            } else if (gap_seen) {
                return false;
            }
        }
        return true;
    }

    // "0,1,3,4,6,7". Lets a report or a test name the set it solved with, and
    // lets a reader compare it to the factory table without re-deriving it from
    // an allocation (which is the only way to see it today).
    [[nodiscard]] std::string to_layer_list() const {
        std::string out;
        for (std::size_t i = 0; i < allowed.size(); ++i) {
            if (!allowed[i]) { continue; }
            if (!out.empty()) { out += ","; }
            out += std::to_string(i);
        }
        return out;
    }
};

// The default construction: the leading window a COUNT can express.
[[nodiscard]] inline KvBitBudgetRk4v4Set kv_bit_budget_rk4v4_prefix_set(
    std::int32_t rk4v4_limit) noexcept {
    KvBitBudgetRk4v4Set set;
    for (std::int32_t i = 0;
         i < rk4v4_limit && i < static_cast<std::int32_t>(kKvLayerStorageSlots); ++i) {
        set.allowed[static_cast<std::size_t>(i)] = true;
    }
    return set;
}

// An ARBITRARY set. Returns false -- and writes nothing -- when a layer index is
// out of range, rather than silently dropping the layer, because a silently
// dropped constraint is exactly the failure mode this struct exists to remove.
[[nodiscard]] inline bool kv_bit_budget_rk4v4_set_from_layers(
    KvBitBudgetRk4v4Set* set, const std::vector<std::int32_t>& layers) noexcept {
    if (set == nullptr) { return false; }
    KvBitBudgetRk4v4Set built;
    for (const std::int32_t layer : layers) {
        if (layer < 0 || layer >= static_cast<std::int32_t>(kKvLayerStorageSlots)) {
            return false;
        }
        built.allowed[static_cast<std::size_t>(layer)] = true;
    }
    *set = built;
    return true;
}

// 0.01-bit resolution, matching the Python tool.
inline constexpr std::int32_t kKvBitBudgetScale = 100;

struct KvBitBudgetTier {
    const char* spec_name;
    std::int32_t bits_x100;
    std::int32_t penalty_x100;
    // THE REACHABILITY GATE. `selectable == false` keeps a row in the ladder -- so its
    // cost, its geometry and its name are all readable and pinnable -- while the DP
    // refuses to CHOOSE it, which is what stops the two rk4v4 widths from being deployed
    // before their K code plate has an encoder and a decoder branch. Without this flag
    // the rows would be worse than absent: a ceiling between 3.25 and 4.25 has no other
    // feasible allocation, so the DP would be FORCED onto them and the engine would
    // build a DType it cannot read. Flip this one field when the codec lands.
    bool selectable = true;
};

// Cost ladder (cheapest first is not needed; the DP explores all of them).
//
// =============================================================================================
// [F1231 2026-09-29] ITEM 1 -- THE UNIT OF `bits_x100`, DECLARED PER ROW, BECAUSE THE COLUMN WAS
// IN TWO OF THEM AND THE MIXING WAS INVISIBLE.
// =============================================================================================
//
// THE UNIT IS: **bits per KV value, averaged over K and V, of THE ROW'S OWN GEOMETRY.** That
// sentence was already here ("per KV element, K+V averaged") and it was already TRUE for every
// row -- what it did not say is **WHICH V EACH ROW AVERAGES WITH**, and that is where the mixing
// lived. MEASURED (dl/rework/arms/tier_column_probe.cpp, PRE-EDIT, so this is a reading and not a
// re-derivation):
//
//   * FIVE ROWS ARE TWO-PLANE-SYMMETRIC CELLS: `bf16 1600, int8 825, nvfp4 450, rk4v4 425,
//     iso4e 450`. For these the declared bits EQUAL `cell_rung(mode).plane_bytes * 100 / 2048`
//     exactly -- i.e. K and V are the same tier, which is what a `CellMode` cell IS
//     (`cell_vector_charge` sums one `plane_bytes` per plane). **This is the descent's own
//     currency**, and it is the currency a solve must use.
//   * TWO ROWS ARE K-NARROWED PAIRS: `rk3v4 375` and `rk2v4 325`, whose V STAYS i4 (the row
//     comments said so; the audit now proves it: `375 == e8_kv_pair_bits_x100(B3, B4)` and
//     `325 == e8_kv_pair_bits_x100(B2, B4)`, both asserted below). **This is the CLI storage
//     grammar's currency** -- `--kv-layer-storage`'s `rk3v4`/`rk2v4` ARE "K narrowed, V still
//     i4" -- and it is NOT the geometry the descent's `CellMode` can express.
//   * ONE ROW HAS NO GEOMETRY AT ALL: `fp8`. There is no `CellMode::Fp8`, so the descent can
//     neither price nor choose it; its 850 is the symmetric fp8 plane (17,408 B) by derivation.
//   ⚠⚠ AND THE ROW WHERE THE TWO CURRENCIES MEET IS `rk3v4`: the ladder declares **375**
//   ((K 6,656 + V i4 8,704)/2) while the descent's own `CellMode::Rk3v4` is a SYMMETRIC cell
//   priced **325** (6,656/2,048 x 100). **One name, two objects, 50 units apart.** The audit says
//   this is the mixing's ENTIRE extent: of the eight rows, seven are priced by exactly one
//   geometry and only `rk3v4` is priced by one geometry while a DIFFERENT object of the same name
//   exists in the engine. Both numbers are pinned below so neither can drift, and the choice of
//   which one a solve should charge is NOT taken here (see the OWED note).
//
// ⚠ AND NOTHING NUMERIC CHANGES IN THIS EDIT, on purpose. The numbers are the shipped answers'
// basis: the probe's own sweep shows SHIPPED SOLVES ALREADY CHOOSING THESE ROWS
// (`ladder_coldcap[lim8]` at 4.00 bits -> `0:rk4v4,1-2:rk3v4,3-7:rk2v4,8-15:nvfp4`;
// `gear_default[lim8]` -> `0-2:rk2v4,3-4:rk4v4,5-7:rk2v4,8-15:nvfp4`). A "correction" of 375 -> 325
// would move those plans, and the rework's rule is that the change in the shipped answers must be
// NAMED before it is made -- so it is named here and NOT made. The diff of the pre/post sweeps is
// the proof that nothing moved.
//
// THE OWED DECISION, NAMED RATHER THAN TAKEN: whether the ladder should price `rk3v4` as the
// storage pair (375, today) or as the symmetric cell the engine would build for it (325). It is a
// question about which OBJECT a solve is choosing -- a storage spelling or a `CellMode` -- and it
// belongs to the solve's design, not to a comment. Until it is answered, a solve that compares
// `rk3v4` against any symmetric row is comparing two geometries, and the conversion it must use is
// written down below.
//
// THE CONVERSION, SO THE NEXT LINE DOES NOT RE-DERIVE IT:
//     symmetric cell  : bits_x100 = cell_rung(mode).plane_bytes * 100 / 2048
//     K-narrowed pair : bits_x100 = (K_plane_bytes + V_i4_plane_bytes) / 2 * 100 / 2048
//                                      = e8_kv_pair_bits_x100(K, E8KvPlaneFormat::B4)
// A solve must convert BOTH sides into the symmetric-cell currency before comparing them, because
// that is the currency the engine's charge (`cell_vector_charge`) and the descent's budget are in.
//
// Bit costs are the ENGINE's plane geometry (per KV element, K+V averaged), not nominal
// format widths - see src/targets/qwen3_6/impl/state/decoder_state.cpp: every tier is a
// nibble or byte code plane plus a scale plane, so int8 pays 16 bits per 64 elements of
// scale (8.25), rk4v4 pays 16 per 64 (4.25), nvfp4 pays 8 per 16 (4.50), and fp8 pays 8 bits
// per element of code plus 8 per 16 of E4M3FN scale, i.e. 8.50.
// fp8's scale plane is E4M3FN at group 16 - the same shape nvfp4/iso4e use and the only one
// the fp8 attention kernels read (gqa_attention_decode_fp8.cuh:159 kFp8Groups = D/16,
// gqa_attention_prefill.cu:200/360 fp8 arms) and the only one ops/wrapper/gqa_attention.cpp
// accepts for a packed-16 tier (:80-82, :121-133, :208-221). The tier is still dominated by
// int8 on cost (8.50 > 8.25 at a worse penalty), so the DP will not pick it on its own; it
// is now a runnable reference point rather than a refusing one.
// iso4e SHARES nvfp4's plane geometry (two codes per byte over per-16 E4M3FN scales, hence
// 4.50) but is NOT the nvfp4 kernel and NOT the same K format: Iso4eGroup16 maps to
// DType::ISO4E (decoder_state.cpp:28-33 gives it the same quant_group and nothing more), whose
// decode/encode paths are ops/kernel/gqa_attention_decode_iso3.cuh - K carries sign-magnitude
// ISO4E nibbles, not E2M1 - and which never touches the nvfp4 second-stage residual plane set,
// while the DType::NVFP4 decode arm runs an extra residual QK pass (decoder_state.cpp:156-185).
// Sharing planes therefore does NOT make iso4e an alias of nvfp4: it can never beat nvfp4 on
// bits (equal) and the measured speed column is a tie (40.3 vs 40.4 tok/s below). Its penalty
// is left pinned PAST the scale so the DP treats it as unusable rather than as a cost-equal
// duplicate of nvfp4 - that pin is a naming/policy choice, not a measured quality claim, and
// the bit column alone (450 == 450) is what makes the two indistinguishable to the DP.
// Measured, 64k NIAH, all 16 layers at one tier (tools/archkit/kv_tier_matrix.py):
//   int8 86.2 > bf16 80.7 > shipped mix 57.1 > nvfp4 40.3 ~ iso4e 40.4 > rk4v4 27.1 tok/s,
//   with rk4v4 answering 0/27 (long-context retrieval breaks). fp8 was not measured in that
//   sweep: every fp8 arm of it died in the planner's FP16 scale plane before reaching the
//   kernels, so its speed column is still open.
inline constexpr std::array<KvBitBudgetTier, 8> kKvBitBudgetTiers{{
    {"bf16", 1600, 0},
    {"int8", 825, 2},
    {"fp8", 850, 3},     // E4M3 scales at group 16 (8.50 bits); still dominated by int8
    {"nvfp4", 450, 30},
    {"rk4v4", 425, 8},      // nibble + FP16/g64
    {"iso4e", 450, 30},   // == nvfp4 planes; THE 200 PIN IS RETIRED -- F1229, see the note below
    // The rk4v4 family's narrower K planes. bits_x100 is DERIVED (product/kv_e8_width.h:
    // layer bytes 15360 / 13312 * 8 * 100 / 32768 -> 375 / 325), not written down, and
    // the w=4 row above is the derivation's check against the shipped plane.
    // THE GATE IS OPEN (dl/e8mixwire/land/patch_flip_batch.py, one item, eight clusters).
    // selectable=false was "the hard gate" for as long as the tier had no codec, no planner arm,
    // no reader in a target and no caller. The codec landed (dl/e8wire), the planner arm landed
    // (dl/e8decarm: decoder_state.cpp:422, sha16 42f5c98ef5f17f99), and the reader landed in the
    // ninfer_ops closure (src/ops/kernel/e8_lattice_kv_plane_inst.cu, 44a78e174cbe88ca). The
    // penalty is no longer pinned past the scale: 990 was the preference that went with the
    // prohibition, and leaving it would make the two rows unusable for a second reason while
    // claiming they are available. NOTE what this pair still does NOT buy -- see the precondition
    // notice at the top of patch_flip_batch.py: no caller and no append arm exist yet.
    // [F1231 item 1] THESE TWO ROWS ARE IN THE OTHER CURRENCY, AND THE COMMENTS NOW SAY SO.
    // `375 = (K 6,656 + V i4 8,704)/2 x 100/2,048` and `325 = (K 4,608 + V i4 8,704)/2 x 100/2,048`
    // -- i.e. V is NOT the row's own tier, which is what makes them different objects from every
    // row above. `rk3v4` additionally disagrees with the engine's own `CellMode::Rk3v4` (a
    // SYMMETRIC cell at 325); `rk2v4` has no `CellMode` at all. Both facts are pinned below.
    {"rk3v4", 375, 12, true},   // K at 3 bits (96 B/row); V still i4  [pair ruler: see the banner]
    {"rk2v4", 325, 16, true},   // K at 2 bits (64 B/row); V still i4  [pair ruler: see the banner]
}};

// =============================================================================================
// [F1231 2026-09-29] ITEM 6 -- `fp8` IS DOMINATED AND IS NAMED AS DOMINATED HERE. And a second,
// LIVE defect rides on the same row, which is why this is two statements and not one.
// =============================================================================================
//
// (1) DOMINATION, ON EVERY AXIS THE ENGINE HAS, MEASURED (`tier_column_probe.cpp`, three tables):
//       * COST     : fp8 850 > int8 825 -- fp8 costs MORE bits;
//       * QUALITY  : default (int8 2, fp8 3); measured_short (int8 49, fp8 49); measured_long
//                    (int8 2, fp8 3) -- fp8 is never BETTER, and on two of three tables it is worse;
//       * SPEED    : int8 0, fp8 0 on all three tables -- a tie, and int8 is the measured fastest.
//     ⇒ **NO WEIGHT OF `--kv-quality-weight` CAN CHOOSE fp8 WHILE int8 IS ADMISSIBLE**: the blend is
//     a convex combination, so int8's blended penalty is <= fp8's at every weight, and where they
//     tie the BITS axis decides -- against fp8. The `static_assert` below pins the default table's
//     half of that; the probe proves it for both measured tables.
//     ⚠ WHAT IS *NOT* DONE HERE, AND WHY: `selectable` is left `true`. Flipping it would (a) retire
//     the deliberate decision that made this row "a runnable reference point rather than a refusing
//     one", and (b) OVERRIDE AN OPERATOR-SUPPLIED `--kv-tier-scores` TABLE, which is an INPUT in
//     this tree's own terms. The exclusion is therefore stated and pinned rather than enforced,
//     because enforcement would take a decision that is not this line's to take. Named, not hidden.
//
// (2) AND THE LIVE DEFECT: `--kv-dtype fp8` BUILDS THE DOCUMENTED-DEFECTIVE ROTATION-ON STATE BY
//     DEFAULT, AND NOTHING REFUSES IT. The site is `gqa_attention_decode_fp8.cuh:268-271`, whose
//     own comment reads as if the gate protects the call -- *"the gate lives inside
//     `gqa_isoquant_rot_block4()`, so 'rotation off' stays identity on both sides"* -- while the
//     call `gqa_prefill_nvfp4_rotate_8(x, d)` is made unconditionally. MEASURED BY A SIBLING LINE
//     2026-09-29 (`ninfer-perplexity`, same instrument family as the nats column above): the cost
//     of rotation ON at fp8 is **+0.31550 nats worse**. So the tier is not merely dominated: its
//     DEFAULT SPELLING is defective, and no refusal names it.
//     ⚠ SCOPE: the DESCENT cannot reach fp8 at all (no `CellMode`), so this binds (a) the ladder/DP
//     above, which may choose fp8 when nothing cheaper is admissible, and (b) every front end that
//     accepts `--kv-dtype fp8`. The descent's own `kv_bit_budget_solve` calls are unaffected.
//     ⚠ THIS LINE DID NOT MEASURE IT AND DOES NOT RE-MEASURE IT; the reading is the sibling's, it is
//     dated, and the reproducible spelling is `--kv-dtype fp8` against the fp8 rotation-off control.
static_assert(kKvBitBudgetTiers[2].bits_x100 > kKvBitBudgetTiers[1].bits_x100,
              "F1231 item 6: fp8 must cost MORE bits than int8 (850 > 825), or its domination "
              "argument above is no longer the argument that holds");
// (The DEFAULT-table half of the fp8 domination pin lives at the FOOT of
// `kv_bit_budget_default_scores()` below, where the table it asserts about exists -- it was first
// written here and could not compile, because this point in the file is ~1,080 lines earlier.)

// =============================================================================================
// [F1231] THE QUARANTINE, IN THE FILE'S OWN WORDS, RE-STATED WHERE THE GATE FLIP IS RECORDED.
// =============================================================================================
// `rk3v4`/`rk2v4` are, by this file's own note on their score rows, *"PRIORS, NOT MEASUREMENTS --
// the only invented numbers in this batch"*. `selectable = true` (the gate flip) means the DP MAY
// CHOOSE THEM, and the probe shows shipped solves DO (`ladder_coldcap[lim8]` and
// `gear_default[lim8]` above). **THE FLIP CHANGED REACHABILITY, NOT EVIDENCE**: a row whose cost
// column is derived but whose QUALITY and SPEED columns are invented cannot be the basis of a
// solve, and it must stay out of any allocation whose objective or constraint is a measured
// quantity until a measurement exists. Keep this block beside the gate-flip note it qualifies.

// =============================================================================================
// THE PINS. THE COLUMN IS NOW SELF-CHECKING: any future edit that moves a row's bits, or its
// geometry, fails to compile HERE with the geometry's owner quoted.
// =============================================================================================
// (1) THE FIVE TWO-PLANE-SYMMETRIC ROWS ARE THE DESCENT'S OWN PRICE. `iso4e` is pinned against
// `CellMode::Nvfp4` because it shares that plane geometry (`decoder_state.cpp` gives it the same
// quant_group and nothing more) -- and there is no `CellMode::Iso4e`, which is why that is the pin.
static_assert(kKvBitBudgetTiers[0].bits_x100 ==
                      cell_rung(CellMode::Bf16).plane_bytes * 100 / 2048 &&
                  kKvBitBudgetTiers[1].bits_x100 ==
                      cell_rung(CellMode::Int8).plane_bytes * 100 / 2048 &&
                  kKvBitBudgetTiers[3].bits_x100 ==
                      cell_rung(CellMode::Nvfp4).plane_bytes * 100 / 2048 &&
                  kKvBitBudgetTiers[4].bits_x100 ==
                      cell_rung(CellMode::Rk4v4).plane_bytes * 100 / 2048 &&
                  kKvBitBudgetTiers[5].bits_x100 ==
                      cell_rung(CellMode::Nvfp4).plane_bytes * 100 / 2048,
              "F1231 item 1: bf16/int8/nvfp4/rk4v4/iso4e are TWO-PLANE-SYMMETRIC cells and their "
              "bits_x100 IS the descent's own price (cell_rung(mode).plane_bytes*100/2048). If "
              "this fires, either a plane price moved in kv_cell_modes.h or a row changed ruler "
              "-- and the second case is the mixing this block exists to make impossible in "
              "silence");
// (2) `fp8` HAS NO `CellMode`: its 850 is the symmetric fp8 plane (8 code bits + 8 per 16 of
// E4M3FN scale over 64 tokens = 8.50 b/el = 17,408 B), so it is pinned against that literal and
// the provenance is named. A `CellMode::Fp8` would make this pin redundant, not wrong.
static_assert(kKvBitBudgetTiers[2].bits_x100 == 17408 * 100 / 2048,
              "F1231 item 1: fp8's 850 is the symmetric fp8 plane (17,408 B = 8.50 b/el)");
// (3) THE TWO K-NARROWED ROWS ARE THE CLI STORAGE GRAMMAR'S PAIR, AND THEY ARE PINNED AGAINST THE
// HEADER THAT OWNS THAT GEOMETRY (`kv_e8_width.h`), not against a typed number.
static_assert(kKvBitBudgetTiers[6].bits_x100 ==
                  e8_kv_pair_bits_x100(E8KvPlaneFormat::B3, E8KvPlaneFormat::B4),
              "F1231 item 1: rk3v4's 375 is the (K at 3 bits, V still i4) PAIR -- the CLI's "
              "storage geometry, not the CellMode cell");
static_assert(kKvBitBudgetTiers[7].bits_x100 ==
                  e8_kv_pair_bits_x100(E8KvPlaneFormat::B2, E8KvPlaneFormat::B4),
              "F1231 item 1: rk2v4's 325 is the (K at 2 bits, V still i4) PAIR -- and note it "
              "equals e8_kv_pair_bits_x100(B3,B3), which is why the two rows can be confused");
// (4) THE DISAGREEMENT ITSELF, PINNED ON BOTH SIDES: the ladder's rk3v4 (375, pair) versus the
// engine's `CellMode::Rk3v4` (325, symmetric cell). If a later round converts one of them, it must
// delete this assertion deliberately rather than discover the mixing again.
static_assert(cell_rung(CellMode::Rk3v4).plane_bytes * 100 / 2048 == 325 &&
                  kKvBitBudgetTiers[6].bits_x100 == 375 &&
                  cell_rung(CellMode::Rk3v4).plane_bytes * 100 / 2048 !=
                      kKvBitBudgetTiers[6].bits_x100,
              "F1231 item 1: `rk3v4` is THE ONE ROW where the ladder's pair ruler and the "
              "descent's symmetric-cell ruler disagree (375 vs 325). This assertion is the "
              "mixing's extent, as a fact: seven rows are priced by one geometry, this one by two");
// (5) THE DESCENT'S REACHABLE PRICE SET, SO THE CENSUS CANNOT DRIFT SILENTLY: six `CellMode`s, and
// the ladder's bits column intersects them at exactly five rows (bf16/int8/nvfp4/rk4v4 and rk3v4
// only through its SYMMETRIC price, which the ladder does not use). `fp8`, `iso4e` and `rk2v4` have
// no `CellMode` at all -- two of them are ladder rows, which is why the ladder and the descent have
// different admissible sets and a solve must not read one as the other.
static_assert(cell_all_modes().count() == 6,
              "F1231: the descent's admissible set is exactly its six CellModes; the ladder has "
              "eight rows, so three of them (fp8, iso4e, rk2v4) are outside the descent's space");

// [F1229, line `landing`, 2026-09-29] THE iso4e PENALTY IS 30, NOT 200. The 200 was pinned as
// "a naming/policy choice, not a measured quality claim" -- the ladder's own sentence, and it was
// true: its only work was to make the bit-equal partner of nvfp4 unreachable through
// kv_gear_solve_preferring ("prefer iso4e at 4.50 returns nvfp4: THE PIN IS A PENALTY"). 30 is
// also what BOTH measured columns in this file already carry (kv_bit_budget_measured_scores:
// iso4e == nvfp4 == (30,114)), so this edit makes the default column agree with the measured ones
// instead of disagreeing with them by a policy margin.
//
// MEASURED TWO-SIDED BEFORE LANDING, because the tree held two contradictory claims about it.
//   * NO SHIPPED PLAN MOVES. dl/landing/probe/iso_pin_probe.cpp solves 61 ceilings (3.75..18.75
//     in 0.25 steps) x 2 rk4v4 windows (8, the pinned test's argument, and 16, this file's own
//     kKvBitBudgetE8LayerLimit) x 7 entries: kv_bit_budget_solve, kv_bit_budget_solve_audited
//     (cold pool on), kv_bit_budget_solve_scored at w=1 over the default table and over BOTH
//     measured tables, kv_gear_solve_default, and kv_gear_solve_preferring("iso4e"). 854 solves
//     per side; the diff of before-vs-after is 18 lines, ALL of them gear_prefer_iso4e solves,
//     and every one of the six shipped entries is byte-identical at every ceiling and both
//     windows. In particular kv_bit_budget_solve(16, 4.50, 8, 0) still returns `0-15:nvfp4` at
//     penalty 4.80 -- the contract tests/test_kv_budget_saturation.cpp:382 pins -- and
//     kv_gear_solve_default's own answers are unmoved too.
//   * SO THIS IS A STRICT WIDENING: it unblocks the preference axis and moves nothing else.
//
// WARNING: iso4e AND nvfp4 ARE NOT QUALITY-EQUAL, and this number should not be read as claiming
// they are. Measured on ninfer-perplexity (binary 54588b93..., 13,318-token zh corpus, ctx 4096 /
// stride 2048, resolution 0.00000): all-nvfp4 mean_nll 0.39090, all-iso4e 0.41330, so iso4e costs
// +0.02240 nats. In this ladder's own unit that is not zero: the ladder prices 0 -> 30 as the
// measured bf16-vs-nvfp4 step (0.12962 nats, same instrument), so the measured price of iso4e is
// ~30 + 30*(0.02240/0.12962) ~ 35, and 30 is its floor. 30 was landed because this is the DEFAULT
// PRIOR column, both measured columns already carry 30, and 35 would leave the preference
// unreachable (a gear scored STRICTLY worse is never chosen). The delta is dated and reproducible
// with `--kv-layer-storage all:iso4e` vs `all:nvfp4`; moving the two integers 30 -> 35 is the
// honest-price variant and is the OWNER'S call, not a comment repair.
//
// RENAME-INVARIANCE PIN (dl/isoname, iso3 -> iso4e). The rename is a NAME change: it may
// touch a spec_name and nothing else. The bit column is what the slider's whole axis
// rests on -- nvfp4 and iso4e are the ONLY two rows that cost the same bits, and that
// equality is why "different KIND at the same bit width" is even expressible -- so the
// one number a rename must never move is pinned here as a literal, in the idiom of
// product/kv_storage_dtype.h:200-209. The runtime half of this pin is
// tests/test_kv_bit_mix_menu.cpp (17 codec-different mixes at 450*16 == 7200, both
// selections |d(achieved_bits)| < 1e-12). Moving either number is a BITS change, not a
// rename, and needs the slider re-measured.
static_assert(kKvBitBudgetTiers[3].bits_x100 == 450 &&
              kKvBitBudgetTiers[5].bits_x100 == 450,
              "nvfp4 (3) and iso4e (5) must both cost 450 = 4.5000 b/element: they are the "
              "only equal-bit pair in the candidate grammar, and the iso3->iso4e rename "
              "moves no bits");

// Cold pseudo-tier. It deliberately lives OUTSIDE kKvBitBudgetTiers so engine consumers of
// the hot tier table are untouched. It is kKvBitBudgetTiers.size() and is asserted to be
// so below: the two rk4v4 width rows were appended at 6 and 7, which is exactly the index the
// cold pseudo-tier used to occupy, so this constant HAD to move with them.
inline constexpr std::int32_t kKvBitBudgetColdTierIndex = 8;
// One cold slot, in bytes, per (page, kv_head, K|V plane). DERIVED, not spelled: see the
// binding note above. The record is the rANS slot at the ceiling decoder_state.cpp fixes
// (kColdSlotRansBitsPerCodeX100 = 404, the MEASURED minimum), i.e.
// 320 B header + 32 streams x ceil(512 x 4.04 / 8) = 259 B + a 1024 B scale tail = 9632 B.
// It is NOT the int8 raw record (9232 B): a cold LAYER is charged the record its own
// dtype's codec allocates (kv_tier_formats.h kv_cold_class_bytes_of). This constant is the
// DEFAULT grid point the ladder prices cold at, because the cold pseudo-tier is a single
// grid point and the DP state has no per-layer axis to charge two of them from.
//
// ⭐ THE PRICE THIS PRICES, stated where the number is, because the number is a COST and
// the ladder's cold option is only worth taking where a codec PAYS. At 9632 B the cold
// rANS slot is a net device-memory COST on every class that can reach it (nvfp4/iso4e
// +416 B/head-page, rk4v4 +928); the cold records that really save are the RAW 9232 B slot
// on int8/fp8/bf16 (-7664 / -7776 / -23136). The previous 6688 B grid point was cheaper
// only because the codec it described produced NOTHING (measured: 0 of 64 valid K slots at
// a 167 B per-stream budget), and a slot that never validates has no price.
inline constexpr std::int32_t kKvBitBudgetColdSlotBytes = kKvColdPoolStrideBytes;
// Elements covered by one head-page of the slot codec. The 1024 B uncompressed scale tail holds
// one E4M3 scale per 16-channel group of a 64-token row, so it describes exactly 16384 KV
// elements, and at the 4-bit no-expansion ceiling the code plane matches it with 16384 4-bit
// codes = 8192 B. Both counts are properties of the PLANE GEOMETRY: the rANS ceiling only
// shrinks the code payload (to 32 x 167 B at 2.60 b/c), never the element count, so one cold
// slot still replaces 16384 KV elements - the same element count the 9216 B nvfp4 head-page
// covers, which is what puts the cold cost on the ladder's own per-element scale.
inline constexpr std::int32_t kKvBitBudgetElementsPerHeadPage = 16384;

// ---------------------------------------------------------------------------
// The cold tier is NOT free - the correction of the original model, which charged a cold
// layer "0.00 hot bits/element" on the theory that "its GPU bit-budget footprint is freed".
// That theory only ever described the RESIDENT page pool; the slot pool a cold page moves
// into is device memory as well:
//   * plan_decoder_state() adds one U8 [slot_bytes, kv_heads, 2, max_cold_pages] tensor plus
//     an I32 validity plane PER FULL-ATTENTION LAYER to the same LayoutBuilder that holds the
//     resident page pool (decoder_state.cpp:232-249, builder.add_tensor);
//   * effective_cold_pages() sends ColdPolicy::None to 0 pages with the comment "slots would
//     be dead device memory" (layouts_impl.h:124-155), i.e. the engine itself treats a slot as
//     resident device bytes;
//   * the slot codec doc says the head-page group "can be returned to the pool while the page
//     is cold" (entropy_nvfp4_slot.h:29-31): the RESIDENT page is recycled, the SLOT is
//     allocated.
// So a cold layer is charged its slot geometry in the SAME unit the ladder uses (bits per KV
// element, K+V averaged):
//   kKvBitBudgetColdSlotBytes * 8 / kKvBitBudgetElementsPerHeadPage = 4.703125
// and that is what it must be compared against, per head-page, for each hot tier:
//   tier   bits/el  resident B  cold B  delta B   delta   verdict
//   bf16    16.00      32768     9632   -23136   -70.6%  cold saves (raw 9232 B record)
//   int8     8.25      16896     9632    -7264   -43.0%  cold saves (the int8 LAYER's own
//                                                        raw record is 9232 B, so the plan
//                                                        really saves 7664 (-45.4%))
//   fp8      8.50      17408     9632    -7776   -44.7%  cold saves (the tier RUNS; [RK4V4-CONTROL 2026-09-18 vs PATCHSET/RK4V4-CONTROL/REPORT.md])
//   nvfp4    4.50       9216     9632     +416    +4.5%  cold COSTS
//   rk4v4    4.25       8704     9632     +928   +10.7%  cold COSTS
//   iso4e    4.50       9216     9632     +416    +4.5%  cold COSTS
// ⭐ The three COST rows are not a regression, they are the measurement: the K plane's rANS
// streams need 259 B each (dl/ransceil/rans_probe.cu, 2048 real K streams), so the record
// that ENCODES is 9632 B, and 9632 > 9216 > 8704 means it cannot also SAVE. The record that
// would save (6688, or the 4-bit 9536) is one the encoder does not fill: 0 of 64 and 61 of
// 64 valid K slots respectively. Encode-and-save is unsatisfiable for every class that can
// reach this codec; the cold tier's real lever is the RAW slot on int8/fp8/bf16.
// The cold column is not a free parameter: it is the rANS record the measured 4.04 bits/code
// ceiling produces (decoder_state.cpp kColdSlotRansBitsPerCodeX100 = 404, pinned against the
// measurement). Break-even ceilings: 3.84 b/c for the nvfp4 plane pair (9216 B), 3.59 b/c
// for rk4v4's 8704 B plane - both BELOW the measured requirement, which is why no ceiling
// both encodes and saves.
// resident B == bits_x100 * kKvBitBudgetElementsPerHeadPage / 800, i.e. the very plane
// geometry the bits_x100 column was derived from, so the two can never drift apart.
//
// Consequences, all deliberate:
//   * achieved_bits is a true per-element device footprint for ANY plan, cold included: a
//     caller can no longer be handed a cold plan that silently costs more bytes than it says.
//   * the DP still returns the minimum-penalty plan inside the budget, but it now PRICES cold
//     instead of subsidising it. With the cold grid point at 327 the ladder reads
//     (rk4v4 425, nvfp4 450, iso4e 450, cold 327), so cold is now CHEAPER per element than every
//     tier it can cache for, and a cold-using plan is a byte decrease for the first time.
//     Before the ceiling was tightened the grid point was 466 and every cold-using plan was a
//     byte increase - that is the same fact as the old "+320 B/head-page" verdict, priced on
//     the ladder's own scale.
//   * kv_bit_budget_solve_audited() additionally compares the cold plan against the SAME
//     budget solved with the cold pool OFF, in real head-page bytes, and refuses the cold plan
//     when it is a net increase; the raw numbers stay in the result (cold_net_head_page_bytes,
//     cold_report) so a caller can print or override them.
//   * cold_cap == 0 (ColdPolicy::None, or --max-cold-pages 0) makes ALL of the above
//     unreachable: the cold candidate is enumerated only under `cold_cap > 0`, so the cold-off
//     allocation is bit-identical to the pre-fix header.
inline constexpr std::int32_t kKvBitBudgetColdPenaltyX100 = 25;  // 0.25 restore prior

namespace detail {

// 9632 * 8 * 100 / 16384 == 470.3125, the 0.01-bit grid point 470 (.3125 is not a tie, so there is
// no tie-to-even subtlety here) -- that IS kKvBitBudgetColdBitsX100, which the static_assert under
// its definition below pins. It was 326.5625 -> 327 while the rANS record was 6688, a record the
// codec no longer produces. The grid point is derived from a STRIDE rather than from one literal
// because the pool does not allocate a single record width: an int8 layer reserves 9232 B and an
// nvfp4 layer the kKvColdPoolStrideBytes rANS record (see kKvBitBudgetColdSlotBytesInt8Raw below),
// so the ladder's single cold grid point is a choice about which of the two it prices at, not a
// property of the geometry.
[[nodiscard]] constexpr std::int32_t cold_bits_x100_from_stride(std::int32_t slot_bytes) noexcept {
    const std::int64_t scaled = static_cast<std::int64_t>(slot_bytes) * 8 * 100;
    return static_cast<std::int32_t>((scaled + kKvBitBudgetElementsPerHeadPage / 2) /
                                     kKvBitBudgetElementsPerHeadPage);
}

[[nodiscard]] constexpr std::int32_t cold_bits_x100_from_slot_geometry() noexcept {
    return cold_bits_x100_from_stride(kKvBitBudgetColdSlotBytes);
}

} // namespace detail

// Per-element device cost of a cold layer, on the ladder's own scale. Derived, never
// hand-written: change the slot geometry and this follows.
inline constexpr std::int32_t kKvBitBudgetColdBitsX100 =
    detail::cold_bits_x100_from_slot_geometry();
// Tripwire: if this fires, the slot geometry changed and the cold-vs-resident table above (and
// every recorded cold verification value) has to be re-derived deliberately.
static_assert(kKvBitBudgetColdBitsX100 == 470,
              "cold slot geometry changed: 9632 B over 16384 elements must be 4.703125 b/el. "
              "This is derived from kKvColdPoolStrideBytes, so if it moved, the cold record "
              "moved in product/kv_tier_formats.h and every recorded cold verification value "
              "has to be re-derived deliberately");

// The SECOND record the pool can allocate, and the one kKvBitBudgetColdSlotBytes is not. The
// cold pool sizes each layer's tensor from that layer's own resolved dtype
// (decoder_state.cpp cold_slot_stride_for), so an int8 layer reserves 9232 B
// (ops::kColdI8SlotBytes, mirrored host-only as kv_tier_formats.h kKvColdInt8PayloadBytes)
// while an nvfp4 layer reserves the kKvColdPoolStrideBytes (9632 B) rANS record the grid point
// above prices. 9232 is NARROWER than 9632, so a single cold grid point OVER-states an int8 layer
// by 400 B/head-page. It UNDER-stated it by 2544 while the rANS record was 6688, which is why the
// static_assert below states the sign of this difference as diagnostic rather than as a number;
// the cold DP state has no per-layer axis to charge the two apart, so the datum is named here
// instead of living in a comment, and detail::cold_bits_x100_from_stride() is the other half of
// what a per-layer charge would be built from. Meanwhile kv_bit_budget_solve_audited() is the
// check that catches the difference in real head-page bytes.
inline constexpr std::int32_t kKvBitBudgetColdSlotBytesInt8Raw = 9232;
static_assert(kKvBitBudgetColdSlotBytesInt8Raw == 9232,
              "int8 raw cold record == ops::kColdI8SlotBytes");
static_assert(detail::cold_bits_x100_from_stride(kKvBitBudgetColdSlotBytesInt8Raw) == 451,
              "9232 B over 16384 elements is 4.5078125 b/el, the 0.01-bit grid point 451");
static_assert(kKvBitBudgetColdSlotBytesInt8Raw - kKvBitBudgetColdSlotBytes == -400,
              "the int8 record is 400 B/head-page NARROWER than the one the grid point "
              "prices. The sign of this difference is diagnostic: while the rANS record was "
              "6688 the single cold grid point UNDER-stated an int8 layer by 2544 B; at the "
              "measured record (9632) it OVER-states it by 400. It is not zero at any record "
              "the codec can produce, which is exactly why kv_cold_class_bytes_of() prices "
              "each class against its OWN record and this constant is only the DP's default "
              "grid point");

// Resident plane bytes per (page, kv_head, K|V plane) implied by a bits_x100 cost:
// kv_bit_budget_plane_bytes(450) == 9216, (425) == 8704, (825) == 16896, (850) == 17408,
// (1600) == 32768 - the numbers the cold table above is built from.
[[nodiscard]] constexpr std::int32_t kv_bit_budget_plane_bytes(std::int32_t bits_x100) noexcept {
    return static_cast<std::int32_t>((static_cast<std::int64_t>(bits_x100) *
                                      kKvBitBudgetElementsPerHeadPage) / 800);
}

// THE COLD GRID POINT'S SHORTFALL, PINNED AS A NUMBER -- and the per-record half of the
// charge, DERIVED. This block replaced an earlier draft of its own author's that claimed
// `kv_bit_budget_plane_bytes_ceil(kKvBitBudgetColdBitsX100) == kKvBitBudgetColdSlotBytes`;
// that assertion FIRED on the real tree, which is the point: a 0.01-bit grid has a step of
// kKvBitBudgetElementsPerHeadPage/800 = 20.48 B, and the shipped 9,632 B record sits BETWEEN
// 470 -> 9,625 B and 471 -> 9,646 B. So no grid point at this granularity is exact for the
// record, and the DP's cold charge is 7 B/head-page BELOW the pool it prices -- the residual
// under-price left after the grid point was derived from the measured record (before that
// derivation it described the retired 2.60 b/c record, 6,688 B, and under-priced the same pool
// by 4.7862 GiB at 1M: dl/pagemath/REPORT.md section 2, dl/fallbackbytes/REPORT.md:622-623).
static_assert(kKvBitBudgetColdSlotBytes -
                      kv_bit_budget_plane_bytes(kKvBitBudgetColdBitsX100) ==
                  7,
              "the cold grid point's byte price moved relative to the record it prices: at the "
              "shipped pair the grid point is 9,625 B and the arena allocates 9,632 B, i.e. the "
              "DP's cold charge is 7 B/head-page below the pool (14,140,672 B = 0.013 GiB at "
              "1M). Re-derive kKvBitBudgetColdBitsX100 from kKvColdPoolStrideBytes and this "
              "shortfall together, and re-run the 1M arithmetic in dl/pagemath/REPORT.md. A "
              "consumer that spends POOL BYTES must take them from kKvBitBudgetColdSlotBytes "
              "-- which is what the audit at kv_bit_budget_head_page_bytes does -- or from "
              "kv_bit_budget_cold_bits_x100_covering(), NEVER from "
              "kv_bit_budget_plane_bytes(kKvBitBudgetColdBitsX100)");

// The PER-RECORD half of the cold charge: the covering 0.01-bit price of a byte record, for a
// caller that knows which record the pool will allocate for a layer (decoder_state.cpp
// cold_slot_stride_for: 9,232 B for int8/fp8/bf16/rk4v4, 9,632 B for nvfp4). The DP's single
// grid point cannot price both -- its state has no per-layer axis (the cold branch of
// kv_bit_budget_solve_impl adds ONE kKvBitBudgetColdBitsX100 per cold layer) -- so this is the
// other half a per-layer charge would be built from. DERIVED from the record, never spelled,
// and rounded UP so a bits price can never come out narrower than the record it prices.
[[nodiscard]] constexpr std::int32_t
kv_bit_budget_cold_bits_x100_covering(std::int32_t record_bytes) noexcept {
    const std::int64_t scaled = static_cast<std::int64_t>(record_bytes) * 8 * 100;
    return static_cast<std::int32_t>((scaled + kKvBitBudgetElementsPerHeadPage - 1) /
                                     kKvBitBudgetElementsPerHeadPage);
}
static_assert(kv_bit_budget_cold_bits_x100_covering(kKvBitBudgetColdSlotBytes) == 471,
              "the covering price of the shipped rANS record is ceil(9632 * 800 / 16384) = "
              "ceil(470.3125) = 471, one step above the round-to-nearest 470 the DP charges");
static_assert(kv_bit_budget_cold_bits_x100_covering(kKvBitBudgetColdSlotBytesInt8Raw) == 451,
              "the covering price of the int8 raw record is ceil(9232 * 800 / 16384) = "
              "ceil(450.78) = 451");
static_assert(kv_bit_budget_plane_bytes(
                  kv_bit_budget_cold_bits_x100_covering(kKvBitBudgetColdSlotBytes)) >=
                  kKvBitBudgetColdSlotBytes,
              "the covering price of the rANS record must not fall below 9,632 B");
static_assert(kv_bit_budget_plane_bytes(
                  kv_bit_budget_cold_bits_x100_covering(kKvBitBudgetColdSlotBytesInt8Raw)) >=
                  kKvBitBudgetColdSlotBytesInt8Raw,
              "the covering price of the int8 raw record must not fall below 9,232 B");

// Resident plane bytes of ladder row `tier_index`.
[[nodiscard]] constexpr std::int32_t
kv_bit_budget_tier_plane_bytes(std::size_t tier_index) noexcept {
    return kv_bit_budget_plane_bytes(kKvBitBudgetTiers[tier_index].bits_x100);
}

// Net device bytes per head-page of one cold slot against that tier's resident plane:
// < 0 = the cold slot is smaller (a real saving), > 0 = the cold slot is BIGGER, i.e. putting
// a layer of that tier on cold would INCREASE device memory.
[[nodiscard]] constexpr std::int32_t
kv_bit_budget_cold_delta_bytes(std::int32_t bits_x100) noexcept {
    return kKvBitBudgetColdSlotBytes - kv_bit_budget_plane_bytes(bits_x100);
}

[[nodiscard]] constexpr std::int32_t
kv_bit_budget_tier_cold_delta_bytes(std::size_t tier_index) noexcept {
    return kv_bit_budget_cold_delta_bytes(kKvBitBudgetTiers[tier_index].bits_x100);
}

[[nodiscard]] constexpr bool kv_bit_budget_cold_saves_bytes(std::int32_t bits_x100) noexcept {
    return kv_bit_budget_cold_delta_bytes(bits_x100) < 0;
}

[[nodiscard]] constexpr bool
kv_bit_budget_tier_cold_saves_bytes(std::size_t tier_index) noexcept {
    return kv_bit_budget_tier_cold_delta_bytes(tier_index) < 0;
}

// Packing order: rk4v4 first so the verified leading layers receive it (the DP only counts
// tiers, the packing decides which layer gets which).
// ⚠ THE TWO NARROW K-PLANE ROWS MUST BE IN THIS LIST (line `redkv`, dl/redkv/TRIAGE.md #53/#52).
// This is a POSITIONAL table for the legacy multiset solver (kv_bit_budget_solve_impl, reached
// whenever cold_cap > 0): it decides a COUNT per tier and the packing here places the blocks. A
// tier the DP may choose but this list does not name is a tier whose count is emitted NOWHERE, so
// the spec silently covers fewer layers than the plan has. That is exactly what happened when
// rk3v4/rk2v4 became selectable (dl/e8mixwire/land/patch_flip_batch.py): measured at 16 layers,
// cold_cap=6, request 4.30, the solver returned `0-5:rk4v4,6-9:cold,10-13:nvfp4` -- FOURTEEN
// layers -- and kv_kv_bits_entry_joint refused it with "the allocator returned a spec that does
// not cover layer 14". 45 of the 101 entry-agreement requests disagreed this way.
// PLACED BESIDE rk4v4, not appended at the end: the 4-bit row is first because its leading window
// is the one that was measured (`0-7:rk4v4`), and a 3-bit/2-bit rk4v4 is the SAME family -- the
// exposure limit at kv_bit_budget.h:730-738 is explicitly a family count -- so the family stays
// contiguous in the shallow block. Appending them last would have put the weakest codecs on the
// deepest layers, which is a positional policy claim this line has no measurement for.
inline constexpr std::array<const char*, 8> kKvBitBudgetPackOrder{
    {"rk4v4", "rk3v4", "rk2v4", "bf16", "int8", "fp8", "nvfp4", "iso4e"}};

// Cold-mode packing: rk4v4 STILL first (keeps its verified leading window), then its two narrower
// siblings, then cold on the next-shallow block; deep layers keep the hot tail.
inline constexpr std::array<const char*, 9> kKvBitBudgetPackOrderCold{
    {"rk4v4", "rk3v4", "rk2v4", "cold", "bf16", "int8", "fp8", "nvfp4", "iso4e"}};

namespace detail {

// Deprecated spellings, accepted for ONE release and resolved to the canonical row.
// Before this rename the row was called `e8` (never the E8 lattice -- it is one signed
// integer code per element over a shared scale plane, src/product/kv_e8_width.h:17-19) and
// the shipped iso row was called `iso3` (never 3 bits -- it is the 4-bit ISO code over an
// E4M3FN scale plane, 4.00 + 8/16 = 4.50 b/el == 9216 B/head-page == the ladder's 450).
// Resolving them here, and not at each call site, keeps the deferred front ends
// (apps/cli/options.cpp, options.h, main.cpp -- another live line) working while every
// EMITTED name is canonical. Delete when those files are renamed.
[[nodiscard]] inline std::string_view canonical_tier_name(std::string_view name) noexcept {
    if (name == "e8")   { return "rk4v4"; }
    if (name == "e8k3") { return "rk3v4"; }
    if (name == "e8k2") { return "rk2v4"; }
    if (name == "iso3") { return "iso4e"; }
    return name;
}

[[nodiscard]] inline std::int32_t tier_index(std::string_view raw) noexcept {
    const std::string_view name = canonical_tier_name(raw);
    for (std::size_t i = 0; i < kKvBitBudgetTiers.size(); ++i) {
        if (name == kKvBitBudgetTiers[i].spec_name) { return static_cast<std::int32_t>(i); }
    }
    return -1;
}

[[nodiscard]] inline std::int32_t pack_index(std::string_view raw) noexcept {
    const std::string_view name = canonical_tier_name(raw);
    for (std::size_t i = 0; i < kKvBitBudgetPackOrder.size(); ++i) {
        if (name == kKvBitBudgetPackOrder[i]) { return static_cast<std::int32_t>(i); }
    }
    return -1;
}

// Index inside kKvBitBudgetTiers for a pack-order name ("cold" -> the cold pseudo-tier).
[[nodiscard]] inline std::int32_t pack_tier_index(std::string_view name) noexcept {
    if (name == "cold") { return kKvBitBudgetColdTierIndex; }
    return tier_index(name);
}

} // namespace detail

struct KvBitBudgetSolution {
    std::string spec;            // --kv-layer-storage grammar ("cold" marks cold-planned slots)
    double achieved_bits = 0.0;  // per-element DEVICE cost of the whole plan: hot tiers at
                                 // their plane geometry, cold layers at
                                 // kKvBitBudgetColdBitsX100 (their real slot bytes, not 0)
    double penalty = 0.0;        // total quality penalty (exact x100 sum / 100)
    // SATURATION ACCOUNTING (BUDGETSAT). `requested_bits` is the ceiling the caller asked
    // for, ON THIS SAME per-element scale; `shortfall_bits` = requested - achieved, >= 0 by
    // construction because no plan may exceed the ceiling. The solver FILLS the ceiling from
    // below -- it maximises `achieved_bits` subject to `achieved_bits <= requested_bits` and
    // only then minimises `penalty` -- so a nonzero shortfall is a property of the LADDER's
    // granularity (the smallest step that still fits under the ceiling), NEVER of the
    // objective trading bits away. A caller must PRINT it: a silently undershooting request
    // is precisely the defect this field exists to make visible.
    double requested_bits = 0.0;
    double shortfall_bits = 0.0;
    std::map<std::string, std::int32_t> counts;  // tier -> layer count, may hold "cold"
    // Cold accounting. All zero for a cold-free plan, which is what keeps the cold-off path
    // byte-identical to the pre-fix header.
    std::int32_t cold_layers = 0;      // layers placed in the cold pool
    std::int32_t cold_bits_x100 = 0;   // per-element cost charged per cold layer (0 if none)
    // Real head-page bytes (K+V planes, all layers, per page per kv_head) of a cold plan minus
    // the same number for the SAME budget solved with the cold pool OFF: > 0 means the cold
    // plan really does take more device memory than the best all-hot plan. Filled by
    // kv_bit_budget_solve_audited(); when that call REFUSES a net increase this field and
    // cold_report describe the rejected cold plan, not the returned one.
    std::int64_t cold_net_head_page_bytes = 0;
    std::string cold_report;  // one line, empty unless cold was used and audited
};

// Real device footprint of a plan: bytes per (page, kv_head), K+V planes, summed over every
// full-attention layer. Cold layers pay kKvBitBudgetColdSlotBytes, hot layers the resident
// plane geometry of their tier - the same numbers the DP charged. `ladder` must be the ladder
// the solution was solved with.
[[nodiscard]] inline std::int64_t kv_bit_budget_head_page_bytes(
    const KvBitBudgetSolution& solution, const std::array<KvBitBudgetTier, 8>& ladder) {
    std::int64_t total = 0;
    for (std::size_t i = 0; i < ladder.size(); ++i) {
        const auto it = solution.counts.find(ladder[i].spec_name);
        if (it == solution.counts.end()) { continue; }
        total += 2LL * static_cast<std::int64_t>(it->second) *
                 kv_bit_budget_plane_bytes(ladder[i].bits_x100);
    }
    const auto cold = solution.counts.find("cold");
    if (cold != solution.counts.end()) {
        total += 2LL * static_cast<std::int64_t>(cold->second) * kKvBitBudgetColdSlotBytes;
    }
    return total;
}

// ===========================================================================
// THE DEPLOYABILITY CENSUS -- DERIVED FROM THE DEPLOY LAYER'S OWN DECISION POINT
// ===========================================================================
// `KvBitBudgetTier::selectable` answers "may the DP CHOOSE this row". It does not answer
// "can the engine RUN the row it chose", and on this tree the two answers differ for
// exactly two rows:
//
//   product/kv_storage_dtype.h:107-123 REFUSES KvCacheStorage::E8K3Group64 / E8K2Group64
//   BY NAME -- "'rk3v4' is a DEFINED rk4v4 tier whose codec, K plate layout and WRITER all
//   exist but which NO RUNTIME PATH CAN READ" -- and
//   src/targets/qwen3_6/impl/runtime/layouts_impl.h:681-684 refuses the same two rows one
//   step later. So a plan the SOLVER calls `deployed` can be a plan the BUILD layer throws
//   on.
//
// MEASURED on this tree (dl/kdslider/logs/21_probe_v2_baseline.txt, section D): every
// `--kv-bits` landing at 4.25 / 4.80 / 5.00 / 6.00 -- and the 4.50 default -- carried
// `rk2v4` or `rk3v4` and was refused by the build, while the solver answered
// `deployed=1 refused=0`. The 4.50 default was
// `0-2:rk2v4,3-4:rk4v4,5:rk3v4,6-7:rk2v4,8-9:int8,10-15:nvfp4`.
//
// WHY THIS IS A CALL AND NOT A LIST: a hand-written set of "rows the engine can read" is a
// SECOND spelling of a question product/kv_storage_dtype.h already answers, and this
// project's worst outcome is a knob (or a gate) accepted and read by nothing.
// `parse_kv_storage` (product/kv_options.h:46-61) is the ladder's own name->storage
// spelling -- the same vocabulary the --kv-layer-storage parser uses -- and
// `kv_dtype_for_storage` (product/kv_storage_dtype.h:45) is the deploy layer's ONE decision
// point. So this census cannot drift from the set of rows an operator can actually run: the
// batch that wires a 3-bit/2-bit K-plate reader deletes the throw THERE, and these two rows
// become admissible here with no edit to this file.
[[nodiscard]] inline const std::array<bool, 8>& kv_bit_budget_deployable_rows() {
    static const std::array<bool, 8> rows = [] {
        std::array<bool, 8> out{};
        for (std::size_t i = 0; i < kKvBitBudgetTiers.size() && i < out.size(); ++i) {
            const std::optional<KvCacheStorage> storage =
                parse_kv_storage(kKvBitBudgetTiers[i].spec_name);
            if (!storage) { continue; }   // no storage for this row: not deployable
            try {
                (void)kv_dtype_for_storage(*storage, "kv-bit-budget");
                out[i] = true;
            } catch (const std::invalid_argument&) {
                out[i] = false;
            }
        }
        return out;
    }();
    return rows;
}

// The deployable rows by name, in ladder order.
[[nodiscard]] inline std::string kv_bit_budget_deployable_list() {
    const std::array<bool, 8>& ok = kv_bit_budget_deployable_rows();
    std::string out;
    for (std::size_t i = 0; i < kKvBitBudgetTiers.size() && i < ok.size(); ++i) {
        if (!ok[i]) { continue; }
        if (!out.empty()) { out += ", "; }
        out += kKvBitBudgetTiers[i].spec_name;
    }
    return out;
}

// The rows the census withholds, in the ladder's own spelling.
[[nodiscard]] inline std::string kv_bit_budget_withheld_rows() {
    const std::array<bool, 8>& ok = kv_bit_budget_deployable_rows();
    std::string out;
    for (std::size_t i = 0; i < kKvBitBudgetTiers.size() && i < ok.size(); ++i) {
        if (ok[i]) { continue; }
        if (!out.empty()) { out += ", "; }
        out += kKvBitBudgetTiers[i].spec_name;
    }
    return out;
}

// `cold` -> `nvfp4`: THE ROW A COLD-PLANNED LAYER'S HOT WINDOW ACTUALLY RESOLVES TO.
// A cold layer is planned as the NVFP4 hot window and never reaches the deploy layer under
// the `cold` token -- product::parse_kv_storage has no `cold` row, and the target plan
// resolves the cold ranges separately (layouts_impl.h: "cold-planned layers keep a HOT
// window at NVFP4"; product/kv_kv_bits.h detail::kv_bits_hot_window_tier is that same fact
// as one function). Walking the RAW spec therefore declared a deployable cold plan NOT
// deployable -- MEASURED on the first version of this line
// (dl/kvreach/logs/50_after_test_kv_budget_saturation.txt: "entry spellings agree on the
// spec at ...: (empty) vs 0-5:cold,6-15:fp8"). The rewrite is what makes the walk see the
// plan the POOL will have rather than the plan's shorthand.
[[nodiscard]] inline std::string kv_bit_budget_hot_window_spec(std::string_view spec) {
    std::string out;
    std::size_t begin = 0;
    while (begin <= spec.size()) {
        const std::size_t comma = spec.find(',', begin);
        const std::string_view item =
            spec.substr(begin, comma == std::string_view::npos ? spec.size() - begin
                                                               : comma - begin);
        const std::size_t colon = item.rfind(':');
        const std::string_view type =
            colon == std::string_view::npos ? item : item.substr(colon + 1);
        if (!out.empty()) { out += ','; }
        if (type == "cold") {
            out += std::string(colon == std::string_view::npos ? std::string_view{}
                                                               : item.substr(0, colon + 1));
            out += "nvfp4";
        } else {
            out += std::string(item);
        }
        if (comma == std::string_view::npos) { break; }
        begin = comma + 1;
    }
    return out;
}

// WHY THIS WALK EXISTS SEPARATELY FROM THE CENSUS. The census answers about a CLASS of
// rows; this answers about a PLAN -- the one the solver is about to hand over -- using the
// same two deploy-layer calls the target plan uses on the same grammar
// (product/kv_options.h:72). So the counter in product/kv_kv_bits.h cannot disagree with
// what the build does, and it names the FIRST refused layer rather than a row set.
[[nodiscard]] inline bool kv_bit_budget_spec_is_deployable(std::string_view spec,
                                                           std::string* why = nullptr,
                                                           std::int32_t* bad_layer = nullptr) {
    if (spec.empty()) { return false; }
    const std::string hot = kv_bit_budget_hot_window_spec(spec);
    KvLayerStorageSpec parsed;
    try {
        parsed = parse_kv_layer_storage_spec(hot);
    } catch (const std::exception& e) {
        if (why != nullptr) { *why = e.what(); }
        return false;
    }
    for (std::size_t i = 0; i < parsed.set.size(); ++i) {
        if (!parsed.set[i]) { continue; }
        try {
            (void)kv_dtype_for_storage(parsed.table[i], "--kv-bits[" + std::to_string(i) + "]");
        } catch (const std::exception& e) {
            if (why != nullptr) { *why = e.what(); }
            if (bad_layer != nullptr) { *bad_layer = static_cast<std::int32_t>(i); }
            return false;
        }
    }
    return true;
}

// Full solution. Structurally IDENTICAL to the Python DP (tools/archkit/kv_bit_budget.py)
// kvreach-k3-census
// so the two agree byte-for-byte, including tie-breaks:
//   * state key = (bits, cold_used); the rk4v4 count is carried in the surviving path's
//     value (NOT an extra dimension), exactly like the Python counts dict;
//   * states are expanded in first-insertion order and candidates in the ladder order
//     ORDER + ["cold"]; an equal-penalty candidate never displaces an inserted one;
//   * the final winner is the minimum-penalty state in first-insertion order.
// cold_cap > 0 enables the cold pseudo-tier; cold_cap == 0 keeps the candidate set on
// the pre-cold code path (cold_states == 1 makes the cold index inert).
// The DP body takes the ladder as a parameter: the default ladder carries the shipped
// per-tier penalties, while the scored variant below hands in a ladder whose penalties are
// the weighted sum of a PRIOR quality column and a measured speed column. spec_name is
// never reinterpreted, so every name-keyed path (packing, --kv-layer-storage emission)
// works unchanged.
[[nodiscard]] inline KvBitBudgetSolution kv_bit_budget_solve_impl(
    std::int32_t layers, double budget_bits, const std::array<KvBitBudgetTier, 8>& ladder,
    std::int32_t rk4v4_limit = kKvBitBudgetE8LayerLimit, std::int32_t cold_cap = 0,
    std::int32_t cold_bits_x100 = kKvBitBudgetColdBitsX100) {
    if (layers <= 0) { throw std::invalid_argument("kv-bit-budget: layer count must be positive"); }
    if (!(budget_bits > 0.0)) {
        throw std::invalid_argument("kv-bit-budget: bit budget must be positive");
    }
    if (rk4v4_limit < 0) { rk4v4_limit = 0; }
    if (rk4v4_limit > layers) { rk4v4_limit = layers; }
    if (cold_cap < 0) { cold_cap = 0; }
    if (cold_cap > layers) { cold_cap = layers; }

    // Rounding must reproduce Python's round() = banker's (round-half-to-EVEN), found
    // the hard way by C's S18 adversarial grid (_collab/C_s14_verify.md): budgets that
    // are k/8 with odd k (4.125, 5.625, 6.625, ...) make budget*100 land on a
    // binary-exact .5 (4.125*100 == 412.5 exactly), where the previous half-up
    // `(int)(x + 0.5)` gave 413 while Python round(412.5) = 412 — a capacity shift of
    // `layers` 0.01-bit units and 8 divergent allocations at cold >= 8, layers >= 48.
    // std::nearbyint honours the default FE_TONEAREST mode = ties-to-even (matches
    // Python on all spot values: 412.5->412, 413.5->414, 562.5->562).
    const std::int32_t budget_x100 = static_cast<std::int32_t>(
        std::nearbyint(budget_bits * kKvBitBudgetScale));
    const std::int32_t capacity = budget_x100 * layers;   // total 0.01-bit units available
    const std::int32_t cold_states = cold_cap + 1;        // 1 when cold is off
    const std::int32_t span = capacity + 1;
    const std::size_t states = static_cast<std::size_t>(span) * static_cast<std::size_t>(cold_states);

    const double kInf = 1.0e18;
    std::vector<double> current(states, kInf);
    std::vector<double> next(states, kInf);
    // Surviving path's rk4v4 usage per state (the Python tool carries it inside `counts`).
    std::vector<std::int8_t> rk4v4_count(states, 0);
    std::vector<std::int8_t> next_rk4v4_count(states, 0);
    // choice[layer][state] = tier chosen for the (layer+1)-th layer when the state was
    // reached (kKvBitBudgetColdTierIndex marks the cold pseudo-tier).
    std::vector<std::vector<std::int8_t>> choice(static_cast<std::size_t>(layers));
    for (auto& layer_choice : choice) { layer_choice.assign(states, -1); }
    // States in first-insertion order (mirrors Python dict iteration order).
    std::vector<std::size_t> order_cur{0};
    std::vector<std::size_t> order_next;
    order_next.reserve(states);

    current[0] = 0.0;
    for (std::int32_t layer = 0; layer < layers; ++layer) {
        std::fill(next.begin(), next.end(), kInf);
        std::fill(next_rk4v4_count.begin(), next_rk4v4_count.end(), 0);
        order_next.clear();
        auto& layer_choice = choice[static_cast<std::size_t>(layer)];
        for (const std::size_t from : order_cur) {
            const double base = current[from];
            if (base >= kInf) { continue; }
            const std::int32_t from_rk4v4 = rk4v4_count[from];
            const std::int32_t from_bits = static_cast<std::int32_t>(from / cold_states);
            const std::int32_t from_cold = static_cast<std::int32_t>(from % cold_states);
            for (std::size_t t = 0; t < ladder.size(); ++t) {
                const KvBitBudgetTier& tier = ladder[t];
                // THE REACHABILITY GATE, read here rather than only priced above: a row
                // that is not selectable is skipped outright, so no ceiling -- not even
                // one low enough that the DP has no other allocation -- can put the
                // engine on a codec whose K plate it cannot read.
                if (!tier.selectable) { continue; }
                // The whole rk4v4 FAMILY, not row 4 alone. The rk4v4 exposure limit exists to
                // bound how many layers may use the weakest codec in the set; a 3-bit or
                // 2-bit rk4v4 is strictly weaker than the 4-bit one, so counting only row 4
                // would let a plan spend its rk4v4 budget and then add unbounded rk2v4 layers
                // on top. COEXISTENCE ITEM: this makes rk4v4_limit cover all three rows.
                const bool is_rk4v4 = static_cast<std::int32_t>(t) == 4 ||
                                   static_cast<std::int32_t>(t) == 6 ||
                                   static_cast<std::int32_t>(t) == 7;
                if (is_rk4v4 && from_rk4v4 >= rk4v4_limit) { continue; }
                const std::int32_t new_bits = from_bits + tier.bits_x100;
                if (new_bits > capacity) { continue; }
                const std::size_t to = static_cast<std::size_t>(new_bits) * cold_states + from_cold;
                // Accumulate in the Python float domain (x100/100.0 == the Python literal),
                // in path order: the tool's comparisons and printed penalties are float
                // sums, and near-ties resolve identically only if the arithmetic matches.
                const double value = base + tier.penalty_x100 / 100.0;
                if (next[to] >= kInf) {
                    next[to] = value;
                    next_rk4v4_count[to] = static_cast<std::int8_t>(from_rk4v4 + (is_rk4v4 ? 1 : 0));
                    layer_choice[to] = static_cast<std::int8_t>(t);
                    order_next.push_back(to);
                } else if (value < next[to]) {
                    next[to] = value;
                    next_rk4v4_count[to] = static_cast<std::int8_t>(from_rk4v4 + (is_rk4v4 ? 1 : 0));
                    layer_choice[to] = static_cast<std::int8_t>(t);
                }
            }
            const std::int32_t cold_bits = from_bits + cold_bits_x100;
            if (cold_cap > 0 && from_cold < cold_cap && cold_bits <= capacity) {
                // Cold pseudo-tier (candidate LAST, after the ladder): it pays the REAL slot
                // geometry (kKvBitBudgetColdBitsX100 per element) instead of the old "0 hot
                // bits" - the slot pool is device memory, see the geometry block above. The
                // branch is not entered at all when cold_cap == 0, so the cold-off path is
                // untouched.
                const std::size_t to = static_cast<std::size_t>(cold_bits) * cold_states +
                                       (from_cold + 1);
                const double value = base + kKvBitBudgetColdPenaltyX100 / 100.0;
                if (next[to] >= kInf) {
                    next[to] = value;
                    next_rk4v4_count[to] = static_cast<std::int8_t>(from_rk4v4);
                    layer_choice[to] = static_cast<std::int8_t>(kKvBitBudgetColdTierIndex);
                    order_next.push_back(to);
                } else if (value < next[to]) {
                    next[to] = value;
                    next_rk4v4_count[to] = static_cast<std::int8_t>(from_rk4v4);
                    layer_choice[to] = static_cast<std::int8_t>(kKvBitBudgetColdTierIndex);
                }
            }
        }
        current.swap(next);
        rk4v4_count.swap(next_rk4v4_count);
        order_cur.swap(order_next);
    }

    // SATURATION (BUDGETSAT). The winner is the state that USES THE MOST BITS while staying
    // inside the ceiling; the penalty only breaks ties among equal-bit states. The previous
    // rule here was "the cheapest reachable final state" (minimum penalty alone), which made
    // the ceiling a ONE-SIDED constraint the objective was free to undershoot: measured at
    // the engine, `--kv-bits 4.5` returned achieved=4.38 while all-nvfp4 at exactly 4.50 was
    // feasible, and a 4.60 ceiling returned 4.38 as well.
    //   * NO OVERSHOOT is structural, not checked here: every state transition above is
    //     guarded by `new_bits <= capacity`, so no state at or past the ceiling exists.
    //   * MONOTONICITY of request -> achieved follows from the rule rather than from luck:
    //     a larger ceiling admits a superset of plans, so its maximum is at least as large.
    //   * Ties inside one bit count keep the FIRST-INSERTED state (order_cur is in
    //     first-insertion order), so the answer stays deterministic.
    std::size_t best = states;
    std::int32_t best_final_bits = -1;
    double best_penalty = kInf;
    for (const std::size_t state : order_cur) {
        if (current[state] >= kInf) { continue; }
        const std::int32_t state_bits = static_cast<std::int32_t>(state / cold_states);
        if (state_bits > best_final_bits ||
            (state_bits == best_final_bits && current[state] < best_penalty)) {
            best_final_bits = state_bits;
            best_penalty = current[state];
            best = state;
        }
    }
    if (best == states) {
        throw std::invalid_argument("kv-bit-budget: no feasible allocation for this budget");
    }
    const std::int32_t best_bits = static_cast<std::int32_t>(best / cold_states);
    const std::int32_t best_cold = static_cast<std::int32_t>(best % cold_states);

    // Backtrack to per-layer tier indices (rk4v4 is not part of the state walk; the guard
    // was applied forward, mirroring the Python counts-carrying value).
    std::vector<std::int32_t> per_layer(static_cast<std::size_t>(layers), -1);
    std::int32_t bits = best_bits;
    std::int32_t cold_used = best_cold;
    for (std::int32_t layer = layers - 1; layer >= 0; --layer) {
        const auto& layer_choice = choice[static_cast<std::size_t>(layer)];
        const std::size_t at = static_cast<std::size_t>(bits) * cold_states + cold_used;
        const std::int32_t tier = layer_choice[at];
        if (tier < 0) { throw std::logic_error("kv-bit-budget: broken DP backtrack"); }
        per_layer[static_cast<std::size_t>(layer)] = tier;
        if (tier == kKvBitBudgetColdTierIndex) {
            bits -= cold_bits_x100;  // the same real slot cost the forward pass charged
            --cold_used;
        } else {
            bits -= ladder[static_cast<std::size_t>(tier)].bits_x100;
        }
    }

    // Counts per tier name (hot names + "cold").
    std::map<std::string, std::int32_t> counts;
    std::int32_t cold_count = 0;
    for (const std::int32_t tier : per_layer) {
        if (tier == kKvBitBudgetColdTierIndex) {
            ++counts["cold"];
            ++cold_count;
        } else {
            ++counts[ladder[static_cast<std::size_t>(tier)].spec_name];
        }
    }

    // Pack: rk4v4 first (cold mode: rk4v4, then cold, then the rest), then emit layer ranges.
    std::vector<const char*> pack_order;
    if (cold_count > 0) {
        pack_order.assign(kKvBitBudgetPackOrderCold.begin(), kKvBitBudgetPackOrderCold.end());
    } else {
        pack_order.assign(kKvBitBudgetPackOrder.begin(), kKvBitBudgetPackOrder.end());
    }
    std::string spec;
    std::int32_t cursor = 0;
    for (const char* pack_name : pack_order) {
        const std::int32_t tier = detail::pack_tier_index(pack_name);
        if (tier < 0) { continue; }
        std::int32_t count = 0;
        if (tier == kKvBitBudgetColdTierIndex) {
            count = cold_count;
        } else {
            const auto it = counts.find(pack_name);
            count = it == counts.end() ? 0 : it->second;
        }
        if (count == 0) { continue; }
        const std::int32_t begin = cursor;
        const std::int32_t end = cursor + count - 1;
        cursor += count;
        if (!spec.empty()) { spec += ","; }
        spec += std::to_string(begin);
        if (end != begin) { spec += "-" + std::to_string(end); }
        spec += ":";
        spec += pack_name;
    }

    KvBitBudgetSolution solution;
    solution.spec = std::move(spec);
    solution.achieved_bits = static_cast<double>(best_bits) / (layers * kKvBitBudgetScale);
    solution.penalty = best_penalty;  // already the Python float-domain accumulated sum
    solution.counts = std::move(counts);
    solution.cold_layers = cold_count;
    solution.cold_bits_x100 = cold_count == 0 ? 0 : cold_bits_x100;
    solution.requested_bits = static_cast<double>(budget_x100) / kKvBitBudgetScale;
    solution.shortfall_bits = solution.requested_bits - solution.achieved_bits;
    if (solution.shortfall_bits < 0.0) { solution.shortfall_bits = 0.0; }
    return solution;
}

// Returns the --kv-layer-storage spec covering `layers` full-attention layers.
//
// DECLARED here, DEFINED with the gearbox below (search "D6"). The definition
// moved there so that this entry -- the one --kv-bits / --kv-bit-budget and
// product/kv_kv_bits.h both reach -- is served by the LAYER-EXACT solver instead
// of by kv_bit_budget_solve_impl's multiset DP. The two differ in a way that
// matters: solve_impl decides a COUNT per tier and then lets a fixed packing order
// decide which layer gets what, so its rk4v4 layers are always the prefix 0..k-1, and
// it cannot return the shipped table at any budget. kv_bit_budget_solve_impl is
// deliberately left in place and unchanged -- it is the documented legacy path, it
// is what the cold-audit entry below still uses, and its cold_cap == 0 output is
// what the cold-mode byte-identity notes are written against.
[[nodiscard]] inline KvBitBudgetSolution kv_bit_budget_solve(
    std::int32_t layers, double budget_bits,
    std::int32_t rk4v4_limit = kKvBitBudgetE8LayerLimit, std::int32_t cold_cap = 0,
    std::int32_t cold_bits_x100 = kKvBitBudgetColdBitsX100,
    // SLIDERWIRE: the caller's candidate PREFERENCE -- the ladder slots to try FIRST among
    // plans the objective already rates equal. DEFAULTED AND APPENDED, so every existing
    // caller (including the pinned tools/archkit probe and every engine call site) compiles
    // and behaves exactly as before, and an empty list reproduces the shipped pack order
    // line for line.
    const std::vector<std::int32_t>& candidate_order = {});

// kv_bit_budget_solve() plus a REAL-BYTE audit of the cold usage, for the reason documented
// above the geometry block: a cold layer's slot is device memory and, on a sub-int8 stack, it
// is BIGGER than the resident plane it replaces, so "cold is free" used to let the DP recommend
// a net device-memory increase without saying so.
//
// When cold_cap > 0 and the solver used cold, the same budget is solved once more with the cold
// pool disabled and the two real footprints are compared (per page per kv_head, K+V, all
// layers). If the cold plan is a net increase and allow_net_increase is false, the cold-off
// plan is returned instead and cold_report says so: the DP never recommends a configuration
// that grows device memory unless it is told to. The refused cold plan's numbers
// (cold_net_head_page_bytes, cold_report) are still reported, so the trade-off the operator
// gave up - a lower restore prior for more bytes - stays visible. With allow_net_increase =
// true the cold plan is returned with the increase stated. Cold-free plans and cold plans that
// save bytes come back untouched, with a report line.
[[nodiscard]] inline KvBitBudgetSolution kv_bit_budget_solve_impl_audited(
    std::int32_t layers, double budget_bits, const std::array<KvBitBudgetTier, 8>& ladder,
    std::int32_t rk4v4_limit = kKvBitBudgetE8LayerLimit, std::int32_t cold_cap = 0,
    std::int32_t cold_bits_x100 = kKvBitBudgetColdBitsX100,
    bool allow_net_increase = false) {
    KvBitBudgetSolution cold_plan =
        kv_bit_budget_solve_impl(layers, budget_bits, ladder, rk4v4_limit, cold_cap, cold_bits_x100);
    if (cold_plan.cold_layers == 0) { return cold_plan; }  // nothing to audit
    // The cold-off baseline can be absent in principle (only with a caller-supplied ladder
    // whose every row costs more than a cold slot): then there is no all-hot plan to compare
    // against and cold is what makes the budget fit, so the plan stands with the reason stated
    // instead of a fabricated delta.
    KvBitBudgetSolution hot_plan;
    bool hot_feasible = true;
    try {
        hot_plan = kv_bit_budget_solve_impl(layers, budget_bits, ladder, rk4v4_limit, 0,
                                            cold_bits_x100);
    } catch (const std::invalid_argument&) {
        hot_feasible = false;
    }
    const std::int64_t cold_bytes = kv_bit_budget_head_page_bytes(cold_plan, ladder);
    std::ostringstream line;
    line.precision(1);
    line << std::fixed;
    line << "--kv-bit-budget cold: " << cold_plan.cold_layers << " layer(s) x "
         << kKvBitBudgetColdSlotBytes << " B/head-page (" << cold_bits_x100
         << " x100 b/element, real slot geometry); plan " << cold_bytes << " B/head-page";
    if (!hot_feasible) {
        line << "; there is no cold-off plan at this budget, so the cold plan is the only "
                "feasible one and its net increase is unmeasurable - cold is what makes it fit "
                "at all";
        cold_plan.cold_report = line.str();
        return cold_plan;
    }
    const std::int64_t hot_bytes = kv_bit_budget_head_page_bytes(hot_plan, ladder);
    const std::int64_t delta     = cold_bytes - hot_bytes;
    cold_plan.cold_net_head_page_bytes = delta;
    const double pct = hot_bytes == 0
                           ? 0.0
                           : 100.0 * static_cast<double>(delta) / static_cast<double>(hot_bytes);
    line << " vs same-budget cold-off plan " << hot_bytes << " B/head-page -> "
         << (delta > 0 ? "+" : "") << delta << " B (" << (pct > 0.0 ? "+" : "") << pct << "%)";
    if (delta > 0) {
        line << " NET DEVICE INCREASE";
        line << (allow_net_increase
                     ? " (allowed by the caller: cold buys a lower restore prior, and the "
                       "increase is this line)"
                     : " -> refused, returning the cold-off plan");
        line << "; the tiers whose cold slot is BIGGER than their resident plane are "
                "nvfp4/iso4e (+416 B) and rk4v4 (+928 B), while int8 (-7264 B on its own "
                "9232 B record: -7664 B), fp8 (-7776 B) and bf16 (-23136 B) really save";
    } else {
        line << " (cold saves device bytes)";
    }
    cold_plan.cold_report = line.str();
    if (delta <= 0 || allow_net_increase) { return cold_plan; }
    KvBitBudgetSolution refused_winner = hot_plan;
    refused_winner.cold_net_head_page_bytes = delta;  // the REJECTED cold plan's numbers
    refused_winner.cold_report = cold_plan.cold_report;
    return refused_winner;
}

[[nodiscard]] inline KvBitBudgetSolution kv_bit_budget_solve_audited(
    std::int32_t layers, double budget_bits,
    std::int32_t rk4v4_limit = kKvBitBudgetE8LayerLimit, std::int32_t cold_cap = 0,
    std::int32_t cold_bits_x100 = kKvBitBudgetColdBitsX100,
    bool allow_net_increase = false) {
    return kv_bit_budget_solve_impl_audited(layers, budget_bits, kKvBitBudgetTiers, rk4v4_limit,
                                            cold_cap, cold_bits_x100, allow_net_increase);
}

[[nodiscard]] inline std::string kv_bit_budget_spec(std::int32_t layers, double budget_bits,
                                                    std::int32_t rk4v4_limit =
                                                        kKvBitBudgetE8LayerLimit,
                                                    std::int32_t cold_cap = 0) {
    return kv_bit_budget_solve(layers, budget_bits, rk4v4_limit, cold_cap).spec;
}

// ---------------------------------------------------------------------------
// Separable per-range bit ceilings ("分开约束"): the user may cap different layer
// ranges independently, e.g. "0-7:8,8-15:4.5" (leading layers 8 bits, the rest 4.5): the ranges must tile every FULL-ATTENTION layer, so the example follows the variant (16 here), NOT a total layer count.
//
// Optimality: the objective is the sum of per-layer penalties and the constraints are
// per-range capacities, so the feasible set is the Cartesian product of the ranges'
// feasible sets and minimising each range independently attains the global optimum.
// Implementation is therefore the single-budget DP above, run once per range, with the
// emitted layer indices shifted into absolute coordinates. The rk4v4 leading-window limit
// is applied on ABSOLUTE layer indices, and the cold pool is shared: ranges are solved
// deepest-first (the pack order already sends cold to the deep block) and each range may
// only spend what the pool has left, so the total cold layers can never exceed cold_cap.
// ---------------------------------------------------------------------------
// Two-score selection: speed and quality are separate columns -- the speed column is
// measured, the quality column is a prior -- and the operator
// chooses the trade-off with a weight. Both columns are "lower is better, x100" and must be
// normalised onto ONE scale by the caller (quality = relative output error x100, speed =
// relative time cost x100 with the fastest tier at 0), because the weight only means
// anything if the two are commensurable.
struct KvTierScoreRow {
    std::int32_t quality_x100 = 0;   // precision loss (0 = lossless); a PRIOR unless the caller measured it
    std::int32_t speed_x100   = 0;   // measured time cost    (0 = fastest available path)
};
using KvTierScoreTable = std::array<KvTierScoreRow, 8>;   // same order as the default ladder
// GREW 6 -> 8 WITH THE GATE (dl/e8mixwire/land/patch_flip_batch.py, one item). The paragraph below
// used to explain why the table could stay at six while the ladder had eight: the two appended
// rows were unselectable, so no ladder-bounded loop could reach them. The gate flip retires that
// reason, and scores_cover_every_selectable_gear() (:1048) makes the consequence explicit --
// the type must grow in the same edit or the header does not compile.

// THE INVARIANT THAT MAKES THE SCORE TABLE SAFE TO INDEX WITH A LADDER BOUND.
// The table describes the SCOREABLE gears, and it is a PREFIX of the ladder. The ladder
// grew from six rows to eight when rk3v4/rk2v4 were appended; the table did not, so any
// loop bounded by `ladder.size()` while indexing `scores[i]` reads two rows past the end.
// Measured: `ninfer --kv-score-table show` segfaulted (rc=139) on its SUCCESS path for
// exactly this reason, and -D_GLIBCXX_ASSERTIONS reports
//   std::array<KvTierScoreRow, 6>::operator[]: Assertion '__n < this->size()' failed.
// (The compiler had already noticed: "iteration 6 invokes undefined behavior
// [-Waggressive-loop-optimizations]" at kv_bit_budget_scored_ladder's bound.)
// The rows the table does not cover must therefore be gears the DP cannot choose; if a
// selectable row is ever appended past the table, this fails to compile instead of
// reading past the end.
namespace detail {

// [dl/backlog item8-guard] THIS LOOP CANNOT RUN, AND THAT IS THE DEFECT.
// State the population: KvTierScoreTable is std::array<KvTierScoreRow, 8> (:1184) and
// kKvBitBudgetTiers is std::array<KvBitBudgetTier, 8> (:281), so the loop below was
// `for (i = 8; i < 8; ++i)` -- ZERO iterations, and the function returned `true` without
// evaluating a single row. It was NOT vacuous when it was written: the table was six rows
// and the ladder eight, and the two appended rows were the unselectable pair. The gate
// flip made the table eight rows and the predicate became unconditional.
// The honest replacement is the PRECONDITION the loop relied on -- table and ladder are
// the same set -- so that a future edit which makes the ladder grow past the table fails
// TO COMPILE here instead of silently restoring the vacuous pass.
[[nodiscard]] constexpr bool scores_cover_every_selectable_gear() noexcept {
    return KvTierScoreTable{}.size() == kKvBitBudgetTiers.size();
}

}  // namespace detail

static_assert(KvTierScoreTable{}.size() <= kKvBitBudgetTiers.size(),
              "the score table must be a prefix of the ladder");
static_assert(detail::scores_cover_every_selectable_gear(),
              "every ladder row the score table does not cover must be unselectable: "
              "otherwise the DP could choose a gear that has no score, and every "
              "ladder-bounded loop indexing the score table would read past its end");

// Combined ladder: penalty = round(w*quality + (1-w)*speed), clamped to >= 0.
[[nodiscard]] inline std::array<KvBitBudgetTier, 8>
kv_bit_budget_scored_ladder(const KvTierScoreTable& scores, double quality_weight) {
    if (!(quality_weight >= 0.0) || !(quality_weight <= 1.0)) {
        throw std::invalid_argument("kv-bit-budget: quality weight must be in [0,1]");
    }
    std::array<KvBitBudgetTier, 8> ladder = kKvBitBudgetTiers;
    // BOUNDED BY THE SCORE TABLE, NOT BY THE LADDER. The two are different sizes (6 score
    // rows, 8 ladder rows) and this loop used to take the ladder's bound while indexing
    // `scores`, i.e. two out-of-bounds reads per call -- on a path reached whenever the
    // slider is used, and fatal on the `--kv-score-table` success path (rc=139).
    // The rows the table does not cover keep their DECLARED penalty from kKvBitBudgetTiers;
    // the static_assert above pins that they are exactly the unselectable gears, so no
    // score can ever be silently missing from a row the DP is allowed to choose.
    for (std::size_t i = 0; i < scores.size(); ++i) {
        const double combined = quality_weight * scores[i].quality_x100 +
                                (1.0 - quality_weight) * scores[i].speed_x100;
        ladder[i].penalty_x100 = static_cast<std::int32_t>(std::nearbyint(combined));
        if (ladder[i].penalty_x100 < 0) { ladder[i].penalty_x100 = 0; }
    }
    // [dl/backlog item8b-loop] ZERO ITERATIONS TODAY (scores.size() == ladder.size() == 8),
    // so this body never runs. Kept because it is correct the moment a score table is
    // SHORTER than the ladder again -- but its old comment claimed the two appended rows
    // were unselectable and that that was what kept the DP off them. Both halves are false
    // now (kKvBitBudgetTiers[6].selectable and [7].selectable are `true`), and the thing
    // that actually keeps the DP off them is their being DOMINATED in the score numbers --
    // which is a property of the table's CONTENTS and not of this guard. See the withheld
    // rows' notes in kv_bit_budget_measured_scores() (dl/backlog item7).
    for (std::size_t i = scores.size(); i < ladder.size(); ++i) {
        ladder[i].penalty_x100 = kKvBitBudgetTiers[i].penalty_x100;
    }
    return ladder;
}

// DECLARED here, DEFINED with the gearbox below (search "D6"), for the same reason
// as kv_bit_budget_solve: the two-score path is the one --kv-quality-weight drives,
// and it must reach the layer-exact candidate space rather than a prefix packing.
[[nodiscard]] inline KvBitBudgetSolution kv_bit_budget_solve_scored(
    std::int32_t layers, double budget_bits, const KvTierScoreTable& scores,
    double quality_weight, std::int32_t rk4v4_limit = kKvBitBudgetE8LayerLimit,
    std::int32_t cold_cap = 0, const std::vector<std::int32_t>& candidate_order = {});

// Provisional default score table. The quality column carries the shipped priors until the
// automated calibration (tools/archkit/kv_tier_matrix.py -> JSON) supplies measured values;
// the speed column starts from the measured uniform-tier decode rates once they exist
// (fastest path = 0, others scaled by their relative per-token time cost). Both columns are
// x100 and normalised onto one scale by whoever builds the table.
// constexpr so detail::default_quality_column_is_the_shipped_penalty_column() can pin it at
// COMPILE time; its body is a single return of a constexpr aggregate, so this is a strict widening.
[[nodiscard]] constexpr KvTierScoreTable kv_bit_budget_default_scores() {
    // speed_x100 is (fastest/v - 1)*100 from the measured uniform-tier rates below, so the
    // fastest path is 0 and slower tiers scale up. quality_x100 still carries the shipped
    // priors until the offline replay (tools/calib) supplies measured per-layer errors.
    return KvTierScoreTable{{
        {/*bf16 */ 0, 7},    // 80.7 tok/s
        {/*int8 */ 2, 0},    // 86.2 tok/s - the measured fastest
        {/*fp8  */ 3, 0},    // speed UNMEASURED, not "does not run". [dl/backlog item1-R1] The 0 is the
                             // field default, not a measurement; the tier is still correctly excluded
                             // by cost (8.50 > int8's 8.25).
                             // (dl/tierclose R1, applied verbatim) The RK4V4-CONTROL 2026-09-18
                             // citation this used to carry (`--kv-layer-storage 0-15:fp8` ->
                             // fp8-e4m3-row256, 4.75 GiB, 27/27) does NOT reproduce on pin
                             // 8c566fba8843b15b3a4632e009d5cd936a6981d8db93f64a498639c3d0bd15 under
                             // examples/cli/messages/long_niah_64k.json: that exact spelling is
                             // refused by the engine's cold-host window guard (layer 0 declares
                             // sliding_window_tokens = 646720 and fp8's decode path does not read the
                             // field), and idem-perplexity refuses --kv-dtype fp8 the same way. The
                             // measurement above stands for the configuration it was taken on; cite
                             // THAT configuration, or cite nothing.
        {/*nvfp4*/ 30, 114}, // 40.3 tok/s: QK runs twice + software V decode
        {/*rk4v4   */ 8, 218},  // 27.1 tok/s: lattice projection + nibble unpack
        {/*iso4e */ 30, 114}, // == nvfp4, and == the ladder row row-for-row (F1229 retired the pin)
        // ⚠ PRIORS, NOT MEASUREMENTS -- the only invented numbers in this batch. Derived from
        // rk4v4's {8, 218} in this table's own idiom: a narrower K plane costs a little
        // precision (quality_x100 up) and a little speed (speed_x100 up), so the two rows are
        // DOMINATED by rk4v4 and can only be CHOSEN when the ceiling leaves no other feasible
        // allocation -- which is exactly the case the gate exists for. Replace these two rows
        // with measured columns before quoting any slider result that depends on the tie-break.
        // ⚠ ALIGNED TO THE LADDER ROW'S OWN DECLARED COLUMN (slider line `kdslider`).
        // These two rows are still PRIORS, not measurements -- but they now carry the SAME number
        // the ladder row declares (12 / 16, this file's kKvBitBudgetTiers above), and that is
        // load bearing rather than cosmetic: the quality column of THIS table then IS the shipped
        // penalty column row for row, which is what product/kv_perlayer_policy.h:52 has asserted
        // since before the e8 gate flip ("BYTE-IDENTICAL ... and --kv-quality-weight 1 is a
        // provable no-op"). The flip appended these two rows with the PRE-flip prior (10 / 13) and
        // broke that identity in silence. detail::default_quality_column_is_the_shipped_penalty_
        // column() below now fails to COMPILE if it breaks again. With the identity restored,
        // w = 1.0 reproduces kKvBitBudgetTiers exactly, so resolving an ABSENT --kv-quality-weight
        // to the quality end changes no plan (measured: 104/104 budgets,
        // dl/kdslider/probe/w_sweep.cpp).
        {/*rk3v4 */ 12, 225},// the ladder row's own declared penalty; PRIOR (narrower K than rk4v4)
        {/*rk2v4 */ 16, 232},// the ladder row's own declared penalty; PRIOR (narrowest K)
    }};
}

// [F1231 item 6] THE DEFAULT TABLE'S HALF OF THE `fp8` DOMINATION PIN, at the one place in this
// file where the table it asserts about exists. `fp8` costs more bits than int8 (850 > 825, pinned
// at the ladder) and is never better on either score column here; the blend is convex, so at every
// `--kv-quality-weight` int8's combined penalty is <= fp8's and the bits axis decides the ties --
// against fp8. The two MEASURED tables carry `int8 (49,0) / fp8 (49,0)` and `(2,0) / (3,0)`, both
// verified by dl/rework/arms/tier_column_probe.cpp rather than by this comment, because a runtime
// table cannot be asserted at compile time.
static_assert(kv_bit_budget_default_scores()[1].quality_x100 <=
                      kv_bit_budget_default_scores()[2].quality_x100 &&
                  kv_bit_budget_default_scores()[1].speed_x100 <=
                      kv_bit_budget_default_scores()[2].speed_x100,
              "F1231 item 6: on the DEFAULT score table int8 must not be worse than fp8 on either "
              "axis, or some weight could choose fp8 -- re-read the domination block at the "
              "ladder's definition, and re-run dl/rework/arms/tier_column_probe.cpp");

// ===========================================================================
// THE SLIDER'S DEFAULT: AN ABSENT --kv-quality-weight IS THE QUALITY END
// ===========================================================================
// The owner's rule, verbatim: the slider exists so that at the SAME bit count one allocates
// different KINDS of quantisation, not fewer bits -- THE BIT COUNT IS AN INPUT. The knob that
// acts on it is --kv-quality-weight, whose two ends are the two readings of the columns:
//
//   w = 1.0  -- the QUALITY end: penalty = the quality column. THE SHIPPED ANSWER.
//   w = 0.0  -- the SPEED   end: penalty = the speed  column. The fastest answer.
//
// ABSENCE USED TO MEAN "do not run the table at all", i.e. the shipped single-penalty ladder.
// That made the scored table a trick the operator had to know, and it is why every refusal in
// this tree has to say "add --kv-quality-weight 0". It now means THE QUALITY END, and that is NOT
// a behaviour change -- a reason this header can state and then CHECK, rather than assert:
//
//   * both engine entries funnel into the SAME kv_gear_solve(request) and differ ONLY in
//     request.ladder: kv_bit_budget_solve passes kKvBitBudgetTiers, kv_bit_budget_solve_scored
//     passes kv_bit_budget_scored_ladder(scores, w) (this file's two entries, at the bottom);
//   * at w = 1.0 the combined penalty is round(1*quality + 0*speed), i.e. the quality column;
//   * so the two entries agree at w = 1.0 IFF this table's quality column IS kKvBitBudgetTiers'
//     penalty column -- asserted at COMPILE time just below.
//
// Measured, not argued: dl/kdslider/probe/w_sweep.cpp section A compares the two entries over 104
// budgets (3.05 .. 9.05 step 0.05) on BOTH columns, and reports 88 of them differing BEFORE the
// alignment and 0 AFTER. Those 88 are the RED CONTROL: a comparison that cannot fail proves
// nothing.
inline constexpr double kKvQualityWeightQualityEnd = 1.0;
// The sentinel stays -1.0, so "was the flag NAMED?" -- the ceiling requirement in
// apps/cli/options.cpp, src/serve/serve_options.cpp and layouts_impl.h -- keeps meaning what it
// means: an ABSENT flag does not demand a ceiling, an EXPLICIT one does.
inline constexpr double kKvQualityWeightAbsent = -1.0;

// The one place the sentinel is given a meaning. Every entry that chooses between the shipped
// ladder and the scored table calls this instead of comparing to 0.0 itself.
[[nodiscard]] inline double kv_quality_weight_resolved(double weight) noexcept {
    return weight < 0.0 ? kKvQualityWeightQualityEnd : weight;
}

namespace detail {

// THE PIN. `false` means an absent weight and w = 1.0 would resolve DIFFERENT plans, i.e. that
// making the scored table the default WOULD move every existing --kv-bits run. That is the one
// thing this change was required not to do, so it is a compile error rather than a note.
[[nodiscard]] constexpr bool default_quality_column_is_the_shipped_penalty_column() noexcept {
    const KvTierScoreTable scores = kv_bit_budget_default_scores();
    for (std::size_t i = 0; i < scores.size(); ++i) {
        if (scores[i].quality_x100 != kKvBitBudgetTiers[i].penalty_x100) { return false; }
    }
    return true;
}

}  // namespace detail
static_assert(detail::default_quality_column_is_the_shipped_penalty_column(),
              "the DEFAULT score table's quality column must equal kKvBitBudgetTiers' penalty "
              "column row for row: --kv-quality-weight's ABSENCE resolves to the quality end "
              "(kKvQualityWeightQualityEnd), so any other pair of columns would silently change "
              "the default plan of every --kv-bits / --kv-bit-budget run");

// "tier quality_x100 speed_x100" per line, '#' comments; must name all EIGHT tiers.
[[nodiscard]] inline KvTierScoreTable kv_bit_budget_parse_scores(std::string_view text) {
    KvTierScoreTable table = kv_bit_budget_default_scores();
    std::array<bool, 8> seen{};
    std::size_t cursor = 0;
    while (cursor < text.size()) {
        const std::size_t eol = text.find('\n', cursor);
        std::string line(text.substr(cursor, eol == std::string_view::npos ? text.size() - cursor
                                                                          : eol - cursor));
        cursor = eol == std::string_view::npos ? text.size() : eol + 1;
        const std::size_t hash = line.find('#');
        if (hash != std::string::npos) { line.resize(hash); }
        std::istringstream stream(line);
        std::string name;
        double quality = 0.0;
        double speed   = 0.0;
        if (!(stream >> name >> quality >> speed)) { continue; }
        const std::int32_t index = detail::tier_index(name);
        if (index < 0) {
            throw std::invalid_argument("kv-tier-scores: unknown tier '" + name + "'");
        }
        // [dl/backlog item6-consumer] A REFUSAL MUST NOT ARRIVE AS A MEASURED ZERO. The
        // producer (tools/archkit/kv_tier_matrix.py) writes `# UNMEASURED ... -1` for a
        // column it has no arm for, and this parser STRIPS the comment (line.find('#') then
        // resize, above), so the sentinel is the only channel left -- and the range check
        // below used to report it as a generic range error. Name it instead, and refuse the
        // row rather than accepting it: a tier the producer could not classify has no score,
        // and the caller's alternative is to omit the tier, which the missing-tier loop at
        // the end of this function also refuses. Measured population of the old shape: 2 of
        // 6 emitted rows in --quick on pin 8c566fba (fp8, rk4v4), each emitted as `0 0`.
        if (quality < 0.0 || speed < 0.0) {
            throw std::invalid_argument(
                "kv-tier-scores: tier '" + name + "' is UNMEASURED (sentinel -1), not a "
                "score of -1: 'nobody looked' and 'the error is zero' are different values, "
                "and this loader refuses the first rather than loading it as a zero");
        }
        if (quality > 10000.0 || speed > 10000.0) {
            throw std::invalid_argument("kv-tier-scores: scores must be in [0,10000] x100");
        }
        table[static_cast<std::size_t>(index)] = KvTierScoreRow{
            static_cast<std::int32_t>(std::nearbyint(quality)),
            static_cast<std::int32_t>(std::nearbyint(speed))};
        seen[static_cast<std::size_t>(index)] = true;
    }
    for (std::size_t i = 0; i < seen.size(); ++i) {
        if (!seen[i]) {
            throw std::invalid_argument(std::string("kv-tier-scores: missing tier '") +
                                        kKvBitBudgetTiers[i].spec_name + "'");
        }
    }
    return table;
}

struct KvBitBudgetRange {
    std::int32_t first = 0;   // inclusive, absolute layer index
    std::int32_t last = 0;    // inclusive, absolute layer index
    double bits = 0.0;        // ceiling for the average bits/element inside this range
};

// "N" (single ceiling for every layer) or "lo-hi:bits[,lo-hi:bits...]".
// Ranges must tile [0, layers) in order; gaps/overlaps/out-of-order are rejected.
[[nodiscard]] inline std::vector<KvBitBudgetRange>
kv_bit_budget_parse_ranges(std::string_view text, std::int32_t layers) {
    std::vector<KvBitBudgetRange> ranges;
    const auto parse_number = [&](std::string_view piece, const char* what) {
        try {
            std::size_t used = 0;
            const double value = std::stod(std::string(piece), &used);
            if (used != piece.size()) { throw std::invalid_argument("trailing characters"); }
            return value;
        } catch (const std::exception&) {
            throw std::invalid_argument(std::string("kv-bit-budget: invalid ") + what + ": " +
                                        std::string(piece));
        }
    };
    bool has_colon = text.find(':') != std::string_view::npos;
    if (!has_colon) {
        if (text.empty()) { throw std::invalid_argument("kv-bit-budget: empty specification"); }
        ranges.push_back(KvBitBudgetRange{0, layers - 1, parse_number(text, "budget")});
        return ranges;
    }
    std::size_t cursor = 0;
    while (cursor <= text.size()) {
        const std::size_t comma = text.find(',', cursor);
        const std::string_view item =
            text.substr(cursor, comma == std::string_view::npos ? text.size() - cursor
                                                                : comma - cursor);
        if (item.empty()) { throw std::invalid_argument("kv-bit-budget: empty range item"); }
        const std::size_t colon = item.find(':');
        if (colon == std::string_view::npos) {
            throw std::invalid_argument("kv-bit-budget: range item needs 'lo-hi:bits': " +
                                        std::string(item));
        }
        const std::string_view layers_part = item.substr(0, colon);
        const std::string_view bits_part   = item.substr(colon + 1);
        const std::size_t dash             = layers_part.find('-');
        KvBitBudgetRange range;
        if (dash == std::string_view::npos) {
            const double only = parse_number(layers_part, "layer index");
            range.first = range.last = static_cast<std::int32_t>(only);
        } else {
            range.first = static_cast<std::int32_t>(parse_number(layers_part.substr(0, dash),
                                                                "layer index"));
            range.last = static_cast<std::int32_t>(parse_number(layers_part.substr(dash + 1),
                                                               "layer index"));
        }
        range.bits = parse_number(bits_part, "budget");
        ranges.push_back(range);
        if (comma == std::string_view::npos) { break; }
        cursor = comma + 1;
    }
    std::int32_t expect = 0;
    for (const KvBitBudgetRange& range : ranges) {
        if (range.first != expect || range.last < range.first || range.last >= layers) {
            throw std::invalid_argument(
                "kv-bit-budget: ranges must tile layers 0.." + std::to_string(layers - 1) +
                " in order (expected to start at " + std::to_string(expect) + ")");
        }
        if (!(range.bits > 0.0)) {
            throw std::invalid_argument("kv-bit-budget: range budget must be positive");
        }
        expect = range.last + 1;
    }
    if (expect != layers) {
        throw std::invalid_argument("kv-bit-budget: ranges must cover every layer (last covered " +
                                    std::to_string(expect - 1) + " of " +
                                    std::to_string(layers - 1) + ")");
    }
    return ranges;
}

// Shifts the layer indices of a single-range spec by `offset` (grammar: "lo-hi:tier,...").
[[nodiscard]] inline std::string kv_bit_budget_shift_spec(std::string_view spec,
                                                         std::int32_t offset) {
    if (offset == 0) { return std::string(spec); }
    std::string out;
    std::size_t cursor = 0;
    while (cursor < spec.size()) {
        const std::size_t comma = spec.find(',', cursor);
        const std::string_view item =
            spec.substr(cursor, comma == std::string_view::npos ? spec.size() - cursor
                                                               : comma - cursor);
        const std::size_t colon = item.find(':');
        const std::string_view layers_part = item.substr(0, colon);
        const std::string_view tier_part   = item.substr(colon + 1);
        const std::size_t dash             = layers_part.find('-');
        const std::int32_t lo = static_cast<std::int32_t>(
            std::stol(std::string(layers_part.substr(0, dash == std::string_view::npos
                                                            ? layers_part.size()
                                                            : dash)))) + offset;
        std::string shifted = std::to_string(lo);
        if (dash != std::string_view::npos) {
            const std::int32_t hi = static_cast<std::int32_t>(
                std::stol(std::string(layers_part.substr(dash + 1)))) + offset;
            shifted += "-" + std::to_string(hi);
        }
        shifted += ":";
        shifted += std::string(tier_part);
        if (!out.empty()) { out += ","; }
        out += shifted;
        if (comma == std::string_view::npos) { break; }
        cursor = comma + 1;
    }
    return out;
}

// Shared driver for the two range-resolving ladders. The constraint structure (which layers
// share a ceiling) and the ladder (which penalty breaks ties) are orthogonal, so both
// combinations must exist; keeping one driver is what stops them from drifting apart on the
// three things that are easy to get subtly wrong: the tiling validation, the deepest-first
// order that spends the shared cold pool where the pack order puts cold, and the per-range
// rk4v4 window (rk4v4 is a leading-layer window in ABSOLUTE coordinates, so a range that starts
// above the limit gets none).
template <typename Solve>
[[nodiscard]] inline std::string kv_bit_budget_ranges_driver(
    std::int32_t layers, const std::vector<KvBitBudgetRange>& ranges, std::int32_t rk4v4_limit,
    std::int32_t cold_cap, Solve solve) {
    if (ranges.empty()) {
        throw std::invalid_argument("kv-bit-budget: no ranges given");
    }
    std::int32_t expect = 0;
    for (const KvBitBudgetRange& range : ranges) {
        if (range.first != expect || range.last < range.first || range.last >= layers) {
            throw std::invalid_argument("kv-bit-budget: ranges must tile the layer stack");
        }
        expect = range.last + 1;
    }
    if (expect != layers) {
        throw std::invalid_argument("kv-bit-budget: ranges must cover every layer");
    }
    // Deepest range first so the shared cold pool is spent where the pack order puts cold.
    std::vector<std::size_t> order(ranges.size());
    for (std::size_t i = 0; i < ranges.size(); ++i) { order[i] = i; }
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        return ranges[a].first > ranges[b].first;
    });
    std::vector<std::string> pieces(ranges.size());
    std::int32_t cold_left = cold_cap;
    for (const std::size_t index : order) {
        const KvBitBudgetRange& range = ranges[index];
        const std::int32_t count      = range.last - range.first + 1;
        const std::int32_t within     = rk4v4_limit - range.first;
        const std::int32_t local_rk4v4   = within <= 0 ? 0 : (within < count ? within : count);
        const KvBitBudgetSolution solved = solve(count, range.bits, local_rk4v4, cold_left);
        for (const auto& [name, used] : solved.counts) {
            if (name == "cold") { cold_left -= used; }
        }
        pieces[index] = kv_bit_budget_shift_spec(solved.spec, range.first);
    }
    std::string out;
    for (std::size_t i = 0; i < pieces.size(); ++i) {
        if (pieces[i].empty()) { continue; }
        if (!out.empty()) { out += ","; }
        out += pieces[i];
    }
    return out;
}

[[nodiscard]] inline std::string
kv_bit_budget_spec_ranges(std::int32_t layers, const std::vector<KvBitBudgetRange>& ranges,
                          std::int32_t rk4v4_limit = kKvBitBudgetE8LayerLimit,
                          std::int32_t cold_cap = 0) {
    return kv_bit_budget_ranges_driver(
        layers, ranges, rk4v4_limit, cold_cap,
        [](std::int32_t count, double bits, std::int32_t local_rk4v4, std::int32_t cold_left) {
            return kv_bit_budget_solve(count, bits, local_rk4v4, cold_left);
        });
}

// The scored ladder under per-range ceilings. Without this, a caller that passes both knobs
// silently loses the ceilings: the range form leaves the scalar budget at 0, so a scored run
// would resolve every layer against a zero-bit ceiling instead of the range's own.
// F911 -- THE MISSING PARAMETER, ADDED. This function's scored sibling
// (kv_bit_budget_solve_scored) has ALWAYS taken a `candidate_order` as its last parameter; the
// range form simply had none to hand it and hard-called the scalar solver with the shipped
// order, which is why --kv-codec-preference was refused by name under the RANGE spelling of
// --kv-bit-budget. The parameter is DEFAULTED and empty means the shipped order, so every
// existing call site is byte-identical; the operator's preference now reaches the same fit that
// the scalar spelling already reached.
[[nodiscard]] inline std::string kv_bit_budget_scored_ranges(
    std::int32_t layers, const std::vector<KvBitBudgetRange>& ranges,
    const KvTierScoreTable& scores, double quality_weight,
    std::int32_t rk4v4_limit = kKvBitBudgetE8LayerLimit, std::int32_t cold_cap = 0,
    const std::vector<std::int32_t>& candidate_order = {}) {
    return kv_bit_budget_ranges_driver(
        layers, ranges, rk4v4_limit, cold_cap,
        [&scores, quality_weight, &candidate_order](std::int32_t count, double bits,
                                                    std::int32_t local_rk4v4,
                                                    std::int32_t cold_left) {
            return kv_bit_budget_solve_scored(count, bits, scores, quality_weight, local_rk4v4,
                                              cold_left, candidate_order);
        });
}

// ===========================================================================
// D6 -- THE FULL GEARBOX: arbitrary per-layer plans, codec combinations and
// incumbents.
// ===========================================================================
// WHAT WAS MISSING BEFORE THIS SECTION, stated so it can be refuted:
//   * The decision variable of kv_bit_budget_solve_impl is a MULTISET (a count
//     per tier). Layer identity was then assigned by a fixed packing order
//     (kKvBitBudgetPackOrder: rk4v4 first, then bf16/int8/fp8/nvfp4/iso4e), so the
//     layers a tier could occupy were always a contiguous run and the rk4v4 layers
//     were always the leading prefix 0..k-1. Measured on the pre-change header
//     (host probe, 16 layers, every budget on a 0.25-bit grid x every rk4v4_limit
//     in 0..16): the reachable rk4v4 sets are 17 and EVERY ONE IS A PREFIX. That is
//     exactly the D4/B5 note above ("the rk4v4 constraint is a SET, not a COUNT"),
//     and it is why the shipped table could not be reached.
//   * The shipped/factory table is rk4v4 on {0,1,3,4,6,7} -- HOLES at 2 and 5
//     (src/targets/qwen3_6_27b/impl/variant.cpp:42-46). A prefix family cannot
//     name it AT ANY BUDGET, which the probe confirms: an exhaustive sweep of
//     (budget, rk4v4_limit) never returns it. A user who passes ANY --kv-bits
//     therefore could not reach the one configuration that was measured good.
//   * A tier's penalty in the scored path is per-TIER, never per-layer, so the
//     objective had no place for per-layer evidence to live, and the packing was
//     still the thing that decided which layer got what.
//   * iso4e was PINNED at penalty 200 ("pinned out of the candidate set", see the
//     ladder comment) -- and the ladder comment itself says that pin "is a
//     naming/policy choice, not a measured quality claim". In the scored path the
//     pin is overwritten by the caller's columns, so iso4e is scoreable, but the
//     prefix packing still kept it a contiguous block instead of a per-layer mix
//     with nvfp4 -- which is the iso4+nvfp4 combination the caller asked for.
//
// WHAT THIS SECTION ADDS (all additive; nothing below is read unless a caller
// calls it, so every allocation the header already produced is unchanged):
//   * F_l, an ARBITRARY per-layer allowed set of gears (KvGearSet), so holes are
//     representable and the optimality argument is a property of the state key
//     (each layer's constraint is a function of the layer, not a side value
//     carried in the surviving path -- the heuristic kv_adapt_solver.h:33-39
//     names in kv_bit_budget_solve_impl).
//   * A layer-indexed exact DP whose backtrack yields a per-layer gear array,
//     emitted as a run-length spec in ASCENDING layer order. The spec grammar
//     (product/kv_options.h:61) accepts A-B:name and A:name in any order, so an
//     interleaved plan like "0:iso4e,1:nvfp4,2:iso4e,..." is deployable and the
//     engine's single resolution path is untouched.
//   * SEEDS/INCUMBENTS: any plan expressible in --kv-layer-storage grammar (the
//     factory table, a measured winner, an operator's incumbent) is EVALUATED on
//     the same ladder and returned when it is at least as good as the DP answer.
//     This is what makes the shipped table reachable AND selected-when-best
//     instead of merely representable.
//   * COMPONENT (KVarN) MODES: the V-plane codec (--kv-v-codec), the rotation
//     (--kv-rotation) and the row scale (--kv-row-scale) are plan-level switches,
//     so the selector enumerates them and scores each plan with the measured
//     deltas instead of assuming one codec per tier.
//
// MEASURED COLUMNS USED BELOW (every number is cited; none is invented here):
//   * 64k NIAH, uniform tiers, tools/archkit/kv_tier_matrix.py:
//       int8 86.2 > bf16 80.7 > shipped mix 57.1 > nvfp4 40.3 ~ iso4e 40.4 >
//       rk4v4 27.1 tok/s, all-layers-rk4v4 answers 0/27.
//   * retrieval by rk4v4 placement, docs/maintainer/kv-strategy-matrix.md section 2
//     rule 2: 0-7:rk4v4 = 8/8 (twice)  8-15:rk4v4 = 0/8  8-11:rk4v4 = 2/8  12-15:rk4v4 = 0/8
//     14-15:rk4v4 = 4/8  0-11:rk4v4 = 0/8  all:rk4v4 = 0/8. So rk4v4 is measured GOOD on the
//     leading window and measured BROKEN above it; the single all-rk4v4 speed datum
//     (27.1 tok/s, 0/27) is the ALL-LAYERS arm and must not be read as a
//     statement about 0-7:rk4v4.
//   * short context (13.3k zh perplexity, ctx 4096, variant.cpp:26-27): shipped
//     mix 1.020 (best of all mixes), all-rk4v4 1.112, all-i8 1.522, all-nvfp4 1.706.
//   * row scale / rotation (research/notes/TODO.md section "row-scale / kvarn",
//     ninfer-perplexity, noise floor = 0):
//       all-NVFP4 64k   baked 1.40990842  identity 1.40831856  delta -0.00159
//       all-NVFP4 2k    baked 1.50889187  identity 1.50900833  delta +0.00012
//       dense 130k      baked 1.99035283  identity 1.98828477  delta -0.00207
//       all-Rk4v4Kv        2.15060161 == 2.15060161               delta  0.00000
//     Two consequences that the scorer must encode: (a) the row-scale/rotation
//     effect is SECOND ORDER -- 180-490x smaller than a tier choice (0.775 nats),
//     so it may break a tie and must never outrank a tier difference; (b) it is
//     BIT-IDENTICAL on rk4v4 (the negative control), so it must not be applied to
//     the rk4v4 gear at all.
//     WARNING [F1229, line `landing`, 2026-09-29] THE COLUMN ABOVE DOES NOT REPRODUCE AND ITS
//     CONCLUSION IS INVERTED. Re-measured by line `kvarnk` (dl/kvarnk/blob_F1222.md, binary
//     24ef9e0e..., SAME harness, SAME 13,318-token zh corpus, ctx 4096 / stride 2048): base
//     (baked row scale, rotation ON) mean_nll 0.39090 and base_repeat 0.39090, i.e. the
//     instrument's resolution is exactly 0.00000; against that floor the identity table costs
//     +0.34966 nats, the INVERTED table (rs_wrong_rec) +0.44654, rotation OFF +0.23439, and a
//     WHOLE TIER CHANGE IN THE SAME BATTERY (bf16) +0.12962. So the row scale's error is about
//     170x the 0.00207 recorded here, and LARGER than the tier change it is compared against:
//     the sentence "the row-scale/rotation effect is SECOND ORDER -- 180-490x smaller than a
//     tier choice (0.775 nats)" is FALSE on the current binary. The recorded runs are from
//     2026-09-12; the mechanism gained its decode Q-read site (gqa_attention_decode_nvfp4.cuh:647)
//     and its prefill sites since, which is a CANDIDATE explanation and is NOT established.
//     THE CREDIT BELOW IS STILL APPLIED AT 5. It is now known to be a POLICY value resting on a
//     measurement that did not survive re-measurement -- NOT the "Derived, not chosen" value its
//     own comment claims (see kKvRowScaleCreditX100). Re-deriving it is OWED and is NOT done by
//     F1229, because it moves the scored default path and that is the owner's call.
//   * fp8 rotation defect (research/notes/TODO.md section 4): the fp8 decode
//     kernel applies R to K but not to Q, so with rotation ON the fp8 tier
//     computes q^T R k instead of (Rq)^T(Rk) -- measured O(1) wrong (mean 1.97 vs
//     1.0e-07 when both sides rotate). fp8 with rotation ON is therefore
//     INADMISSIBLE, not merely penalised; with rotation OFF it is harmless. This
//     is a measured admissibility gate, and it is the clearest example of why the
//     scorer cannot ignore the KVarN switches.
//
// The factory set, verbatim, so a reader can diff it against variant.cpp.
inline constexpr std::array<std::int32_t, 6> kKvBitBudgetFactoryE8Layers{{0, 1, 3, 4, 6, 7}};

// ---------------------------------------------------------------------------
// F_l -- the per-layer allowed set. The faithful encoding of a constraint that
// is a SET; a COUNT is its lossy special case (KvBitBudgetRk4v4Set::is_prefix).
// ---------------------------------------------------------------------------
inline constexpr std::size_t kKvGearCount = kKvBitBudgetTiers.size();

struct KvGearSet {
    std::array<bool, kKvGearCount> allowed{};

    [[nodiscard]] static KvGearSet all() noexcept {
        KvGearSet set;
        set.allowed.fill(true);
        return set;
    }
    [[nodiscard]] bool has(std::size_t index) const noexcept { return allowed[index]; }
    void drop(std::size_t index) noexcept { allowed[index] = false; }
    [[nodiscard]] std::int32_t count() const noexcept {
        std::int32_t n = 0;
        for (const bool a : allowed) { if (a) { ++n; } }
        return n;
    }
    // "rk4v4,nvfp4" -- so a report can name the candidate space it solved over.
    [[nodiscard]] std::string to_name_list() const {
        std::string out;
        for (std::size_t i = 0; i < allowed.size(); ++i) {
            if (!allowed[i]) { continue; }
            if (!out.empty()) { out += ","; }
            out += kKvBitBudgetTiers[i].spec_name;
        }
        return out;
    }
};

// The rk4v4 restriction as an F_l family. `set` is the arbitrary allowed rk4v4 set
// (holes and all); a gear that is not rk4v4 is untouched. This is the bridge from
// the D4/B5 representation to the DP: the same struct that was only a report
// helper above is now the constraint the solver reads.
[[nodiscard]] inline std::vector<KvGearSet> kv_gear_sets_from_rk4v4_set(
    std::int32_t layers, const KvBitBudgetRk4v4Set& rk4v4) {
    // Ladder slots 4 (rk4v4), 6 (rk3v4) and 7 (rk2v4) -- the family, not row 4 alone.
    // The exposure limit is an rk4v4-FAMILY limit (see the is_rk4v4 predicate in the DP), so the
    // gear set has to drop the whole family, not row 4 alone: dropping only row 4 would
    // leave rk3v4/rk2v4 allowed on a layer the operator explicitly barred rk4v4 from.
    constexpr std::size_t kRkFamily[3] = {4, 6, 7};
    std::vector<KvGearSet> per_layer(static_cast<std::size_t>(layers), KvGearSet::all());
    for (std::int32_t l = 0; l < layers; ++l) {
        if (!rk4v4.allowed[static_cast<std::size_t>(l)]) {
            for (const std::size_t slot : kRkFamily) {
                per_layer[static_cast<std::size_t>(l)].drop(slot);
            }
        }
    }
    return per_layer;
}

// ---------------------------------------------------------------------------
// COMPONENT (KVarN) MODES. These are the plan-level switches the engine already
// has -- --kv-v-codec, --kv-rotation, --kv-row-scale -- so they are named here
// with the same three states rather than collapsed into a single score.
// ---------------------------------------------------------------------------
enum class KvVCodecMode : std::uint8_t {
    Iso4e = 0,   // --kv-v-codec iso4e (default): nvfp4 = E2M1 K + ISO4E V
    E2m1 = 1,   // --kv-v-codec e2m1
};
enum class KvRotationMode : std::uint8_t { On = 0, Off = 1 };
enum class KvRowScaleMode : std::uint8_t { Baked = 0, Identity = 1 };

struct KvGearComponentMode {
    KvVCodecMode v_codec  = KvVCodecMode::Iso4e;
    KvRotationMode rotation = KvRotationMode::On;
    KvRowScaleMode row_scale = KvRowScaleMode::Baked;

    // Which ladder slots this mode makes INADMISSIBLE, and why. A gate, not a
    // penalty: a penalty can be outvoted by the budget at a tight ceiling, and
    // the outcome would be a plan built on a measured-wrong kernel.
    [[nodiscard]] KvGearSet admissible() const {
        KvGearSet set = KvGearSet::all();
        if (rotation == KvRotationMode::On) {
            // fp8 decode rotates K but not Q (TODO section 4) => O(1) wrong.
            // Drop fp8 (ladder slot 2) until the Q read path is fixed.
            set.drop(2);
        }
        return set;
    }
    [[nodiscard]] std::string to_string() const {
        return std::string("v-codec=") + (v_codec == KvVCodecMode::E2m1 ? "e2m1" : "iso4e") +
               ",rotation=" + (rotation == KvRotationMode::On ? "on" : "off") +
               ",row-scale=" + (row_scale == KvRowScaleMode::Baked ? "baked" : "identity");
    }
};

// The row-scale/rotation quality credit, in the ladder's own x100 penalty units.
// z100 == 0.01 loss. WARNING [F1229]: THIS IS NO LONGER "DERIVED". The deltas it was derived
// from are refuted by re-measurement (see the row-scale block above: the identity table costs
// +0.34966 nats against a 0.00000 floor, not +0.00012..0.00207), so the 5 below is a POLICY
// value standing on a stale measurement. Re-deriving it -- which under the same 0->30 ==
// 0.12962-nats mapping would be ~35, i.e. a credit ~7x larger than the tier step the comment
// below says it must never outrank -- MOVES the scored default path and is OWED, not done.
// PRE-F1229 TEXT (kept so the change is auditable): "Derived, not chosen: the measured deltas
// above are
// |-0.00159|, |-0.00207| and |+0.00012| nats on the mean NLL, i.e. at most 0.00207
// nats. The ladder's own reference loss is nvfp4's 0.30 for the whole 4-bit step,
// and the largest measured tier-choice gap is 0.775 nats, so expressing the row
// scale as 5 (0.05) keeps it (a) far below a tier step and (b) non-zero so it can
// break a tie between two iso-quant-family gears. It is applied ONLY to the
// iso-quant family (nvfp4/iso4e/fp8, which read gqa_kv_row_scale) and never to rk4v4,
// whose baked-vs-identity arm is BIT-IDENTICAL (delta 0.00000).
inline constexpr std::int32_t kKvRowScaleCreditX100 = 5;      // baked, long context
inline constexpr std::int32_t kKvRowScaleCreditShortX100 = 0; // the 2k arm: mixed direction
inline constexpr std::int32_t kKvRowScaleE8CreditX100 = 0;    // negative control: bit-identical

// Ladder slots that read the row scale / rotation (gqa_isoquant_row_scale{,_loader}
// plus the nvfp4 decode/prefill arms). rk4v4 is deliberately NOT here: its
// baked-vs-identity arm is bit-identical, so applying the credit to it would be
// charging a difference that was measured to be zero.
[[nodiscard]] inline bool kv_gear_is_iso_quant_family(std::size_t index) noexcept {
    return index == 2 /*fp8*/ || index == 3 /*nvfp4*/ || index == 5 /*iso4e*/;
}

// ---------------------------------------------------------------------------
// The measured quality column. context = Long carries the 64k needle/ppl
// evidence; context = Short carries the ctx-4096 13.3k-zh column (where the
// shipped mix is the measured best of all mixes).
// ---------------------------------------------------------------------------
enum class KvScoreContext : std::uint8_t { Long = 0, Short = 1 };

// Long-context measured column, x100, lower = better, 0 = lossless:
//   bf16 0 (lossless, and 8/8 needles)   int8 2   fp8 3
//   nvfp4 30   iso4e 30   (both 8/8 needles; iso4e's pin was a POLICY choice, not
//   a measurement -- kv_bit_budget.h ladder comment -- and its measured speed is
//   a tie with nvfp4, so it is scored where nvfp4 is scored instead of pinned)
//   rk4v4: the all-layers arm answers 0/27. At LONG context rk4v4 is only measured good
//   on the leading window, which the F_l rk4v4 set already encodes; the residual
//   quality cost of the window itself is the 13.3k/ctx-4096 delta scaled down by
//   the fact that the leading window is the measured-safe half. It is kept at the
//   SHORT column's value (9) so this column does not double-charge a constraint
//   the set already carries -- and rk4v4 is NEVER preferred on this column, because
//   every other 4-bit gear scores 30 and speed separates the rest.
[[nodiscard]] inline KvTierScoreTable kv_bit_budget_measured_scores(KvScoreContext context) {
    if (context == KvScoreContext::Short) {
        // ctx-4096 13.3k zh perplexity, relative to the shipped mix (1.020):
        //   all-rk4v4 1.112 -> +9.0%   all-i8 1.522 -> +49.2%   all-nvfp4 1.706 -> +67.3%
        // bf16 is the lossless anchor (0); fp8 is not in that measurement, so it
        // inherits int8's (its plane cost is int8's + the E4M3 scale) and is
        // marked unmeasured by kv_bit_budget_measured_notes().
        return KvTierScoreTable{{
            {/*bf16 */ 0, 7},
            {/*int8 */ 49, 0},
            {/*fp8  */ 49, 0},
            {/*nvfp4*/ 67, 114},
            {/*rk4v4   */ 9, 218},
            {/*iso4e */ 67, 114},   // shares nvfp4's planes and speed; unpinned
            // [dl/backlog item7 + item10] THE TWO ROWS THE DEPLOY LAYER REFUSES, CARRIED AT THE
            // LADDER'S OWN DECLARED PENALTY. This initialiser had SIX entries for a
            // std::array<KvTierScoreRow, 8>, so rows 6/7 (rk3v4, rk2v4) were value-initialised
            // to (0, 0) -- the global MINIMUM on both slider ends. A PASSING shipped test then
            // emitted `0-4:rk3v4,5-7:rk2v4,...` and the deploy layer REFUSED it at layer 0
            // (product/kv_storage_dtype.h:117-119: 'rk3v4 is a DEFINED rk4v4 tier ... which NO
            // RUNTIME PATH CAN READ'). That is a correctness defect, not a doc defect.
            // NO VALUE IS INVENTED HERE. These are the SAME two numbers kKvBitBudgetTiers
            // declares for these two rows and kv_bit_budget_default_scores() already carries
            // (12/225 and 16/232). They are PRIORS, and they are load-bearing for exactly one
            // reason: rk4v4 is (9, 218), so 12 > 9 AND 225 > 218 means rk3v4 is DOMINATED on
            // BOTH columns for every w in [0,1] -- and likewise 16 > 9 and 232 > 218 for
            // rk2v4. A dominated row can never be CHOSEN, so the DP can no longer emit a plan
            // the deploy layer refuses, at any --kv-quality-weight.
            {/*rk3v4 */ 12, 225},  // the ladder's declared penalty; PRIOR, dominated by rk4v4
            {/*rk2v4 */ 16, 232},  // the ladder's declared penalty; PRIOR, dominated by rk4v4
        }};
    }
    // Long context. The 64k sweep carries the SPEED column and the retrieval
    // result; the quality column orders the gears by the retrieval result first
    // (any gear that answers the needles outranks one that does not) and by the
    // 4-bit family loss second.
    return KvTierScoreTable{{
        {/*bf16 */ 0, 7},
        {/*int8 */ 2, 0},
        {/*fp8  */ 3, 0},
        {/*nvfp4*/ 30, 114},
        {/*rk4v4   */ 9, 218},
        {/*iso4e */ 30, 114},
        // [dl/backlog item7 + item10] same reason as the Short arm above: six
        // initialisers for a std::array<KvTierScoreRow, 8> left rows 6/7 at (0, 0),
        // the global minimum on both ends. Carried at the ladder's declared penalty
        // (12/225, 16/232) they are dominated by rk4v4's (9, 218) on BOTH columns and
        // can never be chosen. No value is invented: both are already in this file.
        {/*rk3v4   */ 12, 225},
        {/*rk2v4   */ 16, 232},
    }};
}

// One line per column naming what is measured and what is inherited, so a report
// cannot present an inherited value as a measurement.
[[nodiscard]] inline std::string kv_bit_budget_measured_notes(KvScoreContext context) {
    if (context == KvScoreContext::Short) {
        return "ctx-4096 13.3k zh ppl (variant.cpp:26-27): mix 1.020 rk4v4 1.112 i8 1.522 "
               "nvfp4 1.706; fp8 and iso4e INHERIT (not in that measurement)";
    }
    return "64k: speed column measured (kv_tier_matrix.py); retrieval per rk4v4 placement "
           "measured (kv-strategy-matrix.md sec2 rule2); rk4v4 rows inherit the short-context "
           "value because the leading-window constraint is carried by F_l, not by the score";
}

// ---------------------------------------------------------------------------
// The plan: a per-layer gear array. The layer index is IN the objective's domain,
// which is the necessary condition for per-layer evidence to live anywhere.
// ---------------------------------------------------------------------------
struct KvGearPlan {
    std::vector<std::int32_t> gear;   // per layer; index into the ladder
    std::vector<bool> cold;           // per layer; true = deployed via the cold pool

    [[nodiscard]] std::size_t layers() const noexcept { return gear.size(); }
};

// Emit --kv-layer-storage grammar from a per-layer gear array. Runs ascend, which
// is what product/kv_options.h:61 accepts and what a reader can diff by eye
// against the plan. Cold layers are emitted as the run name "cold", which
// layouts_impl.h routes to the cold pool (deploy_kv_budget_spec) instead of to a
// per-layer dtype.
[[nodiscard]] inline std::string kv_gear_emit_spec(
    const KvGearPlan& plan, const std::array<KvBitBudgetTier, 8>& ladder) {
    std::string out;
    for (std::size_t i = 0; i < plan.gear.size();) {
        std::size_t j = i;
        const bool cold = i < plan.cold.size() && plan.cold[i];
        while (j + 1 < plan.gear.size() && plan.gear[j + 1] == plan.gear[i] &&
               (j + 1 < plan.cold.size() ? plan.cold[j + 1] : false) == cold) {
            ++j;
        }
        if (!out.empty()) { out += ","; }
        out += std::to_string(i);
        if (j != i) { out += "-" + std::to_string(j); }
        out += ":";
        out += cold ? "cold"
                    : ladder[static_cast<std::size_t>(plan.gear[i])].spec_name;
        i = j + 1;
    }
    return out;
}

// Ladder slot of a --kv-layer-storage tier name; -1 when the name is not a hot
// ladder tier ("cold" is handled by the caller, which knows the pool).
[[nodiscard]] inline std::int32_t kv_gear_slot_of(std::string_view name) noexcept {
    return detail::tier_index(name);
}

// Evaluate an EXISTING plan (any --kv-layer-storage spec, e.g. the factory table
// or an operator's incumbent) on the ladder the solver was handed, so an
// incumbent can be compared to the DP answer on the SAME objective. Returns false
// when the spec names a tier outside the ladder, or a layer twice, or a slot at or
// past `layers` -- refused loudly rather than silently scored as something else.
struct KvPlanEvaluation {
    bool parsed = false;
    bool within_budget = false;
    double achieved_bits = 0.0;
    double penalty = 0.0;
    std::string reason;
    // The plan, layer by layer, so a caller can check it against F_l instead of
    // re-parsing the spec and hoping the two agree.
    std::array<std::int32_t, kKvLayerStorageSlots> slot{};
    std::array<bool, kKvLayerStorageSlots> cold{};
};

[[nodiscard]] inline KvPlanEvaluation kv_gear_evaluate_spec(
    std::string_view spec, std::int32_t layers, double budget_bits,
    const std::array<KvBitBudgetTier, 8>& ladder, std::int32_t cold_bits_x100 = 0) {
    KvPlanEvaluation out;
    std::size_t begin = 0;
    std::array<bool, kKvLayerStorageSlots> seen{};
    std::array<std::int32_t, kKvLayerStorageSlots> slot{};
    slot.fill(-1);
    std::array<bool, kKvLayerStorageSlots> is_cold{};
    while (begin < spec.size()) {
        const std::size_t comma = spec.find(',', begin);
        const std::string_view item =
            spec.substr(begin, comma == std::string_view::npos ? spec.size() - begin
                                                               : comma - begin);
        if (item.empty()) { out.reason = "empty entry in '" + std::string(spec) + "'"; return out; }
        const std::size_t colon = item.find(':');
        if (colon == std::string_view::npos) {
            out.reason = "entry needs 'lo-hi:name': '" + std::string(item) + "'";
            return out;
        }
        const std::string_view where = item.substr(0, colon);
        const std::string_view name  = item.substr(colon + 1);
        std::size_t first = 0;
        std::size_t last  = 0;
        const std::size_t dash = where.find('-');
        try {
            if (where == "all") {
                first = 0;
                last  = static_cast<std::size_t>(layers) - 1;
            } else if (dash == std::string_view::npos) {
                first = last = static_cast<std::size_t>(std::stoul(std::string(where)));
            } else {
                first = static_cast<std::size_t>(std::stoul(std::string(where.substr(0, dash))));
                last  = static_cast<std::size_t>(std::stoul(std::string(where.substr(dash + 1))));
            }
        } catch (const std::exception&) {
            out.reason = "bad layer range '" + std::string(where) + "'";
            return out;
        }
        if (first > last || last >= static_cast<std::size_t>(layers)) {
            out.reason = "layer range " + std::string(where) + " is outside 0.." +
                         std::to_string(layers - 1);
            return out;
        }
        std::int32_t index = kv_gear_slot_of(name);
        const bool cold = name == "cold";
        if (index < 0 && !cold) {
            out.reason = "tier '" + std::string(name) + "' is not in this ladder";
            return out;
        }
        if (cold && cold_bits_x100 <= 0) {
            out.reason = "the plan names 'cold' but the cold pool is off in this solve";
            return out;
        }
        for (std::size_t l = first; l <= last; ++l) {
            if (seen[l]) {
                out.reason = "layer " + std::to_string(l) + " is named twice in '" +
                             std::string(spec) + "'";
                return out;
            }
            seen[l] = true;
            slot[l] = index;
            is_cold[l] = cold;
        }
        if (comma == std::string_view::npos) { break; }
        begin = comma + 1;
    }
    double bits = 0.0;
    double penalty = 0.0;
    for (std::int32_t l = 0; l < layers; ++l) {
        const std::size_t ul = static_cast<std::size_t>(l);
        if (!seen[ul]) {
            out.reason = "layer " + std::to_string(l) + " is not named by '" +
                         std::string(spec) + "'";
            return out;
        }
        if (is_cold[ul]) {
            bits += cold_bits_x100 / 100.0;
            penalty += kKvBitBudgetColdPenaltyX100 / 100.0;
            continue;
        }
        const auto& tier = ladder[static_cast<std::size_t>(slot[ul])];
        bits += tier.bits_x100 / 100.0;
        penalty += tier.penalty_x100 / 100.0;
    }
    out.parsed = true;
    out.achieved_bits = bits / static_cast<double>(layers);
    out.penalty = penalty;
    for (std::int32_t l = 0; l < layers; ++l) {
        out.slot[static_cast<std::size_t>(l)] = slot[static_cast<std::size_t>(l)];
        out.cold[static_cast<std::size_t>(l)] = is_cold[static_cast<std::size_t>(l)];
    }
    out.within_budget = out.achieved_bits <= budget_bits + 1e-9;
    if (!out.within_budget) {
        out.reason = "achieved " + std::to_string(out.achieved_bits) + " bits exceeds the " +
                     std::to_string(budget_bits) + " ceiling";
    }
    return out;
}

// The shipped table as a spec: the factory rk4v4 set over the layer count, nvfp4
// elsewhere. Built, not copied, so it cannot drift from kKvBitBudgetFactoryE8Layers.
[[nodiscard]] inline std::string kv_bit_budget_factory_spec(std::int32_t layers) {
    const std::int32_t nvfp4 = kv_gear_slot_of("nvfp4");
    KvGearPlan plan;
    plan.gear.assign(static_cast<std::size_t>(layers), nvfp4);
    for (const std::int32_t layer : kKvBitBudgetFactoryE8Layers) {
        if (layer >= 0 && layer < layers) {
            plan.gear[static_cast<std::size_t>(layer)] = kv_gear_slot_of("rk4v4");
        }
    }
    return kv_gear_emit_spec(plan, kKvBitBudgetTiers);
}

// ---------------------------------------------------------------------------
// THE REQUEST. Everything the gearbox needs, with the two properties that make
// the answer defensible: the constraints are per-layer (F_l), and the incumbents
// are evaluated on the SAME ladder the DP used.
// ---------------------------------------------------------------------------
struct KvGearSolveRequest {
    std::int32_t layers = 0;
    double budget_bits = 0.0;
    // The objective AND its row order (the rk4v4 slot is index 4 by construction).
    std::array<KvBitBudgetTier, 8> ladder = kKvBitBudgetTiers;
    // F_l. Empty => every gear on every layer. Size 1 => the same set everywhere.
    // Size `layers` => one set per layer (holes representable).
    std::vector<KvGearSet> per_layer;
    // The plan-level switches. They gate gears (fp8 x rotation) and adjust the
    // iso-quant family's score by the measured row-scale delta.
    KvGearComponentMode mode{};
    // OFF by default: the credit is a correction to the shipped PRIOR ladder, so a
    // caller that supplied its own columns must not have them silently adjusted.
    // kv_gear_solve_default() turns it on when it hands in the shipped ladder.
    bool apply_row_scale_credit = false;
    // Incumbents. Each is any --kv-layer-storage spec; it is returned when it is
    // at least as good as the DP answer, and REFUSED with a reason when it names
    // a gear the constraints or the mode forbid.
    std::vector<std::string> seeds;
    // The shipped/factory table is an incumbent by default: "reachable" is not the
    // same as "considered", and the measurement that made it good is in the tree.
    bool admit_factory_seed = true;
    // ---- THE SLIDER'S SELECTOR (BITSLIDER) --------------------------------------------
    // The rule this axis exists for: THE BIT COUNT IS AN INPUT, and the knob picks among
    // codecs that cost the SAME bits -- "allocate different KINDS of quantization at the
    // same bit count, not compress bits". Reachability was never the blockage: iso4e is
    // `selectable` and shares nvfp4's plane geometry exactly (450 == 450), so an all-iso4e
    // stack costs exactly what all-nvfp4 costs. The blockage is the TIE-BREAK: among plans
    // the objective already rates EQUAL, the DP keeps whichever candidate the fixed pack
    // order inserted first (detail::gear_candidates: rk4v4, bf16, int8, fp8, nvfp4, iso4e), so
    // nvfp4 displaced iso4e on EVERY tie and the equal-bit option set was unreachable
    // through the solver at any weight (measured: 15 knob combinations at 4.50, ONE spec).
    //
    // This field names the ladder slots to try FIRST. A slot it does not name keeps its
    // shipped position AFTER the named ones, so the candidate SET is unchanged and no gear
    // is dropped. Empty == the shipped order, byte-identical to before this field existed.
    //
    // WHAT IT CANNOT DO, stated here so it is not implied by a knob:
    //   * it does NOT lower or raise any penalty. A gear the ladder scores STRICTLY worse
    //     is still not chosen -- so under the shipped ladder (`iso4e` penalty 200 against
    //     `nvfp4` 30) `prefer iso4e` at 4.50 returns nvfp4: THE PIN IS A PENALTY and a
    //     preference is not a penalty override. It acts exactly where the caller's own
    //     columns have already made two candidates equal (kv_bit_budget_measured_scores:
    //     iso4e == nvfp4 == (30,114) on both columns, so the tie is real and measured).
    //   * it does NOT change which bit totals are reachable, so the achieved bits of the
    //     preferred answer are IDENTICAL to the default answer's at every budget. That
    //     equality is the property the landed test asserts.
    std::vector<std::int32_t> candidate_order;
    std::int32_t cold_cap = 0;
    std::int32_t cold_bits_x100 = kKvBitBudgetColdBitsX100;
};

struct KvGearSolution {
    bool feasible = false;
    std::string spec;               // --kv-layer-storage grammar, arbitrary runs
    double achieved_bits = 0.0;     // per element, K+V averaged
    double penalty = 0.0;
    // The ceiling this answer was solved against, on the same per-element scale. The
    // shortfall is derived from it at the single funnel into KvBitBudgetSolution
    // (kv_gear_to_budget_solution), so it cannot disagree with the returned plan.
    double requested_bits = 0.0;
    KvGearPlan plan;
    // "dp" or "seed" -- WHICH mechanism produced the answer. A caller that cannot
    // tell these apart cannot tell "the DP found this" from "the caller's
    // incumbent survived", and those are different claims.
    std::string winner;
    std::string seed_spec;          // the incumbent that won, empty when winner=="dp"
    std::vector<std::string> refusals;   // one line per seed the constraints refused
    std::vector<std::string> losses;     // one line per seed that was feasible but worse
    std::int64_t states_expanded = 0;
    std::string candidate_space;    // the union of F_l, for the report
};

namespace detail {

// The ladder indices in PACK order, restricted to what this layer allows. The
// order is not cosmetic: the DP keeps the first-inserted path on equal penalty,
// so iterating rk4v4 first is what makes the emitted spec read the way the shipped
// tables read (rk4v4 shallow, precision deep).
[[nodiscard]] inline std::vector<std::int32_t> gear_candidates(
    const KvGearSet& set, const std::array<KvBitBudgetTier, 8>& ladder) {
    // EXTENDED WITH kKvGearCandidateSlots, and it is not optional: this local table and the
    // file-level one at :1873 are the two spellings the comment below claims were consolidated
    // into one. Leaving this at six COMPILES and makes gear_candidates() disagree with
    // gear_candidates_ordered() -- the exact failure the consolidation was for.
    static constexpr std::array<std::size_t, 8> kOrder{{4, 0, 1, 2, 3, 5, 6, 7}};  // rk4v4 first
    std::vector<std::int32_t> out;
    for (const std::size_t index : kOrder) {
        if (set.allowed[index]) { out.push_back(static_cast<std::int32_t>(index)); }
    }
    return out;
}

// The ladder slots the layer-exact solver is allowed to CHOOSE. Was "the six the pack order
// names, which is also exactly the six the score table covers"; now EIGHT, because both of those
// facts moved with the gate (the score table grew in the same item). rk3v4/rk2v4 (slots 6,7) were
// outside this grammar on every path for two reasons, and BOTH have retired: `selectable == false`
// is now true-selectable, and a decode kernel for the 3-bit/2-bit plate now exists in the
// ninfer_ops closure (src/ops/kernel/e8_lattice_kv_plane_inst.cu, 44a78e174cbe88ca). The grammar
// extension does NOT claim the tier is reachable end to end -- patch_flip_batch.py's precondition
// notice records the two walls that remain (no caller, no append arm).
inline constexpr std::array<std::size_t, 8> kKvGearCandidateSlots{{4, 0, 1, 2, 3, 5, 6, 7}};

// THE candidate-grammar membership test. ONE expression, so that "which slots are
// candidates" cannot be spelled twice with two different answers. Before this, the
// same question was answered by an inline loop in kv_gear_solve_preferring (:2237-2240,
// its own bool and its own throw), by kv_gear_candidate_slot (:2285), by the
// in_grammar array in gear_candidates_ordered (:1869-1870) and by the rendering loop
// in kv_gear_candidate_list (:2272) -- five spellings over one constant, with nothing
// asserting they agreed. The claim at :2267-2268 was true of :2272/:2285 and false of
// :2238.
[[nodiscard]] constexpr bool kv_gear_slot_in_grammar(std::int32_t slot) noexcept {
    for (const std::size_t index : kKvGearCandidateSlots) {
        if (static_cast<std::int32_t>(index) == slot) { return true; }
    }
    return false;
}

// The literal half of the two-sided pin, in the idiom of product/kv_storage_dtype.h:200-209:
// the six slots that are IN are named, and the two rk4v4 width rows that are OUT are named,
// so a change to kKvGearCandidateSlots alone fails HERE. The test half (a new check
// evaluating BOTH acceptance paths over all eight ladder rows, with an on-side and an
// off-side non-vacuity control) belongs to the test owner -- this file's author must not
// add it.
static_assert(kv_gear_slot_in_grammar(4) && kv_gear_slot_in_grammar(0) &&
              kv_gear_slot_in_grammar(1) && kv_gear_slot_in_grammar(2) &&
              kv_gear_slot_in_grammar(3) && kv_gear_slot_in_grammar(5) &&
              kv_gear_slot_in_grammar(6) && kv_gear_slot_in_grammar(7),
              "the candidate slots are 4,0,1,2,3,5,6,7 -- every ladder row the DP may choose");
// REBUILT, and the rebuild is the reason this assert exists: it is the OFF side of a two-sided
// pin, so with the grammar extended there is no longer an off side to assert. It is replaced by
// the positive statement that the grammar covers EVERY selectable row -- which is the property
// the old off-side assert was protecting, now expressed against the gate instead of against two
// hard-coded slots. A row that flips selectable without entering the grammar fails HERE.
namespace detail {

[[nodiscard]] constexpr bool grammar_covers_every_selectable_gear() noexcept {
    for (std::size_t i = 0; i < kKvBitBudgetTiers.size(); ++i) {
        if (kKvBitBudgetTiers[i].selectable && !kv_gear_slot_in_grammar(static_cast<std::int32_t>(i))) {
            return false;
        }
    }
    return true;
}

}  // namespace detail
static_assert(detail::grammar_covers_every_selectable_gear(),
              "every SELECTABLE ladder row must be inside the candidate grammar: a row the DP may "
              "choose but a preference cannot name is a row the slider silently cannot reach");
static_assert(!kv_gear_slot_in_grammar(99),
              "rk3v4/rk2v4 are rows of the LADDER but outside the candidate grammar on "
              "EVERY path (selectable == false, and no kernel reads a 3-bit or 2-bit K "
              "code plate): a preference must never make the DP choose them");

// The pack order with the CALLER'S named slots first. The candidate SET is exactly
// gear_candidates()'s -- only the insertion order moves, which is the whole mechanism:
// the DP keeps the first-inserted path on an equal value (`value < next[to]` is false for
// an equal candidate), so naming a gear first is what lets it survive a tie it used to
// always lose. An empty `preferred` returns gear_candidates() verbatim.
[[nodiscard]] inline std::vector<std::int32_t> gear_candidates_ordered(
    const KvGearSet& set, const std::array<KvBitBudgetTier, 8>& ladder,
    const std::vector<std::int32_t>& preferred) {
    if (preferred.empty()) { return gear_candidates(set, ladder); }
    std::array<bool, kKvBitBudgetTiers.size()> in_grammar{};
    for (const std::size_t index : kKvGearCandidateSlots) { in_grammar[index] = true; }
    std::array<bool, kKvBitBudgetTiers.size()> taken{};
    std::vector<std::int32_t> out;
    for (const std::int32_t slot : preferred) {
        if (slot < 0 || slot >= static_cast<std::int32_t>(kKvBitBudgetTiers.size())) { continue; }
        const std::size_t index = static_cast<std::size_t>(slot);
        if (taken[index] || !in_grammar[index] || !set.allowed[index]) { continue; }
        taken[index] = true;
        out.push_back(slot);
    }
    for (const std::size_t index : kKvGearCandidateSlots) {
        if (taken[index] || !set.allowed[index]) { continue; }
        taken[index] = true;
        out.push_back(static_cast<std::int32_t>(index));
    }
    return out;
}

} // namespace detail

// The exact layer-indexed DP. State = (bits used, cold layers used); the choice
// at layer l is restricted to F_l, so every constraint -- the rk4v4 set, the
// component gates, a per-range ceiling -- is a function of the state key rather
// than a value carried in the surviving path. That is the whole difference from
// kv_bit_budget_solve_impl, whose rk4v4 count is a path value and therefore only a
// heuristic bound (kv_adapt_solver.h:33-39 states the same defect).
[[nodiscard]] inline KvGearSolution kv_gear_solve(const KvGearSolveRequest& request) {
    const std::int32_t layers = request.layers;
    if (layers <= 0) { throw std::invalid_argument("kv-gear: layer count must be positive"); }
    if (!(request.budget_bits > 0.0)) {
        throw std::invalid_argument("kv-gear: bit budget must be positive");
    }
    std::int32_t cold_cap = request.cold_cap;
    if (cold_cap < 0) { cold_cap = 0; }
    if (cold_cap > layers) { cold_cap = layers; }

    // F_l, intersected with the component mode's hard admissibility.
    const KvGearSet mode_ok = request.mode.admissible();
    std::vector<KvGearSet> per_layer;
    if (request.per_layer.empty()) {
        per_layer.assign(static_cast<std::size_t>(layers), KvGearSet::all());
    } else if (request.per_layer.size() == 1) {
        per_layer.assign(static_cast<std::size_t>(layers), request.per_layer.front());
    } else if (static_cast<std::int32_t>(request.per_layer.size()) == layers) {
        per_layer = request.per_layer;
    } else {
        throw std::invalid_argument(
            "kv-gear: per_layer must be empty, size 1, or one set per layer");
    }
    for (KvGearSet& set : per_layer) {
        for (std::size_t i = 0; i < set.allowed.size(); ++i) {
            if (!mode_ok.allowed[i]) { set.allowed[i] = false; }
        }
    }

    std::array<KvBitBudgetTier, 8> ladder = request.ladder;
    if (request.apply_row_scale_credit) {
        for (std::size_t i = 0; i < ladder.size(); ++i) {
            if (!kv_gear_is_iso_quant_family(i)) { continue; }
            if (request.mode.row_scale != KvRowScaleMode::Baked) { continue; }
            ladder[i].penalty_x100 -= kKvRowScaleCreditX100;
            if (ladder[i].penalty_x100 < 0) { ladder[i].penalty_x100 = 0; }
        }
    }

    const std::int32_t budget_x100 = static_cast<std::int32_t>(
        std::nearbyint(request.budget_bits * kKvBitBudgetScale));
    const std::int32_t capacity = budget_x100 * layers;
    const std::int32_t cold_states = cold_cap + 1;
    const std::int32_t span = capacity + 1;
    const std::size_t states =
        static_cast<std::size_t>(span) * static_cast<std::size_t>(cold_states);

    const double kInf = 1.0e18;
    std::vector<double> current(states, kInf);
    std::vector<double> next(states, kInf);
    std::vector<std::int8_t> choice(static_cast<std::size_t>(layers) * states, -1);
    std::vector<std::size_t> order_cur{0};
    std::vector<std::size_t> order_next;
    order_next.reserve(states);

    KvGearSolution solution;
    solution.states_expanded = 0;
    solution.candidate_space = per_layer[0].to_name_list();
    solution.requested_bits = static_cast<double>(budget_x100) / kKvBitBudgetScale;
    current[0] = 0.0;
    for (std::int32_t layer = 0; layer < layers; ++layer) {
        std::fill(next.begin(), next.end(), kInf);
        order_next.clear();
        const auto candidates = detail::gear_candidates_ordered(
            per_layer[static_cast<std::size_t>(layer)], ladder,
            request.candidate_order);
        const double cold_penalty = kKvBitBudgetColdPenaltyX100 / 100.0;
        for (const std::size_t from : order_cur) {
            const double base = current[from];
            if (base >= kInf) { continue; }
            ++solution.states_expanded;
            const std::int32_t from_bits = static_cast<std::int32_t>(from / cold_states);
            const std::int32_t from_cold = static_cast<std::int32_t>(from % cold_states);
            for (const std::int32_t t : candidates) {
                const std::int32_t new_bits = from_bits + ladder[static_cast<std::size_t>(t)].bits_x100;
                if (new_bits > capacity) { continue; }
                const std::size_t to = static_cast<std::size_t>(new_bits) * cold_states + from_cold;
                const double value = base + ladder[static_cast<std::size_t>(t)].penalty_x100 / 100.0;
                if (next[to] >= kInf) {
                    next[to] = value;
                    choice[static_cast<std::size_t>(layer) * states + to] =
                        static_cast<std::int8_t>(t);
                    order_next.push_back(to);
                } else if (value < next[to]) {
                    next[to] = value;
                    choice[static_cast<std::size_t>(layer) * states + to] =
                        static_cast<std::int8_t>(t);
                }
            }
            if (cold_cap > 0 && from_cold < cold_cap) {
                const std::int32_t cold_bits = from_bits + request.cold_bits_x100;
                if (cold_bits <= capacity) {
                    const std::size_t to = static_cast<std::size_t>(cold_bits) * cold_states +
                                           (from_cold + 1);
                    const double value = base + cold_penalty;
                    if (next[to] >= kInf) {
                        next[to] = value;
                        choice[static_cast<std::size_t>(layer) * states + to] =
                            static_cast<std::int8_t>(kKvBitBudgetColdTierIndex);
                        order_next.push_back(to);
                    } else if (value < next[to]) {
                        next[to] = value;
                        choice[static_cast<std::size_t>(layer) * states + to] =
                            static_cast<std::int8_t>(kKvBitBudgetColdTierIndex);
                    }
                }
            }
        }
        current.swap(next);
        order_cur.swap(order_next);
    }

    // SATURATION (BUDGETSAT), the same rule as the legacy solver above and for the same
    // measured reason: MAXIMUM BITS under the ceiling first, minimum penalty only as the
    // tie-break. The `new_bits > capacity` guard above is what keeps this from ever
    // exceeding the ceiling, and it is also why a larger ceiling can only raise the
    // maximum -- i.e. why request -> achieved is monotone by construction.
    std::size_t best = states;
    std::int32_t best_final_bits = -1;
    double best_penalty = kInf;
    for (const std::size_t state : order_cur) {
        if (current[state] >= kInf) { continue; }
        const std::int32_t state_bits = static_cast<std::int32_t>(state / cold_states);
        if (state_bits > best_final_bits ||
            (state_bits == best_final_bits && current[state] < best_penalty)) {
            best_final_bits = state_bits;
            best_penalty = current[state];
            best = state;
        }
    }

    // The DP answer, if one exists. It is reported even when a seed wins, so the
    // caller can see what the incumbent displaced.
    const bool dp_feasible = best != states;
    KvGearPlan dp_plan;
    double dp_bits = 0.0;
    std::string dp_spec;
    if (dp_feasible) {
        dp_plan.gear.assign(static_cast<std::size_t>(layers), -1);
        dp_plan.cold.assign(static_cast<std::size_t>(layers), false);
        std::int32_t bits = static_cast<std::int32_t>(best / cold_states);
        std::int32_t cold_used = static_cast<std::int32_t>(best % cold_states);
        for (std::int32_t layer = layers - 1; layer >= 0; --layer) {
            const std::size_t at = static_cast<std::size_t>(bits) * cold_states + cold_used;
            const std::int32_t t =
                choice[static_cast<std::size_t>(layer) * states + at];
            if (t < 0) { throw std::logic_error("kv-gear: broken DP backtrack"); }
            if (t == kKvBitBudgetColdTierIndex) {
                dp_plan.cold[static_cast<std::size_t>(layer)] = true;
                bits -= request.cold_bits_x100;
                --cold_used;
            } else {
                dp_plan.gear[static_cast<std::size_t>(layer)] = t;
                bits -= ladder[static_cast<std::size_t>(t)].bits_x100;
            }
        }
        dp_bits = static_cast<double>(static_cast<std::int32_t>(best / cold_states)) /
                  (layers * kKvBitBudgetScale);
        dp_spec = kv_gear_emit_spec(dp_plan, ladder);
    }

    // Seeds. An incumbent is usable only when it is INSIDE the constraints (F_l and
    // the component gates) and inside the ceiling; otherwise it is refused with the
    // reason named, because accepting it would deploy a plan the constraints exist
    // to forbid.
    KvPlanEvaluation best_seed;
    std::string best_seed_spec;
    std::vector<std::string> seed_pool;
    if (request.admit_factory_seed) { seed_pool.push_back(kv_bit_budget_factory_spec(layers)); }
    for (const std::string& s : request.seeds) { seed_pool.push_back(s); }
    for (const std::string& seed : seed_pool) {
        const KvPlanEvaluation ev =
            kv_gear_evaluate_spec(seed, layers, request.budget_bits, ladder,
                                  cold_cap > 0 ? request.cold_bits_x100 : 0);
        if (!ev.parsed) {
            solution.refusals.push_back("seed " + seed + ": " + ev.reason);
            continue;
        }
        if (!ev.within_budget) {
            solution.refusals.push_back("seed " + seed + ": " + ev.reason);
            continue;
        }
        // Constraint check per layer, on the gear the seed actually names: an
        // incumbent inside the ceiling but outside F_l would deploy a plan the
        // constraint exists to forbid (an rk4v4 layer outside its verified set, an
        // fp8 layer with rotation on), so it is REFUSED rather than scored.
        bool allowed = true;
        std::string why;
        for (std::int32_t l = 0; l < layers; ++l) {
            const std::size_t ul = static_cast<std::size_t>(l);
            if (ev.cold[ul]) {
                if (cold_cap <= 0) {
                    allowed = false;
                    why = "layer " + std::to_string(l) + " is cold but the cold pool is off";
                    break;
                }
                continue;
            }
            const std::int32_t slot = ev.slot[ul];
            if (slot < 0 || slot >= static_cast<std::int32_t>(kKvGearCount)) {
                allowed = false;
                why = "layer " + std::to_string(l) + " names no ladder tier";
                break;
            }
            if (!per_layer[ul].allowed[static_cast<std::size_t>(slot)]) {
                allowed = false;
                why = "layer " + std::to_string(l) + " names '" +
                      ladder[static_cast<std::size_t>(slot)].spec_name +
                      "', which F_" + std::to_string(l) + " excludes (mode: " +
                      request.mode.to_string() + "; candidate space: " +
                      per_layer[ul].to_name_list() + ")";
                break;
            }
        }
        if (!allowed) {
            solution.refusals.push_back("seed " + seed + ": " + why);
            continue;
        }
        // SATURATION (BUDGETSAT): an incumbent is judged on the SAME lexicographic order the
        // DP now optimises -- MORE BITS first, penalty only on an equal-bit tie. With the
        // old penalty-only test the factory table (4.41 bits, penalty 3.48) displaced the
        // saturating plan at a 4.50 request (penalty 4.80), i.e. the undershoot could come
        // back through the incumbent path even after the DP itself was fixed. The tolerance
        // is 1e-6 because the achievable grid step is 1/(layers * 100) = 0.000625 at 16
        // layers: below that a comparison would be between two DIFFERENT allocations.
        const bool seed_loses =
            dp_feasible && (ev.achieved_bits < dp_bits - 1e-6 ||
                            (std::fabs(ev.achieved_bits - dp_bits) <= 1e-6 &&
                             ev.penalty > best_penalty + 1e-9));
        if (seed_loses) {
            std::ostringstream loss;
            loss << "seed " << seed << ": " << ev.achieved_bits << " bits / penalty "
                 << ev.penalty << " loses to the DP answer " << dp_bits << " bits / penalty "
                 << best_penalty << " (the ceiling is filled from below)";
            solution.losses.push_back(loss.str());
            continue;
        }
        if (!best_seed.parsed || ev.penalty < best_seed.penalty) {
            best_seed = ev;
            best_seed_spec = seed;
        }
    }

    // Same lexicographic order as the loss test above: a seed may only win by matching the
    // DP's BIT count (it can never exceed it -- the DP maximises bits) and then beating it
    // on penalty. Anything else would re-introduce the undershoot via the incumbent.
    if (best_seed.parsed &&
        (!dp_feasible || (std::fabs(best_seed.achieved_bits - dp_bits) <= 1e-6 &&
                          best_seed.penalty <= best_penalty + 1e-9))) {
        solution.feasible = true;
        solution.spec = best_seed_spec;
        solution.achieved_bits = best_seed.achieved_bits;
        solution.penalty = best_seed.penalty;
        solution.winner = "seed";
        solution.seed_spec = best_seed_spec;
        solution.plan.gear.assign(static_cast<std::size_t>(layers), -1);
        solution.plan.cold.assign(static_cast<std::size_t>(layers), false);
        return solution;
    }
    if (!dp_feasible) {
        std::ostringstream why;
        why << "kv-gear: no feasible allocation at " << request.budget_bits
            << " bits/element over " << layers << " layer(s)";
        // Name the cheapest gear that IS admissible, since that is the ceiling the
        // caller has to raise to.
        std::int32_t cheapest = -1;
        for (std::size_t i = 0; i < ladder.size(); ++i) {
            if (per_layer[0].allowed[i] && (cheapest < 0 || ladder[i].bits_x100 < cheapest)) {
                cheapest = ladder[i].bits_x100;
            }
        }
        if (cheapest > 0) {
            why << "; the cheapest admissible gear here costs " << cheapest / 100.0
                << " bits/element"
                << " (candidate space at layer 0: " << per_layer[0].to_name_list() << ")";
        } else {
            why << "; no gear is admissible at all: check the component mode ("
                << request.mode.to_string() << ") and the rk4v4 set";
        }
        throw std::invalid_argument(why.str());
    }
    solution.feasible = true;
    solution.spec = dp_spec;
    solution.achieved_bits = dp_bits;
    solution.penalty = best_penalty;
    solution.plan = dp_plan;
    solution.winner = "dp";
    solution.candidate_space = per_layer[0].to_name_list();
    return solution;
}

// The gearbox with the shipped defaults: every gear allowed, the rk4v4 window as the
// measured leading prefix, the component mode as configured, the factory table as
// an incumbent. This is the entry a caller that only wants the full candidate
// space -- and not a new constraint language -- should use.
[[nodiscard]] inline KvGearSolution kv_gear_solve_default(
    std::int32_t layers, double budget_bits, const std::array<KvBitBudgetTier, 8>& ladder,
    const KvBitBudgetRk4v4Set& rk4v4, KvGearComponentMode mode = KvGearComponentMode{},
    std::int32_t cold_cap = 0) {
    KvGearSolveRequest request;
    request.layers = layers;
    request.budget_bits = budget_bits;
    request.ladder = ladder;
    request.per_layer = kv_gear_sets_from_rk4v4_set(layers, rk4v4);
    request.mode = mode;
    request.cold_cap = cold_cap;
    request.apply_row_scale_credit = true;   // the shipped-default entry: measured credit ON
    return kv_gear_solve(request);
}

// THE SLIDER'S OWN ENTRY. The same solve, with `tier` tried FIRST among candidates that
// score equally. This is the opt-in shape, chosen over the two alternatives for reasons
// that are measured, not stylistic:
//   * NOT "unpin iso4e in the auto set" -- AND THE REASON THIS BULLET USED TO GIVE WAS WRONG.
//     [F1229, line `landing`, 2026-09-29, measured, not argued] It said unpinning "would move
//     the DEFAULT answer at 4.50 from `0-15:nvfp4` to `0-15:iso4e`". It does not. With the pin
//     retired, kv_bit_budget_solve(16, 4.50, 8, 0) still returns `0-15:nvfp4` at penalty 4.80,
//     and across 61 ceilings x 2 rk4v4 windows the six SHIPPED entries (ladder cold-off,
//     ladder cold-audited, default-table scored at w=1, both measured columns,
//     kv_gear_solve_default) are byte-identical before and after the edit
//     (dl/landing/probe/iso_pin_probe.cpp, 854 solves per side, 18 lines of diff and all 18
//     are gear_prefer_iso4e). The reason is the one this file already documents two hundred
//     lines up: nvfp4 is inserted BEFORE iso4e in detail::gear_candidates, so an equal-penalty
//     tie is kept by nvfp4 whatever the penalty column says. What the pin actually blocked was
//     THIS function's own axis -- see the bullet below, whose text is now measured-true where
//     it was previously only argued: `prefer iso4e` returned nvfp4 BECAUSE of the pin.
//   * NOT "drive it from --kv-quality-weight": measured incapable. At 4.50 the solver
//     returns `0-15:nvfp4` for every w in {0,0.25,0.5,0.75,1} under BOTH measured score
//     tables and under the default table -- 15/15 knob combinations, one spec -- because
//     the only bit-equal partner (iso4e) is exactly his own speed/quality mirror and the
//     fixed order then decides the tie the same way every time.
//   * SO: an explicit, labelled preference, default off, that cannot move a penalty.
//
// `ladder` is the caller's authority for the penalties, exactly as in kv_gear_solve; the
// measured row-scale credit is NOT applied here (same as kv_bit_budget_solve), so a caller
// that wants it can carry it in the ladder it passes.
//
// An unknown name, or a gear outside the candidate grammar (rk3v4/rk2v4), is REFUSED rather
// than silently ignored.
// The candidate-gear vocabulary, defined below with the rest of the candidate grammar.
// The refusal in kv_gear_solve_preferring is quoted from THIS function rather than written
// out by hand: the list and the acceptance test are the same expression over the same
// table, which is the whole failure mode a hand-written list has.
[[nodiscard]] inline std::string kv_gear_candidate_list();

[[nodiscard]] inline KvGearSolution kv_gear_solve_preferring(
    std::int32_t layers, double budget_bits, const std::array<KvBitBudgetTier, 8>& ladder,
    const KvBitBudgetRk4v4Set& rk4v4, std::string_view tier,
    KvGearComponentMode mode = KvGearComponentMode{}, std::int32_t cold_cap = 0) {
    const std::int32_t slot = detail::tier_index(tier);
    const bool in_grammar = detail::kv_gear_slot_in_grammar(slot);
    if (!in_grammar) {
        throw std::invalid_argument(
            "kv-gear: prefer '" + std::string(tier) + "' is not a candidate gear (want " +
            kv_gear_candidate_list() +
            "); every SELECTABLE ladder row is inside the candidate grammar and nothing else is "
            "(dl/e8mixwire/land/patch_flip_batch.py clusters 5-7 grew"
            " kKvGearCandidateSlots 6 -> 8 with the gate, and grammar_covers_every_selectable_gear "
            "keeps the two sets equal)");
    }
    KvGearSolveRequest request;
    request.layers = layers;
    request.budget_bits = budget_bits;
    request.ladder = ladder;
    request.per_layer = kv_gear_sets_from_rk4v4_set(layers, rk4v4);
    request.mode = mode;
    request.cold_cap = cold_cap;
    request.candidate_order = {slot};
    return kv_gear_solve(request);
}

// ---- The CANDIDATE GRAMMAR as a user-facing vocabulary (SLIDERWIRE) ----------------
// A preference (KvGearSolveRequest::candidate_order) may name ONLY the ladder slots the
// layer-exact solver is allowed to CHOOSE. That set is now EIGHT: rk3v4/rk2v4 left the "outside
// this grammar on EVERY path" clause when the gate opened in the same item (see
// kKvGearCandidateSlots above, and patch_flip_batch.py's precondition notice for what the grammar
// extension still does not buy). The list below is still DERIVED from the same table the
// acceptance test walks, so a front end quoting kv_gear_candidate_list() cannot drift from what
// the solver accepts -- which is the whole failure mode a hand-written list has, and the reason
// this comment's "the six" became "the eight" rather than a second spelling.
//
// The list and the acceptance test are the SAME expression over the SAME table, so a front
// end that quotes kv_gear_candidate_list() in an error message cannot drift from what the
// solver actually accepts -- which is the whole failure mode a hand-written list has.
[[nodiscard]] inline std::string kv_gear_candidate_list() {
    std::string out;
    for (const std::size_t index : detail::kKvGearCandidateSlots) {
        if (!out.empty()) { out += ", "; }
        out += kKvBitBudgetTiers[index].spec_name;
    }
    return out;
}

// The ladder slot for `name` when it is inside the candidate grammar, else -1. -1 is a
// REFUSAL, not a fallback: every caller in this tree turns it into an error carrying
// kv_gear_candidate_list(), because "accepted and silently ignored" is this project's
// worst outcome.
[[nodiscard]] inline std::int32_t kv_gear_candidate_slot(std::string_view name) noexcept {
    const std::int32_t slot = detail::tier_index(name);
    return detail::kv_gear_slot_in_grammar(slot) ? slot : -1;
}

// ---------------------------------------------------------------------------
// The gearbox answer, expressed in the type every existing caller already speaks.
// counts is derived from the per-layer plan, so it can no longer disagree with the
// spec the way a separately-accumulated count can.
// ---------------------------------------------------------------------------
namespace detail {

// ===========================================================================
// THE GATED FALLBACK -- TWO PASSES, AND THE FIRST PASS IS THE SHIPPED ONE
// ===========================================================================
// WHY NOT NARROW THE CANDIDATE SET UP FRONT. Measured on a shadow tree
// (dl/kvreach/logs/44_zero_move.txt): ANDing the deployability census into the candidate
// set BEFORE solving moved 47 landings that were ALREADY DEPLOYABLE -- same achieved_bits,
// different codec mix, different penalty -- because shrinking the candidate set
// re-optimises the ladder's own objective EVERYWHERE, not only on the ceilings where the
// withheld row was chosen. A landing that works must not move, so the shipped candidate set
// gets the first word and the census only breaks a tie the engine cannot build.
//
// THE RULE, IN FOUR LINES: solve as shipped; if the answer is a plan the deploy layer
// accepts, return it UNTOUCHED; otherwise solve again with the withheld rows made
// unselectable and return that. The second pass can never be reached on a ceiling whose
// shipped answer was already runnable, which is what makes the zero-move property
// STRUCTURAL rather than measured.
//
// WHAT IT DOES NOT DO: it lowers no penalty, moves no bit count, and does NOT touch
// `selectable` -- a withheld row is still a ladder row, still priced, still spellable by
// --kv-layer-storage (where the deploy layer refuses it by name), and still inside the
// candidate GRAMMAR, so --kv-codec-preference rk3v4 is still rejected for the reason it
// always was (the row loses the fit) rather than newly accepted-and-ignored.
[[nodiscard]] inline KvBitBudgetSolution kv_bit_budget_solve_impl_gated(
    std::int32_t layers, double budget_bits, const std::array<KvBitBudgetTier, 8>& ladder,
    std::int32_t rk4v4_limit, std::int32_t cold_cap,
    std::int32_t cold_bits_x100 = kKvBitBudgetColdBitsX100) {
    KvBitBudgetSolution first =
        kv_bit_budget_solve_impl(layers, budget_bits, ladder, rk4v4_limit, cold_cap,
                                 cold_bits_x100);
    if (kv_bit_budget_spec_is_deployable(first.spec)) { return first; }
    std::array<KvBitBudgetTier, 8> masked = ladder;
    const std::array<bool, 8>& ok = kv_bit_budget_deployable_rows();
    for (std::size_t i = 0; i < masked.size() && i < ok.size(); ++i) {
        if (!ok[i]) { masked[i].selectable = false; }
    }
    return kv_bit_budget_solve_impl(layers, budget_bits, masked, rk4v4_limit, cold_cap,
                                    cold_bits_x100);
}

[[nodiscard]] inline KvGearSolution kv_gear_solve_gated(const KvGearSolveRequest& request) {
    KvGearSolution first = kv_gear_solve(request);
    if (kv_bit_budget_spec_is_deployable(first.spec)) { return first; }
    KvGearSolveRequest masked = request;
    if (masked.per_layer.empty()) {
        masked.per_layer.assign(static_cast<std::size_t>(request.layers), KvGearSet::all());
    }
    const std::array<bool, 8>& ok = kv_bit_budget_deployable_rows();
    for (KvGearSet& set : masked.per_layer) {
        for (std::size_t i = 0; i < set.allowed.size() && i < ok.size(); ++i) {
            if (!ok[i]) { set.allowed[i] = false; }
        }
    }
    return kv_gear_solve(masked);
}

}  // namespace detail

[[nodiscard]] inline KvBitBudgetSolution kv_gear_to_budget_solution(
// kvreach-k4-funnel
    const KvGearSolution& solved, const std::array<KvBitBudgetTier, 8>& ladder,
    std::int32_t layers, std::int32_t cold_bits_x100) {
    KvBitBudgetSolution out;
    out.spec = solved.spec;
    out.achieved_bits = solved.achieved_bits;
    out.penalty = solved.penalty;
    for (std::int32_t l = 0; l < layers; ++l) {
        const std::size_t ul = static_cast<std::size_t>(l);
        if (ul < solved.plan.cold.size() && solved.plan.cold[ul]) {
            ++out.counts["cold"];
            ++out.cold_layers;
            continue;
        }
        const std::int32_t gear = ul < solved.plan.gear.size() ? solved.plan.gear[ul] : -1;
        if (gear < 0) { continue; }
        ++out.counts[ladder[static_cast<std::size_t>(gear)].spec_name];
    }
    out.cold_bits_x100 = out.cold_layers == 0 ? 0 : cold_bits_x100;
    out.requested_bits = solved.requested_bits;
    out.shortfall_bits = solved.requested_bits - solved.achieved_bits;
    if (out.shortfall_bits < 0.0) { out.shortfall_bits = 0.0; }
    return out;
}

// --- the two engine-facing entries, now served by the layer-exact solver -------
// Both keep their exact signature and their exact meaning; what changes is the
// candidate space behind them. Each is handed the SAME constraints the old path
// used, expressed in the faithful representation: the rk4v4 count becomes the rk4v4 SET
// (the leading window it always meant), every gear is allowed, and the shipped
// table is evaluated as an incumbent rather than assumed either way.
[[nodiscard]] inline KvBitBudgetSolution kv_bit_budget_solve(
    std::int32_t layers, double budget_bits, std::int32_t rk4v4_limit, std::int32_t cold_cap,
    std::int32_t cold_bits_x100, const std::vector<std::int32_t>& candidate_order) {
    // SLIDERWIRE: a preference and a cold pool are mutually unrepresentable, not merely
    // unusual. The cold plan comes from kv_bit_budget_solve_impl (see the note below), whose
    // decision variable is a COUNT per tier placed by a fixed packing -- there is no candidate
    // ORDER in it for a preference to act on, so a preference here would be accepted and read
    // by nothing. Refused by name instead.
    if (!candidate_order.empty() && cold_cap > 0) {
        throw std::invalid_argument(
            "kv-bit-budget: a candidate preference was given together with a cold pool "
            "(cold_cap > 0). With cold layers the plan is built by the legacy multiset solver, "
            "which decides a COUNT per tier and lets a fixed packing place them, so it has no "
            "candidate order for a preference to act on and the preference would be accepted "
            "and read by nothing. Set --max-cold-pages 0, or drop the preference.");
    }
    // The COLD path stays on the legacy solver, deliberately. Which layers receive the
    // cold pool is a POSITIONAL policy in this header, not an objective term: the pack
    // order puts rk4v4 first (its verified leading window), cold on the next-shallow block,
    // and the hot high-precision tail deepest. A per-layer DP does not encode that rule
    // -- it would place cold wherever it scores best -- and no measurement here says the
    // pool is indifferent to the placement, so the cold plan is left bit-identical to
    // what this header produced before rather than silently re-laid-out. The gearbox is
    // reached whenever cold is off, which is the default (`--max-cold-pages 0`).
    if (cold_cap > 0) {
        // THE GATED FUNNEL, not kv_bit_budget_solve_impl: the cold plan is built by the
        // legacy multiset DP and must reach the same deployability rule as the gear path,
        // or the two entry SPELLINGS would disagree about the ceilings that need it.
        return detail::kv_bit_budget_solve_impl_gated(layers, budget_bits, kKvBitBudgetTiers,
                                                      rk4v4_limit, cold_cap, cold_bits_x100);
    }
// kvreach-k5-cold-legacy
    KvGearSolveRequest request;
    request.layers = layers;
    request.budget_bits = budget_bits;
    request.ladder = kKvBitBudgetTiers;
    request.per_layer = kv_gear_sets_from_rk4v4_set(layers, kv_bit_budget_rk4v4_prefix_set(rk4v4_limit));
    request.cold_cap = cold_cap;
    request.cold_bits_x100 = cold_bits_x100;
    // SLIDERWIRE: the caller's preference, verbatim. Empty keeps the shipped order.
    request.candidate_order = candidate_order;
    // THE GATED FUNNEL: the shipped candidate set gets the first word (so no already-runnable
    // landing moves), and the deployability census breaks a tie only when the first answer is
    // a plan the build layer refuses.
    const KvGearSolution solved = detail::kv_gear_solve_gated(request);
    return kv_gear_to_budget_solution(solved, kKvBitBudgetTiers, layers, cold_bits_x100);
// kvreach-k7-gear-legacy
}

[[nodiscard]] inline KvBitBudgetSolution kv_bit_budget_solve_scored(
    std::int32_t layers, double budget_bits, const KvTierScoreTable& scores,
    double quality_weight, std::int32_t rk4v4_limit, std::int32_t cold_cap,
    const std::vector<std::int32_t>& candidate_order) {
    // SLIDERWIRE: the same refusal as kv_bit_budget_solve's (one rule, both entries).
    if (!candidate_order.empty() && cold_cap > 0) {
        throw std::invalid_argument(
            "kv-bit-budget: a candidate preference was given together with a cold pool "
            "(cold_cap > 0). With cold layers the plan is built by the legacy multiset solver, "
            "which decides a COUNT per tier and lets a fixed packing place them, so it has no "
            "candidate order for a preference to act on and the preference would be accepted "
            "and read by nothing. Set --max-cold-pages 0, or drop the preference.");
    }
    // Same cold-path note as kv_bit_budget_solve above.
    if (cold_cap > 0) {
        // Same gated funnel as kv_bit_budget_solve's cold arm, one rule both entries.
        return detail::kv_bit_budget_solve_impl_gated(
            layers, budget_bits, kv_bit_budget_scored_ladder(scores, quality_weight), rk4v4_limit,
            cold_cap);
    }
// kvreach-k6-cold-scored
    KvGearSolveRequest request;
    request.layers = layers;
    request.budget_bits = budget_bits;
    // The caller's two columns ARE the score here, so the measured row-scale credit
    // is left OFF: it is a correction to the shipped PRIOR ladder, and a caller that
    // hands in its own columns is the authority for those columns.
    request.ladder = kv_bit_budget_scored_ladder(scores, quality_weight);
    request.per_layer = kv_gear_sets_from_rk4v4_set(layers, kv_bit_budget_rk4v4_prefix_set(rk4v4_limit));
    request.cold_cap = cold_cap;
    // SLIDERWIRE: the caller's preference, verbatim. Empty keeps the shipped order.
    request.candidate_order = candidate_order;
    // Same gated funnel as kv_bit_budget_solve's gear arm: one rule, both spellings.
    const KvGearSolution solved = detail::kv_gear_solve_gated(request);
    return kv_gear_to_budget_solution(solved, request.ladder, layers, kKvBitBudgetColdBitsX100);
// kvreach-k8-gear-scored
}

} // namespace ninfer::product
