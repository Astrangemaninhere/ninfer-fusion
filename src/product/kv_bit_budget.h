#pragma once

// Fractional KV bit-budget allocator (the C++ side; tools/archkit/kv_bit_budget.py, which
// the original port mirrored and which the notes below name as the parity partner, is no
// longer in the tree - see the cold-tier note for what that means for cross-checks).
//
// Given a target average bit budget per KV element, pick one KV tier per full-attention
// layer so that the total bit cost stays inside the budget while the total quality
// penalty is minimised. Solved exactly with a small DP whose state is (bits used, cold
// layers used); the e8 count is carried in the surviving path rather than in the state key.
// The resulting per-layer assignment is packed into the existing --kv-layer-storage grammar,
// so the rest of the engine keeps its single resolution path.
//
// Tier costs (bits/element, K+V averaged) and penalties (needle-sweep prior, lower is
// better) follow the ENGINE's plane geometry and the measured needle/speed columns, not the
// deleted Python tool's nominal widths. The ladder below is the authority; the summary here
// is only a reading aid, and it has been stale before (it kept the Python tool's fp8 8.03 /
// e8 4.06 / iso3 3.00 after the geometry fix moved them): bf16 16.00/0.00, int8 8.25/0.02,
// fp8 8.50/0.03, nvfp4 4.50/0.30, e8 4.25/0.08, iso3 4.50/2.00 (pinned out).
//
// e8 is only allowed on the leading layers: the e8 tier is verified for layers 0..7 while
// the shipped high-layer e8 path degrades (_TODO.md 96/116); iso3/nvfp4 take over above.
//
// Opt-in cold tier (engine knobs: --cold-policy none/window/host/disk + max_cold_pages):
// with a cold capacity in pages, a layer may instead live in the cold pool. At most
// cold_cap layers may be cold; cold_used is an exact DP dimension. Packing keeps e8 FIRST
// (the e8 limit is a leading-layer constraint, e8 must keep slots 0..count-1 inside its
// verified window), then places cold on the next-shallow block; deep layers keep the hot
// high-precision tail (_TODO.md 46/47). "cold" is NOT a --kv-layer-storage tier (grammar:
// bf16/int8/fp8/nvfp4/iso3/e8): cold-planned layers are deployed via
// --cold-policy host|disk --max-cold-pages N.
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
// pre-fix "cold is free" model AND the pre-correction ladder (fp8 803 / e8 406 / iso3 300,
// e8_limit 8), so neither can be replayed against this header - the ladder was later corrected
// to fp8 850 / e8 425 / iso3 450 with e8_limit 10. Cross-check with a fresh Python mirror
// instead; the cold-off column is the part that must stay identical, and it is, by the
// cold_cap == 0 argument above.
//
// kKvBitBudgetColdSlotBytes mirrors ops::kEntropyNvfp4SlotBytes, defined at
// include/ninfer/ops/entropy_nvfp4_slot.h:14 and consumed at
// src/targets/qwen3_6/impl/state/decoder_state.cpp:232-249 (the cold slot tensors); the value
// is spelled out here so this header stays host-only (that header includes cuda_runtime.h).

#include "ninfer/types.h"

#include <array>
#include <cmath>
#include <sstream>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::product {

// How many layers the DP may place on e8, anywhere in the stack. This is a count, not a
// leading window: the solver carries the running count in its state and refuses to exceed the
// cap, so the cap is the only thing that bounds e8 exposure.
//
// The value is measurement-backed rather than chosen. The shipped default policy places e8 in
// exactly 10 layers (qwen3_6_27b variant.cpp: {0,1,3,4,6,7,8,9,13,14}, the rest NVFP4) and is
// verified at a 131072-token context: 27/27 needle, AL 10.00, 56.90 tok/s, 2.67 GiB. All 16
// layers on e8 is not usable at that context (0/27, and acceptance collapses to 2.82), so the
// safe count is at least 10 and below 16; 11..15 are unmeasured. The previous cap of 8 was
// therefore more conservative than the evidence supports.
//
// Note the DP's own placement at this cap still has to be re-validated: the measured safe set is
// ten specific layers, and the DP is free to choose a different ten.
inline constexpr std::int32_t kKvBitBudgetE8LayerLimit = 10;
// 0.01-bit resolution, matching the Python tool.
inline constexpr std::int32_t kKvBitBudgetScale = 100;

struct KvBitBudgetTier {
    const char* spec_name;
    std::int32_t bits_x100;
    std::int32_t penalty_x100;
};

// Cost ladder (cheapest first is not needed; the DP explores all of them).
// Bit costs are the ENGINE's plane geometry (per KV element, K+V averaged), not nominal
// format widths - see src/targets/qwen3_6/impl/state/decoder_state.cpp: every tier is a
// nibble or byte code plane plus a scale plane, so int8 pays 16 bits per 64 elements of
// scale (8.25), e8 pays 16 per 64 (4.25), nvfp4 pays 8 per 16 (4.50), and fp8 pays 8 bits
// per element of code plus 8 per 16 of E4M3FN scale, i.e. 8.50.
// fp8's scale plane is E4M3FN at group 16 - the same shape nvfp4/iso3 use and the only one
// the fp8 attention kernels read (gqa_attention_decode_fp8.cuh:159 kFp8Groups = D/16,
// gqa_attention_prefill.cu:200/360 fp8 arms) and the only one ops/wrapper/gqa_attention.cpp
// accepts for a packed-16 tier (:80-82, :121-133, :208-221). The tier is still dominated by
// int8 on cost (8.50 > 8.25 at a worse penalty), so the DP will not pick it on its own; it
// is now a runnable reference point rather than a refusing one.
// iso3 SHARES nvfp4's plane geometry (two codes per byte over per-16 E4M3FN scales, hence
// 4.50) but is NOT the nvfp4 kernel and NOT the same K format: Iso3Group16 maps to
// DType::ISO3 (decoder_state.cpp:28-33 gives it the same quant_group and nothing more), whose
// decode/encode paths are ops/kernel/gqa_attention_decode_iso3.cuh - K carries sign-magnitude
// ISO3 nibbles, not E2M1 - and which never touches the nvfp4 second-stage residual plane set,
// while the DType::NVFP4 decode arm runs an extra residual QK pass (decoder_state.cpp:156-185).
// Sharing planes therefore does NOT make iso3 an alias of nvfp4: it can never beat nvfp4 on
// bits (equal) and the measured speed column is a tie (40.3 vs 40.4 tok/s below). Its penalty
// is left pinned PAST the scale so the DP treats it as unusable rather than as a cost-equal
// duplicate of nvfp4 - that pin is a naming/policy choice, not a measured quality claim, and
// the bit column alone (450 == 450) is what makes the two indistinguishable to the DP.
// Measured, 64k NIAH, all 16 layers at one tier (tools/archkit/kv_tier_matrix.py):
//   int8 86.2 > bf16 80.7 > shipped mix 57.1 > nvfp4 40.3 ~ iso3 40.4 > e8 27.1 tok/s,
//   with e8 answering 0/27 (long-context retrieval breaks). fp8 was not measured in that
//   sweep: every fp8 arm of it died in the planner's FP16 scale plane before reaching the
//   kernels, so its speed column is still open.
inline constexpr std::array<KvBitBudgetTier, 6> kKvBitBudgetTiers{{
    {"bf16", 1600, 0},
    {"int8", 825, 2},
    {"fp8", 850, 3},     // E4M3 scales at group 16 (8.50 bits); still dominated by int8
    {"nvfp4", 450, 30},
    {"e8", 425, 8},      // nibble + FP16/g64
    {"iso3", 450, 200},  // == nvfp4 planes; pinned out of the candidate set
}};

// Cold pseudo-tier. It deliberately lives OUTSIDE kKvBitBudgetTiers (index 6) so engine
// consumers of the hot tier table are untouched.
inline constexpr std::int32_t kKvBitBudgetColdTierIndex = 6;
// One cold slot, in bytes, per (page, kv_head, K|V plane). Mirrors
// ops::kEntropyNvfp4SlotBytes (include/ninfer/ops/entropy_nvfp4_slot.h:14).
inline constexpr std::int32_t kKvBitBudgetColdSlotBytes = 9536;
// Elements covered by one head-page of the slot codec: the slot's payload is 8192 code bytes
// plus a 1024 B uncompressed scale tail (entropy_nvfp4_slot.h:12, :17) and a code byte holds
// two 4-bit KV codes, so one slot replaces exactly 16384 KV elements - the same element count
// the 9216 B nvfp4 head-page covers, which is what puts the cold cost on the ladder's own
// per-element scale.
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
//   kKvBitBudgetColdSlotBytes * 8 / kKvBitBudgetElementsPerHeadPage = 4.65625
// and that is what it must be compared against, per head-page, for each hot tier:
//   tier   bits/el  resident B  cold B  delta B   delta   verdict
//   bf16    16.00      32768     9536   -23232   -70.9%  cold saves
//   int8     8.25      16896     9536    -7360   -43.6%  cold saves (the only cold codec
//                                                        reachable in this tree)
//   fp8      8.50      17408     9536    -7872   -45.2%  cold saves (tier does not run)
//   nvfp4    4.50       9216     9536     +320    +3.5%  cold COSTS MORE
//   e8       4.25       8704     9536     +832    +9.6%  cold COSTS MORE
//   iso3     4.50       9216     9536     +320    +3.5%  cold COSTS MORE
// resident B == bits_x100 * kKvBitBudgetElementsPerHeadPage / 800, i.e. the very plane
// geometry the bits_x100 column was derived from, so the two can never drift apart.
//
// Consequences, all deliberate:
//   * achieved_bits is a true per-element device footprint for ANY plan, cold included: a
//     caller can no longer be handed a cold plan that silently costs more bytes than it says.
//   * the DP still returns the minimum-penalty plan inside the budget, but it now PRICES cold
//     instead of subsidising it; on a sub-int8 (nvfp4/e8/iso3) stack cold can only win by
//     paying extra bits for the lower restore prior, and with the ladder as it stands
//     (e8 425 < nvfp4 450 < cold 466) every cold-using plan is a byte increase.
//   * kv_bit_budget_solve_audited() additionally compares the cold plan against the SAME
//     budget solved with the cold pool OFF, in real head-page bytes, and refuses the cold plan
//     when it is a net increase; the raw numbers stay in the result (cold_net_head_page_bytes,
//     cold_report) so a caller can print or override them.
//   * cold_cap == 0 (ColdPolicy::None, or --max-cold-pages 0) makes ALL of the above
//     unreachable: the cold candidate is enumerated only under `cold_cap > 0`, so the cold-off
//     allocation is bit-identical to the pre-fix header.
inline constexpr std::int32_t kKvBitBudgetColdPenaltyX100 = 25;  // 0.25 restore prior

namespace detail {

// 9536 * 8 * 100 / 16384 == 465.625 -> the 0.01-bit grid point 466 (.625 is not a tie, so
// there is no tie-to-even subtlety here).
[[nodiscard]] constexpr std::int32_t cold_bits_x100_from_slot_geometry() noexcept {
    const std::int64_t scaled = static_cast<std::int64_t>(kKvBitBudgetColdSlotBytes) * 8 * 100;
    return static_cast<std::int32_t>((scaled + kKvBitBudgetElementsPerHeadPage / 2) /
                                     kKvBitBudgetElementsPerHeadPage);
}

} // namespace detail

// Per-element device cost of a cold layer, on the ladder's own scale. Derived, never
// hand-written: change the slot geometry and this follows.
inline constexpr std::int32_t kKvBitBudgetColdBitsX100 =
    detail::cold_bits_x100_from_slot_geometry();
// Tripwire: if this fires, the slot geometry changed and the cold-vs-resident table above (and
// every recorded cold verification value) has to be re-derived deliberately.
static_assert(kKvBitBudgetColdBitsX100 == 466,
              "cold slot geometry changed: 9536 B over 16384 elements must be 4.65625 b/el");

// Resident plane bytes per (page, kv_head, K|V plane) implied by a bits_x100 cost:
// kv_bit_budget_plane_bytes(450) == 9216, (425) == 8704, (825) == 16896, (850) == 17408,
// (1600) == 32768 - the numbers the cold table above is built from.
[[nodiscard]] constexpr std::int32_t kv_bit_budget_plane_bytes(std::int32_t bits_x100) noexcept {
    return static_cast<std::int32_t>((static_cast<std::int64_t>(bits_x100) *
                                      kKvBitBudgetElementsPerHeadPage) / 800);
}

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

// Packing order: e8 first so the verified leading layers receive it (the DP only counts
// tiers, the packing decides which layer gets which).
inline constexpr std::array<const char*, 6> kKvBitBudgetPackOrder{
    {"e8", "bf16", "int8", "fp8", "nvfp4", "iso3"}};

// Cold-mode packing: e8 STILL first (keeps its verified leading window), then cold on
// the next-shallow block; deep layers keep the hot tail.
inline constexpr std::array<const char*, 7> kKvBitBudgetPackOrderCold{
    {"e8", "cold", "bf16", "int8", "fp8", "nvfp4", "iso3"}};

namespace detail {

[[nodiscard]] inline std::int32_t tier_index(std::string_view name) noexcept {
    for (std::size_t i = 0; i < kKvBitBudgetTiers.size(); ++i) {
        if (name == kKvBitBudgetTiers[i].spec_name) { return static_cast<std::int32_t>(i); }
    }
    return -1;
}

[[nodiscard]] inline std::int32_t pack_index(std::string_view name) noexcept {
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
    const KvBitBudgetSolution& solution, const std::array<KvBitBudgetTier, 6>& ladder) {
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

// Full solution. Structurally IDENTICAL to the Python DP (tools/archkit/kv_bit_budget.py)
// so the two agree byte-for-byte, including tie-breaks:
//   * state key = (bits, cold_used); the e8 count is carried in the surviving path's
//     value (NOT an extra dimension), exactly like the Python counts dict;
//   * states are expanded in first-insertion order and candidates in the ladder order
//     ORDER + ["cold"]; an equal-penalty candidate never displaces an inserted one;
//   * the final winner is the minimum-penalty state in first-insertion order.
// cold_cap > 0 enables the cold pseudo-tier; cold_cap == 0 keeps the candidate set on
// the pre-cold code path (cold_states == 1 makes the cold index inert).
// The DP body takes the ladder as a parameter: the default ladder carries the shipped
// per-tier penalties, while the scored variant below hands in a ladder whose penalties are
// the weighted sum of a measured quality column and a measured speed column. spec_name is
// never reinterpreted, so every name-keyed path (packing, --kv-layer-storage emission)
// works unchanged.
[[nodiscard]] inline KvBitBudgetSolution kv_bit_budget_solve_impl(
    std::int32_t layers, double budget_bits, const std::array<KvBitBudgetTier, 6>& ladder,
    std::int32_t e8_limit = kKvBitBudgetE8LayerLimit, std::int32_t cold_cap = 0,
    std::int32_t cold_bits_x100 = kKvBitBudgetColdBitsX100) {
    if (layers <= 0) { throw std::invalid_argument("kv-bit-budget: layer count must be positive"); }
    if (!(budget_bits > 0.0)) {
        throw std::invalid_argument("kv-bit-budget: bit budget must be positive");
    }
    if (e8_limit < 0) { e8_limit = 0; }
    if (e8_limit > layers) { e8_limit = layers; }
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
    // Surviving path's e8 usage per state (the Python tool carries it inside `counts`).
    std::vector<std::int8_t> e8_count(states, 0);
    std::vector<std::int8_t> next_e8_count(states, 0);
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
        std::fill(next_e8_count.begin(), next_e8_count.end(), 0);
        order_next.clear();
        auto& layer_choice = choice[static_cast<std::size_t>(layer)];
        for (const std::size_t from : order_cur) {
            const double base = current[from];
            if (base >= kInf) { continue; }
            const std::int32_t from_e8 = e8_count[from];
            const std::int32_t from_bits = static_cast<std::int32_t>(from / cold_states);
            const std::int32_t from_cold = static_cast<std::int32_t>(from % cold_states);
            for (std::size_t t = 0; t < ladder.size(); ++t) {
                const KvBitBudgetTier& tier = ladder[t];
                const bool is_e8 = static_cast<std::int32_t>(t) == 4;  // ladder slot of "e8"
                if (is_e8 && from_e8 >= e8_limit) { continue; }
                const std::int32_t new_bits = from_bits + tier.bits_x100;
                if (new_bits > capacity) { continue; }
                const std::size_t to = static_cast<std::size_t>(new_bits) * cold_states + from_cold;
                // Accumulate in the Python float domain (x100/100.0 == the Python literal),
                // in path order: the tool's comparisons and printed penalties are float
                // sums, and near-ties resolve identically only if the arithmetic matches.
                const double value = base + tier.penalty_x100 / 100.0;
                if (next[to] >= kInf) {
                    next[to] = value;
                    next_e8_count[to] = static_cast<std::int8_t>(from_e8 + (is_e8 ? 1 : 0));
                    layer_choice[to] = static_cast<std::int8_t>(t);
                    order_next.push_back(to);
                } else if (value < next[to]) {
                    next[to] = value;
                    next_e8_count[to] = static_cast<std::int8_t>(from_e8 + (is_e8 ? 1 : 0));
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
                    next_e8_count[to] = static_cast<std::int8_t>(from_e8);
                    layer_choice[to] = static_cast<std::int8_t>(kKvBitBudgetColdTierIndex);
                    order_next.push_back(to);
                } else if (value < next[to]) {
                    next[to] = value;
                    next_e8_count[to] = static_cast<std::int8_t>(from_e8);
                    layer_choice[to] = static_cast<std::int8_t>(kKvBitBudgetColdTierIndex);
                }
            }
        }
        current.swap(next);
        e8_count.swap(next_e8_count);
        order_cur.swap(order_next);
    }

    // Pick the cheapest reachable final state, first-inserted wins on ties (Python scan).
    std::size_t best = states;
    double best_penalty = kInf;
    for (const std::size_t state : order_cur) {
        if (current[state] < best_penalty) {
            best_penalty = current[state];
            best = state;
        }
    }
    if (best == states) {
        throw std::invalid_argument("kv-bit-budget: no feasible allocation for this budget");
    }
    const std::int32_t best_bits = static_cast<std::int32_t>(best / cold_states);
    const std::int32_t best_cold = static_cast<std::int32_t>(best % cold_states);

    // Backtrack to per-layer tier indices (e8 is not part of the state walk; the guard
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

    // Pack: e8 first (cold mode: e8, then cold, then the rest), then emit layer ranges.
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
    return solution;
}

// Returns the --kv-layer-storage spec covering `layers` full-attention layers (thin
// wrapper; signature-compatible with the pre-cold header, cold_cap defaults to off).
[[nodiscard]] inline KvBitBudgetSolution kv_bit_budget_solve(
    std::int32_t layers, double budget_bits,
    std::int32_t e8_limit = kKvBitBudgetE8LayerLimit, std::int32_t cold_cap = 0,
    std::int32_t cold_bits_x100 = kKvBitBudgetColdBitsX100) {
    return kv_bit_budget_solve_impl(layers, budget_bits, kKvBitBudgetTiers, e8_limit, cold_cap,
                                    cold_bits_x100);
}

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
    std::int32_t layers, double budget_bits, const std::array<KvBitBudgetTier, 6>& ladder,
    std::int32_t e8_limit = kKvBitBudgetE8LayerLimit, std::int32_t cold_cap = 0,
    std::int32_t cold_bits_x100 = kKvBitBudgetColdBitsX100,
    bool allow_net_increase = false) {
    KvBitBudgetSolution cold_plan =
        kv_bit_budget_solve_impl(layers, budget_bits, ladder, e8_limit, cold_cap, cold_bits_x100);
    if (cold_plan.cold_layers == 0) { return cold_plan; }  // nothing to audit
    // The cold-off baseline can be absent in principle (only with a caller-supplied ladder
    // whose every row costs more than a cold slot): then there is no all-hot plan to compare
    // against and cold is what makes the budget fit, so the plan stands with the reason stated
    // instead of a fabricated delta.
    KvBitBudgetSolution hot_plan;
    bool hot_feasible = true;
    try {
        hot_plan = kv_bit_budget_solve_impl(layers, budget_bits, ladder, e8_limit, 0,
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
                "nvfp4/iso3 (+320 B) and e8 (+832 B), while int8 (-7360 B), fp8 (-7872 B) and "
                "bf16 (-23232 B) really save";
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
    std::int32_t e8_limit = kKvBitBudgetE8LayerLimit, std::int32_t cold_cap = 0,
    std::int32_t cold_bits_x100 = kKvBitBudgetColdBitsX100,
    bool allow_net_increase = false) {
    return kv_bit_budget_solve_impl_audited(layers, budget_bits, kKvBitBudgetTiers, e8_limit,
                                            cold_cap, cold_bits_x100, allow_net_increase);
}

[[nodiscard]] inline std::string kv_bit_budget_spec(std::int32_t layers, double budget_bits,
                                                    std::int32_t e8_limit =
                                                        kKvBitBudgetE8LayerLimit,
                                                    std::int32_t cold_cap = 0) {
    return kv_bit_budget_solve(layers, budget_bits, e8_limit, cold_cap).spec;
}

// ---------------------------------------------------------------------------
// Separable per-range bit ceilings ("分开约束"): the user may cap different layer
// ranges independently, e.g. "0-7:8,8-63:4.5" (leading layers 8 bits, the rest 4.5).
//
// Optimality: the objective is the sum of per-layer penalties and the constraints are
// per-range capacities, so the feasible set is the Cartesian product of the ranges'
// feasible sets and minimising each range independently attains the global optimum.
// Implementation is therefore the single-budget DP above, run once per range, with the
// emitted layer indices shifted into absolute coordinates. The e8 leading-window limit
// is applied on ABSOLUTE layer indices, and the cold pool is shared: ranges are solved
// deepest-first (the pack order already sends cold to the deep block) and each range may
// only spend what the pool has left, so the total cold layers can never exceed cold_cap.
// ---------------------------------------------------------------------------
// Two-score selection: speed and quality are separate, measured columns and the operator
// chooses the trade-off with a weight. Both columns are "lower is better, x100" and must be
// normalised onto ONE scale by the caller (quality = relative output error x100, speed =
// relative time cost x100 with the fastest tier at 0), because the weight only means
// anything if the two are commensurable.
struct KvTierScoreRow {
    std::int32_t quality_x100 = 0;   // measured precision loss (0 = lossless)
    std::int32_t speed_x100   = 0;   // measured time cost    (0 = fastest available path)
};
using KvTierScoreTable = std::array<KvTierScoreRow, 6>;   // same order as the default ladder

// Combined ladder: penalty = round(w*quality + (1-w)*speed), clamped to >= 0.
[[nodiscard]] inline std::array<KvBitBudgetTier, 6>
kv_bit_budget_scored_ladder(const KvTierScoreTable& scores, double quality_weight) {
    if (!(quality_weight >= 0.0) || !(quality_weight <= 1.0)) {
        throw std::invalid_argument("kv-bit-budget: quality weight must be in [0,1]");
    }
    std::array<KvBitBudgetTier, 6> ladder = kKvBitBudgetTiers;
    for (std::size_t i = 0; i < ladder.size(); ++i) {
        const double combined = quality_weight * scores[i].quality_x100 +
                                (1.0 - quality_weight) * scores[i].speed_x100;
        ladder[i].penalty_x100 = static_cast<std::int32_t>(std::nearbyint(combined));
        if (ladder[i].penalty_x100 < 0) { ladder[i].penalty_x100 = 0; }
    }
    return ladder;
}

[[nodiscard]] inline KvBitBudgetSolution kv_bit_budget_solve_scored(
    std::int32_t layers, double budget_bits, const KvTierScoreTable& scores,
    double quality_weight, std::int32_t e8_limit = kKvBitBudgetE8LayerLimit,
    std::int32_t cold_cap = 0) {
    return kv_bit_budget_solve_impl(layers, budget_bits,
                                    kv_bit_budget_scored_ladder(scores, quality_weight),
                                    e8_limit, cold_cap);
}

// Provisional default score table. The quality column carries the shipped priors until the
// automated calibration (tools/archkit/kv_tier_matrix.py -> JSON) supplies measured values;
// the speed column starts from the measured uniform-tier decode rates once they exist
// (fastest path = 0, others scaled by their relative per-token time cost). Both columns are
// x100 and normalised onto one scale by whoever builds the table.
[[nodiscard]] inline KvTierScoreTable kv_bit_budget_default_scores() {
    // speed_x100 is (fastest/v - 1)*100 from the measured uniform-tier rates below, so the
    // fastest path is 0 and slower tiers scale up. quality_x100 still carries the shipped
    // priors until the offline replay (tools/calib) supplies measured per-layer errors.
    return KvTierScoreTable{{
        {/*bf16 */ 0, 7},    // 80.7 tok/s
        {/*int8 */ 2, 0},    // 86.2 tok/s - the measured fastest
        {/*fp8  */ 3, 0},    // does not run; excluded by the ladder's cost anyway
        {/*nvfp4*/ 30, 114}, // 40.3 tok/s: QK runs twice + software V decode
        {/*e8   */ 8, 218},  // 27.1 tok/s: lattice projection + nibble unpack
        {/*iso3 */ 200, 114},// == nvfp4
    }};
}

// "tier quality_x100 speed_x100" per line, '#' comments; must name all six tiers.
[[nodiscard]] inline KvTierScoreTable kv_bit_budget_parse_scores(std::string_view text) {
    KvTierScoreTable table = kv_bit_budget_default_scores();
    std::array<bool, 6> seen{};
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
        if (quality < 0.0 || speed < 0.0 || quality > 10000.0 || speed > 10000.0) {
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
// e8 window (e8 is a leading-layer window in ABSOLUTE coordinates, so a range that starts
// above the limit gets none).
template <typename Solve>
[[nodiscard]] inline std::string kv_bit_budget_ranges_driver(
    std::int32_t layers, const std::vector<KvBitBudgetRange>& ranges, std::int32_t e8_limit,
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
        const std::int32_t within     = e8_limit - range.first;
        const std::int32_t local_e8   = within <= 0 ? 0 : (within < count ? within : count);
        const KvBitBudgetSolution solved = solve(count, range.bits, local_e8, cold_left);
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
                          std::int32_t e8_limit = kKvBitBudgetE8LayerLimit,
                          std::int32_t cold_cap = 0) {
    return kv_bit_budget_ranges_driver(
        layers, ranges, e8_limit, cold_cap,
        [](std::int32_t count, double bits, std::int32_t local_e8, std::int32_t cold_left) {
            return kv_bit_budget_solve(count, bits, local_e8, cold_left);
        });
}

// The scored ladder under per-range ceilings. Without this, a caller that passes both knobs
// silently loses the ceilings: the range form leaves the scalar budget at 0, so a scored run
// would resolve every layer against a zero-bit ceiling instead of the range's own.
[[nodiscard]] inline std::string kv_bit_budget_scored_ranges(
    std::int32_t layers, const std::vector<KvBitBudgetRange>& ranges,
    const KvTierScoreTable& scores, double quality_weight,
    std::int32_t e8_limit = kKvBitBudgetE8LayerLimit, std::int32_t cold_cap = 0) {
    return kv_bit_budget_ranges_driver(
        layers, ranges, e8_limit, cold_cap,
        [&scores, quality_weight](std::int32_t count, double bits, std::int32_t local_e8,
                                 std::int32_t cold_left) {
            return kv_bit_budget_solve_scored(count, bits, scores, quality_weight, local_e8,
                                              cold_left);
        });
}

} // namespace ninfer::product
