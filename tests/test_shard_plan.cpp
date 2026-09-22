// test_shard_plan.cpp -- unit predicates for src/core/shard_plan.h, and the ONE
// counterexample the multi-device KV question needs.
//
// The load-bearing part is section 4 below: "can a page's residency bit still be ONE bit
// once the KV cache is split across cards?" Everything else here is arithmetic and naming.
//
// Host-only. No CUDA include, no device, no collective. The counterexample is a closed-form
// attention over two heads with exact powers of two, so its expected values are written as
// literals rather than produced by the code under test.

#include "core/shard_plan.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace {

using ninfer::multi::axis_name;
using ninfer::multi::global_admit;
using ninfer::multi::ModelGeometry;
using ninfer::multi::page_content_is_layer_complete;
using ninfer::multi::ParallelAxis;
using ninfer::multi::plan_shards;
using ninfer::multi::Range;
using ninfer::multi::read_free_predicate_factors_over;
using ninfer::multi::recall_record_shape_survives;
using ninfer::multi::residency_decision_needs_replication;
using ninfer::multi::residency_state;
using ninfer::multi::ResidencyState;
using ninfer::multi::sharded_artifact_name;
using ninfer::multi::sharded_name_acceptance;
using ninfer::multi::ShardPlan;
using ninfer::multi::WorldShape;
using ninfer::multi::world_shape_refusal;
using ninfer::multi::world_shape_valid;

int failures = 0;
int checks   = 0;
int reported = 0;
constexpr int kReportLimit = 25;

void check(bool condition, const std::string& what) {
    ++checks;
    if (!condition) {
        ++failures;
        if (reported < kReportLimit) {
            ++reported;
            std::cout << "FAIL: " << what << "\n";
        } else if (reported == kReportLimit) {
            ++reported;
            std::cout << "FAIL: ... further failures suppressed; the count below is exact\n";
        }
    }
}

constexpr std::string_view kRowScaleSuffix = ".kvrowscale.bin";

// Declared here, defined in section 2, so section 1 can use it. (It was defined after its
// first use, which made the whole translation unit fail to compile -- and the harness then
// silently ran a STALE binary and reported a false green. See the stale-binary guard in
// sh/MULTIDEV_run_all.sh.)
ModelGeometry geometry();

// ---------------------------------------------------------------------------
// 1. World shape validation
// ---------------------------------------------------------------------------

void test_world_shape() {
    check(world_shape_valid(WorldShape{1, 0, ParallelAxis::None}), "the identity world is valid");
    const ShardPlan single = plan_shards(WorldShape{1, 0, ParallelAxis::None}, geometry());
    check(single.ok && single.residency_decision_is_replicated() &&
              single.page_is_layer_complete(),
          "the single-device plan must be both replicated and layer-complete");
    check(!world_shape_refusal(WorldShape{0, 0, ParallelAxis::None}).empty(),
          "world_size 0 must be refused");
    check(world_shape_refusal(WorldShape{2, 2, ParallelAxis::Tensor}).find("outside") !=
              std::string::npos,
          "rank >= world_size must be refused by name");

    // The refusal that matters: a multi-device world MUST name its axis, because the KV
    // granularity contract depends on which axis it is.
    const std::string unnamed = world_shape_refusal(WorldShape{4, 0, ParallelAxis::None});
    check(unnamed.find("axis") != std::string::npos,
          "a 4-device world with axis=none must be refused naming the axis, got: " + unnamed);
    const std::string named_but_single =
        world_shape_refusal(WorldShape{1, 0, ParallelAxis::Tensor});
    check(!named_but_single.empty(),
          "axis=tp with world_size 1 must be refused as a contradiction");
}

// ---------------------------------------------------------------------------
// 2. Shard arithmetic
// ---------------------------------------------------------------------------

// A synthetic GQA geometry. `kv_heads = 4` is the number the tree already carries
// (tools/archkit/qpn_port/placement_planner.py: `qwen27_layers(..., kv_heads=4,
// head_dim=256)`); the q_heads / text_layers / weight_columns values here are chosen to be
// divisible so the arithmetic is exercised, not measured claims.
ModelGeometry geometry() {
    ModelGeometry g;
    g.text_layers    = 64;
    g.q_heads        = 32;
    g.kv_heads       = 4;
    g.head_dim       = 256;
    g.weight_columns = 65536;
    return g;
}

void test_tensor_axis_split() {
    for (const std::uint32_t world : {2U, 4U}) {
        const ShardPlan plan = plan_shards(WorldShape{world, 0, ParallelAxis::Tensor}, geometry());
        check(plan.ok, "tp=" + std::to_string(world) + " must be legal: " + plan.reason);
        check(plan.kv_heads == Range{0, 4 / world},
              "rank 0 of tp=" + std::to_string(world) + " must own the first kv heads");
        check(plan.text_layers == Range{0, 64},
              "tp must leave TEXT LAYERS whole on every rank, got [" +
                  std::to_string(plan.text_layers.begin) + "," +
                  std::to_string(plan.text_layers.end) + ")");
    }
    const ShardPlan last = plan_shards(WorldShape{4, 3, ParallelAxis::Tensor}, geometry());
    check(last.ok && last.kv_heads == Range{3, 4}, "rank 3 of tp=4 must own the last kv head");
    check(last.weight_columns == Range{49152, 65536},
          "rank 3 of tp=4 must own the last quarter of N");
    // The ShardPlan's own two verdict fields, asserted directly: the injection harness found
    // that a mutant of residency_decision_is_replicated() survived every other check, because
    // the axis tests only exercised the free predicates.
    for (const std::uint32_t world : {2U, 4U}) {
        const ShardPlan plan = plan_shards(WorldShape{world, 0, ParallelAxis::Tensor}, geometry());
        check(plan.ok, "tp=" + std::to_string(world) + " must be legal: " + plan.reason);
        check(!plan.residency_decision_is_replicated(),
              "ShardPlan::residency_decision_is_replicated() must be FALSE on the tensor axis");
        check(plan.page_is_layer_complete(),
              "ShardPlan::page_is_layer_complete() must be TRUE on the tensor axis");
    }

    // THE CONSTRAINT THAT ACTUALLY BITES: with 4 KV heads, tp can never exceed 4 on the
    // HEAD axis. This is not a soft preference; a larger world would cut a GQA group.
    const ShardPlan over = plan_shards(WorldShape{8, 0, ParallelAxis::Tensor}, geometry());
    check(!over.ok, "tp=8 with kv_heads=4 must be refused");
    check(over.reason.find("GQA group") != std::string::npos,
          "the tp=8 refusal must name the GQA group, got: " + over.reason);

    ModelGeometry bad_n = geometry();
    bad_n.weight_columns = 65535;
    const ShardPlan n_plan = plan_shards(WorldShape{2, 0, ParallelAxis::Tensor}, bad_n);
    check(!n_plan.ok && n_plan.reason.find("weight N") != std::string::npos,
          "an N that does not divide must be refused naming N, got: " + n_plan.reason);

    ModelGeometry bad_q = geometry();
    bad_q.q_heads        = 30; // not a multiple of kv_heads=4
    const ShardPlan q_plan = plan_shards(WorldShape{2, 0, ParallelAxis::Tensor}, bad_q);
    check(!q_plan.ok && q_plan.reason.find("GQA group size") != std::string::npos,
          "q_heads not a multiple of kv_heads must be refused, got: " + q_plan.reason);
}

void test_pipeline_axis_split() {
    const ShardPlan even = plan_shards(WorldShape{4, 2, ParallelAxis::Pipeline}, geometry());
    check(even.ok && even.text_layers == Range{32, 48}, "pp=4 stage 2 must own layers 32..47");
    check(even.kv_heads == Range{0, 4},
          "pp must leave KV HEADS whole on every stage, got [" +
              std::to_string(even.kv_heads.begin) + "," + std::to_string(even.kv_heads.end) +
              ")");
    check(even.residency_decision_is_replicated(),
          "ShardPlan::residency_decision_is_replicated() must be TRUE on the pipeline axis");
    check(!even.page_is_layer_complete(),
          "ShardPlan::page_is_layer_complete() must be FALSE on the pipeline axis");

    // An uneven split is legal on the layer axis (unlike the head axis) because no single
    // layer is ever cut.
    const ShardPlan uneven = plan_shards(WorldShape{3, 0, ParallelAxis::Pipeline}, geometry());
    check(uneven.ok && uneven.text_layers == Range{0, 22},
          "pp=3 stage 0 must take 22 layers (64 = 22 + 21 + 21)");
    // Contiguity and total coverage across every stage.
    std::uint32_t expected_begin = 0;
    for (std::uint32_t rank = 0; rank < 3; ++rank) {
        const ShardPlan stage = plan_shards(WorldShape{3, rank, ParallelAxis::Pipeline}, geometry());
        check(stage.ok && stage.text_layers.begin == expected_begin,
              "pp=3 stage " + std::to_string(rank) + " must abut the previous stage at " +
                  std::to_string(expected_begin));
        expected_begin = stage.text_layers.end;
    }
    check(expected_begin == 64, "pp=3 stages must cover all 64 layers, covered " +
                                    std::to_string(expected_begin));

    const ShardPlan too_many =
        plan_shards(WorldShape{65, 0, ParallelAxis::Pipeline}, geometry());
    check(!too_many.ok && too_many.reason.find("smaller than") != std::string::npos,
          "pp > layers must be refused, got: " + too_many.reason);

    // An empty stage is what makes the read-free conjunction's identity matter, so a legal
    // plan must never contain one.
    for (std::uint32_t world = 1; world <= 64; ++world) {
        for (std::uint32_t rank = 0; rank < world; ++rank) {
            const ShardPlan stage =
                plan_shards(WorldShape{world, rank, ParallelAxis::Pipeline}, geometry());
            if (!stage.ok) { continue; }
            check(!stage.text_layers.empty(),
                  "a legal pp plan must never hand a stage zero layers (world=" +
                      std::to_string(world) + " rank=" + std::to_string(rank) + ")");
        }
    }
}

// ---------------------------------------------------------------------------
// 3. Per-rank persistence naming
// ---------------------------------------------------------------------------

void test_sharded_naming() {
    const WorldShape single{1, 0, ParallelAxis::None};
    check(sharded_artifact_name("m.kvrowscale.bin", single, kRowScaleSuffix) ==
              "m.kvrowscale.bin",
          "a single-device world must leave the existing artifact name untouched");

    const WorldShape tp{4, 1, ParallelAxis::Tensor};
    const std::string named = sharded_artifact_name("m.kvrowscale.bin", tp, kRowScaleSuffix);
    check(named == "m.tp1of4.kvrowscale.bin",
          "tp rank 1 of 4 must produce m.tp1of4.kvrowscale.bin, got " + named);
    check(named.size() > kRowScaleSuffix.size() &&
              named.compare(named.size() - kRowScaleSuffix.size(), kRowScaleSuffix.size(),
                            kRowScaleSuffix) == 0,
          "the suffix must stay at the end so existing globs still match, got " + named);

    // The round trip, and every refusal the reader owes.
    check(sharded_name_acceptance(named, tp).empty(),
          "a table written by this shard must be accepted");
    check(sharded_name_acceptance(named, WorldShape{4, 2, ParallelAxis::Tensor})
              .find("different shard") != std::string::npos,
          "a table from another rank must be REFUSED");
    check(sharded_name_acceptance(named, WorldShape{2, 0, ParallelAxis::Tensor})
              .find("different shard") != std::string::npos,
          "a table from a different world size must be REFUSED");
    check(sharded_name_acceptance("m.kvrowscale.bin", tp).find("no shard tag") !=
              std::string::npos,
          "an UNSHARDED table in a multi-device run must be REFUSED");
    check(sharded_name_acceptance("m.kvrowscale.bin", single).empty(),
          "an unsharded table in a single-device run must be accepted");
    check(sharded_name_acceptance(named, single).empty() == false,
          "a sharded table in a single-device run must be REFUSED");
    check(sharded_name_acceptance(
              sharded_artifact_name("m.kvrowscale.bin",
                                    WorldShape{4, 1, ParallelAxis::Pipeline}, kRowScaleSuffix),
              tp)
              .find("written on axis pp") != std::string::npos,
          "a pp table in a tp run must be REFUSED naming the axis");
    // Malformed tags must not be silently treated as unsharded.
    check(sharded_name_acceptance("m.tpxof4.kvrowscale.bin", tp).empty() == false,
          "a malformed tag must be refused, not ignored");
}

// ---------------------------------------------------------------------------
// 4. The residency bit across a partition  <-- the counterexample
// ---------------------------------------------------------------------------

void test_residency_states() {
    check(residency_state({1, 1}) == ResidencyState::FullyResident, "{1,1} is fully resident");
    check(residency_state({0, 0}) == ResidencyState::FullyAbsent, "{0,0} is fully absent");
    check(residency_state({1, 0}) == ResidencyState::PartiallyResident,
          "{1,0} is partially resident -- a state the model does not define");
    check(residency_state({0, 1}) == ResidencyState::PartiallyResident,
          "{0,1} is partially resident");
    check(global_admit({1, 1}) && !global_admit({1, 0}) && !global_admit({0, 0}),
          "global_admit must be the AND across ranks");
    check(residency_decision_needs_replication(ParallelAxis::Tensor),
          "the tensor axis needs the replicated admit decision");
    check(!residency_decision_needs_replication(ParallelAxis::Pipeline),
          "the pipeline axis does not (each stage's verdict is complete for its own pages)");
    check(!residency_decision_needs_replication(ParallelAxis::None),
          "single device needs no replication");
}

// The read-free predicate, RE-DERIVED from the text of
// src/targets/qwen3_6/impl/runtime/cold_host_tier.h rather than included, so this test does
// not share the subject's own code path:
//   "layer L attends [frontier+1-W_L, frontier], so page p is unread by L iff
//    (p+1)*P + W_L <= frontier. A layer with window 0 is full attention and reads every
//    committed token, so it makes the predicate false for every page."
// It is a pure CONJUNCTION over layers. That is the whole point of the next test.
bool read_free(std::uint32_t page, std::uint32_t page_tokens, std::uint32_t frontier,
               const std::vector<std::uint32_t>& windows) {
    if (page_tokens == 0 || windows.empty()) { return false; }
    const std::uint64_t page_end = (static_cast<std::uint64_t>(page) + 1U) * page_tokens;
    for (const std::uint32_t window : windows) {
        if (window == 0) { return false; }
        if (page_end + window > frontier) { return false; }
    }
    return true;
}

void test_read_free_factors_over_a_layer_partition() {
    const std::vector<std::uint32_t> window_values{0, 1, 2, 4};
    std::vector<std::vector<std::vector<std::uint32_t>>> partitions;
    // All 2-colourings of 3 layers (a 2-stage pipeline), plus the all-singleton partition
    // (a 3-stage pipeline).
    for (std::uint32_t a = 0; a < 2; ++a) {
        for (std::uint32_t b = 0; b < 2; ++b) {
            for (std::uint32_t c = 0; c < 2; ++c) {
                std::vector<std::vector<std::uint32_t>> groups(2);
                groups[a].push_back(0);
                groups[b].push_back(1);
                groups[c].push_back(2);
                partitions.push_back(groups);
            }
        }
    }
    partitions.push_back({{0}, {1}, {2}});

    std::size_t exhaustive_cases = 0;
    for (const std::uint32_t page : {0U, 1U, 2U}) {
        for (const std::uint32_t frontier : {0U, 4U, 8U, 12U, 16U, 20U, 24U}) {
            for (const std::uint32_t w0 : window_values) {
                for (const std::uint32_t w1 : window_values) {
                    for (const std::uint32_t w2 : window_values) {
                        const std::vector<std::uint32_t> windows{w0, w1, w2};
                        const bool global      = read_free(page, 4, frontier, windows);
                        for (const auto& groups : partitions) {
                            bool grouped = true;
                            for (const auto& group : groups) {
                                // A group with no layers contributes the IDENTITY of the
                                // conjunction, not a refusal. This is not a convenience: the
                                // re-derived predicate above returns false for an empty window
                                // list (matching the header's `layer_windows.empty()` guard),
                                // so a stage that owned no layer would veto every eviction.
                                // plan_shards() refuses to build such a stage at all, which
                                // the empty-stage check below pins.
                                if (group.empty()) { continue; }
                                std::vector<std::uint32_t> subset;
                                for (const std::uint32_t index : group) {
                                    subset.push_back(windows[index]);
                                }
                                grouped = grouped && read_free(page, 4, frontier, subset);
                            }
                            ++exhaustive_cases;
                            check(global == grouped,
                                  "the read-free predicate must factor over a LAYER "
                                  "partition, and it did not: page=" + std::to_string(page) +
                                      " frontier=" + std::to_string(frontier) + " windows=" +
                                      std::to_string(w0) + "/" + std::to_string(w1) + "/" +
                                      std::to_string(w2));
                        }
                    }
                }
            }
        }
    }
    check(exhaustive_cases > 3000,
          "the factoring sweep must be exhaustive, ran only " +
              std::to_string(exhaustive_cases) + " cases");
    check(read_free_predicate_factors_over(ParallelAxis::Pipeline),
          "the predicate must be declared to factor over the pipeline axis");
    check(!read_free_predicate_factors_over(ParallelAxis::Tensor),
          "the predicate must NOT be declared to factor over the tensor axis");
}

// THE COUNTEREXAMPLE. Two query heads, two KV heads, tp = 2, head_dim = 1, so rank r owns
// exactly head r. Every number is an exact power of two, so the expected values below are
// literals, not the subject's own output.
//
//   head 0: K = [ln 4, 0], V = [4, 1]   -> present: softmax([ln4,0]) = [0.8, 0.2] -> 3.4
//   head 1: K = [0, ln 4], V = [1, 4]   -> present: softmax([0,ln4]) = [0.2, 0.8] -> 3.4
//   page absent for a head: only the current token remains -> out = V_cur : 1.0 / 4.0
//
// Why this is the right model of tensor parallelism: rank r computes its OWN heads'
// attention COMPLETELY (a rank holds exactly the KV heads its queries attend to), so there
// is no softmax combine across ranks. What crosses ranks is the ROW-PARALLEL o_proj sum,
// and that is where two different contexts get added into one output element.
constexpr double kLn4 = 1.3862943611198906;

struct HeadKv {
    double k_page;
    double k_cur;
    double v_page;
    double v_cur;
};

const HeadKv kHeads[2] = {
    {kLn4, 0.0, 4.0, 1.0},
    {0.0, kLn4, 1.0, 4.0},
};

double attention_for_head(const HeadKv& head, bool page_resident) {
    const double keys[2]   = {head.k_page, head.k_cur};
    const double values[2] = {head.v_page, head.v_cur};
    const int first        = page_resident ? 0 : 1;
    double max_score       = -1e300;
    for (int i = first; i < 2; ++i) { max_score = std::max(max_score, keys[i]); }
    double denominator = 0.0;
    double numerator   = 0.0;
    for (int i = first; i < 2; ++i) {
        const double weight = std::exp(keys[i] - max_score);
        denominator += weight;
        numerator += weight * values[i];
    }
    return numerator / denominator;
}

// Single-card semantics: the page is a property of the SEQUENCE, so every head sees the
// same context. This is the only state the engine can express today.
std::vector<double> single_card_attention(bool page_resident) {
    return {attention_for_head(kHeads[0], page_resident),
            attention_for_head(kHeads[1], page_resident)};
}

// Tensor-parallel semantics, WITH and WITHOUT the replicated admit, behind one switch so the
// fix is one line and the difference is the whole test.
std::vector<double> tp_attention(const std::vector<std::uint8_t>& per_rank_admit,
                                 bool replicate_admit) {
    std::vector<std::uint8_t> effective = per_rank_admit;
    if (replicate_admit) {
        const bool admit = global_admit(per_rank_admit);
        for (std::uint8_t& decision : effective) { decision = admit ? 1 : 0; }
    }
    return {attention_for_head(kHeads[0], effective[0] != 0),
            attention_for_head(kHeads[1], effective[1] != 0)};
}

bool agrees(double a, double b) { return std::fabs(a - b) < 1e-9; }

bool contexts_agree_across_heads(const std::vector<std::uint8_t>& per_rank_admit) {
    return residency_state(per_rank_admit) != ResidencyState::PartiallyResident;
}

void test_head_split_breaks_page_atomicity() {
    const std::vector<double> present = single_card_attention(true);
    const std::vector<double> absent  = single_card_attention(false);
    check(agrees(present[0], 3.4) && agrees(present[1], 3.4),
          "the two legal states must be the literals 3.4 / 3.4 for 'present'");
    check(agrees(absent[0], 1.0) && agrees(absent[1], 4.0),
          "the two legal states must be the literals 1.0 / 4.0 for 'absent'");

    // Agreement is the invariant the single-card engine gets for free: the context belongs
    // to the sequence, not to the head.
    check(contexts_agree_across_heads({1, 1}) && contexts_agree_across_heads({0, 0}),
          "a unanimous decision keeps the context a property of the sequence");

    // Unanimous TP reproduces the single-card states -- so the SPLIT is not what breaks; the
    // DISAGREEMENT is.
    const auto tp_all_present = tp_attention({1, 1}, false);
    const auto tp_all_absent  = tp_attention({0, 0}, false);
    check(agrees(tp_all_present[0], present[0]) && agrees(tp_all_present[1], present[1]),
          "tp with both ranks admitting must reproduce the present state");
    check(agrees(tp_all_absent[0], absent[0]) && agrees(tp_all_absent[1], absent[1]),
          "tp with both ranks refusing must reproduce the absent state");

    // THE BREAK.
    for (const auto& split : {std::vector<std::uint8_t>{1, 0}, std::vector<std::uint8_t>{0, 1}}) {
        check(!contexts_agree_across_heads(split),
              "a split admit must be reported as partially resident");
        const auto out = tp_attention(split, false);
        const bool differs_from_present =
            !agrees(out[0], present[0]) || !agrees(out[1], present[1]);
        const bool differs_from_absent =
            !agrees(out[0], absent[0]) || !agrees(out[1], absent[1]);
        check(differs_from_present && differs_from_absent,
              "a split admit must produce a result that is NEITHER legal state; got ["
              + std::to_string(out[0]) + ", " + std::to_string(out[1]) + "]");
        // Not even a benign interpolation: {1,0} sums ABOVE the present state and {0,1}
        // BELOW the absent one, so no "it is roughly one of the two" escape exists.
        const double sum = out[0] + out[1];
        const bool outside = sum > present[0] + present[1] + 1e-9 ||
                             sum < absent[0] + absent[1] - 1e-9;
        check(outside,
              "the split output must fall outside the [absent, present] interval, got sum " +
                  std::to_string(sum));
    }

    // THE FIX: route every rank's verdict through global_admit. The first-rank-only admit
    // now collapses onto a legal state, and no split can be expressed.
    for (const auto& split : {std::vector<std::uint8_t>{1, 0}, std::vector<std::uint8_t>{0, 1},
                              std::vector<std::uint8_t>{0, 0}, std::vector<std::uint8_t>{1, 1}}) {
        const auto fixed = tp_attention(split, true);
        const bool lawful = global_admit(split);
        const auto expected = single_card_attention(lawful);
        check(agrees(fixed[0], expected[0]) && agrees(fixed[1], expected[1]),
              "with the replicated admit, every per-rank verdict must collapse onto the "
              "matching legal state");
    }
}

// ---------------------------------------------------------------------------
// 5. Which axis keeps the page's declared unit
// ---------------------------------------------------------------------------

void test_which_axis_keeps_the_declared_unit() {
    // cold_host_tier.h: "one logical page ACROSS ALL TEXT LAYERS, never a slice of one".
    check(page_content_is_layer_complete(ParallelAxis::None), "single device: layer-complete");
    check(page_content_is_layer_complete(ParallelAxis::Tensor),
          "a head split leaves every layer on every rank, so each rank's page IS "
          "layer-complete -- the tensor axis costs the DECISION, not the LAYOUT");
    check(!page_content_is_layer_complete(ParallelAxis::Pipeline),
          "a layer split puts one page's layers on different stages, so the declared unit "
          "cannot be one resident object anywhere");

    // turn_recall_journal.h: one `layer_bytes` per page, and RecallCodec::Mixed is refused by
    // a static_assert. A head split scales every layer's stride by the same 1/world_size, so
    // the record shape survives; a layer split makes Mixed REACHABLE, because different
    // stages may legitimately pick different per-layer codecs (which is exactly what
    // `--kv-layer-storage 0-7:rk4v4,4-7:int8` does today).
    check(recall_record_shape_survives(ParallelAxis::Tensor),
          "a head split keeps ONE stride for the page, so the recall record shape survives");
    check(!recall_record_shape_survives(ParallelAxis::Pipeline),
          "a layer split makes per-layer mixed codecs reachable, which the recall journal "
          "refuses at compile time");

    // The cost ordering, asserted so the recommendation cannot drift from the code.
    check(residency_decision_needs_replication(ParallelAxis::Tensor) &&
              page_content_is_layer_complete(ParallelAxis::Tensor) &&
              recall_record_shape_survives(ParallelAxis::Tensor),
          "TP on heads: LAYOUT survives, only the DECISION needs replication");
    check(!residency_decision_needs_replication(ParallelAxis::Pipeline) &&
              !page_content_is_layer_complete(ParallelAxis::Pipeline) &&
              !recall_record_shape_survives(ParallelAxis::Pipeline),
          "PP on layers: decision is free, but the declared page unit does NOT survive");
}

} // namespace

int main() {
    test_world_shape();
    test_tensor_axis_split();
    test_pipeline_axis_split();
    test_sharded_naming();
    test_residency_states();
    test_read_free_factors_over_a_layer_partition();
    test_head_split_breaks_page_atomicity();
    test_which_axis_keeps_the_declared_unit();

    std::cout << "shard_plan: " << checks << " checks, " << failures << " failures -> "
              << (failures == 0 ? "PASS" : "FAIL") << "\n";
    return failures == 0 ? 0 : 1;
}
