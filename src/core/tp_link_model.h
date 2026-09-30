#pragma once

// tp_link_model.h -- WHAT A RANK BOUNDARY COSTS, MODELLED FROM A SOURCED LINK AND THE ENGINE'S OWN
// SHARD ARITHMETIC.
//
// WHY THIS FILE EXISTS, AND WHY IT IS NOT INSIDE core/tp_transport.h
// ---------------------------------------------------------------
// Two files already refuse to hold this number, and both refusals are correct:
//
//   * core/tp_transport.h section 5, verbatim: "THE COST OF ANY COLLECTIVE. Not measured, not
//     estimated, not apologised for." Its section 4 says out loud "No cost is computed, no bandwidth
//     is assumed, and nothing here is allowed to report a throughput number."
//   * core/shard_rank_axis.h:549-550, about the attention combine: "The COST of the collective is
//     not checkable here: it is a transport property, and this box has one GPU."
//
// So the transport cost is a THIRD thing, and this file is it: a model, with its inputs named, that
// turns (payload, rank count, link) into a time. It is a COST MODEL and never a measurement --
// `is_measurement` below is `false` as a value rather than as a promise, and the caller prints it.
//
// THE MODEL. A rank boundary transfer costs
//
//     t = latency_per_call  +  payload_bytes / effective_bytes_per_second
//
// The bandwidth term is the load-bearing one for long payloads and it is DERIVED, not guessed: it is
// a link's per-direction figure times the number of links a peer pair actually has. The latency term
// is a knob with its origin named and its uncertainty carried, because the datasheet does not state
// it -- see `LinkSpec` below.
//
// Host-only, like its neighbours: no CUDA header, no device query. Every input is a parameter, so
// the whole model is exercisable on a machine with no GPU at all.

#include "core/shard_plan.h"
#include "core/shard_rank_axis.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <string_view>

namespace ninfer::multi {

// ---------------------------------------------------------------------------
// 1. THE SOURCED LINK
// ---------------------------------------------------------------------------
//
// ⚠ EVERY NUMBER HERE CARRIES ITS ORIGIN. This project has been burned by an invented anchor, so a
// figure without a source is not admissible and the struct makes the source a FIELD.
//
// SOURCE OF THE PER-LINK FIGURE, named exactly:
//
//   NVIDIA, "NVIDIA Tesla V100 GPU Architecture" / Volta Architecture Whitepaper, WP-08608-001_v1.1,
//   August 2017, section on NVLink: "Each link now provides 25 GB/second in each direction. The
//   number of links supported has been increased from four to six, pushing the supported GPU NVLink
//   bandwidth to 300 GB/second."
//
// The three facts that sentence carries, separated because they are routinely conflated:
//   * PER LINK, PER DIRECTION: 25 GB/s. This is the number a rate model must divide a payload by.
//   * LINKS PER GPU: 6. So the per-GPU AGGREGATE is 6 x 25 = 150 GB/s per DIRECTION, and
//     300 GB/s BIDIRECTIONAL. The "300 GB/s" everyone quotes is the BIDIRECTIONAL AGGREGATE and
//     using it as a one-way rate overstates a one-way transfer by 2x.
//   * IT IS A LINK FIGURE, NOT A PEER-PAIR FIGURE. How many of a GPU's 6 links point at ONE peer
//     depends on the topology, which is why `links_to_peer` is a separate field and not folded in.
//
// V100's own "GB" is decimal (10^9), which is the convention NVIDIA's bandwidth figures use; the
// model takes bytes and seconds and never converts a unit the caller did not name.
struct LinkSpec {
    // The card this link models. Named so a reader cannot re-apply the numbers to a different rung.
    std::string_view generation = "nvlink2 (Volta / V100, sm_70)";
    // Per-direction, per-link. SOURCED: Volta whitepaper, 25 GB/s, above.
    double per_link_gigabytes_per_second_per_direction = 25.0;
    // Total links the GPU has. SOURCED: Volta whitepaper, 6, above.
    std::uint32_t links_per_gpu = 6;
    // HOW MANY OF THEM POINT AT ONE PEER. NOT a datasheet number: it is a TOPOLOGY input, and the
    // model refuses to guess it. The two shapes this project actually discusses:
    //   * 2 ranks, every link on the single peer:      links_to_peer = 6  (150 GB/s per direction)
    //   * 4 ranks, hybrid cube-mesh, 6 links over 3 peers: links_to_peer = 2  (50 GB/s per direction)
    // See `v100_hybrid_cube_mesh()` below, which derives the 4-rank figure rather than asserting it.
    std::uint32_t links_to_peer = 6;
    // A FRACTION, not a datasheet number, and it is here because the sourced figure is a SIGNALLING
    // rate and no kernel reaches it. Efficiency is what a caller carries their own measured value
    // in. 1.0 says "the link runs at its source figure", which is the honest default for a MODEL
    // whose job is to bound: the model then reports the BEST case the interconnect permits, which
    // is the floor the owner asked for, not a prediction of a real kernel.
    double efficiency = 1.0;
    // ⚠ THE LATENCY TERM, AND WHY IT IS A KNOB RATHER THAN A NUMBER.
    // The Volta datasheet states a BANDWIDTH and no latency. Latency is a property of the protocol
    // stack (a signal round trip, a driver-staged copy, a collective's two barriers), not of the
    // link, so an invented microsecond figure would be exactly the unsourced anchor this project
    // catches. So: the DEFAULT is 0 and the model's residual is reported as a function of whatever
    // the caller sets. A caller who has a measured value passes it; a caller who does not gets a
    // bandwidth-only model, which is stated rather than hidden.
    // The one thing the number IS anchored to when set: core/tp_transport.h's slotted call has TWO
    // barriers per call (readiness and acknowledgement, `tp_slotted_call_plan()`), so a per-CALL
    // latency is the round trip a caller measures and the model multiplies it by the call count.
    double latency_per_call_nanoseconds = 0.0;
    // Where a non-zero latency came from. Empty is allowed only when the default 0 is in use.
    std::string_view latency_origin = "unset (bandwidth-only model)";
    // ⚠ AND WHETHER THE LATENCY TERM IS A MEASUREMENT OR AN ASSUMPTION, as a VALUE. The two must
    // not be confused in a report, and a caller who cannot say must pass false.
    bool latency_is_measured = false;

    [[nodiscard]] double per_direction_bytes_per_second() const noexcept {
        const double decimal_g = 1.0e9;
        return per_link_gigabytes_per_second_per_direction * decimal_g *
               static_cast<double>(links_to_peer) * efficiency;
    }

    // The bidirectional aggregate, for a reader who wants to compare against the quoted 300 GB/s.
    [[nodiscard]] double bidirectional_gigabytes_per_second() const noexcept {
        return per_link_gigabytes_per_second_per_direction * static_cast<double>(links_per_gpu) * 2.0;
    }
};

// THE 4-RANK V100 FIGURE, DERIVED RATHER THAN REMEMBERED. In the DGX-1/DGX-2 hybrid cube-mesh a
// V100's 6 links are spread over its 3 peers, 2 links each, so `links_to_peer` is `links_per_gpu /
// peers` when that divides. Written as a function so the arithmetic is auditable and so a caller can
// see that the 4-rank figure is HALF the 2-rank figure per direction -- which is the whole reason
// N=4 and N=2 cannot be compared by dividing by rank count.
[[nodiscard]] inline std::uint32_t v100_links_to_peer(const LinkSpec& spec, std::uint32_t world_size) {
    if (world_size <= 1U) { return 0U; }
    const std::uint32_t peers = world_size - 1U;
    if (peers == 0U) { return 0U; }
    // 2 ranks: one peer, all the links. Otherwise an even spread when it divides; when it does not,
    // the FLOOR, because a model must not assume a link it cannot place.
    const std::uint32_t spread = spec.links_per_gpu / peers;
    return spread == 0U ? 1U : spread;
}

// ---------------------------------------------------------------------------
// 2. THE PAYLOAD -- WHAT CROSSES, PER RANK BOUNDARY, PER STEP
// ---------------------------------------------------------------------------
//
// ⚠ THIS SECTION IS WHERE THE BRIEF'S HYPOTHESIS IS TESTED, AND THE ANSWER IS IN THE ENGINE'S OWN
// ARITHMETIC RATHER THAN IN AN EXPECTATION ABOUT IT.
//
// The expectation to check (from docs/maintainer/multi-device-and-shard-plan.md's ss1.1 and the brief
// built on it): under tp the per-step payload GROWS with context because attention combines across
// ranks; under pp it is a per-layer-boundary handoff that is CONTEXT-INDEPENDENT.
//
// WHAT THE ENGINE ACTUALLY SPLITS (core/shard_plan.h:188-223, `plan_shards`, Tensor arm):
//   * `kv_heads`: rank r owns kv_heads/world_size CONSECUTIVE heads, and the split is only legal
//     when kv_heads % world_size == 0, because a straddling GQA group "is simply not on that card".
//   * `weight_columns`: rank r owns weight_columns/world_size columns of the N dimension.
//   * `text_layers`: ALL layers on EVERY rank.
// And what the combine carries (core/shard_rank_axis.h:551-557): a `AttentionCombine` is
// {m, l, o} -- a ROW MAX, a ROW SUMEXP and a WEIGHTED VALUE SUM. Those are per-query-position
// quantities over the row's head dimension. They are NOT the KV cache.
//
// ⇒ THE CONTEXT LENGTH DOES NOT ENTER THE COMBINE'S SIZE. It enters the cost of COMPUTING each
//   rank's partial, which is each rank's own attention over its own KV heads -- local work. So for
//   this engine, tp's per-step cross-rank payload is CONTEXT-INDEPENDENT, and the brief's ss1
//   expectation does NOT hold. That is a finding, and it is checkable here rather than argued: the
//   payload functions below take the context length and the axes' payloads come out flat.
//
// ⚠ WHERE CONTEXT DOES ENTER, AND IT IS NOT A DECODE STEP. It enters the KV CACHE, and the KV cache
// crosses a rank boundary only for the page/eviction traffic the recall journal owns -- not for
// attention, because a rank attends to the KV heads it holds. So a model that said "tp's per-step
// payload is O(context)" would be describing a DIFFERENT tp (one that replicates KV and gathers it
// per step), not this engine's.

// The model's geometry, taken from the target's own `TextConfig` rather than paraphrased. The
// fields are exactly the ones `plan_shards()` reads plus the two the payload needs.
struct LinkModelGeometry {
    std::uint32_t text_layers  = 0;
    std::uint32_t q_heads      = 0;
    std::uint32_t kv_heads     = 0;
    std::uint32_t head_dim     = 0;
    std::uint32_t hidden       = 0;
    // The N dimension `plan_shards()` splits for the tensor axis: the number of columns a rank's
    // weight block covers. The engine reads it as `geometry.weight_columns`.
    std::uint32_t weight_columns = 0;
    // Bytes per element of the ACTIVATION that crosses. Not the weight dtype and not the KV dtype:
    // it is what a collective's buffer holds.
    std::uint32_t activation_bytes = 2; // bf16
    // How many of the layers are FULL (global) attention vs windowed. Named because a windowed layer
    // bounds its own attention cost at the window, which is a real term in the COMPUTE side; it does
    // not change a payload. Carried so a report cannot silently drop it.
    std::uint32_t full_attention_layers = 0;
    std::uint32_t windowed_layers       = 0;
    std::uint32_t sliding_window        = 0;
};

[[nodiscard]] inline bool geometry_is_complete(const LinkModelGeometry& g) noexcept {
    return g.text_layers != 0U && g.q_heads != 0U && g.kv_heads != 0U && g.head_dim != 0U &&
           g.hidden != 0U && g.weight_columns != 0U && g.activation_bytes != 0U;
}

// The payload of ONE attention combine, per rank boundary, for ONE query position.
//
// WHY THIS SIZE. Each rank contributes a partial {m, l, o} and the merge needs every OTHER rank's
// partial. `o` is the numerator over the row's value dimension, so a partial carries one `o` per
// (query head the rank owns) x (head_dim), plus one m and one l per query head. `plan_shards` gives
// each rank q_heads/world_size query heads (the same division the kv split must respect).
//
// ⚠ `m` and `l` are one value per query HEAD, not per head-dim element: a row max and a row sumexp
// are scalars for the row. Getting that wrong by a factor of head_dim would be a silent 256x error,
// which is why the two terms are separate lines here.
[[nodiscard]] inline std::uint64_t attention_combine_payload_bytes(const LinkModelGeometry& g,
                                                                  std::uint32_t world_size,
                                                                  std::uint32_t partial_bytes) {
    if (!geometry_is_complete(g) || world_size < 2U) { return 0U; }
    const std::uint64_t heads_per_rank = g.q_heads / world_size;
    if (heads_per_rank == 0U) { return 0U; }
    const std::uint64_t o_elements    = heads_per_rank * static_cast<std::uint64_t>(g.head_dim);
    const std::uint64_t ml_elements   = heads_per_rank * 2ULL; // m and l, one each per head
    const std::uint64_t bytes_per_partial =
        (o_elements * partial_bytes) + (ml_elements * static_cast<std::uint64_t>(partial_bytes));
    // A rank must receive every OTHER rank's partial, and it sends its own once.
    return bytes_per_partial * static_cast<std::uint64_t>(world_size - 1U);
}

// The payload of the WEIGHT-N all-reduce, per rank boundary, for ONE crossing activation:
// `world_size - 1` partial blocks of the split output width. This is the term the plan's ss1.1
// sentence "splits KV heads and weight N" implies, and it does not mention it.
//
// It is split into the two crossings a decoder layer actually has: the attention output projection
// and the MLP down projection. Both produce a hidden-width activation, so both are `hidden`.
[[nodiscard]] inline std::uint64_t layer_allreduce_payload_bytes(const LinkModelGeometry& g,
                                                                std::uint32_t world_size,
                                                                std::uint32_t crossings_per_layer) {
    if (!geometry_is_complete(g) || world_size < 2U) { return 0U; }
    return static_cast<std::uint64_t>(g.hidden) * g.activation_bytes *
           static_cast<std::uint64_t>(crossings_per_layer) *
           static_cast<std::uint64_t>(world_size - 1U);
}

// ONE DECODE STEP's total cross-rank payload on the tensor axis.
//
// ⚠ THE CONTEXT LENGTH IS A PARAMETER AND IT DOES NOT APPEAR ON THE RIGHT-HAND SIDE. That is the
// point of the signature: a caller who believes tp's cost grows with context calls this and sees a
// flat number, and the flatness is the finding rather than an omission. `context_tokens` is carried
// only so the caller can print it beside the result and so a future axis whose payload DOES depend
// on it can be added without a signature change.
[[nodiscard]] inline std::uint64_t tp_decode_step_payload_bytes(const LinkModelGeometry& g,
                                                               std::uint32_t world_size,
                                                               std::uint64_t context_tokens,
                                                               std::uint32_t partial_bytes = 8) {
    (void)context_tokens; // see the note above: deliberately unused
    if (!geometry_is_complete(g) || world_size < 2U) { return 0U; }
    // Attention combine: once per layer that has attention to combine. Every layer under this
    // engine's tp keeps its attention, so it is every layer.
    const std::uint64_t combine =
        attention_combine_payload_bytes(g, world_size, partial_bytes) *
        static_cast<std::uint64_t>(g.text_layers);
    // Two weight-N crossings per layer: attention output projection, MLP down projection.
    const std::uint64_t reductions = layer_allreduce_payload_bytes(g, world_size, 2U) *
                                     static_cast<std::uint64_t>(g.text_layers);
    return combine + reductions;
}

// ONE DECODE STEP's total cross-rank payload on the PIPELINE axis.
//
// ⚠ THE PIPELINE AXIS IS REFUSED BY THE GUARD ON THIS BOX -- `multi::VirtualRefusal::BadAxis`, see
// core/virtual_device.h:385-393 -- so this function's output is a MODEL THE GUARD WILL NOT LET
// ANYONE MEASURE HERE. It is kept because the crossover prediction the owner asks for is per axis and
// the pp number is the CONTRAST that makes the tp number legible, and because it is the one axis a
// future real-processes line can check. It is NOT a measurement and no run on this box can make it
// one.
//
// The payload: each stage's LAST layer hands its output activation to the next stage. Context does
// not enter, for the same reason as tp: the handoff is one token's hidden vector.
[[nodiscard]] inline std::uint64_t pp_decode_step_payload_bytes(const LinkModelGeometry& g,
                                                              std::uint32_t world_size,
                                                              std::uint64_t context_tokens) {
    (void)context_tokens;
    if (!geometry_is_complete(g) || world_size < 2U) { return 0U; }
    return static_cast<std::uint64_t>(g.hidden) * g.activation_bytes *
           static_cast<std::uint64_t>(world_size - 1U);
}

// ---------------------------------------------------------------------------
// 3. THE TIME, AND THE CROSSOVER
// ---------------------------------------------------------------------------

// One transfer as the model prices it, with the two terms KEPT APART so a report can subtract the
// deliberate part and name the rest.
struct LinkCost {
    std::uint64_t payload_bytes          = 0;
    std::uint32_t boundary_crossings     = 0;
    double bandwidth_ns                  = 0.0; // payload / effective rate, summed over crossings
    double latency_ns                    = 0.0; // latency_per_call * crossings
    double total_ns                      = 0.0;
    double effective_bytes_per_second    = 0.0; // printed so a leg names its own rate
    double latency_per_call_ns           = 0.0; // printed so a leg names its own latency
};

[[nodiscard]] inline LinkCost price_transfer(const LinkSpec& link, std::uint64_t payload_bytes,
                                             std::uint32_t boundary_crossings) {
    LinkCost out;
    out.payload_bytes        = payload_bytes;
    out.boundary_crossings   = boundary_crossings;
    out.effective_bytes_per_second = link.per_direction_bytes_per_second();
    out.latency_per_call_ns  = link.latency_per_call_nanoseconds;
    if (out.effective_bytes_per_second > 0.0) {
        out.bandwidth_ns =
            (static_cast<double>(payload_bytes) / out.effective_bytes_per_second) * 1.0e9 *
            static_cast<double>(boundary_crossings);
    }
    out.latency_ns = link.latency_per_call_nanoseconds * static_cast<double>(boundary_crossings);
    out.total_ns   = out.bandwidth_ns + out.latency_ns;
    return out;
}

// THE CROSSOVER, as a value with its inputs attached rather than as a number in prose.
//
// DEFINITION, stated once because a differently-defined crossover is a different number: the
// crossover context length is the smallest context at which the MODELLED transport time for ONE
// decode step reaches `fraction_of_compute` of a MEASURED single-card per-token time. Below it the
// interconnect is a rounding error; above it, it is the budget.
//
// ⚠ AND IT IS A BOUND, NOT A PREDICTION OF A REAL RUN. On one card with N simulated ranks, all N
// ranks' COMPUTE shares one piece of silicon, so the compute component of any measurement here is
// N-times too slow while the interconnect component is right. The crossover computed against a
// single-card per-token time is therefore the interconnect-bound FLOOR: a lower bound on what the
// interconnect permits. It is NOT the real speedup and it is not compared against one.
struct Crossover {
    bool defined = false;
    std::string reason;

    // Inputs, all named so a reader can re-apply a different link or a different compute time.
    double   compute_ns_per_step      = 0.0; // the MEASURED single-card per-token time, an INPUT
    double   fraction_of_compute      = 0.0;
    double   transport_ns_per_step    = 0.0; // from the payload functions, at ANY context (flat)
    std::uint64_t payload_bytes       = 0;
    std::uint32_t boundary_crossings  = 0;
    double   effective_bytes_per_second = 0.0;
    double   latency_per_call_ns      = 0.0;
    std::string_view axis             = "unknown";

    // THE ANSWER. Infinite when the payload cannot be made to reach the threshold by changing the
    // context -- which is the truthful outcome for a payload that does not depend on it, and the
    // single most important thing this struct can say.
    bool   crossover_exists       = false;
    double crossover_context_tokens = 0.0;
    // True when the transport is ALREADY above the threshold at the shortest context, i.e. there is
    // no regime in which the interconnect is negligible on this axis and this link.
    bool   already_over           = false;
};

// `per_step_payload_is_context_dependent` is an INPUT rather than something this function infers,
// because inferring it from the payload functions would be circular: the caller states what the
// engine's arithmetic gave it, and the crossover function says what follows.
[[nodiscard]] inline Crossover compute_crossover(const LinkSpec& link, std::uint64_t payload_bytes,
                                                 std::uint32_t boundary_crossings,
                                                 double compute_ns_per_step,
                                                 double fraction_of_compute,
                                                 std::string_view axis,
                                                 bool per_step_payload_is_context_dependent) {
    Crossover out;
    out.axis                        = axis;
    out.compute_ns_per_step         = compute_ns_per_step;
    out.fraction_of_compute         = fraction_of_compute;
    out.payload_bytes               = payload_bytes;
    out.boundary_crossings          = boundary_crossings;
    out.effective_bytes_per_second  = link.per_direction_bytes_per_second();
    out.latency_per_call_ns         = link.latency_per_call_nanoseconds;

    if (compute_ns_per_step <= 0.0) {
        out.reason = "no measured single-card per-token time was supplied: the crossover is a ratio "
                     "against one, so without it there is nothing to cross over";
        return out;
    }
    if (out.effective_bytes_per_second <= 0.0) {
        out.reason = "the link's effective per-direction rate is zero: no payload can be priced";
        return out;
    }
    out.defined = true;

    const LinkCost cost = price_transfer(link, payload_bytes, boundary_crossings);
    out.transport_ns_per_step = cost.total_ns;

    const double threshold = compute_ns_per_step * fraction_of_compute;
    if (!per_step_payload_is_context_dependent) {
        // THE ANSWER THIS PROJECT ACTUALLY GETS. A flat payload crosses the threshold at exactly one
        // context or at none, so there is no length at which it STARTS to matter -- it either
        // matters from the first token or never.
        out.crossover_exists = false;
        out.already_over     = out.transport_ns_per_step >= threshold;
        out.reason = out.already_over
                         ? "the per-step payload does not depend on the context length and is ALREADY "
                           "at or above the threshold at the shortest context: on this axis and this "
                           "link there is no regime in which the interconnect is negligible"
                         : "the per-step payload does not depend on the context length and is BELOW "
                           "the threshold at every context: no context length makes this axis's "
                           "interconnect the bottleneck per step, so a crossover does not exist. "
                           "The context length can still make it matter IN TOTAL, because the total "
                           "is per-step cost times the number of steps -- see the note below.";
        return out;
    }
    // A context-dependent payload: solve payload(L) = threshold, with the payload linear in L.
    // `payload_bytes` is then taken to be the payload at ONE token of context, and the caller's
    // contract is that L multiplies it.
    if (payload_bytes == 0U) {
        out.reason = "a context-dependent payload of zero bytes cannot cross any threshold";
        return out;
    }
    const double per_token_ns =
        (static_cast<double>(payload_bytes) * 1.0e9 / out.effective_bytes_per_second) *
        static_cast<double>(boundary_crossings);
    if (per_token_ns <= 0.0) {
        out.reason = "the per-token payload prices to zero nanoseconds";
        return out;
    }
    const double latency_total = link.latency_per_call_nanoseconds * boundary_crossings;
    if (latency_total >= threshold) {
        out.crossover_exists = true;
        out.crossover_context_tokens = 0.0;
        out.already_over             = true;
        out.reason = "the latency term alone already exceeds the threshold at zero context";
        return out;
    }
    out.crossover_exists        = true;
    out.crossover_context_tokens = (threshold - latency_total) / per_token_ns;
    return out;
}

// ---------------------------------------------------------------------------
// 4. WHAT THIS MODEL IS NOT, AS A VALUE
// ---------------------------------------------------------------------------

// ⚠ Rule 4 of this project's instrument discipline: a printed number may be a COST MODEL and not a
// measurement. This function is that statement in the type system -- a caller gets `false` and has
// to print it beside every number it derived from here.
[[nodiscard]] inline bool link_model_is_measurement() noexcept { return false; }

// The one-paragraph caveat, printed once per report and not re-hedged afterwards. The two sentences
// are the two things that must not be lost:
[[nodiscard]] inline std::string link_model_caveat() {
    return "tp_link_model is a BANDWIDTH-LIMITED COST MODEL over a SOURCED link, not a measurement. "
           "It gives an INTERCONNECT-BOUND FLOOR -- a lower bound on what the interconnect permits "
           "-- and NOT the real speedup: on one card with N simulated ranks every rank's compute "
           "shares one piece of silicon, so the compute component of any single-card measurement is "
           "N-times too slow while the interconnect component is right. The link figures are the "
           "Volta whitepaper's (25 GB/s per direction per link, 6 links per GPU, 300 GB/s "
           "bidirectional aggregate); the latency term is a KNOB whose origin is printed per leg "
           "and which defaults to 0 rather than to an invented microsecond.";
}

} // namespace ninfer::multi
