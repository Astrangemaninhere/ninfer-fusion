#pragma once
// =============================================================================
// THE STAGE PLAN -- the layer range the decode loop actually walks
// =============================================================================
// WHY THIS FILE EXISTS. `src/core/shard_plan.h` answers "WHICH SLICES does this rank own"
// (WorldShape in, ShardPlan out) and `src/core/shard_rank_axis.h` answers "what does a rank
// owe its neighbours". Neither was REACHABLE from a command line. MEASURED 2026-09-25 at
// HEAD 7785fc3, population `apps/`:
//   * `grep -ohE '"--[a-z0-9-]+"' apps/cli/options.{h,cpp} | sort -u | wc -l` = **72** long
//     options, and
//   * `--world-size` 0, `--rank` 0, `--axis` 0, `--layer-range` 0, `--stage-layers` 0.
// So a `pp` world could be asked for from a farm probe and from nothing a user runs. This
// file is the ONE grammar and the ONE set of refusals for that missing surface.
//
// HOST-ONLY BY CONSTRUCTION, exactly like shard_plan.h / shard_rank_axis.h / tp_transport.h:
// no CUDA header, no device query, no tensor, no allocation. That is what lets the same
// parser be called by two callers that must not be able to disagree:
//   1. the FRONT DOOR (`apps/cli/options.cpp`) -- refuses a mis-shaped SPEC before any
//      artifact is opened, the rule `--kv-layer-storage` states at options.cpp:686-693;
//   2. the RUNTIME (`targets/qwen3_6/impl/runtime/text_context_impl.h`) -- refuses a
//      well-shaped SPEC that is not a shard of the world `plan_shards()` derives for the
//      geometry the artifact actually declares, i.e. the check that makes "refused BY NAME
//      when it does not describe a world the rank axis accepts" true rather than a promise.
// A second spelling of this grammar is the exact defect class apps/cli/options.h:58-60 says
// this project keeps paying for, which is why there is one parser and not two.
//
// GRAMMAR, in the tree's own dialect. `--kv-layer-storage` is documented as "lo-hi:dtype
// entries like 0-7:bf16, or all:dtype" (apps/cli/options.cpp:691-693) and `--kv-bit-budget`
// as "a ceiling per KV element, or per layer range: \"0-7:8,8-15:4.5\"" (:252, 2026-09-26). A layer set
// spelled `lo-hi`, comma-separated, is therefore already how this tree writes one:
//
//     SPEC  := RANGE ("," RANGE)*
//     RANGE := LO "-" HI                     0-based, HI INCLUSIVE (the tree's convention)
//
// A SPEC of N ranges IS a pipeline world of N stages: stage i owns RANGE i. So
//   `--stage-layers 0-17,18-35`  asks for a `pp` world of 2 over 36 layers;
//   `--stage-layers 0-35`        is the IDENTITY -- one stage, i.e. axis `none`, and the run
//                                is byte-for-byte a run with no flag at all. That equivalence
//                                is arm T2 of this line's landq entry, and it is what makes
//                                "the flag is read" a reading instead of a claim.
//
// THE BOUNDARY. A `pp` world has a hidden state crossing at each stage boundary. `--stage-handoff
// DIR` names the directory the boundary payload is written to and read from (stage k writes
// `stage_k.bin`, reads `stage_{k-1}.bin`). The payload is the stage's input/output hidden
// tensor as raw bytes plus a header carrying a magic, the layer index it was produced at and
// an FNV-1a of the payload, so a file left over from another prompt is DETECTABLE rather than
// silently consumed -- the property dl/ppaxis (F-739) proved the mechanism has. With no
// `--stage-handoff` the boundary is carried in-process and nothing touches the disk.
//
// What this file deliberately does NOT decide: whether a `pp` world should run at all. That
// verdict lives in `virtual_device.h`'s `VirtualRefusal::BadAxis` and is untouched by this
// header. This surface makes `pp` REACHABLE, not universally SUPPORTED; §the spec refusal
// below names the three run shapes it refuses by name.

#include "core/shard_plan.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::multi {

// The flags. Named once so the help text, the refusals and the parser cannot disagree.
inline constexpr std::string_view kStageLayersFlag     = "--stage-layers";
inline constexpr std::string_view kStageHandoffFlag    = "--stage-handoff";
// The NEGATIVE-CONTROL knob, and nothing else. With it, stage k does not write the boundary
// payload, so stage k+1 reads whatever was already in the file (stale) or nothing at all. It
// exists so that "the handoff is load-bearing" is FALSIFIABLE from the command line on the
// shipped binary -- the falsifier dl/ppaxis ran inside a farm probe. It is not a feature: a
// `pp` run with it is a deliberately wrong run, and the ids must differ from the run without it.
inline constexpr std::string_view kStageHandoffCutFlag = "--stage-handoff-cut";

// Every refusal this surface can produce, named once. The token is what reaches stderr; the
// sentence follows it. `refused-stage-layers` is the grammar, `-partition` is the cover,
// `-axis` is the world the rank axis will not accept, `-spec` is the run shape (see below).
inline constexpr std::string_view kStageLayersGrammarRefusal   = "refused-stage-layers";
inline constexpr std::string_view kStageLayersPartitionRefusal = "refused-stage-layers-partition";
inline constexpr std::string_view kStageLayersAxisRefusal      = "refused-stage-layers-axis";
inline constexpr std::string_view kStageLayersSpecRefusal      = "refused-stage-layers-spec";
inline constexpr std::string_view kStageLayersGraphRefusal     = "refused-stage-layers-graph";
inline constexpr std::string_view kStageLayersW13Refusal       = "refused-stage-layers-w13";
inline constexpr std::string_view kStageLayersHandoffRefusal   = "refused-stage-handoff";

// The bounds. 4096 is shard_plan.h's own parse ceiling (`parse_world_u32`); 64 is
// virtual_device.h's rank ceiling, reused so the two surfaces cannot disagree about how wide
// a world may be.
inline constexpr std::uint32_t kMaximumStageLayerValue = 4096U;
inline constexpr std::size_t kMaximumStages            = 64U;

// ONE stage's layers. `last` is INCLUSIVE, which is the spelling `--kv-layer-storage`'s
// worked example `0-7` uses and the one a user typed the range off. It is NOT shard_plan.h's
// half-open `Range` -- the conversion happens once, in stage_range_of(), so the wire spelling
// and the axis spelling cannot drift apart in two places.
struct StageRange {
    std::uint32_t first = 0; // inclusive, 0-based
    std::uint32_t last  = 0; // INCLUSIVE

    [[nodiscard]] std::uint32_t count() const noexcept { return last - first + 1U; }
    friend bool operator==(const StageRange&, const StageRange&) = default;
};

// The parsed request. `requested == false` means the flag was absent, which is the state every
// existing invocation is in, and in which every reader below is a no-op.
struct StagePlan {
    bool requested = false;
    std::string spec_raw;
    std::vector<StageRange> stages; // ascending, contiguous, covering [0, text_layers)

    [[nodiscard]] std::uint32_t world_size() const noexcept {
        return static_cast<std::uint32_t>(stages.size());
    }
    // ONE stage is the identity (axis `none`). The runtime must be byte-identical to a
    // flagless run in this state, and arm T2 measures exactly that.
    [[nodiscard]] bool is_identity() const noexcept { return stages.size() == 1U; }
    // A run whose stages are not all the same width is a real pipeline and is allowed --
    // shard_plan.h's Pipeline arm says an uneven split "costs only balance".
    [[nodiscard]] bool uniform() const noexcept {
        for (const StageRange& stage : stages) {
            if (stage.count() != stages.front().count()) { return false; }
        }
        return true;
    }
};

// ---------------------------------------------------------------------------
// The conversion to the axis's own spelling, in ONE place
// ---------------------------------------------------------------------------
// shard_plan.h's `Range` is half-open; this surface's is inclusive. One function, so a reader
// that compares a StageRange against a ShardPlan::text_layers cannot get the off-by-one right
// in one place and wrong in another.
[[nodiscard]] inline Range stage_range_of(const StageRange& stage) noexcept {
    return Range{stage.first, stage.last + 1U};
}

[[nodiscard]] inline StageRange stage_range_from(const Range& range) noexcept {
    return StageRange{range.begin, range.end == 0U ? 0U : range.end - 1U};
}

// ---------------------------------------------------------------------------
// The grammar
// ---------------------------------------------------------------------------
// Every grammar refusal opens with the token AND the flag, so a reader who sees one line of
// stderr knows which flag was wrong and which class of wrong it was, and so a grep for the
// token finds the refusal rather than a paraphrase of it.
[[nodiscard]] inline std::string stage_layers_grammar_head() {
    return std::string(kStageLayersGrammarRefusal) + ": " + std::string(kStageLayersFlag) + " ";
}

// Strict, and strict on purpose: a typo must be a refusal rather than a quiet default. Same
// shape as shard_plan.h's `parse_world_u32` and virtual_device.h's own parse ("a typo is a
// refusal rather than rank 0"). Returns an EMPTY string iff the spec parsed; otherwise the
// refusal sentence, which the caller prefixes with kStageLayersGrammarRefusal.
[[nodiscard]] inline std::string parse_stage_layers(std::string_view spec, StagePlan& out) {
    out              = StagePlan{};
    out.requested    = true;
    out.spec_raw.assign(spec);
    if (spec.empty()) {
        return stage_layers_grammar_head() +
               "needs a value: lo-hi layer ranges, one per stage, like 0-17,18-35. An empty "
               "value is refused rather than read as \"no stages\", because a run that quietly "
               "became single-device is indistinguishable from a successful world at the point "
               "where the numbers are read.";
    }
    std::size_t cursor = 0;
    while (cursor <= spec.size()) {
        const std::size_t comma = spec.find(',', cursor);
        const std::size_t stop  = comma == std::string_view::npos ? spec.size() : comma;
        const std::string_view term = spec.substr(cursor, stop - cursor);
        if (term.empty()) {
            return stage_layers_grammar_head() + " \"" + std::string(spec) +
                   "\" has an empty range in it. A trailing or doubled comma reads as a stage "
                   "that owns nothing, which is not a world any axis accepts.";
        }
        // LO "-" HI, or a bare LO. A bare number is a one-layer stage, which is a legal
        // pipeline (shard_plan.h's Pipeline arm allows an uneven split) and is refused only
        // if it does not match the axis.
        const std::size_t dash = term.find('-');
        const std::string_view lo_text = dash == std::string_view::npos ? term : term.substr(0, dash);
        const std::string_view hi_text =
            dash == std::string_view::npos ? term : term.substr(dash + 1);
        const bool bare = dash == std::string_view::npos;
        if (lo_text.empty() || (!bare && hi_text.empty())) {
            return stage_layers_grammar_head() + " \"" + std::string(term) +
                   "\" is not a layer range. The accepted spelling is lo-hi with lo <= hi, or a "
                   "bare layer index for a one-layer stage (so 0-17 and 3 are both ranges).";
        }
        if (!bare && term.find('-', dash + 1) != std::string_view::npos) {
            return stage_layers_grammar_head() + " \"" + std::string(term) +
                   "\" has more than one '-'. A layer range is lo-hi; nesting or a negative "
                   "index is not part of this grammar.";
        }
        std::uint32_t lo = 0;
        std::uint32_t hi = 0;
        bool good        = true;
        for (const char c : lo_text) {
            if (c < '0' || c > '9') { good = false; break; }
            lo = lo * 10U + static_cast<std::uint32_t>(c - '0');
            if (lo > kMaximumStageLayerValue) { good = false; break; }
        }
        if (good) {
            for (const char c : hi_text) {
                if (c < '0' || c > '9') { good = false; break; }
                hi = hi * 10U + static_cast<std::uint32_t>(c - '0');
                if (hi > kMaximumStageLayerValue) { good = false; break; }
            }
        }
        if (!good) {
            return stage_layers_grammar_head() + " \"" + std::string(term) +
                   "\" is not a layer number: every character must be a decimal digit and no "
                   "value may exceed " + std::to_string(kMaximumStageLayerValue) +
                   ". A value that cannot be read is refused rather than defaulted.";
        }
        if (hi < lo) {
            return stage_layers_grammar_head() + " \"" + std::string(term) +
                   "\" runs backwards (lo " + std::to_string(lo) + " > hi " +
                   std::to_string(hi) +
                   "). A stage's layers are lo..hi INCLUSIVE; an inverted range would be a "
                   "stage that owns nothing.";
        }
        if (out.stages.size() >= kMaximumStages) {
            return stage_layers_grammar_head() + " names more than " +
                   std::to_string(kMaximumStages) +
                   " stages. That is the rank ceiling core/virtual_device.h enforces "
                   "(world_size > 64 is refused there), reused here so the two surfaces cannot "
                   "disagree about how wide a world may be.";
        }
        out.stages.push_back(StageRange{lo, hi});
        if (comma == std::string_view::npos) { break; }
        cursor = comma + 1;
    }
    if (out.stages.empty()) {
        return stage_layers_grammar_head() + " \"" + std::string(spec) + "\" names no stage.";
    }
    return {};
}

// ---------------------------------------------------------------------------
// The cover: is this a partition of the layer axis at all
// ---------------------------------------------------------------------------
// Independent of any artifact, so the front door can run it. Rules, all of them from the
// tree's own Pipeline arm: ranges ASCENDING, CONTIGUOUS (no hole, no overlap) and covering
// [0, text_layers) exactly. `text_layers == 0` means "not known yet" and skips the cover
// check -- the front door has no artifact and must not guess one.
[[nodiscard]] inline std::string stage_layers_partition_refusal(const StagePlan& plan,
                                                                std::uint32_t text_layers) {
    if (!plan.requested || plan.stages.empty()) { return {}; }
    if (plan.stages[0].first != 0U && text_layers != 0U) {
        return std::string(kStageLayersPartitionRefusal) + ": the first stage starts at layer " +
               std::to_string(plan.stages[0].first) +
               ", not 0. A pipeline world covers every text layer of the model once; a stage "
               "set that begins above 0 leaves layers nobody runs.";
    }
    for (std::size_t i = 1; i < plan.stages.size(); ++i) {
        const StageRange& previous = plan.stages[i - 1];
        const StageRange& current  = plan.stages[i];
        if (current.first <= previous.last) {
            return std::string(kStageLayersPartitionRefusal) + ": stage " + std::to_string(i) +
                   " (" + std::to_string(current.first) + "-" + std::to_string(current.last) +
                   ") does not start after stage " + std::to_string(i - 1) + " (" +
                   std::to_string(previous.first) + "-" + std::to_string(previous.last) +
                   "). The stages must be ascending and disjoint, because each layer is owned "
                   "by exactly one stage.";
        }
        if (current.first != previous.last + 1U && text_layers != 0U) {
            return std::string(kStageLayersPartitionRefusal) + ": stage " + std::to_string(i - 1) +
                   " ends at " + std::to_string(previous.last) + " and stage " +
                   std::to_string(i) + " begins at " + std::to_string(current.first) +
                   "; layer " + std::to_string(previous.last + 1U) +
                   " belongs to no stage. A hole is not a pipeline: the layers in it would be "
                   "skipped by every stage silently.";
        }
    }
    if (text_layers != 0U) {
        const std::uint32_t covered = plan.stages.back().last + 1U;
        if (covered != text_layers) {
            return std::string(kStageLayersPartitionRefusal) + ": the spec covers layers 0.." +
                   std::to_string(plan.stages.back().last) + " but the model has " +
                   std::to_string(text_layers) +
                   " text layers. A pipeline world covers every layer of the model, so a spec "
                   "that stops short would run a prefix of the stack and then sample from a "
                   "hidden state no layer produced.";
        }
    }
    return {};
}

// ---------------------------------------------------------------------------
// The axis: is this a world the RANK AXIS accepts
// ---------------------------------------------------------------------------
// THIS IS THE CHECK THE SURFACE OWES, and it is the rank axis's OWN function doing it --
// `plan_shards()`, not a second derivation of the balanced split. For every rank of the
// declared world the plan must equal the stage the spec names, exactly. Anything else is a
// spec that describes a world `plan_shards` will not hand out, and it is refused BY NAME
// rather than accepted and ignored.
//
// What this does NOT do: it does not call `validate_virtual_request()`. That guard's answer
// for `axis=pp` is `active = 0` and STAYS 0 -- `pp` is refused as a *virtual device*, because a
// rank in one address space is a budget and not a partition. The layer RANGE is a fact
// `plan_shards` already computes and accepts, and it is the only thing this surface needs.
[[nodiscard]] inline std::string stage_plan_axis_refusal(const StagePlan& plan,
                                                         const ModelGeometry& geometry) {
    if (!plan.requested || plan.stages.empty()) { return {}; }
    // ⭐ A SINGLE STAGE IS NOT AUTOMATICALLY THE IDENTITY, and this arm is where that nearly
    // became a hole: `--stage-layers 5-35` is one stage, so an "is_identity() -> accept"
    // short-circuit would have accepted a range that starts at layer 5 and silently drops
    // layers 0..4. The identity is the whole stack, and "the whole stack" is what
    // plan_shards()' NONE arm returns (`plan.text_layers = Range{0, geometry.text_layers}`,
    // shard_plan.h:180), so the same function decides this case too -- no second rule.
    if (plan.world_size() == 1U) {
        const ShardPlan whole = plan_shards(WorldShape{1U, 0U, ParallelAxis::None}, geometry);
        if (!whole.ok) {
            return std::string(kStageLayersAxisRefusal) +
                   ": the rank axis refuses the single-device world itself: " + whole.reason;
        }
        const StageRange derived = stage_range_from(whole.text_layers);
        if (!(derived == plan.stages[0])) {
            return std::string(kStageLayersAxisRefusal) + ": " + std::string(kStageLayersFlag) +
                   " names one stage as " + std::to_string(plan.stages[0].first) + "-" +
                   std::to_string(plan.stages[0].last) +
                   ", but a single-stage world is the WHOLE stack, which for this model is " +
                   std::to_string(derived.first) + "-" + std::to_string(derived.last) +
                   " (core/shard_plan.h plan_shards, axis=none arm). A one-stage world that is "
                   "not the whole stack would walk a prefix and then sample from a hidden state "
                   "no layer produced.";
        }
        return {};
    }
    const std::uint32_t world_size = plan.world_size();
    for (std::uint32_t rank = 0; rank < world_size; ++rank) {
        const ShardPlan derived = plan_shards(WorldShape{world_size, rank, ParallelAxis::Pipeline},
                                              geometry);
        if (!derived.ok) {
            return std::string(kStageLayersAxisRefusal) + ": the rank axis refuses this world at "
                   "rank " + std::to_string(rank) + ": " + derived.reason;
        }
        const StageRange derived_stage = stage_range_from(derived.text_layers);
        if (!(derived_stage == plan.stages[rank])) {
            return std::string(kStageLayersAxisRefusal) + ": " + std::string(kStageLayersFlag) +
                   " names stage " + std::to_string(rank) + " as " +
                   std::to_string(plan.stages[rank].first) + "-" +
                   std::to_string(plan.stages[rank].last) + ", but a pp world of " +
                   std::to_string(world_size) + " over " + std::to_string(geometry.text_layers) +
                   " text layers gives that rank layers " + std::to_string(derived_stage.first) +
                   "-" + std::to_string(derived_stage.last) +
                   " (core/shard_plan.h plan_shards, Pipeline arm). A layer range that is not a "
                   "shard of the world the axis derives is a world this build does not have, so "
                   "it is refused by name instead of being run and reported as something it is "
                   "not.";
        }
    }
    return {};
}

// ---------------------------------------------------------------------------
// The run shape: the three things a partial range cannot carry
// ---------------------------------------------------------------------------
// A layer range bounds the FORWARD pass. Three other per-layer walks in this tree are NOT
// bounded by it, and a partial range over any of them computes from a layer set the model
// never ran. Each is refused by name rather than silently mis-run. These are findings of this
// line (they are the answer to "does your change expose a further defect"), and the file:line
// behind each is in the message.
[[nodiscard]] inline std::string stage_layers_run_shape_refusal(const StagePlan& plan,
                                                                bool speculation_enabled,
                                                                bool weight_offload_active,
                                                                bool cuda_graph_enabled) {
    if (!plan.requested || plan.is_identity()) { return {}; }
    if (speculation_enabled) {
        return std::string(kStageLayersSpecRefusal) +
               ": a partial layer range with speculation on. The drafters walk the layer axis "
               "themselves and this range does not bound them: dflash_impl.h:143 and :236 "
               "(for (int layer = 0; layer < Config::layers; ++layer)), dflash2_impl.h:190 and "
               ":248, and mtp_impl.h:264 (bounded by state.text_cache.layers()). Those loops "
               "are populated from the SAME weights while the target's forward pass walks only "
               "the stage's range, so a speculative round under a partial range would verify "
               "against a stack this stage did not run. Run the stage with --spec none, or run "
               "the identity range.";
    }
    if (weight_offload_active) {
        return std::string(kStageLayersW13Refusal) +
               ": a partial layer range with the W13 weight host-offload budget set. "
               "src/product/weight_residency.h:374-383 states the residency contract as \"every "
               "pass over the layer order enters every offloaded layer through note_layer()\", "
               "and the arena slot arithmetic at text_context_impl.h:1543-1557 depends on it. A "
               "pass that walks only the stage's range enters a SUBSET of the offloaded layers, "
               "so the prefetch for the layers it skipped is never issued and the arena's "
               "occupancy bookkeeping no longer describes the run. The range breaks a contract "
               "that is asserted, not merely assumed, so the pair is refused rather than run.";
    }
    if (cuda_graph_enabled) {
        return std::string(kStageLayersGraphRefusal) +
               ": a partial layer range with CUDA-graph capture on. The decode batch is "
               "captured as a graph (schedule.h:243-246 capture_ordinary_decode_batch, "
               "decode_impl.h:60-81) and the captured body fixes the layer walk at capture "
               "time; the boundary payload a partial range needs is a HOST-side "
               "write/read at the stage seam, which a captured graph would replay with stale "
               "data -- the same reason W13's H2D is prefill-only (text_context_impl.h:1547-1552). "
               "Pass --no-cuda-graph, or run the identity range.";
    }
    return {};
}

// ---------------------------------------------------------------------------
// The boundary payload's own grammar
// ---------------------------------------------------------------------------
// `--stage-handoff DIR`: the directory the seam's hidden state is written to and read from.
// Refused BY NAME when it is asked for by a run that has no seam (a single stage), because a
// directory that is written and never read is exactly the "accepted and ignored" shape.
[[nodiscard]] inline std::string stage_handoff_refusal(const StagePlan& plan,
                                                       std::string_view dir) {
    if (dir.empty()) {
        if (plan.requested && !plan.is_identity()) {
            return std::string(kStageLayersHandoffRefusal) + ": a world of " +
                   std::to_string(plan.world_size()) +
                   " stages needs a boundary, and --stage-handoff names the directory the hidden "
                   "state crosses through. Without it the stages would have to share one address "
                   "space's tensor, which is the very thing that makes the split unauditable.";
        }
        return {};
    }
    if (!plan.requested || plan.is_identity()) {
        return std::string(kStageLayersHandoffRefusal) + ": " + std::string(kStageHandoffFlag) +
               " \"" + std::string(dir) +
               "\" was given but the run has ONE stage, so there is no boundary to carry. A "
               "directory written and never read is a flag accepted and ignored, which this "
               "tree refuses by name.";
    }
    return {};
}

} // namespace ninfer::multi
