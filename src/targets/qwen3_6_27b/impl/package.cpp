#include <ninfer/targets/qwen3_6_27b/package.h>
#include <ninfer/targets/qwen3_6/frontend_resources.h>
#include <ninfer/targets/qwen3_6/prepared_prompt.h>

#include <cuda_runtime.h>

#include "artifact/reader.h"
#include "targets/qwen3_6_27b/impl/load/bindings.h"
#include "targets/qwen3_6_27b/impl/variant.h"

#include <array>
#include <stdexcept>
#include <vector>
#include <cstdlib>
#include <cstdio>
#include <utility>

namespace ninfer::targets::qwen3_6_27b::detail {

class LoadPlan::Impl {
public:
    Impl(WeightsProfile weights_profile_in, ArtifactLoadPlan target_plan)
        : weights_profile(weights_profile_in), plan(std::move(target_plan)) {}

    WeightsProfile weights_profile;
    ArtifactLoadPlan plan;
};

LoadPlan::LoadPlan(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}

LoadPlan::LoadPlan(LoadPlan&&) noexcept            = default;
LoadPlan& LoadPlan::operator=(LoadPlan&&) noexcept = default;
LoadPlan::~LoadPlan()                              = default;

const artifact::MaterializationPlan& LoadPlan::materialization() const {
    if (impl_ == nullptr) { throw std::logic_error("target load plan is empty"); }
    return impl_->plan.materialization;
}

LoadedModel::LoadedModel(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}

LoadedModel::~LoadedModel() = default;

} // namespace ninfer::targets::qwen3_6_27b::detail

namespace ninfer::targets::qwen3_6_27b {
namespace {

// The declaration and the binder must agree about which profile carries a draft tensor group:
// `detail::Variant::dspark_weights` / `dflash2_weights` (`impl/variant.h`) are the pre-existing
// single source of truth for that, and `resolved_auto_speculative` below still keys on the same
// profile value, so a row whose identity is repointed would otherwise take the draft dispatch with
// it. Ported from id1's `speculative_declarations_match_payload`, which cross-checked the
// speculative field of its realization table against these same two predicates. The two flavours
// are named here and nowhere else on this package's resolution path.
constexpr bool declared_draft_identities_match_payload() {
    for (const targets::WeightsDeclaration<detail::WeightsProfile>& row :
         Package::accepted_weights) {
        const bool names_dspark  = row.identity.weights_id == "nvfp4-dspark";
        const bool names_dflash2 = row.identity.weights_id == "nvfp4-dflash2";
        if (names_dspark != detail::Variant::dspark_weights(row.profile)) { return false; }
        if (names_dflash2 != detail::Variant::dflash2_weights(row.profile)) { return false; }
    }
    return true;
}
static_assert(declared_draft_identities_match_payload(),
              "the declared draft identities must agree with the payload predicates");

// WHICH ARTIFACTS CARRY THE SHORTLIST PROPOSAL HEAD.
//
// `has_shortlist_head` is an assertion about the ARTIFACT'S OWN OBJECT PLAN: it means
// `text/draft_head` + `text/draft_head_token_ids` are in this artifact
// (load/bindings.cpp binds both for EVERY row of this package -- the bind is
// unconditional and only the PLACEMENT is gated by features.optimized_proposal()).
// It is NOT a property of the quantization flavour, and it was read as one:
// `has_shortlist_head` used to be spelled `weights_profile == Qwen38Nvfp4DFlash2`,
// i.e. the field that names a quantization flavour was asked a question about the
// payload. On the shipped nvfp4 artifact that answer is wrong -- measured, by reading
// the container's own objects[] (magic 4e494e4645520002):
//     qwen3_8_27b_nvfp4.ninfer            weights_id=nvfp4            1124 objects
//         -> text/draft_head + text/draft_head_token_ids PRESENT
//     qwen3_8_27b_nvfp4_modelopt.ninfer   weights_id=nvfp4-modelopt  1103 objects
//         -> text/draft_head {Q4G64_F16S, [131072,5120], 356515840 B} PRESENT
//     qwen3_8_27b_nvfp4_dspark.ninfer     weights_id=nvfp4-dspark    1179 objects -> PRESENT
//     qwen3_8_27b_nvfp4_dflash2.ninfer    weights_id=nvfp4-dflash2   1200 objects -> PRESENT
//     qwen3_8_27b_nvfp4_dflash2_bf16head  weights_id=...-bf16head    1200 objects -> PRESENT
// so the old spelling denied the shortlist head to an artifact that HAS it, and Auto
// could never reach Optimized on it -- which is exactly why the measured grid reads
// `full` for k=5/7/9 on nvfp4 while `--lm-head-draft` (which returns at
// startup_features.h:Explicit) reads `optimized` on the same artifact.
//
// The rows this table leaves false are the ones with NO artifact to read here
// (groupwise-int). They stay false, and plan_load PRINTS the disagreement if such an
// artifact turns out to carry the head -- so the unverified case is a named line in the
// log rather than a silent denial.
[[nodiscard]] constexpr bool shortlist_head_declared(detail::WeightsProfile profile) noexcept {
    switch (profile) {
    case detail::WeightsProfile::Qwen38Nvfp4:
    case detail::WeightsProfile::Qwen38Nvfp4Dspark:
    case detail::WeightsProfile::Qwen38Nvfp4DFlash2:
    case detail::WeightsProfile::Qwen38Nvfp4DFlash2Bf16Head:
    case detail::WeightsProfile::Qwen38Nvfp4ModelOpt:
        return true;
    case detail::WeightsProfile::Qwen36GroupwiseInt:
    case detail::WeightsProfile::Qwen38GroupwiseInt:
    case detail::WeightsProfile::Qwen36Nvfp4:
        return false;
    }
    return false;
}

} // namespace

ModelSamplingDefaults Package::sampling_defaults(std::string_view model) {
    return sampling_presets().require(model).defaults;
}

Package::WeightsProfile Package::resolve_weights(const artifact::ArtifactIdentity& identity) {
    return weights_declarations().require(identity.model_id, identity.weights_id).profile;
}

std::string_view Package::declare_identity(std::string_view model, std::string_view weights) {
    return weights_declarations().require(model, weights).target_key;
}


EngineOptions Package::resolved_auto_speculative(const EngineOptions& options,
                                                WeightsProfile weights_profile) {
    return resolved_auto_speculative(options, weights_profile, nullptr);
}

EngineOptions Package::resolved_auto_speculative(const EngineOptions& options,
                                                    WeightsProfile weights_profile,
                                                    std::string_view* reason) {
    EngineOptions resolved = options;
    // REPORT SITE FOR THE HEAD DECISION (instrument only -- it computes no value that is
    // used downstream; the returned options are bit-identical to what they were before).
    //
    // This function is the ONLY frame in which the user's `ProposalHead::Auto` spelling still
    // exists. registry.cpp:109-110 resolves `--spec auto` here and then hands the *resolved*
    // options on -- the same `resolved_options` it reads `.host_pinned_bytes` off a few lines
    // earlier is what reaches `Target::plan_load` at registry.cpp:138. So the guard that used
    // to sit in plan_load (`options.speculative.proposal_head == ProposalHead::Auto`) was
    // identically false on the cold-start path: it counted zero lines on 10/10 arms while the
    // tensor plane proved the head HAD changed (673/6 tensors and 19.73 GiB against the
    // full-head cell's 671/6 and 19.40 GiB). The decision is reported here instead, where the
    // Auto spelling is still alive, and the `== Auto` gate keeps it to exactly ONE line per
    // resolution -- the re-resolution inside plan_load() sees a concrete head and stays quiet.
    //
    // The reason vocabulary is startup_features.h's own `proposal_head_reason`, so this line
    // and the resolver cannot disagree about WHY. The artifact's side of the same decision
    // (the binder's payload truth) is reported separately by plan_load(), the only place
    // that has a binder.
    // `head_reason` is filled by resolved_proposal_head's OWN out-parameter at the two call
    // sites below -- NOT by a second call to proposal_head_reason. That distinction is
    // load-bearing: the fourth argument the resolver receives is NOT the raw window, because an
    // adaptive window (0) is passed as kMtpShortlistMinimumDrafts. Recomputing the reason from
    // `draft_tokens` therefore produced `head=optimized reason=mtp-below-minimum` on
    // `--spec auto` (window 0) -- a line that printed but did not tell the truth. Reading the
    // reason out of the deciding call makes that disagreement impossible by construction.
    // (startup_features.h:100's `reason` out-parameter had no consumer anywhere in the tree
    // before this change; proposal_head_reason's only call site was the dead guard that used to
    // sit in plan_load. It now has exactly ONE call site -- inside resolved_proposal_head -- and
    // the deciding calls below plus plan_load's in-force line are its three consumers.)
    std::string_view head_reason = "unresolved";
    const auto report_head_resolution = [&](const EngineOptions& final_options) {
        // The authority hand-off: whoever asked for the reason gets the DECIDER's own
        // out-parameter, never a recomputation. Written before the early return so that a caller
        // with an already-concrete head (the re-resolution inside plan_load) also gets a value.
        if (reason != nullptr) { *reason = head_reason; }
        if (options.speculative.proposal_head != ProposalHead::Auto) { return; }
        std::fprintf(stderr,
                     "ninfer: qwen3_6_27b proposal head auto-resolved head=%s reason=%.*s "
                     "(profile=%u, backend=%u, window=%u, declared shortlist head=%s)\n",
                     final_options.speculative.proposal_head == ProposalHead::Optimized
                         ? "optimized"
                         : "full",
                     static_cast<int>(head_reason.size()), head_reason.data(),
                     static_cast<unsigned>(weights_profile),
                     static_cast<unsigned>(final_options.speculative.backend),
                     final_options.speculative.draft_tokens,
                     shortlist_head_declared(weights_profile) ? "present" : "absent");
    };
    if (options.speculative.backend != SpeculativeBackend::Auto) {
        // An explicit backend still needs Auto resolved: it must never reach the planner,
        // the load plan or the program, and a disabled run must end on the full head.
        //
        // THE GATE ON THE FOURTH ARGUMENT IS has_shortlist_head, NOT THE DRAFT WINDOW.
        // resolved_proposal_head (startup_features.h:61-72) reaches its draft_tokens test only
        // after the early return at startup_features.h:66
        //     if (!has_shortlist_head) { return ProposalHead::Full; }
        // and that third argument is `weights_profile == WeightsProfile::Qwen38Nvfp4DFlash2`
        // (line 97 below) -- false for every other profile, including the nvfp4 artifact this
        // target ships. On such an artifact the `0 -> 5` rewrite in the fourth argument is a
        // no-op for every pinned width as well as for the adaptive spelling: the measured grid
        // reads `full` for k=5 (B1), k=7 (B3a) and k=9 (B3b), and the only two `optimized` cells
        // (B2o, C1) name the head explicitly with --lm-head-draft, which returns at
        // startup_features.h:65 before the draft_tokens test.
        // Note also that for k >= kMtpShortlistMinimumDrafts the ternary is the identity, so it
        // cannot be what makes those three cells full.
        //
        // It is NOT a write of the draft window either: the rewrite is an ARGUMENT to
        // resolved_proposal_head and only its result lands in .proposal_head.
        // resolved.speculative.draft_tokens keeps the raw value, so the adaptive gate
        // (layouts_impl.h:1648-1650 needs draft_tokens == 0 && draft_tree_paths == 0) and the
        // captured ladder are untouched by it.  draft_tokens == 0 stays the ADAPTIVE spelling.
        //
        // It is nevertheless not dead code in general and is kept deliberately: for an artifact
        // whose profile IS Qwen38Nvfp4DFlash2, run with `--spec mtp`, head Auto and an adaptive
        // window, 5 >= kMtpShortlistMinimumDrafts is the only thing that takes the shortlist
        // head, and program_impl.h:10463-10471 then floors every chosen rung at
        // kMtpShortlistMinimumDrafts because that head is active. Deleting the rewrite would
        // change exactly that combination and nothing else.
        // SUPERSEDED (HEAD-FLASHNEXT): that "combination" is not the only thing the proxy
        // changed -- it also denied the head to every nvfp4 artifact that HAS it. See
        // shortlist_head_declared() above and the payload cross-check in plan_load().
        resolved.speculative.proposal_head = qwen3_6::resolved_proposal_head(
            resolved.speculative.proposal_head, resolved.speculative.backend,
            shortlist_head_declared(weights_profile),
            resolved.speculative.draft_tokens == 0 ? qwen3_6::kMtpShortlistMinimumDrafts
                                                  : resolved.speculative.draft_tokens,
            &head_reason);
        report_head_resolution(resolved);
        return resolved;
    }
    // The artifact weights decide the backend, not the context length: a
    // DFlash2 artifact has no MTP draft head (the two are mutually
    // exclusive), so auto must pick DFlash2 even at long contexts - a memory
    // shortfall then surfaces as a clear reservation error instead of a
    // confusing weights mismatch. A non-DFlash2 artifact defaults to MTP; the
    // single-source ModelOpt profile lands here, which is what its object plan
    // declares (twelve mtp/* objects and no dflash/dflash2 object at all).
    // landq/unlock -- THE VISION CONJUNCT BELOW, NAMED.
    //
    // FIRST, A HARD CORRECTION TO THE COMMENT ABOVE, because this landing must not endorse it. The
    // sentence above states an ARTIFACT fact -- "a DFlash2 artifact has no MTP draft head (the two
    // are mutually exclusive)" -- and THIS BOX'S ARTIFACT CONTRADICTS IT. Read off the container's
    // own objects[] (byte caliber): qwen3_8_27b_nvfp4_dflash2.ninfer declares all TWELVE objects
    // the mtp group requires (bindings.cpp:591-611), 451,267,584 B, formats W8G32_F16S/BF16 --
    // byte-for-byte the same MTP block the plain nvfp4 artifact declares, alongside its 76
    // dflash2/* objects and its text/draft_head. Readings: dl/unlock/logs/22_mtp_sizes.txt and
    // 21_mtp_probe.txt. So "a DFlash2 artifact has no MTP draft head" is true of the ONE-SLOT
    // BACKEND ENUM (startup_features.h:19 makes mtp() and dflash2() exclusive by construction) and
    // FALSE of the container. The sentence above is left exactly as it stands -- correcting it is a
    // separate finding that this line reports and does NOT land -- and this note is here so the two
    // facts are never read as one.
    //
    // NOW THE CONJUNCT BELOW. Nothing in the tree says why `--vision` vetoes that choice: the
    // conjunct was added with the rest of this chain (commit 582d979e, 2026-09-12) whose message
    // never mentions vision, and the veto is SILENT -- the run simply becomes MTP.
    //
    // What the readings support is that the pair is refused because it was NEVER CO-VALIDATED,
    // which is a policy and not an artifact fact:
    //   * NOT an artifact limit: 9 of the 91 readable .ninfer artifacts on this box declare a
    //     complete DFlash2 head and a complete vision tower at once, and all 9 also declare a
    //     complete MTP block (byte caliber, the containers' own objects[];
    //     dl/unlock/logs/20_artifact_census.txt).
    //   * NOT a designed pairing: the upstream family ships its Vision companion as a separate
    //     `vision-mtp-bf16` head, and its acceptance-rate plan never spells vision beside DFlash2.
    // So the fall-through keeps its behaviour exactly, and gains a NAMED line: a silent veto states
    // no reason at all, and "not validated" is the reason there is.
    if (options.enable_vision &&
        (weights_profile == detail::WeightsProfile::Qwen38Nvfp4DFlash2 ||
         weights_profile == detail::WeightsProfile::Qwen38Nvfp4Dspark)) {
        std::fprintf(stderr,
                     "ninfer: qwen3_6_27b --spec auto with --vision: profile %u carries a %s "
                     "draft head, and the draft-head + Vision pair is not co-validated, so auto "
                     "resolves to MTP. POLICY refusal, not an artifact limit -- this artifact "
                     "holds the draft head, the Vision tower and the MTP block (dl/unlock). "
                     "Spell --spec %s --vision explicitly to run the unvalidated pair.\n",
                     static_cast<unsigned>(weights_profile),
                     weights_profile == detail::WeightsProfile::Qwen38Nvfp4DFlash2 ? "DFlash2"
                                                                                   : "DFlash",
                     weights_profile == detail::WeightsProfile::Qwen38Nvfp4DFlash2 ? "dflash2"
                                                                                   : "dflash");
    }
    if (weights_profile == detail::WeightsProfile::Qwen38Nvfp4DFlash2 &&
        !options.enable_vision) {
        resolved.speculative.backend = SpeculativeBackend::DFlash2;
        if (resolved.speculative.draft_tokens == 0) { resolved.speculative.draft_tokens = 7; }
    } else if (weights_profile == detail::WeightsProfile::Qwen38Nvfp4Dspark &&
               !options.enable_vision) {
        resolved.speculative.backend = SpeculativeBackend::DFlash;
        if (resolved.speculative.draft_tokens == 0) { resolved.speculative.draft_tokens = 7; }
    } else {
        // MTP, and the draft window is left at 0 = ADAPTIVE. This is the whole point of the
        // change: auto used to freeze a constant 3 here, and 3 is wrong for most content
        // (measured optima: code 9, high entropy 5, strong repetition 15, Chinese 5). With 0 the
        // planner captures the width ladder and the survival/cost criterion picks the rung per
        // round, so `--spec auto` is content adaptive. A pinned width is spelled
        // `--spec mtp --draft-tokens k`.
        //
        // Cost: the MTP decode frame and the capture ladder are sized from the ladder top
        // (kMtpWindowLadderTop = 15), not from 3, and the graph allowance grows to one
        // executable per rung per batch size. That is the price of the ladder; it is bounded by
        // --graph-capture-ceiling, which defers the rung captures a short run never needs.
        //
        // The proposal head stays Full, and the reason is has_shortlist_head -- not the window.
        // The third argument at line 144 is false for this artifact, so the early return at
        // startup_features.h:66 fires before the `draft_tokens >= kMtpShortlistMinimumDrafts`
        // test at startup_features.h:68 can run.  (The parent report cited that early return as
        // startup_features.h:62; in this tree :62 is the `SpeculativeBackend backend,` parameter
        // and the return is at :66.)
        // The old text here said "adaptive passes 0".  What reaches the *window* is 0, but what
        // reaches the *head argument* 14 lines below is 5: the same `0 -> 5` rewrite the
        // explicit-backend branch applies.  It is inert only because of that early return, which
        // is why the conclusion was right and the reason was not.  Readings: an adaptive MTP run
        // on this artifact reads `full` with realized mean 15.00 (A1), and the `optimized`
        // cells (C1, realized mean 9.00) come from --lm-head-draft, not from this branch.
        // CORRECTED (HEAD-FLASHNEXT): those three cells read `full` because the third
        // argument was the quantization-flavour proxy, NOT because this artifact lacks the
        // head. The parsed "false for this artifact" was an artifact of the spelling; the
        // artifact carries text/draft_head (see shortlist_head_declared()).
        // What stays deliberately undone is taking the shortlist head on MTP for a non-DFlash2
        // artifact at all.  The per-round floor the old text called missing is not missing where
        // that head is active: program_impl.h:10463-10471 sets
        // head_floor = kMtpShortlistMinimumDrafts when proposal_head == Optimized, and the
        // companion check at program_impl.h:10483-10487 refuses a rung outside the captured
        // ladder.
        resolved.speculative.backend = SpeculativeBackend::Mtp;
    }
    // Resolved after the backend: an auto run that ends disabled must land on the full
    // head (layouts_impl.h requires it). Only the DFlash2 profile is wired to the
    // shortlist draft head, so the bf16-head profile keeps the full head.
    resolved.speculative.proposal_head = qwen3_6::resolved_proposal_head(
        resolved.speculative.proposal_head, resolved.speculative.backend,
        shortlist_head_declared(weights_profile),
        resolved.speculative.draft_tokens == 0 ? qwen3_6::kMtpShortlistMinimumDrafts
                                              : resolved.speculative.draft_tokens,
        &head_reason);
    report_head_resolution(resolved);
    return resolved;
}

Package::LoadPlan Package::plan_load(artifact::Binder& binder, const EngineOptions& options,
                                     WeightsProfile weights_profile) {
    // PAYLOAD CROSS-CHECK. `has_shortlist_head` is a claim about this artifact's object
    // plan, so it is checked against that plan here, where the binder exists, instead of
    // being trusted (the same "never silently trust the constant" rule the qwen4_exp
    // identity header applies to geometry). Both directions are named:
    //   * declared TRUE, artifact does NOT carry the objects -> throw. The head decision
    //     would take Optimized and the run would either mis-size the draft workspace or
    //     die later inside the draft step with a message that names neither side.
    //   * declared FALSE, artifact DOES carry them -> one stderr line and the run
    //     proceeds, the convention registry.cpp:39-42 already uses for a context-cost
    //     preset miss. The shortlist head exists and is going UNUSED; that is now a
    //     fact on stderr before the first token instead of having to be inferred from a
    //     log that reads `full`.
    const bool artifact_has_shortlist_head =
        binder.find_tensor("text/draft_head") != nullptr &&
        binder.find_tensor("text/draft_head_token_ids") != nullptr;
    if (shortlist_head_declared(weights_profile) && !artifact_has_shortlist_head) {
        throw std::invalid_argument(
            "qwen3_6_27b: this weights profile is declared to carry the shortlist proposal "
            "head but the artifact has no text/draft_head / text/draft_head_token_ids; the "
            "declared and the loaded sides disagree, refusing to size the draft path either way");
    }
    if (!shortlist_head_declared(weights_profile) && artifact_has_shortlist_head) {
        std::fprintf(stderr,
                     "ninfer: qwen3_6_27b artifact carries text/draft_head "
                     "(text/draft_head_token_ids) but weights profile %u does not declare it; "
                     "the shortlist proposal head stays unused for this run\n",
                     static_cast<unsigned>(weights_profile));
    }
    // The reason for the head that will be IN FORCE, taken from the decider instead of being
    // recomputed here. This frame is where the Auto spelling has ALREADY left (the registry
    // resolves first), so on the cold-start path it cannot name a cause -- but when this frame
    // IS the decider (a direct package-API caller) the cause exists, and the out-parameter is the
    // only way to get it without spelling the decision a second time.
    std::string_view in_force_reason = "unresolved";

    const EngineOptions resolved =
        Package::resolved_auto_speculative(options, weights_profile, &in_force_reason);
    // WHY THIS IS NO LONGER GATED ON `options.speculative.proposal_head == Auto`.
    //
    // That guard was right about the HEAD and wrong about the CALLER. registry.cpp:109-110
    // resolves `--spec auto` before handing the options on -- the same `resolved_options` it
    // reads `.host_pinned_bytes` off four lines earlier is what reaches this function at
    // registry.cpp:138 -- so the Auto spelling has already been turned into a concrete head
    // by the time plan_load runs. The condition was therefore identically false on the
    // cold-start path: 10/10 arms counted zero lines here while the tensor plane proved the
    // head HAD changed (673/6 tensors, 19.73 GiB, against the full-head cell's 671/6 and
    // 19.40 GiB). The head actually IN FORCE is reported unconditionally now, and `reason`
    // names which side consumed the spelling: the Auto head still visible here (a direct
    // plan_load caller) versus already consumed upstream. Two different facts that used to be
    // one silence.
    //
    // The auto->concrete mapping and its WHY are reported by resolved_auto_speculative()
    // above -- the only frame where the Auto spelling exists. This line adds the ARTIFACT's
    // side of the same decision, which needs the binder and so can only be answered here.
    //
    // The field is `intent=`, NOT `reason=`. In the `consumed-upstream` case the value asserts
    // nothing about the cause: this site CANNOT tell "registry consumed an `Auto` spelling" from
    // "the user pinned --lm-head-draft / --no-lm-head-draft": both arrive as a concrete head, and
    // the backend cannot be used as a tell either because resolved_auto_speculative() resolves
    // that too. In the `auto:<reason>` case the cause IS named, and it is the decider's own
    // out-parameter rather than a recomputation performed here.
    // A line reading `reason=auto-...` in the pinned case would print without telling the
    // truth, so the pinned case is reported as what it actually is here -- the spelling is
    // gone -- and the WHY is left to the site that can see it. That site prints IFF the
    // spelling was Auto, so an absent `auto-resolved` line means the head was pinned.
    {
        const bool auto_spelling_visible = options.speculative.proposal_head == ProposalHead::Auto;
        char intent[64];
        if (auto_spelling_visible) {
            // `in_force_reason` is the DECIDER's own out-parameter, so this line cannot name a
            // reason that disagrees with the `head=` printed beside it. The direct
            // `proposal_head_reason` call that used to be here re-derived the reason from the RAW
            // window (0 for the adaptive spelling) and would have printed `mtp-below-minimum`
            // next to `head=optimized` -- the one reading this instrument exists to rule out.
            std::snprintf(intent, sizeof intent, "auto:%.*s",
                          static_cast<int>(in_force_reason.size()), in_force_reason.data());
        } else {
            std::snprintf(intent, sizeof intent, "consumed-upstream");
        }
        std::fprintf(stderr,
                     "ninfer: qwen3_6_27b proposal head in force head=%s intent=%s "
                     "(artifact shortlist head=%s, profile=%u, backend=%u, window=%u, "
                     "auto-spelling=%s)\n",
                     resolved.speculative.proposal_head == ProposalHead::Optimized ? "optimized"
                                                                                   : "full",
                     intent,
                     artifact_has_shortlist_head ? "present" : "absent",
                     static_cast<unsigned>(weights_profile),
                     static_cast<unsigned>(resolved.speculative.backend),
                     resolved.speculative.draft_tokens,
                     auto_spelling_visible ? "visible" : "already-consumed-upstream");
    }
    return LoadPlan(std::make_unique<LoadPlan::Impl>(
        weights_profile,
        detail::bind_artifact(binder, weights_profile, qwen3_6::startup_features(resolved))));
}
std::unique_ptr<Package::LoadedModel>
Package::construct_loaded_model(LoadPlan&& plan, artifact::MaterializedArtifact&& materialized) {
    if (plan.impl_ == nullptr) { throw std::invalid_argument("target load plan is empty"); }
    auto impl = std::make_unique<LoadedModel::Impl>(
        plan.impl_->weights_profile, std::move(plan.impl_->plan.bindings), std::move(materialized));
    plan.impl_.reset();
    return std::unique_ptr<LoadedModel>(new LoadedModel(std::move(impl)));
}

Package::Frontend Package::make_frontend(const LoadedModel& model, const EngineOptions& options) {
    if (model.impl_ == nullptr) { throw std::invalid_argument("loaded model is empty"); }
    return qwen3_6::make_frontend(model.impl_->data.frontend,
                                  qwen3_6::FrontendOptions{
                                      .vision_enabled = model.impl_->data.runtime.features.vision,
                                      .max_context    = options.max_context,
                                      .media_cache_bytes        = options.media_cache_bytes,
                                      .media_live_bytes         = options.media_live_bytes,
                                      .media_preprocess_threads = options.media_preprocess_threads,
                                  });
}

Package::SequencePlanner Package::make_sequence_planner(DeviceContext& device,
                                                        const EngineOptions& options,
                                                        WeightsProfile weights_profile) {
    return qwen3_6::make_sequence_planner<detail::Variant>(device, options, weights_profile);
}

std::unique_ptr<Package::Program>
Package::create_program(const LoadedModel& model, SequencePlan&& plan, DeviceContext& device) {
    if (model.impl_ == nullptr) { throw std::invalid_argument("loaded model is empty"); }
    return qwen3_6::create_program<detail::Variant>(
        model.impl_->data.runtime, model.impl_->weights_profile, std::move(plan), device);
}

void Package::export_head_weights(const LoadedModel& model, const char* directory) {
    if (model.impl_ == nullptr) { throw std::invalid_argument("loaded model is empty"); }
    const auto& view = model.impl_->data.runtime;
    const auto dump = [&](const char* name, const Weight& weight) {
        if (weight.qdata == nullptr || weight.n <= 0 || weight.k <= 0) {
            throw std::runtime_error(std::string("head export: ") + name + " is unavailable");
        }
        const std::string base = std::string(directory) + "/" + name;
        const std::size_t code_bytes =
            static_cast<std::size_t>(weight.n) * static_cast<std::size_t>(weight.k);
        const std::size_t scale_bytes = static_cast<std::size_t>(weight.n) * 2;
        std::vector<std::byte> codes(code_bytes);
        std::vector<std::byte> scales(scale_bytes);
        CUDA_CHECK(cudaMemcpy(codes.data(), weight.qdata, code_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(scales.data(), weight.scales, scale_bytes, cudaMemcpyDeviceToHost));
        std::FILE* f = std::fopen((base + "_codes.bin").c_str(), "wb");
        if (f == nullptr) { throw std::runtime_error("cannot open " + base + "_codes.bin"); }
        std::fwrite(codes.data(), 1, codes.size(), f);
        std::fclose(f);
        f = std::fopen((base + "_scales.bin").c_str(), "wb");
        if (f == nullptr) { throw std::runtime_error("cannot open " + base + "_scales.bin"); }
        std::fwrite(scales.data(), 1, scales.size(), f);
        std::fclose(f);
    };
    dump("embed", view.token_embedding);
    dump("head", view.output_head);
}

} // namespace ninfer::targets::qwen3_6_27b
