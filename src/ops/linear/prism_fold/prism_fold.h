#pragma once
// prism_fold.h -- THE RUNTIME HALF of "who applies the Prism/Bonsai rotation?".
//
// This header is the C++ mirror of `tools/convert/gguf_fold_route.py`, and it is the only
// place on the runtime side where the answer to that question is computed.  The two must
// agree CELL FOR CELL; `tools/ternfold_probe/` dumps both tables and diffs them, because two
// hand-written copies of a truth table are free to drift and a drifted cell is a silent
// wrong matmul rather than a compile error.
//
// WHY THE DECISION IS NOT A FLAG
// ------------------------------
// A rotated checkpoint stores its tensors in the basis of `A = H . D` (normalized Sylvester
// Walsh-Hadamard, block 1024, times a diagonal of +/-1 signs).  The map from that basis into
// the primal one must be applied EXACTLY ONCE, and both error directions are silent:
//
//   zero times -- weights rotated, activation not: the matmul is over the wrong pairing.
//   two  times -- `A` is an orthonormal involution, so `A^-1 A = I`: the second application
//                 CANCELS the first and the result equals the zero-times answer.
//
// Measured on the real 5.95 GB Bonsai-2 ternary file (dl/ternruntime2/10_probe_fold.txt): the
// two-transforms answer and the zero-transforms answer agree to `7.153e-07`, while both differ
// from the correct one by `2.474e+00` -- the concealment is 3.459e+06 times smaller than what
// it conceals.  So no runtime self-check can detect "did we transform twice" from the output,
// and the discriminator must be DECLARED BY THE CHECKPOINT and checked on both sides BEFORE a
// byte moves.  That is what this header does, and it is why `requested_policy` below is
// documented as a FILTER: it can only make the verdict stricter.
//
// THE THREE INPUTS
//   basis                    -- a fact about the CHECKPOINT, read from the checkpoint.
//   runtime_supports_transform -- a fact about THIS BUILD, not a user preference.  It is
//                               hardcoded false in the engine today; see PRISM_FOLD_RUNTIME_
//                               SUPPORTS_TRANSFORM and prism_fold_runtime_supports_transform().
//   requested_policy          -- the operator's rail (`auto` / `require_primal` / `require_folded`).
//
// Host-callable by design, with no CUDA in this header: a truth table that has no host
// instrument is a claim nobody has checked.
//
// ARTIFACT GATE STATUS -- READ THIS BEFORE BELIEVING ANYTHING ABOUT ROUTE A
// -----------------------------------------------------------------------
// `prism_fold_decide` is reachable from the engine ONLY as a refusal.  The device half of
// Route A lives in `prism_fold.cuh` / `prism_fold_launch.cu`, and the latter is NOT in
// `src/CMakeLists.txt`'s `ninfer_ops` source list, which is explicit and glob-free
// (G1/MEASURED, dl/ternruntime3/01_build.txt).  Adding it there is a CMakeLists edit, which
// this line is forbidden to make.  So on the engine side this header's honest contribution is
// the REFUSAL: a folded checkpoint that reaches a `linear` today must be refused by name, and
// PRISM_FOLD_RUNTIME_SUPPORTS_TRANSFORM is `false` so that it is.  See the report's per-route
// ledger for what that does and does not prove.

#include <cstdint>
#include <string>
#include <string_view>

namespace ninfer::ops::prism_fold {

// --------------------------------------------------------------------------- the enums
// The spellings below are the contract with `tools/convert/gguf_fold_route.py` and are compared
// as STRINGS by the probe.  Renaming one without the other is the drift this file warns about.

//: What the checkpoint says about its own weights.  Read from the checkpoint; never a flag.
enum class Basis {
    //: weights are in the natural basis; the converter applied `A^-1` (c = 1)
    Primal,
    //: weights are in `A`'s basis; the runtime must apply `A` to the activation (r = 1)
    Folded,
    //: the checkpoint does not say.  NOT a licence to guess -- see decide().
    Unstated,
};

//: The operator's rail on the runtime.  A FILTER: it can only make the verdict stricter.
enum class Policy {
    Auto,
    RequirePrimal,   //: "I assert this checkpoint needs no runtime transform"
    RequireFolded,   //: "I assert this checkpoint needs the runtime transform"
};

//: What the runtime must do about `linear` activations for this checkpoint.
enum class Verdict {
    RunNoTransform,
    RunTransform,
    Refuse,
};

//: What a converter is asked to do with a rotated source.  Single-valued on purpose: the two
//: routes are two ways to the SAME correctness, so "both" is not a value this enum has.
enum class Route {
    PassThrough,   // Route A: emit the folded weights untouched, and DECLARE them
    FoldBack,      // Route B: apply `A^-1` in the converter, emit primal, runtime does nothing
    Refuse,        // the default: emit nothing
};

inline constexpr std::string_view kPrismFoldBasisPrimal   = "primal";
inline constexpr std::string_view kPrismFoldBasisFolded   = "folded";
inline constexpr std::string_view kPrismFoldBasisUnstated = "unstated";
inline constexpr std::string_view kPrismFoldPolicyAuto          = "auto";
inline constexpr std::string_view kPrismFoldPolicyRequirePrimal = "require_primal";
inline constexpr std::string_view kPrismFoldPolicyRequireFolded = "require_folded";
inline constexpr std::string_view kPrismFoldVerdictNoTransform  = "RUN_NO_TRANSFORM";
inline constexpr std::string_view kPrismFoldVerdictTransform    = "RUN_TRANSFORM";
inline constexpr std::string_view kPrismFoldVerdictRefuse       = "REFUSE";
inline constexpr std::string_view kPrismFoldRoutePassThrough    = "pass_through";
inline constexpr std::string_view kPrismFoldRouteFoldBack       = "fold_back";
inline constexpr std::string_view kPrismFoldRouteRefuse         = "refuse";

//: The key a checkpoint carries its basis under.  A folded checkpoint that does not carry it is
//: inadmissible by construction: without it the runtime cannot tell a folded file from an
//: unrotated one, and for this model family those two differ in one tensor's inner head order
//: UNDER THE SAME NAME (`gguf_hadamard.ssm_out_head_order`), so the difference is a wrong
//: pairing of features and not a scale error.
inline constexpr std::string_view kPrismFoldBasisKey = "prism.fold.basis";

[[nodiscard]] constexpr std::string_view basis_name(Basis b) noexcept {
    switch (b) {
    case Basis::Primal:   return kPrismFoldBasisPrimal;
    case Basis::Folded:   return kPrismFoldBasisFolded;
    case Basis::Unstated: return kPrismFoldBasisUnstated;
    }
    return kPrismFoldBasisUnstated;
}

[[nodiscard]] constexpr std::string_view policy_name(Policy p) noexcept {
    switch (p) {
    case Policy::Auto:          return kPrismFoldPolicyAuto;
    case Policy::RequirePrimal: return kPrismFoldPolicyRequirePrimal;
    case Policy::RequireFolded: return kPrismFoldPolicyRequireFolded;
    }
    return "invalid";
}

[[nodiscard]] constexpr std::string_view verdict_name(Verdict v) noexcept {
    switch (v) {
    case Verdict::RunNoTransform: return kPrismFoldVerdictNoTransform;
    case Verdict::RunTransform:   return kPrismFoldVerdictTransform;
    case Verdict::Refuse:         return kPrismFoldVerdictRefuse;
    }
    return "REFUSE";
}

[[nodiscard]] constexpr std::string_view route_name(Route r) noexcept {
    switch (r) {
    case Route::PassThrough: return kPrismFoldRoutePassThrough;
    case Route::FoldBack:    return kPrismFoldRouteFoldBack;
    case Route::Refuse:      return kPrismFoldRouteRefuse;
    }
    return "refuse";
}

//: Parse the value a checkpoint carries under `kPrismFoldBasisKey`.  Anything unrecognised,
//: including an absent key, is `Unstated` -- and `Unstated` is refused by decide(), so a typo
//: in a checkpoint degrades to a refusal and never to a default.
[[nodiscard]] constexpr Basis basis_from_string(std::string_view s) noexcept {
    if (s == kPrismFoldBasisPrimal) return Basis::Primal;
    if (s == kPrismFoldBasisFolded) return Basis::Folded;
    return Basis::Unstated;
}

[[nodiscard]] constexpr Policy policy_from_string(std::string_view s) noexcept {
    if (s == kPrismFoldPolicyRequirePrimal) return Policy::RequirePrimal;
    if (s == kPrismFoldPolicyRequireFolded) return Policy::RequireFolded;
    return Policy::Auto;   // `auto` and anything unrecognised; unrecognised cannot OPEN anything
}

// --------------------------------------------------------------------------- this build
//: Does THIS BUILD of the engine carry the activation-side transform (Route A)?
//
// It is `false`, and it is a compile-time FACT rather than a runtime flag, because a runtime
// flag would be a way to claim a capability the binary does not have -- which turns the gate
// into a guard that cannot go red.  Flipping this to `true` without the device half actually
// linked would make the engine accept a folded checkpoint and compute the wrong matmul
// SILENTLY, so the constant is tied to a link-time symbol below rather than to a hope: with
// the device half absent the engine does not define `prism_fold_activation_transform_linked`,
// and the `#if` keeps the constant false.  When `prism_fold_launch.cu` is admitted to
// `ninfer_ops` the symbol exists, and the constant becomes true only in that build.
#if defined(PRISM_FOLD_ACTIVATION_TRANSFORM_LINKED)
inline constexpr bool kPrismFoldRuntimeSupportsTransform = true;
#else
inline constexpr bool kPrismFoldRuntimeSupportsTransform = false;
#endif

[[nodiscard]] constexpr bool prism_fold_runtime_supports_transform() noexcept {
    return kPrismFoldRuntimeSupportsTransform;
}

// --------------------------------------------------------------------------- the decision
//: The total three-input verdict.  Mirrors `gguf_fold_route.decide` cell for cell.
//
// The order of the tests is part of the function: an unstated basis is refused BEFORE the
// policy is consulted, so no policy can rescue it.
[[nodiscard]] constexpr Verdict decide(Basis basis, bool runtime_supports_transform,
                                      Policy policy) noexcept {
    if (basis == Basis::Unstated) {
        return Verdict::Refuse;
    }
    if (basis == Basis::Primal) {
        // The weights are already primal, so the converter has already done its `A^-1` and the
        // runtime must do nothing.  A policy that asks the runtime to transform anyway is
        // asking for the SECOND application: the double-apply door, refused.
        if (policy == Policy::RequireFolded) {
            return Verdict::Refuse;
        }
        return Verdict::RunNoTransform;
    }
    // basis == Folded
    // A policy that forbids the one application this checkpoint needs is the skipped-transform
    // door.  Refused, not silently obeyed.
    if (policy == Policy::RequirePrimal) {
        return Verdict::Refuse;
    }
    if (!runtime_supports_transform) {
        // The runtime has no transform, so nothing can bring the activation into the weights'
        // basis.  A refusal, not a warning: the alternative is the silent wrong.
        return Verdict::Refuse;
    }
    return Verdict::RunTransform;
}

//: The verdict for THIS build, which is the only spelling a call site should use.
[[nodiscard]] constexpr Verdict decide_for_this_build(Basis basis, Policy policy) noexcept {
    return decide(basis, prism_fold_runtime_supports_transform(), policy);
}

//: `(c, r)`: did the converter apply `A^-1`, did the runtime apply `A`.  `c + r == 1` on every
//: admitted cell; a REFUSE applies nothing, so it contributes `(0, 0)`.
[[nodiscard]] constexpr int converter_transforms(Basis basis, Verdict v) noexcept {
    if (v != Verdict::RunNoTransform) return 0;
    return basis == Basis::Primal ? 1 : 0;
}

[[nodiscard]] constexpr int runtime_transforms(Basis basis, Verdict v) noexcept {
    if (v == Verdict::RunTransform) return 1;
    return 0;
}

//: Criterion C1 checked against the executable table rather than against prose: `true` when any
//: input makes the total number of applications two, i.e. when the mechanism is broken.
[[nodiscard]] constexpr bool prism_fold_double_apply_is_reachable() noexcept {
    constexpr Basis bases[] = {Basis::Primal, Basis::Folded, Basis::Unstated};
    constexpr Policy policies[] = {Policy::Auto, Policy::RequirePrimal, Policy::RequireFolded};
    for (const bool supports : {true, false}) {
        for (const Basis b : bases) {
            for (const Policy p : policies) {
                const Verdict v = decide(b, supports, p);
                if (converter_transforms(b, v) + runtime_transforms(b, v) == 2) {
                    return true;
                }
            }
        }
    }
    return false;
}
static_assert(!prism_fold_double_apply_is_reachable(),
              "the decision table admits a second transform; this is the silent wrong");

//: C2's totality, as a compile-time fact: every basis x supports x policy cell returns a value
//: of the enum, and every REFUSE cell is `Basis::Unstated`, the double-apply door, the
//: skipped-transform door, or a runtime that cannot transform.  Nothing falls through.
[[nodiscard]] constexpr bool prism_fold_table_is_total() noexcept {
    constexpr Basis bases[] = {Basis::Primal, Basis::Folded, Basis::Unstated};
    constexpr Policy policies[] = {Policy::Auto, Policy::RequirePrimal, Policy::RequireFolded};
    for (const bool supports : {true, false}) {
        for (const Basis b : bases) {
            for (const Policy p : policies) {
                const Verdict v = decide(b, supports, p);
                bool known = v == Verdict::RunNoTransform || v == Verdict::RunTransform ||
                             v == Verdict::Refuse;
                if (!known) return false;
                if (v == Verdict::Refuse) {
                    const bool justified =
                        b == Basis::Unstated ||
                        (b == Basis::Primal && p == Policy::RequireFolded) ||
                        (b == Basis::Folded && p == Policy::RequirePrimal) ||
                        (b == Basis::Folded && !supports);
                    if (!justified) return false;
                }
            }
        }
    }
    return true;
}
static_assert(prism_fold_table_is_total(), "a cell of the decision table is unjustified");

//: C3, as a compile-time fact: `requested_policy` cannot move the runtime transform off the
//: basis.  `RunTransform` is reachable from `Folded` alone, for every policy that is not the
//: explicit refusal, and from no other basis under any policy.
[[nodiscard]] constexpr bool prism_fold_only_folded_opens_the_transform() noexcept {
    for (const bool supports : {true, false}) {
        for (const Policy p : {Policy::Auto, Policy::RequirePrimal, Policy::RequireFolded}) {
            for (const Basis b : {Basis::Primal, Basis::Unstated}) {
                if (decide(b, supports, p) == Verdict::RunTransform) return false;
            }
        }
    }
    return true;
}
static_assert(prism_fold_only_folded_opens_the_transform(),
              "a policy opened the transform on a checkpoint that is not folded");

// --------------------------------------------------------------------------- refusal by name
//: The reason a folded checkpoint is refused today, in full, so no call site has to invent one.
[[nodiscard]] inline std::string prism_fold_refusal_text(Basis basis, Policy policy) {
    std::string out = "prism.fold: refusing a ";
    out += basis_name(basis);
    out += " checkpoint (policy=";
    out += policy_name(policy);
    out += ", runtime_supports_transform=";
    out += (prism_fold_runtime_supports_transform() ? "true" : "false");
    out += "). ";
    if (basis == Basis::Unstated) {
        out += "The checkpoint does not say which basis its weights are in, and the folded and "
               "unrotated files of this family differ in one tensor's inner head order under the "
               "same name, so neither answer is safe. An absent declaration is the unproven "
               "case, not the permissive one.";
    } else if (basis == Basis::Folded && !prism_fold_runtime_supports_transform()) {
        out += "The weights are in the rotation's basis (" + std::string(kPrismFoldBasisKey) +
               "=" + std::string(kPrismFoldBasisFolded) +
               ") and this build carries no activation-side transform, so the activation cannot "
               "be brought into the weights' basis. Running it anyway would pair features across "
               "the rotation and the matmul would be silently wrong -- not a scale error, a wrong "
               "pairing. Refused rather than approximated.";
    } else if (basis == Basis::Primal) {
        out += "A primal checkpoint needs no runtime transform, and the policy asked for one. "
               "Applying the transform on top of a converter that already applied its inverse is "
               "the DOUBLE application: A is an orthonormal involution, so the second call "
               "cancels the first and the result equals applying neither.";
    } else {
        out += "The policy forbids the one application this checkpoint requires.";
    }
    return out;
}

}   // namespace ninfer::ops::prism_fold
