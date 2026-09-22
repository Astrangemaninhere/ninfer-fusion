#pragma once

// ---------------------------------------------------------------------------
// arch_sim.h -- the TEST-ONLY architecture simulator (the "guard-shaped V100").
//
// WHY THIS EXISTS
// ---------------
// User requirement, verbatim: "5090d 你可以搞一个类似守卫的方式模拟 v100" -- on this
// box's RTX 5090 D (sm_120), provide a guard-shaped mechanism that makes the card
// PRESENT ITSELF AS a V100 / sm_70 to the capability + route layers, so the
// non-Blackwell fallback path can actually be exercised on hardware we own.
// The companion requirement ("v100 的支持应该可以方便的迁移到其他卡上") is what
// makes sm_75 -- the 2080Ti rung, the other rung with a named wall -- simulable by
// the same mechanism rather than by a second, Volta-shaped one.
//
// WHAT IT IS
// ----------
// A capability OVERRIDE. It changes the compute capability the two *table* layers
// see -- caps::arch_rung()/evaluate_artifact_formats() via select_route() -- and
// nothing else. It does not touch the CUDA context, the driver, cudaGetDeviceProperties,
// or any device-side __CUDA_ARCH__. The kernels that execute are the ones THIS binary
// was compiled for; only the ROUTE DECISION is simulated. That distinction is the
// whole subject of the "what a simulated run does and does not prove" section of the
// report, and it is restated in sim_banner() below so it cannot be lost in transit.
//
// THE FOUR GUARD PROPERTIES (each is pinned by a test in tests/test_arch_sim.cpp)
// -----------------------------------------------------------------------------
// G1  IMPOSSIBLE BY ACCIDENT. Activation needs THREE independent things, of which
//     the first is a BUILD-time one:
//       (a) the binary was built with -DNINFER_ENABLE_ARCH_SIM=ON (CMake option,
//           default OFF), which defines NINFER_ARCH_SIM_ENABLED;
//       (b) the environment names a rung:      NINFER_SIM_ARCH=70
//       (c) the environment carries an explicit acknowledgement of what this is:
//           NINFER_SIM_ARCH_ACK=I-UNDERSTAND-THIS-IS-NOT-A-V100
//     A stray NINFER_SIM_ARCH in a user's shell therefore does NOT enable it: with
//     (a) absent it is refused by the build fact, and with (c) absent it is refused
//     by the acknowledgement. Neither refusal is silent.
//
// G2  LOUD. Every use prints sim_banner() to stderr, and every route/gate answer
//     produced under the simulator carries a SIMULATED marker in its text. There is
//     no quiet mode and no suppress flag; sim_banner_for() is the only place the
//     text is built, so the wording cannot drift between the entry points.
//
// G3  FAIL CLOSED. A request that cannot be honoured yields SimStatus::Refused,
//     which is a DISTINCT state from Disabled. Callers must not proceed on the real
//     path when it is Refused: select_route(ArchView) returns NoKernelInTree in that
//     state, and require_artifact_formats_supported(ArchView) refuses. Refused is
//     sticky for the process (see the once-flag) so a failure cannot be observed once
//     and then bypassed by a later call.
//
// G4  NEVER RAISES, NEVER EQUALS. The simulated rung must be STRICTLY LOWER than the
//     physical one. A request to simulate the arch the card already is (or a higher
//     one) is Refused, not honoured: simulating a no-op would let a genuine
//     sm_120-only defect hide behind a "simulated" label, and raising the reported
//     capability is the exact bug class this project has already paid for twice
//     (the B200 nvfp4 false positive at src/core/arch_caps.h's sm_100 row, and
//     mma_nvfp4_e4m3 returning sm_100a rc=255). The override is a DOWNGRADE ONLY.
//
// WHAT A SIMULATED RUN PROVES, AND WHAT IT DOES NOT
// -------------------------------------------------
// It proves the code path is TAKEN (which route, which kernel), that the kernel
// EXECUTES, and that its NUMBERS can be compared against the real path's.
// It proves NOTHING about a V100: the cubin being executed is the sm_120a one, the
// sm_70 instruction matrix in the report was produced by ptxas, and the m8n8k4
// channel's lowering is a per-rung SASS fact (kQpnMmaRungs) that no simulator can
// change. No performance claim of any kind may be sourced from a simulated run.
//
// AND THE SUPPORT PROHIBITION IS A SECOND, DIFFERENT PROHIBITION. The paragraph above forbids
// sourcing a PERFORMANCE number (throughput, latency) from a simulated run. src/core/format_probe.h
// forbids sourcing a FUNCTIONAL support verdict from one, and those are not one rule written
// twice: a support claim says "this rung computes this format within the criterion" (functional)
// and a speed claim says "this rung is fast" (not functional), so a run can be a legitimate speed
// experiment and an illegitimate support claim, or the other way round. Both prohibitions stand as
// written above; neither may be traded for the other, and the support half is implemented and
// single-sourced in src/core/format_probe.h's sim_verdict_refusal(), which refuses a simulated
// verdict by name, names the RUNG that was simulated (or reports that the row did not record one),
// states that the SUPPORT is UNMEASURED and the ARITHMETIC is UNVERIFIED, and ends with the
// literal command that would measure the real hardware.
//
// THAT REFUSAL EXISTS BECAUSE THE SIMULATOR HAS A LEGITIMATE USE THAT MUST NOT BE LOST WITH IT:
// exercising a code path this box owns no hardware for. The CUDA 12.8 chain at /mnt/g/cuda12/tk
// reaches compute_50/52/53 (Maxwell, the GTX 960) and compute_60/61/62 (Pascal), and this ladder's
// two named walls are sm_70 (V100) and sm_75 (2080Ti) -- so on those rungs a simulated run is the
// ONLY instrument this tree has that can see the fallback path TAKEN. The correct output of such a
// run is therefore "this path was exercised on a simulated rung; its SUPPORT is UNMEASURED and its
// ARITHMETIC is UNVERIFIED", which is what the refusal says -- not silence, and not a capability
// claim. Deleting the simulator would remove the instrument; marking and refusing keeps it.
//
// THIS FILE IS HOST-ONLY: no CUDA header, no device query. The caller passes the
// physical compute capability in, exactly like arch_caps.h and kernel_route.h, so a
// single-GPU host can exercise every branch.
// ---------------------------------------------------------------------------

#include "core/arch_caps.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <string_view>

namespace ninfer::caps {

// The build fact. NINFER_ARCH_SIM_ENABLED is defined by CMake ONLY when the option
// NINFER_ENABLE_ARCH_SIM=ON is passed; it is not a user-settable -D on the command
// line of a shipping build, and the gate below is why that matters.
#ifdef NINFER_ARCH_SIM_ENABLED
inline constexpr bool kArchSimBuiltIn = true;
#else
inline constexpr bool kArchSimBuiltIn = false;
#endif

// The two environment names, as constants so the tests and the report quote the same
// strings the code reads.
inline constexpr std::string_view kSimArchEnv = "NINFER_SIM_ARCH";
// Deliberately verbose: it must be impossible to set this by muscle memory, by a
// shell profile, or by copying another project's variable.
inline constexpr std::string_view kSimAckEnv = "NINFER_SIM_ARCH_ACK";
inline constexpr std::string_view kSimAckPhrase = "I-UNDERSTAND-THIS-IS-NOT-A-V100";

enum class SimStatus : std::uint8_t {
    Disabled = 0, // no request was made. The default. Zero behaviour change.
    Active,       // a request was made and honoured. Every answer is marked.
    Refused,      // a request was made and NOT honoured. Fail closed: do not
                  // proceed on the real path either.
};

[[nodiscard]] inline std::string_view sim_status_name(SimStatus status) noexcept {
    switch (status) {
    case SimStatus::Disabled: return "disabled";
    case SimStatus::Active: return "active";
    case SimStatus::Refused: return "refused";
    }
    return "unknown-sim-status";
}

// The three-state view of one device. `effective_sm` is what the table layers must use;
// `physical_sm` is what the hardware is, and is always recorded so that no message can
// present the simulated number as the card's number.
struct ArchView {
    int physical_sm      = 0;
    int effective_sm     = 0;
    SimStatus status     = SimStatus::Disabled;
    std::string reason   = {}; // non-empty whenever status != Disabled

    [[nodiscard]] bool simulated() const noexcept { return status == SimStatus::Active; }
    [[nodiscard]] bool failed() const noexcept { return status == SimStatus::Refused; }
    // True when a caller may use effective_sm: either no simulation (effective == physical)
    // or an honoured one. False ONLY for Refused, which is the fail-closed state.
    [[nodiscard]] bool usable() const noexcept { return status != SimStatus::Refused; }
};

// The one place the loud text is built. Mentions the physical card and the rung, states
// that the kernels are NOT sm_70 binaries, and states that no performance claim is valid.
[[nodiscard]] inline std::string sim_banner(const ArchView& view) {
    if (!view.simulated()) { return {}; }
    const ArchRung* rung = arch_rung(view.effective_sm);
    std::string out = "ninfer: *** SIMULATED ARCHITECTURE -- TEST ONLY ***  ";
    out += "physical device sm_";
    out += std::to_string(view.physical_sm);
    out += " is being reported as sm_";
    out += std::to_string(view.effective_sm);
    if (rung != nullptr) {
        out += " (";
        out.append(rung->label);
        out += ", e.g. ";
        out.append(rung->cards);
        out += ")";
    }
    out += " to the capability and route tables, because ";
    out.append(kSimArchEnv);
    out += "=";
    out += std::to_string(view.effective_sm);
    out += " and ";
    out.append(kSimAckEnv);
    out += " were both set and this binary was built with NINFER_ENABLE_ARCH_SIM. The "
           "kernels that will execute are the ones THIS binary was compiled for, NOT "
           "sm_";
    // THE RUNG, NOT A HARDCODED sm_70. This clause used to be the literal string
    // "sm_70 binaries", which was true only for the V100 request: MEASURED 2026-09-18
    // (SIM-RUN) with all five banners captured, the sm_75, sm_86, sm_89 and sm_100 runs each
    // said "NOT sm_70 binaries" while reporting their own rung -- i.e. 4 of 5 rungs named the
    // wrong rung in the one clause whose whole job is to stop a reader from thinking the
    // cubins are the target card's. G2's promise is that this wording cannot drift between
    // entry points; being single-sourced in a rung-blind literal satisfied that and still
    // produced a false sentence. It is now built from view.effective_sm, which is the same
    // number the rest of the banner and every answer below already reports.
    out += std::to_string(view.effective_sm);
    out += " binaries; only the route decision is simulated. A simulated run proves "
           "the path is taken and computes; it proves NOTHING about a real ";
    if (rung != nullptr) {
        out.append(rung->cards);
    } else {
        out += "device of that capability";
    }
    out += ", and no throughput or latency figure from it may be quoted as that "
           "device's. ";
    out.append(view.reason);
    return out;
}

// ---------------------------------------------------------------------------
// The pure decision function
// ---------------------------------------------------------------------------
//
// Every input is a parameter -- the build fact, the two environment values, and the
// physical compute capability -- so ONE test binary can exercise the enabled and the
// not-enabled worlds, the honoured and the refused requests, without a rebuild and
// without a second process. This is the same idiom arch_caps.h uses for the ladder
// ("every row is exercisable on any machine").
//
// `sim_arch_env` / `sim_ack_env` are the RAW environment values: an empty string_view
// means "unset", which is reproducible from a test. The production wrapper below is
// the only place std::getenv is called.
[[nodiscard]] inline ArchView arch_view_for_device_impl(int physical_sm, bool sim_built_in,
                                                       std::string_view sim_arch_env,
                                                       std::string_view sim_ack_env) noexcept {
    ArchView view;
    view.physical_sm  = physical_sm;
    view.effective_sm = physical_sm;

    // No request at all: the default, and the ONLY path a shipping build takes.
    if (sim_arch_env.empty()) {
        view.status = SimStatus::Disabled;
        return view;
    }

    // A request exists. From here on a refusal is recorded, never ignored, so that
    // "requested but not honoured" can never look like "not requested".
    view.status = SimStatus::Refused;

    // (a) the build fact.
    if (!sim_built_in) {
        view.reason =
            "a simulated architecture was requested (";
        view.reason.append(kSimArchEnv);
        view.reason +=
            "=" + std::string(sim_arch_env) +
            ") but this binary was NOT built with -DNINFER_ENABLE_ARCH_SIM=ON, so the "
            "override does not exist in it. Refusing rather than proceeding: a request "
            "that cannot be honoured must not silently run on the real device. Rebuild "
            "with -DNINFER_ENABLE_ARCH_SIM=ON if a simulated run is really what you want.";
        return view;
    }

    // (c) the acknowledgement. Checked before the value so a typo in the rung and a
    // missing ack cannot be confused for one another in the message.
    if (sim_ack_env != kSimAckPhrase) {
        view.reason = std::string(kSimArchEnv) + "=" + std::string(sim_arch_env) +
                      " was requested but " + std::string(kSimAckEnv) + " is ";
        if (sim_ack_env.empty()) {
            view.reason += "unset";
        } else {
            view.reason += "\"" + std::string(sim_ack_env) + "\"";
        }
        view.reason += ", not the required acknowledgement \"" +
                       std::string(kSimAckPhrase) +
                       "\". The simulator is test-only and every use is loud; it cannot "
                       "be enabled by a single stray environment variable. Refusing.";
        return view;
    }

    // (b) the value. Must be a rung of the ladder: an unlisted number would make the
    // simulator itself the source of a capability claim, which is the one thing this
    // table exists to prevent.
    int wanted = 0;
    {
        bool all_digits = !sim_arch_env.empty();
        for (const char c : sim_arch_env) {
            if (c < '0' || c > '9') { all_digits = false; break; }
        }
        if (!all_digits) {
            view.reason = std::string(kSimArchEnv) + "=\"" + std::string(sim_arch_env) +
                          "\" is not a decimal compute capability. Give a rung this "
                          "table contains, e.g. " + std::string(kSimArchEnv) + "=70 for a "
                          "V100 or " + std::string(kSimArchEnv) + "=75 for a 2080Ti.";
            return view;
        }
        for (const char c : sim_arch_env) { wanted = wanted * 10 + (c - '0'); }
    }
    const ArchRung* rung = arch_rung(wanted);
    if (rung == nullptr) {
        view.reason = "sm_" + std::to_string(wanted) +
                      " has no row in kArchLadder, so simulating it would mean inventing a "
                      "capability set. Refusing; add the row with MEASURED evidence first.";
        return view;
    }

    // G4: downgrade only. Equal is refused too -- see the header.
    if (wanted >= physical_sm) {
        view.reason = "the requested simulated rung sm_" + std::to_string(wanted) +
                      " is not STRICTLY LOWER than the physical device sm_" +
                      std::to_string(physical_sm) +
                      ". This override is a downgrade only: simulating the card's own "
                      "capability is a no-op that could hide a real defect behind a "
                      "\"simulated\" label, and raising the reported capability is the "
                      "false-positive bug class this table has already paid for. Refusing.";
        return view;
    }

    view.status       = SimStatus::Active;
    view.effective_sm = wanted;
    view.reason = std::string("sm_") + std::to_string(physical_sm) + " downgraded to sm_" +
                  std::to_string(wanted) + " (" + std::string(rung->label) +
                  ") by the test-only architecture simulator; the route/capability tables "
                  "now answer for sm_" + std::to_string(wanted) + ".";
    return view;
}

// ---------------------------------------------------------------------------
// The production entry point
// ---------------------------------------------------------------------------

// Reads the two environment variables ONCE and prints sim_banner() ONCE per process
// when the override is honoured (G2: loud on every use -- "every use" is every process
// that activates it, plus the SIMULATED marker that travels inside every answer text).
// Refusals are likewise printed once, because a refusal changes what the caller may do.
[[nodiscard]] inline ArchView arch_view_for_device(int physical_sm) noexcept {
    const char* arch = std::getenv(std::string(kSimArchEnv).c_str());
    const char* ack  = std::getenv(std::string(kSimAckEnv).c_str());
    const ArchView view = arch_view_for_device_impl(
        physical_sm, kArchSimBuiltIn, arch == nullptr ? std::string_view{} : arch,
        ack == nullptr ? std::string_view{} : ack);
    static bool announced = false;
    if (!announced && view.status != SimStatus::Disabled) {
        announced = true;
        if (view.simulated()) {
            std::fprintf(stderr, "%s\n", sim_banner(view).c_str());
        } else {
            std::fprintf(stderr,
                         "ninfer: *** SIMULATED ARCHITECTURE REQUEST REFUSED -- TEST "
                         "ONLY ***  %s\n  This process will NOT run on the real device "
                         "either: a refused simulation fails closed.\n",
                         view.reason.c_str());
        }
    }
    return view;
}

// The reason text a caller appends when it has to refuse an artifact or a shape BECAUSE
// the simulation was refused. Kept here so the fail-closed wording is single-sourced.
[[nodiscard]] inline std::string sim_refusal_reason(const ArchView& view) {
    std::string out = "SIMULATED ARCHITECTURE REQUEST REFUSED (fail-closed, test-only): ";
    out += view.reason;
    out += " No answer is given for the real device sm_" + std::to_string(view.physical_sm) +
           " either, because a caller that requested a simulation and got the real "
           "hardware instead would be reading an answer to a question it did not ask.";
    return out;
}

// ---------------------------------------------------------------------------
// THE GATE OVERLOAD G3 PROMISED, which did not exist until now
// ---------------------------------------------------------------------------
//
// G3 above says: "Callers must not proceed on the real path when it is Refused:
// select_route(ArchView) returns NoKernelInTree in that state, and
// require_artifact_formats_supported(ArchView) refuses." MEASURED 2026-09-18 (SIM-RUN): the
// string "ArchView" occurred 0 times in src/core/arch_caps.h and the only
// require_artifact_formats_supported took an `int sm`. So the second half of G3 named an API
// that did not exist, fail-closed was enforced on the ROUTE side only, and the route side had
// no engine consumer at all -- which is why SIM-RUN's own probe, written from this header,
// passed `view.effective_sm` unconditionally and the gate came back ok=true on the two
// REFUSED rungs while the route refused. A caller convention cannot fail closed on behalf of
// an API that is absent.
//
// IT LIVES HERE, NOT IN arch_caps.h, and the reason is the include order: arch_caps.h does
// not know ArchView (arch_sim.h includes arch_caps.h, and kernel_route.h includes both), so
// placing it there would be a cycle. This header is the one that documents the property, so
// this header is where the property has an implementation.
//
// THE TWO BRANCHES, and neither is silent:
//   * !usable()  -> THROW. Refused is not a fallback to the real device: sim_refusal_reason()
//                   already states why in words ("a caller that requested a simulation and
//                   got the real hardware instead would be reading an answer to a question it
//                   did not ask"), and this overload is what makes that statement executable.
//   * usable()   -> evaluate at view.effective_sm. When the override is Disabled this is
//                   view.physical_sm and the verdict is IDENTICAL to the int overload, which
//                   is the property that lets the engine adopt this call without moving any
//                   shipping verdict.
inline void require_artifact_formats_supported(
    const ArchView& view, std::span<const artifact::NumericFormat> formats,
    std::string_view artifact_identity, bool qpn_in_build = kQpnInBuild) {
    if (!view.usable()) {
        throw std::invalid_argument(
            std::string("artifact '") + std::string(artifact_identity) +
            "' was gated against a SIMULATED architecture whose request was REFUSED, so no "
            "capability verdict exists for it: " +
            std::string(sim_refusal_reason(view)) +
            " The real device's capability set was deliberately NOT used, because a caller "
            "that asked for a simulation and silently got the real hardware is reading an "
            "answer to a question it did not ask (src/core/arch_sim.h, G3).");
    }
    require_artifact_formats_supported(view.effective_sm, formats, artifact_identity,
                                      qpn_in_build);
}

} // namespace ninfer::caps
