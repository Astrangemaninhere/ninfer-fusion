#pragma once

// ---------------------------------------------------------------------------
// vendor_sim.h -- the VENDOR-AXIS capability override: the TABLES answer as if the
// vendor were an AMD class. TEST-ONLY, PREPARED, DEFAULT OFF.
//
// ⛔ WHAT THIS FILE TRANSLATES: NOTHING. NOT ONE INSTRUCTION.
// ---------------------------------------------------------------------------
// READ THIS BEFORE ANYTHING BELOW, because a reader who assumes otherwise will read a green
// from this mechanism as a portability claim, and that is the exact class this fleet spent the
// day catching. The sentence is src/core/arch_sim.h:18-24's own, one axis over, and it is
// repeated here rather than paraphrased:
//
//   "A capability OVERRIDE. It changes the compute capability the two *table* layers see ...
//    and nothing else. It does not touch the CUDA context, the driver, cudaGetDeviceProperties,
//    or any device-side __CUDA_ARCH__. The kernels that execute are the ones THIS binary was
//    compiled for; only the ROUTE DECISION is simulated."
//
// For the VENDOR axis the honest form of that sentence is NARROWER, and the narrowing is the
// finding rather than a caveat: there is NO route decision to simulate. src/core/kernel_route.h's
// selector is `select_route(int sm, ...)` / `select_route_on_detail(const ArchRung*, int sm, ...)`
// -- it is keyed on a LADDER RUNG, and a ladder rung is a compute capability, which an AMD gfx
// target does not have. src/core/arch_caps.h:1707-1720 already considered and refused the edit
// that would have created one:
//
//   "A gfx row in that table would make `--sm 906` a legal way to ask for a simulated gfx906 --
//    an implied claim, and one made in files this landing does not own."
//
// ⛔ THAT ROUTE IS NOT TAKEN. This header does NOT touch kernel_route.h, does NOT add a
// kArchLadder row, does NOT add a Cap bit (arch_caps.h:65-73 has 6 instruction-set bits and no
// AMD bit, pinned by tests/test_arch_caps.cpp:843-854), and does NOT widen VirtualProfile
// (src/core/virtual_device.h, whose six profiles are all NVIDIA). It answers exactly ONE
// question, from tables this tree already owns:
//
//     WHICH OF THIS TREE'S SHIPPED KERNELS CAN EXECUTE THIS FORMAT ON THIS VENDOR TARGET.
//
// Nothing is compiled for the target, nothing executes for the target, and no number of any kind
// is produced for the target. The kernels that run under it are this binary's own sm_120 cubins;
// only the TABLE ANSWER is simulated.
//
// ⭐⭐ AND THE ZERO-COVERAGE STATEMENT, WHICH MUST BE READ BEFORE ANY GREEN FROM THIS MECHANISM.
// dl/simvendor measured it and dl/amdladder's six test groups are the evidence: the AMD branch is
// ALREADY reached ON THIS BOX BY PARAMETER -- tests/test_arch_caps.cpp calls
// `amd_rung("gfx906")`, walks `amd_format_verdict(target, format)` over the six literal targets,
// and calls `render_amd_format_refusal("gfx906", NVFP4)` with the target as an ARGUMENT.
// ⇒ THIS OVERRIDE ADDS **ZERO** ROUTE COVERAGE, **ZERO** ADMISSION COVERAGE AND **ZERO** NEW
//   AMD FACT. It cannot: there is no AMD kernel in this tree to route to, so there is no route
//   decision that a different table answer could change. What it adds is coverage of the GUARD
//   MACHINERY ITSELF -- G1-V..G5-V below -- and that is the ONLY thing a green here means.
//   A reader who reads it as "the AMD path is now exercised" has read it wrong.
//
// WHY THE TABLE-LAYER SHAPE IS NOT A CHOICE ON THIS BOX (MEASURED, dl/cufree §2.2.1 / dl/simvendor
// §0.4): a PCI-ID / /dev/nvidia* / /proc/driver/nvidia vendor detector reports "no NVIDIA
// accelerator" on this box -- which HAS an RTX 5090 D and NO 0x10de in /sys/bus/pci/devices; the
// only GPU surface is /dev/dxg plus /usr/lib/wsl/lib/libcuda.so.1. There is no device identity
// here to spoof, so "present as an AMD card" can only mean "the tables answer as if the vendor
// were X". ⇒ DEVICE-IDENTITY SPOOFING IS REFUSED BY NAME (see REFUSALS at the end of this file).
//
// ---------------------------------------------------------------------------
// THE ONE ENTRY POINT, AND WHAT "A PRODUCER" HAD TO MEAN
// ---------------------------------------------------------------------------
// A vendor value enters this mechanism through EXACTLY ONE function:
//
//     VendorView vendor_view_for_device(VendorClass physical) noexcept;
//
// It consumes two things and nothing else:
//   (1) THE VENDOR VALUE: the process environment value NINFER_SIM_VENDOR. Its shape is the
//       vendor ladder's own key, exactly as `sim_rung`'s value is kArchLadder's own key: the
//       ROCm offload-arch target string from kAmdLadder (src/core/arch_caps.h:2012-2064, 6 rows)
//       -- "gfx906", "gfx908", "gfx90a", "gfx942", "gfx1100", "gfx1201". Nothing else is accepted.
//   (2) `physical`, the vendor class of the device this process is really on -- A REQUIRED
//       PARAMETER WITH NO DEFAULT, NEVER A DEVICE QUERY. ⛔ This header contains NO detector and
//       NO CUDA call, for the measured reason above: on this box a detector is INVERTED, so a
//       class read from one would be a FALSE ATTRIBUTION.
//
// ⭐ WHY `physical` BEING UNVERIFIABLE IS NOT A HOLE, AND THE GUARD THAT MAKES IT ONE: it is the
// one input whose provenance this header cannot check, so it is guarded by consequence instead of
// by inspection. See G5-V: a caller that lies about `physical` can only ever obtain a REFUSAL,
// never an ACTIVE view. WHO SUPPLIES IT IS THE WIRING LINE'S DECISION; a reviewer should look
// there first, and G5-V is what bounds the damage if they look and find a lie.
//
// REFUSAL BEHAVIOUR, in one line: EVERY input that cannot be honoured yields SimStatus::Refused
// with a non-empty reason and a named VendorRefusal -- never Disabled, never a smaller-but-working
// simulation. `Disabled` means "nobody asked"; `Refused` means "somebody asked and was told no",
// and the two are never conflated.
//
// ---------------------------------------------------------------------------
// THE FIVE GUARD PROPERTIES, ONE AXIS OVER FROM src/core/arch_sim.h's G1-G4
// ---------------------------------------------------------------------------
// G1-V  IMPOSSIBLE BY ACCIDENT. Four independent things, of which the first is a BUILD fact:
//         (a) the binary was built with -DNINFER_ENABLE_VENDOR_SIM=ON (CMake option, default OFF),
//             which defines NINFER_VENDOR_SIM_ENABLED. ⭐ A SECOND, SEPARATE BUILD FACT, NOT A
//             REUSE OF NINFER_ARCH_SIM_ENABLED: reusing the arch-sim key would mean a binary
//             already audited for the V100 simulator silently acquired a second axis, which is a
//             WIDENING of a landed gate. Two axes, two keys.
//         (b) the environment names a target:  NINFER_SIM_VENDOR=gfx906
//         (c) the environment carries an explicit acknowledgement, and it is a DIFFERENT PHRASE
//             from the arch-sim one, so the V100 phrase cannot satisfy it:
//             NINFER_SIM_VENDOR_ACK=I-UNDERSTAND-THIS-VENDOR-IS-NOT-ON-THIS-BOX
//         (d) the request must not be a NO-OP -- see G4-V.
//       A stray NINFER_SIM_VENDOR in a user's shell therefore does NOT enable this: with (a)
//       absent it is refused by the build fact; with (c) absent it is refused by the
//       acknowledgement; with a value not in the ladder it is refused by the table. NO REFUSAL IS
//       SILENT.
//
// G2-V  LOUD, AND KEYED PER CLASS -- NOT A ONE-SHOT bool.
//       arch_sim.h's own G2 prose says "Every use prints sim_banner() to stderr" while :324 is
//       `static bool announced = false;`, i.e. ONCE PER PROCESS, and tests/test_sim_no_support.cpp
//       :568 pins that one-shot AS THE SHAPE. Its own two sentences disagree and the test sides
//       with `once`; the measured consequence is that a SECOND, DIFFERENT value in one process is
//       announced to nobody. ⇒ THIS header keys the announcement on the vendor class through
//       src/core/announce_once.h -- dl/cufree's `cufree_class_announce_once()` body -- a
//       mutex-guarded set that can only ADD and can never clear, so a SECOND, DIFFERENT VENDOR
//       CLASS IS ANNOUNCED. It also closes the DATA RACE that arch_sim.h:324 has: that `bool` is
//       not synchronized and the function it sits in has two production callers.
//
// G3-V  FAIL CLOSED, WITH A GATE THAT CAN ACTUALLY THROW.
//       The AMD half has NO gate that can fail closed: render_amd_format_refusal() returns a
//       std::string and a string cannot throw, so a caller that forgets to test emptiness gets
//       SILENCE, and silence reads as "fine". The vendor gate below is therefore a THROWING
//       function, mirroring require_artifact_formats_supported(const ArchView&, ...)
//       (src/core/arch_sim.h:380), which is this tree's own precedent for the shape. It lives
//       HERE and not in arch_caps.h for the reason arch_sim.h:366-369 gives for its own placement
//       (arch_caps.h cannot know this type without an include cycle) plus one more: arch_caps.h is
//       the fleet's hottest file -- 2,319 lines, moved several times in one day -- and this
//       landing adds ZERO new anchors to it.
//
// G4-V  NEVER EQUALS, NEVER WIDENS. ⭐ THE VENDOR AXIS IS NOT ORDERED, AND THAT IS THE POINT.
//       G4 one axis over is "the simulated rung must be STRICTLY LOWER than the physical one",
//       which compares two compute capabilities. `Amd` vs `Nvidia` is NOT an ordering, and this
//       header does not pretend otherwise by inventing one. The replacement is a MONOTONICITY
//       property over the lattice of ADMITTED FORMATS:
//
//         NEVER EQUALS: the requested class must differ from `physical`. Asking for the class the
//           card already is is a NO-OP, and a no-op is refused rather than honoured, for G4's own
//           stated reason transferred verbatim: "simulating a no-op would let a genuine defect
//           hide behind a 'simulated' label".
//
//         NEVER WIDENS: `admitted(simulated) SUBSET-OF admitted(physical)`. For this tree the
//           honest value of `admitted(simulated)` for a non-NVIDIA class is the EMPTY SET, because
//           there is no AMD kernel in it at all -- kAmdLadder's own gfx906 row says "The upstream
//           ENGINE has no gfx906 path". ⇒ The gate below refuses EVERY format for every active
//           non-NVIDIA view. A format the physical answer refuses and the simulated answer admits
//           would be a WIDENING; the gate makes that unreachable.
//       ⚠ The alternative replacement -- "a vendor-simulated answer may never be more permissive
//       than the physical card's, i.e. `sim.ok() ⇒ real.ok()` on select_route" -- is expressible
//       ONLY in kernel_route.h, a file arch_caps.h:1716 says the AMD landing does not own. That
//       is the honest price of the OTHER shape, and it is the reason this one states the
//       monotonicity over the GATE's output instead of over the ROUTE's.
//
// G5-V  ⭐ THE UNVERIFIABLE INPUT FAILS CLOSED. `physical` is a parameter this header cannot
//       check, and it is the only such input. It is guarded by CONSEQUENCE, and the guard is a
//       theorem rather than a check:
//
//         IF the caller declares `physical == Amd`, THEN every request whose value is a real
//         kAmdLadder target resolves to `effective == Amd == physical`, so G4-V NEVER EQUALS
//         refuses it. ⇒ There is NO input that yields an ACTIVE view for an AMD target when the
//         caller has declared the box AMD.
//
//       ⇒ A caller that lies about `physical` cannot obtain a single AMD table answer. The lie's
//       only effect is to convert answers into refusals. This is why the seam is not a hole, and
//       tests/test_vendor_sim.cpp V5 asserts it over every kAmdLadder row rather than over one.
//
// AND THE TWO PROHIBITIONS, WHICH ARE DIFFERENT RULES AND NEITHER MAY BE TRADED FOR THE OTHER
// (src/core/arch_sim.h:59-78, src/core/format_probe.h's sim_verdict_refusal):
//   * NO PERFORMANCE NUMBER from a run under this override. Throughput, latency, tokens/s --
//     none, for any vendor, ever. A run under this override executes this binary's own kernels on
//     this box's own card, so every timing it could produce is THIS card's.
//   * NO FUNCTIONAL SUPPORT CLAIM for AMD, Intel, any domestic accelerator or any NPU -- and the
//     vendor axis adds a THIRD, narrower one: NO CLAIM THAT ANY KERNEL WAS TRANSLATED, COMPILED,
//     PORTED OR EXECUTED FOR THE TARGET. This mechanism reads tables. Every vendor-side fact it
//     quotes from kPtxFamilyAmdStatus is labelled EXTERNAL-UNPROBED, and stays so.
//
// ⛔ AND THE ONE MEASUREMENT THAT CHANGES THIS DELIVERABLE'S OWN SHAPE: "INTEL" HAS NOTHING TO
//    ANSWER FROM.
// ---------------------------------------------------------------------------
// MEASURED 2026-09-19 (this line, and independently by the dead line whose draft this is):
// src/core/arch_caps.h has ZERO hits for intel/xpu/pvc/bmxx/level-zero/sycl/oneapi/dpcpp, and the
// whole C++ tree has exactly ONE `intel` hit -- a COMMENT quoting the owner's requirement
// (src/ops/kernel/gqa_attention_simt_ffma.cuh:85) -- while the `xpu` hits are all the substring
// `SharedPrefi[xPub]lication`. ⇒ THERE IS NO INTEL LADDER, NO INTEL RUNG AND NO INTEL TARGET NAME
// IN THIS TREE. So `VendorClass::Intel` is named in the enum and REFUSED by name with its own
// refusal value (`NoTableForVendor`), which is a DIFFERENT sentence from "unknown vendor": Intel
// is a vendor this tree knows of and has no table for, and conflating the two would be the false
// attribution this fleet refuses. An "as-if-Intel" answer is not merely absent -- it has NO SOURCE
// OF TRUTH to be derived from, and inventing one is REFUSED.
//
// ⚠ CORRECTIONS TO THE DRAFT THIS FILE IS DERIVED FROM (both are defects, found by reading it):
//   (1) ⛔ THE DRAFT DID NOT COMPILE. It used `SimStatus` and `kSimAckPhrase` while including only
//       "core/arch_caps.h". MEASURED: `enum class SimStatus` and `inline constexpr
//       std::string_view kSimAckPhrase` are DEFINED ONLY in src/core/arch_sim.h:121 and :119, and
//       arch_caps.h cannot include arch_sim.h (arch_sim.h includes IT -- the cycle arch_sim.h
//       :366-369 names). ⇒ `#include "core/arch_sim.h"` below is mandatory, not tidiness.
//   (2) ⚠ THE DRAFT DEFAULTED `VendorView::physical` AND `::effective` TO `VendorClass::Nvidia`.
//       That is a DEFAULT FOR THE ONE INPUT WHOSE PROVENANCE IS UNVERIFIABLE, i.e. an invented
//       device identity, in the header whose whole subject is that device identity cannot be read
//       here. Both members now default to `VendorClass::Unknown`, so a view that was never
//       assigned reads as UNKNOWN rather than silently as NVIDIA.
// ---------------------------------------------------------------------------

#include "core/arch_caps.h"
#include "core/arch_sim.h" // MANDATORY: SimStatus (:121) and kSimAckPhrase (:119) live here.
#include "core/announce_once.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>

namespace ninfer::caps {

// ---------------------------------------------------------------------------
// (a) THE BUILD FACT. A SECOND key, deliberately not NINFER_ARCH_SIM_ENABLED -- see G1-V.
// ---------------------------------------------------------------------------
#ifdef NINFER_VENDOR_SIM_ENABLED
inline constexpr bool kVendorSimBuiltIn = true;
#else
inline constexpr bool kVendorSimBuiltIn = false;
#endif

// The environment surface, named once so nothing can spell it differently. THREE names, and the
// VALUE name is the vendor ladder's own key -- see "THE ONE ENTRY POINT" above.
inline constexpr std::string_view kVendorSimEnv    = "NINFER_SIM_VENDOR";
inline constexpr std::string_view kVendorSimAckEnv = "NINFER_SIM_VENDOR_ACK";
// Deliberately verbose, and DELIBERATELY A DIFFERENT PHRASE from kSimAckPhrase
// ("I-UNDERSTAND-THIS-IS-NOT-A-V100"): the arch-sim acknowledgement must not be satisfiable as
// this one, because one environment value must not be able to select two axes (G1-V).
inline constexpr std::string_view kVendorSimAckPhrase =
    "I-UNDERSTAND-THIS-VENDOR-IS-NOT-ON-THIS-BOX";

// ---------------------------------------------------------------------------
// THE VENDOR AXIS. Closed, small, and NOT derivable from an sm -- which is the whole reason it is
// a separate axis and not a rung.
// ---------------------------------------------------------------------------
enum class VendorClass : std::uint8_t {
    // NOT A VENDOR. The "the name matched nothing" sentinel, present so that no caller can
    // default a bad name onto a real one -- the same discipline as
    // AmdFormatVerdict::UnknownTarget and VirtualProfile::None. ⚠ IT IS THE DEFAULT OF EVERY
    // MEMBER BELOW, on purpose: see correction (2) in the header.
    Unknown = 0,
    Nvidia  = 1, // what this box is. Asking for it is the NO-OP G4-V refuses.
    Amd     = 2, // has a string-keyed sibling ladder in this tree: kAmdLadder, 6 rows.
    Intel   = 3, // a vendor this tree NAMES and has NO TABLE FOR. Measured; refused by name.
};

[[nodiscard]] inline std::string_view vendor_class_name(VendorClass cls) noexcept {
    switch (cls) {
    case VendorClass::Unknown: return "unknown-vendor-class";
    case VendorClass::Nvidia: return "nvidia";
    case VendorClass::Amd: return "amd";
    case VendorClass::Intel: return "intel";
    }
    return "unknown-vendor-class";
}

// The class a NAME denotes. Only the three real vendors have names; everything else is Unknown
// and is refused as an unknown vendor rather than defaulted.
[[nodiscard]] inline VendorClass vendor_class_from_name(std::string_view name) noexcept {
    if (name == "nvidia") { return VendorClass::Nvidia; }
    if (name == "amd") { return VendorClass::Amd; }
    if (name == "intel") { return VendorClass::Intel; }
    return VendorClass::Unknown;
}

// TRUE ONLY FOR A CLASS THIS TREE HOLDS A LADDER FOR. MEASURED: kAmdLadder
// (src/core/arch_caps.h:2012-2064, kAmdLadderSize at :2064) is the ONLY vendor ladder in the tree.
// Single-sourced here so a second vendor table cannot be added without this predicate changing
// with it -- the discipline that keeps "two tables answering one question" from drifting.
[[nodiscard]] inline bool vendor_has_ladder(VendorClass cls) noexcept {
    return cls == VendorClass::Amd;
}

// ---------------------------------------------------------------------------
// THE REFUSAL, AS A CLOSED ENUM. A value a caller MUST branch on, not a string it can ignore --
// the same reason src/core/virtual_device.h's VirtualRefusal is an enum rather than prose, and the
// same reason arch_caps.h keeps AmdFormatVerdict an enum with prose beside it.
//
// ⚠ AND IT IS A CLOSED KEYSPACE, which is what lets it be the second key of the announce set
// (G2-V): a refusal that is a STRING could not be announced per-value without becoming unbounded.
// ---------------------------------------------------------------------------
enum class VendorRefusal : std::uint8_t {
    NotRequested = 0,  // nothing was asked. NOT a refusal: the guard is off, zero behaviour change
    NotBuiltIn,        // asked, and this binary lacks the build key
    MissingAck,        // asked, and the acknowledgement is absent or wrong
    NotATarget,        // the value is a class NAME where a target belongs ("amd", not "gfx906")
    NotInVendorLadder, // the vendor HAS a ladder here and the value is not a row of it
    NoTableForVendor,  // a vendor this tree NAMES and has NO TABLE for (Intel -- measured, 0 rows)
    UnknownVendor,     // the value names nothing this tree knows: a typo, or no vendor name here
    NotADowngrade,     // the request is a NO-OP: the effective class equals the physical one (G4-V)
};

[[nodiscard]] inline std::string_view vendor_refusal_name(VendorRefusal refusal) noexcept {
    switch (refusal) {
    case VendorRefusal::NotRequested: return "not-requested";
    case VendorRefusal::NotBuiltIn: return "not-built-in";
    case VendorRefusal::MissingAck: return "missing-acknowledgement";
    case VendorRefusal::NotATarget: return "vendor-class-where-a-target-belongs";
    case VendorRefusal::NotInVendorLadder: return "not-in-vendor-ladder";
    case VendorRefusal::NoTableForVendor: return "no-table-for-vendor";
    case VendorRefusal::UnknownVendor: return "unknown-vendor";
    case VendorRefusal::NotADowngrade: return "no-op-not-a-downgrade";
    }
    return "unknown-refusal";
}

// Every refusal value, as a walked range, so a test can assert over the whole keyspace instead of
// over the values it happened to think of. `Count` is the enum's own count (the discipline
// arch_caps.h:605-609 states: "the enum carries its own count as its last member").
inline constexpr std::size_t kVendorRefusalCount = 8;
static_assert(kVendorRefusalCount == static_cast<std::size_t>(VendorRefusal::NotADowngrade) + 1,
              "kVendorRefusalCount must be the enum's own range: a refusal value appended without "
              "updating the count would be invisible to a range walk, and a test that walks a range "
              "is only as complete as the range.");

// ---------------------------------------------------------------------------
// THE VIEW. ⭐ IT HAS NO COMPUTE-CAPABILITY FIELD, AND THAT IS A PROPERTY, NOT AN OMISSION.
// G4-V: a number is what a caller raises, so this struct carries none -- there is no value to
// raise. The simulated rung's analogue (ArchView::effective_sm) is deliberately ABSENT here rather
// than zeroed, because a zero is a number and a caller can compare it.
// ---------------------------------------------------------------------------
struct VendorView {
    // WHAT THE DEVICE REALLY IS. A REQUIRED PARAMETER of the entry point, never a device query --
    // see the measured reason in "THE ONE ENTRY POINT" above. Recorded ALWAYS, on every path, so
    // that no message can present the simulated class as the card's class -- the job
    // ArchView::physical_sm does one axis over (src/core/arch_sim.h:138-139).
    // ⚠ DEFAULTS TO Unknown, NOT TO Nvidia: see correction (2).
    VendorClass physical = VendorClass::Unknown;

    // WHAT THE TABLES MUST ANSWER FOR. Equals `physical` whenever the override is not Active.
    VendorClass effective = VendorClass::Unknown;

    // The vendor ladder key this view is keyed on. EMPTY for every class that is not AMD and for
    // every refusal: a target string is what an AMD class has instead of an sm, and this tree has
    // none for any other class.
    std::string target = {};

    SimStatus status = SimStatus::Disabled; // REUSED, not re-invented: two vocabularies for one
                                            // fact is how a tree starts disagreeing with itself.
    VendorRefusal refusal = VendorRefusal::NotRequested;
    std::string reason = {}; // NON-EMPTY whenever status == Refused. Never empty on a refusal.

    [[nodiscard]] bool simulated() const noexcept { return status == SimStatus::Active; }
    [[nodiscard]] bool failed() const noexcept { return status == SimStatus::Refused; }
    // True when a caller may use this view. False ONLY for Refused, which is the fail-closed state,
    // and it is false for the SAME reason ArchView::usable() is: a caller that asked for a
    // simulation and silently got the real card is reading an answer to a question it did not ask.
    [[nodiscard]] bool usable() const noexcept { return status != SimStatus::Refused; }
};

// ---------------------------------------------------------------------------
// THE PURE DECISION. Every input is a parameter -- the build fact, the two environment values, the
// physical class -- so ONE test binary exercises the enabled and the not-enabled worlds, the
// honoured and the refused requests, and every refusal value, without a rebuild and without a
// second process. Same idiom as arch_view_for_device_impl (src/core/arch_sim.h:215) and
// validate_virtual_request (src/core/virtual_device.h): the reason is the same in all three places
// -- a guard whose whole job is to be trustworthy has to be checkable on a host with no hardware.
//
// `vendor_env` / `ack_env` are the RAW environment values: an empty string_view means "unset",
// which is reproducible from a test. The production wrapper is the only place std::getenv runs.
// ---------------------------------------------------------------------------
[[nodiscard]] inline VendorView vendor_view_for_device_impl(VendorClass physical,
                                                           bool sim_built_in,
                                                           std::string_view vendor_env,
                                                           std::string_view ack_env) {
    VendorView view;
    view.physical  = physical;
    view.effective = physical;

    // No request at all: the default, and the ONLY path a shipping build takes.
    if (vendor_env.empty()) {
        view.status  = SimStatus::Disabled;
        view.refusal = VendorRefusal::NotRequested;
        return view;
    }

    // A request exists. From here on a refusal is RECORDED, never ignored, so that "requested but
    // not honoured" can never look like "not requested".
    view.status = SimStatus::Refused;

    // (a) the build fact.
    if (!sim_built_in) {
        view.refusal = VendorRefusal::NotBuiltIn;
        view.reason  = std::string("a simulated VENDOR was requested (") + std::string(kVendorSimEnv) +
                      "=" + std::string(vendor_env) +
                      ") but this binary was NOT built with -DNINFER_ENABLE_VENDOR_SIM=ON, so the "
                      "vendor override does not exist in it. Refusing rather than proceeding: a "
                      "request that cannot be honoured must not silently run on the real device's "
                      "own vendor. TWO WAYS TO BUILD A BINARY THAT CAN HONOUR IT, AND THEY "
                      "REACH DIFFERENT READERS, so BOTH are named here. (1) IF YOU BUILD WITH "
                      "CMAKE: reconfigure with -DNINFER_ENABLE_VENDOR_SIM=ON -- that is the "
                      "CMake option, and reconfiguring is the only way it can reach you, "
                      "because the option only exists inside CMake. (2) IF YOU COMPILE THIS "
                      "TEST BY HAND: no CMake option is in your path at all, so add the "
                      "preprocessor key -DNINFER_VENDOR_SIM_ENABLED=1 to your own compile "
                      "line -- defining that key is the one thing the CMake option does, so "
                      "naming the option alone would send you to a tool you are not running. "
                      "Use the route that matches how you build; either one reaches a "
                      "vendor-axis run.";
        return view;
    }

    // (c) the acknowledgement. Checked BEFORE the value so that a missing ack and a bad value
    // cannot be confused for one another in the message -- arch_sim.h:246-247's order, kept.
    if (ack_env != kVendorSimAckPhrase) {
        view.refusal = VendorRefusal::MissingAck;
        view.reason  = std::string(kVendorSimEnv) + "=" + std::string(vendor_env) +
                      " was requested but " + std::string(kVendorSimAckEnv) + " is ";
        if (ack_env.empty()) {
            view.reason += "unset";
        } else {
            view.reason += "\"" + std::string(ack_env) + "\"";
        }
        view.reason += ", not the required acknowledgement \"" + std::string(kVendorSimAckPhrase) +
                      "\". The vendor override is test-only and every use is loud; it cannot be "
                      "enabled by a single stray environment variable. NOTE the arch-sim "
                      "acknowledgement (\"" + std::string(kSimAckPhrase) +
                      "\") does NOT satisfy this one: one environment value must not be able to "
                      "select two axes. Refusing.";
        return view;
    }

    // (b) the value. THE THREE REFUSAL SHAPES ARE KEPT DISTINCT because they are three different
    // sentences about the tree. A single "bad vendor" clause over all three would be the false
    // attribution this fleet refuses (dl/cufree's own rule: "one clause over two situations is a
    // false attribution").
    const VendorClass named = vendor_class_from_name(vendor_env);
    const bool looks_like_a_target =
        vendor_env.size() >= 3 && vendor_env.substr(0, 3) == "gfx";

    // WHAT WAS ASKED FOR, HELD AS LOCALS UNTIL THE VIEW IS KNOWN TO BE ACTIVE. The requested class
    // and the requested target are NOT written to `view` here, and that is the fix for the G5-V
    // hole: this function used to record the target BEFORE the no-op check below, so a REFUSED view
    // carried a target string, and `VendorView::target`'s own contract (see the member comment:
    // "EMPTY for every class that is not AMD and for every refusal") was broken on exactly one
    // path. The consequence was measured, not argued: vendor_format_answer() asks for a table
    // query whenever the target is non-empty, so a caller who lied about `physical` obtained all
    // twelve AMD verdicts through the refusal. With the recording at the commit point instead,
    // `target` is non-empty if and only if the view is ACTIVE.
    VendorClass requested_class  = physical;
    std::string requested_target = {};

    if (looks_like_a_target) {
        // A gfx-prefixed token is an AMD offload-arch target by construction.
        const AmdRung* rung = amd_rung(vendor_env);
        if (rung == nullptr) {
            // ⭐ THIS IS `sim_rung`'s PARSE-TIME REFUSAL, ONE AXIS OVER, AND IT IS THE REASON THE
            //    VALUE IS CHECKABLE AT ALL: the ladder is the closed set, so a target that is not a
            //    row of it cannot be recorded anywhere downstream -- it is refused here and has no
            //    representation to be recorded INTO.
            view.refusal = VendorRefusal::NotInVendorLadder;
            view.reason  = std::string(kVendorSimEnv) + "=\"" + std::string(vendor_env) +
                          "\" is shaped like an AMD offload-arch target but has no row in "
                          "kAmdLadder (src/core/arch_caps.h:2012-2064), so simulating it would mean "
                          "inventing a target and a capability set for it. The rows this tree "
                          "contains are gfx906 gfx908 gfx90a gfx942 gfx1100 gfx1201. Refusing; add "
                          "the row with MEASURED evidence first -- the same rule src/core/arch_sim.h"
                          ":282-286 applies to a rung with no kArchLadder row, and "
                          "src/core/format_probe.h:894-915 applies to a sim_rung value, whose "
                          "stated ground is that a simulated verdict must be RE-PROBEABLE on real "
                          "hardware and a rung this table does not contain cannot be.";
            return view;
        }
        requested_class  = VendorClass::Amd;
        requested_target = std::string(rung->target);
    } else if (named == VendorClass::Intel) {
        // ⭐ THE MEASURED REFUSAL. Intel is a vendor this tree NAMES and has NO TABLE for: ZERO
        // hits for intel/xpu/pvc/bmxx/level-zero/sycl/oneapi/dpcpp in arch_caps.h, and the whole
        // C++ tree's only `intel` is a comment. So there is nothing to be "as-if-Intel" ABOUT.
        // This is a DIFFERENT sentence from "unknown vendor" on purpose.
        view.refusal = VendorRefusal::NoTableForVendor;
        view.reason  = std::string(kVendorSimEnv) + "=\"" + std::string(vendor_env) +
                      "\" names a vendor this tree knows of and holds NO TABLE for. MEASURED "
                      "2026-09-19: src/core/arch_caps.h has zero hits for intel/xpu/pvc/bmxx/"
                      "level-zero/sycl/oneapi/dpcpp, and the only `intel` in the whole C++ tree is "
                      "a comment quoting the requirement "
                      "(src/ops/kernel/gqa_attention_simt_ffma.cuh:85). There is no Intel ladder, "
                      "no Intel rung, no Intel target name and no Intel kernel, so an as-if-Intel "
                      "answer has NO SOURCE OF TRUTH to be derived from. This is not 'no Intel "
                      "card is present' -- it is 'this tree holds no Intel table', and inventing "
                      "one is refused rather than approximated from a neighbouring vendor's table.";
        return view;
    } else if (named != VendorClass::Unknown) {
        // The value is a class NAME. The axis is keyed on a TARGET, so this is the wrong KIND of
        // token -- refused as such rather than resolved into a target nobody typed.
        view.refusal = VendorRefusal::NotATarget;
        view.reason  = std::string(kVendorSimEnv) + "=\"" + std::string(vendor_env) +
                      "\" is a vendor CLASS name, but this override is keyed on a TARGET: the "
                      "vendor's own name for a specific part, which is the same shape "
                      "NINFER_SIM_ARCH takes from kArchLadder. Give a target. For an AMD class the "
                      "targets this tree contains are the six rows of kAmdLadder "
                      "(src/core/arch_caps.h:2012-2064): gfx906 gfx908 gfx90a gfx942 gfx1100 "
                      "gfx1201. Refusing rather than guessing which part you meant.";
        return view;
    } else {
        view.refusal = VendorRefusal::UnknownVendor;
        view.reason  = std::string(kVendorSimEnv) + "=\"" + std::string(vendor_env) +
                      "\" names no vendor this tree knows and is not shaped like a target. Known "
                      "class names are nvidia amd intel (none of which is a target), and AMD "
                      "targets are the gfx-prefixed rows of kAmdLadder. Refusing rather than "
                      "defaulting, because the capability set this value implies is what the tables "
                      "will be asked about.";
        return view;
    }

    // (d) G4-V NEVER EQUALS: the no-op. ⭐ AND THIS IS ALSO G5-V'S GUARD: a caller that declared
    // `physical == Amd` reaches this line for EVERY real kAmdLadder target, so a lie about the one
    // unverifiable input yields a refusal and nothing else.
    if (requested_class == physical) {
        view.refusal = VendorRefusal::NotADowngrade;
        view.reason  = "the requested vendor class '" + std::string(vendor_class_name(requested_class)) +
                      "' is the class this device already is, so honouring it would be a NO-OP. G4 "
                      "one axis over refuses the equal case for the same reason and the reason "
                      "transfers unchanged: a no-op that wears a \"simulated\" label can hide a "
                      "genuine defect of the real card behind it, because every answer it produces "
                      "is the real card's answer while every message says it was simulated. ";
        // THE REQUESTED TARGET IS NAMED IN THE MESSAGE AND NOWHERE ELSE. It is not recorded on the
        // view (G5-V), so the refusal message is the only place it can appear -- a refusal that
        // does not say what was asked for is not actionable.
        view.reason += requested_target.empty()
                           ? std::string("Refusing.")
                           : "The target asked for was '" + requested_target +
                                 "'; it is named here and deliberately NOT recorded on this refused "
                                 "view, so no consumer of a refused view can read a target off it "
                                 "(G5-V). Refusing.";
        return view;
    }

    // THE ONE SITE THAT RECORDS A CLASS AND A TARGET ON A VIEW, and it is below every refusal
    // return above, so both fields are written if and only if the request was honoured.
    view.effective = requested_class;
    view.target    = requested_target;
    view.status = SimStatus::Active;
    view.reason = "the tables now answer for vendor class '" +
                  std::string(vendor_class_name(view.effective)) + "' target '" + view.target +
                  "', requested by " + std::string(kVendorSimEnv) + " and acknowledged by " +
                  std::string(kVendorSimAckEnv) + "; this device is really '" +
                  std::string(vendor_class_name(physical)) +
                  "'. NOTHING is translated, compiled or executed for the target: the kernels that "
                  "run are this binary's own.";
    return view;
}

// ---------------------------------------------------------------------------
// FORWARD DECLARATIONS. These two are used by vendor_view_for_device() below and DEFINED after it
// (the banner belongs next to the other text-builders, and the text-builders need VendorView to be
// complete). Without these two lines the header does NOT compile -- name lookup at the point of
// use does not see a later definition, and this is the class of defect a compiler finds and a
// reader does not. They are declared here rather than reordering the file so that the reading
// order stays "view -> entry point -> text", which is the order a reviewer wants.
// ---------------------------------------------------------------------------
[[nodiscard]] inline std::string vendor_banner(const VendorView& view);
[[nodiscard]] inline std::string vendor_refusal_reason(const VendorView& view);

// ---------------------------------------------------------------------------
// THE ONE ENTRY POINT. THE VENDOR VALUE ENTERS HERE AND NOWHERE ELSE.
// ---------------------------------------------------------------------------
// ⚠ `physical` HAS NO DEFAULT, ON PURPOSE. A default would be an invented device identity in the
// one header that exists because device identity cannot be read here. A caller must write it.
//
// The announcement is keyed PER VENDOR CLASS through src/core/announce_once.h -- dl/cufree's
// `cufree_class_announce_once()` body -- so a SECOND, DIFFERENT class in one process IS announced.
// G2-V, and the reason this is not arch_sim.h:324's `static bool`.
//
// ⚠ THE LATCH IS ON THE MESSAGE, NOT ON THE READ. The environment is read on EVERY call, so a
// caller that changes it between calls gets the new answer (and a second announcement); the MESSAGE
// is suppressed only for a class already announced. That is the honest reading of "loud per use",
// and it is what makes the second-class assertion testable in ONE process.
//
// TWO KEYSPACES, ONE PRIMITIVE: the class keys the banner, and the REFUSAL VALUE keys the refusal
// line, so a second, DIFFERENT refusal reason in one process is not silent either.
[[nodiscard]] inline VendorView vendor_view_for_device(VendorClass physical) noexcept {
    const char* vendor = std::getenv(std::string(kVendorSimEnv).c_str());
    const char* ack    = std::getenv(std::string(kVendorSimAckEnv).c_str());
    const VendorView view = vendor_view_for_device_impl(
        physical, kVendorSimBuiltIn, vendor == nullptr ? std::string_view{} : vendor,
        ack == nullptr ? std::string_view{} : ack);

    if (view.status == SimStatus::Disabled) { return view; }

    if (view.simulated()) {
        if (ninfer::detail::announce_once_keyed(view.effective)) {
            std::fprintf(stderr, "%s\n", vendor_banner(view).c_str());
        }
    } else if (ninfer::detail::announce_once_keyed(view.refusal)) {
        std::fprintf(stderr,
                     "ninfer: *** SIMULATED VENDOR REQUEST REFUSED [%s] -- TEST ONLY ***  %s\n"
                     "  This process will NOT answer for the real card's vendor either: a refused "
                     "simulation fails closed.\n",
                     std::string(vendor_refusal_name(view.refusal)).c_str(), view.reason.c_str());
    }
    return view;
}

// The loud text. Single-sourced here so the wording cannot drift between entry points --
// arch_sim.h:41-42 states that property for the arch axis and this is its counterpart.
[[nodiscard]] inline std::string vendor_banner(const VendorView& view) {
    if (!view.simulated()) { return {}; }
    std::string out = "ninfer: *** SIMULATED VENDOR -- TEST ONLY ***  ";
    out += "this device is really vendor class '";
    out += vendor_class_name(view.physical);
    out += "' and is being reported to the capability and route TABLES as '";
    out += vendor_class_name(view.effective);
    out += "' target '";
    out += view.target;
    out += "', because ";
    out.append(kVendorSimEnv);
    out += "=";
    out.append(view.target);
    out += " and ";
    out.append(kVendorSimAckEnv);
    out += " were both set and this binary was built with NINFER_ENABLE_VENDOR_SIM. ";
    // THE FOUR DENIALS, each naming a different reader's assumption.
    out += "NOTHING is translated: no instruction of this tree's was rewritten, re-encoded or "
           "recompiled for the target, and no toolchain for it exists on this box. NOTHING is "
           "executed for the target: the kernels that will run are the ones THIS binary was "
           "compiled for, on this box's own card, and no AMD kernel is compiled into it at all. "
           "The only question this override changes the answer to is which of THIS TREE'S SHIPPED "
           "KERNELS can execute a given weight format on the target, which it answers from the "
           "tables kAmdLadder and kFormatRequirements already hold -- and for every non-NVIDIA "
           "class in this tree that answer is REFUSED, never admitted. A run under it proves "
           "NOTHING about AMD or Intel silicon of any kind, and NO SUPPORT CLAIM and NO PERFORMANCE "
           "NUMBER may be sourced from it -- not a throughput, not a latency, not a "
           "tokens-per-second figure, and not a support verdict for any format on any vendor. ";
    out.append(view.reason);
    return out;
}

// The reason text a caller appends when it has to refuse an artifact or a shape BECAUSE the vendor
// simulation was refused. Kept here so the fail-closed wording is single-sourced, exactly as
// sim_refusal_reason() is for the arch axis (src/core/arch_sim.h:342).
[[nodiscard]] inline std::string vendor_refusal_reason(const VendorView& view) {
    std::string out = "SIMULATED VENDOR REQUEST REFUSED [";
    out += vendor_refusal_name(view.refusal);
    out += "] (fail-closed, test-only): ";
    out += view.reason;
    out += " No answer is given for the real card's own vendor either ('";
    out += vendor_class_name(view.physical);
    out += "'), because a caller that requested a vendor simulation and got the real card instead "
           "would be reading an answer to a question it did not ask.";
    return out;
}

// ---------------------------------------------------------------------------
// THE TABLE ANSWER UNDER A VENDOR VIEW -- "answer as if the vendor were an AMD class"
// ---------------------------------------------------------------------------
// The question, exactly: which of this tree's SHIPPED KERNELS can execute `format` on the view's
// target. The answer comes from `amd_format_verdict(target, format)` (src/core/arch_caps.h:2222),
// whose derivation is measured and NARROW: it reads kAmdLadder for the target and
// kFormatRequirements for the format's `required` Cap bit, and NOTHING ELSE -- not kPtxSites, not
// kPtxFamilyAmdStatus, not kAmdFormatBlockers, not kAmdLdsBlockerSites. That is stated here because
// a reader will otherwise assume the EXTERNAL-UNPROBED apparatus feeds this answer: it does not, it
// feeds the refusal PROSE only.
struct VendorFormatAnswer {
    bool answered = false; // false when no vendor query was made (the view is Disabled, or refused)
    AmdFormatVerdict verdict = AmdFormatVerdict::UnknownTarget;
    // TRUE ONLY WHEN THE TREE'S OWN ANSWER IS AN AFFIRMATIVE ONE. For every non-NVIDIA class in this
    // tree it is FALSE for every format -- see G4-V NEVER WIDENS, and see the gate below, which is
    // the only place that may act on it.
    bool admitted = false;
    // The operator-facing refusal for this (target, format). NON-EMPTY whenever `answered` is
    // true: where the arch table's own render is silent, it is FILLED by
    // render_vendor_verdict_text() below, so that an emptiness test can never read as "fine".
    // EMPTY only when no query was made at all (the view is Disabled, or refused).
    std::string text = {};
    bool simulated = false;  // true when this answer came from a simulated vendor view
    std::string target = {}; // the target answered for; empty when none
};

// ---------------------------------------------------------------------------
// GAP 3 MADE AUDIBLE: THE LOUD FORM OF A VERDICT THE TABLE'S STRING FORM LEAVES SILENT.
// ---------------------------------------------------------------------------
// `render_amd_format_refusal()` (src/core/arch_caps.h:2240) is EMPTY for every verdict other than
// NoKernelInTree, and that emptiness is DELIBERATE in that file and documented in its own words:
// "Empty for every other verdict, so a caller may print it unconditionally in the same shape
// render_fallback_notice() uses." MEASURED here, 2026-09-19: for the target "gfx906" the render is
// EMPTY for exactly the 4 of 12 formats whose `required` Cap bit is Cap::None -- FP32, I32, I64,
// U4Z8G16_F16S -- and the verdict for those four is NotATensorCoreOperand.
//
// WHY THE SILENCE IS NOT ACCEPTABLE AT THIS LAYER: a refusal that is not loud is not a refusal. A
// caller that tests `text.empty()` -- which is what the arch_caps.h comment invites -- receives
// NOTHING for those four formats, and NOTHING reads as "fine", while this tree's actual position is
// that NO format is admissible on the target at all. The verdict word was never an admission in
// the first place: NotATensorCoreOperand is defined as "the absence of the one specific blocker
// this table knows how to name" (src/core/arch_caps.h:2131-2135).
//
// => This fills the silence at the surfaces THIS header owns, and it is ONE sentence used by both
// of them (the answer struct below and the gate below it), so the two cannot drift. It is not a
// second verdict: it names the verdict word the tree gave and says in the same breath that the word
// is not an admission. It is NEVER called for NoKernelInTree, where the table's own text is
// strictly more specific than anything written here.
[[nodiscard]] inline std::string render_vendor_verdict_text(AmdFormatVerdict verdict) {
    return std::string("ninfer: this tree's AMD table did not refuse this format BY NAME -- it "
                       "answered '") + std::string(amd_format_verdict_name(verdict)) +
           "', and that word is NOT an admission and NOT the absence of a problem: it means only "
           "that the one specific blocker this table knows how to name does not apply to this "
           "format (src/core/arch_caps.h:2131-2135). This tree holds NO AMD kernel for the target "
           "at all, so no format is admissible on it, and the vendor override cannot make one "
           "admissible by answering a table differently. An EMPTY string here would read as "
           "\"fine\".\n";
}

[[nodiscard]] inline VendorFormatAnswer vendor_format_answer(const VendorView& view,
                                                            artifact::NumericFormat format) {
    VendorFormatAnswer answer;
    if (view.status == SimStatus::Disabled) { return answer; }
    // ONLY AN ACTIVE VIEW HAS A TABLE ANSWER, AND THE QUESTION IS ASKED OF THE STATUS -- NOT OF
    // `target`. The guard used to be `view.target.empty()`, i.e. a PROXY for "is this view active",
    // and the proxy was what the G5-V hole was made of: the refused-view recording above made it
    // read true, and a caller who lied about `physical` obtained all twelve AMD verdicts. A proxy
    // can be broken by a distant edit; a status cannot. The caller's move for a refused view is the
    // gate below, which refuses; this answer is deliberately NOT "not supported", because "not
    // supported" is a verdict and no verdict was reached.
    if (!view.simulated()) { return answer; }
    answer.simulated = true;
    answer.target    = view.target;
    if (view.target.empty()) {
        // Backstop, unreachable today by construction: an ACTIVE view is created at one site and
        // that site writes a non-empty kAmdLadder key. Kept so the gate's `!answered` branch keeps
        // a real meaning. NOT claimed as covered by a test.
        return answer;
    }
    answer.answered = true;
    answer.verdict  = amd_format_verdict(view.target, format);
    answer.text     = render_amd_format_refusal(view.target, format);
    if (answer.text.empty()) {
        // GAP 3, CLOSED AT THIS SURFACE. See render_vendor_verdict_text() above for why the
        // silence is not allowed through.
        answer.text = render_vendor_verdict_text(answer.verdict);
    }
    // ⭐ THE ONE PLACE `admitted` IS DECIDED, AND THE HONEST VALUE FOR THIS TREE IS `false` FOR
    // EVERY FORMAT. There is no AMD kernel in this tree -- kAmdLadder's gfx906 row says it in the
    // tree's own words ("The upstream ENGINE has no gfx906 path") -- so no format is admissible on
    // the target, whatever the verdict word is. Setting this from the verdict would make
    // `NotATensorCoreOperand` ("the absence of the one specific blocker", NOT a claim that the
    // format works) an admission, which arch_caps.h's own comment forbids and which would be the
    // G4-V WIDENING this header exists to make unreachable.
    // ⚠ THIS IS `false` AND IT IS NOT DERIVED FROM `answer.verdict`. Mutation M2 sets it from the
    // verdict and dies on tests/test_vendor_sim.cpp V4.
    answer.admitted = false;
    return answer;
}

// ---------------------------------------------------------------------------
// THE GATE -- THE AMD SIDE'S FAIL-CLOSED FORM, WHICH DID NOT EXIST
// ---------------------------------------------------------------------------
// WHY THIS EXISTS AT ALL. The AMD half has no gate that can fail closed.
// `render_amd_format_refusal()` returns a std::string and is EMPTY for every verdict other than
// NoKernelInTree, so a caller that forgets to test emptiness receives SILENCE -- and silence reads
// as "fine". `amd_format_verdict()` does better (it returns a closed enum, and UnknownTarget is a
// genuine fail-closed value), but its middle value, NotATensorCoreOperand, is explicitly "the
// absence of the one specific blocker this table knows how to name" and NOT an admission, with
// nothing in the type system stopping a caller from treating it as one. A string cannot throw and
// an enum a caller may misread.
//
// ⇒ THE FAIL-CLOSED FORM IS THIS: a THROWING function, mirroring
// require_artifact_formats_supported(const ArchView&, ...) (src/core/arch_sim.h:380), which is this
// tree's own precedent for the shape and states the reason it lives where it lives (arch_caps.h
// cannot know the view type without an include cycle). It lives HERE for that same reason, and adds
// ZERO new anchors to arch_caps.h.
//
// ⭐ ITS ONLY NON-THROWING PATHS ARE "NOBODY ASKED" AND "THE REQUEST WAS REFUSED". For every ACTIVE
// non-NVIDIA view it REFUSES EVERY FORMAT, and the refusal names WHICH of the three verdict words
// the tree gave, so the caller learns the specific blocker rather than a generic "unsupported". That
// is not over-refusal: it is G4-V NEVER WIDENS made executable. The set of formats this tree admits
// on an AMD target is EMPTY, and a gate whose job is to answer "may I proceed" must say so.
//
// ⚠ WHERE THIS BELONGS. It is prepared HERE because this header owns the vendor view type and
// because arch_caps.h is the fleet's hottest file and this landing adds no anchor to it. MOVING THE
// GATE NEXT TO amd_format_verdict() IS THE AMD SECTION'S OWNING LINE'S CALL, not this line's: the
// shape to move is a six-line function taking (std::string_view target, std::span<const
// NumericFormat>, identity) with the same throw. That decision is theirs rather than pre-empted.
inline void require_vendor_formats_supported(const VendorView& view,
                                            std::span<const artifact::NumericFormat> formats,
                                            std::string_view artifact_identity) {
    if (!view.usable()) {
        throw std::invalid_argument(
            std::string("artifact '") + std::string(artifact_identity) +
            "' was gated against a SIMULATED VENDOR whose request was REFUSED, so no vendor answer "
            "exists for it: " +
            vendor_refusal_reason(view) +
            " The real card's own vendor was deliberately NOT used, because a caller that asked for "
            "a vendor simulation and silently got the real card is reading an answer to a question "
            "it did not ask (src/core/vendor_sim.h, G3-V).");
    }
    if (!view.simulated()) {
        // Disabled: this gate has nothing to say, and it must not become a second, silent
        // capability gate over the real card -- the real path's verdict is decided by
        // require_artifact_formats_supported(int sm, ...) and must not be duplicated here.
        return;
    }
    for (const artifact::NumericFormat format : formats) {
        const VendorFormatAnswer answer = vendor_format_answer(view, format);
        if (!answer.answered) {
            throw std::invalid_argument(
                std::string("artifact '") + std::string(artifact_identity) +
                "' was gated under a SIMULATED VENDOR view that has no target, so no verdict was "
                "reached for it: " +
                vendor_refusal_reason(view));
        }
        if (answer.admitted) { continue; } // unreachable today by construction; NOT claimed as
                                           // covered by a test, and this comment exists so its
                                           // presence cannot be read as coverage.
        std::string why = "artifact '" + std::string(artifact_identity) +
                          "': REFUSED under a SIMULATED VENDOR view. Target '" + view.target +
                          "' format " + std::string(artifact::format_name(format)) + " answers '" +
                          std::string(amd_format_verdict_name(answer.verdict)) + "'. ";
        // `answer.text` is non-empty by contract whenever `answered` is true (it is filled by
        // render_vendor_verdict_text() when the arch table's render is silent), so the sentence
        // below is single-sourced in ONE place instead of being spelled twice.
        why += answer.text;
        why += "This refusal is a FAIL-CLOSED GATE and not a support verdict: it says which kernel "
               "this tree lacks, and it is NOT a claim about any AMD card. The support of every "
               "format on this target is UNMEASURED and its arithmetic is UNVERIFIED. NOTHING was "
               "translated or executed for the target (src/core/vendor_sim.h).";
        throw std::invalid_argument(why);
    }
}

// ---------------------------------------------------------------------------
// REFUSALS BY NAME -- what this header does NOT do, listed where a reader of the CODE will find
// them and not only where a reader of the report will.
// ---------------------------------------------------------------------------
// ⛔ REFUSED: device-identity or vendor-identity spoofing, and any vendor detector. MEASURED: the
//    PCI-ID surface on this box is INVERTED (an RTX 5090 D present, no 0x10de in
//    /sys/bus/pci/devices), so a detector-read class would be a FALSE ATTRIBUTION, which is worse
//    than no class at all. That is why `physical` is a parameter.
// ⛔ REFUSED: a gfx row in kArchLadder, a Cap bit for AMD, an sm for a vendor. Each of the three
//    would convert a refusal into an implied claim, and two of the three are refused in writing at
//    src/core/arch_caps.h:1707-1720.
// ⛔ REFUSED: any translation, port, hipify, codegen or compile for the target. This header reads
//    tables. There is no instrument on this box that could move an EXTERNAL-UNPROBED row (no
//    amdgcn compiler, no AMD silicon), and HIP_PLATFORM=nvcc would compile the PTX verbatim and
//    confirm nothing -- it would be evidence for the wrong proposition.
// ⛔ REFUSED: any support claim for AMD, Intel, any domestic accelerator or any NPU, and any
//    performance number of any kind, from any run under this override.
// ⛔ REFUSED: a `sim_vendor=` field in src/core/format_probe.h's row schema. It is the recordability
//    path, and the honest disposition is NOT to add the field and then guard it: it is that a field
//    whose every value must be refused has no reason to exist. tests/test_vendor_sim.cpp V6 asserts
//    the field is ABSENT, so the day someone adds it, a test goes red and the parse-time refusal it
//    would then need is named in that test rather than re-derived.
// ⛔ REFUSED: making this mechanism's answer an ADMISSION for any class. `admitted` is `false`
//    unconditionally, and the gate throws unconditionally on an active view.
// ---------------------------------------------------------------------------

} // namespace ninfer::caps
