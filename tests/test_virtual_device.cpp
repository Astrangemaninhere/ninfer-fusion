// test_virtual_device.cpp -- the virtual-device guard's contract, checked on a host with no GPU.
//
// A guard whose whole job is to be trustworthy has to be tested where it cannot lean on the thing
// it guards. Every check below is a pure function of numbers passed in, so it runs on a machine
// with no CUDA device at all -- which is also the only way to exercise the refusals, since the
// point of a refusal is that the situation it names cannot be produced on purpose.
//
// The four properties this file asserts, in the wording the mandate uses:
//
//   1. OFF BY DEFAULT.   Nothing set -> inactive, and every consumer of the binding gets the
//      identity world. The engine's behaviour cannot change because this header was added.
//   2. IMPOSSIBLE BY ACCIDENT. One key without the other is a REFUSAL, not a quiet
//      single-device run: an operator who meant to simulate and mistyped must not be served a
//      green run that looks like a successful simulation.
//   3. FAILS CLOSED.     For EVERY refusal: active() == false and the operator is told, by name,
//      which field failed and why. There is no partial or degraded mode.
//   4. NO GATE WEAKENED. The simulated capability is fed INTO the artifact-format gate as its
//      input. A virtual V100 offered an NVFP4 artifact is refused by the REAL gate, exactly as a
//      physical V100 would be -- the guard changes which question is asked, not the answer.

#include "core/arch_caps.h"
#include "core/shard_plan.h"
#include "core/virtual_device.h"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

using ninfer::artifact::NumericFormat;
using ninfer::multi::ParallelAxis;
using ninfer::multi::validate_virtual_request;
using ninfer::multi::VirtualBinding;
using ninfer::multi::VirtualProfile;
using ninfer::multi::VirtualRefusal;
using ninfer::multi::VirtualRequest;

int failures = 0;
int checks   = 0;
int reported = 0;
constexpr int kReportLimit = 30;

void check(bool condition, const std::string& what) {
    ++checks;
    if (!condition) {
        ++failures;
        if (reported < kReportLimit) {
            ++reported;
            std::cout << "FAIL: " << what << "\n";
        } else if (reported == kReportLimit) {
            ++reported;
            std::cout << "FAIL: ... further failures suppressed\n";
        }
    }
}

// The physical facts of THIS box, pinned as constants so the checks are host checks with no
// device dependency. RTX 5090 D: sm_120, 170 SMs (core/device_sm_count.h names 170 as the number
// two call sites used to hardwire), 32 GiB.
constexpr int           kPhysicalDevices = 1;
constexpr int           kPhysicalSms     = 170;
constexpr std::uint64_t kGiB             = 1024ULL * 1024ULL * 1024ULL;
constexpr std::uint64_t kPhysicalMemory  = 32ULL * kGiB;

[[nodiscard]] VirtualRequest request(std::uint32_t world, VirtualProfile profile = VirtualProfile::V100,
                                     ParallelAxis axis = ParallelAxis::Tensor,
                                     std::uint32_t rank = 0, bool ack = true,
                                     bool requested = true) {
    VirtualRequest out;
    out.requested          = requested;
    out.acknowledged       = ack;
    out.world_size         = world;
    out.rank               = rank;
    out.axis               = axis;
    out.profile            = profile;
    out.world_size_raw     = std::to_string(world);
    out.profile_name_raw   = std::string(ninfer::multi::virtual_profile_name(profile));
    return out;
}

// ---------------------------------------------------------------------------
// 1. OFF BY DEFAULT
// ---------------------------------------------------------------------------

void test_off_by_default() {
    // The environment reader, with nothing set. Explicitly cleared so an ambient value in the
    // test harness cannot turn this check into its own opposite.
    unsetenv("NINFER_VIRTUAL_DEVICES");
    unsetenv("NINFER_VIRTUAL_DEVICES_ACK");
    unsetenv("NINFER_VIRTUAL_DEVICE_PROFILE");
    unsetenv("NINFER_VIRTUAL_RANK");

    const VirtualRequest none = ninfer::multi::virtual_request_from_environment();
    check(!none.requested, "with nothing set, no virtual device may be requested");
    check(!validate_virtual_request(request(2, VirtualProfile::V100, ParallelAxis::Tensor, 0, true,
                                            /*requested=*/false),
                                    kPhysicalDevices, kPhysicalSms, kPhysicalMemory)
               .active,
          "and a request that was never made must not be honoured even if its fields are "
          "well-formed");

    const VirtualBinding binding =
        validate_virtual_request(none, kPhysicalDevices, kPhysicalSms, kPhysicalMemory);
    check(!binding.active, "and the binding must be INACTIVE");
    check(binding.refusal == VirtualRefusal::NotRequested,
          "with the specific refusal 'not-requested' -- 'off' is not a failure");
    check(binding.world.world_size == 1U && binding.world.rank == 0U &&
              binding.world.axis == ParallelAxis::None,
          "and the world must be the identity (0 of 1, axis none), which is what every consumer "
          "of the binding already assumed");
    check(binding.capability_sm == 0 && binding.sm_budget == 0 && binding.memory_budget_bytes == 0,
          "an inactive binding must carry no numbers for a consumer to accidentally use");
    check(ninfer::multi::virtual_banner(binding, "x", kPhysicalSms).empty(),
          "an inactive binding must print no banner");

    // The identity world must be a valid WorldShape, or the engine's own checks would refuse the
    // un-simulated case.
    check(ninfer::multi::world_shape_valid(binding.world),
          "the identity world a default DeviceContext carries must pass the engine's own "
          "world_shape_refusal");
}

// ---------------------------------------------------------------------------
// 2. IMPOSSIBLE BY ACCIDENT
// ---------------------------------------------------------------------------

void test_one_key_without_the_other_is_a_refusal() {
    // Intent without acknowledgement. This is the check that makes the guard impossible to enable
    // by accident: the mistake must be SERVED, not swallowed.
    for (const std::string& ack : {std::string{}, std::string("1"), std::string("true"),
                                   std::string("YES"), std::string("Yes")}) {
        VirtualRequest half = request(2);
        half.acknowledged   = ack == "yes";
        const VirtualBinding binding =
            validate_virtual_request(half, kPhysicalDevices, kPhysicalSms, kPhysicalMemory);
        check(!binding.active,
              "an acknowledgement of \"" + ack + "\" must NOT activate the guard");
        check(binding.refusal == VirtualRefusal::MissingAck,
              "and must be refused as 'missing-acknowledgement', not quietly ignored");
        check(binding.refusal_detail.find("NINFER_VIRTUAL_DEVICES_ACK") != std::string::npos,
              "and the refusal must name the variable to set");
        check(binding.refusal_detail.find("BUDGETS") != std::string::npos ||
                  binding.refusal_detail.find("budget") != std::string::npos,
              "and must say what a virtual device's numbers are NOT: got " +
                  binding.refusal_detail);
    }
    // The environment reader must agree with the validator about what "acknowledged" means.
    setenv("NINFER_VIRTUAL_DEVICES", "2", 1);
    setenv("NINFER_VIRTUAL_DEVICES_ACK", "TRUE", 1);
    const VirtualRequest parsed = ninfer::multi::virtual_request_from_environment();
    check(parsed.requested && !parsed.acknowledged,
          "the reader must treat any acknowledgement other than the exact word as absent");
    check(!validate_virtual_request(parsed, kPhysicalDevices, kPhysicalSms, kPhysicalMemory).active,
          "and the validator must refuse it");
    unsetenv("NINFER_VIRTUAL_DEVICES");
    unsetenv("NINFER_VIRTUAL_DEVICES_ACK");
}

// ---------------------------------------------------------------------------
// 3. THE HAPPY PATH, AND WHAT IT ACTUALLY BINDS
// ---------------------------------------------------------------------------

void test_the_active_binding_binds_a_rank_and_a_budget() {
    for (const std::uint32_t world : {2U, 4U, 8U}) {
        for (std::uint32_t rank = 0; rank < world; ++rank) {
            const VirtualBinding binding =
                validate_virtual_request(request(world, VirtualProfile::V100, ParallelAxis::Tensor,
                                                 rank),
                                         kPhysicalDevices, kPhysicalSms, kPhysicalMemory);
            check(binding.active, "world " + std::to_string(world) + " rank " +
                                      std::to_string(rank) + " must be honoured: " +
                                      binding.refusal_detail);
            if (!binding.active) { continue; }
            check(binding.world.world_size == world && binding.world.rank == rank &&
                      binding.world.axis == ParallelAxis::Tensor,
                  "the binding must carry the world it was asked for");
            check(ninfer::multi::world_shape_valid(binding.world),
                  "and it must be a world the engine's own refusal accepts");
            check(binding.capability_sm == 70,
                  "a v100 profile must report sm_70, the number DeviceContext::sm() returns");
            check(binding.sm_budget == static_cast<std::uint32_t>(kPhysicalSms / world),
                  "the per-rank SM budget must be the physical count divided by the world (" +
                      std::to_string(kPhysicalSms) + "/" + std::to_string(world) + "), got " +
                      std::to_string(binding.sm_budget));
            check(binding.memory_budget_bytes == 16ULL * kGiB,
                  "a v100 profile must stand for 16 GiB");
            check(binding.physical_device_index == 0,
                  "every virtual rank is bound to physical device 0: the world is made of ranks, "
                  "not of devices");
            // THE RANK ROLE. Both forks that built this keep it explicit, and it decides which
            // rank may own a request, which rank a sampling result comes from, and which rank a
            // stall is attributable to -- so a binding carrying only an index would be less than
            // they already have. It must be derived from the rank, not stored beside it.
            check(binding.primary() == (rank == 0U),
                  "rank " + std::to_string(rank) + " must report primary=" +
                      (rank == 0U ? "true" : "false") +
                      ": rank 0 owns tokenization, sampling, the retained MTP target hidden state "
                      "and every request-visible state, and the other ranks own none of it");
            check(binding.primary() == (binding.world.rank == 0U),
                  "and the role must be derived from the rank so the two cannot disagree");
        }
    }
    // Every profile must be reachable and must report its own sm number: a closed set that nobody
    // can enumerate is not a closed set.
    struct Row { VirtualProfile profile; int sm; const char* name; std::uint64_t memory; };
    const Row rows[] = {{VirtualProfile::V100, 70, "v100", 16ULL * kGiB},
                        {VirtualProfile::T4, 75, "t4", 16ULL * kGiB},
                        {VirtualProfile::A100, 80, "a100", 40ULL * kGiB},
                        {VirtualProfile::RTX30, 86, "rtx30", 24ULL * kGiB},
                        {VirtualProfile::RTX40, 89, "rtx40", 24ULL * kGiB},
                        {VirtualProfile::H100, 90, "h100", 80ULL * kGiB}};
    for (const Row& row : rows) {
        check(std::string(ninfer::multi::virtual_profile_name(row.profile)) == row.name,
              "the profile must name itself: " + std::string(row.name));
        check(ninfer::multi::virtual_profile_sm(row.profile) == row.sm,
              std::string(row.name) + " must report sm_" + std::to_string(row.sm));
        check(ninfer::multi::virtual_profile_from_name(row.name) == row.profile,
              "and the name must round trip to the profile: " + std::string(row.name));
        check(ninfer::multi::virtual_profile_memory_bytes(row.profile) == row.memory,
              std::string(row.name) + " must stand for its own memory size");

        // A profile whose memory EXCEEDS this box's card is refused, and that refusal is a real
        // fact rather than a blanket one: a rank told it may spend 40 GiB on a 32 GiB card would
        // make weight-residency decisions that are unreachable on the hardware underneath.
        // Both directions are asserted, so neither "honour everything" nor "refuse everything"
        // survives this loop.
        const bool fits = row.memory <= kPhysicalMemory;
        const VirtualBinding on_this_box = validate_virtual_request(
            request(2, row.profile), kPhysicalDevices, kPhysicalSms, kPhysicalMemory);
        check(on_this_box.active == fits,
              std::string(row.name) + " on a " + std::to_string(kPhysicalMemory / kGiB) +
                  " GiB card must be " + (fits ? "honoured" : "refused as memory-too-small") +
                  "; got " + std::string(ninfer::multi::virtual_refusal_name(on_this_box.refusal)) +
                  ": " + on_this_box.refusal_detail);
        if (!fits) {
            check(on_this_box.refusal == VirtualRefusal::MemoryTooSmall,
                  std::string(row.name) + " must be refused for the SPECIFIC reason that its "
                  "memory does not fit, not for some other field");
            // And the refusal must carry NO numbers, so the two facts (which card it would be,
                  // and whether that card fits here) cannot be half-honoured by a consumer.
            check(on_this_box.capability_sm == 0 && on_this_box.sm_budget == 0 &&
                      on_this_box.memory_budget_bytes == 0,
                  std::string(row.name) +
                      ": a memory refusal must expose no capability number either -- a consumer "
                      "that read capability_sm without reading active() would otherwise run the "
                      "run under a card the guard refused");
            check(ninfer::multi::virtual_profile_sm(row.profile) == row.sm,
                  std::string(row.name) +
                      ": the profile's own capability is still a fact about the profile, asked "
                      "through the profile accessor rather than through a refused binding");
        }

        // On a card big enough for every profile in the set, every one is honoured. This is what
        // proves the memory check is a comparison against the card rather than a hardcoded list.
        const VirtualBinding on_a_bigger_card = validate_virtual_request(
            request(2, row.profile), kPhysicalDevices, kPhysicalSms, 96ULL * kGiB);
        check(on_a_bigger_card.active,
              std::string(row.name) + " must be simulatable on a 96 GiB card; got " +
                  std::string(ninfer::multi::virtual_refusal_name(on_a_bigger_card.refusal)) +
                  ": " + on_a_bigger_card.refusal_detail);
    }
    check(ninfer::multi::virtual_profile_from_name("h200") == VirtualProfile::None,
          "an unknown profile name must map to None, which is then refused -- never defaulted");
}

// ---------------------------------------------------------------------------
// 4. FAILS CLOSED, over the WHOLE refusal space
// ---------------------------------------------------------------------------

void test_every_refusal_fails_closed_and_says_why() {
    struct Case { const char* label; VirtualRequest req; int devices; int sms; std::uint64_t mem;
                  VirtualRefusal want; };
    const std::vector<Case> cases = {
        {"world_size 0", request(0, VirtualProfile::V100, ParallelAxis::Tensor, 0, true),
         kPhysicalDevices, kPhysicalSms, kPhysicalMemory, VirtualRefusal::BadWorldSize},
        {"world_size 1", request(1), kPhysicalDevices, kPhysicalSms, kPhysicalMemory,
         VirtualRefusal::BadWorldSize},
        {"world_size 65", request(65), kPhysicalDevices, kPhysicalSms, kPhysicalMemory,
         VirtualRefusal::BadWorldSize},
        {"unknown profile", request(2, VirtualProfile::None), kPhysicalDevices, kPhysicalSms,
         kPhysicalMemory, VirtualRefusal::BadProfile},
        {"axis pp", request(2, VirtualProfile::V100, ParallelAxis::Pipeline), kPhysicalDevices,
         kPhysicalSms, kPhysicalMemory, VirtualRefusal::BadAxis},
        {"axis none", request(2, VirtualProfile::V100, ParallelAxis::None), kPhysicalDevices,
         kPhysicalSms, kPhysicalMemory, VirtualRefusal::BadAxis},
        {"rank outside world", request(2, VirtualProfile::V100, ParallelAxis::Tensor, 2),
         kPhysicalDevices, kPhysicalSms, kPhysicalMemory, VirtualRefusal::BadRank},
        {"two physical devices", request(2), 2, kPhysicalSms, kPhysicalMemory,
         VirtualRefusal::PhysicalNotSingle},
        {"no SMs", request(2), kPhysicalDevices, 0, kPhysicalMemory, VirtualRefusal::SmsTooFew},
        {"one SM per rank", request(64), kPhysicalDevices, 32, kPhysicalMemory,
         VirtualRefusal::SmsTooFew},
        {"profile memory exceeds the card", request(2, VirtualProfile::H100), kPhysicalDevices,
         kPhysicalSms, 8ULL * kGiB, VirtualRefusal::MemoryTooSmall},
    };
    for (const Case& test_case : cases) {
        const VirtualBinding binding = validate_virtual_request(test_case.req, test_case.devices,
                                                               test_case.sms, test_case.mem);
        check(!binding.active,
              std::string("case '") + test_case.label + "': a refused request must NOT be active");
        check(binding.refusal == test_case.want,
              std::string("case '") + test_case.label + "': expected refusal " +
                  std::string(ninfer::multi::virtual_refusal_name(test_case.want)) + ", got " +
                  std::string(ninfer::multi::virtual_refusal_name(binding.refusal)));
        check(!binding.refusal_detail.empty(),
              std::string("case '") + test_case.label +
                  "': a refusal must tell the operator what failed");
        check(binding.refusal_detail.size() > 40U,
              std::string("case '") + test_case.label +
                  "': and must say enough to act on, not one word");
        check(ninfer::multi::virtual_banner(binding, "x", kPhysicalSms).empty(),
              std::string("case '") + test_case.label + "': a refused binding must print nothing "
              "-- a banner plus a refusal reads as 'it worked, with caveats'");
        // THE PROPERTY, asserted structurally over every case rather than one by one: no refusal
        // may leave a usable number behind.
        check(binding.capability_sm == 0 && binding.sm_budget == 0 &&
                  binding.memory_budget_bytes == 0,
              std::string("case '") + test_case.label +
                  "': a refused binding must expose no numbers, so no consumer can honour half of "
                  "a simulation");
    }
    // Red control: the case list must be able to produce an ACTIVE binding, or every check above
    // would pass on a validator that refuses everything.
    check(validate_virtual_request(request(2), kPhysicalDevices, kPhysicalSms, kPhysicalMemory)
              .active,
          "a well-formed request must be honoured -- otherwise 'everything is refused' would "
          "pass this whole group");
}

// ---------------------------------------------------------------------------
// 5. NO GATE WEAKENED -- the simulated capability goes INTO the real gate
// ---------------------------------------------------------------------------

void test_the_virtual_capability_is_fed_into_the_real_gate() {
    const VirtualBinding v100_rank1 =
        validate_virtual_request(request(2, VirtualProfile::V100, ParallelAxis::Tensor, 1),
                                 kPhysicalDevices, kPhysicalSms, kPhysicalMemory);
    check(v100_rank1.active, "the v100 binding must be active for this check to mean anything");

    // svc note: this is the SAME gate src/targets/registry.cpp reaches, taking the number
    // `DeviceContext::sm()` would return under the binding. The point is that the guard does not
    // answer the question -- it asks a different one.
    //
    // THE FOURTH ARGUMENT IS PASSED EXPLICITLY, and it is the value the parameter's own default
    // already supplies (`bool qpn_in_build = kQpnInBuild`). It is spelled out because
    // `src/core/arch_caps.h` currently declares a second, three-parameter overload that differs
    // ONLY by that trailing defaulted `bool`, which makes every three-argument call AMBIGUOUS:
    // `src/targets/registry.cpp`, `tests/test_multidev_wiring.cpp` and `tests/test_arch_caps.cpp`
    // all fail to compile that way. Selecting the four-parameter overload explicitly changes no
    // behaviour and keeps this check honest while the overload set is repaired by its owner (see
    // the MULTIGPU report; the call site in registry.cpp was fixed the same way).
    using ninfer::caps::kQpnInBuild;

    // THE TWO WORLDS, ONE BUILD FACT APART, through the `qpn_in_build` parameter.
    //
    // (a) NO FALLBACK KERNEL IN THE BUILD. This is the world this check was written in, and it is
    // asserted with the parameter spelled out rather than left to whatever -D this TU happens to
    // receive, so the refusal stays a red control even if the build system's reach changes again.
    bool threw = false;
    std::string message;
    try {
        ninfer::caps::require_artifact_formats_supported(
            v100_rank1.capability_sm, std::vector<NumericFormat>{NumericFormat::NVFP4}, "m/w",
            /*qpn_in_build=*/false);
    } catch (const std::invalid_argument& error) {
        threw = true;
        message = error.what();
    }
    check(threw,
          "a virtual V100 offered an NVFP4 artifact must be REFUSED by the real gate when the "
          "build has NO fp16 fallback kernel: sm_70's ladder row has no kind::mxf4nvf4, and that "
          "is a true fact about V100, not a simulation artefact. A guard that made this pass "
          "would be weakening a gate.");
    check(!message.empty() && message.find("70") != std::string::npos,
          "and the refusal must name the capability it is about");

    // (b) THE WORLD THIS BINARY IS IN. The SECOND FLOOR is what answers the V100 rung here, and
    // the requirement is that the answer SAYS SO rather than being a silent pass -- the same
    // distinction (a) draws, from the other side. A gate that refused here would be the mirror
    // defect of a gate that passed silently: it would hide a route the tree can launch.
    const auto served_by_fallback = ninfer::caps::evaluate_artifact_formats(
        v100_rank1.capability_sm, std::vector<NumericFormat>{NumericFormat::NVFP4},
        /*qpn_in_build=*/true);
    check(served_by_fallback.verdict == ninfer::caps::Verdict::Supported &&
              served_by_fallback.fallbacks.size() == 1U,
          "and with the QPN fp16 fallback kernel in the build the SAME gate must answer "
          "Supported WITH one recorded FormatFallback: the sm_70 rung is servable, and its "
          "fallback route is what has to compute the right answer, not nvfp4");
    const std::string fallback_notice =
        ninfer::caps::render_fallback_notice(served_by_fallback, "m/w");
    check(fallback_notice.find("does NOT meet its own floor") != std::string::npos &&
              fallback_notice.find("qpn") != std::string::npos,
          "and the loud notice must name BOTH the floor the V100 rung misses and the QPN kernel "
          "that carries it, so 'supported' is never read as 'meets the floor'");

    // The real device's number, by contrast, DOES satisfy the gate on this box's artifact set --
    // so the two directions differ, which is what makes the check above a comparison rather than
    // a constant.
    bool physical_threw = false;
    try {
        ninfer::caps::require_artifact_formats_supported(
            120, std::vector<NumericFormat>{NumericFormat::NVFP4}, "m/w", kQpnInBuild);
    } catch (const std::invalid_argument&) { physical_threw = true; }
    check(!physical_threw,
          "sm_120 must still satisfy the NVFP4 floor: the virtual check above is only meaningful "
          "because the physical answer is different");
}

// ---------------------------------------------------------------------------
// 6. LOUD, AND HONEST ABOUT WHAT IT CANNOT PROVE
// ---------------------------------------------------------------------------

void test_the_banner_is_loud_and_names_what_is_pending_hardware() {
    const VirtualBinding binding =
        validate_virtual_request(request(2, VirtualProfile::V100), kPhysicalDevices, kPhysicalSms,
                                 kPhysicalMemory);
    check(binding.active, "the banner check needs an active binding");
    const std::string banner = ninfer::multi::virtual_banner(binding, "NVIDIA GeForce RTX 5090 D",
                                                            kPhysicalSms);
    check(!banner.empty(), "an active binding must print a banner");
    check(banner.find("VIRTUAL DEVICES") != std::string::npos,
          "and it must be impossible to miss");
    check(banner.find("v100") != std::string::npos && banner.find("70") != std::string::npos,
          "and it must name the card being simulated and its capability");
    check(banner.find("NVIDIA GeForce RTX 5090 D") != std::string::npos &&
              banner.find(std::to_string(kPhysicalSms)) != std::string::npos,
          "and the physical device underneath, with its real SM count");
    check(banner.find("not a partition") != std::string::npos ||
              banner.find("BUDGETS") != std::string::npos,
          "and it must say that the SM and memory numbers are budgets the ranks SHARE, not "
          "partitions: nothing else in the output distinguishes the two");
    check(banner.find("CAPACITY") != std::string::npos,
          "and it must name what the simulation IS: a capacity shape, which is what --tp N "
          "gives on N cards");
    check(banner.find("TRANSPORT") != std::string::npos &&
              banner.find("NOT THIS COMPONENT") != std::string::npos,
          "and it must name the transport as the shard interface's counter-party rather than as "
          "a missing capability of this engine: the collectives are somebody else's component, "
          "and this engine owes them a correct interface rather than a measurement");
    check(banner.find("NOT THIS COMPONENT") != std::string::npos &&
              banner.find("needs measuring") != std::string::npos,
          "and must say that nothing about the transport needs measuring here, so the caveat is "
          "not read as an apology for a gap");
    check(banner.find("PENDING HARDWARE") != std::string::npos,
          "and it must name the ONE quantity that is pending hardware: aggregate scaling");
    check(banner.find("AGGREGATE THROUGHPUT") != std::string::npos,
          "naming it as AGGREGATE THROUGHPUT SCALING, not as 'any multi-GPU number' -- per-rank "
          "cost, skew and balance ARE measurable and a blanket disclaimer would throw them away");
    check(banner.find("PER-RANK") != std::string::npos ||
              banner.find("per-rank wall clock") != std::string::npos,
          "and it must say which numbers ARE measurable, or the disclaimer is unusable");
    check(banner.find("globaltimer") != std::string::npos,
          "including the skew clock, which is what makes inter-rank balance checkable");
    check(banner.find("40%") != std::string::npos &&
              banner.find("TWO REAL 5090s") != std::string::npos,
          "and must attribute the sibling fork's '~40% faster at long context' to the two REAL "
          "cards it was measured on -- the failure mode this project keeps catching is exactly "
          "that number being read as ours");

    // The one-line caveat, for a log that wants the warning without the box. It must name the ONE
    // quantity it disclaims, so it cannot be reused to disclaim the measurable ones.
    const std::string caveat = ninfer::multi::virtual_caveat("aggregate tok/s");
    check(caveat.find("aggregate tok/s") != std::string::npos &&
              caveat.find("PENDING HARDWARE") != std::string::npos &&
              caveat.find("aggregate") != std::string::npos,
          "virtual_caveat() must name the quantity, mark it pending hardware, and say which "
          "quantity it is talking about");
    check(caveat.find("Per-rank cost") != std::string::npos,
          "and must state that per-rank cost, skew and balance are NOT pending -- a caveat that "
          "disclaims the whole run is as wrong as no caveat");
}

// ---------------------------------------------------------------------------
// 7. PER-RANK WALL CLOCK AND SKEW -- measurable on one device, and checked here
// ---------------------------------------------------------------------------

void test_rank_timing_is_measurable_and_the_spread_is_arithmetic() {
    using ninfer::multi::RankTiming;
    using ninfer::multi::rank_timing_spread;
    using ninfer::multi::VirtualTimingMode;

    // A perfectly balanced world: same elapsed, same start. Imbalance must be exactly 1.0 and the
    // skew exactly 0 -- asserted as equalities, because "balanced" is not a tolerance.
    std::vector<RankTiming> balanced{{0, 1'000'000, 50'000}, {1, 1'000'000, 50'000}};
    const auto flat = rank_timing_spread(balanced);
    check(flat.ok, "a balanced world's spread must be computable: " + flat.reason);
    check(flat.start_skew_ns == 0, "identical start timestamps must give zero skew");
    check(flat.elapsed_spread_ns == 0 && flat.elapsed_imbalance == 1.0,
          "identical elapsed times must give zero spread and an imbalance of exactly 1.0");

    // A skewed world: rank 1 starts 30 us late and takes twice as long. Both facts are separate.
    std::vector<RankTiming> skewed{{0, 1'000'000, 50'000}, {1, 1'030'000, 100'000}};
    const auto uneven = rank_timing_spread(skewed);
    check(uneven.ok, "the skewed world's spread must be computable");
    check(uneven.start_skew_ns == 30'000,
          "the start skew must be the max-minus-min of the absolute %globaltimer samples, got " +
              std::to_string(uneven.start_skew_ns));
    check(uneven.elapsed_min_ns == 50'000 && uneven.elapsed_max_ns == 100'000,
          "the elapsed spread must report both ends, not just the aggregate");
    check(uneven.elapsed_imbalance == 2.0,
          "and the imbalance must be max/min == 2.0, got " +
              std::to_string(uneven.elapsed_imbalance));

    // Red control, and the reason the two numbers are separate: two worlds can have the SAME
    // skew and DIFFERENT balance, so a single number could not describe both.
    std::vector<RankTiming> same_skew_different_work{{0, 1'000'000, 40'000},
                                                    {1, 1'000'000, 80'000}};
    const auto second = rank_timing_spread(same_skew_different_work);
    check(second.start_skew_ns == 0 && second.elapsed_imbalance == 2.0,
          "zero skew with a 2x imbalance must be representable -- it is the shard plan's defect "
          "and not a scheduling one");

    // An unmeasured rank is a REFUSAL, not a zero: a zero would report a perfectly balanced world
    // in which one rank did nothing.
    std::vector<RankTiming> unmeasured{{0, 1'000'000, 50'000}, {1, 1'000'000, 0}};
    const auto bad = rank_timing_spread(unmeasured);
    check(!bad.ok && bad.reason.find("rank 1") != std::string::npos,
          "a rank with zero elapsed time must be refused by name, got: " + bad.reason);
    check(!rank_timing_spread({}).ok, "timing no ranks at all must be refused");

    // The line must carry the mode, because the two modes answer different questions.
    const std::string serialized = ninfer::multi::rank_timing_spread_line(
        uneven, VirtualTimingMode::Serialized);
    const std::string concurrent = ninfer::multi::rank_timing_spread_line(
        uneven, VirtualTimingMode::Concurrent);
    check(serialized.find("serialized") != std::string::npos &&
              concurrent.find("concurrent") != std::string::npos &&
              serialized != concurrent,
          "the report line must name which of the two runs it came from, because per-rank cost "
          "and per-rank cost under contention are different numbers");
    check(serialized.find("PENDING HARDWARE") != std::string::npos &&
              serialized.find("ARE measurable") != std::string::npos,
          "and it must separate the pending quantity from the measured one in the same line");
}

// ---------------------------------------------------------------------------
// 8. EVERY RUNG OF THE LADDER, DRIVEN -- V100 through 4090 and beyond
// ---------------------------------------------------------------------------
//
// The point of the virtual split is not one profile: it is that the SAME engine, on this one
// 5090 D, can be driven through every rung of the capability ladder and answer coherently at
// each one. Each rung below is asked the same question the engine asks at artifact-load time
// (`device.sm()` -> the format gate), and the assertions are about COHERENCE: the rung exists in
// the ladder, the verdict is one of the four defined ones, the answer is deterministic, and the
// refusing rungs refuse for a named reason rather than by a number that happens to miss a table.
//
// What is asserted HERE is the TABLE's answer per rung. Whether the fp4 W4A4 KERNEL is compiled
// into a given rung's binary is a separate, build-time fact and is not visible from a host
// program; it is measured with nvcc per rung and recorded in the report. The two must agree, and
// the report says where they do.
void test_every_ladder_rung_answers_coherently() {
    struct Rung { VirtualProfile profile; int sm; const char* name; };
    const Rung rungs[] = {{VirtualProfile::V100, 70, "v100"},  {VirtualProfile::T4, 75, "t4"},
                          {VirtualProfile::A100, 80, "a100"},  {VirtualProfile::RTX30, 86, "rtx30"},
                          {VirtualProfile::RTX40, 89, "rtx40"}, {VirtualProfile::H100, 90, "h100"}};

    std::cout << "  rung     sm   ladder_row  nvfp4_verdict\n";
    for (const Rung& rung : rungs) {
        // The profile's own sm number must BE the rung, or the simulator is presenting a card
        // whose number the ladder will answer about differently than the profile claims.
        check(ninfer::multi::virtual_profile_sm(rung.profile) == rung.sm,
              std::string(rung.name) + " must still report sm_" + std::to_string(rung.sm));

        // Every rung the ladder knows must have a row, so its floor is CHECKABLE rather than
        // guessed. (A rung with no row is not a defect -- it is the UnknownArch warning path --
        // but these six are all named cards the engine intends to support, so a missing row here
        // means the ladder lost one.)
        const bool has_row = ninfer::caps::arch_rung(rung.sm) != nullptr;

        const auto first = ninfer::caps::evaluate_artifact_formats(
            rung.sm, std::vector<NumericFormat>{NumericFormat::NVFP4});
        const auto second = ninfer::caps::evaluate_artifact_formats(
            rung.sm, std::vector<NumericFormat>{NumericFormat::NVFP4});

        // A verdict from the four defined values, and the SAME verdict twice: a table whose answer
        // depended on call order would make every downstream decision irreproducible.
        const bool defined = first.verdict == ninfer::caps::Verdict::Supported ||
                             first.verdict == ninfer::caps::Verdict::Unsupported ||
                             first.verdict == ninfer::caps::Verdict::UnknownArch ||
                             first.verdict == ninfer::caps::Verdict::UnknownFormat;
        check(defined, std::string(rung.name) + " (sm_" + std::to_string(rung.sm) +
                           ") must give one of the four defined verdicts");
        check(first.verdict == second.verdict && first.ok() == second.ok(),
              std::string(rung.name) + "'s verdict must be deterministic");

        // Coherence with the non-gate principle, per rung: a rung with NO row must warn, not
        // refuse; a rung WITH a row must answer from its row.
        if (!has_row) {
            check(first.verdict == ninfer::caps::Verdict::UnknownArch,
                  std::string(rung.name) +
                      " has no ladder row, so it must be UnknownArch (the warning path) rather "
                      "than a refusal manufactured from a missing table entry");
        }
        std::cout << "  " << rung.name << std::string(9U - std::string(rung.name).size(), ' ')
                  << "sm_" << rung.sm << "  " << (has_row ? "yes" : "NO ")
                  << "         "
                  << (first.verdict == ninfer::caps::Verdict::Supported     ? "supported"
                      : first.verdict == ninfer::caps::Verdict::Unsupported ? "unsupported"
                      : first.verdict == ninfer::caps::Verdict::UnknownArch ? "unknown-arch"
                                                                            : "unknown-format")
                  << "\n";
    }

    // THE TWO DIRECTIONS, as a pair, so the loop above is a comparison and not a catalogue: the
    // card this box has satisfies the nvfp4 floor and the V100 rung does not -- OUT OF ITS OWN
    // ROW. If both agreed, the per-rung answer would be uninformative.
    const auto card_here = ninfer::caps::evaluate_artifact_formats(
        120, std::vector<NumericFormat>{NumericFormat::NVFP4});
    // `v100` used to be read with the DEFAULTED `qpn_in_build`, which made this check's answer a
    // fact about this TU's -D rather than about the rung. Both calls below pass it explicitly, so
    // the pair separates the two questions that were being conflated: what sm_70's OWN row covers,
    // and what THIS BUILD's second floor does about it.
    const auto v100_bare = ninfer::caps::evaluate_artifact_formats(
        70, std::vector<NumericFormat>{NumericFormat::NVFP4}, /*qpn_in_build=*/false);
    check(card_here.verdict == ninfer::caps::Verdict::Supported,
          "this box's sm_120 must satisfy the nvfp4 floor");
    check(v100_bare.verdict == ninfer::caps::Verdict::Unsupported,
          "sm_70 must NOT satisfy it out of its own row -- and note what that means: the V100 "
          "rung is presentable and its FALLBACK route is what has to compute the right answer, "
          "not nvfp4");
    // And sm_70's refusal must be EVIDENCE-BACKED: it has a ladder row, so the refusal comes from
    // a real capability gap, and the rendering must name the missing capability and its kernel.
    const std::string why = ninfer::caps::render_capability_report(v100_bare, "rung/v100");
    check(!why.empty() && why.find("70") != std::string::npos,
          "the V100 rung's refusal must name the capability it is about");
    check(why.find("mma.cuh") != std::string::npos || why.find("kernel") != std::string::npos,
          "and must name the KERNEL that sets the floor rather than only the card, which is the "
          "difference between an actionable refusal and a number-based one");

    // THE SAME RUNG, THE OTHER ANSWER -- and it is the one this build gives. Asserting only the
    // refusal pinned the ABSENCE of the second floor; asserting this side is what pins the
    // MECHANISM, whose whole point is that a Supported verdict here must be LOUD about the floor
    // it did not meet. A silent Supported is the false positive this table exists to prevent.
    const auto v100_served = ninfer::caps::evaluate_artifact_formats(
        70, std::vector<NumericFormat>{NumericFormat::NVFP4}, /*qpn_in_build=*/true);
    check(v100_served.verdict == ninfer::caps::Verdict::Supported &&
              v100_served.fallbacks.size() == 1U,
          "with the QPN fp16 fallback kernel in the build, sm_70 must answer Supported WITH one "
          "recorded FormatFallback: the V100 rung is servable, and the record is what keeps "
          "'servable' from being read as 'meets the floor'");
    const std::string served_note = ninfer::caps::render_fallback_notice(v100_served, "rung/v100");
    check(served_note.find("does NOT meet its own floor") != std::string::npos &&
              served_note.find("qpn") != std::string::npos &&
              served_note.find("70") != std::string::npos,
          "and the fallback notice must name the V100 rung, the floor it misses, and the QPN "
          "kernel that carries it");
}

} // namespace

int main() {
    test_off_by_default();
    test_one_key_without_the_other_is_a_refusal();
    test_the_active_binding_binds_a_rank_and_a_budget();
    test_every_refusal_fails_closed_and_says_why();
    test_the_virtual_capability_is_fed_into_the_real_gate();
    test_the_banner_is_loud_and_names_what_is_pending_hardware();
    test_rank_timing_is_measurable_and_the_spread_is_arithmetic();
    test_every_ladder_rung_answers_coherently();

    std::cout << "virtual_device: " << checks << " checks, " << failures << " failures -> "
              << (failures == 0 ? "PASS" : "FAIL") << "\n";
    return failures == 0 ? 0 : 1;
}
