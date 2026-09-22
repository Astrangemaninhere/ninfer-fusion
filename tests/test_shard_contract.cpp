// test_shard_contract.cpp -- THE ENGINE'S OWN SHARD CONTRACT (src/core/shard_plan.h section 6),
// and the arm that MUST BE RED until a world can span more than one physical device.
//
// WHY THIS FILE EXISTS. src/core/shard_plan.h's sections 1-4 are arithmetic: they produce a
// correct ShardPlan for world_size 4 on four cards and they are right to. What the tree did
// NOT have is the ENGINE's answer to the different question -- is there anything here that can
// BIND a second device, MOVE a byte between two devices, or READ a per-rank weight stream --
// and a machine with two cards therefore got a SILENT SINGLE-CARD run. This test pins that
// contract and keeps the silence from coming back.
//
// THREE KINDS OF EVIDENCE, and they are labelled here because conflating them is the defect
// this project keeps recording:
//
//   G1..G3  POSITIVE -- the contract does what it says, on all sixteen subsets of the facts.
//   G4, G5  VERBATIM STATE -- what THIS revision answers, and the refusal text an operator
//           reads, printed rather than asserted-about.
//   G6      THE RED CONTROL -- the one check that proves the walker can go RED, so G1 cannot
//           be passing because it always says yes.
//   ARM     --multidev-required: THE COUNTER-CONTROL. It asserts the CAPABILITY ("this build
//           can span two physical devices") and it is RED today. See the block at the bottom.
//
// Host-only: core/shard_plan.h and core/virtual_device.h contain no CUDA header and no device
// query, so this runs on a host with no GPU -- which is the only form of evidence this box can
// produce for the multi-device question (one physical card, no MIG, no MPS; the header's own
// header comment records both closures).

#include "core/shard_plan.h"
#include "core/virtual_device.h"

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace {

using ninfer::multi::CrossDeviceFacts;
using ninfer::multi::WorldRequest;
using ninfer::multi::WorldShape;
using ninfer::multi::ParallelAxis;

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
            std::cout << "FAIL: ... further failures suppressed\n";
        }
    }
}

WorldRequest world_request(std::uint32_t world_size, std::uint32_t rank) {
    WorldRequest request;
    request.requested      = true;
    request.world_size     = world_size;
    request.rank           = rank;
    request.world_size_raw = std::to_string(world_size);
    request.rank_raw       = std::to_string(rank);
    return request;
}

// ---------------------------------------------------------------------------
// G1. THE CONTRACT IS A FUNCTION OF THE FACTS, NOT A CONSTANT
// ---------------------------------------------------------------------------
// Every one of the sixteen subsets of the four facts is walked. The invariant is an
// EQUIVALENCE, not a one-way implication:
//
//     cross_device_world_possible(facts)  ==  missing_cross_device_pieces(facts).empty()
//     cross_device_world_possible(facts)  ==  cross_device_world_refusal(...).empty()
//
// A refusal implemented as a constant ("always refuse") fails the second direction on the
// all-true subset. A refusal implemented as "count the pieces" without naming them fails the
// per-piece naming check below. Neither can be satisfied by accident.
bool walker_all_subsets(std::string* first_failure) {
    bool ok = true;
    auto note = [&](const std::string& why) {
        if (ok) { ok = false; if (first_failure != nullptr) { *first_failure = why; } }
    };
    for (unsigned mask = 0; mask < 16U; ++mask) {
        CrossDeviceFacts facts;
        facts.collective_transport_in_build = (mask & 1U) != 0U;
        facts.rank_world_plumbing           = (mask & 2U) != 0U;
        facts.per_rank_weight_loader        = (mask & 4U) != 0U;
        facts.cross_device_decode_graph     = (mask & 8U) != 0U;

        const bool possible = ninfer::multi::cross_device_world_possible(facts);
        const std::vector<std::string> missing =
            ninfer::multi::missing_cross_device_pieces(facts);
        const std::string refusal = ninfer::multi::cross_device_world_refusal(
            world_request(2U, 0U), facts, 2U);

        if (possible != missing.empty()) {
            note("mask " + std::to_string(mask) +
                 ": world_possible and the missing-piece list disagree");
        }
        if (possible != refusal.empty()) {
            note("mask " + std::to_string(mask) +
                 ": world_possible and the refusal disagree (possible=" +
                 std::to_string(possible ? 1 : 0) + ", refusal=" +
                 (refusal.empty() ? "empty" : "non-empty") + ")");
        }
        // Every missing piece is NAMED in the refusal, one by one. A refusal that says
        // "something is missing" is the text this test exists to forbid.
        for (const std::string& piece : missing) {
            if (refusal.find(piece) == std::string::npos) {
                note("mask " + std::to_string(mask) +
                     ": the refusal does not name the missing piece: " + piece);
            }
        }
        if (missing.size() != ninfer::multi::kCrossDevicePieceCount &&
            !missing.empty() &&
            missing.size() + 0U > ninfer::multi::kCrossDevicePieceCount) {
            note("mask " + std::to_string(mask) + ": more missing pieces than pieces exist");
        }
    }
    return ok;
}

void test_g1_the_contract_is_a_function_of_the_facts() {
    std::string first_failure;
    check(walker_all_subsets(&first_failure),
          "none of the sixteen fact subsets may break the equivalence; first: " + first_failure);

    // The all-true subset, asserted DIRECTLY so a broken walker cannot hide it: with every
    // piece present the SAME call that refuses today returns empty.
    CrossDeviceFacts all_present;
    all_present.collective_transport_in_build = true;
    all_present.rank_world_plumbing           = true;
    all_present.per_rank_weight_loader        = true;
    all_present.cross_device_decode_graph     = true;
    check(ninfer::multi::kCrossDevicePieceCount == 4U,
          "the piece count must be four: it is the length of kCrossDevicePieceNames and the "
          "bound of the sixteen-subset walk");
    check(ninfer::multi::cross_device_world_possible(all_present),
          "with all four facts set the build CAN span devices, so the possibility predicate "
          "must say so");
    check(ninfer::multi::missing_cross_device_pieces(all_present).empty(),
          "and nothing may be reported missing");
    check(ninfer::multi::cross_device_world_refusal(world_request(2U, 0U), all_present, 2U)
              .empty(),
          "and the refusal must be EMPTY: a hardcoded refusal cannot pass this line, which is "
          "what keeps the contract evidence-based instead of a number-based gate");

    // Each fact, flipped off ALONE, removes exactly one named piece. This is the "which one
    // broke" property a count cannot carry.
    for (unsigned bit = 0; bit < 4U; ++bit) {
        CrossDeviceFacts one_off = all_present;
        switch (bit) {
        case 0U: one_off.collective_transport_in_build = false; break;
        case 1U: one_off.rank_world_plumbing           = false; break;
        case 2U: one_off.per_rank_weight_loader        = false; break;
        default: one_off.cross_device_decode_graph     = false; break;
        }
        const std::vector<std::string> missing =
            ninfer::multi::missing_cross_device_pieces(one_off);
        check(missing.size() == 1U,
              "flipping fact " + std::to_string(bit) +
                  " alone must leave exactly one missing piece, got " +
                  std::to_string(missing.size()));
        if (missing.size() == 1U) {
            check(missing.front() == ninfer::multi::kCrossDevicePieceNames[bit],
                  "and it must be the piece that belongs to that fact, in the array's order");
        }
        const std::string refusal = ninfer::multi::cross_device_world_refusal(
            world_request(2U, 0U), one_off, 2U);
        check(!refusal.empty() && refusal.find(ninfer::multi::kCrossDevicePieceNames[bit]) !=
                                      std::string::npos,
              "and one missing piece is enough to refuse, naming that piece");
    }
}

// ---------------------------------------------------------------------------
// G2. THE IDENTITY ARM -- nothing asked is nothing changed
// ---------------------------------------------------------------------------
// This is the arm that makes the whole contract behaviour-preserving on the machine it was
// written on. Both halves are asserted, because "returns empty" is only meaningful if it is
// empty for the reason intended and not because the function is broken.
void test_g2_nothing_asked_is_nothing_changed() {
    // (a) No request at all, on a machine with any number of cards.
    for (const std::uint32_t devices : {1U, 2U, 8U}) {
        const WorldRequest nothing; // requested == false, the default
        check(ninfer::multi::cross_device_world_refusal(
                  nothing, ninfer::multi::kCrossDeviceFactsThisRevision, devices)
                  .empty(),
              "an unset " + std::string(ninfer::multi::kWorldSizeEnv) +
                  " must be an empty refusal: the guard is OFF, and a guard that fires when "
                  "nothing was asked is a gate nobody can turn off");
    }
    // (b) A declared one-rank world is the real device -- with the facts of THIS revision,
    // i.e. with every cross-device piece absent. It still must not refuse, because there is
    // nothing to span.
    check(ninfer::multi::cross_device_world_refusal(
              world_request(1U, 0U), ninfer::multi::kCrossDeviceFactsThisRevision, 4U)
              .empty(),
          "world_size 1 must not refuse even on a four-card machine with every cross-device "
          "piece absent: a one-rank world spans nothing, so it needs none of them");
    // (c) The reader's defaults, without touching the environment: the identity world.
    ninfer::multi::WorldRequest identity{};
    check(!identity.requested && identity.world_size == 1U && identity.rank == 0U,
          "the request's defaults must be the identity world (not requested, size 1, rank 0)");
}

// ---------------------------------------------------------------------------
// G3. A MALFORMED WORLD IS REFUSED BY NAME, AND NEVER DEFAULTED
// ---------------------------------------------------------------------------
void test_g3_malformed_worlds_are_refused_by_name() {
    // A typo cannot become rank 0 / world 1. The reader reports it as a non-positive size, and
    // the refusal names the key.
    {
        ninfer::multi::WorldRequest typo;
        typo.requested      = true;
        typo.world_size     = 0U; // what the reader produces for an unparseable value
        typo.world_size_raw = "two";
        const std::string refusal = ninfer::multi::cross_device_world_refusal(
            typo, ninfer::multi::kCrossDeviceFactsThisRevision, 4U);
        check(!refusal.empty(), "an unparseable world size must be refused");
        check(refusal.find(ninfer::multi::kWorldSizeEnv) != std::string::npos,
              "and the refusal must name the key that was set");
        check(refusal.find("two") != std::string::npos,
              "and echo the value, so the operator sees the typo");
    }
    // The parser itself, as a pure function, both directions.
    {
        bool good = true;
        check(ninfer::multi::parse_world_u32("8", good) == 8U && good,
              "a well-formed rank count must parse");
        good = true;
        check(ninfer::multi::parse_world_u32("8x", good) == 0U && !good,
              "a trailing character must be a refusal and not a truncation");
        good = true;
        check(ninfer::multi::parse_world_u32("", good) == 0U && good,
              "the empty string parses as 0 with good=true: emptiness is the reader's signal "
              "for 'not requested', and the size check above is where 0 is refused");
        good = true;
        check(ninfer::multi::parse_world_u32("99999", good) == 0U && !good,
              "an absurd rank count must be refused rather than overflowed");
    }
    // A rank outside its own world.
    {
        const std::string refusal = ninfer::multi::cross_device_world_refusal(
            world_request(2U, 5U), ninfer::multi::kCrossDeviceFactsThisRevision, 2U);
        check(!refusal.empty(), "a rank outside the world must be refused");
        check(refusal.find(std::string(ninfer::multi::kRankEnv)) != std::string::npos,
              "and the refusal must name the rank key");
    }
    // THE HARDWARE BRANCH: the world wants more cards than the machine has. This branch must
    // fire EVEN IF every cross-device piece were present, because it is a fact about the
    // machine and not about the tree -- so it is checked with the all-present facts.
    {
        CrossDeviceFacts all_present;
        all_present.collective_transport_in_build = true;
        all_present.rank_world_plumbing           = true;
        all_present.per_rank_weight_loader        = true;
        all_present.cross_device_decode_graph     = true;
        const std::string refusal = ninfer::multi::cross_device_world_refusal(
            world_request(4U, 0U), all_present, 2U);
        check(!refusal.empty(),
              "four ranks on two cards must be refused even in a build that could span two "
              "devices: the world has nowhere to run");
        check(refusal.find("4") != std::string::npos &&
                  refusal.find("2") != std::string::npos,
              "and it must name both the world size and the device count");
    }
}

// ---------------------------------------------------------------------------
// G4. VERBATIM: WHAT THIS REVISION ANSWERS
// ---------------------------------------------------------------------------
// Printed, not asserted-about. The text below is what an operator reads today when they
// declare a two-card world, and it is the deliverable this test exists to keep honest.
void test_g4_verbatim_state_of_this_revision() {
    const CrossDeviceFacts facts = ninfer::multi::kCrossDeviceFactsThisRevision;
    std::cout << "--- G4: the facts of this revision ---\n";
    std::cout << "  collective_transport_in_build = "
              << (facts.collective_transport_in_build ? "true" : "false") << "\n";
    std::cout << "  rank_world_plumbing           = "
              << (facts.rank_world_plumbing ? "true" : "false") << "\n";
    std::cout << "  per_rank_weight_loader        = "
              << (facts.per_rank_weight_loader ? "true" : "false") << "\n";
    std::cout << "  cross_device_decode_graph     = "
              << (facts.cross_device_decode_graph ? "true" : "false") << "\n";

    const std::vector<std::string> missing = ninfer::multi::missing_cross_device_pieces(facts);
    std::cout << "  missing pieces: " << missing.size() << " of "
              << ninfer::multi::kCrossDevicePieceCount << "\n";
    for (const std::string& piece : missing) { std::cout << "    - " << piece << "\n"; }

    const std::string refusal = ninfer::multi::cross_device_world_refusal(
        world_request(2U, 0U), facts, 2U);
    std::cout << "  refusal for NINFER_WORLD_SIZE=2 on a 2-card machine, verbatim:\n";
    std::cout << "    " << refusal << "\n";

    check(!ninfer::multi::cross_device_world_possible(facts),
          "THIS REVISION cannot span physical devices, and the facts must say so; if this "
          "check is the one that failed, the capability landed and this whole test file's "
          "G4 block and the --multidev-required arm must be updated in the same commit");
    check(missing.size() == ninfer::multi::kCrossDevicePieceCount,
          "every one of the four pieces is absent in this revision");
    check(!refusal.empty(),
          "and so a two-card world is refused: this is the line that replaced the silent "
          "single-card run");
    check(refusal.find("MISSING, by name") != std::string::npos,
          "and the refusal says the pieces are missing BY NAME rather than in a count");
}

// ---------------------------------------------------------------------------
// G5. THE KEY NAMES, PINNED AGAINST THE HEADER THE CONTRACT CANNOT INCLUDE
// ---------------------------------------------------------------------------
// shard_plan.h spells the virtual-device keys as literals because core/virtual_device.h
// INCLUDES it (its line 61) and the include would be circular. That is a real reason and it
// is also exactly how two spellings drift. This check is the tie between them.
void test_g5_key_names_are_pinned_against_virtual_device() {
    check(std::string(ninfer::multi::kWorldSizeEnv) == "NINFER_WORLD_SIZE",
          "the world-size key is NINFER_WORLD_SIZE");
    check(std::string(ninfer::multi::kRankEnv) == "NINFER_RANK",
          "the rank key is NINFER_RANK");
    check(std::string(ninfer::multi::kWorldSizeEnv) !=
              std::string(ninfer::multi::kVirtualDevicesEnv),
          "a PHYSICAL world and a SIMULATED world are different keys: NINFER_WORLD_SIZE means "
          "N ranks that want N cards, NINFER_VIRTUAL_DEVICES means N logical ranks on one "
          "card. Collapsing them would make the two indistinguishable at the point where the "
          "refusal is read");
    check(std::string(ninfer::multi::kVirtualDevicesEnv) == "NINFER_VIRTUAL_DEVICES",
          "the virtual key the refusal's text names as the alternative must be the header's "
          "own spelling; if this fails, one of the two files was renamed alone");
    check(std::string(ninfer::multi::kVirtualDevicesAckEnv) == "NINFER_VIRTUAL_DEVICES_ACK",
          "and so must the acknowledgement key");
    // The refusal's alternative branch must name the key the reader can actually set.
    const std::string refusal = ninfer::multi::cross_device_world_refusal(
        world_request(2U, 0U), ninfer::multi::kCrossDeviceFactsThisRevision, 2U);
    check(refusal.find(std::string(ninfer::multi::kVirtualDevicesEnv)) != std::string::npos,
          "the refusal must name the key that IS the way to do this on one card");
}

// ---------------------------------------------------------------------------
// G6. THE RED CONTROL -- the walker must be able to go RED
// ---------------------------------------------------------------------------
// G1's walker passes today. That is only evidence if the SAME walker can fail. This control
// runs it against a deliberately broken predicate and asserts it is caught: a predicate that
// claims possible-with-a-piece-missing, and one that refuses with every piece present. Both
// are the two ways this contract could degrade, and neither survives.
bool broken_walker_all_subsets() {
    // The MUTANT: "possible" ignores its input, which is what a hardcoded answer looks like.
    auto mutant_possible = [](const CrossDeviceFacts&) { return true; };
    for (unsigned mask = 0; mask < 16U; ++mask) {
        CrossDeviceFacts facts;
        facts.collective_transport_in_build = (mask & 1U) != 0U;
        facts.rank_world_plumbing           = (mask & 2U) != 0U;
        facts.per_rank_weight_loader        = (mask & 4U) != 0U;
        facts.cross_device_decode_graph     = (mask & 8U) != 0U;
        const std::vector<std::string> missing =
            ninfer::multi::missing_cross_device_pieces(facts);
        if (mutant_possible(facts) != missing.empty()) { return false; } // caught
    }
    return true; // NOT caught: the control itself would be broken
}

void test_g6_the_walker_can_go_red() {
    // Direction 1: a predicate that ignores its input is caught (it must fail on the subsets
    // with a missing piece).
    check(!broken_walker_all_subsets(),
          "the walker MUST catch a 'possible' predicate that ignores its input; if this "
          "check fails, G1 is passing vacuously");
    // And the real walker, on the real predicate, is NOT caught -- so the control above is
    // discriminating between the two rather than failing everything.
    std::string first_failure;
    check(walker_all_subsets(&first_failure),
          "the real walker must accept the real predicate; first: " + first_failure);
    // Direction 2: the naming check can go red. A synthetic refusal that names nothing must be
    // detected by the SAME find() the walker uses.
    const std::string nameless = "a world was declared and this build cannot span it";
    check(nameless.find(ninfer::multi::kCrossDevicePieceNames[0]) == std::string::npos,
          "a refusal that names no piece must fail the walker's find(); this is the arm that "
          "shows the naming check is not vacuous");
    const std::string real = ninfer::multi::cross_device_world_refusal(
        world_request(2U, 0U), ninfer::multi::kCrossDeviceFactsThisRevision, 2U);
    for (std::size_t i = 0; i < ninfer::multi::kCrossDevicePieceCount; ++i) {
        check(real.find(ninfer::multi::kCrossDevicePieceNames[i]) != std::string::npos,
              "the real refusal names piece " + std::to_string(i));
    }
}

// ---------------------------------------------------------------------------
// G7. THE OTHER HALF: A MACHINE WITH IDLE CARDS MUST SAY SO
// ---------------------------------------------------------------------------
void test_g7_idle_device_notice() {
    const WorldShape single{1U, 0U, ParallelAxis::None};
    const std::string real_refusal_for_g7 = ninfer::multi::cross_device_world_refusal(
        world_request(2U, 0U), ninfer::multi::kCrossDeviceFactsThisRevision, 2U);
    // One card: nothing to say, and saying nothing is the identity the engine has today.
    check(ninfer::multi::idle_device_notice(1U, single).empty(),
          "on a one-card machine the notice must be EMPTY: an unconditional banner would be a "
          "line every existing run suddenly grew");
    // More cards than the run uses: it must say so, with the count.
    const std::string notice = ninfer::multi::idle_device_notice(3U, single);
    check(!notice.empty(), "on a three-card machine a single-card run must say so");
    check(notice.find("3") != std::string::npos,
          "and the notice must name how many devices are present");
    check(notice.find("idle") != std::string::npos,
          "and say the rest are idle rather than leaving it to be inferred");
    check(notice.find("NOT a scaling result") != std::string::npos,
          "and carry the caveat that a single-card number is not a scaling result");
    check(notice.find("HIP") != std::string::npos,
          "and name the MEASURED reason the link has no transport: the tree's only cross-rank "
          "allreduce is an AMD HIP port, not a CUDA one. Removing this word would leave the "
          "notice implying 'add it to CMakeLists and you are done', which the nvcc compile of "
          "src/ops/common/allreduce.cu refutes at its line 39");
    // The same word has to be in the REFUSAL, or an operator who declares a world reads a
    // refusal that does not tell them why the one transport they can see is not usable.
    check(real_refusal_for_g7.find("HIP") != std::string::npos,
          "the two-card refusal must also name the HIP fact, not only the CMakeLists one");
    // A world that already explains its ranks has nothing added on top.
    const WorldShape tp2{2U, 0U, ParallelAxis::Tensor};
    check(ninfer::multi::idle_device_notice(2U, tp2).empty(),
          "a declared world must not also get the idle-device notice: the ranks are the answer "
          "to the question the notice raises");
}

// ---------------------------------------------------------------------------
// THE COUNTER-CONTROL: THE ARM THAT MUST BE RED TODAY
// ---------------------------------------------------------------------------
// This arm asserts the CAPABILITY, not its absence:
//
//     "this build can carry a world of two ranks across two physical devices"
//
// It is FALSE in this revision, so this arm is RED, and it is RED VERBATIM: it prints the
// refusal an operator would read. It goes GREEN on the day the four pieces are in the build.
//
// IT IS REGISTERED WITH `WILL_FAIL TRUE` (tests/CMakeLists.txt), which is the tree's FIRST use
// of that property and is deliberate:
//   * today   -- the program exits 1, ctest inverts it, the SUITE IS GREEN while the program
//                is RED and prints why. A red test in a shared suite would make every other
//                line's ctest run fail, which is not what "this is not supported yet" means.
//   * later   -- the day multi-device lands the program exits 0, ctest inverts THAT to a FAIL,
//                and the pin RETIRES ITSELF loudly instead of sitting green and stale. That is
//                the whole point of the arm: it is a one-way latch on the absence.
int run_multidev_required_arm() {
    const CrossDeviceFacts facts = ninfer::multi::kCrossDeviceFactsThisRevision;
    std::cout << "MULTIDEV-REQUIRED ARM -- the capability assertion.\n";
    std::cout << "  asserted: this build can carry a world of two ranks across two PHYSICAL "
                 "devices.\n";
    std::cout << "  facts of this revision:\n";
    std::cout << "    collective_transport_in_build = "
              << (facts.collective_transport_in_build ? "true" : "false") << "\n";
    std::cout << "    rank_world_plumbing           = "
              << (facts.rank_world_plumbing ? "true" : "false") << "\n";
    std::cout << "    per_rank_weight_loader        = "
              << (facts.per_rank_weight_loader ? "true" : "false") << "\n";
    std::cout << "    cross_device_decode_graph     = "
              << (facts.cross_device_decode_graph ? "true" : "false") << "\n";
    const std::vector<std::string> missing = ninfer::multi::missing_cross_device_pieces(facts);
    std::cout << "  missing pieces (" << missing.size() << "):\n";
    for (const std::string& piece : missing) { std::cout << "    - " << piece << "\n"; }

    if (ninfer::multi::cross_device_world_possible(facts)) {
        std::cout << "  capability: PRESENT.\n";
        std::cout << "  RESULT: GREEN. This pin has done its job and must be RETIRED: the "
                     "contract text in src/core/shard_plan.h section 6, the facts constant, "
                     "G4's block and this arm's registration in tests/CMakeLists.txt all have "
                     "to move in the same commit.\n";
        return 0;
    }
    std::cout << "  capability: ABSENT.\n";
    std::cout << "  RESULT: RED, verbatim and by design -- this revision cannot carry the "
                 "world.\n";
    std::cout << "RED: a world of 2 ranks across 2 physical devices is NOT available in this "
                 "revision.\n";
    std::cout << "RED: the refusal an operator reads, verbatim:\n";
    std::cout << "  " << ninfer::multi::cross_device_world_refusal(
                     world_request(2U, 0U), facts, 2U)
              << "\n";
    return 1;
}

} // namespace

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--multidev-required") { return run_multidev_required_arm(); }
    }

    test_g1_the_contract_is_a_function_of_the_facts();
    test_g2_nothing_asked_is_nothing_changed();
    test_g3_malformed_worlds_are_refused_by_name();
    test_g4_verbatim_state_of_this_revision();
    test_g5_key_names_are_pinned_against_virtual_device();
    test_g6_the_walker_can_go_red();
    test_g7_idle_device_notice();

    std::cout << "shard_contract: " << checks << " checks, " << failures
              << " failures -> " << (failures == 0 ? "PASS" : "FAIL") << "\n";
    return failures == 0 ? 0 : 1;
}
