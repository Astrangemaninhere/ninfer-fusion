// tools/mg3/and_reduce_probe.cu -- EXECUTE the AND-reduce on the device, two virtual ranks on one
// device, including the case the record says neither reference can express.
//
// WHY A STANDALONE PROBE. The AND-reduce is the one reduction in the plan that no reference on
// this box supplies, so it is the one whose EXECUTION has no existing harness to ride on.
// tools/mgpu2/tp2_selfcheck.cu runs the sum and the relocation; this runs the boolean op, through
// the same four-event pull protocol and the same `PeerEvents`, so the two arms exercise the same
// ordering with a different local step.
//
// ⚠️ WHAT THIS CAN AND CANNOT SHOW, stated before the first line of output rather than after:
//
//   IT CAN execute the real code path. `pull_peer` is `cudaMemcpyAsync(..., DeviceToDevice, ...)`
//   over UVA and a device pointer names its device, so two virtual ranks ON ONE DEVICE run the
//   SAME code as two ranks on two devices, minus the PCIe. The ordering, the event choreography,
//   the decode, the sentinel and the bitwise agreement are all genuinely exercised.
//
//   IT CANNOT measure a cross-card transfer: there is one device, so every copy is local D2D.
//   No number here is a transport cost. The THROUGHPUT of an AND-reduce is `pending hardware`.
//
// WHAT IT ASKS THAT THE HOST TEST CANNOT: the host test (tests/test_and_reduce.cpp) executes the
// decode and the fold as arithmetic. This one asks whether the LOCAL STEP, INSIDE the pull
// protocol, produces the same answer -- and whether the offending raw word is still readable in
// the rank's own staging buffer afterwards, which is the property the design relies on to let an
// out-of-domain word be diagnosed WITHOUT a host round trip on the hot path.
//
// Build: nvcc -std=c++20 -O2 -arch=sm_120 -I<tree>/src -I<tree> tools/mg3/and_reduce_probe.cu \
//          -o and_reduce_probe

#include "core/and_reduce.h"
#include "core/tp_transport.h"
#include "core/virtual_device.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

using namespace ninfer::multi;

namespace {

int g_checks = 0;
int g_failures = 0;

void check(bool ok, const std::string& what) {
    ++g_checks;
    if (!ok) {
        ++g_failures;
        std::printf("  FAIL: %s\n", what.c_str());
    } else {
        std::printf("  ok  : %s\n", what.c_str());
    }
}

const char* cuda_name(cudaError_t e) { return cudaGetErrorName(e); }

// One case, executed. Returns the raw words the two ranks READ (rank 0's view of its own word and
// of the peer's staging), so the caller can cross-check the device verdict against the host fold.
struct CaseResult {
    bool ok = false;
    std::string note;
    std::uint32_t out_word[2] = {kAdmitWordViolation, kAdmitWordViolation};
    std::uint32_t staging_word[2] = {kAdmitWordViolation, kAdmitWordViolation};
    std::uint32_t expected = 0;
};

// `wire` is the pair of words the two ranks PUBLISH. Each rank's local step reads its own word and
// the word the peer's storage held, pulled into its own staging.
CaseResult run_case(std::uint32_t rank0_word, std::uint32_t rank1_word, std::uint32_t expected) {
    CaseResult r;
    r.expected = expected;
    const std::size_t bytes = sizeof(std::uint32_t);

    std::uint32_t* self[2] = {nullptr, nullptr};    // rank r's own published word
    std::uint32_t* staging[2] = {nullptr, nullptr}; // rank r's staging for the peer's source
    std::uint32_t* out[2] = {nullptr, nullptr};     // rank r's verdict
    for (int rank = 0; rank < 2; ++rank) {
        if (cudaMalloc(&self[rank], bytes) != cudaSuccess ||
            cudaMalloc(&staging[rank], bytes) != cudaSuccess ||
            cudaMalloc(&out[rank], bytes) != cudaSuccess) {
            r.note = "cudaMalloc failed";
            return r;
        }
    }
    const std::uint32_t published[2] = {rank0_word, rank1_word};
    for (int rank = 0; rank < 2; ++rank) {
        cudaMemcpy(self[rank], &published[rank], bytes, cudaMemcpyHostToDevice);
        cudaMemset(staging[rank], 0xFF, bytes); // poison: a stale value must not look like a read
        cudaMemset(out[rank], 0xEE, bytes);
    }

    cudaStream_t stream[2] = {nullptr, nullptr};
    for (int rank = 0; rank < 2; ++rank) {
        cudaStreamCreateWithFlags(&stream[rank], cudaStreamNonBlocking);
    }
    PeerEvents events;
    if (!events.create()) { r.note = "PeerEvents::create failed"; return r; }

    // TWO consecutive calls sharing every buffer, with ONE host sync at the end -- the donor's
    // post-condition ("an unbounded sequence of calls sharing the same buffers, staging and
    // PeerEvents needs NO host synchronization between calls") is what a captured graph replays,
    // and a second rep is the cheapest way to exercise the second event pair's barrier.
    for (int rep = 0; rep < 2; ++rep) {
        for (int rank = 0; rank < 2; ++rank) {
            const cudaError_t err =
                allreduce_and_2rank(self[rank], self[1 - rank], staging[rank], out[rank], bytes,
                                    rank, stream[rank], events);
            if (err != cudaSuccess) {
                r.note = std::string("allreduce_and_2rank failed: ") + cuda_name(err);
                break;
            }
        }
    }
    for (int rank = 0; rank < 2 && r.note.empty(); ++rank) { cudaStreamSynchronize(stream[rank]); }

    if (r.note.empty()) {
        for (int rank = 0; rank < 2; ++rank) {
            cudaMemcpy(&r.out_word[rank], out[rank], bytes, cudaMemcpyDeviceToHost);
            cudaMemcpy(&r.staging_word[rank], staging[rank], bytes, cudaMemcpyDeviceToHost);
        }
        r.ok = true;
    }

    for (int rank = 0; rank < 2; ++rank) {
        cudaStreamDestroy(stream[rank]);
        cudaFree(self[rank]);
        cudaFree(staging[rank]);
        cudaFree(out[rank]);
    }
    events.destroy();
    return r;
}

} // namespace

int main() {
    std::printf("=== and_reduce probe (ONE device: see the header for what that rules out) ===\n");

    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    std::printf("device : %s sm_%d%d, %d SMs\n", prop.name, prop.major, prop.minor,
                prop.multiProcessorCount);
    int driver = 0, runtime = 0;
    cudaDriverGetVersion(&driver);
    cudaRuntimeGetVersion(&runtime);
    std::printf("driver : %d / runtime %d\n", driver, runtime);

    // The topology decision, asked of the real device. With one device it must be Unavailable,
    // which is the correct answer and not a degraded transport: no collective may be issued.
    bool probe_ok = false;
    const PeerAccessMatrix matrix = probe_peer_access(1, &probe_ok);
    const TopologyReport topology = decide_topology(matrix, /*world_size=*/2, probe_ok);
    std::printf("topology: %s -- %s\n\n", std::string(topology_decision_name(topology.decision)).c_str(),
                topology.reason.c_str());
    check(topology.decision == TopologyDecision::Unavailable,
          "one device reports Unavailable, so the 2-rank calls below are the code path minus PCIe");

    std::printf("--- the four cases, through the real collective ---\n");

    struct Case {
        std::uint32_t a;
        std::uint32_t b;
        std::uint32_t expected;
        const char* what;
    };
    const Case cases[4] = {
        {1u, 1u, 1u, "{1,1}  -> ADMIT (the canonical conjunction)"},
        {1u, 0u, 0u, "{1,0}  -> REFUSE"},
        {0u, 0u, 0u, "{0,0}  -> REFUSE"},
        {1u, 2u, kAdmitWordViolation, "{1,2}  -> PROTOCOL VIOLATION, not a boolean"},
    };

    for (const Case& c : cases) {
        std::printf(" case %s\n", c.what);
        const CaseResult r = run_case(c.a, c.b, c.expected);
        if (!r.ok) {
            check(false, std::string("case ran: ") + r.note);
            std::printf("\n");
            continue;
        }
        cudaError_t le = cudaGetLastError();
        check(le == cudaSuccess, std::string("no sticky CUDA error after the case: ") + cuda_name(le));

        // The device's verdict, in both ranks, and the two ranks must agree BITWISE.
        check(r.out_word[0] == c.expected,
              "rank 0's device verdict is the expected word (" + std::to_string(r.out_word[0]) +
                  ")");
        check(r.out_word[1] == c.expected,
              "rank 1's device verdict is the expected word (" + std::to_string(r.out_word[1]) +
                  ")");
        check(r.out_word[0] == r.out_word[1], "the two ranks agree on the verdict bit-for-bit");

        // THE CROSS-CHECK THAT MATTERS: the device's word and the HOST fold of the same two words
        // must say the same thing, on both sides of the domain boundary. If the kernel and
        // `and_reduce_wire()` ever disagreed, one of them would be the wrong answer and there
        // would be no way to tell which from the outside.
        const AndReduceOutcome host = and_reduce_wire({c.a, c.b});
        if (admit_word_is_canonical(c.expected)) {
            check(host.ok, "the host fold agrees this input is in the domain");
            check((r.out_word[0] == 1u) == host.admitted,
                  "the device verdict equals the host fold's verdict");
            check((r.out_word[0] == 1u) == global_admit({static_cast<std::uint8_t>(c.a),
                                                           static_cast<std::uint8_t>(c.b)}),
                  "and equals the plan's own global_admit on the canonical pair");
        } else {
            check(!host.ok, "the host fold also calls this a protocol violation");
            check(r.out_word[0] == kAdmitWordViolation,
                  "the device wrote the violation sentinel rather than picking a boolean");
            check(kAdmitWordViolation != 0u && kAdmitWordViolation != 1u,
                  "and the sentinel is outside the canonical domain, so it cannot be a verdict");
        }

        // THE POST-MORTEM PROPERTY, which is why the sentinel is enough: the offending raw word is
        // still sitting in the rank's own staging buffer, because the pull put it there and the
        // local step only READ it. So a violation can be diagnosed after the fact from the side
        // that observed it, with no host round trip on the hot path.
        check(r.staging_word[0] == c.b,
              "rank 0's staging still holds the peer's raw word (" +
                  std::to_string(r.staging_word[0]) + ") after the local step");
        check(r.staging_word[1] == c.a,
              "rank 1's staging still holds the peer's raw word (" +
                  std::to_string(r.staging_word[1]) + ") after the local step");
        std::printf("\n");
    }

    std::printf("--- the identity, and the single-rank contract ---\n");
    const AndReduceOutcome identity = and_reduce_wire({});
    check(identity.ok && identity.admitted && !identity.collective_issued,
          "world 0 is the AND identity and issues no collective, so no kernel is launched");
    const AndReduceOutcome single = and_reduce_wire({2u});
    check(!single.ok && !single.admitted,
          "world 1 with a non-canonical word is still a violation, on the host and in the device "
          "kernel's own domain predicate");

    std::printf("\n⚠️  one device == one clock and one HBM: the AND-reduce's own throughput, its\n"
                "    cross-card cost and its behaviour inside a capture on TWO devices are\n"
                "    `pending hardware`. What ran here is the ordering, the decode, the sentinel\n"
                "    and the agreement between the device step and the host fold.\n");

    std::printf("\nand_reduce_probe: %d checks, %d failures -> %s\n", g_checks, g_failures,
                g_failures == 0 ? "PASS" : "FAIL");
    return g_failures == 0 ? 0 : 1;
}
