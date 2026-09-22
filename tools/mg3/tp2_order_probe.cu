// tools/mg3/tp2_order_probe.cu -- WHY the tp2 self-check's phase 1 goes red, isolated into one
// experiment with a wrong control on each side.
//
// The self-check (tools/mgpu2/tp2_selfcheck.cu, source 481d971d4161dba4) executed the real
// collective and FAILED "the in-place sum equals a + b elementwise" at both sizes, while
// allgather_rows, the AND-reduce and BOTH capture checks passed. A red with no mechanism is not a
// finding, so this probe isolates the mechanism instead of re-running the same red.
//
// THE HYPOTHESIS, and where it comes from. `allreduce_sum_2rank()` is ONE RANK'S WHOLE CALL: five
// steps (record inputs_ready, wait peer's, pull, record pull_done, wait peer's) and then the
// caller's local combine. The self-check emulates two ranks on ONE host thread by issuing rank 0's
// entire call and then rank 1's entire call. But `cudaStreamWaitEvent` snapshots the event's state
// AT THE TIME OF THE CALL -- from the CUDA docs: "If cudaEventRecord() has not been called on
// event, this call acts as if the record has already completed", and a wait captures the most
// recent record as of the call. So in that order:
//   * rank 0's wait on rank 1's `inputs_ready` is a NO-OP (rank 1 has not recorded it yet), and its
//     wait on rank 1's `pull_done` is a no-op too;
//   * in the second repetition those two waits snapshot rank 0's *first*-repetition records, i.e.
//     they are one repetition STALE.
// Nothing therefore orders rank 0's IN-PLACE combine against rank 1's read of rank 0's source
// buffer. The seam's own claim is that the second event pair is the write-after-read barrier --
// and that claim is a HOST-ISSUE-ORDER property, not a device property.
//
// So the experiment has four arms, and the two failing ones fail for DIFFERENT reasons, which is
// what makes the passing arm evidence rather than luck:
//
//   ARM A  call-by-call, no host sync between the ranks.  Expect FAIL.
//          (the self-check's own order, reproduced)
//   ARM B  call-by-call, a full host sync between the two ranks' calls.  Expect FAIL, and for the
//          OTHER reason: serialising the calls puts rank 0's in-place combine BEFORE rank 1's read
//          of rank 0's source, so rank 1 sums a value that has already been combined. This is the
//          failure that shows "add a barrier" is not the fix.
//   ARM C  FORK/JOIN: both ranks record, both wait, both PULL, both record pull_done, both wait,
//          and only then does either combine.  Expect PASS. This is the shape a per-rank-thread
//          host produces naturally, and it is the only one of the three that is correct.
//   ARM D  arm C with the pull_done pair REMOVED.  Expect FAIL -- the red control that proves
//          arm C's pass is not vacuous and that the second event pair is load-bearing.
//
// The failing arms also print the first mismatching element and its value, because the SHAPE of the
// corruption is the evidence: a wrong-order sum gives exactly `a + 2b` or `2a + b`, whereas a torn
// read gives an arbitrary value. Those are different bugs and this probe's job is to say which.
//
// Build: nvcc -std=c++20 -O2 -arch=sm_120 -I<tree>/src -I<tree> tools/mg3/tp2_order_probe.cu \
//          -o tp2_order_probe

#include "core/tp_transport.h"
#include "core/virtual_device.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
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

__global__ void add_in_place(float* a, const float* b, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { a[i] += b[i]; }
}

enum class Arm { CallByCall, CallByCallSynced, ForkJoin, ForkJoinNoBarrier };

const char* arm_name(Arm a) {
    switch (a) {
    case Arm::CallByCall: return "A call-by-call, no sync between ranks (the self-check's order)";
    case Arm::CallByCallSynced: return "B call-by-call, host sync BETWEEN the two ranks";
    case Arm::ForkJoin: return "C fork/join: both pulls, then both combines";
    case Arm::ForkJoinNoBarrier: return "D fork/join with the pull_done pair REMOVED (red control)";
    }
    return "?";
}

struct ArmResult {
    bool values_ok = false;
    bool bitwise_equal = false;
    std::string note;
    // The first mismatch, so the SHAPE of the corruption is reported rather than only its presence.
    int first_bad_rank = -1;
    int first_bad_index = -1;
    float first_bad_got = 0.0f;
    float first_bad_want = 0.0f;
};

ArmResult run_arm(int elements, Arm arm) {
    ArmResult r;
    const std::size_t bytes = static_cast<std::size_t>(elements) * sizeof(float);
    const int blocks = (elements + 255) / 256;

    std::vector<float> host_a(elements), host_b(elements);
    for (int i = 0; i < elements; ++i) {
        host_a[static_cast<std::size_t>(i)] = 1.0f + static_cast<float>(i % 7);
        host_b[static_cast<std::size_t>(i)] = 0.5f + static_cast<float>(i % 5);
    }
    void* self[2] = {nullptr, nullptr};
    void* stage[2] = {nullptr, nullptr};
    for (int rank = 0; rank < 2; ++rank) {
        cudaMalloc(&self[rank], bytes);
        cudaMalloc(&stage[rank], bytes);
    }
    cudaMemcpy(self[0], host_a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(self[1], host_b.data(), bytes, cudaMemcpyHostToDevice);
    cudaStream_t s[2] = {nullptr, nullptr};
    for (int rank = 0; rank < 2; ++rank) { cudaStreamCreateWithFlags(&s[rank], cudaStreamNonBlocking); }
    PeerEvents ev;
    ev.create();

    const bool barrier = (arm != Arm::ForkJoinNoBarrier);

    if (arm == Arm::CallByCall || arm == Arm::CallByCallSynced) {
        for (int rep = 0; rep < 2; ++rep) {
            for (int rank = 0; rank < 2; ++rank) {
                const cudaError_t err =
                    allreduce_sum_2rank(stage[rank], self[1 - rank], bytes, rank, s[rank], ev);
                if (err != cudaSuccess) { r.note = std::string("call failed: ") + cudaGetErrorName(err); break; }
                if (arm == Arm::CallByCallSynced) {
                    // The "just add a barrier" fix: finish this rank completely before the other.
                    cudaStreamSynchronize(s[rank]);
                }
                add_in_place<<<blocks, 256, 0, s[rank]>>>(
                    static_cast<float*>(self[rank]), static_cast<float*>(stage[rank]), elements);
                if (arm == Arm::CallByCallSynced) { cudaStreamSynchronize(s[rank]); }
            }
            for (int rank = 0; rank < 2; ++rank) { cudaStreamSynchronize(s[rank]); }
        }
    } else {
        // FORK/JOIN, composed from the seam's OWN primitives in the order a per-rank-thread host
        // lets them land. `allreduce_sum_2rank` cannot be used here: it is one rank's whole call,
        // which is exactly why a single-threaded two-rank emulation cannot be built out of it.
        for (int rep = 0; rep < 2; ++rep) {
            for (int rank = 0; rank < 2; ++rank) { cudaEventRecord(ev.inputs_ready[rank], s[rank]); }
            for (int rank = 0; rank < 2; ++rank) {
                cudaStreamWaitEvent(s[rank], ev.inputs_ready[1 - rank], 0);
            }
            // THE FORK: both reads of the peer's source, before either rank writes its own.
            for (int rank = 0; rank < 2; ++rank) {
                const cudaError_t err = pull_peer(stage[rank], self[1 - rank], bytes, s[rank]);
                if (err != cudaSuccess) { r.note = std::string("pull failed: ") + cudaGetErrorName(err); }
            }
            if (barrier) {
                for (int rank = 0; rank < 2; ++rank) { cudaEventRecord(ev.pull_done[rank], s[rank]); }
                for (int rank = 0; rank < 2; ++rank) {
                    cudaStreamWaitEvent(s[rank], ev.pull_done[1 - rank], 0);
                }
            }
            // THE JOIN: only now may either rank write its own source.
            for (int rank = 0; rank < 2; ++rank) {
                add_in_place<<<blocks, 256, 0, s[rank]>>>(
                    static_cast<float*>(self[rank]), static_cast<float*>(stage[rank]), elements);
            }
            for (int rank = 0; rank < 2; ++rank) { cudaStreamSynchronize(s[rank]); }
        }
    }

    if (r.note.empty()) {
        std::vector<float> got[2] = {std::vector<float>(elements), std::vector<float>(elements)};
        for (int rank = 0; rank < 2; ++rank) {
            cudaMemcpy(got[rank].data(), self[rank], bytes, cudaMemcpyDeviceToHost);
        }
        r.values_ok = true;
        for (int rank = 0; rank < 2 && r.values_ok; ++rank) {
            for (int i = 0; i < elements; ++i) {
                const float want = host_a[static_cast<std::size_t>(i)] + host_b[static_cast<std::size_t>(i)];
                if (got[rank][static_cast<std::size_t>(i)] != want) {
                    r.values_ok = false;
                    r.first_bad_rank = rank;
                    r.first_bad_index = i;
                    r.first_bad_got = got[rank][static_cast<std::size_t>(i)];
                    r.first_bad_want = want;
                    break;
                }
            }
        }
        r.bitwise_equal = r.values_ok &&
                          (std::memcmp(got[0].data(), got[1].data(), bytes) == 0);
    }

    for (int rank = 0; rank < 2; ++rank) { cudaStreamDestroy(s[rank]); cudaFree(self[rank]); cudaFree(stage[rank]); }
    ev.destroy();
    return r;
}

// What the reference value would be under the two WRONG-ORDER mechanisms, so the shape of a
// mismatch can be named instead of described.
void report_shape(const ArmResult& r, const std::vector<float>& a, const std::vector<float>& b) {
    if (r.first_bad_index < 0) { return; }
    const std::size_t i = static_cast<std::size_t>(r.first_bad_index);
    const float wrong_a_plus_2b = a[i] + 2.0f * b[i];
    const float wrong_2a_plus_b = 2.0f * a[i] + b[i];
    std::printf("      first mismatch: rank %d elem %d  got %.6f  want %.6f\n", r.first_bad_rank,
                r.first_bad_index, static_cast<double>(r.first_bad_got),
                static_cast<double>(r.first_bad_want));
    std::printf("      a=%.6f b=%.6f | a+2b=%.6f (%.1f%% off the got value) 2a+b=%.6f (%.1f%% off)\n",
                static_cast<double>(a[i]), static_cast<double>(b[i]),
                static_cast<double>(wrong_a_plus_2b),
                100.0 * static_cast<double>(std::abs(wrong_a_plus_2b - r.first_bad_got)) /
                    static_cast<double>(std::abs(wrong_a_plus_2b) + 1e-9),
                static_cast<double>(wrong_2a_plus_b),
                100.0 * static_cast<double>(std::abs(wrong_2a_plus_b - r.first_bad_got)) /
                    static_cast<double>(std::abs(wrong_2a_plus_b) + 1e-9));
}

} // namespace

int main() {
    std::printf("=== tp2 ORDER probe: why phase 1 of the self-check goes red (ONE device) ===\n");
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    int driver = 0, runtime = 0;
    cudaDriverGetVersion(&driver);
    cudaRuntimeGetVersion(&runtime);
    std::printf("device : %s sm_%d%d  |  driver %d / runtime %d\n\n", prop.name, prop.major,
                prop.minor, driver, runtime);

    bool probe_ok = false;
    const PeerAccessMatrix matrix = probe_peer_access(1, &probe_ok);
    const TopologyReport topology = decide_topology(matrix, 2, probe_ok);
    std::printf("topology: %s\n\n", std::string(topology_decision_name(topology.decision)).c_str());

    const int elements = 4096;
    std::vector<float> a(elements), b(elements);
    for (int i = 0; i < elements; ++i) {
        a[static_cast<std::size_t>(i)] = 1.0f + static_cast<float>(i % 7);
        b[static_cast<std::size_t>(i)] = 0.5f + static_cast<float>(i % 5);
    }

    const Arm order[4] = {Arm::CallByCall, Arm::CallByCallSynced, Arm::ForkJoin,
                          Arm::ForkJoinNoBarrier};
    for (const Arm arm : order) {
        std::printf("--- ARM %s\n", arm_name(arm));
        const ArmResult r = run_arm(elements, arm);
        if (!r.note.empty()) {
            std::printf("      did not run: %s\n\n", r.note.c_str());
            check(false, std::string("arm ran: ") + r.note);
            continue;
        }
        std::printf("      values elementwise: %s | ranks bitwise equal: %s\n",
                    r.values_ok ? "CORRECT" : "WRONG", r.bitwise_equal ? "yes" : "no");
        if (!r.values_ok) { report_shape(r, a, b); }
        std::printf("\n");
        // The EXPECTATIONS are the experiment: A and B must fail, C must pass, D must fail.
        if (arm == Arm::CallByCall) {
            check(!r.values_ok,
                  "A fails as the self-check did -- the order is reproduced, so the mechanism "
                  "under test is the one that produced the red");
            check(!r.bitwise_equal, "A: the two ranks also disagree bitwise");
        } else if (arm == Arm::CallByCallSynced) {
            check(!r.values_ok,
                  "B STILL fails with a host barrier between the ranks: 'add a barrier' is not the "
                  "fix, because serialising the calls puts the combine before the peer's read");
        } else if (arm == Arm::ForkJoin) {
            check(r.values_ok, "C is CORRECT: both pulls complete before either combine");
            check(r.bitwise_equal, "C: the two ranks agree bitwise (the reference's own gate)");
            check(cudaGetLastError() == cudaSuccess, "C: no sticky CUDA error");
        } else {
            check(!r.values_ok,
                  "D RED CONTROL fails: without the pull_done pair the JOIN is unordered, so "
                  "arm C's pass is not vacuous and the second event pair is load-bearing");
        }
    }

    std::printf("CONCLUSION, as the four arms force it:\n"
                "  the in-place sum's correctness depends on BOTH pulls completing before EITHER\n"
                "  combine, and on each rank's write-after-read wait being issued AFTER the peer's\n"
                "  record. `allreduce_sum_2rank()` is one rank's whole call, so a single host thread\n"
                "  cannot interleave two ranks correctly with it -- the seam's per-rank call API is\n"
                "  usable from PER-RANK THREADS (the donor's shape: a thread per GPU) or from a\n"
                "  fork/join composition like arm C, and from nothing else.\n"
                "  ⚠️ On TWO devices with one host thread the SAME hazard exists and is worse: the\n"
                "  cross-device form additionally needs the events to be visible to the peer,\n"
                "  which per-process `PeerEvents` are not (1Cat IPC-maps its signal slots and uses\n"
                "  st.release.sys / ld.acquire.sys for exactly this reason). PENDING HARDWARE.\n\n");

    std::printf("tp2_order_probe: %d checks, %d failures -> %s\n", g_checks, g_failures,
                g_failures == 0 ? "PASS" : "FAIL");
    return g_failures == 0 ? 0 : 1;
}
