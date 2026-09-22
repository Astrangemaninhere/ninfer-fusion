// tools/mg3/tp2_order_probe2.cu -- the ORDER probe, with ITS OWN FIRST-PASS ERROR CORRECTED.
//
// WHAT THE FIRST PASS GOT WRONG, stated first because it is the reason this file exists.
// tools/mg3/tp2_order_probe.cu (source 670670a0476edca6) ran four arms with TWO repetitions and
// asserted each rank's buffer against `a + b` -- a ONE-repetition reference. Two all-reduce
// repetitions do not give `a + b`; they give `2(a + b)`. So the arm that was supposed to PASS
// (arm C, fork/join) was reported as FAILING while it was in fact producing exactly the right
// value, and the two "failures" the probe reported were the probe's own miscalibration rather than
// a refutation of the mechanism. The measured numbers, a=1.0 b=0.5:
//
//   ARM A call-by-call          rank0 = 3.5   ARM B call-by-call + sync   rank0 = 3.5
//   ARM C fork/join             rank0 = 3.0   ARM D fork/join, no barrier rank0 = 3.5
//
// and 3.0 = 2(a+b) exactly, while 3.5 = 2a+3b exactly. So arm C was right and the reference was
// wrong, and this file fixes the reference instead of re-asserting the same claim.
//
// WHAT THIS FILE MEASURES, and why the reference is now unambiguous. Each arm is run at BOTH
// repetition counts and the reference is written per count:
//   reps = 1  -> ideal is exactly `a + b`,   and the two wrong-order signatures are exactly
//                `a + 2b` (this rank read the peer AFTER the peer folded in its copy of b) and
//                `2a + b` (the mirror), which are the formulas the first pass used on a 2-rep run.
//   reps = 2  -> ideal is exactly `2(a + b)`, the 1-rep value doubled, so a reader can see the
//                compounding rather than infer it.
// All four arms are run at 1 rep with the correct assertions, and arm C is additionally run at 2
// reps as a second, independent check of the same model.
//
// Build: nvcc -std=c++20 -O2 -arch=sm_120 -I<tree>/src -I<tree> tools/mg3/tp2_order_probe2.cu \
//          -o tp2_order_probe2

#include "core/tp_transport.h"
#include "core/virtual_device.h"

#include <cuda_runtime.h>

#include <cmath>
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
    case Arm::CallByCall: return "A call-by-call, no sync between ranks";
    case Arm::CallByCallSynced: return "B call-by-call, host sync BETWEEN the two ranks";
    case Arm::ForkJoin: return "C fork/join: both pulls, then both combines";
    case Arm::ForkJoinNoBarrier: return "D fork/join with the pull_done pair REMOVED (red control)";
    }
    return "?";
}

struct ArmResult {
    bool ok = false;
    std::string note;
    float got[2] = {0.0f, 0.0f};
};

ArmResult run_arm(int elements, Arm arm, int reps) {
    ArmResult r;
    const std::size_t bytes = static_cast<std::size_t>(elements) * sizeof(float);
    const int blocks = (elements + 255) / 256;

    std::vector<float> ha(elements), hb(elements);
    for (int i = 0; i < elements; ++i) {
        ha[static_cast<std::size_t>(i)] = 1.0f + static_cast<float>(i % 7);
        hb[static_cast<std::size_t>(i)] = 0.5f + static_cast<float>(i % 5);
    }
    void* self[2] = {nullptr, nullptr};
    void* stage[2] = {nullptr, nullptr};
    for (int rank = 0; rank < 2; ++rank) {
        cudaMalloc(&self[rank], bytes);
        cudaMalloc(&stage[rank], bytes);
    }
    cudaMemcpy(self[0], ha.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(self[1], hb.data(), bytes, cudaMemcpyHostToDevice);
    cudaStream_t s[2] = {nullptr, nullptr};
    for (int rank = 0; rank < 2; ++rank) { cudaStreamCreateWithFlags(&s[rank], cudaStreamNonBlocking); }
    PeerEvents ev;
    ev.create();
    const bool barrier = (arm != Arm::ForkJoinNoBarrier);

    if (arm == Arm::CallByCall || arm == Arm::CallByCallSynced) {
        for (int rep = 0; rep < reps; ++rep) {
            for (int rank = 0; rank < 2; ++rank) {
                const cudaError_t err =
                    allreduce_sum_2rank(stage[rank], self[1 - rank], bytes, rank, s[rank], ev);
                if (err != cudaSuccess) { r.note = std::string("call failed: ") + cudaGetErrorName(err); }
                if (arm == Arm::CallByCallSynced) { cudaStreamSynchronize(s[rank]); }
                add_in_place<<<blocks, 256, 0, s[rank]>>>(
                    static_cast<float*>(self[rank]), static_cast<float*>(stage[rank]), elements);
                if (arm == Arm::CallByCallSynced) { cudaStreamSynchronize(s[rank]); }
            }
            for (int rank = 0; rank < 2; ++rank) { cudaStreamSynchronize(s[rank]); }
        }
    } else {
        for (int rep = 0; rep < reps; ++rep) {
            for (int rank = 0; rank < 2; ++rank) { cudaEventRecord(ev.inputs_ready[rank], s[rank]); }
            for (int rank = 0; rank < 2; ++rank) { cudaStreamWaitEvent(s[rank], ev.inputs_ready[1 - rank], 0); }
            for (int rank = 0; rank < 2; ++rank) { pull_peer(stage[rank], self[1 - rank], bytes, s[rank]); }
            if (barrier) {
                for (int rank = 0; rank < 2; ++rank) { cudaEventRecord(ev.pull_done[rank], s[rank]); }
                for (int rank = 0; rank < 2; ++rank) { cudaStreamWaitEvent(s[rank], ev.pull_done[1 - rank], 0); }
            }
            for (int rank = 0; rank < 2; ++rank) {
                add_in_place<<<blocks, 256, 0, s[rank]>>>(
                    static_cast<float*>(self[rank]), static_cast<float*>(stage[rank]), elements);
            }
            for (int rank = 0; rank < 2; ++rank) { cudaStreamSynchronize(s[rank]); }
        }
    }

    if (r.note.empty()) {
        for (int rank = 0; rank < 2; ++rank) {
            cudaMemcpy(&r.got[rank], self[rank], sizeof(float), cudaMemcpyDeviceToHost);
        }
        r.ok = true;
    }
    for (int rank = 0; rank < 2; ++rank) { cudaStreamDestroy(s[rank]); cudaFree(self[rank]); cudaFree(stage[rank]); }
    ev.destroy();
    return r;
}

// The whole model, printed as arithmetic so a reader does not have to trust a verdict.
void report(int reps, const ArmResult& r) {
    if (!r.ok) { std::printf("      did not run: %s\n", r.note.c_str()); return; }
    const double a = 1.0, b = 0.5; // element 0 of the two inputs
    const double ideal_1 = a + b;
    const double read_peer_after_fold = a + std::pow(2.0, reps - 1) * 0.0; // placeholder, unused
    (void)read_peer_after_fold;
    std::printf("      rank0 = %.6f   rank1 = %.6f\n", static_cast<double>(r.got[0]),
                static_cast<double>(r.got[1]));
    std::printf("      for reps=%d: ideal %.6f", reps, std::pow(2.0, reps) * ideal_1);
    if (reps == 1) {
        std::printf("  |  a+2b = %.6f (rank1 read its peer AFTER that peer had folded in b)"
                    "  |  2a+b = %.6f (mirror)", a + 2 * b, 2 * a + b);
    } else {
        std::printf("  |  the 1-rep ideal doubled: %.6f | 2a+3b = %.6f (the wrong-order signature "
                    "carried through two reps)", 2 * ideal_1, 2 * a + 3 * b);
    }
    std::printf("\n");
}

} // namespace

int main() {
    std::printf("=== tp2 ORDER probe v2 (ONE device) -- the first pass's 2-rep/1-rep reference "
                "error corrected ===\n");
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
    const float a = 1.0f, b = 0.5f; // element 0
    const double ideal1 = static_cast<double>(a) + static_cast<double>(b);   // 1.5
    const double ideal2 = 2.0 * ideal1;                                       // 3.0
    const double wrong_a2b = static_cast<double>(a) + 2.0 * static_cast<double>(b); // 2.0
    const double wrong_2ab = 2.0 * static_cast<double>(a) + static_cast<double>(b); // 2.5
    const double wrong_2a3b = 2.0 * static_cast<double>(a) + 3.0 * static_cast<double>(b); // 3.5

    // ---- ONE REPETITION, where the reference is exactly a + b ----
    std::printf("--- reps = 1 (ideal is exactly a + b = %.6f) ---\n", ideal1);
    const Arm arms[4] = {Arm::CallByCall, Arm::CallByCallSynced, Arm::ForkJoin,
                         Arm::ForkJoinNoBarrier};
    double c_value = 0.0;
    for (const Arm arm : arms) {
        std::printf(" ARM %s\n", arm_name(arm));
        const ArmResult r = run_arm(elements, arm, 1);
        report(1, r);
        if (!r.ok) { check(false, std::string("arm ran: ") + r.note); continue; }

        const double v0 = static_cast<double>(r.got[0]);
        const bool exactly_ideal = (v0 == ideal1);
        const bool is_a_plus_2b = (v0 == wrong_a2b);
        const bool is_2a_plus_b = (v0 == wrong_2ab);
        std::printf("      rank0 == a+b exactly: %s | == a+2b: %s | == 2a+b: %s\n\n",
                    exactly_ideal ? "YES" : "no", is_a_plus_2b ? "YES" : "no",
                    is_2a_plus_b ? "YES" : "no");

        if (arm == Arm::CallByCall) {
            check(!exactly_ideal && (is_a_plus_2b || is_2a_plus_b),
                  "A is wrong AND wrong in one of the two predicted ways -- so the mechanism is "
                  "named, not merely 'something is off'");
        } else if (arm == Arm::CallByCallSynced) {
            check(!exactly_ideal && (is_a_plus_2b || is_2a_plus_b),
                  "B is ALSO wrong, and in the same named way: a host barrier between the ranks is "
                  "not the fix, because serialising the calls puts a combine before a read");
        } else if (arm == Arm::ForkJoin) {
            check(exactly_ideal,
                  "C is CORRECT with the reference computed for the right number of repetitions: "
                  "rank0 == a+b exactly");
            check(r.got[0] == r.got[1], "C: the two ranks agree bitwise (the reference's own gate)");
            c_value = v0;
        } else {
            check(!exactly_ideal && (is_a_plus_2b || is_2a_plus_b),
                  "D RED CONTROL is wrong: without the pull_done pair the JOIN is unordered, so "
                  "C's pass is not vacuous and the second event pair is load-bearing");
        }
    }

    // ---- TWO REPETITIONS, the same model at the next power ----
    std::printf("--- reps = 2 (ideal is exactly 2(a+b) = %.6f) ---\n", ideal2);
    std::printf(" ARM %s\n", arm_name(Arm::ForkJoin));
    const ArmResult two = run_arm(elements, Arm::ForkJoin, 2);
    report(2, two);
    if (two.ok) {
        const double v0 = static_cast<double>(two.got[0]);
        std::printf("      rank0 == 2(a+b) exactly: %s | == 2a+3b (wrong-order carried through): %s\n\n",
                    (v0 == ideal2) ? "YES" : "no", (v0 == wrong_2a3b) ? "YES" : "no");
        check(v0 == ideal2, "C at 2 reps is exactly twice its 1-rep value -- the model compounds, "
                            "which is what the first pass read as a failure");
        check(c_value == ideal1 && v0 == ideal2,
              "and the two repetition counts agree with each other, so neither is a coincidence");
        // Arm A at 2 reps is the number the FIRST pass measured, asserted here against the value
        // that pass mis-derived, so the correction is checkable rather than asserted.
        std::printf(" ARM %s\n", arm_name(Arm::CallByCall));
        const ArmResult a2 = run_arm(elements, Arm::CallByCall, 2);
        report(2, a2);
        if (a2.ok) {
            const double v = static_cast<double>(a2.got[0]);
            std::printf("      rank0 == 2a+3b (the wrong-order signature at 2 reps): %s\n\n",
                        (v == wrong_2a3b) ? "YES" : "no");
            check(v != ideal2, "A at 2 reps is NOT the ideal, reproducing the first pass's 3.5");
        }
    }

    std::printf("CONCLUSION, now with the reference corrected:\n"
                "  the in-place sum is correct in fork/join order and wrong in both serial orders,\n"
                "  and the wrong values are the NAMED signatures (a+2b / 2a+b at one repetition,\n"
                "  2a+3b at two) rather than arbitrary garbage -- so the mechanism is the write-after-\n"
                "  read hazard of an in-place combine, and `allreduce_sum_2rank()`'s per-rank call API\n"
                "  is usable from PER-RANK THREADS or from a fork/join composition, and from nothing\n"
                "  else. The first pass's two 'failures' were its own 2-rep/1-rep reference error.\n"
                "  ⚠️ On TWO devices the same hazard exists and per-process `PeerEvents` are not even\n"
                "  visible to the peer; 1Cat IPC-maps its signal slots and uses release.sys/\n"
                "  acquire.sys for exactly this reason. PENDING HARDWARE.\n\n");

    std::printf("tp2_order_probe2: %d checks, %d failures -> %s\n", g_checks, g_failures,
                g_failures == 0 ? "PASS" : "FAIL");
    return g_failures == 0 ? 0 : 1;
}
