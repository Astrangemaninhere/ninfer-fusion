// tools/mg4/tp2_sum_stress.cu -- THE REPRODUCER AND THE FIX, on one device.
//
// THE RED, stated once, because everything below is written against it: with both ranks driven by ONE
// host thread, the in-place 2-rank sum of `core/tp_transport.h` returns `a + 2b`, because the second
// event pair of a call is not a barrier when the peer's `cudaEventRecord` has not been issued yet at
// the moment this rank calls `cudaStreamWaitEvent` -- a wait SNAPSHOTS the event at the time of the
// call. The corruption is intermittent, which is the worst possible property for a TP shard: a short
// run misses it and the wrong number is then attributed to the model.
//
// WHAT THIS FILE DOES, in the order it runs:
//
//   §0 THE SNAPSHOT PROPERTY, MEASURED. The mechanism above is turned from a citation into three
//      numbers on THIS stack: a record enqueued BEFORE the wait blocks (~the spin behind it), an
//      event with no pending record does not block, and the SAME record enqueued AFTER the wait does
//      not block either. The last number is the defect in one measurement.
//
//   §1 ARM 1's PHASE 1, REPLICATED AND ADJUDICATED. `tools/mgpu2/tp2_selfcheck.cu`
//      (`481d971d4161dba4`) reported `14 checks, 6 failures -> FAIL` with phase 2 green. Its phase 1
//      is replicated EXACTLY here -- two repetitions, call-by-call, one host sync per rep, the same
//      elements and element patterns -- and run several times at BOTH element counts, printing every
//      replica's (rank0, rank1) pair and a staging post-mortem. The probe asserted `a + b` on a run
//      whose clean value is `2(a+b)`, so the question is what each of its six failures actually is,
//      and the answer is a distribution rather than a verdict.
//
//   §2 THE STRESS: N ITERATIONS, A QUANTIFIED RATE, OLD AND NEW. For each skew in a list and for the
//      OLD path and BOTH device-observed mechanisms of the fix, N iterations of one call, counting
//      the iterations whose result is the `a + 2b` signature, the mirror `2a + b`, or neither.
//      `--skew` is the AMPLIFIER: a `%globaltimer` spin on the peer's stream between the two ranks'
//      calls, i.e. a controllable LEAD -- the same parameter `core/tp_transport.h`'s own slot-hazard
//      scan takes. With a large skew the hazard fires DETERMINISTICALLY, so a fix is falsifiable;
//      with skew 0 the natural rate is measured, which is what "nondeterministic" means as a number.
//      A back-to-back arm follows, K calls with NO host sync between them, against an exact
//      expectation computed by simulating the recurrence (see `simulate_clean`).
//
//   §3 THE CONTROLS that must stay green, reported whether or not they do: the fork/join order
//      (MG3's arm C, hand-rolled step by step exactly as `tools/mg3/tp2_order_probe2.cu` does it),
//      the NON-IN-PLACE AND-reduce under the SAME serial order AND the SAME skew that corrupts the
//      sum, the bitwise rank-to-rank agreement, and the host-side fold agreement.
//
//   §4 THE CAPTURE QUESTIONS, re-measured rather than inherited: a cross-stream event wait inside a
//      capture, `cudaMemcpyPeerAsync` inside a capture, and -- the two this line adds, because the
//      remedy depends on them -- whether `cuStreamWriteValue32`/`cuStreamWaitValue32` and a
//      device-side `st.release`/`ld.acquire` kernel are admissible inside a capture.
//
// ⚠️ WHAT THIS CANNOT SHOW, before the first line of output rather than after: ONE DEVICE. Every copy
// is a local D2D copy. The ordering, the value semantics, the numerics and the capture behaviour are
// genuinely exercised and the failure mode is real; the cost, the cross-card ordering and the peer
// memory mapping are `pending hardware`, and no number below is a two-card number. In particular the
// per-iteration time of the FIX on one device with one host thread is a SERIALIZATION figure (the
// value waits make rank 1's device phase wait for rank 0's publish, which the host issues later), not
// a cost figure: with one thread per rank the two waits would be satisfied concurrently.
//
// Build:
//   nvcc -std=c++20 -O2 -arch=sm_120 -I<tree>/src -I<tree> tools/mg4/tp2_sum_stress.cu
//        -o tp2_sum_stress -lcuda

#include "core/and_reduce.h"
#include "core/decode_graph_peer.h"
#include "core/tp_transport.h"

#include <cuda_runtime.h>
#include <cuda.h> // cuStreamWriteValue32 / cuStreamWaitValue32: the capture question in §4

#include <cmath>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

using namespace ninfer::multi;

namespace {

int g_checks   = 0;
int g_failures = 0;

// Wall clock for the per-cell timings. Section 2 took 12 minutes of wall time on its first,
// UNINSTRUMENTED run while holding the GPU lock, which starves the other lines; the timings and the
// unbuffered output below exist so that a slow cell is identifiable from the log instead of by
// inference.
double now_s() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

// Declared here, defined with the capture questions in §4, because §0d needs the driver's own name for
// a CUresult and it runs first.
const char* cu_name(CUresult r);

// A BOUNDED wait for TWO streams to drain. Every wait in this file is bounded, because a hang costs
// more than the measurement: it holds the GPU lock while the other lines queue behind it, which is
// the thing the lock exists to prevent. `cudaStreamQuery` does not block, so this is a poll with a
// deadline, and false means the caller must REPORT rather than continue.
bool drain_two(cudaStream_t a, cudaStream_t b, double budget_s) {
    const double t0 = now_s();
    for (;;) {
        const cudaError_t qa = cudaStreamQuery(a);
        const cudaError_t qb = cudaStreamQuery(b);
        if (qa == cudaSuccess && qb == cudaSuccess) { return true; }
        if (now_s() - t0 > budget_s) { return false; }
        std::this_thread::sleep_for(std::chrono::microseconds(200));
    }
}

void check(bool ok, const std::string& what) {
    ++g_checks;
    if (!ok) {
        ++g_failures;
        std::printf("  FAIL: %s\n", what.c_str());
    } else {
        std::printf("  ok  : %s\n", what.c_str());
    }
}

// Element 0 of the two ranks' inputs. 1.0f and 0.5f are exact in binary32, so every candidate value
// below is exact and every comparison is `==` rather than approximate.
constexpr float  kA = 1.0f;
constexpr float  kB = 0.5f;
constexpr double kIdeal1 = 1.5; // a + b        -- ONE call on a fresh accumulator, when it is clean
constexpr double kSigA2B  = 2.0; // a + 2b       -- this rank read its peer AFTER that peer folded in b
constexpr double kSig2AB  = 2.5; // 2a + b       -- the mirror
constexpr double kIdeal2 = 3.0; // 2(a + b)     -- TWO calls on one accumulator, when both are clean
constexpr double kSig2A3B = 3.5; // 2a + 3b     -- the wrong-order signature carried through two calls

long long env_ll(const char* name, long long fallback) {
    const char* v = std::getenv(name);
    if (v == nullptr || *v == '\0') { return fallback; }
    return std::atoll(v);
}

// ---------------------------------------------------------------------------
// kernels
// ---------------------------------------------------------------------------

__global__ void add_in_place_kernel(float* a, const float* b, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { a[i] += b[i]; }
}

// The amplifier. One thread spins on %globaltimer (1 GHz nominal) for `ns` nanoseconds: a deliberate,
// bounded LEAD. Without it the two streams race and the outcome is nondeterministic, which is exactly
// why a reproducer is needed before a fix.
__global__ void spin_ns_kernel(unsigned long long ns) {
    if (threadIdx.x != 0 || blockIdx.x != 0) { return; }
    unsigned long long t0 = 0, t = 0;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    for (;;) {
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
        if (t - t0 >= ns) { break; }
        __nanosleep(200);
    }
}

// A SETTER of one word, for the co-scheduling probe in §0c: one block, one thread, one store.
__global__ void set_u32_kernel(unsigned int* word, unsigned int value) {
    if (threadIdx.x == 0 && blockIdx.x == 0) { *word = value; }
}

// The BOUNDED counterpart of the spin: it polls a word and gives up at a deadline, and it records
// which happened. It is bounded on purpose -- a probe that can hang is not a probe, and the measured
// question ("can a kernel on stream A observe a store from a kernel on stream B while A is
// resident?") is answered by the timeout as well as by the success.
__global__ void bounded_spin_kernel(const unsigned int* word, unsigned int target,
                                    unsigned long long deadline_ns, unsigned int* outcome) {
    if (threadIdx.x != 0 || blockIdx.x != 0) { return; }
    unsigned long long t0 = 0, t = 0;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    for (;;) {
        if (*word == target) { *outcome = 1u; return; }
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
        if (t - t0 >= deadline_ns) { *outcome = 2u; return; }
        __nanosleep(200);
    }
}

// ---------------------------------------------------------------------------
// the world: two virtual ranks on one device
// ---------------------------------------------------------------------------

struct World {
    static constexpr int kSlots = 2;
    int elements = 0;
    std::size_t bytes = 0;
    void* self[2][kSlots]  = {};
    void* stage[2][kSlots] = {};
    cudaStream_t stream[2] = {nullptr, nullptr};
    std::vector<float> host_a, host_b;

    bool make(int n) {
        elements = n;
        bytes    = static_cast<std::size_t>(n) * sizeof(float);
        host_a.resize(static_cast<std::size_t>(n));
        host_b.resize(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) {
            // The self-check's own pattern, copied so §1 replicates arm 1 rather than approximating it.
            host_a[static_cast<std::size_t>(i)] = 1.0f + static_cast<float>(i % 7);
            host_b[static_cast<std::size_t>(i)] = 0.5f + static_cast<float>(i % 5);
        }
        for (int r = 0; r < 2; ++r) {
            for (int s = 0; s < kSlots; ++s) {
                if (cudaMalloc(&self[r][s], bytes) != cudaSuccess) { return false; }
                if (cudaMalloc(&stage[r][s], bytes) != cudaSuccess) { return false; }
            }
            if (cudaStreamCreateWithFlags(&stream[r], cudaStreamNonBlocking) != cudaSuccess) {
                return false;
            }
        }
        return true;
    }

    void destroy() {
        for (int r = 0; r < 2; ++r) {
            for (int s = 0; s < kSlots; ++s) { cudaFree(self[r][s]); cudaFree(stage[r][s]); }
            cudaStreamDestroy(stream[r]);
        }
    }

    // The full patterned inputs, for §1 (arm 1's own element patterns).
    void load_patterns() {
        cudaMemcpy(self[0][0], host_a.data(), bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(self[1][0], host_b.data(), bytes, cudaMemcpyHostToDevice);
    }

    // A zero field with ONLY element 0 set, for §2: a corruption is then unmistakable, because every
    // other element must remain exactly 0.
    void reset_signature() {
        for (int r = 0; r < 2; ++r) {
            for (int s = 0; s < kSlots; ++s) {
                cudaMemsetAsync(self[r][s], 0, bytes, stream[r]);
                cudaMemsetAsync(stage[r][s], 0, bytes, stream[r]);
            }
            const float v = (r == 0) ? kA : kB;
            for (int s = 0; s < kSlots; ++s) {
                cudaMemcpyAsync(self[r][s], &v, sizeof(float), cudaMemcpyHostToDevice, stream[r]);
            }
        }
        // BOUNDED, and it reports: a stream left blocked by a mechanism this platform cannot run must
        // not turn the reset into an unbounded host wait. That is exactly how the third run wedged.
        if (!drain_two(stream[0], stream[1], 2.0)) {
            std::printf("      [reset] the streams did NOT drain within 2 s -- a mechanism has left "
                        "one of them blocked\n");
        }
    }

    void readback(float out[2], unsigned int slot) const {
        for (int r = 0; r < 2; ++r) {
            cudaMemcpy(&out[r], self[r][slot], sizeof(float), cudaMemcpyDeviceToHost);
        }
    }
};

// ONE rank's whole call, the OLD path, exactly the self-check's shape: the seam, then the caller's
// in-place combine on this rank's own stream.
cudaError_t old_path_call(World& w, int rank, const PeerEvents& ev) {
    cudaStream_t s = w.stream[rank];
    const cudaError_t err =
        allreduce_sum_2rank(w.stage[rank][0], w.self[1 - rank][0], w.bytes, rank, s, ev);
    if (err != cudaSuccess) { return err; }
    add_in_place_kernel<<<(w.elements + 255) / 256, 256, 0, s>>>(
        static_cast<float*>(w.self[rank][0]), static_cast<float*>(w.stage[rank][0]), w.elements);
    return cudaGetLastError();
}

// ONE rank's whole call, the NEW path: the seam owns the in-place combine, so the acknowledgement
// gates it inside the call rather than in the caller's next statement.
cudaError_t new_path_call(World& w, TpSlotChannel& ch, int rank, TpSignalMechanism mechanism,
                          CUresult* cu_out) {
    const std::uint32_t slot = ch.slot_for_call();
    return allreduce_sum_2rank_slotted(ch, w.stream[rank], w.self[rank][slot],
                                       w.self[1 - rank][slot], w.stage[rank][slot], w.elements,
                                       w.bytes, mechanism, cu_out);
}

// ---------------------------------------------------------------------------
// The CLEAN reference for a SEQUENCE of calls, computed by simulating the recurrence
// ---------------------------------------------------------------------------
//
// This is the part the pre-fix arms did not have, and it is why the fix is checkable rather than
// merely "not corrupt". With the acknowledgement, each rank's pull is ordered BEFORE the peer's
// combine writes the slot, so each call's combine reads the peer's PRE-CALL value, whatever the
// interleaving is. The call is therefore the pair of assignments
//
//     A <- A + B      B <- B + A       (right-hand sides evaluated at the pre-call values)
//
// on the slot it used -- a deterministic function of the state, not of who won a race. Simulating it
// gives the exact expected value of every call in a back-to-back sequence, so §2 has an arithmetic
// reference and not just a corruption counter.
//
// The consequence worth stating: A and B are EQUAL after every call, and a slot's value doubles with
// each reuse -- so after u uses of a slot both ranks hold 2^(u-1) * (a + b). The self-check's own
// two-repetition clean value is the u = 2 case: 2 * (a + b) = 3.0, which is what the probe and all
// three of MG3's probes asserted to be `a + b`.
struct CleanModel {
    double a[2] = {kIdeal1, kIdeal1};
    double b[2] = {kIdeal1, kIdeal1};
};

CleanModel simulate_clean(int calls) {
    CleanModel m;
    for (int s = 0; s < 2; ++s) {
        m.a[s] = static_cast<double>(kA);
        m.b[s] = static_cast<double>(kB);
    }
    for (int k = 0; k < calls; ++k) {
        const int s = k % 2;
        const double A = m.a[s];
        const double B = m.b[s];
        m.a[s] = A + B;
        m.b[s] = B + A;
    }
    return m;
}

// The SAME recurrence applied to ONE buffer, which is the shape §1 replicates: the pre-fix seam's
// `allreduce_sum_2rank` has a single slot, so its n repetitions all accumulate into one buffer and
// the clean value after n calls is 2^(n-1) * (a + b), not `a + b`. This is the number the probe
// asserted wrongly and the number arm 1's phase 1 must be read against.
[[nodiscard]] inline double clean_after_one_slot_calls(int calls) {
    double v = static_cast<double>(kA) + static_cast<double>(kB);
    for (int k = 1; k < calls; ++k) { v *= 2.0; }
    return v;
}

// ---------------------------------------------------------------------------
// outcome classification
// ---------------------------------------------------------------------------
enum class Sig : int { Clean = 0, AReadAfterFold = 1, Mirror = 2, Other = 3 };
const char* sig_name(Sig s) {
    switch (s) {
    case Sig::Clean: return "clean (a+b in both ranks)";
    case Sig::AReadAfterFold: return "a+2b on rank 1";
    case Sig::Mirror: return "2a+b on rank 0";
    case Sig::Other: return "NEITHER -- unnamed";
    }
    return "?";
}

Sig classify(float r0, float r1) {
    if (r0 == static_cast<float>(kIdeal1) && r1 == static_cast<float>(kIdeal1)) { return Sig::Clean; }
    if (r0 == static_cast<float>(kIdeal1) && r1 == static_cast<float>(kSigA2B)) {
        return Sig::AReadAfterFold;
    }
    if (r0 == static_cast<float>(kSig2AB) && r1 == static_cast<float>(kIdeal1)) { return Sig::Mirror; }
    return Sig::Other;
}

struct Tally {
    int n = 0;
    int clean = 0;
    int a2b = 0;
    int mirror = 0;
    int other = 0;
    bool has_corrupt = false;
    Sig first_sig = Sig::Clean;
    float first_pair[2] = {0.0f, 0.0f};
};

void tally(Tally& t, float r0, float r1, bool count_by_sum_signature) {
    ++t.n;
    const Sig s = count_by_sum_signature ? classify(r0, r1)
                                        : ((r0 == 1.0f && r1 == 1.0f) ? Sig::Clean : Sig::Other);
    switch (s) {
    case Sig::Clean: ++t.clean; break;
    case Sig::AReadAfterFold: ++t.a2b; break;
    case Sig::Mirror: ++t.mirror; break;
    case Sig::Other: ++t.other; break;
    }
    if (s != Sig::Clean && !t.has_corrupt) {
        t.has_corrupt = true;
        t.first_sig   = s;
        t.first_pair[0] = r0;
        t.first_pair[1] = r1;
    }
}

// ---------------------------------------------------------------------------
// §0 THE SNAPSHOT PROPERTY, MEASURED
// ---------------------------------------------------------------------------
struct SnapshotMeasurement {
    double record_before_wait_ms = -1.0;
    double never_recorded_ms     = -1.0;
    double record_after_wait_ms  = -1.0;
};

SnapshotMeasurement measure_snapshot() {
    SnapshotMeasurement out;
    const unsigned long long spin = 60000000ull; // 60 ms
    cudaStream_t s0 = nullptr, s1 = nullptr;
    cudaStreamCreateWithFlags(&s0, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
    cudaEvent_t e0 = nullptr, e1 = nullptr, e2 = nullptr;
    cudaEvent_t t0 = nullptr, t1 = nullptr;
    cudaEventCreateWithFlags(&e0, cudaEventDisableTiming);
    cudaEventCreateWithFlags(&e1, cudaEventDisableTiming);
    cudaEventCreateWithFlags(&e2, cudaEventDisableTiming);
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);

    const auto time_wait = [&](cudaEvent_t e, double* out_ms) {
        cudaEventRecord(t0, s1);
        cudaStreamWaitEvent(s1, e, 0);
        cudaEventRecord(t1, s1);
        cudaEventSynchronize(t1);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, t0, t1);
        *out_ms = static_cast<double>(ms);
    };

    // (i) the record is enqueued FIRST with 60 ms of work ahead of it: the wait must block.
    spin_ns_kernel<<<1, 1, 0, s0>>>(spin);
    cudaEventRecord(e0, s0);
    time_wait(e0, &out.record_before_wait_ms);
    cudaStreamSynchronize(s0);
    cudaStreamSynchronize(s1);

    // (ii) an event that has NEVER been recorded: the wait acts as if the record had already
    // completed, so the first pair of a call is a no-op whenever the peer is behind.
    time_wait(e1, &out.never_recorded_ms);
    cudaStreamSynchronize(s0);
    cudaStreamSynchronize(s1);

    // (iii) THE DEFECT, isolated: the wait is issued first and the record -- with 60 ms of work ahead
    // of it -- is enqueued after. If the wait were a barrier this would cost 60 ms.
    cudaEventRecord(t0, s1);
    cudaStreamWaitEvent(s1, e2, 0);
    cudaEventRecord(t1, s1);
    spin_ns_kernel<<<1, 1, 0, s0>>>(spin);
    cudaEventRecord(e2, s0);
    cudaEventSynchronize(t1);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, t0, t1);
    out.record_after_wait_ms = static_cast<double>(ms);
    cudaStreamSynchronize(s0);
    cudaStreamSynchronize(s1);

    cudaEventDestroy(e0); cudaEventDestroy(e1); cudaEventDestroy(e2);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cudaStreamDestroy(s0);
    cudaStreamDestroy(s1);
    return out;
}

// ---------------------------------------------------------------------------
// §0b THE AMPLIFIER'S CALIBRATION, and §0c whether two streams' kernels can be CO-RESIDENT here
// ---------------------------------------------------------------------------

// The amplifier is only an amplifier if its duration is the duration it was asked for. Measured, per
// request, with the device's own event pair.
struct SpinCalibration {
    double requested_ns[5] = {0.0, 1000.0, 10000.0, 100000.0, 1000000.0};
    double measured_us[5]  = {0.0, 0.0, 0.0, 0.0, 0.0};
};

SpinCalibration calibrate_spin() {
    SpinCalibration out;
    cudaStream_t s = nullptr;
    cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    cudaEvent_t t0 = nullptr, t1 = nullptr;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    for (int i = 0; i < 5; ++i) {
        const unsigned long long ns = static_cast<unsigned long long>(out.requested_ns[i]);
        cudaEventRecord(t0, s);
        spin_ns_kernel<<<1, 1, 0, s>>>(ns);
        cudaEventRecord(t1, s);
        cudaEventSynchronize(t1);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, t0, t1);
        out.measured_us[i] = static_cast<double>(ms) * 1000.0;
    }
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    cudaStreamDestroy(s);
    return out;
}

// CAN A KERNEL ON ONE STREAM OBSERVE A STORE FROM A KERNEL ON ANOTHER WHILE IT IS RESIDENT?
//
// This is not idle curiosity: the DeviceAcquire mechanism of the fix implements BOTH of its barriers
// as kernels, and a barrier only works if the peer's publish kernel can run while this rank's polling
// kernel is resident. If the platform serialises kernels across streams, the polling kernel must
// time out instead of observing the publish, and the mechanism is unusable HERE no matter how correct
// it is in principle. The stream-value mechanism does not need kernel concurrency at all -- its wait
// is a stream-ordered memory operation -- which is why this probe decides between the two on this
// stack rather than leaving it to a preference.
//
// The spin is BOUNDED, so the probe cannot hang; the outcome is recorded by the kernel itself.
struct CoScheduling {
    bool observed = false;   // the store was seen before the deadline
    bool timed_out = false;
    double ms = 0.0;
};

CoScheduling measure_cross_stream_coscheduling() {
    CoScheduling out;
    unsigned int* word = nullptr;
    unsigned int* outcome = nullptr;
    cudaMalloc(&word, sizeof(unsigned int));
    cudaMalloc(&outcome, sizeof(unsigned int));
    cudaMemset(word, 0, sizeof(unsigned int));
    cudaMemset(outcome, 0, sizeof(unsigned int));
    cudaStream_t a = nullptr, b = nullptr;
    cudaStreamCreateWithFlags(&a, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&b, cudaStreamNonBlocking);

    // Latch the event after the spin so the elapsed time covers the whole exchange.
    cudaEvent_t t0 = nullptr, t1 = nullptr;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    const double wall0 = now_s();

    // A's polling kernel is enqueued FIRST, so it is resident (or at least ahead in the queue) when
    // B's store arrives. 200 ms of deadline is far more than the store can possibly take.
    cudaEventRecord(t0, a);
    bounded_spin_kernel<<<1, 1, 0, a>>>(word, 1u, 200000000ull, outcome);
    cudaEventRecord(t1, a);
    set_u32_kernel<<<1, 1, 0, b>>>(word, 1u);
    cudaStreamSynchronize(a);
    cudaStreamSynchronize(b);
    out.ms = now_s() - wall0;
    unsigned int h_outcome = 0u;
    cudaMemcpy(&h_outcome, outcome, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    out.observed = (h_outcome == 1u);
    out.timed_out = (h_outcome == 2u);
    (void)t0;
    (void)t1;
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    cudaStreamDestroy(a);
    cudaStreamDestroy(b);
    cudaFree(word);
    cudaFree(outcome);
    return out;
}


// A BOUNDED wait for a stream to drain, kept as the World-shaped wrapper the §2 arms use.
bool drain_with_bound(World& w, double budget_s) {
    return drain_two(w.stream[0], w.stream[1], budget_s);
}

// ---------------------------------------------------------------------------
// §0d CAN A VALUE WAIT ON ONE STREAM BE SATISFIED BY A WRITE FROM ANOTHER? (bounded)
// ---------------------------------------------------------------------------
//
// Section 4 shows `cuStreamWriteValue32`/`cuStreamWaitValue32` are ACCEPTED inside a capture and work
// EAGERLY ON ONE STREAM. That is not the question the fix asks of them: the fix's readiness barrier
// and its acknowledgement are each a wait on ONE stream satisfied by a peer's publish on ANOTHER, and
// the first instrumented run stalled there with the host at 99.5 % CPU and no progress at all. So the
// question is asked directly, in three bounded cells -- bounded so that the answer cannot be a hang:
//
//   (a) same stream, write then wait -- the control
//   (b) cross stream, the write FIRST and synchronised, so the value is provably in memory already
//   (c) cross stream, the wait enqueued FIRST and the write after, which is the shape the fix uses
//
// If (b) is not satisfied, the mechanism cannot carry a cross-rank barrier on this stack whatever the
// API documents, and that is a measurement rather than an opinion about WSL.
struct ValueWaitAnswers {
    bool same_stream = false;
    double same_stream_ms = 0.0;
    bool cross_stream_value_present = false;
    double cross_stream_value_present_ms = 0.0;
    bool cross_stream_write_after = false;
    double cross_stream_write_after_ms = 0.0;
    std::string errors;
};

bool stream_drains(cudaStream_t s, double budget_s, double* ms) {
    const double t0 = now_s();
    for (;;) {
        const cudaError_t q = cudaStreamQuery(s);
        if (q == cudaSuccess) { *ms = (now_s() - t0) * 1000.0; return true; }
        if (now_s() - t0 > budget_s) { *ms = (now_s() - t0) * 1000.0; return false; }
        std::this_thread::sleep_for(std::chrono::microseconds(200));
    }
}

ValueWaitAnswers measure_value_wait_reach() {
    ValueWaitAnswers out;
    const double budget = 2.0; // seconds: a satisfied wait resolves in microseconds
    cudaStream_t a = nullptr, b = nullptr;
    cudaStreamCreateWithFlags(&a, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&b, cudaStreamNonBlocking);
    unsigned int* word = nullptr;
    cudaMalloc(&word, sizeof(unsigned int));

    // (a) the control: the same stream.
    cudaMemsetAsync(word, 0, sizeof(unsigned int), a);
    cudaStreamSynchronize(a);
    const CUresult r1 = cuStreamWriteValue32(reinterpret_cast<CUstream>(a),
                                            reinterpret_cast<CUdeviceptr>(word), 5u, 0u);
    const CUresult r2 = cuStreamWaitValue32(reinterpret_cast<CUstream>(a),
                                           reinterpret_cast<CUdeviceptr>(word), 5u,
                                           CU_STREAM_WAIT_VALUE_GEQ);
    out.errors = std::string("same-stream write=") + cu_name(r1) + " wait=" + cu_name(r2);
    out.same_stream = stream_drains(a, budget, &out.same_stream_ms);

    // (b) cross stream, the write first AND synchronised, so the wait has nothing left to wait for.
    cudaMemsetAsync(word, 0, sizeof(unsigned int), b);
    cudaStreamSynchronize(b);
    const CUresult r3 = cuStreamWriteValue32(reinterpret_cast<CUstream>(b),
                                            reinterpret_cast<CUdeviceptr>(word), 6u, 0u);
    cudaStreamSynchronize(b);
    unsigned int h_word = 0u;
    cudaMemcpy(&h_word, word, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    const CUresult r4 = cuStreamWaitValue32(reinterpret_cast<CUstream>(a),
                                           reinterpret_cast<CUdeviceptr>(word), 6u,
                                           CU_STREAM_WAIT_VALUE_GEQ);
    out.errors += std::string(" | cross w=") + cu_name(r3) + " wait=" + cu_name(r4) +
                  " word=" + std::to_string(h_word);
    out.cross_stream_value_present = stream_drains(a, budget, &out.cross_stream_value_present_ms);

    // (c) the wait first, the write after: the shape the fix actually uses.
    cudaMemsetAsync(word, 0, sizeof(unsigned int), a);
    cudaStreamSynchronize(a);
    const CUresult r5 = cuStreamWaitValue32(reinterpret_cast<CUstream>(a),
                                           reinterpret_cast<CUdeviceptr>(word), 7u,
                                           CU_STREAM_WAIT_VALUE_GEQ);
    const CUresult r6 = cuStreamWriteValue32(reinterpret_cast<CUstream>(b),
                                            reinterpret_cast<CUdeviceptr>(word), 7u, 0u);
    out.errors += std::string(" | later wait=") + cu_name(r5) + " write=" + cu_name(r6);
    cudaStreamSynchronize(b);
    out.cross_stream_write_after = stream_drains(a, budget, &out.cross_stream_write_after_ms);

    cudaFree(word);
    cudaStreamDestroy(a);
    cudaStreamDestroy(b);
    return out;
}

// ---------------------------------------------------------------------------
// §5 THE MAXIMALLY-ORDERED CONTROL: a host synchronisation after EVERY step, so a race is impossible
// ---------------------------------------------------------------------------
//
// §1's 262144-element cells need this. They do not reproduce the a+2b signature at all: the ranks come
// back at their INITIAL values while the staging holds something else, which no ordering of one pull
// and one combine can produce. So the same call is run with BOTH pulls done before EITHER combine AND
// a host synchronisation after every single step -- the strongest form of the order that was clean in
// all three of MG3's passes -- and swept over the element count. If it is clean at every size, the
// 262144 anomaly is about ordering; if it is not, the defect is not an ordering defect at all and no
// barrier will fix it.
struct SyncedResult {
    // FLOATS, not doubles. The first version read `sizeof(float)` bytes into a `double` and printed
    // 0.000000 for every value while the element-wise counts in the same struct were correct -- a
    // probe's own error, reported here rather than quietly fixed, because it is the second time this
    // line has put a wrong type under a correct comparison.
    float r0 = 0.0f, r1 = 0.0f, s0 = 0.0f, s1 = 0.0f;
    int bad_elements = 0;   // elements where either rank differs from a+b
    int stage_mismatch = 0; // elements where a staging differs from the peer's source
    bool ran = false;
};

SyncedResult fully_synced_call(int elements) {
    SyncedResult out;
    World w;
    if (!w.make(elements)) { return out; }
    w.load_patterns();

    // BOTH pulls first, each followed by a host synchronisation.
    pull_peer(w.stage[0][0], w.self[1][0], w.bytes, w.stream[0]);
    cudaStreamSynchronize(w.stream[0]);
    pull_peer(w.stage[1][0], w.self[0][0], w.bytes, w.stream[1]);
    cudaStreamSynchronize(w.stream[1]);
    // THEN both combines, each followed by a host synchronisation.
    const int blocks = (elements + 255) / 256;
    add_in_place_kernel<<<blocks, 256, 0, w.stream[0]>>>(
        static_cast<float*>(w.self[0][0]), static_cast<float*>(w.stage[0][0]), elements);
    cudaStreamSynchronize(w.stream[0]);
    add_in_place_kernel<<<blocks, 256, 0, w.stream[1]>>>(
        static_cast<float*>(w.self[1][0]), static_cast<float*>(w.stage[1][0]), elements);
    cudaStreamSynchronize(w.stream[1]);

    cudaMemcpy(&out.r0, w.self[0][0], sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&out.r1, w.self[1][0], sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&out.s0, w.stage[0][0], sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&out.s1, w.stage[1][0], sizeof(float), cudaMemcpyDeviceToHost);

    std::vector<float> g0(elements), g1(elements), t0(elements), t1(elements);
    cudaMemcpy(g0.data(), w.self[0][0], w.bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(g1.data(), w.self[1][0], w.bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(t0.data(), w.stage[0][0], w.bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(t1.data(), w.stage[1][0], w.bytes, cudaMemcpyDeviceToHost);
    for (int i = 0; i < elements; ++i) {
        const std::size_t k = static_cast<std::size_t>(i);
        if (g0[k] != w.host_a[k] + w.host_b[k] || g1[k] != w.host_b[k] + w.host_a[k]) {
            ++out.bad_elements;
        }
        if (t0[k] != w.host_b[k] || t1[k] != w.host_a[k]) { ++out.stage_mismatch; }
    }
    out.ran = true;
    w.destroy();
    return out;
}

// ---------------------------------------------------------------------------
// §6 WHICH STEP FAILS AT SIZE: the transport's own copy, swept
// ---------------------------------------------------------------------------
//
// §5's result needs one more measurement before it may be called a finding. With a host
// synchronisation after every step the call is exact at 4096 and 16384 elements and wrong in EVERY
// element at 65536 and above -- and "every element wrong, in both stagings" is the signature of a
// copy that delivered NOTHING, not of one that raced. Three steps could be responsible, and this
// sweep separates them AT EACH SIZE, with every cell fully host-synchronised so that no ordering
// statement is involved: a cell that fails here fails for a reason no barrier can fix.
//
//   H2D load      cudaMemcpy, synchronous            -- does the input even land?
//   async D2D     cudaMemcpyAsync on a non-blocking stream + a synchronise -- the seam's own form
//   sync D2D      cudaMemcpy, D2D                    -- is it the asynchronous form that fails?
struct CopySweep {
    int n = 0;
    std::size_t bytes = 0;
    int h2d_bad = 0;
    int async_d2d_bad = 0;
    int sync_d2d_bad = 0;
    bool async_drained = false;
};

CopySweep copy_sweep(int elements) {
    CopySweep out;
    out.n = elements;
    out.bytes = static_cast<std::size_t>(elements) * sizeof(float);
    const std::size_t bytes = out.bytes;
    std::vector<float> host(elements);
    for (int i = 0; i < elements; ++i) { host[static_cast<std::size_t>(i)] = 100.0f + static_cast<float>(i % 13); }
    void* src = nullptr;
    void* dst = nullptr;
    cudaMalloc(&src, bytes);
    cudaMalloc(&dst, bytes);
    cudaStream_t s = nullptr;
    cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    std::vector<float> back(elements);

    // (A) the host-to-device load.
    cudaMemset(src, 0, bytes);
    cudaMemcpy(src, host.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemset(back.data(), 0, bytes);
    cudaMemcpy(back.data(), src, bytes, cudaMemcpyDeviceToHost);
    for (int i = 0; i < elements; ++i) {
        if (back[static_cast<std::size_t>(i)] != host[static_cast<std::size_t>(i)]) { ++out.h2d_bad; }
    }

    // (B) the pull, the seam's own asynchronous form, then a bounded synchronise.
    cudaMemset(dst, 0, bytes);
    pull_peer(dst, src, bytes, s);
    double wait_ms = 0.0;
    out.async_drained = stream_drains(s, 5.0, &wait_ms);
    cudaMemset(back.data(), 0, bytes);
    cudaMemcpy(back.data(), dst, bytes, cudaMemcpyDeviceToHost);
    for (int i = 0; i < elements; ++i) {
        if (back[static_cast<std::size_t>(i)] != host[static_cast<std::size_t>(i)]) { ++out.async_d2d_bad; }
    }

    // (C) the same transfer, synchronous.
    cudaMemset(dst, 0, bytes);
    cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice);
    cudaMemset(back.data(), 0, bytes);
    cudaMemcpy(back.data(), dst, bytes, cudaMemcpyDeviceToHost);
    for (int i = 0; i < elements; ++i) {
        if (back[static_cast<std::size_t>(i)] != host[static_cast<std::size_t>(i)]) { ++out.sync_d2d_bad; }
    }

    cudaStreamDestroy(s);
    cudaFree(src);
    cudaFree(dst);
    return out;
}

// ---------------------------------------------------------------------------
// §7 THE STEP THAT FAILS, NAMED: §5's sequence with every return value checked
// ---------------------------------------------------------------------------
//
// §6 REFUTES the first hypothesis: the transfer is exact at every size up to 4 MB when it is isolated.
// §5 shows the SAME transfer wrong from 65536 elements up when it runs inside the World's buffers. So
// the failing step is inside the sequence and not in the transfer's size -- and this section runs that
// sequence again with (a) the RETURN VALUE of every call checked and printed, and (b) the staging and
// the sources read back after EVERY step. That is the only way to say WHICH step failed rather than
// which step is suspected. §5 threw `pull_peer`'s return value away; so does the seam's own arms, and
// that is exactly how a silent failure comes to look like a race.
struct StepTrace {
    int load_err = 0;
    int load_bad = 0;
    int pull0_err = 0, pull1_err = 0;
    int sync0_err = 0, sync1_err = 0;
    int stage0_bad = 0, stage1_bad = 0;
    int add0_err = 0, add1_err = 0;
    int add0_sync = 0, add1_sync = 0;
    int final0_bad = 0, final1_bad = 0;
    float st0 = 0.0f, st1 = 0.0f, r0 = 0.0f, r1 = 0.0f;
    bool ran = false;
};

StepTrace step_trace(int elements) {
    StepTrace out;
    World w;
    if (!w.make(elements)) { return out; }
    w.load_patterns();
    out.load_err = static_cast<int>(cudaGetLastError());
    std::vector<float> v(static_cast<std::size_t>(elements));
    cudaMemcpy(v.data(), w.self[0][0], w.bytes, cudaMemcpyDeviceToHost);
    for (int i = 0; i < elements; ++i) {
        if (v[static_cast<std::size_t>(i)] != w.host_a[static_cast<std::size_t>(i)]) { ++out.load_bad; }
    }
    cudaMemcpy(v.data(), w.self[1][0], w.bytes, cudaMemcpyDeviceToHost);
    for (int i = 0; i < elements; ++i) {
        if (v[static_cast<std::size_t>(i)] != w.host_b[static_cast<std::size_t>(i)]) { ++out.load_bad; }
    }

    out.pull0_err = static_cast<int>(pull_peer(w.stage[0][0], w.self[1][0], w.bytes, w.stream[0]));
    out.sync0_err = static_cast<int>(cudaStreamSynchronize(w.stream[0]));
    out.pull1_err = static_cast<int>(pull_peer(w.stage[1][0], w.self[0][0], w.bytes, w.stream[1]));
    out.sync1_err = static_cast<int>(cudaStreamSynchronize(w.stream[1]));
    cudaMemcpy(&out.st0, w.stage[0][0], sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(v.data(), w.stage[0][0], w.bytes, cudaMemcpyDeviceToHost);
    for (int i = 0; i < elements; ++i) {
        if (v[static_cast<std::size_t>(i)] != w.host_b[static_cast<std::size_t>(i)]) { ++out.stage0_bad; }
    }
    cudaMemcpy(&out.st1, w.stage[1][0], sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(v.data(), w.stage[1][0], w.bytes, cudaMemcpyDeviceToHost);
    for (int i = 0; i < elements; ++i) {
        if (v[static_cast<std::size_t>(i)] != w.host_a[static_cast<std::size_t>(i)]) { ++out.stage1_bad; }
    }

    const int blocks = (elements + 255) / 256;
    add_in_place_kernel<<<blocks, 256, 0, w.stream[0]>>>(
        static_cast<float*>(w.self[0][0]), static_cast<float*>(w.stage[0][0]), elements);
    out.add0_err = static_cast<int>(cudaGetLastError());
    out.add0_sync = static_cast<int>(cudaStreamSynchronize(w.stream[0]));
    cudaMemcpy(&out.r0, w.self[0][0], sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(v.data(), w.self[0][0], w.bytes, cudaMemcpyDeviceToHost);
    for (int i = 0; i < elements; ++i) {
        if (v[static_cast<std::size_t>(i)] !=
            w.host_a[static_cast<std::size_t>(i)] + w.host_b[static_cast<std::size_t>(i)]) {
            ++out.final0_bad;
        }
    }
    add_in_place_kernel<<<blocks, 256, 0, w.stream[1]>>>(
        static_cast<float*>(w.self[1][0]), static_cast<float*>(w.stage[1][0]), elements);
    out.add1_err = static_cast<int>(cudaGetLastError());
    out.add1_sync = static_cast<int>(cudaStreamSynchronize(w.stream[1]));
    cudaMemcpy(&out.r1, w.self[1][0], sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(v.data(), w.self[1][0], w.bytes, cudaMemcpyDeviceToHost);
    for (int i = 0; i < elements; ++i) {
        if (v[static_cast<std::size_t>(i)] !=
            w.host_b[static_cast<std::size_t>(i)] + w.host_a[static_cast<std::size_t>(i)]) {
            ++out.final1_bad;
        }
    }
    out.ran = true;
    w.destroy();
    return out;
}

// ---------------------------------------------------------------------------
// §1 ARM 1's PHASE 1, REPLICATED EXACTLY, RUN R TIMES, WITH THE VALUES PRINTED
// ---------------------------------------------------------------------------
// Verbatim the shape of tools/mgpu2/tp2_selfcheck.cu's run_phase1: one PeerEvents, `reps`
// repetitions, `for rank in {0,1}` call-by-call with the combine immediately after that rank's seam
// call, and ONE host synchronisation per rep. The staging is read back too, because "which value did
// the pull deliver" is what separates a race from a copy that never happened.
struct Arm1Replica {
    double r0 = 0.0, r1 = 0.0;
    double s0 = 0.0, s1 = 0.0; // element 0 of each rank's staging afterwards
    int deviating_elements = 0;
};

Arm1Replica run_arm1_replica(int elements, int reps) {
    Arm1Replica out;
    World w;
    w.make(elements);
    w.load_patterns();
    PeerEvents ev;
    ev.create();
    const int blocks = (elements + 255) / 256;

    for (int rep = 0; rep < reps; ++rep) {
        for (int rank = 0; rank < 2; ++rank) {
            const cudaError_t err = allreduce_sum_2rank(w.stage[rank][0], w.self[1 - rank][0],
                                                        w.bytes, rank, w.stream[rank], ev);
            if (err != cudaSuccess) { std::printf("      seam error %s\n", cudaGetErrorName(err)); }
            add_in_place_kernel<<<blocks, 256, 0, w.stream[rank]>>>(
                static_cast<float*>(w.self[rank][0]), static_cast<float*>(w.stage[rank][0]),
                elements);
        }
        cudaStreamSynchronize(w.stream[0]);
        cudaStreamSynchronize(w.stream[1]);
    }

    float g0 = 0.0f, g1 = 0.0f, st0 = 0.0f, st1 = 0.0f;
    cudaMemcpy(&g0, w.self[0][0], sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&g1, w.self[1][0], sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&st0, w.stage[0][0], sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&st1, w.stage[1][0], sizeof(float), cudaMemcpyDeviceToHost);
    out.r0 = static_cast<double>(g0);
    out.r1 = static_cast<double>(g1);
    out.s0 = static_cast<double>(st0);
    out.s1 = static_cast<double>(st1);

    // Element-wise, against the CLEAN reference for this many repetitions, so no verdict rests on
    // element 0 alone. The reference is computed from the model rather than written as a constant,
    // because a 2-repetition clean value is not `a + b` -- which is the whole adjudication.
    std::vector<float> v0(elements), v1(elements);
    cudaMemcpy(v0.data(), w.self[0][0], w.bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(v1.data(), w.self[1][0], w.bytes, cudaMemcpyDeviceToHost);
    for (int i = 0; i < elements; ++i) {
        // ONE slot, so the reference is the one-buffer recurrence: 2^(reps-1) * (a_i + b_i).
        const double acc = std::pow(2.0, reps - 1) * (static_cast<double>(w.host_a[static_cast<std::size_t>(i)]) +
                                                      static_cast<double>(w.host_b[static_cast<std::size_t>(i)]));
        if (static_cast<double>(v0[static_cast<std::size_t>(i)]) != acc) { ++out.deviating_elements; }
        if (static_cast<double>(v1[static_cast<std::size_t>(i)]) != acc) { ++out.deviating_elements; }
    }

    ev.destroy();
    w.destroy();
    return out;
}

// ---------------------------------------------------------------------------
// §2 the stress iterations
// ---------------------------------------------------------------------------
void old_iteration(World& w, const PeerEvents& ev, unsigned long long skew_ns, int calls) {
    w.reset_signature();
    for (int k = 0; k < calls; ++k) {
        old_path_call(w, 0, ev);
        if (skew_ns != 0 && k == 0) { spin_ns_kernel<<<1, 1, 0, w.stream[1]>>>(skew_ns); }
        old_path_call(w, 1, ev);
    }
    cudaStreamSynchronize(w.stream[0]);
    cudaStreamSynchronize(w.stream[1]);
}

// The NEW path: one call per rank per k, slots advanced by the channel itself. Returns false if the
// two streams did not drain inside the bound, which is how a mechanism that cannot work here is
// REPORTED instead of hung on.
bool new_iteration(World& w, TpSlotChannel ch[2], TpSignalMechanism m, unsigned long long skew_ns,
                   int calls, CUresult* cu_out, std::uint32_t* last_slot) {
    w.reset_signature();
    for (int k = 0; k < calls; ++k) {
        const std::uint32_t slot = ch[0].slot_for_call();
        *last_slot = slot;
        cudaError_t err = new_path_call(w, ch[0], 0, m, cu_out);
        if (err != cudaSuccess) { std::printf("      new-path rank0 error %s\n", cudaGetErrorName(err)); }
        if (skew_ns != 0 && k == 0) { spin_ns_kernel<<<1, 1, 0, w.stream[1]>>>(skew_ns); }
        err = new_path_call(w, ch[1], 1, m, cu_out);
        if (err != cudaSuccess) { std::printf("      new-path rank1 error %s\n", cudaGetErrorName(err)); }
    }
    return drain_with_bound(w, 2.0);
}

// The FORK/JOIN order (MG3's arm C), hand-rolled step by step exactly as tp2_order_probe2.cu does it:
// both records, then both waits, then both pulls, then both records, then both waits, then both
// combines.
void forkjoin_iteration(World& w, const PeerEvents& ev) {
    w.reset_signature();
    const int blocks = (w.elements + 255) / 256;
    for (int r = 0; r < 2; ++r) { cudaEventRecord(ev.inputs_ready[r], w.stream[r]); }
    for (int r = 0; r < 2; ++r) { cudaStreamWaitEvent(w.stream[r], ev.inputs_ready[1 - r], 0); }
    for (int r = 0; r < 2; ++r) { pull_peer(w.stage[r][0], w.self[1 - r][0], w.bytes, w.stream[r]); }
    for (int r = 0; r < 2; ++r) { cudaEventRecord(ev.pull_done[r], w.stream[r]); }
    for (int r = 0; r < 2; ++r) { cudaStreamWaitEvent(w.stream[r], ev.pull_done[1 - r], 0); }
    for (int r = 0; r < 2; ++r) {
        add_in_place_kernel<<<blocks, 256, 0, w.stream[r]>>>(
            static_cast<float*>(w.self[r][0]), static_cast<float*>(w.stage[r][0]), w.elements);
    }
    cudaStreamSynchronize(w.stream[0]);
    cudaStreamSynchronize(w.stream[1]);
}

// The non-in-place AND-reduce on the SAME serial order and with the SAME amplifier. This is the
// discriminating control: if the ordering were the defect, this arm would be corrupt too.
struct AndControl {
    Tally t;
    bool ranks_equal = true;
    unsigned int verdict0 = 0u;
    int host_disagreements = 0;
};

AndControl and_control(int iters, unsigned long long skew_ns) {
    AndControl out;
    World w;
    w.make(256);
    void* mine[2]   = {nullptr, nullptr};
    void* peer_w[2] = {nullptr, nullptr};
    void* stag[2]   = {nullptr, nullptr};
    void* outw[2]   = {nullptr, nullptr};
    for (int r = 0; r < 2; ++r) {
        cudaMalloc(&mine[r], sizeof(unsigned int));
        cudaMalloc(&peer_w[r], sizeof(unsigned int));
        cudaMalloc(&stag[r], sizeof(unsigned int));
        cudaMalloc(&outw[r], sizeof(unsigned int));
        const unsigned int one = 1u;
        cudaMemcpy(mine[r], &one, sizeof(unsigned int), cudaMemcpyHostToDevice);
        cudaMemcpy(peer_w[r], &one, sizeof(unsigned int), cudaMemcpyHostToDevice);
    }
    PeerEvents ev;
    ev.create();
    const AndReduceOutcome host_fold = and_reduce_wire({1u, 1u});

    for (int it = 0; it < iters; ++it) {
        for (int r = 0; r < 2; ++r) {
            cudaMemsetAsync(outw[r], 0, sizeof(unsigned int), w.stream[r]);
        }
        const cudaError_t e0 = allreduce_and_2rank(mine[0], peer_w[1], stag[0], outw[0],
                                                   sizeof(unsigned int), 0, w.stream[0], ev);
        if (e0 != cudaSuccess) { std::printf("      AND seam error %s\n", cudaGetErrorName(e0)); }
        if (skew_ns != 0) { spin_ns_kernel<<<1, 1, 0, w.stream[1]>>>(skew_ns); }
        const cudaError_t e1 = allreduce_and_2rank(mine[1], peer_w[0], stag[1], outw[1],
                                                   sizeof(unsigned int), 1, w.stream[1], ev);
        if (e1 != cudaSuccess) { std::printf("      AND seam error %s\n", cudaGetErrorName(e1)); }
        cudaStreamSynchronize(w.stream[0]);
        cudaStreamSynchronize(w.stream[1]);
        unsigned int v[2] = {0u, 0u};
        for (int r = 0; r < 2; ++r) {
            cudaMemcpy(&v[r], outw[r], sizeof(unsigned int), cudaMemcpyDeviceToHost);
        }
        if (v[0] != v[1]) { out.ranks_equal = false; }
        // The AND-reduce's own vocabulary: the verdict is a boolean, and the host fold for {1,1} is
        // ADMIT. Classifying it with the SUM's signature vocabulary is the mistake this line made on
        // its first pass and is corrected here.
        const bool device_admits = (v[0] == 1u && v[1] == 1u);
        if (device_admits != host_fold.admitted) { ++out.host_disagreements; }
        if (it == 0) { out.verdict0 = v[0]; }
        tally(out.t, static_cast<float>(v[0]), static_cast<float>(v[1]),
              /*count_by_sum_signature=*/false);
    }
    for (int r = 0; r < 2; ++r) {
        cudaFree(mine[r]); cudaFree(peer_w[r]); cudaFree(stag[r]); cudaFree(outw[r]);
    }
    ev.destroy();
    w.destroy();
    return out;
}

// ---------------------------------------------------------------------------
// §4 the capture questions
// ---------------------------------------------------------------------------
struct CaptureAnswers {
    bool fork_join_accepted = false;
    std::string fork_join_err;
    bool memcpy_peer_refused = false;
    std::string memcpy_peer_err;
    CUresult write_value_capture = CUDA_SUCCESS;
    CUresult wait_value_capture  = CUDA_SUCCESS;
    std::string write_value_err;
    std::string wait_value_err;
    bool spin_kernel_captured = false;
    std::string spin_kernel_err;
    bool spin_graph_replayed = false;
    std::string value_op_eager;
};

const char* cu_name(CUresult r) {
    const char* n = nullptr;
    if (cuGetErrorName(r, &n) != CUDA_SUCCESS || n == nullptr) { return "unknown"; }
    return n;
}

CaptureAnswers answer_capture_questions() {
    CaptureAnswers a;
    const int elements = 1024;
    const std::size_t bytes = static_cast<std::size_t>(elements) * sizeof(float);
    void* p = nullptr;
    void* q = nullptr;
    cudaMalloc(&p, bytes);
    cudaMalloc(&q, bytes);
    unsigned int* flag = nullptr;
    cudaMalloc(&flag, sizeof(unsigned int));
    cudaMemset(flag, 0, sizeof(unsigned int));

    {
        cudaStream_t s0 = nullptr, s1 = nullptr;
        cudaStreamCreateWithFlags(&s0, cudaStreamNonBlocking);
        cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
        PeerForkJoin fj;
        fj.create();
        cudaGraph_t graph = nullptr;
        const cudaError_t err = capture_fork_join(
            s0, s1, fj,
            [&](cudaStream_t origin, cudaStream_t peer_stream) {
                cudaMemcpyAsync(p, q, bytes, cudaMemcpyDeviceToDevice, origin);
                cudaMemcpyAsync(q, p, bytes, cudaMemcpyDeviceToDevice, peer_stream);
            },
            &graph);
        a.fork_join_err      = cudaGetErrorName(err);
        a.fork_join_accepted = (err == cudaSuccess && graph != nullptr);
        if (graph != nullptr) { cudaGraphDestroy(graph); }
        fj.destroy();
        cudaStreamDestroy(s0);
        cudaStreamDestroy(s1);
    }
    {
        cudaStream_t c = nullptr;
        cudaStreamCreateWithFlags(&c, cudaStreamNonBlocking);
        const cudaError_t be = cudaStreamBeginCapture(c, cudaStreamCaptureModeThreadLocal);
        if (be == cudaSuccess) {
            const cudaError_t inner = cudaMemcpyPeerAsync(p, 0, q, 0, bytes, c);
            a.memcpy_peer_err       = cudaGetErrorName(inner);
            a.memcpy_peer_refused   = (inner == cudaErrorStreamCaptureUnsupported);
            cudaGraph_t discard     = nullptr;
            cudaStreamEndCapture(c, &discard);
            if (discard != nullptr) { cudaGraphDestroy(discard); }
        } else {
            a.memcpy_peer_err = std::string("begin_capture failed: ") + cudaGetErrorName(be);
        }
        cudaStreamDestroy(c);
    }
    {
        cudaStream_t c = nullptr;
        cudaStreamCreateWithFlags(&c, cudaStreamNonBlocking);
        const cudaError_t be = cudaStreamBeginCapture(c, cudaStreamCaptureModeThreadLocal);
        if (be == cudaSuccess) {
            a.write_value_capture = cuStreamWriteValue32(reinterpret_cast<CUstream>(c),
                                                        reinterpret_cast<CUdeviceptr>(flag), 1u, 0u);
            a.write_value_err     = cu_name(a.write_value_capture);
            cudaGraph_t discard   = nullptr;
            cudaStreamEndCapture(c, &discard);
            (void)cudaGetLastError();
            if (discard != nullptr) { cudaGraphDestroy(discard); }
        } else {
            a.write_value_capture = CUDA_ERROR_UNKNOWN;
            a.write_value_err     = std::string("begin_capture failed: ") + cudaGetErrorName(be);
        }
        cudaStreamDestroy(c);
    }
    {
        cudaStream_t c = nullptr;
        cudaStreamCreateWithFlags(&c, cudaStreamNonBlocking);
        const cudaError_t be = cudaStreamBeginCapture(c, cudaStreamCaptureModeThreadLocal);
        if (be == cudaSuccess) {
            a.wait_value_capture = cuStreamWaitValue32(reinterpret_cast<CUstream>(c),
                                                      reinterpret_cast<CUdeviceptr>(flag), 1u,
                                                      CU_STREAM_WAIT_VALUE_EQ);
            a.wait_value_err     = cu_name(a.wait_value_capture);
            cudaGraph_t discard  = nullptr;
            cudaStreamEndCapture(c, &discard);
            (void)cudaGetLastError();
            if (discard != nullptr) { cudaGraphDestroy(discard); }
        } else {
            a.wait_value_capture = CUDA_ERROR_UNKNOWN;
            a.wait_value_err     = std::string("begin_capture failed: ") + cudaGetErrorName(be);
        }
        cudaStreamDestroy(c);
    }
    {
        cudaStream_t c = nullptr;
        cudaStreamCreateWithFlags(&c, cudaStreamNonBlocking);
        const CUresult w = cuStreamWriteValue32(reinterpret_cast<CUstream>(c),
                                                reinterpret_cast<CUdeviceptr>(flag), 7u, 0u);
        const CUresult r = cuStreamWaitValue32(reinterpret_cast<CUstream>(c),
                                               reinterpret_cast<CUdeviceptr>(flag), 7u,
                                               CU_STREAM_WAIT_VALUE_EQ);
        const cudaError_t sy = cudaStreamSynchronize(c);
        unsigned int v = 0u;
        cudaMemcpy(&v, flag, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        a.value_op_eager = std::string("write=") + cu_name(w) + " wait=" + cu_name(r) +
                           " sync=" + cudaGetErrorName(sy) + " word_reads_back=" + std::to_string(v);
        cudaStreamDestroy(c);
    }
    {
        cudaStream_t c = nullptr;
        cudaStreamCreateWithFlags(&c, cudaStreamNonBlocking);
        (void)cudaGetLastError(); // clear anything sticky from the refusal probes above
        const cudaError_t be = cudaStreamBeginCapture(c, cudaStreamCaptureModeThreadLocal);
        if (be == cudaSuccess) {
            // THE CARRIER OF THE DEVICE-ACQUIRE MECHANISM: a release store and a bounded spin, both
            // captured into one graph. They are on the SAME stream, so the graph serialises them and
            // the spin is satisfied by the graph's own publish -- the shape a captured collective
            // needs, and the shape that would deadlock if the two were on branches the graph could
            // not order. Whether a CROSS-BRANCH spin can make progress needs two devices.
            tp_publish_epoch_kernel<<<1, 1, 0, c>>>(flag, 1u);
            const cudaError_t launch_err = cudaGetLastError();
            tp_wait_epoch_kernel<<<1, 1, 0, c>>>(flag, 1u);
            const cudaError_t wait_err = cudaGetLastError();
            cudaGraph_t g          = nullptr;
            const cudaError_t ee   = cudaStreamEndCapture(c, &g);
            a.spin_kernel_err = std::string(cudaGetErrorName(launch_err)) + " / " +
                                cudaGetErrorName(wait_err) + " / end=" + cudaGetErrorName(ee);
            a.spin_kernel_captured = (ee == cudaSuccess && g != nullptr);
            if (a.spin_kernel_captured) {
                cudaGraphExec_t ex = nullptr;
                if (cudaGraphInstantiate(&ex, g, 0) == cudaSuccess) {
                    cudaGraphLaunch(ex, c);
                    const cudaError_t rl = cudaStreamSynchronize(c);
                    unsigned int replay_value = 0u;
                    cudaMemcpy(&replay_value, flag, sizeof(unsigned int), cudaMemcpyDeviceToHost);
                    a.spin_graph_replayed = (rl == cudaSuccess && replay_value == 1u);
                    cudaGraphExecDestroy(ex);
                }
            }
            if (g != nullptr) { cudaGraphDestroy(g); }
        } else {
            a.spin_kernel_err = std::string("begin_capture failed: ") + cudaGetErrorName(be);
        }
        cudaStreamDestroy(c);
    }

    cudaFree(flag);
    cudaFree(p);
    cudaFree(q);
    return a;
}

void print_tally(const char* label, const Tally& t, double secs) {
    std::printf("  %-44s N=%-6d clean=%-6d a+2b=%-6d 2a+b=%-6d other=%-6d  [%.2f s]\n", label,
                t.n, t.clean, t.a2b, t.mirror, t.other, secs);
    if (t.has_corrupt) {
        std::printf("      first corruption: %s  (rank0=%.6f rank1=%.6f)\n", sig_name(t.first_sig),
                    static_cast<double>(t.first_pair[0]), static_cast<double>(t.first_pair[1]));
    }
}

} // namespace

int main() {
    // UNBUFFERED, because this probe takes the GPU lock and a run that stalls must be identifiable
    // from its log WHILE it is stalling. (The first run of this file block-buffered its output to a
    // pipe and the log ended mid-line, which made a 12-minute stall look like a slow cell.)
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    // A DEVICE GATE. Measured before this change: under CUDA_VISIBLE_DEVICES= this binary ran to a
    // 180 s timeout in BOTH arms, because the first device call was an unchecked
    // cudaGetDeviceProperties(&prop, 0) whose failure was ignored and whose zeroed prop let the run
    // continue into kernels that never complete. 77 is the repository's skip code. This file stays a
    // NON-ctest executable on purpose -- see the note above its add_executable() in
    // tests/CMakeLists.txt: a ctest test that grabbed the device without sh/_lock.sh would contend
    // with whichever model chain holds the lock -- so here 77 is a diagnostic, not a ctest verdict.
    {
        int gate_devices             = 0;
        const cudaError_t gate_error = cudaGetDeviceCount(&gate_devices);
        if (gate_error == cudaErrorNoDevice || gate_error == cudaErrorInsufficientDriver ||
            (gate_error == cudaSuccess && gate_devices == 0)) {
            std::printf("SKIP: no usable CUDA device (rc 77)\n");
            return 77;
        }
    }
    std::printf("=== MG4 tp2 in-place sum STRESS + FIX (ONE device: no number here is a two-card "
                "number) ===\n");
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    int driver = 0, runtime = 0;
    cudaDriverGetVersion(&driver);
    cudaRuntimeGetVersion(&runtime);
    std::printf("device : %s sm_%d%d, %d SMs\n", prop.name, prop.major, prop.minor,
                prop.multiProcessorCount);
    std::printf("driver : %d / runtime %d\n", driver, runtime);
    std::printf("header : core/tp_transport.h with the slotted protocol (section 3b/3c/4b)\n");
    bool probe_ok = false;
    const PeerAccessMatrix matrix = probe_peer_access(1, &probe_ok);
    const TopologyReport topology = decide_topology(matrix, 2, probe_ok);
    std::printf("topology: %s\n", std::string(topology_decision_name(topology.decision)).c_str());
    std::printf("the arm under test: ReduceOp::SumInPlace, i.e. the shape "
                "tools/mgpu2/tp2_selfcheck.cu's phase 1 uses.\n\n");

    const long long iters_small = env_ll("NINFER_MG4_ITERS", 200ll);
    const long long iters_zero  = env_ll("NINFER_MG4_ITERS_ZERO", 2000ll);
    const long long arm1_reps   = env_ll("NINFER_MG4_ARM1_REPS", 12ll);
    const long long back_to_back_calls = env_ll("NINFER_MG4_B2B_CALLS", 8ll);

    // ---------------- §0 ----------------
    std::printf("--- section 0: the snapshot property, MEASURED (60 ms of work behind each event) "
                "---\n");
    const SnapshotMeasurement snap = measure_snapshot();
    std::printf("  record enqueued BEFORE the wait, 60 ms of work behind it : wait cost %8.3f ms\n",
                snap.record_before_wait_ms);
    std::printf("  an event that was NEVER recorded                        : wait cost %8.3f ms\n",
                snap.never_recorded_ms);
    std::printf("  the SAME record enqueued AFTER the wait                 : wait cost %8.3f ms\n",
                snap.record_after_wait_ms);
    check(snap.record_before_wait_ms > 10.0,
          "a wait issued AFTER the record IS a barrier: the control that makes the next two numbers "
          "mean something");
    check(snap.never_recorded_ms < 5.0,
          "a wait on an event with no pending record does not block, so the FIRST pair of a call is a "
          "no-op whenever the peer has not reached its record yet");
    check(snap.record_after_wait_ms < 5.0,
          "and the SAME record, enqueued after the wait, is invisible to it: the barrier a "
          "collective's second event pair is supposed to be depends on HOST ISSUE ORDER");
    std::printf("\n");

    // ---------------- §0b / §0c ----------------
    std::printf("--- section 0b: the amplifier's own calibration (what the skew ACTUALLY costs) ---\n");
    const SpinCalibration cal = calibrate_spin();
    for (int i = 0; i < 5; ++i) {
        std::printf("  requested %9.0f ns -> measured %10.1f us  (%.1fx)\n", cal.requested_ns[i],
                    cal.measured_us[i],
                    cal.requested_ns[i] > 0.0 ? cal.measured_us[i] * 1000.0 / cal.requested_ns[i] : 0.0);
    }
    std::printf("\n");

    std::printf("--- section 0c: can a kernel on one stream observe a store from another stream's "
                "kernel WHILE RESIDENT? ---\n");
    const CoScheduling cosched = measure_cross_stream_coscheduling();
    std::printf("  a 200 ms-bounded polling kernel on stream A vs a one-thread store on stream B: "
                "%s (%.1f ms wall)\n",
                cosched.observed ? "OBSERVED" : (cosched.timed_out ? "TIMED OUT -- no kernel "
                                                                     "concurrency across streams here"
                                                                   : "INDETERMINATE"),
                cosched.ms);
    std::printf("  -> this decides WHICH mechanism of the fix is usable on this stack: DeviceAcquire "
                "needs\n     the kernel form of concurrency for both of its barriers, StreamValue "
                "needs none.\n\n");
    check(cosched.observed,
          "the platform co-schedules a polling kernel with another stream's store, so the "
          "kernel-spin mechanism is usable here (if this FAILS, only the stream-value mechanism is)");
    std::printf("\n");

    std::printf("--- section 0d: can a VALUE wait on one stream be satisfied by a write from "
                "another? ---\n");
    const ValueWaitAnswers vw = measure_value_wait_reach();
    std::printf("  (a) same stream, write then wait                        : %s (%.1f ms)\n",
                vw.same_stream ? "SATISFIED" : "NOT SATISFIED within 2 s", vw.same_stream_ms);
    std::printf("  (b) cross stream, the write FIRST and sync'd (value in memory): %s (%.1f ms)\n",
                vw.cross_stream_value_present ? "SATISFIED" : "NOT SATISFIED within 2 s",
                vw.cross_stream_value_present_ms);
    std::printf("  (c) cross stream, wait first then the write                 : %s (%.1f ms)\n",
                vw.cross_stream_write_after ? "SATISFIED" : "NOT SATISFIED within 2 s",
                vw.cross_stream_write_after_ms);
    std::printf("      %s\n", vw.errors.c_str());
    check(vw.same_stream, "a value wait on the SAME stream as its write is satisfied");
    std::printf("  -> (b) is the decisive cell: if a wait cannot be satisfied by a value that is "
                "already in\n     memory and was written from another stream, this mechanism cannot "
                "carry a cross-rank\n     barrier here at all, whatever the API documents.\n\n");

    // ---------------- §5, EARLY ON PURPOSE ----------------
    // It cannot hang (no waits, every step host-synchronised) and it is the diagnostic that decides
    // what a §1/§2 anomaly is, so it runs before the arms that can be slow.
    std::printf("--- section 5: the MAXIMALLY-ORDERED control -- BOTH pulls, host sync, then both "
                "combines, host sync, swept over the element count ---\n");
    std::printf("  the question it answers: §1's 262144-element cells come back at the ranks' INITIAL "
                "values,\n  which no ordering of one pull and one combine can produce. If THIS is "
                "clean at every size,\n  the anomaly is about ordering; if it is not, no barrier "
                "fixes it.\n");
    for (int elements : {4096, 16384, 65536, 262144, 1048576}) {
        const double c0 = now_s();
        const SyncedResult sr = fully_synced_call(elements);
        std::printf("  elements=%-8d rank0=%.6f rank1=%.6f stage0=%.6f stage1=%.6f  "
                    "bad_elements=%d stage_mismatch=%d  [%.2f s]\n",
                    elements, sr.r0, sr.r1, sr.s0, sr.s1, sr.bad_elements, sr.stage_mismatch,
                    now_s() - c0);
        check(sr.ran && sr.bad_elements == 0 && sr.stage_mismatch == 0,
              "with a host synchronisation after EVERY step the sum is exact and BOTH stagings hold "
              "the peer's source, element for element, at this size");
    }
    std::printf("\n");

    // ---------------- §6 ----------------
    std::printf("--- section 6: WHICH STEP FAILS AT SIZE -- the transport's own copy, swept, every "
                "cell host-synchronised ---\n");
    for (int elements : {4096, 16384, 65536, 262144, 1048576}) {
        const CopySweep cs = copy_sweep(elements);
        std::printf("  elements=%-8d bytes=%-9zu H2D_load_bad=%-8d async_D2D_bad=%-8d "
                    "sync_D2D_bad=%-8d\n", elements, cs.bytes, cs.h2d_bad, cs.async_d2d_bad,
                    cs.sync_d2d_bad);
        check(cs.h2d_bad == 0, "the HOST-to-DEVICE load lands every element at this size");
        check(cs.async_drained && cs.async_d2d_bad == 0,
              "the seam's own ASYNC pull delivers every element at this size");
        check(cs.sync_d2d_bad == 0, "and so does the SYNCHRONOUS D2D copy: so the failing step is "
                                    "named rather than suspected");
    }
    std::printf("\n");

    // ---------------- §7 ----------------
    std::printf("--- section 7: the step that fails, NAMED (every return value checked, readback after "
                "every step) ---\n");
    for (int elements : {16384, 65536, 262144}) {
        const StepTrace st = step_trace(elements);
        std::printf("  elements=%-8d\n", elements);
        std::printf("    load      err=%-2d bad=%-8d\n", st.load_err, st.load_bad);
        std::printf("    pull0     err=%-2d sync=%-2d | pull1 err=%-2d sync=%-2d\n", st.pull0_err,
                    st.sync0_err, st.pull1_err, st.sync1_err);
        std::printf("    staging   stage0_bad=%-8d (st0=%.6f)  stage1_bad=%-8d (st1=%.6f)\n",
                    st.stage0_bad, static_cast<double>(st.st0), st.stage1_bad,
                    static_cast<double>(st.st1));
        std::printf("    add       err0=%-2d err1=%-2d sync0=%-2d sync1=%-2d\n", st.add0_err,
                    st.add1_err, st.add0_sync, st.add1_sync);
        std::printf("    result    final0_bad=%-8d (r0=%.6f)  final1_bad=%-8d (r1=%.6f)\n",
                    st.final0_bad, static_cast<double>(st.r0), st.final1_bad,
                    static_cast<double>(st.r1));
        check(st.ran && st.load_err == 0 && st.load_bad == 0,
              "the HOST-to-DEVICE load lands every element at this size");
        check(st.pull0_err == 0 && st.pull1_err == 0 && st.sync0_err == 0 && st.sync1_err == 0,
              "BOTH pulls return success AND their streams synchronise at this size");
        check(st.stage0_bad == 0 && st.stage1_bad == 0,
              "and the staging holds the peer's source, element for element, after the pulls");
        check(st.add0_err == 0 && st.add1_err == 0,
              "both in-place combines launch without error at this size");
        check(st.final0_bad == 0 && st.final1_bad == 0,
              "and the sums are exact element for element: so THIS size is correct end to end");
    }
    std::printf("\n");

    // ---------------- §1 ----------------
    std::printf("--- section 1: arm 1's phase 1 replicated, both element counts, VALUES PRINTED ---\n");
    std::printf("  the probe (_mgpu2_tp2_selfcheck, 481d971d4161dba4) asserted `the in-place sum\n"
                "  equals a + b elementwise` = %.6f on a run of TWO repetitions whose clean value is\n"
                "  2(a+b) = %.6f. 12 replicas per cell; every pair is printed.\n", kIdeal1, kIdeal2);
    for (int reps : {1, 2}) {
        for (int elements : {4096, 262144}) {
            const double ref = clean_after_one_slot_calls(reps);
            std::printf("  reps=%d elements=%-7d the CLEAN value of one slot after %d calls = %.6f\n",
                        reps, elements, reps, ref);
            int dev = 0, nclean = 0, nsig2a3b = 0;
            for (int r = 0; r < arm1_reps; ++r) {
                const Arm1Replica rep = run_arm1_replica(elements, reps);
                const bool clean = (rep.r0 == ref && rep.r1 == ref);
                if (clean) { ++nclean; }
                // The named wrong-order signature carried through two calls, so a reader can see
                // whether the corrupt cells are the MECHANISM rather than arbitrary garbage.
                const bool sig2a3b = (rep.r0 == kSig2A3B && rep.r1 == kSig2A3B);
                if (sig2a3b) { ++nsig2a3b; }
                dev += rep.deviating_elements;
                std::printf("    #%-2d rank0=%.6f rank1=%.6f  stage0=%.6f stage1=%.6f  %s\n", r + 1,
                            rep.r0, rep.r1, rep.s0, rep.s1,
                            clean ? "== the clean reference"
                                  : (sig2a3b ? "== 2a+3b (the wrong-order signature at 2 calls)"
                                             : "<== NOT the clean reference"));
            }
            std::printf("    -> %d/%lld replicas equal the clean reference; %d match 2a+3b (=%.6f); "
                        "%d element comparisons deviate\n\n", nclean, arm1_reps, nsig2a3b, kSig2A3B,
                        dev);
        }
    }

    // ---------------- §2 ----------------
    std::printf("--- section 2: THE STRESS -- old path vs both mechanisms of the fix ---\n");
    World w;
    w.make(4096);
    PeerEvents ev;
    ev.create();
    TpSlotChannel ch[2];
    ch[0].rank = 0;
    ch[1].rank = 1;
    check(ch[0].create() && ch[1].create(),
          "the two channels' words are allocated and ZEROED once, which is what makes the first "
          "epoch of 1 distinguishable from 'no call has happened'");
    ch[0].link_peer(ch[1]);
    ch[1].link_peer(ch[0]);

    // The amplifier's amplitude list. Bounded by NINFER_MG4_MAX_SKEW_NS so that an instrumented
    // run can be kept short without editing the file, and so that a cell that turns out to be
    // pathologically slow can be excluded BY NAME in the log rather than by a silent default.
    const unsigned long long max_skew =
        static_cast<unsigned long long>(env_ll("NINFER_MG4_MAX_SKEW_NS", 1000000ll));
    std::vector<unsigned long long> skews;
    for (unsigned long long cand : {0ull, 1000ull, 10000ull, 100000ull, 1000000ull, 2000000ull}) {
        if (cand <= max_skew) { skews.push_back(cand); }
    }
    std::printf("  skew list: ");
    for (unsigned long long cand : skews) { std::printf("%llu ", static_cast<unsigned long long>(cand)); }
    std::printf("ns (max %llu)\n", static_cast<unsigned long long>(max_skew));
    std::printf("  the amplifier is a %%globaltimer spin on the PEER's stream between the two ranks' "
                "calls:\n  a controllable LEAD. skew=0 is the natural timing, which is the only way "
                "'nondeterministic' becomes a number.\n\n");

    std::printf("  OLD PATH: allreduce_sum_2rank + the caller's in-place combine, ONE data slot\n");
    for (unsigned long long sk : skews) {
        const int n = (sk == 0ull) ? static_cast<int>(iters_zero) : static_cast<int>(iters_small);
        const double cell0 = now_s();
        Tally t;
        for (int it = 0; it < n; ++it) {
            old_iteration(w, ev, sk, 1);
            float v[2] = {0.0f, 0.0f};
            w.readback(v, 0);
            tally(t, v[0], v[1], true);
        }
        char label[96];
        std::snprintf(label, sizeof(label), "old, skew=%llu ns", static_cast<unsigned long long>(sk));
        print_tally(label, t, now_s() - cell0);
        if (sk == skews.back()) {
            check(t.n > 0 && t.clean == 0,
                  "at the largest skew the OLD path is corrupt in EVERY iteration: the intermittent "
                  "is now deterministic, so a fix is falsifiable");
        }
    }
    std::printf("\n");

    // THE DEVICE-ACQUIRE MECHANISM IS NOT ATTEMPTED BY DEFAULT, and the reason is MEASURED rather
    // than assumed: §0c shows this platform does not co-schedule a polling kernel with another
    // stream's store, so a kernel-spin barrier cannot be satisfied here -- and its spin is unbounded
    // by construction (a barrier must wait), so attempting it does not fail, it WEDGES the context
    // and every later cell with it. That is what happened on the third run. The gate exists so the
    // attempt is a decision with a record behind it; the mechanism itself is left in the header
    // because on a platform that does co-schedule kernels it is the one that needs no driver API.
    const long long try_device_acquire = env_ll("NINFER_MG4_TRY_DEVICE_ACQUIRE", 0ll);
    for (TpSignalMechanism m : {TpSignalMechanism::StreamValue, TpSignalMechanism::DeviceAcquire}) {
        if (m == TpSignalMechanism::DeviceAcquire && try_device_acquire == 0) {
            std::printf("  NEW PATH: %s -- NOT ATTEMPTED (set NINFER_MG4_TRY_DEVICE_ACQUIRE=1 to try)\n",
                        std::string(tp_signal_mechanism_name(m)).c_str());
            std::printf("      reason, MEASURED in section 0c, not assumed: a kernel on stream A did "
                        "not observe a\n      store from a kernel on stream B inside 200 ms, so this "
                        "platform does not co-schedule\n      kernels across streams and a kernel "
                        "barrier cannot be satisfied here. Its spin is\n      unbounded by "
                        "construction, so attempting it WEDGES the context rather than failing.\n"
                        "      NOT MEASURED on this stack.\n\n");
            continue;
        }
        std::printf("  NEW PATH: %s -- the same serial order, the same amplifier\n",
                    std::string(tp_signal_mechanism_name(m)).c_str());
        CUresult cu = CUDA_SUCCESS;
        bool all_clean = true;
        bool measurable = true;
        int cells_measured = 0;
        for (unsigned long long sk : skews) {
            if (!measurable) { break; }
            const int n = (sk == 0ull) ? static_cast<int>(iters_zero) : static_cast<int>(iters_small);
            const double cell0 = now_s();
            Tally t;
            std::uint32_t slot = 0u;
            const double per_iter_budget = 2.0; // seconds; a clean iteration is well under a ms
            for (int it = 0; it < n; ++it) {
                const double it0 = now_s();
                if (!new_iteration(w, ch, m, sk, 1, &cu, &slot)) {
                    std::printf("      [iter %d/%d] the two streams did NOT drain within %.1f s: "
                                "%s CANNOT BE MEASURED on this stack at skew=%llu ns\n",
                                it + 1, n, per_iter_budget,
                                std::string(tp_signal_mechanism_name(m)).c_str(),
                                static_cast<unsigned long long>(sk));
                    measurable = false;
                    break;
                }
                if (now_s() - it0 > per_iter_budget) {
                    std::printf("      [iter %d/%d] took %.2f s, over the %.1f s budget: abandoning "
                                "this mechanism's remaining cells\n",
                                it + 1, n, now_s() - it0, per_iter_budget);
                    measurable = false;
                    break;
                }
                float v[2] = {0.0f, 0.0f};
                w.readback(v, slot);
                tally(t, v[0], v[1], true);
            }
            char label[96];
            std::snprintf(label, sizeof(label), "new, skew=%llu ns",
                          static_cast<unsigned long long>(sk));
            print_tally(label, t, now_s() - cell0);
            if (measurable) {
                ++cells_measured;
                all_clean = all_clean && (t.clean == t.n);
            }
        }
        if (measurable) {
            check(all_clean,
                  "the fix is CLEAN in EVERY iteration at EVERY skew, including the skews at which "
                  "the old path is corrupt in every iteration");
            check(cu == CUDA_SUCCESS, "every order-independent operation returned CUDA_SUCCESS");
        } else {
            std::printf("      [not measured] this mechanism did not drain inside the bound at "
                        "%d of %zu cells\n", cells_measured, skews.size());
        }
        std::printf("\n");
    }

    // The back-to-back arm: K calls with NO host synchronisation between them, against the exact
    // simulated reference. This is the shape the donor's post-condition is about ("an unbounded
    // sequence of calls sharing the same buffers ... needs no host synchronization") and the shape a
    // captured graph replays.
    {
        const int K = static_cast<int>(back_to_back_calls);
        const CleanModel cm = simulate_clean(K);
        const std::uint32_t topslot = static_cast<std::uint32_t>((K - 1) % 2);
        std::printf("  BACK-TO-BACK: K=%d calls, no host sync between them\n", K);
        std::printf("    the exact reference by simulating A<-A+B, B<-B+A per call: rank0=%.6f "
                    "rank1=%.6f on slot %u\n", cm.a[topslot], cm.b[topslot], topslot);
        for (TpSignalMechanism m : {TpSignalMechanism::StreamValue, TpSignalMechanism::DeviceAcquire}) {
            if (m == TpSignalMechanism::DeviceAcquire && try_device_acquire == 0) {
                std::printf("    %-44s NOT ATTEMPTED: section 0c measured no cross-stream kernel "
                            "concurrency, and\n%s\n", std::string(tp_signal_mechanism_name(m)).c_str(),
                            "      its spin is unbounded by construction, so it would WEDGE the "
                            "context rather than fail. NOT MEASURED.");
                continue;
            }
            CUresult cu = CUDA_SUCCESS;
            int exact = 0;
            const int n = 40;
            std::uint32_t slot = 0u;
            double last0 = 0.0, last1 = 0.0;
            bool drained = true;
            for (int it = 0; it < n; ++it) {
                drained = new_iteration(w, ch, m, 0ull, K, &cu, &slot);
                if (!drained) { break; }
                float v[2] = {0.0f, 0.0f};
                w.readback(v, slot);
                last0 = static_cast<double>(v[0]);
                last1 = static_cast<double>(v[1]);
                if (last0 == cm.a[topslot] && last1 == cm.b[topslot]) { ++exact; }
            }
            std::printf("    %-44s %d/%d iterations equal the exact reference (last %.6f/%.6f)%s\n",
                        std::string(tp_signal_mechanism_name(m)).c_str(), exact, n, last0, last1,
                        drained ? "" : "  [DID NOT DRAIN -- not measured]");
            if (drained) {
                check(exact == n,
                      "every back-to-back sequence matches the EXACT simulated reference, so the fix "
                      "is a deterministic function of the state and not merely 'not corrupt'");
            }
        }
        // The same shape on the OLD path, for the contrast: it is reported, not asserted, because a
        // short run of an arm that is clean by luck is exactly what this line exists to refuse.
        {
            int exact = 0;
            const int n = 40;
            double last0 = 0.0, last1 = 0.0;
            for (int it = 0; it < n; ++it) {
                old_iteration(w, ev, 0ull, K);
                float v[2] = {0.0f, 0.0f};
                w.readback(v, 0);
                last0 = static_cast<double>(v[0]);
                last1 = static_cast<double>(v[1]);
                // The old path uses slot 0 only, so the reference is the K-th power on one slot.
                const double ref = static_cast<double>(kA + kB) * std::pow(2.0, K - 1);
                if (last0 == ref && last1 == ref) { ++exact; }
            }
            std::printf("    %-44s %d/%d iterations equal the clean reference (last %.6f/%.6f)\n",
                        "old path, same K", exact, n, last0, last1);
        }
        std::printf("\n");
    }

    // The cost question, answered for the WRONG thing on purpose: on one device with one host thread
    // the fix's waits serialise the two ranks, so this is a SERIALIZATION figure.
    {
        const int n = 300;
        cudaEvent_t t0 = nullptr, t1 = nullptr;
        cudaEventCreate(&t0);
        cudaEventCreate(&t1);
        std::uint32_t slot = 0u;
        CUresult cu = CUDA_SUCCESS;
        cudaEventRecord(t0, w.stream[0]);
        for (int it = 0; it < n; ++it) { old_iteration(w, ev, 0ull, 1); }
        cudaEventRecord(t1, w.stream[0]);
        cudaStreamSynchronize(w.stream[0]);
        float ms_old = 0.0f;
        cudaEventElapsedTime(&ms_old, t0, t1);
        cudaEventRecord(t0, w.stream[0]);
        for (int it = 0; it < n; ++it) {
            new_iteration(w, ch, TpSignalMechanism::StreamValue, 0ull, 1, &cu, &slot);
        }
        cudaEventRecord(t1, w.stream[0]);
        cudaStreamSynchronize(w.stream[0]);
        float ms_new = 0.0f;
        cudaEventElapsedTime(&ms_new, t0, t1);
        cudaEventRecord(t0, w.stream[0]);
        bool timed_da = false;
        if (try_device_acquire != 0) {
            timed_da = true;
            for (int it = 0; it < n; ++it) {
                if (!new_iteration(w, ch, TpSignalMechanism::DeviceAcquire, 0ull, 1, &cu, &slot)) {
                    timed_da = false;
                    break;
                }
            }
        }
        cudaEventRecord(t1, w.stream[0]);
        cudaStreamSynchronize(w.stream[0]);
        float ms_da = 0.0f;
        cudaEventElapsedTime(&ms_da, t0, t1);
        std::printf("  per-iteration time on ONE device with ONE host thread, %d iterations each:\n"
                    "    old path %.1f us  |  stream-value %.1f us  |  device-acquire %.1f us%s\n"
                    "    ⚠️ this is a SERIALIZATION figure, not a cost figure: the value waits make "
                    "rank 1's\n    device phase wait for rank 0's publish, which ONE thread issues "
                    "second. With one thread\n    per rank the two waits are satisfied "
                    "concurrently. Cost is pending hardware.\n\n",
                    n,
                    static_cast<double>(ms_old) * 1000.0 / n,
                    static_cast<double>(ms_new) * 1000.0 / n,
                    static_cast<double>(ms_da) * 1000.0 / n,
                    timed_da ? "" : "  [device-acquire NOT MEASURED: see section 0c]");
        cudaEventDestroy(t0);
        cudaEventDestroy(t1);
    }

    // (§5 already ran, early, right after §0d: it cannot hang and it decides what a §1/§2 anomaly
    // is, so it is reported before the arms that can be slow.)
    std::printf("--- section 3: the controls that must stay green ---\n");
    {
        const int n = 200;
        const double cell0 = now_s();
        Tally t;
        for (int it = 0; it < n; ++it) {
            forkjoin_iteration(w, ev);
            float v[2] = {0.0f, 0.0f};
            w.readback(v, 0);
            tally(t, v[0], v[1], true);
        }
        print_tally("control A: fork/join order (old primitives)", t, now_s() - cell0);
        check(t.clean == t.n, "the fork/join order of the old primitives is clean in EVERY iteration");
    }
    {
        std::printf("  control B: the NON-IN-PLACE AND-reduce, same serial order, same amplifier\n");
        bool all_clean = true;
        for (unsigned long long sk : skews) {
            const int n = (sk == 0ull) ? static_cast<int>(iters_zero) : static_cast<int>(iters_small);
            const double cell0 = now_s();
            const AndControl ac = and_control(n, sk);
            char label[96];
            std::snprintf(label, sizeof(label), "AND, skew=%llu ns",
                          static_cast<unsigned long long>(sk));
            print_tally(label, ac.t, now_s() - cell0);
            std::printf("      both ranks bitwise equal: %s | first verdict %u (ADMIT) | "
                        "disagreements with the host fold: %d\n",
                        ac.ranks_equal ? "yes" : "NO", ac.verdict0, ac.host_disagreements);
            check(ac.ranks_equal, "the AND-reduce's two ranks agree bitwise at this skew");
            check(ac.host_disagreements == 0,
                  "and the device verdict equals the host fold (and_reduce_wire({1,1}) -> ADMIT)");
            check(ac.t.other == 0 && ac.t.clean == ac.t.n,
                  "the non-in-place AND-reduce is CLEAN on the very ordering that corrupts the sum");
            all_clean = all_clean && (ac.t.clean == ac.t.n);
        }
        check(all_clean, "the AND control is clean at every skew");
    }
    std::printf("\n");

    // ---------------- §4 ----------------
    std::printf("--- section 4: the capture questions, re-measured ---\n");
    const CaptureAnswers ca = answer_capture_questions();
    std::printf("  cross-stream event wait INSIDE capture : %s (%s)\n",
                ca.fork_join_accepted ? "ACCEPTED" : "REFUSED", ca.fork_join_err.c_str());
    std::printf("  cudaMemcpyPeerAsync INSIDE capture     : %s (%s)\n",
                ca.memcpy_peer_refused ? "REFUSED" : "not refused", ca.memcpy_peer_err.c_str());
    std::printf("  cuStreamWriteValue32 INSIDE capture    : %s (%s)\n",
                ca.write_value_capture == CUDA_SUCCESS ? "ACCEPTED" : "REFUSED",
                ca.write_value_err.c_str());
    std::printf("  cuStreamWaitValue32  INSIDE capture    : %s (%s)\n",
                ca.wait_value_capture == CUDA_SUCCESS ? "ACCEPTED" : "REFUSED",
                ca.wait_value_err.c_str());
    std::printf("  the value ops EAGER                     : %s\n", ca.value_op_eager.c_str());
    std::printf("  device release/acquire KERNEL in capture: %s (%s), replay %s\n",
                ca.spin_kernel_captured ? "CAPTURED" : "not captured", ca.spin_kernel_err.c_str(),
                ca.spin_graph_replayed ? "ok" : "no");
    check(ca.fork_join_accepted, "the cross-stream event wait is accepted inside a capture (NECESSARY "
                                 "for the cross-device form, not sufficient)");
    check(ca.memcpy_peer_refused, "cudaMemcpyPeerAsync is refused inside a capture on THIS stack");
    check(ca.spin_kernel_captured && ca.spin_graph_replayed,
          "the device-side carrier (the release/acquire kernel) captures and replays");
    check(ca.write_value_capture == CUDA_SUCCESS && ca.wait_value_capture == CUDA_SUCCESS,
          "both stream-ordered memory operations are accepted inside a capture, so the cheaper "
          "mechanism is admissible in a captured graph on this stack");
    std::printf("\n");

    ch[0].destroy();
    ch[1].destroy();
    ev.destroy();
    w.destroy();

    std::printf("tp2_sum_stress: %d checks, %d failures -> %s\n", g_checks, g_failures,
                g_failures == 0 ? "PASS" : "FAIL");
    return g_failures == 0 ? 0 : 1;
}
