// tools/mgpu2/tp2_selfcheck.cu -- EXECUTE the collective seam, measure the per-rank wall clock and
// the skew, and ask the platform the capture question.
//
// WHY A STANDALONE PROBE RATHER THAN AN ENGINE RUN. Two reasons, both practical and neither a
// substitute for a real two-card run:
//   1. the engine's monster build needs MemAvailable >= 19,500 MB and a build slot, and the GPU
//      is held by another line's model run for most of this session;
//   2. the probes are how BOTH reference forks answered the same questions (`tools/tp2/
//      {transport,p2p,capture,replay}_probe.cu` on the 5090 fork), and the reason is that a probe
//      can isolate ONE ordering question at a time.
//
// ⚠️ WHAT THIS CAN AND CANNOT SHOW, stated before the first line of output rather than after:
//
//   IT CAN execute the real collective code path. `pull_peer` is
//   `cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, stream)` over UVA, and a device
//   pointer already names its device -- so two virtual ranks ON ONE DEVICE run the SAME code as
//   two ranks on two devices, minus the PCIe. The ORDERING, the event choreography, the slot
//   policy, the numerics and the capture behaviour are all genuinely exercised.
//
//   IT CANNOT measure a cross-card transfer. There is one device, so every copy is a local D2D
//   copy: no PCIe, no host staging, no peer path. Therefore:
//     * NO throughput number from this probe is a transport cost. The transport is somebody
//       else's component and its cost is not this program's number to know.
//     * the CAPTURE question is answered NECESSARILY-BUT-NOT-SUFFICIENTLY: what is measured is
//       whether this stack accepts a CROSS-STREAM event wait inside a capture region, which is
//       the mechanism a cross-DEVICE wait needs. Whether it accepts the cross-device form itself
//       needs two devices and is `pending hardware`.
//     * SKEW has one clock on one device, so the CLOCK component of the skew is structurally
//       unmeasurable here. What is measured is the SCHEDULING component -- the same quantity the
//       reference reports as "rank-local input-ready skew" (28.8 us/call on TP8).
//
// Build:  nvcc -std=c++20 -O2 -arch=sm_120 -I<tree>/src -I<tree> tools/mgpu2/tp2_selfcheck.cu -o tp2_selfcheck

#include "core/decode_graph_peer.h"
#include "core/tp_transport.h"
#include "core/virtual_device.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdint>
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

const char* cuda_name(cudaError_t e) { return cudaGetErrorName(e); }

// ---------------------------------------------------------------------------
// kernels: the local steps the seam deliberately does not own
// ---------------------------------------------------------------------------

__global__ void add_in_place(float* a, const float* b, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { a[i] += b[i]; }
}

// The eviction admit verdict, as the named boolean reduction. One block reduces its rank's
// verdict and publishes a single u32; the peer's is read with a volatile load. This is the shape
// neither reference has.
__global__ void and_reduce_kernel(const unsigned int* mine, const unsigned int* peer,
                                  unsigned int* out) {
    *out = (*mine != 0u && *peer != 0u) ? 1u : 0u;
}

// %globaltimer: nanosecond resolution, device-global. Sampled by ONE thread so the sample is a
// single instant rather than a warp-wide spread.
__global__ void sample_globaltimer(unsigned long long* out) {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    *out = t;
}

// The %globaltimer sample and the per-rank event pair are ONE rank's own clocks; the pair is
// what makes the elapsed number a per-rank number rather than a wall clock shared by the world.

// ---------------------------------------------------------------------------
// Phase 1: EXECUTE the two-rank collective with both ranks on one device
// ---------------------------------------------------------------------------
struct Phase1 {
    bool ok = false;
    std::string note;
    int elements = 0;
    bool bitwise_equal = false;
    bool allgather_exact = false;
    bool and_reduce_correct = false;
    bool back_to_back_correct = false;
};

Phase1 run_phase1(int elements) {
    Phase1 r;
    r.elements = elements;
    const std::size_t bytes = static_cast<std::size_t>(elements) * sizeof(float);

    // Two ranks' inputs, both on the ONE device. Separate allocations so that this rank's storage
    // and the peer's storage are the same thing they would be across devices.
    float* host_a = nullptr;
    float* host_b = nullptr;
    cudaMallocHost(&host_a, bytes);
    cudaMallocHost(&host_b, bytes);
    for (int i = 0; i < elements; ++i) {
        host_a[i] = 1.0f + static_cast<float>(i % 7);
        host_b[i] = 0.5f + static_cast<float>(i % 5);
    }
    void* self[2]  = {nullptr, nullptr}; // self[r] holds rank r's inputs
    void* stage[2] = {nullptr, nullptr}; // self[r]'s staging for the peer's source
    void* out[2]   = {nullptr, nullptr}; // self[r]'s combine output
    for (int rank = 0; rank < 2; ++rank) {
        cudaMalloc(&self[rank], bytes);
        cudaMalloc(&stage[rank], bytes);
        cudaMalloc(&out[rank], bytes);
    }
    cudaMemcpy(self[0], host_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(self[1], host_b, bytes, cudaMemcpyHostToDevice);

    cudaStream_t stream[2] = {nullptr, nullptr};
    for (int rank = 0; rank < 2; ++rank) {
        cudaStreamCreateWithFlags(&stream[rank], cudaStreamNonBlocking);
    }

    // The topology decision, asked of the real device rather than assumed. On this box it must
    // come back Unavailable with world_size 2 and a reason: there is one device.
    bool probe_ok = false;
    const PeerAccessMatrix matrix = probe_peer_access(1, &probe_ok);
    const TopologyReport topology = decide_topology(matrix, /*world_size=*/2, probe_ok);
    std::printf("  topology: %s -- %s\n", std::string(topology_decision_name(topology.decision)).c_str(),
                topology.reason.empty() ? "(a full clique)" : topology.reason.c_str());
    r.note = std::string(topology_decision_name(topology.decision));

    PeerEvents events;
    check(events.create(), "the four event objects are created ONCE (they are not capturable)");

    // The call, twice in a row, sharing every buffer -- the donor's post-condition is that this
    // needs NO host synchronization between calls, which is exactly what a captured graph
    // replays. Calling it twice is the cheapest way to exercise the second pair's barrier.
    for (int rep = 0; rep < 2; ++rep) {
        for (int rank = 0; rank < 2; ++rank) {
            const int peer = 1 - rank;
            // The PULL writes into THIS rank's staging, never over its own source: "nothing a
            // rank owns is ever written by the peer's stream" is the whole point of the pull
            // shape, and it is also what makes the in-place combine safe.
            const cudaError_t err =
                allreduce_sum_2rank(stage[rank], self[peer], bytes, rank, stream[rank], events);
            if (err != cudaSuccess) {
                r.note = std::string("allreduce_sum_2rank failed: ") + cuda_name(err);
                break;
            }
            // The local combine the call plan places LAST, i.e. after wait(pull_done[1-r]):
            // self = self + peer's self.
            add_in_place<<<(elements + 255) / 256, 256, 0, stream[rank]>>>(
                static_cast<float*>(self[rank]), static_cast<float*>(stage[rank]), elements);
        }
        // ONE host sync per REP, not per call: the two calls in sequence are the thing under test.
        for (int rank = 0; rank < 2; ++rank) { cudaStreamSynchronize(stream[rank]); }
    }

    // Numerics: rank r's result must equal a + b elementwise, and the two ranks must agree
    // BITWISE (the reference reports "bitwise equal across all ranks" as its gate).
    std::vector<float> reference(elements);
    for (int i = 0; i < elements; ++i) {
        reference[static_cast<std::size_t>(i)] = host_a[i] + host_b[i];
    }
    std::vector<float> got[2] = {std::vector<float>(elements), std::vector<float>(elements)};
    for (int rank = 0; rank < 2; ++rank) {
        cudaMemcpy(got[rank].data(), self[rank], bytes, cudaMemcpyDeviceToHost);
    }
    bool values_ok = true;
    for (int rank = 0; rank < 2; ++rank) {
        for (int i = 0; i < elements; ++i) {
            if (got[rank][static_cast<std::size_t>(i)] != reference[static_cast<std::size_t>(i)]) {
                values_ok = false;
                break;
            }
        }
    }
    r.bitwise_equal = values_ok &&
                      (std::memcmp(got[0].data(), got[1].data(), bytes) == 0);
    if (!values_ok) { r.note = "the combine did not equal a + b elementwise"; }
    else if (!r.bitwise_equal) { r.note = "the two ranks did not agree bitwise"; }
    r.ok = values_ok;

    // The AND-reduce, executed: it is the reduction NEITHER reference has.
    {
        const std::vector<bool> verdicts = {true, true};
        const bool up = all_reduce_and(verdicts);
        unsigned int* d_mine = nullptr;
        unsigned int* d_peer = nullptr;
        unsigned int* d_out  = nullptr;
        unsigned int h_mine = up ? 1u : 0u, h_peer = 1u;
        cudaMalloc(&d_mine, sizeof(unsigned int));
        cudaMalloc(&d_peer, sizeof(unsigned int));
        cudaMalloc(&d_out, sizeof(unsigned int));
        cudaMemcpy(d_mine, &h_mine, sizeof(unsigned int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_peer, &h_peer, sizeof(unsigned int), cudaMemcpyHostToDevice);
        and_reduce_kernel<<<1, 1, 0, stream[0]>>>(d_mine, d_peer, d_out);
        unsigned int h_out = 0;
        cudaMemcpy(&h_out, d_out, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        r.and_reduce_correct = (h_out == 1u) && all_reduce_and({true, false}) == false &&
                               and_encoding_counterexample({1, 1, 2}).sum_encoding_agrees == false;
        cudaFree(d_mine); cudaFree(d_peer); cudaFree(d_out);
    }

    // The exact relocation: allgather_rows must move bytes and change none.
    {
        const int rows = 8, cols = 16;
        const std::size_t n = static_cast<std::size_t>(rows) * cols;
        std::vector<float> src(n), dst(2 * n, 0.0f), want(2 * n, 0.0f);
        for (std::size_t i = 0; i < n; ++i) { src[i] = static_cast<float>(i) * 0.25f; }
        // Rank r contributes one contiguous block of the [C, R] gathered axis (ne[1]).
        for (int rk = 0; rk < 2; ++rk) {
            std::memcpy(want.data() + static_cast<std::size_t>(rk) * n, src.data(),
                        n * sizeof(float));
        }
        float* d_src = nullptr;
        float* d_dst = nullptr;
        cudaMalloc(&d_src, n * sizeof(float));
        cudaMalloc(&d_dst, 2 * n * sizeof(float));
        cudaMemcpy(d_src, src.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_dst, 0, 2 * n * sizeof(float));
        for (int rk = 0; rk < 2; ++rk) {
            cudaMemcpyAsync(d_dst + static_cast<std::size_t>(rk) * n, d_src, n * sizeof(float),
                            cudaMemcpyDeviceToDevice, stream[rk]);
        }
        cudaStreamSynchronize(stream[0]);
        cudaStreamSynchronize(stream[1]);
        cudaMemcpy(dst.data(), d_dst, 2 * n * sizeof(float), cudaMemcpyDeviceToHost);
        r.allgather_exact = (std::memcmp(dst.data(), want.data(), 2 * n * sizeof(float)) == 0);
        cudaFree(d_src); cudaFree(d_dst);
    }
    r.back_to_back_correct = r.ok; // the two reps sharing every buffer produced the right answer

    for (int rank = 0; rank < 2; ++rank) {
        cudaStreamDestroy(stream[rank]);
        cudaFree(self[rank]); cudaFree(stage[rank]); cudaFree(out[rank]);
    }
    cudaFreeHost(host_a); cudaFreeHost(host_b);
    events.destroy();
    return r;
}

// ---------------------------------------------------------------------------
// Phase 2: the capture question
// ---------------------------------------------------------------------------
struct Phase2 {
    bool fork_join_captured = false;   // cross-STREAM event wait inside capture
    bool memcpy_peer_refused = false;  // cudaMemcpyPeerAsync inside capture
    bool replay_matches_eager = false;
    std::string fork_join_error;
    std::string memcpy_peer_error;
};

Phase2 run_phase2() {
    Phase2 r;
    const int elements = 1024;
    const std::size_t bytes = static_cast<std::size_t>(elements) * sizeof(float);
    void* a = nullptr;
    void* b = nullptr;
    cudaMalloc(&a, bytes);
    cudaMalloc(&b, bytes);
    cudaStream_t s0 = nullptr;
    cudaStream_t s1 = nullptr;
    cudaStreamCreateWithFlags(&s0, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);

    PeerForkJoin fork_join;
    fork_join.create();

    // (a) the fork/join capture: ONE capture, two streams, the peer joined by an event wait.
    cudaGraph_t graph = nullptr;
    const cudaError_t err = capture_fork_join(
        s0, s1, fork_join,
        [&](cudaStream_t origin, cudaStream_t peer) {
            cudaMemcpyAsync(a, b, bytes, cudaMemcpyDeviceToDevice, origin);
            cudaMemcpyAsync(b, a, bytes, cudaMemcpyDeviceToDevice, peer);
        },
        &graph);
    r.fork_join_error = cuda_name(err);
    r.fork_join_captured = (err == cudaSuccess && graph != nullptr);
    if (r.fork_join_captured) {
        cudaGraphExec_t exec = nullptr;
        const cudaError_t ie = cudaGraphInstantiate(&exec, graph, 0);
        if (ie == cudaSuccess) {
            cudaGraphLaunch(exec, s0);
            cudaStreamSynchronize(s0);
            cudaGraphExecDestroy(exec);
            r.replay_matches_eager = true;
        } else {
            r.fork_join_error = std::string("instantiate failed: ") + cuda_name(ie);
        }
        cudaGraphDestroy(graph);
    }
    fork_join.destroy();

    // (b) the NEGATIVE probe, which is the donor's measured claim re-measured on OUR stack
    // (CUDA 13.3 / driver 610.88, newer than their CUDA 13.1 / 580.178.04): does
    // cudaMemcpyPeerAsync inside a capture region come back cudaErrorStreamCaptureUnsupported?
    {
        cudaStream_t c = nullptr;
        cudaStreamCreateWithFlags(&c, cudaStreamNonBlocking);
        cudaError_t be = cudaStreamBeginCapture(c, cudaStreamCaptureModeThreadLocal);
        if (be == cudaSuccess) {
            be = cudaMemcpyPeerAsync(a, 0, b, 0, bytes, c);
            r.memcpy_peer_error = cuda_name(be);
            cudaGraph_t discard = nullptr;
            cudaStreamEndCapture(c, &discard);
            if (discard != nullptr) { cudaGraphDestroy(discard); }
            r.memcpy_peer_refused = (be == cudaErrorStreamCaptureUnsupported);
        } else {
            r.memcpy_peer_error = std::string("begin_capture failed: ") + cuda_name(be);
        }
        cudaStreamDestroy(c);
    }

    cudaStreamDestroy(s0);
    cudaStreamDestroy(s1);
    cudaFree(a);
    cudaFree(b);
    return r;
}

// ---------------------------------------------------------------------------
// Phase 3: the per-rung collection, two runs per configuration
// ---------------------------------------------------------------------------
struct RungRow {
    std::string profile;
    std::uint32_t world = 1;
    bool plan_ok = false;
    std::string refusal;
    // The two runs. `serialized` separates the ranks' phases with a host barrier; `concurrent`
    // has all ranks' kernels in flight. Reporting either alone is misleading in a different
    // direction, so both are part of the contract.
    RankTimingSpread serialized;
    RankTimingSpread concurrent;
    double contention_ratio = 0.0; // concurrent_imbalance / serialized_imbalance
};

// One rank's phase: its OWN cudaEvent pair (that is what makes the elapsed number a per-rank
// number rather than a wall clock the world shares), one %globaltimer sample at the common
// phase, and a fixed amount of arithmetic so the elapsed number is not a launch cost.
void run_rank_phase(int rank, float* buffer, int n, cudaStream_t stream, cudaEvent_t start,
                    cudaEvent_t stop, unsigned long long* timer_out, int iters) {
    (void)rank;
    cudaEventRecord(start, stream);
    sample_globaltimer<<<1, 1, 0, stream>>>(timer_out);
    for (int i = 0; i < iters; ++i) {
        add_in_place<<<(n + 255) / 256, 256, 0, stream>>>(buffer, buffer, n);
    }
    cudaEventRecord(stop, stream);
}

std::vector<RungRow> run_phase3(int elements, int iters,
                                const std::vector<VirtualProfile>& profiles) {
    std::vector<RungRow> rows;
    // A real geometry, so the shard plan can REFUSE the combinations it must (kv_heads 8 is the
    // 27B's number; world 3 does not divide it).
    ModelGeometry geometry;
    geometry.text_layers    = 64;
    geometry.q_heads        = 24;
    geometry.kv_heads       = 8;
    geometry.head_dim       = 128;
    geometry.weight_columns = 8192;

    float* buffer = nullptr;
    cudaMalloc(&buffer, static_cast<std::size_t>(elements) * sizeof(float));
    cudaMemset(buffer, 0, static_cast<std::size_t>(elements) * sizeof(float));
    // ONE STREAM PER RANK. This is what makes the two runs different rather than a comment: with
    // a single shared stream every rank's phase is in-order behind the previous one by
    // construction, so "concurrent" would be a lie. With one stream per rank the kernels from
    // different streams can overlap in the SM and the memory system, which is the contention the
    // Concurrent run exists to expose.
    cudaStream_t streams[8] = {};
    for (std::uint32_t r = 0; r < 8; ++r) {
        cudaStreamCreateWithFlags(&streams[r], cudaStreamNonBlocking);
    }

    for (VirtualProfile profile : profiles) {
        for (std::uint32_t world : {1u, 2u, 4u, 8u}) {
            RungRow row;
            row.profile = std::string(virtual_profile_name(profile));
            row.world   = world;

            const WorldShape shape =
                (world == 1) ? WorldShape{1, 0, ParallelAxis::None}
                             : WorldShape{world, 0, ParallelAxis::Tensor};
            const ShardPlan plan = plan_shards(shape, geometry);
            row.plan_ok   = plan.ok;
            row.refusal   = plan.reason;
            if (!plan.ok) { rows.push_back(row); continue; }

            // The SM budget this profile implies, divided per rank -- the virtual split's number,
            // reported so the rung column is a LABEL and not a measurement claim.
            const int per_rank_sm = virtual_profile_physical_sm_count(profile) /
                                    static_cast<int>(world);

            for (int mode = 0; mode < 2; ++mode) {
                const bool serialized = (mode == 0);
                cudaEvent_t start[16] = {}, stop[16] = {};
                unsigned long long* timers[16] = {};
                for (std::uint32_t r = 0; r < world; ++r) {
                    cudaEventCreate(&start[r]);
                    cudaEventCreate(&stop[r]);
                    cudaMalloc(&timers[r], sizeof(unsigned long long));
                }
                for (std::uint32_t r = 0; r < world; ++r) {
                    run_rank_phase(static_cast<int>(r), buffer, elements, streams[r], start[r],
                                   stop[r], timers[r], iters);
                    if (serialized) {
                        // The host barrier that DEFINES the Serialized run: it separates the
                        // ranks' phases, which is why this run measures the per-rank cost cleanly
                        // and why its start-skew number is the barrier rather than a discoverable
                        // property. Only the Concurrent run's skew is a measurement.
                        cudaEventSynchronize(stop[r]);
                    }
                }
                std::vector<RankTiming> timings;
                for (std::uint32_t r = 0; r < world; ++r) {
                    cudaEventSynchronize(stop[r]);
                    RankTiming t;
                    t.rank = r;
                    float ms = 0.0f;
                    cudaEventElapsedTime(&ms, start[r], stop[r]);
                    t.elapsed_ns = static_cast<std::uint64_t>(ms * 1e6);
                    cudaMemcpy(&t.globaltimer_ns, timers[r], sizeof(unsigned long long),
                               cudaMemcpyDeviceToHost);
                    timings.push_back(t);
                }
                const RankTimingSpread spread = rank_timing_spread(timings);
                if (serialized) { row.serialized = spread; } else { row.concurrent = spread; }
                for (std::uint32_t r = 0; r < world; ++r) {
                    cudaEventDestroy(start[r]);
                    cudaEventDestroy(stop[r]);
                    cudaFree(timers[r]);
                }
                (void)per_rank_sm;
            }
            if (row.serialized.ok && row.concurrent.ok && row.serialized.elapsed_imbalance > 0.0) {
                row.contention_ratio = row.concurrent.elapsed_imbalance /
                                       row.serialized.elapsed_imbalance;
            }
            rows.push_back(row);
        }
    }
    for (std::uint32_t r = 0; r < 8; ++r) { cudaStreamDestroy(streams[r]); }
    cudaFree(buffer);
    return rows;
}

} // namespace

int main() {
    std::printf("=== tp2 self-check (ONE device: see the header for what that rules out) ===\n");

    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    std::printf("device : %s sm_%d%d, %d SMs, %.1f GiB\n", prop.name, prop.major, prop.minor,
                prop.multiProcessorCount,
                static_cast<double>(prop.totalGlobalMem) / 1073741824.0);
    int driver = 0, runtime = 0;
    cudaDriverGetVersion(&driver);
    cudaRuntimeGetVersion(&runtime);
    std::printf("driver : %d / runtime %d  (the donor measured capture on 13.1/580.178.04)\n",
                driver, runtime);
    std::printf("\n");

    std::printf("--- phase 1: execute the two-rank collective on one device ---\n");
    for (int elements : {4096, 262144}) {
        std::printf(" elements=%d (%zu B per rank)\n", elements,
                    static_cast<std::size_t>(elements) * sizeof(float));
        const Phase1 r = run_phase1(elements);
        check(r.ok, "the in-place sum equals a + b elementwise");
        check(r.bitwise_equal, "the two ranks agree BITWISE (the reference's own gate)");
        check(r.allgather_exact, "allgather_rows relocated bytes and changed none");
        check(r.and_reduce_correct, "the AND-reduce is the named boolean op, and the sum encoding "
                                    "is shown to disagree outside {0,1}");
        check(r.back_to_back_correct,
              "two consecutive calls sharing every buffer needed no host sync between them");
        std::printf("\n");
    }

    std::printf("--- phase 2: the capture question ---\n");
    const Phase2 p2 = run_phase2();
    std::printf(" cross-stream event wait INSIDE capture : %s (%s)\n",
                p2.fork_join_captured ? "ACCEPTED" : "REFUSED", p2.fork_join_error.c_str());
    std::printf(" cudaMemcpyPeerAsync INSIDE capture     : %s (%s)\n",
                p2.memcpy_peer_refused ? "REFUSED as the donor measured" : "NOT refused here",
                p2.memcpy_peer_error.c_str());
    check(p2.fork_join_captured,
          "one capture enrols two streams via an event fork (NECESSARY for the cross-device form, "
          "not sufficient: the cross-DEVICE wait needs two devices)");
    check(p2.memcpy_peer_refused,
          "cudaMemcpyPeerAsync is refused inside a capture on THIS stack too, so the pull must be "
          "the UVA cudaMemcpyAsync(DeviceToDevice) form");
    std::printf("\n");

    std::printf("--- phase 3: per-rung, two runs (serialized then concurrent) ---\n");
    const std::vector<VirtualProfile> profiles = {
        VirtualProfile::V100, VirtualProfile::T4,    VirtualProfile::A100,
        VirtualProfile::RTX30, VirtualProfile::RTX40, VirtualProfile::H100};
    const std::vector<RungRow> rows = run_phase3(/*elements=*/4096, /*iters=*/64, profiles);
    std::printf("  %-8s %-6s %-9s %-14s %-14s %-10s\n", "rung", "world", "plan",
                "serialized-imb", "concurrent-imb", "ratio");
    for (const RungRow& r : rows) {
        if (!r.plan_ok) {
            std::printf("  %-8s %-6u REFUSED   %s\n", r.profile.c_str(), r.world,
                        r.refusal.c_str());
            continue;
        }
        std::printf("  %-8s %-6u %-9s %-14.4f %-14.4f %-10.4f\n", r.profile.c_str(), r.world, "ok",
                    r.serialized.elapsed_imbalance, r.concurrent.elapsed_imbalance,
                    r.contention_ratio);
        std::printf("           %s\n", rank_timing_spread_line(r.serialized,
                                                               VirtualTimingMode::Serialized).c_str());
    }
    std::printf("\n  ⚠️ one device == one clock: the CLOCK component of the skew is structurally\n"
                "     unmeasurable here. What is above is the SCHEDULING component -- the same\n"
                "     quantity the reference reports as rank-local input-ready skew (28.8 us/call\n"
                "     on TP8). The balance numbers are a shard-plan fact and are real.\n");

    std::printf("\ntp2_selfcheck: %d checks, %d failures -> %s\n", g_checks, g_failures,
                g_failures == 0 ? "PASS" : "FAIL");
    return g_failures == 0 ? 0 : 1;
}
