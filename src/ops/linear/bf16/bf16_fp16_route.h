#pragma once

// ---------------------------------------------------------------------------
// bf16_fp16_route.h -- THE ENGINE'S fp16-PLANE ROUTE DECISION (host only).
// ---------------------------------------------------------------------------
// The counterpart of ops/linear/qpn/qpn_arch_route.h for the OTHER fp16 family. It exists for
// the same measured reason that one does: the route layer had no engine consumer, so the six
// simulated arms produced byte-identical token ids and the identity proved nothing.
// ops/linear/qpn/qpn_arch_route.h:7-14 records that measurement; this file does not restate it,
// it follows it.
//
// WHAT IT DECIDES, and it is ONE question: for THIS (rung, tokens, n, k), does the BF16 linear
// take the fp16 tensor-core arm that src/ops/linear/bf16/bf16_mma_fp16.cuh implements?
//
// THREE PROPERTIES CARRIED OVER FROM THE QPN SIBLING, all load-bearing:
//   * ON A SHIPPING BUILD IT IS A NO-OP. With no environment set, arch_view_for_device()
//     returns Disabled, effective_sm == the card's own, and select_route(sm_120, BF16, ..)
//     answers mma-bf16 -- rung 120 HAS Cap::Bf16Mma, so the floor is met and this arm is never
//     reached. So use_arm is false, no stderr line is printed, and the native shape table in
//     bf16_dispatch.cpp runs unchanged.
//   * A REFUSED SIMULATION IS NOT A LICENCE TO ANSWER FOR THE REAL DEVICE. `refused` is
//     carried and the caller must not proceed (G3).
//   * IT IS NOT SILENT when it fires: the route answer's own `why` is returned in `plan.why`
//     with the SIMULATED marker already on it, and the caller prints it once per shape.
//
// WHAT IT DOES NOT CLAIM: that the arm is FASTER. That is a measurement
// (dl/fp16route/landq/bf16_fp16_bench.cu) and this file reports a ROUTE, not a throughput.
//
// HOST-ONLY, and deliberately a .cpp: it reads environment variables through arch_sim.h and
// builds std::string reasons, so it has no business in a device translation unit -- the same
// split, for the same reason, that qpn_arch_route.cpp:1-6 states.

#include "ninfer/ops/linear.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <string>

namespace ninfer::ops::detail {

struct Bf16Fp16Plan {
    // Take the fp16 tensor-core arm for this op instead of the shape table's answer.
    bool use_arm = false;
    // A simulation was requested and could NOT be honoured: nothing is answered for the real
    // device either, and the caller must not proceed.
    bool refused = false;
    bool simulated = false;
    int physical_sm = 0;
    int effective_sm = 0;
    // The token count the arm's atom requires; below it the FFMA shape table is the better
    // answer and this plan says so rather than pretending the arm is a win.
    std::int32_t min_tokens = 0;
    // "mma-fp16-plane" when use_arm, else the route the tables actually named.
    std::string route;
    // The kernel FILE the route named, or {} when there is none.
    std::string kernel;
    // Never empty. Carries the SIMULATED marker when simulated.
    std::string why;
};

// One call per BF16 linear op invocation on the decode path, so it is cheap and cached: the
// device and the simulated view cannot change within a process, and the SHAPE can, so the cache
// is keyed on it. thread_local for the same reason qpn_arch_route.cpp's cache is.
[[nodiscard]] Bf16Fp16Plan bf16_fp16_plan(std::int32_t tokens, std::int32_t n, std::int32_t k);

// The device's own compute capability. Declared here rather than borrowed from
// ninfer::ops::qpn so this arm does not depend on the QPN family's header, and returns 0 when no
// device is visible -- which the tables treat as "no row" and answer conservatively for.
[[nodiscard]] int bf16_fp16_physical_sm();

// THE DISPATCH. Called by bf16_dispatch() only when plan.use_arm is true, and it is a HOST
// function so that tests/test_bf16_fp16_plane.cpp can take its address -- which is the tie that
// makes the build fact non-optional. Defined in ops/linear/bf16/bf16_mma_fp16.cu.
void launch_bf16_fp16_mma(const Tensor& x, const Weight& weight, Tensor& out,
                          const Bf16Fp16Plan& plan, cudaStream_t stream);

} // namespace ninfer::ops::detail
