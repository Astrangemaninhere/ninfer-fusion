#pragma once

// ---------------------------------------------------------------------------
// pdl.cuh -- programmatic dependent launch, with a correct NON-PDL behaviour.
//
// THE WALL THIS FILE USED TO BE
// -----------------------------
// Both device helpers below called a PDL intrinsic unconditionally, and the launcher
// set cudaLaunchAttributeProgrammaticStreamSerialization unconditionally. Those lower to
// sm_90+ PTX, so ANY build whose arch list contained a pre-sm_90 target failed at ptxas.
// MEASURED on sm_86 (nvcc 13.3, 148 TUs, 138 rc=0; these four are among the 10 red):
//
//   Modifier '.wait' requires .target sm_90 or higher
//   Modifier '.launch_dependents' requires .target sm_90 or higher
//   Instruction 'griddepcontrol' requires .target sm_90 or higher
//
// in src/ops/sparse_moe/decode/sparse_moe_decode_kernels.cu,
//    src/ops/sparse_moe/small_t/sparse_moe_small_t_kernels.cu,
//    src/ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_independent.cu,
//    src/ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_conv_snapshot.cu.
// This is NOT a Volta/Turing-only wall: it reddens sm_86 (3090/A10) and every rung below
// sm_90 as well, which is why it is in scope for "the fallback must migrate easily to
// other cards" and not for sm_70 alone.
//
// Independent measurement, this line, same tree (hand-written PTX, ptxas 13.3):
//   griddepcontrol   sm_70 X  sm_75 X  sm_80 X  sm_86 X  sm_89 X
//                    sm_90 OK sm_100 OK sm_100a OK sm_120a OK
// so the boundary is exactly sm_90 and the guard below is `__CUDA_ARCH__ >= 900`.
//
// WHY THIS IS A FALLBACK AND NOT A TRAP
// -------------------------------------
// Unlike the missing-mma case in ops/common/mma.cuh, PDL HAS a correct non-PDL
// behaviour: launch WITHOUT the programmatic attribute and let ordinary stream
// serialisation order the producer and the consumer. A consumer that does not wait is
// then correct, because the hardware already made the producer finish. So the device
// helpers become NO-OPS below sm_90 rather than traps -- but the no-op is only correct
// TOGETHER WITH the host-side decision to omit the attribute, which is why both halves
// live in this one file and are decided from the same `>= 900` boundary.
//
// THE TWO HALVES USE DIFFERENT FACTS ON PURPOSE, AND THEY AGREE
// ------------------------------------------------------------
//   device side: __CUDA_ARCH__, i.e. the target the cubin was built for;
//   host side  : the DEVICE's runtime compute capability.
// They agree for any fatbin/device pair that can actually be launched (a cubin is only
// selected for a device of its own capability), which is the case that matters. The
// host half is a runtime fact rather than a build fact so that a build which DOES
// target sm_90+ still falls back correctly when it is run against an older card.
//
// LOUDNESS: the fallback is observable rather than silent. NINFER_PDL_TRACE=1 prints one
// line per process the first time a launch is taken without the attribute, naming the
// device capability and the reason. The default is quiet because the fallback is
// CORRECT, not degraded -- a warning on every launch would train the operator to ignore
// it, which is how a real warning gets lost.
// ---------------------------------------------------------------------------

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <utility>

namespace ninfer::pdl {

// The one boundary both halves use. sm_90 is where griddepcontrol appears (measured).
inline constexpr int kProgrammaticLaunchMinSm = 90;

// Runtime capability of the current device, queried once per process and cached. A
// failed query returns 0, which is the conservative answer: no attribute, no wait.
[[nodiscard]] inline int runtime_sm() noexcept {
    static const int cached = []() noexcept -> int {
        int device = 0;
        if (cudaGetDevice(&device) != cudaSuccess) { return 0; }
        cudaDeviceProp properties{};
        if (cudaGetDeviceProperties(&properties, device) != cudaSuccess) { return 0; }
        return properties.major * 10 + properties.minor;
    }();
    return cached;
}

// True when a launch on this device may use the programmatic attribute. False is not an
// error: it selects ordinary stream serialisation, which is correct.
[[nodiscard]] inline bool programmatic_launch_supported() noexcept {
    return runtime_sm() >= kProgrammaticLaunchMinSm;
}

namespace detail {

inline void note_fallback_once(int sm) noexcept {
    static bool noted = false;
    if (noted) { return; }
    noted = true;
    if (std::getenv("NINFER_PDL_TRACE") == nullptr) { return; }
    std::fprintf(stderr,
                 "ninfer: programmatic dependent launch NOT used on this device "
                 "(compute capability %d.%d, sm_%d < sm_%d): the launch omits "
                 "cudaLaunchAttributeProgrammaticStreamSerialization and the kernels' "
                 "trigger/wait helpers are compiled to no-ops, so ordinary stream "
                 "serialisation orders producer and consumer. This is the correct "
                 "non-PDL behaviour, not a degradation. (src/core/pdl.cuh)\n",
                 sm / 10, sm % 10, sm, kProgrammaticLaunchMinSm);
}

} // namespace detail

struct LaunchConfig {
    dim3 grid;
    dim3 block;
    std::size_t dynamic_smem_bytes = 0;
    cudaStream_t stream            = nullptr;
};

// Launches a consumer kernel as a programmatic dependent of the immediately preceding
// producer kernel in the same stream, WHERE THE DEVICE SUPPORTS IT. On a device below
// sm_90 the same call makes an ordinary launch: the producer is still ordered before the
// consumer by the stream, so every consumer control path that calls
// wait_for_dependencies() is still correct -- the call is simply a no-op there.
template <class... KernelArgs, class... CallArgs>
[[nodiscard]] inline cudaError_t
launch_dependent(const LaunchConfig& launch, void (*kernel)(KernelArgs...), CallArgs&&... args) {
    cudaLaunchConfig_t config{};
    config.gridDim          = launch.grid;
    config.blockDim         = launch.block;
    config.dynamicSmemBytes = launch.dynamic_smem_bytes;
    config.stream           = launch.stream;

    cudaLaunchAttribute attribute{};
    if (programmatic_launch_supported()) {
        attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attribute.val.programmaticStreamSerializationAllowed = 1;
        config.attrs    = &attribute;
        config.numAttrs = 1;
    } else {
        // No attribute: ordinary stream serialisation. Deliberately NOT an error and
        // deliberately not silent when NINFER_PDL_TRACE is set.
        detail::note_fallback_once(runtime_sm());
        config.attrs    = nullptr;
        config.numAttrs = 0;
    }
    return cudaLaunchKernelEx(&config, kernel, std::forward<CallArgs>(args)...);
}

// Every producer CTA must call this at least once or exit. This enables dependent
// scheduling but does not make producer writes visible to the consumer.
//
// Present on every target as a SYMBOL, so no caller needs a guard of its own; empty
// below sm_90, where there is no dependent scheduling to enable and the caller's
// wait_for_dependencies() is likewise empty.
__device__ __forceinline__ void trigger_dependents() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

// Call on every consumer control path before its first access to producer-dependent data.
// Empty below sm_90: the host side of this file did not request dependent scheduling
// there, so ordinary stream serialisation already guarantees the producer completed.
__device__ __forceinline__ void wait_for_dependencies() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    cudaGridDependencySynchronize();
#endif
}

} // namespace ninfer::pdl
