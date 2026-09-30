#pragma once

// ---------------------------------------------------------------------------
// qpn_arch_route.h -- THE ENGINE'S ARCHITECTURE-ROUTE DECISION for the W4A16 path
// ---------------------------------------------------------------------------
//
// THE CALLER THAT WAS MISSING. MEASURED 2026-09-18 (PATCHSET/SIM-RUN, section 1.2): the route
// layer (src/core/kernel_route.h) had ZERO callers under src/ or apps/; the literal
// "qpn-w4a16" -- whose only source is kernel_route.h:242 -- occurred 0 times in the shipping
// engine; and route_log_line(), the function that emits the `[route][SIMULATED]` line, had
// zero callers ANYWHERE in the tree, tests included. The consequence was measured too: six
// engine arms under NINFER_SIM_ARCH=70/75/86 produced BYTE-IDENTICAL token ids, and the
// identity proved nothing, because the simulated rung never ran -- no engine translation unit
// could ask the tables anything.
//
// This header is that caller. select_qpn_arch_route() is the one function the engine's NVFP4
// W4A16 op consults, and it is what turns "the fallback route is taken" from a host-side claim
// into an observable property of a real binary.
//
// THE THREE KEYS, unchanged and all still required (src/core/arch_sim.h, G1):
//   1. the binary was built with the CMake option NINFER_ENABLE_ARCH_SIM=ON (default OFF),
//      which defines NINFER_ARCH_SIM_ENABLED -- root CMakeLists.txt, and applied to the
//      engine targets in src/CMakeLists.txt;
//   2. NINFER_SIM_ARCH=<rung>;
//   3. NINFER_SIM_ARCH_ACK=I-UNDERSTAND-THIS-IS-NOT-A-V100.
// A stray environment variable does not enable anything, and a refused request does not fall
// back to the real device: it is REFUSED, and this decision says so (see `refused`).
//
// WHAT IT DOES NOT PROVE, stated here so a caller cannot forget it: the kernels that execute
// are the ones THIS binary was compiled for. On a 5090 D that is sm_120a code, NOT sm_70
// binaries. A simulated rung proves the route decision and nothing about a real V100, and no
// throughput or latency figure from such a run may be quoted as that device's.

#include <cstdint>
#include <string>

namespace ninfer::ops::qpn {

// One answer to "what should the engine do for this shape on this device".
//
//   use_qpn      true  => the route tables selected the QPN W4A16 fallback and the caller
//                         MUST dispatch gemm_qpn (or refuse; see dispatch_qpn_fallback).
//                 false => the caller proceeds on its own native route, unchanged.
//   simulated    true  => the answer was produced for a SIMULATED rung, not for the card in
//                         hand. `why` carries the SIMULATED marker in that case.
//   refused      true  => a simulation was requested and could NOT be honoured (G3). Nothing
//                         is answered for the real device either, and a caller must not
//                         proceed: the run is over, loudly.
//   kernel_in_build    => caps::kQpnInBuild, i.e. whether the QPN sources are compiled into
//                         THIS binary. It is reported rather than assumed because it is the
//                         fact that was published PRIVATE and therefore absent from this
//                         engine's own gate TU until 2026-09-18 (see src/CMakeLists.txt).
struct QpnArchRoute {
    bool        use_qpn         = false;
    bool        simulated       = false;
    bool        refused         = false;
    int         physical_sm     = 0;
    int         effective_sm    = 0;
    bool        kernel_in_build = false;
    // WHICH LAYOUT THE SELECTED BAND TAKES. false => the M 1..16 bands, which consume the
    // qpn_prepack permutation. true => the M 17..64 band, which consumes the artifact's own
    // [N][K/2] / [N][K/16] planes. A FIELD, because dispatch_qpn_fallback has to branch on it
    // and must not parse it out of `kernel`.
    bool        wide_m          = false;
    std::string route;   // route_name(): "qpn-w4a16" when use_qpn, "none" when refused
    std::string kernel;  // the band kernel the route named, or {} when there is none
    std::string why;     // never empty; carries the SIMULATED marker when simulated
};

// The device's own compute capability, e.g. 120 for the 5090 D. Cached: the device does not
// change within a process. Returns 0 when no device is visible, which the tables treat as
// "no row" and answer conservatively for.
[[nodiscard]] int current_device_sm();

// DOES THIS RUNG CARRY THE NATIVE NVFP4 W4A4 CHANNEL (`kind::mxf4nvf4`, m16n8k64 block scale)?
//
// This is the query nvfp4_dispatch.cpp's W4A4-vs-A16 decision was missing, and the reason the table
// could not name NVFP4's tensor-core-free kernel: `resolve_route()` was SHAPE-BLIND and RUNG-BLIND,
// so an admitted NVFP4 artifact on sm_86 would have reached launch_nvfp4_w4a4 -- whose instruction
// those ISAs do not have.
//
// IT ANSWERS FOR THE SAME RUNG THE GATE ANSWERS FOR: `caps::arch_view_for_device(physical_sm)`,
// i.e. the test-only simulator's `effective_sm` when one is honoured, and the card's own number
// otherwise. That is deliberate: the gate (src/targets/registry.cpp, at the artifact load door) and
// the op must not be able to disagree about which rung they are on -- the drift the two deciders in
// this area exist to prevent.
//
// THREE-STATE, FAIL-CLOSED ON THE AXIS THAT MATTERS: a REFUSED simulation answers `false` for
// everything (nothing may be claimed for the real device either), an unreadable device answers
// `true` so that a host-side plan made before the device is bound keeps today's route, and a KNOWN
// rung answers the covers() fact.
[[nodiscard]] bool nvfp4_w4a4_channel_available(int physical_sm);

// THE DECISION. One call per NVFP4 W4A16 op invocation.
//
// The test-only override is read ONCE per process (arch_view_for_device() has its own latch
// and prints sim_banner() once, G2), and the answer is asked through the ArchView overload
// of select_route(), so the Refused state cannot degrade into the real device's answer.
//
// THE LOG LINE. When the answer is simulated, the `[route][SIMULATED] ...` line is printed to
// stderr -- this is the first and only caller of caps::route_log_line() in the tree. It is
// printed when the SHAPE CHANGES rather than on every call, because this function sits on the
// decode path: a line per invocation at 20 tok/s would bury the marker it exists to make
// visible, and a prefill of 64k tokens would emit five figures of them. The marker also
// travels inside `why` on EVERY answer, which is where G2's "every answer is marked" is
// satisfied. `emit_log_line = false` is for callers that log it themselves.
[[nodiscard]] QpnArchRoute select_qpn_arch_route(int physical_sm, std::int32_t m, std::int32_t n,
                                                std::int32_t k, bool emit_log_line = true);

// THE DISPATCH. Called only when `route.use_qpn` is true.
//
// THE PRECONDITION, and it is NOT satisfied by this engine's weights today. gemm_qpn consumes
// the qpn_prepack layout -- `codes: [tile][group][lane][16B]` contiguous u8, `scales`
// indexed `tile * G * 32 + lane` (MEASURED coordinates: src/ops/linear/qpn/qpn_host.cu:2-3 for
// the layout, src/ops/linear/qpn/qpn_kernels.cuh:917-919 for the scales index) -- while an
// artifact's weights arrive in the plain block-scale layout `[N][K/2]` codes + `[N][K/16]`
// e4m3 scales, which is what the native nvfp4_w4a4_mma_kernel consumes. The two are NOT the
// same bytes in the same order, so feeding the native blob to gemm_qpn would be silent
// numerical corruption. TWO pieces are missing and both are ports rather than flags:
//   (1) a native -> qpn_prepack weight converter. Reference:
//       tools/archkit/qpn_port/qpn_prepack_proto.py; the slot map is machine-verified in
//       src/ops/linear/qpn/qpn_map.cuh.
//   (2) a dtype conversion: gemm_qpn's `x_half`/`y_half` are `half`, and every engine
//       activation and output on this path is `__nv_bfloat16`. bf16 and fp16 are different
//       bit patterns, so even a correct weight prepack would need this second step.
// The call site below hands over the engine's REAL buffers so that the day both pieces exist
// the dispatch is a layout fact and not a rewrite; today it refuses before touching them.
//
// So the caller declares the layout it has. With QpnPrepacked the gemm_qpn call below runs.
// With NativeBlockScale this function REFUSES, loudly and by name, rather than computing the
// wrong answer -- which is the same fail-closed rule the rest of this stack follows.
enum class QpnWeightLayout : std::uint8_t {
    NativeBlockScale, // an artifact's own [N][K/2] / [N][K/16] blob: NOT dispatchable
    QpnPrepacked,     // the qpn_prepack layout gemm_qpn consumes
};

void dispatch_qpn_fallback(const QpnArchRoute& route, QpnWeightLayout layout, const void* x_half,
                          const void* codes, const void* scales, float gscale, void* y_half,
                          std::int32_t m, std::int32_t k, std::int32_t n, void* stream);

} // namespace ninfer::ops::qpn
