#pragma once

// FreeToken gap 1 (free-VRAM axis): the ONE place the capacity probe touches
// CUDA. It lives in its own TU on purpose:
//   * src/serve/kv_auto_relayout.{h,cpp} stay CUDA-free -- the header carries no
//     CUDA type and the decision TU needs no libcudart at link time, which is
//     what keeps tools/kv_relayout_test.cpp buildable with plain g++
//     ("No CUDA device required" is that test's contract), and
//   * apps/serve/main.cpp can inject the probe without including
//     <cuda_runtime.h> itself (the header below has no CUDA type), while
//     ninfer_serve already links CUDA::cudart PRIVATE
//     (src/CMakeLists.txt:415), so the symbol resolves in the server binary.
//
// Precedent: src/targets/registry.cpp:91 current_free_device_bytes() runs the
// same query for the startup capacity plan. That one is file-local to
// registry.cpp (no declaration in any header), which is why this one exists
// rather than being reused.

#include <cstdint>

namespace ninfer::serve {

// Bytes free on the current CUDA device right now.
//
// 0 means "unknown" -- the query failed (no device, driver error, no context on
// this thread). 0 is the documented inert value of
// KvAutoRelayout::Config::free_vram_bytes: a failed probe leaves the VRAM axis
// inactive instead of aborting a decision cycle, because a server must not lose
// its reload loop to a diagnostic reading.
std::uint64_t free_vram_bytes();

} // namespace ninfer::serve
