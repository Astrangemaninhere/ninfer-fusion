// The ONE definition of the binary's arch list. Compiled by ninfer_core; the build gives
// NINFER_BUILD_CUDA_ARCHS to THIS file and to core/device_probe.cu, so the macro's reach is a
// stated, closed set instead of a property of whichever translation unit happens to be compiling
// core/arch_caps.h.
//
// The #if below selects an INITIALIZER inside ONE translation unit. It is deliberately NOT a
// function body, which is what made the old shape an ODR hazard: an inline function whose body
// depended on a per-TU macro is one function with two bodies, and a linker picks one of them.
#include "core/arch_caps.h"

namespace ninfer::caps {

#ifdef NINFER_BUILD_CUDA_ARCHS
// Constant-initialised from a string literal: static storage duration, so a std::string_view
// into it can never dangle and there is no static-initialisation order to get wrong.
const std::string_view kBuildArchList{NINFER_BUILD_CUDA_ARCHS};
#else
const std::string_view kBuildArchList{};
#endif

}  // namespace ninfer::caps
