// PROVENANCE -- FORK-SURVEY borrow (Apache-2.0).
// Source repo : JCraigWasTaken/ninfer-gfx906
// Branch      : gfx906-port
// Commit      : 7a3c18d9fcfc977fc9c35766fd9938c0039e9a78
// Source path : src/compat/gfx906/include/nvtx3/nvToolsExt.h
// sha256(src) : b1cecb999038634f3713080fd03782700b042ac0f04694ec08b9be00e02a21ef
// Landed by   : /home/user/ninfer-fusion gfx906port line, 2026-09-21 (mi50 3-step plan, step 2 of 3: additive port layer; step 3 = CMake wiring, NOT done here).
// Upstream    : Neroued/ninfer, Apache-2.0, ships NO NOTICE -- attribution is on us.
// Status      : ADDITIVE, NOT wired into any build target (src/CMakeLists.txt is explicit, not GLOB), so this file is inert until someone adds it deliberately.
// Note        : CUDA->HIP shim header. It SHADOWS a real CUDA header of the same name the moment src/compat/gfx906/include is put ahead of the CUDA include dir -- which is exactly what the fork does, include_directories(BEFORE ...) at src/CMakeLists.txt:5, tests/CMakeLists.txt:2 and bench/CMakeLists.txt:2. Not adding that line is what keeps it inert here. See docs/gfx906/PORT-AUDIT.md.
#pragma once
// gfx906 port: NVTX no-op stub covering the surface core/nvtx.h,
// core/nvtx_range.h, and the runtime instrumentation use. Profiling ranges
// become free no-ops; roctracer integration can replace this later.

#include <cstdint>

#define NVTX_VERSION 3
#define NVTX_EVENT_ATTRIB_STRUCT_SIZE (static_cast<std::uint16_t>(sizeof(nvtxEventAttributes_t)))
#define NVTX_COLOR_ARGB 1
#define NVTX_PAYLOAD_TYPE_UNSIGNED_INT64 1
#define NVTX_MESSAGE_TYPE_REGISTERED 3

typedef void* nvtxDomainHandle_t;
typedef void* nvtxStringHandle_t;

typedef union nvtxMessageValue_t {
    const char* ascii;
    const wchar_t* unicode;
    nvtxStringHandle_t registered;
} nvtxMessageValue_t;

typedef union nvtxEventPayload_t {
    std::uint64_t ullValue;
    std::int64_t llValue;
    double dValue;
} nvtxEventPayload_t;

typedef struct nvtxEventAttributes_t {
    std::uint16_t version;
    std::uint16_t size;
    std::uint32_t category;
    std::int32_t colorType;
    std::uint32_t color;
    std::int32_t payloadType;
    std::int32_t reserved0;
    nvtxEventPayload_t payload;
    std::int32_t messageType;
    nvtxMessageValue_t message;
} nvtxEventAttributes_t;

inline int nvtxRangePushA(const char*) { return 0; }
inline int nvtxRangePop() { return 0; }
inline nvtxDomainHandle_t nvtxDomainCreateA(const char*) { return nullptr; }
inline void nvtxDomainNameCategoryA(nvtxDomainHandle_t, std::uint32_t, const char*) {}
inline nvtxStringHandle_t nvtxDomainRegisterStringA(nvtxDomainHandle_t, const char*) {
    return nullptr;
}
inline int nvtxDomainRangePushEx(nvtxDomainHandle_t, const nvtxEventAttributes_t*) { return 0; }
inline int nvtxDomainRangePop(nvtxDomainHandle_t) { return 0; }
