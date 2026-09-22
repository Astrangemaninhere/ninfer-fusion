#include "core/dtype.h"

#include <stdexcept>
#include <cstdio>
#include <execinfo.h>

namespace ninfer {

std::size_t dtype_size(DType dtype) {
    switch (dtype) {
    case DType::BF16:
        return 2;
    case DType::FP32:
        return 4;
    case DType::I32:
        return 4;
    case DType::U8:
        return 1;
    case DType::I64:
        return 8;
    case DType::I8:
        return 1;
    case DType::FP16:
        return 2;
    case DType::FP8_E4M3FN:
        return 1;
    // NAMED, and deliberately NOT given a number.  NVFP4 / ISO3 / E8Kv are PACKED
    // plane formats (dtype.h:17-24: two codes per byte, scales in a separate plane),
    // so there is no scalar element size to return.  A caller that multiplied
    // rows*columns by this would size a plane WRONG (core/host_kv_arena.cpp:65,
    // core/tensor.cpp:52).  Naming them here, instead of a `default:`, keeps -Wswitch
    // on this switch; they fall into the SAME refusal an unnamed code gets, but the
    // message no longer calls a perfectly valid DType "invalid".
    case DType::NVFP4:
    case DType::ISO3:
    case DType::E8Kv:
        break;
    }
    {
        void* frames[24];
        const int n = backtrace(frames, 24);
        std::fprintf(stderr, "dtype_size: no scalar element size for DType code=%d "
                             "backtrace (%d frames):\n",
                     static_cast<int>(dtype), n);
        backtrace_symbols_fd(frames, n, 2);
    }
    throw std::invalid_argument(
        "dtype_size: this DType has no scalar element size. A packed plane format "
        "(nvfp4/iso3/e8kv) is sized by the per-layer KV storage table, not by "
        "rows*columns*element_size; any other code names no enumerator at all.");
}

} // namespace ninfer
