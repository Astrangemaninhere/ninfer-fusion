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
    }
    {
        void* frames[24];
        const int n = backtrace(frames, 24);
        std::fprintf(stderr, "invalid DType code=%d backtrace (%d frames):\n",
                     static_cast<int>(dtype), n);
        backtrace_symbols_fd(frames, n, 2);
    }
    throw std::invalid_argument("invalid DType");
}

} // namespace ninfer
