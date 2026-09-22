#include "serve/kv_vram_probe.h"

#include <cuda_runtime.h>

namespace ninfer::serve {

std::uint64_t free_vram_bytes() {
    std::size_t free_bytes  = 0;
    std::size_t total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) {
        // Drop the sticky error before returning: the probe is a read-only
        // diagnostic and must not leave an error for a later CUDA call on this
        // thread to inherit (the reload path checks its own calls).
        (void)cudaGetLastError();
        return 0;
    }
    return static_cast<std::uint64_t>(free_bytes);
}

} // namespace ninfer::serve
