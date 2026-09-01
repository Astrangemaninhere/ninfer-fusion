#pragma once

// SSD-backed PLE n-gram table (Qwen4Exp). Implements the ds4 SSD runtime
// contract: the 95 GiB sidecar never becomes device-resident. Host code
// derives the 16 row ids per token (PleLayout), an async worker prefetches
// the touched 2.5 KB/row spans into a bounded pinned page cache, and a
// device kernel gathers the rows into [n_heads * row_dim, n_tokens] BF16.
//
// Read path (cold miss): pread into pinned staging -> H2D copy. The prefetch
// worker keeps hot rows in the pinned LRU cache so steady-state gathers hit
// cache. See RESEARCH-FLASHNEXT.md for the full contract.

#include "ops/ple/ple_layout.h"

#include <cuda_runtime.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace ninfer::ops::ple {

struct PleTableOptions {
    std::filesystem::path sidecar_root; // directory containing ple-manifest.json
    std::size_t cache_bytes = 512ULL << 20; // pinned LRU cache budget
    unsigned prefetch_workers = 16;
};

// Owns the sidecar files and the pinned cache; one instance per engine.
class PleTable {
public:
    explicit PleTable(PleTableOptions options);
    ~PleTable();

    PleTable(const PleTable&) = delete;
    PleTable& operator=(const PleTable&) = delete;

    const PleLayout& layout() const noexcept { return layout_; }

    // Derives rows for a token batch (see PleLayout::derive_rows).
    void derive_rows(std::span<const std::int32_t> tokens,
                     std::span<const std::int32_t> prevs, std::int32_t eos,
                     std::int32_t* rows_out) const noexcept;

    // Gathers 16 * n_tokens rows into dst (device, [n_heads * row_dim, n_tokens],
    // BF16, contiguous). Blocks until the touched rows are resident in the
    // pinned cache (faults are satisfied synchronously).
    void gather(const std::int32_t* rows, std::size_t n_tokens, void* dst,
                cudaStream_t stream);

private:
    struct CacheEntry {
        std::uint64_t file_index : 2;
        std::uint64_t offset; // bytes within the file
        std::uint64_t bytes;
        void* pinned; // cudaHostAlloc(..., cudaHostAllocMapped)
        std::size_t last_use_seq;
    };

    void open_files();
    void* read_span(std::uint32_t file_index, std::uint64_t offset,
                    std::uint64_t bytes); // returns pinned pointer, faulting on miss
    void* fault_in(std::uint32_t file_index, std::uint64_t offset, std::uint64_t bytes);

    PleLayout layout_;
    PleTableOptions options_;
    std::vector<int> file_fds_; // 4 open sidecar files

    // Pinned cache (LRU, bounded by options_.cache_bytes).
    std::mutex cache_mutex_;
    std::unordered_map<std::uint64_t, std::unique_ptr<CacheEntry>> cache_;
    std::uint64_t cache_bytes_used_ = 0;
    std::size_t use_seq_ = 0;
};

} // namespace ninfer::ops::ple
