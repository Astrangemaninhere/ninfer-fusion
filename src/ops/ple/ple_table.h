#pragma once

// SSD-backed PLE n-gram table (Qwen4Exp). Implements the ds4 SSD runtime
// contract: the 95 GiB sidecar never becomes device-resident. Host code
// derives the 16 row ids per token (PleLayout), the touched 2.5 KB/row spans
// are read into a bounded pinned LRU page cache, and a device kernel gathers
// the rows into [n_heads * row_dim, n_tokens] BF16.
//
// Read path (cold miss): pread into pinned staging, then the kernel reads the
// pinned pointer. The LRU keeps hot rows resident, so steady-state gathers hit
// cache. See RESEARCH-FLASHNEXT.md for the full contract.
//
// NO ASYNC PREFETCH WORKER EXISTS, and this comment used to say one did ("an async worker
// prefetches the touched spans", "The prefetch worker keeps hot rows ... so steady-state
// gathers hit cache"). A by-name census over src/, include/, apps/, tests/ and tools/ found
// every occurrence of PleTableOptions::prefetch_workers to be a declaration, a default, a copy
// or a comment: nothing read it. Rather than keep a claim the code does not honour, the knob
// is now a NAMED refusal -- 0 (the default) means "no prefetch pool", and a non-zero value
// throws from the constructor with a message that says so. A silently inert knob and an absent
// knob are the same defect (F391); the loud version is the one an operator can act on. Wiring
// a real pool is a separate patch, with its own concurrency evidence.

#include "ops/ple/ple_layout.h"
#include "ops/ple/ple_stage.h"

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
    // 0 = no prefetch pool. A NON-ZERO VALUE IS REFUSED IN THE CONSTRUCTOR: the async pool
    // this field used to promise was never written, so accepting a non-zero value here would
    // be accepting a request that nothing acts on.
    unsigned prefetch_workers = 0;
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

    // Gathers 16 * n_tokens rows into dst (device, BF16, contiguous, token-major:
    // token t occupies [t * n_heads * row_dim, (t+1) * n_heads * row_dim) with
    // head h at h * row_dim — matches HF's PLE embedding flatten(-2) layout).
    // Blocks until the touched rows are resident in the pinned cache (faults
    // are satisfied synchronously).
    void gather(const std::int32_t* rows, std::size_t n_tokens, void* dst,
                cudaStream_t stream);

    // The phase-aware entry point. It derives and gathers only the columns
    // the phase actually owes (ple_phase_window, ops/ple/ple_stage.h):
    // Prefill/Draft take every column, Decode/Verify take the last one only,
    // because a Verify pass discards every earlier proposal column before the
    // residual reaches the next layer. dst receives window.tokens * n_heads
    // rows laid out exactly like gather() (token-major, head-minor), so a
    // Verify caller that passed T columns must read dst as
    // [n_heads * row_dim, 1].
    //
    // This is the only place a phase enters the PLE path; callers must not
    // re-derive the window themselves.
    void gather_phase(std::span<const std::int32_t> tokens,
                      std::span<const std::int32_t> prevs, std::int32_t eos,
                      PlePhase phase, void* dst, cudaStream_t stream);

    // Host-side counters (relaxed atomics; a few adds per gather). A wired
    // stage that never gathers is indistinguishable from an unwired one
    // without these -- print them under NINFER_PLE_STATS=1.
    [[nodiscard]] const PleForensics& forensics() const noexcept { return forensics_; }

private:
    struct CacheEntry {
        std::uint64_t file_index : 2;
        std::uint64_t offset; // bytes within the file
        std::uint64_t bytes;
        void* pinned; // cudaHostAlloc(..., cudaHostAllocMapped)
        std::size_t last_use_seq;
        // Faults made during the current gather() must not be evicted before
        // the gather kernel runs: their device pointers are already recorded.
        std::uint64_t gather_epoch;
    };

    void open_files();
    void* read_span(std::uint32_t file_index, std::uint64_t offset,
                    std::uint64_t bytes); // returns pinned pointer, faulting on miss
    void* fault_in(std::uint32_t file_index, std::uint64_t offset, std::uint64_t bytes);

    PleLayout layout_;
    PleTableOptions options_;
    std::vector<int> file_fds_; // 4 open sidecar files
#if defined(_WIN32)
    // The Windows arm's handles for the SAME four sidecar files. A HANDLE is
    // pointer-sized and does not fit the int above, and this class is not willing to
    // assume that truncating a handle is safe, so the Windows path keeps its own vector
    // and file_fds_ is left empty there (POSIX builds never see this member).
    std::vector<void*> file_handles_;
#endif

    // Pinned cache (LRU, bounded by options_.cache_bytes).
    std::mutex cache_mutex_;
    std::unordered_map<std::uint64_t, std::unique_ptr<CacheEntry>> cache_;
    std::uint64_t cache_bytes_used_ = 0;
    std::size_t use_seq_ = 0;
    // Bumped by every gather(); entries faulted in during the current epoch are
    // exempt from eviction until that gather has finished (UVA pointers are
    // captured before the kernel launch).
    std::uint64_t gather_epoch_ = 0;

    PleForensics forensics_;
    // Host scratch for gather_phase: window.tokens * n_heads row ids.
    std::vector<std::int32_t> phase_rows_;
};

} // namespace ninfer::ops::ple
