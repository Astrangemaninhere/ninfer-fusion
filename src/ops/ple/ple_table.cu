#include "ops/ple/ple_table.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <unordered_map>
#include <unistd.h>

namespace ninfer::ops::ple {
namespace {

constexpr std::uint64_t kCacheAlign = 4096;

std::uint64_t align_down(std::uint64_t v) { return v & ~(kCacheAlign - 1); }
std::uint64_t align_up(std::uint64_t v) { return (v + kCacheAlign - 1) & ~(kCacheAlign - 1); }

void check_cuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
    }
}

// Each thread copies one 160-column row (BF16) from a UVA-mapped pinned
// address into the token-major output. row_ptrs holds the device-side
// addresses of the unique rows (cudaHostGetDevicePointer results).
__global__ void ple_gather_rows_kernel(const unsigned long long* row_ptrs,
                                       const int* unique_ids, __half* dst, int total,
                                       int row_dim, int n_heads) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) { return; }
    const int t = i / n_heads;
    const int h = i % n_heads;
    const __half* src = reinterpret_cast<const __half*>(row_ptrs[unique_ids[i]]);
    __half* out = dst + t * (n_heads * row_dim) + h * row_dim;
#pragma unroll 4
    for (int j = 0; j < row_dim; ++j) { out[j] = src[j]; }
}

} // namespace

PleTable::PleTable(PleTableOptions options)
    : layout_(PleLayout::from_manifest((options.sidecar_root / "ple-manifest.json").string())),
      options_(std::move(options)) {
    open_files();
}

PleTable::~PleTable() {
    for (int fd : file_fds_) {
        if (fd >= 0) { ::close(fd); }
    }
    file_fds_.clear();
    cache_.clear();
}

void PleTable::open_files() {
    for (const auto& file : layout_.physical_files) {
        const std::filesystem::path path = options_.sidecar_root / file.path;
        const int fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) {
            throw std::runtime_error("PLE sidecar open failed: " + path.string());
        }
        file_fds_.push_back(fd);
    }
}

void PleTable::derive_rows(std::span<const std::int32_t> tokens,
                           std::span<const std::int32_t> prevs, std::int32_t eos,
                           std::int32_t* rows_out) const noexcept {
    layout_.derive_rows(tokens, prevs, eos, rows_out);
}

void* PleTable::read_span(std::uint32_t file_index, std::uint64_t offset,
                          std::uint64_t bytes) {
    // Resolve to an aligned cache range and look it up.
    const std::uint64_t start = align_down(offset);
    const std::uint64_t end = align_up(offset + bytes);
    const std::uint64_t key = (static_cast<std::uint64_t>(file_index) << 56) | start;
    const std::uint64_t len = end - start;

    {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        auto it = cache_.find(key);
        if (it != cache_.end()) {
            it->second->last_use_seq = use_seq_++;
            return static_cast<char*>(it->second->pinned) + (offset - start);
        }
    }

    void* pinned = fault_in(file_index, start, len);
    return static_cast<char*>(pinned) + (offset - start);
}

void* PleTable::fault_in(std::uint32_t file_index, std::uint64_t offset,
                         std::uint64_t bytes) {
    if (file_index >= file_fds_.size()) {
        throw std::runtime_error("PLE sidecar file index out of range");
    }
    void* mapped = nullptr;
    check_cuda(cudaHostAlloc(&mapped, bytes, cudaHostAllocMapped | cudaHostAllocPortable),
               "PLE cache cudaHostAlloc");

    // Read the span from disk. Partial reads (tail alignment beyond EOF) are
    // zero-filled by the allocation already.
    std::size_t done = 0;
    while (done < bytes) {
        const ssize_t got = ::pread(file_fds_[file_index], static_cast<char*>(mapped) + done,
                                    bytes - done, static_cast<off_t>(offset + done));
        if (got < 0) {
            check_cuda(cudaFreeHost(mapped), "PLE cache free");
            throw std::runtime_error("PLE sidecar pread failed");
        }
        if (got == 0) { break; }
        done += static_cast<std::size_t>(got);
    }

    auto entry = std::make_unique<CacheEntry>();
    entry->file_index = file_index;
    entry->offset = offset;
    entry->bytes = bytes;
    entry->pinned = mapped;
    entry->last_use_seq = 0;

    {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        const std::uint64_t key = (static_cast<std::uint64_t>(file_index) << 56) | offset;
        auto [it, inserted] = cache_.emplace(key, std::move(entry));
        if (!inserted) {
            // Lost a race; keep the existing entry and free ours.
            check_cuda(cudaFreeHost(mapped), "PLE cache free");
            it->second->last_use_seq = use_seq_++;
            return static_cast<char*>(it->second->pinned);
        }
        cache_bytes_used_ += bytes;
        use_seq_++;
        it->second->last_use_seq = use_seq_;

        // Evict LRU entries until the budget is respected.
        while (cache_bytes_used_ > options_.cache_bytes && cache_.size() > 1) {
            auto victim = std::min_element(
                cache_.begin(), cache_.end(),
                [](const auto& a, const auto& b) {
                    return a.second->last_use_seq < b.second->last_use_seq;
                });
            if (victim == it) { break; } // never evict the entry we just inserted
            cache_bytes_used_ -= victim->second->bytes;
            check_cuda(cudaFreeHost(victim->second->pinned), "PLE cache free");
            cache_.erase(victim);
        }
        return static_cast<char*>(it->second->pinned);
    }
}

void PleTable::gather(const std::int32_t* rows, std::size_t n_tokens, void* dst,
                      cudaStream_t stream) {
    const std::size_t total = n_tokens * layout_.n_heads;
    const std::size_t row_bytes = layout_.row_stride_bytes;

    // Deduplicate rows so a hot n-gram span is faulted in once.
    std::vector<std::uint64_t> row_keys(total);
    std::vector<std::uint32_t> file_idx(total);
    std::vector<std::uint64_t> offsets(total);
    std::vector<std::uint64_t> unique_rows;
    std::vector<int> unique_ids(total);
    std::unordered_map<std::uint64_t, int> seen;

    for (std::size_t i = 0; i < total; ++i) {
        const std::uint64_t row = static_cast<std::uint64_t>(rows[i]);
        std::uint32_t fi = 0;
        std::uint64_t off = 0;
        if (!layout_.row_location(row, fi, off)) {
            throw std::runtime_error("PLE row id out of range");
        }
        file_idx[i] = fi;
        offsets[i] = off;
        row_keys[i] = (static_cast<std::uint64_t>(fi) << 56) | off;
        auto it = seen.find(row_keys[i]);
        if (it == seen.end()) {
            unique_ids[i] = static_cast<int>(unique_rows.size());
            seen.emplace(row_keys[i], unique_ids[i]);
            unique_rows.push_back(row_keys[i]);
        } else {
            unique_ids[i] = it->second;
        }
    }

    // Fault every unique row in (sorted by file/offset for sequential SSD IO).
    std::vector<std::size_t> order(unique_rows.size());
    for (std::size_t i = 0; i < order.size(); ++i) { order[i] = i; }
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        return unique_rows[a] < unique_rows[b];
    });
    std::vector<unsigned long long> device_ptrs(unique_rows.size());
    for (std::size_t k = 0; k < order.size(); ++k) {
        const std::size_t u = order[k];
        const std::uint32_t fi = static_cast<std::uint32_t>(unique_rows[u] >> 56);
        const std::uint64_t off = unique_rows[u] & ((1ULL << 56) - 1);
        void* host_ptr = read_span(fi, off, row_bytes);
        void* dev_ptr = nullptr;
        check_cuda(cudaHostGetDevicePointer(&dev_ptr, host_ptr, 0),
                   "PLE cache UVA pointer");
        device_ptrs[u] = reinterpret_cast<unsigned long long>(dev_ptr);
    }

    // Upload the pointer table and row ids, then gather on device.
    void* d_ptrs = nullptr;
    void* d_ids = nullptr;
    check_cuda(cudaMallocAsync(&d_ptrs, device_ptrs.size() * sizeof(unsigned long long),
                               stream),
               "PLE ptrs alloc");
    check_cuda(cudaMallocAsync(&d_ids, total * sizeof(int), stream), "PLE ids alloc");
    check_cuda(cudaMemcpyAsync(d_ptrs, device_ptrs.data(),
                               device_ptrs.size() * sizeof(unsigned long long),
                               cudaMemcpyHostToDevice, stream),
               "PLE ptrs upload");
    check_cuda(cudaMemcpyAsync(d_ids, unique_ids.data(), total * sizeof(int),
                               cudaMemcpyHostToDevice, stream),
               "PLE ids upload");

    const int threads = 256;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    ple_gather_rows_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const unsigned long long*>(d_ptrs),
        static_cast<const int*>(d_ids), static_cast<__half*>(dst),
        static_cast<int>(total), static_cast<int>(layout_.embedding_row_dimension),
        static_cast<int>(layout_.n_heads));

    check_cuda(cudaFreeAsync(d_ptrs, stream), "PLE ptrs free");
    check_cuda(cudaFreeAsync(d_ids, stream), "PLE ids free");
}

} // namespace ninfer::ops::ple
