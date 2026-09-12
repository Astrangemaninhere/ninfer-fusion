#pragma once
#include "targets/qwen3_6/impl/runtime/instance.h"
// Qwen3.6 family runtime implementation; instantiated only by exact variants.

#include "core/device.h"
#include "core/dtype.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <cstdlib>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule {

// Offline KV calibration capture. Gated by kvcalib_enabled() (env
// NINFER_KV_CALIB_DIR; the EngineOptions.kv_calibration_dir field this comment
// used to name no longer exists),
// text prefill copies the exact post-RoPE K and V tensors quantized into the
// paged KV cache (per full-attention layer and per chunk) to the host and
// appends one framed binary record per layer/chunk. The Python analyzer in
// tools/calib consumes these records; inference never reads them back.
class KvCalibrationCapture {
public:
    explicit KvCalibrationCapture(std::filesystem::path directory)
        : directory_(std::move(directory)) {
        if (directory_.empty()) {
            throw std::invalid_argument("KV calibration directory must not be empty");
        }
        std::filesystem::create_directories(directory_);
    }

    // S38: the stream parameter is REQUIRED. A plain cudaMemcpy here
    // would not be ordered against the producing (non-NULL) stream -- a
    // SILENT WRONG-DATA bug, not a slow path. Do not "simplify" the
    // Async+Sync back into a blocking copy. Calibration runs use
    // --no-cuda-graph (same offline-mode precedent as NINFER_KVDUMP_DIR /
    // NINFER_FT_STATS).
    void capture(std::uint32_t full_layer, const Tensor& k, const Tensor& v,
                 const Tensor& positions, cudaStream_t stream) {
        if (k.dtype != DType::BF16 || v.dtype != DType::BF16 || positions.dtype != DType::I32 ||
            !k.is_contiguous() || !v.is_contiguous() || !positions.is_contiguous() ||
            k.data == nullptr || v.data == nullptr || positions.data == nullptr) {
            throw std::invalid_argument(
                "KV calibration capture requires contiguous BF16 K/V and I32 positions");
        }
        if (k.ne[0] != v.ne[0] || k.ne[1] != v.ne[1] || k.ne[2] != v.ne[2] || k.ne[3] != 1 ||
            v.ne[3] != 1 || positions.ne[0] != k.ne[2] || positions.ne[1] != 1 ||
            positions.ne[2] != 1 || positions.ne[3] != 1) {
            throw std::invalid_argument("KV calibration capture tensor shapes do not match");
        }
        const auto head_dim = static_cast<std::uint32_t>(k.ne[0]);
        const auto kv_heads = static_cast<std::uint32_t>(k.ne[1]);
        const auto tokens   = static_cast<std::uint32_t>(k.ne[2]);
        if (head_dim == 0 || kv_heads == 0 || tokens == 0 ||
            tokens > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
            throw std::invalid_argument("KV calibration capture shapes are out of range");
        }

        // S38: ENFORCED calibration bound -- cumulative per-layer token
        // cap (NINFER_KV_CALIB_MAX_TOKENS, default 4096). Records beyond
        // the cap are skipped (logged once per layer): a runaway capture
        // on a 57K prompt must be impossible, not merely discouraged.
        const std::uint64_t token_cap = calib_token_cap();
        if (tokens_per_layer_[full_layer] + tokens > token_cap) {
            if (cap_logged_.insert(full_layer).second) {
                std::fprintf(stderr,
                             "[kvcalib] layer %u token cap %llu reached; "
                             "further chunks skipped\n",
                             full_layer,
                             static_cast<unsigned long long>(token_cap));
            }
            return;
        }
        tokens_per_layer_[full_layer] += tokens;
        std::vector<std::int32_t> positions_host(tokens);
        std::vector<std::uint8_t> k_host(k.bytes());
        std::vector<std::uint8_t> v_host(v.bytes());
        // S38: ordered against `stream` (the stream that produced k/v),
        // so the host bytes are exact; one sync per record.
        CUDA_CHECK(cudaMemcpyAsync(positions_host.data(), positions.data,
                                   positions.bytes(), cudaMemcpyDeviceToHost,
                                   stream));
        CUDA_CHECK(cudaMemcpyAsync(k_host.data(), k.data, k.bytes(),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(v_host.data(), v.data, v.bytes(),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        Header header{};
        std::memcpy(header.magic, kMagic, sizeof(header.magic));
        header.header_bytes  = sizeof(Header);
        header.full_layer    = full_layer;
        header.head_dim      = head_dim;
        header.kv_heads      = kv_heads;
        header.tokens        = tokens;
        header.record_index  = record_index_;
        header.first_position = positions_host.front();
        header.last_position  = positions_host.back();

        const std::filesystem::path path =
            directory_ / (std::to_string(record_index_) + ".kvc");
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        if (!out) { throw std::runtime_error("cannot create KV calibration record: " + path.string()); }
        out.write(reinterpret_cast<const char*>(&header), sizeof(header));
        out.write(reinterpret_cast<const char*>(positions_host.data()),
                  static_cast<std::streamsize>(positions_host.size() * sizeof(std::int32_t)));
        out.write(reinterpret_cast<const char*>(k_host.data()),
                  static_cast<std::streamsize>(k_host.size()));
        out.write(reinterpret_cast<const char*>(v_host.data()),
                  static_cast<std::streamsize>(v_host.size()));
        if (!out) { throw std::runtime_error("failed to write KV calibration record: " + path.string()); }
        ++record_index_;
    }

    [[nodiscard]] std::uint32_t record_count() const noexcept { return record_index_; }

private:
    static constexpr char kMagic[16] = {'N', 'I', 'N', 'F', 'E', 'R', 'K', 'V',
                                        'C', 'A', 'L', '1', 0,   0,   0,   0};
    struct Header {
        char magic[16];
        std::uint32_t header_bytes;
        std::uint32_t full_layer;
        std::uint32_t head_dim;
        std::uint32_t kv_heads;
        std::uint32_t tokens;
        std::uint32_t record_index;
        std::int32_t first_position;
        std::int32_t last_position;
        std::uint32_t reserved[4];
    };
    static_assert(sizeof(Header) == 64);

    static std::uint64_t calib_token_cap() {
        static const std::uint64_t cap = [] {
            const char* e = std::getenv("NINFER_KV_CALIB_MAX_TOKENS");
            const long long v = e != nullptr ? std::atoll(e) : 4096;
            return v > 0 ? static_cast<std::uint64_t>(v) : 4096ULL;
        }();
        return cap;
    }

    std::filesystem::path directory_;
    std::uint32_t record_index_ = 0;
    std::map<std::uint32_t, std::uint64_t> tokens_per_layer_;
    std::set<std::uint32_t> cap_logged_;
};

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule
