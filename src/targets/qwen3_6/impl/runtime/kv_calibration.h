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
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule {

// Offline KV calibration capture. When EngineOptions.kv_calibration_dir is set,
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

    void capture(std::uint32_t full_layer, const Tensor& k, const Tensor& v,
                 const Tensor& positions) {
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

        std::vector<std::int32_t> positions_host(tokens);
        std::vector<std::uint8_t> k_host(k.bytes());
        std::vector<std::uint8_t> v_host(v.bytes());
        CUDA_CHECK(cudaMemcpy(positions_host.data(), positions.data, positions.bytes(),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(k_host.data(), k.data, k.bytes(), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(v_host.data(), v.data, v.bytes(), cudaMemcpyDeviceToHost));

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

    std::filesystem::path directory_;
    std::uint32_t record_index_ = 0;
};

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule
