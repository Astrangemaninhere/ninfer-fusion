// Derived from igorls/ninfer @ 5e4a66d (Apache-2.0), modified for this tree (ninfer-fusion).
#include "targets/qwen3_8_flash_next/impl/expert_bank.h"

#include "artifact/materializer.h"
#include "artifact/reader.h"

#include <array>
#include <stdexcept>

namespace ninfer::targets::qwen3_8_flash_next::detail {

Nvfp4ExpertMatrixView Nvfp4ExpertBankView::expert(std::int32_t index) const {
    if (index < 0 || index >= experts) {
        throw std::out_of_range("NVFP4 expert index is outside the bank");
    }
    return {
        .codes                = codes + static_cast<std::uint64_t>(index) * code_bytes_per_expert,
        .scales               = scales + static_cast<std::uint64_t>(index) * scale_bytes_per_expert,
        .weight_scale_divisor = weight_scale_divisors + index,
        .input_scale_divisor  = 1.0F,
        .rows                 = rows,
        .columns              = columns,
    };
}

Nvfp4ExpertBankView make_nvfp4_expert_bank_view(const void* payload, std::uint64_t payload_bytes,
                                                std::int32_t experts, std::int32_t rows,
                                                std::int32_t columns) {
    if (payload == nullptr || experts <= 0 || rows <= 0 || columns <= 0) {
        throw std::invalid_argument("NVFP4 expert bank has invalid storage or shape");
    }
    const std::array<std::uint64_t, 3> shape = {static_cast<std::uint64_t>(experts),
                                                static_cast<std::uint64_t>(rows),
                                                static_cast<std::uint64_t>(columns)};
    const artifact::BlockScaleBankGeometry geometry =
        artifact::block_scale_bank_geometry(artifact::NumericFormat::NVFP4, shape);
    if (payload_bytes != geometry.encoded_bytes) {
        throw std::invalid_argument("NVFP4 expert bank payload has the wrong size");
    }
    const auto* bytes = static_cast<const std::byte*>(payload);
    return {
        .codes  = bytes,
        .scales = bytes + geometry.scale_plane_offset,
        .weight_scale_divisors =
            reinterpret_cast<const float*>(bytes + geometry.weight_divisor_offset),
        .experts                = experts,
        .rows                   = rows,
        .columns                = columns,
        .code_bytes_per_expert  = geometry.code_plane_bytes / static_cast<std::uint64_t>(experts),
        .scale_bytes_per_expert = geometry.scale_plane_bytes / static_cast<std::uint64_t>(experts),
    };
}

Nvfp4ExpertBankView
materialized_nvfp4_expert_bank_view(const artifact::MaterializedArtifact& materialized,
                                    artifact::ObjectHandle handle, std::int32_t experts,
                                    std::int32_t rows, std::int32_t columns) {
    if (experts <= 0 || rows <= 0 || columns <= 0) {
        throw std::invalid_argument("NVFP4 expert bank has invalid shape");
    }
    const std::array<std::uint64_t, 3> shape = {static_cast<std::uint64_t>(experts),
                                                static_cast<std::uint64_t>(rows),
                                                static_cast<std::uint64_t>(columns)};
    const artifact::BlockScaleBankGeometry geometry =
        artifact::block_scale_bank_geometry(artifact::NumericFormat::NVFP4, shape);
    return make_nvfp4_expert_bank_view(materialized.device_data(handle), geometry.encoded_bytes,
                                       experts, rows, columns);
}

Bf16ExpertMatrixView Bf16ExpertBankView::expert(std::int32_t index) const {
    if (index < 0 || index >= experts) {
        throw std::out_of_range("BF16 expert index is outside the bank");
    }
    return {
        .data    = data + static_cast<std::uint64_t>(index) * bytes_per_expert,
        .rows    = rows,
        .columns = columns,
    };
}

Bf16ExpertBankView make_bf16_expert_bank_view(const void* payload, std::uint64_t payload_bytes,
                                              std::int32_t experts, std::int32_t rows,
                                              std::int32_t columns) {
    if (payload == nullptr || experts <= 0 || rows <= 0 || columns <= 0) {
        throw std::invalid_argument("BF16 expert bank has invalid storage or shape");
    }
    const std::uint64_t bytes_per_expert =
        static_cast<std::uint64_t>(rows) * columns * sizeof(std::uint16_t);
    const std::uint64_t expected_total_bytes =
        bytes_per_expert * static_cast<std::uint64_t>(experts);
    if (payload_bytes < expected_total_bytes) {
        throw std::invalid_argument("BF16 expert bank payload has the wrong size");
    }
    return {
        .data             = static_cast<const std::byte*>(payload),
        .experts          = experts,
        .rows             = rows,
        .columns          = columns,
        .bytes_per_expert = bytes_per_expert,
    };
}

Bf16ExpertBankView
materialized_bf16_expert_bank_view(const artifact::MaterializedArtifact& materialized,
                                   artifact::ObjectHandle handle, std::int32_t experts,
                                   std::int32_t rows, std::int32_t columns) {
    if (experts <= 0 || rows <= 0 || columns <= 0) {
        throw std::invalid_argument("BF16 expert bank has invalid shape");
    }
    const std::uint64_t total_bytes = static_cast<std::uint64_t>(experts) * rows * columns * 2;
    return make_bf16_expert_bank_view(materialized.device_data(handle), total_bytes, experts, rows,
                                      columns);
}

} // namespace ninfer::targets::qwen3_8_flash_next::detail
