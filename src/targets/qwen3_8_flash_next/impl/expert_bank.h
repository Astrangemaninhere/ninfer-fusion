// Derived from igorls/ninfer @ 5e4a66d (Apache-2.0), modified for this tree (ninfer-fusion).
#pragma once

#include <cstddef>
#include <cstdint>

namespace ninfer::artifact {
class MaterializedArtifact;
struct ObjectHandle;
} // namespace ninfer::artifact

namespace ninfer::targets::qwen3_8_flash_next::detail {

struct Nvfp4ExpertMatrixView {
    const std::byte* codes;
    const std::byte* scales;
    const float* weight_scale_divisor;
    float input_scale_divisor;
    std::int32_t rows;
    std::int32_t columns;
};

struct Nvfp4ExpertBankView {
    const std::byte* codes;
    const std::byte* scales;
    const float* weight_scale_divisors;
    std::int32_t experts;
    std::int32_t rows;
    std::int32_t columns;
    std::uint64_t code_bytes_per_expert;
    std::uint64_t scale_bytes_per_expert;

    [[nodiscard]] Nvfp4ExpertMatrixView expert(std::int32_t index) const;
};

[[nodiscard]] Nvfp4ExpertBankView
make_nvfp4_expert_bank_view(const void* payload, std::uint64_t payload_bytes, std::int32_t experts,
                            std::int32_t rows, std::int32_t columns);

[[nodiscard]] Nvfp4ExpertBankView
materialized_nvfp4_expert_bank_view(const artifact::MaterializedArtifact& materialized,
                                    artifact::ObjectHandle handle, std::int32_t experts,
                                    std::int32_t rows, std::int32_t columns);

struct Bf16ExpertMatrixView {
    const std::byte* data;
    std::int32_t rows;
    std::int32_t columns;
};

struct Bf16ExpertBankView {
    const std::byte* data;
    std::int32_t experts;
    std::int32_t rows;
    std::int32_t columns;
    std::uint64_t bytes_per_expert;

    [[nodiscard]] Bf16ExpertMatrixView expert(std::int32_t index) const;
};

[[nodiscard]] Bf16ExpertBankView
make_bf16_expert_bank_view(const void* payload, std::uint64_t payload_bytes, std::int32_t experts,
                           std::int32_t rows, std::int32_t columns);

[[nodiscard]] Bf16ExpertBankView
materialized_bf16_expert_bank_view(const artifact::MaterializedArtifact& materialized,
                                   artifact::ObjectHandle handle, std::int32_t experts,
                                   std::int32_t rows, std::int32_t columns);

} // namespace ninfer::targets::qwen3_8_flash_next::detail
