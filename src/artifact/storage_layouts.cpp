#include "artifact/reader.h"

#include <limits>

namespace ninfer::artifact {
namespace {

constexpr std::uint64_t kTensorAlignment = 256;
constexpr std::uint64_t kKAlignment      = 128;

std::uint64_t checked_add(std::uint64_t a, std::uint64_t b, std::string_view label) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a) {
        throw ArtifactError(std::string(label) + " overflows u64");
    }
    return a + b;
}

std::uint64_t checked_mul(std::uint64_t a, std::uint64_t b, std::string_view label) {
    if (a != 0 && b > std::numeric_limits<std::uint64_t>::max() / a) {
        throw ArtifactError(std::string(label) + " overflows u64");
    }
    return a * b;
}

std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment, std::string_view label) {
    const auto biased = checked_add(value, alignment - 1, label);
    return biased / alignment * alignment;
}

struct QuantGeometry {
    std::uint64_t group_size;
    std::uint64_t base_bytes_per_group;
    std::uint64_t high_bytes_per_group;
};

QuantGeometry quant_geometry(NumericFormat format) {
    switch (format) {
    case NumericFormat::Q4G64_F16S:
        return {64, 32, 0};
    case NumericFormat::Q5G64_F16S:
        return {64, 32, 8};
    case NumericFormat::Q6G64_F16S:
        return {64, 32, 16};
    case NumericFormat::W8G32_F16S:
        return {32, 32, 0};
    default:
        throw ArtifactError("row-split-k128-v1 requires a grouped quantized format");
    }
}

std::uint64_t direct_word_bytes(NumericFormat format) {
    switch (format) {
    case NumericFormat::BF16:
        return 2;
    case NumericFormat::FP32:
    case NumericFormat::I32:
        return 4;
    case NumericFormat::I64:
        return 8;
    default:
        throw ArtifactError("contiguous-le-v1 requires BF16, FP32, I32, or I64");
    }
}

} // namespace

std::string_view format_name(NumericFormat format) noexcept {
    switch (format) {
    case NumericFormat::BF16:
        return "BF16";
    case NumericFormat::FP32:
        return "FP32";
    case NumericFormat::I32:
        return "I32";
    case NumericFormat::Q4G64_F16S:
        return "Q4G64_F16S";
    case NumericFormat::Q5G64_F16S:
        return "Q5G64_F16S";
    case NumericFormat::Q6G64_F16S:
        return "Q6G64_F16S";
    case NumericFormat::W8G32_F16S:
        return "W8G32_F16S";
    case NumericFormat::NVFP4:
        return "NVFP4";
    case NumericFormat::FP8_E4M3FN_ROW_BF16S:
        return "FP8_E4M3FN_ROW_BF16S";
    case NumericFormat::I64:
        return "I64";
    case NumericFormat::FP8_E4M3FN_ROW_F32S:
        return "FP8_E4M3FN_ROW_F32S";
    case NumericFormat::U4Z8G16_F16S:
        return "U4Z8G16_F16S";
    case NumericFormat::Count:
        // Not a format: the enum's own count (artifact/reader.h). Named so that this switch, which
        // has no `default:` on purpose, stays exhaustive under -Wswitch -- and so that a REAL
        // member added above it is still the thing that warns.
        return {};
    }
    return {};
}

std::string_view layout_name(StorageLayout layout) noexcept {
    switch (layout) {
    case StorageLayout::ContiguousLeV1:
        return "contiguous-le-v1";
    case StorageLayout::RowSplitK128V1:
        return "row-split-k128-v1";
    case StorageLayout::BlockScaleK16M128x4V1:
        return "blockscale-k16-m128x4-v1";
    case StorageLayout::RowScaleV1:
        return "row-scale-v1";
    case StorageLayout::RowScaleF32V1:
        return "row-scale-f32-v1";
    case StorageLayout::PackedU4G16V1:
        return "packed-u4-g16-v1";
    case StorageLayout::ExpertBlockScaleK16M128x4V1:
        return "expert-blockscale-k16-m128x4-v1";
    }
    return {};
}

std::string_view encoding_name(ResourceEncoding encoding) noexcept {
    switch (encoding) {
    case ResourceEncoding::RawBytesV1:
        return "raw-bytes-v1";
    }
    return {};
}

std::uint64_t tensor_alignment(StorageLayout) noexcept { return kTensorAlignment; }

std::uint64_t resource_alignment(ResourceEncoding) noexcept { return 1; }

std::uint64_t tensor_encoded_size(StorageLayout layout, NumericFormat format,
                                  std::span<const std::uint64_t> shape) {
    if (layout == StorageLayout::ContiguousLeV1) {
        if (shape.size() > 16) {
            throw ArtifactError("contiguous-le-v1 supports rank 0 through 16");
        }
        std::uint64_t elements = 1;
        for (const auto dim : shape) {
            if (dim == 0) { throw ArtifactError("tensor shape dimensions must be positive"); }
            elements = checked_mul(elements, dim, "tensor element count");
        }
        return checked_mul(elements, direct_word_bytes(format), "tensor encoded size");
    }

    if (layout == StorageLayout::RowSplitK128V1) {
        if (shape.size() != 2 || shape[0] == 0 || shape[1] == 0) {
            throw ArtifactError("row-split-k128-v1 requires a positive rank-two shape");
        }
        return row_split_geometry(format, shape).encoded_bytes;
    }
    if (layout == StorageLayout::BlockScaleK16M128x4V1) {
        return block_scale_geometry(format, shape).encoded_bytes;
    }
    if (layout == StorageLayout::RowScaleV1) {
        if (format != NumericFormat::FP8_E4M3FN_ROW_BF16S) {
            throw ArtifactError("row-scale-v1 requires FP8_E4M3FN_ROW_BF16S");
        }
        return row_scale_geometry(format, shape).encoded_bytes;
    }
    if (layout == StorageLayout::RowScaleF32V1) {
        if (format != NumericFormat::FP8_E4M3FN_ROW_F32S) {
            throw ArtifactError("row-scale-f32-v1 requires FP8_E4M3FN_ROW_F32S");
        }
        return row_scale_geometry(format, shape).encoded_bytes;
    }
    if (layout == StorageLayout::PackedU4G16V1) {
        return packed_u4_geometry(format, shape).encoded_bytes;
    }
    if (layout == StorageLayout::ExpertBlockScaleK16M128x4V1) {
        return block_scale_bank_geometry(format, shape).encoded_bytes;
    }
    throw ArtifactError("unknown tensor layout");
}

RowSplitGeometry row_split_geometry(NumericFormat format, std::span<const std::uint64_t> shape) {
    if (shape.size() != 2 || shape[0] == 0 || shape[1] == 0) {
        throw ArtifactError("row-split-k128-v1 requires a positive rank-two shape");
    }
    const auto format_geometry = quant_geometry(format);
    RowSplitGeometry out;
    out.rows                 = shape[0];
    out.columns              = shape[1];
    out.padded_columns       = align_up(shape[1], kKAlignment, "padded K");
    out.group_size           = format_geometry.group_size;
    out.groups_per_row       = out.padded_columns / out.group_size;
    out.low_bytes_per_group  = format_geometry.base_bytes_per_group;
    out.high_bytes_per_group = format_geometry.high_bytes_per_group;
    const auto groups        = checked_mul(out.rows, out.groups_per_row, "physical group count");
    out.low_plane_bytes      = checked_mul(groups, out.low_bytes_per_group, "base plane bytes");
    out.high_plane_bytes     = checked_mul(groups, out.high_bytes_per_group, "high plane bytes");
    out.scale_plane_bytes    = checked_mul(groups, 2, "scale plane bytes");
    out.high_plane_offset    = align_up(out.low_plane_bytes, kTensorAlignment, "high plane offset");
    const auto aligned_high =
        align_up(out.high_plane_bytes, kTensorAlignment, "scale plane alignment");
    out.scale_plane_offset = checked_add(out.high_plane_offset, aligned_high, "scale plane offset");
    out.encoded_bytes =
        checked_add(out.scale_plane_offset, out.scale_plane_bytes, "tensor encoded size");
    return out;
}

BlockScaleGeometry block_scale_geometry(NumericFormat format,
                                        std::span<const std::uint64_t> shape) {
    if (format != NumericFormat::NVFP4) {
        throw ArtifactError("blockscale-k16-m128x4-v1 requires NVFP4");
    }
    if (shape.size() != 2 || shape[0] == 0 || shape[1] == 0) {
        throw ArtifactError("blockscale-k16-m128x4-v1 requires a positive rank-two shape");
    }
    if (shape[0] % 128 != 0 || shape[1] % 64 != 0) {
        throw ArtifactError(
            "blockscale-k16-m128x4-v1 requires N divisible by 128 and K divisible by 64");
    }

    BlockScaleGeometry out;
    out.rows             = shape[0];
    out.columns          = shape[1];
    out.groups_per_row   = shape[1] / 16;
    out.k_tiles          = shape[1] / 64;
    const auto elements  = checked_mul(out.rows, out.columns, "NVFP4 element count");
    out.code_plane_bytes = elements / 2;
    out.scale_plane_offset =
        align_up(out.code_plane_bytes, kTensorAlignment, "NVFP4 scale plane offset");
    out.scale_plane_bytes = elements / 16;
    out.weight_divisor_offset =
        checked_add(out.scale_plane_offset, out.scale_plane_bytes, "NVFP4 weight divisor offset");
    out.encoded_bytes = checked_add(out.weight_divisor_offset, 4, "NVFP4 tensor encoded size");
    return out;
}

RowScaleGeometry row_scale_geometry(NumericFormat format, std::span<const std::uint64_t> shape) {
    // Scale word width is the ONLY difference between the two row-scaled FP8 encodings, and it is
    // load-bearing: 2 bytes for ..._BF16S, 4 for ..._F32S (donor, same function).
    std::uint64_t scale_word_bytes = 0;
    switch (format) {
    case NumericFormat::FP8_E4M3FN_ROW_BF16S:
        scale_word_bytes = 2;
        break;
    case NumericFormat::FP8_E4M3FN_ROW_F32S:
        scale_word_bytes = 4;
        break;
    default:
        throw ArtifactError("row-scaled layout requires a row-scaled FP8 format");
    }
    if (shape.size() != 2 || shape[0] == 0 || shape[1] == 0) {
        throw ArtifactError("row-scale-v1 requires a positive rank-two shape");
    }

    RowScaleGeometry out;
    out.rows             = shape[0];
    out.columns          = shape[1];
    out.code_plane_bytes = checked_mul(out.rows, out.columns, "FP8 element count");
    out.scale_plane_offset =
        align_up(out.code_plane_bytes, kTensorAlignment, "FP8 scale plane offset");
    out.scale_plane_bytes = checked_mul(out.rows, scale_word_bytes, "FP8 scale plane bytes");
    out.encoded_bytes =
        checked_add(out.scale_plane_offset, out.scale_plane_bytes, "FP8 tensor encoded size");
    return out;
}

// Donor: igorls/ninfer @ 5e4a66d src/artifact/storage_layouts.cpp, verbatim.
BlockScaleBankGeometry block_scale_bank_geometry(NumericFormat format,
                                                 std::span<const std::uint64_t> shape) {
    if (format != NumericFormat::NVFP4) {
        throw ArtifactError("expert-blockscale-k16-m128x4-v1 requires NVFP4");
    }
    if (shape.size() != 3 || shape[0] == 0 || shape[1] == 0 || shape[2] == 0) {
        throw ArtifactError("expert-blockscale-k16-m128x4-v1 requires a positive rank-three shape");
    }
    if (shape[1] % 128 != 0 || shape[2] % 64 != 0) {
        throw ArtifactError(
            "expert-blockscale-k16-m128x4-v1 requires N divisible by 128 and K divisible by 64");
    }

    BlockScaleBankGeometry out;
    out.experts          = shape[0];
    out.rows             = shape[1];
    out.columns          = shape[2];
    out.groups_per_row   = shape[2] / 16;
    out.k_tiles          = shape[2] / 64;
    const auto matrices  = checked_mul(out.experts, out.rows, "NVFP4 bank row count");
    const auto elements  = checked_mul(matrices, out.columns, "NVFP4 bank element count");
    out.code_plane_bytes = elements / 2;
    out.scale_plane_offset =
        align_up(out.code_plane_bytes, kTensorAlignment, "NVFP4 bank scale plane offset");
    out.scale_plane_bytes     = elements / 16;
    out.weight_divisor_offset = checked_add(out.scale_plane_offset, out.scale_plane_bytes,
                                            "NVFP4 bank weight divisor offset");
    out.weight_divisor_bytes  = checked_mul(out.experts, 4, "NVFP4 bank divisor bytes");
    out.encoded_bytes         = checked_add(out.weight_divisor_offset, out.weight_divisor_bytes,
                                            "NVFP4 bank tensor encoded size");
    return out;
}

// Donor: igorls/ninfer @ 5e4a66d src/artifact/storage_layouts.cpp, verbatim.
PackedU4Geometry packed_u4_geometry(NumericFormat format, std::span<const std::uint64_t> shape) {
    if (format != NumericFormat::U4Z8G16_F16S) {
        throw ArtifactError("packed-u4-g16-v1 requires U4Z8G16_F16S");
    }
    if (shape.size() != 2 || shape[0] == 0 || shape[1] == 0 || shape[1] % 16 != 0) {
        throw ArtifactError(
            "packed-u4-g16-v1 requires a positive rank-two shape with K divisible by 16");
    }

    PackedU4Geometry out;
    out.rows             = shape[0];
    out.columns          = shape[1];
    out.groups_per_row   = shape[1] / 16;
    const auto elements  = checked_mul(out.rows, out.columns, "U4 element count");
    out.code_plane_bytes = elements / 2;
    out.scale_plane_offset =
        align_up(out.code_plane_bytes, kTensorAlignment, "U4 scale plane offset");
    out.scale_plane_bytes = checked_mul(checked_mul(out.rows, out.groups_per_row, "U4 group count"),
                                        2, "U4 scale plane bytes");
    out.encoded_bytes =
        checked_add(out.scale_plane_offset, out.scale_plane_bytes, "U4 tensor encoded size");
    return out;
}

} // namespace ninfer::artifact
