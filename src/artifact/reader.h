#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace ninfer::artifact {

class ArtifactError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

enum class NumericFormat {
    BF16,
    FP32,
    I32,
    Q4G64_F16S,
    Q5G64_F16S,
    Q6G64_F16S,
    W8G32_F16S,
    NVFP4,
    FP8_E4M3FN_ROW_BF16S,
    // Appended, not renumbered: the artifact wire format carries formats and layouts as the
    // NAMES below (reader.cpp parse_format/parse_layout), so no enumerator ordinal is persisted.
    // Keeping the existing values pinned is this tree's own convention (see KvCacheStorage).
    I64,
    FP8_E4M3FN_ROW_F32S,
    U4Z8G16_F16S,
    // ---- appended 2026-09-29 (line flowopen, marker F1173) ------------------------------------
    // THE LOW-BIT RUNGS THE OWNER ASKED FOR, and the reason they are here rather than in a
    // converter: the container could not NAME them, so every 1-bit / 2-bit / 3-bit source
    // died on the format gate with a refusal that read as "this model is not supported".
    // MEASURED on the four rungs that exist: quant_geometry() is a two-plane bit-split
    //   bits per group == 8 * (base_bytes_per_group + high_bytes_per_group) / group_size
    //   Q4G64_F16S {64,32, 0} = 4.00   Q5G64_F16S {64,32, 8} = 5.00
    //   Q6G64_F16S {64,32,16} = 6.00   W8G32_F16S {32,32, 0} = 8.00
    // so the struct can already express the lower rungs; only the switch's case list was
    // short.  These three carry the CODES.  They do NOT claim a kernel: src/core/arch_caps.h
    // must give each one a Cap::None row naming the absent consumer, and that row is part of
    // this same change -- a carrier with no named consumer is the "admits a file it then
    // cannot read" failure this tree refuses to make.
    Q1G64_F16S,   // 1 bit  -> quant_geometry {64, 8, 0}   bonsai-family sources (ggml Q1_0)
    Q2G64_F16S,   // 2 bits -> quant_geometry {64,16, 0}   Prism PQ2_0 / upstream Q2_0
    Q3G64_F16S,   // 3 bits -> quant_geometry {64,16, 8}   3-bit packed sources (see the note
                  //                                     below: GSQ's pack-int32-le-v1 with
                  //                                     pack_factor 10 is a DIFFERENT packing
                  //                                     and is NOT carried by this member)
    // THE ENUM'S OWN COUNT, and it exists so that "walk every NumericFormat" is a compiler-checked
    // fact instead of a list someone has to remember to extend. It is the LAST member and NOT a
    // format: no artifact object may carry it, no parse arm names it, and format_name() returns {}
    // for it. src/core/arch_caps.h walks 0..Count-1 and static_asserts that its two tables account
    // for every one of them, so APPENDING A NEW FORMAT ABOVE THIS LINE FAILS THE BUILD until the
    // format is given either a capability row or a named refusal -- which is what the previous
    // arrangement (a count copied into another file, plus a test that compared a hand-written
    // 9-element array against itself) did NOT do: I64, FP8_E4M3FN_ROW_F32S and U4Z8G16_F16S were
    // silently uncovered for as long as they existed. Real members are still appended in place and
    // never renumbered (see the note above); keep this one last.
    Count,
};

enum class StorageLayout {
    ContiguousLeV1,
    RowSplitK128V1,
    BlockScaleK16M128x4V1,
    RowScaleV1,
    // Donor spells these separately from RowScaleV1; they are distinct encodings, not renames.
    RowScaleF32V1,
    PackedU4G16V1,
    ExpertBlockScaleK16M128x4V1,
};

enum class ResourceEncoding {
    RawBytesV1,
};

std::string_view format_name(NumericFormat format) noexcept;
std::string_view layout_name(StorageLayout layout) noexcept;
std::string_view encoding_name(ResourceEncoding encoding) noexcept;

std::uint64_t tensor_alignment(StorageLayout layout) noexcept;
std::uint64_t resource_alignment(ResourceEncoding encoding) noexcept;
std::uint64_t tensor_encoded_size(StorageLayout layout, NumericFormat format,
                                  std::span<const std::uint64_t> shape);

struct RowSplitGeometry {
    std::uint64_t rows                 = 0;
    std::uint64_t columns              = 0;
    std::uint64_t padded_columns       = 0;
    std::uint64_t group_size           = 0;
    std::uint64_t groups_per_row       = 0;
    std::uint64_t low_bytes_per_group  = 0;
    std::uint64_t high_bytes_per_group = 0;
    std::uint64_t low_plane_bytes      = 0;
    std::uint64_t high_plane_offset    = 0;
    std::uint64_t high_plane_bytes     = 0;
    std::uint64_t scale_plane_offset   = 0;
    std::uint64_t scale_plane_bytes    = 0;
    std::uint64_t encoded_bytes        = 0;
};

RowSplitGeometry row_split_geometry(NumericFormat format, std::span<const std::uint64_t> shape);

struct BlockScaleGeometry {
    std::uint64_t rows                  = 0;
    std::uint64_t columns               = 0;
    std::uint64_t groups_per_row        = 0;
    std::uint64_t k_tiles               = 0;
    std::uint64_t code_plane_bytes      = 0;
    std::uint64_t scale_plane_offset    = 0;
    std::uint64_t scale_plane_bytes     = 0;
    std::uint64_t weight_divisor_offset = 0;
    std::uint64_t encoded_bytes         = 0;
};

BlockScaleGeometry block_scale_geometry(NumericFormat format, std::span<const std::uint64_t> shape);

struct RowScaleGeometry {
    std::uint64_t rows               = 0;
    std::uint64_t columns            = 0;
    std::uint64_t code_plane_bytes   = 0;
    std::uint64_t scale_plane_offset = 0;
    std::uint64_t scale_plane_bytes  = 0;
    std::uint64_t encoded_bytes      = 0;
};

RowScaleGeometry row_scale_geometry(NumericFormat format, std::span<const std::uint64_t> shape);

// Per-expert NVFP4 bank.  Differs from BlockScaleGeometry by an explicit expert dimension and a
// per-expert weight-divisor array (donor: expert-blockscale-k16-m128x4-v1).
struct BlockScaleBankGeometry {
    std::uint64_t experts               = 0;
    std::uint64_t rows                  = 0;
    std::uint64_t columns               = 0;
    std::uint64_t groups_per_row        = 0;
    std::uint64_t k_tiles               = 0;
    std::uint64_t code_plane_bytes      = 0;
    std::uint64_t scale_plane_offset    = 0;
    std::uint64_t scale_plane_bytes     = 0;
    std::uint64_t weight_divisor_offset = 0;
    std::uint64_t weight_divisor_bytes  = 0;
    std::uint64_t encoded_bytes         = 0;
};

BlockScaleBankGeometry block_scale_bank_geometry(NumericFormat format,
                                                 std::span<const std::uint64_t> shape);

// Packed U4 codes with one FP16 scale per 16 columns (donor: packed-u4-g16-v1; the PLE sidecar).
struct PackedU4Geometry {
    std::uint64_t rows               = 0;
    std::uint64_t columns            = 0;
    std::uint64_t groups_per_row     = 0;
    std::uint64_t code_plane_bytes   = 0;
    std::uint64_t scale_plane_offset = 0;
    std::uint64_t scale_plane_bytes  = 0;
    std::uint64_t encoded_bytes      = 0;
};

PackedU4Geometry packed_u4_geometry(NumericFormat format, std::span<const std::uint64_t> shape);

struct TensorDescriptor {
    std::string name;
    std::vector<std::uint64_t> shape;
    NumericFormat format;
    StorageLayout layout;
    std::uint64_t offset;
    std::uint64_t bytes;
};

struct ResourceDescriptor {
    std::string name;
    ResourceEncoding encoding;
    std::uint64_t offset;
    std::uint64_t bytes;
};

using ObjectDescriptor = std::variant<TensorDescriptor, ResourceDescriptor>;

std::string_view object_name(const ObjectDescriptor& object) noexcept;
std::uint64_t object_offset(const ObjectDescriptor& object) noexcept;
std::uint64_t object_bytes(const ObjectDescriptor& object) noexcept;

struct PayloadSpan {
    std::uint64_t absolute_offset;
    std::span<const std::byte> data;
};

struct ArtifactIdentity {
    std::string model_id;
    std::string weights_id;

    bool operator==(const ArtifactIdentity&) const = default;
};

class Reader {
public:
    static constexpr std::size_t direct_io_alignment = 4096;

    explicit Reader(const std::filesystem::path& path);
    ~Reader();

    Reader(Reader&&) noexcept;
    Reader& operator=(Reader&&) noexcept;
    Reader(const Reader&)            = delete;
    Reader& operator=(const Reader&) = delete;

    const ArtifactIdentity& identity() const noexcept;
    const std::vector<ObjectDescriptor>& objects() const noexcept;
    const ObjectDescriptor* find(std::string_view name) const noexcept;

    std::uint64_t file_bytes() const noexcept;
    std::uint64_t payload_offset() const noexcept;
    PayloadSpan payload(const ObjectDescriptor& object) const;
    PayloadSpan payload(std::string_view name) const;
    std::size_t read_direct(std::uint64_t absolute_offset, std::span<std::byte> destination) const;

    // A lease on this Reader's file mapping. Holding one keeps the mapped bytes valid after the
    // Reader itself is destroyed; the mapping is released when the last lease drops. Empty for a
    // Reader that opened without a mapping.
    //
    // WHY THIS EXISTS (MEASURED 2026-09-18, FN-GATE): MaterializedArtifact::mapped_tensor_bytes()
    // returns a span that ALIASES this mapping (see the lifetime contract in
    // src/artifact/materializer.h), and MappedFile's destructor munmaps
    // (src/artifact/reader.cpp:224). Without a lease, any caller that drops its Reader -- which is
    // what the FlashNext target's load_from_file does on return -- leaves that span pointing at an
    // unmapped range, so it reads unmapped memory rather than merely stale memory. A probe that
    // destroys the Reader and then reads the span segfaults without this and passes with it
    // (scratch/PATCHSET/FNGATE/probe/mapped_lease_probe.cpp).
    //
    // The donor held the same thing (igorls/ninfer @ 5e4a66d src/artifact/materializer.h:58,
    // `std::shared_ptr<const void> mapping_lease_`); this tree had no way to take one until now.
    //
    // ADDITIVE: this adds an accessor and changes one PRIVATE member's smart-pointer type. No
    // Reader's public shape changes, and a Reader that is never leased is destroyed exactly as
    // before.
    [[nodiscard]] std::shared_ptr<const void> mapping_lease() const;

private:
    struct Impl;
    // shared_ptr, not unique_ptr, only so a lease can outlive the Reader. Everything else about
    // Reader's lifetime is unchanged.
    std::shared_ptr<Impl> impl_;
};

} // namespace ninfer::artifact
