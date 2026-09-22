#include "artifact/reader.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstring>
#include <functional>
#include <limits>
#include <span>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <unordered_map>
#include <utility>

#if defined(_WIN32)
// THE TREE'S FIRST <windows.h> -- src, apps and include contain none today. Two macros, and one of
// them is REQUIRED rather than tidiness:
//   NOMINMAX  windows.h defines a function-like `max` macro. This file calls
//             std::numeric_limits<...>::max() in checked_add/align_up and in the size guard below,
//             and with that macro live those calls are ill-formed.
//   WIN32_LEAN_AND_MEAN  keeps winsock/GDI/comm out of a translation unit that wants only the file
//             and memory-mapping APIs.
// Widening the platform claim is deliberate: this header pair is what the arm needs, and there is no
// narrower supported spelling for CreateFileW/CreateFileMappingW/MapViewOfFile.
#    define WIN32_LEAN_AND_MEAN
#    define NOMINMAX
#    include <windows.h>
#else
#    include <fcntl.h>
#    include <sys/mman.h>
#    include <sys/stat.h>
#    include <unistd.h>
#endif

namespace ninfer::artifact {
namespace {

using Json = nlohmann::json;

constexpr std::array<std::byte, 8> kMagic = {
    std::byte{'N'}, std::byte{'I'}, std::byte{'N'}, std::byte{'F'},
    std::byte{'E'}, std::byte{'R'}, std::byte{0},   std::byte{2},
};
constexpr std::uint64_t kPrefixBytes      = 16;
constexpr std::uint64_t kPayloadAlignment = 4096;

std::uint64_t checked_add(std::uint64_t a, std::uint64_t b, std::string_view label) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a) {
        throw ArtifactError(std::string(label) + " overflows u64");
    }
    return a + b;
}

std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment, std::string_view label) {
    const auto biased = checked_add(value, alignment - 1, label);
    return biased / alignment * alignment;
}

std::uint64_t read_u64_le(const std::byte* data) noexcept {
    std::uint64_t value = 0;
    for (unsigned i = 0; i < 8; ++i) {
        value |= std::uint64_t(std::to_integer<unsigned char>(data[i])) << (i * 8);
    }
    return value;
}

template <std::size_t N>
void require_members(const Json& value, const std::array<const char*, N>& members,
                     std::string_view label) {
    if (!value.is_object() || value.size() != N) {
        throw ArtifactError(std::string(label) + " has missing or extra members");
    }
    for (const char* member : members) {
        if (!value.contains(member)) {
            throw ArtifactError(std::string(label) + " has missing or extra members");
        }
    }
}

const std::string& require_string(const Json& value, std::string_view label) {
    if (!value.is_string()) {
        throw ArtifactError(std::string(label) + " must be a nonempty string");
    }
    const auto& result = value.get_ref<const std::string&>();
    if (result.empty()) { throw ArtifactError(std::string(label) + " must be a nonempty string"); }
    return result;
}

std::uint64_t require_unsigned(const Json& value, std::string_view label, bool positive) {
    if (!value.is_number_unsigned()) {
        throw ArtifactError(std::string(label) + " must be an integer");
    }
    const auto result = value.get<std::uint64_t>();
    if (positive && result == 0) { throw ArtifactError(std::string(label) + " must be positive"); }
    return result;
}

NumericFormat parse_format(std::string_view name) {
    if (name == "BF16") { return NumericFormat::BF16; }
    if (name == "FP32") { return NumericFormat::FP32; }
    if (name == "I32") { return NumericFormat::I32; }
    if (name == "Q4G64_F16S") { return NumericFormat::Q4G64_F16S; }
    if (name == "Q5G64_F16S") { return NumericFormat::Q5G64_F16S; }
    if (name == "Q6G64_F16S") { return NumericFormat::Q6G64_F16S; }
    if (name == "W8G32_F16S") { return NumericFormat::W8G32_F16S; }
    if (name == "NVFP4") { return NumericFormat::NVFP4; }
    if (name == "FP8_E4M3FN_ROW_BF16S") { return NumericFormat::FP8_E4M3FN_ROW_BF16S; }
    // Donor spellings (igorls/ninfer @ 5e4a66d src/artifact/reader.cpp).
    if (name == "I64") { return NumericFormat::I64; }
    if (name == "FP8_E4M3FN_ROW_F32S") { return NumericFormat::FP8_E4M3FN_ROW_F32S; }
    if (name == "U4Z8G16_F16S") { return NumericFormat::U4Z8G16_F16S; }
    throw ArtifactError("unknown tensor format: " + std::string(name));
}

StorageLayout parse_layout(std::string_view name) {
    if (name == "contiguous-le-v1") { return StorageLayout::ContiguousLeV1; }
    if (name == "row-split-k128-v1") { return StorageLayout::RowSplitK128V1; }
    if (name == "blockscale-k16-m128x4-v1") { return StorageLayout::BlockScaleK16M128x4V1; }
    if (name == "row-scale-v1") { return StorageLayout::RowScaleV1; }
    // Donor spellings (igorls/ninfer @ 5e4a66d src/artifact/reader.cpp).
    if (name == "row-scale-f32-v1") { return StorageLayout::RowScaleF32V1; }
    if (name == "packed-u4-g16-v1") { return StorageLayout::PackedU4G16V1; }
    if (name == "expert-blockscale-k16-m128x4-v1") {
        return StorageLayout::ExpertBlockScaleK16M128x4V1;
    }
    throw ArtifactError("unknown tensor layout: " + std::string(name));
}

ResourceEncoding parse_encoding(std::string_view name) {
    if (name == "raw-bytes-v1") { return ResourceEncoding::RawBytesV1; }
    throw ArtifactError("unknown resource encoding: " + std::string(name));
}

TensorDescriptor parse_tensor(const Json& value) {
    static constexpr std::array members = {
        "name", "kind", "shape", "format", "layout", "offset", "bytes",
    };
    require_members(value, members, "tensor entry");

    const auto name        = require_string(value.at("name"), "tensor name");
    const auto format      = parse_format(require_string(value.at("format"), "tensor format"));
    const auto layout      = parse_layout(require_string(value.at("layout"), "tensor layout"));
    const auto offset      = require_unsigned(value.at("offset"), "tensor offset", false);
    const auto stored_size = require_unsigned(value.at("bytes"), "tensor bytes", true);

    const auto& raw_shape = value.at("shape");
    if (!raw_shape.is_array()) { throw ArtifactError("tensor shape must be an array"); }
    std::vector<std::uint64_t> shape;
    shape.reserve(raw_shape.size());
    for (const auto& dim : raw_shape) {
        shape.push_back(require_unsigned(dim, "shape dimension", true));
    }

    const auto expected_size = tensor_encoded_size(layout, format, shape);
    if (stored_size != expected_size) {
        throw ArtifactError("tensor " + name + " stores " + std::to_string(stored_size) +
                            " bytes; layout requires " + std::to_string(expected_size));
    }
    return {name, std::move(shape), format, layout, offset, stored_size};
}

ResourceDescriptor parse_resource(const Json& value) {
    static constexpr std::array members = {
        "name", "kind", "encoding", "offset", "bytes",
    };
    require_members(value, members, "resource entry");
    return {
        require_string(value.at("name"), "resource name"),
        parse_encoding(require_string(value.at("encoding"), "resource encoding")),
        require_unsigned(value.at("offset"), "resource offset", false),
        require_unsigned(value.at("bytes"), "resource bytes", true),
    };
}

ObjectDescriptor parse_object(const Json& value) {
    if (!value.is_object()) { throw ArtifactError("each object entry must be a JSON object"); }
    const auto it = value.find("kind");
    if (it == value.end() || !it->is_string()) {
        throw ArtifactError("object kind must be 'tensor' or 'resource'");
    }
    const auto& kind = it->get_ref<const std::string&>();
    if (kind == "tensor") { return parse_tensor(value); }
    if (kind == "resource") { return parse_resource(value); }
    throw ArtifactError("object kind must be 'tensor' or 'resource'");
}

struct TransparentStringHash {
    using is_transparent = void;

    std::size_t operator()(std::string_view value) const noexcept {
        return std::hash<std::string_view>{}(value);
    }

    std::size_t operator()(const std::string& value) const noexcept {
        return (*this)(std::string_view(value));
    }
};

class MappedFile {
public:
    explicit MappedFile(const std::filesystem::path& path) {
#if defined(_WIN32)
        // TWO handles on this path, and that is a design decision rather than a transliteration of
        // the POSIX arm's one. The POSIX arm opens a single O_DIRECT fd and spends it on both jobs,
        // which works because O_DIRECT constrains read/write/pread and leaves mmap alone. Windows
        // splits those jobs across two flag contracts -- an ordinary handle for
        // CreateFileMapping/MapViewOfFile, FILE_FLAG_NO_BUFFERING for the unbuffered read -- so each
        // job gets the handle whose contract is unambiguous. This port does not depend on the two
        // flag sets being legal together at all; no vendor documentation was consulted under this
        // task's "no support claim" limit, and the combination is therefore not relied on.
        //
        // O_RDONLY maps to GENERIC_READ. O_CLOEXEC needs no separate spelling and gets none: a
        // CreateFileW with NULL SECURITY_ATTRIBUTES returns a NON-INHERITABLE handle, which is the
        // property O_CLOEXEC buys on POSIX.
        file_handle_ = ::CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE,
                                     nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (file_handle_ == INVALID_HANDLE_VALUE) {
            throw std::system_error(static_cast<int>(::GetLastError()), std::system_category(),
                                    "CreateFile " + path.string());
        }

        LARGE_INTEGER file_size{};
        if (::GetFileSizeEx(file_handle_, &file_size) == 0) {
            const int error = static_cast<int>(::GetLastError());
            close_all();
            throw std::system_error(error, std::system_category(), "GetFileSizeEx " + path.string());
        }
        if (file_size.QuadPart < 0 ||
            static_cast<std::uintmax_t>(file_size.QuadPart) >
                std::numeric_limits<std::size_t>::max()) {
            close_all();
            throw ArtifactError("artifact size does not fit the process address space");
        }

        const auto size = static_cast<std::size_t>(file_size.QuadPart);
        if (size != 0) {
            // PAGE_READONLY + FILE_MAP_READ is this arm's pairing for MAP_PRIVATE + PROT_READ. The
            // difference is one-sided: MAP_PRIVATE would let a writer materialise private pages that
            // nobody else sees, and this class never writes, so refusing the write outright is the
            // stronger of the two. The zero-size case is skipped on BOTH arms and the guard is
            // load-bearing on both: the POSIX arm skips mmap of a zero-length file, and
            // CreateFileMapping over a zero-length file fails with ERROR_FILE_INVALID.
            mapping_handle_ =
                ::CreateFileMappingW(file_handle_, nullptr, PAGE_READONLY, 0, 0, nullptr);
            if (mapping_handle_ == nullptr) {
                const int error = static_cast<int>(::GetLastError());
                close_all();
                throw std::system_error(error, std::system_category(),
                                        "CreateFileMapping " + path.string());
            }
            const void* view = ::MapViewOfFile(mapping_handle_, FILE_MAP_READ, 0, 0, 0);
            if (view == nullptr) {
                const int error = static_cast<int>(::GetLastError());
                close_all();
                throw std::system_error(error, std::system_category(),
                                        "MapViewOfFile " + path.string());
            }
            data_ = static_cast<const std::byte*>(view);
        }

        // FILE_FLAG_NO_BUFFERING is this arm's stand-in for O_DIRECT. Opened here, EAGERLY, rather
        // than on the first read_direct call: eager keeps the open path single and free of
        // first-use races, and a Reader opens one artifact once, so the second CreateFileW costs
        // nothing that matters. Its failure is this constructor's failure, which is what O_DIRECT's
        // failure already is on the POSIX arm -- an O_DIRECT open fails outright on a filesystem
        // that does not support it.
        //
        // WHY THE SEMANTICS STILL HOLD, given that the two flags are not the same flag:
        //   O_DIRECT               bypasses the page cache for read/write/pread and requires the
        //                          file offset, the transfer length and the destination address to
        //                          be aligned.
        //   FILE_FLAG_NO_BUFFERING bypasses the cache for ReadFile and requires the offset and the
        //                          length to be multiples of the VOLUME SECTOR SIZE and the
        //                          destination to be sector-aligned.
        // Both demand the same three things, and the caller's guarantee is stronger than either:
        // Reader::direct_io_alignment is 4096 and read_direct REFUSES anything not 4096-aligned,
        // while Windows volumes use 512-byte or 4096-byte sectors, and 4096 % 512 == 0 and
        // 4096 % 4096 == 0 (ARITHMETIC, and those are the sector sizes Windows uses for fixed and
        // removable disks). Every request this class accepts is therefore also a legal unbuffered
        // request. If a volume ever had a sector size that does not divide 4096, ReadFile would fail
        // with ERROR_INVALID_PARAMETER and read_direct would throw -- a LOUD failure rather than a
        // short read or a corrupt one, which is the property this port is responsible for.
        direct_handle_ = ::CreateFileW(path.c_str(), GENERIC_READ,
                                       FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING,
                                       FILE_ATTRIBUTE_NORMAL | FILE_FLAG_NO_BUFFERING, nullptr);
        if (direct_handle_ == INVALID_HANDLE_VALUE) {
            const int error = static_cast<int>(::GetLastError());
            close_all();
            throw std::system_error(error, std::system_category(),
                                    "CreateFile(NO_BUFFERING) " + path.string());
        }

        size_ = size;
#else
        const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
        if (fd < 0) {
            throw std::system_error(errno, std::generic_category(), "open " + path.string());
        }

        struct stat status {};

        if (::fstat(fd, &status) != 0) {
            const int error = errno;
            ::close(fd);
            throw std::system_error(error, std::generic_category(), "fstat " + path.string());
        }
        if (status.st_size < 0 ||
            static_cast<std::uintmax_t>(status.st_size) > std::numeric_limits<std::size_t>::max()) {
            ::close(fd);
            throw ArtifactError("artifact size does not fit the process address space");
        }

        const auto size = static_cast<std::size_t>(status.st_size);
        void* mapping   = nullptr;
        if (size != 0) {
            mapping = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
            if (mapping == MAP_FAILED) {
                const int error = errno;
                ::close(fd);
                throw std::system_error(error, std::generic_category(), "mmap " + path.string());
            }
        }
        fd_   = fd;
        data_ = static_cast<const std::byte*>(mapping);
        size_ = size;
#endif
    }

    ~MappedFile() {
#if defined(_WIN32)
        close_all();
#else
        if (data_ != nullptr) { ::munmap(const_cast<std::byte*>(data_), size_); }
        if (fd_ >= 0) { ::close(fd_); }
#endif
    }

    MappedFile(const MappedFile&)            = delete;
    MappedFile& operator=(const MappedFile&) = delete;

    const std::byte* data() const noexcept { return data_; }

    std::size_t size() const noexcept { return size_; }

    std::size_t read_direct(std::uint64_t absolute_offset, std::span<std::byte> destination) const {
        constexpr std::size_t alignment = Reader::direct_io_alignment;
        if (absolute_offset % alignment != 0 || destination.size() % alignment != 0 ||
            reinterpret_cast<std::uintptr_t>(destination.data()) % alignment != 0) {
            throw ArtifactError("direct artifact read is not 4096-byte aligned");
        }
#if defined(_WIN32)
        // ReadFile's count is 32-bit and this arm's position is 64-bit, which is a narrower count and
        // a wider offset than the POSIX arm's ssize_t/off_t pair, so the platform limit this
        // function enforces is a different one and is spelled out here rather than reused.
        if (destination.size() > static_cast<std::size_t>(std::numeric_limits<DWORD>::max())) {
            throw ArtifactError("direct artifact read exceeds platform I/O limits");
        }

        // pread's load-bearing property is that the read starts at the offset the CALLER names,
        // independent of anything else that touched the handle. That is what this arm reproduces,
        // and it is MEASURED rather than assumed: the Windows probe in this line's record dir
        // (evidence/probe_nobuf.cpp, RUN_RC=0) read at OVERLAPPED.Offset=8192 and got the bytes at
        // 8192, with BYTES_MISMATCHING_VS_PATTERN=0 against a buffered read of the same range.
        //
        // WHAT IT DOES NOT REPRODUCE, and this was the probe's other finding: pread also leaves the
        // file POSITION untouched, and this arm does not. The same probe reported
        // FILE_POSITION_AFTER_ALIGNED_READ=12288, i.e. the pointer advanced to offset+length. That
        // is a real difference from pread, and it is harmless HERE for one reason that is worth
        // naming: nothing in this class ever reads the file pointer. The offset is passed per call,
        // so the returned bytes do not depend on the pointer's value; direct_handle_ exists only for
        // read_direct and is never used any other way. The concurrent case was NOT measured -- the
        // claim is that no read depends on the pointer, not that the pointer is race-free.
        //
        // The POSIX arm's do/while (bytes < 0 && errno == EINTR) retry has no counterpart here and
        // does not need one: a synchronous ReadFile is not interruptible by a signal, so there is no
        // EINTR to retry.
        OVERLAPPED overlapped{};
        overlapped.Offset     = static_cast<DWORD>(absolute_offset & 0xffffffffull);
        overlapped.OffsetHigh = static_cast<DWORD>(absolute_offset >> 32);
        DWORD transferred     = 0;
        if (::ReadFile(direct_handle_, destination.data(), static_cast<DWORD>(destination.size()),
                       &transferred, &overlapped) == 0) {
            throw std::system_error(static_cast<int>(::GetLastError()), std::system_category(),
                                    "direct artifact read");
        }
        return static_cast<std::size_t>(transferred);
#else
        if (absolute_offset > static_cast<std::uint64_t>(std::numeric_limits<off_t>::max()) ||
            destination.size() > static_cast<std::size_t>(std::numeric_limits<ssize_t>::max())) {
            throw ArtifactError("direct artifact read exceeds platform I/O limits");
        }

        ssize_t bytes = -1;
        do {
            bytes = ::pread(fd_, destination.data(), destination.size(),
                            static_cast<off_t>(absolute_offset));
        } while (bytes < 0 && errno == EINTR);
        if (bytes < 0) {
            throw std::system_error(errno, std::generic_category(), "direct artifact read");
        }
        return static_cast<std::size_t>(bytes);
#endif
    }

private:
#if defined(_WIN32)
    // The one place the two arms of this class stop looking alike. All four of these are closed by
    // close_all(), which the destructor and every constructor error path calls, so there is exactly
    // one teardown order to audit.
    void close_all() noexcept {
        // Unmap BEFORE closing the section object: a view is only meaningful while its section
        // exists, which is the ordering the POSIX arm encodes by munmap-ing before close().
        if (data_ != nullptr) {
            ::UnmapViewOfFile(const_cast<std::byte*>(data_));
            data_ = nullptr;
        }
        if (mapping_handle_ != nullptr) {
            ::CloseHandle(mapping_handle_);
            mapping_handle_ = nullptr;
        }
        if (direct_handle_ != INVALID_HANDLE_VALUE) {
            ::CloseHandle(direct_handle_);
            direct_handle_ = INVALID_HANDLE_VALUE;
        }
        if (file_handle_ != INVALID_HANDLE_VALUE) {
            ::CloseHandle(file_handle_);
            file_handle_ = INVALID_HANDLE_VALUE;
        }
    }

    HANDLE file_handle_    = INVALID_HANDLE_VALUE; // ordinary: owns the section object
    HANDLE mapping_handle_ = nullptr;              // the section object itself
    HANDLE direct_handle_  = INVALID_HANDLE_VALUE; // FILE_FLAG_NO_BUFFERING: owns read_direct
#else
    int fd_ = -1;
#endif
    const std::byte* data_ = nullptr;
    std::size_t size_      = 0;
};

} // namespace

std::string_view object_name(const ObjectDescriptor& object) noexcept {
    return std::visit([](const auto& descriptor) -> std::string_view { return descriptor.name; },
                      object);
}

std::uint64_t object_offset(const ObjectDescriptor& object) noexcept {
    return std::visit([](const auto& descriptor) { return descriptor.offset; }, object);
}

std::uint64_t object_bytes(const ObjectDescriptor& object) noexcept {
    return std::visit([](const auto& descriptor) { return descriptor.bytes; }, object);
}

struct Reader::Impl {
    explicit Impl(const std::filesystem::path& path) : file(path) {
        if (file.size() < kPrefixBytes) {
            throw ArtifactError("artifact is shorter than the v2 prefix");
        }
        if (!std::equal(kMagic.begin(), kMagic.end(), file.data())) {
            throw ArtifactError("artifact magic is not NInfer v2");
        }

        const auto json_bytes = read_u64_le(file.data() + 8);
        if (json_bytes == 0) { throw ArtifactError("json_bytes must be positive"); }
        const auto metadata_end = checked_add(kPrefixBytes, json_bytes, "JSON range");
        payload_start           = align_up(metadata_end, kPayloadAlignment, "payload offset");
        if (metadata_end > file.size() || payload_start > file.size()) {
            throw ArtifactError("declared JSON or payload start extends beyond the file");
        }

        Json directory;
        try {
            const auto* begin = reinterpret_cast<const char*>(file.data() + kPrefixBytes);
            directory         = Json::parse(begin, begin + json_bytes);
        } catch (const Json::exception& error) {
            throw ArtifactError(std::string("invalid JSON directory: ") + error.what());
        }

        static constexpr std::array root_members = {"identity", "objects"};
        require_members(directory, root_members, "directory root");
        const auto& raw_identity                     = directory.at("identity");
        static constexpr std::array identity_members = {"model_id", "weights_id"};
        require_members(raw_identity, identity_members, "artifact identity");
        identity.model_id   = require_string(raw_identity.at("model_id"), "model_id");
        identity.weights_id = require_string(raw_identity.at("weights_id"), "weights_id");

        const auto& raw_objects = directory.at("objects");
        if (!raw_objects.is_array() || raw_objects.empty()) {
            throw ArtifactError("objects must be a nonempty array");
        }
        entries.reserve(raw_objects.size());
        index.reserve(raw_objects.size());

        const auto payload_bytes = static_cast<std::uint64_t>(file.size()) - payload_start;
        std::uint64_t cursor     = 0;
        for (const auto& raw_object : raw_objects) {
            auto object          = parse_object(raw_object);
            const auto name      = object_name(object);
            const auto offset    = object_offset(object);
            const auto bytes     = object_bytes(object);
            const auto alignment = std::visit(
                [](const auto& descriptor) {
                    using Descriptor = std::decay_t<decltype(descriptor)>;
                    if constexpr (std::is_same_v<Descriptor, TensorDescriptor>) {
                        return tensor_alignment(descriptor.layout);
                    } else {
                        return resource_alignment(descriptor.encoding);
                    }
                },
                object);

            if (offset < cursor) {
                throw ArtifactError("object " + std::string(name) + " overlaps or is out of order");
            }
            if (offset % alignment != 0) {
                throw ArtifactError("object " + std::string(name) + " is not " +
                                    std::to_string(alignment) + "-byte aligned");
            }
            const auto end = checked_add(offset, bytes, "object payload range");
            if (end > payload_bytes) {
                throw ArtifactError("object " + std::string(name) + " extends beyond the file");
            }
            const auto object_index = entries.size();
            auto [_, inserted]      = index.emplace(std::string(name), object_index);
            if (!inserted) { throw ArtifactError("duplicate object name: " + std::string(name)); }
            entries.push_back(std::move(object));
            cursor = end;
        }
    }

    MappedFile file;
    ArtifactIdentity identity;
    std::vector<ObjectDescriptor> entries;
    std::unordered_map<std::string, std::size_t, TransparentStringHash, std::equal_to<>> index;
    std::uint64_t payload_start = 0;
};

Reader::Reader(const std::filesystem::path& path) : impl_(std::make_shared<Impl>(path)) {}

Reader::~Reader()                            = default;
Reader::Reader(Reader&&) noexcept            = default;
Reader& Reader::operator=(Reader&&) noexcept = default;

std::shared_ptr<const void> Reader::mapping_lease() const {
    // The lease IS a share of the Impl, which owns the MappedFile. Making it a
    // `shared_ptr<const void>` keeps the two properties the callers need and nothing else: the
    // mapping stays alive for as long as any lease does, and no caller can reach through it to
    // mutate a Reader. See the declaration in reader.h for the defect this closes.
    return std::static_pointer_cast<const void>(impl_);
}

const ArtifactIdentity& Reader::identity() const noexcept { return impl_->identity; }

const std::vector<ObjectDescriptor>& Reader::objects() const noexcept { return impl_->entries; }

const ObjectDescriptor* Reader::find(std::string_view name) const noexcept {
    const auto it = impl_->index.find(name);
    return it == impl_->index.end() ? nullptr : &impl_->entries[it->second];
}

std::uint64_t Reader::file_bytes() const noexcept { return impl_->file.size(); }

std::uint64_t Reader::payload_offset() const noexcept { return impl_->payload_start; }

PayloadSpan Reader::payload(const ObjectDescriptor& object) const {
    const auto absolute =
        checked_add(impl_->payload_start, object_offset(object), "absolute payload offset");
    const auto end = checked_add(absolute, object_bytes(object), "absolute payload range");
    if (end > impl_->file.size()) { throw ArtifactError("object payload extends beyond the file"); }
    return {
        absolute,
        std::span<const std::byte>(impl_->file.data() + absolute,
                                   static_cast<std::size_t>(object_bytes(object))),
    };
}

PayloadSpan Reader::payload(std::string_view name) const {
    const auto* object = find(name);
    if (object == nullptr) { throw ArtifactError("unknown artifact object: " + std::string(name)); }
    return payload(*object);
}

std::size_t Reader::read_direct(std::uint64_t absolute_offset,
                                std::span<std::byte> destination) const {
    return impl_->file.read_direct(absolute_offset, destination);
}

} // namespace ninfer::artifact
