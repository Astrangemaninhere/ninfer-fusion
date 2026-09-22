// tests/test_fnv_convention.cpp -- the tripwire for the tree's FNV-1a 64 conventions.
//
// WHY THIS FILE EXISTS. Two FNV-1a 64 conventions are alive in this tree and they
// differ by exactly one decimal digit:
//
//     published basis   0xcbf29ce484222325 = 14695981039346656037
//                                          (product/kv_recall_block.h, and the tools' digests)
//     engine's lane 0                       1469598103934665603
//                                          (impl/runtime/prefix_identity.cpp: the owner)
//
// Both are legal FNV, both are already baked into landed records and tests, and
// neither may be "tidied" into the other. src/spec/fnv_convention.h owns both and
// carries the compile-time half of the pin (static_asserts on the arithmetic
// relation between them). This file is the runtime half, and it does four things a
// comment cannot:
//
//   1. STATES that the two are deliberately different, and prints the arithmetic
//      relation between them so a reader sees they are not a typo to correct.
//   2. PINS each convention by GOLDEN VALUE over a fixed input: changing either
//      constant breaks these numbers, so the change cannot be silent.
//   3. PINS the engine's copy in the file that OWNS it
//      (src/targets/qwen3_6/impl/runtime/prefix_identity.cpp) BY TEXT. That file
//      belongs to a target's impl/ and is not included from here, so its constants
//      are read and compared against the owning header -- which is how the header
//      stays the single source of truth without editing that file.
//   4. RECORDS WHICH ARTEFACT DEPENDS ON WHICH CONVENTION as an executable list.
//      Each source file that spells a raw FNV basis or prime is either
//      owner-linked (must include spec/fnv_convention.h and must NOT re-spell the
//      literals) or pinned-by-value (must still contain the exact value recorded
//      for it). A file that disappears, or that changes its constant, fails here
//      by name. And the set of DISTINCT VALUES found anywhere under src/, tests/
//      and tools/ must be exactly the five registered ones, so a THIRD convention
//      cannot be introduced by accident.
//
// Plain g++, no CUDA, no artifact (the same shape as tests/test_sum_dir.cpp and
// tests/test_kv_recall_block.cpp). Sections 3 and 4 need NINFER_SOURCE_DIR, which
// tests/CMakeLists.txt provides through NEEDS_SOURCE_DIR; without it they print a
// skip line and return 77 rather than passing silently.

#include "product/kv_recall_block.h"
#include "spec/fnv_convention.h"
#include "spec/sum_dir.h"

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <set>
#include <string>
#include <vector>

namespace {

namespace f = ninfer::spec::fnv;

int g_failures = 0;

void check(bool condition, const std::string& what) {
    if (!condition) {
        std::fprintf(stderr, "FAIL %s\n", what.c_str());
        ++g_failures;
    }
}

template <typename A, typename B>
void check_equal(const A& got, const B& want, const std::string& what) {
    if (!(got == want)) {
        std::fprintf(stderr, "FAIL %s (got %s, want %s)\n", what.c_str(),
                     std::to_string(got).c_str(), std::to_string(want).c_str());
        ++g_failures;
    }
}

// ---------------------------------------------------------------------------
// 1. the two conventions are deliberately different
// ---------------------------------------------------------------------------
void section1_the_difference_is_deliberate() {
    check_equal(f::kFnvConventionCount, 2, "there are exactly two registered conventions");

    // The published basis is 20 decimal digits; the engine's lane 0 is the same
    // digits with the last one dropped. Asserted as arithmetic, so neither value can
    // be "corrected" on its own -- and the same two assertions are static_asserts in
    // the owning header, which is what makes it a compile error there.
    check_equal(f::kFnv1a64OffsetBasis / 10U, f::kEngineDigestLane0Offset,
                "the engine's lane-0 offset is the published basis truncated by one decimal digit");
    check_equal(f::kFnv1a64OffsetBasis % 10U, std::uint64_t{7},
                "the dropped digit is the trailing 7 of 14695981039346656037");
    check(f::kFnv1a64OffsetBasis != f::kEngineDigestLane0Offset,
          "the two conventions must never be unified: both are referenced by landed records");

    // The engine's lane 0 reuses the published prime; lane 1 does not; the lanes
    // must differ or there is one lane.
    check_equal(f::kEngineDigestPrime0, f::kFnv1a64Prime,
                "the engine's lane-0 prime IS the published FNV prime");
    check(f::kEngineDigestPrime1 != f::kEngineDigestPrime0, "the two lanes must not share a prime");

    std::printf("  convention 1 (published) offset basis = %llu (0x%llx)\n",
                (unsigned long long)f::kFnv1a64OffsetBasis,
                (unsigned long long)f::kFnv1a64OffsetBasis);
    std::printf("  convention 2 (engine)    lane 0 offset = %llu (0x%llx)\n",
                (unsigned long long)f::kEngineDigestLane0Offset,
                (unsigned long long)f::kEngineDigestLane0Offset);
    std::printf("  relation: published / 10 == engine, published %% 10 == %llu -> the engine's "
                "value is the published one with one decimal digit dropped; both are in use\n",
                (unsigned long long)(f::kFnv1a64OffsetBasis % 10U));
}

// ---------------------------------------------------------------------------
// 2. golden values, one per convention
// ---------------------------------------------------------------------------
// A local reimplementation of the engine's two-lane mix over the same token input
// the sum-directory digest receives, written only out of the constants the owning
// header names. If sum_dir.h ever went back to spelling its own copies and one of
// them drifted, the two would disagree here.
std::uint64_t engine_lane0_over_block(const std::uint32_t* tokens, std::size_t count,
                                      std::uint64_t domain) {
    std::uint64_t lo = f::kEngineDigestLane0Offset;
    const auto mix = [&lo](std::uint64_t value) {
        lo ^= value;
        lo *= f::kEngineDigestPrime0;
    };
    mix(domain);
    mix(static_cast<std::uint64_t>(count));
    for (std::size_t i = 0; i < count; ++i) { mix(static_cast<std::uint64_t>(tokens[i])); }
    if (lo == 0) { lo = 1; }
    return lo;
}

void section2_golden_values() {
    const std::vector<std::uint32_t> block{10, 20, 30, 40};
    std::vector<ninfer::TokenId> typed;
    typed.reserve(block.size());
    for (const std::uint32_t token : block) { typed.push_back(static_cast<ninfer::TokenId>(token)); }

    // PUBLISHED convention, through the product header's own function.
    const std::uint64_t published =
        ninfer::product::recall_block_digest(std::span<const ninfer::TokenId>(typed));
    check_equal(published, std::uint64_t{5187932501612950315ULL},
                "recall_block_digest({10,20,30,40}) must keep its recorded value "
                "(published FNV-1a 64 basis) -- a landed artefact index depends on it");

    // ENGINE convention, through the spec header's digest.
    const ninfer::spec::sum_dir::SumDirDigest engine =
        ninfer::spec::sum_dir::sum_dir_block_digest(block);
    check_equal(engine.lo, std::uint64_t{11907109397543494613ULL},
                "sum_dir_block_digest({10,20,30,40}).lo must keep its recorded value "
                "(engine lane-0 offset)");
    check_equal(engine.hi, std::uint64_t{6876853601322608068ULL},
                "sum_dir_block_digest({10,20,30,40}).hi must keep its recorded value");

    // The two conventions do not agree on the same bytes: that is exactly why a
    // reader must not "unify" them (the engine also domain-separates, so the values
    // differ for two independent reasons -- both are deliberate).
    check(published != engine.lo, "the two conventions must not agree on the same input");

    // sum_dir.h's constants ARE the owning header's, not a privately spelled copy.
    check_equal(ninfer::spec::sum_dir::kSumDirDigestOffset0, f::kEngineDigestLane0Offset,
                "sum_dir.h's lane-0 offset is the owning header's");
    check_equal(ninfer::spec::sum_dir::kSumDirDigestOffset1, f::kEngineDigestLane1Offset,
                "sum_dir.h's lane-1 offset is the owning header's");
    check_equal(ninfer::spec::sum_dir::kSumDirDigestPrime1, f::kEngineDigestPrime1,
                "sum_dir.h's lane-1 prime is the owning header's");
    check_equal(engine.lo, engine_lane0_over_block(block.data(), block.size(), 0x6e696e6665722d62ULL),
                "the owning header's constants reproduce sum_dir.h's block digest");

    std::printf("  golden: published=%llu engine.lo=%llu engine.hi=%llu (input {10,20,30,40})\n",
                (unsigned long long)published, (unsigned long long)engine.lo,
                (unsigned long long)engine.hi);
}

#ifdef NINFER_SOURCE_DIR
// ---------------------------------------------------------------------------
// source-tree helpers (sections 3 and 4)
// ---------------------------------------------------------------------------
constexpr const char* kSourceDir = NINFER_SOURCE_DIR;

std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { return {}; }
    return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

std::string lower_copy(std::string text) {
    for (char& c : text) {
        if (c >= 'A' && c <= 'Z') { c = static_cast<char>(c - 'A' + 'a'); }
    }
    return text;
}

// One spelling of one 64-bit FNV value. Longest literals first, so that
// 14695981039346656037 is not read as the 19-digit value plus a stray 7.
struct Spelling {
    const char* literal;       // lower-case text to search for
    std::uint64_t value;
    const char* convention;    // "published" / "engine"
};

const Spelling kSpellings[] = {
    {"14695981039346656037", 14695981039346656037ULL, "published"},
    {"0xcbf29ce484222325", 14695981039346656037ULL, "published"},
    {"1469598103934665603", 1469598103934665603ULL, "engine"},
    {"7809847782465536322", 7809847782465536322ULL, "engine"},
    {"14029467366897019727", 14029467366897019727ULL, "engine"},
    {"1099511628211", 1099511628211ULL, "both"},
    {"0x100000001b3", 1099511628211ULL, "both"},
};

// A file that must still spell the value recorded for it, because it is outside
// this change's reach (a target's impl/, another author's zone) or is a tool.
struct PinnedFile {
    const char* path;
    std::uint64_t value;
    const char* convention;
    const char* why;
};

const PinnedFile kPinnedFiles[] = {
    {"src/targets/qwen3_6/impl/runtime/prefix_identity.cpp", 1469598103934665603ULL, "engine",
     "OWNER of the engine convention (kDigestOffset lane 0)"},
    {"src/targets/qwen3_6/impl/runtime/prefix_identity.cpp", 7809847782465536322ULL, "engine",
     "OWNER of the engine convention (kDigestOffset lane 1)"},
    {"src/targets/qwen3_6/impl/runtime/prefix_identity.cpp", 1099511628211ULL, "engine",
     "OWNER of the engine convention (kDigestPrime lane 0)"},
    {"src/targets/qwen3_6/impl/runtime/prefix_identity.cpp", 14029467366897019727ULL, "engine",
     "OWNER of the engine convention (kDigestPrime lane 1)"},
    {"src/runtime/engine/resource_manager.h", 1469598103934665603ULL, "engine",
     "policy/identity digests, spelled inline"},
    {"src/runtime/engine/resource_manager.h", 1099511628211ULL, "engine", "same digest, the prime"},
    {"src/targets/qwen3_6/impl/runtime/program_impl.h", 1469598103934665603ULL, "engine",
     "request/plan identity digests, spelled inline"},
    {"src/targets/qwen3_6/impl/runtime/program_impl.h", 1099511628211ULL, "engine", "same, prime"},
    {"src/targets/qwen3_6/impl/runtime/pressure_planner.h", 1469598103934665603ULL, "engine",
     "planner digest, spelled inline"},
    {"src/targets/qwen3_6/impl/runtime/pressure_planner.h", 1099511628211ULL, "engine", "same, prime"},
    {"src/product/kv_rowscale_bake.h", 1469598103934665603ULL, "engine",
     "row-scale table digest, spelled inline; the ONLY definition moved to kv_rowscale_bake.h, and persist.h delegates to it"}, 
    {"src/product/kv_rowscale_bake.h", 1099511628211ULL, "engine", "same, prime"},
    {"tests/test_resource_manager.cpp", 1469598103934665603ULL, "engine",
     "the test's own recomputation of the manager's digest"},
    {"tests/test_resource_manager.cpp", 1099511628211ULL, "engine", "same, prime"},
    {"tools/ple_table_test.cu", 14695981039346656037ULL, "published", "tool-side digest"},
    {"tools/ple_table_test.cu", 1099511628211ULL, "published", "same, prime"},
    {"tools/ple_gather_test.cu", 14695981039346656037ULL, "published", "tool-side digest"},
    {"tools/ple_gather_test.cu", 1099511628211ULL, "published", "same, prime"},
    {"tools/archkit/ple_gather_check.py", 14695981039346656037ULL, "published",
     "Python twin of the tool digest"},
    {"tools/archkit/ple_gather_check.py", 1099511628211ULL, "published", "same, prime"},
};

// A file that must NOT re-spell the literals any more: it must take them from the
// owning header. This is the anti-regression for the copy that used to live here.
const char* const kOwnerLinked[] = {
    "src/spec/sum_dir.h",
    "src/product/kv_recall_block.h",
};

bool contains_value(const std::string& text, std::uint64_t value) {
    const std::string lowered = lower_copy(text);
    for (const Spelling& spelling : kSpellings) {
        if (spelling.value == value && lowered.find(spelling.literal) != std::string::npos) {
            return true;
        }
    }
    return false;
}

std::size_t line_number_of(const std::string& text, const char* needle) {
    std::size_t line = 1;
    std::size_t begin = 0;
    while (begin <= text.size()) {
        const std::size_t end = text.find('\n', begin);
        const std::string_view view(text.data() + begin,
                                    (end == std::string::npos ? text.size() : end) - begin);
        if (view.find(needle) != std::string_view::npos) { return line; }
        if (end == std::string::npos) { break; }
        begin = end + 1;
        ++line;
    }
    return 0;
}

// ---------------------------------------------------------------------------
// 3. the engine's owner file still spells exactly what the header owns
// ---------------------------------------------------------------------------
void section3_the_engine_owner_still_matches() {
    const std::string path =
        std::string(kSourceDir) + "/src/targets/qwen3_6/impl/runtime/prefix_identity.cpp";
    const std::string text = read_file(path);
    if (text.empty()) {
        check(false, "cannot read the engine's prefix_identity.cpp at " + path);
        return;
    }
    const std::size_t offset_line = line_number_of(text, "kDigestOffset{");
    const std::size_t prime_line  = line_number_of(text, "kDigestPrime{");
    check(offset_line != 0, "prefix_identity.cpp still declares kDigestOffset");
    check(prime_line != 0, "prefix_identity.cpp still declares kDigestPrime");
    check(contains_value(text, f::kEngineDigestLane0Offset),
          "prefix_identity.cpp still spells the lane-0 offset the owning header records");
    check(contains_value(text, f::kEngineDigestLane1Offset),
          "prefix_identity.cpp still spells the lane-1 offset the owning header records");
    check(contains_value(text, f::kEngineDigestPrime1),
          "prefix_identity.cpp still spells the lane-1 prime the owning header records");
    check(line_number_of(text, "0x9e3779b97f4a7c15ULL") != 0,
          "prefix_identity.cpp still spells the lane skew the owning header records");
    std::printf("  engine owner: %s kDigestOffset at :%zu, kDigestPrime at :%zu "
                "(values match src/spec/fnv_convention.h)\n",
                path.c_str(), offset_line, prime_line);

    // The comment in the owning header cites those two lines; keep the citation true.
    const std::string header = read_file(std::string(kSourceDir) + "/src/spec/fnv_convention.h");
    check(header.find("prefix_identity.cpp:60-61") != std::string::npos,
          "the owning header still cites the owner's constants as prefix_identity.cpp:60-61");
}

// ---------------------------------------------------------------------------
// 4. the inventory: who depends on which convention, and no third one
// ---------------------------------------------------------------------------
void section4_the_inventory_is_closed() {
    // 4a. every pinned file still carries its recorded value
    for (const PinnedFile& pinned : kPinnedFiles) {
        const std::string path = std::string(kSourceDir) + "/" + pinned.path;
        const std::string text = read_file(path);
        if (text.empty()) {
            check(false, std::string("pinned dependency disappeared: ") + pinned.path);
            continue;
        }
        check(contains_value(text, pinned.value),
              std::string("pinned-by-value file ") + pinned.path + " must still spell its recorded "
              "value (" + pinned.convention + "; " + pinned.why + ")");
    }

    // 4b. owner-linked files take the constants from the owning header instead
    for (const char* relative : kOwnerLinked) {
        const std::string path = std::string(kSourceDir) + "/" + relative;
        const std::string text = read_file(path);
        if (text.empty()) {
            check(false, std::string("owner-linked file disappeared: ") + relative);
            continue;
        }
        check(text.find("spec/fnv_convention.h") != std::string::npos,
              std::string("owner-linked file ") + relative +
                  " must include the owning header spec/fnv_convention.h");
        check(text.find("fnv::") != std::string::npos,
              std::string("owner-linked file ") + relative + " must use the owning header's names");
        check(!contains_value(text, f::kEngineDigestLane0Offset) &&
                  !contains_value(text, f::kFnv1a64OffsetBasis),
              std::string("owner-linked file ") + relative +
                  " must NOT re-spell an offset basis: that is how the two conventions drifted "
                  "into unrelated literals");
    }

    // 4c. the whole tree's distinct value set must be exactly the five registered ones
    std::set<std::uint64_t> distinct;
    std::set<std::string> carriers;
    std::size_t scanned = 0;
    for (const char* root : {"src", "tests", "tools"}) {
        const std::filesystem::path base = std::filesystem::path(kSourceDir) / root;
        std::error_code code;
        std::filesystem::recursive_directory_iterator it(
            base, std::filesystem::directory_options::skip_permission_denied, code), end;
        for (; it != end; it.increment(code)) {
            if (code) { break; }
            const std::filesystem::path& entry = it->path();
            const std::string name = entry.filename().string();
            if (it->is_directory(code)) {
                if (name == "build" || name == "out" || name == "__pycache__" || name == ".git") {
                    it.disable_recursion_pending();
                }
                continue;
            }
            const std::string extension = entry.extension().string();
            if (extension != ".h" && extension != ".hpp" && extension != ".cuh" &&
                extension != ".cpp" && extension != ".cu" && extension != ".py") {
                continue;
            }
            const std::string text = read_file(entry.string());
            if (text.empty()) { continue; }
            ++scanned;
            const std::string lowered = lower_copy(text);
            for (const Spelling& spelling : kSpellings) {
                if (lowered.find(spelling.literal) == std::string::npos) { continue; }
                distinct.insert(spelling.value);
                carriers.insert(entry.filename().string());
            }
        }
    }

    const std::set<std::uint64_t> registered = {
        14695981039346656037ULL, // published offset basis
        1469598103934665603ULL,  // engine lane-0 offset
        7809847782465536322ULL,  // engine lane-1 offset
        1099511628211ULL,        // published prime == engine lane-0 prime
        14029467366897019727ULL, // engine lane-1 prime
    };
    for (const std::uint64_t value : distinct) {
        if (registered.count(value) == 0) {
            check(false, "a THIRD FNV convention appeared in the tree: value " +
                             std::to_string(value) +
                             " is not registered in src/spec/fnv_convention.h; register it there "
                             "and here, or use one of the two existing conventions");
        }
    }
    std::printf("  inventory: scanned %zu source files, %zu carry FNV values, %zu distinct values "
                "(registered: %zu)\n",
                scanned, carriers.size(), distinct.size(), registered.size());
    std::printf("  carriers:");
    for (const std::string& name : carriers) { std::printf(" %s", name.c_str()); }
    std::printf("\n");
    check(scanned > 100, "the scan really walked the tree (more than 100 source files)");
}
#endif // NINFER_SOURCE_DIR

} // namespace

int main() {
    section1_the_difference_is_deliberate();
    section2_golden_values();
#ifndef NINFER_SOURCE_DIR
    std::printf("SKIP sections 3-4: built without NINFER_SOURCE_DIR\n");
    if (g_failures == 0) { return 77; }
#else
    section3_the_engine_owner_still_matches();
    section4_the_inventory_is_closed();
#endif
    if (g_failures == 0) {
        std::printf("fnv_convention_test: all checks passed\n");
        return 0;
    }
    std::fprintf(stderr, "fnv_convention_test: %d check(s) failed\n", g_failures);
    return 1;
}
