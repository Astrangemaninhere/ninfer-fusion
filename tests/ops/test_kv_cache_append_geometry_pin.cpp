// INTEGRATION PIN for fork-survey island #2: src/ops/kv_cache/append/geometry.cuh
//
// The additive merge landed four borrows that include "ops/kv_cache/append/geometry.cuh"
// (kv_cache/append/{k8v4,nvfp4,rk2v4e8,hq}_kernel.cuh) plus two more
// (causal_cache/{prompt_hq,small_t_hq}.cuh) -- six files across two islands -- but it did not
// land the header itself. MEASURED RED, under the project's own device oracle
// (nvcc -O3 -DNDEBUG -std=c++20 -c -rdc=true -arch compute_120a): the FIRST error of all six
// consumers was
//     fatal error: ops/kv_cache/append/geometry.cuh: No such file or directory
// at k8v4_kernel.cuh:19 / nvfp4_kernel.cuh:15 / rk2v4e8_kernel.cuh:18 / hq_kernel.cuh:22 /
// prompt_hq.cuh:25 / small_t_hq.cuh:25. Six files, one root cause.
//
// Why this pin exists, and why it is this shape.
//
// The tree ALREADY carries these four declarations -- inlined, at src/ops/kv_cache/append/
// kernel.cuh:17-26, a tracked file present at HEAD. The fork's kernel.cuh instead #includes
// geometry.cuh (fork kernel.cuh:9) and keeps the declarations there. So the borrow is not a
// new mechanism: it is the fork's own extraction of a block this tree kept inlined, and the
// whole risk of landing it is that the tree ends up with TWO definitions of one vocabulary.
// This pin makes that impossible to be true silently. It asserts a three-way byte agreement:
//
//     the EXPECTED literal block  ==  the landed geometry.cuh's block  ==  kernel.cuh:17-26
//
// ⚠ RE-DERIVED (line `redkv`, dl/redkv/TRIAGE.md #169). The third leg of that agreement HAS moved,
// and it moved for the reason the pin exists. kernel.cuh:17-26 no longer SPELLS the block; it
// INCLUDES geometry.cuh and says why in its own words (kernel.cuh:18-25: the two copies "made this
// header and geometry.cuh mutually un-includable: nvcc 13 answered `variable ... has already been
// defined` on any TU that reached both -- which the narrow e8 append arm does"). So the landed
// state IS the state this pin's own prose asks for -- ONE definition -- and the pin could not say
// so, because it required kernel.cuh to keep the second copy. It exited rc=2 on that tree. The
// agreement is now asserted as what it always meant: geometry.cuh IS the pinned literal block,
// kernel.cuh reaches it by include and does NOT redefine it, and therefore the pair holds exactly
// ONE definition of these four names.
//
// Because the comparison is over the exact declaration text, this is a stronger statement
// than a numeric tolerance: it says the borrow reproduces the tree's shipped block to the
// byte, so "the same vocabulary, finally given a home" is measured rather than asserted.
//
// It ALSO ties the landed header to a LIVE consumer by name: src/ops/kv_cache/append/launch.cu
// -- tracked, at HEAD, and compiled into ninfer_ops -- selects the two aliases by name
// (KVCacheAppendD256Kv4 at :169/:187, KVCacheAppendD256Kv2 at :173/:191). Those two names are
// declarations the landed header must supply, so the pin fails if the header drifts away from
// what the engine's own append dispatcher asks for by name.
//
// Host-only: no CUDA runtime call, no device, no model, no lock, no subprocess, no argv.
#include "ops/kv_cache/append/geometry.cuh"

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#ifndef NINFER_SOURCE_DIR
#error "this test needs NINFER_SOURCE_DIR"
#endif

// ---------------------------------------------------------------------------
// 1. COMPILE-TIME: the four facts the six consumers use, asserted at the
//    project's own flags rather than at a debugger-friendly -O0.
// ---------------------------------------------------------------------------
static_assert(ninfer::ops::kKVCacheAppendFullHeadDim == 256,
              "the append kernels are D256; the borrow must say so");
static_assert(ninfer::ops::KVCacheAppendD256Kv4::KVHeads == 4,
              "D256Kv4 must carry 4 KV heads -- launch.cu selects it by this name");
static_assert(ninfer::ops::KVCacheAppendD256Kv2::KVHeads == 2,
              "D256Kv2 must carry 2 KV heads -- launch.cu selects it by this name");
static_assert(ninfer::ops::KVCacheAppendFullGeometry<4>::KVHeads !=
                  ninfer::ops::KVCacheAppendFullGeometry<2>::KVHeads,
              "the two live geometries must stay distinguishable");

namespace {

std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::cerr << "cannot open " << path << '\n';
        std::exit(2);
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

// The declaration block, located by its first identifier and ending at the last alias, so
// the pin does not break when unrelated lines move around it.
std::string declaration_block(const std::string& text, const char* what) {
    const std::string begin = "inline constexpr int kKVCacheAppendFullHeadDim";
    const std::string end   = "using KVCacheAppendD256Kv2 = KVCacheAppendFullGeometry<2>;\n";
    const std::size_t a = text.find(begin);
    if (a == std::string::npos) {
        std::cerr << what << ": declaration start not found\n";
        std::exit(2);
    }
    const std::size_t b = text.find(end, a);
    if (b == std::string::npos) {
        std::cerr << what << ": declaration end not found\n";
        std::exit(2);
    }
    return text.substr(a, b + end.size() - a);
}

std::size_t count_of(const std::string& text, const std::string& needle) {
    std::size_t n = 0;
    for (std::size_t at = text.find(needle); at != std::string::npos;
         at = text.find(needle, at + 1)) {
        ++n;
    }
    return n;
}

} // namespace

int main() {
    const std::string root = NINFER_SOURCE_DIR;
    const std::string landed =
        read_file(root + "/src/ops/kv_cache/append/geometry.cuh");
    const std::string kernel_cuh =
        read_file(root + "/src/ops/kv_cache/append/kernel.cuh");
    const std::string launch_cu =
        read_file(root + "/src/ops/kv_cache/append/launch.cu");

    // The tree's shipped block, verbatim. Hard-coded so the pin states the contract in the
    // test itself rather than deriving it from either file it is checking.
    const std::string expected =
        "inline constexpr int kKVCacheAppendFullHeadDim = 256;\n"
        "\n"
        "template <int KVHeadsValue>\n"
        "struct KVCacheAppendFullGeometry {\n"
        "    static_assert(KVHeadsValue == 4 || KVHeadsValue == 2);\n"
        "    static constexpr int KVHeads = KVHeadsValue;\n"
        "};\n"
        "\n"
        "using KVCacheAppendD256Kv4 = KVCacheAppendFullGeometry<4>;\n"
        "using KVCacheAppendD256Kv2 = KVCacheAppendFullGeometry<2>;\n";

    int failures = 0;

    const std::string got_landed = declaration_block(landed, "geometry.cuh");
    const std::string what = "inline constexpr int kKVCacheAppendFullHeadDim";
    const std::string alias = "using KVCacheAppendD256Kv2 = KVCacheAppendFullGeometry<2>;";
    const std::string include_line = "#include \"ops/kv_cache/append/geometry.cuh\"";

    // 2. The landed borrow must BE the tree's block, byte for byte.
    if (got_landed != expected) {
        std::cerr << "the landed geometry.cuh block is not the expected literal block\n"
                  << "--- expected ---\n" << expected << "--- got ---\n" << got_landed;
        ++failures;
    }
    // 3. ⭐ THE ONE-DEFINITION DIRECTION (the pin's own stated risk, now measured in the shape the
    //    tree actually took). kernel.cuh must NOT spell the block a second time -- that is the
    //    state that made the two headers mutually un-includable -- and it must reach it by
    //    INCLUDE, because a header that neither defines nor includes it would leave the six
    //    consumers of geometry.cuh and the four users of this header with two different worlds.
    const std::size_t kernel_defs = count_of(kernel_cuh, what);
    if (kernel_defs != 0) {
        std::cerr << "kernel.cuh defines '" << what << "' " << kernel_defs
                  << " time(s); it must reach the ONE definition in geometry.cuh instead\n";
        ++failures;
    }
    if (kernel_cuh.find(include_line) == std::string::npos) {
        std::cerr << "kernel.cuh does not include ops/kv_cache/append/geometry.cuh, and it no "
                     "longer defines the block itself: the four names have no reachable home\n";
        ++failures;
    }
    // ... and across the PAIR there is exactly one definition, which is the whole claim.
    const std::size_t pair_defs = count_of(landed, what) + count_of(kernel_cuh, what);
    if (pair_defs != 1) {
        std::cerr << "the pair (geometry.cuh, kernel.cuh) holds " << pair_defs
                  << " definition(s) of '" << what << "', want exactly 1\n";
        ++failures;
    }
    if (count_of(landed, alias) != 1) {
        std::cerr << "geometry.cuh does not hold the Kv2 alias exactly once\n";
        ++failures;
    }

    // 4. The LIVE dispatcher must still ask for the two aliases by name, or this header is
    //    supplying a vocabulary nothing selects.
    for (const char* symbol : {"KVCacheAppendD256Kv4", "KVCacheAppendD256Kv2"}) {
        if (launch_cu.find(symbol) == std::string::npos) {
            std::cerr << "live consumer launch.cu no longer names " << symbol << '\n';
            ++failures;
        }
    }
    // ... and the SIX consumers the pin's header names must reach the landed header, or the
    // "same vocabulary, finally given a home" claim is about a header nobody includes.
    for (const char* consumer : {
             "src/ops/kv_cache/append/k8v4_kernel.cuh",
             "src/ops/kv_cache/append/nvfp4_kernel.cuh",
             "src/ops/kv_cache/append/rk2v4e8_kernel.cuh",
             "src/ops/kv_cache/append/hq_kernel.cuh",
             "src/ops/softmax_attention/dense/causal_cache/prompt_hq.cuh",
             "src/ops/softmax_attention/dense/causal_cache/small_t_hq.cuh",
         }) {
        const std::string body = read_file(root + "/" + consumer);
        if (body.find("ops/kv_cache/append/geometry.cuh") == std::string::npos) {
            std::cerr << "consumer " << consumer << " no longer includes geometry.cuh\n";
            ++failures;
        }
    }

    std::cout << (failures == 0 ? "PASS" : "FAIL")
              << " kv_cache/append geometry: ONE definition of the tree's own block in "
                 "geometry.cuh, reached by include from kernel.cuh and from all six consumers\n";
    return failures == 0 ? 0 : 1;
}
