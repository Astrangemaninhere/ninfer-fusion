// INTEGRATION PIN for the first fork-survey island that is WIRED, not merely landed.
//
// The 2026-09-18 additive merge landed the cometkim/feat/1m-context rope table builder
// (src/models/qwen3_5/program/rope_scaling.h) and its test, but not the vocabulary they consume
// (ops::RopeFrequencies / ops::rope_linear_frequencies / ops::rope_vision_frequencies), so the
// borrow could not compile at all. That half is landed now, and this file is the pin that the
// borrow AGREES WITH THIS TREE rather than being a parallel universe of its own:
//
//   * rope_linear_frequencies(1e7, 64)  must reproduce src/ops/kernel/rope.cuh's baked
//     kTextRopeInvFrequency[32]      (float, bit-exact)
//   * rope_linear_frequencies(1e7, 128) must reproduce kDflashRopeInvFrequency[64]  (double, 1 ulp)
//   * rope_vision_frequencies(1e4)      must reproduce kVisionRopeInvFrequency[18]  (float, bit-exact)
//   * rope_yarn_frequencies(1e7, 64, 262144, 4, 0.1) must reproduce kTextRopeYarn4InvFrequency[32]
//   * the YaRN attention factor must reproduce the kernel's kYarn4AttentionScaling
//
// The kernel's numbers are read from the header AS TEXT, the same way
// tests/ops/test_gqa_decode_split_exact.cu reads its kernel header, because they are
// __device__ __constant__ arrays that no host TU can link against. That is also why the
// comparison against the double yarn4 table carries a 1e-9 relative tolerance: the tree's
// literals are decimal with 10 significant digits, whose rounding bound is
// 0.5 * 10^-9 = 5e-10. MEASURED on this tree: the worst pair is 29, builder
// 1.1328959094002045e-07 against the kernel literal 1.132895909e-07, relative deviation
// 3.533e-10 -- inside the bound, so the shipped table IS the HF value rounded to 10 digits and
// 1e-9 is the tightest honest bound. The exactness claim is carried separately by
// tests/models/qwen3_5/test_rope_scaling.cpp, whose references are 17-digit HF values and which
// the builder reproduces bit-exactly (measured max relative deviation 0).
//
// Host-only: no CUDA runtime call, no device, no model, no lock, no subprocess, no argv.
#include "models/qwen3_5/program/rope_scaling.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#ifndef NINFER_SOURCE_DIR
#error "this test needs NINFER_SOURCE_DIR"
#endif

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

// Numbers inside `static __device__ __constant__ <type> <name>[<n>] = { ... };`, in order.
std::vector<double> kernel_table(const std::string& text, const char* name, int arity) {
    const std::string decl =
        std::string(name) + "[" + std::to_string(arity) + "] = {";
    const std::size_t at = text.find(decl);
    if (at == std::string::npos) {
        std::cerr << "kernel table declaration not found: " << decl << '\n';
        std::exit(2);
    }
    const std::size_t open = text.find('{', at);
    const std::size_t close = text.find('}', open);
    if (open == std::string::npos || close == std::string::npos) {
        std::cerr << "kernel table body not found for " << name << '\n';
        std::exit(2);
    }
    std::vector<double> out;
    const char* p = text.c_str() + open + 1;
    const char* end = text.c_str() + close;
    while (p < end) {
        while (p < end && (*p == ',' || std::isspace(static_cast<unsigned char>(*p)) != 0)) { ++p; }
        if (p >= end) { break; }
        char* stop = nullptr;
        const double v = std::strtod(p, &stop);
        if (stop == p) {
            std::cerr << "unparsable token in " << name << ": " << *p << '\n';
            std::exit(2);
        }
        out.push_back(v);
        p = stop;
        // The float tables spell their literals with a trailing F (1.0e+00F); strtod stops
        // before it, so consume it or the next iteration sees a bare 'F'.
        if (p < end && (*p == 'F' || *p == 'f')) { ++p; }
    }
    if (static_cast<int>(out.size()) != arity) {
        std::cerr << name << ": parsed " << out.size() << " values, expected " << arity << '\n';
        std::exit(2);
    }
    return out;
}

bool same_float(double got, double want) {
    return static_cast<float>(got) == static_cast<float>(want);
}
bool within_ulp(double got, double want) {
    return got == want || std::nextafter(want, 0.0) == got || std::nextafter(want, 1e30) == got;
}
bool close_rel(double got, double want, double rel) {
    const double d = std::fabs(got - want);
    return d <= rel * std::fabs(want);
}

} // namespace

int main() {
    const std::string kcuh =
        read_file(std::string(NINFER_SOURCE_DIR) + "/src/ops/kernel/rope.cuh");
    int failures = 0;

    // 1. Text linear float ladder.
    {
        const auto want = kernel_table(kcuh, "kTextRopeInvFrequency", 32);
        const auto got  = ninfer::ops::rope_linear_frequencies(1.0e7F, 64);
        for (int i = 0; i < 32; ++i) {
            if (!same_float(got.inv_frequency[i], want[i])) {
                std::cerr << "text linear pair " << i << ": builder "
                          << static_cast<float>(got.inv_frequency[i]) << " != kernel "
                          << static_cast<float>(want[i]) << '\n';
                ++failures;
            }
        }
        if (got.attention_factor != 1.0F) {
            std::cerr << "text linear: attention factor must stay 1\n";
            ++failures;
        }
    }
    // 2. DFlash double ladder.
    {
        const auto want = kernel_table(kcuh, "kDflashRopeInvFrequency", 64);
        const auto got  = ninfer::ops::rope_linear_frequencies(1.0e7F, 128);
        for (int i = 0; i < 64; ++i) {
            if (!within_ulp(got.inv_frequency[i], want[i])) {
                std::cerr << "dflash linear pair " << i << ": builder " << got.inv_frequency[i]
                          << " != kernel " << want[i] << " (beyond one ulp)\n";
                ++failures;
            }
        }
    }
    // 3. Vision wrapped ladder.
    {
        const auto want = kernel_table(kcuh, "kVisionRopeInvFrequency", 18);
        const auto got  = ninfer::ops::rope_vision_frequencies(10'000.0F);
        for (int i = 0; i < 36; ++i) {
            if (!same_float(got.inv_frequency[i], want[i % 18])) {
                std::cerr << "vision pair " << i << ": builder "
                          << static_cast<float>(got.inv_frequency[i]) << " != kernel "
                          << static_cast<float>(want[i % 18]) << '\n';
                ++failures;
            }
        }
    }
    // 4. The shipped YaRN factor-4 table the engine actually rotates with.
    {
        const auto want = kernel_table(kcuh, "kTextRopeYarn4InvFrequency", 32);
        const auto got = ninfer::models::qwen3_5::rope_yarn_frequencies(1.0e7F, 64, 262144, 4.0F);
        for (int i = 0; i < 32; ++i) {
            if (!close_rel(got.inv_frequency[i], want[i], 1.0e-9)) {
                std::cerr << "yarn4 vs kernel pair " << i << ": builder " << got.inv_frequency[i]
                          << " != kernel " << want[i] << '\n';
                ++failures;
            }
        }
        // The kernel folds the attention scaling into the sincos coefficients:
        //   constexpr float kYarn4AttentionScaling = 1.138629436111989f;  == 0.1*ln(4)+1
        const std::string decl = "kYarn4AttentionScaling = ";
        const std::size_t at  = kcuh.find(decl);
        if (at == std::string::npos) {
            std::cerr << "kernel kYarn4AttentionScaling not found\n";
            ++failures;
        } else {
            const double want_af = std::strtod(kcuh.c_str() + at + decl.size(), nullptr);
            if (!same_float(got.attention_factor, want_af)) {
                std::cerr << "yarn4 attention factor: builder " << got.attention_factor
                          << " != kernel " << static_cast<float>(want_af) << '\n';
                ++failures;
            }
        }
    }

    std::cout << (failures == 0 ? "PASS" : "FAIL")
              << " rope frequencies vs the shipped kernel tables\n";
    return failures == 0 ? 0 : 1;
}
