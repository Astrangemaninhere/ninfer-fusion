// tools/kv_relayout_test.cpp — host-only unit test for the FreeToken step-2
// decision logic (src/serve/kv_auto_relayout.cpp). No CUDA device required.
//
// Cross-checks the C++ policy against tools/archkit/ft_tiers.py (the Python
// reference) on the same synthetic energy table, plus the hysteresis seam.
//
// Build (WSL or Windows):
//   g++ -std=c++20 -I <repo>/src -I <repo>/include tools/kv_relayout_test.cpp \
//       <repo>/src/serve/kv_auto_relayout.cpp -o kv_relayout_test
// Run:
//   ./kv_relayout_test --expect "<spec printed by ft_tiers.py>"
#include "serve/kv_auto_relayout.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace {

std::vector<std::pair<int, double>> synthetic_energy() {
    // 16 full-attention layers, strictly ascending energy so the tertile split
    // is unambiguous: low third (0-4) -> e8, middle (5-9) -> iso3, high -> nvfp4;
    // layers 13-15 are the deep band and must stay nvfp4 regardless.
    std::vector<std::pair<int, double>> energy;
    for (int layer = 0; layer < 16; ++layer) {
        energy.emplace_back(layer, 1.0 + 0.5 * layer);
    }
    return energy;
}

} // namespace

int main(int argc, char** argv) {
    std::string expect;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--expect") == 0 && i + 1 < argc) { expect = argv[++i]; }
    }

    const auto energy = synthetic_energy();
    const std::string spec = ninfer::serve::build_ft_spec(energy, 16, 0.2);
    std::printf("spec=%s\n", spec.c_str());

    bool ok = true;
    if (!expect.empty()) {
        const bool match = expect == spec;
        std::printf("python-reference: %s\n", match ? "MATCH" : "MISMATCH");
        if (!match) {
            std::printf("  expected=%s\n  actual  =%s\n", expect.c_str(), spec.c_str());
            ok = false;
        }
    }

    // Hysteresis: the first cycle only records the candidate; the second cycle
    // applies it; an identical third cycle is a no-op.
    int applies = 0;
    ninfer::serve::KvAutoRelayout::Config config;
    config.interval_secs = 0; // decide_once is called directly
    config.full_attn_layers = 16;
    ninfer::serve::KvAutoRelayout relayout(config, [&](std::string_view applied) {
        ++applies;
        std::printf("apply[%d]=%s\n", applies, std::string(applied).c_str());
        return true;
    });
    const std::string first = relayout.decide_once(energy);
    const std::string second = relayout.decide_once(energy);
    const std::string third = relayout.decide_once(energy);
    std::printf("hysteresis: first=%s second=%s third=%s\n",
                first.empty() ? "(none)" : first.c_str(),
                second.empty() ? "(none)" : second.c_str(),
                third.empty() ? "(none)" : third.c_str());
    if (!first.empty() || second != spec || !third.empty() || applies != 1) {
        std::printf("hysteresis FAIL\n");
        ok = false;
    }

    // format_kv_table round trip: the applied table renders back to the spec.
    std::array<ninfer::KvCacheStorage, ninfer::kKvLayerStorageSlots> table{};
    for (int layer = 0; layer < 16; ++layer) { table[static_cast<std::size_t>(layer)] = ninfer::KvCacheStorage::BFloat16; }
    table[0] = ninfer::KvCacheStorage::E8Group64;
    table[1] = ninfer::KvCacheStorage::E8Group64;
    table[2] = ninfer::KvCacheStorage::Nvfp4Group16;
    const std::string formatted = ninfer::serve::format_kv_table(table, 3);
    std::printf("format_kv_table=%s\n", formatted.c_str());
    if (formatted != "0-1:e8,2:nvfp4") {
        std::printf("format_kv_table FAIL\n");
        ok = false;
    }

    std::printf("%s\n", ok ? "KV_RELAYOUT_TEST PASS" : "KV_RELAYOUT_TEST FAIL");
    return ok ? 0 : 1;
}
