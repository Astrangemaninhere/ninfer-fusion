#include "ninfer/engine.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

int main() {
    const char* artifact = std::getenv("NINFER_QWEN3_6_27B_WEIGHTS");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "SKIP: NINFER_QWEN3_6_27B_WEIGHTS is not set\n";
        return 77;
    }

    ninfer::EngineOptions options;
    options.artifact_path = artifact;
    options.purpose       = ninfer::EnginePurpose::CausalScoring;
    options.max_context   = 2048;
    options.kv_cache      = ninfer::KvCacheStorage::Fp8E4M3Row256;
    ninfer::Engine engine(options);
    const auto& effective = engine.options();
    // prefill_chunk: this window names no --prefill-chunk, so what it asks for is the
    // EngineOptions default, and 3072 is the value CausalScoring has to leave standing.
    // 3072 is written in exactly TWO places in the tree and they are two spellings of the
    // same default, not two competing decisions:
    //   * include/ninfer/types.h:230 -- `std::uint32_t prefill_chunk = 3072;`, the default
    //     member initialiser. THIS is the one that supplies the value here: the test never
    //     writes the field, so engine.cpp's fallback branch is not even entered.
    //   * src/runtime/engine/engine.cpp:60 -- `if (prefill_chunk == 0) { prefill_chunk =
    //     3072; }`, the sentinel for a caller that spells "unset" as 0 (apps/cli/options.h:23
    //     carries the same 3072 as the CLI default, so the flag and the default agree).
    // 1024 was never a default of this struct: it is the SERVING front door's chunk
    // (src/serve/serve_options.h:34), so asserting it here described a value no
    // CausalScoring engine ever produced -- which is why this test returned rc=1 the
    // moment it was given a real artifact instead of skipping. CausalScoring also no
    // longer overwrites the caller's tile (engine.cpp:56-62 keeps the request and
    // normalises only the 0 / off-128-alignment spellings), so what this clause now
    // asserts is "the requested score tile survives normalisation" -- the contract.
    if (effective.max_concurrency != 1 || effective.prefill_chunk != 3072 ||
        effective.kv_capacity.mode != ninfer::KvCapacityMode::Explicit ||
        effective.kv_capacity.explicit_tokens != effective.max_context ||
        effective.context_cache.enabled ||
        effective.speculative.backend != ninfer::SpeculativeBackend::None ||
        effective.kv_cache != ninfer::KvCacheStorage::Fp8E4M3Row256) {
        std::cerr << "causal scoring options were not normalized correctly\n";
        return 1;
    }

    std::string text;
    const std::string paragraph =
        "NInfer scores each target token from the preceding hidden state. "
        "Every evaluation window owns fresh state and a fresh KV address space.\n";
    std::vector<ninfer::TokenId> tokens;
    while (tokens.size() < 1537) {
        text += paragraph;
        tokens = engine.tokenize_text(text);
    }
    tokens.resize(1537);

    const std::vector<float> all      = engine.score_tokens(tokens, 1);
    const std::vector<float> suffix   = engine.score_tokens(tokens, 513);
    const std::vector<float> repeated = engine.score_tokens(tokens, 513);
    if (all.size() != 1536 || suffix.size() != 1024 || repeated.size() != suffix.size()) {
        std::cerr << "causal scoring returned an invalid result shape\n";
        return 1;
    }
    float maximum_overlap_error = 0.0F;
    for (std::size_t i = 0; i < suffix.size(); ++i) {
        if (!std::isfinite(all[i + 512]) || !std::isfinite(suffix[i])) {
            std::cerr << "causal scoring returned a non-finite logprob\n";
            return 1;
        }
        maximum_overlap_error = std::max(maximum_overlap_error, std::abs(all[i + 512] - suffix[i]));
        if (suffix[i] != repeated[i]) {
            std::cerr << "a repeated score window inherited prior State/KV\n";
            return 1;
        }
    }
    if (maximum_overlap_error > 0.25F) {
        std::cerr << "overlapping target suffix changed by " << maximum_overlap_error << '\n';
        return 1;
    }
    std::cout << "OK causal_score_real max_overlap_error=" << maximum_overlap_error << '\n';
    return 0;
}
