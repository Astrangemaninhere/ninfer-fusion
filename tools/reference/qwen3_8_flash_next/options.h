// Derived from igorls/ninfer @ 5e4a66d (Apache-2.0), modified for this tree (ninfer-fusion).
#pragma once

#include "ninfer/types.h"

#include <cstdint>
#include <span>
#include <string>
#include <string_view>

namespace ninfer::targets::qwen3_8_flash_next::detail {

struct ReferenceToolOptions {
    std::string model_path;
    std::string mode =
        "preflight"; // "preflight", "execute-token", "materialize-full", "materialize-vision", "materialize-mtp", "mtp-step", "chat-diagnostic", "execute-vision", "continuation-check", "oracle-chat23-logits", or "mtp-acceptance-trace"
    std::uint32_t max_context     = 4096;
    std::uint32_t max_concurrency = 1;
    std::uint32_t page_groups     = 0;
    std::uint32_t state_slots     = 0;
    std::int32_t token_id         = 0;
    std::uint32_t prefill_chunk   = 0;
    bool use_cuda_graph           = true; // --no-cuda-graph forces the eager decode body
    std::string dump_gen_logits;          // per-round logits of the generation loop
    std::string oracle_chat23_logits;     // dump posNNNN_logits.bin for the 23 oracle tokens
    bool qsa_prefill_mma              = false;
    std::uint32_t repeat_prefill  = 1;    // --repeat-prefill N: prefill N times, compare logits
    bool do_commit                = true;
    bool json_output              = false;
    ninfer::KvCacheStorage kv_cache = ninfer::KvCacheStorage::BFloat16;

    // Chat diagnostic options
    std::string prompt;
    std::string system_prompt;
    std::string dump_states;
    float temperature             = 1.0f;
    std::int32_t top_k            = 20;
    float top_p                   = 0.95f;
    unsigned long long seed       = 0ULL;
    std::uint32_t max_tokens      = 512;
    std::uint32_t thinking_budget = 0;
    std::string reasoning_effort  = "medium";
    std::string continuation_check;
};

void print_reference_tool_usage(std::string_view prog);

[[nodiscard]] ReferenceToolOptions
parse_reference_tool_options(std::span<const std::string_view> args);

[[nodiscard]] ReferenceToolOptions parse_reference_tool_options(int argc, char** argv);

} // namespace ninfer::targets::qwen3_8_flash_next::detail
