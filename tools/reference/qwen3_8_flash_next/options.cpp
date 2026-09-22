// Derived from igorls/ninfer @ 5e4a66d (Apache-2.0), modified for this tree (ninfer-fusion).
#include "options.h"

#include <charconv>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::targets::qwen3_8_flash_next::detail {
namespace {

std::uint32_t parse_strict_u32(std::string_view str, std::string_view option_name) {
    if (str.empty()) { throw std::invalid_argument(std::string(option_name) + ": empty value"); }
    if (str.front() == '+' || str.front() == '-') {
        throw std::invalid_argument(std::string(option_name) + ": invalid unsigned integer '" +
                                    std::string(str) + "'");
    }
    std::uint32_t val    = 0;
    const auto [ptr, ec] = std::from_chars(str.data(), str.data() + str.size(), val);
    if (ec != std::errc{} || ptr != str.data() + str.size()) {
        throw std::invalid_argument(std::string(option_name) +
                                    ": invalid unsigned integer or trailing characters in '" +
                                    std::string(str) + "'");
    }
    return val;
}

std::int32_t parse_strict_i32(std::string_view str, std::string_view option_name) {
    if (str.empty()) { throw std::invalid_argument(std::string(option_name) + ": empty value"); }
    std::int32_t val     = 0;
    const auto [ptr, ec] = std::from_chars(str.data(), str.data() + str.size(), val);
    if (ec != std::errc{} || ptr != str.data() + str.size()) {
        throw std::invalid_argument(std::string(option_name) +
                                    ": invalid integer or trailing characters in '" +
                                    std::string(str) + "'");
    }
    return val;
}

std::uint64_t parse_strict_u64(std::string_view str, std::string_view option_name) {
    if (str.empty()) { throw std::invalid_argument(std::string(option_name) + ": empty value"); }
    if (str.front() == '+' || str.front() == '-') {
        throw std::invalid_argument(std::string(option_name) + ": invalid unsigned integer '" +
                                    std::string(str) + "'");
    }
    std::uint64_t val    = 0;
    const auto [ptr, ec] = std::from_chars(str.data(), str.data() + str.size(), val);
    if (ec != std::errc{} || ptr != str.data() + str.size()) {
        throw std::invalid_argument(std::string(option_name) +
                                    ": invalid unsigned integer or trailing characters in '" +
                                    std::string(str) + "'");
    }
    return val;
}

float parse_strict_f32(std::string_view str, std::string_view option_name) {
    if (str.empty()) { throw std::invalid_argument(std::string(option_name) + ": empty value"); }
    try {
        std::size_t pos = 0;
        const std::string s(str);
        const float val = std::stof(s, &pos);
        if (pos != str.size()) {
            throw std::invalid_argument(std::string(option_name) + ": trailing characters in '" +
                                        s + "'");
        }
        return val;
    } catch (const std::invalid_argument&) {
        throw std::invalid_argument(std::string(option_name) + ": invalid float '" +
                                    std::string(str) + "'");
    } catch (const std::out_of_range&) {
        throw std::invalid_argument(std::string(option_name) + ": float out of range '" +
                                    std::string(str) + "'");
    }
}

} // namespace

void print_reference_tool_usage(std::string_view prog) {
    std::cout
        << "Usage: " << prog << " [options]\n\n"
        << "Target-private reference and inspection tool for Qwen3.8-Flash-Next native "
           "artifacts.\n\n"
        << "Options:\n"
        << "  --model, -m <path>         Path to .ninfer artifact (required)\n"
        << "  --mode <mode>              Execution mode: 'preflight' (default), 'execute-token', "
           "'materialize-full', 'materialize-vision', 'chat-diagnostic', or 'execute-vision'\n"
        << "  --preflight                Shortcut for --mode preflight\n"
        << "  --execute-token            Shortcut for --mode execute-token\n"
        << "  --materialize-full         Shortcut for --mode materialize-full\n"
        << "  --materialize-vision       Shortcut for --mode materialize-vision\n"
        << "  --chat-diagnostic          Shortcut for --mode chat-diagnostic\n"
        << "  --execute-vision           Shortcut for --mode execute-vision\n"
        << "  --continuation-check <text> Run Turn 1, Turn 2 resumed and from-scratch, compare logits\n"
        << "  --prompt <string>          User prompt text for chat-diagnostic\n"
        << "  --system <string>          Optional system prompt for chat-diagnostic\n"
        << "  --dump-states <dir>        Directory to dump raw state tensors for oracle comparison\n"
        << "  --dump-gen-logits <dir>    Write the logits of every generation round (gen_NNN.bin)\n"
        << "  --oracle-chat23-logits <dir> Prefill tokens[0..p] on a fresh lane for p=0..22 and dump\n"
        << "                             posNNNN_logits.bin (BF16 vocab). Pair with --qsa-prefill-mma\n"
        << "                             on vs off in one production window.\n"
        << "  --qsa-prefill-mma          Set NINFER_FLASH_NEXT_QSA_PREFILL_MMA=1 before plan build\n"
        << "  --kv-dtype <dtype>         KV cache storage dtype: 'bf16' (default) or 'fp8'\n"
        << "  --no-cuda-graph            Run decode rounds eagerly instead of replaying CUDA graphs\n"
        << "  --repeat-prefill <n>       Run the prompt prefill n times on fresh lanes and compare logits\n"
        << "  --temperature <float>      Sampling temperature (default: 1.0, 0 = greedy)\n"
        << "  --top-k <N>                Sampling top-k (default: 20)\n"
        << "  --top-p <float>            Sampling top-p (default: 0.95)\n"
        << "  --seed <N>                 Sampling seed (default: 0)\n"
        << "  --greedy                   Shortcut for deterministic greedy sampling (--temperature 0)\n"
        << "  --max-tokens, -n <N>       Maximum generated tokens (default: 512)\n"
        << "  --thinking-budget <N>      Thinking token budget forcing (not supported)\n"
        << "  --reasoning-effort <effort> Reasoning effort: 'low', 'medium' (default), 'high', 'none'\n"
        << "  --max-context <tokens>     Maximum context length in tokens (default: 4096, max: "
           "262144)\n"
        << "  --max-concurrency <B>      Maximum concurrent decode requests (default: 1, range: "
           "[1, 8])\n"
        << "  --page-groups <N>          Exact physical page group count (default: 0 = max for "
           "context)\n"
        << "  --state-slots <N>          State slot capacity (default: 2 * concurrency, range: [2 "
           "* B, 64])\n"
        << "  --token-id <id>            Input token ID for execute-token mode in [0, 248320) "
           "(default: 0)\n"
        << "  --commit                   Commit the executed token round (default: true)\n"
        << "  --abort                    Abort the executed token round without committing\n"
        << "  --json                     Output structured JSON\n"
        << "  --help, -h                 Show this help message\n";
}

ReferenceToolOptions parse_reference_tool_options(std::span<const std::string_view> args) {
    ReferenceToolOptions opts;
    for (std::size_t i = 0; i < args.size(); ++i) {
        const std::string_view arg = args[i];
        if (arg == "--help" || arg == "-h") {
            print_reference_tool_usage(args.empty() ? "ninfer_qwen3_8_flash_next_reference"
                                                     : args[0]);
            std::exit(0);
        } else if (arg == "--model" || arg == "-m") {
            if (++i >= args.size()) throw std::invalid_argument("Missing argument for --model");
            opts.model_path = std::string(args[i]);
        } else if (arg == "--mode") {
            if (++i >= args.size()) throw std::invalid_argument("Missing argument for --mode");
            opts.mode = std::string(args[i]);
        } else if (arg == "--preflight") {
            opts.mode = "preflight";
        } else if (arg == "--execute-token") {
            opts.mode = "execute-token";
        } else if (arg == "--materialize-full") {
            opts.mode = "materialize-full";
        } else if (arg == "--materialize-vision") {
            opts.mode = "materialize-vision";
        } else if (arg == "--materialize-mtp") {
            opts.mode = "materialize-mtp";
        } else if (arg == "--mtp-step") {
            opts.mode = "mtp-step";
        } else if (arg == "--mtp-acceptance-trace") {
            opts.mode = "mtp-acceptance-trace";
        } else if (arg == "--chat-diagnostic") {
            opts.mode = "chat-diagnostic";
        } else if (arg == "--prompt") {
            if (++i >= args.size()) throw std::invalid_argument("Missing argument for --prompt");
            opts.prompt = std::string(args[i]);
        } else if (arg == "--system") {
            if (++i >= args.size()) throw std::invalid_argument("Missing argument for --system");
            opts.system_prompt = std::string(args[i]);
        } else if (arg == "--dump-states") {
            if (++i >= args.size()) throw std::invalid_argument("Missing argument for --dump-states");
            opts.dump_states = std::string(args[i]);
        } else if (arg == "--temperature") {
            if (++i >= args.size())
                throw std::invalid_argument("Missing argument for --temperature");
            opts.temperature = parse_strict_f32(args[i], "--temperature");
        } else if (arg == "--top-k") {
            if (++i >= args.size()) throw std::invalid_argument("Missing argument for --top-k");
            opts.top_k = parse_strict_i32(args[i], "--top-k");
        } else if (arg == "--top-p") {
            if (++i >= args.size()) throw std::invalid_argument("Missing argument for --top-p");
            opts.top_p = parse_strict_f32(args[i], "--top-p");
        } else if (arg == "--seed") {
            if (++i >= args.size()) throw std::invalid_argument("Missing argument for --seed");
            opts.seed = parse_strict_u64(args[i], "--seed");
        } else if (arg == "--greedy") {
            opts.temperature = 0.0f;
        } else if (arg == "--max-tokens" || arg == "-n") {
            if (++i >= args.size())
                throw std::invalid_argument("Missing argument for --max-tokens");
            opts.max_tokens = parse_strict_u32(args[i], "--max-tokens");
        } else if (arg == "--thinking-budget") {
            if (++i >= args.size())
                throw std::invalid_argument("Missing argument for --thinking-budget");
            opts.thinking_budget = parse_strict_u32(args[i], "--thinking-budget");
        } else if (arg == "--reasoning-effort") {
            if (++i >= args.size())
                throw std::invalid_argument("Missing argument for --reasoning-effort");
            opts.reasoning_effort = std::string(args[i]);
        } else if (arg == "--execute-vision") {
            opts.mode = "execute-vision";
        } else if (arg == "--max-context") {
            if (++i >= args.size())
                throw std::invalid_argument("Missing argument for --max-context");
            opts.max_context = parse_strict_u32(args[i], "--max-context");
        } else if (arg == "--max-concurrency") {
            if (++i >= args.size())
                throw std::invalid_argument("Missing argument for --max-concurrency");
            opts.max_concurrency = parse_strict_u32(args[i], "--max-concurrency");
        } else if (arg == "--page-groups") {
            if (++i >= args.size())
                throw std::invalid_argument("Missing argument for --page-groups");
            opts.page_groups = parse_strict_u32(args[i], "--page-groups");
        } else if (arg == "--state-slots") {
            if (++i >= args.size())
                throw std::invalid_argument("Missing argument for --state-slots");
            opts.state_slots = parse_strict_u32(args[i], "--state-slots");
        } else if (arg == "--token-id") {
            if (++i >= args.size()) throw std::invalid_argument("Missing argument for --token-id");
            opts.token_id = parse_strict_i32(args[i], "--token-id");
        } else if (arg == "--repeat-prefill") {
            if (++i >= args.size())
                throw std::invalid_argument("Missing argument for --repeat-prefill");
            opts.repeat_prefill = parse_strict_u32(args[i], "--repeat-prefill");
        } else if (arg == "--no-cuda-graph") {
            opts.use_cuda_graph = false;
        } else if (arg == "--dump-gen-logits") {
            if (++i >= args.size())
                throw std::invalid_argument("Missing argument for --dump-gen-logits");
            opts.dump_gen_logits = std::string(args[i]);
        } else if (arg == "--prefill-chunk") {
            if (++i >= args.size())
                throw std::invalid_argument("Missing argument for --prefill-chunk");
            opts.prefill_chunk = parse_strict_u32(args[i], "--prefill-chunk");
        } else if (arg == "--continuation-check") {
            if (++i >= args.size())
                throw std::invalid_argument("Missing argument for --continuation-check");
            opts.mode = "continuation-check";
            opts.continuation_check = std::string(args[i]);
        } else if (arg == "--commit") {
            opts.do_commit = true;
        } else if (arg == "--abort") {
            opts.do_commit = false;
        } else if (arg == "--oracle-chat23-logits") {
            if (++i >= args.size())
                throw std::invalid_argument("Missing argument for --oracle-chat23-logits");
            opts.oracle_chat23_logits = std::string(args[i]);
            opts.mode                 = "oracle-chat23-logits";
        } else if (arg == "--qsa-prefill-mma") {
            opts.qsa_prefill_mma = true;
        } else if (arg == "--kv-dtype") {
            if (++i >= args.size())
                throw std::invalid_argument("Missing argument for --kv-dtype");
            if (args[i] == "bf16") {
                opts.kv_cache = ninfer::KvCacheStorage::BFloat16;
            } else if (args[i] == "fp8") {
                opts.kv_cache = ninfer::KvCacheStorage::Fp8E4M3Row256;
            } else {
                throw std::invalid_argument("Invalid --kv-dtype: " + std::string(args[i]) +
                                            " (Flash-Next supports bf16 and fp8)");
            }
        } else if (arg == "--json") {
            opts.json_output = true;
        } else {
            throw std::invalid_argument("Unknown argument: " + std::string(arg));
        }
    }

    if (opts.model_path.empty()) { throw std::invalid_argument("--model <path> is required"); }
    if (opts.mode != "preflight" && opts.mode != "execute-token" &&
        opts.mode != "materialize-full" && opts.mode != "materialize-vision" &&
        opts.mode != "chat-diagnostic" && opts.mode != "execute-vision" &&
        opts.mode != "continuation-check" && opts.mode != "oracle-chat23-logits") {
        throw std::invalid_argument(
            "Invalid --mode: must be 'preflight', 'execute-token', 'materialize-full', "
            "'materialize-vision', 'chat-diagnostic', 'execute-vision', 'continuation-check', "
            "or 'oracle-chat23-logits'");
    }
    if (opts.thinking_budget > 0) {
        throw std::invalid_argument(
            "--thinking-budget is not supported: thinking budget forcing is not yet implemented");
    }
    if (opts.temperature < 0.0f) {
        throw std::invalid_argument("--temperature must be non-negative");
    }
    if (opts.top_p <= 0.0f || opts.top_p > 1.0f) {
        throw std::invalid_argument("--top-p must be in range (0.0, 1.0]");
    }
    if (opts.max_tokens == 0) {
        throw std::invalid_argument("--max-tokens must be positive");
    }
    if (opts.max_concurrency < 1 || opts.max_concurrency > 8) {
        throw std::invalid_argument("--max-concurrency must be between 1 and 8");
    }
    if (opts.max_context < 1 || opts.max_context > 262'144) {
        throw std::invalid_argument("--max-context must be between 1 and 262144");
    }
    if (opts.state_slots != 0 &&
        (opts.state_slots < 2U * opts.max_concurrency || opts.state_slots > 64)) {
        throw std::invalid_argument("--state-slots must be between 2 * max_concurrency and 64");
    }
    if (opts.token_id < 0 || opts.token_id >= 248'320) {
        throw std::invalid_argument("--token-id must be in range [0, 248320)");
    }
    if (!opts.dump_states.empty() && opts.mode != "execute-token" &&
        opts.mode != "chat-diagnostic") {
        throw std::invalid_argument(
            "--dump-states is only valid with --execute-token or --chat-diagnostic");
    }

    return opts;
}

ReferenceToolOptions parse_reference_tool_options(int argc, char** argv) {
    std::vector<std::string_view> args;
    args.reserve(argc > 1 ? static_cast<std::size_t>(argc - 1) : 0);
    for (int i = 1; i < argc; ++i) { args.emplace_back(argv[i]); }
    return parse_reference_tool_options(args);
}

} // namespace ninfer::targets::qwen3_8_flash_next::detail
