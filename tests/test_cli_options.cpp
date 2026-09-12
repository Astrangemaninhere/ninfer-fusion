#include "options.h"

#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

ninfer::cli::Options parse(std::vector<std::string> arguments) {
    std::vector<char*> argv;
    argv.reserve(arguments.size());
    for (std::string& argument : arguments) { argv.push_back(argument.data()); }
    return ninfer::cli::parse_options(static_cast<int>(argv.size()), argv.data());
}

bool rejects(const std::function<void()>& operation) {
    try {
        operation();
    } catch (const std::invalid_argument&) { return true; }
    return false;
}

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

} // namespace

int main() {
    int failures = 0;
    const ninfer::cli::Options configured =
        parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--thinking-budget", "37"});
    failures += check(configured.thinking_budget == 37,
                      "--thinking-budget did not preserve its positive value");
    failures +=
        check(ninfer::cli::usage_text("ninfer-cli").find("--thinking-budget") != std::string::npos,
              "CLI help omits --thinking-budget");
    failures += check(rejects([] {
                          (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello",
                                       "--thinking-budget", "0"});
                      }),
                      "zero --thinking-budget was accepted");
    failures += check(rejects([] {
                          (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello",
                                       "--thinking-budget", "8", "--no-thinking"});
                      }),
                      "--thinking-budget was accepted with --no-thinking");
    const ninfer::cli::Options with_effort =
        parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--thinking-budget", "8",
               "--reasoning-effort", "medium"});
    failures += check(with_effort.thinking_budget == 8 && with_effort.reasoning_effort,
                      "thinking budget did not coexist with reasoning effort");
    failures +=
        check(rejects([] {
                  (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--top-k", "21"});
              }),
              "CLI accepted top_k beyond the executable candidate domain");
    // --kv-tier-formats: absent means the engine keeps every existing path (the flag is
    // the only thing that turns the tier resolution on), and the vocabulary's own rules
    // are checked at parse time.
    const ninfer::cli::Options no_tiers =
        parse({"ninfer-cli", "model.ninfer", "--prompt", "hello"});
    failures += check(!no_tiers.kv_tier_formats_explicit && no_tiers.kv_tier_formats_spec.empty() &&
                          !no_tiers.kv_nvfp4_pure,
                      "the tier vocabulary is off unless asked for");
    const ninfer::cli::Options tiers = parse({"ninfer-cli", "model.ninfer", "--prompt", "hello",
                                              "--kv-tier-formats", "hot=int8,cold=int8",
                                              "--nvfp4-mode", "pure"});
    failures += check(tiers.kv_tier_formats_explicit &&
                          tiers.kv_tier_formats_spec == "hot=int8,cold=int8" && tiers.kv_nvfp4_pure,
                      "--kv-tier-formats did not preserve its spec, or --nvfp4-mode its value");
    failures += check(parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--nvfp4-mode",
                             "fusion"})
                          .kv_nvfp4_pure == false,
                      "--nvfp4-mode fusion did not resolve to the fusion default");
    failures += check(rejects([] {
                          (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello",
                                       "--kv-tier-formats", "hot=int4"});
                      }),
                      "the CLI accepted hot=int4 (below int8)");
    failures += check(rejects([] {
                          (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello",
                                       "--nvfp4-mode", "mixed"});
                      }),
                      "the CLI accepted an unknown nvfp4 mode");
    failures += check(ninfer::cli::usage_text("ninfer-cli").find("--kv-tier-formats") !=
                          std::string::npos,
                      "CLI help omits --kv-tier-formats");
    return failures == 0 ? 0 : 1;
}
