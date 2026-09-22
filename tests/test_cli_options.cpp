#include "options.h"

#include <functional>
#include <cstdlib>
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
    // Cold tier: the layered policy must be selectable from this entry point too,
    // and every pre-existing token must keep its own meaning.
    failures += check(parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--cold-policy",
                             "host-then-disk"})
                          .cold_policy == ninfer::ColdPolicy::HostThenDisk,
                      "--cold-policy host-then-disk did not select the layered policy");
    failures += check(parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--cold-policy",
                             "disk"})
                          .cold_policy == ninfer::ColdPolicy::Disk,
                      "--cold-policy disk no longer selects the disk policy");
    failures += check(parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--cold-policy",
                             "host"})
                          .cold_policy == ninfer::ColdPolicy::Host,
                      "--cold-policy host no longer selects the host policy");
    failures += check(ninfer::cli::usage_text("ninfer-cli").find("host-then-disk") !=
                          std::string::npos,
                      "CLI help omits --cold-policy host-then-disk");

    // --cold-host-bytes: usage_text advertises `N[g|m|k]`, and the parser is the thing that has
    // to accept exactly that grammar. A bare decimal keeps the value it always had, so no
    // command line that parses today changes meaning; the suffixed spellings the usage promised
    // are the ones that used to be refused by name (`32m`, `4g`, `256m`).
    failures += check(parse({"ninfer-cli", "model.ninfer", "--prompt", "hello",
                             "--cold-host-bytes", "33554432"})
                          .cold_host_bytes == (33554432ULL),
                      "--cold-host-bytes lost its bare decimal meaning");
    failures += check(parse({"ninfer-cli", "model.ninfer", "--prompt", "hello",
                             "--cold-host-bytes", "32m"})
                          .cold_host_bytes == (32ULL << 20),
                      "--cold-host-bytes 32m was not read as 32 MiB");
    failures += check(parse({"ninfer-cli", "model.ninfer", "--prompt", "hello",
                             "--cold-host-bytes", "4g"})
                          .cold_host_bytes == (4ULL << 30),
                      "--cold-host-bytes 4g was not read as 4 GiB");
    failures += check(parse({"ninfer-cli", "model.ninfer", "--prompt", "hello",
                             "--cold-host-bytes", "256K"})
                          .cold_host_bytes == (256ULL << 10),
                      "--cold-host-bytes did not accept an upper-case suffix");
    failures += check(rejects([] {
                          (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello",
                                       "--cold-host-bytes", "32x"});
                      }),
                      "the CLI accepted a --cold-host-bytes suffix the usage does not list");
    failures += check(rejects([] {
                          (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello",
                                       "--cold-host-bytes", "m"});
                      }),
                      "the CLI accepted a --cold-host-bytes suffix with no digits");
    failures += check(ninfer::cli::usage_text("ninfer-cli").find("cold-host-bytes N[g|m|k]") !=
                          std::string::npos,
                      "CLI help no longer documents the --cold-host-bytes suffix grammar");

    // FreeToken observation switch: off by default, on|off only, help documented,
    // and an explicit flag committed to the variable the observation reads
    // (ft::enabled() caches NINFER_FT_STATS on first use, so the parser is the
    // last place that can set it).
    failures += check(!parse({"ninfer-cli", "model.ninfer", "--prompt", "hello"}).ft_stats.has_value(),
                      "--ft-stats has a non-deferred default");
    failures += check(parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--ft-stats", "on"})
                          .ft_stats.value_or(false),
                      "--ft-stats on was not recorded");
    failures += check(!parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--ft-stats", "off"})
                           .ft_stats.value_or(true),
                      "--ft-stats off was not recorded");
    failures += check(rejects([] {
                          (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello",
                                       "--ft-stats", "1"});
                      }),
                      "the CLI accepted an unknown ft-stats mode");
    failures += check(ninfer::cli::usage_text("ninfer-cli").find("--ft-stats") != std::string::npos,
                      "CLI help omits --ft-stats");
#if !defined(_WIN32)
    const char* committed = std::getenv("NINFER_FT_STATS");
    failures += check(committed != nullptr && std::string(committed) == "0",
                      "--ft-stats off did not commit NINFER_FT_STATS=0");
    (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--ft-stats", "on"});
    committed = std::getenv("NINFER_FT_STATS");
    failures += check(committed != nullptr && std::string(committed) == "1",
                      "--ft-stats on did not commit NINFER_FT_STATS=1");
    unsetenv("NINFER_FT_STATS");
#endif
    return failures == 0 ? 0 : 1;
}
