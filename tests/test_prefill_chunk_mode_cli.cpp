// The two FRONT ENDS' half of the prefill-chunk mode: that `--prefill-chunk-mode manual` actually
// reaches their own `Options` struct. tests/test_prefill_chunk_mode.cpp covers the mechanism and the
// vocabulary (header-only, no front end in the graph); this file covers the step between them, which
// is where a flag normally dies -- parsed, validated, and then never copied anywhere.
//
// Why it is a separate program: it compiles the front ends' own translation units
// (apps/cli/options.cpp, src/serve/serve_options.cpp) exactly like tests/test_cli_options.cpp and
// tests/test_serve_options.cpp do, so what is exercised is the real parser at the real call site,
// not a copy of its rules.
//
// Negative controls (all four run every time, see REPORT.md):
//   * an unknown mode must be REJECTED by both front ends (a parser that accepted anything would
//     fail this);
//   * the mode must be UNSET when the flag is absent (so "manual by default" cannot pass);
//   * `--prefill-chunk-mode` must not be mistakable for `--prefill-chunk` (the two flags have
//     different grammars; a front end that folded them together would fail one of the two rows).
#include "options.h"
#include "serve/serve_options.h"

#include <cstdlib>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using ninfer::PrefillChunkMode;

ninfer::cli::Options parse_cli(std::vector<std::string> arguments) {
    std::vector<char*> argv;
    argv.reserve(arguments.size());
    for (std::string& argument : arguments) { argv.push_back(argument.data()); }
    return ninfer::cli::parse_options(static_cast<int>(argv.size()), argv.data());
}

ninfer::serve::ServeOptions parse_serve(std::vector<std::string> arguments) {
    std::vector<char*> argv;
    argv.reserve(arguments.size());
    for (std::string& argument : arguments) { argv.push_back(argument.data()); }
    return ninfer::serve::parse_serve_options(static_cast<int>(argv.size()), argv.data());
}

bool rejects(const std::function<void()>& operation) {
    try {
        operation();
    } catch (const std::invalid_argument&) { return true; }
    return false;
}

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

} // namespace

int main() {
    int failures = 0;

    // Flag absent: the mode must be UNSET, not manual. Unset is what makes the engine's own
    // "CLI > environment > default" resolution reachable at all; a front end that defaulted the
    // mode would freeze NINFER_FT_BW_GOV out of the picture.
    const ninfer::cli::Options bare =
        parse_cli({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--prefill-chunk", "1024"});
    failures += check(!bare.prefill_chunk_mode.has_value(),
                      "ninfer-cli without --prefill-chunk-mode did not leave the mode unset");
    failures += check(bare.prefill_chunk == 1024,
                      "ninfer-cli lost --prefill-chunk while the mode was absent");

    const ninfer::cli::Options manual =
        parse_cli({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--prefill-chunk", "1024",
                   "--prefill-chunk-mode", "manual"});
    failures += check(manual.prefill_chunk_mode == PrefillChunkMode::Manual,
                      "ninfer-cli --prefill-chunk-mode manual did not reach Options::prefill_chunk_mode");
    failures += check(manual.prefill_chunk == 1024,
                      "ninfer-cli --prefill-chunk-mode manual changed --prefill-chunk");

    const ninfer::cli::Options dynamic =
        parse_cli({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--prefill-chunk-mode",
                   "dynamic"});
    failures += check(dynamic.prefill_chunk_mode == PrefillChunkMode::Dynamic,
                      "ninfer-cli --prefill-chunk-mode dynamic did not reach Options::prefill_chunk_mode");

    failures += check(rejects([] {
                          (void)parse_cli({"ninfer-cli", "model.ninfer", "--prompt", "hello",
                                           "--prefill-chunk-mode", "Manual"});
                      }),
                      "ninfer-cli accepted --prefill-chunk-mode Manual (wrong case)");
    failures += check(rejects([] {
                          (void)parse_cli({"ninfer-cli", "model.ninfer", "--prompt", "hello",
                                           "--prefill-chunk-mode", "1024"});
                      }),
                      "ninfer-cli accepted --prefill-chunk-mode 1024: the mode flag is not the chunk flag");

    // The server's handle, which used to be the front end with no KV/prefill knobs at all: the flag
    // has to reach ITS struct too, and it must parse the same vocabulary (one vocabulary, two front
    // ends -- BandwidthGovernor::parse_mode).
    const ninfer::serve::ServeOptions serve_bare =
        parse_serve({"ninfer-serve", "model.ninfer", "--prefill-chunk", "1024"});
    failures += check(!serve_bare.prefill_chunk_mode.has_value(),
                      "ninfer-serve without --prefill-chunk-mode did not leave the mode unset");

    const ninfer::serve::ServeOptions serve_manual =
        parse_serve({"ninfer-serve", "model.ninfer", "--prefill-chunk", "1024",
                     "--prefill-chunk-mode", "manual"});
    failures += check(serve_manual.prefill_chunk_mode == PrefillChunkMode::Manual,
                      "ninfer-serve --prefill-chunk-mode manual did not reach ServeOptions");
    failures += check(serve_manual.prefill_chunk == 1024,
                      "ninfer-serve --prefill-chunk-mode manual changed --prefill-chunk");
    failures += check(rejects([] {
                          (void)parse_serve({"ninfer-serve", "model.ninfer",
                                             "--prefill-chunk-mode", "on"});
                      }),
                      "ninfer-serve accepted --prefill-chunk-mode on (there is no on/off spelling)");

    // The usage text has to name both modes, or the flag is undiscoverable: the two front ends
    // printed `--prefill-chunk N` only, which is how a two-mode surface stays invisible.
    failures += check(ninfer::cli::usage_text("ninfer-cli").find("--prefill-chunk-mode") !=
                          std::string::npos,
                      "ninfer-cli usage text does not mention --prefill-chunk-mode");
    failures += check(ninfer::cli::usage_text("ninfer-cli").find("manual") != std::string::npos,
                      "ninfer-cli usage text does not name the manual mode");
    failures += check(ninfer::serve::serve_usage_text("ninfer-serve").find("--prefill-chunk-mode") !=
                          std::string::npos,
                      "ninfer-serve usage text does not mention --prefill-chunk-mode");

    if (failures != 0) {
        std::cerr << failures << " prefill-chunk front-end checks failed\n";
        return 1;
    }
    std::cout << "prefill-chunk front-end checks passed\n";
    return 0;
}
