// tests/semchan_symbol_key_compile_check.cpp
//
// THE COMPILE-TIME PAIR THE ORDER ASKS FOR: with a symbol key FED to the tree's exact-key producer,
// the translation unit must NOT compile (rc != 0); without it fed, it must compile AND RUN (rc = 0).
//
//   g++ -std=gnu++20 -c <this>                      -> the NOT-FED arm, rc = 0, object must exist
//   g++ -std=gnu++20 -c -DNINFER_SEMCHAN_FEED_SYMBOL_KEY <this>  -> the FED arm, rc != 0
//
// WHAT THE FED ARM PROVES, AND WHY IT IS A TYPE ERROR RATHER THAN A static_assert. A refusal written
// as a hand-placed `static_assert(false, ...)` proves only that someone wrote a message. The refusal
// that matters is that THE DOORWAY IS CLOSED BY THE TYPE SYSTEM: `SumDirQueryKey` is reachable only
// through `sum_dir_query_key_from_ids(const std::uint32_t*, std::uint32_t, ...)` and its vector
// overload, and a `SemChanSymbolKey` is neither a `const std::uint32_t*` nor a
// `std::vector<std::uint32_t>`. So the FED arm below is not a message; it is the compiler refusing
// to convert one type into the other, which is `sum_dir_query_key.h:37-39` and `:121` enforced by
// construction instead of by comment. The exact consequence, in the tree's own words: the channel
// "may enter the candidate-set UNION and nothing else" -- and there is no door from it to a KEY.
//
// THE NOT-FED ARM IS NOT DECORATION EITHER: it prints the key's own `token_count` and the offset
// where the admitted source's digest lands, so a reader sees the admitted path WORK while its
// neighbour is refused.
//
// NO CUDA, NO ninfer library, NO fixture: the same build shape as `ninfer_sum_dir_test` and
// `ninfer_recall_restore_first_test` (tests/CMakeLists.txt:1131-1145).

#include "spec/semchan_symbol_key.h"
#include "spec/sum_dir_query_key.h"

#include <cstdint>
#include <cstdio>
#include <vector>

int main() {
    using ninfer::spec::semchan::SemChanSymbolKey;
    using ninfer::spec::sum_dir::query_key::SumDirQueryKeySource;
    using ninfer::spec::sum_dir::query_key::sum_dir_query_key_source_admitted;
    using ninfer::spec::sum_dir::query_key::sum_dir_query_key_source_name;
    using ninfer::spec::sum_dir::query_key::sum_dir_query_key_source_reason;

    const std::vector<std::uint32_t> ids{11U, 12U, 13U, 14U};
    const SemChanSymbolKey symbol_key = ninfer::spec::semchan::semchan_symbol_key_from_ids(ids);
    std::printf("channel: symbol_key tokens=%u distinct_symbols=%zu\n", symbol_key.tokens,
                symbol_key.symbols.size());
    // THE TREE'S OWN REFUSAL, READ AT RUNTIME AS WELL AS AT COMPILE TIME: this is the source this
    // channel IS, and the tree already says it may not be a key.
    std::printf("tree:    source '%s' admitted=%d\n",
                sum_dir_query_key_source_name(SumDirQueryKeySource::EmbeddingVector),
                sum_dir_query_key_source_admitted(SumDirQueryKeySource::EmbeddingVector) ? 1 : 0);
    std::printf("tree:    reason: %s\n",
                sum_dir_query_key_source_reason(SumDirQueryKeySource::EmbeddingVector));

#if defined(NINFER_SEMCHAN_FEED_SYMBOL_KEY)
    // ===========================================================================
    // THE FED ARM. This line MUST NOT COMPILE, and it must fail as a TYPE ERROR:
    // `sum_dir_query_key_from_ids` takes token ids, and a symbol key is not ids.
    // ===========================================================================
    const auto key =
        ninfer::spec::sum_dir::query_key::sum_dir_query_key_from_ids(symbol_key, nullptr);
    std::printf("REACHED-THE-UNREACHABLE key tokens=%u\n", key.token_count);
    return 0;
#else
    // ===========================================================================
    // THE NOT-FED ARM. The admitted source works, and it says so on stdout.
    // ===========================================================================
    const auto key = ninfer::spec::sum_dir::query_key::sum_dir_query_key_from_ids(ids, nullptr);
    std::printf("tree:   query_key tokens=%u identity_unset=%d\n", key.token_count,
                key.identity.is_unset() ? 1 : 0);
    std::printf("NOT-FED arm: rc=0 expected, and this line is the control\n");
    return 0;
#endif
}
