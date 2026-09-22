#pragma once

#include "ninfer/types.h"
#include "product/kv_storage_dtype.h"

#include <array>
#include <cstdint>
#include <iomanip>
#include <ostream>
#include <sstream>
#include <string>
#include <string_view>

namespace ninfer::product {

// The KV summary vocabulary: how a RESOLVED KV store is spelled, and how a byte count is
// spelled, in the `summary` block a front end prints.
//
// Why this is a shared header and not a local helper in each app: `ninfer` and
// `ninfer-perplexity` are two observations of the SAME run, and the whole point of the
// byte column is that ONE regex reads it out of either app's stderr. Two implementations
// of "how many bytes is this" or "which tiers is this table" would make that regex
// silently app-specific, which is the shape this project keeps paying for (a rule that
// lands in one front end and misses the others).
//
// THE NAME IS NOT SPELLED HERE. This header calls `kv_operator_token` -- the single
// canonical lookup for KvCacheStorage, whose table and totality are pinned by
// tests/test_kv_operator_name.cpp. An earlier draft of this header carried its OWN
// switch with the pre-rename spellings and arms for only the first eight enumerators;
// that is a second, drifted copy of the vocabulary, and it printed "unknown" for the
// enumerators the enum names (E8K3Group64 / E8K2Group64 today, seven in all). The
// refusal is inherited as well: a storage code no enumerator names throws the canonical
// lookup's std::invalid_argument instead of printing something that looks like a tier.
//
// ⚠️ KNOWN, DELIBERATE DUPLICATION, and why it is not fixed in this patch:
// apps/cli/main.cpp still carries its OWN `format_kv_cache` / `format_kv_layer_store` /
// `format_bytes`. Those definitions are NOT in HEAD -- `git show HEAD:apps/cli/main.cpp`
// is 388 lines against 660 in the worktree, and the `KvCacheStorage::Dropped` arm,
// `format_kv_layer_store` and the `print_metric("kv cache dtype", ...)` call site are all
// absent from HEAD, i.e. they are ANOTHER LINE'S UNCOMMITTED WORK (the L26 / patch-C
// reflection). Deleting them from a patch whose subject is "the scoring front end needs a
// byte column" is precisely the §12.32 failure this project has already hit three times:
// `git apply --check` returns 0 while another line's landing is silently removed. So this
// patch touches apps/cli/main.cpp NOT AT ALL.
// ⇒ FOLLOW-UP (must be dispatched after that work is committed, not by this patch): delete
// the three local helpers from apps/cli/main.cpp, replace them with
// `using ninfer::product::...` + the one-line format_bytes forwarder, and add a contract
// test that the two spellings of the resolved table agree over a synthesized MemorySummary
// (the `Dropped` arm included) -- the test is what keeps the single-regex premise true.
//
// The number this formats is always the ENGINE's own reflection (MemorySummary), never a
// re-derivation from the flags: --kv-layer-storage / --kv-bit-budget / --kv-bits all
// resolve into one per-layer table in the planner, and only the engine knows which one
// won. A front end that formatted its own argv would report the request, not the run.

inline std::string format_kv_bytes(std::uint64_t bytes) {
    constexpr double kKiB = 1024.0;
    constexpr double kMiB = 1024.0 * kKiB;
    constexpr double kGiB = 1024.0 * kMiB;
    std::ostringstream output;
    output << std::fixed << std::setprecision(2);
    if (bytes >= static_cast<std::uint64_t>(kGiB)) {
        output << static_cast<double>(bytes) / kGiB << " GiB";
    } else if (bytes >= static_cast<std::uint64_t>(kMiB)) {
        output << static_cast<double>(bytes) / kMiB << " MiB";
    } else if (bytes >= static_cast<std::uint64_t>(kKiB)) {
        output << static_cast<double>(bytes) / kKiB << " KiB";
    } else {
        output << bytes << " B";
    }
    return output.str();
}

// DERIVED, not copied: the canonical token of the one table the engine has for this enum.
// The name is deliberately the same one apps/cli/main.cpp routes its operator line
// through, so a reader of either front end sees the same word for the same state, and
// `Dropped` reaches the resolved table as the word "dropped" instead of reading as the
// view's default BF16 (patch C's D5).
inline std::string format_kv_cache(KvCacheStorage storage) {
    return kv_operator_token(storage);
}

// The RESOLVED per-layer KV store, compressed into
// "0,1,3,4,6,7:rk4v4-g64 2,5,8-15:nvfp4-g16" (canonical tokens).
// One tier across every full-attention layer keeps the old single-name form, so a
// uniform run reads exactly as it did.
inline std::string format_kv_layer_store(const MemorySummary& memory) {
    const std::uint32_t layers =
        memory.kv_full_attention_layers < memory.kv_layer_storage.size()
            ? memory.kv_full_attention_layers
            : static_cast<std::uint32_t>(memory.kv_layer_storage.size());
    if (layers == 0) { return format_kv_cache(memory.kv_cache); }
    std::ostringstream runs;
    bool uniform = true;
    for (std::uint32_t first = 0; first < layers;) {
        std::uint32_t last = first;
        while (last + 1 < layers &&
               memory.kv_layer_storage[last + 1] == memory.kv_layer_storage[first]) {
            ++last;
        }
        if (first != 0) {
            uniform = false;
            runs << ' ';
        }
        runs << first;
        if (last != first) { runs << '-' << last; }
        runs << ':' << format_kv_cache(memory.kv_layer_storage[first]);
        first = last + 1;
    }
    if (uniform) { return format_kv_cache(memory.kv_layer_storage[0]); }
    return "per-layer " + runs.str() + " (" + std::to_string(layers) +
           " full-attention layers)";
}

// A set of layer indices as the same compact run syntax the table above uses: "0-2 5 9-11",
// or "-" for the empty set. THE single implementation for every set a record wants to name
// (the discarded layers, the requested residual layers), so a reader can compare two of
// them with one parser.
inline std::string format_layer_set(const std::array<bool, kKvLayerStorageSlots>& table,
                                    std::uint32_t layers) {
    std::ostringstream runs;
    bool any = false;
    for (std::uint32_t first = 0; first < layers;) {
        if (!table[first]) {
            ++first;
            continue;
        }
        std::uint32_t last = first;
        while (last + 1 < layers && table[last + 1]) { ++last; }
        if (any) { runs << ' '; }
        any = true;
        runs << (last == first ? std::to_string(first)
                               : std::to_string(first) + "-" + std::to_string(last));
        first = last + 1;
    }
    return any ? runs.str() : std::string("-");
}

// The layers the planner RESOLVED as discarded (patch C reflects a dropped layer as
// KvCacheStorage::Dropped in the per-layer table). Computed HERE, from the engine's own
// table, and never from NINFER_KV_DROP_LAYERS: copying the environment string would
// report the request, and the env parser has already been caught reading "0-15" as the
// single layer 0 and "abc" as layer 0 (l26b D1/D2), so the string is the one thing a
// record must not trust.
inline std::string format_kv_dropped_layers(const MemorySummary& memory) {
    const std::uint32_t layers =
        memory.kv_full_attention_layers < memory.kv_layer_storage.size()
            ? memory.kv_full_attention_layers
            : static_cast<std::uint32_t>(memory.kv_layer_storage.size());
    std::array<bool, kKvLayerStorageSlots> dropped{};
    for (std::uint32_t layer = 0; layer < layers; ++layer) {
        dropped[layer] = memory.kv_layer_storage[layer] == KvCacheStorage::Dropped;
    }
    return format_layer_set(dropped, layers);
}

// A `summary` line. Both front ends must produce these bytes identically -- the columns
// above are only useful if one regex reads them out of either app -- so the writer lives
// here next to the formatters instead of once per app.
inline void print_summary_metric(std::ostream& out, std::string_view label,
                                 std::string_view value) {
    out << std::left << std::setw(12) << "summary" << std::setw(26) << label << value << '\n';
}

} // namespace ninfer::product
