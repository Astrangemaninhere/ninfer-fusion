#pragma once

#include "ninfer/types.h"

#include <array>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

namespace ninfer::product {

// Per-layer KV storage spec parsing for CLI and serving.
//
// Spec grammar (comma separated):
//   all:<type>     every registered full-attention layer
//   <type>         shorthand for all:<type>
//   A:<type>       one layer, A in [0, 15]
//   A-B:<type>     inclusive layer range A..B, A <= B in [0, 15]
// where <type> is bf16, int8, fp8, or nvfp4. Unlisted slots stay BFloat16,
// which means "inherit the global --kv-dtype". A slot may be written exactly
// once. Cold-policy parsing lives with the cold-pool change, not here.

[[nodiscard]] inline std::optional<KvCacheStorage> parse_kv_storage(std::string_view text) {
    if (text == "bf16") { return KvCacheStorage::BFloat16; }
    if (text == "int8") { return KvCacheStorage::Int8Group64; }
    if (text == "nvfp4") { return KvCacheStorage::Nvfp4Group16; }
    if (text == "fp8") { return KvCacheStorage::Fp8Group16; }
    if (text == "iso3") { return KvCacheStorage::Iso3Group16; }
    return std::nullopt;
}

[[nodiscard]] inline std::array<KvCacheStorage, kKvLayerStorageSlots>
parse_kv_layer_storage(std::string_view spec) {
    std::array<KvCacheStorage, kKvLayerStorageSlots> table{};
    if (spec.empty()) { return table; }

    std::size_t begin = 0;
    while (begin < spec.size()) {
        const std::size_t comma = spec.find(',', begin);
        const std::string_view item =
            spec.substr(begin, comma == std::string_view::npos ? spec.size() - begin
                                                               : comma - begin);
        if (item.empty()) { throw std::invalid_argument("kv-layer-storage has an empty entry"); }

        const std::size_t colon = item.find(':');
        std::size_t first = 0;
        std::size_t last = kKvLayerStorageSlots - 1;
        std::string_view type = item;
        if (colon != std::string_view::npos) {
            const std::string_view layers = item.substr(0, colon);
            type = item.substr(colon + 1);
            if (layers == "all") {
                first = 0;
                last = kKvLayerStorageSlots - 1;
            } else {
                const std::size_t dash = layers.find('-');
                if (dash == std::string_view::npos) {
                    first = last =
                        static_cast<std::size_t>(std::stoul(std::string(layers)));
                } else {
                    first = static_cast<std::size_t>(std::stoul(std::string(
                        layers.substr(0, dash))));
                    last = static_cast<std::size_t>(std::stoul(std::string(
                        layers.substr(dash + 1))));
                }
                if (first > last || last >= kKvLayerStorageSlots) {
                    throw std::invalid_argument("kv-layer-storage layer index out of range");
                }
            }
        }
        const auto value = parse_kv_storage(type);
        if (!value) { throw std::invalid_argument("kv-layer-storage has an invalid type"); }
        for (std::size_t slot = first; slot <= last; ++slot) {
            if (table[slot] != KvCacheStorage::BFloat16) {
                throw std::invalid_argument("kv-layer-storage slot written twice");
            }
            table[slot] = *value;
        }
        if (comma == std::string_view::npos) { break; }
        begin = comma + 1;
    }
    return table;
}

} // namespace ninfer::product
