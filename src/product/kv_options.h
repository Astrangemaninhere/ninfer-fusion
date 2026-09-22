#pragma once

#include "ninfer/types.h"

#include <array>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>

namespace ninfer::product {

// Per-layer KV storage spec parsing for CLI and serving.
//
// Spec grammar (comma separated):
//   all:<type>     every registered full-attention layer
//   <type>         shorthand for all:<type>
//   A:<type>       one layer, A in [0, 63]
//   A-B:<type>     inclusive layer range A..B, A <= B in [0, 63]
//
// CORRECTION (l26c): the bound is the FAMILY-WIDE slot count kKvLayerStorageSlots
// (64), not [0, 15]. 15 is only the smallest member's full-attention layer count;
// muse_glimmer_30b has 52, so its per-layer table legitimately names slots above
// 15 and a parser that rejected them would make that target's table unwritable.
// Slots at or past the ACTIVE target's full-attention count are accepted and
// inert for that model -- layouts_impl.h reports them ("[kv] --kv-layer-storage
// names slot(s) ... have NO effect"). The old text here claimed [0, 15].
// where <type> is bf16, int8, fp8, nvfp4, iso4e, rk4v4, rk3v4 or rk2v4. Unlisted slots stay
// BFloat16, which means "inherit the global --kv-dtype". A slot may be written
// exactly once. Cold-policy parsing lives with the cold-pool change, not here.
//
// "Was this slot written?" is carried by KvLayerStorageSpec::set, because
// BFloat16 doubles as the "unset" sentinel: without the mask an explicit
// `all:bf16` is indistinguishable from an empty spec and the layers keep
// inheriting the global dtype. With it, `0-11:bf16` IS a per-layer BF16 baseline
// (kv_resolve_slot_dtype, product/kv_component_switch.h), and a slot written
// twice is DETECTED instead of silently won by the second write.
// `--kv-dtype bf16` still forces a BF16 baseline for the WHOLE stack and stays
// the cheaper spelling when every layer is BF16.

[[nodiscard]] inline std::optional<KvCacheStorage> parse_kv_storage(std::string_view text) {
    if (text == "bf16") { return KvCacheStorage::BFloat16; }
    if (text == "int8") { return KvCacheStorage::Int8Group64; }
    if (text == "nvfp4") { return KvCacheStorage::Nvfp4Group16; }
    if (text == "fp8") { return KvCacheStorage::Fp8Group16; }
    if (text == "iso4e") { return KvCacheStorage::Iso3Group16; }
    if (text == "rk4v4") { return KvCacheStorage::E8Group64; }
    // rk4v4 family, K plane at 3 / 2 bits. SELECTABLE HERE (the vocabulary, the ladder cost
    // and the plane geometry all exist and are pinned) but NOT RUNNABLE: no decode or
    // append kernel reads a 3-bit or 2-bit K code plate in this tree, so a plan that
    // resolves to one of these is refused by name at the resolver rather than run through
    // the 4-bit reader, which would misread every row. See product/kv_e8_width.h.
    if (text == "rk3v4") { return KvCacheStorage::E8K3Group64; }
    if (text == "rk2v4") { return KvCacheStorage::E8K2Group64; }
    return std::nullopt;
}

// A parsed per-layer table PLUS the mask of which slots the spec actually wrote.
// Both halves are needed: `table` alone cannot distinguish "written bf16" from
// "never mentioned", and the planner resolves a slot differently in those two
// cases (kv_resolve_slot_dtype).
struct KvLayerStorageSpec {
    std::array<KvCacheStorage, kKvLayerStorageSlots> table{};
    std::array<bool, kKvLayerStorageSlots> set{};
};

[[nodiscard]] inline KvLayerStorageSpec parse_kv_layer_storage_spec(std::string_view spec) {
    KvLayerStorageSpec out;
    if (spec.empty()) { return out; }

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
        std::string where = "all";
        if (colon != std::string_view::npos) {
            const std::string_view layers = item.substr(0, colon);
            type = item.substr(colon + 1);
            where = std::string(layers);
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
        if (!value) {
            // Name the offending text and the layers it was meant for. The old
            // text ("has an invalid type") left the operator to guess which of
            // five comma-separated entries was the typo.
            throw std::invalid_argument(
                "kv-layer-storage: unknown dtype '" + std::string(type) + "' for layers '" +
                where + "' (want bf16, int8, fp8, nvfp4, iso4e, rk4v4, rk3v4 or rk2v4)");
        }
        for (std::size_t slot = first; slot <= last; ++slot) {
            // The MASK, not the table, decides "written twice": the old table
            // check (`table[slot] != BFloat16`) could not see a first write of
            // bf16, so "0:bf16,0:int8" was silently accepted and int8 won.
            if (out.set[slot]) {
                throw std::invalid_argument(
                    "kv-layer-storage slot " + std::to_string(slot) +
                    " written twice (layer " + std::to_string(slot) +
                    " appears in more than one entry of '" + std::string(spec) + "')");
            }
            out.set[slot] = true;
            out.table[slot] = *value;
        }
        if (comma == std::string_view::npos) { break; }
        begin = comma + 1;
    }
    return out;
}

// Table-only view, for callers that need nothing but the storages. NOTE: this
// DROPS the mask, so a spec's explicit `bf16` and a slot the spec never mentioned
// are indistinguishable in the result. Prefer parse_kv_layer_storage_spec()
// wherever the RESOLVED dtype of a slot matters (the planner, /reload_kv).
[[nodiscard]] inline std::array<KvCacheStorage, kKvLayerStorageSlots>
parse_kv_layer_storage(std::string_view spec) {
    return parse_kv_layer_storage_spec(spec).table;
}

// --kv-residual-layers SPEC: the per-layer NVFP4 SECOND-STAGE RESIDUAL planes.
//
// Grammar: a BARE layer list, comma separated, single indices and inclusive ranges
// ("0,3,7", "2-5"). It is deliberately the same token grammar NINFER_KV_DROP_LAYERS
// uses (targets/qwen3_6/impl/state/decoder_state.cpp parse_kv_layer_drop), because
// these two are the only flags in the tree that name a set of FULL-ATTENTION layers
// outright, and an operator who has learned one has learned both. It is NOT the
// kv-layer-storage grammar above: there is no `:<type>`, and a repeat is a no-op
// rather than the "written twice" error that grammar raises.
//
// STRICTER THAN THE SERVE PARSER IT REPLACES, in exactly two spellings: a token with
// leading whitespace and a token with a leading `+`. The old loop went through
// strtoull (via parse_u64), which skips leading whitespace and accepts a leading `+`;
// this one is pure decimal, which is what the env parser already requires. A spelling
// that is a typo in one of the two and accepted by the other is the failure mode this
// whole work package is about, so the stricter of the two grammars is the one that
// survives, and the loose pair is named here rather than left to be discovered.
//
// Bound: the FAMILY-WIDE slot count kKvLayerStorageSlots (64), for the same reason
// parse_kv_layer_storage_spec uses it -- the table covers the whole family, and the
// smallest member's 16 full-attention layers must not make a larger member's table
// unwritable. Whether an index is in range for THIS cache is decided where the count
// lives (decoder_state.cpp, plan time), and so is a token that is not a pure decimal.
//
// Before this parser existed the flag was reachable from ninfer-serve ONLY
// (src/serve/serve_options.cpp), so the one handle in the tree that expresses a real
// plane SUBSET -- an NVFP4 layer going from 4 planes to 8 -- could not be driven from
// either app that can be pointed at a corpus or a prompt.
//
// It is still a measurement instrument, not a product knob: the extra planes are
// NVFP4-only (naming any other tier is accepted and inert, decoder_state.cpp gates on
// `selected == DType::NVFP4`), and a residual-bearing layer is NOT cold-capable --
// impl/runtime/program_impl.h abandons the WHOLE cold pass for it -- so this flag buys
// volume at the price of the layer's admissibility, a trade no ppl number shows.
struct KvResidualLayersSpec {
    std::array<bool, kKvLayerStorageSlots> table{};
    std::uint32_t count = 0;
};

// One layer index out of a residual spec token. Pure decimal, non-empty, in
// [0, kKvLayerStorageSlots). Error texts are the ones the serve front end has always
// emitted for this flag, so a caller that switched to this parser cannot tell.
[[nodiscard]] inline std::size_t parse_kv_residual_layer_index(std::string_view token) {
    std::uint64_t value = 0;
    const auto parsed =
        std::from_chars(token.data(), token.data() + token.size(), value);
    if (token.empty() || parsed.ec != std::errc{} ||
        parsed.ptr != token.data() + token.size()) {
        throw std::invalid_argument("invalid kv-residual-layers: " + std::string(token));
    }
    if (value >= static_cast<std::uint64_t>(kKvLayerStorageSlots)) {
        throw std::invalid_argument("--kv-residual-layers layer out of range");
    }
    return static_cast<std::size_t>(value);
}

[[nodiscard]] inline KvResidualLayersSpec
parse_kv_residual_layers_spec(std::string_view spec) {
    KvResidualLayersSpec out;
    std::size_t begin = 0;
    while (begin <= spec.size()) {
        const std::size_t comma = spec.find(',', begin);
        const std::string_view token =
            spec.substr(begin, comma == std::string_view::npos ? spec.size() - begin
                                                               : comma - begin);
        // An empty entry is skipped, not an error, exactly as the env parser reads
        // ",,". A stricter reading here would turn a spelling that has always meant
        // "no extra planes" into a refusal.
        if (!token.empty()) {
            const std::size_t dash = token.find('-');
            // "A-B" is a RANGE. The env parser's D1 lesson is that strtol read
            // "0-15" as the single layer 0, i.e. as a silent no-op; the same shape
            // is why the dash is looked for before any number is read.
            const std::size_t first = parse_kv_residual_layer_index(
                dash == std::string_view::npos ? token : token.substr(0, dash));
            const std::size_t last = parse_kv_residual_layer_index(
                dash == std::string_view::npos ? token : token.substr(dash + 1));
            // A reversed range ("5-3") matches the serve parser's behaviour: it
            // selects nothing. That is a silent no-op, and it is NOT fixed here --
            // this patch moves three callers onto one parser without changing what
            // the parser accepts. Tightening it is a separate, one-line change with
            // its own negative control.
            for (std::size_t layer = first; layer <= last; ++layer) {
                if (!out.table[layer]) {
                    out.table[layer] = true;
                    ++out.count;
                }
            }
        }
        if (comma == std::string_view::npos) { break; }
        begin = comma + 1;
    }
    return out;
}

} // namespace ninfer::product
