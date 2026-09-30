#pragma once

// ninfer::ops::detail -- THE SMALL-T DECODE GEOMETRY DECISION, BY IDENTITY.
//
// WHY THIS FILE EXISTS (line `musefix`, marker F897; framework: dl/xfer F-879's (A)/(B)/(C)).
//
// The question this file answers is "which registered GqaGeometry is THIS launch?" and the defect
// it closes is that the small-T DECODE dispatcher was answering it with a BARE Q-HEAD COUNT.
//
// (A) A GENUINELY MODEL-SPECIFIC FACT, AND IT BELONGS IN THE MODEL'S OWN DECLARATION. Each target
//     already declares its own geometry in its own `impl/config.h` -- `query_heads`, `kv_heads`,
//     `head_dim` (muse_glimmer_30b: 32/2/128, spark_x2_5_4b: 16/4/256, qwen3_6_27b: 24/4/256,
//     qwen3_6_35b_a3b: 16/2/256). Nothing here re-states those numbers: the table below reads
//     them OFF the GqaGeometry instantiations, which are the ops-level spelling of the same facts.
//
// (B) A MODEL-SHAPED INTERFACE WITH NO SEAM -- THE ACTUAL DEFECT, MEASURED. `(q_heads, head_dim)` is
//     NOT an identity: it is many-to-one. The tree's own header says so in writing
//     (ops/launcher/gqa_attention_route_contract.h:61-67): "Gqa16x4Geometry (Spark-X2.5, 16
//     q-heads / 4 kv-heads) is covered by `q_heads != 16`: the small-T launcher dispatches 16
//     q-heads at head_dim 256 to the Gqa35Geometry (16/2) instance and HAS NO KV-HEAD SIGNATURE TO
//     SEPARATE THE TWO". And ops/wrapper/gqa_attention.cpp:528-534 states the same collision from
//     the sizer: "(16 q-heads, head_dim 256) is both 35B (16/2) and Spark-X2.5 (16/4)".
//     MEASURED CONSEQUENCE, from the source and not from a claim: `Gqa16x4Geometry` appears as a
//     `_launch_for<...>` template argument in exactly ONE launcher family (the PREFILL,
//     gqa_attention_prefill.cu:540 and :647) and in NO small-T decode tier, and
//     `gqa_attention_cached_small_t_launch` ended with an UNTESTED fall-through
//     (`gqa_attention_small_t_launch_for<Gqa35Geometry>(...)`, no test, no refusal) where its
//     append sibling threw by name. So a 16 q-head / 4 kv-head cache on this tier was handed to a
//     16 q-head / 2 kv-head kernel silently.
//
// (C) A SEAM THAT ALREADY EXISTS AND IS ALREADY DRIVEN -- SOMEWHERE ELSE. The kv-head signature is
//     not new and it is not an invention here: `cache.num_kv_heads == Gqa16x4Geometry::KVHeads` is
//     ALREADY read by the E8 decode dispatcher (ops/launcher/gqa_attention_decode_e8.cu:296 and
//     :327) and by the prefill dispatcher (ops/launcher/gqa_attention_prefill.cu:644, and :652 for
//     the 35B form). What was missing is that the small-T decode dispatcher -- the tier a bf16 /
//     fp8 / iso3 / nvfp4 KV cache takes on every decode round -- did not read it. This file is
//     that one reader, and the two dispatch sites forward to it, so the rule is spelled ONCE and a
//     host test can drive the dispatcher's own decision with no GPU and no ops library, which is
//     the same contract ops/launcher/gqa_attention_route_contract.h already keeps for the route.
//
// HOW A NEW MODEL ADAPTS, WITHOUT EDITING SHARED CODE. Its own `impl/config.h` already declares the
// triple. If a `GqaGeometry` instantiation exists for that triple, its row is added HERE and in the
// per-tier instantiation lists -- that is the whole path, and the model's name never appears in a
// call chain. If no instantiation exists, the launch REFUSES BY NAME and names the missing kernels,
// which is strictly better than running a neighbouring geometry and printing nothing.
//
// NAMED ABSENCE, WITH ITS POPULATION, NOT A SHRUG. `Gqa16x4Geometry` (16/4/256, Spark-X2.5-4B) has
// no small-T decode instantiation in this tree (measured: `_launch_for<Gqa16x4Geometry>` occurs only
// in gqa_attention_prefill.cu). Adding the row here WITHOUT that instantiation would be an
// undefined-symbol link error, so the row is deliberately NOT in the small-T table and the refusal
// below says exactly which kernels a 16/4 small-T decode still needs. The population is
// spark_x2_5_4b on a bf16 / fp8_e4m3fn / iso3 / nvfp4 KV tier.

#include "ops/kernel/gqa_attention_geometry.cuh"  // Gqa27Geometry, GqaMuseGeometry, Gqa35Geometry

#include <cstdint>
#include <string>

namespace ninfer::ops::detail {

enum class GqaGeometryRoute : std::uint8_t {
    Gqa27 = 0,
    GqaMuse,
    Gqa35,
    Unregistered,
};

struct GqaGeometryRow {
    GqaGeometryRoute route;
    std::int32_t q_heads;
    std::int32_t kv_heads;
    std::int32_t head_dim;
    const char* name;
};

// ONE ROW PER SMALL-T DECODE INSTANTIATION. The numbers are read off the instantiation, so this
// table cannot disagree with the kernels it names.
inline constexpr GqaGeometryRow kGqaSmallTGeometryTable[] = {
    {GqaGeometryRoute::Gqa27, Gqa27Geometry::QHeads, Gqa27Geometry::KVHeads, Gqa27Geometry::HeadDim,
     "qwen3_6_27b (24 q / 4 kv / 256)"},
    {GqaGeometryRoute::GqaMuse, GqaMuseGeometry::QHeads, GqaMuseGeometry::KVHeads,
     GqaMuseGeometry::HeadDim, "muse_glimmer_30b (32 q / 2 kv / 128)"},
    {GqaGeometryRoute::Gqa35, Gqa35Geometry::QHeads, Gqa35Geometry::KVHeads, Gqa35Geometry::HeadDim,
     "qwen3_6_35b_a3b (16 q / 2 kv / 256)"},
};

// THE ONE READER. The identity is the TRIPLE; a bare head count is not enough to select a kernel,
// because (16, 256) is both 35B and Spark-X2.5.
[[nodiscard]] inline GqaGeometryRoute gqa_small_t_geometry_route(std::int32_t q_heads,
                                                                 std::int32_t kv_heads,
                                                                 std::int32_t head_dim) noexcept {
    for (const GqaGeometryRow& row : kGqaSmallTGeometryTable) {
        if (row.q_heads == q_heads && row.kv_heads == kv_heads && row.head_dim == head_dim) {
            return row.route;
        }
    }
    return GqaGeometryRoute::Unregistered;
}

[[nodiscard]] inline const char* gqa_small_t_geometry_route_name(GqaGeometryRoute route) noexcept {
    switch (route) {
    case GqaGeometryRoute::Gqa27:
        return "Gqa27Geometry";
    case GqaGeometryRoute::GqaMuse:
        return "GqaMuseGeometry";
    case GqaGeometryRoute::Gqa35:
        return "Gqa35Geometry";
    case GqaGeometryRoute::Unregistered:
        break;
    }
    return "unregistered";
}

// "24/4/256, 32/2/128, 16/2/256" -- the registered triples, built from the table above so a
// refusal can never list a geometry the dispatcher cannot actually launch.
[[nodiscard]] inline std::string gqa_small_t_registered_geometry_list() {
    std::string out;
    for (const GqaGeometryRow& row : kGqaSmallTGeometryTable) {
        if (!out.empty()) { out += ", "; }
        out += std::to_string(row.q_heads) + "/" + std::to_string(row.kv_heads) + "/" +
               std::to_string(row.head_dim);
    }
    return out;
}

// THE REFUSAL, BY NAME, WITH THE NUMBERS AND WITH THE MISSING KERNELS. It replaces (a) a message
// that named only q-heads, and (b) in the cached path, no message at all.
[[nodiscard]] inline std::string gqa_small_t_geometry_refusal(const char* op, std::int32_t q_heads,
                                                             std::int32_t kv_heads,
                                                             std::int32_t head_dim) {
    std::string out = std::string(op) + ": no registered small-T decode geometry for this cache (q=" +
                      std::to_string(q_heads) + " kv=" + std::to_string(kv_heads) +
                      " head_dim=" + std::to_string(head_dim) +
                      "); registered triples: " + gqa_small_t_registered_geometry_list() +
                      ". The identity is the TRIPLE: (q_heads, head_dim) alone is many-to-one -- "
                      "(16 q, 256 d) is both qwen3_6_35b_a3b (16/2) and spark_x2_5_4b (16/4) -- so a "
                      "cache whose kv-head count matches no row is REFUSED rather than handed to a "
                      "neighbouring geometry. If this cache is 16 q / 4 kv / 256, the geometry "
                      "exists (Gqa16x4Geometry) and only the small-T decode instantiation is "
                      "missing: add gqa_attention_small_t_launch_for<Gqa16x4Geometry> to the tier "
                      "TUs (the prefill family already carries it).";
    return out;
}

} // namespace ninfer::ops::detail
