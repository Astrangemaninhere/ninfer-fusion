#include "targets/qwen3_6/impl/runtime/instance.h"
#include "targets/qwen3_6/impl/runtime/layouts.h"
#include "targets/qwen3_6/impl/runtime/vision_context.h"
#include "targets/qwen3_6/impl/runtime/workspace_recipe.h"

#include "core/device.h"
#include "ninfer/ops/gated_delta_net.h"
#include "ninfer/ops/gdn_gating_proj.h"
#include "ninfer/ops/gdn_input_proj.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/ops/sliding_window_attention.h"
#include "ninfer/ops/softmax_attention.h"
#include "ninfer/ops/speculative_round.h"
#include "product/kv_bit_budget.h"
#include "product/kv_cold_tier_budget.h"
#include "product/kv_options.h"
#include "product/kv_tier_formats.h"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS {
namespace {

constexpr std::size_t kMiB        = 1024ULL * 1024ULL;
constexpr std::size_t kArenaAlign = 256ULL;

enum class GdnWorkspacePath : std::uint8_t {
    Prefill,
    Snapshot,
    ReplayRecord,
};

std::size_t checked_add(std::size_t a, std::size_t b, const char* label) {
    if (b > std::numeric_limits<std::size_t>::max() - a) { throw std::overflow_error(label); }
    return a + b;
}

std::size_t checked_mul(std::size_t a, std::size_t b, const char* label) {
    if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b) {
        throw std::overflow_error(label);
    }
    return a * b;
}

std::int32_t checked_i32(std::uint64_t value, const char* label) {
    if (value == 0 ||
        value > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error(label);
    }
    return static_cast<std::int32_t>(value);
}

std::uint32_t page_count(std::uint32_t capacity) {
    if (capacity == 0) { throw std::invalid_argument("Paged KV capacity must be positive"); }
    return 1U + (capacity - 1U) / static_cast<std::uint32_t>(kPagedKVPageSize);
}

struct TargetKVCacheProfile {
    DType dtype;
    std::int32_t quant_group;
};

TargetKVCacheProfile target_kv_cache_profile(KvCacheStorage storage) {
    switch (storage) {
    case KvCacheStorage::BFloat16:
        return {DType::BF16, 0};
    case KvCacheStorage::Int8Group64:
        return {DType::I8, qwen3_6::kKvInt8QuantGroup};
    case KvCacheStorage::Fp8E4M3Row256:
        return {DType::FP8_E4M3FN, qwen3_6::kKvFp8QuantGroup};
    case KvCacheStorage::Nvfp4Group16:
        return {DType::NVFP4, qwen3_6::kNvfp4KvQuantGroup};
    // ISO3 is a tier of its own: identical planes to NVFP4 but a distinct
    // 3-bit sign-magnitude codec, so the resolved dtype must stay ISO3 and the
    // K/V decode kernel is the ISO3 one (gqa_attention_decode_iso3.cuh).
    case KvCacheStorage::Iso3Group16:
        return {DType::ISO3, qwen3_6::kNvfp4KvQuantGroup};
    case KvCacheStorage::E8Group64:
        return {DType::E8Kv, qwen3_6::kKvInt8QuantGroup};
    }
    throw std::invalid_argument("unknown KV-cache storage profile");
}

template <class ProfileAllowance>
std::size_t graph_topology_allowance(const std::vector<GraphExecutionProfile>& profiles,
                                     ProfileAllowance&& profile_allowance, const char* label) {
    std::vector<std::pair<std::uint32_t, std::size_t>> classes;
    for (const GraphExecutionProfile profile : profiles) {
        const std::size_t allowance = profile_allowance(profile);
        const auto existing = std::find_if(classes.begin(), classes.end(), [&](const auto& entry) {
            return entry.first == profile.topology_class;
        });
        if (existing == classes.end()) {
            classes.emplace_back(profile.topology_class, allowance);
        } else {
            existing->second = std::max(existing->second, allowance);
        }
    }

    std::size_t total = 0;
    for (const auto& [topology_class, allowance] : classes) {
        (void)topology_class;
        total = checked_add(total, allowance, label);
    }
    return total;
}

TensorLayout add_tensor(LayoutBuilder& builder, DType dtype,
                        std::initializer_list<std::int32_t> shape, const char* label) {
    return builder.add_tensor(dtype, shape, kArenaAlign, label);
}

// S30: single source of truth for the effective cold-pool size (S25 precedence).
// ColdPolicy::None always forces 0 (no cold path; slots would be dead device
// memory). ColdPolicy::Host keeps 0 as well, and for a better reason than "the
// consumer is missing": the Host tier's medium is pinned host memory, its pages
// are READ-FREE (no kernel ever reads them again), so it needs no device cold
// slot at all -- see cold_host_tier.h. Window/Disk take the explicit
// --max-cold-pages cap when non-zero, else the keep-tokens derivation. Used by
// BOTH the layout reservation and the --kv-bit-budget DP so the DP's cold
// capacity always equals the reserved pool.
[[nodiscard]] inline std::uint32_t effective_cold_pages(ColdPolicy policy,
                                                        std::uint32_t keep_tokens,
                                                        std::uint32_t explicit_pages) {
    // The device cold-slot pool is the *slot* tiers' resource. Under the Host
    // policy a non-zero --max-cold-pages would reserve device slots the Host tier
    // never uses (it may not: its pages are the ones nothing reads, so there is no
    // slot to read them from), and --cold-host-bytes is the knob that bounds that
    // tier. Asking for both is a contradiction; make it loud instead of quietly
    // reserving a pool that stays empty.
    if (policy == ColdPolicy::Host && explicit_pages != 0) {
        throw std::invalid_argument(
            "cold policy 'host' uses pinned host memory, not the device cold-slot pool, "
            "so --max-cold-pages cannot be honoured; bound the tier with --cold-host-bytes "
            "or use --cold-policy window|disk");
    }
    switch (policy) {
    case ColdPolicy::Window:
    case ColdPolicy::Disk:
        return explicit_pages != 0
                   ? explicit_pages
                   : static_cast<std::uint32_t>(keep_tokens / kPagedKVPageSize + 16);
    default:
        return 0;
    }
}

PersistentLayout persistent_layout(const SequencePlanImpl& plan) {
    if (!plan.context_cache.device_state_slots) {
        throw std::logic_error("Qwen3.6 context cache options are not normalized");
    }
    const std::int32_t state_image_slots = checked_i32(
        static_cast<std::uint64_t>(plan.max_concurrency) + *plan.context_cache.device_state_slots,
        "Qwen3.6 StateImage slot count exceeds int32");
    const auto effective_prefill_chunk =
        static_cast<std::int32_t>(std::min(plan.prefill_chunk, plan.capacity));
    const std::uint32_t logical_pages  = page_count(plan.capacity);
    const std::uint32_t physical_pages = plan.main_page_groups;
    const std::uint64_t mtp_extra_pages =
        plan.features.mtp()
            ? static_cast<std::uint64_t>(plan.max_concurrency) *
                  ((static_cast<std::uint64_t>(plan.draft_window - 1U) + kPagedKVPageSize - 1U) /
                   static_cast<std::uint32_t>(kPagedKVPageSize))
            : 0ULL;
    const std::uint32_t mtp_physical_pages = static_cast<std::uint32_t>(
        checked_i32(static_cast<std::uint64_t>(physical_pages) + mtp_extra_pages,
                    "MTP Paged KV physical pages exceed int32"));
    LayoutBuilder builder;
    PersistentLayout out;
        // E3/S24: per-layer SWA windows from the target config. Muse declares
        // sliding_window (2048) + is_swa_attention() (39 of its 52 layers); the qwen
        // family declares 0 + false, i.e. every layer keeps window 0 = full attention,
        // byte-identical to the behaviour before this wiring (kernels guard on > 0).
        std::array<std::uint32_t, 64> layer_windows{};
        for (std::int32_t i = 0;
             i < static_cast<std::int32_t>(TextConfig::full_attention_layers()); ++i) {
            layer_windows[static_cast<std::size_t>(i)] =
                TextConfig::is_swa_attention(i)
                    ? static_cast<std::uint32_t>(TextConfig::sliding_window)
                    : 0U;
        }
    out.decoder = qwen3_6::plan_decoder_state(
        builder, qwen3_6::DecoderStateSpec{
                     .full_attention_layers     = TextConfig::full_attention_layers(),
                     .mtp_layers                = TextConfig::mtp_layers,
                     .capacity                  = plan.capacity,
                     .kv_heads                  = TextConfig::kv_heads,
                     .attention_head_dim        = TextConfig::head_dim,
                     .kv_dtype                  = plan.kv_dtype,
                     .kv_quant_group            = plan.kv_quant_group,
                     .layer_kv_dtypes           = plan.layer_kv_dtypes,
                     .layer_residual            = plan.kv_residual_layers,
                     .layer_sliding_windows    = layer_windows,
                     // SEPARATION: the three KV component switches reach the
                     // decoder plan through here; plan_decoder_state() is the
                     // single place that commits them to the device (rotation
                     // gate, row-scale gate, V codec + its hard-fail checks).
                     .kv_v_codec                = plan.kv_v_codec,
                     .kv_rotation_off           = plan.kv_rotation_off,
                     .kv_row_scale_spec         = plan.kv_row_scale_spec,
                     .enable_mtp                = plan.features.mtp(),
                     .kv_table_rows             = static_cast<std::int32_t>(plan.max_concurrency + 1),
                     .text_physical_page_groups = physical_pages,
                     .mtp_physical_page_groups  = mtp_physical_pages,
                     // --max-cold-pages via effective_cold_pages (same S25
                     // precedence, now shared with the --kv-bit-budget DP so the
                     // DP's cold capacity always equals the pool reserved here:
                     // None->0 always; Host->0 because the Host tier lives in
                     // pinned host memory and holds read-free pages only (no
                     // device slot is involved, --cold-host-bytes is its cap);
                     // Window/Disk: explicit cap else keep-tokens derivation).
                     .max_cold_pages            = effective_cold_pages(
                         plan.cold_policy, plan.cold_keep_tokens, plan.max_cold_pages),
                 });
    qwen3_6::StateImageSpec state_image_spec{
        .linear =
            {
                // 纯 softmax 族 (gdn==0) 无线性状态: 池规划要求非零几何,
                // 用最小假层占位 (运行时永不触碰; qwen 家族不受影响).
                .layers         = std::max(1, TextConfig::gdn_layers()),
                .conv_channels  = std::max(1, TextConfig::convolution_dim),
                .conv_width     = std::max(1, TextConfig::gdn_conv_state_width),
                .value_heads    = std::max(1, TextConfig::gdn_value_heads),
                .value_head_dim = std::max(1, TextConfig::gdn_value_head_dim),
                .key_head_dim   = std::max(1, TextConfig::gdn_key_head_dim),
                .slot_count     = state_image_slots,
                .conv_dtype     = DType::BF16,
            },
        .hidden = TextConfig::hidden,
    };
    if constexpr (Variant::supports_dflash) {
        if (plan.features.dflash()) {
            state_image_spec.dflash_local = qwen3_6::DFlashLocalStateSpec{
                .layers   = DFlashConfig::local_layers,
                .capacity = DFlashConfig::local_capacity,
                .kv_heads = DFlashConfig::kv_heads,
                .head_dim = DFlashConfig::head_dim,
            };
        }
    }
    out.state_images = qwen3_6::plan_state_image_device_pool(builder, state_image_spec);
    if (plan.speculative_backend != SpeculativeBackend::None) {
        out.replay_records = plan_gdn_replay_records(
            builder, GdnReplayRecordSpec{
                         .layers          = TextConfig::gdn_layers(),
                         .record_capacity = static_cast<std::int32_t>(plan.max_concurrency),
                         .width           = static_cast<std::int32_t>(plan.draft_window + 1U),
                         .conv_channels   = TextConfig::convolution_dim,
                         .qk_heads        = TextConfig::gdn_key_heads,
                         .value_heads     = TextConfig::gdn_value_heads,
                         .key_dim         = TextConfig::gdn_key_head_dim,
                         .value_dim       = TextConfig::gdn_value_head_dim,
                     });
    }
    if constexpr (Variant::supports_dflash) {
        if (plan.features.dflash()) {
            DFlashPersistentLayout& dflash = out.dflash.emplace();
            // Every DSpark layer keeps its own BF16 context-K/V plane pair in
            // one shared page pool; attention selects the layer plane base.
            KVPageGeometry full_geometry{
                .page_tokens        = kPagedKVPageSize,
                .device_plane_order = PagedKVPlaneOrder::HeadMajor,
                .planes             = {},
            };
            for (int layer = 0; layer < DFlashConfig::full_layers; ++layer) {
                full_geometry.planes.push_back(
                    {DType::BF16, DFlashConfig::head_dim, DFlashConfig::kv_heads, 256});
                full_geometry.planes.push_back(
                    {DType::BF16, DFlashConfig::head_dim, DFlashConfig::kv_heads, 256});
            }
            qwen3_6::PagedKVCacheLayout full_layout{
                .pages = plan_device_kv_page_pool(
                    builder, DeviceKVPagePoolSpec{.page_group_count = physical_pages,
                                                  .geometry         = std::move(full_geometry)}),
                .execution_tables = plan_kv_execution_tables(
                    builder,
                    KVExecutionTableSpec{
                        .logical_page_capacity = logical_pages,
                        .table_rows            = static_cast<std::int32_t>(plan.max_concurrency + 1),
                    }),
                .layers      = static_cast<std::uint32_t>(DFlashConfig::full_layers),
                .max_context = plan.capacity,
                .kv_heads    = DFlashConfig::kv_heads,
                .head_dim    = DFlashConfig::head_dim,
                .dtype       = DType::BF16,
                .quant_group = 0,
            };
            for (int layer = 0; layer < DFlashConfig::full_layers; ++layer) {
                full_layout.layer_plane_base[static_cast<std::size_t>(layer)] =
                    static_cast<std::uint32_t>(2 * layer);
            }
            dflash.full = full_layout;
            dflash.prefill_features = add_tensor(
                builder, DType::BF16, {DFlashConfig::feature_rows, effective_prefill_chunk},
                "DFlash prefill target features");
            dflash.prefill_positions = add_tensor(builder, DType::I32, {effective_prefill_chunk},
                                                  "DFlash prefill target positions");
            dflash.pending_features  = add_tensor(builder, DType::BF16,
                                                  {DFlashConfig::feature_rows,
                                                   static_cast<std::int32_t>(plan.draft_window + 1U),
                                                   static_cast<std::int32_t>(plan.max_concurrency)},
                                                  "DFlash pending target features");
        }
    }
    if constexpr (Variant::supports_dflash2) {
        if (plan.features.dflash2()) {
            DFlash2PersistentLayout& dflash2 = out.dflash2.emplace();
            dflash2.local = plan_cyclic_kv_cache(builder, DFlash2Config::local_layers,
                                                 DFlash2Config::local_capacity,
                                                 DFlash2Config::kv_heads, DFlash2Config::head_dim,
                                                 static_cast<std::int32_t>(plan.max_concurrency));
            dflash2.rewrite_checkpoint_local = plan_cyclic_kv_cache(
                builder, DFlash2Config::local_layers, DFlash2Config::local_capacity,
                DFlash2Config::kv_heads, DFlash2Config::head_dim,
                static_cast<std::int32_t>(plan.max_concurrency));
            dflash2.prefill_features = add_tensor(
                builder, DType::BF16, {DFlash2Config::feature_rows, effective_prefill_chunk},
                "DFlash2 prefill target features");
            dflash2.prefill_positions = add_tensor(builder, DType::I32, {effective_prefill_chunk},
                                                   "DFlash2 prefill target positions");
            dflash2.pending_features = add_tensor(
                builder, DType::BF16,
                {DFlash2Config::feature_rows,
                 static_cast<std::int32_t>(plan.draft_window + 1U),
                 static_cast<std::int32_t>(plan.max_concurrency)},
                "DFlash2 pending target features");
        }
    }

    out.round = qwen3_6::begin_round_state_layout(
        builder, qwen3_6::RoundStateSpec{.hidden         = TextConfig::hidden,
                                         .output_rows    = TextConfig::output_rows,
                                         .batch_capacity = plan.max_concurrency,
                                         .draft_window   = plan.draft_window,
                                         .enable_mtp     = plan.features.mtp(),
                                         .enable_dflash  = plan.features.dflash_like()});
    out.prefill_hidden = add_tensor(
        builder, DType::BF16, {TextConfig::hidden, effective_prefill_chunk}, "step prefill hidden");
    if (plan.causal_scoring) {
        out.score_hidden = add_tensor(
            builder, DType::BF16, {TextConfig::hidden, static_cast<std::int32_t>(kCausalScoreTile)},
            "causal score hidden staging");
    }
    qwen3_6::complete_round_state_layout(builder, out.round);
    const auto i32 = [&](std::size_t n, const char* label) {
        return add_tensor(builder, DType::I32, {static_cast<std::int32_t>(n)}, label);
    };
    out.token_counts =
        add_tensor(builder, DType::I32,
                   {TextConfig::token_domain, static_cast<std::int32_t>(plan.max_concurrency)},
                   "sampling token counts");
    const auto config_words = static_cast<std::int32_t>(
        (sizeof(ops::SamplingConfig) + sizeof(std::int32_t) - 1) / sizeof(std::int32_t));
    out.sampling_config = add_tensor(
        builder, DType::I32, {config_words, static_cast<std::int32_t>(plan.max_concurrency)},
        "sampling config");
    out.bytes = builder.finish(kArenaAlign, "persistent layout");
    out.kv_payload_bytes =
        out.decoder.kv_payload_bytes() + (out.dflash ? out.dflash->kv_payload_bytes() : 0);
    return out;
}

WorkspacePlan build_workspace_plan(const SequencePlanImpl& plan) {
    const std::uint32_t chunk_u32 = std::min(plan.prefill_chunk, plan.capacity);
    if (chunk_u32 == 0 ||
        chunk_u32 > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) ||
        plan.draft_window >= static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument("sequence workspace dimensions are invalid");
    }
    const auto chunk  = static_cast<std::int32_t>(chunk_u32);
    const auto drafts = static_cast<std::int32_t>(plan.draft_window);
    const auto verify = drafts + 1;
    const ops::GqaExecutionEnvelope text_envelope{1, plan.capacity};

    const auto matrix  = [](WorkspaceLayoutBuilder& layout, DType dtype, std::int32_t rows,
                           std::int32_t tokens) { (void)layout.alloc(dtype, {rows, tokens}); };
    const auto scratch = [](WorkspaceLayoutBuilder& layout, std::size_t bytes) {
        if (bytes == 0) { return; }
        auto scope = layout.scope();
        (void)layout.alloc_bytes(bytes);
    };
    const auto finish = [](const WorkspaceLayoutBuilder& layout) { return layout.peak_bytes(1); };

    const auto text_common_root = [&](WorkspaceLayoutBuilder& layout, std::int32_t tokens) {
        (void)workspace_recipe::text_prefill_roots<TextConfig>(
            layout, tokens, plan.features.vision ? 3 : 0, plan.features.vision ? tokens : 0);
    };
    const auto attention_stage = [&](WorkspaceLayoutBuilder& layout, std::int32_t first,
                                     std::int32_t last, qwen3_6::TextPhase phase,
                                     std::int32_t batch_size, std::int32_t min_width,
                                     std::int32_t max_width,
                                     ops::GqaExecutionEnvelope envelope) {
        auto stage = layout.scope();
        (void)workspace_recipe::text_attention_projection<TextConfig>(layout, last);
        scratch(layout, Variant::attention_projection_workspace_capacity_bytes(plan.weights_profile,
                                                                               phase, first, last));
        (void)workspace_recipe::text_attention_results<TextConfig>(layout, last);
        scratch(layout, ops::gqa_attention_workspace_capacity_bytes(TextConfig::head_dim, 
                            TextConfig::query_heads, plan.kv_dtype, envelope, batch_size,
                            min_width, max_width));
        scratch(layout, Variant::attention_output_projection_workspace_capacity_bytes(
                            plan.weights_profile, phase, first, last));
    };
    const auto gdn_stage = [&](WorkspaceLayoutBuilder& layout, std::int32_t first,
                               std::int32_t last, qwen3_6::TextPhase phase, GdnWorkspacePath path,
                               std::int32_t batch_size, std::int32_t min_width,
                               std::int32_t max_width) {
        auto stage = layout.scope();
        (void)workspace_recipe::gdn_control<TextConfig>(layout, last);
        scratch(layout, Variant::gdn_norm_control_projection_workspace_capacity_bytes(first, last));
        (void)workspace_recipe::gdn_projection<TextConfig>(layout, last);
        if (path == GdnWorkspacePath::Snapshot) {
            scratch(layout, Variant::gdn_input_projection_snapshot_workspace_capacity_bytes(
                                plan.weights_profile, phase, batch_size, min_width, max_width));
        } else if (path == GdnWorkspacePath::ReplayRecord) {
            scratch(layout, Variant::gdn_input_projection_record_workspace_capacity_bytes(
                                plan.weights_profile, phase, batch_size, min_width, max_width));
        } else {
            (void)workspace_recipe::gdn_prefill_conv<TextConfig>(layout, last);
            scratch(layout, Variant::gdn_input_projection_workspace_capacity_bytes(
                                plan.weights_profile, phase, first, last));
        }
        (void)workspace_recipe::gdn_recurrent_output<TextConfig>(layout, last);
        if (path == GdnWorkspacePath::Prefill) {
            scratch(layout,
                    ops::gated_delta_net_workspace_capacity_bytes(
                        TextConfig::gdn_key_heads, TextConfig::gdn_value_heads, true, first, last));
        }
        (void)workspace_recipe::gdn_normalized_output<TextConfig>(layout, last);
        scratch(layout, Variant::gdn_output_projection_workspace_capacity_bytes(
                            plan.weights_profile, phase, first, last));
    };
    const auto post_mixer_stage = [&](WorkspaceLayoutBuilder& layout, std::int32_t first,
                                      std::int32_t last, qwen3_6::TextPhase phase) {
        auto stage = layout.scope();
        (void)workspace_recipe::post_mixer_hidden<TextConfig>(layout, last);
        scratch(layout, Variant::post_mixer_workspace_capacity_bytes(plan.weights_profile, phase,
                                                                     first, last));
    };
    const auto target_body = [&](WorkspaceLayoutBuilder& layout, std::int32_t first,
                                 std::int32_t last, qwen3_6::TextPhase phase, GdnWorkspacePath path,
                                 std::int32_t batch_size, std::int32_t min_width,
                                 std::int32_t max_width,
                                 ops::GqaExecutionEnvelope envelope) {
        attention_stage(layout, first, last, phase, batch_size, min_width, max_width, envelope);
        gdn_stage(layout, first, last, phase, path, batch_size, min_width, max_width);
        post_mixer_stage(layout, first, last, phase);
    };
    const auto proposal_scratch = [&](WorkspaceLayoutBuilder& layout, std::int32_t columns) {
        if (plan.proposal_head == ProposalHead::Optimized) {
            matrix(layout, DType::BF16, Variant::draft_head_rows, columns);
        }
    };
    const auto mtp_stem = [&](WorkspaceLayoutBuilder& layout, std::int32_t tokens,
                              bool preembedded) {
        (void)workspace_recipe::mtp_stem<TextConfig>(layout, tokens, !preembedded);
    };
    const auto mtp_full_core = [&](WorkspaceLayoutBuilder& layout, std::int32_t tokens,
                                   ops::GqaExecutionEnvelope envelope) {
        auto core = layout.scope();
        mtp_stem(layout, tokens, false);
        (void)workspace_recipe::mtp_attention_projection<TextConfig>(layout, tokens);
        scratch(layout, Variant::mtp_attention_projection_workspace_capacity_bytes(tokens, tokens));
        (void)workspace_recipe::mtp_attention_results<TextConfig>(layout, tokens);
        scratch(layout, ops::gqa_attention_workspace_capacity_bytes(TextConfig::head_dim, 
                            TextConfig::query_heads, plan.kv_dtype, envelope, 1, tokens, tokens));
        (void)workspace_recipe::mtp_post_attention<TextConfig>(layout, tokens);
        scratch(layout, Variant::mtp_post_mixer_workspace_capacity_bytes(tokens, tokens));
    };
    const auto mtp_full_call = [&](WorkspaceLayoutBuilder& layout, std::int32_t tokens,
                                   ops::GqaExecutionEnvelope envelope,
                                   bool build_proposal) {
        auto call = layout.scope();
        matrix(layout, DType::I32, 1, tokens);
        mtp_full_core(layout, tokens, envelope);
        if (build_proposal) {
            auto proposal = layout.scope();
            proposal_scratch(layout, 1);
        }
    };
    const auto mtp_prefill_chunk = [&](WorkspaceLayoutBuilder& layout, std::int32_t first,
                                       std::int32_t last, bool preembedded) {
        auto call = layout.scope();
        matrix(layout, DType::BF16, TextConfig::hidden, 1);
        matrix(layout, DType::BF16, TextConfig::hidden, 1);
        {
            auto bulk = layout.scope();
            mtp_stem(layout, last, preembedded);
            matrix(layout, DType::BF16, TextConfig::kv_size, last);
            matrix(layout, DType::BF16, TextConfig::kv_size, last);
            scratch(layout, Variant::mtp_kv_projection_workspace_capacity_bytes(first, last));
            matrix(layout, DType::BF16, TextConfig::kv_size, last);
        }
        matrix(layout, DType::BF16, TextConfig::query_size, 1);
        matrix(layout, DType::BF16, TextConfig::query_size, 1);
        scratch(layout, Variant::mtp_q_gate_projection_workspace_capacity_bytes(1, 1));
        matrix(layout, DType::BF16, TextConfig::query_size, 1);
        matrix(layout, DType::I32, 3, 1);
        matrix(layout, DType::BF16, TextConfig::query_size, 1);
        scratch(layout, ops::gqa_attention_workspace_capacity_bytes(TextConfig::head_dim, 
                            TextConfig::query_heads, plan.kv_dtype, text_envelope, 1, 1, 1));
        matrix(layout, DType::BF16, TextConfig::hidden, 1);
        matrix(layout, DType::BF16, TextConfig::hidden, 1);
        scratch(layout, Variant::mtp_post_mixer_workspace_capacity_bytes(1, 1));
        proposal_scratch(layout, 1);
    };

    WorkspacePlan out;
    WorkspaceLayoutBuilder text_prefill;
    text_common_root(text_prefill, chunk);
    target_body(text_prefill, 1, chunk, qwen3_6::TextPhase::Prefill, GdnWorkspacePath::Prefill, 1,
                1, chunk, text_envelope);
    scratch(text_prefill, ops::sampling_workspace_capacity_bytes(TextConfig::token_domain, 1, 1));
    out.text_prefill = finish(text_prefill);

    if (plan.causal_scoring) {
        WorkspaceLayoutBuilder causal_score;
        matrix(causal_score, DType::BF16, TextConfig::output_rows,
               static_cast<std::int32_t>(kCausalScoreTile));
        matrix(causal_score, DType::I32, 1, static_cast<std::int32_t>(kCausalScoreTile));
        matrix(causal_score, DType::FP32, 1, static_cast<std::int32_t>(kCausalScoreTile));
        out.causal_score = finish(causal_score);
    }

    for (std::int32_t batch = 1; batch <= static_cast<std::int32_t>(plan.max_concurrency);
         ++batch) {
        WorkspaceLayoutBuilder ordinary;
        matrix(ordinary, DType::BF16, TextConfig::hidden, batch);
        target_body(ordinary, batch, batch, qwen3_6::TextPhase::Verify, GdnWorkspacePath::Snapshot,
                    batch, 1, 1, text_envelope);
        scratch(ordinary,
                ops::sampling_workspace_capacity_bytes(TextConfig::token_domain, batch, batch));
        out.ordinary_round = std::max(out.ordinary_round, finish(ordinary));
    }

    if (plan.features.mtp()) {
        WorkspaceLayoutBuilder mtp_prefill;
        text_common_root(mtp_prefill, chunk);
        target_body(mtp_prefill, 1, chunk, qwen3_6::TextPhase::Prefill, GdnWorkspacePath::Prefill,
                    1, 1, chunk, text_envelope);
        matrix(mtp_prefill, DType::I32, 1, chunk);
        if (plan.features.vision) {
            matrix(mtp_prefill, DType::BF16, TextConfig::hidden, chunk);
            (void)workspace_recipe::visual_scatter_indices(mtp_prefill, chunk);
        }
        mtp_prefill_chunk(mtp_prefill, 1, chunk, plan.features.vision);
        for (std::int32_t i = 1; i < drafts; ++i) {
            matrix(mtp_prefill, DType::BF16, TextConfig::hidden, 1);
            mtp_full_call(mtp_prefill, 1, text_envelope, true);
        }
        out.mtp_prefill = finish(mtp_prefill);

        WorkspaceLayoutBuilder mtp_batch;
        mtp_full_call(mtp_batch, verify, text_envelope, false);
        WorkspaceLayoutBuilder mtp_ar;
        mtp_full_call(mtp_ar, 1, text_envelope, true);
        WorkspaceLayoutBuilder mtp_align;
        mtp_full_call(mtp_align, 1, text_envelope, false);
        WorkspaceLayoutBuilder mtp_proposal;
        proposal_scratch(mtp_proposal, 1);
        const std::size_t accept = ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(
            TextConfig::token_domain, drafts, drafts, 1, 1);
        out.mtp_round = std::max({accept, finish(mtp_batch), finish(mtp_ar), finish(mtp_proposal)});
        out.ordinary_round = std::max(out.ordinary_round, finish(mtp_align));

        for (std::int32_t batch = 1; batch <= static_cast<std::int32_t>(plan.max_concurrency);
             ++batch) {
            const std::int32_t aggregate = batch * verify;
            WorkspaceLayoutBuilder target;
            matrix(target, DType::BF16, TextConfig::hidden, aggregate);
            target_body(target, aggregate, aggregate, qwen3_6::TextPhase::Verify,
                        GdnWorkspacePath::ReplayRecord, batch, verify, verify, text_envelope);

            const auto mtp_decode_core = [&](WorkspaceLayoutBuilder& layout, std::int32_t width) {
                const std::int32_t tokens = batch * width;
                auto core                 = layout.scope();
                mtp_stem(layout, tokens, false);
                (void)workspace_recipe::mtp_attention_projection<TextConfig>(layout, tokens);
                scratch(layout,
                        Variant::mtp_attention_projection_workspace_capacity_bytes(tokens, tokens));
                (void)workspace_recipe::mtp_attention_results<TextConfig>(layout, tokens);
                scratch(layout,
                        ops::gqa_attention_workspace_capacity_bytes(TextConfig::head_dim, 
                            TextConfig::query_heads, plan.kv_dtype, text_envelope, batch, width, width));
                (void)workspace_recipe::mtp_post_attention<TextConfig>(layout, tokens);
                scratch(layout, Variant::mtp_post_mixer_workspace_capacity_bytes(tokens, tokens));
            };

            WorkspaceLayoutBuilder alignment;
            mtp_decode_core(alignment, verify);
            WorkspaceLayoutBuilder ar;
            mtp_decode_core(ar, 1);
            WorkspaceLayoutBuilder proposal;
            proposal_scratch(proposal, batch);
            const std::size_t batch_accept =
                ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(
                    TextConfig::token_domain, drafts, drafts, batch, batch);
            out.mtp_round = std::max({out.mtp_round, finish(target), finish(alignment), finish(ar),
                                      finish(proposal), batch_accept});
        }
    }

    if (plan.features.dflash()) {
        if constexpr (!Variant::supports_dflash) {
            throw std::logic_error("unsupported target reached DFlash scratch planning");
        } else {
            const auto dflash_context_capacity = [&](std::int32_t tokens, bool compact_input) {
                WorkspaceLayoutBuilder layout;
                if (compact_input) {
                    matrix(layout, DType::BF16, DFlashConfig::feature_rows, tokens);
                }
                (void)workspace_recipe::dflash_context<DFlashConfig>(layout, tokens);
                (void)ops::linear_workspace_capacity_bytes(
                    QType::BF16_CTRL, DFlashConfig::hidden, DFlashConfig::feature_rows,
                    ops::LinearPolicy::A16Only, tokens, tokens);
                {
                    auto layer = layout.scope();
                    (void)workspace_recipe::dflash_context_layer<DFlashConfig>(layout, tokens);
                    (void)ops::linear_workspace_capacity_bytes(
                        QType::BF16_CTRL, DFlashConfig::kv_size, DFlashConfig::hidden,
                        ops::LinearPolicy::A16Only, tokens, tokens);
                }
                return finish(layout);
            };
            const auto dflash_proposal_capacity = [&](std::int32_t width, std::int32_t batch) {
                WorkspaceLayoutBuilder layout;
                const std::int32_t tokens = width * batch;
                matrix(layout, DType::BF16, DFlashConfig::hidden, tokens);
                {
                    auto attention = layout.scope();
                    (void)workspace_recipe::dflash_attention<DFlashConfig>(layout, tokens);
                    scratch(layout,
                            std::max(ops::swa_workspace_capacity_bytes(
                                         {0, plan.capacity}, width, width, batch,
                                         DFlashConfig::local_window),
                                     ops::bidirectional_gqa_attention_workspace_capacity_bytes(
                                         {0, plan.capacity}, width, width, batch)));
                    (void)ops::linear_workspace_capacity_bytes(
                        QType::BF16_CTRL, DFlashConfig::query_size + 2 * DFlashConfig::kv_size,
                        DFlashConfig::hidden, ops::LinearPolicy::A16Only, tokens, tokens);
                    (void)ops::linear_workspace_capacity_bytes(
                        QType::BF16_CTRL, DFlashConfig::hidden, DFlashConfig::query_size,
                        ops::LinearPolicy::A16Only, tokens, tokens);
                }
                {
                    auto mlp = layout.scope();
                    (void)workspace_recipe::dflash_mlp<DFlashConfig>(layout, tokens);
                    (void)ops::linear_workspace_capacity_bytes(
                        QType::BF16_CTRL, 2 * DFlashConfig::intermediate, DFlashConfig::hidden,
                        ops::LinearPolicy::A16Only, tokens, tokens);
                    (void)ops::linear_workspace_capacity_bytes(
                        QType::BF16_CTRL, DFlashConfig::hidden, DFlashConfig::intermediate,
                        ops::LinearPolicy::A16Only, tokens, tokens);
                }
                matrix(layout, DType::BF16, DFlashConfig::hidden, drafts * batch);
                matrix(layout, DType::BF16, DFlashConfig::hidden, drafts * batch);
                if (plan.proposal_head == ProposalHead::Optimized) {
                    matrix(layout, DType::BF16, Variant::draft_head_rows, drafts * batch);
                } else {
                    matrix(layout, DType::BF16, TextConfig::output_rows, drafts * batch);
                }
                // DSpark Markov-argmax/SVIP per-row scratch.
                matrix(layout, DType::I32, batch, 1);
                matrix(layout, DType::I32, batch, 1);
                matrix(layout, DType::I32, batch, 1);
                return finish(layout);
            };

            out.dflash_context = dflash_context_capacity(chunk, false);
            for (std::int32_t batch = 1; batch <= static_cast<std::int32_t>(plan.max_concurrency);
                 ++batch) {
                const std::int32_t aggregate = verify * batch;
                WorkspaceLayoutBuilder target;
                matrix(target, DType::BF16, TextConfig::hidden, aggregate);
                target_body(target, aggregate, aggregate, qwen3_6::TextPhase::Verify,
                            GdnWorkspacePath::ReplayRecord, batch, verify, verify, text_envelope);
                const std::size_t accept =
                    ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(
                        TextConfig::token_domain, drafts, drafts, batch, batch);
                const std::size_t proposal = dflash_proposal_capacity(verify, batch);
                out.dflash_round           = std::max({out.dflash_round, finish(target), accept,
                                                       dflash_context_capacity(aggregate, true), proposal});
            }
        }
    }

    if (plan.features.dflash2()) {
        if constexpr (!Variant::supports_dflash2) {
            throw std::logic_error("unsupported target reached DFlash2 scratch planning");
        } else {
            const auto dflash2_context_capacity = [&](std::int32_t tokens, bool compact_input) {
                WorkspaceLayoutBuilder layout;
                if (compact_input) {
                    matrix(layout, DType::BF16, DFlash2Config::feature_rows, tokens);
                }
                (void)workspace_recipe::dflash_context<DFlash2Config>(layout, tokens);
                (void)ops::linear_workspace_capacity_bytes(
                    QType::BF16_CTRL, DFlash2Config::hidden, DFlash2Config::feature_rows,
                    ops::LinearPolicy::A16Only, tokens, tokens);
                {
                    auto layer = layout.scope();
                    (void)workspace_recipe::dflash_context_layer<DFlash2Config>(layout, tokens);
                    (void)ops::linear_workspace_capacity_bytes(
                        QType::BF16_CTRL, DFlash2Config::kv_size, DFlash2Config::hidden,
                        ops::LinearPolicy::A16Only, tokens, tokens);
                }
                return finish(layout);
            };
            const auto dflash2_proposal_capacity = [&](std::int32_t width, std::int32_t batch) {
                WorkspaceLayoutBuilder layout;
                const std::int32_t tokens = width * batch;
                matrix(layout, DType::BF16, DFlash2Config::hidden, tokens);
                {
                    auto attention = layout.scope();
                    (void)workspace_recipe::dflash_attention<DFlash2Config>(layout, tokens);
                    matrix(layout, DType::BF16, 1280, tokens);
                    matrix(layout, DType::BF16, DFlash2Config::hidden, tokens);
                    matrix(layout, DType::BF16, DFlash2Config::hidden, tokens);
                    scratch(layout, ops::swa_workspace_capacity_bytes(
                                       {0, plan.capacity}, width, width, batch,
                                       DFlash2Config::local_window));
                    (void)ops::linear_workspace_capacity_bytes(
                        QType::BF16_CTRL, DFlash2Config::query_size + 2 * DFlash2Config::kv_size,
                        DFlash2Config::hidden, ops::LinearPolicy::A16Only, tokens, tokens);
                    (void)ops::linear_workspace_capacity_bytes(
                        QType::BF16_CTRL, DFlash2Config::hidden, DFlash2Config::query_size,
                        ops::LinearPolicy::A16Only, tokens, tokens);
                    (void)ops::linear_workspace_capacity_bytes(
                        QType::BF16_CTRL, 1280, DFlash2Config::hidden, ops::LinearPolicy::A16Only,
                        tokens, tokens);
                }
                {
                    auto mlp = layout.scope();
                    (void)workspace_recipe::dflash_mlp<DFlash2Config>(layout, tokens);
                    matrix(layout, DType::BF16, 1280, tokens);
                    matrix(layout, DType::BF16, DFlash2Config::hidden, tokens);
                    matrix(layout, DType::BF16, DFlash2Config::hidden, tokens);
                    (void)ops::linear_workspace_capacity_bytes(
                        QType::BF16_CTRL, 2 * DFlash2Config::intermediate, DFlash2Config::hidden,
                        ops::LinearPolicy::A16Only, tokens, tokens);
                    (void)ops::linear_workspace_capacity_bytes(
                        QType::BF16_CTRL, DFlash2Config::hidden, DFlash2Config::intermediate,
                        ops::LinearPolicy::A16Only, tokens, tokens);
                    (void)ops::linear_workspace_capacity_bytes(
                        QType::BF16_CTRL, 1280, DFlash2Config::hidden, ops::LinearPolicy::A16Only,
                        tokens, tokens);
                }
                matrix(layout, DType::BF16, DFlash2Config::hidden, drafts * batch);
                matrix(layout, DType::BF16, DFlash2Config::hidden, drafts * batch);
                matrix(layout, DType::BF16, TextConfig::output_rows, drafts * batch);
                matrix(layout, DType::BF16, DFlash2Config::selector_rank, drafts * batch);
                (void)ops::linear_workspace_capacity_bytes(
                    QType::BF16_CTRL, DFlash2Config::selector_rank, DFlash2Config::hidden,
                    ops::LinearPolicy::A16Only, drafts * batch, drafts * batch);
                // Scratch shapes must match the proposal step count the schedule
                // actually runs (dflash2_impl.h), i.e. the runtime draft window.
                matrix(layout, DType::I32, batch * drafts * DFlash2Config::selector_top_k,
                       1);
                matrix(layout, DType::FP32, batch * drafts * DFlash2Config::selector_top_k,
                       1);
                matrix(layout, DType::FP32,
                       batch * drafts * DFlash2Config::selector_top_k *
                           DFlash2Config::selector_top_k,
                       1);
                return finish(layout);
            };

            out.dflash2_context = dflash2_context_capacity(chunk, false);
            for (std::int32_t batch = 1; batch <= static_cast<std::int32_t>(plan.max_concurrency);
                 ++batch) {
                const std::int32_t aggregate = verify * batch;
                WorkspaceLayoutBuilder target;
                matrix(target, DType::BF16, TextConfig::hidden, aggregate);
                target_body(target, aggregate, aggregate, qwen3_6::TextPhase::Verify,
                            GdnWorkspacePath::ReplayRecord, batch, verify, verify, text_envelope);
                const std::size_t accept =
                    ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(
                        TextConfig::token_domain, drafts, drafts, batch, batch);
                const std::size_t proposal = dflash2_proposal_capacity(verify, batch);
                out.dflash2_round          = std::max({out.dflash2_round, finish(target), accept,
                                                       dflash2_context_capacity(aggregate, true),
                                                       proposal});
            }
        }
    }

    out.general_capacity =
        std::max({out.text_prefill, out.ordinary_round, out.mtp_prefill, out.mtp_round,
                  out.dflash_context, out.dflash_round, out.dflash2_context, out.dflash2_round,
                  out.causal_score});
    // Stopgap for the Muse multi-chunk prefill bad_alloc (see _TODO.md 89): the text_prefill
    // recipe under-estimates the per-call peak (34.5 MiB planned vs >=55 MiB used at
    // prefill_chunk=512), so the bump arena ran out. The recipe still needs reconciling; until
    // then the arena gets headroom, which is safe because every buffer keeps its own size.
    // NINFER_WS_HEADROOM_PCT overrides the default 100% (i.e. 2x) for experiments.
    {
        static const std::uint32_t headroom_pct = [] {
            const char* value = std::getenv("NINFER_WS_HEADROOM_PCT");
            const long parsed = value == nullptr ? 100 : std::atol(value);
            return parsed >= 0 && parsed <= 1000 ? static_cast<std::uint32_t>(parsed) : 100U;
        }();
        out.general_capacity += out.general_capacity * headroom_pct / 100U;
    }
    out.capacity = out.general_capacity;
    if (std::getenv("NINFER_WS_DUMP") != nullptr) {
        std::fprintf(stderr,
                     "[ws] chunk=%u text_prefill=%zu ordinary=%zu mtp_prefill=%zu mtp_round=%zu "
                     "general=%zu\n",
                     chunk_u32, out.text_prefill, out.ordinary_round, out.mtp_prefill,
                     out.mtp_round, out.general_capacity);
    }
    if (plan.features.vision) {
        const std::uint32_t merged = static_cast<std::uint32_t>(
            std::min<std::uint64_t>(plan.capacity, kMaximumVisionItemTokens));
        out.vision   = schedule::VisionContext::plan_workspace(merged, out.general_capacity);
        out.capacity = std::max(out.capacity, out.vision->capacity_bytes);
    }
    return out;
}

void validate_target_options(DeviceContext& device, const EngineOptions& options) {
    // Static YaRN factor-4 extends the rope domain to 4x the native context.
    const std::uint32_t context_limit =
        options.yarn_enabled ? 4ULL * Variant::maximum_context : Variant::maximum_context;
    if (options.max_context == 0 || options.max_context > context_limit) {
        throw std::invalid_argument(
            options.yarn_enabled
                ? "max_context exceeds the variant YaRN-extended context capacity"
                : "max_context exceeds the variant native context capacity");
    }
    if (options.prefill_chunk == 0 || options.prefill_chunk % kPrefillChunkAlignment != 0) {
        throw std::invalid_argument("prefill_chunk must be a nonzero multiple of 128");
    }
    if (options.max_concurrency == 0 || options.max_concurrency > kMaximumConcurrency) {
        throw std::invalid_argument("max_concurrency must be in [1,8]");
    }
    const std::uint32_t logical_pages = page_count(options.max_context);
    // An explicit --kv-capacity may floor the device page pool below max_context
    // (YaRN extends the rope domain beyond the pool; the cold pool recycles pages).
    std::uint32_t minimum_pages = std::max(logical_pages, options.max_concurrency);
    if (options.kv_capacity.mode == KvCapacityMode::Explicit) {
        minimum_pages = std::max(page_count(options.kv_capacity.explicit_tokens),
                                 options.max_concurrency);
    }
    const std::uint64_t maximum_pages64 =
        static_cast<std::uint64_t>(options.max_concurrency) * logical_pages;
    if (maximum_pages64 > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("maximum Main KV page count exceeds uint32");
    }
    switch (options.kv_capacity.mode) {
    case KvCapacityMode::Explicit: {
        if (options.kv_capacity.explicit_tokens == 0) {
            throw std::invalid_argument("kv_capacity must be positive");
        }
        const std::uint32_t requested_pages = page_count(options.kv_capacity.explicit_tokens);
        if (requested_pages < minimum_pages || requested_pages > maximum_pages64) {
            throw std::invalid_argument(
                "kv_capacity is outside the usable range for max_context and max_concurrency");
        }
        break;
    }
    case KvCapacityMode::Automatic:
        break;
    default:
        throw std::invalid_argument("unknown kv_capacity policy");
    }
    switch (options.speculative.backend) {
    case SpeculativeBackend::None:
        if (options.speculative.draft_tokens != 0 ||
            options.speculative.proposal_head != ProposalHead::Full) {
            throw std::invalid_argument(
                "disabled speculative decoding requires draft_tokens=0 and the full proposal head");
        }
        break;
    case SpeculativeBackend::Mtp:
        if (options.speculative.draft_tokens == 0 ||
            options.speculative.draft_tokens > kMaximumMtpDraftTokens) {
            throw std::invalid_argument("MTP draft window must be in [1,5]");
        }
        break;
    case SpeculativeBackend::DFlash2:
        // --draft-tokens picks the startup-fixed draft window K; 0 keeps the
        // historical 7-draft default, normalized by the planner below. The bound
        // is enforced here too so the planner never sees an out-of-frame draft
        // window (empty graph profiles / over-wide layouts).
        if (options.speculative.draft_tokens > qwen3_6::kDFlashDecodeMaximumDrafts) {
            throw std::invalid_argument(
                "DFlash2 draft window must be 0 (default 7) or in [1,15]");
        }
        break;
    case SpeculativeBackend::DFlash:
        if (kMaximumDFlashDraftTokens == 0) {
            throw std::invalid_argument("DFlash is not supported by this target");
        }
        if (options.speculative.draft_tokens == 0 ||
            options.speculative.draft_tokens > kMaximumDFlashDraftTokens) {
            throw std::invalid_argument("DFlash draft window must be in [1,15]");
        }
        if (options.enable_vision) {
            throw std::invalid_argument("DFlash and Vision cannot be enabled together");
        }
        break;
    }
    // Multi-GPU gate (see tools/archkit/_GPU_MATRIX.md): this binary is
    // compiled for the archs in CMAKE_CUDA_ARCHITECTURES; the nvfp4 weight
    // profile needs fp4 tensor cores (sm_100a/120a) while the groupwise-int
    // profile targets Turing+ (sm_75+). Until the per-arch dist matrix is
    // built, the runtime only accepts the native sm_120 build.
    if (device.sm() != 120) {
        throw std::invalid_argument(
            "this ninfer build targets compute capability 12.0 only; your GPU is "
            "sm_" + std::to_string(device.sm()) +
            " - use the per-arch dist (groupwise-int for sm_75..sm_89, see "
            "_GPU_MATRIX.md) or rebuild with CMAKE_CUDA_ARCHITECTURES");
    }
}

std::unique_ptr<SequencePlanImpl> build_sequence_candidate(const SequencePlanningInputs& inputs,
                                                           std::uint32_t main_page_groups) {
    if (main_page_groups == 0) {
        throw std::invalid_argument("Main KV physical page count must be positive");
    }
    auto impl                 = std::make_unique<SequencePlanImpl>();
    impl->weights_profile     = inputs.weights_profile;
    impl->capacity            = inputs.capacity;
    impl->main_page_groups    = main_page_groups;
    impl->kv_capacity         = static_cast<std::uint32_t>(checked_i32(
        static_cast<std::uint64_t>(main_page_groups) * static_cast<std::uint32_t>(kPagedKVPageSize),
        "resolved Paged KV capacity exceeds int32"));
    impl->max_concurrency     = inputs.max_concurrency;
    impl->prefill_chunk       = inputs.prefill_chunk;
    impl->draft_window        = inputs.draft_window;
    impl->speculative_backend = inputs.speculative_backend;
    impl->proposal_head       = inputs.proposal_head;
    impl->features            = inputs.features;
    impl->use_cuda_graph      = inputs.use_cuda_graph;
    impl->cold_policy      = inputs.cold_policy;
    impl->cold_keep_tokens = inputs.cold_keep_tokens;
    impl->max_cold_pages   = inputs.max_cold_pages;
    impl->cold_host_bytes  = inputs.cold_host_bytes;
    impl->cold_disk_path   = inputs.cold_disk_path;
    impl->cold_disk_bytes  = inputs.cold_disk_bytes;
    // The two cold byte budgets are config, not runtime state: a tier the
    // operator explicitly selected whose cap can never admit one page is
    // rejected here, before anything is laid out (the same loud-contradiction
    // rule effective_cold_pages applies to --max-cold-pages + host). The
    // capacities in pages are derived later, where the strides are known.
    product::validate_cold_tier_budget(product::cold_tier_budget_from(
        impl->cold_policy, impl->cold_host_bytes, impl->cold_disk_bytes));
    impl->causal_scoring      = inputs.causal_scoring;
    impl->graph_capture_ceiling = inputs.graph_capture_ceiling;
    impl->device              = inputs.device;
    impl->context_cache       = inputs.context_cache;
    impl->kv_dtype            = inputs.kv_dtype;
    impl->kv_quant_group      = inputs.kv_quant_group;
    impl->layer_kv_dtypes     = inputs.layer_kv_dtypes;
    impl->kv_residual_layers  = inputs.kv_residual_layers;
    impl->kv_v_codec          = inputs.kv_v_codec;
    impl->kv_rotation_off     = inputs.kv_rotation_off;
    impl->kv_row_scale_spec   = inputs.kv_row_scale_spec;
    impl->persistent          = persistent_layout(*impl);
    impl->workspace           = build_workspace_plan(*impl);
    if (impl->use_cuda_graph) {
        // Definitions remain per execution profile, but only one executable is instantiated for
        // each reachable node-topology class. These bounds cover the largest profile installed in
        // each class and the driver/module state materialized while qualifying all definitions.
        if (impl->speculative_backend == SpeculativeBackend::None) {
            impl->graph_allowance_bytes = checked_mul(12ULL * kMiB, impl->max_concurrency,
                                                      "ordinary exact-b graph allowance");
        } else if (impl->speculative_backend == SpeculativeBackend::Mtp) {
            const auto profiles = mtp_graph_profiles(impl->capacity, impl->draft_window);
            const std::size_t per_batch_allowance = graph_topology_allowance(
                profiles,
                [&](GraphExecutionProfile profile) {
                    const std::uint64_t final_visible = std::min<std::uint64_t>(
                        impl->capacity,
                        static_cast<std::uint64_t>(profile.max) + 2ULL * impl->draft_window);
                    return (final_visible <= 4096 ? 12ULL : 82ULL) * kMiB;
                },
                "MTP graph allowance");
            impl->graph_allowance_bytes = checked_mul(per_batch_allowance, impl->max_concurrency,
                                                      "MTP exact-b graph allowance");
        } else {
            const auto class_allowance = [&](std::uint32_t batch_size) {
                const auto profiles =
                    dflash_graph_profiles(impl->capacity, impl->draft_window, batch_size);
                return graph_topology_allowance(
                    profiles,
                    [&](GraphExecutionProfile profile) {
                        const std::uint64_t final_visible = std::min<std::uint64_t>(
                            impl->capacity,
                            static_cast<std::uint64_t>(profile.max) + impl->draft_window + 1ULL);
                        return (final_visible <= 4096 ? 64ULL : 96ULL) * kMiB;
                    },
                    "DFlash graph allowance");
            };
            for (std::uint32_t batch_size = 1; batch_size <= impl->max_concurrency; ++batch_size) {
                impl->graph_allowance_bytes =
                    checked_add(impl->graph_allowance_bytes, class_allowance(batch_size),
                                "DFlash exact-b graph allowance");
            }
        }
    }

    impl->device_reservation_bytes = checked_add(
        checked_add(impl->persistent.bytes, impl->workspace.capacity, "sequence memory plan"),
        impl->graph_allowance_bytes, "sequence graph allowance");
    return impl;
}

} // namespace

std::unique_ptr<qwen3_6::detail::SequencePlannerImpl<Variant>>
make_sequence_planner_impl(DeviceContext& device, const EngineOptions& options,
                           WeightsProfile weights_profile) {
    validate_target_options(device, options);
    const TargetKVCacheProfile kv_profile = target_kv_cache_profile(options.kv_cache);

    // N1/S30: --kv-bit-budget deferred resolution. The CLI parse site has no
    // model knowledge; here TextConfig::full_attention_layers() is available.
    // The DP (product/kv_bit_budget.h, parity-proven against
    // tools/archkit/kv_bit_budget.py, _collab/A_kvbit_cold_cpp.md) resolves the
    // budget TOGETHER with the cold capacity: cold_cap = effective_cold_pages,
    // the same helper (and precedence) the layout reservation uses, so the DP
    // can never plan more cold layers than the pool that will exist. With no
    // cold capacity (cold_cap = 0: no policy, ColdPolicy::None/Host, or a zero
    // cap) the DP runs hot-only and the resolved table is byte-identical to
    // the N1 path (S12/S14 goldens).
    std::array<KvCacheStorage, 64> budget_storage_table = options.kv_layer_storage;
    bool storage_explicit = options.kv_layer_storage_explicit;

    // --kv-tier-formats / --nvfp4-mode: the KV tier vocabulary (hot/tail/cold +
    // nvfp4 mode). Resolved HERE, not at the parse site, for the same reason as
    // --kv-bit-budget below: `hot` is the resident format of every full-attention
    // layer, so the landing needs this target's layer count, and the mode/cold
    // gates must judge the table this plan will really build. Stage 1 (here)
    // applies `hot` to the same storage table the other knobs produce, so the
    // engine keeps exactly one resolution path into the per-layer dtypes; stage 2
    // (after layer_overrides) judges mode/cold on the effective per-layer dtype.
    // Whatever the engine cannot express -- the `tail` tier, the unreachable cold
    // codecs -- is refused with the missing mechanism named, never approximated.
    std::optional<product::KvTierPlan> tier_plan;
    if (options.kv_tier_formats_explicit) {
        tier_plan = product::kv_tier_formats_plan(options.kv_tier_formats_spec,
                                                  options.kv_nvfp4_pure, budget_storage_table);
        if (tier_plan->override_hot &&
            (storage_explicit || options.kv_bit_budget_explicit || options.kv_cache_explicit)) {
            throw std::invalid_argument(
                "--kv-tier-formats with an explicit hot format cannot be combined with "
                "--kv-layer-storage / --kv-bit-budget / --kv-dtype: hot already names the "
                "resident format of every full-attention layer, so the two describe the same "
                "table");
        }
        if (tier_plan->override_hot) {
            budget_storage_table = tier_plan->table;
            storage_explicit     = true;
        }
    }

    if (options.kv_bit_budget_explicit && !storage_explicit) {
        const std::int32_t full_layers =
            static_cast<std::int32_t>(TextConfig::full_attention_layers());
        const std::uint32_t cold_pages = effective_cold_pages(
            options.cold_policy, options.cold_keep_tokens, options.max_cold_pages);
        // Two forms of the same knob: a single ceiling for every full-attention layer, or
        // separable per-range ceilings ("0-7:8,8-63:4.5") which the DP minimises per range
        // (globally optimal: additive objective, per-range constraints).
        // Two-score path: when a quality weight is given, the per-tier penalty becomes the
        // weighted sum of the measured quality and speed columns, and the same DP resolves it.
        // The two knobs are orthogonal - one describes the constraint structure, the other the
        // ladder that breaks ties - so all four combinations must exist. The ranges branch
        // comes first for a concrete reason: the range form leaves options.kv_bit_budget_bits
        // at 0, so a scored run that ignored the ranges would resolve every layer against a
        // zero-bit ceiling instead of the ceiling the operator actually gave.
        const bool scored_active = options.kv_quality_weight >= 0.0;
        const bool ranges_given = !options.kv_bit_budget_ranges.empty();
        const std::int32_t e8_limit = product::kKvBitBudgetE8LayerLimit;
        const std::int32_t cold_cap = static_cast<std::int32_t>(cold_pages);
        product::KvTierScoreTable scores = product::kv_bit_budget_default_scores();
        product::KvBitBudgetSolution scored;
        if (scored_active) {
            if (!options.kv_tier_scores.empty()) {
                scores = options.kv_tier_scores.find('\n') != std::string::npos
                             ? product::kv_bit_budget_parse_scores(options.kv_tier_scores)
                             : product::kv_bit_budget_parse_scores([&] {
                                   std::ifstream in(options.kv_tier_scores);
                                   if (!in) {
                                       throw std::invalid_argument(
                                           "kv-tier-scores: cannot open " +
                                           options.kv_tier_scores);
                                   }
                                   return std::string(std::istreambuf_iterator<char>(in),
                                                      std::istreambuf_iterator<char>());
                               }());
            }
            if (!ranges_given) {
                scored = product::kv_bit_budget_solve_scored(
                    full_layers, options.kv_bit_budget_bits, scores, options.kv_quality_weight,
                    e8_limit, cold_cap);
                std::fprintf(stderr,
                             "[kv-score] quality_weight=%.2f achieved_bits=%.2f penalty=%.2f "
                             "spec=%s (speed column is the measured per-tier time cost, quality "
                             "column the measured precision loss)\n",
                             options.kv_quality_weight, scored.achieved_bits, scored.penalty,
                             scored.spec.c_str());
            }
        }
        std::string spec;
        if (ranges_given) {
            const auto ranges =
                product::kv_bit_budget_parse_ranges(options.kv_bit_budget_ranges, full_layers);
            spec = scored_active
                       ? product::kv_bit_budget_scored_ranges(
                             full_layers, ranges, scores, options.kv_quality_weight, e8_limit,
                             cold_cap)
                       : product::kv_bit_budget_spec_ranges(full_layers, ranges, e8_limit,
                                                            cold_cap);
            if (scored_active) {
                std::fprintf(stderr,
                             "[kv-score] quality_weight=%.2f ranges=%s spec=%s (the weighted "
                             "speed/quality ladder ran inside each range's own ceiling)\n",
                             options.kv_quality_weight, options.kv_bit_budget_ranges.c_str(),
                             spec.c_str());
            }
        } else {
            spec = scored_active
                       ? scored.spec
                       : product::kv_bit_budget_spec(full_layers, options.kv_bit_budget_bits,
                                                     e8_limit, cold_cap);
        }
        // Split the DP plan. "cold" is NOT a --kv-layer-storage tier (S12 note):
        // cold-planned layers keep a HOT window at NVFP4 (cold slots hold
        // requantized E2M1-family data, decoder_state.cpp cold-pool note), the
        // remaining ranges feed the existing override path verbatim.
        std::string hot_spec;
        std::string cold_ranges;
        std::size_t cursor = 0;
        while (cursor < spec.size()) {
            const std::size_t comma = spec.find(',', cursor);
            const std::string item =
                spec.substr(cursor, comma == std::string::npos ? spec.size() - cursor
                                                               : comma - cursor);
            const std::size_t colon = item.rfind(':');
            const std::string ranges = item.substr(0, colon);
            const std::string tier = item.substr(colon + 1);
            const bool is_cold = tier == "cold";
            if (!hot_spec.empty()) { hot_spec += ","; }
            if (is_cold) {
                if (!cold_ranges.empty()) { cold_ranges += ","; }
                cold_ranges += ranges;
                hot_spec += ranges + ":nvfp4";
            } else {
                hot_spec += item;
            }
            if (comma == std::string::npos) { break; }
            cursor = comma + 1;
        }
        const auto table = product::parse_kv_layer_storage(hot_spec);
        for (std::size_t i = 0; i < budget_storage_table.size(); ++i) {
            budget_storage_table[i] = table[i];
        }
        storage_explicit = true;
        std::fprintf(stderr,
                     "[kv-bit-budget] full_attention_layers=%d bits=%.2f ranges=%s "
                     "cold_pages=%u cold_placed=%s hot=%s (cold residency needs "
                     "--cold-policy window|disk; pool sized by --max-cold-pages >= %u)\n",
                     static_cast<int>(full_layers), options.kv_bit_budget_bits,
                     options.kv_bit_budget_ranges.empty() ? "-"
                                                          : options.kv_bit_budget_ranges.c_str(),
                     static_cast<unsigned>(cold_pages),
                     cold_ranges.empty() ? "-" : cold_ranges.c_str(), hot_spec.c_str(),
                     static_cast<unsigned>(cold_pages));
    }

    // Both fp8 spellings name the SAME target tier: KvCacheStorage carries an old
    // Fp8E4M3Row256 (the standalone ops/kv_cache row-scaled codec's name, kept for
    // --kv-dtype / request-log naming) and Fp8Group16 (what product::parse_kv_storage
    // emits for the per-layer spec), and target_kv_cache_profile() maps both to
    // (DType::FP8_E4M3FN, kKvFp8QuantGroup). Only Fp8Group16 used to be handled here, so
    // a table written with the other spelling resolved to DType::BF16 - a silent downgrade
    // that drops the fp8 request instead of building it.
    std::array<DType, 64> layer_overrides{};
    const bool has_override = storage_explicit;
    if (has_override) {        for (std::size_t i = 0; i < layer_overrides.size(); ++i) {
            const auto v = budget_storage_table[i];
            layer_overrides[i] = v == KvCacheStorage::BFloat16
                                     ? DType::BF16
                                     : (v == KvCacheStorage::Int8Group64
                                            ? DType::I8
                                            : (v == KvCacheStorage::Nvfp4Group16
                                                   ? DType::NVFP4
                                                   : (v == KvCacheStorage::Iso3Group16
                                                          ? DType::ISO3
                                                          : (v == KvCacheStorage::E8Group64
                                                                 ? DType::E8Kv
                                                                 : ((v == KvCacheStorage::Fp8Group16 ||
                                                                     v == KvCacheStorage::Fp8E4M3Row256)
                                                                        ? DType::FP8_E4M3FN
                                                                        : DType::BF16)))));
        }
    } else if (options.kv_cache_explicit) {
        // An explicit global --kv-dtype replaces the target's registered per-layer default table
        // for every layer (the table is only consulted when the user pinned layers). Without
        // this the global dtype never reached the KV page geometry (_TODO.md 97).
        layer_overrides.fill(kv_profile.dtype);
    } else if constexpr (Variant::supports_per_layer_kv_defaults) {
        layer_overrides = Variant::default_layer_kv_dtypes(
            weights_profile);
    }
    if (tier_plan.has_value()) {
        // Stage 2: mode and cold, judged on the EFFECTIVE dtype of every layer. A
        // BF16 override slot inherits the global dtype, exactly as
        // PagedKVCache::plan_cache resolves it (decoder_state.cpp layer_dtype()), so
        // resolving it here is what makes the check describe the built plan and not
        // just the option text. The per-layer class carries the two facts the gates
        // need: whether the layer sits on a fusion tier (mode=pure) and which
        // cold-slot codec it can feed (cold=), mirroring the two branches of
        // enqueue_cold_compressions (program_impl.h) and the all-INT8 gate above them.
        const std::int32_t tier_layers =
            static_cast<std::int32_t>(TextConfig::full_attention_layers());
        std::array<product::KvLayerClass, kKvLayerStorageSlots> tier_classes{};
        for (std::size_t i = 0; i < tier_classes.size(); ++i) {
            const DType selected =
                layer_overrides[i] == DType::BF16 ? kv_profile.dtype : layer_overrides[i];
            tier_classes[i] = product::kv_layer_class_of(selected);
        }
        const bool tier_cold_pool =
            effective_cold_pages(options.cold_policy, options.cold_keep_tokens,
                                 options.max_cold_pages) > 0;
        const std::string tier_report = product::kv_tier_formats_check(
            *tier_plan, tier_classes, tier_layers, tier_cold_pool);
        std::fprintf(stderr, "%s\n", tier_report.c_str());
    }
    SequencePlanningInputs inputs{
        .weights_profile     = weights_profile,
        .capacity            = options.max_context,
        .kv_capacity_tokens  = options.kv_capacity.mode == KvCapacityMode::Explicit
                                   ? std::optional<std::uint32_t>(options.kv_capacity.explicit_tokens)
                                   : std::nullopt,
        .max_concurrency     = options.max_concurrency,
        .prefill_chunk       = std::min(options.prefill_chunk, options.max_context),
        .draft_window        = options.speculative.backend == SpeculativeBackend::DFlash2 &&
                                       options.speculative.draft_tokens == 0
                                       ? 7U
                                       : options.speculative.draft_tokens,
        .speculative_backend = options.speculative.backend,
        .kv_dtype            = kv_profile.dtype,
        .kv_quant_group      = kv_profile.quant_group,
        .layer_kv_dtypes     = layer_overrides,
        .kv_residual_layers  = options.kv_residual_explicit
                                   ? options.kv_residual_layers
                                   : std::array<bool, 64>{},
        .kv_v_codec          = options.kv_v_codec,
        .kv_rotation_off     = options.kv_rotation_explicit && options.kv_rotation_off,
        .kv_row_scale_spec   = options.kv_row_scale_explicit ? options.kv_row_scale_spec
                                                             : std::string{},
        .proposal_head       = options.speculative.proposal_head,
        .features            = qwen3_6::startup_features(options),
        .use_cuda_graph      = options.use_cuda_graph,
        .graph_capture_ceiling = options.graph_capture_ceiling,
        .causal_scoring      = options.purpose == EnginePurpose::CausalScoring,
        .cold_policy         = options.cold_policy,
        // Designators must follow declaration order (layouts.h): max_cold_pages
        // is declared before cold_keep_tokens.
        .max_cold_pages      = options.max_cold_pages,
        .cold_keep_tokens    = options.cold_keep_tokens,
        .cold_host_bytes     = options.cold_host_bytes,
        .cold_disk_path      = options.cold_disk_path,
        .cold_disk_bytes     = options.cold_disk_bytes,
        .device              = options.device,
        .context_cache       = options.context_cache,
    };
    const std::uint32_t logical_pages = page_count(inputs.capacity);
    // The device page pool normally covers the full max_context; an explicit
    // --kv-capacity below max_context instead floors the pool at that size so
    // the rope domain (4x under YaRN) can exceed what the pool can hold.
    std::uint32_t minimum_pages = std::max(logical_pages, inputs.max_concurrency);
    if (inputs.kv_capacity_tokens) {
        minimum_pages = std::max(page_count(*inputs.kv_capacity_tokens), inputs.max_concurrency);
    }
    const std::uint64_t maximum_pages64 =
        static_cast<std::uint64_t>(inputs.max_concurrency) * logical_pages;
    if (maximum_pages64 > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("maximum Main KV page count exceeds uint32");
    }
    const auto maximum_pages = static_cast<std::uint32_t>(maximum_pages64);

    auto planner     = std::make_unique<qwen3_6::detail::SequencePlannerImpl<Variant>>();
    planner->inputs  = inputs;
    planner->minimum = build_sequence_candidate(inputs, minimum_pages);
    planner->curve   = runtime::SequenceCapacityCurve{
          .main_page_tokens                     = static_cast<std::uint32_t>(kPagedKVPageSize),
          .minimum_main_page_groups             = minimum_pages,
          .maximum_main_page_groups             = maximum_pages,
          .minimum_device_reservation_bytes     = planner->minimum->device_reservation_bytes,
          .bytes_per_additional_main_page_group = 0,
    };
    if (minimum_pages < maximum_pages) {
        auto adjacent = build_sequence_candidate(inputs, minimum_pages + 1U);
        if (adjacent->device_reservation_bytes <= planner->minimum->device_reservation_bytes) {
            throw std::logic_error("Qwen3.6 sequence layout has a nonpositive KV capacity stride");
        }
        planner->curve.bytes_per_additional_main_page_group =
            adjacent->device_reservation_bytes - planner->minimum->device_reservation_bytes;
    }
    return planner;
}

std::unique_ptr<SequencePlanImpl>
finalize_sequence_plan_impl(std::unique_ptr<qwen3_6::detail::SequencePlannerImpl<Variant>> planner,
                            std::uint32_t main_page_groups) {
    if (planner == nullptr || planner->minimum == nullptr) {
        throw std::invalid_argument("Qwen3.6 sequence planner is empty");
    }
    const std::size_t expected = planner->curve.reservation_bytes(main_page_groups);
    std::unique_ptr<SequencePlanImpl> plan;
    if (main_page_groups == planner->curve.minimum_main_page_groups) {
        plan = std::move(planner->minimum);
    } else {
        plan = build_sequence_candidate(planner->inputs, main_page_groups);
    }
    if (plan->device_reservation_bytes != expected) {
        throw std::logic_error(
            "Qwen3.6 physical sequence layout is not affine in Main KV page capacity");
    }
    return plan;
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS
