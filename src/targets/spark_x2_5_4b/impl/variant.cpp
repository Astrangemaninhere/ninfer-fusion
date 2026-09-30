#include "targets/spark_x2_5_4b/impl/variant.h"

// src/targets/spark_x2_5_4b/impl/variant.cpp
//
// The execution leaves, and the macro instantiation that closes the Qwen3.6 family
// runtime over this namespace.  Derived from
// src/targets/muse_glimmer_30b/impl/variant.cpp -- the same three leaves are
// meaningful and the same nine are refusals -- with the Spark-specific bodies marked
// `SPARK:`.
//
// The nine refusals are NOT placeholders-until-later: they are the declared interface
// the shared runtime compiles against for every variant, and a target that has no GDN
// and no MTP must still present them.  muse_glimmer_30b/impl/variant.cpp:397-409 is
// the precedent for the wording, and it is copied here because the alternative -- a
// silent zero -- is exactly what a reader would mistake for "wired".

#include "ninfer/ops/gelu_mul.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_pair.h"
#include "ninfer/ops/residual_add.h"

#include <algorithm>
#include <stdexcept>

#define NINFER_QWEN36_VARIANT    ::ninfer::targets::spark_x2_5_4b::detail::Variant
#define NINFER_QWEN36_RUNTIME_NS spark_x2_5_4b_runtime
#include "targets/qwen3_6/impl/runtime/instantiate.h"

namespace ninfer::targets::spark_x2_5_4b::detail {

std::array<DType, 64> Variant::default_layer_kv_dtypes(WeightsProfile) {
    // supports_per_layer_kv_defaults is false, so the planner never calls this.  A
    // uniform BF16 table is the measured default (serve --kv-dtype bf16) and the only
    // stored format v1 has: the artifact is BF16 throughout
    // (tools/convert/spark_x2_5_4b/inventory.py declares one format, BF16, for all
    // 290 objects).
    std::array<DType, 64> table{};
    table.fill(DType::BF16);
    return table;
}

namespace {

std::vector<GraphExecutionProfile>
graph_profiles_through(std::uint32_t max_frontier,
                       const std::vector<std::uint32_t>& preferred_ends) {
    std::vector<GraphExecutionProfile> out;
    std::uint32_t begin = 0;
    for (const std::uint32_t preferred_end : preferred_ends) {
        if (begin > max_frontier) { break; }
        const std::uint32_t end = std::min(preferred_end, max_frontier);
        out.push_back({begin, end});
        if (end == max_frontier) { return out; }
        begin = end + 1;
    }
    if (begin <= max_frontier) { out.push_back({begin, max_frontier}); }
    return out;
}

void validate_token_interval(std::int32_t first, std::int32_t last) {
    if (first <= 0 || last < first) {
        throw std::invalid_argument("invalid target leaf token interval");
    }
}

std::vector<GraphExecutionProfile>
dflash_base_profiles(std::uint32_t capacity, std::uint32_t draft_window) {
    if (draft_window == 0 || capacity == 0) { return {}; }
    const std::uint32_t block        = draft_window + 1;
    const std::uint32_t max_frontier = capacity - 1;
    std::vector<std::uint32_t> ends{
        96U, 127U, 511U, 1023U, 2047U, 4095U, 8191U, 16383U, 32767U, 65536U, 131072U, 196608U,
    };
    const auto add_target_boundary = [&](std::uint32_t visible_end) {
        if (visible_end >= block) { ends.push_back(visible_end - block); }
    };
    for (const std::uint32_t visible_end : {128U, 512U, 2048U, 4096U, 8198U, 16390U, 32768U}) {
        add_target_boundary(visible_end);
    }
    if (draft_window >= 6 && draft_window <= 15) {
        add_target_boundary(draft_window <= 11 ? 512U : 1024U);
    }
    std::sort(ends.begin(), ends.end());
    ends.erase(std::unique(ends.begin(), ends.end()), ends.end());
    return graph_profiles_through(max_frontier, ends);
}

bool dflash_target_uses_chunked_small_t(std::uint32_t draft_window, std::uint32_t batch_size,
                                        std::uint32_t max_visible_keys) {
    const std::uint32_t tokens = draft_window + 1;
    if (tokens <= 6) { return false; }
    if (batch_size > 1) { return true; }
    const std::uint32_t prompt_visible_limit = tokens <= 12 ? 512U : 1024U;
    return max_visible_keys > prompt_visible_limit;
}

} // namespace

// --------------------------------------------------------------------------- //
// graph profiles.  The ordinary ladder is copied verbatim from the two sibling
// softmax targets: it is a measured set of split-policy transitions for the SHARED
// generator (text_context_impl.h's producer grid), not a property of the checkpoint,
// so re-deriving it for Spark would be inventing a second measurement of the same
// thing.  The three draft ladders are unreachable (draft_window is always 0 for a
// target whose Variant reports supports_dflash/dflash2 = false) and are kept as the
// same declared interface the refusals below are kept as.
// --------------------------------------------------------------------------- //
std::vector<GraphExecutionProfile> Variant::ordinary_graph_profiles(std::uint32_t capacity) {
    // E+1 is the one-token visible window. Early ranges limit empty producer CTAs; later ranges
    // follow measured split-policy transitions until the producer grid reaches its fixed cap.
    return graph_profiles_through(capacity - 1, {127, 511, 2047, 4095, 8197, 16389, 32767});
}

std::vector<GraphExecutionProfile> Variant::mtp_graph_profiles(std::uint32_t capacity,
                                                               std::uint32_t draft_window,
                                                               bool ladder_capture) {
    if (draft_window == 0 || capacity == 0) { return {}; }
    std::vector<std::uint32_t> ends;
    const auto add_shifted = [&](std::uint32_t visible_end, std::uint32_t offset) {
        if (visible_end >= offset) { ends.push_back(visible_end - offset); }
    };
    for (const std::uint32_t visible_end : {128U, 512U, 2048U, 4096U, 8198U, 16390U, 32768U}) {
        add_shifted(visible_end, 2 * draft_window);
    }
    if (ladder_capture) {
        for (const std::uint32_t visible_end : {128U, 160U, 512U, 1029U, 2054U, 8198U}) {
            add_shifted(visible_end, draft_window + 1);
        }
    } else if (draft_window == 3) {
        add_shifted(1029, draft_window + 1);
    } else if (draft_window == 4) {
        for (const std::uint32_t visible_end : {128U, 512U, 1029U}) {
            add_shifted(visible_end, draft_window + 1);
        }
    } else if (draft_window == 5) {
        for (const std::uint32_t visible_end : {128U, 160U, 2054U, 8198U}) {
            add_shifted(visible_end, draft_window + 1);
        }
    }
    std::sort(ends.begin(), ends.end());
    ends.erase(std::unique(ends.begin(), ends.end()), ends.end());
    return graph_profiles_through(capacity - 1, ends);
}

std::vector<GraphExecutionProfile> Variant::dflash_graph_profiles(std::uint32_t capacity,
                                                                   std::uint32_t draft_window,
                                                                   std::uint32_t batch_size) {
    std::vector<GraphExecutionProfile> profiles = dflash_base_profiles(capacity, draft_window);
    for (GraphExecutionProfile& profile : profiles) {
        const std::uint32_t target_max = static_cast<std::uint32_t>(std::min<std::uint64_t>(
            capacity, static_cast<std::uint64_t>(profile.max) + draft_window + 1ULL));
        const bool chunked_target =
            dflash_target_uses_chunked_small_t(draft_window, batch_size, target_max);
        profile.topology_class = chunked_target ? 2U : 0U;
    }
    return profiles;
}

std::vector<GraphExecutionProfile> Variant::dflash2_graph_profiles(std::uint32_t capacity,
                                                                   std::uint32_t draft_window,
                                                                   std::uint32_t batch_size) {
    std::vector<GraphExecutionProfile> profiles = dflash_base_profiles(capacity, draft_window);
    for (GraphExecutionProfile& profile : profiles) {
        const std::uint32_t target_max = static_cast<std::uint32_t>(std::min<std::uint64_t>(
            capacity, static_cast<std::uint64_t>(profile.max) + draft_window + 1ULL));
        const bool split_swa           = profile.max > 96U;
        const bool chunked_target =
            dflash_target_uses_chunked_small_t(draft_window, batch_size, target_max);
        profile.topology_class = (chunked_target ? 2U : 0U) | (split_swa ? 1U : 0U);
    }
    return profiles;
}

// --------------------------------------------------------------------------- //
// the leaves that exist
// --------------------------------------------------------------------------- //

void Variant::attention_projection(const Tensor& hidden,
                                   const FullAttentionProjectionWeights& weights, Tensor& query,
                                   Tensor& gate, Tensor& key, Tensor& value, qwen3_6::TextPhase,
                                   WorkspaceArena& workspace, cudaStream_t stream) {
    // SPARK: four independent projections, in the order the family's views name them.
    //
    // The checkpoint ships ONE `self_attn.q_k_v_proj` of 6144 rows = 4096 q + 1024 k +
    // 1024 v, and one `self_attn.g_proj` of 16 rows.  The engine's fused
    // attn_input_proj op is a fixed-geometry qwen op (parent [14336,5120], q/gate
    // [6144,T], k/v [1024,T]; include/ninfer/ops/attn_input_proj.h:18-50 and
    // src/ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.cpp:30-31), so it is not
    // usable here; muse reached the same conclusion and split into ops::linear calls.
    //
    // ⚠ THE GATE IS NOT THE SAME SHAPE AS ITS SIBLINGS, AND THE FAMILY HANDED IT IN
    //   AS IF IT WERE.  Read this before touching the three lines:
    //
    //   * what the family passes: text_context_impl.h:1180-1187 builds
    //       `Tensor gate = projection.gate.view({kCfg.head_dim, kCfg.n_q, T});`
    //     then `gate_flat = gate.view({kCfg.q_size, T})` and passes THAT as this
    //     leaf's `gate`.  So `gate.ne[0] == query_size == 4096` on arrival.
    //   * what Spark's gate object is: [attention_gate_rows, hidden] = [16, 2560], so
    //     the projection only has 16 rows to write.
    //   * the view below is therefore a SHRINK of the buffer the family handed in: the
    //     gate lands in rows 0..15, exactly where a headwise gate belongs, and rows
    //     16..4095 of that buffer are left as they were.
    //   * text_context_impl.h:1284 then calls `ops::sigmoid_mul(gate, a, s)` with
    //     `gate` still viewed {head_dim, n_q, T}.  ops::sigmoid_mul chooses its
    //     headwise route BY SHAPE and the test it uses
    //     (src/ops/wrapper/sigmoid_mul.cpp:222-225) is `gate.ne[0] == x.ne[1]`, i.e.
    //     256 == 16, which is FALSE for a {256,16,T} gate.  It therefore falls
    //     through to the per-element route, whose shape test ACCEPTS the pair
    //     ({256,16,T} against {256,16,T}) and multiplies `a` by rows 16..4095 of the
    //     gate buffer -- numbers this leaf never wrote.  That is a SILENT WRONG
    //     NUMBER, and it is why impl/package.cpp refuses at plan_load instead of
    //     letting the engine start.
    //   * the fix is one family-side branch: when the target declares a headwise
    //     gate, view gate as {n_q, T} and not {head_dim, n_q, T}.  Named as companion
    //     land (1) in dl/sparktarget/REPORT.md, with the other two hooks.
    //
    // The three lines below are written in the FINAL correct form, i.e. they are
    // right the moment (1) lands and they are the reason it must land.
    const int cols = hidden.ne[1];
    Tensor gate_rows = gate.view({TextConfig::attention_gate_rows, cols});
    ops::linear(hidden, weights.query, query, stream);
    ops::linear(hidden, weights.key, key, stream);
    ops::linear(hidden, weights.value, value, stream);
    ops::linear(hidden, weights.gate, gate_rows, stream);
    (void)workspace;
}

void Variant::attention_output_projection(const Tensor& attention, const Weight& weight,
                                          Tensor& residual, qwen3_6::TextPhase,
                                          WorkspaceArena& workspace, cudaStream_t stream) {
    // ops::linear + residual, not ops::linear_add: linear_add's registered profiles are
    // qwen geometries, and this projection's is [2560, 4096] with a [4096, T] input --
    // the contraction dimension is the QUERY width, not hidden (the checkpoint's
    // `out_proj` is [hidden, q_heads*head_dim]).  Muse took the same route for the same
    // reason (muse_glimmer_30b/impl/variant.cpp:255-263).
    (void)workspace;
    Tensor o = workspace.alloc(DType::BF16, {TextConfig::hidden, attention.ne[1]});
    ops::linear(attention, weight, o, stream);
    ops::residual_add(o, residual, stream);
}

void Variant::post_mixer(const Tensor& hidden, const PostMixerWeights& weights, Tensor& residual,
                         qwen3_6::TextPhase, WorkspaceArena& workspace, cudaStream_t stream) {
    // SPARK: `down(gelu(gate(x)) * up(x))` with gelu = the EXACT erf form, not the tanh
    // approximation and not SiLU.
    //
    // Three separate differences from the sibling, all measured:
    //   * the activation is gelu (config.json hidden_act = "gelu"), so the fused
    //     ops::linear_swiglu path 27b uses is unavailable -- it hard-codes SiLU
    //     (include/ninfer/ops/linear_swiglu.h:38-53) -- and muse's ops::silu_mul
    //     substitution would compute the wrong function silently.
    //   * the two-input multiply needs an op of its own.  It EXISTS:
    //     include/ninfer/ops/gelu_mul.h:27,
    //       `ideal[i] = gelu(gate[i]) * up[i]`, exact erf, gate and up same-shaped.
    //     One difference in its contract matters here: unlike ops::silu_mul it does NOT
    //     write in place -- it takes three tensors, so this body allocates `act`.
    //   * the leaf is therefore four calls plus a residual, and `act` and the down
    //     output are live at the same time as gate and up (hence the workspace number
    //     below, which is muse's with the same reasoning).
    auto scope     = workspace.scope();
    const int cols = hidden.ne[1];
    Tensor gate = workspace.alloc(DType::BF16, {TextConfig::intermediate, cols});
    Tensor up   = workspace.alloc(DType::BF16, {TextConfig::intermediate, cols});
    ops::linear(hidden, weights.gate, gate, stream);
    ops::linear(hidden, weights.up, up, stream);
    Tensor act = workspace.alloc(DType::BF16, {TextConfig::intermediate, cols});
    ops::gelu_mul(gate, up, act, stream);
    Tensor o = workspace.alloc(DType::BF16, {TextConfig::hidden, cols});
    ops::linear(act, weights.down, o, stream);
    ops::residual_add(o, residual, stream);
}

void Variant::post_mixer_double_norm(const Tensor&, const PostMixerWeights&, const Tensor&,
                                     Tensor&, qwen3_6::TextPhase, WorkspaceArena&,
                                     cudaStream_t) {
    // Single-norm architecture.  TextConfig declares neither attn_out_post_norm() nor
    // mlp_out_post_norm(), so the family's detection idioms
    // (text_context.h:193-206) are compile-time false and this leaf is unreachable.
    // Declared because the shared runtime compiles for every variant.
    throw std::logic_error("post_mixer_double_norm is not enabled for spark_x2_5_4b");
}

void Variant::mtp_post_mixer(const Tensor&, const MtpPostMixerWeights&, Tensor&,
                             WorkspaceArena&, cudaStream_t) {
    throw std::logic_error("MTP is not enabled for spark_x2_5_4b");
}

// --------------------------------------------------------------------------- //
// the leaves that refuse.  GDN: the family's run_layers has exactly two branches and
// TextConfig::gdn_layers() is 0, so the gdn branch is never entered; these exist
// because the shared runtime names them.  MTP: mtp_layers is 0 and the checkpoint has
// no MTP head (the index's 290 tensors are 36*8 + embedding + final norm), so
// --spec has nothing to resolve to and Package::resolved_auto_speculative below
// resolves Auto to None.
// --------------------------------------------------------------------------- //

void Variant::mtp_attention_projection(const Tensor&, const MtpAttentionProjectionWeights&,
                                       Tensor&, Tensor&, Tensor&, Tensor&, WorkspaceArena&,
                                       cudaStream_t) {
    throw std::logic_error("MTP is not enabled for spark_x2_5_4b");
}

void Variant::mtp_kv_projection(const Tensor&, const MtpAttentionProjectionWeights&, Tensor&,
                                Tensor&, WorkspaceArena&, cudaStream_t) {
    throw std::logic_error("MTP is not enabled for spark_x2_5_4b");
}

void Variant::mtp_q_gate_projection(const Tensor&, const MtpAttentionProjectionWeights&, Tensor&,
                                    Tensor&, WorkspaceArena&, cudaStream_t) {
    throw std::logic_error("MTP is not enabled for spark_x2_5_4b");
}

void Variant::gdn_input_projection(const Tensor&, const GdnProjectionWeights&, Tensor&, Tensor&,
                                   qwen3_6::TextPhase, WorkspaceArena&, cudaStream_t) {
    throw std::logic_error("GDN is not enabled for spark_x2_5_4b");
}

void Variant::gdn_input_projection_snapshot(const Tensor&, const GdnProjectionWeights&,
                                            const Tensor&, Tensor&, const Tensor&, const Tensor&,
                                            const Tensor&, Tensor&, Tensor&, Tensor&, Tensor&,
                                            qwen3_6::TextPhase, WorkspaceArena&, cudaStream_t) {
    throw std::logic_error("GDN is not enabled for spark_x2_5_4b");
}

void Variant::gdn_input_projection_record(const Tensor&, const GdnProjectionWeights&,
                                          const Tensor&, const Tensor&, const Tensor&,
                                          const Tensor&, Tensor&, Tensor&, Tensor&, Tensor&,
                                          Tensor&, qwen3_6::TextPhase, WorkspaceArena&,
                                          cudaStream_t) {
    throw std::logic_error("GDN is not enabled for spark_x2_5_4b");
}

void Variant::gdn_output_projection(const Tensor&, const Weight&, Tensor&, qwen3_6::TextPhase,
                                    WorkspaceArena&, cudaStream_t) {
    throw std::logic_error("GDN is not enabled for spark_x2_5_4b");
}

void Variant::gdn_norm_control_projection(const Tensor&, const Tensor&, float,
                                          const GdnProjectionWeights&, Tensor&, Tensor&, Tensor&,
                                          WorkspaceArena&, cudaStream_t) {
    throw std::logic_error("GDN is not enabled for spark_x2_5_4b");
}

std::size_t Variant::mtp_attention_projection_workspace_capacity_bytes(std::int32_t,
                                                                       std::int32_t) {
    // mtp_layers is 0; capacity 0, the same number muse returns for the same reason.
    return 0;
}

std::size_t Variant::mtp_kv_projection_workspace_capacity_bytes(std::int32_t, std::int32_t) {
    return 0;
}

std::size_t Variant::mtp_q_gate_projection_workspace_capacity_bytes(std::int32_t, std::int32_t) {
    return 0;
}

std::size_t Variant::attention_projection_workspace_capacity_bytes(WeightsProfile,
                                                                   qwen3_6::TextPhase,
                                                                   std::int32_t first,
                                                                   std::int32_t last) {
    validate_token_interval(first, last);
    // Four independent ops::linear calls, each of which owns whatever transient it
    // needs through its own registered profile; this leaf asks for nothing extra.
    return 0;
}

std::size_t Variant::attention_output_projection_workspace_capacity_bytes(
    WeightsProfile, qwen3_6::TextPhase, std::int32_t first, std::int32_t last) {
    validate_token_interval(first, last);
    return 0;
}

std::size_t Variant::gdn_input_projection_workspace_capacity_bytes(WeightsProfile,
                                                                   qwen3_6::TextPhase,
                                                                   std::int32_t, std::int32_t) {
    return 0;
}

std::size_t Variant::gdn_input_projection_snapshot_workspace_capacity_bytes(
    WeightsProfile, qwen3_6::TextPhase, std::int32_t, std::int32_t, std::int32_t) {
    return 0;
}

std::size_t Variant::gdn_input_projection_record_workspace_capacity_bytes(
    WeightsProfile, qwen3_6::TextPhase, std::int32_t, std::int32_t, std::int32_t) {
    return 0;
}

std::size_t Variant::gdn_output_projection_workspace_capacity_bytes(WeightsProfile,
                                                                    qwen3_6::TextPhase,
                                                                    std::int32_t, std::int32_t) {
    return 0;
}

std::size_t Variant::gdn_norm_control_projection_workspace_capacity_bytes(std::int32_t,
                                                                          std::int32_t) {
    return 0;
}

std::size_t Variant::post_mixer_workspace_capacity_bytes(
    WeightsProfile, qwen3_6::TextPhase, std::int32_t first, std::int32_t last) {
    validate_token_interval(first, last);
    // Spark's gate, up and act are live simultaneously (gelu_mul is not in place, so
    // it needs a third buffer), plus the down-projection output, all BF16, and the
    // layer scope releases them only after the layer completes.  The same accounting
    // muse_glimmer_30b/impl/variant.cpp:481-492 arrived at after returning zero made
    // multi-chunk prefill overflow the arena with std::bad_alloc (_TODO.md 100): three
    // intermediate-width BF16 tensors plus one hidden-width tensor, per token.
    const std::size_t tokens = static_cast<std::size_t>(last);
    return 3ULL * 2ULL * TextConfig::intermediate * tokens +
           2ULL * TextConfig::hidden * tokens;
}

std::size_t Variant::mtp_post_mixer_workspace_capacity_bytes(std::int32_t, std::int32_t) {
    return 0;
}

} // namespace ninfer::targets::spark_x2_5_4b::detail
