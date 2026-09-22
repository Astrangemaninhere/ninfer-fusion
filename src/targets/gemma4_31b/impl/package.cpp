// src/targets/gemma4_31b/impl/package.cpp
//
// The compilation owner of gemma4_31b::validate_stage_a_geometry, declared in
// targets/gemma4_31b/export/ninfer/targets/gemma4_31b/package.h and called from
// src/targets/registry.cpp before its generic "no registered target" refusal.
//
// Host-only: no CUDA header, no device. It reads an already-open artifact's
// object table, which is what makes it testable and compilable without a GPU --
// the same reason src/targets/qwen4_exp/CMakeLists.txt:9-13 gives for putting
// its ngram TU on a plain host compiler.
//
// It is NOT the load path. Nothing here materialises a weight, and no fail here
// can leave a half-loaded model: the function runs before any load plan exists
// (src/targets/registry.cpp:513-514 takes the arch gate first, :520-525 the
// table lookup, and this runs after both).

#include <ninfer/targets/gemma4_31b/package.h>

namespace ninfer::targets::gemma4_31b {

std::string validate_stage_a_geometry(const artifact::Reader& reader) {
    using TC = detail::TextConfig;

    const std::vector<std::uint64_t> embed =
        detail::tensor_shape(reader, "text/token_embedding", "hidden/vocab");
    const std::vector<std::uint64_t> sliding_q =
        detail::tensor_shape(reader, "text/layers/0/attention/query", "sliding q rows");
    const std::vector<std::uint64_t> sliding_k =
        detail::tensor_shape(reader, "text/layers/0/attention/key", "sliding kv rows");
    const std::vector<std::uint64_t> full_q =
        detail::tensor_shape(reader, "text/layers/5/attention/query", "full q rows");
    const std::vector<std::uint64_t> full_k =
        detail::tensor_shape(reader, "text/layers/5/attention/key", "full kv rows");
    const std::vector<std::uint64_t> gate =
        detail::tensor_shape(reader, "text/layers/0/mlp/gate", "intermediate");
    const std::vector<std::uint64_t> input_norm =
        detail::tensor_shape(reader, "text/layers/0/input_norm", "hidden");
    const std::vector<std::uint64_t> layer_scalar =
        detail::tensor_shape(reader, "text/layers/0/layer_scalar", "per-layer scalar");

    if (embed.size() != 2 || sliding_q.size() != 2 || sliding_k.size() != 2 ||
        full_q.size() != 2 || full_k.size() != 2 || gate.size() != 2 ||
        input_norm.size() != 1 || layer_scalar.size() != 1) {
        throw std::runtime_error(
            "gemma4-31b stage (a): geometry tensors have the wrong rank "
            "(text/token_embedding, text/layers/{0,5}/attention/{query,key} and "
            "text/layers/0/mlp/gate must be rank 2; text/layers/0/input_norm and "
            "text/layers/0/layer_scalar rank 1)");
    }

    // shared geometry
    detail::expect_dim("hidden", embed[1], static_cast<std::uint64_t>(TC::hidden),
                       "text/token_embedding");
    detail::expect_dim("vocab", embed[0], static_cast<std::uint64_t>(TC::vocab),
                       "text/token_embedding");
    detail::expect_dim("hidden(query)", sliding_q[1], static_cast<std::uint64_t>(TC::hidden),
                       "text/layers/0/attention/query");
    detail::expect_dim("hidden(gate)", gate[1], static_cast<std::uint64_t>(TC::hidden),
                       "text/layers/0/mlp/gate");
    detail::expect_dim("intermediate", gate[0], static_cast<std::uint64_t>(TC::intermediate),
                       "text/layers/0/mlp/gate");
    detail::expect_dim("hidden(input_norm)", input_norm[0],
                       static_cast<std::uint64_t>(TC::hidden), "text/layers/0/input_norm");
    detail::expect_dim("layer_scalar rows", layer_scalar[0], 1,
                       "text/layers/0/layer_scalar");

    // the sliding geometry, off layer 0
    detail::expect_dim("sliding q rows", sliding_q[0],
                       static_cast<std::uint64_t>(TC::query_rows_at(0)),
                       "text/layers/0/attention/query");
    detail::expect_dim("sliding kv rows", sliding_k[0],
                       static_cast<std::uint64_t>(TC::kv_rows_at(0)),
                       "text/layers/0/attention/key");

    // the full geometry, off layer 5 -- the first full_attention slot. These two
    // are the quantities no kernel can serve
    // (src/ops/kernel/gqa_attention_geometry.cuh:15), so a mismatch here and a
    // mismatch there are different diagnoses and both are named.
    detail::expect_dim("full q rows", full_q[0],
                       static_cast<std::uint64_t>(TC::query_rows_at(5)),
                       "text/layers/5/attention/query");
    detail::expect_dim("full kv rows", full_k[0],
                       static_cast<std::uint64_t>(TC::kv_rows_at(5)),
                       "text/layers/5/attention/key");

    // layers: max N over "text/layers/<N>/..." plus one. The plan's object names
    // are the only place the count of layers is observable from an artifact.
    std::uint64_t max_layer = 0;
    for (const artifact::ObjectDescriptor& object : reader.objects()) {
        const std::string_view name = artifact::object_name(object);
        constexpr std::string_view prefix = "text/layers/";
        if (name.rfind(prefix, 0) != 0) { continue; }
        std::uint64_t value   = 0;
        std::size_t pos       = prefix.size();
        bool any_digit        = false;
        while (pos < name.size() && name[pos] >= '0' && name[pos] <= '9') {
            value = value * 10 + static_cast<std::uint64_t>(name[pos] - '0');
            ++pos;
            any_digit = true;
        }
        if (any_digit && value > max_layer) { max_layer = value; }
    }
    detail::expect_dim("layers", max_layer + 1, static_cast<std::uint64_t>(TC::layers),
                       "text/layers/<N>/* names");

    return "geometry cross-check PASS: hidden=" + std::to_string(TC::hidden) +
           " vocab=" + std::to_string(TC::vocab) +
           " layers=" + std::to_string(TC::layers) +
           " sliding(q=" + std::to_string(TC::query_rows_at(0)) +
           ",kv=" + std::to_string(TC::kv_rows_at(0)) +
           ",head_dim=" + std::to_string(TC::head_dim_at(0)) +
           ",rotary=" + std::to_string(TC::rotary_dim_at(0)) + ")" +
           " full(q=" + std::to_string(TC::query_rows_at(5)) +
           ",kv=" + std::to_string(TC::kv_rows_at(5)) +
           ",head_dim=" + std::to_string(TC::head_dim_at(5)) +
           ",rotary=" + std::to_string(TC::rotary_dim_at(5)) + ")" +
           " intermediate=" + std::to_string(TC::intermediate) +
           " schedule=" + std::to_string(TC::full_attention_layers()) + " full + " +
           std::to_string(TC::swa_attention_layers()) + " sliding" +
           " layer_scalar=present";
}

} // namespace ninfer::targets::gemma4_31b
