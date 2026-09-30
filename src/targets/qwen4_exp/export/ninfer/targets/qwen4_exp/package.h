#pragma once

// S37 阶段 (a)：FlashNext (qwen4_exp) 目标的身份头 + 几何交叉核验。
//
// 只有常量和一个纯函数：运行时类型（Package / LoadedModel / Frontend / SequencePlan /
// Program / Variant）都还不存在，属阶段 (b) 起的工作（分阶段计划见
// _collab/B_s37_flashnext_p1.md §2：a 身份+可编译+几何核验 → b BF16 单 QSA 层前向 →
// c MoE 512x10 → d GDN 36 层 + hyper-connection → e PLE → f MTP）。
//
// 为什么要有一次形状交叉核验
// --------------------------
// artifact manifest 里没有语义几何：src/artifact/reader.cpp 的 parse_tensor 只认
// name/kind/shape/format/layout/offset/bytes。于是 **张量形状是 artifact 侧唯一的
// 几何来源**。impl/config.h 里的常量是 spec 侧期望值；下面每个量都必须能从一个具名
// 张量推出来，并且与常量相等。不符或缺失 -> 抛 std::runtime_error，报文里同时给出
// 两侧数值（这样 S28 writer 换了名字或形状时，报错直接指向要改的那一侧）。
// **从不静默信任常量。**
//
// 名字与形状的出处（不是猜的）
// ----------------------------
//   * artifact 形状规则：tools/archkit/flashnext_convert.py:119-179 artifact_shape()，
//     其注释明确"形状一律用 (out, in) 的 artifact 序书写"（:39）。
//   * 契约张量名：tools/archkit/flashnext_bindings.py 的 e(...) 条目与
//     flashnext_convert.py 的 _rule 表（:48-103）。
//   * 端到端装载映射：src/artifact/reader.h（Reader::find / object_name）。
//
// 调用时机
// --------
// registry.cpp 在命中 qwen4-exp 身份之后、通用拒绝之前调用它，把判词拼进"拒绝装载
// stub 目标"的报文。于是：非 qwen4_exp 的 artifact 完全走原来的路径（行为不变）；
// qwen4_exp 的 artifact 一旦被误当可装载目标，拿到的是具名几何判词而不是静默装载或
// 者内核里的崩溃；几何不符的 artifact 当场报双侧数值。

#include "artifact/reader.h"
#include "targets/qwen4_exp/impl/config.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace ninfer::targets::qwen4_exp {

inline constexpr std::string_view kModelId      = "qwen4-exp";
inline constexpr std::string_view kTargetKey    = "qwen4-exp";
inline constexpr std::string_view kRuntimeStage = "a-identity-only";

// ---------------------------------------------------------------------------------------
// STAGE (b) COMPLETENESS -- the predicate src/targets/registry.cpp's load gate reads.
// ---------------------------------------------------------------------------------------
// PUBLISHED BY THE BUILD. This directory's stage_b.cmake derives it from an existence test
// over the three sources CMakeLists.txt names as stage (b) (impl/package.cpp,
// impl/variant.cpp, impl/load/bindings.cpp) and CMakeLists.txt turns that verdict into the
// two definitions below.
//
// WHERE THE FAIL-SAFE DIRECTION IS: a TU that sees NEITHER definition -- a test, a tool, or
// any future translation unit that compiles this header without the engine's definitions --
// reads false, and the registry refuses BY NAME. "The gate is open" is never a default; it is
// stated by the build that owns the implementation.
//
// WHY IT IS A PREDICATE AND NOT THE UNCONDITIONAL throw THAT WAS HERE: that throw was taken
// whether or not a runtime existed, so landing stage (b) would have needed a second edit in
// another file before anything could load. Inverted, the open arm is the arm that has to
// exist, and it carries a static_assert that demands the family's registration row.
#if defined(NINFER_QWEN4_EXP_STAGE_B_COMPLETE)
inline constexpr bool kStageBImplemented = NINFER_QWEN4_EXP_STAGE_B_COMPLETE != 0;
#else
inline constexpr bool kStageBImplemented = false;
#endif

// PER-SOURCE INTEGER FLAGS, NOT ONE STRING LIST, and the reason is measured rather than
// stylistic: a definition carrying a LIST cannot be written as `-DNAME=a.cpp b.cpp` -- the
// preprocessor body is then a token sequence, not a string literal, and the refusal that is
// supposed to NAME the missing files fails to compile ("error: 'MISS_A' was not declared in
// this scope", measured). Three integers need no quoting, and the list is composed below.
//
// The fail-safe direction is the same as the predicate's: a TU that sees NONE of the three
// definitions reads all three as 0, i.e. "every stage (b) source is absent", and refuses.
#ifndef NINFER_QWEN4_EXP_STAGE_B_HAS_IMPL_PACKAGE
#define NINFER_QWEN4_EXP_STAGE_B_HAS_IMPL_PACKAGE 0
#endif
#ifndef NINFER_QWEN4_EXP_STAGE_B_HAS_IMPL_VARIANT
#define NINFER_QWEN4_EXP_STAGE_B_HAS_IMPL_VARIANT 0
#endif
#ifndef NINFER_QWEN4_EXP_STAGE_B_HAS_IMPL_BINDINGS
#define NINFER_QWEN4_EXP_STAGE_B_HAS_IMPL_BINDINGS 0
#endif

// The stage (b) sources this build did NOT find, named one by one. Empty exactly when
// kStageBImplemented is true, so the verdict and the message cannot disagree about WHICH files
// are gone; built at runtime from three compile-time flags, which is why no quoting is involved.
[[nodiscard]] inline std::string stage_b_missing_sources() {
    std::string missing;
    if (!NINFER_QWEN4_EXP_STAGE_B_HAS_IMPL_PACKAGE) { missing += "impl/package.cpp "; }
    if (!NINFER_QWEN4_EXP_STAGE_B_HAS_IMPL_VARIANT) { missing += "impl/variant.cpp "; }
    if (!NINFER_QWEN4_EXP_STAGE_B_HAS_IMPL_BINDINGS) { missing += "impl/load/bindings.cpp "; }
    if (missing.empty()) { missing = "(none: the declared stage (b) sources are all present)"; }
    return missing;
}

namespace detail {

// 取一个具名张量的形状。缺失或不是张量 -> 抛错指名（这就是 S28 writer 的契约面）。
inline std::vector<std::uint64_t> tensor_shape(const artifact::Reader& reader,
                                               std::string_view name,
                                               std::string_view quantity) {
    const artifact::ObjectDescriptor* object = reader.find(name);
    if (object == nullptr) {
        throw std::runtime_error(
            "qwen4-exp stage (a): artifact has no tensor '" + std::string(name) +
            "', the only shape source for " + std::string(quantity) +
            " - the S28 writer must emit it; refusing to guess");
    }
    const auto* tensor = std::get_if<artifact::TensorDescriptor>(object);
    if (tensor == nullptr) {
        throw std::runtime_error(
            "qwen4-exp stage (a): artifact object '" + std::string(name) +
            "' is not a tensor, so " + std::string(quantity) +
            " is not derivable; refusing to guess");
    }
    return tensor->shape;
}

inline void expect_dim(std::string_view quantity, std::uint64_t from_artifact,
                       std::uint64_t from_config, std::string_view tensor_name) {
    if (from_artifact != from_config) {
        throw std::runtime_error(
            "qwen4-exp stage (a): geometry mismatch for " + std::string(quantity) +
            ": artifact tensor '" + std::string(tensor_name) + "' says " +
            std::to_string(from_artifact) + ", config expects " +
            std::to_string(from_config) + "; refusing to load");
    }
}

} // namespace detail

// 返回一行 PASS 判词（拼进装载拒绝报文）；任一不符或不可推导即抛 std::runtime_error。
//
// 张量 -> 量 的映射（artifact 2-D 线性是 (out, in)，出处 flashnext_convert.py:139-166）：
//   token_embd          (vocab, hidden)      -> vocab, hidden
//   layer.3.qsa.q       (24*256, 2560)       -> q_dim = query_heads*head_dim, hidden
//   layer.3.qsa.k       (2*256, 2560)        -> kv_dim = kv_heads*head_dim, hidden
//   layer.0.moe.e0.down (2560, 640)          -> hidden, intermediate（每专家 MLP 宽度）
//   layer.<N>.*         名字里最大序号 + 1    -> layers
//
// layer.3 是第一个 QSA 层（flashnext_bindings.py:36，QSA_LAYERS[0] == 3），所以
// layer.3.qsa.* 一定存在，且它是 48 层里唯一同时带 q/k 与全维 head 的那种层。
inline std::string validate_stage_a_geometry(const artifact::Reader& reader) {
    using TC = detail::TextConfig;

    const std::vector<std::uint64_t> embed =
        detail::tensor_shape(reader, "token_embd", "hidden/vocab");
    const std::vector<std::uint64_t> q = detail::tensor_shape(reader, "layer.3.qsa.q", "q_dim");
    const std::vector<std::uint64_t> k = detail::tensor_shape(reader, "layer.3.qsa.k", "kv_dim");
    const std::vector<std::uint64_t> down =
        detail::tensor_shape(reader, "layer.0.moe.e0.down", "intermediate");

    if (embed.size() != 2 || q.size() != 2 || k.size() != 2 || down.size() != 2) {
        throw std::runtime_error(
            "qwen4-exp stage (a): geometry tensors must be rank two "
            "(token_embd / layer.3.qsa.q / layer.3.qsa.k / layer.0.moe.e0.down)");
    }
    detail::expect_dim("hidden", embed[1], static_cast<std::uint64_t>(TC::hidden), "token_embd");
    detail::expect_dim("vocab", embed[0], static_cast<std::uint64_t>(TC::vocab), "token_embd");
    detail::expect_dim("q_dim", q[0],
                       static_cast<std::uint64_t>(TC::query_heads) *
                           static_cast<std::uint64_t>(TC::head_dim),
                       "layer.3.qsa.q");
    detail::expect_dim("hidden(q)", q[1], static_cast<std::uint64_t>(TC::hidden), "layer.3.qsa.q");
    detail::expect_dim("kv_dim", k[0],
                       static_cast<std::uint64_t>(TC::kv_heads) *
                           static_cast<std::uint64_t>(TC::head_dim),
                       "layer.3.qsa.k");
    detail::expect_dim("hidden(k)", k[1], static_cast<std::uint64_t>(TC::hidden), "layer.3.qsa.k");
    detail::expect_dim("hidden(down)", down[0], static_cast<std::uint64_t>(TC::hidden),
                       "layer.0.moe.e0.down");
    detail::expect_dim("intermediate", down[1], static_cast<std::uint64_t>(TC::intermediate),
                       "layer.0.moe.e0.down");

    // layers：manifest 里 "layer.<N>." 的最大 N + 1。
    std::uint64_t max_layer = 0;
    for (const artifact::ObjectDescriptor& object : reader.objects()) {
        const std::string_view name = artifact::object_name(object);
        if (name.rfind("layer.", 0) != 0) { continue; }
        std::uint64_t value = 0;
        std::size_t pos     = 6; // strlen("layer.")
        bool any_digit      = false;
        while (pos < name.size() && name[pos] >= '0' && name[pos] <= '9') {
            value = value * 10 + static_cast<std::uint64_t>(name[pos] - '0');
            ++pos;
            any_digit = true;
        }
        if (any_digit && value > max_layer) { max_layer = value; }
    }
    detail::expect_dim("layers", max_layer + 1, static_cast<std::uint64_t>(TC::layers),
                       "layer.<N>.* names");

    return "geometry cross-check PASS: hidden=" + std::to_string(TC::hidden) +
           " vocab=" + std::to_string(TC::vocab) +
           " q=" + std::to_string(TC::query_heads * TC::head_dim) +
           " kv=" + std::to_string(TC::kv_heads * TC::head_dim) +
           " intermediate=" + std::to_string(TC::intermediate) +
           " layers=" + std::to_string(TC::layers);
}

} // namespace ninfer::targets::qwen4_exp
