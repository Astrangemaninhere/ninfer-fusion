#include "ninfer/ops/dflash2_selector.h"

#include "ops/launcher/dflash2_selector.h"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops {

void dflash2_selector(const Tensor& unary_logits, const Tensor& projected_hidden,
                      const Weight& predecessor_codebook, const Weight& successor_codebook,
                      const Tensor& anchors, Tensor& candidates, Tensor& unary, Tensor& scores,
                      Tensor& drafts, std::int32_t steps, std::int32_t top_k, cudaStream_t stream) {
    const auto invalid = [](const char* message) { throw std::invalid_argument(message); };
    if (unary_logits.dtype != DType::BF16 || projected_hidden.dtype != DType::BF16 ||
        anchors.dtype != DType::I32 || candidates.dtype != DType::I32 ||
        unary.dtype != DType::FP32 || scores.dtype != DType::FP32 || drafts.dtype != DType::I32) {
        invalid("dflash2_selector: logits/projected must be BF16, ids I32, scores F32");
    }
    const auto contiguous = [](const Tensor& tensor) {
        return tensor.is_contiguous() && tensor.data != nullptr;
    };
    if (!contiguous(unary_logits) || !contiguous(projected_hidden) || !contiguous(anchors) ||
        !contiguous(candidates) || !contiguous(unary) || !contiguous(scores) ||
        !contiguous(drafts)) {
        invalid("dflash2_selector: tensors must be contiguous and non-null");
    }
    const std::int32_t vocab  = unary_logits.ne[0];
    const std::int32_t tokens = unary_logits.ne[1];
    const std::int32_t rank   = projected_hidden.ne[0];
    const std::int32_t batch  = anchors.ne[0];
    if (vocab != 248320 || rank != detail::kDflash2SelectorRank || steps != 7 ||
        top_k != detail::kDflash2SelectorTopK || batch < 1 || batch > 8 ||
        tokens != steps * batch ||
        projected_hidden.ne[1] != tokens) {
        invalid("dflash2_selector: registered domain is V=248320, R=256, S=7, B=1..8, K=16");
    }
    if (candidates.ne[0] != batch || candidates.ne[1] != steps || candidates.ne[2] != top_k ||
        unary.ne[0] != batch || unary.ne[1] != steps || unary.ne[2] != top_k ||
        scores.ne[0] != batch || scores.ne[1] != steps || scores.ne[2] != top_k ||
        scores.ne[3] != top_k || drafts.ne[0] != tokens) {
        invalid("dflash2_selector: scratch or draft shapes are inconsistent");
    }
    if (predecessor_codebook.qtype != QType::BF16_CTRL ||
        successor_codebook.qtype != QType::BF16_CTRL || predecessor_codebook.n != vocab ||
        predecessor_codebook.k != rank || successor_codebook.n != vocab ||
        successor_codebook.k != rank || predecessor_codebook.qdata == nullptr ||
        successor_codebook.qdata == nullptr) {
        invalid("dflash2_selector: codebooks must be BF16 [V,256]");
    }

    detail::dflash2_selector_launch(unary_logits, projected_hidden, predecessor_codebook,
                                    successor_codebook, anchors, candidates, unary, scores, drafts,
                                    steps, top_k, stream);
}

} // namespace ninfer::ops
