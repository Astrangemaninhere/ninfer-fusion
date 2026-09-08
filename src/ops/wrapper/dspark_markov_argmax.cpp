#include "ninfer/ops/dspark_markov_argmax.h"

#include "ops/launcher/dspark_markov_argmax.h"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {

void dspark_markov_argmax(const Tensor& logits, const Weight& markov_w1,
                          const Weight& markov_w2, const Tensor& anchors, Tensor& drafts,
                          Tensor& best_value, Tensor& best_index, Tensor& extents,
                          float entropy_threshold, cudaStream_t stream) {
    if (logits.dtype != DType::BF16 || anchors.dtype != DType::I32 || drafts.dtype != DType::I32 ||
        best_value.dtype != DType::I32 || best_index.dtype != DType::I32 ||
        extents.dtype != DType::I32) {
        throw std::invalid_argument("dspark_markov_argmax: logits must be BF16 and ids I32");
    }
    if (!logits.is_contiguous() || logits.data == nullptr || !anchors.is_contiguous() ||
        anchors.data == nullptr || !drafts.is_contiguous() || drafts.data == nullptr ||
        !best_value.is_contiguous() || best_value.data == nullptr || !best_index.is_contiguous() ||
        best_index.data == nullptr || !extents.is_contiguous() || extents.data == nullptr) {
        throw std::invalid_argument("dspark_markov_argmax: tensors must be contiguous and non-null");
    }
    if (!(entropy_threshold > 0.0f) || !std::isfinite(entropy_threshold)) {
        throw std::invalid_argument("dspark_markov_argmax: entropy threshold must be positive");
    }
    const std::int32_t vocab  = logits.ne[0];
    const std::int32_t tokens = logits.ne[1];
    if (vocab != 248320 || tokens < 1 || tokens > 7 * 8) {
        throw std::invalid_argument("dspark_markov_argmax: registered domain is V=248320, K*B<=56");
    }
    const std::int32_t batch = anchors.ne[0];
    if (batch < 1 || batch > 8 || tokens % batch != 0 || tokens / batch > 7 ||
        drafts.ne[0] != tokens || best_value.ne[0] != batch || best_index.ne[0] != batch ||
        extents.ne[0] != batch) {
        throw std::invalid_argument("dspark_markov_argmax: inconsistent batch or draft shape");
    }
    if (markov_w1.qtype != QType::BF16_CTRL || markov_w2.qtype != QType::BF16_CTRL ||
        markov_w1.n != vocab || markov_w1.k != detail::kDsparkMarkovRank ||
        markov_w2.n != vocab || markov_w2.k != detail::kDsparkMarkovRank ||
        markov_w1.qdata == nullptr || markov_w2.qdata == nullptr) {
        throw std::invalid_argument("dspark_markov_argmax: Markov weights must be BF16 [V,256]");
    }

    detail::dspark_markov_argmax_launch(logits, markov_w1, markov_w2, anchors, drafts, best_value,
                                        best_index, extents, tokens / batch, entropy_threshold,
                                        stream);
}

} // namespace ninfer::ops
