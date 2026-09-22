#include "ops/linear_add/fp8/fp8_linear_add_plan.h"

#include "ops/linear/fp8/fp8_a8_plan.h"
#include "ops/linear/fp8/fp8_config.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

enum class Fp8LinearAddRoute : std::uint8_t {
    A16,
    A8,
};

// ⚠️ ACCEPT2 — THE VERIFY DOMAIN THIS FILE PROTECTS. `linear_add` carries the SAME defect shape as
// the three sites FIX-C / FIX-C2 closed (fp8_gdn_input_plan.cpp, fp8_gdn_conv_plan.cpp,
// fp8_attn_input_plan.cpp): A8 quantises the ACTIVATIONS to FP8, so it is a DIFFERENT ARITHMETIC,
// not a re-ordering of the batch-1 decode's. Those three are domain-scoped to <= kVerifyWidthCeiling,
// because `kMtpDecodeMaximumDrafts + 1 = 16` is the widest a chain-verify round can present.
// kVerifyWidthCeiling mirrors them; kWideVerifyWidthCeiling is the WIDENED domain the ngram-draft
// line introduces (it widens verify to 1..63 and names these thresholds).
inline constexpr std::int32_t kVerifyWidthCeiling     = 16;
inline constexpr std::int32_t kWideVerifyWidthCeiling = 64;

Fp8LinearAddRoute resolve_route(std::int32_t output_rows, std::int32_t input_rows,
                                LinearPolicy policy, std::int32_t tokens) {
    if (tokens <= 0 || output_rows != Fp8Residual6144Geometry::kOutputRows ||
        (input_rows != Fp8Residual6144Geometry::kInputRows &&
         input_rows != Fp8Residual17408Geometry::kInputRows)) {
        throw std::invalid_argument("fp8 linear_add: unsupported shape");
    }
    if (policy == LinearPolicy::A16Only) { return Fp8LinearAddRoute::A16; }
    if (policy != LinearPolicy::AllowA8) {
        throw std::invalid_argument("fp8 linear_add: unsupported policy");
    }
    // ⚠️ ACCEPT2 — THE FOURTH A8 CROSSOVER, CLOSED. This line carried the family's defect shape
    // with NO domain guard at all, and its crossover sits at 22/25 — ABOVE the verify domain, so it
    // is LATENT today (the widest chain-verify round is kMtpDecodeMaximumDrafts + 1 = 16, and
    // --draft-tree's node budget is L*d <= 15). It stops being latent the moment verification is
    // widened past 16: ngram widens verify to 1..63 and names THESE thresholds
    // (docs/ngram.md:51-54 in the pulled fork), and the fork resolves it with an explicit
    // A16-retention branch. This is that branch, so the two trees converge:
    //   * [17, 64] is the widened verify domain => the decode's own tier, stated as a DOMAIN rather
    //     than left to where a number happens to sit;
    //   * the measured 22/25 crossovers are KEPT and still apply above 64, the column counts that
    //     only prefill and the draft model reach, so no measured calibration is discarded;
    //   * a future widening past 64 meets a NAMED constant instead of silently entering A8.
    // (The literal family form `tokens > kVerifyWidthCeiling ? A8 : A16` is NOT applicable here:
    //  22 already exceeds 16, so that form would LOWER the crossover to 17 and move widths 17..21
    //  from A16 to A8 -- a precision regression, not a fix.)
    if (tokens > kVerifyWidthCeiling && tokens <= kWideVerifyWidthCeiling) {
        return Fp8LinearAddRoute::A16;
    }
    const std::int32_t first_a8 = input_rows == Fp8Residual6144Geometry::kInputRows ? 22 : 25;
    return tokens >= first_a8 ? Fp8LinearAddRoute::A8 : Fp8LinearAddRoute::A16;
}

void launch_a16(const Tensor& x, const Weight& weight, Tensor& residual, cudaStream_t stream) {
    for (std::int32_t token_begin = 0; token_begin < x.ne[1]; token_begin += kFp8LastSmallT) {
        const std::int32_t active = std::min(kFp8LastSmallT, x.ne[1] - token_begin);
        auto* input               = static_cast<std::uint8_t*>(x.data) +
                      static_cast<std::int64_t>(token_begin) * weight.k * sizeof(std::uint16_t);
        auto* output = static_cast<std::uint8_t*>(residual.data) +
                       static_cast<std::int64_t>(token_begin) * weight.n * sizeof(std::uint16_t);
        Tensor input_chunk(input, DType::BF16, {weight.k, active});
        Tensor residual_chunk(output, DType::BF16, {weight.n, active});
        if (active == 1) {
            fp8_linear_add_decode_launch(input_chunk, weight, residual_chunk, stream);
        } else {
            fp8_linear_add_small_t_launch(input_chunk, weight, residual_chunk, stream);
        }
    }
}

} // namespace

std::size_t fp8_linear_add_workspace_capacity_bytes(std::int32_t output_rows,
                                                    std::int32_t input_rows, LinearPolicy policy,
                                                    std::int32_t min_tokens,
                                                    std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("fp8 linear_add workspace: invalid token interval");
    }
    (void)resolve_route(output_rows, input_rows, policy, min_tokens);
    return resolve_route(output_rows, input_rows, policy, max_tokens) == Fp8LinearAddRoute::A8
               ? fp8_a8_workspace_capacity_bytes(max_tokens, input_rows)
               : 0;
}

void fp8_linear_add_dispatch(const Tensor& x, const Weight& weight, Tensor& residual,
                             LinearPolicy policy, WorkspaceArena& workspace, cudaStream_t stream) {
    const Fp8LinearAddRoute route = resolve_route(weight.n, weight.k, policy, x.ne[1]);
    if (route == Fp8LinearAddRoute::A16) {
        launch_a16(x, weight, residual, stream);
        return;
    }
    fp8_linear_add_a8_launch(x, weight, residual, workspace, stream);
}

} // namespace ninfer::ops::detail
