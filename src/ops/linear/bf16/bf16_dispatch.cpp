#include "ops/linear/bf16/bf16_dispatch.h"

#include "ops/linear/bf16/bf16_config.h"
#include "ops/linear/bf16/bf16_fp16_route.h"
#include "ops/linear/bf16/bf16_launch.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>
#include <string>

namespace ninfer::ops::detail {

Bf16Launch select_bf16_a16_launch(std::int32_t n, std::int32_t k, std::int32_t t) {
    const bool legacy_problem  = (n == 14336 && k == 5120) || (n == 5120 && k == 6144);
    const bool dspark_problem  = (n == 5120 && k == 25600) || (n == 7168 && k == 5120) ||
                                (n == 5120 && k == 5120) || (n == 20480 && k == 5120) ||
                                (n == 5120 && k == 10240) || (n == 1024 && k == 5120);
    const bool dflash2_problem = (n == 5120 && k == 25600) || (n == 6144 && k == 5120) ||
                                 (n == 4096 && k == 5120) || (n == 1024 && k == 5120) ||
                                 (n == 5120 && k == 4096) || (n == 1280 && k == 5120) ||
                                 (n == 34816 && k == 5120) || (n == 5120 && k == 17408) ||
                                 (n == 256 && k == 5120) ||
                                 // 全词表/短表 vocabulary 头：BF16 档的 text/output_head
                                 // 与 text/draft_head（供 head 量化误差的非循环参考）。
                                 (n == 248320 && k == 5120) || (n == 131072 && k == 5120);
    if ((!legacy_problem && !dspark_problem && !dflash2_problem) || t <= 0) {
        throw std::invalid_argument("bf16 linear: unsupported shape or T");
    }
    if (dspark_problem || dflash2_problem) {
        // The generic MMA core already tiles arbitrary admitted n/k with the
        // 64x128x64 production schedule; the DSpark/DFlash2 drafts never need
        // the fixed-shape decode/small-T specializations (T = verify width 8..64).
        return launch_bf16_mma;
    }
    if (t == 1) { return launch_bf16_decode; }
    const std::int32_t small_t_end =
        n == 5120 ? kBf16SmallTMaxTokens : kBf16LinearSmallTDispatchEnd;
    if (t <= small_t_end) { return launch_bf16_small_t; }
    return launch_bf16_mma;
}

Bf16Launch select_bf16_launch(std::int32_t n, std::int32_t k, std::int32_t t, LinearPolicy policy) {
    switch (policy) {
    case LinearPolicy::A16Only:
        return select_bf16_a16_launch(n, k, t);
    case LinearPolicy::AllowA8:
    case LinearPolicy::AllowA4:
        break;
    }
    throw std::invalid_argument("bf16 linear: unsupported policy");
}

void bf16_dispatch(const Tensor& x, const Weight& weight, Tensor& out, LinearPolicy policy,
                   cudaStream_t stream) {
    // ---------------------------------------------------------------------------
    // THE ARCHITECTURE ROUTE DECISION FOR THE BF16 LINE (dl/fp16route, F-736)
    // ---------------------------------------------------------------------------
    // Before this call the BF16 route layer had no engine consumer either: MEASURED, this
    // function selected BY SHAPE and WITHOUT CONSULTING THE ARCH (t == 1 -> bf16_gemv; t <=
    // kBf16LinearSmallTDispatchEnd -> bf16_small_t), which is exactly what the BF16 row of
    // kFormatRequirements records -- "on this box the BF16 decode path is this kernel on every
    // card". So a simulated rung could not change what ran, and the wall could not be seen, let
    // alone measured. The order of the two properties is deliberate and worth stating:
    //   * ON A SHIPPING BUILD THIS IS A NO-OP. No environment set -> arch_view_for_device()
    //     returns Disabled, effective_sm == the card's own (120), and select_route(120, BF16, ..)
    //     answers mma-bf16 because rung 120 HAS Cap::Bf16Mma. So use_arm is false, nothing is
    //     printed, and the shape table below runs unchanged. This is the same shape (and the
    //     same reasoning) as nvfp4_dispatch.cpp's own route call.
    //   * ON A SIMULATED RUN IT IS NOT SILENT and it is NOT a free pass: a REFUSED request
    //     throws here rather than falling back to the real device, and the arm is taken only
    //     when the tables selected it AND the op is at least one atom wide
    //     (Bf16Fp16Plan::use_arm; see bf16_fp16_route.cpp for why T < 8 stays on FFMA).
    const Bf16Fp16Plan fp16_plan = bf16_fp16_plan(x.ne[1], weight.n, weight.k);
    if (fp16_plan.refused) {
        throw std::invalid_argument(
            "bf16 linear: the architecture request was REFUSED, so the native shape table is not "
            "taken either, because answering for the real device would answer a question nobody "
            "asked. " + fp16_plan.why);
    }
    if (fp16_plan.use_arm) {
        // Loud ONCE PER SHAPE, following the routing line's own reasoning: a line per invocation
        // at 20 tok/s would bury the marker it exists to make visible, and a prefill would emit
        // five figures of them.
        //
        // AND NOW THE CODE DOES WHAT THAT SENTENCE SAYS. The pre-image compared the key against
        // the PREVIOUS key only, which is once per shape CHANGE and not once per shape: an engine
        // whose per-round op cycle revisits a small set of shapes re-announces on every call.
        // MEASURED (dl/armlaunch, from the preserved stderr of every cell of record): a 4-round
        // decode on this artifact class announces 150 times for 9 distinct shapes -- 37.5 lines per
        // round, 16.7x the stated intent -- and the line is 4,518 bytes, so the write is not free.
        // The set below makes the announcement O(distinct shapes) instead of O(calls). It changes
        // NOTHING about the route, the plan or the kernel, so the emitted ids are unchanged.
        // ARMLAUNCH_F822_ANNOUNCE_ONCE_PER_SHAPE
        //
        // The set is BOUNDED, and past the bound the announcement falls back to per-invocation so
        // that a shape stream larger than the bound degrades to the pre-image's behaviour rather
        // than losing the diagnostic.
        static thread_local std::vector<std::string> announced;
        const std::string key = fp16_plan.route + "|" + std::to_string(x.ne[1]) + "|" +
                                std::to_string(weight.n) + "|" + std::to_string(weight.k);
        constexpr std::size_t kAnnouncedShapesMax = 64;
        const bool already_announced =
            announced.size() < kAnnouncedShapesMax &&
            std::find(announced.begin(), announced.end(), key) != announced.end();
        const bool over_bound = announced.size() >= kAnnouncedShapesMax;
        if (!already_announced) {
            if (!over_bound) { announced.push_back(key); }
            std::fprintf(stderr,
                         "[bf16] the fp16 tensor-core arm is taken for %dt x %dx%d: %s. %s\n",
                         x.ne[1], weight.n, weight.k, fp16_plan.kernel.c_str(),
                         fp16_plan.why.c_str());
        }
        launch_bf16_fp16_mma(x, weight, out, fp16_plan, stream);
        return;
    }
    const Bf16Launch launch = select_bf16_launch(weight.n, weight.k, x.ne[1], policy);
    launch(x, weight, out, stream);
}

} // namespace ninfer::ops::detail
