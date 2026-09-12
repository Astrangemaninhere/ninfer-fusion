#!/usr/bin/env python3
"""Wire the checkpoint's dedicated proposal head (text/draft_head + draft_head_token_ids)
into the dflash2 proposal path.

Contract (ninfer-upstream/docs/maintainer/qwen3.8-27b-dflash2.md 6.1, line 254):
the top-16 candidates must be mapped through `text/draft_head_token_ids` to global token
ids BEFORE they index the selector codebooks. Upstream implements both heads
(`if (proposal_head == Full) ... else linear_topk(head.head, head.token_ids, ...)`); our
dflash2 path only had the full-head branch and its CLI rejected the other one outright.

Every edit asserts its anchor occurs exactly once. Originals are backed up and a unified
diff is written for `patch -b` / `patch -R` rollback (P2).
"""

import difflib
import hashlib
import pathlib
import shutil
import sys

BUILD = "/home/user/ninfer-fusion"
BAK = "/home/user/df2head_bak"
OUT_DIFF = "/mnt/c/Users/User/Documents/ziqinzhang/_collab/F1_df2_draft_head.diff"

CONFIG = "src/targets/qwen3_6_27b/impl/config.h"
OPTS = "src/product/speculative_options.h"
PUB = "include/ninfer/ops/dflash2_selector.h"
LHDR = "src/ops/launcher/dflash2_selector.h"
LCU = "src/ops/launcher/dflash2_selector.cu"
WRAP = "src/ops/wrapper/dflash2_selector.cpp"
KERN = "src/ops/kernel/dflash2_selector.cuh"
IMPL = "src/targets/qwen3_6/impl/runtime/dflash2_impl.h"

EDITS = []


def add(path, old, new, note):
    EDITS.append((path, old.encode(), new.encode(), note))


# ---------------------------------------------------------------- config.h
add(CONFIG,
    "    static constexpr int selector_rank   = 256;\n"
    "    static constexpr int selector_top_k  = 16;\n",
    "    static constexpr int selector_rank   = 256;\n"
    "    static constexpr int selector_top_k  = 16;\n"
    "    // Rows of the checkpoint's dedicated proposal head (`text/draft_head`). Its\n"
    "    // rows are shortlist entries: only the global ids reached through\n"
    "    // `text/draft_head_token_ids` may index the selector codebooks (contract 6.1).\n"
    "    static constexpr int draft_head_rows = 131072;\n",
    "DFlash2Config::draft_head_rows")

# ------------------------------------------------- speculative_options.h
add(OPTS,
    "        if (options.proposal_head != ProposalHead::Full) {\n"
    "            throw std::invalid_argument(\"--spec dflash2 requires the full proposal "
    "head\");\n"
    "        }\n",
    "        // Both proposal heads are supported (propose_batch_impl): the full\n"
    "        // vocabulary head, and the checkpoint's dedicated shortlist head whose rows\n"
    "        // are mapped to global ids through text/draft_head_token_ids.\n",
    "dflash2 accepts ProposalHead::Optimized")

# --------------------------------------------------- public selector header
add(PUB,
    " * The registered domain is V=248320, R=256, S=7, B=1..8, K=16. `domain` bounds the\n"
    " * token ids admitted into the top-K candidate set (the contract only allows ids\n"
    " * `0..token_domain-1`; 248077 for this checkpoint). Inputs and weights are unchanged\n"
    " * and the Op owns no workspace or persistent state.\n",
    " * `head_token_ids` is contiguous I32 [V] and maps every candidate row to the global\n"
    " * token id used for the draft output, for the codebook lookup, and for score ties.\n"
    " * Pass a default Tensor for the identity map (the full-vocabulary head). With a\n"
    " * shortlist head, V is the shortlist row count and `domain` is the number of rows\n"
    " * scanned; with the full head V=248320 and `domain` bounds the addressable token ids\n"
    " * (the contract only allows ids `0..token_domain-1`; 248077 for this checkpoint).\n"
    " *\n"
    " * The registered geometry is R=256, S=1..15, B=1..8, K=16. Inputs and weights are\n"
    " * unchanged and the Op owns no workspace or persistent state.\n",
    "public doc: head mapping semantics")
add(PUB,
    "                      Tensor& candidate_probs, std::int32_t steps, std::int32_t top_k,\n"
    "                      std::int32_t domain, cudaStream_t stream);\n",
    "                      Tensor& candidate_probs, std::int32_t steps, std::int32_t top_k,\n"
    "                      std::int32_t domain, const Tensor& head_token_ids,\n"
    "                      cudaStream_t stream);\n",
    "public signature")

# ------------------------------------------------------ launcher header
add(LHDR,
    "inline constexpr int kDflash2SelectorRank = 256;\n"
    "inline constexpr int kDflash2SelectorTopK = 16;\n",
    "inline constexpr int kDflash2SelectorRank = 256;\n"
    "inline constexpr int kDflash2SelectorTopK = 16;\n"
    "// The selector codebooks are indexed by GLOBAL token ids, so their row count is the\n"
    "// global vocabulary even when the candidates come from the shortlist head.\n"
    "inline constexpr int kDflash2GlobalVocab  = 248320;\n",
    "launcher: global vocabulary constant")
add(LHDR,
    "                             Tensor& candidate_probs, std::int32_t steps, std::int32_t top_k,\n"
    "                             std::int32_t domain, cudaStream_t stream);\n",
    "                             Tensor& candidate_probs, std::int32_t steps, std::int32_t top_k,\n"
    "                             std::int32_t domain, const Tensor& head_token_ids,\n"
    "                             cudaStream_t stream);\n",
    "launcher signature")

# --------------------------------------------------------- launcher .cu
add(LCU,
    "                             Tensor& candidate_probs, std::int32_t steps, std::int32_t top_k,\n"
    "                             std::int32_t domain, cudaStream_t stream) {\n",
    "                             Tensor& candidate_probs, std::int32_t steps, std::int32_t top_k,\n"
    "                             std::int32_t domain, const Tensor& head_token_ids,\n"
    "                             cudaStream_t stream) {\n",
    "launcher .cu signature")
add(LCU,
    "        static_cast<const __nv_bfloat16*>(unary_logits.data),\n"
    "        static_cast<std::int32_t*>(candidates.data), static_cast<float*>(unary.data),\n"
    "        unary_logits.ne[0], domain, batch, steps, columns);\n",
    "        static_cast<const __nv_bfloat16*>(unary_logits.data),\n"
    "        head_token_ids.data != nullptr\n"
    "            ? static_cast<const std::int32_t*>(head_token_ids.data)\n"
    "            : nullptr,\n"
    "        static_cast<std::int32_t*>(candidates.data), static_cast<float*>(unary.data),\n"
    "        unary_logits.ne[0], domain, batch, steps, columns);\n",
    "launcher: pass the shortlist map")

# ------------------------------------------------------- wrapper .cpp
add(WRAP,
    "                      Tensor& candidate_probs, std::int32_t steps, std::int32_t top_k,\n"
    "                      std::int32_t domain, cudaStream_t stream) {\n",
    "                      Tensor& candidate_probs, std::int32_t steps, std::int32_t top_k,\n"
    "                      std::int32_t domain, const Tensor& head_token_ids,\n"
    "                      cudaStream_t stream) {\n",
    "wrapper signature")
add(WRAP,
    "    if (vocab != 248320 || rank != detail::kDflash2SelectorRank || steps < 1 ||\n"
    "        steps > 15 || top_k != detail::kDflash2SelectorTopK || batch < 1 || batch > 8 ||\n"
    "        tokens != steps * batch || domain < 1 || domain > vocab ||\n"
    "        projected_hidden.ne[1] != tokens) {\n"
    "        invalid(\n"
    "            \"dflash2_selector: registered domain is V=248320, R=256, S=1..15, B=1..8, \"\n"
    "            \"K=16, domain in [1,V]\");\n"
    "    }\n",
    "    if (rank != detail::kDflash2SelectorRank || steps < 1 || steps > 15 ||\n"
    "        top_k != detail::kDflash2SelectorTopK || batch < 1 || batch > 8 ||\n"
    "        tokens != steps * batch || domain < 1 || domain > vocab ||\n"
    "        projected_hidden.ne[1] != tokens) {\n"
    "        invalid(\n"
    "            \"dflash2_selector: registered domain is V=131072|248320, R=256, S=1..15, \"\n"
    "            \"B=1..8, K=16, domain in [1,V]\");\n"
    "    }\n"
    "    if (head_token_ids.data != nullptr &&\n"
    "        (head_token_ids.dtype != DType::I32 || !contiguous(head_token_ids) ||\n"
    "         head_token_ids.ne[0] != vocab)) {\n"
    "        invalid(\"dflash2_selector: head_token_ids must be a contiguous I32 [V] map\");\n"
    "    }\n",
    "wrapper: allow both head geometries + validate the map")
add(WRAP,
    "        successor_codebook.qtype != QType::BF16_CTRL || predecessor_codebook.n != vocab ||\n"
    "        predecessor_codebook.k != rank || successor_codebook.n != vocab ||\n",
    "        successor_codebook.qtype != QType::BF16_CTRL ||\n"
    "        predecessor_codebook.n != detail::kDflash2GlobalVocab ||\n"
    "        predecessor_codebook.k != rank ||\n"
    "        successor_codebook.n != detail::kDflash2GlobalVocab ||\n",
    "wrapper: codebooks are global-vocabulary")
add(WRAP,
    "                                    configs, candidate_ids, candidate_probs, steps, top_k, domain,\n"
    "                                    stream);\n",
    "                                    configs, candidate_ids, candidate_probs, steps, top_k, domain,\n"
    "                                    head_token_ids, stream);\n",
    "wrapper: forward the map")

# ------------------------------------------------------------ kernel
add(KERN,
    "namespace ninfer::ops {\n",
    "namespace ninfer::ops {\n"
    "\n"
    "// Sentinel for an unfilled shortlist slot: every real global token id is below it.\n"
    "inline constexpr std::int32_t kDflash2NoToken = 0x7fffffff;\n",
    "kernel: no-candidate sentinel")
add(KERN,
    "    const __nv_bfloat16* __restrict__ logits, std::int32_t* __restrict__ candidates,\n"
    "    float* __restrict__ unary, int vocab, int domain, int batch, int steps, int columns) {\n",
    "    const __nv_bfloat16* __restrict__ logits,\n"
    "    const std::int32_t* __restrict__ row_to_global_ids,\n"
    "    std::int32_t* __restrict__ candidates, float* __restrict__ unary, int vocab, int domain,\n"
    "    int batch, int steps, int columns) {\n",
    "kernel: topk takes the shortlist map")
add(KERN,
    "    // Per-thread top-K over a strided slice of the vocabulary column.\n"
    "    float local_value[K];\n"
    "    std::int32_t local_index[K];\n"
    "#pragma unroll\n"
    "    for (int k = 0; k < K; ++k) {\n"
    "        local_value[k] = -CUDART_INF_F;\n"
    "        local_index[k] = vocab;\n"
    "    }\n"
    "    const __nv_bfloat16* col_logits = logits + static_cast<std::int64_t>(column) * vocab;\n"
    "    // Contract: only token ids [0, domain) may enter the top-K; ids at or above the\n"
    "    // token domain (248077 here) are not addressable sampling results.\n"
    "    for (int v = tid; v < domain; v += Block) {\n"
    "        const float value = __bfloat162float(col_logits[v]);\n"
    "        // Fast reject: below the current K-th best (with id tie-break).\n"
    "        if (value < local_value[K - 1] ||\n"
    "            (value == local_value[K - 1] && v >= local_index[K - 1])) {\n"
    "            continue;\n"
    "        }\n"
    "        for (int k = 0; k < K; ++k) {\n"
    "            const bool better =\n"
    "                value > local_value[k] || (value == local_value[k] && v < local_index[k]);\n"
    "            if (better) {\n"
    "                for (int tail = K - 1; tail > k; --tail) {\n"
    "                    local_value[tail] = local_value[tail - 1];\n"
    "                    local_index[tail] = local_index[tail - 1];\n"
    "                }\n"
    "                local_value[k] = value;\n"
    "                local_index[k] = v;\n"
    "                break;\n"
    "            }\n"
    "        }\n"
    "    }\n",
    "    // Per-thread top-K over a strided slice of the head rows. `row_to_global_ids`\n"
    "    // maps a shortlist row to the global token id used for the draft output and for\n"
    "    // score ties (contract 6.1); a null map is the identity, i.e. the full head.\n"
    "    float local_value[K];\n"
    "    std::int32_t local_index[K];\n"
    "#pragma unroll\n"
    "    for (int k = 0; k < K; ++k) {\n"
    "        local_value[k] = -CUDART_INF_F;\n"
    "        local_index[k] = kDflash2NoToken;\n"
    "    }\n"
    "    const __nv_bfloat16* col_logits = logits + static_cast<std::int64_t>(column) * vocab;\n"
    "    // Contract: only rows [0, domain) may enter the top-K. A full head additionally\n"
    "    // excludes ids at or above the token domain (248077) as unaddressable results.\n"
    "    for (int v = tid; v < domain; v += Block) {\n"
    "        const float value = __bfloat162float(col_logits[v]);\n"
    "        const std::int32_t id =\n"
    "            row_to_global_ids != nullptr ? row_to_global_ids[v] : static_cast<std::int32_t>(v);\n"
    "        // Fast reject: below the current K-th best (with id tie-break).\n"
    "        if (value < local_value[K - 1] ||\n"
    "            (value == local_value[K - 1] && id >= local_index[K - 1])) {\n"
    "            continue;\n"
    "        }\n"
    "        for (int k = 0; k < K; ++k) {\n"
    "            const bool better =\n"
    "                value > local_value[k] || (value == local_value[k] && id < local_index[k]);\n"
    "            if (better) {\n"
    "                for (int tail = K - 1; tail > k; --tail) {\n"
    "                    local_value[tail] = local_value[tail - 1];\n"
    "                    local_index[tail] = local_index[tail - 1];\n"
    "                }\n"
    "                local_value[k] = value;\n"
    "                local_index[k] = id;\n"
    "                break;\n"
    "            }\n"
    "        }\n"
    "    }\n",
    "kernel: map rows to global ids + tie-break by id")

# ------------------------------------------------------- dflash2_impl.h
add(IMPL,
    "    Tensor logits = state.execution.work.alloc(\n"
    "        DType::BF16, {TextConfig::output_rows, static_cast<std::int32_t>(k) * batch_size});\n"
    "    ops::linear(proposal_hidden, state.execution.model.output_head, logits,\n"
    "                state.execution.device.stream);\n"
    "    kCfg.apply_final_logit_policy(logits, state.execution.device.stream);\n",
    "    // The unary candidates come from the checkpoint's dedicated proposal head when the\n"
    "    // run selects it (contract 6.1): its rows are shortlist entries, so only the global\n"
    "    // token ids reached through `text/draft_head_token_ids` may index the codebooks.\n"
    "    // The full-vocabulary head stays available for the default path.\n"
    "    const bool optimized_head = state.execution.proposal_head == ProposalHead::Optimized;\n"
    "    if (optimized_head && !state.execution.model.optimized_proposal.has_value()) {\n"
    "        throw std::logic_error(\"optimized DFlash2 proposal head is unavailable\");\n"
    "    }\n"
    "    const std::int32_t head_rows =\n"
    "        optimized_head ? Config::draft_head_rows : TextConfig::output_rows;\n"
    "    const std::int32_t head_domain =\n"
    "        optimized_head ? Config::draft_head_rows : TextConfig::token_domain;\n"
    "    const Tensor empty_head_ids;\n"
    "    const Tensor& head_token_ids = optimized_head\n"
    "                                       ? state.execution.model.optimized_proposal->token_ids\n"
    "                                       : empty_head_ids;\n"
    "    Tensor logits = state.execution.work.alloc(\n"
    "        DType::BF16, {head_rows, static_cast<std::int32_t>(k) * batch_size});\n"
    "    ops::linear(proposal_hidden,\n"
    "                optimized_head ? state.execution.model.optimized_proposal->head\n"
    "                               : state.execution.model.output_head,\n"
    "                logits, state.execution.device.stream);\n"
    "    // The shortlist head carries no output multiplier or softcap (contract 6.1).\n"
    "    if (!optimized_head) { kCfg.apply_final_logit_policy(logits, state.execution.device.stream); }\n",
    "dflash2: proposal head branch")
add(IMPL,
    "                          frame.draft_candidate_probs, steps, Config::selector_top_k,\n"
    "                          TextConfig::token_domain, state.execution.device.stream);\n",
    "                          frame.draft_candidate_probs, steps, Config::selector_top_k,\n"
    "                          head_domain, head_token_ids, state.execution.device.stream);\n",
    "dflash2: selector gets the mapped candidates")


def main() -> int:
    pathlib.Path(BAK).mkdir(parents=True, exist_ok=True)
    files = sorted({p for p, _, _, _ in EDITS})
    originals = {}
    for rel in files:
        raw = pathlib.Path(f"{BUILD}/{rel}").read_bytes()
        originals[rel] = raw
        dest = pathlib.Path(BAK) / rel.replace("/", "__")
        dest.write_bytes(raw)
        print(f"backup {rel:<58} {len(raw):>7} B  md5={hashlib.md5(raw).hexdigest()[:12]}")

    print()
    edited = dict(originals)
    for rel, old, new, note in EDITS:
        raw = edited[rel]
        if raw.count(old) != 1:
            for variant_old, variant_new in ((old, new),
                                             (old.replace(b"\n", b"\r\n"),
                                              new.replace(b"\n", b"\r\n"))):
                if raw.count(variant_old) == 1:
                    old, new = variant_old, variant_new
                    break
            else:
                print(f"FAIL {rel}: anchor occurs {raw.count(old)} times ({note})")
                print("     head:", old[:100])
                return 2
        edited[rel] = raw.replace(old, new)
        print(f"ok   {rel:<58} {note}")

    print()
    chunks = []
    for rel in files:
        before = originals[rel].decode("utf-8", "replace").splitlines(keepends=True)
        after = edited[rel].decode("utf-8", "replace").splitlines(keepends=True)
        chunks.append("".join(difflib.unified_diff(
            before, after, f"a/{rel}", f"b/{rel}", n=3)))
        pathlib.Path(f"{BUILD}/{rel}").write_bytes(edited[rel])

    patch = "".join(chunks)
    pathlib.Path(OUT_DIFF).write_text(patch, encoding="utf-8")
    added = sum(1 for line in patch.splitlines()
                if line.startswith("+") and not line.startswith("+++"))
    removed = sum(1 for line in patch.splitlines()
                  if line.startswith("-") and not line.startswith("---"))
    print(f"diff -> {OUT_DIFF}  (+{added}/-{removed}, {len(files)} files)")
    print("backups: " + BAK)
    return 0


if __name__ == "__main__":
    sys.exit(main())
