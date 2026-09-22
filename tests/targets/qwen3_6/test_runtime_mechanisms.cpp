#include "core/layout.h"
#include <ninfer/targets/qwen3_6/decoder_state.h>
#include <ninfer/targets/qwen3_6/hybrid_topology.h>
#include <ninfer/targets/qwen3_6/mtp_alignment.h>
#include <ninfer/targets/qwen3_6/round_state.h>
#include <ninfer/targets/qwen3_6/vision_control.h>

#include "targets/qwen3_6/impl/runtime/prefix_identity.h"
#include "targets/qwen3_6/impl/runtime/rebuild_work.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

namespace q36 = ninfer::targets::qwen3_6;

int failures = 0;

void expect(bool condition, std::string_view message) {
    if (condition) { return; }
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
}

void test_topology() {
    static_assert(q36::kHybridAttentionInterval == 4);
    static_assert(q36::full_attention_layers(64) == 16);
    static_assert(q36::gdn_layers(64) == 48);
    for (std::int32_t layer = 0; layer < 64; ++layer) {
        expect(q36::is_full_attention_layer(layer) == ((layer + 1) % 4 == 0), "hybrid layer kind");
        if (q36::is_full_attention_layer(layer)) {
            expect(q36::full_attention_index(layer) == layer / 4, "full-attention index");
        } else {
            expect(q36::gdn_index(layer) == layer - layer / 4, "GDN index");
        }
    }
}

q36::DecoderStateSpec decoder_spec(ninfer::DType dtype, bool mtp) {
    return q36::DecoderStateSpec{
        .full_attention_layers     = 2,
        .mtp_layers                = 1,
        .capacity                  = 129,
        .kv_heads                  = 2,
        .attention_head_dim        = 64,
        .kv_dtype                  = dtype,
        .kv_quant_group            = dtype == ninfer::DType::I8 ? q36::kKvInt8QuantGroup : 0,
        .enable_mtp                = mtp,
        .text_physical_page_groups = 5,
        .mtp_physical_page_groups  = mtp ? 4U : 0U,
    };
}

void test_decoder_layout() {
    ninfer::LayoutBuilder bf16_builder;
    const q36::DecoderStateLayout bf16 =
        q36::plan_decoder_state(bf16_builder, decoder_spec(ninfer::DType::BF16, false));
    (void)bf16_builder.finish(256);
    expect(bf16.text_kv.pages.planes.size() == 4, "BF16 Text KV has K/V planes per layer");
    expect(bf16.text_kv.pages.spec.page_group_count == 5 &&
               bf16.text_kv.execution_tables.spec.logical_page_capacity == 3 &&
               bf16.text_kv.execution_tables.spec.table_rows == 1,
           "Text KV separates five physical pages from three logical pages");
    expect(std::all_of(bf16.text_kv.pages.planes.begin(), bf16.text_kv.pages.planes.end(),
                       [](const ninfer::DeviceKVPlaneLayout& plane) {
                           return plane.geometry.dtype == ninfer::DType::BF16;
                       }),
           "BF16 KV has no scale planes");
    expect(!bf16.mtp_kv.has_value(), "disabled MTP omits KV storage");
    expect(bf16.kv_payload_bytes() == bf16.text_kv.payload_bytes(), "BF16 KV payload accounting");

    ninfer::LayoutBuilder int8_builder;
    const q36::DecoderStateLayout int8 =
        q36::plan_decoder_state(int8_builder, decoder_spec(ninfer::DType::I8, true));
    (void)int8_builder.finish(256);
    expect(int8.text_kv.pages.planes.size() == 8 &&
               int8.text_kv.pages.planes[2].geometry.dtype == ninfer::DType::FP16 &&
               int8.text_kv.pages.planes[3].geometry.dtype == ninfer::DType::FP16,
           "INT8 Text KV has code and scale planes per layer");
    expect(int8.mtp_kv.has_value() && int8.mtp_kv->layers == 1 &&
               int8.mtp_kv->pages.planes.size() == 4 &&
               int8.mtp_kv->pages.spec.page_group_count == 4 &&
               int8.mtp_kv->execution_tables.spec.logical_page_capacity == 3,
           "enabled MTP has one paged KV layer");
    expect(int8.mtp_kv && int8.mtp_kv->pages.planes[2].geometry.dtype == ninfer::DType::FP16 &&
               int8.mtp_kv->pages.planes[3].geometry.dtype == ninfer::DType::FP16,
           "INT8 MTP KV has scale planes");
    expect(int8.kv_payload_bytes() == int8.text_kv.payload_bytes() + int8.mtp_kv->payload_bytes(),
           "INT8 Text/MTP KV payload accounting");

    q36::DecoderStateSpec fp8_spec = decoder_spec(ninfer::DType::FP8_E4M3FN, true);
    // Two groups of head_dim: keeping the leading extent > 1 is what makes the assertion
    // below discriminating -- at head_dim == kKvFp8QuantGroup the extent is 1 either way,
    // so a row-scaled (one scale per token row) plane and a group-16 plane look alike.
    fp8_spec.attention_head_dim    = 2 * q36::kKvFp8QuantGroup;
    fp8_spec.kv_quant_group        = q36::kKvFp8QuantGroup;
    ninfer::LayoutBuilder fp8_builder;
    const q36::DecoderStateLayout fp8 = q36::plan_decoder_state(fp8_builder, fp8_spec);
    (void)fp8_builder.finish(256);
    // The fp8 tier is a packed-16 dtype like nvfp4/iso4e: FP8-E4M3FN codes plus E4M3FN group
    // scales (ops/wrapper/gqa_attention.cpp:80-82/:121-133, read by
    // gqa_attention_decode_fp8.cuh:159). An FP16 scale plane here is the bug that made every
    // fp8 run fail at that guard with "invalid NVFP4 KV cache scale dtype".
    expect(fp8.text_kv.pages.planes.size() == 8 &&
               fp8.text_kv.pages.planes[0].geometry.dtype == ninfer::DType::FP8_E4M3FN &&
               fp8.text_kv.pages.planes[0].geometry.leading_extent == 2 * q36::kKvFp8QuantGroup &&
               fp8.text_kv.pages.planes[2].geometry.dtype == ninfer::DType::FP8_E4M3FN &&
               fp8.text_kv.pages.planes[2].geometry.leading_extent == 2,
           "FP8 Text KV has E4M3FN code and group-16 E4M3FN scale planes per layer");
    expect(fp8.mtp_kv && fp8.mtp_kv->pages.planes.size() == 4 &&
               fp8.mtp_kv->pages.planes[0].geometry.dtype == ninfer::DType::FP8_E4M3FN &&
               fp8.mtp_kv->pages.planes[2].geometry.dtype == ninfer::DType::FP8_E4M3FN &&
               fp8.mtp_kv->pages.planes[2].geometry.leading_extent == 2,
           "FP8 MTP KV has E4M3FN code and group-16 E4M3FN scale planes");
    expect(fp8.kv_payload_bytes() == fp8.text_kv.payload_bytes() + fp8.mtp_kv->payload_bytes(),
           "FP8 Text/MTP KV payload accounting");
}

// kv_v_codec_check() must resolve a BF16 per-layer slot the same way plan_cache() does
// (selected = override == BF16 ? global : override, decoder_state.cpp layer_dtype()), because
// the layers that reach the kernels are the RESOLVED ones. A global nvfp4 dtype with an
// all-BF16 override table is therefore a stack of NVFP4 layers, and if such a layer keeps a V
// residual plane or sits in the entropy cold pool, its E2M1 V plane is decoded by a mechanism
// that only knows ISO4E -- V is silently mis-decoded. The guard used to read the raw override
// table, saw BF16, and continued, so the exact configuration both refusals were written for
// was accepted. Cases 1 and 2 below are the negative control: they MUST be refused, and
// before the fix they returned no refusal at all (the positive controls are case 0, which
// must stay accepted, and cases 3/4, which pins that the fix neither narrowed nor widened
// the guard).
void test_kv_v_codec_guard_resolves_bf16_slots() {
    const auto refusal = [](bool explicit_nvfp4_slots, bool residual, std::uint32_t cold_pages,
                            ninfer::KvVCodec codec,
                            ninfer::DType global = ninfer::DType::NVFP4) {
        q36::DecoderStateSpec spec = decoder_spec(global, false);
        spec.kv_v_codec     = codec;
        spec.max_cold_pages = cold_pages;
        if (explicit_nvfp4_slots) { spec.layer_kv_dtypes.fill(ninfer::DType::NVFP4); }
        if (residual) { spec.layer_residual[0] = true; }
        ninfer::LayoutBuilder builder;
        try {
            (void)q36::plan_decoder_state(builder, spec);
        } catch (const std::invalid_argument& error) {
            return std::string(error.what());
        }
        return std::string{};
    };
    const auto names_residual = [](const std::string& text) {
        return text.find("kv-v-codec e2m1: NVFP4 layer 0") != std::string::npos &&
               text.find("residual") != std::string::npos;
    };
    const auto names_cold = [](const std::string& text) {
        return text.find("cold pool") != std::string::npos;
    };

    // 0. Positive control: the ablation itself is legal, so the guard must stay silent.
    expect(refusal(false, false, 0, ninfer::KvVCodec::E2M1).empty(),
           "E2M1 V on an inherited NVFP4 stack with no residual and no cold pool is legal");
    // 1. The bypassed combination: an all-BF16 slot table on a global nvfp4 dtype IS an
    //    NVFP4 layer 0, whose residual plane decodes ISO4E only.
    const std::string inherited_residual = refusal(false, true, 0, ninfer::KvVCodec::E2M1);
    expect(!inherited_residual.empty(),
           "E2M1 V + a residual plane on a layer that is NVFP4 only by BF16 inheritance must "
           "be refused (it was silently accepted before the fix)");
    expect(names_residual(inherited_residual),
           "the refusal names layer 0 and the residual plane that would mis-decode V");
    // 2. The same hole in the cold-pool refusal.
    const std::string inherited_cold = refusal(false, false, 4, ninfer::KvVCodec::E2M1);
    expect(!inherited_cold.empty(),
           "E2M1 V + the cold pool on a layer that is NVFP4 only by BF16 inheritance must be "
           "refused (it was silently accepted before the fix)");
    expect(names_cold(inherited_cold),
           "the cold-pool refusal names the pool whose eviction requant is Iso4eVG16");
    // 3. The explicit table must keep being refused: the fix did not narrow the guard.
    const std::string explicit_residual = refusal(true, true, 0, ninfer::KvVCodec::E2M1);
    expect(!explicit_residual.empty() && names_residual(explicit_residual),
           "an explicitly NVFP4 slot with a residual plane is still refused, by name");
    expect(!refusal(true, false, 4, ninfer::KvVCodec::E2M1).empty(),
           "an explicitly NVFP4 slot in the cold pool is still refused");
    // 4. ISO4E V is what those two mechanisms expect, so it stays accepted.
    expect(refusal(false, true, 0, ninfer::KvVCodec::Iso3).empty(),
           "ISO4E V with a residual plane is accepted (the check is codec-specific)");
    expect(refusal(true, false, 4, ninfer::KvVCodec::Iso3).empty(),
           "ISO4E V with the cold pool is accepted");
    // 5. A BF16 stack has no NVFP4 layer at all: the DOMAIN refusal must fire, not the loops.
    const std::string bf16_stack =
        refusal(false, true, 0, ninfer::KvVCodec::E2M1, ninfer::DType::BF16);
    expect(!bf16_stack.empty() && !names_residual(bf16_stack),
           "an all-BF16 stack is refused by the domain check, not by the layer loops");
}

// The per-layer BF16 MASK (DecoderStateSpec::layer_kv_dtypes_set).
//
// DType::BF16 is also the "inherit the global --kv-dtype" sentinel, so the table
// alone cannot express "make THIS layer BF16": `--kv-layer-storage 0-11:bf16`
// under `--kv-dtype nvfp4` used to resolve every one of those layers to NVFP4,
// silently. The mask is the missing bit, and this test pins the resolution at the
// only place that commits it (plan_cache, through kv_resolve_slot_dtype).
//
// The assertion is the PLANE GEOMETRY, not just the dtype table, because that is
// what the pool is actually built from and what the kernels are launched against:
// a BF16 layer contributes 2 planes, an NVFP4 layer 4.
//
// It can fail in both directions, and the second half is the injection: the SAME
// table with the mask cleared MUST come out all-NVFP4. Deleting the mask from
// plan_cache()'s layer_dtype() turns the first half red; honouring the mask where
// the historical rule is meant to apply turns the second half red.
void test_per_layer_bf16_mask() {
    const auto plan = [](bool mask_written) {
        q36::DecoderStateSpec spec = decoder_spec(ninfer::DType::NVFP4, false);
        spec.full_attention_layers        = 2;
        spec.layer_kv_dtypes[0]           = ninfer::DType::BF16;
        spec.layer_kv_dtypes[1]           = ninfer::DType::NVFP4;
        spec.layer_kv_dtypes_set[0]       = mask_written;
        spec.layer_kv_dtypes_set[1]       = true;
        ninfer::LayoutBuilder builder;
        q36::DecoderStateLayout layout = q36::plan_decoder_state(builder, spec);
        (void)builder.finish(256);
        return layout;
    };

    // Written: layer 0 is a REAL BF16 layer under a global NVFP4 dtype.
    const q36::DecoderStateLayout written = plan(true);
    expect(written.text_kv.layer_dtypes[0] == ninfer::DType::BF16,
           "a slot the spec WROTE as bf16 stays bf16 under --kv-dtype nvfp4 "
           "(this is the per-layer BF16 baseline that had no spelling before the mask)");
    expect(written.text_kv.layer_dtypes[1] == ninfer::DType::NVFP4,
           "the next layer of the same spec keeps the nvfp4 it named");
    expect(written.text_kv.layer_plane_base[0] == 0 && written.text_kv.layer_plane_base[1] == 2,
           "the mixed BF16 + NVFP4 pool gives layer 0 two planes and layer 1 four, "
           "so layer 1 starts at plane 2");
    expect(written.text_kv.pages.planes.size() == 6 &&
               written.text_kv.pages.planes[0].geometry.dtype == ninfer::DType::BF16 &&
               written.text_kv.pages.planes[2].geometry.dtype == ninfer::DType::U8,
           "the page geometry really carries 2 BF16 planes then 4 NVFP4 planes");

    // Not written: the SAME table is the historical inheritance rule -- both layers
    // are NVFP4. This is the injection's target: if plan_cache() honours the table
    // without the mask, this half goes red, and with it the claim that every
    // pre-mask configuration is unchanged.
    const q36::DecoderStateLayout inherited = plan(false);
    expect(inherited.text_kv.layer_dtypes[0] == ninfer::DType::NVFP4,
           "an UNWRITTEN bf16 slot still inherits the global dtype (all-false mask is the "
           "pre-mask rule, bit-for-bit)");
    expect(inherited.text_kv.layer_plane_base[1] == 4 &&
               inherited.text_kv.pages.planes.size() == 8,
           "with the mask clear both layers are NVFP4: 4 + 4 planes");
}

void test_round_layout() {
    ninfer::LayoutBuilder builder;
    q36::RoundStateLayout round = q36::begin_round_state_layout(
        builder, q36::RoundStateSpec{
                     .hidden = 32, .output_rows = 128, .draft_window = 5, .enable_mtp = true});
    const ninfer::TensorRegion exact_prefill =
        builder.add_tensor(ninfer::DType::BF16, {32, 16}, 256, "exact prefill hidden");
    q36::complete_round_state_layout(builder, round);
    (void)builder.finish(256);
    expect(round.complete, "round layout completes");
    expect(round.logits.shape[0] == 128 && round.logits.shape[1] == 1, "round logits shape");
    expect(round.mtp.has_value() && round.mtp->draft_tokens.shape[0] == 5 &&
               round.mtp->target_input_ids.shape[0] == 6,
           "MTP prefill scratch shapes");
    expect(round.logits.region.offset < exact_prefill.region.offset &&
               exact_prefill.region.offset < round.mtp->draft_tokens.region.offset,
           "exact prefill extension retains established round-region order");
    expect(round.mtp.has_value() && round.mtp->position.shape[0] == 1,
           "MTP prefill scratch is explicit");
    expect(round.mtp_decode.has_value() && round.mtp_decode->alignment_ids.shape[0] == 6 &&
               round.mtp_decode->alignment_ids.shape[1] == 1,
           "MTP decode frame is explicit");

    ninfer::LayoutBuilder speculative_builder;
    q36::RoundStateLayout dflash = q36::begin_round_state_layout(
        speculative_builder,
        q36::RoundStateSpec{
            .hidden = 32, .output_rows = 128, .draft_window = 15, .enable_dflash = true});
    q36::complete_round_state_layout(speculative_builder, dflash);
    (void)speculative_builder.finish(256);
    expect(dflash.logits.shape[1] == 1 && dflash.dflash_prefill.has_value() &&
               dflash.dflash_prefill->produced_count.shape[0] == 1 &&
               dflash.dflash_decode.has_value() &&
               dflash.dflash_decode->draft_tokens.shape[0] == 15,
           "K=15 DFlash storage is backend-owned");
    expect(!dflash.mtp.has_value() && !dflash.mtp_decode.has_value(),
           "DFlash layout does not allocate MTP storage");
}

void test_mtp_alignment() {
    const std::vector<std::int32_t> scatter{2, 4, 7};
    const q36::MtpAlignmentWindow first = q36::plan_mtp_alignment_window(8, 0, 4);
    expect(first.hidden_begin == 0 && first.position_begin == 0 &&
               first.shifted_embedding_begin == 1 && first.columns == 4 &&
               !first.final_column_uses_generated_token,
           "non-final MTP alignment window");
    const q36::MtpVisualOverlap first_visual = q36::shifted_visual_overlap(scatter, 8, first);
    expect(first_visual.source_begin == 0 &&
               first_visual.destination_columns == std::vector<std::int32_t>({1, 3}),
           "non-final shifted visual overlap");

    const q36::MtpAlignmentWindow final = q36::plan_mtp_alignment_window(8, 4, 4);
    expect(final.shifted_embedding_begin == 5 && final.final_column_uses_generated_token,
           "final MTP alignment window");
    const q36::MtpVisualOverlap final_visual = q36::shifted_visual_overlap(scatter, 8, final);
    expect(final_visual.source_begin == 2 &&
               final_visual.destination_columns == std::vector<std::int32_t>({2}),
           "final shifted visual overlap excludes generated-token column");
}

void test_vision_control() {
    q36::PreparedPromptData prompt;
    prompt.token_ids.resize(7);
    prompt.token_types           = {0, static_cast<std::uint8_t>(q36::PromptModality::Image),
                                    0, static_cast<std::uint8_t>(q36::PromptModality::Video),
                                    0, static_cast<std::uint8_t>(q36::PromptModality::Video),
                                    0};
    prompt.prepare.media_items   = 2;
    prompt.prepare.raw_patches   = 12;
    prompt.prepare.vision_tokens = 3;
    prompt.vision_items          = {
        q36::VisionItem{.modality    = q36::PromptModality::Image,
                                 .grid        = {.temporal = 1, .height = 2, .width = 2},
                                 .patch_begin = 0,
                                 .patch_count = 4,
                                 .token_spans = {{.begin = 1, .count = 1}}},
        q36::VisionItem{.modality    = q36::PromptModality::Video,
                                 .grid        = {.temporal = 2, .height = 2, .width = 2},
                                 .patch_begin = 4,
                                 .patch_count = 8,
                                 .token_spans = {{.begin = 3, .count = 1}, {.begin = 5, .count = 1}}},
    };

    const q36::VisionControlPlan plan = q36::plan_vision_control(prompt);
    const q36::VisionControl control  = q36::build_vision_control(prompt, plan, 0);
    expect(control.items.size() == 2, "Vision per-item control count");
    expect(control.items[0].patch_begin == 0 && control.items[0].patch_count == 4 &&
               control.items[0].merged_count == 1 && control.items[0].segment_length == 4 &&
               control.items[0].segment_count == 1 &&
               control.items[0].scatter_indices == std::vector<std::int32_t>({1}) &&
               control.items[0].position_ids.size() == 8 &&
               control.items[0].position_table_indices.size() == 16 &&
               control.items[0].position_table_weights.size() == 16,
           "image item control offsets");
    expect(control.items[1].patch_begin == 4 && control.items[1].patch_count == 8 &&
               control.items[1].merged_count == 2 && control.items[1].segment_length == 4 &&
               control.items[1].segment_count == 2 &&
               control.items[1].scatter_indices == std::vector<std::int32_t>({3, 5}) &&
               control.items[1].position_ids.size() == 16 &&
               control.items[1].position_table_indices.size() == 32 &&
               control.items[1].position_table_weights.size() == 32,
           "video item control offsets");

    const q36::VisionControl suffix = q36::build_vision_control(prompt, plan, 1);
    expect(suffix.prepared_item_begin == 1 && suffix.items.size() == 1 &&
               suffix.items[0].patch_begin == control.items[1].patch_begin &&
               suffix.items[0].scatter_indices == control.items[1].scatter_indices &&
               suffix.items[0].position_ids == control.items[1].position_ids,
           "Vision suffix control contents");
}

q36::PreparedPromptData identity_prompt(std::uint8_t digest_byte = 1) {
    q36::PreparedPromptData prompt;
    prompt.token_ids   = {10, 248056, 248056, 11};
    prompt.token_types = {0, static_cast<std::uint8_t>(q36::PromptModality::Image),
                          static_cast<std::uint8_t>(q36::PromptModality::Image), 0};
    prompt.positions   = {0, 1, 1, 3, 0, 1, 1, 3, 0, 1, 2, 3};
    prompt.rope_delta  = 0;
    q36::VisionItem item{.modality    = q36::PromptModality::Image,
                         .grid        = {.temporal = 1, .height = 2, .width = 4},
                         .patch_begin = 0,
                         .patch_count = 8,
                         .token_spans = {{.begin = 1, .count = 2}}};
    item.content_digest.fill(digest_byte);
    prompt.vision_items.push_back(std::move(item));
    return prompt;
}

void append_text_token(q36::PreparedPromptData& prompt, ninfer::TokenId token,
                       std::int32_t position) {
    const std::size_t old_tokens = prompt.token_ids.size();
    std::vector<std::int32_t> positions;
    positions.reserve(3 * (old_tokens + 1));
    for (std::size_t axis = 0; axis < 3; ++axis) {
        const auto begin =
            prompt.positions.begin() + static_cast<std::ptrdiff_t>(axis * old_tokens);
        positions.insert(positions.end(), begin, begin + static_cast<std::ptrdiff_t>(old_tokens));
        positions.push_back(position);
    }
    prompt.token_ids.push_back(token);
    prompt.token_types.push_back(0);
    prompt.positions = std::move(positions);
}

void test_prefix_identity() {
    q36::PreparedPromptData original    = identity_prompt();
    std::vector<ninfer::TokenId> ledger = original.token_ids;
    q36::detail::ResidentPrefixIdentity resident;
    q36::detail::PrefixShortlistDigests digests;
    resident.reserve(16);
    resident.assign(original);
    digests.reserve(16);
    digests.assign(original);

    expect(q36::detail::prefix_matches(original, ledger, resident, original.token_ids.size()),
           "identical multimodal prefix identity");

    q36::PreparedPromptData changed_media = identity_prompt(2);
    expect(!q36::detail::prefix_matches(changed_media, ledger, resident,
                                        changed_media.token_ids.size()),
           "different media content must not reuse placeholder tokens");
    expect(q36::detail::prefix_matches(changed_media, ledger, resident, 1),
           "media wholly after the frontier does not affect prefix identity");
    expect(!q36::detail::prefix_matches(original, ledger, resident, 2),
           "frontier must not divide one Vision item");

    q36::PreparedPromptData changed_position = identity_prompt();
    changed_position.positions[0] += 1;
    expect(!q36::detail::prefix_matches(changed_position, ledger, resident,
                                        changed_position.token_ids.size()),
           "different MRoPE positions must not reuse resident state");

    q36::PreparedPromptData changed_decomposition              = identity_prompt();
    changed_decomposition.identity.rewrite_execution_frontiers = {1};
    expect(!q36::detail::prefix_matches(changed_decomposition, ledger, resident,
                                        changed_decomposition.token_ids.size()),
           "different GDN execution decomposition must not reuse resident state");
    changed_decomposition.identity.rewrite_execution_frontiers = {4};
    expect(q36::detail::prefix_matches(changed_decomposition, ledger, resident, 3),
           "execution decomposition wholly after the frontier changed prefix identity");

    q36::PreparedPromptData resident_future              = identity_prompt();
    resident_future.identity.rewrite_execution_frontiers = {1, 4};
    q36::detail::ResidentPrefixIdentity resident_with_future;
    resident_with_future.assign(resident_future);
    q36::PreparedPromptData incoming_future              = identity_prompt();
    incoming_future.identity.rewrite_execution_frontiers = {1, 3};
    expect(q36::detail::prefix_matches(incoming_future, ledger, resident_with_future, 1),
           "resident execution decomposition after the frontier changed prefix identity");
    expect(!q36::detail::prefix_matches(incoming_future, ledger, resident_with_future, 3),
           "different execution decomposition inside the frontier reused resident state");

    q36::detail::PrefixShortlistDigests future_digest;
    future_digest.assign(resident_future);
    q36::detail::PrefixShortlistDigests incoming_digest;
    incoming_digest.assign(incoming_future);
    expect(future_digest.at(1) == incoming_digest.at(1),
           "future execution boundaries changed an earlier content shortlist");
    expect(future_digest.at(3) != incoming_digest.at(3),
           "different in-prefix execution boundaries shared a shortlist digest");

    resident.append_generated(1, original.rope_delta);
    ledger.push_back(12);
    const std::array<ninfer::TokenId, 1> generated{12};
    digests.append_generated(generated, original.rope_delta);
    append_text_token(original, 12, 4);
    q36::detail::PrefixShortlistDigests rebuilt;
    rebuilt.assign(original);
    expect(digests.at(ledger.size()) == rebuilt.at(ledger.size()),
           "incremental generated-token shortlist diverged from a full rebuild");
    expect(q36::detail::prefix_matches(original, ledger, resident, ledger.size()),
           "generated multimodal continuation identity");

    const q36::PreparedPromptData prompt_only = identity_prompt();
    resident.truncate(prompt_only.token_ids.size());
    digests.truncate(prompt_only.token_ids.size());
    ledger.resize(prompt_only.token_ids.size());
    q36::detail::PrefixShortlistDigests prompt_digest;
    prompt_digest.assign(prompt_only);
    expect(digests.at(ledger.size()) == prompt_digest.at(ledger.size()),
           "truncated shortlist did not restore the original frontier digest");
    expect(q36::detail::prefix_matches(prompt_only, ledger, resident, ledger.size()),
           "truncated multimodal continuation identity");
}

void test_rebuild_work_prompt_frontier_boundary() {
    constexpr std::uint32_t prompt_tokens = 100;
    constexpr std::uint32_t prefill_chunk = 2048;
    std::uint32_t tail_begin              = 0;
    q36::runtime_support::include_rebuild_boundary(tail_begin, prompt_tokens, prompt_tokens);
    expect(tail_begin == prompt_tokens,
           "prompt-frontier rebuild boundary was not retained for continuation growth");

    ninfer::runtime::PrefillWork work =
        ninfer::runtime::make_prefill_work(0, prompt_tokens, 0, 0, prefill_chunk);
    q36::runtime_support::advance_segmented_rebuild_work(work, tail_begin, prompt_tokens,
                                                         prompt_tokens + 1, prefill_chunk);
    const ninfer::runtime::PrefillWork exact =
        ninfer::runtime::make_prefill_work(0, prompt_tokens + 1, 0, 0, prefill_chunk);
    expect(work.chunks == 2 && work.tokens == exact.tokens &&
               work.attention_pairs == exact.attention_pairs,
           "continuation growth did not preserve the prompt-frontier rebuild split");
}


// The prefill unit is not a constant. bandwidth_governor_.prefill_chunk_for() installs
// a value in [128, prefill_chunk_capacity()] between two engine steps
// (engine_core.h:2150-2172), so a co-resident decode request makes it SHRINK while a
// prefill is in flight. `advance_segmented_rebuild_work` used to recompute the OLD
// tail's contribution at the SMALLER unit, overshoot the recorded count and throw
// "sequence rebuild chunk accounting is invalid" -- which killed the request
// (observed twice, A3 and V3; two other shrinks to 128 finished, so it is timing
// dependent). A shrink is a legal event and must not be an error.
void test_rebuild_work_prefill_unit_shrink() {
    constexpr std::uint32_t wide   = 3072;
    constexpr std::uint32_t narrow = 128;
    ninfer::runtime::PrefillWork work = ninfer::runtime::make_prefill_work(0, wide, 0, 0, wide);
    expect(work.chunks == 1, "one wide unit did not charge exactly one chunk");

    // the frontier advances one narrow unit while the unit collapses to its floor
    q36::runtime_support::advance_segmented_rebuild_work(work, /*tail_begin=*/0, wide, wide + narrow,
                                                         narrow);
    const ninfer::runtime::PrefillWork exact =
        ninfer::runtime::make_prefill_work(0, wide + narrow, 0, 0, narrow);
    expect(work.chunks == exact.chunks && work.tokens == exact.tokens &&
               work.attention_pairs == exact.attention_pairs,
           "a mid-stream prefill-unit shrink did not re-account the rebuild work at the unit "
           "in force");

    // and the re-accounted state is exact for the advances that follow it
    q36::runtime_support::advance_segmented_rebuild_work(work, /*tail_begin=*/0, wide + narrow,
                                                         wide + 2 * narrow, narrow);
    const ninfer::runtime::PrefillWork exact2 =
        ninfer::runtime::make_prefill_work(0, wide + 2 * narrow, 0, 0, narrow);
    expect(work.chunks == exact2.chunks && work.tokens == exact2.tokens &&
               work.attention_pairs == exact2.attention_pairs,
           "the rebuild accounting was not stable after a prefill-unit shrink");
}

} // namespace

int main() {
    test_topology();
    test_decoder_layout();
    test_kv_v_codec_guard_resolves_bf16_slots();
    test_per_layer_bf16_mask();
    test_round_layout();
    test_mtp_alignment();
    test_vision_control();
    test_prefix_identity();
    test_rebuild_work_prompt_frontier_boundary();
    test_rebuild_work_prefill_unit_shrink();
    if (failures != 0) {
        std::cerr << failures << " Qwen3.6 runtime mechanism checks failed\n";
        return 1;
    }
    std::cout << "Qwen3.6 runtime mechanism checks passed\n";
    return 0;
}
