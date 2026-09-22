// Derived from igorls/ninfer @ 5e4a66d (Apache-2.0), modified for this tree (ninfer-fusion).
#pragma once

#include "targets/qwen3_8_flash_next/impl/gdn_workspace.h"
#include "targets/qwen3_8_flash_next/impl/model_view.h"

#include <cuda_runtime.h>

namespace ninfer::targets::qwen3_8_flash_next::detail {

void flash_next_gdn_conv_launch(const FlashNextGdnWorkspace& scratch, const Tensor& convolution,
                                const Tensor& source_slots, const Tensor& destination_slots,
                                Tensor& convolution_states, cudaStream_t stream,
                                int batch_count = -1, int batch_offset = 0);
void flash_next_gdn_controls_launch(const Tensor& input, const GdnWeights& weights,
                                    FlashNextGdnWorkspace& scratch, cudaStream_t stream);
void flash_next_gdn_output_gate_launch(const FlashNextGdnWorkspace& scratch, const Tensor& norm,
                                       cudaStream_t stream);

} // namespace ninfer::targets::qwen3_8_flash_next::detail
