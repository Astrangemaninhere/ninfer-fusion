#include "ninfer/ops/cold_i8.h"

#include "ops/launcher/cold_i8.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops {

void cold_i8_slot_pack_raw(const std::uint8_t* src_codes, const std::uint8_t* src_scales,
                           int kv_heads, int page_count, std::uint8_t* slots,
                           std::int32_t* slot_valid, int slot_bytes, cudaStream_t stream) {
    detail::cold_i8_slot_pack_launch(src_codes, src_scales, kv_heads, page_count, slots,
                                     slot_valid, slot_bytes, stream);
}

void cold_i8_slot_restore_raw(const std::uint8_t* slots, int kv_heads, int page_count,
                              std::int8_t* dst_codes, void* dst_scales_fp16,
                              int slot_bytes, cudaStream_t stream) {
    detail::cold_i8_slot_restore_launch(slots, kv_heads, page_count, dst_codes,
                                        static_cast<__half*>(dst_scales_fp16), slot_bytes,
                                        stream);
}

// FIX-WOS-2 (WINDOW-ONE-SHOT). The two public entry points that
// include/ninfer/ops/cold_i8.h declares (BF16-COLD-LAND A5 / E5) and that
// program_impl.h:12563/12565/12585/12588 calls. The delivery shipped the declaration
// and the detail::..._launch definition but never this forwarder, so the link failed
// with 'undefined reference to ninfer::ops::cold_i8_slot_restore_{bf16,e8}_raw'.
// NAMEFIX5 (2026-09-18). `_e8_raw` -> `_rk4v4_raw` above, and why a second spelling was REFUSED.
// The rename to the Rk4v4 vocabulary had landed on the declaration
// (include/ninfer/ops/cold_i8.h:46) and on the caller (program_impl.h:12793/12796, four
// ProgramImplCore::restore_cold_page instantiations inside variant.cpp) but not here, so the
// engine link died at 100% of the build with
//     undefined reference to `ninfer::ops::cold_i8_slot_restore_rk4v4_raw(...)'
// This is the THIRD half-landing of this one rename in one night (the first left the forwarder
// out entirely; the second renamed the declaration and the caller; this one is the definition).
// So the fix is a spelling change and NOT a definition under the old name: an alias would keep
// BOTH vocabularies alive in this file, which is the state that produced the first two.
// NAMEFIX6 (2026-09-18, later the same night). The debt this note recorded is PAID, and the note
// is rewritten rather than deleted because it named COORDINATES that then stopped existing -- a
// comment naming what is not there is the same defect class as a symbol that does not resolve.
//   `cold_i8_slot_restore_e8_launch` -> `cold_i8_slot_restore_rk4v4_launch`
//        decl ops/launcher/cold_i8.h:24, def ops/launcher/cold_i8.cu:176, call :60 below
//   `cold_i8_slot_restore_e8_kernel` -> `cold_i8_slot_restore_rk4v4_kernel`
//        decl ops/kernel/cold_i8_kernels.cuh:67, def ops/launcher/cold_i8.cu:112, launch :186
// Six code sites in four files, landed together; NO alias, for the reason given above (an alias
// would keep both vocabularies resolvable in this very file). The `_e8_` RATCHET in
// tests/ops/test_cold_i8_symbol_pin.cpp is now pinned at 0, not 6 -- the cap was lowered with the
// landing, which is the only thing that may lower it.
// Deliberately NOT renamed: ops/launcher/cold_i8.cu:181 still READS "e8 cold restore needs the
// fixed int8 raw record". That is a runtime message, not a symbol and not a comment; nothing in
// the tree asserts its text, and changing it is a separate, separately-testable decision. It is
// listed here so the next reader is not left to find it by surprise.
// Signature and argument order mirror the declarations and the launchers exactly;
// the fixed-record (kColdI8SlotBytes) precondition is enforced inside the launcher.
void cold_i8_slot_restore_bf16_raw(const std::uint8_t* slots, int kv_heads, int page_count,
                                   void* dst_bf16, int slot_bytes, cudaStream_t stream) {
    detail::cold_i8_slot_restore_bf16_launch(slots, kv_heads, page_count,
                                            static_cast<__nv_bfloat16*>(dst_bf16), slot_bytes,
                                            stream);
}

void cold_i8_slot_restore_rk4v4_raw(const std::uint8_t* slots, int kv_heads, int page_count,
                                 std::int8_t* dst_codes, void* dst_scales_fp16,
                                 int slot_bytes, cudaStream_t stream) {
    detail::cold_i8_slot_restore_rk4v4_launch(slots, kv_heads, page_count, dst_codes,
                                           dst_scales_fp16, slot_bytes, stream);
}

} // namespace ninfer::ops
