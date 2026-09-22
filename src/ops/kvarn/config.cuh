// PROVENANCE -- FORK-SURVEY borrow (Apache-2.0).
// Source repo : MirkoCovizzi/ninfer-rtx5090-mobile
// Branch      : feat/kvarn-production
// Commit      : 750aac554d52b758c3ee33eff81f08bf1dceb201
// Source path : src/ops/kvarn/config.cuh
// sha256(src) : e6c5afd98a07359242f2662038e822e63c96d15c3bddbd5db9958151614fcc39
// Landed by   : /home/user/scratch/PATCHSET/MERGE (fork-survey merge, 2026-09-18)
// Upstream    : Neroued/ninfer, Apache-2.0, ships NO NOTICE -- attribution is on us.
// Status      : ADDITIVE, NOT wired into any build target (src/CMakeLists.txt is explicit, not GLOB), so this file is inert until someone adds it deliberately.
// Note        : the codecs in this family label a rotation axis -- rk<N>v<M> = rotated K at N bits against V at M bits (rk4v4-e8 = rotated K 4b, V 4b, rk4v4 in the mix), which is a DIFFERENT axis from src/kvcfg/kv_formats.h's tier vocabulary (hot/tail/cold, Rk4v4).
#pragma once

// Fixed native profile from Huawei KVarN commit 7586257f1c632e63187bfacbbe21ccb51540f7b3,
// vllm/model_executor/layers/quantization/kvarn/config.py: kvarn_k4v2_g128, D256.

namespace ninfer::ops::kvarn {

inline constexpr int D                 = 256;
inline constexpr int Group             = 128;
inline constexpr int KBits             = 4;
inline constexpr int VBits             = 2;
inline constexpr int Iterations        = 8;
inline constexpr int PrefillSlabTokens = 16384;
inline constexpr int MtpPackedWindow   = 1024;
inline constexpr int PackedQueryChunk  = 16;
inline constexpr int DecodeMidWindow   = 122880;
inline constexpr int DecodeMidSplits   = 41;
inline constexpr int DecodeLongSplits  = 82;

inline constexpr float StdMin = 1.0e-3F;
inline constexpr float StdMax = 1.0e3F;
inline constexpr float LogMin = -0.3F;
inline constexpr float LogMax = 10.0F;

} // namespace ninfer::ops::kvarn
