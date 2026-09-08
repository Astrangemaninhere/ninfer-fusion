// ninfer QPN port: SM70-family NVFP4 skinny GEMM device kernels.
// Source: v100-skinny kernels/skinny_kernels.cu (Apache-2.0, 1Cat lineage),
// device section extracted verbatim (no torch dependencies).
// Packed layout (0.5625 B/weight): codes u8 [N][K/2] e2m1x2, scales u8
// [N][K/16] e4m3 per-16, gscale f32 epilogue. See qpn_host.cu wrappers.
#pragma once

#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops::qpn {


#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

#define DEV_INLINE __device__ __forceinline__

// PRMT-LUT decoder (compile with -DSKINNY_LUT_CVT to select): loses
// to the TurboMind-derived shift+rebias decoder below by ~28% at M=1
// (504 vs 647 GB/s) and ~22% at M=16 - longer PRMT dependency chain
// and more INT-pipe ops per value. Kept for A/B reference.
// fp16 high bytes of the e2m1 magnitudes {0,.5,1,1.5} and {2,3,4,6};
// the low bytes are all 0x00, so one PRMT yields two packed halves.
constexpr unsigned LUT_LO = 0x3E3C3800u;
constexpr unsigned LUT_HI = 0x46444240u;

// Dequant one byte-pair position of an 8-code word. `q` holds 8 nibbles;
// byte `pi` gives codes (2p, 2p+1) -> returns them as one half2.
DEV_INLINE half2 dequant_pair(unsigned q, int pi, half2 sc2) {
  const unsigned mq = (q & 0x77777777u) >> (8 * pi);
  const unsigned sq = (q & 0x88888888u) >> (8 * pi);
  unsigned sel = ((mq & 0x7u) << 4) | ((mq & 0x70u) << 8);
  unsigned h = __byte_perm(LUT_LO, LUT_HI, sel);
  h |= ((sq & 0x8u) << 12) | ((sq & 0x80u) << 24);
  return __hmul2(*reinterpret_cast<half2 *>(&h), sc2);
}

DEV_INLINE half2 fp8e4m3_to_half2(unsigned char b) {
  const unsigned short hb =
      (((unsigned short)b & 0x80u) << 8) | (((unsigned short)b & 0x7Fu) << 7);
  const half hs = __hmul(__ushort_as_half(hb), __ushort_as_half(0x5C00));  // *256
  return __halves2half2(hs, hs);
}

// XOR swizzle on the low 3 bits of a k-pair index; conflict-free for the
// simt read pattern (lane-groups sharing a bank base differ in p>>5).
DEV_INLINE int swz(int p) { return (p & ~7) | ((p ^ (p >> 5)) & 7); }

#ifndef SKINNY_LUT_CVT
// Alternative e2m1 decoder derived from TurboMind's cvt_f16x8_e2m1
// (Apache-2.0; 1Cat-vLLM csrc/sm70_turbomind/lmdeploy/src/turbomind/
// kernels/attention/quantization.h). Shifts sign/EM bits into fp16
// positions; the 2^14 exponent re-bias is folded into the caller's
// scale, so no extra multiply. Output half2 pairing is INTERLEAVED:
// out[i] holds codes (i, i+4) of the 8-code word.
DEV_INLINE void dequant8_tm(unsigned q, half2 sc2p, half2 out[4]) {
  constexpr unsigned S = 0x80008000u, EM = 0x0E000E00u;
  unsigned v0 = ((q << 12) & S) | ((q << 9) & EM);
  unsigned v1 = ((q << 8) & S) | ((q << 5) & EM);
  unsigned v2 = ((q << 4) & S) | ((q << 1) & EM);
  unsigned v3 = (q & S) | ((q >> 3) & EM);
  out[0] = __hmul2(*reinterpret_cast<half2 *>(&v0), sc2p);
  out[1] = __hmul2(*reinterpret_cast<half2 *>(&v1), sc2p);
  out[2] = __hmul2(*reinterpret_cast<half2 *>(&v2), sc2p);
  out[3] = __hmul2(*reinterpret_cast<half2 *>(&v3), sc2p);
}
#endif

// Stage 8 contiguous activation halves as four half2 pairs. Default
// pairing is adjacent-k; the TM decoder variant needs (k, k+4) pairs to
// match dequant8_tm's interleaved output.
DEV_INLINE void stage_pairs(half2 *dst, int base_pair, const uint4 &v) {
#ifndef SKINNY_LUT_CVT
  const unsigned *r = reinterpret_cast<const unsigned *>(&v);
  unsigned o[4] = {__byte_perm(r[0], r[2], 0x5410),
                   __byte_perm(r[0], r[2], 0x7632),
                   __byte_perm(r[1], r[3], 0x5410),
                   __byte_perm(r[1], r[3], 0x7632)};
#pragma unroll
  for (int j = 0; j < 4; j++)
    dst[swz(base_pair + j)] = *reinterpret_cast<half2 *>(&o[j]);
#else
  const half2 *hv = reinterpret_cast<const half2 *>(&v);
#pragma unroll
  for (int j = 0; j < 4; j++) dst[swz(base_pair + j)] = hv[j];
#endif
}

// ---------------------------------------------------------------------------
// SIMT kernel: 8 warps/block, one output row per warp.
// ---------------------------------------------------------------------------
template <int M, int KC, int R = 1, bool ARGMAX = false>
__global__ void skinny_nvfp4_simt(const uint8_t *__restrict__ codes,
                                  const uint8_t *__restrict__ scales,
                                  const half *__restrict__ x,
                                  half *__restrict__ y, int N, int K,
                                  float gscale, half *__restrict__ bvals,
                                  int *__restrict__ bidxs) {
  extern __shared__ char smem_raw[];
  half2 *xs = reinterpret_cast<half2 *>(smem_raw);  // [M][KC/2] swizzled
  constexpr int P2 = KC / 2;

  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int n = (blockIdx.x * 8 + warp) * R;
  const uint8_t *crow[R];
  const uint8_t *srow[R];
#pragma unroll
  for (int r = 0; r < R; r++) {
    crow[r] = codes + (size_t)(n + r) * (K >> 1);
    srow[r] = scales + (size_t)(n + r) * (K >> 4);
  }
  // Fold the global scale into the group scales so in-kernel weights sit
  // at their true O(0.1) magnitudes; otherwise code*fp8scale reaches
  // ~2.7e3 and fp16 products overflow on real activation outliers.
#ifndef SKINNY_LUT_CVT
  // dequant8_tm needs a 2^14 exponent re-bias; fold it here for free.
  const half2 gm2 = __float2half2_rn(gscale * 16384.f);
#else
  const half2 gm2 = __float2half2_rn(gscale);
#endif

  float accf[R][M];
#pragma unroll
  for (int r = 0; r < R; r++)
#pragma unroll
    for (int m = 0; m < M; m++) accf[r][m] = 0.f;

  int k0 = 0;
  for (; k0 + KC <= K; k0 += KC) {
    __syncthreads();
    for (int idx = threadIdx.x; idx < M * (KC / 8); idx += blockDim.x) {
      const int m = idx / (KC / 8), j4 = idx % (KC / 8);
      const uint4 v =
          *reinterpret_cast<const uint4 *>(x + (size_t)m * K + k0 + j4 * 8);
      stage_pairs(xs + m * P2, j4 * 4, v);
    }
    __syncthreads();

#pragma unroll
    for (int i = 0; i < KC / 512; i++) {
      const int s = lane + 32 * i;  // 16-code segment == one scale group
      uint2 q2[R];
      half2 sc2[R];
#pragma unroll
      for (int r = 0; r < R; r++) {
        q2[r] = *reinterpret_cast<const uint2 *>(crow[r] + (k0 >> 1) + s * 8);
        sc2[r] = __hmul2(fp8e4m3_to_half2(srow[r][(k0 >> 4) + s]), gm2);
      }
      // fp16 accumulation window is one 16-code segment (8 products per
      // half2 lane); flushed to fp32 so real activation outliers cannot
      // overflow half range.
      half2 acch[R][M];
#pragma unroll
      for (int r = 0; r < R; r++)
#pragma unroll
        for (int m = 0; m < M; m++) acch[r][m] = __float2half2_rn(0.f);
#pragma unroll
      for (int w = 0; w < 2; w++) {
        half2 w4[R][4];
#pragma unroll
        for (int r = 0; r < R; r++) {
          const unsigned qw = w == 0 ? q2[r].x : q2[r].y;
#ifndef SKINNY_LUT_CVT
          dequant8_tm(qw, sc2[r], w4[r]);
#else
#pragma unroll
          for (int pi = 0; pi < 4; pi++)
            w4[r][pi] = dequant_pair(qw, pi, sc2[r]);
#endif
        }
#pragma unroll
        for (int pi = 0; pi < 4; pi++) {
          const int psw = swz(s * 8 + w * 4 + pi);
#pragma unroll
          for (int m = 0; m < M; m++) {
            const half2 xv = xs[m * P2 + psw];
#pragma unroll
            for (int r = 0; r < R; r++)
              acch[r][m] = __hfma2(w4[r][pi], xv, acch[r][m]);
          }
        }
      }
#pragma unroll
      for (int r = 0; r < R; r++)
#pragma unroll
        for (int m = 0; m < M; m++) {
          const float2 f = __half22float2(acch[r][m]);
          accf[r][m] += f.x + f.y;
        }
    }
  }

  // Tail chunk: K % KC remainder (any multiple of 128). Same layout and
  // swizzle, runtime segment bound with idle-lane guard.
  const int tail = K - k0;
  if (tail > 0) {
    __syncthreads();
    for (int idx = threadIdx.x; idx < M * (tail / 8); idx += blockDim.x) {
      const int m = idx / (tail / 8), j4 = idx % (tail / 8);
      const uint4 v =
          *reinterpret_cast<const uint4 *>(x + (size_t)m * K + k0 + j4 * 8);
      stage_pairs(xs + m * P2, j4 * 4, v);
    }
    __syncthreads();
    const int nseg = tail >> 4;
    for (int s = lane; s < nseg; s += 32) {
      uint2 q2[R];
      half2 sc2[R];
#pragma unroll
      for (int r = 0; r < R; r++) {
        q2[r] = *reinterpret_cast<const uint2 *>(crow[r] + (k0 >> 1) + s * 8);
        sc2[r] = __hmul2(fp8e4m3_to_half2(srow[r][(k0 >> 4) + s]), gm2);
      }
      half2 acch[R][M];
#pragma unroll
      for (int r = 0; r < R; r++)
#pragma unroll
        for (int m = 0; m < M; m++) acch[r][m] = __float2half2_rn(0.f);
#pragma unroll
      for (int w = 0; w < 2; w++) {
        half2 w4[R][4];
#pragma unroll
        for (int r = 0; r < R; r++) {
          const unsigned qw = w == 0 ? q2[r].x : q2[r].y;
#ifndef SKINNY_LUT_CVT
          dequant8_tm(qw, sc2[r], w4[r]);
#else
#pragma unroll
          for (int pi = 0; pi < 4; pi++)
            w4[r][pi] = dequant_pair(qw, pi, sc2[r]);
#endif
        }
#pragma unroll
        for (int pi = 0; pi < 4; pi++) {
          const int psw = swz(s * 8 + w * 4 + pi);
#pragma unroll
          for (int m = 0; m < M; m++) {
            const half2 xv = xs[m * P2 + psw];
#pragma unroll
            for (int r = 0; r < R; r++)
              acch[r][m] = __hfma2(w4[r][pi], xv, acch[r][m]);
          }
        }
      }
#pragma unroll
      for (int r = 0; r < R; r++)
#pragma unroll
        for (int m = 0; m < M; m++) {
          const float2 f = __half22float2(acch[r][m]);
          accf[r][m] += f.x + f.y;
        }
    }
  }

  if constexpr (!ARGMAX) {
#pragma unroll
    for (int r = 0; r < R; r++)
#pragma unroll
      for (int m = 0; m < M; m++) {
        float v = accf[r][m];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(~0u, v, o);
        if (lane == 0) y[(size_t)m * N + n + r] = __float2half(v);
      }
  } else {
    // Fused greedy argmax (M=1): identical reduce, identical
    // __float2half rounding, then compare halfs with strict > so ties
    // keep the LOWEST index — matching argmax-over-half semantics of
    // the separate path. One (val, idx) pair per block; no logits hit
    // HBM.
    __shared__ half wval[8];
    __shared__ int widx[8];
    half best_h = __float2half(-3.0e38f);
    int best_i = -1;
#pragma unroll
    for (int r = 0; r < R; r++) {
      float v = accf[r][0];
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(~0u, v, o);
      if (lane == 0) {
        const half h = __float2half(v);
        if (__hgt(h, best_h)) { best_h = h; best_i = n + r; }
      }
    }
    if (lane == 0) { wval[warp] = best_h; widx[warp] = best_i; }
    __syncthreads();
    if (threadIdx.x == 0) {
      half bh = wval[0];
      int bi = widx[0];
#pragma unroll
      for (int w = 1; w < 8; w++)
        if (__hgt(wval[w], bh)) { bh = wval[w]; bi = widx[w]; }
      bvals[blockIdx.x] = bh;
      bidxs[blockIdx.x] = bi;
    }
  }
}

// ---------------------------------------------------------------------------
// WMMA kernel: WN x WM warps of 16x16 output tiles, KC-deep smem staging.
// Software-pipelined: while the tensor cores chew on chunk i, each thread's
// gmem loads for chunk i+1 are already in flight into registers.
// ---------------------------------------------------------------------------
template <int WN, int WM, int KC>
__global__ void skinny_nvfp4_wmma(const uint8_t *__restrict__ codes,
                                  const uint8_t *__restrict__ scales,
                                  const half *__restrict__ x,
                                  half *__restrict__ y, int N, int K,
                                  int m_real, float gscale) {
  constexpr int NT = WN * 16, MT = WM * 16;
  constexpr int PW = KC + 16, PX = KC + 16;  // padded smem pitches (halfs)
  constexpr int NTHREADS = WN * WM * 32;
  constexpr int CSEG = NT * (KC / 16) / NTHREADS;  // code segs per thread
  constexpr int XSEG = MT * (KC / 8) / NTHREADS;   // x uint4s per thread
  static_assert(CSEG * NTHREADS == NT * (KC / 16), "code seg split");
  static_assert(XSEG * NTHREADS == MT * (KC / 8), "x seg split");

  extern __shared__ char smem_raw[];
  half *ws = reinterpret_cast<half *>(smem_raw);  // [NT][PW]
  half *xs = ws + NT * PW;                        // [MT][PX]

  const int tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31;
  const int wn = warp % WN, wm = warp / WN;
  const int nb = blockIdx.x * NT;

  uint2 st_c[CSEG];
  unsigned char st_s[CSEG];
  uint4 st_x[XSEG];

  auto load_stage = [&](int k0) {
#pragma unroll
    for (int i = 0; i < CSEG; i++) {
      const int idx = tid + i * NTHREADS;
      const int n = idx / (KC / 16), s = idx % (KC / 16);
      st_c[i] = __ldcs(reinterpret_cast<const uint2 *>(
          codes + (size_t)(nb + n) * (K >> 1) + (k0 >> 1) + s * 8));
      st_s[i] = __ldcs(scales + (size_t)(nb + n) * (K >> 4) + (k0 >> 4) + s);
    }
#pragma unroll
    for (int i = 0; i < XSEG; i++) {
      const int idx = tid + i * NTHREADS;
      const int m = idx / (KC / 8), j4 = idx % (KC / 8);
      st_x[i] = (m < m_real)
                    ? *reinterpret_cast<const uint4 *>(x + (size_t)m * K + k0 +
                                                       j4 * 8)
                    : make_uint4(0, 0, 0, 0);
    }
  };

  auto store_stage = [&]() {
#pragma unroll
    for (int i = 0; i < CSEG; i++) {
      const int idx = tid + i * NTHREADS;
      const int n = idx / (KC / 16), s = idx % (KC / 16);
      const half2 sc2 = fp8e4m3_to_half2(st_s[i]);
      half2 *wrow = reinterpret_cast<half2 *>(ws + n * PW + s * 16);
      const unsigned qs[2] = {st_c[i].x, st_c[i].y};
#pragma unroll
      for (int w = 0; w < 2; w++) {
#ifndef SKINNY_LUT_CVT
        half2 t[4];
        dequant8_tm(qs[w], sc2, t);  // values carry a 2^-14 factor here
        const unsigned *tr = reinterpret_cast<const unsigned *>(t);
        unsigned lin[4] = {__byte_perm(tr[0], tr[1], 0x5410),
                           __byte_perm(tr[2], tr[3], 0x5410),
                           __byte_perm(tr[0], tr[1], 0x7632),
                           __byte_perm(tr[2], tr[3], 0x7632)};
#pragma unroll
        for (int pi = 0; pi < 4; pi++)
          wrow[w * 4 + pi] = *reinterpret_cast<half2 *>(&lin[pi]);
#else
#pragma unroll
        for (int pi = 0; pi < 4; pi++)
          wrow[w * 4 + pi] = dequant_pair(qs[w], pi, sc2);
#endif
      }
    }
#pragma unroll
    for (int i = 0; i < XSEG; i++) {
      const int idx = tid + i * NTHREADS;
      const int m = idx / (KC / 8), j4 = idx % (KC / 8);
      *reinterpret_cast<uint4 *>(xs + m * PX + j4 * 8) = st_x[i];
    }
  };

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> cfrag;
  wmma::fill_fragment(cfrag, 0.f);

  load_stage(0);
  for (int k0 = 0; k0 < K; k0 += KC) {
    __syncthreads();  // previous chunk's mma done; smem free to overwrite
    store_stage();
    __syncthreads();
    if (k0 + KC < K) load_stage(k0 + KC);  // in flight during the mma loop

    // Double-buffered fragments: kk+1's shared loads overlap kk's mma.
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a[2];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b[2];
    wmma::load_matrix_sync(a[0], ws + wn * 16 * PW, PW);
    wmma::load_matrix_sync(b[0], xs + wm * 16 * PX, PX);
#pragma unroll
    for (int kk = 0; kk < KC / 16; kk++) {
      const int cur = kk & 1, nxt = cur ^ 1;
      if (kk + 1 < KC / 16) {
        wmma::load_matrix_sync(a[nxt], ws + wn * 16 * PW + (kk + 1) * 16, PW);
        wmma::load_matrix_sync(b[nxt], xs + wm * 16 * PX + (kk + 1) * 16, PX);
      }
      wmma::mma_sync(cfrag, a[cur], b[cur], cfrag);
    }
  }

  __syncthreads();  // done with ws/xs; reuse for the fp32 epilogue stage
  float *cs = reinterpret_cast<float *>(smem_raw) + warp * 256;
  wmma::store_matrix_sync(cs, cfrag, 16, wmma::mem_row_major);
  __syncwarp();
  for (int e = lane; e < 256; e += 32) {
    const int i = e >> 4, j = e & 15;  // i: n within tile, j: m within tile
    const int gm = wm * 16 + j, gn = nb + wn * 16 + i;
    #ifndef SKINNY_LUT_CVT
    const float gs_eff = gscale * 16384.f;  // undo dequant8_tm's 2^-14
#else
    const float gs_eff = gscale;
#endif
    if (gm < m_real) y[(size_t)gm * N + gn] = __float2half(cs[e] * gs_eff);
  }
}

// ---------------------------------------------------------------------------
// DP4A kernel: int8 SIMT path for the compute-bound M band (4..16).
//
// The e2m1 lattice x2 is integer-exact ({0,1,2,3,4,6,8,12}), so the
// weight side is lossless int8; only activations are quantized
// (symmetric per-16-group int8, done host-side per GEMM call and
// amortized over all N rows). dp4a delivers 4 MACs per issue slot vs
// HFMA2's 2, which is the whole bet: the M-scaling penalty of the
// fp16 SIMT path is issue-bound MAC work.
// ---------------------------------------------------------------------------
DEV_INLINE unsigned dp4a_unpack_mag(unsigned nib4) {
  // nib4 holds 4 e2m1 codes in the low nibbles of each byte
  // (0x0c0c0c0c layout). Returns 4 packed uint8 magnitudes x2 via the
  // shift identity: mag2(c) = c < 2 ? c : (2 + (c&1)) << ((c>>1) - 1).
  unsigned out = 0;
#pragma unroll
  for (int b = 0; b < 4; b++) {
    const unsigned c = (nib4 >> (8 * b)) & 0x7u;
    const unsigned m = c < 2 ? c : (2u + (c & 1u)) << ((c >> 1) - 1u);
    out |= m << (8 * b);
  }
  return out;
}

// One thread per 16-element group: absmax -> scale -> int8 quantize.
// Single launch replaces an 8-op torch chain (which cost 168us of
// launch overhead at these sizes).
__global__ void skinny_quant_a8(const half *__restrict__ x,
                                int8_t *__restrict__ xq,
                                float *__restrict__ xs, int total_groups,
                                int k) {
  const int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= total_groups) return;
  const half2 *xp = reinterpret_cast<const half2 *>(x) + g * 8;
  float amax = 1e-8f;
#pragma unroll
  for (int i = 0; i < 8; i++) {
    const half2 v = xp[i];
    amax = fmaxf(amax, fmaxf(fabsf(__half2float(__low2half(v))),
                             fabsf(__half2float(__high2half(v)))));
  }
  const float s = amax / 127.f, inv = 127.f / amax;
  char4 *out = reinterpret_cast<char4 *>(xq) + g * 4;
#pragma unroll
  for (int i = 0; i < 4; i++) {
    const half2 a = xp[i * 2], b = xp[i * 2 + 1];
    char4 c;
    c.x = (signed char)__float2int_rn(__half2float(__low2half(a)) * inv);
    c.y = (signed char)__float2int_rn(__half2float(__high2half(a)) * inv);
    c.z = (signed char)__float2int_rn(__half2float(__low2half(b)) * inv);
    c.w = (signed char)__float2int_rn(__half2float(__high2half(b)) * inv);
    out[i] = c;
  }
  xs[g] = s;
  (void)k;
}

template <int M, int KC>
__global__ void skinny_nvfp4_dp4a(const uint8_t *__restrict__ codes,
                                  const uint8_t *__restrict__ scales,
                                  const int8_t *__restrict__ xq,
                                  const float *__restrict__ xs_scale,
                                  half *__restrict__ y, int N, int K,
                                  float gscale) {
  extern __shared__ char smem_raw[];
  // staged activations: int32 words [M][KC/4] then scales [M][KC/16]
  int *xw = reinterpret_cast<int *>(smem_raw);
  float *xsc = reinterpret_cast<float *>(xw + M * (KC / 4));
  constexpr int PW4 = KC / 4, PS = KC / 16;

  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int n = blockIdx.x * 8 + warp;
  const uint8_t *crow = codes + (size_t)n * (K >> 1);
  const uint8_t *srow = scales + (size_t)n * (K >> 4);

  float accf[M];
#pragma unroll
  for (int m = 0; m < M; m++) accf[m] = 0.f;

  for (int k0 = 0; k0 < K; k0 += KC) {
    __syncthreads();
    for (int idx = threadIdx.x; idx < M * (KC / 4); idx += blockDim.x) {
      const int m = idx / (KC / 4), j = idx % (KC / 4);
      const int js = (j & ~3) | ((j ^ (j >> 5)) & 3);  // bank swizzle
      xw[m * PW4 + js] = reinterpret_cast<const int *>(
          xq + (size_t)m * K + k0)[j];
    }
    for (int idx = threadIdx.x; idx < M * (KC / 16); idx += blockDim.x) {
      const int m = idx / (KC / 16), g = idx % (KC / 16);
      xsc[m * PS + g] = xs_scale[(size_t)m * (K / 16) + (k0 / 16) + g];
    }
    __syncthreads();

#pragma unroll
    for (int i = 0; i < (KC / 16 + 31) / 32; i++) {
      const int s = lane + 32 * i;  // 16-code segment == one scale group
      if (KC / 16 < 32 && s >= KC / 16) break;
      const uint2 q2 = *reinterpret_cast<const uint2 *>(
          crow + (k0 >> 1) + s * 8);
      // fp8 scale -> float, x0.5 compensates the x2 integer lattice
      const half2 sch = fp8e4m3_to_half2(srow[(k0 >> 4) + s]);
      const float wsc = __half2float(__low2half(sch)) * 0.5f;

      // unpack 16 codes -> 4 dp4a words (int8, sign applied)
      unsigned w4[4];
#pragma unroll
      for (int hw = 0; hw < 2; hw++) {
        const unsigned q = hw == 0 ? q2.x : q2.y;
        const unsigned lo = q & 0x0F0F0F0Fu;         // codes 0,2,4,6
        const unsigned hi = (q >> 4) & 0x0F0F0F0Fu;  // codes 1,3,5,7
        // interleave back to k-order nibble words: bytes of `lo` are
        // even k, bytes of `hi` odd k -> two words of 4 consecutive k
        const unsigned w01 = __byte_perm(lo, hi, 0x5140);  // k0..k3
        const unsigned w23 = __byte_perm(lo, hi, 0x7362);  // k4..k7
        const unsigned wds[2] = {w01, w23};
#pragma unroll
        for (int t = 0; t < 2; t++) {
          const unsigned mag = dp4a_unpack_mag(wds[t]);
          const unsigned sgn = (wds[t] >> 3) & 0x01010101u;  // sign bits
          const unsigned msk = sgn * 0xFFu;  // 0x00 or 0xFF per byte
          // two's complement negate where sign: (mag ^ msk) + (msk & 1)
          w4[hw * 2 + t] = __vadd4(mag ^ msk, sgn);
        }
      }

      int acc32[M];
#pragma unroll
      for (int m = 0; m < M; m++) acc32[m] = 0;
#pragma unroll
      for (int m = 0; m < M; m++) {
        const int *xrow = xw + m * PW4;
#pragma unroll
        for (int t = 0; t < 4; t++) {
          const int j = s * 4 + t;
          const int js = (j & ~3) | ((j ^ (j >> 5)) & 3);
          acc32[m] = __dp4a((int)w4[t], xrow[js], acc32[m]);
        }
      }
#pragma unroll
      for (int m = 0; m < M; m++)
        accf[m] += (float)acc32[m] * (wsc * xsc[m * PS + s]);
    }
  }

  // warp-reduce the per-lane segment partials before the single write
  const float gs = gscale;
#pragma unroll
  for (int m = 0; m < M; m++) {
    float v = accf[m];
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      v += __shfl_down_sync(0xffffffffu, v, off);
    if (lane == 0) y[(size_t)m * N + n] = __float2half(v * gs);
  }
}

// ---------------------------------------------------------------------------
// Host dispatch
// ---------------------------------------------------------------------------

} // namespace ninfer::ops::qpn
