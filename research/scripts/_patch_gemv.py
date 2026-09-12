p = '/home/user/ninfer-fusion/src/ops/linear/nvfp4/nvfp4_gemv.cuh'
s = open(p).read()
old = '''#pragma unroll
        for (int pair = 0; pair < Schedule::kPairsPerLane; ++pair) {
            const int activation_index =
                phase * (kValuesPerPhase / 2) + lane * Schedule::kPairsPerLane + pair;
            const float2 activation = bf16x2_bits_to_float2(activation_pairs[activation_index]);
            const int group         = ((lane * Schedule::kValuesPerLane & 15) + pair * 2) / 16;
#pragma unroll
            for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
                const std::uint32_t word  = row_codes[local_row].words[pair / 4];
                const std::uint8_t packed = static_cast<std::uint8_t>(word >> (8 * (pair & 3)));
                const float2 code         = decode_nvfp4_e2m1x2(packed);
                const float coefficient   = coefficients[local_row][group];
                constexpr int kChainMask  = Schedule::kAccumulatorChains - 1;
                accumulators[local_row][(2 * pair) & kChainMask] =
                    fmaf(code.x * coefficient, activation.x,
                         accumulators[local_row][(2 * pair) & kChainMask]);
                accumulators[local_row][(2 * pair + 1) & kChainMask] =
                    fmaf(code.y * coefficient, activation.y,
                         accumulators[local_row][(2 * pair + 1) & kChainMask]);
            }
        }'''
new = '''        // Each lane's 16 values in a phase share one scale coefficient
        // (kGroupsPerLane == 1).  Accumulate code*activation per chain with
        // plain FMAs, then fold the shared coefficient once per chain at the
        // end of the phase: replaces one FMUL+FFMA per value with a single
        // FFMA (the scale is common to all 16 values in the phase).
        if constexpr (kGroupsPerLane == 1) {
            float phase_accum[Schedule::kRowsPerWarp][Schedule::kAccumulatorChains] = {};
#pragma unroll
            for (int pair = 0; pair < Schedule::kPairsPerLane; ++pair) {
                const int activation_index =
                    phase * (kValuesPerPhase / 2) + lane * Schedule::kPairsPerLane + pair;
                const float2 activation = bf16x2_bits_to_float2(activation_pairs[activation_index]);
                constexpr int kChainMask = Schedule::kAccumulatorChains - 1;
#pragma unroll
                for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
                    const std::uint32_t word = row_codes[local_row].words[pair / 4];
                    const std::uint8_t packed =
                        static_cast<std::uint8_t>(word >> (8 * (pair & 3)));
                    const float2 code = decode_nvfp4_e2m1x2(packed);
                    phase_accum[local_row][(2 * pair) & kChainMask] =
                        fmaf(code.x, activation.x,
                             phase_accum[local_row][(2 * pair) & kChainMask]);
                    phase_accum[local_row][(2 * pair + 1) & kChainMask] =
                        fmaf(code.y, activation.y,
                             phase_accum[local_row][(2 * pair + 1) & kChainMask]);
                }
            }
#pragma unroll
            for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
                const float coefficient = coefficients[local_row][0];
#pragma unroll
                for (int chain = 0; chain < Schedule::kAccumulatorChains; ++chain) {
                    accumulators[local_row][chain] =
                        fmaf(coefficient, phase_accum[local_row][chain],
                             accumulators[local_row][chain]);
                }
            }
        } else {
#pragma unroll
            for (int pair = 0; pair < Schedule::kPairsPerLane; ++pair) {
                const int activation_index =
                    phase * (kValuesPerPhase / 2) + lane * Schedule::kPairsPerLane + pair;
                const float2 activation = bf16x2_bits_to_float2(activation_pairs[activation_index]);
                const int group = ((lane * Schedule::kValuesPerLane & 15) + pair * 2) / 16;
#pragma unroll
                for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
                    const std::uint32_t word  = row_codes[local_row].words[pair / 4];
                    const std::uint8_t packed = static_cast<std::uint8_t>(word >> (8 * (pair & 3)));
                    const float2 code         = decode_nvfp4_e2m1x2(packed);
                    const float coefficient   = coefficients[local_row][group];
                    constexpr int kChainMask  = Schedule::kAccumulatorChains - 1;
                    accumulators[local_row][(2 * pair) & kChainMask] =
                        fmaf(code.x * coefficient, activation.x,
                             accumulators[local_row][(2 * pair) & kChainMask]);
                    accumulators[local_row][(2 * pair + 1) & kChainMask] =
                        fmaf(code.y * coefficient, activation.y,
                             accumulators[local_row][(2 * pair + 1) & kChainMask]);
                }
            }
        }'''
if old not in s:
    print('PATTERN NOT FOUND')
    raise SystemExit(1)
s = s.replace(old, new)
open(p, 'w').write(s)
print('patched ok')
