// qpn_map.cuh — qpn fragment layouts (machine-derived from exhaustive
// single-code probes over the live kernels; 2048/2048 entries verified each).
//
// qpn_simt slot(g, lane, ci):
//   col = 2*(lane%16) + ci/8
//   k   = 32*g + 16*(lane/16) + 8*((ci/4)%2) + [0,4,1,5][ci%4] + nib*2
// qpn2 slot(g, lane, ci):
//   col = 2*(lane%16) + ci/8
//   k   = 32*g + 16*(lane/16) + 8*((ci/4)%2) + [0,4,1,5][ci%4] + nib*2
// (qpn2 differs from simt only in its M<=8 row mapping and split-K schedule;
//  the nibble slot -> (col,k) map is identical, probe-verified.)
#pragma once

namespace ninfer::ops::qpn {

constexpr int qpn_slot_col(int lane, int ci) {
    return 2 * (lane % 16) + (ci / 8);
}
constexpr int qpn_slot_k(int g, int lane, int ci, int nib) {
    constexpr int kBase[4] = {0, 4, 1, 5};
    return 32 * g + 16 * (lane / 16) + 8 * ((ci / 4) % 2) + kBase[ci % 4] + nib * 2;
}
constexpr int qpn2_slot_col(int lane, int ci) {
    return 2 * (lane % 16) + (ci / 8);
}
constexpr int qpn2_slot_k(int g, int lane, int ci, int nib) {
    constexpr int kBase[4] = {0, 4, 1, 5};
    return 32 * g + 16 * (lane / 16) + 8 * ((ci / 4) % 2) + kBase[ci % 4] + nib * 2;
}

} // namespace ninfer::ops::qpn
