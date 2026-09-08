// qpn_map.cuh — qpn_simt fragment layout (machine-derived from exhaustive
// single-code probes over the live kernel; 2048/2048 entries verified).
// slot(g, lane, ci) covers column col_of(lane,ci) at positions k_of(...), +1.
// col = 2*(lane%16) + ci/8
// k   = 32*g + 16*(lane/16) + 8*((ci/4)%2) + [0,4,1,5][ci%4] + nib*2
#pragma once

namespace ninfer::ops::qpn {

// Closed-form helpers (compile-time friendly).
constexpr int qpn_slot_col(int lane, int ci) {
    return 2 * (lane % 16) + (ci / 8);
}
constexpr int qpn_slot_k(int g, int lane, int ci, int nib) {
    constexpr int kBase[4] = {0, 4, 1, 5};
    return 32 * g + 16 * (lane / 16) + 8 * ((ci / 4) % 2) + kBase[ci % 4] + nib * 2;
}

} // namespace ninfer::ops::qpn
