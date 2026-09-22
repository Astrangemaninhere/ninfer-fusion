// probe TU: does the W3/W2 device K-plane arm actually compile as a real TU under nvcc?
// Lives in dl/e8k3k2 (this line's own directory) -- nothing in the product tree is touched.
#include "ops/kernel/e8_lattice_kv_plane.cuh"

int main() { return 0; }
