// cuda_runtime.h -- host shim so the REAL src/ops/kernel/e8_lattice*.cuh compile on the
// host. e8_lattice.cuh includes <cuda_runtime.h>; this satisfies it without CUDA.
#pragma once
#include <cmath>
#include <cstdint>
