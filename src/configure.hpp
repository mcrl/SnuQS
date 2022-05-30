#pragma once

#include <thrust/complex.h>
namespace snuqs {

constexpr size_t MAX_INMEM = 30;
constexpr size_t MAX_IOBUF_SHIFT = 33;
constexpr size_t IOBUF_SHIFT = MAX_IOBUF_SHIFT-MAX_INMEM;
constexpr size_t FUSION_LIMIT = 4;
constexpr size_t MAX_DIAGONAL_FUSION_SIZE = 5;
constexpr bool ALLOW_DIAGONAL_FUSION = true;

using real_t = double;
using amp_t = thrust::complex<real_t>;
} // namespace snuqs


#include "configure_cpu.hpp"
#include "configure_gpu.hpp"
