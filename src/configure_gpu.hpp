#pragma once


namespace snuqs {
namespace gpu {

constexpr size_t NQUBIT = 30;
constexpr size_t NBUF = 20;
constexpr size_t MAX_NGPU = 4;


constexpr size_t MAX_QUBIT = 30;
constexpr size_t WARPSHIFT = 5;
constexpr size_t WARPSIZE = 32;
constexpr size_t SMEM_SHIFT = 11;
constexpr size_t SMEM_NELEM = (1ul << SMEM_SHIFT);
constexpr size_t NTHREADS = 256;
constexpr size_t MAX_CACHE = (SMEM_SHIFT-WARPSHIFT);

} // namespace gpu
} // namespace snuqs
