#pragma once

namespace snuqs {
namespace cpu {

constexpr size_t NQUBIT = 30; // >= 25
constexpr size_t MEMORY_LIMIT= (1ULL << NQUBIT);
constexpr size_t MAX_QUBIT = 34;
constexpr size_t NBUF = 20;
constexpr size_t CACHE_SHIFT = 21;
constexpr size_t CACHE_LIMIT = (1ULL << CACHE_SHIFT);

} // namespace cpu
} // namespace snuqs
