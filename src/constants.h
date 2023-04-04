#pragma once

namespace snuqs {

  constexpr size_t MAX_GPUS = 4;
  constexpr size_t NTHREADS = 256;

  constexpr size_t SMEM_SHIFT = 11;
  constexpr size_t SMEM_NELEM = (1ul << SMEM_SHIFT);

  constexpr size_t WARP_SHIFT = 5;
  constexpr size_t WARP_NELEM = (1ul << WARP_SHIFT);


  //FIXME
  constexpr size_t MAX_INMEM = 30;
};
