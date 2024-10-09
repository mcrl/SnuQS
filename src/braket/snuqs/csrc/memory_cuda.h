#ifndef _MEMORY_CUDA_H_
#define _MEMORY_CUDA_H_

#include <cuda_runtime.h>
#include <spdlog/spdlog.h>

#include <complex>

#include "utils_cuda.h"

class MemoryCUDA {
 public:
  MemoryCUDA(size_t count) {
    if (count > 0) {
      CUDA_CHECK(cudaMalloc(&buffer_, sizeof(std::complex<double>) * count));
    }
  }
  ~MemoryCUDA() {
    if (buffer_ != nullptr) CUDA_CHECK(cudaFree(buffer_));
  }
  std::complex<double>* buffer() { return buffer_; }
  std::complex<double>* buffer_ = nullptr;
};
#endif  //_MEMORY_CUDA_H_
