#include "utils_cuda.h"

namespace cu {
std::pair<size_t, size_t> mem_info() {
  size_t free, total;
  CUDA_CHECK(cudaMemGetInfo(&free, &total));
  return {free, total};
}

int device_count() {
  int count;
  CUDA_CHECK(cudaGetDeviceCount(&count));
  return count;
}
};  // namespace cu
