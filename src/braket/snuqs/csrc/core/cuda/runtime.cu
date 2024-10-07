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

void set_device(int device) { CUDA_CHECK(cudaSetDevice(device)); }
int get_device() {
  int device;
  CUDA_CHECK(cudaGetDevice(&device));
  return device;
}
};  // namespace cu
