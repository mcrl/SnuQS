#include "api.h"

#include <cuda_runtime.h>

namespace snurt {

int GetDeviceCount() {
  cudaError_t err;
  int count;
  err = cudaGetDeviceCount(&count);
  if (err != cudaSuccess)
    return -EINVAL;
  return count;
}

addr_t MallocHost(size_t count) {
  addr_t addr;
  cudaError_t err;
  void *ptr;
  err = cudaMallocHost(&ptr, count);
  addr.ptr = (err == cudaSuccess) ? ptr : nullptr;
  return addr;
}

addr_t MallocDevice(size_t count, size_t devno) {
  addr_t addr;
  cudaError_t err;
  err = cudaSetDevice(devno);
  if (err != cudaSuccess) {
    addr.ptr = nullptr;
    return addr;
  }

  void *ptr;
  err = cudaMalloc(&ptr, count);
  addr.ptr = (err == cudaSuccess) ? ptr : nullptr;
  return addr;
}

} // namespace snurt
