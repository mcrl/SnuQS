#include <thrust/complex.h>

#include "initialization/initialization_impl_cuda.h"
#include "utils_cuda.h"
namespace cu {

static constexpr size_t BLOCKDIM = 256;

static __global__ void initializeZero_kernel(thrust::complex<double>* buf,
                                             size_t nelems) {
  size_t i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i >= nelems) return;

  buf[i] = 0;
}

static __global__ void initializeBasis_Z_kernel(thrust::complex<double>* buf,
                                                size_t nelems) {
  size_t i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i >= nelems) return;

  buf[i] = (i == 0) ? 1 : 0;
}

void initializeZero(void* _buf, size_t nelems) {
  initializeZero_kernel<<<(nelems + BLOCKDIM - 1) / BLOCKDIM, BLOCKDIM>>>(
      reinterpret_cast<thrust::complex<double>*>(_buf), nelems);
  CUDA_CHECK(cudaGetLastError());
}

void initializeBasis_Z(void* _buf, size_t nelems) {
  initializeBasis_Z_kernel<<<(nelems + BLOCKDIM - 1) / BLOCKDIM, BLOCKDIM>>>(
      reinterpret_cast<thrust::complex<double>*>(_buf), nelems);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace cu
