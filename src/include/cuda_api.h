#ifndef __CUDA_API_H__
#define __CUDA_API_H__

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

namespace snuqs {
namespace cuda {
namespace api {

static inline void assertCudaSuccess(cudaError_t err) {
  if (err != cudaSuccess) {
    printf("[%s:%d] CUDA ERROR: %s\n", __FILE__, __LINE__,
           cudaGetErrorString(err));
    std::exit(EXIT_FAILURE);
  }
}

static void assertKernelLaunch() { assertCudaSuccess(cudaGetLastError()); }

using Stream = cudaStream_t;
using MemcpyKind = cudaMemcpyKind;

static void malloc(void **pptr, size_t count) {
  assertCudaSuccess(cudaMalloc(pptr, count));
}

static void free(void *ptr) { assertCudaSuccess(cudaFree(ptr)); }

static void memcpy(void *dst, const void *src, size_t count, MemcpyKind kind) {
  assertCudaSuccess(cudaMemcpy(dst, src, count, kind));
}

static void memcpyAsync(void *dst, const void *src, size_t count,
                        MemcpyKind kind, Stream stream) {
  assertCudaSuccess(cudaMemcpyAsync(dst, src, count, kind, stream));
}

static void streamCreate(Stream *stream) {
  assertCudaSuccess(cudaStreamCreate(stream));
}

static void streamDestroy(Stream stream) {
  assertCudaSuccess(cudaStreamDestroy(stream));
}

static void streamSynchronize(Stream stream) {
  assertCudaSuccess(cudaStreamSynchronize(stream));
}

static void deviceSynchronize() { assertCudaSuccess(cudaDeviceSynchronize()); }

} // namespace api
} // namespace cuda
} // namespace snuqs
#endif //__CUDA_API_H__
