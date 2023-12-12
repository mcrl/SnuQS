#ifndef __CUDA_API_H__
#define __CUDA_API_H__

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

namespace snuqs {
namespace cuda {
namespace api {

// static inline void assertCudaSuccess(cudaError_t err) {
//}
#define assertCudaSuccess(err)                                                 \
  if ((err) != cudaSuccess) {                                                  \
    printf("[%s:%d] CUDA ERROR: %s\n", __FILE__, __LINE__,                     \
           cudaGetErrorString(err));                                           \
    std::exit(EXIT_FAILURE);                                                   \
  }

using ret_t = void;

static ret_t assertKernelLaunch() { assertCudaSuccess(cudaGetLastError()); }

using Stream = cudaStream_t;
using MemcpyKind = cudaMemcpyKind;

static ret_t malloc(void **pptr, size_t count) {
  assertCudaSuccess(cudaMalloc(pptr, count));
}

static ret_t mallocHost(void **pptr, size_t count) {
  assertCudaSuccess(cudaMallocHost(pptr, count));
}

static ret_t free(void *ptr) { assertCudaSuccess(cudaFree(ptr)); }
static ret_t freeHost(void *ptr) { assertCudaSuccess(cudaFreeHost(ptr)); }

static ret_t memset(void *dst, int value, size_t count) {
  assertCudaSuccess(cudaMemset(dst, value, count));
}

static ret_t memcpy(void *dst, const void *src, size_t count, MemcpyKind kind) {
  assertCudaSuccess(cudaMemcpy(dst, src, count, kind));
}

static ret_t memcpyAsync(void *dst, const void *src, size_t count,
                         MemcpyKind kind, Stream stream) {
  assertCudaSuccess(cudaMemcpyAsync(dst, src, count, kind, stream));
}

static ret_t streamCreate(Stream *stream) {
  assertCudaSuccess(cudaStreamCreate(stream));
}

static ret_t streamDestroy(Stream stream) {
  assertCudaSuccess(cudaStreamDestroy(stream));
}

static ret_t streamSynchronize(Stream stream) {
  assertCudaSuccess(cudaStreamSynchronize(stream));
}

static ret_t getDevice(int *devp) { assertCudaSuccess(cudaGetDevice(devp)); }

static ret_t setDevice(int dev) { assertCudaSuccess(cudaSetDevice(dev)); }

static ret_t getDeviceCount(int *countp) {
  assertCudaSuccess(cudaGetDeviceCount(countp));
}

static ret_t memGetInfo(size_t *freep, size_t *totalp) {
  assertCudaSuccess(cudaMemGetInfo(freep, totalp));
}

static ret_t deviceSynchronize() { assertCudaSuccess(cudaDeviceSynchronize()); }

} // namespace api
} // namespace cuda
} // namespace snuqs
#endif //__CUDA_API_H__
