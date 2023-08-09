#include "rt.h"
#include "rt_device.h"

#include <cstdlib>
#include <set>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <cassert>

#include <cuda_runtime.h>
#include <iostream>

#define CHECK_CUDA(e)                  \
  do {                                 \
    cudaError_t error = (e);           \
    if (error != cudaSuccess) {        \
      std::cout                        \
        << "CUDA ERROR!["              \
        << __FILE__                    \
        << ":"                         \
        << __LINE__                    \
        << "]  \n"                     \
        << cudaGetErrorString(error);  \
      std::exit(EXIT_FAILURE);         \
    }                                  \
  } while (0)

namespace snuqs {
namespace rt {

device_handle_t::device_handle_t(handle_t *_handle)
: handle(_handle) {

  CHECK_CUDA(cudaGetDeviceCount(&num_devices));
  CHECK_CUDA(cudaSetDevice(0));

  for (int i = 0; i < num_devices; ++i) {
    cudaStream_t h2d, kern, d2h;
    CHECK_CUDA(cudaStreamCreate(&h2d));
    CHECK_CUDA(cudaStreamCreate(&kern));
    CHECK_CUDA(cudaStreamCreate(&d2h));
    h2d_streams.push_back(h2d);
    kern_streams.push_back(kern);
    d2h_streams.push_back(d2h);
  }
}

device_handle_t::~device_handle_t() {
  for (int i = 0; i < h2d_streams.size(); ++i) {
    CHECK_CUDA(cudaStreamDestroy(h2d_streams[i]));
  }
  for (int i = 0; i < kern_streams.size(); ++i) {
    CHECK_CUDA(cudaStreamDestroy(kern_streams[i]));
  }
  for (int i = 0; i < d2h_streams.size(); ++i) {
    CHECK_CUDA(cudaStreamDestroy(d2h_streams[i]));
  }
}

RuntimeError device_handle_t::get_num_devices(int *num_devices_p) {
  if (num_devices_p == nullptr) {
    return RT_NULL_POINTER;
  }
  CHECK_CUDA(cudaGetDeviceCount(num_devices_p));
  return RT_SUCCESS;
}

RuntimeError device_handle_t::get_device(int *device_p) {
  if (device_p == nullptr) {
    return RT_NULL_POINTER;
  }
  CHECK_CUDA(cudaGetDevice(device_p));
  return RT_SUCCESS;
}

RuntimeError device_handle_t::set_device(int device) {
  CHECK_CUDA(cudaSetDevice(device));
  return RT_SUCCESS;
}

RuntimeError device_handle_t::alloc(addr_t *addr_p, uint64_t size) { 
  void *ptr;
  CHECK_CUDA(cudaMalloc(&ptr, size));
  *addr_p = ptr;
  return RT_SUCCESS;
}

RuntimeError device_handle_t::alloc_pinned(addr_t *addr_p, uint64_t size) {
  void *ptr;
  CHECK_CUDA(cudaMallocHost(&ptr, size));
  *addr_p = ptr;
  return RT_SUCCESS;
}

RuntimeError device_handle_t::free(addr_t addr) {
  void *ptr = reinterpret_cast<void*>(addr);
  CHECK_CUDA(cudaFree(ptr));
  return RT_SUCCESS;
}

RuntimeError device_handle_t::free_pinned(addr_t addr) {
  void *ptr = reinterpret_cast<void*>(addr);
  CHECK_CUDA(cudaFreeHost(ptr));
  return RT_SUCCESS;
}

device_handle_t::device_stream_t device_handle_t::get_current_h2d_stream() {
  RuntimeError error;
  int device;
  error = get_device(&device);
  if (error != RT_SUCCESS)
    return nullptr;

  return h2d_streams[device];
}

device_handle_t::device_stream_t device_handle_t::get_current_kern_stream() {
  RuntimeError error;
  int device;
  error = get_device(&device);
  if (error != RT_SUCCESS)
    return nullptr;

  return kern_streams[device];
}

device_handle_t::device_stream_t device_handle_t::get_current_d2h_stream() {
  RuntimeError error;
  int device;
  error = get_device(&device);
  if (error != RT_SUCCESS)
    return nullptr;

  return d2h_streams[device];
}

RuntimeError device_handle_t::memcpy_h2d(addr_t dst, addr_t src, uint64_t size) {
  cudaStream_t stream = get_current_h2d_stream();
  if (!stream) return RT_INVALID_DEVICE;

  CHECK_CUDA(cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));
  return RT_SUCCESS;
}

RuntimeError device_handle_t::memcpy_d2h(addr_t dst, addr_t src, uint64_t size) {
  cudaStream_t stream = get_current_d2h_stream();
  if (!stream) return RT_INVALID_DEVICE;

  CHECK_CUDA(cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));
  return RT_SUCCESS;
}

RuntimeError device_handle_t::launch_kernel(const void *func, dim3 grid_dim, dim3 block_dim,
    void **args, uint64_t shared_mem) {
  cudaStream_t stream = get_current_kern_stream();
  if (!stream) return RT_INVALID_DEVICE;

  CHECK_CUDA(cudaLaunchKernel(func, grid_dim, block_dim, args, shared_mem, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));
  return RT_SUCCESS;
}

} // namespace rt
} // namespace snuqs
