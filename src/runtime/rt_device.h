#ifndef __RT_DEVICE_H__
#define __RT_DEVICE_H__

#include "rt_handle.h"
#include "rt_error.h"
#include "rt_type.h"

#include <vector>
#include <mutex>
#include <thread>

#include <cuda_runtime.h>

namespace snuqs {
namespace rt {

struct queue;
struct task_t;
struct device_handle_t {
  using device_stream_t = cudaStream_t;

  device_handle_t(handle_t *_handle);
  ~device_handle_t();

  handle_t *handle;
  std::mutex mutex;
  int num_devices;
  std::vector<device_stream_t> h2d_streams;
  std::vector<device_stream_t> kern_streams;
  std::vector<device_stream_t> d2h_streams;

  RuntimeError get_num_devices(int *num_devices_p);
  RuntimeError get_device(int *device_p);
  RuntimeError set_device(int device);

  RuntimeError alloc(addr_t *addr_p, uint64_t size);
  RuntimeError alloc_pinned(addr_t *addr_p, uint64_t size);
  RuntimeError free(addr_t addr);
  RuntimeError free_pinned(addr_t addr);

  device_stream_t get_current_h2d_stream();
  device_stream_t get_current_kern_stream();
  device_stream_t get_current_d2h_stream();
  RuntimeError memcpy_h2d(addr_t dst, addr_t src, uint64_t size);
  RuntimeError memcpy_d2h(addr_t dst, addr_t src, uint64_t size);
  RuntimeError launch_kernel(const void *func, dim3 grid_dim, dim3 block_dim,
      void **args, uint64_t shared_mem);
};

} // namespace rt
} // namespace snuqs

#endif //__RT_DEVICE_H__
