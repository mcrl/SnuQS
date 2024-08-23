#ifndef __RT_HANDLE_H__
#define __RT_HANDLE_H__

#include "rt_error.h"
#include "rt_type.h"

#include <vector>
#include <string>

namespace snuqs {
namespace rt {

struct stream_t;
struct kernel_t;
struct handle_t {
  handle_t();
  handle_t(std::vector<std::string> paths);
  ~handle_t();

  void *host_handle;
  void *device_handle;
  void *storage_handle;
  void *stream_handle;

  RuntimeError create_stream(stream_t **stream_p);
  RuntimeError destroy_stream(stream_t *stream);
  RuntimeError stream_synchronize(stream_t *stream);

  RuntimeError malloc_host(addr_t *addr_p, uint64_t size);
  RuntimeError malloc_storage(addr_t *addr_p, uint64_t size); 
  RuntimeError malloc_device(addr_t *addr_p, uint64_t size);
  RuntimeError malloc_pinned(addr_t *addr_p, uint64_t size);

  RuntimeError free_host(addr_t addr);
  RuntimeError free_storage(addr_t addr);
  RuntimeError free_device(addr_t addr);
  RuntimeError free_pinned(addr_t addr);

  RuntimeError get_num_devices(int *num_devices_p);
  RuntimeError get_device(int *device_p);
  RuntimeError set_device(int device);

  RuntimeError memcpy_h2d(addr_t dst, addr_t src, uint64_t size, stream_t *stream);
  RuntimeError memcpy_d2h(addr_t dst, addr_t src, uint64_t size, stream_t *stream);
  RuntimeError create_kernel(kernel_t **kernelp, const void *func);
  RuntimeError set_kernel_arg(kernel_t *kernel, int idx, uint64_t size, void *ptr);
  RuntimeError launch_kernel(kernel_t *kernel, dim3 grid_dim, dim3 block_dim, uint64_t shared_mem, stream_t *stream);
  RuntimeError memcpy_h2s(addr_t dst, addr_t src, uint64_t size, stream_t *stream);
  RuntimeError memcpy_s2h(addr_t dst, addr_t src, uint64_t size, stream_t *stream);
};

} // namespace rt
} // namespace snuqs

#endif //__RT_HANDLE_H__
