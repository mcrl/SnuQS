#ifndef _RUNTIME_CONTEXT_H_
#define _RUNTIME_CONTEXT_H_

#include "runtime_error.h"
#include "runtime_types.h"

#include <string>
#include <vector>

namespace runtime {

struct stream_t;
struct kernel_t;

class RuntimeContext {
public:
  RuntimeContext();
  RuntimeContext(std::vector<std::string> paths);
  ~RuntimeContext();

  void *host_handle;
  void *device_handle;
  void *storage_handle;
  void *stream_handle;

  RuntimeError mallocHost(host_addr_t *addr_p, uint64_t size);
  RuntimeError mallocStorage(storage_addr_t *addr_p, uint64_t size);
  RuntimeError mallocDevice(device_addr_t *addr_p, uint64_t size);
  RuntimeError mallocPinned(host_addr_t *addr_p, uint64_t size);

  RuntimeError freeHost(host_addr_t addr);
  RuntimeError freeStorage(storage_addr_t addr);
  RuntimeError freeDevice(device_addr_t addr);
  RuntimeError freePinned(host_addr_t addr);

  RuntimeError createStream(stream_t **stream_p);
  RuntimeError destroyStream(stream_t *stream);
  RuntimeError streamSynchronize(stream_t *stream);

  RuntimeError getNumDevices(int *num_devices_p);
  RuntimeError getDevice(int *device_p);
  RuntimeError setDevice(int device);

  RuntimeError memcpyH2D(device_addr_t dst, host_addr_t src, uint64_t size,
                         stream_t *stream);
  RuntimeError memcpyD2H(host_addr_t dst, device_addr_t src, uint64_t size,
                         stream_t *stream);

  RuntimeError createKernel(kernel_t **kernelp, const void *func);
  RuntimeError setKernelArg(kernel_t *kernel, int idx, uint64_t size,
                            void *ptr);
  RuntimeError launchKernel(kernel_t *kernel, grid grid_dim, grid block_dim,
                            uint64_t shared_mem, stream_t *stream);
  RuntimeError memcpyH2S(storage_addr_t dst, host_addr_t src, uint64_t size,
                         stream_t *stream);
  RuntimeError memcpyS2H(host_addr_t dst, storage_addr_t src, uint64_t size,
                         stream_t *stream);
};

} // namespace runtime

#endif //_RUNTIME_CONTEXT_H_
