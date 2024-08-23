#ifndef __RT_HOST_H__
#define __RT_HOST_H__

#include "rt_type.h"
#include "rt_handle.h"
namespace snuqs {
namespace rt {

struct host_handle_t {
  static constexpr uint64_t MEM_ALIGN = (1ul << 12);

  host_handle_t(handle_t *_handle);
  ~host_handle_t();

  RuntimeError alloc(addr_t *addr_p, uint64_t size);
  RuntimeError free(addr_t addr);
  handle_t *handle;
};

} // namespace rt
} // namespace snuqs
  
#endif // __RT_HOST_H__
