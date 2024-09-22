#include "rt_host.h"

#include <malloc.h>

namespace snuqs {
namespace rt {

host_handle_t::host_handle_t(handle_t *_handle)
  : handle(_handle) {
}

host_handle_t::~host_handle_t() {
}

RuntimeError host_handle_t::alloc(addr_t *addr_p, uint64_t size) {
  *addr_p = reinterpret_cast<addr_t>(memalign(MEM_ALIGN, size));
  return RT_SUCCESS;
}

RuntimeError host_handle_t::free(addr_t addr) {
  void *ptr = reinterpret_cast<void*>(addr);
  std::free(ptr);
  return RT_SUCCESS;
}

} // namespace rt
} // namespace snuqs
