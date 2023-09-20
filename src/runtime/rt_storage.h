#ifndef __RT_STORAGE_H__
#define __RT_STORAGE_H__

#include "rt_handle.h"
#include "rt_type.h"
#include "rt_soft_raid0.h"
#include <vector>
#include <string>
#include <mutex>

namespace snuqs {
namespace rt {

struct soft_raid0_t;

struct storage_handle_t {
  storage_handle_t(handle_t *_handle, std::vector<std::string> paths);
  ~storage_handle_t();

  RuntimeError alloc(addr_t *addr_p, uint64_t size);
  RuntimeError free(addr_t addr);
  RuntimeError memcpy_h2s(addr_t dst, addr_t src, uint64_t size);
  RuntimeError memcpy_s2h(addr_t dst, addr_t src, uint64_t size);

  soft_raid0_t raid0;
  handle_t *handle;
};

} // namespace rt
} // namespace snuqs


#endif // __RT_STORAGE_H__
