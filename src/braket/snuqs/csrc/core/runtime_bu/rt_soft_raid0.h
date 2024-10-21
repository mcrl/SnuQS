#ifndef __RT_SOFT_RAID0_H__
#define __RT_SOFT_RAID0_H__

#include "rt_type.h"
#include "rt_error.h"
#include <vector>
#include <string>
#include <mutex>

namespace snuqs {
namespace rt {

struct soft_raid0_t {
  static constexpr uint64_t MAX_IO = 0x7ffff000;
  static constexpr uint64_t BLOCK_SIZE = (1ul << 21);
  addr_t base = reinterpret_cast<addr_t>(0xdead000000000000ul);

  soft_raid0_t(std::vector<std::string> _paths);
  ~soft_raid0_t();

  std::mutex mutex;
  std::vector<std::string> paths;
  std::vector<int> fds;
  RuntimeError alloc(addr_t *addr_p, uint64_t size);
  RuntimeError free(addr_t addr);
  RuntimeError write(addr_t dst, addr_t src, uint64_t size);
  RuntimeError read(addr_t dst, addr_t src, uint64_t size);
};

} // namespace rt
} // namespace snuqs

#endif //__RT_SOFT_RAID0_H__
