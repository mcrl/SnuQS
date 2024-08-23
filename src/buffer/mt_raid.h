#ifndef __MT_RAID_H__
#define __MT_RAID_H__

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace snuqs {

class MTRaid {
public:
  static constexpr uint64_t MAX_IO = 0x7ffff000;
  static constexpr uint64_t BLOCK_SIZE = (1ul << 21);
  void *base = reinterpret_cast<void *>(0xdead000000000000ul);

  MTRaid(std::vector<std::string> paths);
  ~MTRaid();

  std::mutex mutex_;
  std::vector<std::string> paths_;
  std::vector<int> fds_;
  int alloc(void **addr_p, uint64_t size);
  int free(void *addr);
  int write(void *dst, void *src, uint64_t size);
  int read(void *dst, void *src, uint64_t size);
};

} // namespace snuqs
#endif //__MT_RAID0_H__
