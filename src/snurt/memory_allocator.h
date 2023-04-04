#pragma once

#include <vector>
#include <mutex>

namespace snurt {

class MemoryAllocator {
  public:
  uint64_t Allocate(size_t count);
  private:
  std::mutex mutex;
  std::vector<std::pair<uint64_t, size_t>> free_list;
};

} // namespace snurt
