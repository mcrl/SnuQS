#include "api.h"

#include "memory_allocator.h"

namespace snurt {

static MemoryAllocator io_allocator;

addr_t MallocIO(size_t count) {
  addr_t addr;
  uint64_t off = io_allocator.Allocate(count);
  addr.off = off;
  return addr;
}

} // namespace snurt
