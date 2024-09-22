#include "rt_storage.h"
#include "rt_soft_raid0.h"
#include <cstdlib>
#include <mutex>

namespace snuqs {
namespace rt {


storage_handle_t::storage_handle_t(handle_t *_handle, std::vector<std::string> paths)
  : handle(_handle), raid0(paths) {
}

storage_handle_t::~storage_handle_t() {
}

RuntimeError storage_handle_t::alloc(addr_t *addr_p, uint64_t size) {
  return this->raid0.alloc(addr_p, size);
}

RuntimeError storage_handle_t::free(addr_t addr) {
  return this->raid0.free(addr);
}

RuntimeError storage_handle_t::memcpy_h2s(addr_t dst, addr_t src, uint64_t size) {
  return this->raid0.write(dst, src, size);
}

RuntimeError storage_handle_t::memcpy_s2h(addr_t dst, addr_t src, uint64_t size) {
  return this->raid0.read(dst, src, size);
}

} // namespace rt
} // namespace snuqs
