#include "buffer/buffer_storage.h"

#include <fcntl.h>
#include <spdlog/spdlog.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <cassert>
#include <cstdlib>

#include "buffer/buffer_cpu.h"
#include "core/cuda/runtime.h"
#include "core/runtime.h"
#define SECTOR_SIZE (512)

BufferStorage::BufferStorage(size_t count) : count_(count) {
  assert(count >= blk_count);
  assert(count % blk_count == 0);
  assert(is_attached_fs());
  fs_ = get_fs();
  fs_addr_ = fs_->alloc(count);
}
BufferStorage::~BufferStorage() { fs_->free(fs_addr_); }

void* BufferStorage::ptr() { return fs_addr_.ptr; }
size_t BufferStorage::count() const { return count_; }

std::string BufferStorage::formatted_string() const {
  return "BufferStorage<" + std::to_string(count_) + ">";
}
fs_addr_t BufferStorage::addr() { return fs_addr_; }

std::shared_ptr<Buffer> BufferStorage::cpu() { return nullptr; }

std::shared_ptr<Buffer> BufferStorage::cuda() { return nullptr; }

std::shared_ptr<Buffer> BufferStorage::storage() { return shared_from_this(); }
