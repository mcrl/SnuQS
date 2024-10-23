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
#include "buffer/buffer_cuda.h"
#include "core/cuda/runtime.h"
#include "core/runtime.h"
#define SECTOR_SIZE (512)

BufferStorage::BufferStorage(size_t count) : count_(count) {
  assert(count >= blk_count);
  assert(count % blk_count == 0);
  assert(is_attached_fs());
  spdlog::info("BufferStorage({})", count);
  fs_ = get_fs();
  fs_addr_ = fs_->alloc(count);
}
BufferStorage::~BufferStorage() { fs_->free(fs_addr_); }

void* BufferStorage::ptr() {
  int ret = msync(fs_addr_.ptr, fs_addr_.end - fs_addr_.start, MS_SYNC);
  return fs_addr_.ptr;
}

size_t BufferStorage::count() const { return count_; }

std::string BufferStorage::formatted_string() const {
  return "BufferStorage<" + std::to_string(count_) + ">";
}
fs_addr_t BufferStorage::addr() { return fs_addr_; }
void BufferStorage::sync() {
  int ret = msync(fs_addr_.ptr, fs_addr_.end - fs_addr_.start, MS_SYNC);
  assert(ret == 0);
}

std::shared_ptr<Buffer> BufferStorage::cpu(std::shared_ptr<Stream> stream) {
  auto buf = std::make_shared<BufferCPU>(count_);
  memcpyS2H(buf->ptr(), fs_addr_, count_, 0, stream);
  return buf;
}
std::shared_ptr<Buffer> BufferStorage::cuda(std::shared_ptr<Stream> stream) {
  auto buf_cuda = std::make_shared<BufferCUDA>(count_);
  memcpyS2D(buf_cuda->ptr(), fs_addr_, count_, 0, stream);
  return buf_cuda;
}
std::shared_ptr<Buffer> BufferStorage::storage(std::shared_ptr<Stream> stream) {
  return shared_from_this();
}
