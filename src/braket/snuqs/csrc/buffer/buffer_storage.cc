#include "buffer/buffer_storage.h"

#include <fcntl.h>
#include <spdlog/spdlog.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <cassert>
#include <complex>
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

void* BufferStorage::ptr() { return ptr_; }
size_t BufferStorage::count() const { return count_; }

std::string BufferStorage::formatted_string() const {
  return "BufferStorage<" + std::to_string(count_) + ">";
}

std::shared_ptr<Buffer> BufferStorage::cpu() {
  //  auto info = mem_info();
  //  size_t free = info.first;
  //  assert(free >= count());
  //
  //  auto buf = std::make_shared<BufferCPU>(count_);
  //
  //  size_t nbytes = count();
  //  size_t nbytes_read = 0;
  //
  //  char* buffer = reinterpret_cast<char*>(buf->buffer());
  //  while (nbytes_read < nbytes) {
  //    auto off_info = get_offset(nbytes_read);
  //    int fd = off_info.first;
  //    size_t offset = off_info.second;
  //    size_t bytes_to_read =
  //        std::min((size_t)SSIZE_MAX,
  //                 std::min((size_t)(blk_count_ - (offset % blk_count_)),
  //                          (size_t)(nbytes - nbytes_read)));
  //    ssize_t ret = pread(fd, (void*)buffer, bytes_to_read, offset);
  //    assert(ret != -1);
  //    nbytes_read += ret;
  //    buffer += ret;
  //  }
  //
  //  return buf;
  return nullptr;
}

std::shared_ptr<Buffer> BufferStorage::cuda() {
  auto info = cu::mem_info();
  size_t free = info.first;

  assert(free >= count());

  return cpu()->cuda();
}

std::shared_ptr<Buffer> BufferStorage::storage() { return shared_from_this(); }

std::pair<int, size_t> BufferStorage::get_offset(size_t pos) const {
  //  size_t num_devices = fds_.size();
  //  size_t row_size = blk_count_ * num_devices;
  //  size_t device = (pos / blk_count_) % num_devices;
  //  size_t offset = (pos / row_size) * blk_count_;
  //  return {device, offset};
  return {0, 0};
}
