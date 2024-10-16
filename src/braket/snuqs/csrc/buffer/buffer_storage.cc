#include "buffer/buffer_storage.h"

#include <fcntl.h>
#include <spdlog/spdlog.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <cassert>
#include <cstdlib>

BufferStorage::BufferStorage(size_t count, size_t blk_count,
                             const std::vector<std::string>& path)
    : count_(count), blk_count_(blk_count), path_(path) {
  assert(count >= blk_count);
  assert(count % blk_count == 0);

  for (auto& p : path_) {
    int fd = open(p.c_str(), O_RDWR | O_DIRECT);
    spdlog::info("fd: {}", fd);
    assert(fd != -1);
    fds_.push_back(fd);
  }

  void* addr = nullptr;
  size_t offset = 0;
  size_t num_blks = count / blk_count;
  size_t mapped_blks = 0;
  while (mapped_blks < num_blks) {
    for (auto fd : fds_) {
      addr =
          mmap(addr, blk_count, PROT_READ | PROT_WRITE, MAP_SHARED, fd, offset);
      assert(addr != MAP_FAILED);
      spdlog::info("addr: {}, fd: {}, offset: {}", addr, fd, offset);
    }
    offset += blk_count;
    mapped_blks++;
  }
  buffer_ = reinterpret_cast<std::complex<double>*>(addr);
}

BufferStorage::~BufferStorage() {
  int ret = munmap(buffer_, count_);
  assert(ret == 0);
  for (auto fd : fds_) {
    close(fd);
  }
}
std::complex<double>* BufferStorage::buffer() { return buffer_; }
size_t BufferStorage::count() const { return count_; }
size_t BufferStorage::itemsize() const { return sizeof(std::complex<double>); }

std::string BufferStorage::formatted_string() const {
  return "BufferStorage<" + std::to_string(count_) + ">";
}
