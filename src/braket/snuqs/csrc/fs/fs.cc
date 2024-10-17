#include "fs.h"

#include <fcntl.h>
#include <spdlog/spdlog.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <cassert>

#define SECTOR_SIZE (512)
#define ALIGNMENT (512)
std::string fs_addr_t::formatted_string() const {
  return "fs_addr_t<" + std::to_string(start) + "-" + std::to_string(end) + ">";
}

FS::FS(size_t count, size_t blk_count, const std::vector<std::string>& path)
    : count_(count), blk_count_(blk_count), path_(path) {
  assert(blk_count % SECTOR_SIZE == 0);

  for (auto& p : path_) {
    int fd = open(p.c_str(), O_RDWR | O_DIRECT);
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
    }
    offset += blk_count;
    mapped_blks++;
  }
  ptr_ = addr;

  free_list_.push_back({0, count, ptr_});
}

FS::~FS() {
  assert(free_list_.size() == 1);
  auto info = free_list_[0];
  size_t size = info.end - info.start;
  assert(size == count_);

  int ret = munmap(ptr_, count_);
  assert(ret == 0);
  for (auto fd : fds_) {
    close(fd);
  }
}

fs_addr_t FS::alloc(size_t count) {
  if (free_list_.size() == 0) {
    assert(false && "Cannot allocate memory");
  }

  count = (count + ALIGNMENT - 1) / ALIGNMENT * ALIGNMENT;

  size_t num_frees = free_list_.size();
  for (size_t i = 0; i < num_frees; ++i) {
    auto info = free_list_[i];
    size_t start = info.start;
    size_t end = info.end;
    size_t free_count = end - start;
    if (count <= free_count) {
      size_t new_end = start + count;
      if (count == free_count) {
        auto it = (free_list_.begin() + i);
        free_list_.erase(it);
      } else {
        free_list_[i] = {new_end, end};
      }
      return {start, new_end, &(((char*)ptr_)[start])};
    }
  }

  assert(false && "Cannot allocate memory");
  return {0, 0, nullptr};
}

void FS::free(fs_addr_t addr) {
  size_t num_frees = free_list_.size();
  if (num_frees == 0) {
    free_list_.push_back(addr);
    return;
  }

  int i = 0;
  for (i = 0; i < num_frees; ++i) {
    auto info = free_list_[i];
    size_t start = info.start;
    if (addr.end <= info.start) {
      free_list_.push_back(free_list_[free_list_.size() - 1]);
      for (int j = num_frees - 1; j >= i; --j) {
        free_list_[j + 1] = free_list_[j];
      }
      free_list_[i] = addr;
      break;
    }
  }

  if (i == num_frees) {
    free_list_.push_back(addr);
  }

  assert(free_list_[free_list_.size() - 1].end <= addr.start);

  int start_idx = i;
  int end_idx = i;
  if (i > 0 && free_list_[i - 1].end == free_list_[i].start) {
    start_idx = i - 1;
  }
  if (i < num_frees && free_list_[i].end == free_list_[i + 1].start) {
    end_idx = i + 1;
  }

  if (start_idx != end_idx) {
    free_list_[start_idx].end = free_list_[end_idx].end;
    int dist = end_idx - start_idx;
    size_t nfrees = free_list_.size();
    for (int j = start_idx + 1; j < nfrees - dist; ++j) {
      free_list_[j] = free_list_[j + dist];
    }
    free_list_.resize(nfrees - dist);
  }
}

void FS::dump() const {
  int i = 0;
  for (auto& addr : free_list_) {
    spdlog::info("addr {}: {}-{}", i, addr.start, addr.end);
    ++i;
  }
}
