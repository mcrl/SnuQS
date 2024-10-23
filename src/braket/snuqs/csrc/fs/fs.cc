#include "fs.h"

#include <cuda_runtime.h>
#include <fcntl.h>
#include <limits.h>
#include <spdlog/spdlog.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <cassert>
#include <complex>

#include "utils_cuda.h"

#define SECTOR_SIZE (512)
#define ALIGNMENT (512)
#define IO_MAX (0x7ffff000)
std::string fs_addr_t::formatted_string() const {
  return "fs_addr_t<" + std::to_string(start) + "-" + std::to_string(end) + ">";
}

FS::FS(size_t count, size_t blk_count, const std::vector<std::string>& path)
    : row_size_(blk_count * path.size()),
      blk_count_(blk_count),
      count_((count + row_size_ - 1) / row_size_ * row_size_),
      path_(path) {
  assert(blk_count % SECTOR_SIZE == 0);

  for (auto& p : path_) {
    int fd = open(p.c_str(), O_RDWR | O_DIRECT | O_SYNC);
    assert(fd != -1);
    fds_.push_back(fd);
  }

  void* addr = nullptr;
  size_t offset = 0;
  size_t num_blks = count_ / row_size_;
  size_t mapped_blks = 0;

  ptr_ = mmap(nullptr, count, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  munmap(ptr_, count_);
  addr = ptr_;
  while (mapped_blks < num_blks) {
    for (auto fd : fds_) {
      addr = mmap(addr, blk_count, PROT_READ | PROT_WRITE,
                  MAP_SHARED | MAP_FIXED, fd, offset);
      assert(addr != MAP_FAILED);
      addr = reinterpret_cast<void*>(reinterpret_cast<char*>(addr) + blk_count);
    }
    offset += blk_count;
    mapped_blks++;
  }

  for (size_t i = 0; i < count / sizeof(unsigned int); ++i) {
    reinterpret_cast<unsigned int*>(ptr_)[i] = 0xdeadbeaf;
  }

  free_list_.push_back({0, count_, ptr_});
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

void FS::read(fs_addr_t addr, void* buf, size_t count, size_t offset,
              std::shared_ptr<Stream> stream) {
  CUDA_CHECK(cudaStreamSynchronize(
      reinterpret_cast<cudaStream_t>(stream == nullptr ? 0 : stream->get())));

  size_t nbytes = count;
  size_t nbytes_read = 0;

  size_t base_addr = addr.start + offset;
#pragma omp parallel num_threads(fds_.size())
#pragma omp single
  {
    while (nbytes_read < nbytes) {
      auto off_info = get_offset(base_addr + nbytes_read);
      int fd = off_info.first;
      size_t device_offset = off_info.second;
      size_t bytes_to_read =
          std::min((size_t)(blk_count_ - (device_offset % blk_count_)),
                   (size_t)(nbytes - nbytes_read));
#pragma omp task
      {
        size_t current = 0;
        while (current < bytes_to_read) {
          size_t bytes_per_request =
              std::min((size_t)IO_MAX, bytes_to_read - current);
          ssize_t ret = pread(fd, reinterpret_cast<char*>(buf) + current,
                              bytes_to_read, device_offset);
          assert(ret != -1);
          current += ret;
        }
//        spdlog::info(
//            "Reading fd: {}, device_offset: {}, bytes_to_read: {}, elem_a: "
//            "{}+{}i",
//            fd, device_offset, bytes_to_read,
//            reinterpret_cast<std::complex<double>*>(buf)[0].real(),
//            reinterpret_cast<std::complex<double>*>(buf)[0].imag());
      }
      nbytes_read += bytes_to_read;
      buf = reinterpret_cast<char*>(buf) + bytes_to_read;
    }
  }
}

void FS::write(fs_addr_t addr, void* buf, size_t count, size_t offset,
               std::shared_ptr<Stream> stream) {
  CUDA_CHECK(cudaStreamSynchronize(
      reinterpret_cast<cudaStream_t>(stream == nullptr ? 0 : stream->get())));

  size_t nbytes = count;
  size_t nbytes_written = 0;

  size_t base_addr = addr.start + offset;
#pragma omp parallel num_threads(fds_.size())
#pragma omp single
  {
    while (nbytes_written < nbytes) {
      auto off_info = get_offset(base_addr + nbytes_written);
      int fd = off_info.first;
      size_t device_offset = off_info.second;
      size_t bytes_to_write =
          std::min((size_t)(blk_count_ - (device_offset % blk_count_)),
                   (size_t)(nbytes - nbytes_written));
#pragma omp task
      {
//        spdlog::info(
//            "Writing fd: {}, device_offset: {}, bytes_to_write: {}, elem_a: "
//            "{}+{}i",
//            fd, device_offset, bytes_to_write,
//            reinterpret_cast<std::complex<double>*>(buf)[0].real(),
//            reinterpret_cast<std::complex<double>*>(buf)[0].imag());
        size_t current = 0;
        while (current < bytes_to_write) {
          size_t bytes_per_request =
              std::min((size_t)IO_MAX, bytes_to_write - current);
          ssize_t ret = pwrite(fd, reinterpret_cast<char*>(buf) + current,
                               bytes_per_request, device_offset);
          assert(ret != -1);
          current += ret;
        }
      }
      nbytes_written += bytes_to_write;
      buf = reinterpret_cast<char*>(buf) + bytes_to_write;
    }
  }
}

std::pair<int, size_t> FS::get_offset(size_t pos) const {
  size_t num_devices = fds_.size();
  size_t row_size = blk_count_ * num_devices;
  size_t device = (pos / blk_count_) % num_devices;
  size_t offset = (pos / row_size) * blk_count_;
  return {fds_[device], offset};
}
