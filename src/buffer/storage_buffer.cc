#include "assertion.h"
#include "buffer.h"
#include "buffer/mt_raid.h"

#include <cstdlib>
#include <iostream>
#include <vector>

namespace snuqs {

//
// Storage Buffer
//
StorageBuffer::StorageBuffer(size_t count, std::vector<std::string> devices)
    : count_(count), raid_(devices) {
  small_buf_ =
      reinterpret_cast<std::complex<double> *>(aligned_alloc(512, 512));
  if (small_buf_ == nullptr) {
    throw std::bad_alloc();
  }
  raid_.alloc(reinterpret_cast<void **>(&buf_),
              sizeof(std::complex<double>) * count);
}

StorageBuffer::~StorageBuffer() { raid_.free(reinterpret_cast<void **>(buf_)); }

std::complex<double> StorageBuffer::__getitem__(size_t key) {
  uint64_t addr = (uint64_t)&buf_[key];
  uint64_t base_addr = (addr / 512) * 512;
  uint64_t idx = (addr - base_addr) / sizeof(std::complex<double>);

  int err = raid_.read(small_buf_, (void *)base_addr, 512);
  if (err) {
    throw std::domain_error("raid_.read() failed");
  }

  return small_buf_[idx];
}

void StorageBuffer::__setitem__(size_t key, std::complex<double> val) {
  uint64_t addr = (uint64_t)&buf_[key];
  uint64_t base_addr = (addr / 512) * 512;
  uint64_t idx = (addr - base_addr) / sizeof(std::complex<double>);

  int err = raid_.read(small_buf_, (void *)base_addr, 512);
  if (err) {
    throw std::domain_error("raid_.read() failed");
  }
  small_buf_[idx] = val;

  err = raid_.write((void *)base_addr, small_buf_, 512);
  if (err) {
    throw std::domain_error("raid_.read() failed");
  }
}

void StorageBuffer::read(void *buf, size_t count, size_t offset) {
  NOT_IMPLEMENTED();
}

void StorageBuffer::write(void *buf, size_t count, size_t offset) {
  NOT_IMPLEMENTED();
}

} // namespace snuqs
