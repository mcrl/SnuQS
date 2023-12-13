#include "assertion.h"
#include "buffer.h"
#include "buffer/mt_raid.h"

#include <cstdlib>
#include <vector>

namespace snuqs {

//
// Storage Buffer
//
template <typename T>
StorageBuffer<T>::StorageBuffer(size_t count, std::vector<std::string> devices)
    : count_(count), raid_(devices) {
  small_buf_ = reinterpret_cast<std::complex<T> *>(aligned_alloc(512, 512));
  if (small_buf_ == nullptr) {
    throw std::bad_alloc();
  }
  raid_.alloc(reinterpret_cast<void **>(&buf_),
              sizeof(std::complex<T>) * count);
}

template <typename T> StorageBuffer<T>::~StorageBuffer() {
  raid_.free(reinterpret_cast<void **>(buf_));
}

template <typename T> std::complex<T> *StorageBuffer<T>::ptr() {
  NOT_SUPPORTED();
  return buf_;
}

template <typename T>
std::complex<T> StorageBuffer<T>::__getitem__(size_t key) {
  uint64_t addr = (uint64_t)&buf_[key];
  uint64_t base_addr = (addr / 512) * 512;
  uint64_t idx = (addr - base_addr) / sizeof(std::complex<T>);

  int err = raid_.read(small_buf_, (void *)base_addr, 512);
  if (err) {
    throw std::domain_error("raid_.read() failed");
  }

  return small_buf_[idx];
}

template <typename T>
void StorageBuffer<T>::__setitem__(size_t key, std::complex<T> val) {
  uint64_t addr = (uint64_t)&buf_[key];
  uint64_t base_addr = (addr / 512) * 512;
  uint64_t idx = (addr - base_addr) / sizeof(std::complex<T>);

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

template <typename T>
void StorageBuffer<T>::read(void *buf, size_t count, size_t offset) {
  NOT_IMPLEMENTED();
}

template <typename T>
void StorageBuffer<T>::write(void *buf, size_t count, size_t offset) {
  NOT_IMPLEMENTED();
}

//template class StorageBuffer<float>;
template class StorageBuffer<double>;

} // namespace snuqs
