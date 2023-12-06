#include "assertion.h"
#include "buffer.h"

#include <cstdlib>
#include <new>

#include <iostream>

namespace snuqs {

//
// Memory Buffer
//
MemoryBuffer::MemoryBuffer(size_t count) : count_(count) {
  buf_ = reinterpret_cast<std::complex<double> *>(
      malloc(sizeof(std::complex<double>) * count));
  if (buf_ == nullptr) {
    throw std::bad_alloc();
  }
}

MemoryBuffer::~MemoryBuffer() {
  if (buf_ != nullptr)
    free(buf_);
}
void *MemoryBuffer::ptr() { return buf_; }

std::complex<double> MemoryBuffer::__getitem__(size_t key) { return buf_[key]; }

void MemoryBuffer::__setitem__(size_t key, std::complex<double> val) {
  buf_[key] = val;
}

} // namespace snuqs
