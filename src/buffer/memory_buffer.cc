#include "assertion.h"
#include "buffer.h"

#include <cstdlib>
#include <new>

#include <iostream>

namespace snuqs {

//
// Memory Buffer
//
MemoryBuffer::MemoryBuffer(size_t size) : size_(size) {
  buf_ = reinterpret_cast<double *>(malloc(size));
  if (buf_ == nullptr) {
    throw std::bad_alloc();
  }
}

MemoryBuffer::~MemoryBuffer() {
  if (buf_ != nullptr)
    free(buf_);
}

double MemoryBuffer::__getitem__(size_t key) {
  return buf_[key];
}

void MemoryBuffer::__setitem__(size_t key, double val) {
  buf_[key] = val;
}

} // namespace snuqs
