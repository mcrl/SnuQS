#include "assertion.h"
#include "buffer.h"
#include "cuda_api.h"

#include <cstdlib>
#include <new>

namespace snuqs {

//
// Memory Buffer
//
template <typename T>
MemoryBuffer<T>::MemoryBuffer(size_t count) : count_(count), pinned_(false) {
  buf_ = reinterpret_cast<std::complex<T> *>(
      malloc(sizeof(std::complex<T>) * count));
  if (buf_ == nullptr) {
    throw std::bad_alloc();
  }
}

template <typename T>
MemoryBuffer<T>::MemoryBuffer(size_t count, bool pinned)
    : count_(count), pinned_(pinned) {
  if (pinned_) {
    cuda::api::mallocHost(reinterpret_cast<void **>(&buf_),
                          sizeof(std::complex<T>) * count);
  } else {
    buf_ = reinterpret_cast<std::complex<T> *>(
        malloc(sizeof(std::complex<T>) * count));
  }

  if (buf_ == nullptr) {
    throw std::bad_alloc();
  }
}

template <typename T> MemoryBuffer<T>::~MemoryBuffer() {
  if (buf_ == nullptr)
    return;

  if (pinned_) {
    cuda::api::freeHost(buf_);
  } else {
    free(buf_);
  }
}

template <typename T> std::complex<T> *MemoryBuffer<T>::ptr() { return buf_; }

template <typename T> std::complex<T> MemoryBuffer<T>::__getitem__(size_t key) {
  return buf_[key];
}

template <typename T>
void MemoryBuffer<T>::__setitem__(size_t key, std::complex<T> val) {
  buf_[key] = val;
}

template class MemoryBuffer<float>;
template class MemoryBuffer<double>;

} // namespace snuqs
