#include "assertion.h"
#include "buffer.h"
#include "cuda_api.h"

#include <cstdlib>
#include <new>

#include <iostream>

namespace snuqs {
namespace cuda {

//
// Cuda Buffer
//
template <typename T> CudaBuffer<T>::CudaBuffer(size_t count) : count_(count) {
  api::malloc((void **)&buf_, sizeof(std::complex<T>) * count);
  if (buf_ == nullptr) {
    throw std::bad_alloc();
  }
}

template <typename T> CudaBuffer<T>::~CudaBuffer() {
  if (buf_ != nullptr)
    cudaFree(buf_);
}
template <typename T> void *CudaBuffer<T>::ptr() { return buf_; }

template <typename T>
std::complex<double> CudaBuffer<T>::__getitem__(size_t key) {
  NOT_IMPLEMENTED();
}

template <typename T>
void CudaBuffer<T>::__setitem__(size_t key, std::complex<double> val) {
  NOT_IMPLEMENTED();
}

template <typename T>
void CudaBuffer<T>::read(void *buf, size_t count, size_t offset) {
  cudaMemcpy(buf, &buf_[offset], sizeof(std::complex<T>) * count,
             cudaMemcpyDeviceToHost);
}

template <typename T>
void CudaBuffer<T>::write(void *buf, size_t count, size_t offset) {
  cudaMemcpy(&buf_[offset], buf, sizeof(std::complex<T>) * count,
             cudaMemcpyHostToDevice);
}

template class CudaBuffer<double>;
template class CudaBuffer<float>;

} // namespace cuda
} // namespace snuqs
