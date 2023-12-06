#include "assertion.h"
#include "buffer.h"

#include <cstdlib>
#include <cuda_runtime.h>
#include <new>

#include <iostream>

namespace snuqs {
namespace cuda {

//
// Cuda Buffer
//
template <typename T> CudaBuffer<T>::CudaBuffer(size_t count) : count_(count) {
  CUDA_ASSERT(cudaMalloc(&buf_, sizeof(std::complex<T>) * count));
  if (buf_ == nullptr) {
    throw std::bad_alloc();
  }
}

template <typename T> CudaBuffer<T>::~CudaBuffer() {
  if (buf_ != nullptr)
    CUDA_ASSERT(cudaFree(buf_));
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
  CUDA_ASSERT(cudaMemcpy(buf, &buf_[offset], sizeof(std::complex<T>) * count,
                         cudaMemcpyDeviceToHost));
}

template <typename T>
void CudaBuffer<T>::write(void *buf, size_t count, size_t offset) {
  CUDA_ASSERT(cudaMemcpy(&buf_[offset], buf, sizeof(std::complex<T>) * count,
                         cudaMemcpyHostToDevice));
}

template class CudaBuffer<double>;
template class CudaBuffer<float>;

} // namespace cuda
} // namespace snuqs
