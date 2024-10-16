#include "buffer/buffer_cuda.h"

#include <cassert>
#include <cstdlib>

#include "utils_cuda.h"

BufferCUDA::BufferCUDA(size_t count) : count_(count) {
  CUDA_CHECK(cudaMalloc(&buffer_, sizeof(std::complex<double>) * count));
  assert(buffer_ != nullptr);
}
BufferCUDA::~BufferCUDA() { CUDA_CHECK(cudaFree(buffer_)); }
std::complex<double>* BufferCUDA::buffer() { return buffer_; }
size_t BufferCUDA::count() const { return count_; }
size_t BufferCUDA::itemsize() const { return sizeof(std::complex<double>); }
std::string BufferCUDA::formatted_string() const {
  return "BufferCUDA<" + std::to_string(count_) + ">";
}
