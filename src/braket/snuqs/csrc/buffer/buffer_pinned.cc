#include "buffer/buffer_pinned.h"

#include <cassert>
#include <cstdlib>

#include "utils_cuda.h"

BufferPinned::BufferPinned(size_t count) : count_(count) {
  CUDA_CHECK(cudaMallocHost(&buffer_, sizeof(std::complex<double>) * count));
  assert(buffer_ != nullptr);
}
BufferPinned::~BufferPinned() { CUDA_CHECK(cudaFreeHost(buffer_)); }

std::complex<double>* BufferPinned::buffer() { return buffer_; }
size_t BufferPinned::count() const { return count_; }
size_t BufferPinned::itemsize() const { return sizeof(std::complex<double>); }
std::string BufferPinned::formatted_string() const {
  return "BufferPinned<" + std::to_string(count_) + ">";
}
