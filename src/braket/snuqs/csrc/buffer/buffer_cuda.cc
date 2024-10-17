#include "buffer/buffer_cuda.h"

#include <cassert>
#include <cstdlib>

#include "buffer/buffer_cpu.h"
#include "core/runtime.h"
#include "utils_cuda.h"

BufferCUDA::BufferCUDA(size_t count) : count_(count) {
  CUDA_CHECK(cudaMalloc(&buffer_, count));
  assert(buffer_ != nullptr);
}
BufferCUDA::~BufferCUDA() { CUDA_CHECK(cudaFree(buffer_)); }
void* BufferCUDA::buffer() { return buffer_; }
size_t BufferCUDA::count() const { return count_; }
std::string BufferCUDA::formatted_string() const {
  return "BufferCUDA<" + std::to_string(count_) + ">";
}

std::shared_ptr<Buffer> BufferCUDA::cpu() {
  auto buf = std::make_shared<BufferCPU>(count_);
  memcpyD2H(buf->buffer(), buffer_, count_);
  return buf;
}

std::shared_ptr<Buffer> BufferCUDA::cuda() {
  auto buf = std::make_shared<BufferCUDA>(count_);
  memcpyD2D(buf->buffer(), buffer_, count_);
  return buf;
}
