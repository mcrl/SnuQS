#include "buffer/buffer_pinned.h"

#include <cassert>
#include <cstdlib>

#include "buffer/buffer_cpu.h"
#include "buffer/buffer_cuda.h"
#include "core/runtime.h"
#include "utils_cuda.h"

BufferPinned::BufferPinned(size_t count) : count_(count) {
  CUDA_CHECK(cudaMallocHost(&buffer_, count));
  assert(buffer_ != nullptr);
}
BufferPinned::~BufferPinned() { CUDA_CHECK(cudaFreeHost(buffer_)); }

void* BufferPinned::buffer() { return buffer_; }
size_t BufferPinned::count() const { return count_; }
std::string BufferPinned::formatted_string() const {
  return "BufferPinned<" + std::to_string(count_) + ">";
}

std::shared_ptr<Buffer> BufferPinned::cpu() {
  auto buf = std::make_shared<BufferCPU>(count_);
  memcpyH2H(buf->buffer(), buffer_, count_);
  return buf;
}

std::shared_ptr<Buffer> BufferPinned::cuda() {
  auto buf = std::make_shared<BufferCUDA>(count_);
  memcpyH2D(buf->buffer(), buffer_, count_);
  return buf;
}
