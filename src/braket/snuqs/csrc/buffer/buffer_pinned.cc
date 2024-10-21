#include "buffer/buffer_pinned.h"

#include <cassert>
#include <cstdlib>

#include "buffer/buffer_cpu.h"
#include "buffer/buffer_cuda.h"
#include "core/runtime.h"
#include "utils_cuda.h"

BufferPinned::BufferPinned(size_t count) : count_(count) {
  CUDA_CHECK(cudaMallocHost(&ptr_, count));
  assert(ptr_ != nullptr);
}
BufferPinned::~BufferPinned() { CUDA_CHECK(cudaFreeHost(ptr_)); }

void* BufferPinned::ptr() { return ptr_; }
size_t BufferPinned::count() const { return count_; }
std::string BufferPinned::formatted_string() const {
  return "BufferPinned<" + std::to_string(count_) + ">";
}

std::shared_ptr<Buffer> BufferPinned::cpu() { return shared_from_this(); }

std::shared_ptr<Buffer> BufferPinned::cuda() {
  auto buf = std::make_shared<BufferCUDA>(count_);
  memcpyH2D(buf->ptr(), ptr_, count_);
  return buf;
}

std::shared_ptr<Buffer> BufferPinned::storage() {
  assert(false);
  return nullptr;
}
