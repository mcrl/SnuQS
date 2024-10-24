#include "buffer/buffer_cuda.h"

#include <spdlog/spdlog.h>

#include <cassert>
#include <cstdlib>

#include "buffer/buffer_cpu.h"
#include "buffer/buffer_storage.h"
#include "core/runtime.h"
#include "utils_cuda.h"

BufferCUDA::BufferCUDA(size_t count) : count_(count) {
  spdlog::info("BufferCUDA({})", count);
  CUDA_CHECK(cudaMalloc(&ptr_, count));
  assert(ptr_ != nullptr);
}
BufferCUDA::~BufferCUDA() {
  spdlog::info("~BufferCUDA({})", count_);
  CUDA_CHECK(cudaFree(ptr_));
}
void* BufferCUDA::ptr() { return ptr_; }
size_t BufferCUDA::count() const { return count_; }
std::string BufferCUDA::formatted_string() const {
  return "BufferCUDA<" + std::to_string(count_) + ">";
}

std::shared_ptr<Buffer> BufferCUDA::cpu(std::shared_ptr<Stream> stream) {
  auto buf = std::make_shared<BufferCPU>(count_);
  memcpyD2H(buf->ptr(), ptr_, count_, stream);
  return buf;
}

std::shared_ptr<Buffer> BufferCUDA::cuda(std::shared_ptr<Stream> stream) {
  return shared_from_this();
}

std::shared_ptr<Buffer> BufferCUDA::storage(std::shared_ptr<Stream> stream) {
  auto buf_storage = std::make_shared<BufferStorage>(count_);
  memcpyD2S(buf_storage->addr(), ptr_, count_, 0, stream);
  return buf_storage;
}
