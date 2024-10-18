#include "buffer/buffer_cpu.h"

#include <spdlog/spdlog.h>

#include <cassert>
#include <cstdlib>

#include "buffer/buffer_cuda.h"
#include "core/runtime.h"
#include "utils_cuda.h"

BufferCPU::BufferCPU(size_t count) : count_(count), pinned_(false) {
  spdlog::info("BufferCPU({})", count);
  ptr_ = aligned_alloc(1ul << 12, count);
  assert(ptr_ != nullptr);
}

BufferCPU::BufferCPU(size_t count, bool pinned)
    : count_(count), pinned_(pinned) {
  spdlog::info("BufferCPU({}, {})", count, pinned);
  if (pinned_) {
    CUDA_CHECK(cudaMallocHost(&ptr_, count));
  } else {
    ptr_ = aligned_alloc(1ul << 12, count);
  }
  assert(ptr_ != nullptr);
}

BufferCPU::~BufferCPU() {
  if (pinned_) {
    CUDA_CHECK(cudaFreeHost(ptr_));
  } else {
    free(ptr_);
  }
}

void* BufferCPU::ptr() { return ptr_; }
size_t BufferCPU::count() const { return count_; }
std::string BufferCPU::formatted_string() const {
  return "BufferCPU<" + std::to_string(count_) +
         (pinned_ ? ", pinned" : ", not pinned") + ">";
}

bool BufferCPU::pinned() const { return pinned_; }

std::shared_ptr<Buffer> BufferCPU::cpu() { return shared_from_this(); }

std::shared_ptr<Buffer> BufferCPU::cuda() {
  auto buf = std::make_shared<BufferCUDA>(count_);
  memcpyH2D(buf->ptr(), ptr_, count_);
  return buf;
}

std::shared_ptr<Buffer> BufferCPU::storage() {
  assert(false);
  return nullptr;
}
