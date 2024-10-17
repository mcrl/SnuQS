#include "buffer/buffer_cpu.h"

#include <cassert>
#include <complex>
#include <cstdlib>

#include "buffer/buffer_cuda.h"
#include "core/runtime.h"

BufferCPU::BufferCPU(size_t count) : count_(count) {
  buffer_ = aligned_alloc(1ul << 12, count);
  assert(buffer_ != nullptr);
}
BufferCPU::~BufferCPU() { free(buffer_); }

void* BufferCPU::buffer() { return buffer_; }
size_t BufferCPU::count() const { return count_; }
std::string BufferCPU::formatted_string() const {
  return "BufferCPU<" + std::to_string(count_) + ">";
}

std::shared_ptr<Buffer> BufferCPU::cpu() {
  auto buf = std::make_shared<BufferCPU>(count_);
  memcpyH2H(buf->buffer(), buffer_, count_);
  return buf;
}

std::shared_ptr<Buffer> BufferCPU::cuda() {
  auto buf = std::make_shared<BufferCUDA>(count_);
  memcpyH2D(buf->buffer(), buffer_, count_);
  return buf;
}
