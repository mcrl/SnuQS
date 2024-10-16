#include "buffer/buffer_cpu.h"

#include <cassert>
#include <cstdlib>

BufferCPU::BufferCPU(size_t count) : count_(count) {
  buffer_ =
      reinterpret_cast<std::complex<double>*>(aligned_alloc(1ul << 12, count));
  assert(buffer_ != nullptr);
}
BufferCPU::~BufferCPU() { free(buffer_); }

std::complex<double>* BufferCPU::buffer() { return buffer_; }
size_t BufferCPU::count() const { return count_; }
size_t BufferCPU::itemsize() const { return sizeof(std::complex<double>); }
std::string BufferCPU::formatted_string() const {
  return "BufferCPU<" + std::to_string(count_) + ">";
}
