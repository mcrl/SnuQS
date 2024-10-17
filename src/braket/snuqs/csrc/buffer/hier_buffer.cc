#include "buffer/hier_buffer.h"

#include <cuda_runtime.h>
#include <fcntl.h>
#include <spdlog/spdlog.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <cstdlib>

#include "buffer/buffer_cpu.h"
#include "buffer/buffer_cuda.h"
#include "buffer/buffer_storage.h"

HierBuffer::HierBuffer(size_t count) : count_(count) {
  ptr_ = std::make_shared<BufferCPU>(count);
  assert(ptr_ != nullptr);
}

HierBuffer::HierBuffer(size_t count, size_t l1_count)
    : count_(count), l1_count_(l1_count) {
  ptr_ = std::make_shared<BufferCPU>(count);
  ptr_cuda_ = std::make_shared<BufferCUDA>(l1_count);
  assert(ptr_ != nullptr);
  assert(ptr_cuda_ != nullptr);
}

HierBuffer::HierBuffer(size_t count, size_t l1_count, size_t l1_blk_count)
    : count_(count), l1_count_(l1_count), l1_blk_count_(l1_blk_count) {
  assert(l1_count <= count);
  assert(l1_blk_count <= l1_count);

  ptr_ = std::make_shared<BufferCPU>(count);
  ptr_cuda_ = std::make_shared<BufferCUDA>(l1_count);
  assert(ptr_ != nullptr);
  assert(ptr_cuda_ != nullptr);
}

HierBuffer::HierBuffer(size_t count, size_t l1_count, size_t l1_blk_count,
                       const std::vector<std::string> &path)
    : count_(count),
      l1_count_(l1_count),
      l1_blk_count_(l1_blk_count),
      path_(path) {
  assert(l1_count <= count);
  assert(l1_blk_count <= l1_count);

  assert(count % l1_count == 0);
  assert(l1_count % l1_blk_count == 0);

  ptr_storage_ = std::make_shared<BufferStorage>(count, l1_blk_count, path);
  ptr_ = std::make_shared<BufferCPU>(l1_count);

  assert(ptr_storage_ != nullptr);
  assert(ptr_ != nullptr);
}

HierBuffer::HierBuffer(size_t count, size_t l1_count, size_t l1_blk_count,
                       size_t l2_count, const std::vector<std::string> &path)
    : count_(count), l1_count_(l1_count), l2_count_(l2_count), path_(path) {
  assert(l1_count <= count);
  assert(l1_blk_count <= l1_count);
  assert(l2_count <= l1_count);

  ptr_storage_ = std::make_shared<BufferStorage>(count, l1_blk_count, path);
  ptr_ = std::make_shared<BufferCPU>(l1_count);
  ptr_cuda_ = std::make_shared<BufferCUDA>(l2_count);

  assert(ptr_storage_ != nullptr);
  assert(ptr_ != nullptr);
  assert(ptr_cuda_ != nullptr);
}

HierBuffer::~HierBuffer() {}

std::string HierBuffer::formatted_string() const {
  return "HierBuffer<" + std::to_string(count_) + "," +
         std::to_string(l1_count_) + "," + std::to_string(l1_blk_count_) + "," +
         std::to_string(l2_count_) + ">";
}
DeviceType HierBuffer::device() const { return device_; }
std::shared_ptr<Buffer> HierBuffer::cpu() { return ptr_; }
std::shared_ptr<Buffer> HierBuffer::cuda() { return ptr_cuda_; }
std::shared_ptr<Buffer> HierBuffer::storage() { return ptr_storage_; }

void HierBuffer::to_cpu() {}
void HierBuffer::to_cuda() {}
void HierBuffer::to_storage() {}
