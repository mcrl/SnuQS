#include "buffer/pbuffer.h"

#include "buffer/buffer_cpu.h"
#include "buffer/buffer_cuda.h"
#include "buffer/buffer_storage.h"

PBuffer::PBuffer(size_t count)
    : device_(DeviceType::CPU),
      count_(count),
      buffer_(std::make_shared<BufferCPU>(count)),
      offset_(0) {}

PBuffer::PBuffer(DeviceType device, size_t count)
    : device_(device), count_(count), offset_(0) {
  switch (device) {
    case DeviceType::CPU:
      buffer_ = std::make_shared<BufferCPU>(count);
      break;
    case DeviceType::CUDA:
      buffer_ = std::make_shared<BufferCUDA>(count);
      break;
    case DeviceType::STORAGE:
      buffer_ = std::make_shared<BufferStorage>(count);
      break;
    default:
      assert(false);
  }
}

PBuffer::PBuffer(DeviceType device, size_t count,
                 std::shared_ptr<Buffer> buffer, size_t offset)
    : device_(device), count_(count), buffer_(buffer), offset_(offset) {
  assert(count_ <= buffer_->count());
}
DeviceType PBuffer::device() const { return device_; }
size_t PBuffer::count() const { return count_; }
void* PBuffer::buffer() {
  return &reinterpret_cast<char*>(buffer_->buffer())[offset_];
}
size_t PBuffer::offset() const { return offset_; }

std::shared_ptr<PBuffer> PBuffer::cpu() {
  return std::make_shared<PBuffer>(DeviceType::CPU, count_, buffer_->cpu(),
                                   offset_);
}

std::shared_ptr<PBuffer> PBuffer::cuda() {
  return std::make_shared<PBuffer>(DeviceType::CUDA, count_, buffer_->cuda(),
                                   offset_);
}
