#include "buffer/pbuffer.h"

#include <spdlog/spdlog.h>

#include "buffer/buffer_cpu.h"
#include "buffer/buffer_cuda.h"
#include "buffer/buffer_storage.h"
#include "core/runtime.h"

PBuffer::PBuffer(size_t count)
    : device_(DeviceType::CPU),
      count_(count),
      buffer_(std::make_shared<BufferCPU>(count)),
      offset_(0) {}

PBuffer::PBuffer(size_t count, bool pinned)
    : device_(DeviceType::CPU),
      count_(count),
      buffer_(std::make_shared<BufferCPU>(count, pinned)),
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
void* PBuffer::ptr() {
  return &reinterpret_cast<char*>(buffer_->ptr())[offset_];
}
std::shared_ptr<Buffer> PBuffer::buffer() { return buffer_; }
size_t PBuffer::offset() const { return offset_; }

std::shared_ptr<PBuffer> PBuffer::cpu() {
  if (device_ == DeviceType::CPU) return shared_from_this();
  return std::make_shared<PBuffer>(DeviceType::CPU, count_, buffer_->cpu(),
                                   offset_);
}

std::shared_ptr<PBuffer> PBuffer::cuda() {
  if (device_ == DeviceType::CUDA) return shared_from_this();
  return std::make_shared<PBuffer>(DeviceType::CUDA, count_, buffer_->cuda(),
                                   offset_);
}

std::shared_ptr<PBuffer> PBuffer::storage() {
  if (device_ == DeviceType::STORAGE) return shared_from_this();
  return std::make_shared<PBuffer>(DeviceType::STORAGE, count_,
                                   buffer_->storage(), offset_);
}

std::shared_ptr<PBuffer> PBuffer::slice(size_t count, size_t offset) {
  assert(count + offset_ + offset <= count_);
  return std::make_shared<PBuffer>(device_, count, buffer_, offset_ + offset);
}

void PBuffer::copy(std::shared_ptr<PBuffer> other) {
  switch (other->device()) {
    case DeviceType::CPU:
      copy_from_cpu(other);
      break;
    case DeviceType::CUDA:
      copy_from_cuda(other);
      break;
    case DeviceType::STORAGE:
      copy_from_storage(other);
      break;
    default:
      assert(false);
  }
}

void PBuffer::copy_from_cpu(std::shared_ptr<PBuffer> other) {
  assert(other->device() == Device::CPU);
  assert(count_ == other->count());
  switch (device_) {
    case DeviceType::CPU:
      memcpyH2H(ptr(), other->ptr(), count_);
      break;
    case DeviceType::CUDA:
      memcpyH2D(ptr(), other->ptr(), count_);
      break;
    case DeviceType::STORAGE: {
      auto bs = dynamic_cast<BufferStorage*>(buffer_.get());
      memcpyH2S(bs->addr(), other->ptr(), count_);
    } break;
    default:
      assert(false);
  }
}

void PBuffer::copy_from_cuda(std::shared_ptr<PBuffer> other) {
  assert(other->device() == Device::CUDA);
  assert(count_ == other->count());
  switch (device_) {
    case DeviceType::CPU:
      memcpyD2H(ptr(), other->ptr(), count_);
      break;
    case DeviceType::CUDA:
      memcpyD2D(ptr(), other->ptr(), count_);
      break;
    case DeviceType::STORAGE: {
      auto bs = dynamic_cast<BufferStorage*>(other->buffer().get());
      auto buf_cpu = std::make_shared<BufferCPU>(count_, true);  // pinned
      spdlog::info("Allocating auxiliary pinned CPU buffer of {} bytes",
                   count_);
      memcpyD2H(buf_cpu->ptr(), ptr(), count_);
      memcpyH2S(bs->addr(), buf_cpu->ptr(), count_);
    } break;
    default:
      assert(false);
  }
}
void PBuffer::copy_from_storage(std::shared_ptr<PBuffer> other) {
  assert(other->device() == Device::STORAGE);
  assert(count_ == other->count());
  switch (device_) {
    case DeviceType::CPU: {
      auto bs = dynamic_cast<BufferStorage*>(other->buffer().get());
      memcpyS2H(ptr(), bs->addr(), count_);
    } break;
    case DeviceType::CUDA: {
      auto bs = dynamic_cast<BufferStorage*>(other->buffer().get());
      auto buf_cpu = std::make_shared<BufferCPU>(count_, true);  // pinned
      spdlog::info("Allocating auxiliary pinned CPU buffer of {} bytes",
                   count_);
      memcpyS2H(buf_cpu->ptr(), bs->addr(), count_);
      memcpyH2D(ptr(), buf_cpu->ptr(), count_);
    } break;
    case DeviceType::STORAGE: {
      auto bs_src = dynamic_cast<BufferStorage*>(other->buffer().get());
      auto bs_dst = dynamic_cast<BufferStorage*>(buffer_.get());
      auto buf_cpu = std::make_shared<BufferCPU>(count_, true);  // pinned
      spdlog::info("Allocating auxiliary pinned CPU buffer of {} bytes",
                   count_);
      memcpyS2H(buf_cpu->ptr(), bs_src->addr(), count_);
      memcpyH2S(bs_dst->addr(), buf_cpu->ptr(), count_);
    } break;
    default:
      assert(false);
  }
}
