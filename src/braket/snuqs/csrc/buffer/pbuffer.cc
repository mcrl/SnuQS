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
  spdlog::info("Pbuffer(device: {}, count: {}, offset: {})", (int)device, count,
               offset);
}
DeviceType PBuffer::device() const { return device_; }
size_t PBuffer::count() const { return count_; }
void* PBuffer::ptr() {
  return &reinterpret_cast<char*>(buffer_->ptr())[offset_];
}
std::shared_ptr<Buffer> PBuffer::buffer() { return buffer_; }
size_t PBuffer::offset() const { return offset_; }

std::shared_ptr<PBuffer> PBuffer::cpu(std::shared_ptr<Stream> stream) {
  if (device_ == DeviceType::CPU) return shared_from_this();
  return std::make_shared<PBuffer>(DeviceType::CPU, count_,
                                   buffer_->cpu(stream), offset_);
}

std::shared_ptr<PBuffer> PBuffer::cuda(std::shared_ptr<Stream> stream) {
  if (device_ == DeviceType::CUDA) return shared_from_this();
  return std::make_shared<PBuffer>(DeviceType::CUDA, count_,
                                   buffer_->cuda(stream), offset_);
}

std::shared_ptr<PBuffer> PBuffer::storage(std::shared_ptr<Stream> stream) {
  if (device_ == DeviceType::STORAGE) return shared_from_this();
  return std::make_shared<PBuffer>(DeviceType::STORAGE, count_,
                                   buffer_->storage(stream), offset_);
}

std::shared_ptr<PBuffer> PBuffer::slice(size_t count, size_t offset) {
  assert(count + offset_ + offset <= count_);
  return std::make_shared<PBuffer>(device_, count, buffer_, offset_ + offset);
}

void PBuffer::copy(std::shared_ptr<PBuffer> other,
                   std::shared_ptr<Stream> stream) {
  switch (other->device()) {
    case DeviceType::CPU:
      copy_from_cpu(other, stream);
      break;
    case DeviceType::CUDA:
      copy_from_cuda(other, stream);
      break;
    case DeviceType::STORAGE:
      copy_from_storage(other, stream);
      break;
    default:
      assert(false);
  }
}

void PBuffer::copy_from_cpu(std::shared_ptr<PBuffer> other,
                            std::shared_ptr<Stream> stream) {
  assert(other->device() == Device::CPU);
  assert(count_ == other->count());
  switch (device_) {
    case DeviceType::CPU:
      memcpyH2H(ptr(), other->ptr(), count_, stream);
      break;
    case DeviceType::CUDA:
      memcpyH2D(ptr(), other->ptr(), count_, stream);
      break;
    case DeviceType::STORAGE: {
      auto bs = dynamic_cast<BufferStorage*>(buffer_.get());
      memcpyH2S(bs->addr(), other->ptr(), count_, stream);
    } break;
    default:
      assert(false);
  }
}

void PBuffer::copy_from_cuda(std::shared_ptr<PBuffer> other,
                             std::shared_ptr<Stream> stream) {
  assert(other->device() == Device::CUDA);
  assert(count_ == other->count());
  switch (device_) {
    case DeviceType::CPU:
      memcpyD2H(ptr(), other->ptr(), count_, stream);
      break;
    case DeviceType::CUDA:
      memcpyD2D(ptr(), other->ptr(), count_, stream);
      break;
    case DeviceType::STORAGE: {
      auto bs = dynamic_cast<BufferStorage*>(other->buffer().get());
      memcpyD2S(bs->addr(), ptr(), count_, stream);
    } break;
    default:
      assert(false);
  }
}
void PBuffer::copy_from_storage(std::shared_ptr<PBuffer> other,
                                std::shared_ptr<Stream> stream) {
  assert(other->device() == Device::STORAGE);
  assert(count_ == other->count());
  switch (device_) {
    case DeviceType::CPU: {
      auto bs = dynamic_cast<BufferStorage*>(other->buffer().get());
      memcpyS2H(ptr(), bs->addr(), count_, stream);
    } break;
    case DeviceType::CUDA: {
      auto bs = dynamic_cast<BufferStorage*>(other->buffer().get());
      memcpyS2D(ptr(), bs->addr(), count_, stream);
    } break;
    case DeviceType::STORAGE: {
      auto bs_src = dynamic_cast<BufferStorage*>(other->buffer().get());
      auto bs_dst = dynamic_cast<BufferStorage*>(buffer_.get());
      memcpyS2S(bs_dst->addr(), bs_src->addr(), count_, stream);
    } break;
    default:
      assert(false);
  }
}
