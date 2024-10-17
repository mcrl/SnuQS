#include "buffer/pbuffer.h"

PBuffer::PBuffer(size_t count, std::shared_ptr<Buffer> buffer, size_t offset)
    : count_(count), buffer_(buffer), offset_(offset) {
  assert(count_ <= buffer_->count());
}
size_t PBuffer::count() const { return count_; }
void* PBuffer::buffer() {
  return &reinterpret_cast<char*>(buffer_->buffer())[offset_];
}
size_t PBuffer::offset() const { return offset_; }
