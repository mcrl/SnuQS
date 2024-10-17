#ifndef _BUFFER_PINNED_H_
#define _BUFFER_PINNED_H_

#include "buffer/buffer.h"

class BufferPinned : public Buffer {
 public:
  BufferPinned(size_t count);
  ~BufferPinned();

  virtual void* buffer() override;
  virtual size_t count() const override;
  virtual std::string formatted_string() const override;
  virtual std::shared_ptr<Buffer> cpu() override;
  virtual std::shared_ptr<Buffer> cuda() override;

 private:
  size_t count_;
  void* buffer_ = nullptr;
};
#endif  //_BUFFER_PINNED_H_
