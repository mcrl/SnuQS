#ifndef _BUFFER_PINNED_H_
#define _BUFFER_PINNED_H_

#include "buffer/buffer.h"

class BufferPinned : public Buffer {
 public:
  BufferPinned(size_t count);
  ~BufferPinned();

  virtual std::complex<double>* buffer() override;
  virtual size_t count() const override;
  virtual size_t itemsize() const override;
  virtual std::string formatted_string() const override;

 private:
  size_t count_;
  std::complex<double>* buffer_ = nullptr;
};
#endif  //_BUFFER_PINNED_H_
