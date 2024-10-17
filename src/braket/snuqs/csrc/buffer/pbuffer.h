
#ifndef _PBUFFER_H_
#define _PBUFFER_H_
#include <memory>

#include "buffer/buffer.h"

class PBuffer {
 public:
  PBuffer(size_t count, std::shared_ptr<Buffer> buffer, size_t offset);
  size_t count() const;
  void* buffer();
  size_t offset() const;

 private:
  size_t count_;
  std::shared_ptr<Buffer> buffer_;
  size_t offset_;
};

#endif  //_PBUFFER_H_
