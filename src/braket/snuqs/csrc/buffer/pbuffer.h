
#ifndef _PBUFFER_H_
#define _PBUFFER_H_
#include <memory>

#include "buffer/buffer.h"
#include "device_types.h"

class PBuffer {
 public:
  PBuffer(size_t count);
  PBuffer(DeviceType device, size_t count);
  PBuffer(DeviceType device, size_t count, std::shared_ptr<Buffer> buffer,
          size_t offset);
  DeviceType device() const;
  size_t count() const;
  void* buffer();
  size_t offset() const;

  std::shared_ptr<PBuffer> cpu();
  std::shared_ptr<PBuffer> cuda();

 private:
  DeviceType device_;
  size_t count_;
  std::shared_ptr<Buffer> buffer_;
  size_t offset_;
};

#endif  //_PBUFFER_H_
