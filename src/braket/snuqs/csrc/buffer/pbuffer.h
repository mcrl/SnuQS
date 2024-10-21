
#ifndef _PBUFFER_H_
#define _PBUFFER_H_
#include <memory>

#include "buffer/buffer.h"
#include "device_types.h"
#include "stream/stream.h"

class PBuffer : public std::enable_shared_from_this<PBuffer> {
 public:
  PBuffer(size_t count);
  PBuffer(size_t count, bool pinned);
  PBuffer(DeviceType device, size_t count);
  PBuffer(DeviceType device, size_t count, std::shared_ptr<Buffer> buffer,
          size_t offset);
  DeviceType device() const;
  size_t count() const;
  void* ptr();
  std::shared_ptr<Buffer> buffer();
  size_t offset() const;

  void copy(std::shared_ptr<PBuffer> other, std::shared_ptr<Stream> stream);
  void copy_from_cpu(std::shared_ptr<PBuffer> other,
                     std::shared_ptr<Stream> stream);
  void copy_from_cuda(std::shared_ptr<PBuffer> other,
                      std::shared_ptr<Stream> stream);
  void copy_from_storage(std::shared_ptr<PBuffer> other,
                         std::shared_ptr<Stream> stream);

  std::shared_ptr<PBuffer> cpu(std::shared_ptr<Stream> stream);
  std::shared_ptr<PBuffer> cuda(std::shared_ptr<Stream> stream);
  std::shared_ptr<PBuffer> storage(std::shared_ptr<Stream> stream);

  std::shared_ptr<PBuffer> slice(size_t count, size_t offset);

 private:
  DeviceType device_;
  size_t count_;
  std::shared_ptr<Buffer> buffer_;
  size_t offset_;
};

#endif  //_PBUFFER_H_
