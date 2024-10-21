#ifndef _BUFFER_CUDA_H_
#define _BUFFER_CUDA_H_

#include <cuda_runtime.h>

#include "buffer/buffer.h"

class BufferCUDA : public Buffer {
 public:
  BufferCUDA(size_t count);
  ~BufferCUDA();
  virtual void* ptr() override;
  virtual size_t count() const override;
  virtual std::string formatted_string() const override;
  virtual std::shared_ptr<Buffer> cpu(std::shared_ptr<Stream> stream) override;
  virtual std::shared_ptr<Buffer> cuda(std::shared_ptr<Stream> stream) override;
  virtual std::shared_ptr<Buffer> storage(std::shared_ptr<Stream> stream) override;

 private:
  size_t count_;
  void* ptr_ = nullptr;
};
#endif  //_BUFFER_CUDA_H_
