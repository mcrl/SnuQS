#ifndef _BUFFER_CPU_H_
#define _BUFFER_CPU_H_

#include "buffer/buffer.h"

class BufferCPU : public Buffer {
 public:
  BufferCPU(size_t count);
  ~BufferCPU();

  virtual void* buffer() override;
  virtual size_t count() const override;
  virtual std::string formatted_string() const override;
  virtual std::shared_ptr<Buffer> cpu() override;
  virtual std::shared_ptr<Buffer> cuda() override;

 private:
  size_t count_;
  void* buffer_ = nullptr;
};
#endif  //_BUFFER_CPU_H_
