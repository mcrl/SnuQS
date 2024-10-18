#ifndef _BUFFER_CPU_H_
#define _BUFFER_CPU_H_

#include "buffer/buffer.h"

class BufferCPU : public Buffer {
 public:
  BufferCPU(size_t count);
  BufferCPU(size_t count, bool pinned);
  BufferCPU(size_t count, void *ptr);
  ~BufferCPU();

  virtual void* ptr() override;
  virtual size_t count() const override;
  virtual std::string formatted_string() const override;
  bool pinned() const;
  virtual std::shared_ptr<Buffer> cpu() override;
  virtual std::shared_ptr<Buffer> cuda() override;
  virtual std::shared_ptr<Buffer> storage() override;

 private:
  size_t count_;
  void* ptr_ = nullptr;
  bool pinned_ = false;
};
#endif  //_BUFFER_CPU_H_
