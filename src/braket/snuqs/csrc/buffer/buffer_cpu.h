#ifndef _BUFFER_CPU_H_
#define _BUFFER_CPU_H_

#include "buffer/buffer.h"

class BufferCPU : public Buffer {
 public:
  BufferCPU(size_t count);
  ~BufferCPU();

  virtual std::complex<double>* buffer() override;
  virtual size_t count() const override;
  virtual size_t itemsize() const override;
  virtual std::string formatted_string() const override;

 private:
  size_t count_;
  std::complex<double>* buffer_ = nullptr;
};
#endif  //_BUFFER_CPU_H_
