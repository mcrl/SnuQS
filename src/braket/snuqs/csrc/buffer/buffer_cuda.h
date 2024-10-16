#ifndef _BUFFER_CUDA_H_
#define _BUFFER_CUDA_H_

#include <cuda_runtime.h>
#include <spdlog/spdlog.h>

#include <complex>

#include "buffer/buffer.h"

class BufferCUDA : public Buffer {
 public:
  BufferCUDA(size_t count);
  ~BufferCUDA();
  virtual std::complex<double>* buffer() override;
  virtual size_t count() const override;
  virtual size_t itemsize() const override;
  virtual std::string formatted_string() const override;

 private:
  size_t count_;
  std::complex<double>* buffer_ = nullptr;
};
#endif  //_BUFFER_CUDA_H_
