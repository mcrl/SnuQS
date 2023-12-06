#ifndef __BUFFER_H__
#define __BUFFER_H__

#include "buffer/mt_raid.h"
#include <complex>
#include <cstddef>

namespace snuqs {

class Buffer {
public:
  virtual ~Buffer() {}
  virtual std::complex<double> __getitem__(size_t key) = 0;
  virtual void __setitem__(size_t key, std::complex<double> val) = 0;
};

class MemoryBuffer : public Buffer {
public:
  MemoryBuffer(size_t count);
  virtual ~MemoryBuffer() override;
  virtual std::complex<double> __getitem__(size_t key) override;
  virtual void __setitem__(size_t key, std::complex<double> val) override;

  void *ptr();

private:
  size_t count_;
  std::complex<double> *buf_;
};

class StorageBuffer : public Buffer {
public:
  StorageBuffer(size_t count, std::vector<std::string> devices);
  virtual ~StorageBuffer() override;
  virtual std::complex<double> __getitem__(size_t key) override;
  virtual void __setitem__(size_t key, std::complex<double> val) override;

  void read(void *buf, size_t count, size_t offset);
  void write(void *buf, size_t count, size_t offset);

private:
  size_t count_;
  std::complex<double> *buf_;
  std::complex<double> *small_buf_;
  MTRaid raid_;
};

namespace cuda {

template <typename T> class CudaBuffer : public Buffer {
public:
  CudaBuffer(size_t count);
  virtual ~CudaBuffer() override;
  virtual std::complex<double> __getitem__(size_t key) override;
  virtual void __setitem__(size_t key, std::complex<double> val) override;

  void *ptr();

  void read(void *buf, size_t count, size_t offset);
  void write(void *buf, size_t count, size_t offset);

private:
  size_t count_;
  std::complex<T> *buf_;
};

} // namespace cuda

} // namespace snuqs

#endif //__BUFFER_H__
