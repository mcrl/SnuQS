#ifndef __BUFFER_H__
#define __BUFFER_H__

#include "buffer/mt_raid.h"
#include <complex>
#include <cstddef>

namespace snuqs {

template <typename T> class Buffer {
public:
  virtual ~Buffer() = default;
  virtual std::complex<T> *ptr() = 0;
  virtual std::complex<T> __getitem__(size_t key) = 0;
  virtual void __setitem__(size_t key, std::complex<T> val) = 0;
  virtual size_t __len__() = 0;
};

template <typename T> class MemoryBuffer : public Buffer<T> {
public:
  MemoryBuffer(size_t count);
  MemoryBuffer(size_t count, bool pinned);
  virtual ~MemoryBuffer() override;
  virtual std::complex<T> *ptr() override;
  virtual std::complex<T> __getitem__(size_t key) override;
  virtual void __setitem__(size_t key, std::complex<T> val) override;
  virtual size_t __len__() override;

private:
  bool pinned_;
  size_t count_;
  std::complex<T> *buf_;
};

template <typename T> class StorageBuffer : public Buffer<T> {
public:
  StorageBuffer(size_t count, std::vector<std::string> devices);
  virtual ~StorageBuffer() override;
  virtual std::complex<T> *ptr() override;
  virtual std::complex<T> __getitem__(size_t key) override;
  virtual void __setitem__(size_t key, std::complex<T> val) override;
  virtual size_t __len__() override;

  void read(void *buf, size_t count, size_t offset);
  void write(void *buf, size_t count, size_t offset);

private:
  size_t count_;
  std::complex<T> *buf_;
  std::complex<T> *small_buf_;
  MTRaid raid_;
};

namespace cuda {

template <typename T> class CudaBuffer : public Buffer<T> {
public:
  CudaBuffer(size_t count);
  virtual ~CudaBuffer() override;
  virtual std::complex<T> *ptr() override;
  virtual std::complex<T> __getitem__(size_t key) override;
  virtual void __setitem__(size_t key, std::complex<T> val) override;
  virtual size_t __len__() override;

  void read(void *buf, size_t count, size_t offset);
  void write(void *buf, size_t count, size_t offset);

private:
  size_t count_;
  std::complex<T> *buf_;
};

} // namespace cuda

} // namespace snuqs

#endif //__BUFFER_H__
