#ifndef __BUFFER_H__
#define __BUFFER_H__

#include <cstddef>

namespace snuqs {

class Buffer {
public:
  virtual ~Buffer() {}
  virtual double __getitem__(size_t key) = 0;
  virtual void __setitem__(size_t key, double val) = 0;
};

class MemoryBuffer : public Buffer {
public:
  MemoryBuffer(size_t size);
  virtual ~MemoryBuffer() override;
  virtual double __getitem__(size_t key) override;
  virtual void __setitem__(size_t key, double val) override;

private:
  size_t size_;
  double *buf_;
};

class StorageBuffer : public Buffer {
public:
  StorageBuffer(size_t size);
  virtual ~StorageBuffer() override;
  virtual double __getitem__(size_t key) override;
  virtual void __setitem__(size_t key, double val) override;

private:
  size_t size_;
  double *buf_;
};

} // namespace snuqs

#endif //__BUFFER_H__
