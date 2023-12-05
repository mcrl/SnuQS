#ifndef __BUFFER_H__
#define __BUFFER_H__

#include "buffer/mt_raid.h"
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
  MemoryBuffer(size_t count);
  virtual ~MemoryBuffer() override;
  virtual double __getitem__(size_t key) override;
  virtual void __setitem__(size_t key, double val) override;

private:
  size_t count_;
  double *buf_;
};

class StorageBuffer : public Buffer {
public:
  StorageBuffer(size_t count);
  virtual ~StorageBuffer() override;
  virtual double __getitem__(size_t key) override;
  virtual void __setitem__(size_t key, double val) override;

private:
  size_t count_;
  double *buf_;
  double *small_buf_;
  MTRaid raid_;
};

} // namespace snuqs

#endif //__BUFFER_H__
