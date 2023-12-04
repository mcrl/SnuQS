#ifndef __BUFFER_H__
#define __BUFFER_H__

#include <cstddef>

namespace snuqs {

class Buffer {
public:
  virtual ~Buffer() = 0;
  virtual void allocate(size_t size) = 0;
};

class MemoryBuffer : public Buffer {
public:
  virtual ~MemoryBuffer();
  virtual void allocate(size_t size);
};

class StorageBuffer : public Buffer {
public:
  virtual ~StorageBuffer();
  virtual void allocate(size_t size);
};

} // namespace snuqs

#endif //__BUFFER_H__
