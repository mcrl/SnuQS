#ifndef _BUFFER_STORAGE_H_
#define _BUFFER_STORAGE_H_

#include <string>
#include <vector>

#include "buffer/buffer.h"
class BufferStorage : public Buffer {
 public:
  BufferStorage(size_t count, size_t blk_count,
                const std::vector<std::string>& path);
  virtual ~BufferStorage();
  virtual std::complex<double>* buffer() override;
  virtual size_t count() const override;
  virtual size_t itemsize() const override;
  virtual std::string formatted_string() const override;

 private:
  size_t count_;
  size_t blk_count_;
  std::complex<double>* buffer_ = nullptr;
  std::vector<std::string> path_;
  std::vector<size_t> fds_;
};
#endif  //_BUFFER_STORAGE_H_
