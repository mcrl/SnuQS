#ifndef _BUFFER_STORAGE_H_
#define _BUFFER_STORAGE_H_

#include <memory>
#include <string>
#include <vector>

#include "buffer/buffer.h"
#include "fs/fs.h"

class BufferStorage : public Buffer {
 public:
  BufferStorage(size_t count);
  virtual ~BufferStorage();
  virtual void* ptr() override;
  virtual size_t count() const override;
  virtual std::string formatted_string() const override;
  fs_addr_t addr();
  virtual std::shared_ptr<Buffer> cpu() override;
  virtual std::shared_ptr<Buffer> cuda() override;
  virtual std::shared_ptr<Buffer> storage() override;

 private:
  size_t count_;
  std::shared_ptr<FS> fs_;
  fs_addr_t fs_addr_;
};
#endif  //_BUFFER_STORAGE_H_
