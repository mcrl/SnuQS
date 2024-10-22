#ifndef _FS_H_
#define _FS_H_
#include <cstddef>
#include <string>
#include <vector>

#include "stream/stream.h"

typedef struct {
  size_t start;
  size_t end;
  void* ptr;
  std::string formatted_string() const;
} fs_addr_t;

class FS {
 public:
  FS(size_t count, size_t blk_count, const std::vector<std::string>& path);
  ~FS();
  fs_addr_t alloc(size_t count);
  void free(fs_addr_t addr);
  void read(fs_addr_t addr, void* buf, size_t count, size_t offset,
            std::shared_ptr<Stream> stream);
  void write(fs_addr_t addr, void* buf, size_t count, size_t offset,
             std::shared_ptr<Stream> stream);
  void dump() const;
  std::pair<int, size_t> get_offset(size_t pos) const;

 private:
  size_t row_size_;
  size_t blk_count_;
  size_t count_;
  std::vector<std::string> path_;
  std::vector<size_t> fds_;
  std::vector<fs_addr_t> free_list_;
  void* ptr_;
};
#endif  //_FS_H_
