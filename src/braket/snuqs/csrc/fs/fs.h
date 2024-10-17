#ifndef _FS_H_
#define _FS_H_
#include <cstddef>
#include <string>
#include <vector>

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
  void dump() const;

 private:
  size_t count_;
  size_t blk_count_;
  std::vector<std::string> path_;
  std::vector<size_t> fds_;
  std::vector<fs_addr_t> free_list_;
  void* ptr_;
};
#endif  //_FS_H_
