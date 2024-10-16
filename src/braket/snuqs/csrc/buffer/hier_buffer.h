#ifndef _HIER_BUFFER_H_
#define _HIER_BUFFER_H_

#include <memory>
#include <string>
#include <vector>

#include "buffer/buffer.h"

class HierBuffer {
 public:
  HierBuffer(size_t count);                                        // CPU-Only
  HierBuffer(size_t count, size_t l1_count, size_t l1_blk_count);  // CPU + CUDA
  HierBuffer(size_t count, size_t l1_count, size_t l1_blk_count,
             const std::vector<std::string> &path);  // STORAGE + CPU
  HierBuffer(size_t count, size_t l1_count, size_t l1_blk_count,
             size_t l2_count, size_t l2_blk_count,
             const std::vector<std::string> &path);  // STORAGE + CPU + CUDA
  ~HierBuffer();

 private:
  std::vector<std::string> path_;
  std::vector<size_t> fds_;

  size_t count_ = 0;
  size_t l1_count_ = 0;
  size_t l1_blk_count_ = 0;
  size_t l2_count_ = 0;
  size_t l2_blk_count_ = 0;

  std::shared_ptr<Buffer> ptr_ = nullptr;
  std::shared_ptr<Buffer> ptr_cuda_ = nullptr;
  std::shared_ptr<Buffer> ptr_storage_ = nullptr;
};
#endif  //_HIER_BUFFER_H_
