#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "core/runtime.h"

#include <cuda_runtime.h>
#include <spdlog/spdlog.h>
#include <stdio.h>
#include <sys/sysinfo.h>
#include <unistd.h>

#include <cassert>
#include <cstring>

#include "utils_cuda.h"

static std::shared_ptr<FS> fs_ = nullptr;
void attach_fs(size_t count, size_t blk_count,
               const std::vector<std::string> &path) {
  assert(fs_ == nullptr);
  spdlog::info("Attaching FS, count: {}, blk_count: {}", count, blk_count);
  for (auto &p : path) {
    spdlog::info("\t{}", p);
  }
  fs_ = std::make_shared<FS>(count, blk_count, path);
}
bool is_attached_fs() { return fs_ != nullptr; }
void detach_fs() {
  assert(fs_ != nullptr);
  spdlog::info("Detaching FS");
  fs_ = nullptr;
}
std::shared_ptr<FS> get_fs() { return fs_; }

std::pair<size_t, size_t> mem_info() {
  /* PAGESIZE is POSIX: http://pubs.opengroup.org/onlinepubs/9699919799/
   * but PHYS_PAGES and AVPHYS_PAGES are glibc extensions. I bet those are
   * parsed from /proc/meminfo. */
  size_t free, total;
  total = get_phys_pages() * sysconf(_SC_PAGESIZE);
  free = get_avphys_pages() * sysconf(_SC_PAGESIZE);
  return {free, total};
}

void memcpyH2H(void *dst, void *src, size_t count) { memcpy(dst, src, count); }

void memcpyH2D(void *dst, void *src, size_t count) {
  CUDA_CHECK(cudaMemcpy(dst, src, count, cudaMemcpyHostToDevice));
}

void memcpyD2H(void *dst, void *src, size_t count) {
  CUDA_CHECK(cudaMemcpy(dst, src, count, cudaMemcpyDeviceToHost));
}

void memcpyD2D(void *dst, void *src, size_t count) {
  CUDA_CHECK(cudaMemcpy(dst, src, count, cudaMemcpyDeviceToDevice));
}
