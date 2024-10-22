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

#include "buffer/buffer_cpu.h"
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

void memcpyH2H(void *dst, void *src, size_t count,
               std::shared_ptr<Stream> stream) {
  memcpy(dst, src, count);
}

void memcpyH2D(void *dst, void *src, size_t count,
               std::shared_ptr<Stream> stream) {
  CUDA_CHECK(cudaMemcpyAsync(dst, src, count, cudaMemcpyHostToDevice,
                             reinterpret_cast<cudaStream_t>(
                                 stream == nullptr ? nullptr : stream->get())));
}

void memcpyD2H(void *dst, void *src, size_t count,
               std::shared_ptr<Stream> stream) {
  CUDA_CHECK(cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToHost,
                             reinterpret_cast<cudaStream_t>(
                                 stream == nullptr ? nullptr : stream->get())));
}

void memcpyD2D(void *dst, void *src, size_t count,
               std::shared_ptr<Stream> stream) {
  CUDA_CHECK(cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToDevice,
                             reinterpret_cast<cudaStream_t>(
                                 stream == nullptr ? nullptr : stream->get())));
}

void memcpyS2H(void *dst, fs_addr_t src, size_t count,
               std::shared_ptr<Stream> stream) {
  spdlog::info("memcpyS2H({}, {}, {})", dst, src.start, count);
  std::shared_ptr<FS> fs = get_fs();
  fs->read(src, dst, count, stream);
}

void memcpyH2S(fs_addr_t dst, void *src, size_t count,
               std::shared_ptr<Stream> stream) {
  spdlog::info("memcpyH2S({}, {}, {})", dst.start, src, count);
  std::shared_ptr<FS> fs = get_fs();
  fs->write(dst, src, count, stream);
}

void memcpyD2S(fs_addr_t dst, void *src, size_t count,
               std::shared_ptr<Stream> stream) {
  auto buf_cpu = std::make_shared<BufferCPU>(count, true);  // pinned
  spdlog::info("Allocating auxiliary pinned CPU buffer of {} bytes", count);
  memcpyD2H(buf_cpu->ptr(), src, count, stream);
  memcpyH2S(dst, buf_cpu->ptr(), count, stream);
}

void memcpyS2D(void *dst, fs_addr_t src, size_t count,
               std::shared_ptr<Stream> stream) {
  auto buf_cpu = std::make_shared<BufferCPU>(count, true);  // pinned
  spdlog::info("Allocating auxiliary pinned CPU buffer of {} bytes", count);
  memcpyS2H(buf_cpu->ptr(), src, count, stream);
  memcpyH2D(dst, buf_cpu->ptr(), count, stream);
}

void memcpyS2S(fs_addr_t dst, fs_addr_t src, size_t count,
               std::shared_ptr<Stream> stream) {
  auto buf_cpu = std::make_shared<BufferCPU>(count, true);  // pinned
  spdlog::info("Allocating auxiliary pinned CPU buffer of {} bytes", count);
  memcpyS2H(buf_cpu->ptr(), src, count, stream);
  memcpyH2S(dst, buf_cpu->ptr(), count, stream);
}
