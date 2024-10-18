#ifndef _RUNTIME_H_
#define _RUNTIME_H_

#include <cstddef>
#include <memory>

#include "fs/fs.h"

void attach_fs(size_t count, size_t blk_count,
               const std::vector<std::string> &path);
bool is_attached_fs();
void detach_fs();
std::shared_ptr<FS> get_fs();

void memcpyH2H(void *dst, void *src, size_t count);
void memcpyH2D(void *dst, void *src, size_t count);
void memcpyD2H(void *dst, void *src, size_t count);
void memcpyD2D(void *dst, void *src, size_t count);

void memcpyS2H(void *dst, fs_addr_t src, size_t count);
void memcpyH2S(fs_addr_t dst, void *src, size_t count);

std::pair<size_t, size_t> mem_info();

#endif  //_RUNTIME_H_
