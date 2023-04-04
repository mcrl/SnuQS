#pragma once

#include "addr.h"
#include "command_queue.h"

namespace snurt {

int Init();
void Deinit();

int GetDeviceCount();
addr_t Malloc(size_t count);
addr_t MallocAligned(size_t count, size_t align);
addr_t MallocHost(size_t count);
addr_t MallocDevice(size_t count, size_t devno);
addr_t MallocIO(size_t count);

int MemcpyH2S(CommandQueue &queue, addr_t dst, addr_t src, size_t count);
int MemcpyS2H(CommandQueue &queue, addr_t dst, addr_t src, size_t count);
int MemcpyH2D(CommandQueue &queue, addr_t dst, addr_t src, size_t count);
int MemcpyD2H(CommandQueue &queue, addr_t dst, addr_t src, size_t count);
int MemcpyS2D(CommandQueue &queue, addr_t dst, addr_t src, size_t count);
int MemcpyD2S(CommandQueue &queue, addr_t dst, addr_t src, size_t count);
int Synchronize(CommandQueue &queue);

} // namespace snurt
