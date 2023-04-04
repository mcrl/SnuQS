#pragma once

#include <string>
#include <vector>
#include <memory>
#include <thread>
#include <mutex>
#include <atomic>

#include <deque>

#include <cstdint>
#include <condition_variable>

#include "libio.h"

namespace snurt {
namespace blkio {

constexpr size_t kMaxTransferSize = (1ul << 30);
constexpr unsigned kNumEvents = 256;

struct DeviceContext {
  void (*callback) (void*);
  void *user_data;
	std::atomic<size_t> var;
};

struct DeviceHandle {
  DeviceHandle(const char *fname);
  DeviceHandle(DeviceHandle &&other);
  ~DeviceHandle();

	void EnqueueWrite(void *buf, uint64_t off, size_t nbytes);
	void EnqueueRead(void *buf, uint64_t off, size_t nbytes);
  int Submit(size_t num_entries);
  int Wait(size_t num_entries);
  int Submit(std::shared_ptr<DeviceContext> ctx);

  std::string name;
  int fd;
  size_t size_in_GB;

  std::mutex mutex;
  std::condition_variable cv;

  IOContext ctx;
  std::deque<IOControlBlock> iocbs;
  IOEvent io_event;

  std::deque<std::shared_ptr<DeviceContext>> contexts;

  std::atomic<bool> running;
	std::thread thread;

};

struct IOHandle {
  IOHandle(const char *config_file, size_t _block_size=(1ul << 21));
	std::vector<DeviceHandle> devices;
	std::string config_file;

	int Read(void *buf, uint64_t off, size_t count, void (*callback)(void*), void* user_data);
	int Write(void *buf, uint64_t off, size_t count, void (*callback)(void*), void* user_data);

	size_t block_size;
};

std::unique_ptr<IOHandle> CreateIOHandle(const char *config_file);

} // namespace blkio
} // namespace snurt
