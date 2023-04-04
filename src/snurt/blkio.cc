#include "blkio.h"

#include <fstream>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <atomic>
#include <libaio.h>

#include "logger.h"

namespace snurt {
namespace blkio {

// FIXME
std::unique_ptr<IOHandle> CreateIOHandle(const char *config_file)
{
  return std::make_unique<IOHandle>(config_file);
}

IOHandle::IOHandle(const char *config_file, size_t _block_size)
: block_size(_block_size)
{
  this->devices.clear();

  this->config_file = std::string(config_file);
  std::ifstream fs(config_file);
  if (!fs.is_open()) {
    throw std::domain_error("Cannot open the configuration file");
  }

  std::string line;

  // TODO
  this->devices.reserve(2);
  while (std::getline(fs, line)) {
    this->devices.emplace_back(line.c_str());
  }
}

static std::shared_ptr<DeviceContext> GetDeviceContext(size_t num_devices, void (*callback)(void*), void *user_data)
{
  std::shared_ptr<DeviceContext> ctx = std::make_shared<DeviceContext>();
  ctx->var = num_devices;
  ctx->callback = callback;
  ctx->user_data = user_data;
  return ctx;
}

static int get_device_idx(uint64_t off, size_t block_size, size_t num_devices) {
  return (off / block_size) % num_devices;
}

static uint64_t get_device_offset(uint64_t off, size_t block_size, size_t num_devices) {
  return (off / (block_size * num_devices)) * block_size + (off % block_size);
}

static size_t get_device_nbytes(uint64_t off, size_t block_size, size_t count) {
  return std::min(block_size - (off % block_size), count);
}

int IOHandle::Read(void *buf, uint64_t off, size_t count, void (callback)(void*), void *user_data) {
	size_t num_devices = devices.size();
	while (count > 0) {
	  int didx = get_device_idx(off, block_size, num_devices);
	  uint64_t doff = get_device_offset(off, block_size, num_devices);
	  size_t nbytes = get_device_nbytes(off, block_size, count);
	  devices[didx].EnqueueRead(buf, doff, nbytes);
	  buf = reinterpret_cast<void*>(reinterpret_cast<std::uintptr_t>(buf) + nbytes);
	  off += nbytes;
	  count -= nbytes;
  }

  int err;
  std::shared_ptr<DeviceContext> ctx = GetDeviceContext(devices.size(), callback, user_data);
  for (auto &device : devices) {
    err = device.Submit(ctx);
    if (err) {
      return err;
    }
  }

  return 0;
}

int IOHandle::Write(void *buf, uint64_t off, size_t count, void (callback)(void*), void *user_data) {
	size_t num_devices = devices.size();
	while (count > 0) {
	  int didx = get_device_idx(off, block_size, num_devices);
	  uint64_t doff = get_device_offset(off, block_size, num_devices);
	  size_t nbytes = get_device_nbytes(off, block_size, count);
	  devices[didx].EnqueueWrite(buf, doff, nbytes);
	  buf = reinterpret_cast<void*>(reinterpret_cast<std::uintptr_t>(buf) + nbytes);
	  off += nbytes;
	  count -= nbytes;
  }

  int err;
  std::shared_ptr<DeviceContext> ctx = GetDeviceContext(devices.size(), callback, user_data);
  for (auto &device : devices) {
    err = device.Submit(ctx);
    if (err) {
      return err;
    }
  }

  return 0;
}

//
// DeviceHandle
//
static void SubmitLoop(void *arg)
{
  DeviceHandle *device = reinterpret_cast<DeviceHandle*>(arg);

  int err;
  while (device->running) {
    // FIXME: HERE
    std::unique_lock<std::mutex> lk(device->mutex);
    ((device->cv)).wait(lk, [&] { return !device->running || device->iocbs.size() > 0; });
    if (!device->running) {
      break;
    }
    size_t num_entries = device->iocbs.size();
    err = device->Submit(num_entries);
    if (err) {
      throw std::runtime_error("SubmitAll Failed");
    }

    err = device->Wait(num_entries);
    if (err) {
      throw std::runtime_error("WaitAll Failed");
    }

    auto &ctx = device->contexts.front();
    device->contexts.pop_front();
    if (std::atomic_fetch_sub(&ctx->var, (size_t)1) == 1) {
      ctx->callback(ctx->user_data);
    }
  }
}

DeviceHandle::DeviceHandle(const char *fname)
  :
  ctx(kNumEvents),
  running(true)
{
  name = std::string(fname);
  fd = open(fname, O_RDWR | O_DIRECT);
  if (fd == -1) {
    throw std::domain_error("Cannot open the given file");
  }


  thread = std::thread(SubmitLoop, this);
  thread.detach();
}

DeviceHandle::DeviceHandle(DeviceHandle &&other)
{
  if (this != &other) {
    this->name = std::move(other.name);
    this->fd = std::move(other.fd);
    this->size_in_GB = std::move(other.size_in_GB);

    this->ctx = std::move(other.ctx);
    this->iocbs = std::move(other.iocbs);
    this->io_event = std::move(other.io_event);

    this->contexts = std::move(other.contexts);

    //this->running = other.running;
    this->thread = std::move(other.thread);
  }
}

DeviceHandle::~DeviceHandle() {
  running = false;
  ((this->cv)).notify_all();
}

void DeviceHandle::EnqueueWrite(void *buf, uint64_t off, size_t nbytes) {
  IOControlBlock cb;
  IOPrepPwrite(&cb, fd, buf, nbytes, off);
  iocbs.push_back(cb);
}

void DeviceHandle::EnqueueRead(void *buf, uint64_t off, size_t nbytes) {
  IOControlBlock cb;
  IOPrepPread(&cb, fd, buf, nbytes, off);
  iocbs.push_back(cb);
}

int DeviceHandle::Submit(std::shared_ptr<DeviceContext> ctx) {
  {
    std::lock_guard<std::mutex> lk(this->mutex);
    this->contexts.push_back(ctx);
  }
  ((this->cv)).notify_all();
  return 0;
}

int DeviceHandle::Submit(size_t num_entries) {
  int err;

  int i = 0;
  while (num_entries > 0) {
    IOControlBlock *iocbp = reinterpret_cast<IOControlBlock*>(&iocbs[i]);
    err = IOSubmitOne(ctx, iocbp);
    if (err != 1) {
      //TODO:: Logging
      return -EAGAIN;
    }
    num_entries--;
    i++;
  }
  return 0;
}

int DeviceHandle::Wait(size_t num_entries) {
  int nevents = 0;
  IOEvent io_event;
  while (num_entries > 0) {
    nevents = IOGetEventsOne(ctx, &io_event, NULL);
    if (nevents < 0) {
      //TODO: Logging
      return -EFAULT;
    }
    for (int i = 0; i < nevents; ++i) {
      int res = io_event.Result();
      if (res < 0) {
        return -EAGAIN;
      } 
    }
    num_entries -= nevents;
  }
  iocbs.clear();
  return 0;
}

} // namespace blkio
} // namespace snurt
