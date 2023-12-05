#include "buffer/mt_raid.h"
#include <cassert>
#include <tuple>

extern "C" {
#include <cstring>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
};

namespace snuqs {

inline static void *align_up(void *addr, uint64_t up) {
  uint64_t base = reinterpret_cast<uint64_t>(addr);
  return reinterpret_cast<void *>((base + up - 1) / up * up);
}

MTRaid::MTRaid(std::vector<std::string> paths) : paths_(paths) {
  if (paths.size() == 0)
    throw "Paths of files must be provided";

  for (int i = 0; i < paths_.size(); ++i) {
    int fd = open(paths_[i].c_str(), O_RDWR | O_DIRECT);
    if (fd == -1) {
      throw "Cannot open the given file";
    }
    fds_.push_back(fd);
  }
}

MTRaid::~MTRaid() {
  for (int i = 0; i < fds_.size(); ++i) {
    close(fds_[i]);
  }
}

int MTRaid::alloc(void **addr_p, uint64_t size) {
  if (size == 0)
    return 0;

  const std::lock_guard<std::mutex> lock(mutex_);
  *addr_p = this->base;
  this->base = align_up(this->base, MTRaid::BLOCK_SIZE);
  return 0;
}

int MTRaid::free(void *addr) {
  /* Do nothing */
  return 0;
}

int MTRaid::write(void *dst, void *src, uint64_t size) {
  if (size == 0)
    return 0;

  uint64_t num_devices = fds_.size();
  uint64_t row_size = BLOCK_SIZE * num_devices;

  std::vector<std::vector<std::tuple<uint64_t, uint64_t, uint64_t>>>
      transactions(num_devices);
  uint64_t offset =
      reinterpret_cast<uint64_t>(dst) - reinterpret_cast<uint64_t>(this->base);
  uint64_t from = reinterpret_cast<uint64_t>(src);

  while (size > 0) {
    int dev = (offset / BLOCK_SIZE) % num_devices;
    uint64_t dev_off = (offset / row_size) * BLOCK_SIZE + offset % BLOCK_SIZE;
    uint64_t to_send = std::min(BLOCK_SIZE - offset % BLOCK_SIZE, size);

    transactions[dev].emplace_back(std::make_tuple(dev_off, to_send, from));

    size -= to_send;
    offset += to_send;
    from += to_send;
  }

#pragma omp parallel for num_threads(num_devices)
  for (int d = 0; d < num_devices; ++d) {
    for (int i = 0; i < transactions[d].size(); ++i) {
      uint64_t offset = std::get<0>(transactions[d][i]);
      uint64_t to_send = std::get<1>(transactions[d][i]);
      void *from = reinterpret_cast<void *>(std::get<2>(transactions[d][i]));

      while (to_send > 0) {
        ssize_t sent = pwrite(fds_[d], from, to_send, offset);
        if (sent == -1 && errno != EAGAIN) {
          printf("Cannot be here");
          assert(false);
          break;
        } else {
          to_send -= sent;
        }
      }
    }
  }
  return 0;
}

int MTRaid::read(void *dst, void *src, uint64_t size) {
  uint64_t num_devices = fds_.size();
  uint64_t row_size = BLOCK_SIZE * num_devices;

  std::vector<std::vector<std::tuple<uint64_t, uint64_t, uint64_t>>>
      transactions(num_devices);
  uint64_t offset =
      reinterpret_cast<uint64_t>(src) - reinterpret_cast<uint64_t>(this->base);
  uint64_t from = reinterpret_cast<uint64_t>(dst);

  while (size > 0) {
    int dev = (offset / BLOCK_SIZE) % num_devices;
    uint64_t dev_off = (offset / row_size) * BLOCK_SIZE + offset % BLOCK_SIZE;
    uint64_t to_send = std::min(BLOCK_SIZE - offset % BLOCK_SIZE, size);

    transactions[dev].emplace_back(std::make_tuple(dev_off, to_send, from));

    size -= to_send;
    offset += to_send;
    from += to_send;
  }

#pragma omp parallel for num_threads(num_devices)
  for (int d = 0; d < num_devices; ++d) {
    for (int i = 0; i < transactions[d].size(); ++i) {
      uint64_t offset = std::get<0>(transactions[d][i]);
      uint64_t to_send = std::get<1>(transactions[d][i]);
      void *from = reinterpret_cast<void *>(std::get<2>(transactions[d][i]));

      while (to_send > 0) {
        ssize_t sent = pread(fds_[d], from, to_send, offset);
        if (sent == -1 && errno != EAGAIN) {
          printf("Cannot be here");
          break;
        } else {
          to_send -= sent;
        }
      }
    }
  }
  return 0;
}

} // namespace snuqs
