#include "rt_soft_raid0.h"
#include <tuple>
#include <cassert>

extern "C" {
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
};

namespace snuqs {
namespace rt {

inline static addr_t align_up(addr_t addr, uint64_t up) {
  uint64_t base = reinterpret_cast<uint64_t>(addr);
  return reinterpret_cast<addr_t>((base + up - 1) / up * up);
}

soft_raid0_t::soft_raid0_t(std::vector<std::string> _paths)
: paths(_paths) {
  if (_paths.size() == 0)
    throw "Paths of files must be provided";

  for (int i = 0; i < paths.size(); ++i) {
    int fd = open(paths[i].c_str(), O_RDWR | O_DIRECT);
    if (fd == -1) {
      throw "Cannot open the given file";
    }
    fds.push_back(fd);
  }
}

soft_raid0_t::~soft_raid0_t() {
  for (int i = 0; i < fds.size(); ++i) {
    close(fds[i]);
  }
}

RuntimeError soft_raid0_t::alloc(addr_t *addr_p, uint64_t size) {
  if (size == 0)
    return RT_ZERO_ALLOCATION;

  const std::lock_guard<std::mutex> lock(this->mutex);
  *addr_p = this->base;
  this->base = align_up(this->base, soft_raid0_t::BLOCK_SIZE);
  return RT_SUCCESS;
}

RuntimeError soft_raid0_t::free(addr_t addr) {
  /* Do nothing */
  return RT_SUCCESS;
}

RuntimeError soft_raid0_t::write(addr_t dst, addr_t src, uint64_t size) {
  if (size == 0) 
    return RT_ZERO_IO_REQUEST;

  uint64_t num_devices = fds.size();
  uint64_t row_size = BLOCK_SIZE * num_devices;

  std::vector<std::vector<std::tuple<uint64_t, uint64_t, uint64_t>>> transactions(num_devices);
  uint64_t offset = reinterpret_cast<uint64_t>(dst) - reinterpret_cast<uint64_t>(this->base);
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
        void* from = reinterpret_cast<void*>(std::get<2>(transactions[d][i]));

        while (to_send > 0) {
          ssize_t sent = pwrite(fds[d], from, to_send, offset);
          if (sent == -1 && errno != EAGAIN) {
            assert(false);
            break;
          } else {
            to_send -= sent;
          }
        }
    }
  }
  return RT_SUCCESS;
}

RuntimeError soft_raid0_t::read(addr_t dst, addr_t src, uint64_t size) {
  uint64_t num_devices = fds.size();
  uint64_t row_size = BLOCK_SIZE * num_devices;

  std::vector<std::vector<std::tuple<uint64_t, uint64_t, uint64_t>>> transactions(num_devices);
  uint64_t offset = reinterpret_cast<uint64_t>(src) - reinterpret_cast<uint64_t>(this->base);
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
        void* from = reinterpret_cast<void*>(std::get<2>(transactions[d][i]));

        while (to_send > 0) {
          ssize_t sent = pread(fds[d], from, to_send, offset);
          if (sent == -1 && errno != EAGAIN) {
            assert(false);
            break;
          } else {
            to_send -= sent;
          }
        }
    }
  }
  return RT_SUCCESS;
}

} // namespace rt
} // namespace snuqs
