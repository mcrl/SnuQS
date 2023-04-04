#pragma once

#include <cstdint>

namespace snurt {
struct addr_t {
  union {
    void *ptr;
    uint64_t off;
  };
  operator void*() const { return this->ptr; }
  operator uint64_t() const { return this->off; }
  addr_t& operator+(uint64_t offset) {
    this->off += offset;
    return *this;
  }

  addr_t& operator=(uint64_t offset) {
    this->off = offset;
    return *this;
  }

  addr_t& operator+=(uint64_t offset) {
    this->off += offset;
    return *this;
  }
};

} // namespace snurt 
