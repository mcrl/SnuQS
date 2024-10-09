#ifndef _MEMORY_H_
#define _MEMORY_H_

#include <spdlog/spdlog.h>

#include <cassert>
#include <complex>
#include <cstdlib>

class Memory {
 public:
  Memory(size_t count) {
    if (count > 0) {
      buffer_ = reinterpret_cast<std::complex<double>*>(
          malloc(sizeof(std::complex<double>) * count));
    }
    assert(buffer_ != nullptr);
  }
  ~Memory() {
    if (buffer_ != nullptr) free(buffer_);
  }
  std::complex<double>* buffer() { return buffer_; }
  size_t itemsize() const { return sizeof(std::complex<double>); }

 private:
  std::complex<double>* buffer_ = nullptr;
};

#endif  //_MEMORY_H_
