#ifndef _BUFFER_H_
#define _BUFFER_H_

#include <cassert>
#include <complex>
#include <cstdlib>
#include <string>

class Buffer {
 public:
  virtual std::complex<double>* buffer() = 0;
  virtual size_t count() const = 0;
  virtual size_t itemsize() const = 0;
  virtual std::string formatted_string() const = 0;
};

#endif  //_BUFFER_H_
