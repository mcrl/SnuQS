#ifndef _BUFFER_H_
#define _BUFFER_H_

#include <cassert>
#include <cstdlib>
#include <memory>
#include <string>

class Buffer : public std::enable_shared_from_this<Buffer> {
 public:
  virtual size_t count() const = 0;
  virtual std::string formatted_string() const = 0;
  virtual void* ptr() = 0;
  virtual std::shared_ptr<Buffer> cpu() = 0;
  virtual std::shared_ptr<Buffer> cuda() = 0;
  virtual std::shared_ptr<Buffer> storage() = 0;
};

#endif  //_BUFFER_H_
