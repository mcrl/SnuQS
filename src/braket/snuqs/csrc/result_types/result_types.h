#ifndef _RESULT_TYPES_H_
#define _RESULT_TYPES_H_

#include <cstddef>
#include <vector>

class ResultType {
 public:
  virtual void *data() = 0;
  virtual size_t dim() const = 0;
  virtual std::vector<size_t> shape() const = 0;
};

#endif  // _RESULT_TYPES_H_
