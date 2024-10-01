#ifndef _OPERATION_H_
#define _OPERATION_H_

#include <complex>
#include <vector>

class Operation {};
class GateOperation : public Operation {
 public:
  virtual void *data();
  virtual void *data_cuda();
  virtual size_t num_elems() const;
  virtual size_t dim() const = 0;
  virtual std::vector<size_t> shape() const = 0;
  virtual std::vector<size_t> stride() const = 0;
  std::complex<double> *data_ = nullptr;
  std::complex<double> *data_cuda_ = nullptr;
  bool copied_to_cuda = false;
};

#endif  //_OPERATION_H_
