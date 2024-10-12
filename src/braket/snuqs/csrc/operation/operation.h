#ifndef _OPERATION_H_
#define _OPERATION_H_

#include <complex>
#include <string>
#include <vector>

class Operation {
 public:
  Operation(const std::vector<size_t> &targets);
  ~Operation();
  virtual std::vector<size_t> get_targets() const;
  virtual void set_targets(const std::vector<size_t> &targets);
  std::vector<size_t> targets_;
};

class GateOperation : public Operation {
 public:
  GateOperation(const std::vector<size_t> &targets,
                const std::vector<size_t> &ctrl_modifiers, size_t power);
  virtual ~GateOperation();

  virtual void *data();
  void *ptr();
  void *ptr_cuda();
  virtual size_t num_elems() const;
  std::vector<size_t> ctrl_modifiers_;
  size_t power_;

  virtual size_t dim() const;
  virtual std::vector<size_t> shape() const;
  virtual std::vector<size_t> stride() const;
  virtual void slice(size_t idx);

  virtual std::string name() const;
  virtual std::string formatted_string() const;

  virtual bool symmetric() const;
  virtual bool sliceable() const;

  std::complex<double> *ptr_ = nullptr;
  std::complex<double> *ptr_cuda_ = nullptr;
  bool copied_to_cuda = false;
};

#endif  //_OPERATION_H_
