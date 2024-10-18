#ifndef _OPERATION_H_
#define _OPERATION_H_

#include <complex>
#include <memory>
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

enum class GateOperationType {
  Identity,
  Hadamard,
  PauliX,
  PauliY,
  PauliZ,
  CX,
  CY,
  CZ,
  S,
  Si,
  T,
  Ti,
  V,
  Vi,
  PhaseShift,
  CPhaseShift,
  CPhaseShift00,
  CPhaseShift01,
  CPhaseShift10,
  RotX,
  RotY,
  RotZ,
  Swap,
  ISwap,
  PSwap,
  XY,
  XX,
  YY,
  ZZ,
  CCNot,
  CSwap,
  U,
  GPhase,
  SwapA2A,
};

class GateOperation : public Operation {
 public:
  GateOperation(GateOperationType type, const std::vector<size_t> &targets,
                const std::vector<double> &angles,
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

  std::string name() const;
  virtual std::string formatted_string() const;

  virtual bool diagonal() const;
  virtual bool anti_diagonal() const;
  virtual bool sliceable() const;
  virtual std::shared_ptr<GateOperation> slice(size_t idx) const;

  std::vector<double> angles_;
  GateOperationType type_;
  std::complex<double> *ptr_ = nullptr;
  std::complex<double> *ptr_cuda_ = nullptr;
  bool copied_to_cuda = false;
};

#endif  //_OPERATION_H_
