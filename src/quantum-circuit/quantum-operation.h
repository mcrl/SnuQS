#ifndef __QUANTUM_OPERATION_H__
#define __QUANTUM_OPERATION_H__

#include "misc/types.h"
#include <vector>

namespace snuqs {

class QuantumOperation {
public:
  QuantumOperation(std::vector<qidx> qubits);
  QuantumOperation(std::vector<qidx> qubits, std::vector<double> params);
  virtual ~QuantumOperation();

protected:
  std::vector<qidx> qubits_;
  std::vector<double> params_;
};

class UGate : public QuantumOperation {
public:
  UGate(qidx target, double theta, double phi, double lambda);
};

class CXGate : public QuantumOperation {
public:
  CXGate(qidx control, qidx target);
};

class Measurement : public QuantumOperation {
  Measurement(qidx target);
};

class Reset : public QuantumOperation {
  Reset(qidx target);
};

} // namespace snuqs

#endif //__QUANTUM_OPERATION_H__
