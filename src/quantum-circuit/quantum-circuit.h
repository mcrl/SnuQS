#ifndef __QUANTUM_CIRCUIT_H__
#define __QUANTUM_CIRCUIT_H__

#include "misc/types.h"
#include "quantum-circuit/classical-register.h"
#include "quantum-circuit/quantum-operation.h"
#include "quantum-circuit/quantum-register.h"

#include <vector>

namespace snuqs {

class QuantumCircuit {
public:
  QuantumCircuit(qidx num_qubits, qidx num_bits);

  QuantumRegister& qreg();
  ClassicalRegister& creg();
  std::vector<QuantumOperation>& qops();

private:
  QuantumRegister qreg_;
  ClassicalRegister creg_;
  std::vector<QuantumOperation> qops_;
};
} // namespace snuqs

#endif //__QUANTUM_CIRCUIT_H__
