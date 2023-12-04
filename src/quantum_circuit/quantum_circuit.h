#ifndef __CIRCUIT_H__
#define __CIRCUIT_H__

#include "quantum_circuit/creg.h"
#include "quantum_circuit/qop.h"
#include "quantum_circuit/qreg.h"
#include "types.h"

#include <vector>

namespace snuqs {

class QuantumCircuit {
public:
  QuantumCircuit(qidx num_qubits, qidx num_bits);

  Qreg &qreg();
  Creg &creg();
  std::vector<Qop> &qops();

private:
  Qreg qreg_;
  Creg creg_;
  std::vector<Qop> qops_;
};
} // namespace snuqs

#endif //__CIRCUIT_H__
