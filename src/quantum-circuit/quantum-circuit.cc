#include "quantum-circuit/quantum-circuit.h"

namespace snuqs {

QuantumCircuit::QuantumCircuit(qidx num_qubits, qidx num_bits)
    : qreg_(num_qubits), creg_(num_bits) {}

QuantumRegister &QuantumCircuit::qreg() { return qreg_; }
ClassicalRegister &QuantumCircuit::creg() { return creg_; }
std::vector<QuantumOperation> &QuantumCircuit::qops() { return qops_; }

} // namespace snuqs
