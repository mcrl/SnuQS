#include "quantum_circuit/quantum_circuit.h"

namespace snuqs {

QuantumCircuit::QuantumCircuit(qidx num_qubits, qidx num_bits)
    : qreg_(num_qubits), creg_(num_bits) {}

Qreg &QuantumCircuit::qreg() { return qreg_; }
Creg &QuantumCircuit::creg() { return creg_; }
std::vector<Qop> &QuantumCircuit::qops() { return qops_; }

} // namespace snuqs
