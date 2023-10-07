#include "quantum-operation.h"
#include <assert.h>

namespace snuqs {

//
// QuantumOperation
//
QuantumOperation::QuantumOperation(std::vector<qidx> qubits)
    : qubits_(qubits) {}
QuantumOperation::QuantumOperation(std::vector<qidx> qubits,
                                   std::vector<double> params)
    : qubits_(qubits), params_(params) {}

//
// UGate
//
UGate::UGate(qidx target, double theta, double phi, double lambda)
    : QuantumOperation({target}, {theta, phi, lambda}) {}

//
// CXGate
//
CXGate::CXGate(qidx control, qidx target)
    : QuantumOperation({control, target}) {}

//
//
// Measurement
//
Measurement::Measurement(qidx target) : QuantumOperation({target}) {}

//
// Reset
//
Reset::Reset(qidx target) : QuantumOperation({target}) {}

} // namespace snuqs
