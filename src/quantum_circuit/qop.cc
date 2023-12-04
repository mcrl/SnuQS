#include "qop.h"
#include <assert.h>

namespace snuqs {

//
// Qop
//
Qop::Qop(std::vector<qidx> qubits) : qubits_(qubits) {}
Qop::Qop(std::vector<qidx> qubits, std::vector<double> params)
    : qubits_(qubits), params_(params) {}

//
// UGate
//
UGate::UGate(qidx target, double theta, double phi, double lambda)
    : Qop({target}, {theta, phi, lambda}) {}

//
// CXGate
//
CXGate::CXGate(qidx control, qidx target) : Qop({control, target}) {}

//
//
// Measurement
//
Measurement::Measurement(qidx target) : Qop({target}) {}

//
// Reset
//
Reset::Reset(qidx target) : Qop({target}) {}

} // namespace snuqs
