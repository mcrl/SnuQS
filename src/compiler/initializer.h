#pragma once

#include "circuit_optimizer.h"

#include "circuit/quantum_circuit.h"

namespace snuqs {

class Initializer : public OptimizationPass {
	public:
	virtual QuantumCircuit optimize(const QuantumCircuit &circ) override;
};

} // namespace snuqs
