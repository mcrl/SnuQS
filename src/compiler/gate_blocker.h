#pragma once 

#include "circuit/quantum_circuit.h"
#include "circuit_optimizer.h"

namespace snuqs {

class GateBlocker : public OptimizationPass {
	public:
	virtual QuantumCircuit optimize(const QuantumCircuit &circ) override;
};

} // namespace snuqs
