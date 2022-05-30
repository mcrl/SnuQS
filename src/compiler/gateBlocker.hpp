#pragma once 

#include "quantumCircuit.hpp"
#include "circuitOptimizer.hpp"

namespace snuqs {

class GateBlocker : public OptimizationPass {
	public:
	virtual QuantumCircuit optimize(const QuantumCircuit &circ) override;
};

} // namespace snuqs
