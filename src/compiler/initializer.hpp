#pragma once

#include "quantumCircuit.hpp"
#include "circuitOptimizer.hpp"

namespace snuqs {

class Initializer : public OptimizationPass {
	public:
	virtual QuantumCircuit optimize(const QuantumCircuit &circ) override;
};

} // namespace snuqs
