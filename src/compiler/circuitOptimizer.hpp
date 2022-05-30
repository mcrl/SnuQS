#pragma once

#include "quantumCircuit.hpp"

namespace snuqs {

class OptimizationPass {
	public:
	virtual QuantumCircuit optimize(const QuantumCircuit &circ) = 0;
};

class CircuitOptimizer {
	private:
	std::vector<std::shared_ptr<OptimizationPass>> passes_;

	public:
	CircuitOptimizer();
	QuantumCircuit optimize(const QuantumCircuit &circ);
};

} //namespace snuqs
