#pragma once 

#include "quantumCircuit.hpp"
#include "circuitOptimizer.hpp"

namespace snuqs {

class CircuitPartitioner {
	public:
	std::vector<QuantumCircuit> partition(const QuantumCircuit &circ);
};

} // namespace snuqs
