#pragma once 

#include "circuit/quantum_circuit.h"
#include "circuit_optimizer.h"

namespace snuqs {

class CircuitPartitioner {
	public:
	std::vector<QuantumCircuit> partition(const QuantumCircuit &circ);
};

} // namespace snuqs
