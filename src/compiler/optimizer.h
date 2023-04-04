#pragma once

#include <vector>
#include "circuit/quantum_circuit.h"

namespace snuqs {

class Optimizer {
	private:
	std::vector<QuantumCircuit> circs_;

	public:
	void Optimize(const QuantumCircuit &circ);
	const std::vector<QuantumCircuit>& GetQuantumCircuits();
};

} // namespace snuqs
