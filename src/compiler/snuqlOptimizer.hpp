#pragma once

#include <vector>
#include "quantumCircuit.hpp"

namespace snuqs {

class SnuQLOptimizer {
	private:
	std::vector<QuantumCircuit> circs_;

	public:
	void optimize(const QuantumCircuit &circ);
	const std::vector<snuqs::QuantumCircuit>& getQuantumCircuits();
};

} // namespace snuqs
