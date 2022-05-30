#include "snuqlOptimizer.hpp"
#include "circuitOptimizer.hpp"
#include "circuitPartitioner.hpp"


namespace snuqs {

void SnuQLOptimizer::optimize(const QuantumCircuit &circ) {
	CircuitPartitioner part;
	circs_ = part.partition(circ);

	CircuitOptimizer opt;
	for (auto && circ : circs_) {
		circ = opt.optimize(circ);
	}
}

const std::vector<snuqs::QuantumCircuit>& SnuQLOptimizer::getQuantumCircuits() {
	return circs_;
}

} // namespace snuqs
