#include "optimizer.h"

#include "circuit_optimizer.h"
#include "circuit_partitioner.h"


namespace snuqs {

void Optimizer::Optimize(const QuantumCircuit &circ) {
	CircuitPartitioner part;
	circs_ = part.partition(circ);

	CircuitOptimizer opt;
	for (auto && circ : circs_) {
		circ = opt.optimize(circ);
	}
}

const std::vector<QuantumCircuit>& Optimizer::GetQuantumCircuits() {
	return circs_;
}

} // namespace snuqs
