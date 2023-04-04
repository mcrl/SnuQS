#include "circuit_optimizer.h"
#include "initializer.h"
#include "gate_blocker.h"
#include "circuit_partitioner.h"

namespace snuqs {

//
// OptimizationPass
//
QuantumCircuit OptimizationPass::optimize(const QuantumCircuit &circ) {
	return circ;
}

//
// Optimizer
//
CircuitOptimizer::CircuitOptimizer()
: passes_{
#ifdef _USE_BLOCKER_
	std::make_shared<GateBlocker>()
#endif
	}
{
}

QuantumCircuit CircuitOptimizer::optimize(const QuantumCircuit &circ) {

	std::vector<QuantumCircuit> circs(1);
	QuantumCircuit opt_circ = circ;
	for (auto &&pass : passes_) {
		opt_circ = pass->optimize(opt_circ);
	}

	return opt_circ;
};

} //namespace snuqs
