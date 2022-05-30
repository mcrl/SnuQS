#include "snuqlTranspiler.hpp"

namespace snuqs {

SnuQLTranspiler::SnuQLTranspiler(SimulationMethod method)
: method_(method),
transpiler_(nullptr)
{
	switch (method_) {
		case SimulationMethod::STATEVECTOR:
			transpiler_ = new NoTranspiler();
			break;
		case SimulationMethod::DENSITY:
			transpiler_ = new DensityTranspiler();
			break;
		case SimulationMethod::CONTRACTION:
			break;
	}
}

SnuQLTranspiler::~SnuQLTranspiler() {
	if (transpiler_) {
		delete transpiler_;
	}
}

void SnuQLTranspiler::transpile(const QuantumCircuit &circ) {
	if (transpiler_) {
		transpiler_->transpile(circ);
	} else {
		circ_ = circ;
	}
}

const QuantumCircuit& SnuQLTranspiler::getQuantumCircuit() {
	if (transpiler_) {
		return transpiler_->getQuantumCircuit();
	} else {
		return circ_;
	}
}

} // namespace snuqs
