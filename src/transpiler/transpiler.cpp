#include "transpiler.hpp"

namespace snuqs {

//
// Transpiler
//
Transpiler::~Transpiler() {
}

//
// NoTranspiler
//
NoTranspiler::~NoTranspiler() {
}

void NoTranspiler::transpile(const QuantumCircuit &circ) {
	circ_ = circ;
}

const QuantumCircuit& NoTranspiler::getQuantumCircuit() {
	return circ_;
}

//
// DensityTranspiler
//
DensityTranspiler::~DensityTranspiler() {
}

void DensityTranspiler::transpile(const QuantumCircuit &circ) {
	size_t num_qubits = circ.num_qubits();
	circ_.set_num_qubits(num_qubits*2);
	circ_.set_permutation(circ.permutation());

	circ_.clearGate();
	for (auto g : circ.gates()) {
		std::vector<size_t> qubits = g->qubits();
		for (auto &q : qubits) {
			q += num_qubits;
		}
		circ_.addGate(g);
		circ_.addGate(gateFactory(g->name(),
					qubits,
					g->params()));
	}
}

const QuantumCircuit& DensityTranspiler::getQuantumCircuit() {
	return circ_;
}

} // namespace snuqs 
