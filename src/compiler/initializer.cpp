#include "initializer.h"
#include "circuit/gate_factory.h"

namespace snuqs {

//
// Initializer
//
QuantumCircuit Initializer::optimize(const QuantumCircuit &circ) {
	size_t cnt = 0;
	for (size_t i = 0; i < circ.num_qubits(); i++) {
		for (auto &&g : circ.gates()) {
			auto &qubits = g->qubits();
			if (std::find(qubits.begin(), qubits.end(), i) != qubits.end()) {
				if (g->type() == Gate::Type::H)
					cnt++;
				break;
			}
		}
	}

	QuantumCircuit opt_circ;
	opt_circ.set_num_qubits(circ.num_qubits());
	if (cnt == circ.num_qubits()) {
		std::vector<size_t> qubits(circ.num_qubits());
		for (size_t i = 0; i < circ.num_qubits(); i++) {
			qubits[i] = i;
		}
		opt_circ.addGate(GateFactory::CreateGate("uniform", std::move(qubits), {}));
		std::vector<bool> bitmap(circ.num_qubits());

		for (size_t i = 0; i < circ.num_qubits(); i++) {
			bitmap[i] = false;
		}

		for (auto &&g : circ.gates()) {
			if (g->type() == Gate::Type::H) {
				size_t q = g->qubits()[0];
				if (!bitmap[q]) {
					bitmap[q] = true;
				} else {
					opt_circ.addGate(g);
				}
			} else {
				opt_circ.addGate(g);
			}
		}
	} else {
		opt_circ.addGate(GateFactory::CreateGate("zerostate", {}, {}));
		opt_circ = circ;
	}
	return opt_circ;
}

} // namespace snuqs

