#include "simulator.hpp"
#include "tensor.hpp"

namespace snuqs {

void ContractionCPUSimulator::init(size_t num_qubits) {
}

void ContractionCPUSimulator::deinit() {
}

void ContractionCPUSimulator::run(const std::vector<QuantumCircuit> &circs) {

	size_t num_qubits = circs[0].num_qubits();

	std::vector<unsigned int> index_map(num_qubits);
	std::vector<Tensor*> tensors;


	for (auto &circ: circs) {

		unsigned int nedges = 0;
		for (size_t i = 0; i < num_qubits; ++i) {
			tensors.push_back(new Tensor({}, {nedges}, {1, 0}));
			index_map[i] = nedges++;
		}

		for (auto g : circ.gates()) {
			Tensor *t = g->toTensor();
			tensors.push_back(t);

			std::vector<unsigned int> inedges;
			std::vector<unsigned int> outedges;
			for (auto q : g->qubits()) {
				inedges.push_back(index_map[q]);
			}

			for (auto q : g->qubits()) {
				outedges.push_back(nedges);
				index_map[q] = nedges;
				nedges++;
			}

			t->setInEdges(inedges);
			t->setOutEdges(outedges);
		}


		for (auto t : tensors) {
			std::cout << *t << "\n";
		}
	}
}

} // namespace snuqs
