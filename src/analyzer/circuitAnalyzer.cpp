#include "circuitAnalyzer.hpp"
#include <iostream>


#include <map>
#include <set>
#include <string>

namespace snuqs {

struct EntangleContext {
	std::set<unsigned int> target;
	std::set<unsigned int> control;
};

void CircuitAnalyzer::analyze(const QuantumCircuit &circ) {

	std::cout << "number of qubits: " << circ.num_qubits() << "\n";

	//
	// Entanglement Analysis
	//
	std::vector<std::set<unsigned int>> entanglements(circ.num_qubits());
	for (unsigned int i = 0; i < circ.num_qubits(); ++i) {
		std::set<unsigned int> s;
		s.insert(i);
		entanglements[i] = s;
	}

	for (auto g : circ.gates()) {
		std::set<unsigned int> new_set;
		for (auto q : g->qubits()) {
			for (auto p : entanglements[q]) {
				new_set.insert(p);
			}
		}
		for (auto q : g->qubits()) {
			entanglements[q] = new_set;
		}
	}
	std::cout << "Entanglement\n";
	for (unsigned int i = 0; i < circ.num_qubits(); ++i) {
		std::cout << "qubit " << i << ": ";
		for (auto q : entanglements[i]) {
			std::cout << q << " ";
		}
		std::cout << "\n";
	}

	//
	// Resource
	//
	std::map<std::string, unsigned int> name_cnt_map;
	for (auto g : circ.gates()) {
		if (name_cnt_map.find(g->name()) == name_cnt_map.end()) {
			name_cnt_map.insert(name_cnt_map.begin(), {g->name(), 0});
		}
		name_cnt_map[g->name()]++;
	}

	std::cout << "Gate counts\n";
	for (auto &[n, c]: name_cnt_map) {
		std::cout << n << ": " << c << "\n";
	}

	//
	// Timing Analysis
	//
	std::vector<unsigned int> timestamp(circ.num_qubits());
	for (unsigned int i = 0; i < circ.num_qubits(); ++i) {
		timestamp[i] = 0;
	}

	for (auto g : circ.gates()) {
		unsigned int lt = 0;
		for (auto q : g->qubits()) {
			lt = std::max(timestamp[q], lt);
		}
		for (auto q : g->qubits()) {
			timestamp[q] = lt+1;
		}
	}
	std::cout << "Number of cycles required: " ;
	for (unsigned int i = 0; i < circ.num_qubits(); ++i) {
		std::cout << "qubits " <<  i << ": " << timestamp[i] << "\n";
	}
}

} // namespace snuqs 
