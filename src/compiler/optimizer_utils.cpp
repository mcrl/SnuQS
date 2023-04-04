#include "optimizer_utils.h"

namespace snuqs {

bool lessthanOrder(Gate::shared_ptr &g, Gate::shared_ptr &h, size_t num_qubits, const std::vector<size_t> order) {
	for (auto &&i : g->qubits()) {
		for (auto &&j : h->qubits()) {
			if (order[j] <= order[i]) {
				return false;
			}
		}
	}
	return true;
}

bool lessthan(Gate::shared_ptr &g, Gate::shared_ptr &h, size_t num_qubits, size_t base) {
	for (auto &&i : g->qubits()) {
		for (auto &&j : h->qubits()) {
			if ((j+num_qubits-base) % num_qubits <= (i+num_qubits-base) % num_qubits)
				return false;
		}
	}
	return true;
}

bool greaterthan(Gate::shared_ptr &g, Gate::shared_ptr &h, size_t num_qubits, size_t base) {
	for (auto &&i : g->qubits()) {
		for (auto &&j : h->qubits()) {
			if ((j+num_qubits-base) % num_qubits >= (i+num_qubits-base) % num_qubits)
				return false;
		}
	}
	return true;
}

void sortGatesInv(std::vector<Gate::shared_ptr>::iterator start, std::vector<Gate::shared_ptr>::iterator end, size_t num_qubits, size_t base)
{
	for (auto it = start+1; it != end; ++it) {
		auto jt = it;
		while (jt > start && greaterthan(*jt, *(jt-1), num_qubits, base)) {
			std::swap(*jt, *(jt-1));
			jt--;
		}
	}
}

void sortGatesOrder(std::vector<Gate::shared_ptr>::iterator start, std::vector<Gate::shared_ptr>::iterator end, size_t num_qubits, const std::vector<size_t> &order)
{
	for (auto it = start+1; it != end; ++it) {
		auto jt = it;
		while (jt > start && lessthanOrder(*jt, *(jt-1), num_qubits, order)) {
			std::swap(*jt, *(jt-1));
			jt--;
		}
	}
}

void sortGates(std::vector<Gate::shared_ptr>::iterator start, std::vector<Gate::shared_ptr>::iterator end, size_t num_qubits, size_t base)
{
	for (auto it = start+1; it != end; ++it) {
		auto jt = it;
		while (jt > start && lessthan(*jt, *(jt-1), num_qubits, base)) {
			std::swap(*jt, *(jt-1));
			jt--;
		}
	}
}

void sortGates(QuantumCircuit &circ)
{
	auto &gates = circ.gates();
	sortGates(gates.begin(), gates.end(), circ.num_qubits());
}

} // namespace snuqs
