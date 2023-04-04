#include "circuit_partitioner.h"

#include <cassert>

#include "optimizer_utils.h"
#include "permutation.h"
#include "circuit/gate.h"
#include "circuit/gate_factory.h"

namespace snuqs {

Permutation getNextPermutation(
		QuantumCircuit::gate_iterator start,
		QuantumCircuit::gate_iterator end,
		size_t n) {
	std::vector<size_t> p;

	for (size_t i = 0; i < n; i++) {
		p.push_back(n-i-1);
	}
	return Permutation(p);
}


void mapGates(
		QuantumCircuit::gate_iterator start,
		QuantumCircuit::gate_iterator end,
		const Permutation &perm)
{
	for (auto it = start; it != end; it++) {
		auto &&gp = *it;
		std::vector<size_t> new_qubits;
		for (auto q : gp->qubits()) {
			new_qubits.push_back(perm[q]);
		}
		gp->set_qubits(new_qubits);
	}
}

std::vector<Permutation> decomposeCycles(const Permutation &perm) {
	std::vector<Permutation> cycles;
	std::vector<bool> checked(perm.size());

	for (size_t i = 0; i < perm.size(); i++) {
		checked[i] = false;
	}

	for (size_t i = 0; i < perm.size(); i++) {
		if (!checked[i]) {
			checked[i] = true;
			std::vector<size_t> cycle;
			cycle.push_back(i);

			size_t j = perm[i];
			while (j != i) {
				assert(!checked[j]);
				checked[j] = true;
				cycle.push_back(j);
				j = perm[j];
			}

			if (cycle.size() > 1) 
				cycles.emplace_back(cycle);
		}
	}
	return cycles;
}

Permutation getPermuationFromTo(const Permutation &from, const Permutation &to) {
	assert(from.size() == to.size());

	std::vector<size_t> perm(to.size());
	auto to_inv = to.inv();
	for (size_t i = 0; i < to.size(); i++) {
		perm[i] = to_inv[from[i]];
	}
	return perm;
}

QuantumCircuit::gate_iterator
insertSwapGates(
		std::vector<Gate::shared_ptr> &gates,
		QuantumCircuit::gate_iterator start,
		const Permutation &perm,
		size_t nlocal
		) {

	const std::vector<size_t> & p = perm.perm_;

	std::vector<size_t> ident(p.size());
	std::vector<size_t> inv_map(p.size());
	for (size_t i = 0; i < p.size(); i++) {
		ident[i] = i;
		inv_map[p[i]] = i;
	}
	
	std::vector<size_t> memlay;
	std::vector<size_t> blocklay;

	size_t nglobals = 0;
	for (size_t i = 0; i < nlocal; i++) {
		if (inv_map[i] >= nlocal) {
			nglobals++;
		}
	}

	memlay = ident;
	size_t j = nlocal-nglobals;
	for (size_t i = nlocal; i < p.size(); i++) {
		if (inv_map[i] < nlocal) {
			memlay[j] = inv_map[i];
			j++;
		}
	}

	j = nlocal-nglobals;
	for (size_t i = 0; i < 2* nlocal - p.size(); i++) {
		if (inv_map[i] >= nlocal) {
			memlay[i] = j++;
		}
	}

	blocklay = memlay;
	j = nlocal-nglobals;
	for (size_t i = nlocal; i < p.size(); i++) {
		if (inv_map[i] < nlocal) {
			std::swap(blocklay[j], blocklay[i]);
			j++;
		}
	}

	auto inmem0 = getPermuationFromTo(Permutation(ident), Permutation(memlay));
	auto block = getPermuationFromTo(Permutation(memlay), Permutation(blocklay));
	auto inmem1 = getPermuationFromTo(Permutation(blocklay), perm);
	for (size_t i = 0; i < inmem1.size(); i++) {
		assert(p[i] == inmem1[block[inmem0[i]]]);
	}
	
	auto it = start;
	for (auto &&perm : decomposeCycles(inmem0)) {
		if (perm.size() == 2) {
			it = gates.insert(it, GateFactory::CreateGate("swap", perm.perm_, {})) + 1;
		} else {
			it = gates.insert(it, GateFactory::CreateGate("nswap", perm.perm_, {})) + 1;
		}
	}
	it = gates.insert(it, GateFactory::CreateGate("placeholder", std::vector<size_t>(block.perm_), {})) + 1;
	for (auto &&perm : decomposeCycles(inmem1)) {
		if (perm.size() == 2) {
			it = gates.insert(it, GateFactory::CreateGate("swap", perm.perm_, {})) + 1;
		} else {
			it = gates.insert(it, GateFactory::CreateGate("nswap", perm.perm_, {})) + 1;
		}
	}

	return it;
}

std::vector<QuantumCircuit> CircuitPartitioner::partition(const QuantumCircuit &circ) {
	size_t nlocal = MAX_INMEM;
	QuantumCircuit opt_circ = circ;
	sortGates(opt_circ);

	if (circ.num_qubits() <= MAX_INMEM) {
		return std::vector<QuantumCircuit>{opt_circ};
	}

	auto &gates = opt_circ.gates();

	for (auto it = gates.begin(); it != gates.end(); ++it) {
		auto &&gp = *it;
		if (!gp->diagonal()) {
			for (auto q : gp->qubits()) {
				if (q >= nlocal) {
					auto perm = getNextPermutation(it, gates.end(), circ.num_qubits());
					mapGates(it, gates.end(), perm);
					it = insertSwapGates(gates, it, perm, nlocal);
					sortGates(it, gates.end(), circ.num_qubits());
				}
			}
		}
	}

	std::vector<QuantumCircuit> circs(1);
	std::vector<std::vector<size_t>> perms(1);

	for (size_t i = 0; i < opt_circ.num_qubits(); i++)
		perms[0].push_back(i);

	circs[0].set_num_qubits(opt_circ.num_qubits());
	int j = 0;
	for (auto &&g : opt_circ.gates()) {
		if (g->type() != Gate::Type::PLACEHOLDER) {
			circs[j].addGate(g);
		} else {
			j++;
			circs.emplace_back();
			circs[j].set_num_qubits(opt_circ.num_qubits());
			perms.emplace_back();
			perms[j] = g->qubits();
		}
	}
	if (circs[j].gates().empty()) {
		circs.erase(circs.begin() + j-1);
	}


	for (size_t i = 1; i < perms.size(); i++) {
		std::vector<size_t> perm(perms[i].size());
		for (size_t k = 0; k < perms[i].size(); k++) {
			size_t p = k;
			for (size_t j = 0; j <= i; j++) {
				p = perms[j][p];
			}
			perm[k] = p;
		}
		perms[i] = perm;
	}

	j = 0;
	for (auto &&circ : circs) {
		circ.set_permutation(perms[j++]);
	}

	return circs;
}

};
