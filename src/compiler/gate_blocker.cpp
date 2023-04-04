#include "gate_blocker.h"
#include "optimizer_utils.h"
#include "circuit/gate_factory.h"

namespace snuqs {

bool mergable(const Gate::shared_ptr &g)
{
  //	for (auto q : g->qubits()) {
  //		if (q >= MAX_INMEM) {
  //			return false;
  //		}
  //	}
  //	return (g->type() != Gate::Type::NSWAP) && (g->type() != Gate::Type::PLACEHOLDER);
  if (g->type() == Gate::Type::NSWAP) {
    if (g->qubits().size() >= SMEM_NELEM) {
      return false;
    }
  }
  return true;
}



Gate::shared_ptr makeBlockGate(const QuantumCircuit &circ, const std::set<size_t> &equbits, size_t s, size_t e) {
  if (s + 1 >= e) {
    return circ.gates()[s];
  }
  auto block_gate = GateFactory::CreateGate("block", std::vector<size_t>(equbits.begin(), equbits.end()), {});
  //BlockGate* ptr = reinterpret_cast<BlockGate*>(block_gate.get());

  auto &&gates = circ.gates();
  std::vector<Gate::shared_ptr> block;
  for (size_t j = s; j < e; j++) {
    block_gate->AddSubgate(gates[j]);
    //block.push_back(gates[j]);
  }
  //ptr->addGates(block);
  //block_gate->addGates(block);
  return block_gate;
}

QuantumCircuit GateBlocker::optimize(const QuantumCircuit &circ) {
#ifdef __CUDA_ARCH__
	QuantumCircuit opt_circ;
	opt_circ.set_num_qubits(circ.num_qubits());
	opt_circ.set_permutation(circ.permutation());

	auto &gates = circ.gates();


	size_t s = 0;
	size_t i = 0;
	std::set<size_t> equbits;
	for (; i < gates.size(); i++) {
		/*
		if (mergable(gates[i])) {
			for (auto q : gates[i]->qubits()) {
				if (q < MAX_INMEM && q >= WARPSHIFT) {
					equbits.insert(q);
				}
			}
			Gate::shared_ptr block = makeBlockGate(circ, equbits, i, i+1);
			equbits.clear();
			opt_circ.addGate(block);
		} else {
			opt_circ.addGate(gates[i]);
		}
		*/
		if (mergable(gates[i])) {
			size_t to_be_added = 0;
			for (auto q : gates[i]->qubits()) {
				if (q < MAX_INMEM && q >= WARPSHIFT) {
					if (equbits.find(q) == equbits.end()) {
						to_be_added++;
					}
				}
			}

			if (equbits.size() + to_be_added <= MAX_CACHE) {
				for (auto q : gates[i]->qubits()) {
					if (q < MAX_INMEM && q >= WARPSHIFT) {
						equbits.insert(q);
					}
				}
			} else {
				Gate::shared_ptr block = makeBlockGate(circ, equbits, s, i);
				opt_circ.addGate(block);
				equbits.clear();

				s = i;
				for (auto q : gates[i]->qubits()) {
					if (q < MAX_INMEM && q >= WARPSHIFT) {
						equbits.insert(q);
					}
				}
			}
		} else {
			if (s < i) {
				Gate::shared_ptr block = makeBlockGate(circ, equbits, s, i);
				opt_circ.addGate(block);
				equbits.clear();
			}

			opt_circ.addGate(gates[i]);
			s = i+1;
		}
	}

	if (s < i) {
		Gate::shared_ptr block = makeBlockGate(circ, equbits, s, i);
		opt_circ.addGate(block);
		equbits.clear();
	}
	return opt_circ;
#else
	return circ;
#endif
}

} // namespace snuqs
