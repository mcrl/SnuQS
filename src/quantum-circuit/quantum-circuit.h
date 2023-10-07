#ifndef __QUANTUM_CIRCUIT_H__
#define __QUANTUM_CIRCUIT_H__

#include "misc/types.h"

#include <vector>

namespace snuqs {

class QuantumCircuit {
    public:
        QuantumCircuit();

        void Reorder(std::vector<int> perm);

        qidx num_qubits_;
};

} // namespace snuqs

#endif //__QUANTUM_CIRCUIT_H__

