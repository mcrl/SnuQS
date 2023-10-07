#ifndef __QUANTUM_REGISTER_H__
#define __QUANTUM_REGISTER_H__

#include "misc/types.h"

namespace snuqs {

class QuantumRegister {
    public:
    QuantumRegister(qidx num_qubits);

    qidx num_qubits_;
};

} // namespace snuqs

#endif //__QUANTUM_REGISTER_H__
