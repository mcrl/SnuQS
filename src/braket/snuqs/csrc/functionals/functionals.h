#ifndef _FUNCTIONALS_H_
#define _FUNCTIONALS_H_
#include <vector>

#include "operation/operation.h"
#include "result_types/state_vector.h"
#include "stream/stream.h"

namespace functionals {
void apply(StateVector &state_vector, GateOperation &op, size_t num_qubits,
           std::vector<size_t> targets, std::shared_ptr<Stream> stream);
void initialize_zero(StateVector &state_vector,
                     std::shared_ptr<Stream> &stream);
void initialize_basis_z(StateVector &state_vector,
                        std::shared_ptr<Stream> &stream);
}  // namespace functionals

#endif  //_FUNCTIONALS_H_
