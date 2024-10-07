#ifndef _FUNCTIONALS_H_
#define _FUNCTIONALS_H_
#include <vector>

#include "operation/operation.h"
#include "result_types/state_vector.h"

namespace functionals {
void apply(StateVector &state_vector, GateOperation &op,
           std::vector<size_t> targets);
void initialize_zero(StateVector &state_vector);
void initialize_basis_z(StateVector &state_vector);
}  // namespace functionals

#endif  //_FUNCTIONALS_H_
