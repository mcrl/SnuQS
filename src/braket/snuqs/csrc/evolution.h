#ifndef _EVOLUTION_H_
#define _EVOLUTION_H_
#include <vector>
#include "operation.h"
#include "state_vector.h"

void evolve(StateVector &state_vector, GateOperation &op,
            std::vector<size_t> targets, bool use_cuda);

#endif  //_EVOLUTION_H_
