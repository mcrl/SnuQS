#ifndef _INTERFACE_H_
#define _INTERFACE_H_
#include "gate_operation.h"
#include "state_vector.h"
#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

namespace py = pybind11;

void evolve(StateVector &state_vector, GateOperation &op,
            std::vector<size_t> targets, bool use_cuda);

#endif //_INTERFACE_H_
