#ifndef _INTERFACE_H_
#define _INTERFACE_H_
#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

namespace py = pybind11;

void evolve(GateOepration &op, py::buffer buffer, std::vector<size_t> targets);
#endif //_INTERFACE_H_
