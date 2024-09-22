#ifndef _UTILS_H_
#define _UTILS_H_

#include <vector>
#include <pybind11/pybind11.h>
namespace py = pybind11;
void multiply_matrix(py::buffer state, py::buffer matrix,
                     std::vector<int> targets);

#endif //_UTILS_H_
