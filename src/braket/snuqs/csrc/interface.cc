#include "interface.h"
#include "gate_operation.h"

namespace py = pybind11;
void evolve(GateOperation &op, py::buffer buffer, std::vector<size_t> targets) {
  py::buffer_info info = buffer.request();
  size_t nelem = 1;
  for (int i = 0; i < info.ndim; ++i) {
    nelem *= info.shape[i];
  }

  size_t num_qubits = info.ndim; // FIXME later
  std::complex<double> *buf =
      reinterpret_cast<std::complex<double> *>(info.ptr);
  op.evolve(buf, targets);
}
