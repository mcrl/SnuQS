#include "interface.h"
#include "gate_operation.h"
#include "gate_operation_impl.h"
#include "gate_operation_impl_cuda.h"
#include <memory>
#include <thrust/complex.h>

namespace py = pybind11;
void evolve(StateVector &state_vector, GateOperation &op,
            std::vector<size_t> targets, bool use_cuda) {
  size_t nelem = state_vector.num_elems();
  size_t nqubits = state_vector.num_qubits();

  if (!use_cuda) {
    applyGate(reinterpret_cast<std::complex<double> *>(state_vector.data()),
              reinterpret_cast<std::complex<double> *>(op.data()), targets,
              nqubits, nelem);
  } else {
    cu::applyGate(
        reinterpret_cast<thrust::complex<double> *>(state_vector.data_cuda()),
        reinterpret_cast<thrust::complex<double> *>(op.data_cuda()), targets,
        nqubits, nelem);
  }
}
