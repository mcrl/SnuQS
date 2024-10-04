#include "functionals.h"

#include <thrust/complex.h>

#include <cassert>

#include "gate_operations_impl_cpu.h"
#include "gate_operations_impl_cuda.h"

namespace functionals {
void apply(StateVector &state_vector, GateOperation &op,
           std::vector<size_t> targets, bool use_cuda) {
  assert(state_vector.allocated());
  size_t nelems = state_vector.num_elems();
  size_t nqubits = state_vector.num_qubits();

  if (!use_cuda) {
    cpu::applyGate(
        reinterpret_cast<std::complex<double> *>(state_vector.data()),
        reinterpret_cast<std::complex<double> *>(op.data()), targets, nqubits,
        nelems);
  } else {
    cu::applyGate(
        reinterpret_cast<thrust::complex<double> *>(state_vector.data_cuda()),
        reinterpret_cast<thrust::complex<double> *>(op.data_cuda()), targets,
        nqubits, nelems);
  }
}

void initialize_zero(StateVector &state_vector) {
  assert(state_vector.allocated());
  assert(state_vector.device() == Device::CPU);
  size_t nelems = state_vector.num_elems();
  size_t nqubits = state_vector.num_qubits();

  auto buf = reinterpret_cast<std::complex<double> *>(state_vector.data());
  for (size_t i = 0; i < nelems; ++i) {
    buf[i] = 0;
  }
}

void initialize_basis_z(StateVector &state_vector) {
  assert(state_vector.allocated());
  assert(state_vector.device() == Device::CPU);

  auto buf = reinterpret_cast<std::complex<double> *>(state_vector.data());
  initialize_zero(state_vector);
  buf[0] = 1;
}
}  // namespace functionals
