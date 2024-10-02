#include "initialization.h"

#include <cassert>

void initialize_zero(StateVector &state_vector) {
  assert(state_vector.device() == Device::CPU);
  size_t nelems = state_vector.num_elems();
  size_t nqubits = state_vector.num_qubits();

  auto buf = reinterpret_cast<std::complex<double> *>(state_vector.data());
  for (size_t i = 0; i < nelems; ++i) {
    buf[i] = 0;
  }
}

void initialize_basis_z(StateVector &state_vector) {
  assert(state_vector.device() == Device::CPU);

  auto buf = reinterpret_cast<std::complex<double> *>(state_vector.data());
  initialize_zero(state_vector);
  buf[0] = 1;
}
