#include "functionals/functionals.h"

#include <thrust/complex.h>

#include <cassert>

#include "device_type.h"
#include "initialization/initialization_impl_cpu.h"
#include "initialization/initialization_impl_cuda.h"
#include "operation/gate_operations_impl_cpu.h"
#include "operation/gate_operations_impl_cuda.h"

namespace functionals {
void apply(StateVector &state_vector, GateOperation &op,
           std::vector<size_t> targets) {
  assert(state_vector.allocated());
  assert(state_vector.initialized());

  if (state_vector.device() == DeviceType::CPU) {
    cpu::applyGate(state_vector.data(), op.data(), targets,
                   state_vector.num_qubits(), state_vector.num_elems());
  } else if (state_vector.device() == DeviceType::CUDA) {
    cu::applyGate(state_vector.data_cuda(), op.data_cuda(), targets,
                  state_vector.num_qubits(), state_vector.num_elems());
  } else {
    assert(false);
  }
}

void initialize_zero(StateVector &state_vector) {
  assert(state_vector.allocated());
  if (state_vector.device() == DeviceType::CPU) {
    cpu::initializeZero(state_vector.data(), state_vector.num_elems());
  } else if (state_vector.device() == DeviceType::CUDA) {
    cu::initializeZero(state_vector.data_cuda(), state_vector.num_elems());
  } else {
    assert(false);
  }
  state_vector.set_initialized();
}

void initialize_basis_z(StateVector &state_vector) {
  assert(state_vector.allocated());
  if (state_vector.device() == DeviceType::CPU) {
    cpu::initializeBasis_Z(state_vector.data(), state_vector.num_elems());
  } else if (state_vector.device() == DeviceType::CUDA) {
    cu::initializeBasis_Z(state_vector.data_cuda(), state_vector.num_elems());
  } else {
    assert(false);
  }
  state_vector.set_initialized();
}
}  // namespace functionals
