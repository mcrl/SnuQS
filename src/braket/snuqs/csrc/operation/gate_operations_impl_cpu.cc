#include "operation/gate_operations_impl_cpu.h"

#include <spdlog/spdlog.h>

#include <cassert>
#include <complex>

namespace cpu {
static void applyGlobalPhase(std::complex<double> *buffer,
                             std::complex<double> *gate,
                             std::vector<size_t> targets, size_t nqubits,
                             size_t nelems) {
  std::complex<double> gphase = gate[0];
  for (size_t i = 0; i < nelems; ++i) {
    buffer[i] = buffer[i] * gphase;
  }
}

static void applyOneQubitGate(std::complex<double> *buffer,
                              std::complex<double> *gate,
                              std::vector<size_t> targets, size_t nqubits,
                              size_t nelems) {
  size_t target = targets[0];
  size_t st = (1ull << (nqubits - target - 1));
  for (size_t i = 0; i < nelems; ++i) {
    if ((i & st) == 0) {
      std::complex<double> a0 = buffer[i];
      std::complex<double> a1 = buffer[i + st];
      buffer[i] = gate[0 * 2 + 0] * a0 + gate[0 * 2 + 1] * a1;
      buffer[i + st] = gate[1 * 2 + 0] * a0 + gate[1 * 2 + 1] * a1;
    }
  }
}

static void applyTwoQubitGate(std::complex<double> *buffer,
                              std::complex<double> *gate,
                              std::vector<size_t> targets, size_t nqubits,
                              size_t nelems) {
  size_t t0 = targets[0];
  size_t t1 = targets[1];
  size_t target0 = nqubits - t1 - 1;
  size_t target1 = nqubits - t0 - 1;
  size_t st0 = (1ull << target0);
  size_t st1 = (1ull << target1);
  for (size_t i = 0; i < nelems; ++i) {
    if ((i & st0) == 0 && (i & st1) == 0) {
      std::complex<double> a0 = buffer[i + 0];
      std::complex<double> a1 = buffer[i + st0];
      std::complex<double> a2 = buffer[i + st1];
      std::complex<double> a3 = buffer[i + st1 + st0];
      buffer[i + 0] = gate[0 * 4 + 0] * a0 + gate[0 * 4 + 1] * a1 +
                      gate[0 * 4 + 2] * a2 + gate[0 * 4 + 3] * a3;
      buffer[i + st0] = gate[1 * 4 + 0] * a0 + gate[1 * 4 + 1] * a1 +
                        gate[1 * 4 + 2] * a2 + gate[1 * 4 + 3] * a3;
      buffer[i + st1] = gate[2 * 4 + 0] * a0 + gate[2 * 4 + 1] * a1 +
                        gate[2 * 4 + 2] * a2 + gate[2 * 4 + 3] * a3;
      buffer[i + st1 + st0] = gate[3 * 4 + 0] * a0 + gate[3 * 4 + 1] * a1 +
                              gate[3 * 4 + 2] * a2 + gate[3 * 4 + 3] * a3;
    }
  }
}

static void applyThreeQubitGate(std::complex<double> *buffer,
                                std::complex<double> *gate,
                                std::vector<size_t> targets, size_t nqubits,
                                size_t nelems) {
  size_t t0 = targets[0];
  size_t t1 = targets[1];
  size_t t2 = targets[2];
  size_t target0 = nqubits - t2 - 1;
  size_t target1 = nqubits - t1 - 1;
  size_t target2 = nqubits - t0 - 1;
  size_t st0 = (1ull << target0);
  size_t st1 = (1ull << target1);
  size_t st2 = (1ull << target2);
  for (size_t i = 0; i < nelems; ++i) {
    if ((i & st0) == 0 && (i & st1) == 0 && (i & st2) == 0) {
      std::complex<double> a0 = buffer[i + 0];
      std::complex<double> a1 = buffer[i + st0];
      std::complex<double> a2 = buffer[i + st1];
      std::complex<double> a3 = buffer[i + st1 + st0];
      std::complex<double> a4 = buffer[i + st2];
      std::complex<double> a5 = buffer[i + st2 + st0];
      std::complex<double> a6 = buffer[i + st2 + st1];
      std::complex<double> a7 = buffer[i + st2 + st1 + st0];
      buffer[i + 0] = gate[0 * 8 + 0] * a0 + gate[0 * 8 + 1] * a1 +
                      gate[0 * 8 + 2] * a2 + gate[0 * 8 + 3] * a3 +
                      gate[0 * 8 + 4] * a4 + gate[0 * 8 + 5] * a5 +
                      gate[0 * 8 + 6] * a6 + gate[0 * 8 + 7] * a7;
      buffer[i + st0] = gate[1 * 8 + 0] * a0 + gate[1 * 8 + 1] * a1 +
                        gate[1 * 8 + 2] * a2 + gate[1 * 8 + 3] * a3 +
                        gate[1 * 8 + 4] * a4 + gate[1 * 8 + 5] * a5 +
                        gate[1 * 8 + 6] * a6 + gate[1 * 8 + 7] * a7;
      buffer[i + st1] = gate[2 * 8 + 0] * a0 + gate[2 * 8 + 1] * a1 +
                        gate[2 * 8 + 2] * a2 + gate[2 * 8 + 3] * a3 +
                        gate[2 * 8 + 4] * a4 + gate[2 * 8 + 5] * a5 +
                        gate[2 * 8 + 6] * a6 + gate[2 * 8 + 7] * a7;
      buffer[i + st1 + st0] = gate[3 * 8 + 0] * a0 + gate[3 * 8 + 1] * a1 +
                              gate[3 * 8 + 2] * a2 + gate[3 * 8 + 3] * a3 +
                              gate[3 * 8 + 4] * a4 + gate[3 * 8 + 5] * a5 +
                              gate[3 * 8 + 6] * a6 + gate[3 * 8 + 7] * a7;
      buffer[i + st2] = gate[4 * 8 + 0] * a0 + gate[4 * 8 + 1] * a1 +
                        gate[4 * 8 + 2] * a2 + gate[4 * 8 + 3] * a3 +
                        gate[4 * 8 + 4] * a4 + gate[4 * 8 + 5] * a5 +
                        gate[4 * 8 + 6] * a6 + gate[4 * 8 + 7] * a7;
      buffer[i + st2 + st0] = gate[5 * 8 + 0] * a0 + gate[5 * 8 + 1] * a1 +
                              gate[5 * 8 + 2] * a2 + gate[5 * 8 + 3] * a3 +
                              gate[5 * 8 + 4] * a4 + gate[5 * 8 + 5] * a5 +
                              gate[5 * 8 + 6] * a6 + gate[5 * 8 + 7] * a7;
      buffer[i + st2 + st1] = gate[6 * 8 + 0] * a0 + gate[6 * 8 + 1] * a1 +
                              gate[6 * 8 + 2] * a2 + gate[6 * 8 + 3] * a3 +
                              gate[6 * 8 + 4] * a4 + gate[6 * 8 + 5] * a5 +
                              gate[6 * 8 + 6] * a6 + gate[6 * 8 + 7] * a7;
      buffer[i + st2 + st1 + st0] =
          gate[7 * 8 + 0] * a0 + gate[7 * 8 + 1] * a1 + gate[7 * 8 + 2] * a2 +
          gate[7 * 8 + 3] * a3 + gate[7 * 8 + 4] * a4 + gate[7 * 8 + 5] * a5 +
          gate[7 * 8 + 6] * a6 + gate[7 * 8 + 7] * a7;
    }
  }
}

void applyGate(void *_buffer, void *_gate, std::vector<size_t> targets,
               size_t nqubits, size_t nelems) {
  assert(targets.size() == 1 || targets.size() == 2 || targets.size() == 3);
  auto buffer = reinterpret_cast<std::complex<double> *>(_buffer);
  auto gate = reinterpret_cast<std::complex<double> *>(_gate);

  if (targets.size() == 0) {
    applyGlobalPhase(buffer, gate, targets, nqubits, nelems);
  } else if (targets.size() == 1) {
    applyOneQubitGate(buffer, gate, targets, nqubits, nelems);
  } else if (targets.size() == 2) {
    applyTwoQubitGate(buffer, gate, targets, nqubits, nelems);
  } else if (targets.size() == 3) {
    applyThreeQubitGate(buffer, gate, targets, nqubits, nelems);
  } else {
    assert(false);
  }
}
}  // namespace cpu
