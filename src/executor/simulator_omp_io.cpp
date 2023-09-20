#include "simulator_omp_io.h"

#include <cmath>
#include <cstdlib>
#include <omp.h>

#include "runtime/rt.h"

using namespace std::complex_literals;
namespace snuqs {

SimulatorOMPIO::SimulatorOMPIO(std::vector<std::string> paths) {
}

SimulatorOMPIO::~SimulatorOMPIO() {
}

void SimulatorOMPIO::init(Qop *qop, Qreg &qreg, Creg &creg) {
  Qreg::amp_t *buf = qreg.get_buf();
  int num_qubits = qreg.get_num_qubits();

  uint64_t num_amps = (1ul << num_qubits);
#pragma omp parallel for
  for (uint64_t i = 0; i < num_amps; ++i) {
    buf[i] = 0.;
  }
  buf[0] = 1.;
}

void SimulatorOMPIO::fini(Qop *qop, Qreg &qreg, Creg &creg) {
  /* Do nothing */
}

void SimulatorOMPIO::cond(Qop *qop, Qreg &qreg, Creg &creg) {
  int val = 0;
  int k = 0;
  for (int i = qop->get_base(); i < qop->get_limit(); ++i) {
    if (creg.get_buf()[i] == 1) {
      val += (1ul << k);
    }
    k += 1;
  }
  if (val == qop->get_value()) {
    run_op(qop->get_op(), qreg, creg);
  }
}

void SimulatorOMPIO::measure(Qop *qop, Qreg &qreg, Creg &creg) {
  Qreg::amp_t *buf = qreg.get_buf();
  int num_qubits = qreg.get_num_qubits();
  const std::vector<int> &qubits = qop->get_qubits();
  int *creg_buf = creg.get_buf();
  int num_bits = creg.get_num_bits();
  const std::vector<int> &bits = qop->get_bits();

  uint64_t target = qubits[0];
  uint64_t tbit = bits[0];
  uint64_t st = (1ul << target);

  double threshold = (double)std::rand() / RAND_MAX;
  double prob0 = 0.;

  uint64_t num_amps = (1ul << num_qubits);

#pragma omp parallel for collapse(2) reduction(+:prob0)
  for (uint64_t i = 0; i < num_amps; i += 2*st) {
    for (uint64_t _j = 0; _j < st; ++_j) {
      uint64_t j = i + _j;
      prob0 += std::abs(buf[j]) * std::abs(buf[j]);
    }
  }


  int collapse_to_zero = false;
  if (threshold <= prob0) {
    collapse_to_zero = true;
    creg_buf[tbit] = 0;
  } else {
    collapse_to_zero = false;
    creg_buf[tbit] = 1;
  }


#pragma omp parallel for collapse(2)
  for (uint64_t i = 0; i < num_amps; i += 2*st) {
    for (uint64_t _j = 0; _j < st; ++_j) {
      uint64_t j = i + _j;
      if (collapse_to_zero) {
        buf[j] = buf[j] / std::sqrt(prob0);
        buf[j+st] = 0;
      } else {
        buf[j] = 0;
        buf[j+st] = buf[j+st] / std::sqrt(1-prob0);
      }
    }
  }
}

void SimulatorOMPIO::reset(Qop *qop, Qreg &qreg, Creg &creg) {
  Qreg::amp_t *buf = qreg.get_buf();
  int num_qubits = qreg.get_num_qubits();
  const std::vector<int> &qubits = qop->get_qubits();


  uint64_t target = qubits[0];
  uint64_t st = (1ul << target);
  uint64_t num_amps = (1ul << num_qubits);
  double prob0 = 0.;

#pragma omp parallel for collapse(2) reduction(+:prob0)
  for (uint64_t i = 0; i < num_amps; i += 2*st) {
    for (uint64_t _j = 0; _j < st; ++_j) {
      uint64_t j = i + _j;
      prob0 += std::abs(buf[j]) * std::abs(buf[j]);
    }
  }

#pragma omp parallel for collapse(2)
  for (uint64_t i = 0; i < num_amps; i += 2*st) {
    for (uint64_t _j = 0; _j < st; ++_j) {
      uint64_t j = i + _j;
      buf[j] = buf[j] / std::sqrt(prob0);
      buf[j+st] = 0;
    }
  }
}

void SimulatorOMPIO::barrier(Qop *qop, Qreg &qreg, Creg &creg) {
  /* Do nothing */
}

void SimulatorOMPIO::ugate(Qop *qop, Qreg &qreg, Creg &creg) {
  Qreg::amp_t *buf = qreg.get_buf();
  int num_qubits = qreg.get_num_qubits();
  const std::vector<int> &qubits = qop->get_qubits();
  const std::vector<double> &params = qop->get_params();

  uint64_t target = qubits[0];
  uint64_t st = (1ul << target);

  double theta = params[0];
  double phi = params[1];
  double lambda = params[2];

  Qreg::amp_t a00 = std::cos(theta/2);
  Qreg::amp_t a01 = -std::exp(1i*lambda) * std::sin(theta/2);
  Qreg::amp_t a10 = std::exp(1i*phi) * std::sin(theta/2);
  Qreg::amp_t a11 = std::exp(1i*(phi+lambda)) * std::cos(theta/2);

  uint64_t num_amps = (1ul << num_qubits);

#pragma omp parallel for collapse(2)
  for (uint64_t i = 0; i < num_amps; i += 2*st) {
    for (uint64_t _j = 0; _j < st; ++_j) {
      uint64_t j = i + _j;
      Qreg::amp_t v0 = a00 * buf[j] + a01 * buf[j+st];
      Qreg::amp_t v1 = a10 * buf[j] + a11 * buf[j+st];
      buf[j] = v0;
      buf[j+st] = v1;
    }
  }
}

void SimulatorOMPIO::cxgate(Qop *qop, Qreg &qreg, Creg &creg) {
  Qreg::amp_t *buf = qreg.get_buf();
  int num_qubits = qreg.get_num_qubits();
  const std::vector<int> &qubits = qop->get_qubits();

  uint64_t ctrl = qubits[0];
  uint64_t target = qubits[1];
  uint64_t t0 = qubits[0];
  uint64_t t1 = qubits[1];

  if (ctrl > target) {
    uint64_t tmp = t0;
    t0 = t1;
    t1 = tmp;
  }

  uint64_t st0 = (1ul << t0);
  uint64_t st1 = (1ul << t1);
  uint64_t cst = (1ul << ctrl);
  uint64_t tst = (1ul << target);
  uint64_t num_amps = (1ul << num_qubits);

#pragma omp parallel for collapse(3)
  for (uint64_t i = 0; i < num_amps; i += 2*st1) {
    for (uint64_t _j = 0; _j < st1; _j += 2*st0) {
      for (uint64_t _k = 0; _k < st0; ++_k) {
        uint64_t j = i + _j;
        uint64_t k = j + _k;
        Qreg::amp_t tmp = buf[k+cst+tst];
        buf[k+cst+tst] = buf[k+cst];
        buf[k+cst] = tmp;
      }
    }
  }
}

void SimulatorOMPIO::run_op(Qop* qop, Qreg &qreg, Creg &creg) {
    switch (qop->get_type()) {
      case QopType::Init: init(qop, qreg, creg); break;
      case QopType::Fini: fini(qop, qreg, creg); break;
      case QopType::Cond: cond(qop, qreg, creg); break;
      case QopType::Measure: measure(qop, qreg, creg); break;
      case QopType::Reset: reset(qop, qreg, creg); break;
      case QopType::Barrier: barrier(qop, qreg, creg);
      case QopType::UGate: ugate(qop, qreg, creg); break;
      case QopType::CXGate: cxgate(qop, qreg, creg); break;
    }
}

void SimulatorOMPIO::run(std::vector<Qop*> &circ, Qreg &qreg, Creg &creg) {
  std::cout << "[SnuQS] Running OMP IO Simulator...\n";
  for (auto qop : circ) {
    run_op(qop, qreg, creg);
  }
}

} // namespace snuqs
