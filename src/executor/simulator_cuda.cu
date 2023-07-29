#include "simulator_cuda.h"

#include <cmath>
#include <cstdlib>


#include <thrust/complex.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <iostream>

#define CHECK_RT(e)               \
  do {                            \
    if ((e) != rt::RT_SUCCESS) {  \
      std::cout                   \
        << "Runtime ERROR!["      \
        << __FILE__               \
        << ":"                    \
        << __LINE__               \
        << "]   \n";              \
      std::exit(EXIT_FAILURE);    \
    }                             \
  } while (0)

using namespace std::complex_literals;
namespace snuqs {
  
struct int_param_t {
  int params[3];
  int num_params;
};

struct double_param_t {
  double params[3];
  int num_params;
};

__global__ static void kernel_init(
    Qreg::device_amp_t *buf,
    int num_qubits
    )
{
  uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (1ul << num_qubits))
    return;

  buf[i] = 0.;
  if (i == 0)
    buf[i] = 1.;
}

__global__ static void kernel_ugate(
    Qreg::device_amp_t *buf,
    int num_qubits, 
    struct int_param_t target_param,
    struct double_param_t gate_params) {
  uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  int num_params = gate_params.num_params;
  int num_targets = target_param.num_params;

  int *qubits = target_param.params;
  double *params = gate_params.params;
  if (i >= (1ul << num_qubits) / 2)
    return;

  uint64_t target = qubits[0];
  uint64_t st = (1ul << target);
	uint64_t j = ((i >> target) << (target+1)) + (i & (st-1));

  double theta = params[0];
  double phi = params[1];
  double lambda = params[2];
  Qreg::device_amp_t _1i{0, 1};
  Qreg::device_amp_t a00 = cos(theta/2);
  Qreg::device_amp_t a01 = -exp(_1i*lambda) * sin(theta/2);
  Qreg::device_amp_t a10 = exp(_1i*phi) * sin(theta/2);
  Qreg::device_amp_t a11 = exp(_1i*(phi+lambda)) * cos(theta/2);

  Qreg::device_amp_t v0 = a00 * buf[j] + a01 * buf[j+st];
  Qreg::device_amp_t v1 = a10 * buf[j] + a11 * buf[j+st];
  buf[j] = v0;
  buf[j+st] = v1;
}

__global__ static void kernel_cxgate(
    Qreg::device_amp_t *buf,
    int num_qubits,
    struct int_param_t target_param) {
  uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (1ul << num_qubits) / 4)
    return;

  int *qubits = target_param.params;
  int num_targets = target_param.num_params;

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

  uint64_t j = ((i >> t0) << (t0+1)) + (i & (st0-1));
  uint64_t k = ((j >> t1) << (t1+1)) + (j & (st1-1));

  Qreg::device_amp_t tmp = buf[k+cst+tst];
  buf[k+cst+tst] = buf[k+cst];
  buf[k+cst] = tmp;
}

SimulatorCUDA::SimulatorCUDA() {
  CHECK_RT(rt.create_stream(&stream));
}

SimulatorCUDA::~SimulatorCUDA() {
  CHECK_RT(rt.destroy_stream(stream));
}

void SimulatorCUDA::init(Qop *qop, Qreg &qreg, Creg &creg) {
  Qreg::device_amp_t *buf = qreg.get_device_buf(0);
  int num_qubits = qreg.get_num_qubits();

  uint64_t num_amps = (1ul << num_qubits);

  dim3 block_dim(128);
  dim3 grid_dim(((num_amps) + block_dim.x - 1) / block_dim.x);

  rt::kernel_t *kernel;
  CHECK_RT(rt.create_kernel(&kernel, reinterpret_cast<const void*>(kernel_init)));
  CHECK_RT(rt.set_kernel_arg(kernel, 0, sizeof(buf), &buf));
  CHECK_RT(rt.set_kernel_arg(kernel, 1, sizeof(int), &num_qubits));
  CHECK_RT(rt.launch_kernel(kernel, grid_dim, block_dim, 0, stream));
}

void SimulatorCUDA::fini(Qop *qop, Qreg &qreg, Creg &creg) {
  Qreg::amp_t *host_buf = qreg.get_buf();
  Qreg::device_amp_t *device_buf = qreg.get_device_buf(0);
  int num_qubits = qreg.get_num_qubits();

  uint64_t num_amps = (1ul << num_qubits);
  CHECK_RT(rt.memcpy_d2h(
      host_buf,
      device_buf,
      sizeof(Qreg::amp_t) * num_amps,
      stream
      ));
  CHECK_RT(rt.stream_synchronize(stream));
}

void SimulatorCUDA::cond(Qop *qop, Qreg &qreg, Creg &creg) {
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

void SimulatorCUDA::measure(Qop *qop, Qreg &qreg, Creg &creg) {
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

void SimulatorCUDA::reset(Qop *qop, Qreg &qreg, Creg &creg) {
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

void SimulatorCUDA::barrier(Qop *qop, Qreg &qreg, Creg &creg) {
  /* Do nothing */
}

void SimulatorCUDA::ugate(Qop *qop, Qreg &qreg, Creg &creg) {
  Qreg::device_amp_t *buf = qreg.get_device_buf(0);
  int num_qubits = qreg.get_num_qubits();
  const std::vector<int> &qubits = qop->get_qubits();
  const std::vector<double> &params = qop->get_params();

  uint64_t num_amps = (1ul << num_qubits);

  dim3 block_dim(128);
  dim3 grid_dim(((num_amps / 2) + block_dim.x - 1) / block_dim.x);

  struct int_param_t target_param;
  for (int i = 0; i < qubits.size(); ++i) {
    target_param.params[i] = qubits[i];
  }
  target_param.num_params = qubits.size();

  struct double_param_t gate_param;
  for (int i = 0; i < params.size(); ++i) {
    gate_param.params[i] = params[i];
  }
  gate_param.num_params = params.size();

  rt::kernel_t *kernel;
  CHECK_RT(rt.create_kernel(&kernel, reinterpret_cast<const void*>(kernel_ugate)));
  CHECK_RT(rt.set_kernel_arg(kernel, 0, sizeof(buf), &buf));
  CHECK_RT(rt.set_kernel_arg(kernel, 1, sizeof(int), &num_qubits));
  CHECK_RT(rt.set_kernel_arg(kernel, 2, sizeof(struct int_param_t), &target_param));
  CHECK_RT(rt.set_kernel_arg(kernel, 3, sizeof(struct double_param_t), &gate_param));
  CHECK_RT(rt.launch_kernel(kernel, grid_dim, block_dim, 0, stream));
}

void SimulatorCUDA::cxgate(Qop *qop, Qreg &qreg, Creg &creg) {
  Qreg::device_amp_t *buf = qreg.get_device_buf(0);
  int num_qubits = qreg.get_num_qubits();
  const std::vector<int> &qubits = qop->get_qubits();

  uint64_t num_amps = (1ul << num_qubits);
  dim3 block_dim(128);
  dim3 grid_dim(((num_amps / 4) + block_dim.x - 1) / block_dim.x);

  struct int_param_t target_param;
  for (int i = 0; i < qubits.size(); ++i) {
    target_param.params[i] = qubits[i];
  }
  target_param.num_params = qubits.size();

  rt::kernel_t *kernel;
  CHECK_RT(rt.create_kernel(&kernel, reinterpret_cast<const void*>(kernel_cxgate)));
  CHECK_RT(rt.set_kernel_arg(kernel, 0, sizeof(buf), &buf));
  CHECK_RT(rt.set_kernel_arg(kernel, 1, sizeof(int), &num_qubits));
  CHECK_RT(rt.set_kernel_arg(kernel, 2, sizeof(struct int_param_t), &target_param));
  CHECK_RT(rt.launch_kernel(kernel, grid_dim, block_dim, 0, stream));
}

void SimulatorCUDA::run_op(Qop *qop, Qreg &qreg, Creg &creg) {
  switch (qop->get_type()) {
    case QopType::Init: init(qop, qreg, creg); break;
    case QopType::Fini: fini(qop, qreg, creg); break;
    case QopType::Cond: cond(qop, qreg, creg); break;
    case QopType::Measure: measure(qop, qreg, creg); break;
    case QopType::Reset: reset(qop, qreg, creg); break;
    case QopType::Barrier: barrier(qop, qreg, creg); break;
    case QopType::UGate: ugate(qop, qreg, creg); break;
    case QopType::CXGate: cxgate(qop, qreg, creg); break;
  }
}

void SimulatorCUDA::run(std::vector<Qop*> &circ, Qreg &qreg, Creg &creg) {
  std::cout << "[SnuQS] Running CUDA Simulator...\n";

  uint64_t num_amps = (1ul << qreg.get_num_qubits());
  Qreg::device_amp_t *device_buf;
  rt.malloc_device(
      reinterpret_cast<rt::addr_t*>(&device_buf),
      sizeof(Qreg::device_amp_t) * num_amps);
  qreg.set_device_buf(device_buf, 0);

  for (auto qop : circ) {
    run_op(qop, qreg, creg);
    rt.stream_synchronize(stream);
  }

  qreg.set_device_buf(nullptr, 0);
  rt.free_device(reinterpret_cast<rt::addr_t*>(device_buf));
}

} // namespace snuqs
