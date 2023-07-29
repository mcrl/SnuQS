#include "executor.h"

#include <pybind11/pybind11.h>

#include <iostream>

#include "qop.h"
#include "simulator_interface.h"
#include "simulator_omp.h"
#include "simulator_cuda.h"
#include "simulator_omp_io.h"

namespace py = pybind11;

namespace snuqs {

static Qop* build_qop(py::handle op) {
  py::object typ = op.attr("typ").cast<py::object>();
  int typi = typ.attr("value").cast<int>();
  QopType type = static_cast<QopType>(typi);

  std::vector<int> qubits;
  py::object _qubits = op.attr("qubits");
  for (auto q : _qubits.cast<py::list>()) {
    qubits.push_back(q.cast<int>());
  }

  switch (type) {
    case QopType::Barrier:
      /* Do nothing */
      return new Qop(type, qubits);
      break;
    case QopType::Reset:
      return new Qop(type, qubits);
      break;
    case QopType::Measure:
      {
        std::vector<int> bits;
        for (auto p : op.attr("bits").cast<py::list>()) {
          bits.push_back(p.cast<int>());
        }
        return new Qop(type, qubits, bits);
        break;
      }
    case QopType::Cond:
      {
        int base = op.attr("base").cast<int>();
        int limit = op.attr("limit").cast<int>();
        int val = op.attr("val").cast<int>();
        return new Qop(type, qubits, base, limit, val, build_qop(op.attr("op")));
      }
      break;
    case QopType::Init:
      return new Qop(type, qubits);
      break;
    case QopType::Fini:
      return new Qop(type, qubits);
      break;
    case QopType::UGate:
      {
        std::vector<double> params;
        for (auto p : op.attr("params").cast<py::list>()) {
          params.push_back(p.cast<double>());
        }
      return new Qop(type, qubits, params);
      }
    case QopType::CXGate:
      return new Qop(type, qubits);
      break;
  }
  assert(false);
  return new Qop(type, qubits);
}

Executor::Executor() {
}

Executor::~Executor() {
}

void Executor::run(py::object circ, py::object qreg, py::object creg, py::object method) {
  // Circ
  py::list ops = circ.attr("ops").cast<py::list>();
  for (auto &op : ops) {
    circ_.emplace_back(build_qop(op));
  }

  // Qreg
  int num_qubits = qreg.attr("nqubits").cast<int>();
  auto qbuf = reinterpret_cast<std::complex<double>*>(
      qreg.attr("buf").cast<py::buffer>().request().ptr
      );
  qreg_.set_buf(qbuf);
  qreg_.set_num_qubits(num_qubits);

  // Creg
  int num_bits = creg.attr("nbits").cast<int>();
  auto cbuf = reinterpret_cast<int*>(
      creg.attr("buf").cast<py::buffer>().request().ptr
      );
  creg_.set_buf(cbuf);
  creg_.set_num_bits(num_bits);

  // Method
  switch (static_cast<ExecutionMethod>(method.attr("value").cast<int>())) {
    case ExecutionMethod::OMP: 
      simulator_ = std::make_unique<SimulatorOMP>();
      break;
    case ExecutionMethod::CUDA:
      simulator_ = std::make_unique<SimulatorCUDA>();
      break;
    case ExecutionMethod::OMP_IO: 
      simulator_ = std::make_unique<SimulatorOMPIO>(std::vector<std::string>{"/dev/nvme0n1", "/dev/nvme1n1"});
    case ExecutionMethod::CUDA_IO:
      // TODO
      assert(false);
  }


  // Execute
  simulator_->run(circ_, qreg_, creg_);

  // Release
  for (auto op : circ_) {
    delete op;
  }
}

} // namespace snuqs
