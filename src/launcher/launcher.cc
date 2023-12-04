#include "launcher/launcher.h"

#include "assertion.h"
#include <complex>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace snuqs {

Launcher::Launcher() { NOT_IMPLEMENTED(); }

Launcher::~Launcher() { NOT_IMPLEMENTED(); }

static void *build_qop(py::handle op) {
  NOT_IMPLEMENTED();
  return nullptr;
  /*
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
      // Do nothing
      return new Qop(type, qubits);
      break;
    case QopType::Reset:
      return new Qop(type, qubits);
      break;
    case QopType::Measure: {
      std::vector<int> bits;
      for (auto p : op.attr("bits").cast<py::list>()) {
        bits.push_back(p.cast<int>());
      }
      return new Qop(type, qubits, bits);
      break;
    }
    case QopType::Cond: {
      int base = op.attr("base").cast<int>();
      int limit = op.attr("limit").cast<int>();
      int val = op.attr("val").cast<int>();
      return new Qop(type, qubits, base, limit, val, build_qop(op.attr("op")));
    } break;
    case QopType::Init:
      return new Qop(type, qubits);
      break;
    case QopType::Fini:
      return new Qop(type, qubits);
      break;
    case QopType::UGate: {
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
    */
}

void Launcher::run(py::object circ) {
  NOT_IMPLEMENTED();

  /*
// Circ
py::list ops = circ.attr("ops").cast<py::list>();
for (auto &op : ops) {
  circ_.emplace_back(build_qop(op));
}

// Qreg
py::object qreg = circ.attr("qreg").cast<py::object>();
int num_qubits = qreg.attr("nqubits").cast<int>();
auto qbuf = reinterpret_cast<std::complex<double> *>(
    qreg.attr("buf").cast<py::buffer>().request().ptr);
qreg_.set_buf(qbuf);
qreg_.set_num_qubits(num_qubits);

// Creg
py::object creg = circ.attr("creg").cast<py::object>();
int num_bits = creg.attr("nbits").cast<int>();
auto cbuf = reinterpret_cast<int *>(
    creg.attr("buf").cast<py::buffer>().request().ptr);
creg_.set_buf(cbuf);
creg_.set_num_bits(num_bits);

// Build
simulator_ = std::make_unique<SimulatorCUDA>();

// Execute
simulator_->run(circ_, qreg_, creg_);

// Release
for (auto op : circ_) {
  delete op;
}
*/
}

} // namespace snuqs
