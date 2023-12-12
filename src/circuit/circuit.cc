#include "circuit/circuit.h"
#include <sstream>

namespace snuqs {

Circuit::Circuit(const std::string &name) {}
void Circuit::append_qreg(std::shared_ptr<Qreg> qreg) {
  qregs_.push_back(qreg);
}
void Circuit::append_creg(std::shared_ptr<Creg> creg) {
  cregs_.push_back(creg);
}
void Circuit::append(std::shared_ptr<Qop> qop) { qops_.push_back(qop); }
std::string Circuit::__repr__() {

  std::ostringstream s;
  s << "Circuit<'" << name_ << "'>\n";

  s << "qregs:\n";
  for (auto &qreg : qregs_) {
    s << "    " << qreg->__repr__() << "\n";
  }

  s << "cregs:\n";
  for (auto &creg : cregs_) {
    s << "    " << creg->__repr__() << "\n";
  }

  s << "qops:\n";
  for (auto &qop : qops_) {
    s << "    " << qop->__repr__() << "\n";
  }

  return s.str();
}

std::vector<std::shared_ptr<Qop>> &Circuit::qops() { return qops_; }
const std::vector<std::shared_ptr<Qop>> &Circuit::qops() const { return qops_; }

std::vector<std::shared_ptr<Qreg>> &Circuit::qregs() { return qregs_; }
const std::vector<std::shared_ptr<Qreg>> &Circuit::qregs() const {
  return qregs_;
}

std::vector<std::shared_ptr<Creg>> &Circuit::cregs() { return cregs_; }
const std::vector<std::shared_ptr<Creg>> &Circuit::cregs() const {
  return cregs_;
}

} // namespace snuqs
