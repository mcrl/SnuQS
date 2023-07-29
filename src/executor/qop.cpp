#include "qop.h"

#include <iostream>

namespace snuqs {

Qop::Qop(QopType type, std::vector<int> qubits)
  : type_(type), qubits_(qubits) {
}

Qop::Qop(QopType type, std::vector<int> qubits, std::vector<int> bits)
  : type_(type), qubits_(qubits), bits_(bits) {
}

Qop::Qop(QopType type, std::vector<int> qubits, std::vector<double> params)
  : type_(type), qubits_(qubits), params_(params) {
}

Qop::Qop(QopType type, std::vector<int> qubits, int base, int limit, int value, Qop *qop)
  : type_(type), qubits_(qubits), base_(base), limit_(limit), value_(value), qop_(qop) {
}

Qop::~Qop() {
  if (qop_ != nullptr) {
    delete qop_;
    qop_ = nullptr;
  }
}

QopType Qop::get_type() const {
  return type_;
}

const std::vector<int>& Qop::get_qubits() const {
  return qubits_;
}

const std::vector<double>& Qop::get_params() const {
  return params_;
}

const std::vector<int>& Qop::get_bits() const {
  return bits_;
}

int Qop::get_base() const {
  return base_;
}

int Qop::get_limit() const {
  return limit_;
}

int Qop::get_value() const {
  return value_;
}

Qop* Qop::get_op() const {
  return qop_;
}

std::ostream& Qop::operator<<(std::ostream &os) const {
  os << static_cast<int>(type_);

  if (!params_.empty()) {
    os << "(";
    for (size_t i = 0; i < params_.size(); i++) {
      os << params_[i];
      if (i+1 != params_.size()) {
        os << ",";
      }
    }
    os << ")";
  }

  os << " ";
  if (qubits_.size() == 0) {
    os << "all ";
  } else {
    for (size_t i = 0; i < qubits_.size(); i++) {
      os << qubits_[i];
      if (i+1 != qubits_.size()) {
        os << " ";
      }
    }
  }

  return os;
}

std::ostream &operator<<(std::ostream &os, const Qop &qop)
{
    return qop.operator<<(os);
}


} // namespace snuqs
