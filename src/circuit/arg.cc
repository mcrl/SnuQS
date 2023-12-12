#include "arg.h"

namespace snuqs {
Qarg::Qarg(std::shared_ptr<const Qreg> qreg) : qreg_(qreg) {}
Qarg::Qarg(std::shared_ptr<const Qreg> qreg, size_t index)
    : index_(index), qreg_(qreg) {}
std::shared_ptr<const Qreg> Qarg::qreg() const { return qreg_; }
std::string Qarg::__repr__() const {
  std::string name = qreg_->name();
  if (index_ == -1) {
    return name;
  } else {
    return name + "[" + std::to_string(index_) + "]";
  }
}

size_t Qarg::index() const { return index_; }
size_t Qarg::globalIndex() const { return qreg_->baseDim() + index_; }

bool Qarg::operator<(const Qarg &other) const {
  return this->globalIndex() < other.globalIndex();
}

bool Qarg::operator==(const Qarg &other) const {
  return (this->qreg() == other.qreg()) && (this->index() && other.index());
}

Carg::Carg(const Creg &creg) : creg_(creg) {}
Carg::Carg(const Creg &creg, size_t index) : index_(index), creg_(creg) {}
std::string Carg::__repr__() const {
  std::string name = creg_.name();
  if (index_ == -1) {
    return name;
  } else {
    return name + "[" + std::to_string(index_) + "]";
  }
}

} // namespace snuqs
