#include "circuit/qreg.h"

namespace snuqs {

Qreg::Qreg(std::string name, size_t dim) : name_(name), dim_(dim) {
  for (int i = 0; i < dim; ++i) {
    qbits_.push_back(new Qbit(this, i));
  }
}

Qreg::~Qreg() {
  for (auto p : qbits_)
    delete p;
}

Qbit::Qbit(Qreg *qreg, size_t index)
    : Qreg(qreg->name_, 1), qreg_(qreg), index_(index) {}

} // namespace snuqs
