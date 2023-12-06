#include "arg.h"

namespace snuqs {
Arg::Arg(const Reg &reg, size_t index) : reg_(reg), index_(index) {}
const Reg &Arg::reg() const { return reg_; }
size_t Arg::index() const { return index_; }
size_t Arg::dim() const { return dim_; }
size_t Arg::value() const { return value_; }
std::string Arg::__repr__() const {
  return reg_.name() + "[" + std::to_string(index_) + "]";
}

Qarg::Qarg(const Qreg &qreg, size_t index) : Arg(qreg, index) {}

Carg::Carg(const Creg &creg, size_t index) : Arg(creg, index) {}

} // namespace snuqs
