#include "reg.h"

#include <stdexcept>
namespace snuqs {

Reg::Reg(std::string name, size_t dim) : name_(name), dim_(dim), base_dim_(0) {
  if (dim == 0) {
    throw std::invalid_argument("Reg of dimension 0 is not allowed");
  }

  if (dim > Reg::MAX_BITS) {
    throw std::invalid_argument("Reg dimension is too large");
  }
}
std::string Reg::name() const { return name_; }
size_t Reg::dim() const { return dim_; }
size_t Reg::baseDim() const { return base_dim_; }
void Reg::setBaseDim(size_t base_dim) { base_dim_ = base_dim; }
std::string Reg::__repr__() const {
  return name_ + "[" + std::to_string(dim_) + "]";
}

Qreg::Qreg(std::string name, size_t dim) : Reg(name, dim) {}
std::string Qreg::__repr__() const {
  return name_ + "[" + std::to_string(dim_) + "]";
}

bool Qreg::operator==(const Qreg &other) const {
  return (this->name() == other.name());
}

size_t Creg::value() const { return bitset_.to_ulong(); }
Creg::Creg(std::string name, size_t dim) : Reg(name, dim), bitset_{0} {}
std::string Creg::__repr__() const {
  return name_ + "[" + std::to_string(dim_) + "] <'" +
         bitset_.to_string().substr(0, dim_) + "'>";
}
bool Creg::__getitem__(size_t key) { return bitset_[key]; }
void Creg::__setitem__(size_t key, bool val) { bitset_[key] = val; }
} // namespace snuqs
