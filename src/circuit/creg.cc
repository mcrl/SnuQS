#include "circuit/creg.h"

#include <iostream>
#include <stdexcept>

namespace snuqs {

Creg::Creg() {}
Creg::Creg(std::string name, size_t num_bits)
    : name_(name), num_bits_(num_bits), bitset_{0} {
  if (num_bits == 0) {
    throw std::invalid_argument("Creg of dimension 0 is not allowed");
  }

  if (num_bits > Creg::MAX_BITS) {
    throw std::invalid_argument("Creg dimension is too large");
  }

  for (auto i = 0; i < num_bits; ++i) {
    cbits_.push_back(new Cbit(*this, i, bitset_[i]));
  }
}

Creg::~Creg() {
  std::cout << "Creg Destructor\n";
  for (auto p : cbits_)
    delete p;
}

std::string Creg::name() const { return name_; }
size_t Creg::value() const { return bitset_.to_ulong(); }

Cbit &Creg::__getitem__(size_t key) { return *cbits_[key]; }
void Creg::__setitem__(size_t key, bool val) { bitset_[key] = val; }

Cbit::Cbit(const Creg &creg, size_t index, Creg::bitset::reference bit)
    : creg_(creg), index_(index), bit_(bit) {}

Cbit::~Cbit() {}

} // namespace snuqs
