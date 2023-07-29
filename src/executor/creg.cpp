#include "creg.h"

namespace snuqs {

int* Creg::get_buf() const {
  return buf_;
}

void Creg::set_buf(int *buf) {
  buf_ = buf;
}

int Creg::get_num_bits() const {
  return num_bits_;
}

void Creg::set_num_bits(int num_bits) {
  num_bits_ = num_bits;
}


} //namespace snuqs
