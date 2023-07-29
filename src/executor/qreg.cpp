#include "qreg.h"

namespace snuqs {

Qreg::amp_t* Qreg::get_buf() const {
  return buf_;
}

void Qreg::set_buf(Qreg::amp_t *buf) {
  buf_ = buf;
}

std::vector<Qreg::device_amp_t*> Qreg::get_device_bufs() const {
  return device_bufs_;
}

Qreg::device_amp_t* Qreg::get_device_buf(int i) const {
  return device_bufs_[i];
}

void Qreg::set_device_buf(Qreg::device_amp_t *device_buf, int i) {
  if (device_bufs_.size() <= i) {
    device_bufs_.resize(i+1);
  }

  device_bufs_[i] = device_buf;
}

int Qreg::get_num_qubits() const {
  return num_qubits_;
}

void Qreg::set_num_qubits(int num_qubits) {
  num_qubits_ = num_qubits;
}

} //namespace snuqs
