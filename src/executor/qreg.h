#ifndef __QREG_H__
#define __QREG_H__

#include <complex>
#include <vector>
#include <thrust/complex.h>

namespace snuqs {

class Qreg {
  public:
    using amp_t = std::complex<double>;
    using device_amp_t = thrust::complex<double>;

    amp_t* get_buf() const;
    void set_buf(amp_t* buf);

    std::vector<device_amp_t*> get_device_bufs() const;
    device_amp_t* get_device_buf(int i) const;
    void set_device_buf(device_amp_t *device_buf, int i);

    int get_num_qubits() const;
    void set_num_qubits(int num_qubits);

  private:
      amp_t *buf_;
      int num_qubits_;
      std::vector<device_amp_t*> device_bufs_;
};

} //namespace snuqs

#endif // __QREG_H__
