#ifndef __CIRCUIT_H__
#define __CIRCUIT_H__

#include "circuit/creg.h"
#include "circuit/qop.h"
#include "circuit/qreg.h"

#include <vector>

namespace snuqs {

class Circuit {
public:
  Circuit(size_t num_qbits, size_t num_bits);
  void append(Qop qop);

private:
  size_t num_qbits_;
  size_t num_bits_;
  std::vector<Qop> qops_;
};
} // namespace snuqs

#endif //__CIRCUIT_H__
