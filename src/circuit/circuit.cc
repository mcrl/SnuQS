#include "circuit/circuit.h"

namespace snuqs {

Circuit::Circuit(size_t num_qbits, size_t num_bits)
    : num_qbits_(num_qbits), num_bits_(num_bits) {}

void Circuit::append(Qop qop) { qops_.push_back(qop); }

} // namespace snuqs
