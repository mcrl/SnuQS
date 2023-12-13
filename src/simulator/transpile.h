#ifndef __TRANSPILE_H__
#define __TRANSPILE_H__

#include "circuit/circuit.h"

namespace snuqs {
std::shared_ptr<Circuit> transpileSingleGPU(Circuit &_circ);
std::shared_ptr<Circuit> transpileMultiGPU(Circuit &_circ, int num_qubits,
                                           int device, int num_devices);
std::shared_ptr<Circuit> transpileCPU(Circuit &_circ, int num_qubits_per_device,
                                      int num_qubits, int device,
                                      int num_devices);
} // namespace snuqs

#endif // __TRANSPILE_H__
