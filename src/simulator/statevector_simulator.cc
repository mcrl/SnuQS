#include "simulator/statevector_simulator.h"
#include "assertion.h"
#include "buffer/buffer.h"
#include "circuit/circuit.h"
#include "circuit/creg.h"
#include "circuit/qreg.h"

#include "simulator/statevector_simulator_cuda.h"

namespace snuqs {

StatevectorSimulator::StatevectorSimulator() {}

StatevectorSimulator::~StatevectorSimulator() {}

void StatevectorSimulator::run(Circuit &circ) { NOT_IMPLEMENTED(); }

void StatevectorSimulator::test() {

  cuda::CudaBuffer<double> buffer(512);
  std::complex<double> *buf =
      reinterpret_cast<std::complex<double> *>(buffer.ptr());

  cuda::gate<double>::initZero(buf, 512, {0}, {});
  cuda::synchronize();

  std::complex<double> *cmem = reinterpret_cast<std::complex<double>*>(malloc(512 * sizeof(std::complex<double>)));
  buffer.read(cmem, 512, 0);

  for (int i = 0; i < 512; ++i) {
      printf("%d: %lf+%lfJ\n",
              i, cmem[i].real(), cmem[i].imag());
  }
  printf("HERE Allocated\n");
}
} // namespace snuqs
