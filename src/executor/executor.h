#ifndef __EXECUTOR_H__
#define __EXECUTOR_H__

#include <pybind11/pybind11.h>
#include <memory>
#include "simulator_interface.h"
#include "qreg.h"
#include "creg.h"

namespace snuqs {

enum class ExecutionMethod {
  OMP = 0,
  CUDA = 1,
  OMP_IO = 2,
  CUDA_IO = 3
};

class Executor {
  public:
    Executor();
    ~Executor();

    void run(pybind11::object circ, pybind11::object qreg, pybind11::object creg, pybind11::object method);

  private:
    std::vector<Qop*> circ_;
    Qreg qreg_;
    Creg creg_;
    std::unique_ptr<SimulatorInterface> simulator_;
};

} // namespace snuqs

#endif // __EXECUTOR_H__
