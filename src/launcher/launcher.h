#ifndef __LAUCNHER_H__
#define __LAUCNHER_H__

#include <memory>
#include <pybind11/pybind11.h>

namespace snuqs {

enum class ExecutionMethod { OMP = 0, CUDA = 1, OMP_IO = 2, CUDA_IO = 3 };

class Launcher {
public:
  Launcher();
  ~Launcher();

  void run(pybind11::object circ);

private:
  // std::vector<Qop*> circ_;
  // std::unique_ptr<SimulatorInterface> simulator_;
};

} // namespace snuqs

#endif // __LAUCNHER_H__
