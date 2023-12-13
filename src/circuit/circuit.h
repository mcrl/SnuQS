#ifndef __CIRCUIT_H__
#define __CIRCUIT_H__

#include "circuit/qop.h"
#include "circuit/reg.h"

#include <memory>
#include <vector>

namespace snuqs {

class Circuit {
public:
  Circuit(const std::string &name);
  void append_qreg(std::shared_ptr<Qreg> qreg);
  void append_creg(std::shared_ptr<Creg> creg);
  void append(std::shared_ptr<Qop> qop);
  void prepend(std::shared_ptr<Qop> qop);

  std::shared_ptr<Qreg> getQregForIndex(size_t index) const;

  std::string __repr__();
  std::string name() const;

  std::vector<std::shared_ptr<Qop>> &qops();
  const std::vector<std::shared_ptr<Qop>> &qops() const;

  std::vector<std::shared_ptr<Qreg>> &qregs();
  const std::vector<std::shared_ptr<Qreg>> &qregs() const;

  std::vector<std::shared_ptr<Creg>> &cregs();
  const std::vector<std::shared_ptr<Creg>> &cregs() const;

private:
  std::string name_;
  std::vector<std::shared_ptr<Qreg>> qregs_;
  std::vector<std::shared_ptr<Creg>> cregs_;
  std::vector<std::shared_ptr<Qop>> qops_;
};
} // namespace snuqs

#endif //__CIRCUIT_H__
