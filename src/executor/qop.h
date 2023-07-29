#ifndef __QOP_H__
#define __QOP_H__

#include <vector>
#include <iostream>
#include "qreg.h"
#include "creg.h"

namespace snuqs {

enum class QopType {
    Init = 0,
    Fini = 1,
    Cond = 2,
    Measure = 3,
    Reset = 4,
    Barrier = 5,
    UGate = 6,
    CXGate = 7
};

class Qop {
  friend std::ostream &operator<<(std::ostream &os, const Qop &qop);

  public:
  Qop(QopType type, std::vector<int> qubits);
  Qop(QopType type, std::vector<int> qubits, std::vector<int> bits);
  Qop(QopType type, std::vector<int> qubits, std::vector<double> params);
  Qop(QopType type, std::vector<int> qubits, int base, int limit, int val, Qop *qop);
  ~Qop();

  QopType get_type() const;
  const std::vector<int>& get_qubits() const;
  const std::vector<double>& get_params() const;
  const std::vector<int>& get_bits() const;
  int get_base() const;
  int get_limit() const;
  int get_value() const;
  Qop* get_op() const;

  std::ostream& operator<<(std::ostream &os) const;

  private:
  QopType type_;
  std::vector<int> qubits_;
  std::vector<int> bits_;
  std::vector<double> params_;

  int base_;
  int limit_;
  int value_;
  Qop *qop_ = nullptr;
};


} // namespace snuqs

#endif //__QOP_H__
