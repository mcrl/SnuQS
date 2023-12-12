#ifndef __QOP_H__
#define __QOP_H__

#include "arg.h"
#include "parameter.h"
#include "types.h"
#include <memory>
#include <vector>

namespace snuqs {

enum class QopType {
  BARRIER = 1,
  RESET = 2,
  MEASURE = 3,
  COND = 4,
  CUSTOM = 5,
  QGATE = 6,
  GLOBAL_SWAP = 7,
};

class Qop {
public:
  Qop(QopType type);
  Qop(QopType type, std::vector<std::shared_ptr<Qarg>> qargs);
  Qop(QopType type, std::vector<std::shared_ptr<Qarg>> qargs,
      std::vector<std::shared_ptr<Parameter>> params);
  virtual ~Qop();
  void setQargs(std::vector<std::shared_ptr<Qarg>> qargs);
  std::vector<std::shared_ptr<Qarg>> qargs();
  std::vector<std::shared_ptr<Parameter>> params();
  QopType type() const;
  virtual std::string __repr__() const;

  std::vector<std::shared_ptr<Qarg>> qargs_;
  std::vector<std::shared_ptr<Parameter>> params_;

private:
  QopType type_;
};

class Barrier : public Qop {
public:
  Barrier(std::vector<std::shared_ptr<Qarg>> qargs);
  virtual std::string __repr__() const override;
};

class Reset : public Qop {
public:
  Reset(std::vector<std::shared_ptr<Qarg>> qargs);
  virtual std::string __repr__() const override;
};

class Measure : public Qop {
public:
  Measure(std::vector<std::shared_ptr<Qarg>> qargs, std::vector<Carg> cbits);
  virtual std::string __repr__() const override;

private:
  std::vector<Carg> cbits_;
};

class Cond : public Qop {
public:
  Cond(std::shared_ptr<Qop> op, std::shared_ptr<Creg> creg, size_t val);
  virtual std::string __repr__() const override;

private:
  std::shared_ptr<Qop> op_;
  std::shared_ptr<Creg> creg_;
  size_t val_;
};

class Custom : public Qop {
public:
  Custom(const std::string &name, std::vector<std::shared_ptr<Qop>> qops,
         std::vector<std::shared_ptr<Qarg>> qargs,
         std::vector<std::shared_ptr<Parameter>> params);
  std::vector<std::shared_ptr<Qop>> qops();
  virtual std::string __repr__() const override;

private:
  std::string name_;
  std::vector<std::shared_ptr<Qop>> qops_;
};

enum class QgateType {
  ID = 0,
  X = 1,
  Y = 2,
  Z = 3,
  H = 4,
  S = 5,
  SDG = 6,
  T = 7,
  TDG = 8,
  SX = 9,
  SXDG = 10,
  P = 11,
  RX = 12,
  RY = 13,
  RZ = 14,
  U0 = 15,
  U1 = 16,
  U2 = 17,
  U3 = 18,
  U = 19,
  CX = 20,
  CY = 21,
  CZ = 22,
  SWAP = 23,
  CH = 24,
  CSX = 25,
  CRX = 26,
  CRY = 27,
  CRZ = 28,
  CP = 29,
  CU1 = 30,
  RXX = 31,
  RZZ = 32,
  CU3 = 33,
  CU = 34,
  CCX = 35,
  CSWAP = 36,
};

class Qgate : public Qop {
public:
  Qgate(QgateType type, std::vector<std::shared_ptr<Qarg>> qargs);
  Qgate(QgateType type, std::vector<std::shared_ptr<Qarg>> qargs,
        std::vector<std::shared_ptr<Parameter>> params);
  QgateType gate_type() const;
  virtual std::string __repr__() const override;
  virtual size_t numQargs() const = 0;
  virtual size_t numParams() const = 0;

private:
  QgateType gate_type_;
};

class ID : public Qgate {
public:
  ID(std::vector<std::shared_ptr<Qarg>> qargs);
  ID(std::vector<std::shared_ptr<Qarg>> qargs,
     std::vector<std::shared_ptr<Parameter>> params);
  virtual size_t numQargs() const override;
  virtual size_t numParams() const override;
};

class X : public Qgate {
public:
  X(std::vector<std::shared_ptr<Qarg>> qargs);
  X(std::vector<std::shared_ptr<Qarg>> qargs,
    std::vector<std::shared_ptr<Parameter>> params);
  virtual size_t numQargs() const override;
  virtual size_t numParams() const override;
};

class Y : public Qgate {
public:
  Y(std::vector<std::shared_ptr<Qarg>> qargs);
  Y(std::vector<std::shared_ptr<Qarg>> qargs,
    std::vector<std::shared_ptr<Parameter>> params);
  virtual size_t numQargs() const override;
  virtual size_t numParams() const override;
};

class Z : public Qgate {
public:
  Z(std::vector<std::shared_ptr<Qarg>> qargs);
  Z(std::vector<std::shared_ptr<Qarg>> qargs,
    std::vector<std::shared_ptr<Parameter>> params);
  virtual size_t numQargs() const override;
  virtual size_t numParams() const override;
};

class H : public Qgate {
public:
  H(std::vector<std::shared_ptr<Qarg>> qargs);
  H(std::vector<std::shared_ptr<Qarg>> qargs,
    std::vector<std::shared_ptr<Parameter>> params);
  virtual size_t numQargs() const override;
  virtual size_t numParams() const override;
};

class S : public Qgate {
public:
  S(std::vector<std::shared_ptr<Qarg>> qargs);
  S(std::vector<std::shared_ptr<Qarg>> qargs,
    std::vector<std::shared_ptr<Parameter>> params);
  virtual size_t numQargs() const override;
  virtual size_t numParams() const override;
};

class SDG : public Qgate {
public:
  SDG(std::vector<std::shared_ptr<Qarg>> qargs);
  SDG(std::vector<std::shared_ptr<Qarg>> qargs,
      std::vector<std::shared_ptr<Parameter>> params);
  virtual size_t numQargs() const override;
  virtual size_t numParams() const override;
};

class T : public Qgate {
public:
  T(std::vector<std::shared_ptr<Qarg>> qargs);
  T(std::vector<std::shared_ptr<Qarg>> qargs,
    std::vector<std::shared_ptr<Parameter>> params);
  virtual size_t numQargs() const override;
  virtual size_t numParams() const override;
};

class TDG : public Qgate {
public:
  TDG(std::vector<std::shared_ptr<Qarg>> qargs);
  TDG(std::vector<std::shared_ptr<Qarg>> qargs,
      std::vector<std::shared_ptr<Parameter>> params);
  virtual size_t numQargs() const override;
  virtual size_t numParams() const override;
};

class SX : public Qgate {
public:
  SX(std::vector<std::shared_ptr<Qarg>> qargs);
  SX(std::vector<std::shared_ptr<Qarg>> qargs,
     std::vector<std::shared_ptr<Parameter>> params);
  virtual size_t numQargs() const override;
  virtual size_t numParams() const override;
};

class SXDG : public Qgate {
public:
  SXDG(std::vector<std::shared_ptr<Qarg>> qargs);
  SXDG(std::vector<std::shared_ptr<Qarg>> qargs,
       std::vector<std::shared_ptr<Parameter>> params);
  virtual size_t numQargs() const override;
  virtual size_t numParams() const override;
};

class P : public Qgate {
public:
  P(std::vector<std::shared_ptr<Qarg>> qargs);
  P(std::vector<std::shared_ptr<Qarg>> qargs,
    std::vector<std::shared_ptr<Parameter>> params);
  virtual size_t numQargs() const override;
  virtual size_t numParams() const override;
};

class RX : public Qgate {
public:
  RX(std::vector<std::shared_ptr<Qarg>> qargs);
  RX(std::vector<std::shared_ptr<Qarg>> qargs,
     std::vector<std::shared_ptr<Parameter>> params);
  virtual size_t numQargs() const override;
  virtual size_t numParams() const override;
};

class RY : public Qgate {
public:
  RY(std::vector<std::shared_ptr<Qarg>> qargs);
  RY(std::vector<std::shared_ptr<Qarg>> qargs,
     std::vector<std::shared_ptr<Parameter>> params);
  virtual size_t numQargs() const override;
  virtual size_t numParams() const override;
};

class RZ : public Qgate {
public:
  RZ(std::vector<std::shared_ptr<Qarg>> qargs);
  RZ(std::vector<std::shared_ptr<Qarg>> qargs,
     std::vector<std::shared_ptr<Parameter>> params);
  virtual size_t numQargs() const override;
  virtual size_t numParams() const override;
};

class U0 : public Qgate {
public:
  U0(std::vector<std::shared_ptr<Qarg>> qargs);
  U0(std::vector<std::shared_ptr<Qarg>> qargs,
     std::vector<std::shared_ptr<Parameter>> params);
  virtual size_t numQargs() const override;
  virtual size_t numParams() const override;
};

class U1 : public Qgate {
public:
  U1(std::vector<std::shared_ptr<Qarg>> qargs);
  U1(std::vector<std::shared_ptr<Qarg>> qargs,
     std::vector<std::shared_ptr<Parameter>> params);
  virtual size_t numQargs() const override;
  virtual size_t numParams() const override;
};

class U2 : public Qgate {
public:
  U2(std::vector<std::shared_ptr<Qarg>> qargs);
  U2(std::vector<std::shared_ptr<Qarg>> qargs,
     std::vector<std::shared_ptr<Parameter>> params);
  virtual size_t numQargs() const override;
  virtual size_t numParams() const override;
};

class U3 : public Qgate {
public:
  U3(std::vector<std::shared_ptr<Qarg>> qargs);
  U3(std::vector<std::shared_ptr<Qarg>> qargs,
     std::vector<std::shared_ptr<Parameter>> params);
  virtual size_t numQargs() const override;
  virtual size_t numParams() const override;
};

class U : public Qgate {
public:
  U(std::vector<std::shared_ptr<Qarg>> qargs);
  U(std::vector<std::shared_ptr<Qarg>> qargs,
    std::vector<std::shared_ptr<Parameter>> params);
  virtual size_t numQargs() const override;
  virtual size_t numParams() const override;
};

class CX : public Qgate {
public:
  CX(std::vector<std::shared_ptr<Qarg>> qargs);
  CX(std::vector<std::shared_ptr<Qarg>> qargs,
     std::vector<std::shared_ptr<Parameter>> params);
  virtual size_t numQargs() const override;
  virtual size_t numParams() const override;
};

class CZ : public Qgate {
public:
  CZ(std::vector<std::shared_ptr<Qarg>> qargs);
  CZ(std::vector<std::shared_ptr<Qarg>> qargs,
     std::vector<std::shared_ptr<Parameter>> params);
  virtual size_t numQargs() const override;
  virtual size_t numParams() const override;
};

class CY : public Qgate {
public:
  CY(std::vector<std::shared_ptr<Qarg>> qargs);
  CY(std::vector<std::shared_ptr<Qarg>> qargs,
     std::vector<std::shared_ptr<Parameter>> params);
  virtual size_t numQargs() const override;
  virtual size_t numParams() const override;
};

class SWAP : public Qgate {
public:
  SWAP(std::vector<std::shared_ptr<Qarg>> qargs);
  SWAP(std::vector<std::shared_ptr<Qarg>> qargs,
       std::vector<std::shared_ptr<Parameter>> params);
  virtual size_t numQargs() const override;
  virtual size_t numParams() const override;
};

class CH : public Qgate {
public:
  CH(std::vector<std::shared_ptr<Qarg>> qargs);
  CH(std::vector<std::shared_ptr<Qarg>> qargs,
     std::vector<std::shared_ptr<Parameter>> params);
  virtual size_t numQargs() const override;
  virtual size_t numParams() const override;
};

class CSX : public Qgate {
public:
  CSX(std::vector<std::shared_ptr<Qarg>> qargs);
  CSX(std::vector<std::shared_ptr<Qarg>> qargs,
      std::vector<std::shared_ptr<Parameter>> params);
  virtual size_t numQargs() const override;
  virtual size_t numParams() const override;
};

class CRX : public Qgate {
public:
  CRX(std::vector<std::shared_ptr<Qarg>> qargs);
  CRX(std::vector<std::shared_ptr<Qarg>> qargs,
      std::vector<std::shared_ptr<Parameter>> params);
  virtual size_t numQargs() const override;
  virtual size_t numParams() const override;
};

class CRY : public Qgate {
public:
  CRY(std::vector<std::shared_ptr<Qarg>> qargs);
  CRY(std::vector<std::shared_ptr<Qarg>> qargs,
      std::vector<std::shared_ptr<Parameter>> params);
  virtual size_t numQargs() const override;
  virtual size_t numParams() const override;
};

class CRZ : public Qgate {
public:
  CRZ(std::vector<std::shared_ptr<Qarg>> qargs);
  CRZ(std::vector<std::shared_ptr<Qarg>> qargs,
      std::vector<std::shared_ptr<Parameter>> params);
  virtual size_t numQargs() const override;
  virtual size_t numParams() const override;
};

class CU1 : public Qgate {
public:
  CU1(std::vector<std::shared_ptr<Qarg>> qargs);
  CU1(std::vector<std::shared_ptr<Qarg>> qargs,
      std::vector<std::shared_ptr<Parameter>> params);
  virtual size_t numQargs() const override;
  virtual size_t numParams() const override;
};

class CP : public Qgate {
public:
  CP(std::vector<std::shared_ptr<Qarg>> qargs);
  CP(std::vector<std::shared_ptr<Qarg>> qargs,
     std::vector<std::shared_ptr<Parameter>> params);
  virtual size_t numQargs() const override;
  virtual size_t numParams() const override;
};

class RXX : public Qgate {
public:
  RXX(std::vector<std::shared_ptr<Qarg>> qargs);
  RXX(std::vector<std::shared_ptr<Qarg>> qargs,
      std::vector<std::shared_ptr<Parameter>> params);
  virtual size_t numQargs() const override;
  virtual size_t numParams() const override;
};

class RZZ : public Qgate {
public:
  RZZ(std::vector<std::shared_ptr<Qarg>> qargs);
  RZZ(std::vector<std::shared_ptr<Qarg>> qargs,
      std::vector<std::shared_ptr<Parameter>> params);
  virtual size_t numQargs() const override;
  virtual size_t numParams() const override;
};

class CU3 : public Qgate {
public:
  CU3(std::vector<std::shared_ptr<Qarg>> qargs);
  CU3(std::vector<std::shared_ptr<Qarg>> qargs,
      std::vector<std::shared_ptr<Parameter>> params);
  virtual size_t numQargs() const override;
  virtual size_t numParams() const override;
};

class CU : public Qgate {
public:
  CU(std::vector<std::shared_ptr<Qarg>> qargs);
  CU(std::vector<std::shared_ptr<Qarg>> qargs,
     std::vector<std::shared_ptr<Parameter>> params);
  virtual size_t numQargs() const override;
  virtual size_t numParams() const override;
};

class CCX : public Qgate {
public:
  CCX(std::vector<std::shared_ptr<Qarg>> qargs);
  CCX(std::vector<std::shared_ptr<Qarg>> qargs,
      std::vector<std::shared_ptr<Parameter>> params);
  virtual size_t numQargs() const override;
  virtual size_t numParams() const override;
};

class CSWAP : public Qgate {
public:
  CSWAP(std::vector<std::shared_ptr<Qarg>> qargs);
  CSWAP(std::vector<std::shared_ptr<Qarg>> qargs,
        std::vector<std::shared_ptr<Parameter>> params);
  virtual size_t numQargs() const override;
  virtual size_t numParams() const override;
};

class GlobalSwap : public Qop {
public:
  GlobalSwap(std::vector<std::shared_ptr<Qarg>> qargs);
  virtual std::string __repr__() const override;
};

} // namespace snuqs
#endif //__QOP_H__
