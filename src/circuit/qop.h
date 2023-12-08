#ifndef __QOP_H__
#define __QOP_H__

#include "arg.h"
#include "parameter.h"
#include "types.h"
#include <memory>
#include <vector>

namespace snuqs {

class Qop {
public:
  Qop();
  Qop(std::vector<Qarg> qbits);
  Qop(std::vector<Qarg> qbits, std::vector<std::shared_ptr<Parameter>> params);
  virtual ~Qop();
  virtual std::string __repr__() const;

  std::vector<Qarg> qbits_;
  std::vector<std::shared_ptr<Parameter>> params_;
};

class Barrier : public Qop {
public:
  Barrier(std::vector<Qarg> qbits);
  virtual std::string __repr__() const override;
};

class Reset : public Qop {
public:
  Reset(std::vector<Qarg> qbits);
  virtual std::string __repr__() const override;
};

class Measure : public Qop {
public:
  Measure(std::vector<Qarg> qbits, std::vector<Carg> cbits);
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
  Custom(const std::string &name, std::vector<std::shared_ptr<Qop>> qops, std::vector<Qarg> qbits,
         std::vector<std::shared_ptr<Parameter>> params);
  virtual std::string __repr__() const override;

private:
  std::string name_;
  std::vector<std::shared_ptr<Qop>> qops_;
};

class Qgate : public Qop {
public:
  Qgate(std::vector<Qarg> qbits);
  Qgate(std::vector<Qarg> qbits,
        std::vector<std::shared_ptr<Parameter>> params);
  virtual size_t numQubits() const;
  virtual size_t numParams() const;
  virtual std::string name() const;
  virtual std::string __repr__() const;
};

class ID : public Qgate {
public:
  ID(std::vector<Qarg> qbits);
  ID(std::vector<Qarg> qbits, std::vector<std::shared_ptr<Parameter>> params);
  virtual size_t numQubits() const override;
  virtual size_t numParams() const override;
  virtual std::string name() const override;
};

class X : public Qgate {
public:
  X(std::vector<Qarg> qbits);
  X(std::vector<Qarg> qbits, std::vector<std::shared_ptr<Parameter>> params);
  virtual size_t numQubits() const override;
  virtual size_t numParams() const override;
  virtual std::string name() const override;
};

class Y : public Qgate {
public:
  Y(std::vector<Qarg> qbits);
  Y(std::vector<Qarg> qbits, std::vector<std::shared_ptr<Parameter>> params);
  virtual size_t numQubits() const override;
  virtual size_t numParams() const override;
  virtual std::string name() const override;
};

class Z : public Qgate {
public:
  Z(std::vector<Qarg> qbits);
  Z(std::vector<Qarg> qbits, std::vector<std::shared_ptr<Parameter>> params);
  virtual size_t numQubits() const override;
  virtual size_t numParams() const override;
  virtual std::string name() const override;
};

class H : public Qgate {
public:
  H(std::vector<Qarg> qbits);
  H(std::vector<Qarg> qbits, std::vector<std::shared_ptr<Parameter>> params);
  virtual size_t numQubits() const override;
  virtual size_t numParams() const override;
  virtual std::string name() const override;
};

class S : public Qgate {
public:
  S(std::vector<Qarg> qbits);
  S(std::vector<Qarg> qbits, std::vector<std::shared_ptr<Parameter>> params);
  virtual size_t numQubits() const override;
  virtual size_t numParams() const override;
  virtual std::string name() const override;
};

class SDG : public Qgate {
public:
  SDG(std::vector<Qarg> qbits);
  SDG(std::vector<Qarg> qbits, std::vector<std::shared_ptr<Parameter>> params);
  virtual size_t numQubits() const override;
  virtual size_t numParams() const override;
  virtual std::string name() const override;
};

class T : public Qgate {
public:
  T(std::vector<Qarg> qbits);
  T(std::vector<Qarg> qbits, std::vector<std::shared_ptr<Parameter>> params);
  virtual size_t numQubits() const override;
  virtual size_t numParams() const override;
  virtual std::string name() const override;
};

class TDG : public Qgate {
public:
  TDG(std::vector<Qarg> qbits);
  TDG(std::vector<Qarg> qbits, std::vector<std::shared_ptr<Parameter>> params);
  virtual size_t numQubits() const override;
  virtual size_t numParams() const override;
  virtual std::string name() const override;
};

class SX : public Qgate {
public:
  SX(std::vector<Qarg> qbits);
  SX(std::vector<Qarg> qbits, std::vector<std::shared_ptr<Parameter>> params);
  virtual size_t numQubits() const override;
  virtual size_t numParams() const override;
  virtual std::string name() const override;
};

class SXDG : public Qgate {
public:
  SXDG(std::vector<Qarg> qbits);
  SXDG(std::vector<Qarg> qbits, std::vector<std::shared_ptr<Parameter>> params);
  virtual size_t numQubits() const override;
  virtual size_t numParams() const override;
  virtual std::string name() const override;
};

class P : public Qgate {
public:
  P(std::vector<Qarg> qbits);
  P(std::vector<Qarg> qbits, std::vector<std::shared_ptr<Parameter>> params);
  virtual size_t numQubits() const override;
  virtual size_t numParams() const override;
  virtual std::string name() const override;
};

class RX : public Qgate {
public:
  RX(std::vector<Qarg> qbits);
  RX(std::vector<Qarg> qbits, std::vector<std::shared_ptr<Parameter>> params);
  virtual size_t numQubits() const override;
  virtual size_t numParams() const override;
  virtual std::string name() const override;
};

class RY : public Qgate {
public:
  RY(std::vector<Qarg> qbits);
  RY(std::vector<Qarg> qbits, std::vector<std::shared_ptr<Parameter>> params);
  virtual size_t numQubits() const override;
  virtual size_t numParams() const override;
  virtual std::string name() const override;
};

class RZ : public Qgate {
public:
  RZ(std::vector<Qarg> qbits);
  RZ(std::vector<Qarg> qbits, std::vector<std::shared_ptr<Parameter>> params);
  virtual size_t numQubits() const override;
  virtual size_t numParams() const override;
  virtual std::string name() const override;
};

class U0 : public Qgate {
public:
  U0(std::vector<Qarg> qbits);
  U0(std::vector<Qarg> qbits, std::vector<std::shared_ptr<Parameter>> params);
  virtual size_t numQubits() const override;
  virtual size_t numParams() const override;
  virtual std::string name() const override;
};

class U1 : public Qgate {
public:
  U1(std::vector<Qarg> qbits);
  U1(std::vector<Qarg> qbits, std::vector<std::shared_ptr<Parameter>> params);
  virtual size_t numQubits() const override;
  virtual size_t numParams() const override;
  virtual std::string name() const override;
};

class U2 : public Qgate {
public:
  U2(std::vector<Qarg> qbits);
  U2(std::vector<Qarg> qbits, std::vector<std::shared_ptr<Parameter>> params);
  virtual size_t numQubits() const override;
  virtual size_t numParams() const override;
  virtual std::string name() const override;
};

class U3 : public Qgate {
public:
  U3(std::vector<Qarg> qbits);
  U3(std::vector<Qarg> qbits, std::vector<std::shared_ptr<Parameter>> params);
  virtual size_t numQubits() const override;
  virtual size_t numParams() const override;
  virtual std::string name() const override;
};

class U : public Qgate {
public:
  U(std::vector<Qarg> qbits);
  U(std::vector<Qarg> qbits, std::vector<std::shared_ptr<Parameter>> params);
  virtual size_t numQubits() const override;
  virtual size_t numParams() const override;
  virtual std::string name() const override;
};

class CX : public Qgate {
public:
  CX(std::vector<Qarg> qbits);
  CX(std::vector<Qarg> qbits, std::vector<std::shared_ptr<Parameter>> params);
  virtual size_t numQubits() const override;
  virtual size_t numParams() const override;
  virtual std::string name() const override;
};

class CZ : public Qgate {
public:
  CZ(std::vector<Qarg> qbits);
  CZ(std::vector<Qarg> qbits, std::vector<std::shared_ptr<Parameter>> params);
  virtual size_t numQubits() const override;
  virtual size_t numParams() const override;
  virtual std::string name() const override;
};

class CY : public Qgate {
public:
  CY(std::vector<Qarg> qbits);
  CY(std::vector<Qarg> qbits, std::vector<std::shared_ptr<Parameter>> params);
  virtual size_t numQubits() const override;
  virtual size_t numParams() const override;
  virtual std::string name() const override;
};

class SWAP : public Qgate {
public:
  SWAP(std::vector<Qarg> qbits);
  SWAP(std::vector<Qarg> qbits, std::vector<std::shared_ptr<Parameter>> params);
  virtual size_t numQubits() const override;
  virtual size_t numParams() const override;
  virtual std::string name() const override;
};

class CH : public Qgate {
public:
  CH(std::vector<Qarg> qbits);
  CH(std::vector<Qarg> qbits, std::vector<std::shared_ptr<Parameter>> params);
  virtual size_t numQubits() const override;
  virtual size_t numParams() const override;
  virtual std::string name() const override;
};

class CSX : public Qgate {
public:
  CSX(std::vector<Qarg> qbits);
  CSX(std::vector<Qarg> qbits, std::vector<std::shared_ptr<Parameter>> params);
  virtual size_t numQubits() const override;
  virtual size_t numParams() const override;
  virtual std::string name() const override;
};

class CRX : public Qgate {
public:
  CRX(std::vector<Qarg> qbits);
  CRX(std::vector<Qarg> qbits, std::vector<std::shared_ptr<Parameter>> params);
  virtual size_t numQubits() const override;
  virtual size_t numParams() const override;
  virtual std::string name() const override;
};

class CRY : public Qgate {
public:
  CRY(std::vector<Qarg> qbits);
  CRY(std::vector<Qarg> qbits, std::vector<std::shared_ptr<Parameter>> params);
  virtual size_t numQubits() const override;
  virtual size_t numParams() const override;
  virtual std::string name() const override;
};

class CRZ : public Qgate {
public:
  CRZ(std::vector<Qarg> qbits);
  CRZ(std::vector<Qarg> qbits, std::vector<std::shared_ptr<Parameter>> params);
  virtual size_t numQubits() const override;
  virtual size_t numParams() const override;
  virtual std::string name() const override;
};

class CU1 : public Qgate {
public:
  CU1(std::vector<Qarg> qbits);
  CU1(std::vector<Qarg> qbits, std::vector<std::shared_ptr<Parameter>> params);
  virtual size_t numQubits() const override;
  virtual size_t numParams() const override;
  virtual std::string name() const override;
};

class CP : public Qgate {
public:
  CP(std::vector<Qarg> qbits);
  CP(std::vector<Qarg> qbits, std::vector<std::shared_ptr<Parameter>> params);
  virtual size_t numQubits() const override;
  virtual size_t numParams() const override;
  virtual std::string name() const override;
};

class RXX : public Qgate {
public:
  RXX(std::vector<Qarg> qbits);
  RXX(std::vector<Qarg> qbits, std::vector<std::shared_ptr<Parameter>> params);
  virtual size_t numQubits() const override;
  virtual size_t numParams() const override;
  virtual std::string name() const override;
};

class RZZ : public Qgate {
public:
  RZZ(std::vector<Qarg> qbits);
  RZZ(std::vector<Qarg> qbits, std::vector<std::shared_ptr<Parameter>> params);
  virtual size_t numQubits() const override;
  virtual size_t numParams() const override;
  virtual std::string name() const override;
};

class CU3 : public Qgate {
public:
  CU3(std::vector<Qarg> qbits);
  CU3(std::vector<Qarg> qbits, std::vector<std::shared_ptr<Parameter>> params);
  virtual size_t numQubits() const override;
  virtual size_t numParams() const override;
  virtual std::string name() const override;
};

class CU : public Qgate {
public:
  CU(std::vector<Qarg> qbits);
  CU(std::vector<Qarg> qbits, std::vector<std::shared_ptr<Parameter>> params);
  virtual size_t numQubits() const override;
  virtual size_t numParams() const override;
  virtual std::string name() const override;
};

class CCX : public Qgate {
public:
  CCX(std::vector<Qarg> qbits);
  CCX(std::vector<Qarg> qbits, std::vector<std::shared_ptr<Parameter>> params);
  virtual size_t numQubits() const override;
  virtual size_t numParams() const override;
  virtual std::string name() const override;
};

class CSWAP : public Qgate {
public:
  CSWAP(std::vector<Qarg> qbits);
  CSWAP(std::vector<Qarg> qbits,
        std::vector<std::shared_ptr<Parameter>> params);
  virtual size_t numQubits() const override;
  virtual size_t numParams() const override;
  virtual std::string name() const override;
};

class RCCX : public Qgate {
public:
  RCCX(std::vector<Qarg> qbits);
  RCCX(std::vector<Qarg> qbits, std::vector<std::shared_ptr<Parameter>> params);
  virtual size_t numQubits() const override;
  virtual size_t numParams() const override;
  virtual std::string name() const override;
};

class RC3X : public Qgate {
public:
  RC3X(std::vector<Qarg> qbits);
  RC3X(std::vector<Qarg> qbits, std::vector<std::shared_ptr<Parameter>> params);
  virtual size_t numQubits() const override;
  virtual size_t numParams() const override;
  virtual std::string name() const override;
};

class C3X : public Qgate {
public:
  C3X(std::vector<Qarg> qbits);
  C3X(std::vector<Qarg> qbits, std::vector<std::shared_ptr<Parameter>> params);
  virtual size_t numQubits() const override;
  virtual size_t numParams() const override;
  virtual std::string name() const override;
};

class C3SQRTX : public Qgate {
public:
  C3SQRTX(std::vector<Qarg> qbits);
  C3SQRTX(std::vector<Qarg> qbits,
          std::vector<std::shared_ptr<Parameter>> params);
  virtual size_t numQubits() const override;
  virtual size_t numParams() const override;
  virtual std::string name() const override;
};

class C4X : public Qgate {
public:
  C4X(std::vector<Qarg> qbits);
  C4X(std::vector<Qarg> qbits, std::vector<std::shared_ptr<Parameter>> params);
  virtual size_t numQubits() const override;
  virtual size_t numParams() const override;
  virtual std::string name() const override;
};

} // namespace snuqs
#endif //__QOP_H__
