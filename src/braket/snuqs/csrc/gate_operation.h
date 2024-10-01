#ifndef _GATE_OPERATION_H_
#define _GATE_OPERATION_H_

#include <vector>

#include "operation.h"

class OneQubitGate : public GateOperation {
 public:
  OneQubitGate();
  virtual ~OneQubitGate();
  virtual size_t dim() const override;
  virtual std::vector<size_t> shape() const override;
  virtual std::vector<size_t> stride() const override;
};

class ThreeQubitGate : public GateOperation {
 public:
  ThreeQubitGate();
  virtual ~ThreeQubitGate();
  virtual size_t dim() const override;
  virtual std::vector<size_t> shape() const override;
  virtual std::vector<size_t> stride() const override;
};

class TwoQubitGate : public GateOperation {
 public:
  TwoQubitGate();
  virtual ~TwoQubitGate();
  virtual size_t dim() const override;
  virtual std::vector<size_t> shape() const override;
  virtual std::vector<size_t> stride() const override;
};

class Identity : public OneQubitGate {
 public:
  Identity();
  virtual ~Identity();
};

class Hadamard : public OneQubitGate {
 public:
  Hadamard();
  virtual ~Hadamard();
};

class PauliX : public OneQubitGate {
 public:
  PauliX();
  virtual ~PauliX();
};

class PauliY : public OneQubitGate {
 public:
  PauliY();
  virtual ~PauliY();
};

class PauliZ : public OneQubitGate {
 public:
  PauliZ();
  virtual ~PauliZ();
};

class CX : public TwoQubitGate {
 public:
  CX();
  virtual ~CX();
};

class CY : public TwoQubitGate {
 public:
  CY();
  virtual ~CY();
};

class CZ : public TwoQubitGate {
 public:
  CZ();
  virtual ~CZ();
};

class S : public OneQubitGate {
 public:
  S();
  virtual ~S();
};

class Si : public OneQubitGate {
 public:
  Si();
  virtual ~Si();
};

class T : public OneQubitGate {
 public:
  T();
  virtual ~T();
};

class Ti : public OneQubitGate {
 public:
  Ti();
  virtual ~Ti();
};

class V : public OneQubitGate {
 public:
  V();
  virtual ~V();
};

class Vi : public OneQubitGate {
 public:
  Vi();
  virtual ~Vi();
};

class PhaseShift : public OneQubitGate {
 public:
  PhaseShift(double angle);
  virtual ~PhaseShift();
  double angle_;
};

class CPhaseShift : public TwoQubitGate {
 public:
  CPhaseShift(double angle);
  virtual ~CPhaseShift();
  double angle_;
};

class CPhaseShift00 : public TwoQubitGate {
 public:
  CPhaseShift00(double angle);
  virtual ~CPhaseShift00();
  double angle_;
};

class CPhaseShift01 : public TwoQubitGate {
 public:
  CPhaseShift01(double angle);
  virtual ~CPhaseShift01();
  double angle_;
};

class CPhaseShift10 : public TwoQubitGate {
 public:
  CPhaseShift10(double angle);
  virtual ~CPhaseShift10();
  double angle_;
};

class RotX : public OneQubitGate {
 public:
  RotX(double angle);
  virtual ~RotX();
  double angle_;
};

class RotY : public OneQubitGate {
 public:
  RotY(double angle);
  virtual ~RotY();
  double angle_;
};

class RotZ : public OneQubitGate {
 public:
  RotZ(double angle);
  virtual ~RotZ();
  double angle_;
};

class Swap : public TwoQubitGate {
 public:
  Swap();
  virtual ~Swap();
};

class ISwap : public TwoQubitGate {
 public:
  ISwap();
  virtual ~ISwap();
};

class PSwap : public TwoQubitGate {
 public:
  PSwap(double angle);
  virtual ~PSwap();
  double angle_;
};

class XY : public TwoQubitGate {
 public:
  XY(double angle);
  virtual ~XY();
  double angle_;
};

class XX : public TwoQubitGate {
 public:
  XX(double angle);
  virtual ~XX();
  double angle_;
};

class YY : public TwoQubitGate {
 public:
  YY(double angle);
  virtual ~YY();
  double angle_;
};

class ZZ : public TwoQubitGate {
 public:
  ZZ(double angle);
  virtual ~ZZ();
  double angle_;
};

class CCNot : public ThreeQubitGate {
 public:
  CCNot();
  virtual ~CCNot();
};

class CSwap : public ThreeQubitGate {
 public:
  CSwap();
  virtual ~CSwap();
};

class U : public OneQubitGate {
 public:
  U(double theta, double phi, double lambda);
  virtual ~U();
  double theta_;
  double phi_;
  double lambda_;
};

class GPhase : public OneQubitGate {
 public:
  GPhase(double angle);
  virtual ~GPhase();
  double angle_;
};

class Unitary : public GateOperation {
 public:
  Unitary();
  virtual ~Unitary();
};
#endif  //_GATE_OPERATION_H_
