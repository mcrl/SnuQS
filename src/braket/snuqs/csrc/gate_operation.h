#ifndef _GATE_OPERATION_H_
#define _GATE_OPERATION_H_

#include "state_vector.h"
#include <complex>
#include <map>
#include <vector>

class GateOperation {
public:
  virtual std::complex<double> *data() = 0;
  virtual size_t dim() const = 0;
  virtual std::vector<size_t> shape() const = 0;
  virtual std::vector<size_t> stride() const = 0;
  std::complex<double> *data_;
};

class OneQubitGate : public GateOperation {
public:
  OneQubitGate();
  virtual ~OneQubitGate();
  virtual std::complex<double> *data() override;
  virtual size_t dim() const override;
  virtual std::vector<size_t> shape() const override;
  virtual std::vector<size_t> stride() const override;
};

class TwoQubitGate : public GateOperation {
public:
  TwoQubitGate();
  virtual ~TwoQubitGate();
  virtual std::complex<double> *data() override;
  virtual size_t dim() const override;
  virtual std::vector<size_t> shape() const override;
  virtual std::vector<size_t> stride() const override;
};

class ThreeQubitGate : public GateOperation {
public:
  ThreeQubitGate();
  virtual ~ThreeQubitGate();
  virtual std::complex<double> *data() override;
  virtual size_t dim() const override;
  virtual std::vector<size_t> shape() const override;
  virtual std::vector<size_t> stride() const override;
};

class ControlledGate : public GateOperation {
public:
  ControlledGate();
  virtual ~ControlledGate();
  virtual std::complex<double> *data() override;
  virtual size_t dim() const override;
  virtual std::vector<size_t> shape() const override;
  virtual std::vector<size_t> stride() const override;
};

class Identity : public OneQubitGate {
public:
  Identity();
  ~Identity();
};

class Hadamard : public OneQubitGate {
public:
  Hadamard();
  ~Hadamard();
};

class PauliX : public OneQubitGate {
public:
  PauliX();
  ~PauliX();
};

class PauliY : public OneQubitGate {
public:
  PauliY();
  ~PauliY();
};

class PauliZ : public OneQubitGate {
public:
  PauliZ();
  ~PauliZ();
};

class CX : public ControlledGate {
public:
  CX();
  ~CX();
};

class CY : public ControlledGate {
public:
  CY();
  ~CY();
};

class CZ : public ControlledGate {
public:
  CZ();
  ~CZ();
};

class S : public OneQubitGate {
public:
  S();
  ~S();
};

class Si : public OneQubitGate {
public:
  Si();
  ~Si();
};

class T : public OneQubitGate {
public:
  T();
  ~T();
};

class Ti : public OneQubitGate {
public:
  Ti();
  ~Ti();
};

class V : public OneQubitGate {
public:
  V();
  ~V();
};

class Vi : public OneQubitGate {
public:
  Vi();
  ~Vi();
};

class PhaseShift : public OneQubitGate {
public:
  PhaseShift(double angle);
  ~PhaseShift();
  double angle_;
};

class CPhaseShift : public ControlledGate {
public:
  CPhaseShift(double angle);
  ~CPhaseShift();
  double angle_;
};

class CPhaseShift00 : public ControlledGate {
public:
  CPhaseShift00(double angle);
  ~CPhaseShift00();
  double angle_;
};

class CPhaseShift01 : public ControlledGate {
public:
  CPhaseShift01(double angle);
  ~CPhaseShift01();
  double angle_;
};

class CPhaseShift10 : public ControlledGate {
public:
  CPhaseShift10(double angle);
  ~CPhaseShift10();
  double angle_;
};

class RotX : public OneQubitGate {
public:
  RotX(double angle);
  ~RotX();
  double angle_;
};

class RotY : public OneQubitGate {
public:
  RotY(double angle);
  ~RotY();
  double angle_;
};

class RotZ : public OneQubitGate {
public:
  RotZ(double angle);
  ~RotZ();
  double angle_;
};

class Swap : public TwoQubitGate {
public:
  Swap();
  ~Swap();
};

class ISwap : public TwoQubitGate {
public:
  ISwap();
  ~ISwap();
};

class PSwap : public TwoQubitGate {
public:
  PSwap(double angle);
  ~PSwap();
  double angle_;
};

class XY : public TwoQubitGate {
public:
  XY(double angle);
  ~XY();
  double angle_;
};

class XX : public TwoQubitGate {
public:
  XX(double angle);
  ~XX();
  double angle_;
};

class YY : public TwoQubitGate {
public:
  YY(double angle);
  ~YY();
  double angle_;
};

class ZZ : public TwoQubitGate {
public:
  ZZ(double angle);
  ~ZZ();
  double angle_;
};

class CCNot : public ThreeQubitGate {
public:
  CCNot();
  ~CCNot();
};

class CSwap : public ThreeQubitGate {
public:
  CSwap();
  ~CSwap();
};

class Unitary : public GateOperation {
public:
  Unitary();
  ~Unitary();
  virtual std::complex<double> *data() override;
  virtual size_t dim() const override;
  virtual std::vector<size_t> shape() const override;
  virtual std::vector<size_t> stride() const override;
};

class U : public OneQubitGate {
public:
  U(double theta, double phi, double lambda);
  ~U();
  double theta_;
  double phi_;
  double lambda_;
};

class GPhase : public OneQubitGate {
public:
  GPhase(double angle);
  ~GPhase();
  double angle_;
};
#endif //_GATE_OPERATION_H_
