#ifndef _GATE_OPERATION_H_
#define _GATE_OPERATION_H_

#include "state_vector.h"
#include <complex>
#include <map>
#include <vector>

class GateOperation {
public:
  virtual void run(StateVector &sv) = 0;
};

class Identity : public GateOperation {
public:
  virtual void run(StateVector &sv) override;
};

class Hadamard : public GateOperation {
public:
  virtual void run(StateVector &sv) override;
};

class PauliX : public GateOperation {
public:
  virtual void run(StateVector &sv) override;
};

class PauliY : public GateOperation {
public:
  virtual void run(StateVector &sv) override;
};

class PauliZ : public GateOperation {
public:
  virtual void run(StateVector &sv) override;
};

class CV : public GateOperation {
public:
  virtual void run(StateVector &sv) override;
};

class CX : public GateOperation {
public:
  virtual void run(StateVector &sv) override;
};

class CY : public GateOperation {
public:
  virtual void run(StateVector &sv) override;
};

class CZ : public GateOperation {
public:
  virtual void run(StateVector &sv) override;
};

class ECR : public GateOperation {
public:
  virtual void run(StateVector &sv) override;
};

class S : public GateOperation {
public:
  virtual void run(StateVector &sv) override;
};

class Si : public GateOperation {
public:
  virtual void run(StateVector &sv) override;
};

class T : public GateOperation {
public:
  virtual void run(StateVector &sv) override;
};

class Ti : public GateOperation {
public:
  virtual void run(StateVector &sv) override;
};

class V : public GateOperation {
public:
  virtual void run(StateVector &sv) override;
};

class Vi : public GateOperation {
public:
  virtual void run(StateVector &sv) override;
};

class PhaseShift : public GateOperation {
public:
  virtual void run(StateVector &sv) override;
};

class CPhaseShift : public GateOperation {
public:
  virtual void run(StateVector &sv) override;
};

class CPhaseShift00 : public GateOperation {
public:
  virtual void run(StateVector &sv) override;
};

class CPhaseShift01 : public GateOperation {
public:
  virtual void run(StateVector &sv) override;
};

class CPhaseShift10 : public GateOperation {
public:
  virtual void run(StateVector &sv) override;
};

class RotX : public GateOperation {
public:
  virtual void run(StateVector &sv) override;
};

class RotY : public GateOperation {
public:
  virtual void run(StateVector &sv) override;
};

class RotZ : public GateOperation {
public:
  virtual void run(StateVector &sv) override;
};

class Swap : public GateOperation {
public:
  virtual void run(StateVector &sv) override;
};

class ISwap : public GateOperation {
public:
  virtual void run(StateVector &sv) override;
};

class PSwap : public GateOperation {
public:
  virtual void run(StateVector &sv) override;
};

class XY : public GateOperation {
public:
  virtual void run(StateVector &sv) override;
};

class XX : public GateOperation {
public:
  virtual void run(StateVector &sv) override;
};

class YY : public GateOperation {
public:
  virtual void run(StateVector &sv) override;
};

class ZZ : public GateOperation {
public:
  virtual void run(StateVector &sv) override;
};

class CCNot : public GateOperation {
public:
  virtual void run(StateVector &sv) override;
};

class CSwap : public GateOperation {
public:
  virtual void run(StateVector &sv) override;
};

class PRx : public GateOperation {
public:
  virtual void run(StateVector &sv) override;
};

class Unitary : public GateOperation {
public:
  virtual void run(StateVector &sv) override;
};

class U : public GateOperation {
public:
  virtual void run(StateVector &sv) override;
};

class GPhase : public GateOperation {
public:
  virtual void run(StateVector &sv) override;
};

#endif //_GATE_OPERATION_H_
