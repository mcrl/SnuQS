#ifndef _GATE_OPERATIONS_H_
#define _GATE_OPERATIONS_H_

#include <vector>

#include "operation/operation.h"

class Identity : public GateOperation {
 public:
  Identity(std::vector<size_t> targets, std::vector<size_t> ctrl_modifiers,
           size_t power);
  virtual ~Identity();
};

class Hadamard : public GateOperation {
 public:
  Hadamard(std::vector<size_t> targets, std::vector<size_t> ctrl_modifiers,
           size_t power);
  virtual ~Hadamard();
};

class PauliX : public GateOperation {
 public:
  PauliX(std::vector<size_t> targets, std::vector<size_t> ctrl_modifiers,
         size_t power);
  virtual ~PauliX();
};

class PauliY : public GateOperation {
 public:
  PauliY(std::vector<size_t> targets, std::vector<size_t> ctrl_modifiers,
         size_t power);
  virtual ~PauliY();
};

class PauliZ : public GateOperation {
 public:
  PauliZ(std::vector<size_t> targets, std::vector<size_t> ctrl_modifiers,
         size_t power);
  virtual ~PauliZ();
};

class CX : public GateOperation {
 public:
  CX(std::vector<size_t> targets, std::vector<size_t> ctrl_modifiers,
     size_t power);
  virtual ~CX();
};

class CY : public GateOperation {
 public:
  CY(std::vector<size_t> targets, std::vector<size_t> ctrl_modifiers,
     size_t power);
  virtual ~CY();
};

class CZ : public GateOperation {
 public:
  CZ(std::vector<size_t> targets, std::vector<size_t> ctrl_modifiers,
     size_t power);
  virtual ~CZ();
};

class S : public GateOperation {
 public:
  S(std::vector<size_t> targets, std::vector<size_t> ctrl_modifiers,
    size_t power);
  virtual ~S();
};

class Si : public GateOperation {
 public:
  Si(std::vector<size_t> targets, std::vector<size_t> ctrl_modifiers,
     size_t power);
  virtual ~Si();
};

class T : public GateOperation {
 public:
  T(std::vector<size_t> targets, std::vector<size_t> ctrl_modifiers,
    size_t power);
  virtual ~T();
};

class Ti : public GateOperation {
 public:
  Ti(std::vector<size_t> targets, std::vector<size_t> ctrl_modifiers,
     size_t power);
  virtual ~Ti();
};

class V : public GateOperation {
 public:
  V(std::vector<size_t> targets, std::vector<size_t> ctrl_modifiers,
    size_t power);
  virtual ~V();
};

class Vi : public GateOperation {
 public:
  Vi(std::vector<size_t> targets, std::vector<size_t> ctrl_modifiers,
     size_t power);
  virtual ~Vi();
};

class PhaseShift : public GateOperation {
 public:
  PhaseShift(std::vector<size_t> targets, double angle,
             std::vector<size_t> ctrl_modifiers, size_t power);
  virtual ~PhaseShift();
  double angle_;
};

class CPhaseShift : public GateOperation {
 public:
  CPhaseShift(std::vector<size_t> targets, double angle,
              std::vector<size_t> ctrl_modifiers, size_t power);
  virtual ~CPhaseShift();
  double angle_;
};

class CPhaseShift00 : public GateOperation {
 public:
  CPhaseShift00(std::vector<size_t> targets, double angle,
                std::vector<size_t> ctrl_modifiers, size_t power);
  virtual ~CPhaseShift00();
  double angle_;
};

class CPhaseShift01 : public GateOperation {
 public:
  CPhaseShift01(std::vector<size_t> targets, double angle,
                std::vector<size_t> ctrl_modifiers, size_t power);
  virtual ~CPhaseShift01();
  double angle_;
};

class CPhaseShift10 : public GateOperation {
 public:
  CPhaseShift10(std::vector<size_t> targets, double angle,
                std::vector<size_t> ctrl_modifiers, size_t power);
  virtual ~CPhaseShift10();
  double angle_;
};

class RotX : public GateOperation {
 public:
  RotX(std::vector<size_t> targets, double angle,
       std::vector<size_t> ctrl_modifiers, size_t power);
  virtual ~RotX();
  double angle_;
};

class RotY : public GateOperation {
 public:
  RotY(std::vector<size_t> targets, double angle,
       std::vector<size_t> ctrl_modifiers, size_t power);
  virtual ~RotY();
  double angle_;
};

class RotZ : public GateOperation {
 public:
  RotZ(std::vector<size_t> targets, double angle,
       std::vector<size_t> ctrl_modifiers, size_t power);
  virtual ~RotZ();
  double angle_;
};

class Swap : public GateOperation {
 public:
  Swap(std::vector<size_t> targets, std::vector<size_t> ctrl_modifiers,
       size_t power);
  virtual ~Swap();
};

class ISwap : public GateOperation {
 public:
  ISwap(std::vector<size_t> targets, std::vector<size_t> ctrl_modifiers,
        size_t power);
  virtual ~ISwap();
};

class PSwap : public GateOperation {
 public:
  PSwap(std::vector<size_t> targets, double angle,
        std::vector<size_t> ctrl_modifiers, size_t power);
  virtual ~PSwap();
  double angle_;
};

class XY : public GateOperation {
 public:
  XY(std::vector<size_t> targets, double angle,
     std::vector<size_t> ctrl_modifiers, size_t power);
  virtual ~XY();
  double angle_;
};

class XX : public GateOperation {
 public:
  XX(std::vector<size_t> targets, double angle,
     std::vector<size_t> ctrl_modifiers, size_t power);
  virtual ~XX();
  double angle_;
};

class YY : public GateOperation {
 public:
  YY(std::vector<size_t> targets, double angle,
     std::vector<size_t> ctrl_modifiers, size_t power);
  virtual ~YY();
  double angle_;
};

class ZZ : public GateOperation {
 public:
  ZZ(std::vector<size_t> targets, double angle,
     std::vector<size_t> ctrl_modifiers, size_t power);
  virtual ~ZZ();
  double angle_;
};

class CCNot : public GateOperation {
 public:
  CCNot(std::vector<size_t> targets, std::vector<size_t> ctrl_modifiers,
        size_t power);
  virtual ~CCNot();
};

class CSwap : public GateOperation {
 public:
  CSwap(std::vector<size_t> targets, std::vector<size_t> ctrl_modifiers,
        size_t power);
  virtual ~CSwap();
};

class U : public GateOperation {
 public:
  U(std::vector<size_t> targets, double theta, double phi, double lambda,
    std::vector<size_t> ctrl_modifiers, size_t power);
  virtual ~U();
  double theta_;
  double phi_;
  double lambda_;
};

class GPhase : public GateOperation {
 public:
  GPhase(std::vector<size_t> targets, double angle,
         std::vector<size_t> ctrl_modifiers, size_t power);
  virtual ~GPhase();
  double angle_;
};

#endif  //_GATE_OPERATION_H_
