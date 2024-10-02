#ifndef _GATE_OPERATION_H_
#define _GATE_OPERATION_H_

#include <vector>

#include "operation.h"

class Identity : public GateOperation {
 public:
  Identity(const std::vector<size_t> &targets,
           const std::vector<size_t> &ctrl_modifiers, size_t power);
  virtual ~Identity();
};

class Hadamard : public GateOperation {
 public:
  Hadamard(const std::vector<size_t> &targets,
           const std::vector<size_t> &ctrl_modifiers, size_t power);
  virtual ~Hadamard();
};

class PauliX : public GateOperation {
 public:
  PauliX(const std::vector<size_t> &targets,
         const std::vector<size_t> &ctrl_modifiers, size_t power);
  virtual ~PauliX();
};

class PauliY : public GateOperation {
 public:
  PauliY(const std::vector<size_t> &targets,
         const std::vector<size_t> &ctrl_modifiers, size_t power);
  virtual ~PauliY();
};

class PauliZ : public GateOperation {
 public:
  PauliZ(const std::vector<size_t> &targets,
         const std::vector<size_t> &ctrl_modifiers, size_t power);
  virtual ~PauliZ();
};

class CX : public GateOperation {
 public:
  CX(const std::vector<size_t> &targets,
     const std::vector<size_t> &ctrl_modifiers, size_t power);
  virtual ~CX();
};

class CY : public GateOperation {
 public:
  CY(const std::vector<size_t> &targets,
     const std::vector<size_t> &ctrl_modifiers, size_t power);
  virtual ~CY();
};

class CZ : public GateOperation {
 public:
  CZ(const std::vector<size_t> &targets,
     const std::vector<size_t> &ctrl_modifiers, size_t power);
  virtual ~CZ();
};

class S : public GateOperation {
 public:
  S(const std::vector<size_t> &targets,
    const std::vector<size_t> &ctrl_modifiers, size_t power);
  virtual ~S();
};

class Si : public GateOperation {
 public:
  Si(const std::vector<size_t> &targets,
     const std::vector<size_t> &ctrl_modifiers, size_t power);
  virtual ~Si();
};

class T : public GateOperation {
 public:
  T(const std::vector<size_t> &targets,
    const std::vector<size_t> &ctrl_modifiers, size_t power);
  virtual ~T();
};

class Ti : public GateOperation {
 public:
  Ti(const std::vector<size_t> &targets,
     const std::vector<size_t> &ctrl_modifiers, size_t power);
  virtual ~Ti();
};

class V : public GateOperation {
 public:
  V(const std::vector<size_t> &targets,
    const std::vector<size_t> &ctrl_modifiers, size_t power);
  virtual ~V();
};

class Vi : public GateOperation {
 public:
  Vi(const std::vector<size_t> &targets,
     const std::vector<size_t> &ctrl_modifiers, size_t power);
  virtual ~Vi();
};

class PhaseShift : public GateOperation {
 public:
  PhaseShift(const std::vector<size_t> &targets, double angle,
             const std::vector<size_t> &ctrl_modifiers, size_t power);
  virtual ~PhaseShift();
  double angle_;
};

class CPhaseShift : public GateOperation {
 public:
  CPhaseShift(const std::vector<size_t> &targets, double angle,
              const std::vector<size_t> &ctrl_modifiers, size_t power);
  virtual ~CPhaseShift();
  double angle_;
};

class CPhaseShift00 : public GateOperation {
 public:
  CPhaseShift00(const std::vector<size_t> &targets, double angle,
                const std::vector<size_t> &ctrl_modifiers, size_t power);
  virtual ~CPhaseShift00();
  double angle_;
};

class CPhaseShift01 : public GateOperation {
 public:
  CPhaseShift01(const std::vector<size_t> &targets, double angle,
                const std::vector<size_t> &ctrl_modifiers, size_t power);
  virtual ~CPhaseShift01();
  double angle_;
};

class CPhaseShift10 : public GateOperation {
 public:
  CPhaseShift10(const std::vector<size_t> &targets, double angle,
                const std::vector<size_t> &ctrl_modifiers, size_t power);
  virtual ~CPhaseShift10();
  double angle_;
};

class RotX : public GateOperation {
 public:
  RotX(const std::vector<size_t> &targets, double angle,
       const std::vector<size_t> &ctrl_modifiers, size_t power);
  virtual ~RotX();
  double angle_;
};

class RotY : public GateOperation {
 public:
  RotY(const std::vector<size_t> &targets, double angle,
       const std::vector<size_t> &ctrl_modifiers, size_t power);
  virtual ~RotY();
  double angle_;
};

class RotZ : public GateOperation {
 public:
  RotZ(const std::vector<size_t> &targets, double angle,
       const std::vector<size_t> &ctrl_modifiers, size_t power);
  virtual ~RotZ();
  double angle_;
};

class Swap : public GateOperation {
 public:
  Swap(const std::vector<size_t> &targets,
       const std::vector<size_t> &ctrl_modifiers, size_t power);
  virtual ~Swap();
};

class ISwap : public GateOperation {
 public:
  ISwap(const std::vector<size_t> &targets,
        const std::vector<size_t> &ctrl_modifiers, size_t power);
  virtual ~ISwap();
};

class PSwap : public GateOperation {
 public:
  PSwap(const std::vector<size_t> &targets, double angle,
        const std::vector<size_t> &ctrl_modifiers, size_t power);
  virtual ~PSwap();
  double angle_;
};

class XY : public GateOperation {
 public:
  XY(const std::vector<size_t> &targets, double angle,
     const std::vector<size_t> &ctrl_modifiers, size_t power);
  virtual ~XY();
  double angle_;
};

class XX : public GateOperation {
 public:
  XX(const std::vector<size_t> &targets, double angle,
     const std::vector<size_t> &ctrl_modifiers, size_t power);
  virtual ~XX();
  double angle_;
};

class YY : public GateOperation {
 public:
  YY(const std::vector<size_t> &targets, double angle,
     const std::vector<size_t> &ctrl_modifiers, size_t power);
  virtual ~YY();
  double angle_;
};

class ZZ : public GateOperation {
 public:
  ZZ(const std::vector<size_t> &targets, double angle,
     const std::vector<size_t> &ctrl_modifiers, size_t power);
  virtual ~ZZ();
  double angle_;
};

class CCNot : public GateOperation {
 public:
  CCNot(const std::vector<size_t> &targets,
        const std::vector<size_t> &ctrl_modifiers, size_t power);
  virtual ~CCNot();
};

class CSwap : public GateOperation {
 public:
  CSwap(const std::vector<size_t> &targets,
        const std::vector<size_t> &ctrl_modifiers, size_t power);
  virtual ~CSwap();
};

class U : public GateOperation {
 public:
  U(const std::vector<size_t> &targets, double theta, double phi, double lambda,
    const std::vector<size_t> &ctrl_modifiers, size_t power);
  virtual ~U();
  double theta_;
  double phi_;
  double lambda_;
};

class GPhase : public GateOperation {
 public:
  GPhase(const std::vector<size_t> &targets, double angle,
         const std::vector<size_t> &ctrl_modifiers, size_t power);
  virtual ~GPhase();
  double angle_;
};

#endif  //_GATE_OPERATION_H_
