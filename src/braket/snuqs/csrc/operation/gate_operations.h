#ifndef _GATE_OPERATIONS_H_
#define _GATE_OPERATIONS_H_

#include <string>
#include <vector>

#include "operation/operation.h"

class Identity : public GateOperation {
 public:
  Identity(std::vector<size_t> targets, std::vector<size_t> ctrl_modifiers,
           size_t power);
  virtual ~Identity();
  virtual std::string name() const override;
  virtual bool sliceable() const override;
};

class Hadamard : public GateOperation {
 public:
  Hadamard(std::vector<size_t> targets, std::vector<size_t> ctrl_modifiers,
           size_t power);
  virtual ~Hadamard();
  virtual std::string name() const override;
  virtual bool sliceable() const override;
};

class PauliX : public GateOperation {
 public:
  PauliX(std::vector<size_t> targets, std::vector<size_t> ctrl_modifiers,
         size_t power);
  virtual ~PauliX();
  virtual std::string name() const override;
  virtual bool sliceable() const override;
};

class PauliY : public GateOperation {
 public:
  PauliY(std::vector<size_t> targets, std::vector<size_t> ctrl_modifiers,
         size_t power);
  virtual ~PauliY();
  virtual std::string name() const override;
  virtual bool sliceable() const override;
};

class PauliZ : public GateOperation {
 public:
  PauliZ(std::vector<size_t> targets, std::vector<size_t> ctrl_modifiers,
         size_t power);
  virtual ~PauliZ();
  virtual std::string name() const override;
  virtual bool sliceable() const override;
};

class CX : public GateOperation {
 public:
  CX(std::vector<size_t> targets, std::vector<size_t> ctrl_modifiers,
     size_t power);
  virtual ~CX();
  virtual std::string name() const override;
  virtual bool sliceable() const override;
};

class CY : public GateOperation {
 public:
  CY(std::vector<size_t> targets, std::vector<size_t> ctrl_modifiers,
     size_t power);
  virtual ~CY();
  virtual std::string name() const override;
  virtual bool sliceable() const override;
};

class CZ : public GateOperation {
 public:
  CZ(std::vector<size_t> targets, std::vector<size_t> ctrl_modifiers,
     size_t power);
  virtual ~CZ();
  virtual std::string name() const override;
  virtual bool sliceable() const override;
};

class S : public GateOperation {
 public:
  S(std::vector<size_t> targets, std::vector<size_t> ctrl_modifiers,
    size_t power);
  virtual ~S();
  virtual std::string name() const override;
  virtual bool sliceable() const override;
};

class Si : public GateOperation {
 public:
  Si(std::vector<size_t> targets, std::vector<size_t> ctrl_modifiers,
     size_t power);
  virtual ~Si();
  virtual std::string name() const override;
  virtual bool sliceable() const override;
};

class T : public GateOperation {
 public:
  T(std::vector<size_t> targets, std::vector<size_t> ctrl_modifiers,
    size_t power);
  virtual ~T();
  virtual std::string name() const override;
  virtual bool sliceable() const override;
};

class Ti : public GateOperation {
 public:
  Ti(std::vector<size_t> targets, std::vector<size_t> ctrl_modifiers,
     size_t power);
  virtual ~Ti();
  virtual std::string name() const override;
  virtual bool sliceable() const override;
};

class V : public GateOperation {
 public:
  V(std::vector<size_t> targets, std::vector<size_t> ctrl_modifiers,
    size_t power);
  virtual ~V();
  virtual std::string name() const override;
  virtual bool sliceable() const override;
};

class Vi : public GateOperation {
 public:
  Vi(std::vector<size_t> targets, std::vector<size_t> ctrl_modifiers,
     size_t power);
  virtual ~Vi();
  virtual std::string name() const override;
  virtual bool sliceable() const override;
};

class PhaseShift : public GateOperation {
 public:
  PhaseShift(std::vector<size_t> targets, double angle,
             std::vector<size_t> ctrl_modifiers, size_t power);
  virtual ~PhaseShift();
  double angle_;
  virtual std::string name() const override;
  virtual bool sliceable() const override;
};

class CPhaseShift : public GateOperation {
 public:
  CPhaseShift(std::vector<size_t> targets, double angle,
              std::vector<size_t> ctrl_modifiers, size_t power);
  virtual ~CPhaseShift();
  double angle_;
  virtual std::string name() const override;
  virtual bool sliceable() const override;
};

class CPhaseShift00 : public GateOperation {
 public:
  CPhaseShift00(std::vector<size_t> targets, double angle,
                std::vector<size_t> ctrl_modifiers, size_t power);
  virtual ~CPhaseShift00();
  double angle_;
  virtual std::string name() const override;
  virtual bool sliceable() const override;
};

class CPhaseShift01 : public GateOperation {
 public:
  CPhaseShift01(std::vector<size_t> targets, double angle,
                std::vector<size_t> ctrl_modifiers, size_t power);
  virtual ~CPhaseShift01();
  double angle_;
  virtual std::string name() const override;
  virtual bool sliceable() const override;
};

class CPhaseShift10 : public GateOperation {
 public:
  CPhaseShift10(std::vector<size_t> targets, double angle,
                std::vector<size_t> ctrl_modifiers, size_t power);
  virtual ~CPhaseShift10();
  double angle_;
  virtual std::string name() const override;
  virtual bool sliceable() const override;
};

class RotX : public GateOperation {
 public:
  RotX(std::vector<size_t> targets, double angle,
       std::vector<size_t> ctrl_modifiers, size_t power);
  virtual ~RotX();
  double angle_;
  virtual std::string name() const override;
  virtual bool sliceable() const override;
};

class RotY : public GateOperation {
 public:
  RotY(std::vector<size_t> targets, double angle,
       std::vector<size_t> ctrl_modifiers, size_t power);
  virtual ~RotY();
  double angle_;
  virtual std::string name() const override;
  virtual bool sliceable() const override;
};

class RotZ : public GateOperation {
 public:
  RotZ(std::vector<size_t> targets, double angle,
       std::vector<size_t> ctrl_modifiers, size_t power);
  virtual ~RotZ();
  double angle_;
  virtual std::string name() const override;
  virtual bool sliceable() const override;
};

class Swap : public GateOperation {
 public:
  Swap(std::vector<size_t> targets, std::vector<size_t> ctrl_modifiers,
       size_t power);
  virtual ~Swap();
  virtual std::string name() const override;
  virtual bool sliceable() const override;
};

class ISwap : public GateOperation {
 public:
  ISwap(std::vector<size_t> targets, std::vector<size_t> ctrl_modifiers,
        size_t power);
  virtual ~ISwap();
  virtual std::string name() const override;
  virtual bool sliceable() const override;
};

class PSwap : public GateOperation {
 public:
  PSwap(std::vector<size_t> targets, double angle,
        std::vector<size_t> ctrl_modifiers, size_t power);
  virtual ~PSwap();
  double angle_;
  virtual std::string name() const override;
  virtual bool sliceable() const override;
};

class XY : public GateOperation {
 public:
  XY(std::vector<size_t> targets, double angle,
     std::vector<size_t> ctrl_modifiers, size_t power);
  virtual ~XY();
  double angle_;
  virtual std::string name() const override;
  virtual bool sliceable() const override;
};

class XX : public GateOperation {
 public:
  XX(std::vector<size_t> targets, double angle,
     std::vector<size_t> ctrl_modifiers, size_t power);
  virtual ~XX();
  double angle_;
  virtual std::string name() const override;
  virtual bool sliceable() const override;
};

class YY : public GateOperation {
 public:
  YY(std::vector<size_t> targets, double angle,
     std::vector<size_t> ctrl_modifiers, size_t power);
  virtual ~YY();
  double angle_;
  virtual std::string name() const override;
  virtual bool sliceable() const override;
};

class ZZ : public GateOperation {
 public:
  ZZ(std::vector<size_t> targets, double angle,
     std::vector<size_t> ctrl_modifiers, size_t power);
  virtual ~ZZ();
  double angle_;
  virtual std::string name() const override;
  virtual bool sliceable() const override;
};

class CCNot : public GateOperation {
 public:
  CCNot(std::vector<size_t> targets, std::vector<size_t> ctrl_modifiers,
        size_t power);
  virtual ~CCNot();
  virtual std::string name() const override;
  virtual bool sliceable() const override;
};

class CSwap : public GateOperation {
 public:
  CSwap(std::vector<size_t> targets, std::vector<size_t> ctrl_modifiers,
        size_t power);
  virtual ~CSwap();
  virtual std::string name() const override;
  virtual bool sliceable() const override;
};

class U : public GateOperation {
 public:
  U(std::vector<size_t> targets, double theta, double phi, double lambda,
    std::vector<size_t> ctrl_modifiers, size_t power);
  virtual ~U();
  double theta_;
  double phi_;
  double lambda_;
  virtual std::string name() const override;
  virtual bool sliceable() const override;
};

class GPhase : public GateOperation {
 public:
  GPhase(std::vector<size_t> targets, double angle,
         std::vector<size_t> ctrl_modifiers, size_t power);
  virtual ~GPhase();
  double angle_;
  virtual std::string name() const override;
  virtual bool sliceable() const override;
};

#endif  //_GATE_OPERATION_H_
