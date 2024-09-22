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

class Identity : public GateOperation {
public:
  Identity();
  ~Identity();
  virtual std::complex<double> *data() override;
  virtual size_t dim() const override;
  virtual std::vector<size_t> shape() const override;
  virtual std::vector<size_t> stride() const override;
};

 class Hadamard : public GateOperation {
 public:
  Hadamard();
  ~Hadamard();
  virtual std::complex<double> *data() override;
  virtual size_t dim() const override;
  virtual std::vector<size_t> shape() const override;
  virtual std::vector<size_t> stride() const override;
};
//
// class PauliX : public GateOperation {
// public:
//  PauliX();
//  virtual std::complex<double> *data() override;
//  virtual size_t dim() const override;
//  virtual std::vector<size_t> shape() const override;
//};
//
// class PauliY : public GateOperation {
// public:
//  PauliY();
//  virtual std::complex<double> *data() override;
//  virtual size_t dim() const override;
//  virtual std::vector<size_t> shape() const override;
//};
//
// class PauliZ : public GateOperation {
// public:
//  PauliZ();
//  virtual std::complex<double> *data() override;
//  virtual size_t dim() const override;
//  virtual std::vector<size_t> shape() const override;
//};
//
// class CV : public GateOperation {
// public:
//  CV();
//  virtual std::complex<double> *data() override;
//  virtual size_t dim() const override;
//  virtual std::vector<size_t> shape() const override;
//};
//
// class CX : public GateOperation {
// public:
//  CX();
//  virtual std::complex<double> *data() override;
//  virtual size_t dim() const override;
//  virtual std::vector<size_t> shape() const override;
//};
//
// class CY : public GateOperation {
// public:
//  CY();
//  virtual std::complex<double> *data() override;
//  virtual size_t dim() const override;
//  virtual std::vector<size_t> shape() const override;
//};
//
// class CZ : public GateOperation {
// public:
//  CZ();
//  virtual std::complex<double> *data() override;
//  virtual size_t dim() const override;
//  virtual std::vector<size_t> shape() const override;
//};
//
// class ECR : public GateOperation {
// public:
//  ECR();
//  virtual std::complex<double> *data() override;
//  virtual size_t dim() const override;
//  virtual std::vector<size_t> shape() const override;
//};
//
// class S : public GateOperation {
// public:
//  S();
//  virtual std::complex<double> *data() override;
//  virtual size_t dim() const override;
//  virtual std::vector<size_t> shape() const override;
//};
//
// class Si : public GateOperation {
// public:
//  Si();
//  virtual std::complex<double> *data() override;
//  virtual size_t dim() const override;
//  virtual std::vector<size_t> shape() const override;
//};
//
// class T : public GateOperation {
// public:
//  T();
//  virtual std::complex<double> *data() override;
//  virtual size_t dim() const override;
//  virtual std::vector<size_t> shape() const override;
//};
//
// class Ti : public GateOperation {
// public:
//  Ti();
//  virtual std::complex<double> *data() override;
//  virtual size_t dim() const override;
//  virtual std::vector<size_t> shape() const override;
//};
//
// class V : public GateOperation {
// public:
//  V();
//  virtual std::complex<double> *data() override;
//  virtual size_t dim() const override;
//  virtual std::vector<size_t> shape() const override;
//};
//
// class Vi : public GateOperation {
// public:
//  Vi();
//  virtual std::complex<double> *data() override;
//  virtual size_t dim() const override;
//  virtual std::vector<size_t> shape() const override;
//};
//
// class PhaseShift : public GateOperation {
// public:
//  PhaseShift();
//  virtual std::complex<double> *data() override;
//  virtual size_t dim() const override;
//  virtual std::vector<size_t> shape() const override;
//};
//
// class CPhaseShift : public GateOperation {
// public:
//  CPhaseShift();
//  virtual std::complex<double> *data() override;
//  virtual size_t dim() const override;
//  virtual std::vector<size_t> shape() const override;
//};
//
// class CPhaseShift00 : public GateOperation {
// public:
//  CPhaseShift00();
//  virtual std::complex<double> *data() override;
//  virtual size_t dim() const override;
//  virtual std::vector<size_t> shape() const override;
//};
//
// class CPhaseShift01 : public GateOperation {
// public:
//  CPhaseShift01();
//  virtual std::complex<double> *data() override;
//  virtual size_t dim() const override;
//  virtual std::vector<size_t> shape() const override;
//};
//
// class CPhaseShift10 : public GateOperation {
// public:
//  CPhaseShift10();
//  virtual std::complex<double> *data() override;
//  virtual size_t dim() const override;
//  virtual std::vector<size_t> shape() const override;
//};
//
// class RotX : public GateOperation {
// public:
//  RotX();
//  virtual std::complex<double> *data() override;
//  virtual size_t dim() const override;
//  virtual std::vector<size_t> shape() const override;
//};
//
// class RotY : public GateOperation {
// public:
//  RotY();
//  virtual std::complex<double> *data() override;
//  virtual size_t dim() const override;
//  virtual std::vector<size_t> shape() const override;
//};
//
// class RotZ : public GateOperation {
// public:
//  RotZ();
//  virtual std::complex<double> *data() override;
//  virtual size_t dim() const override;
//  virtual std::vector<size_t> shape() const override;
//};
//
// class Swap : public GateOperation {
// public:
//  Swap();
//  virtual std::complex<double> *data() override;
//  virtual size_t dim() const override;
//  virtual std::vector<size_t> shape() const override;
//};
//
// class ISwap : public GateOperation {
// public:
//  ISwap();
//  virtual std::complex<double> *data() override;
//  virtual size_t dim() const override;
//  virtual std::vector<size_t> shape() const override;
//};
//
// class PSwap : public GateOperation {
// public:
//  PSwap();
//  virtual std::complex<double> *data() override;
//  virtual size_t dim() const override;
//  virtual std::vector<size_t> shape() const override;
//};
//
// class XY : public GateOperation {
// public:
//  XY();
//  virtual std::complex<double> *data() override;
//  virtual size_t dim() const override;
//  virtual std::vector<size_t> shape() const override;
//};
//
// class XX : public GateOperation {
// public:
//  XX();
//  virtual std::complex<double> *data() override;
//  virtual size_t dim() const override;
//  virtual std::vector<size_t> shape() const override;
//};
//
// class YY : public GateOperation {
// public:
//  YY();
//  virtual std::complex<double> *data() override;
//  virtual size_t dim() const override;
//  virtual std::vector<size_t> shape() const override;
//};
//
// class ZZ : public GateOperation {
// public:
//  ZZ();
//  virtual std::complex<double> *data() override;
//  virtual size_t dim() const override;
//  virtual std::vector<size_t> shape() const override;
//};
//
// class CCNot : public GateOperation {
// public:
//  CCNot();
//  virtual std::complex<double> *data() override;
//  virtual size_t dim() const override;
//  virtual std::vector<size_t> shape() const override;
//};
//
// class CSwap : public GateOperation {
// public:
//  CSwap();
//  virtual std::complex<double> *data() override;
//  virtual size_t dim() const override;
//  virtual std::vector<size_t> shape() const override;
//};
//
// class PRx : public GateOperation {
// public:
//  PRx();
//  virtual std::complex<double> *data() override;
//  virtual size_t dim() const override;
//  virtual std::vector<size_t> shape() const override;
//};
//
// class Unitary : public GateOperation {
// public:
//  Unitary();
//  virtual std::complex<double> *data() override;
//  virtual size_t dim() const override;
//  virtual std::vector<size_t> shape() const override;
//};
//
// class U : public GateOperation {
// public:
//  U();
//  virtual std::complex<double> *data() override;
//  virtual size_t dim() const override;
//  virtual std::vector<size_t> shape() const override;
//};
//
// class GPhase : public GateOperation {
// public:
//  GPhase();
//  virtual std::complex<double> *data() override;
//  virtual size_t dim() const override;
//  virtual std::vector<size_t> shape() const override;
//};

#endif //_GATE_OPERATION_H_
