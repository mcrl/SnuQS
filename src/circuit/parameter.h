#ifndef __PARAMETER_H__
#define __PARAMETER_H__

#include "circuit/arg.h"
#include <complex>
#include <memory>

namespace snuqs {

class Parameter {
public:
  Parameter(){};
  virtual std::complex<double> eval() const { return 0.0; };
};

class Identifier : public Parameter {
public:
  Identifier(const Creg &creg);
  virtual std::complex<double> eval() const override;

  const Creg &creg_;
};

enum class BinOpType {
  ADD = 1,
  SUB = 2,
  MUL = 3,
  DIV = 4,
};

class BinOp : public Parameter {
public:
  BinOp(BinOpType op, std::shared_ptr<Parameter> param0,
        std::shared_ptr<Parameter> param1);
  virtual std::complex<double> eval() const override;

  BinOpType op_;
  std::shared_ptr<Parameter> param0_;
  std::shared_ptr<Parameter> param1_;
};

class NegOp : public Parameter {
public:
  NegOp(std::shared_ptr<Parameter> param);
  virtual std::complex<double> eval() const override;

  std::shared_ptr<Parameter> param_;
};

enum class UnaryOpType {
  SIN = 1,
  COS = 2,
  TAN = 3,
  EXP = 4,
  LN = 5,
  SQRT = 6,
};

class UnaryOp : public Parameter {
public:
  UnaryOp(UnaryOpType op, std::shared_ptr<Parameter> param);
  virtual std::complex<double> eval() const override;

  UnaryOpType op_;
  std::shared_ptr<Parameter> param_;
};

class Parenthesis : public Parameter {
public:
  Parenthesis(std::shared_ptr<Parameter> param);
  virtual std::complex<double> eval() const override;

  std::shared_ptr<Parameter> param_;
};

class Constant : public Parameter {
public:
  Constant(double value);
  Constant(std::complex<double> value);
  virtual std::complex<double> eval() const override;

  std::complex<double> value_;
};

class Pi : public Constant {
public:
  Pi();
  virtual std::complex<double> eval() const override;
};

} // namespace snuqs
#endif //__PARAMETER_H__
