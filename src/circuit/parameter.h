#ifndef __PARAMETER_H__
#define __PARAMETER_H__

#include "circuit/arg.h"

namespace snuqs {

class Parameter {
public:
  virtual double eval() const = 0;
};

class Identifier : public Parameter {
public:
  Identifier(const Carg &creg);
  virtual double eval() const override;

  const Carg &carg_;
};


enum class BinOpType {
  ADD = 1,
  SUB = 2,
  MUL = 3,
  DIV = 4,
};

class BinOp : public Parameter {
public:
  BinOp(BinOpType op, const Parameter &param0, const Parameter &param1);
  virtual double eval() const override;

  BinOpType op_;
  const Parameter &param0_;
  const Parameter &param1_;
};

class NegOp : public Parameter {
public:
  NegOp(const Parameter &param);
  virtual double eval() const override;

  const Parameter &param_;
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
  UnaryOp(UnaryOpType op, const Parameter &param);
  virtual double eval() const override;

  UnaryOpType op_;
  const Parameter &param_;
};

class Parenthesis : public Parameter {
public:
  Parenthesis(const Parameter &param);
  virtual double eval() const override;

  const Parameter &param_;
};

class Constant : public Parameter {
public:
  Constant(double value);
  virtual double eval() const override;

  double value_;
};

class Pi : public Constant {
public:
  Pi();
  virtual double eval() const override;
};

} // namespace snuqs
#endif //__PARAMETER_H__
