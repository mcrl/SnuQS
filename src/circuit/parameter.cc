#include "circuit/parameter.h"

#include "assertion.h"
#include <cmath>

namespace snuqs {

Identifier::Identifier(const Creg &creg) : creg_(creg) {}
double Identifier::eval() const { return creg_.value(); }

BinOp::BinOp(BinOpType op, const Parameter &param0, const Parameter &param1)
    : op_(op), param0_(param0), param1_(param1) {}

double BinOp::eval() const {
  switch (op_) {
  case BinOpType::ADD:
    return param0_.eval() + param1_.eval();
  case BinOpType::SUB:
    return param0_.eval() - param1_.eval();
  case BinOpType::MUL:
    return param0_.eval() * param1_.eval();
  case BinOpType::DIV:
    return param0_.eval() / param1_.eval();
  }

  return 0.;
}

NegOp::NegOp(const Parameter &param) : param_(param) {}
double NegOp::eval() const { return -param_.eval(); }

UnaryOp::UnaryOp(UnaryOpType op, const Parameter &param)
    : op_(op), param_(param) {}

double UnaryOp::eval() const {
  switch (op_) {
  case UnaryOpType::SIN:
    return sin(param_.eval());
  case UnaryOpType::COS:
    return cos(param_.eval());
  case UnaryOpType::TAN:
    return tan(param_.eval());
  case UnaryOpType::EXP:
    return exp(param_.eval());
  case UnaryOpType::LN:
    return log(param_.eval());
  case UnaryOpType::SQRT:
    return sqrt(param_.eval());
  }
  return 0.;
}

Parenthesis::Parenthesis(const Parameter &param) : param_(param) {}
double Parenthesis::eval() const { return param_.eval(); }

Constant::Constant(double value) : value_(value) {}
double Constant::eval() const { return value_; }

Pi::Pi() : Constant(M_PI) {}
double Pi::eval() const { return value_; }

} // namespace snuqs
