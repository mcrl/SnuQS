
// Generated from QASM.g4 by ANTLR 4.13.1

#pragma once


#include "antlr4-runtime.h"
#include "QASMParser.h"


/**
 * This interface defines an abstract listener for a parse tree produced by QASMParser.
 */
class  QASMListener : public antlr4::tree::ParseTreeListener {
public:

  virtual void enterMainprogram(QASMParser::MainprogramContext *ctx) = 0;
  virtual void exitMainprogram(QASMParser::MainprogramContext *ctx) = 0;

  virtual void enterVersion(QASMParser::VersionContext *ctx) = 0;
  virtual void exitVersion(QASMParser::VersionContext *ctx) = 0;

  virtual void enterProgram(QASMParser::ProgramContext *ctx) = 0;
  virtual void exitProgram(QASMParser::ProgramContext *ctx) = 0;

  virtual void enterStatement(QASMParser::StatementContext *ctx) = 0;
  virtual void exitStatement(QASMParser::StatementContext *ctx) = 0;

  virtual void enterDeclStatement(QASMParser::DeclStatementContext *ctx) = 0;
  virtual void exitDeclStatement(QASMParser::DeclStatementContext *ctx) = 0;

  virtual void enterRegDeclStatement(QASMParser::RegDeclStatementContext *ctx) = 0;
  virtual void exitRegDeclStatement(QASMParser::RegDeclStatementContext *ctx) = 0;

  virtual void enterQregDeclStatement(QASMParser::QregDeclStatementContext *ctx) = 0;
  virtual void exitQregDeclStatement(QASMParser::QregDeclStatementContext *ctx) = 0;

  virtual void enterCregDeclStatement(QASMParser::CregDeclStatementContext *ctx) = 0;
  virtual void exitCregDeclStatement(QASMParser::CregDeclStatementContext *ctx) = 0;

  virtual void enterGateDeclStatement(QASMParser::GateDeclStatementContext *ctx) = 0;
  virtual void exitGateDeclStatement(QASMParser::GateDeclStatementContext *ctx) = 0;

  virtual void enterOpaqueStatement(QASMParser::OpaqueStatementContext *ctx) = 0;
  virtual void exitOpaqueStatement(QASMParser::OpaqueStatementContext *ctx) = 0;

  virtual void enterGateStatement(QASMParser::GateStatementContext *ctx) = 0;
  virtual void exitGateStatement(QASMParser::GateStatementContext *ctx) = 0;

  virtual void enterGoplist(QASMParser::GoplistContext *ctx) = 0;
  virtual void exitGoplist(QASMParser::GoplistContext *ctx) = 0;

  virtual void enterGop(QASMParser::GopContext *ctx) = 0;
  virtual void exitGop(QASMParser::GopContext *ctx) = 0;

  virtual void enterGopUGate(QASMParser::GopUGateContext *ctx) = 0;
  virtual void exitGopUGate(QASMParser::GopUGateContext *ctx) = 0;

  virtual void enterGopCXGate(QASMParser::GopCXGateContext *ctx) = 0;
  virtual void exitGopCXGate(QASMParser::GopCXGateContext *ctx) = 0;

  virtual void enterGopBarrier(QASMParser::GopBarrierContext *ctx) = 0;
  virtual void exitGopBarrier(QASMParser::GopBarrierContext *ctx) = 0;

  virtual void enterGopCustomGate(QASMParser::GopCustomGateContext *ctx) = 0;
  virtual void exitGopCustomGate(QASMParser::GopCustomGateContext *ctx) = 0;

  virtual void enterGopReset(QASMParser::GopResetContext *ctx) = 0;
  virtual void exitGopReset(QASMParser::GopResetContext *ctx) = 0;

  virtual void enterIdlist(QASMParser::IdlistContext *ctx) = 0;
  virtual void exitIdlist(QASMParser::IdlistContext *ctx) = 0;

  virtual void enterParamlist(QASMParser::ParamlistContext *ctx) = 0;
  virtual void exitParamlist(QASMParser::ParamlistContext *ctx) = 0;

  virtual void enterQopStatement(QASMParser::QopStatementContext *ctx) = 0;
  virtual void exitQopStatement(QASMParser::QopStatementContext *ctx) = 0;

  virtual void enterQopUGate(QASMParser::QopUGateContext *ctx) = 0;
  virtual void exitQopUGate(QASMParser::QopUGateContext *ctx) = 0;

  virtual void enterQopCXGate(QASMParser::QopCXGateContext *ctx) = 0;
  virtual void exitQopCXGate(QASMParser::QopCXGateContext *ctx) = 0;

  virtual void enterQopMeasure(QASMParser::QopMeasureContext *ctx) = 0;
  virtual void exitQopMeasure(QASMParser::QopMeasureContext *ctx) = 0;

  virtual void enterQopReset(QASMParser::QopResetContext *ctx) = 0;
  virtual void exitQopReset(QASMParser::QopResetContext *ctx) = 0;

  virtual void enterQopCustomGate(QASMParser::QopCustomGateContext *ctx) = 0;
  virtual void exitQopCustomGate(QASMParser::QopCustomGateContext *ctx) = 0;

  virtual void enterIfStatement(QASMParser::IfStatementContext *ctx) = 0;
  virtual void exitIfStatement(QASMParser::IfStatementContext *ctx) = 0;

  virtual void enterBarrierStatement(QASMParser::BarrierStatementContext *ctx) = 0;
  virtual void exitBarrierStatement(QASMParser::BarrierStatementContext *ctx) = 0;

  virtual void enterArglist(QASMParser::ArglistContext *ctx) = 0;
  virtual void exitArglist(QASMParser::ArglistContext *ctx) = 0;

  virtual void enterQarg(QASMParser::QargContext *ctx) = 0;
  virtual void exitQarg(QASMParser::QargContext *ctx) = 0;

  virtual void enterCarg(QASMParser::CargContext *ctx) = 0;
  virtual void exitCarg(QASMParser::CargContext *ctx) = 0;

  virtual void enterExplist(QASMParser::ExplistContext *ctx) = 0;
  virtual void exitExplist(QASMParser::ExplistContext *ctx) = 0;

  virtual void enterExp(QASMParser::ExpContext *ctx) = 0;
  virtual void exitExp(QASMParser::ExpContext *ctx) = 0;

  virtual void enterComplex(QASMParser::ComplexContext *ctx) = 0;
  virtual void exitComplex(QASMParser::ComplexContext *ctx) = 0;

  virtual void enterAddsub(QASMParser::AddsubContext *ctx) = 0;
  virtual void exitAddsub(QASMParser::AddsubContext *ctx) = 0;

  virtual void enterBinop(QASMParser::BinopContext *ctx) = 0;
  virtual void exitBinop(QASMParser::BinopContext *ctx) = 0;

  virtual void enterNegop(QASMParser::NegopContext *ctx) = 0;
  virtual void exitNegop(QASMParser::NegopContext *ctx) = 0;

  virtual void enterUnaryop(QASMParser::UnaryopContext *ctx) = 0;
  virtual void exitUnaryop(QASMParser::UnaryopContext *ctx) = 0;


};

