
// Generated from QASM.g4 by ANTLR 4.13.1

#pragma once


#include "antlr4-runtime.h"
#include "QASMListener.h"


/**
 * This class provides an empty implementation of QASMListener,
 * which can be extended to create a listener which only needs to handle a subset
 * of the available methods.
 */
class  QASMBaseListener : public QASMListener {
public:

  virtual void enterMainprogram(QASMParser::MainprogramContext * /*ctx*/) override { }
  virtual void exitMainprogram(QASMParser::MainprogramContext * /*ctx*/) override { }

  virtual void enterVersion(QASMParser::VersionContext * /*ctx*/) override { }
  virtual void exitVersion(QASMParser::VersionContext * /*ctx*/) override { }

  virtual void enterProgram(QASMParser::ProgramContext * /*ctx*/) override { }
  virtual void exitProgram(QASMParser::ProgramContext * /*ctx*/) override { }

  virtual void enterStatement(QASMParser::StatementContext * /*ctx*/) override { }
  virtual void exitStatement(QASMParser::StatementContext * /*ctx*/) override { }

  virtual void enterDeclStatement(QASMParser::DeclStatementContext * /*ctx*/) override { }
  virtual void exitDeclStatement(QASMParser::DeclStatementContext * /*ctx*/) override { }

  virtual void enterRegDeclStatement(QASMParser::RegDeclStatementContext * /*ctx*/) override { }
  virtual void exitRegDeclStatement(QASMParser::RegDeclStatementContext * /*ctx*/) override { }

  virtual void enterQregDeclStatement(QASMParser::QregDeclStatementContext * /*ctx*/) override { }
  virtual void exitQregDeclStatement(QASMParser::QregDeclStatementContext * /*ctx*/) override { }

  virtual void enterCregDeclStatement(QASMParser::CregDeclStatementContext * /*ctx*/) override { }
  virtual void exitCregDeclStatement(QASMParser::CregDeclStatementContext * /*ctx*/) override { }

  virtual void enterGateDeclStatement(QASMParser::GateDeclStatementContext * /*ctx*/) override { }
  virtual void exitGateDeclStatement(QASMParser::GateDeclStatementContext * /*ctx*/) override { }

  virtual void enterOpaqueStatement(QASMParser::OpaqueStatementContext * /*ctx*/) override { }
  virtual void exitOpaqueStatement(QASMParser::OpaqueStatementContext * /*ctx*/) override { }

  virtual void enterGateStatement(QASMParser::GateStatementContext * /*ctx*/) override { }
  virtual void exitGateStatement(QASMParser::GateStatementContext * /*ctx*/) override { }

  virtual void enterGoplist(QASMParser::GoplistContext * /*ctx*/) override { }
  virtual void exitGoplist(QASMParser::GoplistContext * /*ctx*/) override { }

  virtual void enterGop(QASMParser::GopContext * /*ctx*/) override { }
  virtual void exitGop(QASMParser::GopContext * /*ctx*/) override { }

  virtual void enterGopUGate(QASMParser::GopUGateContext * /*ctx*/) override { }
  virtual void exitGopUGate(QASMParser::GopUGateContext * /*ctx*/) override { }

  virtual void enterGopCXGate(QASMParser::GopCXGateContext * /*ctx*/) override { }
  virtual void exitGopCXGate(QASMParser::GopCXGateContext * /*ctx*/) override { }

  virtual void enterGopBarrier(QASMParser::GopBarrierContext * /*ctx*/) override { }
  virtual void exitGopBarrier(QASMParser::GopBarrierContext * /*ctx*/) override { }

  virtual void enterGopCustomGate(QASMParser::GopCustomGateContext * /*ctx*/) override { }
  virtual void exitGopCustomGate(QASMParser::GopCustomGateContext * /*ctx*/) override { }

  virtual void enterGopReset(QASMParser::GopResetContext * /*ctx*/) override { }
  virtual void exitGopReset(QASMParser::GopResetContext * /*ctx*/) override { }

  virtual void enterIdlist(QASMParser::IdlistContext * /*ctx*/) override { }
  virtual void exitIdlist(QASMParser::IdlistContext * /*ctx*/) override { }

  virtual void enterParamlist(QASMParser::ParamlistContext * /*ctx*/) override { }
  virtual void exitParamlist(QASMParser::ParamlistContext * /*ctx*/) override { }

  virtual void enterQopStatement(QASMParser::QopStatementContext * /*ctx*/) override { }
  virtual void exitQopStatement(QASMParser::QopStatementContext * /*ctx*/) override { }

  virtual void enterQopUGate(QASMParser::QopUGateContext * /*ctx*/) override { }
  virtual void exitQopUGate(QASMParser::QopUGateContext * /*ctx*/) override { }

  virtual void enterQopCXGate(QASMParser::QopCXGateContext * /*ctx*/) override { }
  virtual void exitQopCXGate(QASMParser::QopCXGateContext * /*ctx*/) override { }

  virtual void enterQopMeasure(QASMParser::QopMeasureContext * /*ctx*/) override { }
  virtual void exitQopMeasure(QASMParser::QopMeasureContext * /*ctx*/) override { }

  virtual void enterQopReset(QASMParser::QopResetContext * /*ctx*/) override { }
  virtual void exitQopReset(QASMParser::QopResetContext * /*ctx*/) override { }

  virtual void enterQopCustomGate(QASMParser::QopCustomGateContext * /*ctx*/) override { }
  virtual void exitQopCustomGate(QASMParser::QopCustomGateContext * /*ctx*/) override { }

  virtual void enterIfStatement(QASMParser::IfStatementContext * /*ctx*/) override { }
  virtual void exitIfStatement(QASMParser::IfStatementContext * /*ctx*/) override { }

  virtual void enterBarrierStatement(QASMParser::BarrierStatementContext * /*ctx*/) override { }
  virtual void exitBarrierStatement(QASMParser::BarrierStatementContext * /*ctx*/) override { }

  virtual void enterArglist(QASMParser::ArglistContext * /*ctx*/) override { }
  virtual void exitArglist(QASMParser::ArglistContext * /*ctx*/) override { }

  virtual void enterQarg(QASMParser::QargContext * /*ctx*/) override { }
  virtual void exitQarg(QASMParser::QargContext * /*ctx*/) override { }

  virtual void enterCarg(QASMParser::CargContext * /*ctx*/) override { }
  virtual void exitCarg(QASMParser::CargContext * /*ctx*/) override { }

  virtual void enterExplist(QASMParser::ExplistContext * /*ctx*/) override { }
  virtual void exitExplist(QASMParser::ExplistContext * /*ctx*/) override { }

  virtual void enterExp(QASMParser::ExpContext * /*ctx*/) override { }
  virtual void exitExp(QASMParser::ExpContext * /*ctx*/) override { }

  virtual void enterComplex(QASMParser::ComplexContext * /*ctx*/) override { }
  virtual void exitComplex(QASMParser::ComplexContext * /*ctx*/) override { }

  virtual void enterAddsub(QASMParser::AddsubContext * /*ctx*/) override { }
  virtual void exitAddsub(QASMParser::AddsubContext * /*ctx*/) override { }

  virtual void enterBinop(QASMParser::BinopContext * /*ctx*/) override { }
  virtual void exitBinop(QASMParser::BinopContext * /*ctx*/) override { }

  virtual void enterNegop(QASMParser::NegopContext * /*ctx*/) override { }
  virtual void exitNegop(QASMParser::NegopContext * /*ctx*/) override { }

  virtual void enterUnaryop(QASMParser::UnaryopContext * /*ctx*/) override { }
  virtual void exitUnaryop(QASMParser::UnaryopContext * /*ctx*/) override { }


  virtual void enterEveryRule(antlr4::ParserRuleContext * /*ctx*/) override { }
  virtual void exitEveryRule(antlr4::ParserRuleContext * /*ctx*/) override { }
  virtual void visitTerminal(antlr4::tree::TerminalNode * /*node*/) override { }
  virtual void visitErrorNode(antlr4::tree::ErrorNode * /*node*/) override { }

};

