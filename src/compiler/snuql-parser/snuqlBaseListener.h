
// Generated from snuql.g4 by ANTLR 4.7.2

#pragma once


#include "antlr4-runtime.h"
#include "snuqlListener.h"


/**
 * This class provides an empty implementation of snuqlListener,
 * which can be extended to create a listener which only needs to handle a subset
 * of the available methods.
 */
class  snuqlBaseListener : public snuqlListener {
public:

  virtual void enterMainprogram(snuqlParser::MainprogramContext * /*ctx*/) override { }
  virtual void exitMainprogram(snuqlParser::MainprogramContext * /*ctx*/) override { }

  virtual void enterHeader(snuqlParser::HeaderContext * /*ctx*/) override { }
  virtual void exitHeader(snuqlParser::HeaderContext * /*ctx*/) override { }

  virtual void enterVersion(snuqlParser::VersionContext * /*ctx*/) override { }
  virtual void exitVersion(snuqlParser::VersionContext * /*ctx*/) override { }

  virtual void enterInclude(snuqlParser::IncludeContext * /*ctx*/) override { }
  virtual void exitInclude(snuqlParser::IncludeContext * /*ctx*/) override { }

  virtual void enterProgram(snuqlParser::ProgramContext * /*ctx*/) override { }
  virtual void exitProgram(snuqlParser::ProgramContext * /*ctx*/) override { }

  virtual void enterStatement(snuqlParser::StatementContext * /*ctx*/) override { }
  virtual void exitStatement(snuqlParser::StatementContext * /*ctx*/) override { }

  virtual void enterDecl(snuqlParser::DeclContext * /*ctx*/) override { }
  virtual void exitDecl(snuqlParser::DeclContext * /*ctx*/) override { }

  virtual void enterQuantumDecl(snuqlParser::QuantumDeclContext * /*ctx*/) override { }
  virtual void exitQuantumDecl(snuqlParser::QuantumDeclContext * /*ctx*/) override { }

  virtual void enterClassicalDecl(snuqlParser::ClassicalDeclContext * /*ctx*/) override { }
  virtual void exitClassicalDecl(snuqlParser::ClassicalDeclContext * /*ctx*/) override { }

  virtual void enterGatedeclStatement(snuqlParser::GatedeclStatementContext * /*ctx*/) override { }
  virtual void exitGatedeclStatement(snuqlParser::GatedeclStatementContext * /*ctx*/) override { }

  virtual void enterGatedecl(snuqlParser::GatedeclContext * /*ctx*/) override { }
  virtual void exitGatedecl(snuqlParser::GatedeclContext * /*ctx*/) override { }

  virtual void enterGoplist(snuqlParser::GoplistContext * /*ctx*/) override { }
  virtual void exitGoplist(snuqlParser::GoplistContext * /*ctx*/) override { }

  virtual void enterQopStatement(snuqlParser::QopStatementContext * /*ctx*/) override { }
  virtual void exitQopStatement(snuqlParser::QopStatementContext * /*ctx*/) override { }

  virtual void enterUopStatement(snuqlParser::UopStatementContext * /*ctx*/) override { }
  virtual void exitUopStatement(snuqlParser::UopStatementContext * /*ctx*/) override { }

  virtual void enterMeasureQop(snuqlParser::MeasureQopContext * /*ctx*/) override { }
  virtual void exitMeasureQop(snuqlParser::MeasureQopContext * /*ctx*/) override { }

  virtual void enterResetQop(snuqlParser::ResetQopContext * /*ctx*/) override { }
  virtual void exitResetQop(snuqlParser::ResetQopContext * /*ctx*/) override { }

  virtual void enterIfStatement(snuqlParser::IfStatementContext * /*ctx*/) override { }
  virtual void exitIfStatement(snuqlParser::IfStatementContext * /*ctx*/) override { }

  virtual void enterBarrierStatement(snuqlParser::BarrierStatementContext * /*ctx*/) override { }
  virtual void exitBarrierStatement(snuqlParser::BarrierStatementContext * /*ctx*/) override { }

  virtual void enterUnitaryOp(snuqlParser::UnitaryOpContext * /*ctx*/) override { }
  virtual void exitUnitaryOp(snuqlParser::UnitaryOpContext * /*ctx*/) override { }

  virtual void enterCustomOp(snuqlParser::CustomOpContext * /*ctx*/) override { }
  virtual void exitCustomOp(snuqlParser::CustomOpContext * /*ctx*/) override { }

  virtual void enterAnylist(snuqlParser::AnylistContext * /*ctx*/) override { }
  virtual void exitAnylist(snuqlParser::AnylistContext * /*ctx*/) override { }

  virtual void enterIdlist(snuqlParser::IdlistContext * /*ctx*/) override { }
  virtual void exitIdlist(snuqlParser::IdlistContext * /*ctx*/) override { }

  virtual void enterMixedlist(snuqlParser::MixedlistContext * /*ctx*/) override { }
  virtual void exitMixedlist(snuqlParser::MixedlistContext * /*ctx*/) override { }

  virtual void enterArglist(snuqlParser::ArglistContext * /*ctx*/) override { }
  virtual void exitArglist(snuqlParser::ArglistContext * /*ctx*/) override { }

  virtual void enterArgument(snuqlParser::ArgumentContext * /*ctx*/) override { }
  virtual void exitArgument(snuqlParser::ArgumentContext * /*ctx*/) override { }

  virtual void enterExplist(snuqlParser::ExplistContext * /*ctx*/) override { }
  virtual void exitExplist(snuqlParser::ExplistContext * /*ctx*/) override { }

  virtual void enterExp(snuqlParser::ExpContext * /*ctx*/) override { }
  virtual void exitExp(snuqlParser::ExpContext * /*ctx*/) override { }

  virtual void enterBinop(snuqlParser::BinopContext * /*ctx*/) override { }
  virtual void exitBinop(snuqlParser::BinopContext * /*ctx*/) override { }

  virtual void enterUnaryop(snuqlParser::UnaryopContext * /*ctx*/) override { }
  virtual void exitUnaryop(snuqlParser::UnaryopContext * /*ctx*/) override { }


  virtual void enterEveryRule(antlr4::ParserRuleContext * /*ctx*/) override { }
  virtual void exitEveryRule(antlr4::ParserRuleContext * /*ctx*/) override { }
  virtual void visitTerminal(antlr4::tree::TerminalNode * /*node*/) override { }
  virtual void visitErrorNode(antlr4::tree::ErrorNode * /*node*/) override { }

};

