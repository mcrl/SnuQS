
// Generated from snuql.g4 by ANTLR 4.7.2

#pragma once


#include "antlr4-runtime.h"
#include "snuqlParser.h"


/**
 * This interface defines an abstract listener for a parse tree produced by snuqlParser.
 */
class  snuqlListener : public antlr4::tree::ParseTreeListener {
public:

  virtual void enterMainprogram(snuqlParser::MainprogramContext *ctx) = 0;
  virtual void exitMainprogram(snuqlParser::MainprogramContext *ctx) = 0;

  virtual void enterHeader(snuqlParser::HeaderContext *ctx) = 0;
  virtual void exitHeader(snuqlParser::HeaderContext *ctx) = 0;

  virtual void enterVersion(snuqlParser::VersionContext *ctx) = 0;
  virtual void exitVersion(snuqlParser::VersionContext *ctx) = 0;

  virtual void enterInclude(snuqlParser::IncludeContext *ctx) = 0;
  virtual void exitInclude(snuqlParser::IncludeContext *ctx) = 0;

  virtual void enterProgram(snuqlParser::ProgramContext *ctx) = 0;
  virtual void exitProgram(snuqlParser::ProgramContext *ctx) = 0;

  virtual void enterStatement(snuqlParser::StatementContext *ctx) = 0;
  virtual void exitStatement(snuqlParser::StatementContext *ctx) = 0;

  virtual void enterDecl(snuqlParser::DeclContext *ctx) = 0;
  virtual void exitDecl(snuqlParser::DeclContext *ctx) = 0;

  virtual void enterQuantumDecl(snuqlParser::QuantumDeclContext *ctx) = 0;
  virtual void exitQuantumDecl(snuqlParser::QuantumDeclContext *ctx) = 0;

  virtual void enterClassicalDecl(snuqlParser::ClassicalDeclContext *ctx) = 0;
  virtual void exitClassicalDecl(snuqlParser::ClassicalDeclContext *ctx) = 0;

  virtual void enterGatedeclStatement(snuqlParser::GatedeclStatementContext *ctx) = 0;
  virtual void exitGatedeclStatement(snuqlParser::GatedeclStatementContext *ctx) = 0;

  virtual void enterGatedecl(snuqlParser::GatedeclContext *ctx) = 0;
  virtual void exitGatedecl(snuqlParser::GatedeclContext *ctx) = 0;

  virtual void enterGoplist(snuqlParser::GoplistContext *ctx) = 0;
  virtual void exitGoplist(snuqlParser::GoplistContext *ctx) = 0;

  virtual void enterQopStatement(snuqlParser::QopStatementContext *ctx) = 0;
  virtual void exitQopStatement(snuqlParser::QopStatementContext *ctx) = 0;

  virtual void enterUopStatement(snuqlParser::UopStatementContext *ctx) = 0;
  virtual void exitUopStatement(snuqlParser::UopStatementContext *ctx) = 0;

  virtual void enterMeasureQop(snuqlParser::MeasureQopContext *ctx) = 0;
  virtual void exitMeasureQop(snuqlParser::MeasureQopContext *ctx) = 0;

  virtual void enterResetQop(snuqlParser::ResetQopContext *ctx) = 0;
  virtual void exitResetQop(snuqlParser::ResetQopContext *ctx) = 0;

  virtual void enterIfStatement(snuqlParser::IfStatementContext *ctx) = 0;
  virtual void exitIfStatement(snuqlParser::IfStatementContext *ctx) = 0;

  virtual void enterBarrierStatement(snuqlParser::BarrierStatementContext *ctx) = 0;
  virtual void exitBarrierStatement(snuqlParser::BarrierStatementContext *ctx) = 0;

  virtual void enterUnitaryOp(snuqlParser::UnitaryOpContext *ctx) = 0;
  virtual void exitUnitaryOp(snuqlParser::UnitaryOpContext *ctx) = 0;

  virtual void enterCustomOp(snuqlParser::CustomOpContext *ctx) = 0;
  virtual void exitCustomOp(snuqlParser::CustomOpContext *ctx) = 0;

  virtual void enterAnylist(snuqlParser::AnylistContext *ctx) = 0;
  virtual void exitAnylist(snuqlParser::AnylistContext *ctx) = 0;

  virtual void enterIdlist(snuqlParser::IdlistContext *ctx) = 0;
  virtual void exitIdlist(snuqlParser::IdlistContext *ctx) = 0;

  virtual void enterMixedlist(snuqlParser::MixedlistContext *ctx) = 0;
  virtual void exitMixedlist(snuqlParser::MixedlistContext *ctx) = 0;

  virtual void enterArglist(snuqlParser::ArglistContext *ctx) = 0;
  virtual void exitArglist(snuqlParser::ArglistContext *ctx) = 0;

  virtual void enterArgument(snuqlParser::ArgumentContext *ctx) = 0;
  virtual void exitArgument(snuqlParser::ArgumentContext *ctx) = 0;

  virtual void enterExplist(snuqlParser::ExplistContext *ctx) = 0;
  virtual void exitExplist(snuqlParser::ExplistContext *ctx) = 0;

  virtual void enterExp(snuqlParser::ExpContext *ctx) = 0;
  virtual void exitExp(snuqlParser::ExpContext *ctx) = 0;

  virtual void enterBinop(snuqlParser::BinopContext *ctx) = 0;
  virtual void exitBinop(snuqlParser::BinopContext *ctx) = 0;

  virtual void enterUnaryop(snuqlParser::UnaryopContext *ctx) = 0;
  virtual void exitUnaryop(snuqlParser::UnaryopContext *ctx) = 0;


};

