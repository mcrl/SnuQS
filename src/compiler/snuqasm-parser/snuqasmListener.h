
// Generated from snuqasm.g4 by ANTLR 4.12.0

#pragma once


#include "antlr4-runtime.h"
#include "snuqasmParser.h"


/**
 * This interface defines an abstract listener for a parse tree produced by snuqasmParser.
 */
class  snuqasmListener : public antlr4::tree::ParseTreeListener {
public:

  virtual void enterMainprogram(snuqasmParser::MainprogramContext *ctx) = 0;
  virtual void exitMainprogram(snuqasmParser::MainprogramContext *ctx) = 0;

  virtual void enterVersion(snuqasmParser::VersionContext *ctx) = 0;
  virtual void exitVersion(snuqasmParser::VersionContext *ctx) = 0;

  virtual void enterProgram(snuqasmParser::ProgramContext *ctx) = 0;
  virtual void exitProgram(snuqasmParser::ProgramContext *ctx) = 0;

  virtual void enterInclude(snuqasmParser::IncludeContext *ctx) = 0;
  virtual void exitInclude(snuqasmParser::IncludeContext *ctx) = 0;

  virtual void enterStatement(snuqasmParser::StatementContext *ctx) = 0;
  virtual void exitStatement(snuqasmParser::StatementContext *ctx) = 0;

  virtual void enterDecl(snuqasmParser::DeclContext *ctx) = 0;
  virtual void exitDecl(snuqasmParser::DeclContext *ctx) = 0;

  virtual void enterQuantumDecl(snuqasmParser::QuantumDeclContext *ctx) = 0;
  virtual void exitQuantumDecl(snuqasmParser::QuantumDeclContext *ctx) = 0;

  virtual void enterClassicalDecl(snuqasmParser::ClassicalDeclContext *ctx) = 0;
  virtual void exitClassicalDecl(snuqasmParser::ClassicalDeclContext *ctx) = 0;

  virtual void enterGatedeclStatement(snuqasmParser::GatedeclStatementContext *ctx) = 0;
  virtual void exitGatedeclStatement(snuqasmParser::GatedeclStatementContext *ctx) = 0;

  virtual void enterGoplist(snuqasmParser::GoplistContext *ctx) = 0;
  virtual void exitGoplist(snuqasmParser::GoplistContext *ctx) = 0;

  virtual void enterOpaqueDeclStatement(snuqasmParser::OpaqueDeclStatementContext *ctx) = 0;
  virtual void exitOpaqueDeclStatement(snuqasmParser::OpaqueDeclStatementContext *ctx) = 0;

  virtual void enterQopStatement(snuqasmParser::QopStatementContext *ctx) = 0;
  virtual void exitQopStatement(snuqasmParser::QopStatementContext *ctx) = 0;

  virtual void enterUopStatement(snuqasmParser::UopStatementContext *ctx) = 0;
  virtual void exitUopStatement(snuqasmParser::UopStatementContext *ctx) = 0;

  virtual void enterMeasureQop(snuqasmParser::MeasureQopContext *ctx) = 0;
  virtual void exitMeasureQop(snuqasmParser::MeasureQopContext *ctx) = 0;

  virtual void enterResetQop(snuqasmParser::ResetQopContext *ctx) = 0;
  virtual void exitResetQop(snuqasmParser::ResetQopContext *ctx) = 0;

  virtual void enterIfStatement(snuqasmParser::IfStatementContext *ctx) = 0;
  virtual void exitIfStatement(snuqasmParser::IfStatementContext *ctx) = 0;

  virtual void enterBarrierStatement(snuqasmParser::BarrierStatementContext *ctx) = 0;
  virtual void exitBarrierStatement(snuqasmParser::BarrierStatementContext *ctx) = 0;

  virtual void enterUnitaryOp(snuqasmParser::UnitaryOpContext *ctx) = 0;
  virtual void exitUnitaryOp(snuqasmParser::UnitaryOpContext *ctx) = 0;

  virtual void enterCustomOp(snuqasmParser::CustomOpContext *ctx) = 0;
  virtual void exitCustomOp(snuqasmParser::CustomOpContext *ctx) = 0;

  virtual void enterAnylist(snuqasmParser::AnylistContext *ctx) = 0;
  virtual void exitAnylist(snuqasmParser::AnylistContext *ctx) = 0;

  virtual void enterIdlist(snuqasmParser::IdlistContext *ctx) = 0;
  virtual void exitIdlist(snuqasmParser::IdlistContext *ctx) = 0;

  virtual void enterDesignatedIdentifier(snuqasmParser::DesignatedIdentifierContext *ctx) = 0;
  virtual void exitDesignatedIdentifier(snuqasmParser::DesignatedIdentifierContext *ctx) = 0;

  virtual void enterMixedlist(snuqasmParser::MixedlistContext *ctx) = 0;
  virtual void exitMixedlist(snuqasmParser::MixedlistContext *ctx) = 0;

  virtual void enterArglist(snuqasmParser::ArglistContext *ctx) = 0;
  virtual void exitArglist(snuqasmParser::ArglistContext *ctx) = 0;

  virtual void enterArgument(snuqasmParser::ArgumentContext *ctx) = 0;
  virtual void exitArgument(snuqasmParser::ArgumentContext *ctx) = 0;

  virtual void enterExplist(snuqasmParser::ExplistContext *ctx) = 0;
  virtual void exitExplist(snuqasmParser::ExplistContext *ctx) = 0;

  virtual void enterExp(snuqasmParser::ExpContext *ctx) = 0;
  virtual void exitExp(snuqasmParser::ExpContext *ctx) = 0;

  virtual void enterBinop(snuqasmParser::BinopContext *ctx) = 0;
  virtual void exitBinop(snuqasmParser::BinopContext *ctx) = 0;

  virtual void enterUnaryop(snuqasmParser::UnaryopContext *ctx) = 0;
  virtual void exitUnaryop(snuqasmParser::UnaryopContext *ctx) = 0;


};

