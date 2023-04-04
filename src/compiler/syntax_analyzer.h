#pragma once

#include "snuqasm-parser/snuqasmBaseListener.h"

namespace snuqs {

class SyntaxAnalyzer : public snuqasmBaseListener {
	public:
	void analyze(const std::string &filename);

  virtual void enterMainprogram(snuqasmParser::MainprogramContext * ctx) override;
  virtual void exitMainprogram(snuqasmParser::MainprogramContext * ctx) override;

  virtual void enterVersion(snuqasmParser::VersionContext * ctx) override;
  virtual void exitVersion(snuqasmParser::VersionContext * ctx) override;

  virtual void enterProgram(snuqasmParser::ProgramContext * ctx) override;
  virtual void exitProgram(snuqasmParser::ProgramContext * ctx) override;

  virtual void enterInclude(snuqasmParser::IncludeContext * ctx) override;
  virtual void exitInclude(snuqasmParser::IncludeContext * ctx) override;

  virtual void enterStatement(snuqasmParser::StatementContext * ctx) override;
  virtual void exitStatement(snuqasmParser::StatementContext * ctx) override;

  virtual void enterDecl(snuqasmParser::DeclContext * ctx) override;
  virtual void exitDecl(snuqasmParser::DeclContext * ctx) override;

  virtual void enterQuantumDecl(snuqasmParser::QuantumDeclContext * ctx) override;
  virtual void exitQuantumDecl(snuqasmParser::QuantumDeclContext * ctx) override;

  virtual void enterClassicalDecl(snuqasmParser::ClassicalDeclContext * ctx) override;
  virtual void exitClassicalDecl(snuqasmParser::ClassicalDeclContext * ctx) override;

  virtual void enterGatedeclStatement(snuqasmParser::GatedeclStatementContext * ctx) override;
  virtual void exitGatedeclStatement(snuqasmParser::GatedeclStatementContext * ctx) override;

  virtual void enterGoplist(snuqasmParser::GoplistContext * ctx) override;
  virtual void exitGoplist(snuqasmParser::GoplistContext * ctx) override;

  virtual void enterOpaqueDeclStatement(snuqasmParser::OpaqueDeclStatementContext * ctx) override;
  virtual void exitOpaqueDeclStatement(snuqasmParser::OpaqueDeclStatementContext * ctx) override;

  virtual void enterQopStatement(snuqasmParser::QopStatementContext * ctx) override;
  virtual void exitQopStatement(snuqasmParser::QopStatementContext * ctx) override;

  virtual void enterUopStatement(snuqasmParser::UopStatementContext * ctx) override;
  virtual void exitUopStatement(snuqasmParser::UopStatementContext * ctx) override;

  virtual void enterMeasureQop(snuqasmParser::MeasureQopContext * ctx) override;
  virtual void exitMeasureQop(snuqasmParser::MeasureQopContext * ctx) override;

  virtual void enterResetQop(snuqasmParser::ResetQopContext * ctx) override;
  virtual void exitResetQop(snuqasmParser::ResetQopContext * ctx) override;

  virtual void enterIfStatement(snuqasmParser::IfStatementContext * ctx) override;
  virtual void exitIfStatement(snuqasmParser::IfStatementContext * ctx) override;

  virtual void enterBarrierStatement(snuqasmParser::BarrierStatementContext * ctx) override;
  virtual void exitBarrierStatement(snuqasmParser::BarrierStatementContext * ctx) override;

  virtual void enterUnitaryOp(snuqasmParser::UnitaryOpContext * ctx) override;
  virtual void exitUnitaryOp(snuqasmParser::UnitaryOpContext * ctx) override;

  virtual void enterCustomOp(snuqasmParser::CustomOpContext * ctx) override;
  virtual void exitCustomOp(snuqasmParser::CustomOpContext * ctx) override;

  virtual void enterAnylist(snuqasmParser::AnylistContext * ctx) override;
  virtual void exitAnylist(snuqasmParser::AnylistContext * ctx) override;

  virtual void enterIdlist(snuqasmParser::IdlistContext * ctx) override;
  virtual void exitIdlist(snuqasmParser::IdlistContext * ctx) override;

  virtual void enterMixedlist(snuqasmParser::MixedlistContext * ctx) override;
  virtual void exitMixedlist(snuqasmParser::MixedlistContext * ctx) override;

  virtual void enterArglist(snuqasmParser::ArglistContext * ctx) override;
  virtual void exitArglist(snuqasmParser::ArglistContext * ctx) override;

  virtual void enterArgument(snuqasmParser::ArgumentContext * ctx) override;
  virtual void exitArgument(snuqasmParser::ArgumentContext * ctx) override;

  virtual void enterExplist(snuqasmParser::ExplistContext * ctx) override;
  virtual void exitExplist(snuqasmParser::ExplistContext * ctx) override;

  virtual void enterExp(snuqasmParser::ExpContext * ctx) override;
  virtual void exitExp(snuqasmParser::ExpContext * ctx) override;

  virtual void enterBinop(snuqasmParser::BinopContext * ctx) override;
  virtual void exitBinop(snuqasmParser::BinopContext * ctx) override;

  virtual void enterUnaryop(snuqasmParser::UnaryopContext * ctx) override;
  virtual void exitUnaryop(snuqasmParser::UnaryopContext * ctx) override;


  virtual void enterEveryRule(antlr4::ParserRuleContext * ctx) override;
  virtual void exitEveryRule(antlr4::ParserRuleContext * ctx) override;
  virtual void visitTerminal(antlr4::tree::TerminalNode * node) override;
  virtual void visitErrorNode(antlr4::tree::ErrorNode * node) override;
};

} // namespace snuqs
