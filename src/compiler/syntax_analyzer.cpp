#include "syntax_analyzer.h"

#include "antlr4-runtime.h"
#include "snuqasm-parser/snuqasmLexer.h"
#include "snuqasm-parser/snuqasmParser.h"
#include "snuqasm-parser/snuqasmBaseListener.h"

#include "logger.h"

namespace snuqs {

template<typename... Args>
void CannotBeHere( Args... args)
{
  Logger::error(args...);
  std::exit(EXIT_FAILURE);
}

void SyntaxAnalyzer::analyze(const std::string &filename) {
  /* Do nothing */
  std::ifstream stream(filename);
  antlr4::ANTLRInputStream input(stream);
  snuqasmLexer lexer(&input);
  antlr4::CommonTokenStream tokens(&lexer);
  snuqasmParser parser(&tokens);
  antlr4::tree::ParseTree *tree = parser.mainprogram();
  antlr4::tree::ParseTreeWalker::DEFAULT.walk(this, tree);
}

void SyntaxAnalyzer::enterMainprogram(snuqasmParser::MainprogramContext * ctx)  {
  /* Do nothing */
}

void SyntaxAnalyzer::exitMainprogram(snuqasmParser::MainprogramContext * ctx)  {
  /* Do nothing */
}

void SyntaxAnalyzer::enterVersion(snuqasmParser::VersionContext * ctx)  {
  /* Do nothing */
}

void SyntaxAnalyzer::exitVersion(snuqasmParser::VersionContext * ctx)  {
  /* Do nothing */
}

void SyntaxAnalyzer::enterProgram(snuqasmParser::ProgramContext * ctx)  {
  /* Do nothing */
}

void SyntaxAnalyzer::exitProgram(snuqasmParser::ProgramContext * ctx)  {
  /* Do nothing */
}

void SyntaxAnalyzer::enterInclude(snuqasmParser::IncludeContext * ctx)  {
  /* Do nothing */
}

void SyntaxAnalyzer::exitInclude(snuqasmParser::IncludeContext * ctx)  {
  /* Do nothing */
}

void SyntaxAnalyzer::enterStatement(snuqasmParser::StatementContext * ctx)  {
  /* Do nothing */
}

void SyntaxAnalyzer::exitStatement(snuqasmParser::StatementContext * ctx)  {
  /* Do nothing */
}

void SyntaxAnalyzer::enterDecl(snuqasmParser::DeclContext * ctx)  {
  /* Do nothing */
}

void SyntaxAnalyzer::exitDecl(snuqasmParser::DeclContext * ctx)  {
  /* Do nothing */
}

void SyntaxAnalyzer::enterQuantumDecl(snuqasmParser::QuantumDeclContext * ctx)  {
  /* Do nothing */
}

void SyntaxAnalyzer::exitQuantumDecl(snuqasmParser::QuantumDeclContext * ctx)  {
  /* Do nothing */
}

void SyntaxAnalyzer::enterClassicalDecl(snuqasmParser::ClassicalDeclContext * ctx)  {
  /* Do nothing */
}

void SyntaxAnalyzer::exitClassicalDecl(snuqasmParser::ClassicalDeclContext * ctx)  {
  /* Do nothing */
}

void SyntaxAnalyzer::enterGatedeclStatement(snuqasmParser::GatedeclStatementContext * ctx)  {
  /* Do nothing */
}

void SyntaxAnalyzer::exitGatedeclStatement(snuqasmParser::GatedeclStatementContext * ctx)  {
  /* Do nothing */
}

void SyntaxAnalyzer::enterGoplist(snuqasmParser::GoplistContext * ctx)  {
  /* Do nothing */
}

void SyntaxAnalyzer::exitGoplist(snuqasmParser::GoplistContext * ctx)  {
  /* Do nothing */
}

void SyntaxAnalyzer::enterOpaqueDeclStatement(snuqasmParser::OpaqueDeclStatementContext * ctx) {
  CannotBeHere("Opaque declaration is not suppored.\n");
}

void SyntaxAnalyzer::exitOpaqueDeclStatement(snuqasmParser::OpaqueDeclStatementContext * ctx) {
  /* Do nothing */
}

void SyntaxAnalyzer::enterQopStatement(snuqasmParser::QopStatementContext * ctx)  {
  /* Do nothing */
}

void SyntaxAnalyzer::exitQopStatement(snuqasmParser::QopStatementContext * ctx)  {
  /* Do nothing */
}

void SyntaxAnalyzer::enterUopStatement(snuqasmParser::UopStatementContext * ctx)  {
  /* Do nothing */
}

void SyntaxAnalyzer::exitUopStatement(snuqasmParser::UopStatementContext * ctx)  {
  /* Do nothing */
}

void SyntaxAnalyzer::enterMeasureQop(snuqasmParser::MeasureQopContext * ctx)  {
  /* Do nothing */
}

void SyntaxAnalyzer::exitMeasureQop(snuqasmParser::MeasureQopContext * ctx)  {
  /* Do nothing */
}

void SyntaxAnalyzer::enterResetQop(snuqasmParser::ResetQopContext * ctx)  {
  /* Do nothing */
}

void SyntaxAnalyzer::exitResetQop(snuqasmParser::ResetQopContext * ctx)  {
  /* Do nothing */
}

void SyntaxAnalyzer::enterIfStatement(snuqasmParser::IfStatementContext * ctx)  {
  /* Do nothing */
}

void SyntaxAnalyzer::exitIfStatement(snuqasmParser::IfStatementContext * ctx)  {
  /* Do nothing */
}

void SyntaxAnalyzer::enterBarrierStatement(snuqasmParser::BarrierStatementContext * ctx)  {
  /* Do nothing */
}

void SyntaxAnalyzer::exitBarrierStatement(snuqasmParser::BarrierStatementContext * ctx)  {
  /* Do nothing */
}

void SyntaxAnalyzer::enterUnitaryOp(snuqasmParser::UnitaryOpContext * ctx)  {
  /* Do nothing */
}

void SyntaxAnalyzer::exitUnitaryOp(snuqasmParser::UnitaryOpContext * ctx)  {
  /* Do nothing */
}

void SyntaxAnalyzer::enterCustomOp(snuqasmParser::CustomOpContext * ctx)  {
  /* Do nothing */
}

void SyntaxAnalyzer::exitCustomOp(snuqasmParser::CustomOpContext * ctx)  {
  /* Do nothing */
}

void SyntaxAnalyzer::enterAnylist(snuqasmParser::AnylistContext * ctx)  {
  /* Do nothing */
}

void SyntaxAnalyzer::exitAnylist(snuqasmParser::AnylistContext * ctx)  {
  /* Do nothing */
}

void SyntaxAnalyzer::enterIdlist(snuqasmParser::IdlistContext * ctx)  {
  /* Do nothing */
}

void SyntaxAnalyzer::exitIdlist(snuqasmParser::IdlistContext * ctx)  {
  /* Do nothing */
}

void SyntaxAnalyzer::enterMixedlist(snuqasmParser::MixedlistContext * ctx)  {
  /* Do nothing */
}

void SyntaxAnalyzer::exitMixedlist(snuqasmParser::MixedlistContext * ctx)  {
  /* Do nothing */
}

void SyntaxAnalyzer::enterArglist(snuqasmParser::ArglistContext * ctx)  {
  /* Do nothing */
}

void SyntaxAnalyzer::exitArglist(snuqasmParser::ArglistContext * ctx)  {
  /* Do nothing */
}

void SyntaxAnalyzer::enterArgument(snuqasmParser::ArgumentContext * ctx)  {
  /* Do nothing */
}

void SyntaxAnalyzer::exitArgument(snuqasmParser::ArgumentContext * ctx)  {
  /* Do nothing */
}

void SyntaxAnalyzer::enterExplist(snuqasmParser::ExplistContext * ctx)  {
  /* Do nothing */
}

void SyntaxAnalyzer::exitExplist(snuqasmParser::ExplistContext * ctx)  {
  /* Do nothing */
}

void SyntaxAnalyzer::enterExp(snuqasmParser::ExpContext * ctx)  {
  /* Do nothing */
}

void SyntaxAnalyzer::exitExp(snuqasmParser::ExpContext * ctx)  {
  /* Do nothing */
}

void SyntaxAnalyzer::enterBinop(snuqasmParser::BinopContext * ctx)  {
  /* Do nothing */
}

void SyntaxAnalyzer::exitBinop(snuqasmParser::BinopContext * ctx)  {
  /* Do nothing */
}

void SyntaxAnalyzer::enterUnaryop(snuqasmParser::UnaryopContext * ctx)  {
  /* Do nothing */
}

void SyntaxAnalyzer::exitUnaryop(snuqasmParser::UnaryopContext * ctx)  {
  /* Do nothing */
}

void SyntaxAnalyzer::enterEveryRule(antlr4::ParserRuleContext * ctx)  {
  /* Do nothing */
}

void SyntaxAnalyzer::exitEveryRule(antlr4::ParserRuleContext * ctx)  {
  /* Do nothing */
}

void SyntaxAnalyzer::visitTerminal(antlr4::tree::TerminalNode * node)  {
  /* Do nothing */
}

void SyntaxAnalyzer::visitErrorNode(antlr4::tree::ErrorNode * node)  {
  /* Do nothing */
}

} // namespace snuqs
