#pragma once

#include "antlr4-runtime.h"
#include "snuql-parser/snuqlLexer.h"
#include "snuql-parser/snuqlParser.h"
#include "snuql-parser/snuqlBaseListener.h"

#include "symbolTable.hpp"

namespace snuqs {

class SemanticChecker : public snuqlBaseListener {
	public:

	private:
	SymbolTable symtab_;

	void checkIdentRedef(antlr4::tree::TerminalNode *) const;
	void checkIdentUndefined(antlr4::tree::TerminalNode *) const;
	void checkIdentType(antlr4::tree::TerminalNode *, SymbolTable::sym_type) const;


	size_t getExplistSize(snuqlParser::ExplistContext *ctx) const;
	size_t getIdlistSize(snuqlParser::IdlistContext *ctx) const;
	size_t getMixedlistSize(snuqlParser::MixedlistContext *ctx) const;
	size_t getAnylistSize(snuqlParser::AnylistContext *ctx) const;

	void checkUnitaryGateExp(snuqlParser::ExpContext *) const;
	void checkUnitaryGateExplist(snuqlParser::ExplistContext *) const;
	void checkUnitaryGateArg(snuqlParser::ArgumentContext*) const;
	void checkUnitaryGateArgs(std::vector<snuqlParser::ArgumentContext*> &&) const;
	void checkUnitaryGate(snuqlParser::UnitaryOpContext *) const;
	void checkCustomGate(snuqlParser::CustomOpContext *) const;

	void insertIdlistToSymtab(snuqlParser::IdlistContext *idlist, SymbolTable::sym_type type);

	void checkIdlist(snuqlParser::IdlistContext *ctx) const;
	void checkMixedlist(snuqlParser::MixedlistContext *ctx) const;

	public:
	void check(antlr4::tree::ParseTree *tree);
	const SymbolTable& getSymbolTable() const;

	void enterVersion(snuqlParser::VersionContext *ctx) override;
	void enterQuantumDecl(snuqlParser::QuantumDeclContext *) override;
	void enterClassicalDecl(snuqlParser::ClassicalDeclContext *) override;
	void enterGatedeclStatement(snuqlParser::GatedeclStatementContext *) override;
	void exitGatedeclStatement(snuqlParser::GatedeclStatementContext *ctx) override;
	void enterUnitaryOp(snuqlParser::UnitaryOpContext *ctx) override;

	void enterIfStatement(snuqlParser::IfStatementContext *ctx) override;
	void enterBarrierStatement(snuqlParser::BarrierStatementContext *ctx) override;
	void enterAnylist(snuqlParser::AnylistContext *ctx) override;

	void enterCustomOp(snuqlParser::CustomOpContext *ctx) override;
	void enterMeasureQop(snuqlParser::MeasureQopContext *ctx) override;
	void enterResetQop(snuqlParser::ResetQopContext *ctx) override;
};

} // namespace snuqs
