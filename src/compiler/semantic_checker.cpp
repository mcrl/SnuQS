#include "semantic_checker.h"

#include <cassert>


#include "logger.h"

namespace snuqs {

void SemanticChecker::check(antlr4::tree::ParseTree *tree) {
	antlr4::tree::ParseTreeWalker::DEFAULT.walk(this, tree);
}

const SymbolTable& SemanticChecker::getSymbolTable() const {
	return symtab_;
}

template<typename... Args>
void CheckWithMessage(bool e, Args... args)
{
	if (!e) {
		Logger::error(args...);
		std::exit(EXIT_FAILURE);
	}
}

template<typename... Args>
void CannotBeHere(Args... args)
{
  Logger::error(args...);
  std::exit(EXIT_FAILURE);
}

void DoNothing() {
  //
  // Do nothing deliberately.
  //
}

void SemanticChecker::checkIdentRedef(antlr4::tree::TerminalNode *id) const {
	CheckWithMessage(!symtab_.hasSymbol(id->getText()),
			"Redifinition of symbol {}.\n", id->getText());
}

void SemanticChecker::checkIdentUndefined(antlr4::tree::TerminalNode *id) const {
	CheckWithMessage(symtab_.hasSymbol(id->getText()),
			"Reference to undefinfed symbol {}.\n", id->getText());
}

void SemanticChecker::checkIdentType(antlr4::tree::TerminalNode *id, SymbolTable::SymbolType type) const {
	CheckWithMessage(symtab_.find(id->getText()).first == type,
			"Illegal reference to wrong type variable {}.\n", id->getText());
}

void SemanticChecker::checkUnitaryGateExp(snuqasmParser::ExpContext *exp) const {
	for (auto &&e: exp->exp()) {
		checkUnitaryGateExp(e);
	}

	if (exp->Identifier() != nullptr) {
		//checkIdentUndefined(exp->Identifier());
		checkIdentType(exp->Identifier(), SymbolTable::SymbolType::CPARAM);
	}
}

void SemanticChecker::checkUnitaryGateExplist(snuqasmParser::ExplistContext *explist) const {
	while (explist) {
		checkUnitaryGateExp(explist->exp());
		explist = explist->explist();
	}
}

void SemanticChecker::checkUnitaryGateArg(snuqasmParser::ArgumentContext* arg) const {
  auto ident = arg->Identifier();
	auto sym_type = symtab_.find(ident->getText()).first;
  CheckWithMessage(
      sym_type == SymbolTable::SymbolType::QPARAM 
      || sym_type == SymbolTable::SymbolType::QREG,
      "Illegal reference to wrong type variable {}: quatnum register or parameter expected.\n",
      arg->getText());

  if (sym_type == SymbolTable::SymbolType::QREG) {
    if (arg->Integer()) {
      auto qdecl = dynamic_cast<snuqasmParser::QuantumDeclContext*>(
          symtab_.find(ident->getText()).second);
      CheckWithMessage(
          std::stoul(arg->Integer()->getText()) < std::stoul(qdecl->Integer()->getText()),
          "Qreg reference dimension error.\n");
    }
  }
}

void SemanticChecker::checkUnitaryGateArgs(std::vector<snuqasmParser::ArgumentContext*> &&args) const {
	for (auto && arg : args) {
		checkUnitaryGateArg(arg);
	}

	if (args.size() > 0) {
		for (size_t i = 0; i < args.size()-1; ++i) {
			for (size_t j = i+1; j < args.size(); j++) {
				if (args[i]->Identifier()->getText() == 
						args[j]->Identifier()->getText()) {
					if (args[i]->Integer() && args[j]->Integer()) {
						CheckWithMessage(
								std::stoul(args[i]->Integer()->getText())
								!= std::stoul(args[j]->Integer()->getText()),
								"Qubits of two- or more qubit gates must be distinct.\n");
					}
				}
			}
		}
	}
}

size_t SemanticChecker::getExplistSize(snuqasmParser::ExplistContext *ctx) const {
	size_t cnt = 0;
	auto explist = ctx;
	while (explist) {
		cnt++;
		explist = explist->explist();
	}
	return cnt;
}

size_t SemanticChecker::getIdlistSize(snuqasmParser::IdlistContext *ctx) const {
	size_t cnt = 0;
	auto idlist = ctx;
	while (idlist) {
		cnt++;
		idlist = idlist->idlist();
	}
	return cnt;
}

size_t SemanticChecker::getMixedlistSize(snuqasmParser::MixedlistContext *ctx) const {
	if (ctx == nullptr)
		return 0;

  return 1+getMixedlistSize(ctx->mixedlist());
}

size_t SemanticChecker::getAnylistSize(snuqasmParser::AnylistContext *ctx) const {
	if (ctx->idlist()) {
		return getIdlistSize(ctx->idlist());
	} else {
		return getMixedlistSize(ctx->mixedlist());
	}
}

void SemanticChecker::checkUnitaryGate(snuqasmParser::UnitaryOpContext *ctx) const {
	checkUnitaryGateExplist(ctx->explist());
	checkUnitaryGateArgs(ctx->argument());

//	auto arglist = ctx->arglist();
//	while (arglist) {
//		checkUnitaryGateArg(arglist->argument());
//		arglist = arglist->arglist();
//	}
}

void SemanticChecker::checkCustomGate(snuqasmParser::CustomOpContext *ctx) const {
	checkIdentUndefined(ctx->Identifier());

	auto gdecl = dynamic_cast<snuqasmParser::GatedeclStatementContext*>(symtab_.find(ctx->Identifier()->getText()).second);
	if (gdecl->idlist().size() == 1) { // No Params
		CheckWithMessage(ctx->explist() == nullptr, "Gate {} does not have parameters.\n", ctx->Identifier()->getText());

		size_t narg = getAnylistSize(ctx->anylist());
		size_t nparam = getIdlistSize(gdecl->idlist()[0]);
		CheckWithMessage(narg == nparam, "Calling gate {} with wrong number of arguments.\n", ctx->Identifier()->getText());
	} else {
		CheckWithMessage(ctx->explist() != nullptr, "Gate {} must have parameters.\n", ctx->Identifier()->getText());

		size_t nexp = getExplistSize(ctx->explist());
		size_t nexp_required = getIdlistSize(gdecl->idlist()[0]);
		CheckWithMessage(nexp == nexp_required, "Calling gate {} with wrong number of expressions.\n", ctx->Identifier()->getText());

		size_t narg = getAnylistSize(ctx->anylist());
		size_t nparam = getIdlistSize(gdecl->idlist()[1]);
		CheckWithMessage(narg == nparam, "Calling gate {} with wrong number of arguments.\n", ctx->Identifier()->getText());
	}
	checkUnitaryGateExplist(ctx->explist());
}

void SemanticChecker::insertIdlistToSymtab(snuqasmParser::IdlistContext *idlist, SymbolTable::SymbolType type) {
	if (idlist == nullptr)
		return;

	symtab_.insert({idlist->Identifier()->getText(), {type, idlist}});
	insertIdlistToSymtab(idlist->idlist(), type);
}

void SemanticChecker::checkIdlist(snuqasmParser::IdlistContext *ctx) const {
	if (ctx == nullptr)
		return;

	checkIdentUndefined(ctx->Identifier());
	checkIdlist(ctx->idlist());
}

void SemanticChecker::checkMixedlist(snuqasmParser::MixedlistContext *ctx) const {
	/*
	if (ctx == nullptr)
		return;

	checkIdentUndefined(ctx->Identifier());
	if (ctx->designatedIdentifier()) {
	  ctx->designatedIdentifier()->Integer()
		auto qdecl = dynamic_cast<snuqasmParser::QuantumDeclContext*>(symtab_.find(ctx->Identifier()->getText()).second);
		CheckWithMessage(std::stoul(ctx->Integer()->getText()) < std::stoul(qdecl->Integer()->getText()),
						"Dimension mismatch.\n");
	}

	//checkIdlist(ctx->idlist());
	checkMixedlist(ctx->mixedlist());
	*/
}

template<typename... Args>
void CheckPositivity(antlr4::tree::TerminalNode* node, Args... args) {
  auto dim = std::stoul(node->getText());
  CheckWithMessage(
      dim > 0,
      args...
      );
}

template<typename... Args>
void CheckFirstDefinition(antlr4::tree::TerminalNode* node,
    const SymbolTable &symtab,
    Args... args) {
  CheckWithMessage(
      !symtab.HasSymbol(node->getText()),
      args...
      );
}

template<typename... Args>
void CheckDefinedSymbol(antlr4::tree::TerminalNode* node,
    const SymbolTable &symtab,
    Args... args) {
  CheckWithMessage(
      symtab.HasSymbol(node->getText()),
      args...
      );
}

template<typename... Args>
void CheckSymbolType(antlr4::tree::TerminalNode* node,
    const SymbolTable &symtab,
    Symbol::SymbolType type,
    Args... args) {
  CheckWithMessage(
      symtab.GetSymbolType(node->getText()) == type,
      args...
      );
}

void SemanticChecker::enterMainprogram(snuqasmParser::MainprogramContext * ctx) {
  DoNothing();
}

void SemanticChecker::exitMainprogram(snuqasmParser::MainprogramContext * ctx) {
  DoNothing();
}

void SemanticChecker::enterVersion(snuqasmParser::VersionContext *ctx) {
  double version = std::stod(ctx->Real()->getText());
	CheckWithMessage(version == 2.0, "Only version 2.0 is supported.\n");
}

void SemanticChecker::exitVersion(snuqasmParser::VersionContext * ctx) {
  DoNothing();
}

void SemanticChecker::enterProgram(snuqasmParser::ProgramContext * ctx) {
  DoNothing();
}

void SemanticChecker::exitProgram(snuqasmParser::ProgramContext * ctx) {
  DoNothing();
}

void SemanticChecker::enterInclude(snuqasmParser::IncludeContext *ctx) {
  CannotBeHere("All includes must be handled by preprocessor.");
}

void SemanticChecker::exitInclude(snuqasmParser::IncludeContext * ctx) {
  CannotBeHere("All includes must be handled by preprocessor.");
}

void SemanticChecker::enterStatement(snuqasmParser::StatementContext * ctx) {
  DoNothing();
}

void SemanticChecker::exitStatement(snuqasmParser::StatementContext * ctx) {
  DoNothing();
}

void SemanticChecker::enterDecl(snuqasmParser::DeclContext * ctx) {
  DoNothing();
}

void SemanticChecker::exitDecl(snuqasmParser::DeclContext * ctx) {
  DoNothing();
}

void SemanticChecker::enterQuantumDecl(snuqasmParser::QuantumDeclContext *ctx) {
  CheckPositivity(ctx->Integer(),
      "Qreg has to be larger than 0-dimension.\n"
      );

  CheckFirstDefinition(ctx->Identifier(), symtab_,
      "Redefinition of qreg {}\n", 
      ctx->Identifier()->getText()
      );

  symtab_.Insert({ctx->Identifier()->getText(), Symbol::SymbolType::QREG, ctx});
	symtab_.insert({ctx->Identifier()->getText(), {SymbolTable::SymbolType::QREG, ctx}});
}

void SemanticChecker::exitQuantumDecl(snuqasmParser::QuantumDeclContext * ctx) {
  DoNothing();
}

void SemanticChecker::enterClassicalDecl(snuqasmParser::ClassicalDeclContext *ctx) {
  CheckPositivity(ctx->Integer(),
      "Qreg has to be larger than 0-dimension.\n"
      );

  CheckFirstDefinition(ctx->Identifier(), symtab_,
      "Redefinition of creg {}\n", 
      ctx->Identifier()->getText()
      );

  symtab_.Insert({ctx->Identifier()->getText(), Symbol::SymbolType::CREG, ctx});
	symtab_.insert({ctx->Identifier()->getText(), {SymbolTable::SymbolType::CREG, ctx}});
}

void SemanticChecker::exitClassicalDecl(snuqasmParser::ClassicalDeclContext * ctx) {
  DoNothing();
}

void SemanticChecker::enterGatedeclStatement(snuqasmParser::GatedeclStatementContext *ctx) {
	/*
	snuqasmParser::GatedeclContext *gatedecl = ctx->gatedecl();

	checkIdentRedef(gatedecl->Identifier());
	symtab_.insert({gatedecl->Identifier()->getText(), {SymbolTable::SymbolType::GATE, ctx}});

	snuqasmParser::IdlistContext *arg_list = nullptr;
	snuqasmParser::IdlistContext *param_list = nullptr;
	if (gatedecl->idlist().size() == 2) {
		param_list = gatedecl->idlist(0);
		arg_list = gatedecl->idlist(1);
	} else { 
		arg_list = gatedecl->idlist(0);
	}

	symtab_.pushContext();

	insertIdlistToSymtab(arg_list, SymbolTable::SymbolType::QPARAM);
	insertIdlistToSymtab(param_list, SymbolTable::SymbolType::CPARAM);
	*/
  CheckFirstDefinition(ctx->Identifier(), symtab_,
      "Redefinition of gate {}\n", 
      ctx->Identifier()->getText()
      );
  symtab_.Insert({ctx->Identifier()->getText(), Symbol::SymbolType::GATE, ctx});
	symtab_.insert({ctx->Identifier()->getText(), {SymbolTable::SymbolType::GATE, ctx}});
}

void SemanticChecker::exitGatedeclStatement(snuqasmParser::GatedeclStatementContext *ctx) {  
	//symtab_.popContext();
	DoNothing();
}

void SemanticChecker::enterGoplist(snuqasmParser::GoplistContext * ctx) {
}

void SemanticChecker::exitGoplist(snuqasmParser::GoplistContext * ctx) {
}

void SemanticChecker::enterOpaqueDeclStatement(snuqasmParser::OpaqueDeclStatementContext * ctx) {
  CheckFirstDefinition(ctx->Identifier(), 
      symtab_,
      "Redefinition of opaque gate {}\n", 
      ctx->Identifier()->getText()
      );
  symtab_.Insert({ctx->Identifier()->getText(), Symbol::SymbolType::OPAQUE_GATE, ctx});
	symtab_.insert({ctx->Identifier()->getText(), {SymbolTable::SymbolType::OPAQUE_GATE, ctx}});
}

void SemanticChecker::exitOpaqueDeclStatement(snuqasmParser::OpaqueDeclStatementContext * ctx) {
  DoNothing();
}

void SemanticChecker::enterQopStatement(snuqasmParser::QopStatementContext * ctx) {
}

void SemanticChecker::exitQopStatement(snuqasmParser::QopStatementContext * ctx) {
}

void SemanticChecker::enterUopStatement(snuqasmParser::UopStatementContext * ctx) {
}

void SemanticChecker::exitUopStatement(snuqasmParser::UopStatementContext * ctx) {
}

void SemanticChecker::enterMeasureQop(snuqasmParser::MeasureQopContext *ctx) {

	auto qarg = ctx->argument(0);
	auto qtype = symtab_.find(qarg->Identifier()->getText()).first;
	CheckWithMessage(qtype == SymbolTable::SymbolType::QPARAM || qtype == SymbolTable::SymbolType::QREG,
			"Illegal reference to wrong type variable {}: quantum register expected.\n", qarg->Identifier()->getText());

	if (qarg->Integer()) {
		checkIdentType(qarg->Identifier(), SymbolTable::SymbolType::QREG);
		auto qdecl = dynamic_cast<snuqasmParser::QuantumDeclContext*>(symtab_.find(qarg->Identifier()->getText()).second);
		CheckWithMessage(std::stoul(qarg->Integer()->getText()) < std::stoul(qdecl->Integer()->getText()),
				"Qreg reference dimension error.\n");
	}

	auto carg = ctx->argument(1);
	auto ctype = symtab_.find(carg->Identifier()->getText()).first;
	CheckWithMessage(ctype == SymbolTable::SymbolType::CPARAM || ctype == SymbolTable::SymbolType::CREG,
			"Illegal reference to wrong type variable {}: classical register expected.\n", carg->Identifier()->getText());
	if (carg->Integer()) {
		checkIdentType(carg->Identifier(), SymbolTable::SymbolType::CREG);
		auto qdecl = dynamic_cast<snuqasmParser::ClassicalDeclContext*>(symtab_.find(carg->Identifier()->getText()).second);
		CheckWithMessage(std::stoul(carg->Integer()->getText()) < std::stoul(qdecl->Integer()->getText()),
				"Creg reference dimension error.\n");
	}

	CheckWithMessage(
		(qarg->Integer() == nullptr && carg->Integer() == nullptr)
		|| (qarg->Integer() != nullptr && carg->Integer() != nullptr),
		"Illegal measure operation.\n");

}

void SemanticChecker::exitMeasureQop(snuqasmParser::MeasureQopContext * ctx) {
}

void SemanticChecker::enterResetQop(snuqasmParser::ResetQopContext *ctx) {
	assert(false);
}

void SemanticChecker::exitResetQop(snuqasmParser::ResetQopContext * ctx) {
}

void SemanticChecker::enterIfStatement(snuqasmParser::IfStatementContext *ctx) {
  CheckDefinedSymbol(ctx->Identifier(), symtab_,
      "Undefined symbol {}.\n",
      ctx->Identifier()->getText()
      );

  CheckSymbolType(ctx->Identifier(), symtab_, Symbol::SymbolType::CREG,
    "Symbol {} must be of creg type\n",
    ctx->Identifier()->getText()
  );
}

void SemanticChecker::exitIfStatement(snuqasmParser::IfStatementContext * ctx) {
}

void SemanticChecker::enterBarrierStatement(snuqasmParser::BarrierStatementContext *ctx) {
  /*
  CheckAlreadyDefined(ctx->Identifier(), symtab_,
      "Redefinition of qreg {}\n", 
      ctx->Identifier()->getText()
      );
  */
}

void SemanticChecker::exitBarrierStatement(snuqasmParser::BarrierStatementContext * ctx) {
  DoNothing();
}

void SemanticChecker::enterUnitaryOp(snuqasmParser::UnitaryOpContext *ctx) {
	checkUnitaryGate(ctx);
}

void SemanticChecker::exitUnitaryOp(snuqasmParser::UnitaryOpContext * ctx) {
}

void SemanticChecker::enterCustomOp(snuqasmParser::CustomOpContext *ctx) {
	checkCustomGate(ctx);
}

void SemanticChecker::exitCustomOp(snuqasmParser::CustomOpContext * ctx) {
}

void SemanticChecker::enterAnylist(snuqasmParser::AnylistContext *ctx) {
	if (ctx->idlist()) {
		checkIdlist(ctx->idlist());
	} else {
		checkMixedlist(ctx->mixedlist());
	}
}

void SemanticChecker::exitAnylist(snuqasmParser::AnylistContext * ctx) {
}

void SemanticChecker::enterIdlist(snuqasmParser::IdlistContext * ctx) {
}

void SemanticChecker::exitIdlist(snuqasmParser::IdlistContext * ctx) {
}

void SemanticChecker::enterMixedlist(snuqasmParser::MixedlistContext * ctx) {
}

void SemanticChecker::exitMixedlist(snuqasmParser::MixedlistContext * ctx) {
}

void SemanticChecker::enterArglist(snuqasmParser::ArglistContext * ctx) {
}

void SemanticChecker::exitArglist(snuqasmParser::ArglistContext * ctx) {
}

void SemanticChecker::enterArgument(snuqasmParser::ArgumentContext * ctx) {
}

void SemanticChecker::exitArgument(snuqasmParser::ArgumentContext * ctx) {
}

void SemanticChecker::enterExplist(snuqasmParser::ExplistContext * ctx) {
  DoNothing();
}

void SemanticChecker::exitExplist(snuqasmParser::ExplistContext * ctx) {
  DoNothing();
}

void SemanticChecker::enterExp(snuqasmParser::ExpContext * ctx) {
  DoNothing();
}

void SemanticChecker::exitExp(snuqasmParser::ExpContext * ctx) {
  DoNothing();
}

void SemanticChecker::enterBinop(snuqasmParser::BinopContext * ctx) {
  DoNothing();
}

void SemanticChecker::exitBinop(snuqasmParser::BinopContext * ctx) {
  DoNothing();
}

void SemanticChecker::enterUnaryop(snuqasmParser::UnaryopContext * ctx) {
  DoNothing();
}

void SemanticChecker::exitUnaryop(snuqasmParser::UnaryopContext * ctx) {
  DoNothing();
}

void SemanticChecker::enterEveryRule(antlr4::ParserRuleContext * ctx) {
  DoNothing();
}

void SemanticChecker::exitEveryRule(antlr4::ParserRuleContext * ctx) {
  DoNothing();
}

void SemanticChecker::visitTerminal(antlr4::tree::TerminalNode * ctx) {
  DoNothing();
}

void SemanticChecker::visitErrorNode(antlr4::tree::ErrorNode * ctx) {
  DoNothing();
}


} // namespace snuqs
