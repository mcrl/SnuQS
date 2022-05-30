#include "semanticChecker.hpp"

#include <spdlog/spdlog.h>

namespace snuqs {

void SemanticChecker::check(antlr4::tree::ParseTree *tree) {
	antlr4::tree::ParseTreeWalker::DEFAULT.walk(this, tree);
}

const SymbolTable& SemanticChecker::getSymbolTable() const {
	return symtab_;
}

template<typename... Args>
void checkWithMessage(bool e, Args... args)
{
	if (!e) {
		spdlog::error(args...);
		std::exit(EXIT_FAILURE);
	}
}

void SemanticChecker::checkIdentRedef(antlr4::tree::TerminalNode *id) const {
	checkWithMessage(!symtab_.hasSymbol(id->getText()),
			"Redifinition of symbol {}.", id->getText());
}

void SemanticChecker::checkIdentUndefined(antlr4::tree::TerminalNode *id) const {
	checkWithMessage(symtab_.hasSymbol(id->getText()),
			"Reference to undefinfed symbol {}.", id->getText());
}

void SemanticChecker::checkIdentType(antlr4::tree::TerminalNode *id, SymbolTable::sym_type type) const {
	checkWithMessage(symtab_.find(id->getText()).first == type,
			"Illegal reference to wrong type variable {}", id->getText());
}

void SemanticChecker::checkUnitaryGateExp(snuqlParser::ExpContext *exp) const {
	for (auto &&e: exp->exp()) {
		checkUnitaryGateExp(e);
	}

	if (exp->Identifier() != nullptr) {
		//checkIdentUndefined(exp->Identifier());
		checkIdentType(exp->Identifier(), SymbolTable::sym_type::CPARAM);
	}
}

void SemanticChecker::checkUnitaryGateExplist(snuqlParser::ExplistContext *explist) const {
	while (explist) {
		checkUnitaryGateExp(explist->exp());
		explist = explist->explist();
	}
}

void SemanticChecker::checkUnitaryGateArg(snuqlParser::ArgumentContext* arg) const {
	if (arg->Integer() == nullptr) {
		checkIdentType(arg->Identifier(), SymbolTable::sym_type::QPARAM);
	} else {
		checkIdentType(arg->Identifier(), SymbolTable::sym_type::QREG);

		auto qdecl = dynamic_cast<snuqlParser::QuantumDeclContext*>(symtab_.find(arg->Identifier()->getText()).second);
		checkWithMessage(std::stoul(arg->Integer()->getText()) < std::stoul(qdecl->Integer()->getText()),
				"Qreg reference dimension error");
	}
}

void SemanticChecker::checkUnitaryGateArgs(std::vector<snuqlParser::ArgumentContext*> &&args) const {
	for (auto && arg : args) {
		checkUnitaryGateArg(arg);
	}

	if (args.size() > 0) {
		for (size_t i = 0; i < args.size()-1; ++i) {
			for (size_t j = i+1; j < args.size(); j++) {
				if (args[i]->Identifier()->getText() == 
						args[j]->Identifier()->getText()) {
					if (args[i]->Integer() && args[j]->Integer()) {
						checkWithMessage(
								std::stoul(args[i]->Integer()->getText())
								!= std::stoul(args[j]->Integer()->getText()),
								"Qubits of two- or more qubit gates must be distinct.");
					}
				}
			}
		}
	}
}

size_t SemanticChecker::getExplistSize(snuqlParser::ExplistContext *ctx) const {
	size_t cnt = 0;
	auto explist = ctx;
	while (explist) {
		cnt++;
		explist = explist->explist();
	}
	return cnt;
}

size_t SemanticChecker::getIdlistSize(snuqlParser::IdlistContext *ctx) const {
	size_t cnt = 0;
	auto idlist = ctx;
	while (idlist) {
		cnt++;
		idlist = idlist->idlist();
	}
	return cnt;
}

size_t SemanticChecker::getMixedlistSize(snuqlParser::MixedlistContext *ctx) const {
	if (ctx == nullptr)
		return 0;

	if (ctx->idlist()) {
		return 1+getIdlistSize(ctx->idlist());
	} else {
		return 1+getMixedlistSize(ctx->mixedlist());
	}
}

size_t SemanticChecker::getAnylistSize(snuqlParser::AnylistContext *ctx) const {
	if (ctx->idlist()) {
		return getIdlistSize(ctx->idlist());
	} else {
		return getMixedlistSize(ctx->mixedlist());
	}
}

void SemanticChecker::checkUnitaryGate(snuqlParser::UnitaryOpContext *ctx) const {
	checkUnitaryGateExplist(ctx->explist());
	checkUnitaryGateArgs(ctx->argument());

	auto arglist = ctx->arglist();
	while (arglist) {
		checkUnitaryGateArg(arglist->argument());
		arglist = arglist->arglist();
	}
}

void SemanticChecker::checkCustomGate(snuqlParser::CustomOpContext *ctx) const {
	checkIdentUndefined(ctx->Identifier());

	auto gdecl = dynamic_cast<snuqlParser::GatedeclStatementContext*>(symtab_.find(ctx->Identifier()->getText()).second);
	if (gdecl->gatedecl()->idlist().size() == 1) {
		checkWithMessage(ctx->explist() == nullptr, "Gate {} does not have parameters.", ctx->Identifier()->getText());

		size_t narg = getAnylistSize(ctx->anylist());
		size_t nparam = getIdlistSize(gdecl->gatedecl()->idlist()[0]);
		checkWithMessage(narg == nparam, "Calling gate {} with wrong number of arguments.", ctx->Identifier()->getText());
	} else {
		checkWithMessage(ctx->explist() != nullptr, "Gate {} must have parameters", ctx->Identifier()->getText());

		size_t nexp = getExplistSize(ctx->explist());
		size_t nexp_required = getIdlistSize(gdecl->gatedecl()->idlist()[0]);
		checkWithMessage(nexp == nexp_required, "Calling gate {} with wrong number of expressions.", ctx->Identifier()->getText());

		size_t narg = getAnylistSize(ctx->anylist());
		size_t nparam = getIdlistSize(gdecl->gatedecl()->idlist()[1]);
		checkWithMessage(narg == nparam, "Calling gate {} with wrong number of arguments.", ctx->Identifier()->getText());
	}
	checkUnitaryGateExplist(ctx->explist());
}

void SemanticChecker::enterVersion(snuqlParser::VersionContext *ctx) {
	double version = 0.0;
	
	if (ctx->Integer()) {
		version = std::stod(ctx->Integer()->getText());
	} else { // if (ctx->Real())
		version = std::stod(ctx->Real()->getText());
	}

	checkWithMessage(version == 2.0, "The version mismatch.");
}

void SemanticChecker::enterQuantumDecl(snuqlParser::QuantumDeclContext *ctx) {

	checkWithMessage(std::stoul(ctx->Integer()->getText()) > 0,
			"Qreg has to be larger than 0-dimension");

	symtab_.insert({ctx->Identifier()->getText(), {SymbolTable::sym_type::QREG, ctx}});
}

void SemanticChecker::enterClassicalDecl(snuqlParser::ClassicalDeclContext *ctx) {
	checkWithMessage(std::stoul(ctx->Integer()->getText()) > 0,
			"Creg has to be larger than 0-dimension");

	checkIdentRedef(ctx->Identifier());

	symtab_.insert({ctx->Identifier()->getText(), {SymbolTable::sym_type::CREG, ctx}});
}

void SemanticChecker::insertIdlistToSymtab(snuqlParser::IdlistContext *idlist, SymbolTable::sym_type type) {
	if (idlist == nullptr)
		return;

	symtab_.insert({idlist->Identifier()->getText(), {type, idlist}});
	insertIdlistToSymtab(idlist->idlist(), type);
}

void SemanticChecker::enterGatedeclStatement(snuqlParser::GatedeclStatementContext *ctx) {
	snuqlParser::GatedeclContext *gatedecl = ctx->gatedecl();

	checkIdentRedef(gatedecl->Identifier());
	symtab_.insert({gatedecl->Identifier()->getText(), {SymbolTable::sym_type::GATE, ctx}});

	snuqlParser::IdlistContext *arg_list = nullptr;
	snuqlParser::IdlistContext *param_list = nullptr;
	if (gatedecl->idlist().size() == 2) {
		param_list = gatedecl->idlist(0);
		arg_list = gatedecl->idlist(1);
	} else { 
		arg_list = gatedecl->idlist(0);
	}

	symtab_.pushContext();

	insertIdlistToSymtab(arg_list, SymbolTable::sym_type::QPARAM);
	insertIdlistToSymtab(param_list, SymbolTable::sym_type::CPARAM);
}

void SemanticChecker::exitGatedeclStatement(snuqlParser::GatedeclStatementContext *ctx) {  
	symtab_.popContext();
}

void SemanticChecker::enterIfStatement(snuqlParser::IfStatementContext *ctx) {
	checkIdentUndefined(ctx->Identifier());
	checkIdentType(ctx->Identifier(), SymbolTable::sym_type::CREG);
}

void SemanticChecker::enterBarrierStatement(snuqlParser::BarrierStatementContext *ctx) {
}

void SemanticChecker::checkIdlist(snuqlParser::IdlistContext *ctx) const {
	if (ctx == nullptr)
		return;

	checkIdentUndefined(ctx->Identifier());
	checkIdlist(ctx->idlist());
}

void SemanticChecker::checkMixedlist(snuqlParser::MixedlistContext *ctx) const {
	if (ctx == nullptr)
		return;

	checkIdentUndefined(ctx->Identifier());
	if (ctx->Integer()) {
		auto qdecl = dynamic_cast<snuqlParser::QuantumDeclContext*>(symtab_.find(ctx->Identifier()->getText()).second);
		checkWithMessage(std::stoul(ctx->Integer()->getText()) < std::stoul(qdecl->Integer()->getText()),
						"Dimension mismatch.");
	}

	checkIdlist(ctx->idlist());
	checkMixedlist(ctx->mixedlist());
}

void SemanticChecker::enterUnitaryOp(snuqlParser::UnitaryOpContext *ctx) {
	checkUnitaryGate(ctx);
}

void SemanticChecker::enterCustomOp(snuqlParser::CustomOpContext *ctx) {
	checkCustomGate(ctx);
}

void SemanticChecker::enterMeasureQop(snuqlParser::MeasureQopContext *ctx) {

	auto qarg = ctx->argument(0);
	auto qtype = symtab_.find(qarg->Identifier()->getText()).first;
	checkWithMessage(qtype == SymbolTable::sym_type::QPARAM || qtype == SymbolTable::sym_type::QREG,
			"Illegal reference to wrong type variable {}", qarg->Identifier()->getText());

	if (qarg->Integer()) {
		checkIdentType(qarg->Identifier(), SymbolTable::sym_type::QREG);
		auto qdecl = dynamic_cast<snuqlParser::QuantumDeclContext*>(symtab_.find(qarg->Identifier()->getText()).second);
		checkWithMessage(std::stoul(qarg->Integer()->getText()) < std::stoul(qdecl->Integer()->getText()),
				"Qreg reference dimension error");
	}

	auto carg = ctx->argument(1);
	auto ctype = symtab_.find(carg->Identifier()->getText()).first;
	checkWithMessage(ctype == SymbolTable::sym_type::CPARAM || ctype == SymbolTable::sym_type::CREG,
			"Illegal reference to wrong type variable {}", carg->Identifier()->getText());
	if (carg->Integer()) {
		checkIdentType(carg->Identifier(), SymbolTable::sym_type::CREG);
		auto qdecl = dynamic_cast<snuqlParser::ClassicalDeclContext*>(symtab_.find(carg->Identifier()->getText()).second);
		checkWithMessage(std::stoul(carg->Integer()->getText()) < std::stoul(qdecl->Integer()->getText()),
				"Creg reference dimension error");
	}

	checkWithMessage(
		(qarg->Integer() == nullptr && carg->Integer() == nullptr)
		|| (qarg->Integer() != nullptr && carg->Integer() != nullptr),
		"Illegal measure operation");

}

void SemanticChecker::enterResetQop(snuqlParser::ResetQopContext *ctx) {
	assert(false);
}

void SemanticChecker::enterAnylist(snuqlParser::AnylistContext *ctx) {
	if (ctx->idlist()) {
		checkIdlist(ctx->idlist());
	} else {
		checkMixedlist(ctx->mixedlist());
	}
}


} // namespace snuqs
