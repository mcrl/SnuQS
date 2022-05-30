#include "circuitGenerator.hpp"
#include "gate.hpp"

#include <cmath>

namespace snuqs {

//
// Var
//
Var::Var(std::string s, size_t i) 
: s(s), i(i) {
	designated = true;
}

Var::Var(std::string s)
: s(s) {
}

Var::Var(size_t i)
: i(i) {
}

bool Var::operator==(const Var &other) const {
	return (s == other.s) && (i == other.i);
}

bool Var::operator<(const Var &other) const {
	return (s < other.s) || ((s == other.s) && (i < other.i));
}


//
// VarMapping
//
void VarMapping::insert(Var k, Var v) {
	map_.insert({k, v});
};

size_t VarMapping::findLast(const std::string &s) const {
	assert(map_.find({s}) != map_.end());
	auto var = map_.find({s})->second;

	if (var.s != "") {
		if (var.designated) {
			return parent->findLast(var.s, var.i);
		} else {
			return parent->findLast(var.s);
		}
	}
	return var.i;
}

size_t VarMapping::findLast(const std::string &s, size_t i) const {
	assert(map_.find({s, i}) != map_.end());
	auto var = map_.find({s, i})->second;

	if (var.s != "") {
		if (var.designated) {
			return parent->findLast(var.s, var.i);
		} else {
			return parent->findLast(var.s);
		}
	}
	return var.i;
}

//
// Expr
//
Expr::Expr(std::string s)
: op(ExprOp::IDENT), s(s) {
}

Expr::Expr(real_t d)
: op(ExprOp::CONST), d(d) {
}

Expr::Expr(ExprOp op, Expr *l)
: op(op), left(l)
{
}

Expr::Expr(ExprOp op, Expr *l, Expr *r)
: op(op), left(l), right(r)
{
}

bool Expr::operator<(const Expr &other) const {
	return ((op == ExprOp::IDENT) && (other.op == ExprOp::IDENT)) && (s < other.s);
}

//
// ExprMapping
//
void ExprMapping::insert(Expr k, Expr v) {
	map_.insert({k, v});
};

real_t ExprMapping::eval(const std::string &s) const {
	assert(map_.find({s}) != map_.end());
	auto expr = map_.find({s})->second;

	switch (expr.op) {
		case ExprOp::CONST: 
			return expr.d;
		case ExprOp::IDENT:
			return parent->eval(expr.s);
		case ExprOp::ADD:
		case ExprOp::SUB:
		case ExprOp::MUL:
		case ExprOp::DIV:
		case ExprOp::XOR:
		case ExprOp::NEG:
		case ExprOp::SIN:
		case ExprOp::COS:
		case ExprOp::TAN:
		case ExprOp::EXP:
		case ExprOp::LOG:
		case ExprOp::SQRT:
		assert(false);
	}

	return expr.d;
}

//
// CircuitGenerator
//
void CircuitGenerator::pushVarContext() {
	auto tmp = new VarMapping;
	tmp->parent = var_map_;
	var_map_ = tmp;
}

void CircuitGenerator::popVarContext() {
	auto tmp = var_map_;
	var_map_ = var_map_->parent;
	delete tmp;
}

void CircuitGenerator::pushExprContext() {
	auto tmp = new ExprMapping;
	tmp->parent = expr_map_;
	expr_map_ = tmp;
}

void CircuitGenerator::popExprContext() {
	auto tmp = expr_map_;
	expr_map_ = expr_map_->parent;
	delete tmp;
}

void CircuitGenerator::initializeMapping(std::ostream &os) {
	size_t qcnt = 0;
	size_t ccnt = 0;

	size_t nqreg = 0;
	size_t ncreg = 0;
	for (auto const& [key, val] : symtab_.ctx_->symmap_) {
		auto type = val.first;

		if (type == SymbolTable::sym_type::QREG) {
			auto qdecl = dynamic_cast<snuqlParser::QuantumDeclContext*>(val.second);
			auto s = qdecl->Identifier()->getText();
			auto n = std::stoul(qdecl->Integer()->getText());
			for (size_t i = 0; i < n; ++i) {
				var_map_->insert({s, i}, qcnt);
				qcnt++;
			}
			nqreg++;
		} else if (type == SymbolTable::sym_type::CREG) {
			auto cdecl = dynamic_cast<snuqlParser::ClassicalDeclContext*>(val.second);
			auto s = cdecl->Identifier()->getText();
			auto n = std::stoul(cdecl->Integer()->getText());
			for (size_t i = 0; i < n; ++i) {
				bit_map_.insert({{s, i}, ccnt});
				ccnt++;
			}
			ncreg++;
		}
	}

	assert(nqreg <= 1);
	assert(ncreg <= 1);

	circ_.set_num_qubits(qcnt);

//	for (auto &[key,val] : var_map_->map_) {
//		spdlog::info("{}[{}] -> {}[{}]", key.s, key.i, val.s, val.i);
//	}
}

const QuantumCircuit& CircuitGenerator::generate(antlr4::tree::ParseTree *tree, std::ostream &os) {
	pushVarContext();
	initializeMapping(os);
	antlr4::tree::ParseTreeWalker::DEFAULT.walk(this, tree);
	popVarContext();

//	for (auto &&g : circ_.gates()) {
//		os << *g << "\n";
//	}
	return circ_;
}

real_t CircuitGenerator::expToReal(snuqlParser::ExpContext *ctx) {
	if (ctx->exp().size() == 0) {
		if (ctx->Real()) {
			return std::stod(ctx->Real()->getText());
		} else if (ctx->Integer()) {
			return std::stod(ctx->Integer()->getText());
		} else if (ctx->getStart()->getText() == "pi") {
			return M_PI;
		} else {
			return expr_map_->eval(ctx->Identifier()->getText());
		}
	} else {
		if (ctx->binop()) {
			if (ctx->binop()->getText() == "+") {
				return expToReal(ctx->exp(0)) + expToReal(ctx->exp(1));
			} else if (ctx->binop()->getText() == "-") {
				return expToReal(ctx->exp(0)) - expToReal(ctx->exp(1));
			} else if (ctx->binop()->getText() == "*") {
				return expToReal(ctx->exp(0)) * expToReal(ctx->exp(1));
			} else if (ctx->binop()->getText() == "/") {
				return expToReal(ctx->exp(0)) / expToReal(ctx->exp(1));
			} else if (ctx->binop()->getText() == "^") {
				return (real_t)((size_t)expToReal(ctx->exp(0)) ^ (size_t)expToReal(ctx->exp(1)));
			}
		} else if (ctx->unaryop()) {
			if (ctx->unaryop()->getText() == "sin") {
				return sin(expToReal(ctx->exp(0)));
			} else if (ctx->unaryop()->getText() == "cos") {
				return cos(expToReal(ctx->exp(0)));
			} else if (ctx->unaryop()->getText() == "tan") {
				return tan(expToReal(ctx->exp(0)));
			} else if (ctx->unaryop()->getText() == "exp") {
				return exp(expToReal(ctx->exp(0)));
			} else if (ctx->unaryop()->getText() == "ln") {
				return log(expToReal(ctx->exp(0)));
			} else if (ctx->unaryop()->getText() == "sqrt") {
				return sqrt(expToReal(ctx->exp(0)));
			}
		} else if (ctx->getStart()->getText() == "(") {
			return expToReal(ctx->exp(0));
		} else {
			assert(ctx->getStart()->getText() == "-");
			return -expToReal(ctx->exp(0));
		}
	}

	assert(false);
	return -1.23456;
}

real_t CircuitGenerator::inlineExpToReal(snuqlParser::ExpContext *ctx) {
	if (ctx->exp().size() == 0) {
		if (ctx->Real()) {
			return std::stod(ctx->Real()->getText());
		} else if (ctx->Integer()) {
			return std::stod(ctx->Integer()->getText());
		} else if (ctx->getStart()->getText() == "pi") {
			return M_PI;
		} else {
			return expr_map_->eval(ctx->Identifier()->getText());
		}
	} else {
		if (ctx->binop()) {
			if (ctx->binop()->getText() == "+") {
				return inlineExpToReal(ctx->exp(0)) + inlineExpToReal(ctx->exp(1));
			} else if (ctx->binop()->getText() == "-") {
				return inlineExpToReal(ctx->exp(0)) - inlineExpToReal(ctx->exp(1));
			} else if (ctx->binop()->getText() == "*") {
				return inlineExpToReal(ctx->exp(0)) * inlineExpToReal(ctx->exp(1));
			} else if (ctx->binop()->getText() == "/") {
				return inlineExpToReal(ctx->exp(0)) / inlineExpToReal(ctx->exp(1));
			} else if (ctx->binop()->getText() == "^") {
				return (real_t)((size_t)inlineExpToReal(ctx->exp(0)) ^ (size_t)inlineExpToReal(ctx->exp(1)));
			}
		} else if (ctx->unaryop()) {
			if (ctx->unaryop()->getText() == "sin") {
				return sin(inlineExpToReal(ctx->exp(0)));
			} else if (ctx->unaryop()->getText() == "cos") {
				return cos(inlineExpToReal(ctx->exp(0)));
			} else if (ctx->unaryop()->getText() == "tan") {
				return tan(inlineExpToReal(ctx->exp(0)));
			} else if (ctx->unaryop()->getText() == "exp") {
				return exp(inlineExpToReal(ctx->exp(0)));
			} else if (ctx->unaryop()->getText() == "ln") {
				return log(inlineExpToReal(ctx->exp(0)));
			} else if (ctx->unaryop()->getText() == "sqrt") {
				return sqrt(inlineExpToReal(ctx->exp(0)));
			}
		} else if (ctx->getStart()->getText() == "(") {
			return inlineExpToReal(ctx->exp(0));
		} else {
			assert(ctx->getStart()->getText() == "-");
			return -inlineExpToReal(ctx->exp(0));
		}
	}

	assert(false);
	return -1.23456;
}

std::vector<real_t> CircuitGenerator::explistToReal(snuqlParser::ExplistContext *ctx) {
	std::vector<real_t> reals;

	auto explist = ctx;
	while (explist) {
		reals.push_back(expToReal(explist->exp()));
		explist = explist->explist();
	}

	return reals;
}

size_t CircuitGenerator::argumentToNumber(snuqlParser::ArgumentContext *ctx) {
	assert(ctx->Integer() != nullptr);

	std::string s = ctx->Identifier()->getText();
	size_t i = std::stoul(ctx->Integer()->getText());
	return var_map_->findLast(s, i);
}

std::vector<size_t> CircuitGenerator::arglistToNumberList(snuqlParser::ArglistContext* arglist) {
	std::vector<size_t> numlist;
	while (arglist) {
		numlist.push_back(argumentToNumber(arglist->argument()));
		arglist = arglist->arglist();
	}
	return numlist;
}

std::vector<size_t> CircuitGenerator::argumentsToNumberList(std::vector<snuqlParser::ArgumentContext*> &&arglist) {
	std::vector<size_t> numlist;
	for (auto &&arg : arglist) {
		numlist.push_back(argumentToNumber(arg));
	}
	return numlist;
}

std::vector<real_t> CircuitGenerator::inlineExplistToReal(snuqlParser::ExplistContext *ctx) {
	std::vector<real_t> reals;

	auto explist = ctx;
	while (explist) {
		reals.push_back(inlineExpToReal(explist->exp()));
		explist = explist->explist();
	}

	return reals;
}

size_t CircuitGenerator::inlineArgumentToNumber(snuqlParser::ArgumentContext *ctx) {
	if (ctx->Integer()) {
		std::string s = ctx->Identifier()->getText();
		size_t i = std::stoul(ctx->Integer()->getText());
		return var_map_->findLast(s, i);
	} else {
		std::string s = ctx->Identifier()->getText();
		return var_map_->findLast(s);
	}
	return 0;
}

std::vector<size_t> CircuitGenerator::inlineArgumentToNumberList(std::vector<snuqlParser::ArgumentContext*> &&arglist) {
	std::vector<size_t> numlist;
	for (auto &&arg : arglist) {
		numlist.push_back(inlineArgumentToNumber(arg));
	}
	return numlist;
}

void CircuitGenerator::generateUnitaryOp(snuqlParser::UnitaryOpContext *ctx) {
	if (ctx->getStart()->getText() == "nswap") {
		circ_.addGate(gateFactory(ctx->getStart()->getText(),
					arglistToNumberList(ctx->arglist()),
					explistToReal(ctx->explist())));
	} else {
		circ_.addGate(gateFactory(ctx->getStart()->getText(),
					argumentsToNumberList(ctx->argument()),
					explistToReal(ctx->explist())));
	}
}

void CircuitGenerator::generateUopStatement(snuqlParser::UopStatementContext *ctx) {
	if (ctx->unitaryOp()) {
		generateUnitaryOp(ctx->unitaryOp());
	} else {
		generateCustomOp(ctx->customOp());
	}
}

void CircuitGenerator::inlineUnitaryOp(snuqlParser::UnitaryOpContext *ctx) {
	circ_.addGate(gateFactory(ctx->getStart()->getText(),
			inlineArgumentToNumberList(ctx->argument()),
			inlineExplistToReal(ctx->explist())));
}

void CircuitGenerator::inlineCustomOp(snuqlParser::CustomOpContext *ctx) {
	generateCustomOp(ctx);
}

void CircuitGenerator::inlineUopStatement(snuqlParser::UopStatementContext *ctx) {
	if (ctx->unitaryOp()) {
		inlineUnitaryOp(ctx->unitaryOp());
	} else {
		inlineCustomOp(ctx->customOp());
	}
}

void CircuitGenerator::idlistToMappings(snuqlParser::IdlistContext* idlist, std::vector<Var> &mappings) {
	while (idlist) {
		mappings.emplace_back(idlist->Identifier()->getText());
		idlist = idlist->idlist();
	}
}

void CircuitGenerator::idlistToExprMappings(snuqlParser::IdlistContext* idlist, std::vector<Expr> &mappings) {
	while (idlist) {
		mappings.emplace_back(idlist->Identifier()->getText());
		idlist = idlist->idlist();
	}
}

void CircuitGenerator::explistToExprMappings(snuqlParser::ExplistContext* explist, std::vector<Expr> &mappings) {
	while (explist) {
		mappings.push_back(expToReal(explist->exp()));
		explist = explist->explist();
	}
}

void CircuitGenerator::mixedlistToMappings(snuqlParser::MixedlistContext* mixedlist, std::vector<Var> &mappings) {
	if (mixedlist == nullptr)
		return;

	if (mixedlist->Integer()) {
		mappings.emplace_back(mixedlist->Identifier()->getText(), std::stoul(mixedlist->Integer()->getText()));
	} else {
		mappings.emplace_back(mixedlist->Identifier()->getText());
	}

	if (mixedlist->idlist()) {
		idlistToMappings(mixedlist->idlist(), mappings);
	} else {
		mixedlistToMappings(mixedlist->mixedlist(), mappings);
	}
}

void CircuitGenerator::anylistToMappings(snuqlParser::AnylistContext* anylist, std::vector<Var> &mappings) {
	if (anylist->idlist()) {
		idlistToMappings(anylist->idlist(), mappings);
	} else {
		mixedlistToMappings(anylist->mixedlist(), mappings);
	}
}

void CircuitGenerator::generateCustomOp(snuqlParser::CustomOpContext *ctx) {
	//assert(false);
	auto gdecl = dynamic_cast<snuqlParser::GatedeclStatementContext*>(symtab_.find(ctx->Identifier()->getText()).second);
	auto gatedecl = gdecl->gatedecl();

	if (gatedecl->idlist().size() == 1) {
		std::vector<Var> qparams;
		idlistToMappings(gatedecl->idlist()[0], qparams);
		std::vector<Var> qargs;
		anylistToMappings(ctx->anylist(), qargs);

		assert(qparams.size() == qargs.size());

		pushVarContext();
		for (size_t i = 0; i < qparams.size(); ++i) {
			var_map_->insert(qparams[i], qargs[i]);
		}
	} else {
		std::vector<Expr> cparams;
		idlistToExprMappings(gatedecl->idlist()[0], cparams);
		std::vector<Expr> cargs;
		explistToExprMappings(ctx->explist(), cargs);

		assert(cparams.size() == cargs.size());
		pushExprContext();
		for (size_t i = 0; i < cparams.size(); ++i) {
			expr_map_->insert(cparams[i], cargs[i]);
		}

		std::vector<Var> qparams;
		idlistToMappings(gatedecl->idlist()[1], qparams);
		std::vector<Var> qargs;
		anylistToMappings(ctx->anylist(), qargs);

		assert(qparams.size() == qargs.size());

		pushVarContext();
		for (size_t i = 0; i < qparams.size(); ++i) {
			var_map_->insert(qparams[i], qargs[i]);
		}
	}


	auto goplist = gdecl->goplist();
	while (goplist) {
		if (goplist->uopStatement()) {
			inlineUopStatement(goplist->uopStatement());
		} else {
			assert(false);
		}
		goplist = goplist->goplist();
	}

	if (gatedecl->idlist().size() != 1) {
		popExprContext();
	}
	popVarContext();
}


void CircuitGenerator::generateMeasureQop(snuqlParser::MeasureQopContext *ctx) {
	assert(false);
}

void CircuitGenerator::generateResetQop(snuqlParser::ResetQopContext *ctx) {
	assert(false);
}

void CircuitGenerator::generateQopStatement(snuqlParser::QopStatementContext *ctx) {
	if (ctx->uopStatement()) {
		generateUopStatement(ctx->uopStatement());
	} else if (ctx->measureQop()) {
		generateMeasureQop(ctx->measureQop());
	} else if (ctx->resetQop()) {
		generateResetQop(ctx->resetQop());
	} else {
		assert(false);
	}
}

void CircuitGenerator::generateIfStatement(snuqlParser::IfStatementContext *ctx) {
	assert(false);
}

void CircuitGenerator::generateBarrierStatement(snuqlParser::BarrierStatementContext *ctx) {
	assert(false);
}

void CircuitGenerator::enterStatement(snuqlParser::StatementContext *ctx) {
	if (ctx->qopStatement()) {
		generateQopStatement(ctx->qopStatement());
	} else if (ctx->ifStatement()) {
		generateIfStatement(ctx->ifStatement());
	} else if (ctx->barrierStatement()) {
		generateBarrierStatement(ctx->barrierStatement());
	}
}

} // namespace snuqs
