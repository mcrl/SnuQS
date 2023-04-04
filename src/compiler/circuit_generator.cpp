#include "circuit_generator.h"

#include <cmath>

#include "circuit/gate.h"
#include "circuit/gate_factory.h"
#include "logger.h"

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
    return parent->findLast(var.s);
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

void CircuitGenerator::InitializeMapping() {
	size_t qcnt = 0;
	size_t ccnt = 0;

	size_t nqreg = 0;
	size_t ncreg = 0;
	for (auto const& it : symtab_.ctx_->symmap_) {
	  auto key = it.first;
	  auto val = it.second;
		auto type = val.first;

		if (type == SymbolTable::SymbolType::QREG) {
			auto qdecl = dynamic_cast<snuqasmParser::QuantumDeclContext*>(val.second);
			auto s = qdecl->Identifier()->getText();
			auto n = std::stoul(qdecl->Integer()->getText());
      var_map_->insert(s, qcnt);
			qcnt += n;
			nqreg++;
		} else if (type == SymbolTable::SymbolType::CREG) {
			auto cdecl = dynamic_cast<snuqasmParser::ClassicalDeclContext*>(val.second);
			auto s = cdecl->Identifier()->getText();
			auto n = std::stoul(cdecl->Integer()->getText());
      bit_map_.insert({s, ccnt});
			ccnt += n;
			ncreg++;
		}
	}

	circ_.set_num_qubits(qcnt);

}

const QuantumCircuit& CircuitGenerator::Generate(antlr4::tree::ParseTree *tree) {
	pushVarContext();
	InitializeMapping();
	antlr4::tree::ParseTreeWalker::DEFAULT.walk(this, tree);
	popVarContext();

	return circ_;
}

real_t CircuitGenerator::expToReal(snuqasmParser::ExpContext *ctx) {
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

real_t CircuitGenerator::inlineExpToReal(snuqasmParser::ExpContext *ctx) {
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

std::vector<real_t> CircuitGenerator::explistToReal(snuqasmParser::ExplistContext *ctx) {
	std::vector<real_t> reals;

	auto explist = ctx;
	while (explist) {
		reals.push_back(expToReal(explist->exp()));
		explist = explist->explist();
	}

	return reals;
}

size_t CircuitGenerator::ArgumentToNumber(snuqasmParser::ArgumentContext *ctx) {
  if (ctx->Integer() != nullptr) {
    std::string s = ctx->Identifier()->getText();
    size_t i = std::stoul(ctx->Integer()->getText());
    return var_map_->findLast(s) + i;
  } else {
    auto qdecl = dynamic_cast<snuqasmParser::QuantumDeclContext*>(
        symtab_.find(ctx->Identifier()->getText()).second);
  }
}

std::vector<size_t> CircuitGenerator::ArgumentToNumberList(std::vector<snuqasmParser::ArgumentContext*> &&arglist) {
	std::vector<size_t> numlist;
	for (auto &&arg : arglist) {
		numlist.push_back(ArgumentToNumber(arg));
	}
	return numlist;
}

std::vector<real_t> CircuitGenerator::inlineExplistToReal(snuqasmParser::ExplistContext *ctx) {
	std::vector<real_t> reals;

	auto explist = ctx;
	while (explist) {
		reals.push_back(inlineExpToReal(explist->exp()));
		explist = explist->explist();
	}

	return reals;
}

size_t CircuitGenerator::inlineArgumentToNumber(snuqasmParser::ArgumentContext *ctx) {
	if (ctx->Integer()) {
		std::string s = ctx->Identifier()->getText();
		size_t i = std::stoul(ctx->Integer()->getText());
		return var_map_->findLast(s) + i;
	} else {
		std::string s = ctx->Identifier()->getText();
		return var_map_->findLast(s);
	}
	return 0;
}

std::vector<size_t> CircuitGenerator::inlineArgumentToNumberList(std::vector<snuqasmParser::ArgumentContext*> &&arglist) {
	std::vector<size_t> numlist;
	for (auto &&arg : arglist) {
		numlist.push_back(inlineArgumentToNumber(arg));
	}
	return numlist;
}

void CircuitGenerator::GenerateUnitaryOp(snuqasmParser::UnitaryOpContext *ctx) {

  const auto &arglist = ctx->argument();

  size_t max_dim = 1;
  for (auto &arg : arglist) {
    if (arg->Integer() == nullptr) {
      auto decl = symtab_.find(arg->Identifier()->getText()).second;
      auto qdecl = dynamic_cast<snuqasmParser::QuantumDeclContext*>(decl);
      max_dim = std::max(max_dim, std::stoul(qdecl->Integer()->getText()));
    }
  }

  if (max_dim == 1) { // Without broadcast
    std::vector<size_t> qubit_ids(arglist.size());
    for (int i = 0; i < arglist.size(); ++i) {
      auto arg  = arglist[i];
      auto idx = std::stoul(arg->Integer()->getText());
      auto qubit_id = var_map_->findLast(arg->Identifier()->getText()) + idx;
      qubit_ids[i] = qubit_id;
    }
    circ_.addGate(GateFactory::CreateGate(
          ctx->getStart()->getText(),
          qubit_ids,
          explistToReal(ctx->explist())));
  } else { // With broadcast
    std::vector<std::vector<size_t>> qubit_id_vecs(arglist.size());
    for (int i = 0; i < arglist.size(); ++i) {
      auto arg  = arglist[i];
      qubit_id_vecs[i].reserve(max_dim);
      if (arg->Integer() != nullptr) {
        auto idx = std::stoul(arg->Integer()->getText());
        auto qubit_id = var_map_->findLast(arg->Identifier()->getText()) + idx;
        for (int j = 0; j < max_dim; ++j) {
          qubit_id_vecs[i][j] = qubit_id;
        }
      } else {
        auto qubit_id = var_map_->findLast(arg->Identifier()->getText());
        for (int j = 0; j < max_dim; ++j) {
          qubit_id_vecs[i][j] = qubit_id + j;
        }
      }
    }
    for (int i = 0; i < arglist.size(); ++i) {
        for (int j = 0; j < max_dim; ++j) {
          Logger::debug("id_vecs[{}][{}]: {}\n", i, j, j);
        }
    }
    auto expr = explistToReal(ctx->explist());
    for (int j = 0; j < max_dim; ++j) {
      std::vector<size_t> qubit_ids(arglist.size());
      for (int i = 0; i < arglist.size(); ++i) {
        qubit_ids[i] = qubit_id_vecs[i][j];
      }
      circ_.addGate(GateFactory::CreateGate(
            ctx->getStart()->getText(),
            qubit_ids,
            expr));
    }
  }

}

void CircuitGenerator::GenerateUopStatement(snuqasmParser::UopStatementContext *ctx) {
	if (ctx->unitaryOp()) {
		GenerateUnitaryOp(ctx->unitaryOp());
	} else {
		GenerateCustomOp(ctx->customOp());
	}
}

void CircuitGenerator::inlineUnitaryOp(snuqasmParser::UnitaryOpContext *ctx) {
	circ_.addGate(GateFactory::CreateGate(ctx->getStart()->getText(),
			inlineArgumentToNumberList(ctx->argument()),
			inlineExplistToReal(ctx->explist())));
}

void CircuitGenerator::inlineCustomOp(snuqasmParser::CustomOpContext *ctx) {
	GenerateCustomOp(ctx);
}

void CircuitGenerator::inlineUopStatement(snuqasmParser::UopStatementContext *ctx) {
	if (ctx->unitaryOp()) {
		inlineUnitaryOp(ctx->unitaryOp());
	} else {
		inlineCustomOp(ctx->customOp());
	}
}

void CircuitGenerator::idlistToMappings(snuqasmParser::IdlistContext* idlist, std::vector<Var> &mappings) {
	while (idlist) {
	  Logger::debug("idlist: {}\n", idlist->Identifier()->getText());
		mappings.emplace_back(idlist->Identifier()->getText());
		idlist = idlist->idlist();
	}
}

void CircuitGenerator::idlistToExprMappings(snuqasmParser::IdlistContext* idlist, std::vector<Expr> &mappings) {
	while (idlist) {
		mappings.emplace_back(idlist->Identifier()->getText());
		idlist = idlist->idlist();
	}
}

void CircuitGenerator::explistToExprMappings(snuqasmParser::ExplistContext* explist, std::vector<Expr> &mappings) {
	while (explist) {
		mappings.push_back(expToReal(explist->exp()));
		explist = explist->explist();
	}
}

void CircuitGenerator::mixedlistToMappings(snuqasmParser::MixedlistContext* mixedlist, std::vector<Var> &mappings) {
	if (mixedlist == nullptr)
		return;

	if (mixedlist->designatedIdentifier()) {
		mappings.emplace_back(mixedlist->designatedIdentifier()->Identifier()->getText(),
        std::stoul(mixedlist->designatedIdentifier()->Integer()->getText()));
	} else {
		mappings.emplace_back(mixedlist->Identifier()->getText());
	}

  mixedlistToMappings(mixedlist->mixedlist(), mappings);
}

void CircuitGenerator::anylistToMappings(snuqasmParser::AnylistContext* anylist, std::vector<Var> &mappings) {
	if (anylist->idlist()) {
		idlistToMappings(anylist->idlist(), mappings);
	} else {
		mixedlistToMappings(anylist->mixedlist(), mappings);
	}
}

void CircuitGenerator::GenerateCustomOp(snuqasmParser::CustomOpContext *ctx) {
	//assert(false);
	Logger::debug("Custom op {}\n", ctx->getText());
	auto gdecl = dynamic_cast<snuqasmParser::GatedeclStatementContext*>(symtab_.find(ctx->Identifier()->getText()).second);

	if (gdecl->idlist().size() == 1) { // No param expression
		std::vector<Var> qparams;
		idlistToMappings(gdecl->idlist()[0], qparams);
		std::vector<Var> qargs;
		anylistToMappings(ctx->anylist(), qargs);

		assert(qparams.size() == qargs.size());

		pushVarContext();
		for (size_t i = 0; i < qparams.size(); ++i) {
			var_map_->insert(qparams[i], qargs[i]);
		}
	} else {
		std::vector<Expr> cparams;
		idlistToExprMappings(gdecl->idlist()[0], cparams);
		std::vector<Expr> cargs;
		explistToExprMappings(ctx->explist(), cargs);

		assert(cparams.size() == cargs.size());
		pushExprContext();
		for (size_t i = 0; i < cparams.size(); ++i) {
			expr_map_->insert(cparams[i], cargs[i]);
		}

		std::vector<Var> qparams;
		idlistToMappings(gdecl->idlist()[1], qparams);
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
		} else if (goplist->barrierStatement()) {
		  GenerateBarrierStatement(goplist->barrierStatement());
    } else {
			assert(false);
		}
		goplist = goplist->goplist();
	}

	if (gdecl->idlist().size() != 1) {
		popExprContext();
	}
	popVarContext();
}


void CircuitGenerator::GenerateMeasureQop(snuqasmParser::MeasureQopContext *ctx) {
  auto qarg = ctx->argument(0);
  auto carg = ctx->argument(1);
  auto qident = qarg->Identifier();
  auto cident = carg->Identifier();
  if (qarg->Integer()) {
    auto s = std::string(qident->getText());
    auto n = std::stoul(qarg->Integer()->getText());
    auto i = var_map_->findLast(s) + n;
		circ_.addGate(GateFactory::CreateGate(Gate::Type::MEASURE, i));
  } else {
    auto s = std::string(qident->getText());
    auto sym = symtab_.find(s);
    assert(sym.first == SymbolTable::SymbolType::QREG);
		circ_.addGate(GateFactory::CreateGate(Gate::Type::MEASURE));
  }
}

void CircuitGenerator::GenerateResetQop(snuqasmParser::ResetQopContext *ctx) {
	assert(false);
}

void CircuitGenerator::GenerateQopStatement(snuqasmParser::QopStatementContext *ctx) {
	if (ctx->uopStatement()) {
		GenerateUopStatement(ctx->uopStatement());
	} else if (ctx->measureQop()) {
		GenerateMeasureQop(ctx->measureQop());
	} else if (ctx->resetQop()) {
		GenerateResetQop(ctx->resetQop());
	} else {
		assert(false);
	}
}

void CircuitGenerator::GenerateIfStatement(snuqasmParser::IfStatementContext *ctx) {
	assert(false);
}

void CircuitGenerator::GenerateBarrierStatement(snuqasmParser::BarrierStatementContext *ctx) {
	assert(false);
}

void CircuitGenerator::enterStatement(snuqasmParser::StatementContext *ctx) {
	if (ctx->qopStatement()) {
		GenerateQopStatement(ctx->qopStatement());
	} else if (ctx->ifStatement()) {
		GenerateIfStatement(ctx->ifStatement());
	} else if (ctx->barrierStatement()) {
		GenerateBarrierStatement(ctx->barrierStatement());
	}
}

} // namespace snuqs
