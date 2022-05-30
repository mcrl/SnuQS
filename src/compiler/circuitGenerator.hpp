#pragma once

#include "antlr4-runtime.h"
#include "snuql-parser/snuqlLexer.h"
#include "snuql-parser/snuqlParser.h"
#include "snuql-parser/snuqlBaseListener.h"


#include "symbolTable.hpp"
#include "quantumCircuit.hpp"

namespace snuqs {

class Var {
	public:
	std::string s = "";
	bool designated = false;
	size_t i = 1000;
	Var(std::string s, size_t i);
	Var(std::string s);
	Var(size_t i);
	bool operator==(const Var &other) const;
	bool operator<(const Var &other) const;
};

class VarMapping {
	public:
	std::map<Var, Var> map_;
	void insert(Var k, Var v);
	size_t findLast(const std::string &s) const;
	size_t findLast(const std::string &s, size_t i) const;
	VarMapping *parent = nullptr;
};

enum class ExprOp {
	CONST,
	IDENT,
	ADD,
	SUB,
	MUL,
	DIV,
	XOR,
	NEG,
	SIN,
	COS,
	TAN,
	EXP,
	LOG,
	SQRT,
};

class Expr {
	public:
	ExprOp op = ExprOp::CONST;
	std::string s;
	real_t d = -1234.;
	Expr(ExprOp op, Expr *l);
	Expr(ExprOp op, Expr *l, Expr *r);
	Expr(std::string s);
	Expr(real_t d);
	bool operator<(const Expr &other) const;

	Expr *left = nullptr;
	Expr *right = nullptr;
};

class ExprMapping {
	public:
	std::map<Expr, Expr> map_;
	void insert(Expr k, Expr v);
	real_t eval(const std::string &s) const;
	ExprMapping *parent = nullptr;
};

class CircuitGenerator : public snuqlBaseListener {
	private:
	const SymbolTable &symtab_;
	QuantumCircuit circ_;

	VarMapping *var_map_ = nullptr;
	ExprMapping *expr_map_ = nullptr;
	std::map<std::pair<std::string, size_t>, size_t> bit_map_;

	void pushVarContext();
	void popVarContext();
	void pushExprContext();
	void popExprContext();
	void initializeMapping(std::ostream &os);

	size_t argumentToNumber(snuqlParser::ArgumentContext *ctx);
	std::vector<size_t> arglistToNumberList(snuqlParser::ArglistContext*);
	std::vector<size_t> argumentsToNumberList(std::vector<snuqlParser::ArgumentContext*> &&arglist);
	size_t inlineArgumentToNumber(snuqlParser::ArgumentContext *ctx);
	std::vector<size_t> inlineArgumentToNumberList(std::vector<snuqlParser::ArgumentContext*> &&arglist);
	real_t expToReal(snuqlParser::ExpContext *ctx);
	real_t inlineExpToReal(snuqlParser::ExpContext *ctx);
	std::vector<real_t> inlineExplistToReal(snuqlParser::ExplistContext *ctx);
	std::vector<real_t> explistToReal(snuqlParser::ExplistContext *ctx);

	void idlistToMappings(snuqlParser::IdlistContext* idlist, std::vector<Var> &mappings);
	void idlistToExprMappings(snuqlParser::IdlistContext* idlist, std::vector<Expr> &mappings);
	void mixedlistToMappings(snuqlParser::MixedlistContext* mixedlist, std::vector<Var> &mappings);
	void anylistToMappings(snuqlParser::AnylistContext* anylist, std::vector<Var> &mappings);
	void explistToExprMappings(snuqlParser::ExplistContext* explist, std::vector<Expr> &mappings);
	void inlineUnitaryOp(snuqlParser::UnitaryOpContext *ctx);
	void inlineCustomOp(snuqlParser::CustomOpContext *ctx);
	void inlineUopStatement(snuqlParser::UopStatementContext *ctx);

	void generateUnitaryOp(snuqlParser::UnitaryOpContext *ctx);
	void generateCustomOp(snuqlParser::CustomOpContext *ctx);
	void generateUopStatement(snuqlParser::UopStatementContext *ctx);
	void generateMeasureQop(snuqlParser::MeasureQopContext *ctx);
	void generateResetQop(snuqlParser::ResetQopContext *ctx);
	void generateQopStatement(snuqlParser::QopStatementContext *ctx);
	void generateIfStatement(snuqlParser::IfStatementContext *ctx);
	void generateBarrierStatement(snuqlParser::BarrierStatementContext *ctx);

	public:
	CircuitGenerator(const SymbolTable &symtab)
	: symtab_(symtab)
	{}

	const QuantumCircuit& generate(antlr4::tree::ParseTree *tree, std::ostream &os);
	void enterStatement(snuqlParser::StatementContext *) override;
};

} // namespace snuqs
