#pragma once

#include "antlr4-runtime.h"
#include "snuqasm-parser/snuqasmLexer.h"
#include "snuqasm-parser/snuqasmParser.h"
#include "snuqasm-parser/snuqasmBaseListener.h"

#include "symbol_table.h"
#include "circuit/quantum_circuit.h"

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

class CircuitGenerator : public snuqasmBaseListener {
	private:
	const SymbolTable &symtab_;
	QuantumCircuit circ_;

	VarMapping *var_map_ = nullptr;
	ExprMapping *expr_map_ = nullptr;
	std::map<std::string, size_t> bit_map_;

	void pushVarContext();
	void popVarContext();
	void pushExprContext();
	void popExprContext();
  void InitializeMapping();

	size_t ArgumentToNumber(snuqasmParser::ArgumentContext *ctx);
	std::vector<size_t> ArgumentToNumberList(std::vector<snuqasmParser::ArgumentContext*> &&arglist);
	size_t inlineArgumentToNumber(snuqasmParser::ArgumentContext *ctx);
	std::vector<size_t> inlineArgumentToNumberList(std::vector<snuqasmParser::ArgumentContext*> &&arglist);
	real_t expToReal(snuqasmParser::ExpContext *ctx);
	real_t inlineExpToReal(snuqasmParser::ExpContext *ctx);
	std::vector<real_t> inlineExplistToReal(snuqasmParser::ExplistContext *ctx);
	std::vector<real_t> explistToReal(snuqasmParser::ExplistContext *ctx);

	void idlistToMappings(snuqasmParser::IdlistContext* idlist, std::vector<Var> &mappings);
	void idlistToExprMappings(snuqasmParser::IdlistContext* idlist, std::vector<Expr> &mappings);
	void mixedlistToMappings(snuqasmParser::MixedlistContext* mixedlist, std::vector<Var> &mappings);
	void anylistToMappings(snuqasmParser::AnylistContext* anylist, std::vector<Var> &mappings);
	void explistToExprMappings(snuqasmParser::ExplistContext* explist, std::vector<Expr> &mappings);
	void inlineUnitaryOp(snuqasmParser::UnitaryOpContext *ctx);
	void inlineCustomOp(snuqasmParser::CustomOpContext *ctx);
	void inlineUopStatement(snuqasmParser::UopStatementContext *ctx);

	void GenerateUnitaryOp(snuqasmParser::UnitaryOpContext *ctx);
	void GenerateCustomOp(snuqasmParser::CustomOpContext *ctx);
	void GenerateUopStatement(snuqasmParser::UopStatementContext *ctx);
	void GenerateMeasureQop(snuqasmParser::MeasureQopContext *ctx);
	void GenerateResetQop(snuqasmParser::ResetQopContext *ctx);
	void GenerateQopStatement(snuqasmParser::QopStatementContext *ctx);
	void GenerateIfStatement(snuqasmParser::IfStatementContext *ctx);
	void GenerateBarrierStatement(snuqasmParser::BarrierStatementContext *ctx);

	public:
	CircuitGenerator(const SymbolTable &symtab)
	: symtab_(symtab)
	{}

	const QuantumCircuit& Generate(antlr4::tree::ParseTree *tree);
	virtual void enterStatement(snuqasmParser::StatementContext *) override;
};

} // namespace snuqs
