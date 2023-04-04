#pragma once

#include "antlr4-runtime.h"
#include "snuqasm-parser/snuqasmLexer.h"
#include "snuqasm-parser/snuqasmParser.h"
#include "snuqasm-parser/snuqasmBaseListener.h"

#include <string>


namespace snuqs {

class DeclContext; 

class Symbol {
	public:
	enum class SymbolType {
		QREG,
		CREG,
		GATE,
		OPAQUE_GATE,
		QPARAM,
		CPARAM,
	};

	Symbol(std::string name, SymbolType type, antlr4::tree::ParseTree *node)
	: name_(name),
	  type_(type),
	  node_(node) {
  }

  std::string name_;
  SymbolType type_;
  antlr4::tree::ParseTree *node_;
};

class SymbolTable {
	public:
	enum class SymbolType {
		QREG,
		CREG,
		GATE,
		OPAQUE_GATE,
		QPARAM,
		CPARAM,
	};

	using key_t = std::string;
	using node_t = antlr4::tree::ParseTree;
	using value_t = std::pair<SymbolType, node_t*>;
	using sym_map_t = std::map<key_t, value_t>;


	public:
	static std::string typeToString(SymbolType t);

	SymbolTable();

	void pushContext();
	void popContext();

	bool hasSymbol(key_t k) const;
	value_t find(key_t k) const;
	void insert(std::pair<key_t, value_t> p);
	void merge(SymbolTable &symtab);
	void dump() const;

	DeclContext *ctx_ = nullptr;
	sym_map_t sym_tab_;


  void EnterContext();
  void ExitContext();

  bool HasSymbol(std::string name) const;
  Symbol GetSymbol(std::string name) const;
  Symbol::SymbolType GetSymbolType(std::string name) const;
  void Insert(Symbol sym);
	std::map<std::string, Symbol> *table_;
	std::map<std::string, Symbol> global_table_;
	std::map<std::string, Symbol> local_table_;
};

class DeclContext {
	public:
	bool hasSymbol(const std::string &key) const;
	std::pair<SymbolTable::SymbolType, antlr4::tree::ParseTree*> find(const std::string &key) const;
	void insert(std::pair<std::string, std::pair<SymbolTable::SymbolType, antlr4::tree::ParseTree*>> p);

	DeclContext *parent = nullptr;
	std::map<std::string, std::pair<SymbolTable::SymbolType, antlr4::tree::ParseTree*>> symmap_;
	void dump() const;
};

} //namespace snuqs
