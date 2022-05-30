#pragma once

#include "antlr4-runtime.h"
#include "snuql-parser/snuqlLexer.h"
#include "snuql-parser/snuqlParser.h"
#include "snuql-parser/snuqlBaseListener.h"

#include <string>


namespace snuqs {

class DeclContext; 

class SymbolTable {
	public:
	enum class sym_type {
		QREG,
		CREG,
		GATE,
		QPARAM,
		CPARAM,
	};

	using key_t = std::string;
	using node_t = antlr4::tree::ParseTree;
	using value_t = std::pair<sym_type, node_t*>;
	using sym_map_t = std::map<key_t, value_t>;


	public:
	static std::string typeToString(sym_type t);

	SymbolTable() {
		pushContext();
	}

	void pushContext();
	void popContext();

	bool hasSymbol(key_t k) const;
	value_t find(key_t k) const;
	void insert(std::pair<key_t, value_t> p);
	void merge(SymbolTable &symtab);
	void dump() const;

	DeclContext *ctx_ = nullptr;
	sym_map_t sym_tab_;
};

class DeclContext {
	public:
	bool hasSymbol(const std::string &key) const;
	std::pair<SymbolTable::sym_type, antlr4::tree::ParseTree*> find(const std::string &key) const;
	void insert(std::pair<std::string, std::pair<SymbolTable::sym_type, antlr4::tree::ParseTree*>> p);

	DeclContext *parent = nullptr;
	std::map<std::string, std::pair<SymbolTable::sym_type, antlr4::tree::ParseTree*>> symmap_;
	void dump() const;
};

} //namespace snuqs
