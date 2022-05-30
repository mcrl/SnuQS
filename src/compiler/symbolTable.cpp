#include "antlr4-runtime.h"
#include "symbolTable.hpp"

#include <spdlog/spdlog.h>


namespace snuqs {

//
// SymbolTable
//
std::string SymbolTable::typeToString(SymbolTable::sym_type t) {
	switch (t) {
		case SymbolTable::sym_type::QREG: return "QREG";
		case SymbolTable::sym_type::CREG: return "CREG";
		case SymbolTable::sym_type::GATE: return "GATE";
		case SymbolTable::sym_type::QPARAM: return "QPARAM";
		case SymbolTable::sym_type::CPARAM: return "CPARAM";
	};
	return "";
}

void SymbolTable::pushContext() {
	auto *ctx = new DeclContext;
	ctx->parent = ctx_;
	ctx_ = ctx;
}

void SymbolTable::popContext() {
	auto *ctx = ctx_;
	ctx_ = ctx_->parent; 
	delete ctx;
}

bool SymbolTable::hasSymbol(key_t k) const {
	return ctx_->hasSymbol(k);
}

SymbolTable::value_t SymbolTable::find(key_t key) const {
	return ctx_->find(key);
}

void SymbolTable::insert(std::pair<key_t, value_t> p) {
	ctx_->insert(p);
}

void SymbolTable::merge(SymbolTable &symtab) {
//	for (auto const& [key, val] : symtab.get()) {
//		sym_tab_.insert({key, val});
//	}
}

void SymbolTable::dump() const {
	ctx_->dump();
}

//
// DeclContext


bool DeclContext::hasSymbol(const std::string &key) const {
	if (symmap_.find(key) != symmap_.end()) {
		return true;
	} else if (parent == nullptr) {
		return false;
	} else {
		return parent->hasSymbol(key);
	}
}

std::pair<SymbolTable::sym_type, antlr4::tree::ParseTree*>
DeclContext::find(const std::string &key) const {
	auto v = symmap_.find(key);
	if (v != symmap_.end()) {
		return v->second;
	} else if (parent == nullptr) {
		spdlog::error("No such symbol: {}", key);
		std::exit(EXIT_FAILURE);
	} else {
		return parent->find(key);
	}
}

void DeclContext::insert(std::pair<std::string, std::pair<SymbolTable::sym_type, antlr4::tree::ParseTree*>> p) {
	const auto &key = p.first;
	if (symmap_.find(key) != symmap_.end()) {
		spdlog::error("redefinition of Symbol conflict: {}", key);
		std::exit(EXIT_FAILURE);
	}
	symmap_.insert(p);
}

void DeclContext::dump() const {
	spdlog::info("Symbol table------------------");
	for (auto const& [key, val] : symmap_) {
		auto type = val.first;
		spdlog::info("{} ({})", key, SymbolTable::typeToString(type));
	}
	spdlog::info("------------------------------");
}

} //namespace snuqs
