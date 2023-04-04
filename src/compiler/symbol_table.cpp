#include "symbol_table.h"

#include "antlr4-runtime.h"

#include "logger.h"


namespace snuqs {

//
// SymbolTable
//

SymbolTable::SymbolTable() {
  table_ = &global_table_;
  pushContext();
}

std::string SymbolTable::typeToString(SymbolTable::SymbolType t) {
	switch (t) {
		case SymbolTable::SymbolType::QREG: return "QREG";
		case SymbolTable::SymbolType::CREG: return "CREG";
		case SymbolTable::SymbolType::GATE: return "GATE";
		case SymbolTable::SymbolType::OPAQUE_GATE: return "OPAQUE_GATE";
		case SymbolTable::SymbolType::QPARAM: return "QPARAM";
		case SymbolTable::SymbolType::CPARAM: return "CPARAM";
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


void SymbolTable::EnterContext() {
  table_ = &local_table_;
}

void SymbolTable::ExitContext() {
  local_table_.clear();
  table_ = &global_table_;
}

bool SymbolTable::HasSymbol(std::string name) const {
  return table_->find(name) != table_->end();
}

Symbol SymbolTable::GetSymbol(std::string name) const {
  assert(HasSymbol(name));
  return table_->find(name)->second;
}

Symbol::SymbolType SymbolTable::GetSymbolType(std::string name) const {
  assert(HasSymbol(name));
  return GetSymbol(name).type_;
}

void SymbolTable::Insert(Symbol sym) {
  assert(!HasSymbol(sym.name_));
  table_->insert({sym.name_, sym});
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

std::pair<SymbolTable::SymbolType, antlr4::tree::ParseTree*>
DeclContext::find(const std::string &key) const {
	auto v = symmap_.find(key);
	if (v != symmap_.end()) {
		return v->second;
	} else if (parent == nullptr) {
		Logger::error("No such symbol: {}", key);
		std::exit(EXIT_FAILURE);
	} else {
		return parent->find(key);
	}
}

void DeclContext::insert(std::pair<std::string, std::pair<SymbolTable::SymbolType, antlr4::tree::ParseTree*>> p) {
	const auto &key = p.first;
	if (symmap_.find(key) != symmap_.end()) {
		Logger::error("redefinition of Symbol conflict: {}", key);
		std::exit(EXIT_FAILURE);
	}
	symmap_.insert(p);
}

void DeclContext::dump() const {
	Logger::info("Symbol table------------------\n");
	for (auto const& it : symmap_) {
	  auto key = it.first;
	  auto val = it.second;
		auto type = val.first;
		Logger::info("{} ({})\n", key, SymbolTable::typeToString(type));
	}
	Logger::info("------------------------------\n");
}

} //namespace snuqs
