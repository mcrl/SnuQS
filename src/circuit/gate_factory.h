#pragma once

#include "gate.h"
#include "types.h"

namespace snuqs {

class GateFactory {
  private:
  GateFactory() = delete;

  public:
  static std::shared_ptr<Gate> CreateGate(std::string name);
  static std::shared_ptr<Gate> CreateGate(Gate::Type type);
  static std::shared_ptr<Gate> CreateGate(std::string name, size_t qubit);
  static std::shared_ptr<Gate> CreateGate(Gate::Type type, size_t qubit);
  static std::shared_ptr<Gate> CreateGate(std::string name,  std::vector<size_t> qubits, std::vector<real_t> params);
  static std::shared_ptr<Gate> CreateGate(Gate::Type type, std::vector<size_t> qubits, std::vector<real_t> params);

  static Gate::Type StringToType(const std::string &s);
  static std::string TypeToString(Gate::Type t);

};

} //namespace snuqs
