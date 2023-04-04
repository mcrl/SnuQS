#include "gate_factory.h"

#include <cassert>

namespace snuqs {

Gate::Type GateFactory::StringToType(const std::string &s) {
	if (s == "id") {
		return Gate::Type::ID;
	} else if (s == "h") {
		return Gate::Type::H;
	} else if (s == "x") {
		return Gate::Type::X;
	} else if (s == "y") {
		return Gate::Type::Y;
	} else if (s == "z") {
		return Gate::Type::Z;
	} else if (s == "sx") {
		return Gate::Type::SX;
	} else if (s == "sy") {
		return Gate::Type::SY;
	} else if (s == "s") {
		return Gate::Type::S;
	} else if (s == "sdg") {
		return Gate::Type::SDG;
	} else if (s == "t") {
		return Gate::Type::T;
	} else if (s == "tdg") {
		return Gate::Type::TDG;
	} else if (s == "rx") {
		return Gate::Type::RX;
	} else if (s == "ry") {
		return Gate::Type::RY;
	} else if (s == "rz") {
		return Gate::Type::RZ;
	} else if (s == "u1") {
		return Gate::Type::U1;
	} else if (s == "u2") {
		return Gate::Type::U2;
	} else if ((s == "U") || (s == "u") || (s == "u3")) {
		return Gate::Type::U3;
	} else if (s == "swap") {
		return Gate::Type::SWAP;
	} else if ((s == "CX") || (s == "cx")) {
		return Gate::Type::CX;
	} else if (s == "cy") {
		return Gate::Type::CY;
	} else if (s == "cz") {
		return Gate::Type::CZ;
	} else if (s == "ch") {
		return Gate::Type::CH;
	} else if (s == "crx") {
		return Gate::Type::CRX;
	} else if (s == "cry") {
		return Gate::Type::CRY;
	} else if (s == "crz") {
		return Gate::Type::CRZ;
	} else if (s == "cu1") {
		return Gate::Type::CU1;
	} else if (s == "cu2") {
		return Gate::Type::CU2;
	} else if (s == "cu3") {
		return Gate::Type::CU3;
	} else if (s == "ccx") {
		return Gate::Type::CCX;
	} else if (s == "zerostate") {
		return Gate::Type::ZEROSTATE;
	} else if (s == "uniform") {
		return Gate::Type::UNIFORM;
	} else if (s == "block") {
		return Gate::Type::BLOCK;
	} else if (s == "fusion") {
		return Gate::Type::FUSION;
	} else if (s == "placeholder") {
		return Gate::Type::PLACEHOLDER;
	} else if (s == "nswap") {
		return Gate::Type::NSWAP;
	} else if (s == "measure") {
		return Gate::Type::MEASURE;
	} 

  assert(false);

	return Gate::Type::ID;
}

std::string GateFactory::TypeToString(Gate::Type t) {
	switch (t) {
		case Gate::Type::ID:  return "id";
		case Gate::Type::H:   return "h";
		case Gate::Type::X:   return "x";
		case Gate::Type::Y:   return "y";
		case Gate::Type::Z:   return "z";
		case Gate::Type::SX:  return "sx";
		case Gate::Type::SY:  return "sy";
		case Gate::Type::S:   return "s";
		case Gate::Type::SDG: return "sdg";
		case Gate::Type::T:   return "t";
		case Gate::Type::TDG: return "tdg";
		case Gate::Type::RX:  return "rx";
		case Gate::Type::RY:  return "ry";
		case Gate::Type::RZ:  return "rz";
		case Gate::Type::U1:  return "u1";
		case Gate::Type::U2:  return "u2";
		case Gate::Type::U3:  return "u3";
		case Gate::Type::SWAP:  return "swap";
		case Gate::Type::CX:  return "cx";
		case Gate::Type::CY:  return "cy";
		case Gate::Type::CZ:  return "cz";
		case Gate::Type::CH:  return "ch";
		case Gate::Type::CRX: return "crx";
		case Gate::Type::CRY: return "cry";
		case Gate::Type::CRZ: return "crz";
		case Gate::Type::CU1: return "cu1";
		case Gate::Type::CU2: return "cu2";
		case Gate::Type::CU3: return "cu3";
		case Gate::Type::CCX: return "ccx";

		case Gate::Type::ZEROSTATE: return "zerostate";
		case Gate::Type::UNIFORM: return "uniform";

		case Gate::Type::FUSION: return "fusion";
		case Gate::Type::BLOCK: return "block";

		case Gate::Type::PLACEHOLDER: return "placeholder";
		case Gate::Type::NSWAP:  return "nswap";

		case Gate::Type::MEASURE: return "measure";
		default: assert(false);
	}
}

std::shared_ptr<Gate> GateFactory::CreateGate(std::string name) {
  return GateFactory::CreateGate(StringToType(name));
}

std::shared_ptr<Gate> GateFactory::CreateGate(Gate::Type type) {
  return GateFactory::CreateGate(type, {}, {});
}

std::shared_ptr<Gate> GateFactory::CreateGate(std::string name, size_t qubit) {
  return GateFactory::CreateGate(StringToType(name), qubit);
}

std::shared_ptr<Gate> GateFactory::CreateGate(Gate::Type type, size_t qubit) {
  return GateFactory::CreateGate(type, {qubit}, {});
}

std::shared_ptr<Gate> GateFactory::CreateGate(std::string name, std::vector<size_t> qubits, std::vector<real_t> params) {
  return GateFactory::CreateGate(StringToType(name), qubits, params);
}

std::shared_ptr<Gate> GateFactory::CreateGate(Gate::Type type, std::vector<size_t> qubits, std::vector<real_t> params) {
	switch (type) {
		case Gate::Type::ID:   return std::make_shared<Gate>(type, qubits, params);
		case Gate::Type::H:    return std::make_shared<Gate>(type, qubits, params);
		case Gate::Type::X:    return std::make_shared<Gate>(type, qubits, params);
		case Gate::Type::Y:    return std::make_shared<Gate>(type, qubits, params);
		case Gate::Type::Z:    return std::make_shared<Gate>(type, qubits, params);
		case Gate::Type::SX:   return std::make_shared<Gate>(type, qubits, params);
		case Gate::Type::SY:   return std::make_shared<Gate>(type, qubits, params);
		case Gate::Type::S:    return std::make_shared<Gate>(type, qubits, params);
		case Gate::Type::SDG:  return std::make_shared<Gate>(type, qubits, params);
		case Gate::Type::T:    return std::make_shared<Gate>(type, qubits, params);
		case Gate::Type::TDG:  return std::make_shared<Gate>(type, qubits, params);
		case Gate::Type::RX:   return std::make_shared<Gate>(type, qubits, params);
		case Gate::Type::RY:   return std::make_shared<Gate>(type, qubits, params);
		case Gate::Type::RZ:   return std::make_shared<Gate>(type, qubits, params);
		case Gate::Type::U1:   return std::make_shared<Gate>(type, qubits, params);
		case Gate::Type::U2:   return std::make_shared<Gate>(type, qubits, params);
		case Gate::Type::U3:   return std::make_shared<Gate>(type, qubits, params);
		case Gate::Type::SWAP: return  std::make_shared<Gate>(type, qubits, params);
		case Gate::Type::CX:   return std::make_shared<Gate>(type, qubits, params);
		case Gate::Type::CY:   return std::make_shared<Gate>(type, qubits, params);
		case Gate::Type::CZ:   return std::make_shared<Gate>(type, qubits, params);
		case Gate::Type::CH:   return std::make_shared<Gate>(type, qubits, params);
		case Gate::Type::CRX:  return std::make_shared<Gate>(type, qubits, params);
		case Gate::Type::CRY:  return std::make_shared<Gate>(type, qubits, params);
		case Gate::Type::CRZ:  return std::make_shared<Gate>(type, qubits, params);
		case Gate::Type::CU1:  return std::make_shared<Gate>(type, qubits, params);
		case Gate::Type::CU2:  return std::make_shared<Gate>(type, qubits, params);
		case Gate::Type::CU3:  return std::make_shared<Gate>(type, qubits, params);
		case Gate::Type::CCX:  return std::make_shared<Gate>(type, qubits, params);

		case Gate::Type::ZEROSTATE: return std::make_shared<Gate>(type, qubits, params);
		case Gate::Type::UNIFORM: return std::make_shared<Gate>(type, qubits, params);

		case Gate::Type::FUSION: return std::make_shared<Gate>(type, qubits, params);
		case Gate::Type::BLOCK: return std::make_shared<Gate>(type, qubits, params);

		case Gate::Type::PLACEHOLDER: return std::make_shared<Gate>(type, qubits, params);
		case Gate::Type::NSWAP:return std::make_shared<Gate>(type, qubits, params);

		case Gate::Type::MEASURE: return std::make_shared<Gate>(type, qubits, params);
	}

	return std::make_shared<Gate>(Gate::Type::ID, qubits, params);
}

}  // namespace snuqs 
