#include "simulator.h"

#include <cassert>

namespace snuqs {

static Simulator::unique_ptr CreateCPUIOSimulator(Simulator::Method method) {
	switch (method) {
    case Simulator::Method::kStateVector:
      return std::move(std::make_unique<StatevectorCPUIOSimulator>());
  }
  return std::move(std::make_unique<NoSimulator>());
}

//static Simulator::unique_ptr CreateGPUIOSimulator(Simulator::Method method) {
//	switch (method) {
//    case Simulator::Method::kStateVector:
//      return std::move(std::make_unique<StatevectorGPUIOSimulator>());
//  }
//  return std::move(std::make_unique<NoSimulator>());
//}

static Simulator::unique_ptr CreateCPUSimulator(Simulator::Method method) {
	switch (method) {
    case Simulator::Method::kStateVector:
      return std::move(std::make_unique<StatevectorCPUSimulator>());
  }
  return std::move(std::make_unique<NoSimulator>());
}

//static Simulator::unique_ptr CreateGPUSimulator(Simulator::Method method) {
//	switch (method) {
//    case Simulator::Method::kStateVector:
//      return std::move(std::make_unique<StatevectorGPUSimulator>());
//  }
//  return std::move(std::make_unique<NoSimulator>());
//}

static Simulator::unique_ptr CreateIOSimulator(
	Simulator::Method method,
	Simulator::Device device) {

	switch (device) {
	  case Simulator::Device::kCPU:
      return std::move(CreateCPUIOSimulator(method));
      /*
	  case Simulator::Device::kGPU:
      return std::move(CreateGPUIOSimulator(method));
      */
  }
  return std::move(std::make_unique<NoSimulator>());
}

static Simulator::unique_ptr CreateMemorySimulator(
	Simulator::Method method,
	Simulator::Device device) {

	switch (device) {
	  case Simulator::Device::kCPU:
      return std::move(CreateCPUSimulator(method));
      /*
	  case Simulator::Device::kGPU:
      return std::move(CreateGPUSimulator(method));
      */
  }
  return std::move(std::make_unique<NoSimulator>());
}

Simulator::unique_ptr Simulator::CreateSimulator(
	Simulator::Method method,
	Simulator::Device device,
	bool useio) {

	if (useio) {
	  return std::move(CreateIOSimulator(method, device));
  } else {
	  return std::move(CreateMemorySimulator(method, device));
  }
}



} // namespace snuqs
