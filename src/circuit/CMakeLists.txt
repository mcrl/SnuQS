add_library(circuit INTERFACE "")

target_sources(circuit
  INTERFACE
  gate.cpp 
  gate_cpu_impl.cpp 
  gate_factory.cpp 
  gate_interface.cpp 
  quantum_circuit.cpp
	)
