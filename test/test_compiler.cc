#include <iostream>
#include <string>

#include "compiler/compiler.h"


void test(const char *filename)
{
  snuqs::Compiler compiler;
  std::cout << "Testing " << filename << "\n";
  compiler.Compile(std::string(filename));
  std::cout << compiler.GetQuantumCircuit() << "\n";
  std::cout << "Done\n\n";
}

int main(int argc, char *argv[])
{
  std::cout << "========================================\n";
  std::cout << "============= Compiler TEST ============\n";
  std::cout << "========================================\n";

  const char* examples[] = {
    "../examples/adder.sq",
    "../examples/bigadder.sq",
    "../examples/invalid_gate_no_found.sq",
    "../examples/invalid_missing_semicolon.sq",
    "../examples/inverseqft1.sq",
    "../examples/inverseqft2.sq",
    "../examples/ipea_3_pi_8.sq",
    "../examples/pea_3_pi_8.sq",
    "../examples/qec.sq",
    "../examples/qft.sq",
    "../examples/qpt.sq",
    "../examples/rb.sq",
    "../examples/teleport.sq",
    "../examples/teleportv2.sq",
    "../examples/W-state.sq",
  };
  for (int i = 0; i < sizeof(examples) / sizeof(const char*); ++i) {
    test(examples[i]);
  }

  return 0;
}
