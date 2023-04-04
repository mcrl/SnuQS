#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "optionparser.h"

#include "circuit/quantum_circuit.h"
#include "simulator/simulator.h"
#include "simulator/job.h"
#include "compiler/compiler.h"
#include "compiler/optimizer.h"


enum optionIndex { UNKNOWN, HELP, SNUQL, METHOD, DEVICE, STORAGE, VERBOSE, MODEL, USEIO };

struct Arg: public option::Arg
{
  static void printError(const char* msg1, const option::Option& opt, const char* msg2)
  {
    fprintf(stderr, "%s", msg1);
    fwrite(opt.name, opt.namelen, 1, stderr);
    fprintf(stderr, "%s", msg2);
  }

  static option::ArgStatus Unknown(const option::Option& option, bool msg)
  {
    if (msg) printError("Unknown option '", option, "'\n");
    return option::ARG_ILLEGAL;
  }

  static option::ArgStatus Required(const option::Option& option, bool msg)
  {
    if (option.arg != 0)
      return option::ARG_OK;

    if (msg) printError("Option '", option, "' requires an argument\n");
    return option::ARG_ILLEGAL;
  }

  static option::ArgStatus NonEmpty(const option::Option& option, bool msg)
  {
    if (option.arg != 0 && option.arg[0] != 0)
      return option::ARG_OK;

    if (msg) printError("Option '", option, "' requires a non-empty argument\n");
    return option::ARG_ILLEGAL;
  }

  static option::ArgStatus Numeric(const option::Option& option, bool msg)
  {
    char* endptr = 0;
    if (option.arg != 0 && strtol(option.arg, &endptr, 10)){};
    if (endptr != option.arg && *endptr == 0)
      return option::ARG_OK;

    if (msg) printError("Option '", option, "' requires a numeric argument\n");
    return option::ARG_ILLEGAL;
  }
};

const option::Descriptor usage[] =
{
  {UNKNOWN, 0, "", "",option::Arg::None,
    "USAGE: example [options]\n\n"
      "Options:" },
  {HELP, 0,"", "help", option::Arg::None,
    "  --help  \tPrint usage and exit." },
  {SNUQL, 0,"s","snuql", Arg::Required,
    "  --snuql, -s  \tsnuql file name."},
  {METHOD, 0,"m","method", Arg::Optional,
    "  --method, -m  \tsimulation method. default: statevector."},
  {DEVICE, 0,"d","device", Arg::Optional,
    "  --device, -d  \tsimulation device. default: cpu."},
  {STORAGE, 0,"s","storage", Arg::Optional,
    "  --storage, -s  \tstorage description file name."},
  {USEIO, 0,"i","io", Arg::Optional,
    "  --io, -i  \tsimulate using io devices."},
  {UNKNOWN, 0, "", "",option::Arg::None,
    "\nExamples:\n"
      "  snuqs -s <snuql_filename> --method=statevector --device=cpu\n"
  },

  {0,0,0,0,0,0}
};

void printUsageAndExit(int retval) {
  option::printUsage(std::cout, usage);
  std::exit(retval);
}

std::string GetFileName(const std::vector<option::Option> &options) {
  return options[SNUQL].arg;
}

snuqs::Simulator::Method GetMethod(const std::vector<option::Option> &options) {
  snuqs::Simulator::Method m = snuqs::Simulator::Method::kStateVector;
  if (options[METHOD]) {
    const std::string &method = options[METHOD].arg;

    if (method == "statevector") {
      m = snuqs::Simulator::Method::kStateVector;
    } else {
      printUsageAndExit(EXIT_FAILURE);
    }
  }
  return m;
}

snuqs::Simulator::Device GetDevice(const std::vector<option::Option> &options) {
  snuqs::Simulator::Device m = snuqs::Simulator::Device::kCPU;
  if (options[DEVICE]) {
    const std::string &device = options[DEVICE].arg;

    if (device == "cpu") {
      m = snuqs::Simulator::Device::kCPU;
    } else if (device == "gpu") {
      m = snuqs::Simulator::Device::kGPU;
    } else {
      printUsageAndExit(EXIT_FAILURE);
    }
  }
  return m;
}

bool GetUseIO(const std::vector<option::Option> &options) {
  if (options[USEIO]) {
    return true;
  }
  return false;
}



int main(int argc, char* argv[])
{
  int num_args = argc-1;
  char **argp = &argv[1];

  // -------------------
  // Parse options
  // -------------------
  option::Stats stats(usage, num_args, argp);
  std::vector<option::Option> options(stats.options_max);
  std::vector<option::Option> buffer(stats.buffer_max);
  option::Parser args(usage, num_args, argp, &options[0], &buffer[0]);

  if (args.error()) {
    exit(EXIT_FAILURE);
  }

  if (options[UNKNOWN]) {
    for (option::Option* opt = options[UNKNOWN]; opt; opt = opt->next()) {
      std::cerr << "Unknown option: " << std::string(opt->name,opt->namelen) << "\n";
    }
    std::exit(EXIT_FAILURE);
  }

  if (args.nonOptionsCount() > 0 ){
    for (int i = 0; i < args.nonOptionsCount(); ++i) {
      std::cerr << "Unknown input: "<< args.nonOption(i) << "\n";
    }
    std::exit(EXIT_FAILURE);
  }

  if (options[HELP]) {
    printUsageAndExit(EXIT_SUCCESS);
  }

  if (!options[SNUQL]) {
    printUsageAndExit(EXIT_FAILURE);
  }

  // -------------------

  //
  // Configure parameters
  //
  std::cout << "Simulating " << options[SNUQL].arg << "\n";
  try {

    //
    // 1. Compile
    //
    snuqs::Compiler compiler;
    compiler.Compile(GetFileName(options));

    //
    // 2. Optimize
    //
    snuqs::Optimizer optimizer;
    optimizer.Optimize(compiler.GetQuantumCircuit());
    const std::vector<snuqs::QuantumCircuit> &circs = optimizer.GetQuantumCircuits();


    //
    // 3. Simulate
    //
    snuqs::Simulator::Method method = GetMethod(options);
    snuqs::Simulator::Device device = GetDevice(options);
    bool useio = GetUseIO(options);
    snuqs::Simulator::unique_ptr sim = snuqs::Simulator::CreateSimulator(method, device, useio);

    snuqs::Job job(sim.get());
    job.Run(circs);

  } catch (std::exception &e) {
    std::cerr << e.what() << "\n";
    std::exit(EXIT_FAILURE);
  }

  return 0;
}
